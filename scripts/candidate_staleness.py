"""Reuse the candidate set or the final selection across decode steps: what is lost?

Decodes greedily with dense attention and, at every step and layer, builds the two-stage
quest+mean candidate set (top `prefilter` of free blocks, shared per KV head) and the
final TopCdf + floor selection from int8 block mass. For each lag it then scores, per
QUERY head, the true attention mass captured by
  * stale_sel{lag}:  the selection made `lag` steps ago (plus this step's sink/local
                     blocks), i.e. reusing the whole selection;
  * stale_cand{lag}: a fresh TopCdf + floor selection over the candidate set made `lag`
                     steps ago (plus sink/local), i.e. reusing only stage 1;
  * xlayer_sel / xlayer_cand: the same two reuses across layers instead of steps: layer
                     i takes layer i-1's selection or candidate set from this step
                     (TidalDecode-style), so stage 1 or both stages run every other layer.
Fresh selection is the reference.

Example:
    python scripts/candidate_staleness.py --model Qwen/Qwen3-4B-Instruct-2507
"""

import json
import math
from collections import defaultdict

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from group_diagnosis import _topcdf
from ruler_kernel_eval import _prompt_ids
from transformers import AutoModelForCausalLM, AutoTokenizer


class StalenessProbe(DecodeLoop):
    def __init__(self, *args, lags, **kwargs):
        super().__init__(*args, **kwargs)
        self.lags = lags
        self.cand_hist = defaultdict(list)
        self.sel_hist = defaultdict(list)
        self.stats = defaultdict(list)
        self.prev_layer = None  # (step, layer, cand, sel) of the last layer attended

    def _attend(self, i, q, attn):
        qs = (q.squeeze(2) * attn.scaling).contiguous()
        n = int(self.seq_len)
        B, Hkv, Hq, bs = self.B, self.Hkv, self.Hq, self.bs
        G = Hq // Hkv
        cur_p = (n + bs - 1) // bs
        pad = cur_p * bs - n
        dev = q.device
        blocks = torch.arange(cur_p, device=dev)
        forced = (blocks < self.sink_blocks) | (blocks >= cur_p - self.local_blocks)
        free = ~forced
        qg = qs.view(B, Hkv, G, -1).float()
        kk = self.k[i][:, :, :n].float()

        att = torch.softmax(torch.einsum("bhgd,bhnd->bhgn", qg, kk), -1)
        true_mass = F.pad(att, (0, pad)).view(B, Hkv, G, cur_p, bs).sum(-1)

        valid_tok = (torch.arange(cur_p * bs, device=dev) < n).view(cur_p, bs, 1)
        kb = F.pad(kk, (0, 0, 0, pad)).view(B, Hkv, cur_p, bs, -1)
        kmax = kb.masked_fill(~valid_tok, float("-inf")).amax(3)
        kmin = kb.masked_fill(~valid_tok, float("inf")).amin(3)
        kmean = (kb * valid_tok).sum(3) / valid_tok.sum(1)
        quest = qg.relu() @ kmax.transpose(-1, -2) + (qg - qg.relu()) @ kmin.transpose(
            -1, -2
        )
        mean_l = qg @ kmean.transpose(-1, -2)
        mix = 0.5 * torch.softmax(quest, -1) + 0.5 * torch.softmax(mean_l, -1)
        shared = mix.mean(2).masked_fill(forced, -1.0)
        n_keep = math.ceil(int(free.sum()) * self.prefilter)
        top = torch.zeros_like(shared, dtype=torch.bool).scatter(
            -1, shared.topk(n_keep, -1).indices, True
        )
        cand = (top & free) | forced

        k8, ks = self.k8[i][:, :, :n], self.ks[i][:, :, :n]
        kd = (k8.float() * ks[..., None]).to(qs.dtype).float()
        l8 = F.pad(
            torch.einsum("bhgd,bhnd->bhgn", qg, kd), (0, pad), value=float("-inf")
        )
        lm8 = torch.logsumexp(l8.view(B, Hkv, G, cur_p, bs), -1)
        cap = min(
            cur_p,
            self.sink_blocks + self.local_blocks + math.ceil(self.max_frac * cur_p),
        )

        def select(c):
            pr = torch.softmax(lm8.masked_fill(~c.unsqueeze(2), float("-inf")), -1)
            return _topcdf(pr.mean(2), c, forced, self.cdf, self.floor, cap)

        def record(name, sel):
            m = (true_mass * sel.unsqueeze(2)).sum(-1)
            self.stats[name].append(m.flatten().cpu())
            self.stats[name + "|grpmin"].append(m.amin(-1).flatten().cpu())
            self.stats[name + "|dens"].append(
                (sel.sum(-1).float() / cur_p).flatten().cpu()
            )

        sel = select(cand)
        record("fresh", sel)

        def grow(x):  # an older [.., p] mask padded to this step's cur_p blocks
            return F.pad(x, (0, cur_p - x.shape[-1]))

        for lag in self.lags:
            if len(self.cand_hist[i]) >= lag:
                record(f"stale_sel{lag}", grow(self.sel_hist[i][-lag]) | forced)
                record(
                    f"stale_cand{lag}", select(grow(self.cand_hist[i][-lag]) | forced)
                )
        pl = self.prev_layer
        if pl is not None and pl[0] == self.step_i and pl[1] == i - 1:
            record("xlayer_sel", grow(pl[3]) | forced)
            record("xlayer_cand", select(grow(pl[2]) | forced))
        self.prev_layer = (self.step_i, i, cand, sel)
        self.cand_hist[i].append(cand)
        self.sel_hist[i].append(sel)
        if len(self.cand_hist[i]) > max(self.lags):
            self.cand_hist[i].pop(0)
            self.sel_hist[i].pop(0)
        valid_k = (self.positions < self.seq_len).view(1, 1, 1, -1)
        return F.scaled_dot_product_attention(
            q,
            self.k[i],
            self.v[i],
            attn_mask=valid_k,
            scale=attn.scaling,
            enable_gqa=True,
        ).squeeze(2)


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    subsets: str = "niah_multikey_2,niah_multikey_3,qa_1",
    n: int = 3,
    new_tokens: int = 40,
    lags=(1, 4, 8, 16),
    cdf: float = 0.95,
    floor: float = 0.02,
    max_frac: float = 0.15,
    prefilter: float = 0.25,
    block_size: int = 64,
    out: str = "",
) -> None:
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    lags = [int(x) for x in lags]
    stats = defaultdict(list)
    subs = subsets.split(",") if isinstance(subsets, str) else list(subsets)
    for subset in subs:
        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset).select(
            range(n)
        )
        for row in rows:
            ids = _prompt_ids(tok, row).cuda()
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            loop = StalenessProbe(
                m,
                pre.past_key_values,
                ids.shape[1],
                1,
                new_tokens + 1,
                block_size,
                "two_stage",
                cdf,
                floor,
                max_frac,
                math.ceil(128 / block_size),
                math.ceil(128 / block_size),
                (layer_pre, layer_post),
                lags=lags,
            )
            loop.prefilter = prefilter
            loop.reset(pre.logits[:, -1].argmax(-1))
            for _ in range(new_tokens):
                loop.step()
            for key, vals in loop.stats.items():
                stats[key] += vals
            del loop, pre
            torch.cuda.empty_cache()
    res = {}
    names = ["fresh", "xlayer_cand", "xlayer_sel"] + [
        f"{kind}{lag}" for lag in lags for kind in ("stale_cand", "stale_sel")
    ]
    for name in names:
        x = torch.cat(stats[name])
        res[name] = dict(
            mean=float(x.mean()),
            p10=float(x.quantile(0.1)),
            p1=float(x.quantile(0.01)),
            below_0p8=float((x < 0.8).float().mean()),
            grpmin=float(torch.cat(stats[name + "|grpmin"]).mean()),
            density=float(torch.cat(stats[name + "|dens"]).mean()),
        )
        r = res[name]
        print(
            f"{name:14s} mass {r['mean']:.3f} p10 {r['p10']:.3f} p1 {r['p1']:.3f} "
            f"<0.8 {r['below_0p8']:.3f} grpmin {r['grpmin']:.3f} density {r['density']:.3f}",
            flush=True,
        )
    if out:
        json.dump(
            dict(model=model, subsets=subs, results=res), open(out, "w"), indent=2
        )


if __name__ == "__main__":
    fire.Fire(main)
