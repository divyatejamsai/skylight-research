"""Attention output error of two-stage selection, with and without a tail estimate.

Selection drops the mass of unselected blocks and renormalizes over what it keeps.
Instead, add each unselected block back as one pseudo-token carrying the block's mean
value vector and an estimated block weight:
  t1  candidates that were not selected: exact block log-mass from the int8 stage-2 keys
  t2  blocks the prefilter dropped: q · mean key + log(block length) (a lower bound)
  g   t1 only for diffuse blocks: top-token share of the int8 block mass below rho
so the softmax denominator stays right and the diffuse tail still contributes.

Decodes greedily with dense attention (identical queries for every variant) and
records, per query head, the relative L2 error of the attention output against dense.

Example:
    python scripts/tail_estimate.py --model Qwen/Qwen2.5-1.5B-Instruct --n 4
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


class TailProbe(DecodeLoop):
    def __init__(self, *args, taus, rhos, **kwargs):
        super().__init__(*args, **kwargs)
        self.taus, self.rhos = taus, rhos
        self.stats = defaultdict(list)

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
        vv = self.v[i][:, :, :n].float()

        logits = torch.einsum("bhgd,bhnd->bhgn", qg, kk)  # exact (bf16 keys)
        dense = torch.softmax(logits, -1) @ vv  # [B, Hkv, G, D]
        lt = F.pad(logits, (0, pad), value=float("-inf")).view(B, Hkv, G, cur_p, bs)

        valid_tok = (torch.arange(cur_p * bs, device=dev) < n).view(cur_p, bs, 1)
        cnt = valid_tok.sum(1).float()  # [cur_p, 1]
        kb = F.pad(kk, (0, 0, 0, pad)).view(B, Hkv, cur_p, bs, -1)
        vb = F.pad(vv, (0, 0, 0, pad)).view(B, Hkv, cur_p, bs, -1)
        kmax = kb.masked_fill(~valid_tok, float("-inf")).amax(3)
        kmin = kb.masked_fill(~valid_tok, float("inf")).amin(3)
        kmean = (kb * valid_tok).sum(3) / cnt
        vmean = (vb * valid_tok).sum(3) / cnt  # [B, Hkv, cur_p, D]

        quest = qg.relu() @ kmax.transpose(-1, -2) + (qg - qg.relu()) @ kmin.transpose(
            -1, -2
        )
        mean_l = qg @ kmean.transpose(-1, -2)  # [B, Hkv, G, cur_p]
        pre_scores = {
            "q": quest,
            "qm": torch.log(
                0.5 * torch.softmax(quest, -1) + 0.5 * torch.softmax(mean_l, -1)
            ),
        }
        k8, ks = self.k8[i][:, :, :n], self.ks[i][:, :, :n]
        kd = (k8.float() * ks[..., None]).to(qs.dtype).float()
        l8 = F.pad(
            torch.einsum("bhgd,bhnd->bhgn", qg, kd), (0, pad), value=float("-inf")
        )
        lm8 = torch.logsumexp(l8.view(B, Hkv, G, cur_p, bs), -1)  # int8 block log-mass
        # Share of a block's mass on its top token: near 1/bs when diffuse, near 1 when
        # one token dominates (then the block-mean value is a poor stand-in).
        peak = torch.exp(l8.view(B, Hkv, G, cur_p, bs).amax(-1) - lm8)
        t2_lm = mean_l + torch.log(cnt).view(1, 1, 1, cur_p)  # lower bound on log-mass
        cap = min(
            cur_p,
            self.sink_blocks + self.local_blocks + math.ceil(self.max_frac * cur_p),
        )
        n_keep = math.ceil(int(free.sum()) * self.prefilter)

        for pname, s in pre_scores.items():
            shared = torch.softmax(s.masked_fill(forced, float("-inf")), -1).mean(2)
            top = torch.zeros_like(shared, dtype=torch.bool).scatter(
                -1, shared.topk(n_keep, -1).indices, True
            )
            cand = (top & free) | forced  # [B, Hkv, cur_p]
            pr = torch.softmax(lm8.masked_fill(~cand.unsqueeze(2), float("-inf")), -1)
            for tau in self.taus:
                sel = _topcdf(pr.mean(2), cand, forced, tau, self.floor, cap)
                selg = sel.unsqueeze(2)  # [B, Hkv, 1, cur_p]
                ls = lt.masked_fill(~selg.unsqueeze(-1), float("-inf"))
                mx = ls.amax((-1, -2), keepdim=True)
                e = torch.exp(ls - mx)  # [B, Hkv, G, cur_p, bs]
                num = torch.einsum("bhgpt,bhptd->bhgd", e, vb)
                den = e.sum((-1, -2))[..., None]
                mxs = mx.squeeze(-1)  # [B, Hkv, G, 1]
                outs = {"none": num / den}
                t1 = (cand & ~sel).unsqueeze(2)
                t2 = (~cand).unsqueeze(2)
                w1 = torch.exp(lm8 - mxs) * t1
                w2 = torch.exp(t2_lm - mxs) * t2
                tails = [("t1", w1), ("t12", w1 + w2)]
                for rho in self.rhos:
                    tails.append((f"t12g{rho}", w1 * (peak < rho) + w2))
                for tname, w in tails:
                    outs[tname] = (num + w @ vmean) / (den + w.sum(-1, keepdim=True))
                dens = float(sel.sum(-1).float().mean()) / cur_p
                for tname, o in outs.items():
                    err = (o - dense).norm(dim=-1) / dense.norm(dim=-1).clamp_min(1e-6)
                    key = f"{pname}|{tau}|{tname}"
                    self.stats[key].append(err.flatten().cpu())
                    self.stats[key + "|dens"].append(torch.tensor([dens]))
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
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    subsets: str = "niah_multikey_2,niah_multikey_3,qa_1",
    n: int = 3,
    new_tokens: int = 16,
    taus=(0.8, 0.9, 0.95),
    rhos=(0.05, 0.1, 0.2),
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
    taus = [float(t) for t in taus]
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
            loop = TailProbe(
                m,
                pre.past_key_values,
                ids.shape[1],
                1,
                new_tokens + 1,
                block_size,
                "two_stage",
                0.95,
                floor,
                max_frac,
                math.ceil(128 / block_size),
                math.ceil(128 / block_size),
                (layer_pre, layer_post),
                taus=taus,
                rhos=[float(r) for r in rhos],
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
    for key in sorted(k for k in stats if not k.endswith("|dens")):
        e = torch.cat(stats[key])
        res[key] = dict(
            err_mean=float(e.mean()),
            err_p90=float(e.quantile(0.9)),
            err_p99=float(e.quantile(0.99)),
            density=float(torch.cat(stats[key + "|dens"]).mean()),
        )
        r = res[key]
        print(
            f"{key:18s} err {r['err_mean']:.4f} p90 {r['err_p90']:.4f} "
            f"p99 {r['err_p99']:.4f} density {r['density']:.3f}",
            flush=True,
        )
    if out:
        json.dump(
            dict(model=model, subsets=subs, results=res), open(out, "w"), indent=2
        )


if __name__ == "__main__":
    fire.Fire(main)
