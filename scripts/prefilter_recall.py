"""Stage-1 recall: how much true attention mass survives each candidate prefilter?

Decodes greedily with dense attention and, at every step and layer, ranks blocks with
several cheap scorers, keeps the top `frac` of free blocks (shared over the GQA group, or
the union of per-query-head tops), and records the true fp32 attention mass of each
QUERY head that lies inside the candidate set (forced sink/local blocks included).

Scorers (bytes read per key token at D=128, bf16 unless noted):
  quest      relu(q)·kmax + min(q,0)·kmin per block              (8 B/token)
  mean       q · block-mean key (SpargeAttn's block score)       (4 B/token)
  quest+mean average of the quest and mean block softmaxes       (12 B/token)
  lr{r}      block logsumexp of rank-r PCA logits; basis from the prompt keys of each
             KV head (Loki-style), int8 per-token                (r B/token)
  oracle     true group-mean block mass (the ceiling)

Example:
    python scripts/prefilter_recall.py --model Qwen/Qwen2.5-1.5B-Instruct --n 4
"""

import json
import math
from collections import defaultdict

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from ruler_kernel_eval import _prompt_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_attention_hub.kernels.block_sparse_decode import quantize_keys


class RecallProbe(DecodeLoop):
    def __init__(self, *args, fracs, ranks, **kwargs):
        super().__init__(*args, **kwargs)
        self.fracs, self.ranks = fracs, ranks
        self.stats = defaultdict(list)
        # Per-layer, per-KV-head PCA of the prompt keys (centered; q·mu cancels in softmax).
        self.mu, self.U = [], []
        for i in range(len(self.k)):
            kk = self.k[i][:, :, : self.ctx].float()
            mu = kk.mean(2, keepdim=True)
            _, _, vh = torch.linalg.svd(kk - mu, full_matrices=False)
            self.mu.append(mu)
            self.U.append(vh[..., : max(ranks), :].transpose(-1, -2))  # [B, Hkv, D, r]

    def _attend(self, i, q, attn):
        qs = (q.squeeze(2) * attn.scaling).contiguous()
        n = int(self.seq_len)
        B, Hkv, Hq, bs = self.B, self.Hkv, self.Hq, self.bs
        G = Hq // Hkv
        cur_p = (n + bs - 1) // bs
        dev = q.device
        blocks = torch.arange(cur_p, device=dev)
        forced = (blocks < self.sink_blocks) | (blocks >= cur_p - self.local_blocks)
        free = ~forced
        n_free = int(free.sum())
        qg = qs.view(B, Hkv, G, -1).float()
        kk = self.k[i][:, :, :n].float()
        pad = cur_p * bs - n

        def lse_blocks(tok_logits):
            t = F.pad(tok_logits, (0, pad), value=float("-inf"))
            return torch.logsumexp(t.view(B, Hkv, G, cur_p, bs), -1)

        logits = torch.einsum("bhgd,bhnd->bhgn", qg, kk)
        true_mass = (
            F.pad(torch.softmax(logits, -1), (0, pad))
            .view(B, Hkv, G, cur_p, bs)
            .sum(-1)
        )
        kb = F.pad(kk, (0, 0, 0, pad)).view(B, Hkv, cur_p, bs, -1)
        valid_tok = (torch.arange(cur_p * bs, device=dev) < n).view(cur_p, bs, 1)
        kmax = kb.masked_fill(~valid_tok, float("-inf")).amax(3)
        kmin = kb.masked_fill(~valid_tok, float("inf")).amin(3)
        kmean = (kb * valid_tok).sum(3) / valid_tok.sum(1).clamp_min(1)
        scores = {
            "quest": qg.relu() @ kmax.transpose(-1, -2)
            + (qg - qg.relu()) @ kmin.transpose(-1, -2),
            "mean": qg @ kmean.transpose(-1, -2),
        }
        scores["quest+mean"] = torch.log(
            0.5 * torch.softmax(scores["quest"], -1)
            + 0.5 * torch.softmax(scores["mean"], -1)
        )
        kc = kk - self.mu[i]
        for r in self.ranks:
            U = self.U[i][..., :r]
            k8, s8 = quantize_keys(kc @ U)  # [B, Hkv, n, r]
            kr = k8.float() * s8[..., None]
            scores[f"lr{r}"] = lse_blocks(torch.einsum("bhgr,bhnr->bhgn", qg @ U, kr))
        scores["oracle"] = torch.log(true_mass.clamp_min(1e-30))

        for name, s in scores.items():
            s = s.masked_fill(forced, float("-inf"))
            for frac in self.fracs:
                n_keep = math.ceil(n_free * frac)
                shared = torch.softmax(s, -1).mean(2)  # [B, Hkv, P]
                for agg in ("shared", "union"):
                    if agg == "shared":
                        top = torch.zeros_like(shared, dtype=torch.bool).scatter(
                            -1, shared.topk(n_keep, -1).indices, True
                        )
                    else:
                        top = (
                            torch.zeros_like(s, dtype=torch.bool)
                            .scatter(-1, s.topk(n_keep, -1).indices, True)
                            .any(2)
                        )
                    cand = (top & free) | forced
                    rec = (true_mass * cand.unsqueeze(2)).sum(-1)  # [B, Hkv, G]
                    key = f"{name}|{frac}|{agg}"
                    self.stats[key].append(rec.flatten().cpu())
                    self.stats[key + "|grpmin"].append(rec.amin(-1).flatten().cpu())
                    self.stats[key + "|dens"].append(
                        (cand.sum(-1).float() / cur_p).flatten().cpu()
                    )
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
    subsets: str = "niah_multikey_2,niah_multikey_3",
    n: int = 4,
    new_tokens: int = 16,
    fracs=(0.1, 0.25),
    ranks=(8, 16, 32),
    block_size: int = 64,
    out: str = "",
) -> None:
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    fracs, ranks = [float(f) for f in fracs], [int(r) for r in ranks]
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
            loop = RecallProbe(
                m,
                pre.past_key_values,
                ids.shape[1],
                1,
                new_tokens + 1,
                block_size,
                "two_stage",
                0.95,
                0.02,
                0.15,
                math.ceil(128 / block_size),
                math.ceil(128 / block_size),
                (layer_pre, layer_post),
                fracs=fracs,
                ranks=ranks,
            )
            loop.reset(pre.logits[:, -1].argmax(-1))
            for _ in range(new_tokens):
                loop.step()
            for key, vals in loop.stats.items():
                stats[key] += vals
            del loop, pre
            torch.cuda.empty_cache()
    res = {}
    for key in sorted({k.rsplit("|", 1)[0] if k.count("|") == 3 else k for k in stats}):
        rec = torch.cat(stats[key])
        res[key] = dict(
            mean=float(rec.mean()),
            p10=float(rec.quantile(0.1)),
            below_0p9=float((rec < 0.9).float().mean()),
            grpmin=float(torch.cat(stats[key + "|grpmin"]).mean()),
            density=float(torch.cat(stats[key + "|dens"]).mean()),
        )
        r = res[key]
        print(
            f"{key:24s} recall {r['mean']:.3f} p10 {r['p10']:.3f} <0.9 {r['below_0p9']:.3f} "
            f"grpmin {r['grpmin']:.3f} cand-density {r['density']:.3f}",
            flush=True,
        )
    if out:
        json.dump(
            dict(model=model, subsets=subs, results=res), open(out, "w"), indent=2
        )


if __name__ == "__main__":
    fire.Fire(main)
