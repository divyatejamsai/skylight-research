"""How well does a stale block selection serve later decode steps?

Runs the two-stage decode loop (dense prefill, then greedy decode through the real
kernels) and, at every step and layer, measures the current query's true attention
mass (fp32, group-averaged per KV head) captured by
  * the fresh selection of this step, and
  * the selection made `lag` steps earlier, plus this step's local blocks (which a
    reuse scheme would always add),
together with the Jaccard overlap of the two block sets. If stale selections keep
nearly all the mass, selection can be reused across steps to amortize its cost.

Example:
    python scripts/selection_stability.py --lags "[1,2,4,8,16]"
"""

import json
import math
from collections import defaultdict
from typing import Dict, List

import fire
import torch
from datasets import load_dataset
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from ruler_kernel_eval import _prompt_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_attention_hub.kernels.block_sparse_decode import (
    block_sparse_decode_fused,
    select_blocks_two_stage,
)


class StabilityProbe(DecodeLoop):
    """Two-stage DecodeLoop that records selections and scores stale ones."""

    def __init__(self, *args, lags: List[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.lags = lags
        self.history: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.stats: Dict[str, List[float]] = defaultdict(list)

    def _attend(self, i: int, q: torch.Tensor, attn) -> torch.Tensor:
        qs = (q.squeeze(2) * attn.scaling).contiguous()
        idx, cnt = select_blocks_two_stage(
            qs,
            self.kmin[i],
            self.kmax[i],
            self.k8[i],
            self.ks[i],
            self.bs,
            self.cdf,
            self.floor,
            self.max_frac,
            self.prefilter,
            self.sink_blocks,
            self.local_blocks,
            seq_len=self.seq_len,
        )
        n = int(self.seq_len)
        cur_p = (n + self.bs - 1) // self.bs
        B, Hkv, G = self.B, self.Hkv, self.Hq // self.Hkv
        sel = torch.zeros(B, Hkv, self.P, dtype=torch.int32, device=q.device)
        slot = torch.arange(idx.shape[-1], device=q.device)
        sel.scatter_add_(-1, idx.long(), (slot < cnt.unsqueeze(-1)).int())
        sel = sel > 0
        # True attention mass per block (fp32), averaged over each KV head's group.
        k = self.k[i][:, :, :n].float().repeat_interleave(G, dim=1)
        att = torch.softmax(torch.einsum("bhd,bhnd->bhn", qs.float(), k), dim=-1)
        att = torch.nn.functional.pad(att, (0, cur_p * self.bs - n))
        mass = (
            att.view(B, self.Hq, cur_p, self.bs).sum(-1).view(B, Hkv, G, cur_p).mean(2)
        )
        mass = torch.nn.functional.pad(mass, (0, self.P - cur_p))
        local = torch.zeros(self.P, dtype=torch.bool, device=q.device)
        local[max(0, cur_p - self.local_blocks) : cur_p] = True
        self.stats["fresh"] += (mass * sel).sum(-1).flatten().tolist()
        self.stats["fresh_density"].append(
            float(sel.sum(-1).float().mean()) * self.bs / n
        )
        hist = self.history[i]
        for lag in self.lags:
            if len(hist) >= lag:
                stale = hist[-lag] | local
                self.stats[f"stale{lag}"] += (mass * stale).sum(-1).flatten().tolist()
                inter = (stale & sel).sum(-1).float()
                union = (stale | sel).sum(-1).float()
                self.stats[f"jaccard{lag}"] += (inter / union).flatten().tolist()
        hist.append(sel)
        if len(hist) > max(self.lags):
            hist.pop(0)
        return block_sparse_decode_fused(
            qs, self.k[i], self.v[i], idx, cnt, self.bs, seq_len=self.seq_len
        )


def _summary(
    stats: Dict[str, List[float]], lags: List[int]
) -> Dict[str, Dict[str, float]]:
    def describe(xs: List[float]) -> Dict[str, float]:
        t = torch.tensor(xs)
        return dict(
            mean=float(t.mean()),
            p10=float(t.quantile(0.1)),
            p1=float(t.quantile(0.01)),
            n=len(xs),
        )

    out = {
        "fresh": describe(stats["fresh"]),
        "fresh_density": dict(
            mean=sum(stats["fresh_density"]) / len(stats["fresh_density"])
        ),
    }
    for lag in lags:
        if stats.get(f"stale{lag}"):
            out[f"stale{lag}"] = describe(stats[f"stale{lag}"])
            out[f"jaccard{lag}"] = describe(stats[f"jaccard{lag}"])
    return out


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    subsets: str = "niah_multikey_3,cwe,qa_1,vt",
    n: int = 3,
    new_tokens: int = 48,
    lags: List[int] = (1, 2, 4, 8, 16),
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
    per_subset = {}
    for subset in subsets.split(",") if isinstance(subsets, str) else list(subsets):
        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset).select(
            range(n)
        )
        stats: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            ids = _prompt_ids(tok, row).cuda()
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            loop = StabilityProbe(
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
            for _ in range(new_tokens):  # fixed length: stability, not answer quality
                loop.step()
            for key, vals in loop.stats.items():
                stats[key] += vals
            del loop, pre
            torch.cuda.empty_cache()
        per_subset[subset] = _summary(stats, lags)
        print(
            subset,
            json.dumps({k: round(v["mean"], 4) for k, v in per_subset[subset].items()}),
            flush=True,
        )
    print(json.dumps(per_subset, indent=2))
    if out:
        json.dump(
            dict(
                model=model,
                cdf=cdf,
                prefilter=prefilter,
                block_size=block_size,
                results=per_subset,
            ),
            open(out, "w"),
            indent=2,
        )


if __name__ == "__main__":
    fire.Fire(main)
