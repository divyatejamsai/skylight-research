"""Which part of two-stage selection loses attention mass, per query head?

Decodes greedily with DENSE attention (so every variant sees the same queries) and, at
every step and layer, scores several selection variants by the true attention mass
(fp32, bf16 keys) that each QUERY head gets from its KV head's block set. A group mean
hides a retrieval head that one shared selection starves; the per-head minimum and the
fraction of heads below 0.8 expose it.

Variants (all share the Quest-bound prefilter and TopCdf + floor + cap):
  kernel     the Triton select_blocks_two_stage (checks the torch variants match it)
  base       torch: int8 per-token keys, shared prefilter, group-mean probabilities
  union_pre  per-query-head union prefilter
  bf16       exact block mass from bf16 keys (no int8)
  centered   int8 of keys minus their per-channel token mean (softmax-invariant)
  int4       4-bit per-token keys; int4g32: 4-bit with one scale per 32 channels
  union_sel  TopCdf + floor per query head, union over the group
  all        centered + union_pre + union_sel

Example:
    python scripts/group_diagnosis.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --subsets niah_multikey_2,niah_multikey_3 --n 4
"""

import json
import math
from collections import defaultdict
from typing import Dict, List

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from ruler_kernel_eval import _prompt_ids
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_attention_hub.kernels.block_sparse_decode import (
    quantize_keys,
    select_blocks_two_stage,
)

VARIANTS = [
    "kernel",
    "base",
    "union_pre",
    "bf16",
    "centered",
    "int4",
    "int4g32",
    "union_sel",
    "all",
]


def _fake_quant(k: torch.Tensor, bits: int, group: int = 0) -> torch.Tensor:
    """Symmetric per-token (or per-token, per-`group`-channel) fake quantization."""
    qmax = 2 ** (bits - 1) - 1
    x = k.unflatten(-1, (-1, group)) if group else k.unsqueeze(-2)
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (torch.round(x / scale).clamp(-qmax, qmax) * scale).flatten(-2)


def _topcdf(probs, cand, forced, tau, floor_frac, cap):
    """Vectorized TopCdf ∪ top-k floor ∪ forced, truncated to `cap` blocks.

    probs/cand/forced: [..., P]. Same rule as select_blocks_two_stage_ref.
    """
    free = cand & ~forced
    fmass = (probs * forced).sum(-1, keepdim=True)
    p_free = probs.masked_fill(~free, -1.0)
    order = p_free.argsort(-1, descending=True)
    ps = p_free.gather(-1, order).clamp_min(0.0)
    before = ps.cumsum(-1) - ps + fmass
    rank = torch.arange(ps.shape[-1], device=ps.device)
    k_floor = torch.ceil(cand.sum(-1, keepdim=True).float() * floor_frac)
    room = cap - forced.sum(-1, keepdim=True)
    keep = ((before < tau) | (rank < k_floor)) & (rank < room)
    keep &= free.gather(-1, order)
    sel = torch.zeros_like(keep).scatter(-1, order, keep)
    return sel | (forced & cand)


class GroupProbe(DecodeLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.mu = []  # per-layer per-channel key mean over the prompt
        for i in range(len(self.k)):
            self.mu.append(self.k[i][:, :, : self.ctx].float().mean(2, keepdim=True))

    def _attend(self, i: int, q: torch.Tensor, attn) -> torch.Tensor:
        qs = (q.squeeze(2) * attn.scaling).contiguous()
        n = int(self.seq_len)
        B, Hkv, Hq, bs, P = self.B, self.Hkv, self.Hq, self.bs, self.P
        G = Hq // Hkv
        cur_p = (n + bs - 1) // bs
        dev = q.device
        blocks = torch.arange(P, device=dev)
        valid = blocks < cur_p
        forced = valid & (
            (blocks < self.sink_blocks) | (blocks >= cur_p - self.local_blocks)
        )
        free = valid & ~forced
        qg = qs.view(B, Hkv, G, -1).float()

        # True per-query-head block mass.
        kk = self.k[i][:, :, :n].float()
        logits = torch.einsum("bhgd,bhnd->bhgn", qg, kk)
        att = torch.softmax(logits, -1)
        att = F.pad(att, (0, cur_p * bs - n)).view(B, Hkv, G, cur_p, bs).sum(-1)
        true_mass = F.pad(att, (0, P - cur_p))  # [B, Hkv, G, P]

        # Stage 1: Quest bound prefilter (shared over the group, or per-head union).
        bound = qg.relu() @ self.kmax[i].float().transpose(-1, -2) + (
            qg - qg.relu()
        ) @ self.kmin[i].float().transpose(-1, -2)
        bound = bound.masked_fill(~valid, float("-inf"))
        n_keep = math.ceil(int(free.sum()) * self.prefilter)
        shared = torch.softmax(bound, -1).mean(2).masked_fill(~free, -1.0)
        top_s = torch.zeros_like(shared, dtype=torch.bool).scatter(
            -1, shared.topk(n_keep, -1).indices, True
        )
        per = bound.masked_fill(~free, float("-inf"))
        top_u = (
            torch.zeros_like(per, dtype=torch.bool)
            .scatter(-1, per.topk(n_keep, -1).indices, True)
            .any(2)
        )
        cand = {"s": forced | (top_s & free), "u": forced | (top_u & free)}

        # Stage 2 block log-mass per query head from different key copies.
        def block_lmass(keys):
            tl_ = torch.einsum("bhgd,bhnd->bhgn", qg, keys)
            tl_ = F.pad(tl_, (0, cur_p * bs - n), value=float("-inf"))
            lm = torch.logsumexp(tl_.view(B, Hkv, G, cur_p, bs), -1)
            return F.pad(lm, (0, P - cur_p), value=float("-inf"))

        k8, ks = self.k8[i][:, :, :n], self.ks[i][:, :, :n]
        k_int8 = (k8.float() * ks[..., None]).to(qs.dtype).float()
        kc8, kcs = quantize_keys(kk - self.mu[i])
        k_cent = (kc8.float() * kcs[..., None]).to(qs.dtype).float()
        lm = {"int8": block_lmass(k_int8), "bf16": block_lmass(kk), "cent": None}
        lm["cent"] = block_lmass(k_cent)
        lm["int4"] = block_lmass(_fake_quant(kk, 4).to(qs.dtype).float())
        lm["int4g32"] = block_lmass(_fake_quant(kk, 4, 32).to(qs.dtype).float())

        cap = min(
            P,
            min(P, self.sink_blocks + self.local_blocks)
            + max(1, math.ceil(self.max_frac * P)),
        )

        def select(keys, pre, union_sel):
            c = cand[pre]
            pr = torch.softmax(lm[keys].masked_fill(~c.unsqueeze(2), float("-inf")), -1)
            if not union_sel:
                return _topcdf(pr.mean(2), c, forced, self.cdf, self.floor, cap)
            per_head = _topcdf(
                pr, c.unsqueeze(2).expand_as(pr), forced, self.cdf, self.floor, cap
            ).any(2)
            # Over the cap: keep the blocks with the highest per-head probability.
            score = pr.amax(2).masked_fill(~per_head, -1.0).masked_fill(forced, 2.0)
            keep = torch.zeros_like(per_head).scatter(
                -1, score.topk(cap, -1).indices, True
            )
            return per_head & keep

        sels = {
            "base": select("int8", "s", False),
            "union_pre": select("int8", "u", False),
            "bf16": select("bf16", "s", False),
            "centered": select("cent", "s", False),
            "int4": select("int4", "s", False),
            "int4g32": select("int4g32", "s", False),
            "union_sel": select("int8", "s", True),
            "all": select("cent", "u", True),
        }
        idx, cnt = select_blocks_two_stage(
            qs,
            self.kmin[i],
            self.kmax[i],
            self.k8[i],
            self.ks[i],
            bs,
            self.cdf,
            self.floor,
            self.max_frac,
            self.prefilter,
            self.sink_blocks,
            self.local_blocks,
            seq_len=self.seq_len,
        )
        ks_sel = torch.zeros(B, Hkv, P, dtype=torch.int32, device=dev)
        slot = torch.arange(idx.shape[-1], device=dev)
        ks_sel.scatter_add_(-1, idx.long(), (slot < cnt.unsqueeze(-1)).int())
        sels["kernel"] = ks_sel > 0
        self.stats["kernel_vs_base_mismatch"].append(
            (sels["kernel"] != sels["base"]).any(-1).float().mean().view(1)
        )
        for name, sel in sels.items():
            cap_mass = (true_mass * sel.unsqueeze(2)).sum(-1)  # [B, Hkv, G]
            self.stats[name].append(cap_mass.flatten().cpu())
            self.stats[name + "_grpmin"].append(cap_mass.amin(-1).flatten().cpu())
            self.stats[name + "_density"].append(
                (sel.sum(-1).float() * bs / n).flatten().cpu()
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


def _summary(stats):
    out = {}
    for name in VARIANTS:
        m = torch.cat(stats[name])
        g = torch.cat(stats[name + "_grpmin"])
        d = torch.cat(stats[name + "_density"])
        out[name] = dict(
            mean=float(m.mean()),
            p10=float(m.quantile(0.1)),
            p1=float(m.quantile(0.01)),
            below_0p8=float((m < 0.8).float().mean()),
            grpmin_mean=float(g.mean()),
            density=float(d.mean()),
        )
    out["kernel_vs_base_mismatch"] = float(
        torch.cat(stats["kernel_vs_base_mismatch"]).mean()
    )
    return out


def main(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    subsets: str = "niah_multikey_2,niah_multikey_3",
    n: int = 4,
    new_tokens: int = 24,
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
    per_subset = {}
    subs = subsets.split(",") if isinstance(subsets, str) else list(subsets)
    for subset in subs:
        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset).select(
            range(n)
        )
        stats = defaultdict(list)
        for row in rows:
            ids = _prompt_ids(tok, row).cuda()
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            loop = GroupProbe(
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
            )
            loop.prefilter = prefilter
            loop.reset(pre.logits[:, -1].argmax(-1))
            for _ in range(new_tokens):
                loop.step()
            for key, vals in loop.stats.items():
                stats[key] += vals
            del loop, pre
            torch.cuda.empty_cache()
        per_subset[subset] = _summary(stats)
        print(subset, flush=True)
        for name in VARIANTS:
            s = per_subset[subset][name]
            print(
                f"  {name:10s} mean {s['mean']:.3f} p10 {s['p10']:.3f} p1 {s['p1']:.3f} "
                f"<0.8 {s['below_0p8']:.3f} grpmin {s['grpmin_mean']:.3f} "
                f"density {s['density']:.3f}",
                flush=True,
            )
        print(
            "  kernel/base block-set mismatch rate",
            round(per_subset[subset]["kernel_vs_base_mismatch"], 4),
            flush=True,
        )
    if out:
        json.dump(
            dict(model=model, cdf=cdf, prefilter=prefilter, results=per_subset),
            open(out, "w"),
            indent=2,
        )


if __name__ == "__main__":
    fire.Fire(main)
