"""Correctness and latency of the block-sparse decode kernel against dense attention.

  check_tail: quest+mean prefilter block sets and the tail-estimate attention vs torch
         references, and the running block K/V sums of append_kv.
  check: kernel output vs masked dense attention (fp32 reference) for the same selection,
         and the all-blocks kernel vs torch SDPA.
  bench: per-layer decode attention latency (one token, Qwen3-4B shapes: 32 q / 8 kv heads,
         d=128) for SDPA dense, the kernel with every block (dense flash-decoding), block
         selection alone, and selection + sparse attention at fixed densities.

Example:
    python scripts/bench_block_sparse_decode.py --ctx "[32768,65536,131072]" --batch "[1,8]"
"""

import json
import math
from typing import Dict, List

import fire
import torch
import torch.nn.functional as F
from triton.testing import do_bench

from sparse_attention_hub.kernels.block_sparse_decode import (
    append_k8,
    append_kv,
    block_min_max,
    block_sparse_decode,
    block_sparse_decode_fused,
    block_sums,
    dense_block_index,
    quantize_keys,
    select_blocks,
    select_blocks_fused,
    select_blocks_two_stage,
    select_blocks_two_stage_ref,
    two_stage_tail_ref,
)

HQ, HKV, D = 32, 8, 128


def _reference(q, k, v, idx, counts, block_size):
    """Masked dense attention in fp32 for the selected blocks."""
    B, Hq, _ = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    P = (N + block_size - 1) // block_size
    r = torch.arange(idx.shape[-1], device=q.device)
    # scatter_add: padding slots past counts may repeat an index and must not clear it.
    sel = torch.zeros(B, Hkv, P, dtype=torch.int32, device=q.device)
    sel.scatter_add_(-1, idx.long(), (r < counts.unsqueeze(-1)).int())
    sel = sel > 0
    tok = sel.repeat_interleave(block_size, dim=-1)[..., :N].repeat_interleave(
        Hq // Hkv, dim=1
    )
    kk = k.float().repeat_interleave(Hq // Hkv, dim=1)
    vv = v.float().repeat_interleave(Hq // Hkv, dim=1)
    s = torch.einsum("bhd,bhnd->bhn", q.float(), kk).masked_fill(~tok, float("-inf"))
    return torch.einsum("bhn,bhnd->bhd", torch.softmax(s, -1), vv)


def check(block_size: int = 16, n: int = 32768 + 37, seed: int = 0) -> None:
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(2, HQ, D, device=dev, dtype=torch.bfloat16) / math.sqrt(D)
    k = torch.randn(2, HKV, n, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(2, HKV, n, D, device=dev, dtype=torch.bfloat16)
    # Plant a few strongly attended keys so selection is not uniform.
    k[:, :, 1000:1010] += 4 * math.sqrt(D) * q.view(2, HKV, HQ // HKV, D)[:, :, :1]
    kmin, kmax = block_min_max(k, block_size)
    idx, counts = select_blocks(q, kmin, kmax, block_size, 0.9, 0.02, 0.25)
    out = block_sparse_decode(q, k, v, idx, counts, block_size)
    ref = _reference(q, k, v, idx, counts, block_size)
    P = (n + block_size - 1) // block_size
    print(
        f"sparse  max|err|={(out.float() - ref).abs().max():.2e}  density={counts.float().mean() / P:.3f}"
    )
    didx, dcnt = dense_block_index(2, HKV, n, block_size, dev)
    dense = block_sparse_decode(q, k, v, didx, dcnt, block_size)
    sdpa = F.scaled_dot_product_attention(
        q.unsqueeze(2), k, v, scale=1.0, enable_gqa=True
    ).squeeze(2)
    print(f"dense   max|err| vs SDPA={(dense.float() - sdpa.float()).abs().max():.2e}")


def _graphed(fn):
    """Capture fn in a CUDA graph (as decode serving does) and return its replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def _as_sets(idx, cnt):
    return [
        set(idx[b, h, : cnt[b, h]].tolist())
        for b in range(idx.shape[0])
        for h in range(idx.shape[1])
    ]


def check_fused(block_size: int = 64, n: int = 32768 + 37, seed: int = 0) -> None:
    """Fused kernels vs the reference path: same block sets, same outputs, same appends."""
    torch.manual_seed(seed)
    dev = "cuda"
    B = 3
    q = torch.randn(B, HQ, D, device=dev, dtype=torch.bfloat16) / math.sqrt(D)
    k = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
    k[:, :, 5000:5100] += 3 * math.sqrt(D) * q.view(B, HKV, HQ // HKV, D)[:, :, :1]
    pad = 7 * block_size  # unused tail capacity, as in a static cache
    kmin, kmax = block_min_max(
        torch.cat([k, k[:, :, -1:].expand(B, HKV, pad, D)], 2), block_size
    )
    seq = torch.full((1,), n, device=dev, dtype=torch.int32)
    for cdf, floor, cap in ((0.9, 0.02, 0.3), (0.98, 0.0, 0.5), (0.5, 0.05, 0.3)):
        ri, rc = select_blocks(
            q, kmin, kmax, block_size, cdf, floor, cap, 2, 2, seq_len=seq
        )
        fi, fc = select_blocks_fused(
            q, kmin, kmax, block_size, cdf, floor, cap, 2, 2, seq_len=seq
        )
        rs, fs = _as_sets(ri, rc), _as_sets(fi, fc)
        sym = sum(len(a ^ b) for a, b in zip(rs, fs))
        # Captured mass under the reference block probabilities: equal sets up to fp32 ties.
        qg = q.view(B, HKV, HQ // HKV, D).float()
        sc = qg.relu() @ kmax.float().transpose(-1, -2) + (
            qg - qg.relu()
        ) @ kmin.float().transpose(-1, -2)
        valid = (
            torch.arange(kmin.shape[2], device=dev) < (n + block_size - 1) // block_size
        )
        pr = (
            torch.softmax(sc.masked_fill(~valid, float("-inf")), -1)
            .mean(2)
            .flatten(0, 1)
        )

        def mass(sets):
            return max(
                abs(pr[i, list(a)].sum().item() - pr[i, list(b)].sum().item())
                for i, (a, b) in enumerate(sets)
            )

        print(
            f"cdf={cdf} floor={floor}: blocks ref={sum(map(len, rs))} fused={sum(map(len, fs))} "
            f"sym-diff={sym} max|mass diff| per head={mass(list(zip(rs, fs))):.1e}"
        )
        ref = _reference(q, k, v, fi, fc, block_size)
        out = block_sparse_decode_fused(q, k, v, fi, fc, block_size, seq_len=seq)
        print(
            f"   fused attention max|err| vs fp32 ref = {(out.float() - ref).abs().max():.2e}"
        )
    # Append: write token n-1 again via the kernel into a copy whose last block was reset.
    kb, vb = k.clone(), v.clone()
    mn, mx = kmin.clone(), kmax.clone()
    pos = torch.full((1,), n - 1, device=dev, dtype=torch.long)
    pblk = (n - 1) // block_size
    mn[:, :, pblk] = (
        k[:, :, pblk * block_size : n - 1].amin(2) if (n - 1) % block_size else 0
    )
    mx[:, :, pblk] = (
        k[:, :, pblk * block_size : n - 1].amax(2) if (n - 1) % block_size else 0
    )
    kb[:, :, n - 1] = 0
    append_kv(k[:, :, n - 1], v[:, :, n - 1], kb, vb, pos, mn, mx, block_size)
    print(
        f"append: kv exact={torch.equal(kb, k)} meta exact={torch.equal(mn[:, :, pblk], kmin[:, :, pblk]) and torch.equal(mx[:, :, pblk], kmax[:, :, pblk])}"
    )


def check_two_stage(
    block_size: int = 64, n: int = 32768 + 37, seed: int = 0, sink: float = 0.0
) -> None:
    """Two-stage kernels vs the torch reference: block sets, captured mass, attention, int8 append.

    sink > 0 adds sink * (sum of the group's queries) to the first block's keys, so its
    bound logit beats every other block's by ~4 * sink and their fp32 softmax underflows.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    B = 2
    q = torch.randn(B, HQ, D, device=dev, dtype=torch.bfloat16) * (
        2.0 / math.sqrt(D)
    )  # sharper, like real heads
    k = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
    for j, pos in enumerate(
        (3000, 9000, 20000)
    ):  # a few attended spans, one per query head pair
        k[:, :, pos : pos + 40] += (
            2.5 * math.sqrt(D) * q.view(B, HKV, HQ // HKV, D)[:, :, j % 4 : j % 4 + 1]
        )
    k[:, :, :block_size] += sink * q.view(B, HKV, HQ // HKV, D).sum(2, keepdim=True)
    kmin, kmax = block_min_max(k, block_size)
    k8, ks = quantize_keys(k)
    seq = torch.full((1,), n, device=dev, dtype=torch.int32)
    for cdf, floor, cap, pre, sh in (
        (0.9, 0.02, 0.3, 0.25, False),
        (0.95, 0.02, 0.3, 0.25, False),
        (0.98, 0.0, 0.5, 0.5, False),
        (0.9, 0.02, 0.15, 0.25, True),
        (0.95, 0.02, 0.15, 0.15, True),
    ):
        ri, rc = select_blocks_two_stage_ref(
            q,
            kmin,
            kmax,
            k8,
            ks,
            block_size,
            cdf,
            floor,
            cap,
            pre,
            2,
            2,
            seq_len=seq,
            prefilter_shared=sh,
        )
        fi, fc = select_blocks_two_stage(
            q,
            kmin,
            kmax,
            k8,
            ks,
            block_size,
            cdf,
            floor,
            cap,
            pre,
            2,
            2,
            seq_len=seq,
            prefilter_shared=sh,
        )
        rs, fs = _as_sets(ri, rc), _as_sets(fi, fc)
        # Captured attention mass per KV head (true fp32 attention, group-averaged).
        att = torch.softmax(
            torch.einsum(
                "bgd,bgnd->bgn", q.float(), k.float().repeat_interleave(HQ // HKV, 1)
            ),
            -1,
        )
        P = (n + block_size - 1) // block_size
        bmass = (
            torch.nn.functional.pad(att, (0, P * block_size - n))
            .view(B, HQ, P, block_size)
            .sum(-1)
        )
        bmass = bmass.view(B, HKV, HQ // HKV, P).mean(2).flatten(0, 1)

        def cap_mass(sets):
            return sum(
                bmass[i, list(st)].sum().item() for i, st in enumerate(sets)
            ) / len(sets)

        print(
            f"cdf={cdf} pre={pre} shared={sh} cap={cap}: blocks ref={sum(map(len, rs))} fused={sum(map(len, fs))} "
            f"sym-diff={sum(len(a ^ b) for a, b in zip(rs, fs))} density={fc.float().mean().item() * block_size / n:.3f} "
            f"true mass ref={cap_mass(rs):.4f} fused={cap_mass(fs):.4f}"
        )
        out = block_sparse_decode_fused(q, k, v, fi, fc, block_size, seq_len=seq)
        print(
            f"   attention max|err| vs fp32 ref = {(out.float() - _reference(q, k, v, fi, fc, block_size)).abs().max():.2e}"
        )
    k8b, ksb = k8.clone(), ks.clone()
    k8b[:, :, n - 1] = 0
    append_k8(
        k[:, :, n - 1], k8b, ksb, torch.full((1,), n - 1, device=dev, dtype=torch.long)
    )
    print(
        f"append_k8: int8 exact={torch.equal(k8b, k8)} scale max|err|={(ksb - ks).abs().max():.1e}"
    )


def check_tail(block_size: int = 64, n: int = 32768 + 37, seed: int = 0) -> None:
    """Quest+mean prefilter and tail-estimate attention vs the torch references."""
    torch.manual_seed(seed)
    dev = "cuda"
    B, G = 2, HQ // HKV
    q = torch.randn(B, HQ, D, device=dev, dtype=torch.bfloat16) * (2.0 / math.sqrt(D))
    k = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
    k += 0.5 * torch.randn(1, HKV, 1, D, device=dev, dtype=torch.bfloat16)  # key bias
    for j, pos in enumerate((3000, 9000, 20000)):
        k[:, :, pos : pos + 40] += (
            2.5 * math.sqrt(D) * q.view(B, HKV, G, D)[:, :, j % G : j % G + 1]
        )
    kmin, kmax = block_min_max(k, block_size)
    ksum, vsum = block_sums(k, block_size), block_sums(v, block_size)
    k8, ks = quantize_keys(k)
    seq = torch.full((1,), n, device=dev, dtype=torch.int32)
    for cdf, pre, sh, rho, t1, t2 in (
        (0.9, 0.25, True, 0.6, True, True),
        (0.95, 0.15, True, 0.2, True, True),
        (0.9, 0.25, False, 0.6, True, True),
        (0.8, 0.25, True, 1.01, True, False),
        (0.8, 0.25, True, 0.6, False, True),
    ):
        args = (q, kmin, kmax, k8, ks, block_size, cdf, 0.02, 0.15, pre, 2, 2)
        ri, rc, rcand = select_blocks_two_stage_ref(
            *args,
            seq_len=seq,
            prefilter_shared=sh,
            ksum=ksum,
            prefilter_mean=True,
            return_cand=True,
        )
        fi, fc, aux = select_blocks_two_stage(
            *args,
            seq_len=seq,
            prefilter_shared=sh,
            ksum=ksum,
            prefilter_mean=True,
            return_aux=True,
        )
        rs, fs = _as_sets(ri, rc), _as_sets(fi, fc)
        fcand = torch.zeros_like(rcand)
        for bi in range(B):
            for h in range(HKV):
                c = aux["cand"][bi, h, : aux["ncand"][bi, h]].long()
                fcand[bi, h, c] = True
        tail = dict(aux=aux, vsum=vsum, rho=rho, t1=t1, t2=t2)
        out = block_sparse_decode_fused(
            q, k, v, fi, fc, block_size, seq_len=seq, tail=tail
        )
        ref = two_stage_tail_ref(
            q, k, v, k8, ks, ksum, vsum, fi, fc, fcand, block_size, rho, seq, t1, t2
        )
        plain = block_sparse_decode_fused(q, k, v, fi, fc, block_size, seq_len=seq)
        dense = (
            F.scaled_dot_product_attention(
                q.unsqueeze(2), k, v, scale=1.0, enable_gqa=True
            )
            .squeeze(2)
            .float()
        )

        def rel(o):
            return ((o.float() - dense).norm(dim=-1) / dense.norm(dim=-1)).mean().item()

        print(
            f"cdf={cdf} pre={pre} shared={sh} rho={rho} t1={t1} t2={t2}: "
            f"cand sym-diff={int((rcand != fcand).sum())} blocks sym-diff="
            f"{sum(len(a ^ b) for a, b in zip(rs, fs))} | tail max|err| vs ref="
            f"{(out.float() - ref).abs().max():.2e} | rel err vs dense: plain "
            f"{rel(plain):.4f} tail {rel(out):.4f}"
        )
    # Running sums: append the last token of a block into sums built without it.
    p_last = (n - 1) // block_size
    ksb, vsb = block_sums(k[:, :, : n - 1], block_size), block_sums(
        v[:, :, : n - 1], block_size
    )
    if ksb.shape[2] < ksum.shape[2]:
        ksb = torch.nn.functional.pad(ksb, (0, 0, 0, 1))
        vsb = torch.nn.functional.pad(vsb, (0, 0, 0, 1))
    kmn, kmx = kmin.clone(), kmax.clone()
    kb, vb = k.clone(), v.clone()
    append_kv(
        k[:, :, n - 1],
        v[:, :, n - 1],
        kb,
        vb,
        torch.full((1,), n - 1, device=dev, dtype=torch.long),
        kmn,
        kmx,
        block_size,
        ksb,
        vsb,
    )
    print(
        f"append sums: block {p_last} max|dK|={(ksb - ksum).abs().max():.1e} "
        f"max|dV|={(vsb - vsum).abs().max():.1e} kmin/kmax exact="
        f"{torch.equal(kmn, kmin) and torch.equal(kmx, kmax)}"
    )


def bench(
    ctx: List[int] = (32768, 65536, 131072),
    batch: List[int] = (1, 8),
    densities: List[float] = (0.025, 0.05, 0.1, 0.2),
    block_size: int = 16,
    out: str = "",
) -> None:
    dev = "cuda"
    rows: List[Dict] = []
    for B in batch:
        for n in ctx:
            q = torch.randn(B, HQ, D, device=dev, dtype=torch.bfloat16) / math.sqrt(D)
            k = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
            v = torch.randn(B, HKV, n, D, device=dev, dtype=torch.bfloat16)
            kmin, kmax = block_min_max(k, block_size)

            def sdpa():
                return F.scaled_dot_product_attention(
                    q.unsqueeze(2), k, v, scale=1.0, enable_gqa=True
                )

            t_sdpa = do_bench(_graphed(sdpa))
            didx, dcnt = dense_block_index(B, HKV, n, block_size, dev)
            t_dense = do_bench(
                _graphed(lambda: block_sparse_decode(q, k, v, didx, dcnt, block_size))
            )
            row = dict(batch=B, ctx=n, sdpa_ms=t_sdpa, kernel_dense_ms=t_dense)
            for dens in densities:
                # floor = cap = dens fixes the budget, so latency reflects that density.
                def sel(dens=dens):
                    return select_blocks(q, kmin, kmax, block_size, 0.9, dens, dens)

                t_sel = do_bench(_graphed(sel))
                idx, cnt = sel()
                t_att = do_bench(
                    _graphed(lambda: block_sparse_decode(q, k, v, idx, cnt, block_size))
                )
                t_both = do_bench(
                    _graphed(lambda: block_sparse_decode(q, k, v, *sel(), block_size))
                )
                row[f"d{dens}"] = dict(
                    select_ms=t_sel,
                    attend_ms=t_att,
                    step_ms=t_both,
                    speedup_vs_sdpa=t_sdpa / t_both,
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
    if out:
        json.dump(rows, open(out, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(
        {
            "check": check,
            "check_fused": check_fused,
            "check_two_stage": check_two_stage,
            "check_tail": check_tail,
            "bench": bench,
        }
    )
