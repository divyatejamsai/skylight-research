"""CUDA-graph timings of the fused decode kernels across launch parameters.

Builds Qwen3-4B-shaped inputs (32 query / 8 KV heads, d=128) with a few planted attended
spans, runs the quest+mean two-stage selection once, then times selection and the
block-sparse attention (with and without the tail) for each setting in kernels.TUNE.

Example:
    python scripts/tune_decode_kernels.py --ctx 32768 --batch 8
"""

import itertools
import math

import fire
import torch
from triton.testing import do_bench_cudagraph

import sparse_attention_hub.kernels.block_sparse_decode as K


def _inputs(B, HQ, HKV, D, n, bs, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16) * (
        2.0 / math.sqrt(D)
    )
    k = torch.randn(B, HKV, n, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, HKV, n, D, device="cuda", dtype=torch.bfloat16)
    G = HQ // HKV
    for j, pos in enumerate((3000, 9000, 20000, n - 5000)):
        k[:, :, pos : pos + 40] += (
            2.5 * math.sqrt(D) * q.view(B, HKV, G, D)[:, :, j % G : j % G + 1]
        )
    kmin, kmax = K.block_min_max(k, bs)
    k8, ks = K.quantize_keys(k)
    return q, k, v, kmin, kmax, k8, ks, K.block_sums(k, bs), K.block_sums(v, bs)


def _breakdown(*fns, reps=50):
    """Per-kernel GPU time (us per call) of each fn, from the torch profiler."""
    for fn in fns:
        g = torch.cuda.CUDAGraph()
        fn()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                g.replay()
            torch.cuda.synchronize()
        tot = 0.0
        for e in sorted(prof.key_averages(), key=lambda e: -e.device_time_total):
            if e.device_time_total <= 0:
                continue
            t = e.device_time_total / reps
            tot += t
            print(f"  {e.key[:60]:60s} {t:7.1f} us  x{e.count // reps}")
        print(f"  {'total kernel time':60s} {tot:7.1f} us")


def main(
    ctx: int = 32768,
    batch: int = 8,
    hq: int = 32,
    hkv: int = 8,
    bs: int = 64,
    profile: bool = False,
):
    q, k, v, kmin, kmax, k8, ks, ksum, vsum = _inputs(batch, hq, hkv, 128, ctx, bs)
    seq = torch.full((1,), ctx, device="cuda", dtype=torch.int32)
    args = (q, kmin, kmax, k8, ks, bs, 0.95, 0.02, 0.15, 0.25, 2, 2)

    def select():
        return K.select_blocks_two_stage(
            *args, seq_len=seq, ksum=ksum, prefilter_mean=True, return_aux=True
        )

    idx, cnt, aux = select()
    print(
        f"ctx {ctx} batch {batch} heads {hq}/{hkv}: mean selected blocks "
        f"{cnt.float().mean():.1f} of cap {idx.shape[-1]}, candidates "
        f"{aux['ncand'].float().mean():.1f}"
    )
    ref = K.block_sparse_decode_fused(q, k, v, idx, cnt, bs, seq_len=seq)
    tail = dict(aux=aux, vsum=vsum, rho=0.6)
    if profile:
        _breakdown(
            select,
            lambda: K.block_sparse_decode_fused(
                q, k, v, idx, cnt, bs, seq_len=seq, tail=tail
            ),
        )
        return
    base = dict(K.TUNE)
    for slots, warps in itertools.product((2, 4, 8), (2, 4, 8)):
        K.TUNE.update(base, mass_slots=slots, mass_warps=warps)
        t = do_bench_cudagraph(select)
        print(f"select       mass_slots={slots} mass_warps={warps}: {1000 * t:7.1f} us")
    K.TUNE.update(base)
    for radix, pw, fw in itertools.product((2, 4, 8, 16), (2, 4, 8), (2, 4, 8)):
        K.TUNE.update(base, search_radix=radix, prefilter_warps=pw, final_warps=fw)
        i2, c2, _ = select()
        same = bool((c2 == cnt).all() and (i2 == idx).all())
        t = do_bench_cudagraph(select)
        print(
            f"select       radix={radix:2d} prefilter_warps={pw} final_warps={fw}: "
            f"{1000 * t:7.1f} us  same blocks {same}"
        )
    K.TUNE.update(base)
    for bps, warps, stages in itertools.product(
        (None, 2, 4, 8, 16), (2, 4, 8), (2, 3, 4)
    ):
        K.TUNE.update(base, decode_blocks_per_split=bps, decode_warps=warps)
        K.TUNE["decode_stages"] = stages
        out = K.block_sparse_decode_fused(q, k, v, idx, cnt, bs, seq_len=seq)
        err = (out.float() - ref.float()).abs().max().item()
        t = do_bench_cudagraph(
            lambda: K.block_sparse_decode_fused(q, k, v, idx, cnt, bs, seq_len=seq)
        )
        tt = do_bench_cudagraph(
            lambda: K.block_sparse_decode_fused(
                q, k, v, idx, cnt, bs, seq_len=seq, tail=tail
            )
        )
        print(
            f"attention    blocks/split={str(bps):4s} warps={warps} stages={stages}: "
            f"{1000 * t:6.1f} us, with tail {1000 * tt:6.1f} us (max|diff| {err:.1e})"
        )
    K.TUNE.update(base)


if __name__ == "__main__":
    fire.Fire(main)
