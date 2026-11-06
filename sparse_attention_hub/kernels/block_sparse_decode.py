"""Block-sparse decode attention: select KV blocks per KV head, then attend only to them.

Selection (``select_blocks``) combines:
  * Quest's per-channel key range as the block score, computed as two GEMMs:
    sum_d max(q_d * min_d, q_d * max_d) = relu(q) . kmax + min(q, 0) . kmin;
  * SpargeAttn's TopCdf budget on the softmax of those scores (adaptive per head),
    with SpargeAttention2's top-k floor so no head is starved;
  * one selection per KV head (probabilities averaged over its query group), so a
    GQA group reads each selected block once;
  * sink and local blocks always kept.

Attention (``block_sparse_decode``) is a split-K flash-decoding Triton kernel that reads
only the first ``counts[b, h]`` entries of ``block_idx[b, h]``; partial results are merged
with a log-sum-exp combine. With every block selected it is a dense flash-decoding kernel,
which is the apples-to-apples baseline for speed.
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Launch parameters of the fused decode path (defaults from scripts/tune_decode_kernels.py
# and e2e timings on an H200, Qwen3-4B shapes). decode_blocks_per_split: selected blocks
# per split-K program; small splits keep many programs busy when the selection is far
# below its cap (None: size splits from the cap and the SM count). search_radix: probes
# per round of the TopCdf / prefilter threshold searches (2 = plain bisection);
# sort_max_blocks: up to this many (padded) blocks a kernel finds its threshold with one
# exact sort instead. prefilter_warps / final_warps: the most warps those one-program-per-
# KV-head kernels get; they use one warp per 64 padded blocks (at least 4) up to it.
# prefilter_chunk: with tile stats, the shared prefilter walks the live blocks in pieces
# this wide (0: one tile padded to the capacity, as before); prefilter_chunk_warps: its
# warps; prefilter_digit: bits per radix-select round; prefilter_region: if > 0, each
# run of this many blocks keeps its own share of the candidates, one program per run,
# instead of the global top (0); prefilter_region_alpha: weight of a region's block count
# in its share, the rest following its share of the ranking probability. final_split: the final selection
# keeps forced candidates apart and sorts only the free ones (False: one padded tile).
# mass_slots / mass_warps / mass_stages: candidate blocks per block-mass program, its
# warps, and its software-pipelining depth.
TUNE = dict(
    decode_blocks_per_split=4,
    decode_warps=4,
    decode_stages=2,
    mass_slots=2,
    mass_warps=2,
    mass_stages=2,
    search_radix=4,
    sort_max_blocks=1024,
    prefilter_stats=True,
    prefilter_warps=16,
    final_warps=16,
    prefilter_chunk=512,
    prefilter_chunk_warps=8,
    prefilter_digit=4,
    prefilter_region=512,
    prefilter_region_alpha=0.5,
    final_split=True,
)


def _rounds(bisection_steps: int, radix: int) -> int:
    """Rounds of a radix-way threshold search that resolve as finely as the bisection."""
    return math.ceil(bisection_steps / math.log2(radix))


def _warps(most: int, padded_blocks: int) -> int:
    """One warp per 64 padded blocks, at least 4 and at most `most`."""
    return max(4, min(most, padded_blocks // 64))


def block_sums(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block fp32 sums of x [B, H, N, D] over tokens -> [B, H, ceil(N / bs), D]."""
    B, H, N, D = x.shape
    pad = (-N) % block_size
    xp = torch.nn.functional.pad(x.float(), (0, 0, 0, pad))
    return xp.view(B, H, -1, block_size, D).sum(3)


def block_min_max(
    keys: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block channel min and max of ``keys`` [B, Hkv, N, D] -> two [B, Hkv, P, D] tensors."""
    B, H, N, D = keys.shape
    P = (N + block_size - 1) // block_size
    pad = P * block_size - N
    k = keys
    if pad:
        # Repeat the last key so padding changes neither min nor max.
        k = torch.cat([k, k[:, :, -1:].expand(B, H, pad, D)], dim=2)
    k = k.view(B, H, P, block_size, D)
    return k.amin(dim=3), k.amax(dim=3)


@torch.no_grad()
def select_blocks(
    q: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    block_size: int,
    cdf_threshold: float,
    floor_fraction: float,
    max_fraction: float,
    sink_blocks: int = 8,
    local_blocks: int = 8,
    seq_len: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose KV blocks for one decode step.

    q: [B, Hq, D] (already scaled by softmax scale); kmin/kmax: [B, Hkv, P, D], where only
    the first ceil(seq_len / block_size) blocks are valid (all of them if seq_len is None).
    Returns block_idx [B, Hkv, Kmax] (int32, selected blocks first) and counts [B, Hkv].
    Everything stays on the GPU (no host sync), so a decode step can be a CUDA graph.
    """
    B, Hq, D = q.shape
    Hkv, P = kmin.shape[1], kmin.shape[2]
    G = Hq // Hkv
    blocks = torch.arange(P, device=q.device)
    if seq_len is None:
        cur_p = torch.full((), P, device=q.device, dtype=torch.int32)
    else:
        cur_p = (seq_len.reshape(()) + block_size - 1) // block_size
    valid = blocks < cur_p

    qg = q.view(B, Hkv, G, D).float()
    pos = torch.relu(qg)
    scores = pos @ kmax.float().transpose(-1, -2) + (qg - pos) @ kmin.float().transpose(
        -1, -2
    )
    scores = scores.masked_fill(~valid, float("-inf"))
    probs = torch.softmax(scores, dim=-1).mean(dim=2)  # [B, Hkv, P]

    forced = valid & ((blocks < sink_blocks) | (blocks >= cur_p - local_blocks))
    n_forced = min(P, sink_blocks + local_blocks)
    kmax_sel = min(P, n_forced + max(1, math.ceil(max_fraction * P)))
    floor_k = n_forced + torch.ceil(cur_p * floor_fraction)

    # Forced blocks rank first; their mass counts toward the threshold.
    ranked = probs + forced.float() * 2.0
    vals, idx = torch.topk(ranked, kmax_sel, dim=-1)
    mass = torch.where(vals > 1.0, vals - 2.0, vals)
    cdf = torch.cumsum(mass, dim=-1)
    rank = torch.arange(kmax_sel, device=q.device)
    keep = (((cdf - mass) < cdf_threshold) | (rank < floor_k)) & (vals > 0)  # a prefix
    counts = keep.sum(dim=-1).to(torch.int32)
    return idx.to(torch.int32), counts


@triton.jit
def _sparse_decode_kernel(
    Q,
    K,
    V,
    IDX,
    CNT,
    NPTR,
    O_ACC,
    O_M,
    O_L,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ib,
    stride_ih,
    stride_ab,
    stride_ah,
    stride_as,
    stride_mb,
    stride_mh,
    stride_ms,
    HKV,
    BLOCKS_PER_SPLIT,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    BPI: tl.constexpr,
):
    # Each iteration gathers BPI selected blocks into one [BPI * BLOCK_N, D] tile.
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = (pid_bh // HKV).to(
        tl.int64
    )  # b * stride_kb exceeds int32 at long context x large batch
    h = pid_bh % HKV

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, BPI * BLOCK_N)
    slot = offs_t // BLOCK_N
    within = offs_t % BLOCK_N
    offs_d = tl.arange(0, D)
    row_ok = offs_m < G
    q = tl.load(
        Q + b * stride_qb + (h * G + offs_m)[:, None] * stride_qh + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    N = tl.load(NPTR)
    cnt = tl.load(CNT + b * HKV + h)
    start = pid_s * BLOCKS_PER_SPLIT
    end = tl.minimum(start + BLOCKS_PER_SPLIT, cnt)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    for j in range(start, end, BPI):
        sel_ok = (j + slot) < end
        blk = tl.load(
            IDX + b * stride_ib + h * stride_ih + j + slot, mask=sel_ok, other=0
        )
        tok = blk * BLOCK_N + within
        tok_ok = sel_ok & (tok < N)
        k = tl.load(
            k_base + tok[:, None] * stride_kn + offs_d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k))
        s = tl.where(tok_ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A fully masked tile (e.g. a look-ahead block not yet written) keeps m at -inf;
        # shift by 0 then, since exp(-inf - -inf) would be NaN.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            v_base + tok[:, None] * stride_vn + offs_d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out_rows = h * G + offs_m
    tl.store(
        O_ACC
        + b * stride_ab
        + out_rows[:, None] * stride_ah
        + pid_s * stride_as
        + offs_d[None, :],
        acc,
        mask=row_ok[:, None],
    )
    tl.store(
        O_M + b * stride_mb + out_rows * stride_mh + pid_s * stride_ms, m_i, mask=row_ok
    )
    tl.store(
        O_L + b * stride_mb + out_rows * stride_mh + pid_s * stride_ms, l_i, mask=row_ok
    )


def block_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    counts: torch.Tensor,
    block_size: int,
    num_splits: Optional[int] = None,
    seq_len: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention of one query token per head over the selected key blocks.

    q: [B, Hq, D] (already scaled); k, v: [B, Hkv, N, D] with unit stride on D;
    block_idx: [B, Hkv, Kmax] int32; counts: [B, Hkv] int32; seq_len: int32 device scalar
    with the number of valid keys (default N). Returns [B, Hq, D] in q.dtype.
    """
    B, Hq, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    G = Hq // Hkv
    kmax_sel = block_idx.shape[-1]
    if num_splits is None:
        # Enough programs to fill the GPU at small batch; each split takes >= 4 blocks.
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = max(
            1, min(triton.cdiv(kmax_sel, 4), triton.cdiv(2 * sms, B * Hkv))
        )
    bpi = max(1, 64 // block_size)
    per_split = triton.cdiv(triton.cdiv(kmax_sel, num_splits), bpi) * bpi
    num_splits = triton.cdiv(kmax_sel, per_split)
    acc = torch.empty(B, Hq, num_splits, D, device=q.device, dtype=torch.float32)
    m = torch.empty(B, Hq, num_splits, device=q.device, dtype=torch.float32)
    l_sum = torch.empty_like(m)
    if seq_len is None:
        seq_len = torch.full((1,), N, device=q.device, dtype=torch.int32)
    _sparse_decode_kernel[(B * Hkv, num_splits)](
        q,
        k,
        v,
        block_idx,
        counts,
        seq_len,
        acc,
        m,
        l_sum,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        block_idx.stride(0),
        block_idx.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        Hkv,
        per_split,
        G=G,
        BLOCK_M=max(16, triton.next_power_of_2(G)),
        BLOCK_N=block_size,
        D=D,
        BPI=bpi,
        num_warps=4,
        num_stages=3,
    )
    # Log-sum-exp merge of the splits; empty splits have m = -inf and l_sum = 0.
    m_max = m.amax(dim=-1, keepdim=True)
    w = torch.exp(m - m_max)
    out = (acc * w.unsqueeze(-1)).sum(dim=2) / (l_sum * w).sum(dim=-1, keepdim=True)
    return out.to(q.dtype)


def dense_block_index(
    B: int, Hkv: int, N: int, block_size: int, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every block selected: turns block_sparse_decode into dense flash-decoding."""
    P = (N + block_size - 1) // block_size
    idx = (
        torch.arange(P, device=device, dtype=torch.int32).expand(B, Hkv, P).contiguous()
    )
    return idx, torch.full((B, Hkv), P, device=device, dtype=torch.int32)


# ---------------------------------------------------------------------------------------
# Fused path: the selection, the KV/metadata append and the split merge as Triton kernels.
# select_blocks above is the readable reference; select_blocks_fused returns the same set
# of blocks (in a different order: forced blocks first, then by block index).
# ---------------------------------------------------------------------------------------


@triton.jit
def _logaddexp(a, b):
    m = tl.maximum(a, b)
    safe = tl.where(m == float("-inf"), 0.0, m)
    return tl.where(
        m == float("-inf"),
        float("-inf"),
        safe + tl.log(tl.exp(a - safe) + tl.exp(b - safe)),
    )


@triton.jit
def _search_round(lp, w, mask, lo, hi, target, R: tl.constexpr, AT_MOST: tl.constexpr):
    """One R-ary step of a threshold search on f(t) = sum(w over mask & lp >= t).

    Probes R evenly spaced thresholds in [lo, hi) at once (one 2-D reduction), so a round
    equals log2(R) bisection steps on the same grid. AT_MOST=False keeps lo at the largest
    t with f(t) >= target; AT_MOST=True keeps hi at the smallest t with f(t) <= target.
    """
    t = lo + (hi - lo) * (tl.arange(0, R).to(tl.float32) / R)
    f = tl.sum(
        tl.where(mask[None, :] & (lp[None, :] >= t[:, None]), w[None, :], 0.0), 1
    )
    if AT_MOST:
        ok = f <= target
        return tl.max(tl.where(ok, lo, t), 0), tl.min(tl.where(ok, t, hi), 0)
    ok = f >= target
    return tl.max(tl.where(ok, t, lo), 0), tl.min(tl.where(ok, hi, t), 0)


@triton.jit
def _kth_largest(x, k):
    """The k-th largest entry (0-based) of a 1-D block, from one bitonic sort."""
    xs = tl.sort(x, descending=True)
    return tl.sum(tl.where(tl.arange(0, x.shape[0]) == k, xs, 0.0), axis=0)


@triton.jit
def _head_log_softmax(S, S2, off, offs_p, valid, MEAN: tl.constexpr):
    """One query head's block log-softmax (blended with the mean-key one if MEAN)."""
    s = tl.load(S + off + offs_p, mask=valid, other=float("-inf"))
    mx = tl.max(s, axis=0)
    lpr = s - mx - tl.log(tl.sum(tl.exp(s - mx), axis=0))  # -inf if invalid
    if MEAN:
        s2 = tl.load(S2 + off + offs_p, mask=valid, other=float("-inf"))
        mx2 = tl.max(s2, axis=0)
        lpr2 = s2 - mx2 - tl.log(tl.sum(tl.exp(s2 - mx2), axis=0))
        lpr = _logaddexp(lpr, lpr2) - 0.6931471805599453
    return lpr


@triton.jit
def _top_log(
    lp, free, ones, n_keep, ITERS: tl.constexpr, RADIX: tl.constexpr, SORT: tl.constexpr
):
    """Free blocks at or above the n_keep-th largest log score among free blocks."""
    if SORT:
        lo = _kth_largest(tl.where(free, lp, -1e30), n_keep.to(tl.int32) - 1)
    else:
        hi = tl.max(tl.where(free, lp, -1e30), axis=0)
        lo = hi - 1e4
        for _ in range(ITERS):
            lo, hi = _search_round(lp, ones, free, lo, hi, n_keep, RADIX, False)
    return free & (lp >= lo)


@triton.jit
def _top_linear(
    pr, free, ones, n_keep, ITERS: tl.constexpr, RADIX: tl.constexpr, SORT: tl.constexpr
):
    """_top_log on probabilities (zero outside free); needs n_keep nonzero entries."""
    if SORT:
        lo = _kth_largest(pr, n_keep.to(tl.int32) - 1)
        top = free & (pr >= lo) & (pr > 0.0)
    else:
        top = _top_log(
            tl.where(pr > 0.0, tl.log(pr), float("-inf")),
            free,
            ones,
            n_keep,
            ITERS,
            RADIX,
            SORT,
        )
    return top


@triton.jit
def _block_scores_kernel(
    Q,
    KMIN,
    KMAX,
    S,
    E1,
    KSUM,
    S2,
    E2,
    ST,
    NPTR,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kp,
    stride_sb,
    stride_sh,
    HKV,
    P,
    BS,
    NT,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    MEAN: tl.constexpr = False,
    STATS: tl.constexpr = False,
    STORE_E: tl.constexpr = True,
):
    # STATS: per query head and tile of BLOCK_P blocks, also store exp(s - tile max) in E1
    # (exp(s2 - tile max) in E2) plus the tile max and sum in ST [B, Hq, NT, 4], so the
    # prefilter gets each block's softmax probability with no per-block exp/log of its
    # own: it runs one program per KV head, this kernel one per tile. STORE_E=False keeps
    # only the tile max and sum (the live-length prefilter needs just the normalizers).
    pid_bh = tl.program_id(0)
    b = (pid_bh // HKV).to(tl.int64)
    h = pid_bh % HKV
    offs_m = tl.arange(0, BLOCK_M)
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < G
    q = tl.load(
        Q + b * stride_qb + (h * G + offs_m)[:, None] * stride_qh + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    cur_p = (tl.load(NPTR) + BS - 1) // BS
    p_ok = offs_p < cur_p
    kp = b * stride_kb + h * stride_kh + offs_p[:, None] * stride_kp + offs_d[None, :]
    kmn = tl.load(KMIN + kp, mask=p_ok[:, None], other=0.0)
    kmx = tl.load(KMAX + kp, mask=p_ok[:, None], other=0.0)
    q_pos = tl.maximum(q, 0.0).to(kmx.dtype)
    q_neg = tl.minimum(q, 0.0).to(kmx.dtype)
    s = tl.dot(q_pos, tl.trans(kmx)) + tl.dot(q_neg, tl.trans(kmn))
    s = tl.where(p_ok[None, :], s, float("-inf"))
    st = ST + ((b * HKV * G + h * G + offs_m) * NT + tl.program_id(1)) * 4
    out = b * stride_sb + (h * G + offs_m)[:, None] * stride_sh + offs_p[None, :]
    out_ok = row_ok[:, None] & (offs_p[None, :] < P)
    tl.store(S + out, s, mask=out_ok)
    if STATS:
        m1 = tl.max(s, axis=1)
        e1 = tl.exp(s - tl.where(m1 == float("-inf"), 0.0, m1)[:, None])
        tl.store(st, m1, mask=row_ok)
        tl.store(st + 1, tl.sum(e1, axis=1), mask=row_ok)
        if STORE_E:
            tl.store(E1 + out, e1, mask=out_ok)
    if MEAN:
        # q . block-mean key (SpargeAttn's block score); the last block may be partial.
        n_tok = tl.minimum(BS, tl.load(NPTR) - offs_p * BS).to(tl.float32)
        ksm = tl.load(KSUM + kp, mask=p_ok[:, None], other=0.0)
        kmean = (ksm / tl.maximum(n_tok, 1.0)[:, None]).to(kmx.dtype)
        s2 = tl.dot(q.to(kmx.dtype), tl.trans(kmean))
        s2 = tl.where(p_ok[None, :], s2, float("-inf"))
        tl.store(
            S2
            + b * stride_sb
            + (h * G + offs_m)[:, None] * stride_sh
            + offs_p[None, :],
            s2,
            mask=row_ok[:, None] & (offs_p[None, :] < P),
        )
        if STATS:
            m2 = tl.max(s2, axis=1)
            e2 = tl.exp(s2 - tl.where(m2 == float("-inf"), 0.0, m2)[:, None])
            tl.store(st + 2, m2, mask=row_ok)
            tl.store(st + 3, tl.sum(e2, axis=1), mask=row_ok)
            if STORE_E:
                tl.store(
                    E2
                    + b * stride_sb
                    + (h * G + offs_m)[:, None] * stride_sh
                    + offs_p[None, :],
                    e2,
                    mask=row_ok[:, None] & (offs_p[None, :] < P),
                )


@triton.jit
def _select_kernel(
    S,
    IDX,
    CNT,
    NPTR,
    stride_sb,
    stride_sh,
    stride_ib,
    stride_ih,
    HKV,
    P,
    BS,
    CAP,
    SINK,
    LOCAL,
    tau,
    floor_frac,
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = (pid_bh // HKV).to(tl.int64)
    h = pid_bh % HKV
    offs_p = tl.arange(0, BLOCK_P)
    cur_p = (tl.load(NPTR) + BS - 1) // BS
    valid = offs_p < cur_p
    # Group-mean of the per-query-head softmax over blocks.
    lmean = tl.full([BLOCK_P], float("-inf"), tl.float32)
    for g in range(G):
        s = tl.load(
            S + b * stride_sb + (h * G + g) * stride_sh + offs_p,
            mask=valid,
            other=float("-inf"),
        )
        mx = tl.max(s, axis=0)
        lmean = _logaddexp(lmean, s - mx - tl.log(tl.sum(tl.exp(s - mx), axis=0)))
    lp_all = lmean - tl.log(G * 1.0)  # log of the group-mean softmax, no underflow
    probs = tl.exp(lp_all)
    forced = valid & ((offs_p < SINK) | (offs_p >= cur_p - LOCAL))
    free = valid & (~forced)
    forced_mass = tl.sum(tl.where(forced, probs, 0.0), axis=0)
    k_floor = tl.ceil(cur_p.to(tl.float32) * floor_frac)
    # Largest thresholds t with mass(probs >= t) + forced mass >= tau, and with
    # count(probs >= t) >= k_floor; keeping probs >= min(t_cdf, t_k) is TopCdf union top-k.
    # Thresholds are searched in log-probability: tail blocks of a peaked head sit far
    # below any fixed absolute resolution, so a search over [0, 1] over-selects them.
    lp = tl.where(lp_all > -1e30, lp_all, -1e30)
    # Bracket [best - 60, best]: blocks e^60 below the best free block carry no mass.
    top = tl.max(tl.where(free, lp, -1e30), axis=0)
    lo_c = top - 60.0
    hi_c = top
    lo_k = top - 60.0
    hi_k = top
    # Smallest threshold whose free-block count fits the buffer: truncation by probability.
    n_forced = tl.sum(forced.to(tl.int32), axis=0)
    cap_free = (CAP - n_forced).to(tl.float32)
    lo_x = top - 60.0
    hi_x = top
    for _ in range(ITERS):
        mid_c = (lo_c + hi_c) * 0.5
        ok_c = (
            forced_mass + tl.sum(tl.where(free & (lp >= mid_c), probs, 0.0), axis=0)
            >= tau
        )
        lo_c = tl.where(ok_c, mid_c, lo_c)
        hi_c = tl.where(ok_c, hi_c, mid_c)
        mid_k = (lo_k + hi_k) * 0.5
        ok_k = tl.sum((free & (lp >= mid_k)).to(tl.float32), axis=0) >= k_floor
        lo_k = tl.where(ok_k, mid_k, lo_k)
        hi_k = tl.where(ok_k, hi_k, mid_k)
        mid_x = (lo_x + hi_x) * 0.5
        ok_x = tl.sum((free & (lp >= mid_x)).to(tl.float32), axis=0) <= cap_free
        hi_x = tl.where(ok_x, mid_x, hi_x)
        lo_x = tl.where(ok_x, lo_x, mid_x)
    t = tl.maximum(tl.minimum(lo_c, lo_k), hi_x)
    chosen = free & (lp >= t) & (lp > -1e30)
    # Compact: forced blocks first, then chosen ones in block order (CAP guards exact ties).
    slot = tl.where(
        forced,
        tl.cumsum(forced.to(tl.int32), axis=0) - 1,
        n_forced + tl.cumsum(chosen.to(tl.int32), axis=0) - 1,
    )
    write = (forced | chosen) & (slot < CAP)
    tl.store(
        IDX + b * stride_ib + h * stride_ih + slot, offs_p.to(tl.int32), mask=write
    )
    total = n_forced + tl.sum(chosen.to(tl.int32), axis=0)
    tl.store(CNT + b * HKV + h, tl.minimum(total, CAP))


@torch.no_grad()
def select_blocks_fused(
    q: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    block_size: int,
    cdf_threshold: float,
    floor_fraction: float,
    max_fraction: float,
    sink_blocks: int = 8,
    local_blocks: int = 8,
    seq_len: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same contract and block set as select_blocks, as two Triton kernels with no sort."""
    B, Hq, D = q.shape
    Hkv, P = kmin.shape[1], kmin.shape[2]
    G = Hq // Hkv
    if seq_len is None:
        seq_len = torch.full((1,), P * block_size, device=q.device, dtype=torch.int32)
    scores = torch.empty(B, Hq, P, device=q.device, dtype=torch.float32)
    bp = 64
    _block_scores_kernel[(B * Hkv, triton.cdiv(P, bp))](
        q,
        kmin,
        kmax,
        scores,
        scores,
        kmin,
        scores,
        scores,
        scores,
        seq_len,
        q.stride(0),
        q.stride(1),
        kmin.stride(0),
        kmin.stride(1),
        kmin.stride(2),
        scores.stride(0),
        scores.stride(1),
        Hkv,
        P,
        block_size,
        triton.cdiv(P, bp),
        G=G,
        BLOCK_M=max(16, triton.next_power_of_2(G)),
        BLOCK_P=bp,
        D=D,
    )
    cap = min(
        P, min(P, sink_blocks + local_blocks) + max(1, math.ceil(max_fraction * P))
    )
    idx = torch.zeros(B, Hkv, cap, device=q.device, dtype=torch.int32)
    counts = torch.empty(B, Hkv, device=q.device, dtype=torch.int32)
    _select_kernel[(B * Hkv,)](
        scores,
        idx,
        counts,
        seq_len,
        scores.stride(0),
        scores.stride(1),
        idx.stride(0),
        idx.stride(1),
        Hkv,
        P,
        block_size,
        cap,
        sink_blocks,
        local_blocks,
        cdf_threshold,
        floor_fraction,
        G=G,
        BLOCK_P=triton.next_power_of_2(P),
        ITERS=20,
        num_warps=8,
    )
    return idx, counts


@triton.jit
def _append_kernel(
    KNEW,
    VNEW,
    KBUF,
    VBUF,
    KMIN,
    KMAX,
    KSUM,
    VSUM,
    POS,
    stride_nb,
    stride_nh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_mb,
    stride_mh,
    stride_mp,
    HKV,
    BS,
    D: tl.constexpr,
    META: tl.constexpr,
    SUMS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    offs_d = tl.arange(0, D)
    pos = tl.load(POS)
    k = tl.load(KNEW + b * stride_nb + h * stride_nh + offs_d)
    v = tl.load(VNEW + b * stride_nb + h * stride_nh + offs_d)
    kv = b * stride_kb + h * stride_kh + pos * stride_kn + offs_d
    tl.store(KBUF + kv, k)
    tl.store(VBUF + kv, v)
    if META:
        mp = b * stride_mb + h * stride_mh + (pos // BS) * stride_mp + offs_d
        if pos % BS == 0:
            tl.store(KMIN + mp, k)
            tl.store(KMAX + mp, k)
        else:
            tl.store(KMIN + mp, tl.minimum(tl.load(KMIN + mp), k))
            tl.store(KMAX + mp, tl.maximum(tl.load(KMAX + mp), k))
    if SUMS:
        sp = b * stride_mb + h * stride_mh + (pos // BS) * stride_mp + offs_d
        if pos % BS == 0:
            tl.store(KSUM + sp, k.to(tl.float32))
            tl.store(VSUM + sp, v.to(tl.float32))
        else:
            tl.store(KSUM + sp, tl.load(KSUM + sp) + k.to(tl.float32))
            tl.store(VSUM + sp, tl.load(VSUM + sp) + v.to(tl.float32))


def append_kv(
    k_new,
    v_new,
    k_buf,
    v_buf,
    pos,
    kmin=None,
    kmax=None,
    block_size: int = 64,
    ksum=None,
    vsum=None,
) -> None:
    """Write one decode token's K/V ([B, Hkv, D] or [B, Hkv, 1, D]) at position pos (device
    scalar) and, when kmin/kmax are given, fold it into its block's channel min/max (and,
    when ksum/vsum [B, Hkv, P, D] fp32 are given, into its block's K and V sums)."""
    B, Hkv = k_buf.shape[0], k_buf.shape[1]
    k_new = k_new.reshape(B, Hkv, -1).contiguous()
    v_new = v_new.reshape(B, Hkv, -1).contiguous()
    meta = kmin is not None
    sums = ksum is not None
    if not meta:
        kmin = kmax = k_buf
    if not sums:
        ksum = vsum = kmin
    assert not (sums and meta) or ksum.stride() == kmin.stride()
    _append_kernel[(B * Hkv,)](
        k_new,
        v_new,
        k_buf,
        v_buf,
        kmin,
        kmax,
        ksum,
        vsum,
        pos,
        k_new.stride(0),
        k_new.stride(1),
        k_buf.stride(0),
        k_buf.stride(1),
        k_buf.stride(2),
        kmin.stride(0),
        kmin.stride(1),
        kmin.stride(2),
        Hkv,
        block_size,
        D=k_buf.shape[-1],
        META=meta,
        SUMS=sums,
    )


@triton.jit
def _merge_kernel(
    ACC,
    M,
    L,
    OUT,
    stride_ab,
    stride_ah,
    stride_as,
    stride_mb,
    stride_mh,
    stride_ob,
    stride_oh,
    HQ,
    NS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = (pid // HQ).to(tl.int64)
    h = pid % HQ
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    s_ok = offs_s < NS
    m = tl.load(
        M + b * stride_mb + h * stride_mh + offs_s, mask=s_ok, other=float("-inf")
    )
    l_sum = tl.load(L + b * stride_mb + h * stride_mh + offs_s, mask=s_ok, other=0.0)
    w = tl.exp(m - tl.max(m, axis=0))
    w = tl.where(s_ok, w, 0.0)
    acc = tl.load(
        ACC
        + b * stride_ab
        + h * stride_ah
        + offs_s[:, None] * stride_as
        + offs_d[None, :],
        mask=s_ok[:, None],
        other=0.0,
    )
    out = tl.sum(acc * w[:, None], axis=0) / tl.sum(l_sum * w, axis=0)
    tl.store(OUT + b * stride_ob + h * stride_oh + offs_d, out.to(OUT.dtype.element_ty))


@triton.jit
def _tail_kernel(
    S2,
    ISC,
    L,
    PK,
    CAND,
    NCAND,
    KEPT,
    VSUM,
    NPTR,
    O_ACC,
    O_M,
    O_L,
    stride_sb,
    stride_sh,
    stride_cb,
    stride_ch,
    stride_vb,
    stride_vh,
    stride_vp,
    stride_ab,
    stride_ah,
    stride_as,
    stride_mb,
    stride_mh,
    stride_ms,
    HKV,
    BS,
    SPLIT0,
    NT2,
    log_rho,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
    D: tl.constexpr,
):
    # Extra split-K partials, one per tile of BLOCK_T blocks: every unselected block enters
    # as a single pseudo-token with the block-mean value and an estimated block weight, so
    # the softmax denominator keeps the dropped mass. Tiles [0, NT2) cover T2 (blocks the
    # prefilter dropped; weight = q . mean key + log(block length)); the rest cover T1
    # (candidate slots not kept; weight = exact int8 block mass, only when no token holds
    # more than rho of it).
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < G
    rows = h * G + offs_m
    N = tl.load(NPTR)
    v_base = VSUM + b * stride_vb + h * stride_vh
    if tile < NT2:
        offs = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        ok = offs < (N + BS - 1) // BS
        isc = tl.load(ISC + b * stride_cb + h * stride_ch + offs, mask=ok, other=1)
        use = ok & (isc == 0)
        logw = tl.load(
            S2 + b * stride_sb + rows[:, None] * stride_sh + offs[None, :],
            mask=row_ok[:, None] & use[None, :],
            other=float("-inf"),
        )
        blk = offs
        n_tok = tl.maximum(tl.minimum(BS, N - blk * BS), 1).to(tl.float32)
        logw = tl.where(use[None, :], logw + tl.log(n_tok)[None, :], float("-inf"))
    else:
        offs = (tile - NT2) * BLOCK_T + tl.arange(0, BLOCK_T)
        ok = offs < tl.load(NCAND + b * HKV + h)
        kept = tl.load(KEPT + b * stride_cb + h * stride_ch + offs, mask=ok, other=1)
        use = ok & (kept == 0)
        rmask = row_ok[:, None] & use[None, :]
        lse = tl.load(
            L + b * stride_sb + rows[:, None] * stride_sh + offs[None, :],
            mask=rmask,
            other=float("-inf"),
        )
        pk = tl.load(
            PK + b * stride_sb + rows[:, None] * stride_sh + offs[None, :],
            mask=rmask,
            other=0.0,
        )
        logw = tl.where(rmask & (pk < log_rho), lse, float("-inf"))
        blk = tl.load(CAND + b * stride_cb + h * stride_ch + offs, mask=use, other=0)
        n_tok = tl.maximum(tl.minimum(BS, N - blk * BS), 1).to(tl.float32)
    vm = (
        tl.load(
            v_base + blk[:, None] * stride_vp + offs_d[None, :],
            mask=use[:, None],
            other=0.0,
        )
        / n_tok[:, None]
    )
    m_i = tl.max(logw, axis=1)
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
    p = tl.exp(logw - m_safe[:, None])
    acc = tl.dot(p, vm)
    split = SPLIT0 + tile
    tl.store(
        O_ACC
        + b * stride_ab
        + rows[:, None] * stride_ah
        + split * stride_as
        + offs_d[None, :],
        acc,
        mask=row_ok[:, None],
    )
    tl.store(
        O_M + b * stride_mb + rows * stride_mh + split * stride_ms, m_i, mask=row_ok
    )
    tl.store(
        O_L + b * stride_mb + rows * stride_mh + split * stride_ms,
        tl.sum(p, axis=1),
        mask=row_ok,
    )


def block_sparse_decode_fused(
    q, k, v, block_idx, counts, block_size, num_splits=None, seq_len=None, tail=None
):
    """block_sparse_decode with the split merge as one Triton kernel instead of torch ops.

    tail: optional dict(aux=<select_blocks_two_stage(..., return_aux=True) aux>,
    vsum=[B, Hkv, P, D] fp32 block value sums, rho=float, t1=bool, t2=bool). Adds the
    unselected blocks back as block-mean pseudo-tokens (see _tail_kernel).
    """
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    kmax_sel = block_idx.shape[-1]
    bpi = max(1, 64 // block_size)
    if num_splits is None and TUNE["decode_blocks_per_split"]:
        num_splits = triton.cdiv(kmax_sel, TUNE["decode_blocks_per_split"])
    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = max(
            1, min(triton.cdiv(kmax_sel, 4), triton.cdiv(2 * sms, B * Hkv))
        )
    per_split = triton.cdiv(triton.cdiv(kmax_sel, num_splits), bpi) * bpi
    num_splits = triton.cdiv(kmax_sel, per_split)
    n_tail = 0
    if tail is not None:
        bt = 64
        nt2 = triton.cdiv(tail["vsum"].shape[2], bt) if tail.get("t2", True) else 0
        nt1 = triton.cdiv(tail["aux"]["max_cand"], bt) if tail.get("t1", True) else 0
        n_tail = nt2 + nt1
    n_parts = num_splits + n_tail
    acc = torch.empty(B, Hq, n_parts, D, device=q.device, dtype=torch.float32)
    m = torch.empty(B, Hq, n_parts, device=q.device, dtype=torch.float32)
    l_sum = torch.empty_like(m)
    if seq_len is None:
        seq_len = torch.full((1,), k.shape[2], device=q.device, dtype=torch.int32)
    if n_tail:
        a, vsum = tail["aux"], tail["vsum"]
        t2 = tail.get("t2", True)
        assert not t2 or a["has_s2"], "the T2 tail needs block key sums (ksum)"
        _tail_kernel[(B * Hkv, n_tail)](
            a["s2"],
            a["isc"],
            a["lse"],
            a["pk"],
            a["cand"],
            a["ncand"],
            a["kept"],
            vsum,
            seq_len,
            acc,
            m,
            l_sum,
            a["lse"].stride(0),
            a["lse"].stride(1),
            a["cand"].stride(0),
            a["cand"].stride(1),
            vsum.stride(0),
            vsum.stride(1),
            vsum.stride(2),
            acc.stride(0),
            acc.stride(1),
            acc.stride(2),
            m.stride(0),
            m.stride(1),
            m.stride(2),
            Hkv,
            block_size,
            num_splits,
            nt2,
            math.log(tail.get("rho", 0.6)),
            G=G,
            BLOCK_M=max(16, triton.next_power_of_2(G)),
            BLOCK_T=bt,
            D=D,
            num_warps=4,
        )
    _sparse_decode_kernel[(B * Hkv, num_splits)](
        q,
        k,
        v,
        block_idx,
        counts,
        seq_len,
        acc,
        m,
        l_sum,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        block_idx.stride(0),
        block_idx.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        Hkv,
        per_split,
        G=G,
        BLOCK_M=max(16, triton.next_power_of_2(G)),
        BLOCK_N=block_size,
        D=D,
        BPI=bpi,
        num_warps=TUNE["decode_warps"],
        num_stages=TUNE["decode_stages"],
    )
    out = torch.empty(B, Hq, D, device=q.device, dtype=q.dtype)
    _merge_kernel[(B * Hq,)](
        acc,
        m,
        l_sum,
        out,
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        m.stride(0),
        m.stride(1),
        out.stride(0),
        out.stride(1),
        Hq,
        NS=n_parts,
        BLOCK_S=triton.next_power_of_2(n_parts),
        D=D,
    )
    return out


# ---------------------------------------------------------------------------------------
# Two-stage selection: the Quest bound prefilters each query head's top blocks (union over
# the GQA group), block mass is then computed exactly from an int8 copy of the keys on
# those candidates only, and TopCdf + floor picks the final blocks from that mass.
# ---------------------------------------------------------------------------------------


def quantize_keys(k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 keys: k ~= k8 * scale, k8 [..., D] int8, scale [...] fp32."""
    scale = k.float().abs().amax(dim=-1).clamp_min(1e-8) / 127.0
    k8 = torch.round(k.float() / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return k8, scale


@triton.jit
def _prefilter_kernel(
    S,
    E1,
    S2,
    E2,
    ST,
    CAND,
    ISC,
    NCAND,
    NFORCED,
    NPTR,
    stride_sb,
    stride_sh,
    stride_cb,
    stride_ch,
    HKV,
    BS,
    SINK,
    LOCAL,
    pre_frac,
    NT,
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ITERS: tl.constexpr,
    SHARED: tl.constexpr,
    MEAN: tl.constexpr = False,
    RADIX: tl.constexpr = 2,
    SORT: tl.constexpr = False,
    STATS: tl.constexpr = False,
    TILE: tl.constexpr = 64,
    BLOCK_T: tl.constexpr = 16,
):
    # SHARED: one top set per KV head from the group-mean bound softmax; otherwise the
    # union of every query head's own top set. MEAN: rank by the average of the bound
    # softmax and the block-mean-key softmax (S2) instead of the bound alone.
    pid = tl.program_id(0)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    offs_p = tl.arange(0, BLOCK_P)
    cur_p = (tl.load(NPTR) + BS - 1) // BS
    valid = offs_p < cur_p
    forced = valid & ((offs_p < SINK) | (offs_p >= cur_p - LOCAL))
    free = valid & (~forced)
    n_keep = tl.ceil(tl.sum(free.to(tl.float32), axis=0) * pre_frac)
    ones = tl.full([BLOCK_P], 1.0, tl.float32)
    union = tl.zeros([BLOCK_P], tl.int1)
    if STATS:
        # SHARED only (the launch gates it; a per-head union gains nothing, measured).
        # Ranking from the scores kernel's tile statistics: each block's softmax
        # probability is exp(s - tile max) * exp(tile max - max) / sum, summed in linear
        # space over the group like the reference (softmax(...).mean over heads). Below
        # ~1e-30 fp32 loses precision and then underflows (one block > ~87 logits above
        # the rest), so if fewer than n_keep blocks clear it the ranking is redone in log
        # space.
        offs_t = tl.arange(0, BLOCK_T)
        t_ok = offs_t < NT
        tile_e = offs_p // TILE
        pm = tl.zeros([BLOCK_P], tl.float32)
        for g in range(G):
            off = b * stride_sb + (h * G + g) * stride_sh
            st = ST + ((b * HKV * G + h * G + g) * NT) * 4
            mt = tl.load(st + offs_t * 4, mask=t_ok, other=float("-inf"))
            zt = tl.load(st + offs_t * 4 + 1, mask=t_ok, other=0.0)
            mx = tl.max(mt, axis=0)
            z = tl.sum(tl.where(zt > 0, zt * tl.exp(mt - mx), 0.0), axis=0)
            me = tl.load(st + tile_e * 4, mask=valid, other=float("-inf"))
            pr = tl.load(E1 + off + offs_p, mask=valid, other=0.0) * tl.exp(me - mx) / z
            if MEAN:
                mt = tl.load(st + offs_t * 4 + 2, mask=t_ok, other=float("-inf"))
                zt = tl.load(st + offs_t * 4 + 3, mask=t_ok, other=0.0)
                mx = tl.max(mt, axis=0)
                z = tl.sum(tl.where(zt > 0, zt * tl.exp(mt - mx), 0.0), axis=0)
                me = tl.load(st + tile_e * 4 + 2, mask=valid, other=float("-inf"))
                pr2 = tl.load(E2 + off + offs_p, mask=valid, other=0.0)
                pr = 0.5 * (pr + pr2 * tl.exp(me - mx) / z)
            pm += pr
        pm = tl.where(free, pm, 0.0)
        if tl.sum((pm > 1e-30).to(tl.float32), axis=0) >= n_keep:
            union = _top_linear(pm, free, ones, n_keep, ITERS, RADIX, SORT)
        else:
            lmean = tl.full([BLOCK_P], float("-inf"), tl.float32)
            for g in range(G):
                off = b * stride_sb + (h * G + g) * stride_sh
                lpr = _head_log_softmax(S, S2, off, offs_p, valid, MEAN)
                lmean = _logaddexp(lmean, lpr)
            union = _top_log(lmean, free, ones, n_keep, ITERS, RADIX, SORT)
    else:
        lmean = tl.full([BLOCK_P], float("-inf"), tl.float32)
        for g in range(G):
            off = b * stride_sb + (h * G + g) * stride_sh
            lpr = _head_log_softmax(S, S2, off, offs_p, valid, MEAN)
            if SHARED:
                lmean = _logaddexp(lmean, lpr)
            else:
                # Wide range: a block far down the bound's ranking can still hold real mass.
                union = union | _top_log(lpr, free, ones, n_keep, ITERS, RADIX, SORT)
        if SHARED:
            union = _top_log(lmean, free, ones, n_keep, ITERS, RADIX, SORT)
    n_forced = tl.sum(forced.to(tl.int32), axis=0)
    slot = tl.where(
        forced,
        tl.cumsum(forced.to(tl.int32), axis=0) - 1,
        n_forced + tl.cumsum(union.to(tl.int32), axis=0) - 1,
    )
    tl.store(
        CAND + b * stride_cb + h * stride_ch + slot,
        offs_p.to(tl.int32),
        mask=forced | union,
    )
    tl.store(NCAND + b * HKV + h, n_forced + tl.sum(union.to(tl.int32), axis=0))
    tl.store(NFORCED + b * HKV + h, n_forced)
    tl.store(
        ISC + b * stride_cb + h * stride_ch + offs_p,
        (forced | union).to(tl.int8),
        mask=valid,
    )


@triton.jit
def _ordered_key(x):
    """int32 keys that compare (signed) in the same order as the fp32 values x."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)


@triton.jit
def _prefilter_live_kernel(
    S,
    S2,
    ST,
    KEY,
    CAND,
    ISC,
    NCAND,
    NFORCED,
    NPTR,
    stride_sb,
    stride_sh,
    stride_cb,
    stride_ch,
    HKV,
    BS,
    SINK,
    LOCAL,
    pre_frac,
    NT,
    alpha,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MEAN: tl.constexpr = False,
    DIGIT: tl.constexpr = 8,
    REGION: tl.constexpr = 0,
    BLOCK_R: tl.constexpr = 1,
    BLOCK_F: tl.constexpr = 1,
):
    # The shared prefilter with work proportional to the live length: block scores are
    # walked in CHUNK-wide pieces up to cur_p (read on device, so CUDA graphs still see a
    # fixed launch) instead of one tile padded to the capacity. Each query head's
    # log-softmax normalizer comes from the scores kernel's tile statistics, so a chunk's
    # ranking key, the log of the group's summed block probability, needs no whole-row
    # pass. The n_keep-th largest key is found exactly by a radix select over DIGIT-bit
    # digits: one 2**DIGIT-bin histogram per digit, restricted to keys matching the
    # digits above. tl.histogram's cost grows with the bin count, so 4-bit digits (eight
    # rounds) beat 8-bit ones (four) and 11-bit ones are slower still. Exactly n_keep
    # blocks are kept: keys tied at the threshold are taken in block order.
    # REGION > 0 (grid axis 1, a multiple of the scores kernel's 64-block tiles): each
    # program keeps the top of its own REGION blocks instead of the global top, so a KV
    # head's selection runs as several programs. Region r's quota mixes its share of the
    # free blocks with its share of the ranking probability (weight alpha on the first),
    # the latter read from the tile statistics minus the forced blocks; a region holding
    # most of the mass so gets most of the candidates. Every program derives all quotas
    # the same way, so slot offsets agree without communication. Quotas are cumulative
    # roundings of n_keep (capped at a region's free blocks), so they never exceed it.
    pid = tl.program_id(0)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    cur_p = (tl.load(NPTR) + BS - 1) // BS
    n_forced = tl.minimum(cur_p, SINK + LOCAL)  # sink and local blocks overlap below
    n_keep = tl.ceil((cur_p - n_forced).to(tl.float32) * pre_frac).to(tl.int32)
    offs_g = tl.arange(0, BLOCK_G)
    g_ok = offs_g < G
    offs_t = tl.arange(0, BLOCK_T)
    st = ST + ((b * HKV * G + h * G + offs_g)[:, None] * NT + offs_t[None, :]) * 4
    st_ok = g_ok[:, None] & (offs_t[None, :] < NT)
    mt = tl.load(st, mask=st_ok, other=float("-inf"))
    zt = tl.load(st + 1, mask=st_ok, other=0.0)
    mx = tl.max(mt, axis=1)
    lse = mx + tl.log(
        tl.sum(tl.where(zt > 0, zt * tl.exp(mt - mx[:, None]), 0.0), axis=1)
    )
    lse2 = lse
    if MEAN:
        mt = tl.load(st + 2, mask=st_ok, other=float("-inf"))
        zt = tl.load(st + 3, mask=st_ok, other=0.0)
        mx = tl.max(mt, axis=1)
        lse2 = mx + tl.log(
            tl.sum(tl.where(zt > 0, zt * tl.exp(mt - mx[:, None]), 0.0), axis=1)
        )
    row = b * stride_sb + (h * G + offs_g)[:, None] * stride_sh
    kb = KEY + b * stride_cb + h * stride_ch
    lo = tl.zeros([], tl.int32)
    hi = cur_p
    keep_lo = tl.zeros([], tl.int32)
    n_take = n_keep
    n_union = n_keep
    if REGION > 0:
        lo = tl.program_id(1) * REGION
        hi = tl.minimum(lo + REGION, cur_p)
        offs_r = tl.arange(0, BLOCK_R)
        r_lo = offs_r * REGION
        r_hi = tl.minimum(r_lo + REGION, cur_p)
        f_lo = tl.maximum(tl.minimum(r_lo, cur_p - LOCAL) - SINK, 0)
        f_hi = tl.maximum(tl.minimum(r_hi, cur_p - LOCAL) - SINK, 0)
        n_free = tl.where(r_lo < cur_p, tl.maximum(f_hi - f_lo, 0), 0)
        # ranking probability per tile, summed over the group's query heads
        mt = tl.load(st, mask=st_ok, other=float("-inf"))
        zt = tl.load(st + 1, mask=st_ok, other=0.0)
        tm = tl.sum(tl.where(st_ok, zt * tl.exp(mt - lse[:, None]), 0.0), axis=0)
        if MEAN:
            mt = tl.load(st + 2, mask=st_ok, other=float("-inf"))
            zt = tl.load(st + 3, mask=st_ok, other=0.0)
            tm += tl.sum(tl.where(st_ok, zt * tl.exp(mt - lse2[:, None]), 0.0), axis=0)
        # minus the forced blocks (sink, then local, overlaps counted once)
        offs_f = tl.arange(0, BLOCK_F)
        fpos = tl.where(offs_f < SINK, offs_f, cur_p - LOCAL + offs_f - SINK)
        f_ok = (offs_f < SINK + LOCAL) & (fpos >= 0) & (fpos < cur_p)
        f_ok = f_ok & ((offs_f < SINK) | (fpos >= SINK))
        okf = g_ok[:, None] & f_ok[None, :]
        sf = tl.load(S + row + fpos[None, :], mask=okf, other=0.0)
        pf = tl.sum(tl.where(okf, tl.exp(sf - lse[:, None]), 0.0), axis=0)
        if MEAN:
            sf = tl.load(S2 + row + fpos[None, :], mask=okf, other=0.0)
            pf += tl.sum(tl.where(okf, tl.exp(sf - lse2[:, None]), 0.0), axis=0)
        tm -= tl.sum(
            tl.where((fpos // 64)[None, :] == offs_t[:, None], pf[None, :], 0.0), 1
        )
        TPR: tl.constexpr = REGION // 64
        rm = tl.sum(
            tl.where(
                (offs_t // TPR)[None, :] == offs_r[:, None],
                tl.maximum(tm, 0.0)[None, :],
                0.0,
            ),
            axis=1,
        )
        fsum = tl.maximum(tl.sum(n_free, axis=0), 1).to(tl.float32)
        msum = tl.sum(rm, axis=0)
        w = n_free.to(tl.float32) / fsum
        w = tl.where(msum > 0, alpha * w + (1.0 - alpha) * rm / msum, w)
        h_r = tl.minimum(tl.ceil(tl.cumsum(w, axis=0) * n_keep.to(tl.float32)), n_keep)
        h_r = h_r.to(tl.int32)
        h_prev = tl.sum(
            tl.where(offs_r[None, :] == offs_r[:, None] - 1, h_r[None, :], 0), axis=1
        )
        k_r = tl.maximum(tl.minimum(h_r - h_prev, n_free), 0)
        o_r = tl.cumsum(k_r, axis=0) - k_r
        me = offs_r == tl.program_id(1)
        n_take = tl.sum(tl.where(me, k_r, 0), axis=0)
        keep_lo = tl.sum(tl.where(me, o_r, 0), axis=0)
        n_union = tl.sum(k_r, axis=0)
    NB: tl.constexpr = 1 << DIGIT
    TOP: tl.constexpr = 32 - DIGIT
    bins = tl.arange(0, NB)
    hist = tl.zeros([NB], tl.int32)
    for c0 in range(lo, hi, CHUNK):
        offs_p = c0 + tl.arange(0, CHUNK)
        valid = offs_p < hi
        free = valid & (offs_p >= SINK) & (offs_p < cur_p - LOCAL)
        ok = g_ok[:, None] & valid[None, :]
        x = tl.load(S + row + offs_p[None, :], mask=ok, other=0.0)
        x = tl.where(ok, x - lse[:, None], float("-inf"))
        m = tl.max(x, axis=0)
        if MEAN:
            x2 = tl.load(S2 + row + offs_p[None, :], mask=ok, other=0.0)
            x2 = tl.where(ok, x2 - lse2[:, None], float("-inf"))
            m = tl.maximum(m, tl.max(x2, axis=0))
        m = tl.where(free, m, 0.0)
        z = tl.sum(tl.exp(x - m[None, :]), axis=0)
        if MEAN:
            z += tl.sum(tl.exp(x2 - m[None, :]), axis=0)
        key = _ordered_key(tl.where(free, m + tl.log(z), -1e30))
        tl.store(kb + offs_p, key, mask=valid)
        hist += tl.histogram(((key >> TOP) + NB // 2) & (NB - 1), NB, mask=free)
    # Radix select, most significant digit first (the top one is signed, hence + NB/2).
    # k counts how many keys are still wanted from the current digit's bin downward.
    k = n_take
    t = tl.zeros([], tl.int32)
    for i in tl.static_range(32 // DIGIT):
        s = TOP - DIGIT * i
        if i > 0:
            hist = tl.zeros([NB], tl.int32)
            for c0 in range(lo, hi, CHUNK):
                offs_p = c0 + tl.arange(0, CHUNK)
                free = (offs_p < hi) & (offs_p >= SINK) & (offs_p < cur_p - LOCAL)
                key = tl.load(kb + offs_p, mask=free, other=0)
                hit = free & ((key >> (s + DIGIT)) == (t >> (s + DIGIT)))
                hist += tl.histogram((key >> s) & (NB - 1), NB, mask=hit)
        ge = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist  # count >= bin
        d = tl.max(tl.where(ge >= k, bins, 0), axis=0)
        k -= tl.sum(tl.where(bins == d, ge - hist, 0), axis=0)
        if i == 0:
            t = (d - NB // 2) * (1 << TOP)
        else:
            t = t | (d << s)
    # t is now the n_take-th largest key: keep every free block above it and the first
    # k of those equal to it. Forced blocks take the leading slots in block order (sink,
    # then local), union blocks the rest; when sink and local overlap every block is
    # forced and its slot is its index.
    nu = tl.zeros([], tl.int32)
    nt = tl.zeros([], tl.int32)
    for c0 in range(lo, hi, CHUNK):
        offs_p = c0 + tl.arange(0, CHUNK)
        valid = offs_p < hi
        forced = valid & ((offs_p < SINK) | (offs_p >= cur_p - LOCAL))
        key = tl.load(kb + offs_p, mask=valid, other=0)
        free = valid & (~forced) & (n_take > 0)
        tie = free & (key == t)
        tie_rank = nt + tl.cumsum(tie.to(tl.int32), axis=0) - 1
        union = free & ((key > t) | (tie & (tie_rank < k)))
        nt += tl.sum(tie.to(tl.int32), axis=0)
        slot = tl.where(
            forced,
            tl.where(offs_p < SINK, offs_p, n_forced - (cur_p - offs_p)),
            n_forced + keep_lo + nu + tl.cumsum(union.to(tl.int32), axis=0) - 1,
        )
        tl.store(
            CAND + b * stride_cb + h * stride_ch + slot,
            offs_p.to(tl.int32),
            mask=forced | union,
        )
        tl.store(
            ISC + b * stride_cb + h * stride_ch + offs_p,
            (forced | union).to(tl.int8),
            mask=valid,
        )
        nu += tl.sum(union.to(tl.int32), axis=0)
    if tl.program_id(1) == 0:
        tl.store(NCAND + b * HKV + h, n_forced + n_union)
        tl.store(NFORCED + b * HKV + h, n_forced)


@triton.jit
def _block_mass_kernel(
    Q,
    K8,
    KS,
    CAND,
    NCAND,
    L,
    PK,
    NPTR,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_sb,
    stride_sh,
    stride_cb,
    stride_ch,
    stride_lb,
    stride_lh,
    HKV,
    SLOTS_PER_SPLIT,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    STAGES: tl.constexpr = 1,
):
    pid_bh = tl.program_id(0)
    b = (pid_bh // HKV).to(tl.int64)
    h = pid_bh % HKV
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < G
    q = tl.load(
        Q + b * stride_qb + (h * G + offs_m)[:, None] * stride_qh + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    N = tl.load(NPTR)
    ncand = tl.load(NCAND + b * HKV + h)
    start = tl.program_id(1) * SLOTS_PER_SPLIT
    end = tl.minimum(start + SLOTS_PER_SPLIT, ncand)
    for j in tl.range(start, end, num_stages=STAGES):
        blk = tl.load(CAND + b * stride_cb + h * stride_ch + j)
        tok = blk * BLOCK_N + offs_n
        ok = tok < N
        k8 = tl.load(
            K8
            + b * stride_kb
            + h * stride_kh
            + tok[:, None] * stride_kn
            + offs_d[None, :],
            mask=ok[:, None],
            other=0,
        )
        ks = tl.load(KS + b * stride_sb + h * stride_sh + tok, mask=ok, other=0.0)
        # int8 values are exact in bf16; each token's scale is applied to its logit after
        # the dot, so no dequantized key is rounded (and nothing is dequantized).
        s = tl.dot(q, tl.trans(k8.to(q.dtype))) * ks[None, :]
        s = tl.where(ok[None, :], s, float("-inf"))
        m = tl.max(s, axis=1)
        lse = m + tl.log(tl.sum(tl.exp(s - m[:, None]), axis=1))
        tl.store(L + b * stride_lb + (h * G + offs_m) * stride_lh + j, lse, mask=row_ok)
        # log of the share of the block's mass on its top token (0 when one token holds it)
        tl.store(
            PK + b * stride_lb + (h * G + offs_m) * stride_lh + j, m - lse, mask=row_ok
        )


@triton.jit
def _final_select_kernel(
    L,
    CAND,
    NCAND,
    NFORCED,
    IDX,
    CNT,
    KEPT,
    stride_lb,
    stride_lh,
    stride_cb,
    stride_ch,
    stride_ib,
    stride_ih,
    HKV,
    CAP,
    tau,
    floor_frac,
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ITERS: tl.constexpr,
    RADIX: tl.constexpr = 2,
    SORT: tl.constexpr = False,
):
    pid = tl.program_id(0)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    offs = tl.arange(0, BLOCK_P)
    ncand = tl.load(NCAND + b * HKV + h)
    nf = tl.load(NFORCED + b * HKV + h)
    valid = offs < ncand
    lmean = tl.full([BLOCK_P], float("-inf"), tl.float32)
    for g in range(G):
        s = tl.load(
            L + b * stride_lb + (h * G + g) * stride_lh + offs,
            mask=valid,
            other=float("-inf"),
        )
        mx = tl.max(s, axis=0)
        lmean = _logaddexp(lmean, s - mx - tl.log(tl.sum(tl.exp(s - mx), axis=0)))
    lp_all = lmean - tl.log(G * 1.0)
    probs = tl.exp(lp_all)
    forced = valid & (offs < nf)
    free = valid & (offs >= nf)
    forced_mass = tl.sum(tl.where(forced, probs, 0.0), axis=0)
    k_floor = tl.ceil(ncand.to(tl.float32) * floor_frac)
    cap_free = (CAP - nf).to(tl.float32)
    if SORT:
        # Exact TopCdf, as in the reference: sort the free candidates by log-probability,
        # take n = min(cap, max(floor, the number needed to reach tau)), threshold at the
        # n-th. Cheaper than the search for small candidate sets; bitonic sort cost grows
        # as n log^2 n, so large ones use the search.
        lp = tl.where(free & (lp_all > -1e30), lp_all, -1e30)
        ls = tl.sort(lp, descending=True)
        real = ls > -1e30
        ps = tl.where(real, tl.exp(ls), 0.0)
        before = tl.cumsum(ps, axis=0) - ps
        n_cdf = tl.sum((real & (forced_mass + before < tau)).to(tl.float32), axis=0)
        n_real = tl.sum(real.to(tl.float32), axis=0)
        n = tl.minimum(tl.minimum(tl.maximum(n_cdf, k_floor), cap_free), n_real)
        t = tl.sum(tl.where(offs == n.to(tl.int32) - 1, ls, 0.0), axis=0)
        t = tl.where(n >= 1, t, float("inf"))
    else:
        # Thresholds are searched in log-probability: tail blocks of a peaked head sit far
        # below any fixed absolute resolution, so a search over [0, 1] over-selects them.
        lp = tl.where(lp_all > -1e30, lp_all, -1e30)
        # Bracket [best - 60, best]: blocks e^60 below the best free block carry no mass.
        top = tl.max(tl.where(free, lp, -1e30), axis=0)
        lo_c = top - 60.0
        hi_c = top
        lo_k = top - 60.0
        hi_k = top
        lo_x = top - 60.0
        hi_x = top
        ones = tl.full([BLOCK_P], 1.0, tl.float32)
        for _ in range(ITERS):
            lo_c, hi_c = _search_round(
                lp, probs, free, lo_c, hi_c, tau - forced_mass, RADIX, False
            )
            lo_k, hi_k = _search_round(
                lp, ones, free, lo_k, hi_k, k_floor, RADIX, False
            )
            lo_x, hi_x = _search_round(
                lp, ones, free, lo_x, hi_x, cap_free, RADIX, True
            )
        t = tl.maximum(tl.minimum(lo_c, lo_k), hi_x)
    chosen = free & (lp >= t) & (lp > -1e30)
    keep = forced | chosen
    slot = tl.cumsum(keep.to(tl.int32), axis=0) - 1  # forced occupy the leading slots
    blk = tl.load(CAND + b * stride_cb + h * stride_ch + offs, mask=valid, other=0)
    tl.store(IDX + b * stride_ib + h * stride_ih + slot, blk, mask=keep & (slot < CAP))
    n_sel = tl.minimum(tl.sum(keep.to(tl.int32), axis=0), CAP)
    tl.store(CNT + b * HKV + h, n_sel)
    # Zero the unused index slots so callers may scatter over all CAP slots.
    tl.store(
        IDX + b * stride_ib + h * stride_ih + offs,
        tl.zeros([BLOCK_P], tl.int32),
        mask=(offs >= n_sel) & (offs < CAP),
    )
    tl.store(
        KEPT + b * stride_cb + h * stride_ch + offs,
        (keep & (slot < CAP)).to(tl.int8),
        mask=valid,
    )


@triton.jit
def _group_log_mean(s, lse, ok, G: tl.constexpr):
    """log of the mean over query heads (rows) of exp(s - lse); -inf where none is ok."""
    x = tl.where(ok, s - lse[:, None], float("-inf"))
    m = tl.max(x, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    return m + tl.log(tl.sum(tl.exp(x - m[None, :]), axis=0) / G)


@triton.jit
def _final_select_split_kernel(
    L,
    CAND,
    NCAND,
    NFORCED,
    IDX,
    CNT,
    KEPT,
    stride_lb,
    stride_lh,
    stride_cb,
    stride_ch,
    stride_ib,
    stride_ih,
    HKV,
    CAP,
    MAXC,
    tau,
    floor_frac,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ITERS: tl.constexpr,
    RADIX: tl.constexpr = 2,
    SORT: tl.constexpr = False,
):
    # _final_select_kernel with the forced candidates (the leading NFORCED slots) in
    # their own small tile, so the sort or search spans only the free candidates, and
    # every query head's block log-softmax taken at once from one [heads, slots] tile.
    # Candidate slots past MAXC were never scored by the mass kernel and are ignored.
    pid = tl.program_id(0)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    ncand = tl.minimum(tl.load(NCAND + b * HKV + h), MAXC)
    nf = tl.load(NFORCED + b * HKV + h)
    offs_g = tl.arange(0, BLOCK_G)
    g_ok = offs_g < G
    rows = L + b * stride_lb + (h * G + offs_g)[:, None] * stride_lh
    offs_c = tl.arange(0, BLOCK_C)
    forced = offs_c < nf
    ok_c = g_ok[:, None] & forced[None, :]
    sc = tl.load(rows + offs_c[None, :], mask=ok_c, other=float("-inf"))
    offs = tl.arange(0, BLOCK_P)  # free candidate j sits in slot nf + j
    free = nf + offs < ncand
    ok_f = g_ok[:, None] & free[None, :]
    sf = tl.load(rows + (nf + offs)[None, :], mask=ok_f, other=float("-inf"))
    mx = tl.maximum(tl.max(sc, axis=1), tl.max(sf, axis=1))
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    lse = mx + tl.log(
        tl.sum(tl.exp(sc - mx[:, None]), axis=1)
        + tl.sum(tl.exp(sf - mx[:, None]), axis=1)
    )
    forced_mass = tl.sum(
        tl.where(forced, tl.exp(_group_log_mean(sc, lse, ok_c, G)), 0.0), axis=0
    )
    lp = _group_log_mean(sf, lse, ok_f, G)
    lp = tl.where(free & (lp > -1e30), lp, -1e30)
    k_floor = tl.ceil(ncand.to(tl.float32) * floor_frac)
    cap_free = (CAP - nf).to(tl.float32)
    if SORT:
        ls = tl.sort(lp, descending=True)
        real = ls > -1e30
        ps = tl.where(real, tl.exp(ls), 0.0)
        before = tl.cumsum(ps, axis=0) - ps
        n_cdf = tl.sum((real & (forced_mass + before < tau)).to(tl.float32), axis=0)
        n_real = tl.sum(real.to(tl.float32), axis=0)
        n = tl.minimum(tl.minimum(tl.maximum(n_cdf, k_floor), cap_free), n_real)
        t = tl.sum(tl.where(offs == n.to(tl.int32) - 1, ls, 0.0), axis=0)
        t = tl.where(n >= 1, t, float("inf"))
    else:
        probs = tl.where(free, tl.exp(lp), 0.0)
        top = tl.max(lp, axis=0)
        lo_c = top - 60.0
        hi_c = top
        lo_k = top - 60.0
        hi_k = top
        lo_x = top - 60.0
        hi_x = top
        ones = tl.full([BLOCK_P], 1.0, tl.float32)
        for _ in range(ITERS):
            lo_c, hi_c = _search_round(
                lp, probs, free, lo_c, hi_c, tau - forced_mass, RADIX, False
            )
            lo_k, hi_k = _search_round(
                lp, ones, free, lo_k, hi_k, k_floor, RADIX, False
            )
            lo_x, hi_x = _search_round(
                lp, ones, free, lo_x, hi_x, cap_free, RADIX, True
            )
        t = tl.maximum(tl.minimum(lo_c, lo_k), hi_x)
    chosen = free & (lp >= t) & (lp > -1e30)
    slot = nf + tl.cumsum(chosen.to(tl.int32), axis=0) - 1
    cb = CAND + b * stride_cb + h * stride_ch
    ib = IDX + b * stride_ib + h * stride_ih
    tl.store(
        ib + offs_c,
        tl.load(cb + offs_c, mask=forced, other=0),
        mask=forced & (offs_c < CAP),
    )
    blk = tl.load(cb + nf + offs, mask=free, other=0)
    tl.store(ib + slot, blk, mask=chosen & (slot < CAP))
    n_sel = tl.minimum(nf + tl.sum(chosen.to(tl.int32), axis=0), CAP)
    tl.store(CNT + b * HKV + h, n_sel)
    # Zero the unused index slots so callers may scatter over all CAP slots (the host
    # makes BLOCK_P >= CAP).
    tl.store(
        ib + offs, tl.zeros([BLOCK_P], tl.int32), mask=(offs >= n_sel) & (offs < CAP)
    )
    kb = KEPT + b * stride_cb + h * stride_ch
    tl.store(kb + offs_c, (offs_c < CAP).to(tl.int8), mask=forced)
    tl.store(kb + nf + offs, (chosen & (slot < CAP)).to(tl.int8), mask=free)


def _keep_count(n: int, frac: float) -> int:
    """ceil(n * frac) in fp32, as the prefilter kernels compute it."""
    return math.ceil(
        (
            torch.tensor(n, dtype=torch.float32)
            * torch.tensor(frac, dtype=torch.float32)
        ).item()
    )


@torch.no_grad()
def select_blocks_two_stage(
    q: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    k8: torch.Tensor,
    kscale: torch.Tensor,
    block_size: int,
    cdf_threshold: float,
    floor_fraction: float,
    max_fraction: float,
    prefilter_fraction: float = 0.25,
    sink_blocks: int = 8,
    local_blocks: int = 8,
    seq_len: Optional[torch.Tensor] = None,
    prefilter_shared: bool = True,
    ksum: Optional[torch.Tensor] = None,
    prefilter_mean: bool = False,
    return_aux: bool = False,
):
    """Bound prefilter -> int8 block mass on candidates -> TopCdf + floor, per KV head.

    q: [B, Hq, D] scaled; kmin/kmax: [B, Hkv, P, D]; k8: [B, Hkv, N, D] int8 and kscale
    [B, Hkv, N] from quantize_keys. Returns block_idx [B, Hkv, cap] int32 and counts.
    ksum: [B, Hkv, P, D] fp32 block key sums; with it the block-mean-key scores are
    computed too, and prefilter_mean ranks candidates by the average of the bound and
    mean-key softmaxes. return_aux also returns the intermediates the tail estimate uses.
    """
    B, Hq, D = q.shape
    Hkv, P = kmin.shape[1], kmin.shape[2]
    G = Hq // Hkv
    dev = q.device
    if seq_len is None:
        seq_len = torch.full(
            (1,), min(P * block_size, k8.shape[2]), device=dev, dtype=torch.int32
        )
    bm = max(16, triton.next_power_of_2(G))
    bp = triton.next_power_of_2(P)
    scores = torch.empty(B, Hq, P, device=dev, dtype=torch.float32)
    has_s2 = ksum is not None
    assert has_s2 or not prefilter_mean, "prefilter_mean needs ksum"
    if has_s2:
        assert ksum.shape == kmin.shape and ksum.stride() == kmin.stride()
    scores2 = torch.empty_like(scores) if has_s2 else scores
    nt = triton.cdiv(P, 64)
    stats = prefilter_shared and bool(TUNE["prefilter_stats"])
    live = stats and TUNE["prefilter_chunk"] > 0
    e1 = e2 = tile_stats = scores
    if stats:
        tile_stats = torch.empty(B, Hq, nt, 4, device=dev, dtype=torch.float32)
    if stats and not live:
        e1 = torch.empty_like(scores)
        e2 = torch.empty_like(scores) if has_s2 else scores
    _block_scores_kernel[(B * Hkv, nt)](
        q,
        kmin,
        kmax,
        scores,
        e1,
        ksum if has_s2 else kmin,
        scores2,
        e2,
        tile_stats,
        seq_len,
        q.stride(0),
        q.stride(1),
        kmin.stride(0),
        kmin.stride(1),
        kmin.stride(2),
        scores.stride(0),
        scores.stride(1),
        Hkv,
        P,
        block_size,
        nt,
        G=G,
        BLOCK_M=bm,
        BLOCK_P=64,
        D=D,
        MEAN=has_s2,
        STATS=stats,
        STORE_E=not live,
    )
    # Every buffer below is written over the range its readers use (candidate slots
    # < ncand, blocks < cur_p, index slots < CAP), so none needs a fill kernel.
    cand = torch.empty(B, Hkv, P, device=dev, dtype=torch.int32)
    isc = torch.empty(B, Hkv, P, device=dev, dtype=torch.int8)
    ncand = torch.empty(B, Hkv, device=dev, dtype=torch.int32)
    nforced = torch.empty_like(ncand)
    if live:
        region = TUNE["prefilter_region"]
        assert region % 64 == 0, "regions are whole tiles of the scores kernel"
        _prefilter_live_kernel[(B * Hkv, triton.cdiv(P, region) if region else 1)](
            scores,
            scores2,
            tile_stats,
            torch.empty_like(cand),
            cand,
            isc,
            ncand,
            nforced,
            seq_len,
            scores.stride(0),
            scores.stride(1),
            cand.stride(0),
            cand.stride(1),
            Hkv,
            block_size,
            sink_blocks,
            local_blocks,
            prefilter_fraction,
            nt,
            float(TUNE["prefilter_region_alpha"]),
            G=G,
            BLOCK_G=triton.next_power_of_2(G),
            CHUNK=TUNE["prefilter_chunk"],
            BLOCK_T=triton.next_power_of_2(nt),
            MEAN=prefilter_mean,
            DIGIT=TUNE["prefilter_digit"],
            REGION=region,
            BLOCK_R=triton.next_power_of_2(triton.cdiv(P, region)) if region else 1,
            BLOCK_F=triton.next_power_of_2(max(1, sink_blocks + local_blocks)),
            num_warps=TUNE["prefilter_chunk_warps"],
        )
    else:
        _prefilter_kernel[(B * Hkv,)](
            scores,
            e1,
            scores2,
            e2,
            tile_stats,
            cand,
            isc,
            ncand,
            nforced,
            seq_len,
            scores.stride(0),
            scores.stride(1),
            cand.stride(0),
            cand.stride(1),
            Hkv,
            block_size,
            sink_blocks,
            local_blocks,
            prefilter_fraction,
            nt,
            G=G,
            BLOCK_P=bp,
            ITERS=_rounds(30, TUNE["search_radix"]),
            SHARED=prefilter_shared,
            MEAN=prefilter_mean,
            RADIX=TUNE["search_radix"],
            SORT=bp <= TUNE["sort_max_blocks"],
            STATS=stats,
            BLOCK_T=triton.next_power_of_2(nt),
            num_warps=_warps(TUNE["prefilter_warps"], bp),
        )
    lse = torch.empty(B, Hq, P, device=dev, dtype=torch.float32)
    pk = torch.empty_like(lse)
    # Size the grid by the candidate bound, not P: ~4 candidate blocks per program.
    n_forced = min(P, sink_blocks + local_blocks)
    if live:  # keeps exactly n_keep free blocks
        max_cand = n_forced + _keep_count(P - n_forced, prefilter_fraction)
    else:
        per_head = math.ceil(prefilter_fraction * P)
        max_cand = min(
            P,
            sink_blocks
            + local_blocks
            + (per_head if prefilter_shared else G * per_head),
        )
    slots = TUNE["mass_slots"]
    splits = triton.cdiv(max_cand, slots)
    _block_mass_kernel[(B * Hkv, splits)](
        q,
        k8,
        kscale,
        cand,
        ncand,
        lse,
        pk,
        seq_len,
        q.stride(0),
        q.stride(1),
        k8.stride(0),
        k8.stride(1),
        k8.stride(2),
        kscale.stride(0),
        kscale.stride(1),
        cand.stride(0),
        cand.stride(1),
        lse.stride(0),
        lse.stride(1),
        Hkv,
        slots,
        G=G,
        BLOCK_M=bm,
        BLOCK_N=block_size,
        D=D,
        STAGES=TUNE["mass_stages"],
        num_warps=TUNE["mass_warps"],
    )
    cap = min(P, n_forced + max(1, math.ceil(max_fraction * P)))
    idx = torch.empty(B, Hkv, cap, device=dev, dtype=torch.int32)
    counts = torch.empty(B, Hkv, device=dev, dtype=torch.int32)
    kept = torch.empty(B, Hkv, P, device=dev, dtype=torch.int8)
    if TUNE["final_split"]:
        # Free candidates number at most max_cand - n_forced (none while the context is
        # shorter than the forced blocks); the tile also spans the CAP index slots.
        bp_sel = triton.next_power_of_2(max(max_cand - n_forced, cap))
        _final_select_split_kernel[(B * Hkv,)](
            lse,
            cand,
            ncand,
            nforced,
            idx,
            counts,
            kept,
            lse.stride(0),
            lse.stride(1),
            cand.stride(0),
            cand.stride(1),
            idx.stride(0),
            idx.stride(1),
            Hkv,
            cap,
            max_cand,
            cdf_threshold,
            floor_fraction,
            G=G,
            BLOCK_G=triton.next_power_of_2(G),
            BLOCK_C=triton.next_power_of_2(max(1, n_forced)),
            BLOCK_P=bp_sel,
            ITERS=_rounds(20, TUNE["search_radix"]),
            RADIX=TUNE["search_radix"],
            SORT=bp_sel <= TUNE["sort_max_blocks"],
            num_warps=_warps(TUNE["final_warps"], bp_sel),
        )
    else:
        bp_sel = triton.next_power_of_2(max(max_cand, cap))
        _final_select_kernel[(B * Hkv,)](
            lse,
            cand,
            ncand,
            nforced,
            idx,
            counts,
            kept,
            lse.stride(0),
            lse.stride(1),
            cand.stride(0),
            cand.stride(1),
            idx.stride(0),
            idx.stride(1),
            Hkv,
            cap,
            cdf_threshold,
            floor_fraction,
            G=G,
            BLOCK_P=bp_sel,  # candidate slots, not P
            ITERS=_rounds(20, TUNE["search_radix"]),
            RADIX=TUNE["search_radix"],
            SORT=bp_sel <= TUNE["sort_max_blocks"],
            num_warps=_warps(TUNE["final_warps"], bp_sel),
        )
    if return_aux:
        aux = dict(
            s2=scores2,
            has_s2=has_s2,
            isc=isc,
            lse=lse,
            pk=pk,
            cand=cand,
            ncand=ncand,
            kept=kept,
            max_cand=max_cand,
        )
        return idx, counts, aux
    return idx, counts


@torch.no_grad()
def select_blocks_two_stage_ref(
    q,
    kmin,
    kmax,
    k8,
    kscale,
    block_size,
    cdf_threshold,
    floor_fraction,
    max_fraction,
    prefilter_fraction=0.25,
    sink_blocks=8,
    local_blocks=8,
    seq_len=None,
    prefilter_shared=True,
    ksum=None,
    prefilter_mean=False,
    return_cand=False,
    region=0,
    region_alpha=0.5,
):
    """Readable torch reference of select_blocks_two_stage (same block sets), for tests.

    region / region_alpha: the shared prefilter's per-region quotas (TUNE prefilter_region
    and prefilter_region_alpha); 0 is the global top.
    """
    B, Hq, D = q.shape
    Hkv, P = kmin.shape[1], kmin.shape[2]
    G = Hq // Hkv
    n = int(seq_len) if seq_len is not None else min(P * block_size, k8.shape[2])
    cur_p = (n + block_size - 1) // block_size
    blocks = torch.arange(P, device=q.device)
    valid = blocks < cur_p
    forced = valid & ((blocks < sink_blocks) | (blocks >= cur_p - local_blocks))
    free = valid & ~forced
    qg = q.view(B, Hkv, G, D).float()
    bound = qg.relu() @ kmax.float().transpose(-1, -2) + (
        qg - qg.relu()
    ) @ kmin.float().transpose(-1, -2)
    n_keep = math.ceil(int(free.sum()) * prefilter_fraction)
    ranked = bound.masked_fill(~valid, float("-inf"))
    if prefilter_mean:
        s2 = _mean_key_scores_ref(q, ksum, block_size, n).masked_fill(
            ~valid, float("-inf")
        )
        ranked = torch.logaddexp(
            torch.log_softmax(ranked, -1), torch.log_softmax(s2, -1)
        ) - math.log(2.0)
    if prefilter_shared:
        shared = (
            torch.softmax(ranked, -1).mean(2).masked_fill(~free, -1.0)
        )  # [B, Hkv, P]
        if region:
            # ranked in log space like the kernel: tail probabilities underflow in fp32
            lshared = torch.logsumexp(torch.log_softmax(ranked, -1), 2).masked_fill(
                ~free, float("-inf")
            )
            top = _region_top_ref(
                shared,
                lshared,
                cur_p,
                n_keep,
                region,
                region_alpha,
                sink_blocks,
                local_blocks,
            )
        else:
            top = torch.zeros_like(shared, dtype=torch.bool).scatter(
                -1, shared.topk(n_keep, dim=-1).indices, True
            )
    else:
        ranked = ranked.masked_fill(~free, float("-inf"))
        top = (
            torch.zeros_like(ranked, dtype=torch.bool)
            .scatter(-1, ranked.topk(n_keep, dim=-1).indices, True)
            .any(2)
        )
    cand = forced | (top & free)  # [B, Hkv, P]
    # Like the kernel: q in q.dtype times the int8 keys, then each token's scale.
    tok_logits = (
        torch.einsum("bhgd,bhnd->bhgn", qg, k8[:, :, :n].float())
        * kscale[:, :, None, :n]
    )
    pad = cur_p * block_size - n
    tok_logits = torch.nn.functional.pad(tok_logits, (0, pad), value=float("-inf"))
    mass = torch.logsumexp(tok_logits.view(B, Hkv, G, cur_p, block_size), -1)
    mass = torch.nn.functional.pad(mass, (0, P - cur_p), value=float("-inf"))
    probs = torch.softmax(mass.masked_fill(~cand.unsqueeze(2), float("-inf")), -1).mean(
        2
    )  # [B, Hkv, P]
    cap = min(
        P, min(P, sink_blocks + local_blocks) + max(1, math.ceil(max_fraction * P))
    )
    idx = torch.zeros(B, Hkv, cap, dtype=torch.int32, device=q.device)
    counts = torch.zeros(B, Hkv, dtype=torch.int32, device=q.device)
    for bi in range(B):
        for h in range(Hkv):
            c = cand[bi, h]
            f_ids = torch.nonzero(forced & c).flatten().tolist()
            free_ids = torch.nonzero(c & ~forced).flatten()
            order = free_ids[probs[bi, h, free_ids].argsort(descending=True)].tolist()
            k_floor = math.ceil(int(c.sum()) * floor_fraction)
            acc = float(probs[bi, h, f_ids].sum())
            chosen = []
            for r, p in enumerate(order):
                if len(f_ids) + len(chosen) >= cap:
                    break
                if acc < cdf_threshold or r < k_floor:
                    chosen.append(p)
                    acc += float(probs[bi, h, p])
                else:
                    break
            sel = f_ids + chosen
            idx[bi, h, : len(sel)] = torch.tensor(sel, dtype=torch.int32)
            counts[bi, h] = len(sel)
    if return_cand:
        return idx, counts, cand
    return idx, counts


def _region_top_ref(shared, lshared, cur_p, n_keep, region, alpha, sink, local):
    """Per-region top sets of the shared prefilter, with the kernel's quotas: each region's
    share mixes its free-block share (weight alpha) and its ranking-probability share
    (from shared), rounded cumulatively from n_keep and capped at its free blocks; blocks
    are ranked within a region by lshared, the log of the same probabilities."""
    starts = list(range(0, cur_p, region))
    nf = torch.tensor(
        [
            max(0, min(min(lo + region, cur_p), cur_p - local) - sink)
            - max(0, min(lo, cur_p - local) - sink)
            for lo in starts
        ],
        device=shared.device,
    )
    mass = torch.stack(
        [
            shared[..., lo : min(lo + region, cur_p)].clamp_min(0).sum(-1)
            for lo in starts
        ],
        -1,
    )  # [B, Hkv, regions]
    w = (nf / nf.sum().clamp_min(1)).expand_as(mass)
    msum = mass.sum(-1, keepdim=True)
    w = torch.where(msum > 0, alpha * w + (1 - alpha) * mass / msum, w)
    hr = torch.ceil(torch.cumsum(w, -1) * n_keep).clamp(max=n_keep).long()
    k = (hr - torch.nn.functional.pad(hr, (1, 0))[..., :-1]).minimum(nf).clamp_min(0)
    top = torch.zeros_like(shared, dtype=torch.bool)
    for r, lo in enumerate(starts):
        part = lshared[..., lo : min(lo + region, cur_p)]
        rank = part.argsort(-1, descending=True).argsort(-1)
        top[..., lo : lo + part.shape[-1]] = rank < k[..., r : r + 1]
    return top


def _block_token_counts(P, block_size, n, device):
    starts = torch.arange(P, device=device) * block_size
    return (n - starts).clamp(min=1, max=block_size).float()


def _mean_key_scores_ref(q, ksum, block_size, n):
    """q . block-mean key per query head, rounded like the kernel -> [B, Hkv, G, P]."""
    B, Hq, D = q.shape
    Hkv, P = ksum.shape[1], ksum.shape[2]
    cnt = _block_token_counts(P, block_size, n, q.device)
    kmean = (ksum / cnt[:, None]).to(q.dtype).float()
    return q.view(B, Hkv, Hq // Hkv, D).float() @ kmean.transpose(-1, -2)


def two_stage_tail_ref(
    q,
    k,
    v,
    k8,
    kscale,
    ksum,
    vsum,
    block_idx,
    counts,
    cand,
    block_size,
    rho=0.6,
    seq_len=None,
    t1=True,
    t2=True,
):
    """Torch reference of block_sparse_decode_fused(..., tail=...): attention over the
    selected blocks plus every unselected block as one block-mean pseudo-token."""
    B, Hq, D = q.shape
    Hkv, P = ksum.shape[1], ksum.shape[2]
    G = Hq // Hkv
    n = int(seq_len) if seq_len is not None else k.shape[2]
    cur_p = (n + block_size - 1) // block_size
    pad = cur_p * block_size - n
    ninf = float("-inf")
    slot = torch.arange(block_idx.shape[-1], device=q.device)
    sel = (
        torch.zeros(B, Hkv, P, dtype=torch.int32, device=q.device).scatter_add_(
            -1, block_idx.long(), (slot < counts.unsqueeze(-1)).int()
        )
        > 0
    )
    sel, cand = sel[..., :cur_p], cand[..., :cur_p]
    qg = q.view(B, Hkv, G, D).float()

    def blocked(x):  # [B, Hkv, G, n] -> [B, Hkv, G, cur_p, bs], -inf padded
        return torch.nn.functional.pad(x, (0, pad), value=ninf).view(
            B, Hkv, G, cur_p, block_size
        )

    lt = blocked(torch.einsum("bhgd,bhnd->bhgn", qg, k[:, :, :n].float()))
    lt = lt.masked_fill(~sel[:, :, None, :, None], ninf)
    mx = lt.amax((-1, -2))  # [B, Hkv, G]
    e = torch.exp(lt - mx[..., None, None])
    vb = torch.nn.functional.pad(v[:, :, :n].float(), (0, 0, 0, pad)).view(
        B, Hkv, cur_p, block_size, D
    )
    num = torch.einsum("bhgpt,bhptd->bhgd", e, vb)
    den = e.sum((-1, -2))
    l8 = blocked(
        torch.einsum("bhgd,bhnd->bhgn", qg, k8[:, :, :n].float())
        * kscale[:, :, None, :n]
    )
    lse = torch.logsumexp(l8, -1)
    peak = l8.amax(-1) - lse
    cnt = _block_token_counts(cur_p, block_size, n, q.device)
    vmean = vsum[:, :, :cur_p] / cnt[:, None]
    w = torch.zeros_like(lse)
    if t1:
        use1 = (cand & ~sel)[:, :, None] & (peak < math.log(rho))
        w = w + torch.where(use1, torch.exp(lse - mx[..., None]), 0.0)
    if t2:
        s2 = _mean_key_scores_ref(q, ksum, block_size, n)[..., :cur_p]
        use2 = (~cand)[:, :, None].expand_as(s2)
        w = w + torch.where(use2, torch.exp(s2 + torch.log(cnt) - mx[..., None]), 0.0)
    out = (num + w @ vmean) / (den + w.sum(-1))[..., None]
    return out.view(B, Hq, D)


@triton.jit
def _append_q8_kernel(
    KNEW,
    K8,
    KS,
    POS,
    stride_nb,
    stride_nh,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_sb,
    stride_sh,
    HKV,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = (pid // HKV).to(tl.int64)
    h = pid % HKV
    offs_d = tl.arange(0, D)
    pos = tl.load(POS)
    k = tl.load(KNEW + b * stride_nb + h * stride_nh + offs_d).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(k), axis=0), 1e-8) / 127.0
    # div_rn: IEEE division, so rounding matches quantize_keys bit for bit.
    q8 = tl.minimum(
        tl.maximum(
            tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(k, scale)),
            -127.0,
        ),
        127.0,
    )
    tl.store(
        K8 + b * stride_kb + h * stride_kh + pos * stride_kn + offs_d, q8.to(tl.int8)
    )
    tl.store(KS + b * stride_sb + h * stride_sh + pos, scale)


def append_k8(
    k_new: torch.Tensor, k8: torch.Tensor, kscale: torch.Tensor, pos: torch.Tensor
) -> None:
    """Quantize one decode token's keys ([B, Hkv, D] or [B, Hkv, 1, D]) into the int8 copy."""
    B, Hkv = k8.shape[0], k8.shape[1]
    k_new = k_new.reshape(B, Hkv, -1).contiguous()
    _append_q8_kernel[(B * Hkv,)](
        k_new,
        k8,
        kscale,
        pos,
        k_new.stride(0),
        k_new.stride(1),
        k8.stride(0),
        k8.stride(1),
        k8.stride(2),
        kscale.stride(0),
        kscale.stride(1),
        Hkv,
        D=k8.shape[-1],
    )
