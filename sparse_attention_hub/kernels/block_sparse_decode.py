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
def _block_scores_kernel(
    Q,
    KMIN,
    KMAX,
    S,
    KSUM,
    S2,
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
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_P: tl.constexpr,
    D: tl.constexpr,
    MEAN: tl.constexpr = False,
):
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
    tl.store(
        S + b * stride_sb + (h * G + offs_m)[:, None] * stride_sh + offs_p[None, :],
        s,
        mask=row_ok[:, None] & (offs_p[None, :] < P),
    )
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
        kmin,
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
    if num_splits is None:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = max(
            1, min(triton.cdiv(kmax_sel, 4), triton.cdiv(2 * sms, B * Hkv))
        )
    bpi = max(1, 64 // block_size)
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
        num_warps=4,
        num_stages=3,
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
    S2,
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
    G: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ITERS: tl.constexpr,
    SHARED: tl.constexpr,
    MEAN: tl.constexpr = False,
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
    union = tl.zeros([BLOCK_P], tl.int1)
    lmean = tl.full([BLOCK_P], float("-inf"), tl.float32)
    for g in range(G):
        s = tl.load(
            S + b * stride_sb + (h * G + g) * stride_sh + offs_p,
            mask=valid,
            other=float("-inf"),
        )
        mx = tl.max(s, axis=0)
        lpr = (
            s - mx - tl.log(tl.sum(tl.exp(s - mx), axis=0))
        )  # log-softmax, -inf if invalid
        if MEAN:
            s2 = tl.load(
                S2 + b * stride_sb + (h * G + g) * stride_sh + offs_p,
                mask=valid,
                other=float("-inf"),
            )
            mx2 = tl.max(s2, axis=0)
            lpr2 = s2 - mx2 - tl.log(tl.sum(tl.exp(s2 - mx2), axis=0))
            lpr = _logaddexp(lpr, lpr2) - 0.6931471805599453
        if SHARED:
            lmean = _logaddexp(lmean, lpr)
        else:
            # Wide range: a block far down the bound's ranking can still hold real mass.
            hi = tl.max(tl.where(free, lpr, -1e30), axis=0)
            lo = hi - 1e4
            for _ in range(ITERS):
                mid = (lo + hi) * 0.5
                ok = tl.sum((free & (lpr >= mid)).to(tl.float32), axis=0) >= n_keep
                lo = tl.where(ok, mid, lo)
                hi = tl.where(ok, hi, mid)
            union = union | (free & (lpr >= lo))
    if SHARED:
        hi = tl.max(tl.where(free, lmean, -1e30), axis=0)
        lo = hi - 1e4
        for _ in range(ITERS):
            mid = (lo + hi) * 0.5
            ok = tl.sum((free & (lmean >= mid)).to(tl.float32), axis=0) >= n_keep
            lo = tl.where(ok, mid, lo)
            hi = tl.where(ok, hi, mid)
        union = free & (lmean >= lo)
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
    for j in range(start, end):
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
        k = (k8.to(tl.float32) * ks[:, None]).to(q.dtype)
        s = tl.dot(q, tl.trans(k))
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
    keep = forced | chosen
    slot = tl.cumsum(keep.to(tl.int32), axis=0) - 1  # forced occupy the leading slots
    blk = tl.load(CAND + b * stride_cb + h * stride_ch + offs, mask=valid, other=0)
    tl.store(IDX + b * stride_ib + h * stride_ih + slot, blk, mask=keep & (slot < CAP))
    tl.store(CNT + b * HKV + h, tl.minimum(tl.sum(keep.to(tl.int32), axis=0), CAP))
    tl.store(
        KEPT + b * stride_cb + h * stride_ch + offs,
        (keep & (slot < CAP)).to(tl.int8),
        mask=valid,
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
    _block_scores_kernel[(B * Hkv, triton.cdiv(P, 64))](
        q,
        kmin,
        kmax,
        scores,
        ksum if has_s2 else kmin,
        scores2,
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
        G=G,
        BLOCK_M=bm,
        BLOCK_P=64,
        D=D,
        MEAN=has_s2,
    )
    cand = torch.zeros(B, Hkv, P, device=dev, dtype=torch.int32)
    isc = torch.zeros(B, Hkv, P, device=dev, dtype=torch.int8)
    ncand = torch.empty(B, Hkv, device=dev, dtype=torch.int32)
    nforced = torch.empty_like(ncand)
    _prefilter_kernel[(B * Hkv,)](
        scores,
        scores2,
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
        G=G,
        BLOCK_P=bp,
        ITERS=30,
        SHARED=prefilter_shared,
        MEAN=prefilter_mean,
        num_warps=8,
    )
    lse = torch.full((B, Hq, P), float("-inf"), device=dev, dtype=torch.float32)
    pk = torch.zeros_like(lse)
    # Size the grid by the candidate bound, not P: ~4 candidate blocks per program.
    per_head = math.ceil(prefilter_fraction * P)
    max_cand = min(
        P, sink_blocks + local_blocks + (per_head if prefilter_shared else G * per_head)
    )
    slots = 4
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
        num_warps=4,
    )
    cap = min(
        P, min(P, sink_blocks + local_blocks) + max(1, math.ceil(max_fraction * P))
    )
    idx = torch.zeros(B, Hkv, cap, device=dev, dtype=torch.int32)
    counts = torch.empty(B, Hkv, device=dev, dtype=torch.int32)
    kept = torch.zeros(B, Hkv, P, device=dev, dtype=torch.int8)
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
        BLOCK_P=triton.next_power_of_2(max_cand),  # candidate slots, not P
        ITERS=20,
        num_warps=8,
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
):
    """Readable torch reference of select_blocks_two_stage (same block sets), for tests."""
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
    # Round like the kernel: dequantized keys and q in q.dtype, fp32 accumulation.
    kd = (k8[:, :, :n].float() * kscale[:, :, :n, None]).to(q.dtype).float()
    tok_logits = torch.einsum("bhgd,bhnd->bhgn", qg, kd)
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
    kd = (k8[:, :, :n].float() * kscale[:, :, :n, None]).to(q.dtype).float()
    l8 = blocked(torch.einsum("bhgd,bhnd->bhgn", qg, kd))
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
