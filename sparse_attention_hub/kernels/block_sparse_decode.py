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
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
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


def append_kv(
    k_new, v_new, k_buf, v_buf, pos, kmin=None, kmax=None, block_size: int = 64
) -> None:
    """Write one decode token's K/V ([B, Hkv, D] or [B, Hkv, 1, D]) at position pos (device
    scalar) and, when kmin/kmax are given, fold it into its block's channel min/max."""
    B, Hkv = k_buf.shape[0], k_buf.shape[1]
    k_new = k_new.reshape(B, Hkv, -1).contiguous()
    v_new = v_new.reshape(B, Hkv, -1).contiguous()
    meta = kmin is not None
    if not meta:
        kmin = kmax = k_buf
    _append_kernel[(B * Hkv,)](
        k_new,
        v_new,
        k_buf,
        v_buf,
        kmin,
        kmax,
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


def block_sparse_decode_fused(
    q, k, v, block_idx, counts, block_size, num_splits=None, seq_len=None
):
    """block_sparse_decode with the split merge as one Triton kernel instead of torch ops."""
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
    acc = torch.empty(B, Hq, num_splits, D, device=q.device, dtype=torch.float32)
    m = torch.empty(B, Hq, num_splits, device=q.device, dtype=torch.float32)
    l_sum = torch.empty_like(m)
    if seq_len is None:
        seq_len = torch.full((1,), k.shape[2], device=q.device, dtype=torch.int32)
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
        NS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
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
    CAND,
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
):
    # SHARED: one top set per KV head from the group-mean bound softmax; otherwise the
    # union of every query head's own top set.
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


@triton.jit
def _block_mass_kernel(
    Q,
    K8,
    KS,
    CAND,
    NCAND,
    L,
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


@triton.jit
def _final_select_kernel(
    L,
    CAND,
    NCAND,
    NFORCED,
    IDX,
    CNT,
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bound prefilter -> int8 block mass on candidates -> TopCdf + floor, per KV head.

    q: [B, Hq, D] scaled; kmin/kmax: [B, Hkv, P, D]; k8: [B, Hkv, N, D] int8 and kscale
    [B, Hkv, N] from quantize_keys. Returns block_idx [B, Hkv, cap] int32 and counts.
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
    _block_scores_kernel[(B * Hkv, triton.cdiv(P, 64))](
        q,
        kmin,
        kmax,
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
        BLOCK_M=bm,
        BLOCK_P=64,
        D=D,
    )
    cand = torch.zeros(B, Hkv, P, device=dev, dtype=torch.int32)
    ncand = torch.empty(B, Hkv, device=dev, dtype=torch.int32)
    nforced = torch.empty_like(ncand)
    _prefilter_kernel[(B * Hkv,)](
        scores,
        cand,
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
        num_warps=8,
    )
    lse = torch.full((B, Hq, P), float("-inf"), device=dev, dtype=torch.float32)
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
    _final_select_kernel[(B * Hkv,)](
        lse,
        cand,
        ncand,
        nforced,
        idx,
        counts,
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
        BLOCK_P=bp,
        ITERS=20,
        num_warps=8,
    )
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
    return idx, counts


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
