"""Tests for the SpargeAttn Top-CDF masker."""

import math

import pytest
import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    ResearchMasker,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    SpargeTopCdfMasker,
    SpargeTopCdfMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask


def _run(masker, queries, keys, attention_mask=None, previous_mask=None):
    B, H, Q, _ = queries.shape
    K = keys.shape[2]
    if previous_mask is None:
        previous_mask = Mask.create_empty_mask(
            (B, H, Q, K), dtype=torch.float32, device=queries.device
        )
    return masker.add_mask(
        keys=keys,
        queries=queries,
        values=keys,
        attention_mask=attention_mask,
        scaling=1.0 / math.sqrt(queries.shape[-1]),
        dropout=0.0,
        sparse_meta_data={},
        previous_mask=previous_mask,
    ).get_dense_mask()


def _reference_blocks(queries, keys, scaling, cfg):
    """Loop reference of block selection for one (b, h) at a time."""
    B, H, Q, D = queries.shape
    K = keys.shape[2]
    bs = cfg.block_size
    P = (K + bs - 1) // bs
    out = torch.zeros(B, H, Q, P, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            kh = keys[b, h].double()
            ks = kh - kh.mean(0) if cfg.smooth_k else kh
            means, sims, counts, varis, mins, maxs = [], [], [], [], [], []
            for p in range(P):
                blk = kh[p * bs : (p + 1) * bs]
                sblk = ks[p * bs : (p + 1) * bs]
                unit = sblk / sblk.norm(dim=-1, keepdim=True)
                means.append(blk.mean(0))
                varis.append(blk.var(0, unbiased=False))
                mins.append(blk.amin(0))
                maxs.append(blk.amax(0))
                sims.append((unit @ unit.T).mean().item())
                counts.append(blk.shape[0])
            for q in range(Q):
                forced = [s < cfg.sim_threshold for s in sims]
                cand = [p for p in range(P) if not forced[p]]
                qv = queries[b, h, q].double()
                if cfg.score == "exact":
                    scores = {
                        p: torch.logsumexp(
                            kh[p * bs : (p + 1) * bs] @ qv * scaling, 0
                        ).item()
                        for p in cand
                    }
                base = {
                    p: (
                        torch.maximum(qv * mins[p], qv * maxs[p]).sum().item()
                        if cfg.score == "quest_bound"
                        else (qv @ means[p]).item()
                    )
                    for p in cand
                }
                scores = (
                    scores
                    if cfg.score == "exact"
                    else {
                        p: base[p] * scaling
                        + (math.log(counts[p]) if cfg.size_weighted else 0.0)
                        + (
                            float(cfg.variance_correction)
                            * 0.5
                            * scaling**2
                            * (queries[b, h, q].double() ** 2 @ varis[p]).item()
                            if cfg.variance_correction
                            else 0.0
                        )
                        for p in cand
                    }
                )
                if cand:
                    mx = max(scores.values())
                    z = sum(math.exp(s - mx) for s in scores.values())
                    n_topk = math.ceil(len(cand) * cfg.topk_fraction)
                    acc = 0.0
                    for r, p in enumerate(sorted(cand, key=lambda p: -scores[p])):
                        mass = math.exp(scores[p] - mx) / z
                        if cfg.include_crossing:
                            in_cdf = acc < cfg.cdf_threshold
                        else:
                            in_cdf = r == 0 or acc + mass <= cfg.cdf_threshold
                        if in_cdf or r < n_topk:
                            out[b, h, q, p] = True
                        acc += mass
                for p in range(P):
                    if forced[p]:
                        out[b, h, q, p] = True
    return out


def _blocks_of(dense, bs):
    K = dense.shape[-1]
    P = (K + bs - 1) // bs
    pad = P * bs - K
    d = torch.nn.functional.pad(dense, (0, pad))
    return d.view(*dense.shape[:-1], P, bs).amax(-1) > 0


class TestSpargeTopCdfMaskerConfig:
    def test_defaults_and_registry(self):
        cfg = SpargeTopCdfMaskerConfig()
        assert isinstance(
            ResearchMasker.create_masker_from_config(cfg), SpargeTopCdfMasker
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"block_size": 0},
            {"cdf_threshold": 0.0},
            {"cdf_threshold": 1.5},
            {"sim_threshold": 2.0},
            {"score": "max"},
            {"prefilter_fraction": 1.5},
            {"key_bits": 1},
            {"max_fraction": 0.0},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            SpargeTopCdfMaskerConfig(**kwargs)


class TestSpargeTopCdfMasker:
    def test_short_sequence_is_full(self):
        masker = SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(block_size=64))
        q = torch.randn(1, 2, 1, 16)
        k = torch.randn(1, 2, 64, 16)
        assert torch.all(_run(masker, q, k) == 1)

    def test_full_previous_mask_passthrough(self):
        masker = SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(block_size=8))
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 64, 8)
        full = Mask.create_full_mask(
            (1, 1, 1, 64), dtype=torch.float32, device=q.device
        )
        assert torch.all(_run(masker, q, k, previous_mask=full) == 1)

    @pytest.mark.parametrize("size_weighted", [True, False])
    @pytest.mark.parametrize("sim", [-1.0, 0.05, 0.2])
    @pytest.mark.parametrize("cdf", [0.3, 0.7, 0.95])
    @pytest.mark.parametrize(
        "smooth_k,include_crossing,topk_fraction,variance_correction,score",
        [
            (False, True, 0.0, False, "mean"),
            (True, True, 0.0, False, "mean"),
            (True, False, 0.0, False, "mean"),
            (True, False, 0.3, False, "mean"),
            (True, True, 0.0, True, "mean"),
            (False, True, 0.3, True, "mean"),
            (True, True, 0.0, 0.5, "mean"),
            (True, True, 0.0, False, "quest_bound"),
            (True, True, 0.3, False, "quest_bound"),
            (True, True, 0.0, True, "exact"),
            (True, False, 0.3, False, "exact"),
        ],
    )
    def test_matches_loop_reference(
        self,
        size_weighted,
        sim,
        cdf,
        smooth_k,
        include_crossing,
        topk_fraction,
        variance_correction,
        score,
    ):
        torch.manual_seed(0)
        cfg = SpargeTopCdfMaskerConfig(
            block_size=8,
            cdf_threshold=cdf,
            sim_threshold=sim,
            size_weighted=size_weighted,
            smooth_k=smooth_k,
            include_crossing=include_crossing,
            topk_fraction=topk_fraction,
            variance_correction=variance_correction,
            score=score,
        )
        q = torch.randn(2, 3, 4, 16)
        # Mix self-similar blocks (shared direction + noise) with random ones.
        base = torch.randn(2, 3, 10, 1, 16)
        k = (base * torch.rand(2, 3, 10, 1, 1) * 3 + torch.randn(2, 3, 10, 8, 16)).view(
            2, 3, 80, 16
        )[
            :, :, :77
        ]  # ragged tail block
        got = _blocks_of(_run(SpargeTopCdfMasker(cfg), q, k), 8)
        want = _reference_blocks(q, k, 1.0 / 4.0, cfg)
        assert torch.equal(got, want)

    def test_cdf_one_selects_everything(self):
        cfg = SpargeTopCdfMaskerConfig(
            block_size=8, cdf_threshold=1.0, sim_threshold=-1.0
        )
        q = torch.randn(1, 2, 3, 16)
        k = torch.randn(1, 2, 70, 16)
        assert torch.all(_run(SpargeTopCdfMasker(cfg), q, k) == 1)

    def test_concentrated_attention_picks_one_block(self):
        D, bs = 32, 16
        direction = torch.randn(D)
        direction /= direction.norm()
        # Every block shares a weak common direction so all are self-similar.
        k = 0.1 * torch.randn(1, 1, 8 * bs, D) + direction
        k[0, 0, 3 * bs : 4 * bs] += 6.0 * direction
        q = (8.0 * direction).view(1, 1, 1, D)
        cfg = SpargeTopCdfMaskerConfig(
            block_size=bs, cdf_threshold=0.9, sim_threshold=0.5
        )
        sel = _blocks_of(_run(SpargeTopCdfMasker(cfg), q, k), bs)[0, 0, 0]
        assert sel.tolist() == [i == 3 for i in range(8)]

    def test_dissimilar_block_is_forced(self):
        D, bs = 32, 16
        direction = torch.randn(D)
        k = 0.05 * torch.randn(1, 1, 4 * bs, D) + direction
        k[0, 0, bs : 2 * bs] = torch.randn(bs, D) * 3  # incoherent block
        q = direction.view(1, 1, 1, D) * 0.0  # uniform compressed scores
        sel_on = _blocks_of(
            _run(
                SpargeTopCdfMasker(
                    SpargeTopCdfMaskerConfig(
                        block_size=bs, cdf_threshold=0.2, sim_threshold=0.5
                    )
                ),
                q,
                k,
            ),
            bs,
        )[0, 0, 0]
        assert sel_on[1]
        sel_off = _blocks_of(
            _run(
                SpargeTopCdfMasker(
                    SpargeTopCdfMaskerConfig(
                        block_size=bs, cdf_threshold=0.2, sim_threshold=-1.0
                    )
                ),
                q,
                k,
            ),
            bs,
        )[0, 0, 0]
        assert sel_off.sum() == 1  # four equal-mass blocks, 0.2 needs just one

    def test_attention_mask_blocks_never_selected(self):
        cfg = SpargeTopCdfMaskerConfig(
            block_size=8, cdf_threshold=1.0, sim_threshold=0.9
        )
        q = torch.randn(1, 2, 1, 16)
        k = torch.randn(1, 2, 64, 16)
        am = torch.zeros(1, 1, 1, 64)
        am[..., 40:] = float("-inf")
        dense = _run(SpargeTopCdfMasker(cfg), q, k, attention_mask=am)
        assert torch.all(dense[..., 40:] == 0)
        assert torch.all(dense[..., :40] == 1)

    @pytest.mark.parametrize("score", ["mean", "quest_bound"])
    def test_share_kv_groups(self, score):
        torch.manual_seed(0)
        kw = dict(
            block_size=8,
            cdf_threshold=0.8,
            sim_threshold=-1.0,
            score=score,
            topk_fraction=0.1,
        )
        shared = SpargeTopCdfMasker(
            SpargeTopCdfMaskerConfig(share_kv_groups=True, **kw)
        )
        plain = SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(**kw))
        q = torch.randn(1, 8, 2, 16)  # 4 query heads per KV head
        k = torch.randn(1, 2, 100, 16)
        got = _run(shared, q, k).view(1, 2, 4, 2, 100)
        assert torch.all(got == got[:, :, :1])  # one selection per KV head
        # With identical queries across each group, sharing changes nothing.
        q_same = (
            q.view(1, 2, 4, 2, 16)[:, :, :1].expand(1, 2, 4, 2, 16).reshape(1, 8, 2, 16)
        )
        assert torch.equal(_run(shared, q_same, k), _run(plain, q_same, k))

    def test_prefilter_and_key_bits(self):
        torch.manual_seed(1)
        kw = dict(block_size=8, cdf_threshold=0.9, sim_threshold=-1.0, score="exact")
        q = torch.randn(1, 2, 3, 16)
        k = torch.randn(1, 2, 160, 16)
        plain = _run(SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(**kw)), q, k)
        # A prefilter that keeps every block, and 16-bit keys, change nothing.
        same = SpargeTopCdfMaskerConfig(prefilter_fraction=1.0, key_bits=16, **kw)
        assert torch.equal(_run(SpargeTopCdfMasker(same), q, k), plain)
        # Selection never leaves the bound's top 25% of blocks.
        pre = _blocks_of(
            _run(
                SpargeTopCdfMasker(
                    SpargeTopCdfMaskerConfig(prefilter_fraction=0.25, **kw)
                ),
                q,
                k,
            ),
            8,
        )
        bound = _blocks_of(
            _run(
                SpargeTopCdfMasker(
                    SpargeTopCdfMaskerConfig(
                        block_size=8,
                        cdf_threshold=1e-6,
                        sim_threshold=-1.0,
                        score="quest_bound",
                        topk_fraction=0.25,
                    )
                ),
                q,
                k,
            ),
            8,
        )
        assert torch.all(bound | ~pre)
        assert pre.sum() <= bound.sum()
        # 8-bit keys barely move block mass; 2-bit keys still give a valid mask.
        q8 = _run(SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(key_bits=8, **kw)), q, k)
        assert (q8 != plain).float().mean() < 0.1
        _run(SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(key_bits=2, **kw)), q, k)

    def test_prefilter_shared_and_cap(self):
        torch.manual_seed(2)
        kw = dict(
            block_size=8,
            cdf_threshold=0.95,
            sim_threshold=-1.0,
            score="exact",
            share_kv_groups=True,
            prefilter_fraction=0.25,
        )
        q = torch.randn(1, 8, 1, 16)
        k = torch.randn(1, 2, 400, 16)
        union = _blocks_of(
            _run(SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(**kw)), q, k), 8
        )
        shared = _blocks_of(
            _run(
                SpargeTopCdfMasker(
                    SpargeTopCdfMaskerConfig(prefilter_shared=True, **kw)
                ),
                q,
                k,
            ),
            8,
        )
        assert shared.sum() <= union.sum()  # one 25% set per KV head vs a union of four
        capped = _blocks_of(
            _run(
                SpargeTopCdfMasker(SpargeTopCdfMaskerConfig(max_fraction=0.05, **kw)),
                q,
                k,
            ),
            8,
        )
        assert int(capped.sum(-1).max()) <= math.ceil(50 * 0.05)

    def test_gqa_and_previous_mask_union(self):
        cfg = SpargeTopCdfMaskerConfig(
            block_size=8, cdf_threshold=0.1, sim_threshold=-1.0
        )
        q = torch.randn(1, 4, 2, 16)
        k = torch.randn(1, 2, 64, 16)
        prev = torch.zeros(1, 4, 2, 64)
        prev[..., :4] = 1.0
        prev_mask = Mask.create_mask_from_dense_mask(
            (1, 4, 2, 64), prev, dtype=torch.float32
        )
        dense = _run(SpargeTopCdfMasker(cfg), q, k, previous_mask=prev_mask)
        assert dense.shape == (1, 4, 2, 64)
        assert torch.all(dense[..., :4] == 1)
        assert torch.all(_blocks_of(dense, 8).sum(-1) >= 1)
