"""Tests for the TwoStageBlockSparse efficient attention."""

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from sparse_attention_hub.sparse_attention import (
    EfficientAttention,
    SparseAttention,
    TwoStageBlockSparse,
    TwoStageBlockSparseConfig,
)
from sparse_attention_hub.sparse_attention.efficient_attention.implementations.two_stage_block_sparse import (
    CARRY_KEY,
    META_KEY,
)
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_true_attention_output,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")


def _qkv(b, hq, hkv, n, d=128, lq=1, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, hq, lq, d, generator=g) / math.sqrt(d)
    k = torch.randn(b, hkv, n, d, generator=g)
    v = torch.randn(b, hkv, n, d, generator=g)
    return (t.to(device=device, dtype=dtype) for t in (q, k, v))


def _call(attn, q, k, v, meta, layer_idx=0, mask=None, **kw):
    return attn.custom_attention(
        module=nn.Module(),
        queries=q,
        keys=k,
        values=v,
        attention_mask=mask,
        scaling=1.0,
        dropout=0.0,
        sparse_meta_data=meta,
        layer_idx=layer_idx,
        **kw,
    )


def _kernel_reference(c, q, k, v, cap, carry_score=None):
    """The selection and decode kernels on fresh metadata for k/v (the live tokens) in
    buffers of capacity cap (the selection cap is a fraction of the capacity's blocks).
    carry_score: carried block scores to select with (updated in place).
    """
    from sparse_attention_hub.kernels.block_sparse_decode import (
        block_min_max,
        block_sparse_decode_fused,
        block_sums,
        quantize_keys,
        select_blocks_two_stage,
    )

    def padded(x, size):
        buf = x.new_zeros(*x.shape[:2], size, *x.shape[3:])
        buf[:, :, : x.shape[2]] = x
        return buf

    n = k.shape[2]
    kmin, kmax = (padded(x, cap // 64) for x in block_min_max(k, 64))
    k8, ks = quantize_keys(k)
    k8, ks = padded(k8, cap), padded(ks, cap)
    ksum = padded(block_sums(k, 64), cap // 64)
    vsum = padded(block_sums(v, 64), cap // 64)
    seq = torch.full((1,), n, device="cuda", dtype=torch.int32)
    sel = select_blocks_two_stage(
        q[:, :, 0].contiguous(),
        kmin,
        kmax,
        k8,
        ks,
        64,
        c.cdf_threshold,
        c.floor_fraction,
        c.max_fraction,
        c.prefilter_fraction,
        2,
        2,
        seq_len=seq,
        ksum=ksum,
        prefilter_mean=True,
        return_aux=c.tail,
        carry_score=carry_score,
        carry_threshold=c.carry_threshold,
    )
    return block_sparse_decode_fused(
        q[:, :, 0].contiguous(),
        k,
        v,
        sel[0],
        sel[1],
        64,
        seq_len=seq,
        tail=dict(aux=sel[2], vsum=vsum, rho=c.tail_rho) if c.tail else None,
    )


@pytest.mark.unit
class TestTwoStageBlockSparseConfig:
    def test_defaults(self):
        c = TwoStageBlockSparseConfig()
        assert (c.block_size, c.cdf_threshold, c.prefilter_fraction) == (64, 0.95, 0.25)
        assert c.prefilter_mean and not c.tail

    def test_factories(self):
        c = TwoStageBlockSparseConfig(cdf_threshold=0.9)
        for factory in (
            TwoStageBlockSparse.create_from_config,
            EfficientAttention.create_from_config,
            SparseAttention.create_from_config,
        ):
            attn = factory(c)
            assert isinstance(attn, TwoStageBlockSparse)
            assert attn.config.cdf_threshold == 0.9

    def test_wrong_config(self):
        from sparse_attention_hub.sparse_attention import HashAttentionConfig

        with pytest.raises(TypeError):
            TwoStageBlockSparse.create_from_config(
                HashAttentionConfig(
                    heavy_size=1,
                    sink_size=1,
                    local_size=1,
                    hat_weights={},
                    hat_bits=1,
                    hat_mlp_layers=1,
                    hat_mlp_hidden_size=1,
                )
            )

    def test_requires_sparse_meta_data(self):
        attn = TwoStageBlockSparse.create_from_config(TwoStageBlockSparseConfig())
        q, k, v = _qkv(1, 4, 2, 32)
        with pytest.raises(ValueError):
            attn.custom_attention(nn.Module(), q, k, v, None, 1.0, 0.0)


@pytest.mark.unit
class TestDenseFallback:
    @pytest.mark.parametrize("lq", [1, 5])
    def test_cpu_and_multi_query_are_dense(self, lq):
        attn = TwoStageBlockSparse.create_from_config(TwoStageBlockSparseConfig())
        q, k, v = _qkv(2, 8, 2, 300, lq=lq)
        out, w = _call(attn, q, k, v, {})
        ref, _ = get_true_attention_output(nn.Module(), q, k, v, None, 1.0, 0.0)
        assert w is None and out.shape == (2, lq, 8, 128)
        torch.testing.assert_close(out, ref)


@cuda
@pytest.mark.unit
class TestKernelPath:
    def test_every_block_matches_dense(self):
        # Prefilter, TopCdf and cap all at 100%: every block is attended.
        attn = TwoStageBlockSparse.create_from_config(
            TwoStageBlockSparseConfig(
                cdf_threshold=1.0,
                floor_fraction=0.0,
                max_fraction=1.0,
                prefilter_fraction=1.0,
            )
        )
        q, k, v = _qkv(2, 16, 4, 4000, device="cuda", dtype=torch.bfloat16)
        out, _ = _call(attn, q, k, v, {})
        ref, _ = get_true_attention_output(
            nn.Module(), q.float(), k.float(), v.float(), None, 1.0, 0.0
        )
        torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=0)

    @pytest.mark.parametrize("tail", [False, True])
    def test_matches_kernels_on_fresh_metadata(self, tail):
        c = TwoStageBlockSparseConfig(tail=tail)
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v = _qkv(1, 32, 8, 20000, device="cuda", dtype=torch.bfloat16)
        meta = {}
        out, _ = _call(attn, q, k, v, meta)
        ref = _kernel_reference(c, q, k, v, meta[META_KEY][0]["k8"].shape[2])
        torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)

    def test_layer_prefilter_fraction(self):
        c = TwoStageBlockSparseConfig(layer_prefilter_fraction={3: 0.75})
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v = _qkv(1, 32, 8, 20000, device="cuda", dtype=torch.bfloat16)
        meta, refs = {}, []
        for layer, frac in ((0, 0.25), (3, 0.75)):
            out, _ = _call(attn, q, k, v, meta, layer_idx=layer)
            cap = meta[META_KEY][layer]["k8"].shape[2]
            refs.append(
                _kernel_reference(replace(c, prefilter_fraction=frac), q, k, v, cap)
            )
            torch.testing.assert_close(out[:, 0], refs[-1], atol=2e-3, rtol=0)
        # The budgets select differently here, so the test sees which one was used.
        assert (refs[0] - refs[1]).abs().max() > 1e-2

    def test_metadata_follows_the_cache(self):
        from sparse_attention_hub.kernels.block_sparse_decode import (
            block_min_max,
            block_sums,
            quantize_keys,
        )

        attn = TwoStageBlockSparse.create_from_config(TwoStageBlockSparseConfig())
        q, k, v = _qkv(1, 8, 2, 3000, device="cuda", dtype=torch.bfloat16)
        meta = {}
        # Prefill-sized first call, then single tokens (the append kernels) up to and
        # across a block boundary (1088), a partial block, and past the first capacity.
        for n in (1000, 1001, 1037, 1087, 1088, 1089, 1130, 2100, 2101, 3000):
            _call(attn, q, k[:, :, :n], v[:, :, :n], meta, layer_idx=3)
            st = meta[META_KEY][3]
            p = math.ceil(n / 64)
            kmin, kmax = block_min_max(k[:, :, :n], 64)
            k8, ks = quantize_keys(k[:, :, :n])
            assert st["n"] == n
            assert torch.equal(st["kmin"][:, :, :p], kmin)
            assert torch.equal(st["kmax"][:, :, :p], kmax)
            torch.testing.assert_close(
                st["ksum"][:, :, :p], block_sums(k[:, :, :n], 64)
            )
            torch.testing.assert_close(
                st["vsum"][:, :, :p], block_sums(v[:, :, :n], 64)
            )
            assert torch.equal(st["k8"][:, :, :n], k8)
            assert torch.equal(st["ks"][:, :, :n], ks)

    def test_layers_and_new_sequences_are_separate(self):
        attn = TwoStageBlockSparse.create_from_config(TwoStageBlockSparseConfig())
        q, k, v = _qkv(1, 8, 2, 2000, device="cuda", dtype=torch.bfloat16)
        meta = {}
        _call(attn, q, k, v, meta, layer_idx=0)
        _call(attn, q, k[:, :, :500], v[:, :, :500], meta, layer_idx=1)
        assert meta[META_KEY][0]["n"] == 2000 and meta[META_KEY][1]["n"] == 500
        # A shorter cache on a used layer is a new sequence: metadata is rebuilt.
        _call(attn, q, k[:, :, :700], v[:, :, :700], meta, layer_idx=0)
        assert meta[META_KEY][0]["n"] == 700


def _static_step(attn, q, kbuf, vbuf, k, v, meta, pos):
    """One decode step on a static cache: write token pos (as HF's StaticCache does), then
    attend with the position on the device."""
    kbuf.index_copy_(2, pos, k.index_select(2, pos))
    vbuf.index_copy_(2, pos, v.index_select(2, pos))
    return _call(attn, q, kbuf, vbuf, meta, position_ids=pos.view(1, 1))[0]


@cuda
@pytest.mark.unit
class TestStaticCache:
    """A preallocated cache longer than the sequence: device-side positions, graph-safe."""

    L, N0 = 3000, 1080  # buffer length; tokens before the first decode step

    def _setup(self, seed=0):
        q, k, v = _qkv(2, 16, 4, self.L, device="cuda", dtype=torch.bfloat16, seed=seed)
        kbuf, vbuf = torch.zeros_like(k), torch.zeros_like(v)
        kbuf[:, :, : self.N0], vbuf[:, :, : self.N0] = (
            k[:, :, : self.N0],
            v[:, :, : self.N0],
        )
        return q, k, v, kbuf, vbuf

    def test_steps_match_kernels_on_fresh_metadata(self):
        from sparse_attention_hub.kernels.block_sparse_decode import (
            block_min_max,
            block_sums,
            quantize_keys,
        )

        # Steps cross a block boundary (1088) and start a new block.
        c = TwoStageBlockSparseConfig()
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v, kbuf, vbuf = self._setup()
        meta = {}
        for n in range(self.N0 + 1, self.N0 + 12):
            pos = torch.tensor([n - 1], device="cuda")
            out = _static_step(attn, q, kbuf, vbuf, k, v, meta, pos)
            st = meta[META_KEY][0]
            assert st["static"] and st["k8"].shape[2] == math.ceil(self.L / 64) * 64
            ref = _kernel_reference(c, q, k[:, :, :n], v[:, :, :n], st["k8"].shape[2])
            torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)
            p = math.ceil(n / 64)
            kmin, kmax = block_min_max(k[:, :, :n], 64)
            k8, ks = quantize_keys(k[:, :, :n])
            assert int(st["seq_len"]) == n
            assert torch.equal(st["kmin"][:, :, :p], kmin)
            assert torch.equal(st["kmax"][:, :, :p], kmax)
            torch.testing.assert_close(
                st["ksum"][:, :, :p], block_sums(k[:, :, :n], 64)
            )
            torch.testing.assert_close(
                st["vsum"][:, :, :p], block_sums(v[:, :, :n], 64)
            )
            assert torch.equal(st["k8"][:, :, :n], k8)
            assert torch.equal(st["ks"][:, :, :n], ks)
        # The append reads the token from the cache and leaves the cache as written.
        assert torch.equal(kbuf[:, :, : self.N0 + 11], k[:, :, : self.N0 + 11])

    def test_multi_token_call_is_folded_in(self):
        c = TwoStageBlockSparseConfig()
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v, kbuf, vbuf = self._setup()
        meta = {}
        _static_step(
            attn, q, kbuf, vbuf, k, v, meta, torch.tensor([self.N0], device="cuda")
        )
        # A 40-token chunk (dense attention), then decode continues after it.
        chunk = torch.arange(self.N0 + 1, self.N0 + 41, device="cuda")
        kbuf[:, :, chunk], vbuf[:, :, chunk] = k[:, :, chunk], v[:, :, chunk]
        qc = q.expand(-1, -1, 40, -1)
        _call(attn, qc, kbuf, vbuf, meta, position_ids=chunk.view(1, -1))
        n = self.N0 + 42
        out = _static_step(
            attn, q, kbuf, vbuf, k, v, meta, torch.tensor([n - 1], device="cuda")
        )
        cap = meta[META_KEY][0]["k8"].shape[2]
        ref = _kernel_reference(c, q, k[:, :, :n], v[:, :, :n], cap)
        torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)

    @pytest.mark.parametrize("carry", [False, True])
    def test_cuda_graph_replay_matches_eager(self, carry):
        # carry: the carried scores (decayed and raised on the device) and a full-scan
        # layer replay like everything else.
        c = TwoStageBlockSparseConfig()
        if carry:
            c = TwoStageBlockSparseConfig(
                carry_threshold=0.05, layer_prefilter_fraction={0: 1.0}
            )
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v, kbuf, vbuf = self._setup()
        eager, scores, meta = [], [], {}
        for n in range(self.N0 + 1, self.N0 + 12):
            pos = torch.tensor([n - 1], device="cuda")
            eager.append(_static_step(attn, q, kbuf, vbuf, k, v, meta, pos))
            if carry:
                scores.append(meta[CARRY_KEY]["score"].clone())

        q, k, v, kbuf, vbuf = self._setup()
        meta = {}
        pos = torch.tensor([self.N0], device="cuda")
        first = _static_step(
            attn, q, kbuf, vbuf, k, v, meta, pos
        )  # eager: sets up state
        torch.testing.assert_close(first, eager[0], atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _static_step(attn, q, kbuf, vbuf, k, v, meta, pos)
        for i in range(1, len(eager)):
            pos.fill_(self.N0 + i)
            graph.replay()
            torch.testing.assert_close(out, eager[i], atol=0, rtol=0)
            if carry:
                assert torch.equal(meta[CARRY_KEY]["score"], scores[i])
        if carry:
            assert scores[-1].max() > 0.05


@cuda
@pytest.mark.unit
class TestMinCachedTokens:
    """Below min_cached_tokens (batch x length) decode steps run dense SDPA with GQA."""

    @staticmethod
    def _attn(min_cached_tokens):
        return TwoStageBlockSparse.create_from_config(
            TwoStageBlockSparseConfig(min_cached_tokens=min_cached_tokens)
        )

    @staticmethod
    def _dense(q, k, v):
        return get_true_attention_output(nn.Module(), q, k, v, None, 1.0, 0.0)[0]

    def test_growing_cache_is_dense_below_and_sparse_above(self):
        attn = self._attn(2 * 3000)  # batch 2: dense below 3000 tokens per row
        q, k, v = _qkv(2, 16, 4, 3200, device="cuda", dtype=torch.bfloat16)
        meta = {}
        out = _call(attn, q, k[:, :, :2999], v[:, :, :2999], meta)[0]
        dense = self._dense(q, k[:, :, :2999], v[:, :, :2999])
        torch.testing.assert_close(out, dense, atol=2e-3, rtol=0)
        assert "k8" not in meta[META_KEY][0]  # no metadata while dense
        out = _call(attn, q, k[:, :, :3100], v[:, :, :3100], meta)[0]
        st = meta[META_KEY][0]
        assert st["n"] == 3100
        ref = _kernel_reference(
            attn.config, q, k[:, :, :3100], v[:, :, :3100], st["k8"].shape[2]
        )
        torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)

    def test_static_cache_decides_at_its_first_decode_step(self):
        L, n0 = 3000, 1080
        q, k, v = _qkv(2, 16, 4, L, device="cuda", dtype=torch.bfloat16)
        for min_tokens, dense in ((2 * 2000, True), (2 * 1000, False)):
            attn = self._attn(min_tokens)
            kbuf, vbuf = torch.zeros_like(k), torch.zeros_like(v)
            meta = {}
            for n in (n0 + 1, n0 + 2):
                kbuf[:, :, :n], vbuf[:, :, :n] = k[:, :, :n], v[:, :, :n]
                mask = (torch.arange(L, device="cuda") < n).view(1, 1, 1, L)
                pos = torch.tensor([[n - 1]], device="cuda")
                out = _call(attn, q, kbuf, vbuf, meta, mask=mask, position_ids=pos)[0]
                st = meta[META_KEY][0]
                assert st["static"] and st["dense"] == dense and ("k8" in st) != dense
                if dense:
                    ref = self._dense(q, k[:, :, :n], v[:, :, :n])
                    torch.testing.assert_close(out, ref, atol=2e-3, rtol=0)
                else:
                    cap = st["k8"].shape[2]
                    ref = _kernel_reference(
                        attn.config, q, k[:, :, :n], v[:, :, :n], cap
                    )
                    torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)


def _selection_inputs(n, cap, b=2, hq=8, hkv=2, d=32, seed=0):
    """Random queries and the block metadata of n live tokens in buffers of capacity cap,
    as select_blocks_two_stage takes them. The queries are large, so attention is peaked
    and TopCdf ends well inside the candidates."""
    from sparse_attention_hub.kernels.block_sparse_decode import (
        block_min_max,
        block_sums,
        quantize_keys,
    )

    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (4 * torch.randn(b, hq, d, generator=g)).to("cuda", torch.bfloat16)
    k = torch.randn(b, hkv, n, d, generator=g).to("cuda", torch.bfloat16)

    def padded(x, size):
        buf = x.new_zeros(*x.shape[:2], size, *x.shape[3:])
        buf[:, :, : x.shape[2]] = x
        return buf

    kmin, kmax = (padded(x, cap // 64) for x in block_min_max(k, 64))
    k8, ks = quantize_keys(k)
    seq = torch.full((1,), n, device="cuda", dtype=torch.int32)
    return dict(
        q=q,
        kmin=kmin,
        kmax=kmax,
        k8=padded(k8, cap),
        ks=padded(ks, cap),
        ksum=padded(block_sums(k, 64), cap // 64),
        seq=seq,
    )


def _select(x, kernel, frac=0.25, **kw):
    from sparse_attention_hub.kernels.block_sparse_decode import (
        TUNE,
        select_blocks_two_stage,
        select_blocks_two_stage_ref,
    )

    args = (x["q"], x["kmin"], x["kmax"], x["k8"], x["ks"], 64, 0.95, 0.02, 0.15, frac)
    common = dict(seq_len=x["seq"], ksum=x["ksum"], prefilter_mean=True, **kw)
    if kernel:
        return select_blocks_two_stage(*args, 2, 2, return_aux=True, **common)
    return select_blocks_two_stage_ref(
        *args,
        2,
        2,
        return_cand=True,
        region=TUNE["prefilter_region"],
        region_alpha=TUNE["prefilter_region_alpha"],
        **common,
    )


def _block_sets(idx, counts):
    return [
        [sorted(idx[b, h, : counts[b, h]].tolist()) for h in range(idx.shape[1])]
        for b in range(idx.shape[0])
    ]


@cuda
@pytest.mark.unit
class TestCarriedBlocks:
    """The model-wide carried set: kernels against the torch reference."""

    N, CAP = 45000, 45056  # 704 blocks: two prefilter regions

    def test_full_scan_keeps_every_block(self):
        x = _selection_inputs(self.N, self.CAP)
        idx, counts, aux = _select(x, True, frac=1.0)
        p = math.ceil(self.N / 64)
        assert (aux["ncand"] == p).all()
        assert aux["isc"][:, :, :p].all()
        ridx, rcounts, rcand = _select(x, False, frac=1.0)
        assert rcand[:, :, :p].all()
        assert _block_sets(idx, counts) == _block_sets(ridx, rcounts)

    def test_kernel_matches_reference(self):
        x = _selection_inputs(self.N, self.CAP)
        p = math.ceil(self.N / 64)
        g = torch.Generator(device="cpu").manual_seed(1)
        score = torch.zeros(2, self.CAP // 64)
        carried = torch.randint(2, p - 2, (2, 24), generator=g)
        score.scatter_(1, carried, 0.5)
        below = torch.randint(2, p - 2, (2, 24), generator=g)
        score.scatter_(1, below, 0.05)  # under the threshold: not carried
        score = score.cuda()
        plain = _select(x, True)
        ks, rs = score.clone(), score.clone()
        idx, counts, aux = _select(x, True, carry_score=ks, carry_threshold=0.2)
        ridx, rcounts, rcand = _select(x, False, carry_score=rs, carry_threshold=0.2)
        # every carried block is a candidate, and the candidate count is unchanged
        isc = aux["isc"][:, :, :p].bool()
        assert torch.equal(isc, rcand[:, :, :p])
        want = (score[:, :p] >= 0.2)[:, None, :].expand_as(isc)
        assert isc[want].all()
        assert torch.equal(aux["ncand"], plain[2]["ncand"])
        assert not torch.equal(isc, plain[2]["isc"][:, :, :p].bool())
        assert _block_sets(idx, counts) == _block_sets(ridx, rcounts)
        # scores only rise, to the same values as the reference, and only on candidates
        torch.testing.assert_close(ks, rs, atol=1e-5, rtol=1e-4)
        assert (ks >= score).all() and (ks > score).any()
        assert torch.equal(ks[:, :p][~isc.any(1)], score[:, :p][~isc.any(1)])

    def test_module_passes_the_scores_between_layers(self):
        c = TwoStageBlockSparseConfig(carry_threshold=0.2)
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v = _qkv(1, 32, 8, 45000, device="cuda", dtype=torch.bfloat16)
        meta = {}
        out, _ = _call(attn, q, k, v, meta)
        cap = meta[META_KEY][0]["k8"].shape[2]
        score = torch.zeros(1, cap // 64, device="cuda")
        ref = _kernel_reference(c, q, k, v, cap, carry_score=score)
        torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)
        torch.testing.assert_close(meta[CARRY_KEY]["score"], score, atol=0, rtol=0)
        assert score.max() > 0
        # another layer of the same step selects with what the first one found
        out, _ = _call(attn, q, k, v, meta, layer_idx=1)
        ref = _kernel_reference(c, q, k, v, cap, carry_score=score)
        torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)
        torch.testing.assert_close(meta[CARRY_KEY]["score"], score, atol=0, rtol=0)


def _planted_layers(blocks, n=45000, hq=8, hkv=2, d=128):
    """(q, k, v) of two layers in which one block per batch row matters, a query that
    looks elsewhere, and the reader's needle values [B, hkv, d]. Every query head of a
    layer points along one direction.

    finder: all 64 keys of the row's block lie along its query, so the prefilter ranks
    the block first and stage 2 gives it nearly all the mass.
    reader: one needle key of the block lies along its query and takes nearly all the
    attention, but the block's other keys point the other way, so the bound and the mean
    key both rank the block last: a prefilter miss.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    b = len(blocks)

    def direction():
        return torch.nn.functional.normalize(torch.randn(b, 1, d, generator=g), dim=-1)

    def query(w):
        return (4.0 * w)[:, None].expand(b, hq, 1, d).clone()

    def kv():
        return torch.randn(b, hkv, n, d, generator=g)

    w0, w1 = direction(), direction()
    k0, v0, k1, v1 = kv(), kv(), kv(), kv()
    for row, x in enumerate(blocks):
        s = slice(64 * x, 64 * x + 64)
        k0[row, :, s] = 6.0 * w0[row] + 0.01 * k0[row, :, s]
        k1[row, :, s] = -0.5 * w1[row] + 0.01 * k1[row, :, s]
        k1[row, :, 64 * x + 5] = 6.0 * w1[row, 0]
    needle = torch.stack([v1[row, :, 64 * x + 5] for row, x in enumerate(blocks)])

    def cast(*ts):
        return tuple(t.to("cuda", torch.bfloat16) for t in ts)

    return (
        cast(query(w0), k0, v0),
        cast(query(w1), k1, v1),
        cast(query(direction()))[0],
        needle.cuda(),
    )


def _needle_cos(out, needle):
    """Cosine of each query head's output (b, 1, h_q, d) with its KV head's needle."""
    g = out.shape[2] // needle.shape[1]
    return torch.nn.functional.cosine_similarity(
        out[:, 0].float(), needle.repeat_interleave(g, dim=1).float(), dim=-1
    )


@cuda
@pytest.mark.unit
class TestCarriedSetAcrossLayers:
    """The module on two layers: a block that one layer's stage 2 finds becomes a
    candidate of a layer whose own prefilter misses it."""

    BLOCKS = (300, 600)  # one per batch row, one in each prefilter region

    @staticmethod
    def _attn(**kw):
        return TwoStageBlockSparse.create_from_config(TwoStageBlockSparseConfig(**kw))

    def test_the_reader_misses_the_needle_on_its_own(self):
        finder, reader, _, needle = _planted_layers(self.BLOCKS)
        q, k, v = reader
        dense = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=1.0, enable_gqa=True
        )
        assert (_needle_cos(dense.transpose(1, 2), needle) > 0.99).all()
        attn, meta = self._attn(), {}
        _call(attn, *finder, meta, layer_idx=0)
        out, _ = _call(attn, *reader, meta, layer_idx=1)
        assert CARRY_KEY not in meta  # threshold 0: no carried state at all
        assert (_needle_cos(out, needle) < 0.5).all()

    def test_a_found_block_reaches_the_other_layer(self):
        finder, reader, _, needle = _planted_layers(self.BLOCKS)
        attn, meta = self._attn(carry_threshold=0.2), {}
        _call(attn, *finder, meta, layer_idx=0)
        score = meta[CARRY_KEY]["score"]
        for row, x in enumerate(self.BLOCKS):
            assert score[row, x] > 0.9
            assert score[row, self.BLOCKS[1 - row]] < 0.2  # batch rows are separate
        out, _ = _call(attn, *reader, meta, layer_idx=1)
        assert (_needle_cos(out, needle) > 0.99).all()

    def test_per_layer_scores_do_not_reach_the_other_layer(self):
        finder, reader, _, needle = _planted_layers(self.BLOCKS)
        attn, meta = self._attn(carry_threshold=0.2, carry_per_layer=True), {}
        _call(attn, *finder, meta, layer_idx=0)
        assert CARRY_KEY not in meta
        assert meta[(CARRY_KEY, 0)]["score"][0, self.BLOCKS[0]] > 0.9
        out, _ = _call(attn, *reader, meta, layer_idx=1)
        assert (_needle_cos(out, needle) < 0.5).all()

    @pytest.mark.parametrize("decay, carried", [(0.0, False), (0.9, True)])
    def test_decay_decides_whether_a_block_outlives_its_step(self, decay, carried):
        finder, reader, away, needle = _planted_layers(self.BLOCKS)
        attn, meta = self._attn(carry_threshold=0.2, carry_decay=decay), {}
        _call(attn, *finder, meta, layer_idx=0)
        out, _ = _call(attn, *reader, meta, layer_idx=1)
        assert (_needle_cos(out, needle) > 0.99).all()  # same step: always carried
        # the next step: the first layer looks elsewhere, so nothing writes the block
        _call(attn, away, *finder[1:], meta, layer_idx=0)
        out, _ = _call(attn, *reader, meta, layer_idx=1)
        assert ((_needle_cos(out, needle) > 0.99) == carried).all()


@pytest.mark.unit
class TestCarryScore:
    def test_per_layer_scores_are_separate_buffers(self):
        attn = TwoStageBlockSparse.create_from_config(
            TwoStageBlockSparseConfig(
                carry_threshold=0.2, carry_decay=0.5, carry_per_layer=True
            )
        )
        meta, kmin = {}, torch.zeros(2, 4, 10, 8)
        a = attn._carry_score(meta, 3, kmin, 100)
        b = attn._carry_score(meta, 5, kmin, 100)
        assert a is not b and CARRY_KEY not in meta
        a.fill_(1.0)
        b.fill_(1.0)
        assert attn._carry_score(meta, 5, kmin, 101) is b  # each layer decays its own
        assert (b == 0.5).all() and (a == 1.0).all()
        assert attn._carry_score(meta, 3, kmin, 101) is a and (a == 0.5).all()

    def test_decays_once_per_step_and_resets_on_a_new_sequence(self):
        attn = TwoStageBlockSparse.create_from_config(
            TwoStageBlockSparseConfig(carry_threshold=0.2, carry_decay=0.5)
        )
        meta, kmin = {}, torch.zeros(2, 4, 10, 8)
        score = attn._carry_score(meta, 3, kmin, 100)
        assert score.shape == (2, 10) and not score.any()
        score.fill_(1.0)
        assert attn._carry_score(meta, 5, kmin, 100) is score  # another layer: no decay
        assert (score == 1.0).all()
        assert attn._carry_score(meta, 3, kmin, 101) is score  # the next step
        assert (score == 0.5).all()
        grown = attn._carry_score(meta, 5, torch.zeros(2, 4, 12, 8), 101)
        assert grown.shape == (2, 12)
        assert (grown[:, :10] == 0.5).all() and not grown[:, 10:].any()
        assert not attn._carry_score(meta, 3, torch.zeros(2, 4, 12, 8), 7).any()

    def test_off_by_default(self):
        c = TwoStageBlockSparseConfig()
        assert c.carry_threshold == 0.0 and c.carry_decay == 0.9
        assert c.carry_per_layer is False
