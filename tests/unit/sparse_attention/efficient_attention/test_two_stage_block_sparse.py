"""Tests for the TwoStageBlockSparse efficient attention."""

import math

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


def _call(attn, q, k, v, meta, layer_idx=0, mask=None):
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
        from sparse_attention_hub.kernels.block_sparse_decode import (
            block_min_max,
            block_sparse_decode_fused,
            block_sums,
            quantize_keys,
            select_blocks_two_stage,
        )

        c = TwoStageBlockSparseConfig(tail=tail)
        attn = TwoStageBlockSparse.create_from_config(c)
        q, k, v = _qkv(1, 32, 8, 20000, device="cuda", dtype=torch.bfloat16)
        meta = {}
        out, _ = _call(attn, q, k, v, meta)
        # Fresh metadata in buffers of the same capacity (the selection cap is a fraction
        # of the capacity's blocks).
        n, cap = k.shape[2], meta[META_KEY][0]["k8"].shape[2]

        def padded(x, size):
            buf = x.new_zeros(*x.shape[:2], size, *x.shape[3:])
            buf[:, :, : x.shape[2]] = x
            return buf

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
            return_aux=tail,
        )
        ref = block_sparse_decode_fused(
            q[:, :, 0].contiguous(),
            k,
            v,
            sel[0],
            sel[1],
            64,
            seq_len=seq,
            tail=dict(aux=sel[2], vsum=vsum, rho=c.tail_rho) if tail else None,
        )
        torch.testing.assert_close(out[:, 0], ref, atol=2e-3, rtol=0)

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
