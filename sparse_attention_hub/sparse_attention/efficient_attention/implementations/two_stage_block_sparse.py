"""Two-stage block-sparse decode attention on the Triton kernels in kernels/block_sparse_decode."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ...utils.mask_attention_utils import get_true_attention_output
from ..base import EfficientAttention, EfficientAttentionConfig, SparseAttentionConfig

META_KEY = "two_stage_block_sparse"


@dataclass
class TwoStageBlockSparseConfig(EfficientAttentionConfig):
    """Configuration for TwoStageBlockSparse.

    Per KV head and decode step: blocks are ranked by the Quest bound (blended with the
    block-mean-key score when prefilter_mean), the top prefilter_fraction become candidates,
    their exact block mass is computed from an int8 copy of the keys, and TopCdf keeps the
    smallest set reaching cdf_threshold of it (at least floor_fraction of the candidates, at
    most max_fraction of the blocks). Sink and local tokens are always attended. tail adds
    the unselected blocks back as block-mean pseudo-tokens.
    """

    block_size: int = 64
    cdf_threshold: float = 0.95
    floor_fraction: float = 0.02
    max_fraction: float = 0.15
    prefilter_fraction: float = 0.25
    sink_tokens: int = 128
    local_tokens: int = 128
    prefilter_mean: bool = True
    tail: bool = False
    tail_rho: float = 0.6


class TwoStageBlockSparse(EfficientAttention):
    """Block-sparse decode attention with two-stage selection (Triton kernels, CUDA only).

    Single-token decode steps run the selection and block-sparse attention kernels; any
    call with more than one query (prefill, a question chunk) runs dense attention. Per
    layer, block metadata (key min/max, key/value sums, int8 keys) is kept in
    sparse_meta_data and brought up to date with whatever tokens the KV cache gained since
    the last call. Assumes unpadded sequences (every batch row has the cache's length).

    Speed: a decode step launches about seven Triton kernels per layer. Under HuggingFace's
    eager generation that launch cost (~0.3 ms of CPU per layer) outweighs the attention
    time saved, so this path is for quality evaluation; the speedups come from capturing
    the decode step in a CUDA graph (see scripts/e2e_decode_bench.py).
    """

    def __init__(
        self,
        sparse_attention_config: SparseAttentionConfig,
        config: TwoStageBlockSparseConfig,
    ) -> None:
        """Initialize from its config.

        Args:
            sparse_attention_config: Configuration for the sparse attention mechanism.
            config: The TwoStageBlockSparse parameters.
        """
        super().__init__(sparse_attention_config)
        self.config = config

    def custom_attention(
        self,
        module: nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute attention: block-sparse for one query token, dense otherwise.

        Args:
            module: The attention module.
            queries: (b, h_q, s_q, d).
            keys: (b, h_kv, s_k, d), the whole KV cache.
            values: (b, h_kv, s_k, d).
            attention_mask: Mask HuggingFace built; used by the dense path only.
            scaling: Scaling factor for attention logits.
            dropout: Dropout probability (dense path only).
            **kwargs: Must include sparse_meta_data; layer_idx keys the per-layer state.

        Returns:
            Attention output (b, s_q, h_q, d) and None.
        """
        sparse_meta_data: Dict[Any, Any] = kwargs.pop("sparse_meta_data", None)  # type: ignore[assignment]
        if sparse_meta_data is None:
            raise ValueError(
                "sparse_meta_data must be provided in kwargs while calling custom_attention()"
            )
        if kwargs.get("s_aux") is not None:
            raise NotImplementedError(
                "TwoStageBlockSparse does not support attention-sink (s_aux) models."
            )
        kwargs.pop("s_aux", None)
        if queries.shape[2] != 1 or not queries.is_cuda:
            return get_true_attention_output(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )

        from ....kernels.block_sparse_decode import (
            block_sparse_decode_fused,
            select_blocks_two_stage,
        )

        c = self.config
        state = self._layer_state(sparse_meta_data, kwargs.get("layer_idx", 0))
        n = keys.shape[2]
        seq_len = torch.full((1,), n, device=keys.device, dtype=torch.int32)
        self._update_metadata(state, keys, values, seq_len)
        q = (queries[:, :, 0] * scaling).contiguous()
        use_sums = c.prefilter_mean or c.tail
        sel = select_blocks_two_stage(
            q,
            state["kmin"],
            state["kmax"],
            state["k8"],
            state["ks"],
            c.block_size,
            c.cdf_threshold,
            c.floor_fraction,
            c.max_fraction,
            c.prefilter_fraction,
            math.ceil(c.sink_tokens / c.block_size),
            math.ceil(c.local_tokens / c.block_size),
            seq_len=seq_len,
            ksum=state["ksum"] if use_sums else None,
            prefilter_mean=c.prefilter_mean,
            return_aux=c.tail,
        )
        tail = dict(aux=sel[2], vsum=state["vsum"], rho=c.tail_rho) if c.tail else None
        out = block_sparse_decode_fused(
            q, keys, values, sel[0], sel[1], c.block_size, seq_len=seq_len, tail=tail
        )
        return out.unsqueeze(1), None

    @staticmethod
    def _layer_state(
        sparse_meta_data: Dict[Any, Any], layer_idx: Any
    ) -> Dict[str, Any]:
        layers: Dict[Any, Dict[str, Any]] = sparse_meta_data.setdefault(META_KEY, {})
        return layers.setdefault(layer_idx, {"n": 0})

    def _update_metadata(
        self,
        state: Dict[str, Any],
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> None:
        """Fold the cache's new tokens into the block metadata, growing it when full.

        One new token (a decode step) is folded in by the append kernels. Otherwise the
        blocks from the one holding the first new token onward are recomputed, so a
        partially filled block picks up its new tokens.
        """
        from ....kernels.block_sparse_decode import (
            append_k8,
            append_kv,
            block_min_max,
            block_sums,
            quantize_keys,
        )

        bs = self.config.block_size
        B, H, n, D = keys.shape
        n_old = state["n"]
        if n_old > n or ("k8" in state and state["k8"].shape[:2] != (B, H)):
            n_old = 0  # a new sequence reuses this sparse_meta_data
        if n_old == n:
            return
        cap = state["k8"].shape[2] if "k8" in state else 0
        if n > cap:
            # Room for 1024 more tokens (or double), whole blocks.
            new_cap = math.ceil(max(n + 1024, 2 * cap) / bs) * bs
            grown = dict(
                kmin=keys.new_empty(B, H, new_cap // bs, D),
                kmax=keys.new_empty(B, H, new_cap // bs, D),
                ksum=torch.empty(B, H, new_cap // bs, D, device=keys.device),
                vsum=torch.empty(B, H, new_cap // bs, D, device=keys.device),
                k8=torch.empty(B, H, new_cap, D, device=keys.device, dtype=torch.int8),
                ks=torch.empty(B, H, new_cap, device=keys.device),
            )
            if n_old:
                p_old = math.ceil(n_old / bs)
                for name in ("kmin", "kmax", "ksum", "vsum"):
                    grown[name][:, :, :p_old] = state[name][:, :, :p_old]
                grown["k8"][:, :, :n_old] = state["k8"][:, :, :n_old]
                grown["ks"][:, :, :n_old] = state["ks"][:, :, :n_old]
            state.update(grown)
        if n == n_old + 1:
            # The append kernel also writes the token back into the cache (same values).
            pos = seq_len - 1
            k_new, v_new = keys[:, :, n - 1], values[:, :, n - 1]
            append_kv(
                k_new,
                v_new,
                keys,
                values,
                pos,
                state["kmin"],
                state["kmax"],
                bs,
                state["ksum"],
                state["vsum"],
            )
            append_k8(k_new, state["k8"], state["ks"], pos)
            state["n"] = n
            return
        b0 = n_old // bs
        p1 = math.ceil(n / bs)
        tail_k = keys[:, :, b0 * bs : n]
        kmin, kmax = block_min_max(tail_k, bs)
        state["kmin"][:, :, b0:p1] = kmin
        state["kmax"][:, :, b0:p1] = kmax
        state["ksum"][:, :, b0:p1] = block_sums(tail_k, bs)
        state["vsum"][:, :, b0:p1] = block_sums(values[:, :, b0 * bs : n], bs)
        k8, ks = quantize_keys(keys[:, :, n_old:n])
        state["k8"][:, :, n_old:n] = k8
        state["ks"][:, :, n_old:n] = ks
        state["n"] = n

    @classmethod
    def create_from_config(cls, config: SparseAttentionConfig) -> "TwoStageBlockSparse":
        """Create a TwoStageBlockSparse instance from its configuration.

        Args:
            config: A TwoStageBlockSparseConfig.

        Returns:
            Instance of TwoStageBlockSparse.

        Raises:
            TypeError: If config is not a TwoStageBlockSparseConfig.
        """
        if not isinstance(config, TwoStageBlockSparseConfig):
            raise TypeError(f"Expected TwoStageBlockSparseConfig, got {type(config)}")
        return cls(sparse_attention_config=config, config=config)
