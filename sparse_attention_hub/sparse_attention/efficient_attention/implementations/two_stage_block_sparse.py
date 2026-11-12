"""Two-stage block-sparse decode attention on the Triton kernels in kernels/block_sparse_decode."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ...utils.mask_attention_utils import get_true_attention_output
from ..base import EfficientAttention, EfficientAttentionConfig, SparseAttentionConfig

META_KEY = "two_stage_block_sparse"
CARRY_KEY = "two_stage_block_sparse_carry"


@dataclass
class TwoStageBlockSparseConfig(EfficientAttentionConfig):
    """Configuration for TwoStageBlockSparse.

    Per KV head and decode step: blocks are ranked by the Quest bound (blended with the
    block-mean-key score when prefilter_mean), the top prefilter_fraction become candidates,
    their exact block mass is computed from an int8 copy of the keys, and TopCdf keeps the
    smallest set reaching cdf_threshold of it (at least floor_fraction of the candidates, at
    most max_fraction of the blocks). Sink and local tokens are always attended. tail adds
    the unselected blocks back as block-mean pseudo-tokens.

    min_cached_tokens: decode steps with fewer cached tokens than this (batch x sequence
    length) run dense SDPA with native GQA instead, which is faster there: on Qwen3-4B /
    H200 under CUDA graphs sparse decoding breaks even at ~32k cached tokens at batch 1
    and 8 alike. 0 (the default) always decodes sparsely; set ~32768 for speed.

    layer_prefilter_fraction: prefilter_fraction for particular layers (layer_idx ->
    fraction); other layers use prefilter_fraction. At long context the prefilter's misses
    concentrate in a few layers, which can be given a larger candidate budget alone. A
    fraction of 1.0 makes a layer scan every block (nothing is dropped before stage 2).

    carry_threshold: a carried set of blocks shared by the whole model (0 disables it).
    At long context the prefilter drops blocks that another layer has already found. One
    score per block is kept for all layers and KV heads: a layer raises the score of each
    of its candidates to the largest stage-2 probability a query head gives it, the layers
    above read that in the same step, and the scores decay by carry_decay once per decode
    step. A block scoring at least carry_threshold is a candidate in every layer whatever
    the prefilter ranks it, in place of the lowest-ranked candidate of its region (so at
    most its region's quota of blocks is carried). With a few early layers scanning every
    block (layer_prefilter_fraction 1.0), what they find reaches every layer above them at
    the default candidate budget. carry_decay 0 clears the scores at every decode step: a
    block then reaches only the layers above the one that found it, in the same step.

    carry_per_layer: an ablation of the carried set. Every layer keeps its own scores, so
    a block is carried across decode steps but never from one layer to another.
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
    min_cached_tokens: int = 0
    layer_prefilter_fraction: Optional[Dict[int, float]] = None
    carry_threshold: float = 0.0
    carry_decay: float = 0.9
    carry_per_layer: bool = False


class TwoStageBlockSparse(EfficientAttention):
    """Block-sparse decode attention with two-stage selection (Triton kernels, CUDA only).

    Single-token decode steps run the selection and block-sparse attention kernels; any
    call with more than one query (prefill, a question chunk) runs dense attention. Per
    layer, block metadata (key min/max, key/value sums, int8 keys) is kept in
    sparse_meta_data and brought up to date with whatever tokens the KV cache gained since
    the last call. Assumes unpadded sequences (every batch row has the cache's length).

    Speed: a decode step launches about seven Triton kernels per layer. Under HuggingFace's
    eager generation that launch cost (~0.3 ms of CPU per layer) outweighs the attention
    time saved; the speedups come from capturing the decode step in a CUDA graph.

    Static caches: when the cache is longer than the sequence at a layer's first decode
    step (a preallocated cache such as HF's StaticCache, sequence length read from
    position_ids), decode steps take the token's position from position_ids on the device
    and append it to the metadata without any host-side read, so a whole HF decode step
    can be captured in a CUDA graph and replayed (see scripts/hf_graph_decode_bench.py).
    Multi-token calls on a static cache fold their tokens in eagerly. A new sequence needs
    a new sparse_meta_data.

    Below config.min_cached_tokens cached tokens, decode steps run dense (no metadata is
    kept meanwhile): per step on a growing cache, and once per sequence, at its first
    decode step, on a static one.
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
        layer_idx = kwargs.get("layer_idx", 0)
        state = self._layer_state(sparse_meta_data, layer_idx)
        position_ids: Optional[torch.Tensor] = kwargs.get("position_ids")  # type: ignore[assignment]
        if state.get("static") and not state["dense"] and queries.shape[2] > 1:
            # Tokens a multi-token call wrote into the static cache: fold them in while
            # eager (decode steps only append their own token).
            self._update_metadata(
                state,
                keys,
                values,
                int(position_ids[0, 0]),
                int(position_ids[0, -1]) + 1,
            )
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
            append_kv,
            block_sparse_decode_fused,
            select_blocks_two_stage,
        )

        c = self.config
        if state["n"] == 0 and not state.get("static") and position_ids is not None:
            # First decode step on this layer: a cache longer than the sequence is static.
            n = int(position_ids[0, -1]) + 1
            if n < keys.shape[2]:
                state["static"] = True
                # Decided once per sequence: a captured graph cannot switch paths later.
                # Dense needs the mask that limits the static cache to its live tokens.
                state["dense"] = (
                    attention_mask is not None
                    and keys.shape[0] * n < c.min_cached_tokens
                )
                if not state["dense"]:
                    state["seq_len"] = torch.empty(
                        1, device=keys.device, dtype=torch.int32
                    )
                    self._update_metadata(state, keys, values, 0, n - 1)
        if state.get("dense") or (
            not state.get("static")
            and keys.shape[0] * keys.shape[2] < c.min_cached_tokens
        ):
            return self._dense_decode(queries, keys, values, attention_mask, scaling)
        if state.get("static"):
            # The cache already holds the token at its position: one append kernel reads
            # it there, folds it into the metadata and writes seq_len = position + 1.
            seq_len = state["seq_len"]
            append_kv(
                None,
                None,
                keys,
                values,
                position_ids[0, -1:],
                state["kmin"],
                state["kmax"],
                c.block_size,
                state["ksum"],
                state["vsum"],
                k8=state["k8"],
                kscale=state["ks"],
                seq_out=seq_len,
            )
        else:
            n = keys.shape[2]
            seq_len = torch.full((1,), n, device=keys.device, dtype=torch.int32)
            n_old = state["n"]
            if n_old > n or ("k8" in state and state["k8"].shape[:2] != keys.shape[:2]):
                n_old = 0  # a new sequence reuses this sparse_meta_data
            if n == n_old + 1 and "k8" in state and state["k8"].shape[2] >= n:
                self._append(
                    state,
                    keys[:, :, n - 1],
                    values[:, :, n - 1],
                    keys,
                    values,
                    seq_len - 1,
                )
                state["n"] = n
            else:
                self._update_metadata(state, keys, values, n_old, n)
        q = (queries[:, :, 0] * scaling).contiguous()
        use_sums = c.prefilter_mean or c.tail
        prefilter = min(
            1.0, (c.layer_prefilter_fraction or {}).get(layer_idx, c.prefilter_fraction)
        )  # 1.0 is already a full scan; the kernels size their buffers for at most that
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
            prefilter,
            math.ceil(c.sink_tokens / c.block_size),
            math.ceil(c.local_tokens / c.block_size),
            seq_len=seq_len,
            ksum=state["ksum"] if use_sums else None,
            prefilter_mean=c.prefilter_mean,
            return_aux=c.tail,
            carry_score=(
                self._carry_score(
                    sparse_meta_data,
                    layer_idx,
                    state["kmin"],
                    None if state.get("static") else keys.shape[2],
                )
                if c.carry_threshold > 0
                else None
            ),
            carry_threshold=c.carry_threshold,
        )
        tail = dict(aux=sel[2], vsum=state["vsum"], rho=c.tail_rho) if c.tail else None
        out = block_sparse_decode_fused(
            q, keys, values, sel[0], sel[1], c.block_size, seq_len=seq_len, tail=tail
        )
        return out.unsqueeze(1), None

    @staticmethod
    def _dense_decode(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
    ) -> Tuple[torch.Tensor, None]:
        """One dense decode step: SDPA with native GQA (K/V are not repeated per query
        head; FlashAttention without a mask, cuDNN with one where enabled)."""
        mask = None if attention_mask is None else attention_mask[..., : keys.shape[2]]
        out = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=mask, scale=scaling, enable_gqa=True
        )
        return out.transpose(1, 2), None

    def _carry_score(
        self,
        sparse_meta_data: Dict[Any, Any],
        layer_idx: Any,
        kmin: torch.Tensor,
        n: Optional[int],
    ) -> torch.Tensor:
        """The carried block scores [B, P]: one buffer for every layer of the model.

        The first layer to ask for it marks the decode step: on each of its calls the
        scores decay once, and a sequence shorter than the last one (a new sequence on a
        growing cache; n is None on a static one, which needs a new sparse_meta_data
        anyway) starts from zero. The buffer follows the metadata's block capacity.
        With carry_per_layer each layer has a buffer of its own, under (CARRY_KEY, layer).
        """
        key: Any = (CARRY_KEY, layer_idx) if self.config.carry_per_layer else CARRY_KEY
        carry: Dict[str, Any] = sparse_meta_data.setdefault(
            key, {"first": layer_idx, "n": 0}
        )
        shape = (kmin.shape[0], kmin.shape[2])
        score: Optional[torch.Tensor] = carry.get("score")
        if score is None or score.shape != shape:
            grown = torch.zeros(shape, device=kmin.device)
            if score is not None and score.shape[0] == shape[0]:
                p = min(shape[1], score.shape[1])
                grown[:, :p] = score[:, :p]
            carry["score"] = score = grown
        if layer_idx == carry["first"]:
            if n is not None:
                if n < carry["n"]:
                    score.zero_()
                carry["n"] = n
            score.mul_(self.config.carry_decay)
        return score

    @staticmethod
    def _layer_state(
        sparse_meta_data: Dict[Any, Any], layer_idx: Any
    ) -> Dict[str, Any]:
        layers: Dict[Any, Dict[str, Any]] = sparse_meta_data.setdefault(META_KEY, {})
        return layers.setdefault(layer_idx, {"n": 0})

    def _append(
        self,
        state: Dict[str, Any],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        pos: torch.Tensor,
    ) -> None:
        """Fold one decode token at device position pos into the metadata (no host reads).

        The append kernel also writes the token back into the cache (same values).
        """
        from ....kernels.block_sparse_decode import append_k8, append_kv

        append_kv(
            k_new,
            v_new,
            keys,
            values,
            pos,
            state["kmin"],
            state["kmax"],
            self.config.block_size,
            state["ksum"],
            state["vsum"],
        )
        append_k8(k_new, state["k8"], state["ks"], pos)

    def _update_metadata(
        self,
        state: Dict[str, Any],
        keys: torch.Tensor,
        values: torch.Tensor,
        n_old: int,
        n: int,
    ) -> None:
        """Fold cache tokens [n_old, n) into the block metadata, growing it when full.

        The blocks from the one holding token n_old onward are recomputed, so a partially
        filled block picks up its new tokens. A static cache gets room for all of it at
        once (its buffers must not move between CUDA-graph replays).
        """
        from ....kernels.block_sparse_decode import (
            block_min_max,
            block_sums,
            quantize_keys,
        )

        bs = self.config.block_size
        B, H, _, D = keys.shape
        cap = state["k8"].shape[2] if "k8" in state and n_old else 0
        static = bool(state.get("static"))
        if (keys.shape[2] if static else n) > cap:
            # Whole blocks: the whole static cache, else room for 1024 more (or double).
            grow_to = keys.shape[2] if static else max(n + 1024, 2 * cap)
            new_cap = math.ceil(grow_to / bs) * bs
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
        state["n"] = n
        if n <= n_old:
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
