"""SpargeAttn block Top-CDF masker (decode-time adaptation).

SpargeAttn (Zhang et al. 2025) predicts a block-sparse mask from a compressed
attention map: every key block is pooled to its mean, blocks whose keys are not
self-similar are always computed (their mean is not a faithful summary), and among
the remaining blocks the smallest set whose compressed attention mass reaches
``cdf_threshold`` is kept (TopCdf).

SpargeAttn targets prefill and pools query blocks too. Here the queries are the
question tokens and generated tokens attending to a KV cache, so every query row
selects its own key blocks (a query block of size 1). The block budget is therefore
adaptive per (batch, head, query) instead of the fixed ``heavy_size`` of Quest.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from ray import tune

from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention.maskers.base import (
    MaskerConfig,
    MaskerRegistry,
)
from sparse_attention_hub.sparse_attention.utils.kv_utils import (
    _get_num_key_value_groups,
    repeat_kv,
)
from sparse_attention_hub.sparse_attention.utils.mask import Mask

from ..base import FixedMasker, FixedMaskerConfig
from .utils.quest_utils import attention_mask_to_allowed_prob

MicroMetricLogger.register_metric("sparge_forced_block_fraction", float)


@dataclass
class SpargeTopCdfMaskerConfig(FixedMaskerConfig):
    """Configuration for SpargeTopCdfMasker.

    Attributes:
        block_size: Number of keys pooled into one block.
        cdf_threshold: Compressed attention mass the selected blocks must cover (tau).
        sim_threshold: Blocks whose mean pairwise cosine similarity is below this are
            always kept (theta). ``-1.0`` disables the check.
        size_weighted: Weight a block's compressed score by its token count, so the
            compressed softmax estimates block mass rather than mean-key affinity.
        smooth_k: Subtract the per-head mean key before the similarity test, as the
            SpargeAttn kernels do. It leaves the compressed softmax unchanged.
        include_crossing: Keep the block whose mass crosses ``cdf_threshold`` so the
            kept mass is at least the threshold. SpargeAttn drops it (``cdf <= tau``)
            and keeps at least one block per row.
        topk_fraction: Also keep the top ``topk_fraction`` of candidate blocks per row
            (the Top-k + Top-p union of SpargeAttention2). ``0.0`` disables it.
        variance_correction: Weight lambda of the second-order term of log-sum-exp over
            the block, ``lambda * 0.5 * scaling**2 * sum_d q_d**2 * Var_block(k_d)``, so
            spread-out blocks are not under-scored by their mean key (a soft alternative to
            the similarity gate). ``0.0`` disables it; ``True`` means 1.0.
        score: Base block score. ``"mean"`` is SpargeAttn's ``q . mean(k_block)``;
            ``"quest_bound"`` is Quest's upper bound ``sum_d max(q_d * min_d, q_d * max_d)``
            over the block's per-channel key range, which keeps a single high-scoring
            token from being averaged away. ``"exact"`` is the true block mass
            ``logsumexp_{i in block}(q . k_i * scaling)`` (an oracle, for ablations; it
            ignores ``size_weighted`` and ``variance_correction``). TopCdf then runs on
            these scores.
        prefilter_fraction: Two-stage selection. Keep only the top fraction of valid
            blocks by the Quest bound as candidates before scoring them with ``score``
            (a cheap bound prunes, a costlier estimate decides). ``0.0`` disables it.
        key_bits: For ``score="exact"``, fake-quantize keys per token (symmetric, absmax
            over the head dim) to this many bits before computing block mass, modelling a
            low-bit key copy used only for selection. ``16`` leaves keys unchanged.
        prefilter_shared: Rank the prefilter on the group-mean of the bound's per-head
            softmax (one candidate set per KV head) instead of each query head's own top
            blocks (union over the group). Needs ``share_kv_groups``.
        max_fraction: Cap on selected candidate blocks per row, as a fraction of valid
            blocks (highest mass first); bounds the worst-case head. ``1.0`` disables it.
        share_kv_groups: Select once per KV head (block probabilities averaged over its
            query group) so a GQA group reads each selected block once, as a real
            block-sparse decode kernel would.
    """

    block_size: int = 64
    cdf_threshold: float = 0.9
    sim_threshold: float = 0.5
    size_weighted: bool = True
    smooth_k: bool = True
    include_crossing: bool = True
    topk_fraction: float = 0.0
    variance_correction: float = 0.0
    score: str = "mean"
    share_kv_groups: bool = False
    prefilter_fraction: float = 0.0
    key_bits: int = 16
    prefilter_shared: bool = False
    max_fraction: float = 1.0
    search_space: Dict[str, Any] = field(
        default_factory=lambda: {
            "block_size": tune.grid_search([16, 32, 64, 128]),
            "cdf_threshold": tune.grid_search([0.5, 0.7, 0.8, 0.9, 0.95, 0.98]),
            "sim_threshold": tune.grid_search([-1.0, 0.3, 0.5, 0.7]),
        }
    )

    def __post_init__(self) -> None:
        """Validate post-initialization constraints."""
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if not 0.0 < self.cdf_threshold <= 1.0:
            raise ValueError(
                f"cdf_threshold must be in (0, 1], got {self.cdf_threshold}"
            )
        if not -1.0 <= self.sim_threshold <= 1.0:
            raise ValueError(
                f"sim_threshold must be in [-1, 1], got {self.sim_threshold}"
            )
        if not 0.0 <= self.topk_fraction <= 1.0:
            raise ValueError(
                f"topk_fraction must be in [0, 1], got {self.topk_fraction}"
            )
        if self.score not in ("mean", "quest_bound", "exact"):
            raise ValueError(
                f"score must be 'mean', 'quest_bound' or 'exact', got {self.score!r}"
            )
        if not 0.0 <= self.prefilter_fraction <= 1.0:
            raise ValueError(
                f"prefilter_fraction must be in [0, 1], got {self.prefilter_fraction}"
            )
        if not 0.0 < self.max_fraction <= 1.0:
            raise ValueError(f"max_fraction must be in (0, 1], got {self.max_fraction}")
        if not 2 <= self.key_bits <= 16:
            raise ValueError(f"key_bits must be in [2, 16], got {self.key_bits}")
        if self.variance_correction < 0.0:
            raise ValueError(
                f"variance_correction must be >= 0, got {self.variance_correction}"
            )


@MaskerRegistry.register(SpargeTopCdfMaskerConfig)
class SpargeTopCdfMasker(FixedMasker):
    """SpargeAttn block selection: mean-pooled keys, self-similarity gate, TopCdf."""

    block_size: int
    cdf_threshold: float
    sim_threshold: float
    size_weighted: bool
    smooth_k: bool
    include_crossing: bool
    topk_fraction: float
    variance_correction: float
    score: str
    share_kv_groups: bool
    prefilter_fraction: float
    key_bits: int
    prefilter_shared: bool
    max_fraction: float

    def __init__(self, config: SpargeTopCdfMaskerConfig) -> None:
        super().__init__(config)
        self.block_size = int(config.block_size)
        self.cdf_threshold = float(config.cdf_threshold)
        self.sim_threshold = float(config.sim_threshold)
        self.size_weighted = bool(config.size_weighted)
        self.smooth_k = bool(config.smooth_k)
        self.include_crossing = bool(config.include_crossing)
        self.topk_fraction = float(config.topk_fraction)
        self.variance_correction = float(config.variance_correction)
        self.score = str(config.score)
        self.share_kv_groups = bool(config.share_kv_groups)
        self.prefilter_fraction = float(config.prefilter_fraction)
        self.key_bits = int(config.key_bits)
        self.prefilter_shared = bool(config.prefilter_shared)
        self.max_fraction = float(config.max_fraction)

    def add_mask(
        self,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float,
        sparse_meta_data: Dict[Any, Any],
        previous_mask: Mask,
        **kwargs: Dict[str, Any],
    ) -> Mask:
        """Add SpargeAttn Top-CDF block mask."""
        if previous_mask.is_full_mask():
            return previous_mask

        dims = self._extract_tensor_dimensions(keys, queries)
        if dims.seq_len_keys <= self.block_size:
            return self._create_full_mask(
                dims, previous_mask.dtype, previous_mask.device
            )

        groups: int = _get_num_key_value_groups(queries, keys)
        keys_rep: torch.Tensor = repeat_kv(keys, groups)
        block_sel: torch.Tensor = self._select_blocks(
            keys_rep, queries, attention_mask, scaling, kwargs.get("layer_idx"), groups
        )  # [B,H,Q,P] bool

        dense_prev: torch.Tensor = previous_mask.get_dense_mask()
        block_union: torch.Tensor = block_sel.to(dense_prev.dtype).repeat_interleave(
            self.block_size, dim=-1
        )[..., : dims.seq_len_keys]
        dense_mask: torch.Tensor = torch.maximum(dense_prev, block_union)

        if attention_mask is not None:
            allowed: torch.Tensor = attention_mask_to_allowed_prob(
                attention_mask, dims.seq_len_keys
            ).to(dense_mask.dtype)
            dense_mask = dense_mask * allowed.expand_as(dense_mask)

        return Mask.create_mask_from_dense_mask(
            (
                dims.batch_size,
                dims.num_heads,
                dims.seq_len_queries,
                dims.seq_len_keys,
            ),
            dense_mask,
            dtype=previous_mask.dtype,
        )

    def _select_blocks(
        self,
        keys_rep: torch.Tensor,
        queries: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        layer_idx: Optional[int],
        groups: int = 1,
    ) -> torch.Tensor:
        """Return the selected key blocks as a [B,H,Q,P] bool tensor.

        Steps:
          1) Pool keys into blocks: block mean and mean pairwise cosine similarity.
          2) Score blocks with the compressed logits q . mean(k_block) * scaling.
          3) Blocks that are invalid under attention_mask are dropped; blocks that are
             not self-similar are forced on and excluded from the compressed softmax.
          4) TopCdf over the remaining blocks: keep the smallest prefix (by score)
             whose softmax mass reaches cdf_threshold.
        """
        B, H, K, D = keys_rep.shape
        Q: int = queries.shape[2]
        bs: int = self.block_size
        num_blocks: int = (K + bs - 1) // bs
        pad: int = num_blocks * bs - K

        k32: torch.Tensor = keys_rep.float()
        k_sim: torch.Tensor = (
            k32 - k32.mean(dim=2, keepdim=True) if self.smooth_k else k32
        )
        if pad > 0:
            k32 = torch.nn.functional.pad(k32, (0, 0, 0, pad))
            k_sim = torch.nn.functional.pad(k_sim, (0, 0, 0, pad))
        k_blocks: torch.Tensor = k32.view(B, H, num_blocks, bs, D)
        counts: torch.Tensor = torch.full(
            (num_blocks,), float(bs), device=keys_rep.device
        )
        counts[-1] = float(bs - pad)

        # 1) Block mean and self-similarity. Mean pairwise cosine (self-pairs included)
        # equals the squared norm of the mean unit vector, so it is O(bs * D) per block.
        # Padding rows normalize to zero and so add nothing to the sum.
        block_mean: torch.Tensor = k_blocks.sum(dim=3) / counts.view(1, 1, -1, 1)
        k_unit: torch.Tensor = torch.nn.functional.normalize(
            k_sim.view(B, H, num_blocks, bs, D), dim=-1
        )
        unit_mean: torch.Tensor = k_unit.sum(dim=3) / counts.view(1, 1, -1, 1)
        block_sim: torch.Tensor = (unit_mean * unit_mean).sum(dim=-1)  # [B,H,P]

        # 2) Compressed logits per query row.
        bound: Optional[torch.Tensor] = None
        if self.score == "quest_bound" or self.prefilter_fraction > 0.0:
            is_pad: torch.Tensor = (
                torch.arange(num_blocks * bs, device=keys_rep.device) >= K
            ).view(1, 1, num_blocks, bs, 1)
            block_max: torch.Tensor = k_blocks.masked_fill(is_pad, float("-inf")).amax(
                dim=3
            )
            block_min: torch.Tensor = k_blocks.masked_fill(is_pad, float("inf")).amin(
                dim=3
            )
            q4: torch.Tensor = queries.float().unsqueeze(3)  # [B,H,Q,1,D]
            bound = (
                torch.maximum(
                    q4 * block_min.unsqueeze(2), q4 * block_max.unsqueeze(2)
                ).sum(-1)
                * scaling
            )
        if self.score == "quest_bound":
            logits: torch.Tensor = bound
        elif self.score == "exact":
            k_score: torch.Tensor = k32
            if self.key_bits < 16:
                qmax: float = float(2 ** (self.key_bits - 1) - 1)
                step: torch.Tensor = (
                    k32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
                )
                k_score = torch.round(k32 / step).clamp_(-qmax, qmax) * step
            token_logits: torch.Tensor = (
                torch.einsum("bhqd,bhkd->bhqk", queries.float(), k_score) * scaling
            )
            if pad > 0:
                token_logits[..., K:] = float("-inf")
            logits = torch.logsumexp(token_logits.view(B, H, Q, num_blocks, bs), dim=-1)
        else:
            logits = (
                torch.einsum("bhqd,bhpd->bhqp", queries.float(), block_mean) * scaling
            )
        if self.size_weighted and self.score != "exact":
            logits = logits + torch.log(counts).view(1, 1, 1, -1)
        if self.variance_correction and self.score != "exact":
            block_var: torch.Tensor = (
                (k_blocks * k_blocks).sum(dim=3) / counts.view(1, 1, -1, 1)
                - block_mean * block_mean
            ).clamp_(min=0.0)
            logits = (
                logits
                + self.variance_correction
                * 0.5
                * scaling
                * scaling
                * torch.einsum("bhqd,bhpd->bhqp", queries.float() ** 2, block_var)
            )

        # 3) Validity and the self-similarity gate.
        valid: torch.Tensor = torch.ones(
            (B, H, Q, num_blocks), dtype=torch.bool, device=keys_rep.device
        )
        if attention_mask is not None:
            allowed: torch.Tensor = attention_mask_to_allowed_prob(attention_mask, K)
            if pad > 0:
                allowed = torch.nn.functional.pad(allowed, (0, pad))
            allowed = allowed.amax(dim=1, keepdim=True)  # [B,1,Q|1,K+pad]
            block_ok: torch.Tensor = (
                allowed.view(B, 1, allowed.shape[2], num_blocks, bs).amax(dim=-1) > 0
            )
            valid = valid & block_ok.expand(B, H, Q, num_blocks)

        forced: torch.Tensor = (block_sim < self.sim_threshold).unsqueeze(2).expand(
            B, H, Q, num_blocks
        ) & valid
        if layer_idx is not None and MicroMetricLogger().is_metric_enabled(
            "sparge_forced_block_fraction"
        ):
            MicroMetricLogger().log(
                "sparge_forced_block_fraction",
                float(forced.float().mean().item()),
                metadata={"layer_idx": layer_idx, "q_len": Q, "k_len": K},
            )

        candidate: torch.Tensor = valid & ~forced
        if self.prefilter_fraction > 0.0:
            ranked: torch.Tensor = bound.masked_fill(~candidate, float("-inf"))
            if self.prefilter_shared and groups > 1:
                grouped_b = (B, H // groups, groups, Q, num_blocks)
                ranked = (
                    torch.softmax(ranked, dim=-1)
                    .nan_to_num(0.0)
                    .view(grouped_b)
                    .mean(dim=2, keepdim=True)
                    .expand(grouped_b)
                    .reshape(B, H, Q, num_blocks)
                    .masked_fill(~candidate, -1.0)
                )
            order_b: torch.Tensor = ranked.argsort(dim=-1, descending=True)
            n_keep: torch.Tensor = torch.ceil(
                candidate.sum(dim=-1, keepdim=True) * self.prefilter_fraction
            )
            in_top: torch.Tensor = torch.zeros_like(candidate).scatter(
                -1,
                order_b,
                torch.arange(num_blocks, device=keys_rep.device).expand_as(order_b)
                < n_keep,
            )
            candidate = candidate & in_top
        logits = logits.masked_fill(~candidate, float("-inf"))

        # 4) TopCdf over blocks sorted by compressed mass. Rows with no candidate
        # blocks produce NaN probabilities and select nothing.
        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        if self.share_kv_groups and groups > 1:
            grouped = (B, H // groups, groups, Q, num_blocks)
            probs = (
                probs.nan_to_num(0.0)
                .view(grouped)
                .mean(dim=2, keepdim=True)
                .expand(grouped)
                .reshape(B, H, Q, num_blocks)
            )
            candidate = (
                candidate.view(grouped)
                .any(dim=2, keepdim=True)
                .expand(grouped)
                .reshape(B, H, Q, num_blocks)
            )
        sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
        cdf: torch.Tensor = torch.cumsum(sorted_probs, dim=-1)
        rank: torch.Tensor = torch.arange(num_blocks, device=probs.device)
        if self.include_crossing:
            keep_sorted: torch.Tensor = cdf - sorted_probs < self.cdf_threshold
        else:
            keep_sorted = (cdf <= self.cdf_threshold) | (rank == 0)
        if self.topk_fraction > 0.0:
            n_candidates: torch.Tensor = candidate.sum(dim=-1, keepdim=True)
            keep_sorted = keep_sorted | (
                rank < torch.ceil(n_candidates * self.topk_fraction)
            )
        if self.max_fraction < 1.0:
            n_valid: torch.Tensor = valid.sum(dim=-1, keepdim=True)
            keep_sorted = keep_sorted & (rank < torch.ceil(n_valid * self.max_fraction))
        keep_sorted = keep_sorted & (sorted_probs > 0)
        keep: torch.Tensor = torch.zeros_like(keep_sorted).scatter(
            -1, order, keep_sorted
        )

        return (keep & candidate) | forced

    @classmethod
    def create_from_config(cls, config: MaskerConfig) -> "SpargeTopCdfMasker":
        if not isinstance(config, SpargeTopCdfMaskerConfig):
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(config)
