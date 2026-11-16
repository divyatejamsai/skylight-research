"""Research maskers for a benchmark run: configs from text, and decode-only masking.

masker_configs_from_text builds the masker configs of a ResearchAttentionConfig from
the names of their classes, in the order they are applied:

    "SinkMaskerConfig(sink_size=128);LocalMaskerConfig(window_size=128);
     OracleTopKConfig(heavy_size=0.05)"

DecodeOnlyResearchAttention applies the maskers to the decode steps alone. In
ModelAdapterHF.process_request a research attention also masks the call that reads the
question, while an attention such as TwoStageBlockSparse attends densely there and is
sparse from the first generated token on. To compare the two on the same footing, give
the adapter this attention before its first request:

    adapter = ModelAdapterHF(model_name, ResearchAttentionConfig(masker_configs=configs))
    adapter.sparse_attention = DecodeOnlyResearchAttention(
        adapter.sparse_attention_config, adapter.sparse_attention.maskers
    )
"""

import ast
from typing import Any, List, Optional, Tuple

import torch

from sparse_attention_hub.metric_logging import MicroMetricLogger
from sparse_attention_hub.sparse_attention import ResearchAttention
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed import (
    implementations as fixed_maskers,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling import (
    implementations as sampling_maskers,
)
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_true_attention_output,
)

DENSITY_METRIC = "research_attention_density"
ERROR_METRIC = "research_attention_output_error"


def masker_configs_from_text(spec: str) -> List[Any]:
    """Masker configs from 'Name(key=value, ...);Name(...)'.

    Args:
        spec: The configs' class names with their arguments by name (Python literals),
            separated by semicolons.

    Returns:
        The configs, in the order given.

    Raises:
        ValueError: For a name that is not a masker config, positional arguments, or
            an empty spec.
    """
    configs: List[Any] = []
    for item in filter(None, (part.strip() for part in str(spec).split(";"))):
        name, _, arguments = item.partition("(")
        cls: Any = getattr(fixed_maskers, name, None) or getattr(
            sampling_maskers, name, None
        )
        if not name.endswith("Config") or cls is None or not item.endswith(")"):
            raise ValueError(f"not a masker config: {item!r}")
        call: Any = ast.parse("f(" + arguments[:-1] + ")", mode="eval").body
        if call.args:
            raise ValueError(f"{name}: give its arguments by name")
        configs.append(
            cls(**{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
        )
    if not configs:
        raise ValueError("no masker config is named")
    return configs


class DecodeOnlyResearchAttention(ResearchAttention):
    """ResearchAttention whose maskers apply to the decode steps only.

    A call with more than one query token (the question of a request) attends densely
    and logs a density of 1 and an output error of 0, as TwoStageBlockSparse does.
    """

    def custom_attention(  # type: ignore[override]
        self,
        module: Any,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Masked attention for one query token, dense attention for more.

        Args:
            module: The attention module.
            queries: (b, h, s_q, d).
            keys: (b, h_kv, s_k, d).
            values: (b, h_kv, s_k, d).
            attention_mask: The mask HuggingFace built.
            scaling: Scaling factor for attention logits.
            dropout: Dropout probability.
            **kwargs: sparse_meta_data and layer_idx, as for ResearchAttention.

        Returns:
            Attention output and attention weights (None for a dense call).
        """
        if queries.shape[2] == 1:
            return super().custom_attention(
                module,
                queries,
                keys,
                values,
                attention_mask,
                scaling,
                dropout,
                **kwargs,
            )
        logger = MicroMetricLogger()
        for name, value in ((DENSITY_METRIC, 1.0), (ERROR_METRIC, 0.0)):
            if logger.is_metric_enabled(name):
                logger.log(
                    name,
                    value,
                    metadata={
                        "layer_idx": kwargs.get("layer_idx"),
                        "q_len": queries.shape[2],
                        "k_len": keys.shape[2],
                    },
                )
        kwargs.pop("sparse_meta_data", None)
        return get_true_attention_output(
            module, queries, keys, values, attention_mask, scaling, dropout, **kwargs
        )
