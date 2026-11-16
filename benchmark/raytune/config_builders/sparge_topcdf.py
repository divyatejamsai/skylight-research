"""Configuration builder for SpargeAttn Top-CDF attention."""

from typing import Dict, List, Optional, Tuple

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    SinkMaskerConfig,
    SpargeTopCdfMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name


@register_builder("sparge_topcdf")
class SpargeTopCdfConfigBuilder(BaseConfigBuilder):
    """Builder for SpargeAttn Top-CDF sparse attention configurations."""

    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs,
    ) -> Tuple[
        List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
        List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
    ]:
        """Get all SpargeAttn Top-CDF attention configurations.

        The block budget is adaptive, so like oracle_topp the sparsity objective is met
        by searching the thresholds rather than by setting a size.

        Uses:
            sparsity_objectives: List[int] - List of sparsity objectives to build the configurations.
        Ignores:
            memory_objectives: List[int] - List of memory objectives
            model_config: Dict[str, str] - Model configuration

        Returns:
            Tuple of (optimal_configs, to_optimize_configs)
        """
        optimal_configs: List[
            Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]
        ] = []
        to_optimize_configs: List[
            Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]
        ] = []

        for sparsity_objective in sparsity_objectives:
            classes = [SinkMaskerConfig, LocalMaskerConfig, SpargeTopCdfMaskerConfig]
            name: str = get_masker_list_name(
                classes,
                other_params={
                    "builder": "sparge_topcdf",
                    "sparsity_obj": sparsity_objective,
                },
            )

            config = ResearchAttentionConfig(
                masker_configs=[
                    SinkMaskerConfig(sink_size=128),
                    LocalMaskerConfig(window_size=128),
                    SpargeTopCdfMaskerConfig(
                        block_size=16, cdf_threshold=0.9, sim_threshold=0.25
                    ),
                ]
            )

            config.masker_configs[2].search_space = {
                "block_size": tune.grid_search([16, 32, 64]),
                "cdf_threshold": tune.grid_search([0.5, 0.7, 0.8, 0.9, 0.95, 0.98]),
                "sim_threshold": tune.grid_search([-1.0, 0.0, 0.25, 0.5]),
            }

            # Set validity to default (doesn't depend on memory objectives)
            config.validity_constraint = lambda config: True
            # Set objective function
            config.objective = sparsity_objective

            to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs
