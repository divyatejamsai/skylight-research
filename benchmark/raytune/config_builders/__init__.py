"""Configuration builders for sparse attention configs."""

from .base import BaseConfigBuilder

# Import builders to trigger registration via decorators
from .dense import DenseConfigBuilder  # noqa: E402, F401
from .double_sparsity import DoubleSparsityTopKConfigBuilder  # noqa: E402, F401
from .factory import get_all_config_builders, get_config_builder, register_builder
from .hashattention_topk import HashAttentionTopKConfigBuilder  # noqa: E402, F401
from .magicpig import MagicPigConfigBuilder  # noqa: E402, F401
from .oracle_topk import OracleTopKConfigBuilder  # noqa: E402, F401
from .oracle_topp import OracleTopPConfigBuilder  # noqa: E402, F401
from .pqcache import PQCacheTopKConfigBuilder  # noqa: E402, F401
from .quest_top_k import QuestTopKConfigBuilder  # noqa: E402, F401
from .random_sampling import RandomSamplingConfigBuilder  # noqa: E402, F401
from .socket_topk import SocketTopKConfigBuilder  # noqa: E402, F401
from .sparge_topcdf import SpargeTopCdfConfigBuilder  # noqa: E402, F401
from .vattention_hashattention import (  # noqa: E402, F401
    VAttentionHashAttentionTopKConfigBuilder,
)
from .vattention_oracle import VAttentionOracleTopKConfigBuilder  # noqa: E402, F401
from .vattention_pqcache import VAttentionPQCacheTopKConfigBuilder  # noqa: E402, F401

__all__ = [
    "BaseConfigBuilder",
    "DenseConfigBuilder",
    "DoubleSparsityTopKConfigBuilder",
    "VAttentionOracleTopKConfigBuilder",
    "VAttentionHashAttentionTopKConfigBuilder",
    "VAttentionPQCacheTopKConfigBuilder",
    "OracleTopKConfigBuilder",
    "OracleTopPConfigBuilder",
    "HashAttentionTopKConfigBuilder",
    "MagicPigConfigBuilder",
    "PQCacheTopKConfigBuilder",
    "QuestTopKConfigBuilder",
    "RandomSamplingConfigBuilder",
    "SocketTopKConfigBuilder",
    "SpargeTopCdfConfigBuilder",
    "get_config_builder",
    "get_all_config_builders",
    "register_builder",
]
