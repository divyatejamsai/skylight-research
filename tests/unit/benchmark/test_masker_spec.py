"""Unit tests for benchmark/utils/masker_spec.py."""

from typing import Any

import pytest
import torch
from torch import nn

from benchmark.utils.masker_spec import (
    DENSITY_METRIC,
    ERROR_METRIC,
    DecodeOnlyResearchAttention,
    masker_configs_from_text,
)
from sparse_attention_hub.sparse_attention import (
    LocalMaskerConfig,
    ResearchAttention,
    ResearchAttentionConfig,
    SinkMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    OracleTopKConfig,
    QuestTopKMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)
from sparse_attention_hub.sparse_attention.utils.mask_attention_utils import (
    get_true_attention_output,
)


@pytest.mark.unit
class TestMaskerConfigsFromText:
    def test_configs_in_the_order_given(self) -> None:
        configs = masker_configs_from_text(
            "SinkMaskerConfig(sink_size=128); LocalMaskerConfig(window_size=64);"
            "OracleTopKConfig(heavy_size=0.05)"
        )
        assert configs == [
            SinkMaskerConfig(sink_size=128),
            LocalMaskerConfig(window_size=64),
            OracleTopKConfig(heavy_size=0.05),
        ]

    def test_several_arguments_and_a_sampling_masker(self) -> None:
        configs = masker_configs_from_text(
            "QuestTopKMaskerConfig(heavy_size=0.1, page_size=16, label_bits=16);"
            "AdaptiveSamplingMaskerConfig(base_rate_sampling=0.05, epsilon=0.25, "
            "delta=0.25, init_offset=128, local_offset=128)"
        )
        assert configs == [
            QuestTopKMaskerConfig(heavy_size=0.1, page_size=16, label_bits=16),
            AdaptiveSamplingMaskerConfig(
                base_rate_sampling=0.05,
                epsilon=0.25,
                delta=0.25,
                init_offset=128,
                local_offset=128,
            ),
        ]

    @pytest.mark.parametrize(
        "spec, message",
        [
            ("", "no masker config"),
            (" ; ", "no masker config"),
            ("NoSuchConfig(x=1)", "not a masker config"),
            ("OracleTopK(heavy_size=0.05)", "not a masker config"),
            ("OracleTopKConfig", "not a masker config"),
            ("OracleTopKConfig(0.05)", "by name"),
        ],
    )
    def test_what_is_not_a_config_is_refused(self, spec: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            masker_configs_from_text(spec)

    def test_arguments_are_literals_not_code(self) -> None:
        with pytest.raises(ValueError):
            masker_configs_from_text("OracleTopKConfig(heavy_size=__import__('os'))")


def _qkv(lq: int, n: int = 200) -> Any:
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, lq, 32, generator=g)
    k = torch.randn(1, 4, n, 32, generator=g)
    v = torch.randn(1, 4, n, 32, generator=g)
    return q, k, v


def _call(attention: ResearchAttention, q: Any, k: Any, v: Any) -> torch.Tensor:
    return attention.custom_attention(
        module=nn.Module(),
        queries=q,
        keys=k,
        values=v,
        attention_mask=None,
        scaling=32**-0.5,
        dropout=0.0,
        sparse_meta_data={},
        layer_idx=0,
    )[0]


@pytest.mark.unit
class TestDecodeOnlyResearchAttention:
    CONFIG = ResearchAttentionConfig(masker_configs=[LocalMaskerConfig(window_size=8)])

    def _attentions(self) -> Any:
        masked = ResearchAttention.create_from_config(self.CONFIG)
        return masked, DecodeOnlyResearchAttention(self.CONFIG, masked.maskers)

    def test_a_decode_step_is_masked(self) -> None:
        masked, decode_only = self._attentions()
        q, k, v = _qkv(1)
        dense, _ = get_true_attention_output(
            nn.Module(), q, k, v, None, 32**-0.5, 0.0
        )
        out = _call(decode_only, q, k, v)
        torch.testing.assert_close(out, _call(masked, q, k, v))
        assert (out - dense).abs().max() > 1e-2

    def test_a_question_is_attended_densely(self) -> None:
        masked, decode_only = self._attentions()
        q, k, v = _qkv(5)
        dense, _ = get_true_attention_output(
            nn.Module(), q, k, v, None, 32**-0.5, 0.0
        )
        torch.testing.assert_close(_call(decode_only, q, k, v), dense)
        assert (_call(masked, q, k, v) - dense).abs().max() > 1e-2

    def test_micro_metrics_of_both_kinds_of_call(self, micro_metric_log: Any) -> None:
        records = micro_metric_log(DENSITY_METRIC, ERROR_METRIC)
        _, decode_only = self._attentions()
        _call(decode_only, *_qkv(5))
        _call(decode_only, *_qkv(1))
        got = [(r["metric"], r["value"], r["metadata"]["q_len"]) for r in records()]
        assert got[:2] == [(DENSITY_METRIC, 1.0, 5), (ERROR_METRIC, 0.0, 5)]
        # The masked step: the 8 local tokens of 200, and an error against dense.
        assert got[2] == (DENSITY_METRIC, pytest.approx(8 / 200), 1)
        assert got[3][0] == ERROR_METRIC and got[3][1] > 0 and got[3][2] == 1
