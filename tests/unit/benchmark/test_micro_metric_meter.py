"""Unit tests for benchmark/utils/micro_metric_meter.py."""

import math
from pathlib import Path
from typing import Any

import pytest

from benchmark.utils.micro_metric_meter import MicroMetricMeter, mean_of
from sparse_attention_hub.metric_logging import MicroMetricLogger

DENSITY = "research_attention_density"
ERROR = "research_attention_output_error"


def _log(metric: str, value: float, q_len: int) -> None:
    MicroMetricLogger().log(
        metric, value, metadata={"layer_idx": 0, "q_len": q_len, "k_len": 100}
    )


@pytest.mark.unit
class TestMicroMetricMeter:
    def test_sums_decode_steps_and_other_calls_apart(
        self, micro_metric_log: Any, tmp_path: Path
    ) -> None:
        micro_metric_log(DENSITY, ERROR)  # registers them; the meter sets its own file
        meter = MicroMetricMeter([DENSITY, ERROR], str(tmp_path / "micro"))
        meter.start()
        _log(DENSITY, 1.0, 7)
        _log(DENSITY, 0.25, 1)
        _log(DENSITY, 0.5, 1)
        _log(ERROR, 0.125, 1)
        assert meter.stop() == {
            DENSITY: {"decode": [0.75, 2], "chunk": [1.0, 1]},
            ERROR: {"decode": [0.125, 1], "chunk": [0.0, 0]},
        }

    def test_each_request_starts_from_nothing(
        self, micro_metric_log: Any, tmp_path: Path
    ) -> None:
        micro_metric_log(DENSITY)
        meter = MicroMetricMeter([DENSITY], str(tmp_path / "micro"))
        meter.start()
        _log(DENSITY, 0.5, 1)
        meter.stop()
        # The file is gone and the logger is off between requests.
        assert not (tmp_path / "micro" / "micro_metrics.jsonl").exists()
        assert not MicroMetricLogger().is_metric_enabled(DENSITY)
        meter.start()
        _log(DENSITY, 0.25, 1)
        assert meter.stop() == {DENSITY: {"decode": [0.25, 1], "chunk": [0.0, 0]}}

    def test_more_records_than_the_logger_keeps_in_memory(
        self, micro_metric_log: Any, tmp_path: Path
    ) -> None:
        micro_metric_log(DENSITY)
        meter = MicroMetricMeter([DENSITY], str(tmp_path / "micro"))
        meter.start()
        for _ in range(2500):
            _log(DENSITY, 0.5, 1)
        assert meter.stop()[DENSITY]["decode"] == [1250.0, 2500]

    def test_only_its_own_metrics_are_measured(
        self, micro_metric_log: Any, tmp_path: Path
    ) -> None:
        micro_metric_log(DENSITY, ERROR)
        meter = MicroMetricMeter([DENSITY], str(tmp_path / "micro"))
        meter.start()
        assert not MicroMetricLogger().is_metric_enabled(ERROR)
        _log(DENSITY, 0.5, 1)
        assert meter.stop() == {DENSITY: {"decode": [0.5, 1], "chunk": [0.0, 0]}}

    def test_an_unregistered_metric_is_refused(
        self, micro_metric_log: Any, tmp_path: Path
    ) -> None:
        micro_metric_log()
        meter = MicroMetricMeter(["no_such_metric"], str(tmp_path / "micro"))
        with pytest.raises(ValueError, match="not registered"):
            meter.start()


@pytest.mark.unit
class TestMeanOf:
    def test_mean_over_the_calls_of_all_requests(self) -> None:
        measured = [
            {DENSITY: {"decode": [0.5, 1], "chunk": [1.0, 1]}},
            {DENSITY: {"decode": [0.25, 3], "chunk": [0.0, 0]}},
        ]
        assert mean_of(measured, DENSITY) == pytest.approx(0.75 / 4)
        assert mean_of(measured, DENSITY, "chunk") == 1.0

    def test_no_call_is_nan(self) -> None:
        assert math.isnan(mean_of([], DENSITY))
        measured = [{DENSITY: {"decode": [0.0, 0], "chunk": [1.0, 2]}}]
        assert math.isnan(mean_of(measured, DENSITY))
