"""Pytest configuration and shared fixtures for sparse attention hub tests."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List

import numpy as np
import pytest


@pytest.fixture
def sample_attention_scores() -> np.ndarray:
    """Generate sample attention scores for testing."""
    return np.random.randn(32, 8, 512, 512)


@pytest.fixture
def sample_sparse_mask() -> np.ndarray:
    """Generate sample sparse attention mask for testing."""
    mask = np.random.rand(32, 8, 512, 512) > 0.5
    return mask.astype(np.bool_)


@pytest.fixture
def small_sequence_length() -> int:
    """Small sequence length for fast tests."""
    return 128


@pytest.fixture
def medium_sequence_length() -> int:
    """Medium sequence length for integration tests."""
    return 512


@pytest.fixture
def large_sequence_length() -> int:
    """Large sequence length for performance tests."""
    return 2048


@pytest.fixture
def num_heads() -> int:
    """Number of attention heads for testing."""
    return 8


@pytest.fixture
def batch_size() -> int:
    """Batch size for testing."""
    return 4


@pytest.fixture
def micro_metric_log(
    tmp_path: Path,
) -> Iterator[Callable[..., Callable[[], List[Dict[str, Any]]]]]:
    """Turns MicroMetricLogger on for one test.

    Call it with the metrics to enable; it returns a function giving the records
    logged so far. The logger is one object for the whole process and other tests
    clear its registry, so the metrics are registered here if they are missing, and
    the logger and the registry are put back afterwards.
    """
    from sparse_attention_hub.metric_logging import MicroMetricLogger

    logger = MicroMetricLogger()
    names = (
        "log_path",
        "enabled_metrics",
        "max_records",
        "sampling_factor",
        "_total_records_logged",
    )
    saved = {name: getattr(logger, name) for name in names}
    registered = dict(MicroMetricLogger._registered_metrics)

    def enable(*metrics: str) -> Callable[[], List[Dict[str, Any]]]:
        for metric in metrics:
            MicroMetricLogger._registered_metrics.setdefault(metric, float)
        logger.configure_logging(str(tmp_path), enabled_metrics=list(metrics))

        def records() -> List[Dict[str, Any]]:
            logger.flush()
            path = tmp_path / "micro_metrics.jsonl"
            if not path.exists():
                return []
            return [json.loads(line) for line in path.read_text().splitlines()]

        return records

    yield enable
    logger.log_queue.clear()
    for name, value in saved.items():
        setattr(logger, name, value)
    MicroMetricLogger._registered_metrics.clear()
    MicroMetricLogger._registered_metrics.update(registered)
