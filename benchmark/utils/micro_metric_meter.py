"""Micro-metrics of one request at a time, added up instead of kept.

MicroMetricLogger writes one line per attention call and metric. At 128k tokens of
context that is megabytes for every request, when what is wanted is usually a mean. A
MicroMetricMeter turns the logger on for one request (start) and, at the end of the
request (stop), adds up what was logged, removes the file and turns the logger off. It
is a recorder for benchmark/utils/answer_log.py:

    meter = MicroMetricMeter(["research_attention_density"], "results/run/micro")
    adapter = LoggedAdapter(ModelAdapterHF(...), "results/run/answers.jsonl", run, meter)

The sums are kept apart for the decode steps (calls with one query token) and for the
other calls (a question processed in one call), which sparse attentions may treat
differently:

    {"research_attention_density": {"decode": [sum, calls], "chunk": [sum, calls]}}

mean_of gives the mean over the calls of many requests.
"""

import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence

from sparse_attention_hub.metric_logging import MicroMetricLogger

Sums = Dict[str, Dict[str, List[float]]]


class MicroMetricMeter:
    """Sums of the micro-metrics logged between start() and stop()."""

    def __init__(self, metrics: Sequence[str], directory: str) -> None:
        """
        Args:
            metrics: Identifiers of registered micro-metrics.
            directory: Where the logger writes meanwhile (a file of this meter's own,
                removed at every stop).
        """
        self.metrics: List[str] = list(metrics)
        self._directory: str = directory
        self._file: str = os.path.join(directory, "micro_metrics.jsonl")

    def start(self) -> None:
        """Turn the logger on, for these metrics alone, with an empty file."""
        logger = MicroMetricLogger()
        logger.flush()  # what was logged before goes where it was meant to go
        if os.path.exists(self._file):
            os.remove(self._file)
        logger.configure_logging(self._directory, enabled_metrics=self.metrics)
        missing = set(self.metrics) - logger.get_enabled_metrics()
        if missing:
            raise ValueError(
                f"micro-metrics that are not registered: {sorted(missing)}"
            )

    def stop(self) -> Sums:
        """Turn the logger off and return the sums of what it logged since start()."""
        logger = MicroMetricLogger()
        logger.flush()
        sums: Sums = {
            metric: {"decode": [0.0, 0], "chunk": [0.0, 0]} for metric in self.metrics
        }
        if os.path.exists(self._file):
            with open(self._file, encoding="utf-8") as f:
                for line in f:
                    record: Dict[str, Any] = json.loads(line)
                    if record["metric"] not in sums:
                        continue
                    one_query: bool = (record.get("metadata") or {}).get("q_len") == 1
                    cell = sums[record["metric"]]["decode" if one_query else "chunk"]
                    cell[0] += float(record["value"])
                    cell[1] += 1
            os.remove(self._file)
        logger.enable_metrics([])
        return sums


def mean_of(measured: Iterable[Sums], metric: str, kind: str = "decode") -> float:
    """The mean of a metric over the calls of many requests (nan without a call).

    Args:
        measured: What MicroMetricMeter.stop returned for each request.
        metric: The metric's identifier.
        kind: "decode" (calls with one query token) or "chunk" (the other calls).
    """
    total: float = 0.0
    calls: float = 0
    for sums in measured:
        total += sums[metric][kind][0]
        calls += sums[metric][kind][1]
    return total / calls if calls else math.nan
