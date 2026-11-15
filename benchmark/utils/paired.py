"""Two runs of a benchmark on the same prompts, compared prompt by prompt.

A sparse attention is judged against dense attention on the same prompts. The scores of
the two runs share most of their noise (a hard prompt is hard for both), so the
difference is known far more precisely than either score, provided it is taken per
prompt. This module gives the three steps:

* row_scores: the score of every row, from the benchmark's own post_run_evaluate applied
  to that row alone, so it is the benchmark's metric and not a copy of it;
* pair_rows: the rows of two runs put in the same order, refusing runs that did not
  answer the same prompts;
* paired_bootstrap: the mean over tasks of the per-task mean difference, with a
  percentile interval from resampling prompts within each task.
"""

import contextlib
import io
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PAIR_KEYS: Tuple[str, ...] = ("task", "context_hash", "question")


def metric_of_task(
    metrics: Dict[str, Any], task: str, metric: Optional[str] = None
) -> float:
    """The score of one task in what post_run_evaluate returned.

    Benchmarks report per-task scores under "task_scores" or "task_metrics", as a number
    or as a dictionary of named metrics. ``metric`` names the one to take; it can be
    left out when the task has a single numeric metric.
    """
    failed: Dict[str, str] = metrics.get("failed_tasks", {})
    if task in failed:
        raise ValueError(f"scoring of task {task} failed: {failed[task]}")
    for field in ("task_scores", "task_metrics"):
        per_task: Dict[str, Any] = metrics.get(field, {})
        if task not in per_task:
            continue
        value: Any = per_task[task]
        if not isinstance(value, dict):
            return float(value)
        if metric is None:
            names: List[str] = [
                name for name, v in value.items() if isinstance(v, float)
            ]
            if len(names) != 1:
                raise ValueError(
                    f"task {task} has the metrics {sorted(value)}: name one of them"
                )
            metric = names[0]
        if metric not in value:
            raise ValueError(f"task {task} has no metric {metric}: {sorted(value)}")
        return float(value[metric])
    raise ValueError(f"no score for task {task} in metrics with keys {sorted(metrics)}")


def _evaluate(benchmark: Any, frame: pd.DataFrame) -> Dict[str, Any]:
    """post_run_evaluate on a copy (some scorers edit the frame), without its prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return dict(benchmark.post_run_evaluate(frame.copy()))


def row_scores(
    benchmark: Any,
    rows: pd.DataFrame,
    metric: Optional[str] = None,
    scale: float = 1.0,
    tolerance: float = 0.02,
) -> np.ndarray:
    """The benchmark's score of every row, times ``scale``, in the order of ``rows``.

    Each row is scored by benchmark.post_run_evaluate on a frame of that row alone. This
    is only the row's share of the task score when the metric is a mean over rows, so it
    is checked: per task, the mean of the row scores must be the score of all its rows
    together within ``tolerance`` (benchmarks round what they report), or a ValueError
    is raised.
    """
    if "task" not in rows.columns:
        raise ValueError("rows need a task column")
    scores: np.ndarray = np.zeros(len(rows), dtype=np.float64)
    tasks: np.ndarray = rows["task"].to_numpy()
    for i in range(len(rows)):
        one: Dict[str, Any] = _evaluate(benchmark, rows.iloc[[i]])
        scores[i] = scale * metric_of_task(one, str(tasks[i]), metric)
    for task in sorted(set(tasks.tolist())):
        of_task: np.ndarray = tasks == task
        together: float = scale * metric_of_task(
            _evaluate(benchmark, rows[of_task]), str(task), metric
        )
        mean: float = float(scores[of_task].mean())
        if abs(mean - together) > tolerance:
            raise ValueError(
                f"task {task}: the mean of the row scores is {mean:.4f} and the score "
                f"of the rows together is {together:.4f}: this metric is not a mean "
                "over rows"
            )
    return scores


def _keyed(rows: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Rows sorted by their keys, with a count that tells repeated keys apart."""
    missing: List[str] = [key for key in keys if key not in rows.columns]
    if missing:
        raise ValueError(f"rows have no column {missing}")
    keyed: pd.DataFrame = rows.copy()
    keyed["_key"] = [
        json.dumps([str(value) for value in values])
        for values in zip(*(keyed[key] for key in keys))
    ]
    keyed["_repeat"] = keyed.groupby("_key").cumcount()
    return keyed.sort_values(["_key", "_repeat"], kind="stable").reset_index(drop=True)


def pair_rows(
    first: pd.DataFrame,
    second: pd.DataFrame,
    keys: Sequence[str] = PAIR_KEYS,
    same: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Both frames in one common order, row i of one being the prompt of row i of the
    other.

    A prompt is identified by ``keys``. The two runs must hold the same prompts, each
    the same number of times, and agree on the columns in ``same`` (the references, for
    instance); otherwise a ValueError says what differs.
    """
    a: pd.DataFrame = _keyed(first, keys)
    b: pd.DataFrame = _keyed(second, keys)
    keys_a: List[Tuple[str, int]] = list(zip(a["_key"], a["_repeat"]))
    keys_b: List[Tuple[str, int]] = list(zip(b["_key"], b["_repeat"]))
    if keys_a != keys_b:
        only_a: int = len(set(keys_a) - set(keys_b))
        only_b: int = len(set(keys_b) - set(keys_a))
        raise ValueError(
            f"the runs differ in their prompts: {len(a)} and {len(b)} rows, "
            f"{only_a} only in the first, {only_b} only in the second"
        )
    for column in same:
        text_a: List[str] = [json.dumps(_listed(v), default=str) for v in a[column]]
        text_b: List[str] = [json.dumps(_listed(v), default=str) for v in b[column]]
        if text_a != text_b:
            differing: int = sum(x != y for x, y in zip(text_a, text_b))
            raise ValueError(f"the runs differ in {column} on {differing} rows")
    drop: List[str] = ["_key", "_repeat"]
    return a.drop(columns=drop), b.drop(columns=drop)


def _listed(value: Any) -> Any:
    """Arrays and tuples as lists, so equal references compare equal."""
    if isinstance(value, np.ndarray):
        return [_listed(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_listed(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def paired_bootstrap(
    differences: Dict[str, np.ndarray],
    reps: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict[str, float]:
    """The mean over tasks of the mean per-prompt difference, with an interval.

    ``differences`` holds, per task, the per-prompt differences of the two runs. Prompts
    are resampled with replacement within each task (tasks in sorted order, numpy's
    default_rng(seed)), ``reps`` times; the interval is the percentile interval of the
    resampled statistic at ``level``.
    """
    if not differences:
        raise ValueError("no differences")
    tasks: List[str] = sorted(differences)
    rng: np.random.Generator = np.random.default_rng(seed)
    resampled: np.ndarray = np.zeros(reps, dtype=np.float64)
    point: float = 0.0
    for task in tasks:
        d: np.ndarray = np.asarray(differences[task], dtype=np.float64)
        if d.ndim != 1 or len(d) == 0:
            raise ValueError(f"task {task} has no differences")
        point += float(d.mean())
        index: np.ndarray = rng.integers(0, len(d), size=(reps, len(d)))
        resampled += d[index].mean(axis=1)
    point /= len(tasks)
    resampled /= len(tasks)
    tail: float = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(resampled, [tail, 100.0 - tail])
    return {
        "difference": point,
        "low": float(low),
        "high": float(high),
        "tasks": float(len(tasks)),
        "prompts": float(sum(len(differences[task]) for task in tasks)),
    }
