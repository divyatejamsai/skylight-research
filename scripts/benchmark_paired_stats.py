"""Two runs of a benchmark compared prompt by prompt, with the benchmark's own scorer.

Reads the results.jsonl files that scripts/benchmark_request_eval.py writes (one
directory per run; several directories or a glob per side are joined), pairs the rows of
the two sides by task, context and question, scores every row with the benchmark's
post_run_evaluate, and prints per task the two scores and their difference, then the mean
over tasks of the difference with a paired-bootstrap interval
(benchmark/utils/paired.py).

--metric names the per-task metric when the benchmark reports several (LOFT:
subspan_em); --scale 100 turns a 0..1 metric into points; rows with a split column are
restricted to --split (LOFT reports its test split). --margin M adds two verdicts: the
lower end of the interval is at least -M, and the interval contains 0 with the
difference at least -M. --exclude names a file of "task context_hash" lines (prompts
used to choose settings, say); the result is then also given without them. --keys
names the columns that identify a prompt (rows that share them are paired in their
order of appearance).

Example:
    python scripts/benchmark_paired_stats.py --benchmark ruler128k \
        --baseline "results/full/qwen_ruler128k_*_dense" \
        --other "results/full/qwen_ruler128k_*_carried" --margin 1.0
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import fire
import numpy as np
import pandas as pd

from benchmark.benchmark_registry import (
    create_benchmark_instance,
    ensure_benchmarks_loaded,
)
from benchmark.utils.paired import PAIR_KEYS, pair_rows, paired_bootstrap, row_scores

REFERENCE_COLUMNS: Tuple[str, ...] = ("answer", "answers")


def _directories(spec: Any) -> List[str]:
    """Directories named by a comma-separated list of paths and globs."""
    parts: List[str] = (
        [str(p) for p in spec]
        if isinstance(spec, (list, tuple))
        else str(spec).split(",")
    )
    found: List[str] = []
    for part in filter(None, parts):
        matches: List[str] = sorted(glob.glob(part))
        if not matches:
            raise FileNotFoundError(f"no directory matches {part}")
        found.extend(matches)
    return found


def _rows(directories: List[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for directory in directories:
        path: str = os.path.join(directory, "results.jsonl")
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return pd.DataFrame(records)


def _excluded(path: str) -> Set[Tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        pairs: List[List[str]] = [line.split() for line in f if line.strip()]
    return {(task, context) for task, context in pairs}


def _report(
    label: str,
    tasks: np.ndarray,
    base: np.ndarray,
    other: np.ndarray,
    reps: int,
    seed: int,
    margin: Optional[float],
) -> Dict[str, Any]:
    names: List[str] = sorted(set(tasks.tolist()))
    differences: Dict[str, np.ndarray] = {
        task: other[tasks == task] - base[tasks == task] for task in names
    }
    result: Dict[str, Any] = dict(paired_bootstrap(differences, reps=reps, seed=seed))
    result.update(
        baseline=float(np.mean([base[tasks == task].mean() for task in names])),
        other=float(np.mean([other[tasks == task].mean() for task in names])),
        worse=int(sum((d < 0).sum() for d in differences.values())),
        better=int(sum((d > 0).sum() for d in differences.values())),
    )
    print(
        f"{label}: baseline {result['baseline']:.2f}  other {result['other']:.2f}  "
        f"other - baseline {result['difference']:+.3f} "
        f"[{result['low']:+.3f}, {result['high']:+.3f}]  "
        f"({int(result['tasks'])} tasks, {int(result['prompts'])} prompts, "
        f"worse on {result['worse']}, better on {result['better']})"
    )
    if margin is not None:
        result["lower_end_at_least_minus_margin"] = bool(result["low"] >= -margin)
        result["contains_zero_and_point_at_least_minus_margin"] = bool(
            result["low"] <= 0.0 <= result["high"] and result["difference"] >= -margin
        )
        print(
            f"  lower end >= {-margin:+.1f}: "
            f"{'MET' if result['lower_end_at_least_minus_margin'] else 'NOT MET'}"
            f"   interval contains 0 and difference >= {-margin:+.1f}: "
            + (
                "MET"
                if result["contains_zero_and_point_at_least_minus_margin"]
                else "NOT MET"
            )
        )
    return result


def main(
    benchmark: str,
    baseline: str,
    other: str,
    metric: str = "",
    scale: float = 1.0,
    split: str = "test",
    margin: Optional[float] = None,
    exclude: str = "",
    keys: str = ",".join(PAIR_KEYS),
    reps: int = 10000,
    seed: int = 0,
    out: str = "",
) -> None:
    ensure_benchmarks_loaded()
    bench = create_benchmark_instance(benchmark, None)
    frames: List[pd.DataFrame] = []
    for spec in (baseline, other):
        directories: List[str] = _directories(spec)
        frame: pd.DataFrame = _rows(directories)
        if "split" in frame.columns and split:
            frame = frame[frame["split"] == split].reset_index(drop=True)
        print(f"{len(frame)} rows from {len(directories)} directories of {spec}")
        frames.append(frame)
    same: List[str] = [c for c in REFERENCE_COLUMNS if c in frames[0].columns]
    key_list: List[str] = (
        [str(k) for k in keys] if isinstance(keys, (list, tuple)) else keys.split(",")
    )
    base_rows, other_rows = pair_rows(frames[0], frames[1], key_list, same)
    base: np.ndarray = row_scores(bench, base_rows, metric or None, scale)
    scored: np.ndarray = row_scores(bench, other_rows, metric or None, scale)
    tasks: np.ndarray = base_rows["task"].to_numpy().astype(str)

    print(
        f"{'task':24s} {'n':>5s} {'baseline':>8s} {'other':>8s} {'diff':>7s}  worse better"
    )
    per_task: Dict[str, Dict[str, Any]] = {}
    for task in sorted(set(tasks.tolist())):
        b, o = base[tasks == task], scored[tasks == task]
        per_task[task] = dict(
            n=int(len(b)),
            baseline=float(b.mean()),
            other=float(o.mean()),
            difference=float((o - b).mean()),
            worse=int((o < b).sum()),
            better=int((o > b).sum()),
        )
        print(
            f"{task:24s} {len(b):5d} {b.mean():8.2f} {o.mean():8.2f} "
            f"{(o - b).mean():+7.2f}  {int((o < b).sum()):5d} {int((o > b).sum()):6d}"
        )
    results: Dict[str, Any] = dict(
        benchmark=benchmark,
        baseline=baseline,
        other=other,
        metric=metric,
        scale=scale,
        reps=reps,
        seed=seed,
        per_task=per_task,
        all=_report("ALL PROMPTS", tasks, base, scored, reps, seed, margin),
    )
    if exclude:
        dropped: Set[Tuple[str, str]] = _excluded(exclude)
        keep: np.ndarray = np.array(
            [
                (task, context) not in dropped
                for task, context in zip(tasks, base_rows["context_hash"])
            ]
        )
        print(f"{int((~keep).sum())} of {len(keep)} prompts are named in {exclude}")
        results["without_excluded"] = _report(
            "WITHOUT THEM", tasks[keep], base[keep], scored[keep], reps, seed, margin
        )
        results["excluded"] = int((~keep).sum())
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
