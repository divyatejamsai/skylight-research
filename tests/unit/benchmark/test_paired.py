"""Unit tests for benchmark/utils/paired.py: two runs compared prompt by prompt."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from benchmark.infinite_bench import InfiniteBench
from benchmark.loft.loft_rag import LoftRag
from benchmark.ruler128k import Ruler128K
from benchmark.utils.paired import (
    metric_of_task,
    pair_rows,
    paired_bootstrap,
    row_scores,
)


def _ruler_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"task": "niah_single_1", "answer": ["4821"], "predicted_answer": " 4821."},
        {"task": "niah_single_1", "answer": ["4821"], "predicted_answer": " 1284."},
        {"task": "niah_multivalue", "answer": ["11", "22"], "predicted_answer": "11"},
        {
            "task": "niah_multivalue",
            "answer": ["11", "22"],
            "predicted_answer": "22 11",
        },
        {"task": "qa_1", "answer": ["Paris", "Lyon"], "predicted_answer": "in Lyon"},
    ]
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestRowScores:
    def test_ruler(self) -> None:
        """One metric per task, on a 0 to 100 scale; qa counts any reference."""
        scores = row_scores(Ruler128K(), _ruler_rows())
        assert scores.tolist() == [100.0, 0.0, 50.0, 100.0, 100.0]

    def test_the_rows_are_left_as_they_were(self) -> None:
        rows = _ruler_rows()
        row_scores(Ruler128K(), rows)
        pd.testing.assert_frame_equal(rows, _ruler_rows())

    def test_infinite_bench(self) -> None:
        rows = pd.DataFrame(
            {
                "task": ["passkey", "passkey", "math_find"],
                "question": ["What is the pass key?"] * 2 + ["The largest number: "],
                "answer": [["71432"], ["71432"], ["88"]],
                "predicted_answer": [" **71432**.", " 71433", "88"],
            }
        )
        scores = row_scores(InfiniteBench(), rows, scale=100.0)
        assert scores.tolist() == [100.0, 0.0, 100.0]

    def test_loft_needs_the_metric_named(self) -> None:
        rows = pd.DataFrame(
            {
                "task": ["nq_128k", "nq_128k"],
                "answers": [["Paris"], ["Paris"]],
                "answer_prefix": ["Final Answer: "] * 2,
                "predicted_answer": [
                    "It is the capital.\nFinal Answer: ['Paris, France']",
                    "Final Answer: ['Lyon']",
                ],
            }
        )
        bench = LoftRag(subsets_to_run=["nq_128k"])
        assert row_scores(bench, rows, "subspan_em", 100.0).tolist() == [100.0, 0.0]
        assert row_scores(bench, rows, "em", 100.0).tolist() == [0.0, 0.0]
        with pytest.raises(ValueError, match="name one of them"):
            row_scores(bench, rows)

    def test_a_metric_that_is_not_a_mean_over_rows_is_refused(self) -> None:
        class BestRow:
            """Reports the best row of a task as its score."""

            def post_run_evaluate(self, frame: pd.DataFrame) -> Dict[str, Any]:
                return {"task_scores": {"t": float(frame["value"].max())}}

        rows = pd.DataFrame({"task": ["t", "t"], "value": [0.0, 1.0]})
        with pytest.raises(ValueError, match="not a mean over rows"):
            row_scores(BestRow(), rows)

    def test_a_task_whose_scoring_failed_has_no_score(self) -> None:
        metrics = {"task_scores": {}, "failed_tasks": {"code_run": "IndexError: x"}}
        with pytest.raises(ValueError, match="IndexError"):
            metric_of_task(metrics, "code_run")
        with pytest.raises(ValueError, match="no score for task"):
            metric_of_task({"task_scores": {"a": 1.0}}, "b")
        with pytest.raises(ValueError, match="no metric f1"):
            metric_of_task({"task_metrics": {"a": {"em": 1.0}}}, "a", "f1")


def _run(order: List[int], predictions: List[str]) -> pd.DataFrame:
    """Four prompts (one context holds two questions) in the given order."""
    prompts = [
        ("vt", "c1", "q1", ["A"]),
        ("vt", "c1", "q2", ["B"]),
        ("vt", "c2", "q1", ["C"]),
        ("cwe", "c1", "q1", ["D"]),
    ]
    return pd.DataFrame(
        [
            dict(
                task=prompts[i][0],
                context_hash=prompts[i][1],
                question=prompts[i][2],
                answer=prompts[i][3],
                predicted_answer=predictions[i],
            )
            for i in order
        ]
    )


@pytest.mark.unit
class TestPairRows:
    def test_two_orders_of_the_same_prompts(self) -> None:
        a, b = pair_rows(
            _run([0, 1, 2, 3], ["a0", "a1", "a2", "a3"]),
            _run([3, 1, 0, 2], ["b0", "b1", "b2", "b3"]),
            same=["answer"],
        )
        assert a["predicted_answer"].str[1:].tolist() == (
            b["predicted_answer"].str[1:].tolist()
        )
        assert sorted(a["predicted_answer"]) == ["a0", "a1", "a2", "a3"]
        assert "_key" not in a.columns

    def test_a_prompt_that_occurs_twice(self) -> None:
        first = _run([0, 0, 1], ["x", "x", "y"])
        a, b = pair_rows(first, _run([1, 0, 0], ["x", "x", "y"]))
        assert a["question"].tolist() == b["question"].tolist()
        with pytest.raises(ValueError, match="differ in their prompts"):
            pair_rows(first, _run([0, 1, 1], ["x", "x", "y"]))

    def test_runs_of_different_prompts_are_refused(self) -> None:
        preds = ["p"] * 4
        with pytest.raises(ValueError, match="1 only in the first, 0 only in the"):
            pair_rows(_run([0, 1, 2], preds), _run([0, 1], preds))
        with pytest.raises(ValueError, match="no column"):
            pair_rows(_run([0], preds).drop(columns=["context_hash"]), _run([0], preds))

    def test_runs_with_different_references_are_refused(self) -> None:
        preds = ["p"] * 4
        other = _run([0, 1], preds)
        other.at[1, "answer"] = ["Z"]
        with pytest.raises(ValueError, match="differ in answer on 1 rows"):
            pair_rows(_run([0, 1], preds), other, same=["answer"])
        as_arrays = _run([0, 1], preds)
        as_arrays["answer"] = [np.array(v, dtype=object) for v in as_arrays["answer"]]
        pair_rows(_run([0, 1], preds), as_arrays, same=["answer"])


@pytest.mark.unit
class TestPairedBootstrap:
    def test_a_constant_difference_has_no_width(self) -> None:
        result = paired_bootstrap({"a": np.full(50, -2.0), "b": np.full(7, 4.0)}, 200)
        assert result["difference"] == result["low"] == result["high"] == 1.0
        assert (result["tasks"], result["prompts"]) == (2.0, 57.0)

    def test_tasks_count_equally_whatever_their_size(self) -> None:
        differences = {"big": np.zeros(1000), "small": np.full(10, 10.0)}
        assert paired_bootstrap(differences, 100)["difference"] == 5.0

    def test_the_interval_of_a_known_case(self) -> None:
        """100 prompts, half +1 and half -1: the standard error of the mean is 0.1."""
        d = np.array([1.0, -1.0] * 50)
        result = paired_bootstrap({"t": d}, reps=20000, seed=0)
        assert result["difference"] == 0.0
        assert -0.23 < result["low"] < -0.16
        assert 0.16 < result["high"] < 0.23

    def test_the_seed_fixes_the_interval(self) -> None:
        d = {"t": np.random.default_rng(1).normal(size=80)}
        assert paired_bootstrap(d, 500, seed=3) == paired_bootstrap(d, 500, seed=3)
        assert paired_bootstrap(d, 500, seed=3) != paired_bootstrap(d, 500, seed=4)

    def test_nothing_to_compare(self) -> None:
        with pytest.raises(ValueError):
            paired_bootstrap({})
        with pytest.raises(ValueError, match="no differences"):
            paired_bootstrap({"t": np.array([])})
