"""Unit tests for the InfiniteBench benchmark: scoring of rows in the Hub's format.

The dataset on the Hub (benchmark/infinite_bench/create_huggingface_dataset.py) stores
every answer as a sequence of strings, and for the two multiple-choice tasks only the
letter of the correct option. A dataframe hands such a sequence over as a numpy array.
The scorers are the original InfiniteBench ones, written for lists with numbers as
numbers and, for the multiple-choice tasks, the option's text before its letter.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from benchmark.infinite_bench import InfiniteBench
from benchmark.infinite_bench.calculate_metrics import calculate_metrics

CODE_DEBUG_QUESTION = (
    "Which funtion has deliberate error?\n\nOptions:\n"
    "A. Resource.pkgname\nB. repack_carchive\nC. cmd_gen\nD. _init"
)
CHOICE_QUESTION = (
    "Where does the story begin?\n\nOnly one of the following options is correct, tell "
    "me the answer using one single letter (A, B, C, or D). Don't say anything else.\n"
    "A. Hayslope\nB. Hall Farm\nC. Snowfield\nD. Stoniton"
)

# task -> (question, answer as stored on the Hub, a right prediction, a wrong one)
ROWS: Dict[str, Tuple[str, List[str], str, str]] = {
    "passkey": ("What is the pass key?", ["71432"], " **71432**.", " 71433"),
    "number_string": (
        "What is the sequence of digits?",
        ["3000077442"],
        " **3000077442**.",
        " 3000077443",
    ),
    "kv_retrieval": (
        '\nKey: "798c2306"\n ',
        ["5e6b7b90-710d"],
        ':\n\n**"5e6b7b90-710d"**.',
        ' "5e6b7b90-710e"',
    ),
    "code_run": (
        "Please give me the exact number of the return value of func_6577(-7).",
        ["-14"],
        " -14",
        " 14",
    ),
    "code_debug": (CODE_DEBUG_QUESTION, ["B"], " **B. repack_carchive**", " C"),
    "math_find": ("The largest number of the list is: ", ["88"], "88", "89"),
    "longbook_qa_eng": (
        "What's the name of the summer home?",
        ["the Hebrides"],
        " The Hebrides",
        " Skye",
    ),
    "longdialogue_qa_eng": (
        "Which character is $$MASK$$ ?",
        ["ACE", "ACE ROTHSTEIN"],
        " Ace Rothstein",
        " Nicky",
    ),
    "longbook_choice_eng": (CHOICE_QUESTION, ["C"], " C.", " A."),
}


def _frame(task: str, predictions: List[str]) -> pd.DataFrame:
    """Result rows of one task as the benchmark runner builds them."""
    question, answer, _, _ = ROWS[task]
    rows: List[Dict[str, Any]] = [
        {
            "question": question,
            "answer": np.array(answer, dtype=object),
            "task": task,
            "predicted_answer": prediction,
        }
        for prediction in predictions
    ]
    return pd.DataFrame(rows)


@pytest.mark.unit
class TestInfiniteBenchScoring:
    """Every task is scored on rows in the Hub's format."""

    def test_all_datasets_have_a_row_here(self) -> None:
        assert sorted(ROWS) == sorted(InfiniteBench.all_datasets)

    @pytest.mark.parametrize("task", sorted(ROWS))
    def test_right_and_wrong_predictions(self, task: str) -> None:
        _, _, right, wrong = ROWS[task]
        assert calculate_metrics(_frame(task, [right])) == 1.0
        assert calculate_metrics(_frame(task, [wrong])) == 0.0
        assert calculate_metrics(_frame(task, [right, wrong])) == 0.5

    @pytest.mark.parametrize("task", sorted(ROWS))
    def test_answer_given_as_a_list(self, task: str) -> None:
        """A frame read back from a file can hold lists instead of arrays."""
        _, answer, right, wrong = ROWS[task]
        frame = _frame(task, [right, wrong])
        frame["answer"] = [list(answer), list(answer)]
        assert calculate_metrics(frame) == 0.5

    def test_math_find_with_a_decimal_answer(self) -> None:
        frame = _frame("math_find", ["3.5", "3"])
        frame["answer"] = [np.array(["3.5"], dtype=object)] * 2
        assert calculate_metrics(frame) == 0.5

    def test_code_debug_by_function_name(self) -> None:
        """The original scorer also accepts the name of the function without a letter."""
        assert calculate_metrics(_frame("code_debug", ["repack_carchive"])) == 1.0
        assert calculate_metrics(_frame("code_debug", ["cmd_gen"])) == 0.0

    def test_choice_by_option_text(self) -> None:
        """The original scorer also accepts the text of the option without a letter."""
        assert calculate_metrics(_frame("longbook_choice_eng", ["Snowfield"])) == 1.0

    def test_choice_without_options_in_the_question(self) -> None:
        """Without option lines the letter alone is scored."""
        frame = _frame("code_debug", [" B", " C"])
        frame["question"] = "Which function has the error?"
        assert calculate_metrics(frame) == 0.5


@pytest.mark.unit
class TestInfiniteBenchPostRunEvaluate:
    """What the benchmark reports for a run."""

    def test_scores_of_every_task(self) -> None:
        frames = [_frame(task, [ROWS[task][2], ROWS[task][3]]) for task in sorted(ROWS)]
        metrics = InfiniteBench().post_run_evaluate(
            pd.concat(frames, ignore_index=True)
        )
        assert metrics["task_scores"] == {task: 0.5 for task in ROWS}
        assert metrics["overall_score"] == 0.5
        assert metrics["failed_tasks"] == {}
        assert metrics["summary"] == {"total_tasks": 9, "total_samples": 18}

    def test_a_scorer_error_is_not_a_score_of_zero(self) -> None:
        """A task whose scoring raises is reported as failed, not as 0.0."""
        good = _frame("passkey", [ROWS["passkey"][2]])
        bad = _frame("math_find", ["88"])
        bad["answer"] = [np.array(["not a number"], dtype=object)]
        metrics = InfiniteBench().post_run_evaluate(
            pd.concat([good, bad], ignore_index=True)
        )
        assert metrics["task_scores"] == {"passkey": 1.0}
        assert metrics["overall_score"] == 1.0
        assert list(metrics["failed_tasks"]) == ["math_find"]
        assert "not a number" in metrics["failed_tasks"]["math_find"]

    def test_no_results(self) -> None:
        empty = pd.DataFrame(columns=["question", "answer", "task", "predicted_answer"])
        assert InfiniteBench().post_run_evaluate(empty) == {
            "error": "No results to evaluate"
        }
