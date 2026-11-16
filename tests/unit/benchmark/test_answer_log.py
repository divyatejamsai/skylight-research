"""Unit tests for benchmark/utils/answer_log.py: a benchmark run that can be continued."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest
import torch

from benchmark.base import Benchmark
from benchmark.utils.answer_log import (
    LoggedAdapter,
    context_hash,
    request_key,
    request_seed,
)
from sparse_attention_hub.adapters.base import Request, RequestResponse

RUN: Dict[str, Any] = {"model": "a-model", "attention": "dense", "subsets": ["x"]}


class CountingAdapter:
    """Answers every question with a text made from it, and counts its requests.

    With ``fail_at`` it raises at that request, as a run that is interrupted there.
    """

    model_name: str = "a-model"

    def __init__(self, fail_at: int = -1) -> None:
        self.requests: int = 0
        self.fail_at: int = fail_at

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
    ) -> RequestResponse:
        if self.requests == self.fail_at:
            raise RuntimeError("interrupted")
        self.requests += 1
        questions: List[str] = (
            request.questions
            if isinstance(request.questions, list)
            else [request.questions]
        )
        return RequestResponse(responses=[f"answer to {q}" for q in questions])


class FiveQuestions(Benchmark):
    """Five questions on four contexts, scored by whether the answer names them."""

    all_datasets: List[str] = ["t"]
    benchmark_name: str = "five_questions"
    huggingface_dataset_id: str = "none/none"

    def _load_datasets(self) -> pd.DataFrame:
        contexts: List[str] = ["c1", "c1", "c2", "c3", "c4"]
        return pd.DataFrame(
            {
                "context": contexts,
                "question": [f"q{i}" for i in range(5)],
                "answer_prefix": [""] * 5,
                "answer": [[f"q{i}"] for i in range(5)],
                "task": ["t"] * 5,
                "max_new_tokens": [8] * 5,
            }
        )

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        right: List[bool] = [
            answer[0] in predicted
            for answer, predicted in zip(
                results_df["answer"], results_df["predicted_answer"]
            )
        ]
        return {"task_scores": {"t": sum(right) / len(right)}}


def _request(context: str = "a context", question: str = "a question") -> Request:
    return Request(context=context, questions=[question], answer_prefix="")


@pytest.mark.unit
class TestLoggedAdapter:
    def test_an_answer_is_logged_and_then_reused(self, tmp_path: Path) -> None:
        path = str(tmp_path / "run" / "answers.jsonl")
        inner = CountingAdapter()
        adapter = LoggedAdapter(inner, path, RUN)
        first = adapter.process_request(_request(), {"max_new_tokens": 8}, {})
        again = adapter.process_request(_request(), {"max_new_tokens": 8}, {})
        assert first.responses == again.responses == ["answer to a question"]
        assert inner.requests == 1
        assert (adapter.answered, adapter.reused, len(adapter)) == (1, 1, 1)

        lines = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert lines[0] == {"run": RUN}
        assert lines[1]["responses"] == ["answer to a question"]
        assert lines[1]["context_hash"] == context_hash("a context")

    def test_a_second_process_reuses_the_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "answers.jsonl")
        LoggedAdapter(CountingAdapter(), path, RUN).process_request(_request(), {}, {})
        inner = CountingAdapter()
        adapter = LoggedAdapter(inner, path, RUN)
        assert len(adapter) == 1
        response = adapter.process_request(_request(), {}, {})
        assert response.responses == ["answer to a question"]
        assert inner.requests == 0

    @pytest.mark.parametrize(
        "other",
        [
            (_request(context="another context"), {}, {}),
            (_request(question="another question"), {}, {}),
            (_request(), {"max_new_tokens": 9}, {}),
            (_request(), {}, {"max_context_length": 100}),
            (Request("a context", ["a question"], "Answer:"), {}, {}),
        ],
    )
    def test_what_changes_the_answer_changes_the_key(self, other: Any) -> None:
        assert request_key(*other) != request_key(_request(), {}, {})

    def test_the_file_of_another_run_is_refused(self, tmp_path: Path) -> None:
        path = str(tmp_path / "answers.jsonl")
        LoggedAdapter(CountingAdapter(), path, RUN)
        with pytest.raises(ValueError, match="another run"):
            LoggedAdapter(CountingAdapter(), path, dict(RUN, attention="sparse"))

    def test_a_file_that_is_not_an_answer_log_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "answers.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="run description"):
            LoggedAdapter(CountingAdapter(), str(path), RUN)

    def test_a_line_cut_short_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "answers.jsonl"
        adapter = LoggedAdapter(CountingAdapter(), str(path), RUN)
        adapter.process_request(_request(question="one"), {}, {})
        whole: bytes = path.read_bytes()
        path.write_bytes(whole + b'{"key": "abc", "respon')

        inner = CountingAdapter()
        adapter = LoggedAdapter(inner, str(path), RUN)
        assert len(adapter) == 1
        assert path.read_bytes() == whole
        adapter.process_request(_request(question="two"), {}, {})
        lines = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert [line.get("responses") for line in lines[1:]] == [
            ["answer to one"],
            ["answer to two"],
        ]

    def test_other_attributes_are_the_adapters(self, tmp_path: Path) -> None:
        adapter = LoggedAdapter(CountingAdapter(), str(tmp_path / "a.jsonl"), RUN)
        assert adapter.model_name == "a-model"
        with pytest.raises(AttributeError):
            adapter.no_such_attribute
        assert copy.copy(adapter).model_name == "a-model"

    def test_answers_with_any_characters_survive_the_file(self, tmp_path: Path) -> None:
        """Model output can hold line breaks, quotes and half of a surrogate pair."""
        text: str = 'line one\nline "two" \u00e9 \U0001f600 \ud83d end'

        class Odd(CountingAdapter):
            def process_request(self, request: Any, *args: Any) -> RequestResponse:
                return RequestResponse(responses=[text])

        path = str(tmp_path / "answers.jsonl")
        LoggedAdapter(Odd(), path, RUN).process_request(_request(), {}, {})
        again = LoggedAdapter(CountingAdapter(), path, RUN)
        assert again.process_request(_request(), {}, {}).responses == [text]

    def test_an_interrupted_benchmark_run_is_continued(self, tmp_path: Path) -> None:
        """The second run answers only what the first did not, with the same result."""
        whole_dir, cut_dir = str(tmp_path / "whole"), str(tmp_path / "cut")
        whole_adapter = CountingAdapter()
        whole = FiveQuestions().run_benchmark(whole_adapter, whole_dir, {}, {})
        assert whole == {"task_scores": {"t": 1.0}}
        assert whole_adapter.requests == 4  # one request per context

        path = f"{cut_dir}/answers.jsonl"
        with pytest.raises(RuntimeError, match="interrupted"):
            FiveQuestions().run_benchmark(
                LoggedAdapter(CountingAdapter(fail_at=2), path, RUN), cut_dir, {}, {}
            )
        assert not (tmp_path / "cut" / "metrics.json").exists()

        inner = CountingAdapter()
        adapter = LoggedAdapter(inner, path, RUN)
        continued = FiveQuestions().run_benchmark(adapter, cut_dir, {}, {})
        assert (adapter.reused, inner.requests) == (2, 2)
        assert continued == whole
        assert (tmp_path / "cut" / "raw_results.csv").read_text() == (
            tmp_path / "whole" / "raw_results.csv"
        ).read_text()


class Stopwatch:
    """A recorder: counts its requests and returns the count of the one it measured."""

    def __init__(self) -> None:
        self.started: int = 0
        self.stopped: int = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> Dict[str, Any]:
        self.stopped += 1
        return {"request": self.stopped, "pair": (1, 2)}


class TestRecorder:
    def test_what_it_measures_is_kept_with_the_answer(self, tmp_path: Path) -> None:
        path = str(tmp_path / "answers.jsonl")
        watch = Stopwatch()
        adapter = LoggedAdapter(CountingAdapter(), path, RUN, watch)
        adapter.process_request(_request("c1"), {}, {})
        adapter.process_request(_request("c2"), {}, {})
        adapter.process_request(_request("c1"), {}, {})  # from the log: not measured
        assert (watch.started, watch.stopped) == (2, 2)
        # As a JSON round trip leaves it, the same as when it is read back.
        assert adapter.measured == {
            context_hash("c1"): [{"request": 1, "pair": [1, 2]}],
            context_hash("c2"): [{"request": 2, "pair": [1, 2]}],
        }
        lines = [json.loads(line) for line in open(path, encoding="utf-8")]
        assert [line.get("measured") for line in lines] == [
            None,
            {"request": 1, "pair": [1, 2]},
            {"request": 2, "pair": [1, 2]},
        ]

    def test_a_continued_run_has_the_measurements_of_the_first(
        self, tmp_path: Path
    ) -> None:
        path = str(tmp_path / "answers.jsonl")
        first = LoggedAdapter(CountingAdapter(), path, RUN, Stopwatch())
        first.process_request(_request("c1"), {}, {})
        watch = Stopwatch()
        second = LoggedAdapter(CountingAdapter(), path, RUN, watch)
        assert second.measured == first.measured
        second.process_request(_request("c1"), {}, {})
        second.process_request(_request("c2"), {}, {})
        assert watch.started == 1
        assert second.measured == {
            context_hash("c1"): [{"request": 1, "pair": [1, 2]}],
            context_hash("c2"): [{"request": 1, "pair": [1, 2]}],
        }

    def test_two_requests_on_one_context_are_both_kept(self, tmp_path: Path) -> None:
        adapter = LoggedAdapter(
            CountingAdapter(), str(tmp_path / "answers.jsonl"), RUN, Stopwatch()
        )
        adapter.process_request(_request("c1", "q1"), {}, {})
        adapter.process_request(_request("c1", "q2"), {}, {})
        assert [m["request"] for m in adapter.measured[context_hash("c1")]] == [1, 2]

    def test_it_is_stopped_when_the_request_fails(self, tmp_path: Path) -> None:
        path = str(tmp_path / "answers.jsonl")
        watch = Stopwatch()
        adapter = LoggedAdapter(CountingAdapter(fail_at=0), path, RUN, watch)
        with pytest.raises(RuntimeError, match="interrupted"):
            adapter.process_request(_request(), {}, {})
        assert (watch.started, watch.stopped) == (1, 1)
        # Nothing of the failed request is logged or kept.
        assert adapter.measured == {} and len(adapter) == 0
        assert len(open(path, encoding="utf-8").readlines()) == 1

    def test_without_one_nothing_is_measured(self, tmp_path: Path) -> None:
        path = str(tmp_path / "answers.jsonl")
        adapter = LoggedAdapter(CountingAdapter(), path, RUN)
        adapter.process_request(_request(), {}, {})
        assert adapter.measured == {}
        assert "measured" not in json.loads(open(path).readlines()[1])


class DrawingAdapter(CountingAdapter):
    """Answers with a number from torch's generator, as a sampling masker draws them."""

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
    ) -> RequestResponse:
        self.requests += 1
        return RequestResponse(responses=[str(torch.rand(1).item())])


class TestSeed:
    def test_a_continued_run_draws_what_the_whole_run_drew(
        self, tmp_path: Path
    ) -> None:
        requests: List[Request] = [_request(f"c{i}") for i in range(4)]
        whole = LoggedAdapter(DrawingAdapter(), str(tmp_path / "whole"), RUN, seed=7)
        expected = [whole.process_request(r, {}, {}).responses for r in requests]
        assert len({answer[0] for answer in expected}) == 4  # a draw of its own each

        path = str(tmp_path / "cut")
        LoggedAdapter(DrawingAdapter(), path, RUN, seed=7).process_request(
            requests[0], {}, {}
        )
        torch.rand(5)  # whatever else the process drew before it went on
        inner = DrawingAdapter()
        continued = LoggedAdapter(inner, path, RUN, seed=7)
        answers = [continued.process_request(r, {}, {}).responses for r in requests]
        assert answers == expected
        assert inner.requests == 3

    def test_the_order_of_the_requests_does_not_matter(self, tmp_path: Path) -> None:
        requests: List[Request] = [_request(f"c{i}") for i in range(3)]
        forward = LoggedAdapter(DrawingAdapter(), str(tmp_path / "a"), RUN, seed=7)
        backward = LoggedAdapter(DrawingAdapter(), str(tmp_path / "b"), RUN, seed=7)
        first = [forward.process_request(r, {}, {}).responses for r in requests]
        second = [backward.process_request(r, {}, {}).responses for r in requests[::-1]]
        assert first == second[::-1]

    def test_another_seed_draws_other_numbers(self, tmp_path: Path) -> None:
        one = LoggedAdapter(DrawingAdapter(), str(tmp_path / "a"), RUN, seed=7)
        two = LoggedAdapter(DrawingAdapter(), str(tmp_path / "b"), RUN, seed=8)
        assert (
            one.process_request(_request(), {}, {}).responses
            != two.process_request(_request(), {}, {}).responses
        )
        key: str = request_key(_request(), {}, {})
        assert request_seed(7, key) != request_seed(8, key)
        assert 0 <= request_seed(2**62, "f" * 64) < 2**63

    def test_without_a_seed_the_generator_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seeded: List[int] = []
        monkeypatch.setattr(torch, "manual_seed", seeded.append)
        adapter = LoggedAdapter(DrawingAdapter(), str(tmp_path / "a"), RUN)
        adapter.process_request(_request(), {}, {})
        assert seeded == []
        LoggedAdapter(
            DrawingAdapter(), str(tmp_path / "b"), RUN, seed=3
        ).process_request(_request(), {}, {})
        assert seeded == [request_seed(3, request_key(_request(), {}, {}))]
