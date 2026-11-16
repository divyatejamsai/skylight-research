"""Answers of a benchmark run kept on disk as they arrive, so a run can be continued.

Benchmark.run_benchmark answers every request, scores them and only then saves anything:
a run that stops at request 400 of 500 leaves nothing behind. LoggedAdapter wraps the
adapter of such a run. Every answered request is appended to a JSON-lines file, and a
request that is already in the file is answered from it without calling the model, so
the same command run again continues where the first one stopped.

    adapter = LoggedAdapter(ModelAdapterHF(...), "results/run/answers.jsonl", run)
    benchmark.run_benchmark(adapter, "results/run", generation_kwargs, request_kwargs)

``run`` describes what produced the answers (model, attention, options). It is written
as the first line of the file, and a file written for a different ``run`` is refused.
Only use it with deterministic decoding: a reused answer is the answer of the first run.

A ``recorder`` measures something while a request is answered (for example
benchmark/utils/micro_metric_meter.py). Its start() and stop() are called around every
request the adapter answers, and what stop() returns is written with the answer, so a
continued run has the measurements of all its requests, each measured once.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Union

from sparse_attention_hub.adapters.base import Request, RequestResponse

Responses = Union[str, List[str]]


def context_hash(context: str) -> str:
    """The first 16 hex digits of the SHA-256 of a context."""
    return hashlib.sha256(str(context).encode("utf-8")).hexdigest()[:16]


def request_key(
    request: Request,
    generation_kwargs: Dict[str, Any],
    request_kwargs: Dict[str, Any],
) -> str:
    """One key per request and per set of options that can change its answer."""
    parts: List[Any] = [
        hashlib.sha256(str(request.context).encode("utf-8")).hexdigest(),
        request.questions,
        request.answer_prefix,
        generation_kwargs,
        request_kwargs,
    ]
    text: str = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    """What a value is after a JSON round trip (tuples become lists, and so on)."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


class LoggedAdapter:
    """An adapter whose answers are written to a file and reused when asked again.

    Everything except process_request is the wrapped adapter's.

    measured: what the recorder returned for each answered request (a JSON value),
    by context_hash, one entry per request with that context; it includes the
    requests read back from the file.
    """

    def __init__(
        self,
        adapter: Any,
        path: str,
        run: Dict[str, Any],
        recorder: Optional[Any] = None,
    ) -> None:
        self._adapter: Any = adapter
        self._path: str = path
        self._recorder: Optional[Any] = recorder
        self._answers: Dict[str, Responses] = {}
        self.measured: Dict[str, List[Any]] = {}
        self.reused: int = 0
        self.answered: int = 0
        run = _plain(run)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self._read(run)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._append({"run": run})

    def _read(self, run: Dict[str, Any]) -> None:
        """Load the file. A last line cut short by a crash is dropped and cut off."""
        with open(self._path, "rb") as f:
            data: bytes = f.read()
        kept: int = 0  # bytes of the file that hold complete entries
        first: bool = True
        for line in data.splitlines(keepends=True):
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("unfinished line")
                entry: Dict[str, Any] = json.loads(line)
            except ValueError:
                break
            if first:
                if entry.get("run") != run:
                    raise ValueError(
                        f"{self._path} holds the answers of another run: "
                        f"{entry.get('run')} (this run: {run})"
                    )
                first = False
            else:
                self._answers[entry["key"]] = entry["responses"]
                if "measured" in entry:
                    self.measured.setdefault(entry["context_hash"], []).append(
                        entry["measured"]
                    )
            kept += len(line)
        if first:
            raise ValueError(f"{self._path} does not start with a run description")
        if kept < len(data):
            with open(self._path, "r+b") as f:
                f.truncate(kept)

    def _append(self, entry: Dict[str, Any]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __len__(self) -> int:
        return len(self._answers)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # its own fields, before __init__ has set them
            raise AttributeError(name)
        return getattr(self._adapter, name)

    def process_request(
        self,
        request: Request,
        generation_kwargs: Dict[str, Any],
        request_kwargs: Dict[str, Any],
        **kwargs: Any,
    ) -> RequestResponse:
        """The logged answer of this request, or the adapter's, which is then logged."""
        key: str = request_key(request, generation_kwargs, request_kwargs)
        if key in self._answers:
            self.reused += 1
            return RequestResponse(responses=self._answers[key])
        response: RequestResponse
        entry: Dict[str, Any] = {
            "key": key,
            "context_hash": context_hash(request.context),
        }
        if self._recorder is None:
            response = self._adapter.process_request(
                request, generation_kwargs, request_kwargs, **kwargs
            )
            entry["responses"] = response.responses
        else:
            self._recorder.start()
            try:
                response = self._adapter.process_request(
                    request, generation_kwargs, request_kwargs, **kwargs
                )
            finally:
                measured: Any = self._recorder.stop()
            entry["responses"] = response.responses
            entry["measured"] = _plain(measured)
            self.measured.setdefault(entry["context_hash"], []).append(
                entry["measured"]
            )
        self._append(entry)
        self._answers[key] = response.responses
        self.answered += 1
        return response
