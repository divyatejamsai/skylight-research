"""Any Skylight benchmark through its own runner and ModelAdapterHF.process_request.

The benchmark is created by name from Skylight's registry and run with
Benchmark.run_benchmark: its data, its prompts, its request loop and its metrics. This
script only chooses the attention (TwoStageBlockSparse with the --sparse overrides, or
none with --dense) and the options of the request loop: generation_kwargs["graph_decode"]
(the static-cache decode path) and, with --compile_layers, "compile_layers" (compiled
decoder layers in that path), and optionally max_requests (the first rows of the
dataset) and max_context_length (tokens of context kept).

Besides the runner's own metrics.json and config.json, the rows are saved without their
contexts in results.jsonl (one row per request, with the prediction, the reference and
context_hash, the first 16 hex digits of the SHA-256 of the context), so that two runs can
be compared prompt by prompt; the runner's raw_results.csv, which repeats every context,
is removed unless --keep_raw.

Example:
    python scripts/benchmark_request_eval.py --benchmark infinite_bench \
        --subsets passkey --max_requests 40 --max_context_length 250000 \
        --sparse "carry_threshold=0.2" --out results/infbench_passkey_carried
    python scripts/benchmark_request_eval.py --benchmark infinite_bench \
        --subsets passkey --max_requests 40 --max_context_length 250000 --dense \
        --out results/infbench_passkey_dense
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List

import fire
import numpy as np
import pandas as pd
import torch
from hf_graph_decode_bench import _config

from benchmark.benchmark_registry import (
    create_benchmark_instance,
    ensure_benchmarks_loaded,
)
from sparse_attention_hub.adapters import ModelAdapterHF


def _plain(value: Any) -> Any:
    """A JSON value for a dataframe cell (arrays become lists)."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(
    benchmark: str,
    subsets: str = "",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    sparse: str = "",
    dense: bool = False,
    max_new_tokens: int = 0,
    max_requests: int = 0,
    max_context_length: int = 0,
    graph_decode: bool = True,
    compile_layers: bool = False,
    keep_raw: bool = False,
    out: str = "results/benchmark_request",
) -> None:
    cfg = None if dense else _config(sparse)
    adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    subset_list: List[str] = (
        [s for s in subsets.split(",") if s]
        if isinstance(subsets, str)
        else list(subsets)
    )
    ensure_benchmarks_loaded()
    bench = create_benchmark_instance(benchmark, subset_list or None)
    generation_kwargs: Dict[str, Any] = {"graph_decode": graph_decode}
    if compile_layers:
        generation_kwargs["compile_layers"] = True
    if max_new_tokens:  # otherwise the benchmark's own limit for each row
        generation_kwargs["max_new_tokens"] = max_new_tokens
    request_kwargs: Dict[str, Any] = {}
    if max_requests:
        request_kwargs["max_requests"] = max_requests
    if max_context_length:
        request_kwargs["max_context_length"] = max_context_length

    # The runner scores the rows and saves them with their contexts; keep the rows it
    # scored to save them without.
    scored: Dict[str, pd.DataFrame] = {}
    evaluate = bench.post_run_evaluate

    def evaluate_and_keep(results_df: pd.DataFrame) -> Dict[str, Any]:
        scored["rows"] = results_df
        return evaluate(results_df)

    bench.post_run_evaluate = evaluate_and_keep  # type: ignore[method-assign]
    t0 = time.time()
    metrics = bench.run_benchmark(adapter, out, generation_kwargs, request_kwargs)
    minutes = (time.time() - t0) / 60

    rows = scored["rows"]
    if "context" in rows.columns:
        rows = rows.assign(
            context_hash=[
                hashlib.sha256(str(context).encode("utf-8")).hexdigest()[:16]
                for context in rows["context"]
            ]
        ).drop(columns=["context"])
    with open(f"{out}/results.jsonl", "w") as f:
        for record in rows.to_dict(orient="records"):
            f.write(json.dumps({k: _plain(v) for k, v in record.items()}) + "\n")
    if not keep_raw and os.path.exists(f"{out}/raw_results.csv"):
        os.remove(f"{out}/raw_results.csv")
    summary = dict(
        benchmark=benchmark,
        subsets=subset_list,
        model=model,
        attention="dense" if dense else vars(cfg),
        generation_kwargs=generation_kwargs,
        request_kwargs=request_kwargs,
        rows=len(rows),
        minutes=round(minutes, 1),
        metrics=metrics,
    )
    print(json.dumps(summary, indent=2, default=str))
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    fire.Fire(main)
