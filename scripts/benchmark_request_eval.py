"""Any Skylight benchmark through its own runner and ModelAdapterHF.process_request.

The benchmark is created by name from Skylight's registry and run with
Benchmark.run_benchmark: its data, its prompts, its request loop and its metrics. This
script only chooses the attention (TwoStageBlockSparse with the --sparse overrides,
research maskers with --maskers, or none with --dense) and the options of the request
loop: generation_kwargs["graph_decode"] (the static-cache decode path) and, with
--compile_layers, "compile_layers" (compiled decoder layers in that path), and
optionally max_requests (the first rows of the dataset), skip_requests (rows left out
before those) and max_context_length (tokens of context kept).

--maskers takes the masker configs by the names of their classes, in the order they
are applied: "SinkMaskerConfig(sink_size=128);LocalMaskerConfig(window_size=128);
OracleTopKConfig(heavy_size=0.05)". Research maskers have no static-cache path: run
them with --graph_decode False. With --decode_only they mask the decode steps only
(calls with one query token) and the question is attended densely, which is what
TwoStageBlockSparse does.

--micro_metrics density (or density,error) measures research_attention_density (and
research_attention_output_error) request by request (benchmark/utils/
micro_metric_meter.py): their means over decode steps and over the other calls go to
summary.json and, per request, to results.jsonl. It needs --graph_decode False too: a
replayed decode step logs nothing.

Besides the runner's own metrics.json and config.json, the rows are saved without their
contexts in results.jsonl (one row per request, with the prediction, the reference and
context_hash, the first 16 hex digits of the SHA-256 of the context), so that two runs can
be compared prompt by prompt (scripts/benchmark_paired_stats.py); the runner's
raw_results.csv, which repeats every context, is removed unless --keep_raw.

Maskers that draw random numbers (AdaptiveSamplingMaskerConfig and the other sampling
maskers) need --seed: torch is seeded from it and from the request before every request,
so that a run gives the same answers again, also when it is continued.

A run can be interrupted. Every answer is appended to answers.jsonl in the output
directory as it arrives (benchmark/utils/answer_log.py), and the same command run again
answers only the requests that are not in it. A directory that already has summary.json
is complete and is left alone.

Example:
    python scripts/benchmark_request_eval.py --benchmark infinite_bench \
        --subsets passkey --max_requests 40 --max_context_length 250000 \
        --sparse "carry_threshold=0.2" --out results/infbench_passkey_carried
    python scripts/benchmark_request_eval.py --benchmark infinite_bench \
        --subsets passkey --max_requests 40 --max_context_length 250000 --dense \
        --out results/infbench_passkey_dense
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import fire
import numpy as np
import pandas as pd
import torch
from hf_graph_decode_bench import _config

from benchmark.benchmark_registry import (
    create_benchmark_instance,
    ensure_benchmarks_loaded,
)
from benchmark.utils.answer_log import LoggedAdapter, context_hash
from benchmark.utils.masker_spec import (
    DecodeOnlyResearchAttention,
    masker_configs_from_text,
)
from benchmark.utils.micro_metric_meter import MicroMetricMeter, mean_of
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention import ResearchAttentionConfig

MICRO_METRICS: Dict[str, str] = {
    "density": "research_attention_density",
    "error": "research_attention_output_error",
}


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
    maskers: str = "",
    decode_only: bool = False,
    micro_metrics: str = "",
    seed: Optional[int] = None,
    max_new_tokens: int = 0,
    max_requests: int = 0,
    skip_requests: int = 0,
    max_context_length: int = 0,
    graph_decode: bool = True,
    compile_layers: bool = False,
    keep_raw: bool = False,
    out: str = "results/benchmark_request",
) -> None:
    if os.path.exists(f"{out}/summary.json"):
        print(f"{out} is complete (it has summary.json), nothing to run")
        return
    if isinstance(micro_metrics, (list, tuple)):  # fire splits a,b
        micro_metrics = ",".join(micro_metrics)
    measured: List[str] = [MICRO_METRICS[m] for m in micro_metrics.split(",") if m]
    if graph_decode and (maskers or measured):
        raise ValueError("--maskers and --micro_metrics need --graph_decode False")
    if decode_only and not maskers:
        raise ValueError("--decode_only is for --maskers")
    if dense and measured:
        raise ValueError("dense attention logs no micro-metrics")
    cfg: Any = None
    if maskers:
        cfg = ResearchAttentionConfig(masker_configs=masker_configs_from_text(maskers))
    elif not dense:
        cfg = _config(sparse)
    subset_list: List[str] = (
        [s for s in subsets.split(",") if s]
        if isinstance(subsets, str)
        else list(subsets)
    )
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
    attention: Any = "dense"
    if maskers:
        # A config's parameters, without the tuner's search space it carries along.
        attention = dict(
            maskers=[
                {
                    type(m).__name__: {
                        k: v for k, v in vars(m).items() if k != "search_space"
                    }
                }
                for m in cfg.masker_configs
            ],
            decode_only=decode_only,
        )
    elif not dense:
        attention = vars(cfg)
    run: Dict[str, Any] = dict(
        benchmark=benchmark,
        subsets=subset_list,
        model=model,
        attention=attention,
        generation_kwargs=generation_kwargs,
        request_kwargs=request_kwargs,
    )
    # Options that older runs did not have are named only when used, so that the
    # answers of an older run are still continued by the command that began them.
    if skip_requests:
        run["skip_requests"] = skip_requests
    if measured:
        run["micro_metrics"] = measured
    if seed is not None:
        run["seed"] = seed
    ensure_benchmarks_loaded()
    bench = create_benchmark_instance(benchmark, subset_list or None)
    if skip_requests:
        load = bench._load_datasets

        def load_after_skipped() -> pd.DataFrame:
            return load().iloc[skip_requests:].reset_index(drop=True)

        bench._load_datasets = load_after_skipped  # type: ignore[method-assign]
    model_adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    if decode_only:  # before the first request: the adapter registers it then
        model_adapter.sparse_attention = DecodeOnlyResearchAttention(
            cfg, model_adapter.sparse_attention.maskers
        )
    adapter = LoggedAdapter(
        model_adapter,
        f"{out}/answers.jsonl",
        run,
        MicroMetricMeter(measured, f"{out}/micro") if measured else None,
        seed,
    )
    if len(adapter):
        print(
            f"continuing: {len(adapter)} requests are answered in {out}/answers.jsonl"
        )

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
            context_hash=[context_hash(context) for context in rows["context"]]
        ).drop(columns=["context"])
    means: Dict[str, Dict[str, float]] = {}
    for name in measured:
        # Per row: the means over the requests of its context (one, as a rule).
        short = [k for k, v in MICRO_METRICS.items() if v == name][0]
        for kind in ("decode", "chunk"):
            rows[f"{short}_{kind}"] = [
                mean_of(adapter.measured.get(h, []), name, kind)
                for h in rows["context_hash"]
            ]
        everything = [m for ms in adapter.measured.values() for m in ms]
        means[name] = {
            kind: mean_of(everything, name, kind) for kind in ("decode", "chunk")
        }
        means[name]["decode_calls"] = sum(m[name]["decode"][1] for m in everything)
        means[name]["chunk_calls"] = sum(m[name]["chunk"][1] for m in everything)
    with open(f"{out}/results.jsonl", "w") as f:
        for record in rows.to_dict(orient="records"):
            f.write(json.dumps({k: _plain(v) for k, v in record.items()}) + "\n")
    if not keep_raw and os.path.exists(f"{out}/raw_results.csv"):
        os.remove(f"{out}/raw_results.csv")
    summary = dict(
        run,
        rows=len(rows),
        requests_answered=adapter.answered,
        requests_from_answers_file=adapter.reused,
        minutes=round(minutes, 1),  # of this invocation
        metrics=metrics,
    )
    if measured:
        summary["micro_metric_means"] = means
    print(json.dumps(summary, indent=2, default=str))
    # summary.json marks the directory as complete: it appears whole or not at all.
    with open(f"{out}/summary.json.part", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    os.replace(f"{out}/summary.json.part", f"{out}/summary.json")


if __name__ == "__main__":
    fire.Fire(main)
