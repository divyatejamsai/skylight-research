"""LOFT RAG through Skylight's own benchmark runner and ModelAdapterHF.process_request.

Nothing here is specific to this branch but two settings: the sparse attention is
TwoStageBlockSparse (or none, with --dense), and generation_kwargs["graph_decode"]
selects the static-cache decode path of process_request. The data, the prompt, the
request loop and the metrics are those of benchmark/loft.

The runner saves raw_results.csv, metrics.json and config.json in --out. LOFT's
comparable figure is the test split (metrics["by_split"]["test"]); it is printed at the
end with the wall-clock time of the run.

Example:
    python scripts/loft_request_eval.py --subsets nq_128k \
        --sparse "carry_threshold=0.2" --out results/loft_request_nq_carried
    python scripts/loft_request_eval.py --subsets nq_128k --dense \
        --out results/loft_request_nq_dense
"""

import json
import time
from typing import List

import fire
import torch
from hf_graph_decode_bench import _config

from benchmark.loft.loft_rag import LoftRag
from sparse_attention_hub.adapters import ModelAdapterHF


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    subsets: str = "nq_128k",
    sparse: str = "",
    dense: bool = False,
    max_new_tokens: int = 4096,
    graph_decode: bool = True,
    out: str = "results/loft_request",
) -> None:
    cfg = None if dense else _config(sparse)
    adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    subset_list: List[str] = (
        subsets.split(",") if isinstance(subsets, str) else list(subsets)
    )
    t0 = time.time()
    metrics = LoftRag(subsets_to_run=subset_list).run_benchmark(
        adapter,
        out,
        generation_kwargs={
            "max_new_tokens": max_new_tokens,
            "graph_decode": graph_decode,
        },
        request_kwargs={},
    )
    minutes = (time.time() - t0) / 60
    test = metrics.get("by_split", {}).get("test", {})
    summary = dict(
        model=model,
        attention="dense" if dense else vars(cfg),
        graph_decode=graph_decode,
        max_new_tokens=max_new_tokens,
        minutes=round(minutes, 1),
        test_split=test,
    )
    print(json.dumps(summary, indent=2))
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
