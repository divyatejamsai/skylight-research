#!/usr/bin/env python3
"""RULER-32k accuracy of SpargeAttn Top-CDF against dense, oracle top-k and Quest.

Usage:
    python benchmark/scripts/run_ruler32k_sparge.py --gpus 0,2,3 --max-requests 25
"""

import os
import sys

import fire
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from benchmark.executor import BenchmarkExecutor  # noqa: E402
from benchmark.executor_config import AdapterConfig, BenchmarkConfig  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    LocalMaskerConfig,
    OracleTopKConfig,
    QuestTopKMaskerConfig,
    SinkMaskerConfig,
    SpargeTopCdfMaskerConfig,
)

SUBSETS = ["niah_single_2", "niah_multikey_2", "niah_multivalue", "vt", "fwe", "qa_1"]


def _base():
    return [SinkMaskerConfig(sink_size=128), LocalMaskerConfig(window_size=128)]


def sparse_configs(heavy: float, cdfs, sim: float, block: int):
    out = [
        ("dense", None),
        (
            f"oracle_topk_h{heavy}",
            ResearchAttentionConfig(_base() + [OracleTopKConfig(heavy_size=heavy)]),
        ),
        (
            f"quest_p{block}_h{heavy}",
            ResearchAttentionConfig(
                _base() + [QuestTopKMaskerConfig(heavy_size=heavy, page_size=block)]
            ),
        ),
    ]
    for cdf in cdfs:
        out.append(
            (
                f"sparge_b{block}_sim{sim}_cdf{cdf}",
                ResearchAttentionConfig(
                    _base()
                    + [
                        SpargeTopCdfMaskerConfig(
                            block_size=block, cdf_threshold=cdf, sim_threshold=sim
                        )
                    ]
                ),
            )
        )
    # SpargeAttention2 hybrid (top-p ∪ top-k), the best variant at 8k.
    out.append(
        (
            f"sparge_b{block}_sim{sim}_cdf{cdfs[0]}_topk0.02",
            ResearchAttentionConfig(
                _base()
                + [
                    SpargeTopCdfMaskerConfig(
                        block_size=block,
                        cdf_threshold=cdfs[0],
                        sim_threshold=sim,
                        topk_fraction=0.02,
                    )
                ]
            ),
        )
    )
    return out


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    gpus: str = "0",
    max_requests: int = 25,
    heavy: float = 0.05,
    cdfs=(0.9, 0.95),
    sim: float = -1.0,
    block: int = 16,
    subsets: str = ",".join(SUBSETS),
    result_dir: str = "./results/ruler32k_sparge",
) -> None:
    # fire turns "0,1" into a tuple and "0" into an int.
    gpu_ids = [
        int(g)
        for g in (gpus if isinstance(gpus, (tuple, list)) else str(gpus).split(","))
    ]
    executor = BenchmarkExecutor(
        gpu_ids=gpu_ids,
        max_concurrent_runs=len(gpu_ids),
        base_result_dir=result_dir,
    )
    results = executor.run_benchmark_matrix(
        model_names=[model],
        sparse_attention_configs=sparse_configs(heavy, list(cdfs), sim, block),
        benchmark_configs=[
            BenchmarkConfig(benchmark_name="ruler32k", subsets=subsets.split(","))
        ],
        adapter_config=AdapterConfig(
            adapter_name="huggingface",
            model_kwargs={"torch_dtype": torch.bfloat16},
            tokenizer_kwargs={"padding_side": "left"},
        ),
        generation_kwargs={"max_new_tokens": 64, "do_sample": False},
        request_kwargs={"max_context_length": 32768, "max_requests": max_requests},
    )
    print(results.execution_summary)


if __name__ == "__main__":
    fire.Fire(main)
