# Archive: the experiment scripts behind TwoStageBlockSparse

This branch is `feat/two-stage-sparse-decode` as it stood at commit `1fc0c69`
(17 Nov 2026), plus this file. It keeps the scripts that produced the measurements
during development. The pull-request branch no longer has the 21 files listed here; it
keeps the attention, the kernels, the decode path, the tests, the documentation and the
scripts that the documentation names.

Each script says in its first lines what it measures and how it was run. Several import
helpers from `scripts/e2e_decode_bench.py`, `scripts/ruler_kernel_eval.py` and
`scripts/group_diagnosis.py`; run them from `scripts/` on this branch.

## Kernel and decode-loop timing

| Script | What it measures |
|---|---|
| `scripts/bench_block_sparse_decode.py` | Correctness and latency of the block-sparse decode kernel against dense attention |
| `scripts/profile_e2e_step.py` | Kernel-time breakdown of one CUDA-graphed decode step |
| `scripts/hf_graph_pair_bench.py` | Paired timing of two decode settings in one process, for a GPU that other jobs share |
| `scripts/hf_graph_pair_stats.py` | Reads the results of `hf_graph_pair_bench.py` |

## What the two selection stages lose, and why

| Script | Question |
|---|---|
| `scripts/prefilter_recall.py` | How much true attention mass survives each candidate prefilter (stage 1)? |
| `scripts/group_diagnosis.py` | Which part of the two-stage selection loses attention mass, per query head? |
| `scripts/tail_estimate.py` | Output error of the selection with and without an estimate of the tail |
| `scripts/certified_prune_probe.py` | How many blocks can stage 1 drop with a proof that they hold almost no attention? |
| `scripts/candidate_staleness.py` | What is lost when the candidate set or the final selection is reused across decode steps? |
| `scripts/selection_stability.py` | How well does a stale block selection serve later decode steps? |
| `scripts/hf_fidelity_probe.py` | Per-step fidelity of a sparse decode under teacher forcing |

## Accuracy runs that `scripts/benchmark_request_eval.py` replaced

| Script | What it ran |
|---|---|
| `scripts/long_niah_eval.py` | Needle retrieval at about 128k tokens through the decode kernels |
| `scripts/loft_eval.py` | `hf_ruler_multi_eval.py` on LOFT RAG at 128k |
| `scripts/loft_score.py` | Scores a LOFT RAG 128k run |
| `scripts/ruler_bootstrap.py` | Paired bootstrap intervals for `ruler_kernel_eval.py` results (now `scripts/benchmark_paired_stats.py`) |

## The SpargeAttn Top-CDF masker

The masker (`SpargeTopCdfMasker`) and its tests stay on the pull-request branch. These
are the runs that compared it with Quest and oracle top-k.

| Script | What it does |
|---|---|
| `scripts/sparge_decode_eval.py` | Decode-time quality against density for Top-CDF, Quest and oracle top-k |
| `scripts/launch_gpu_sweep.sh` | Shards `sparge_decode_eval.py` over idle GPUs |
| `scripts/plot_sparge_eval.py` | Density against quality from those results, with a ranked table |
| `scripts/plot_frontier_32k.py` | Figure: attention error against density at 32k for the main selection rules |
| `benchmark/scripts/run_ruler32k_sparge.py` | RULER-32k accuracy of Top-CDF against dense, oracle top-k and Quest |
| `benchmark/scripts/summarize_ruler32k.py` | Table of that run per config and subset, with density and output error |
