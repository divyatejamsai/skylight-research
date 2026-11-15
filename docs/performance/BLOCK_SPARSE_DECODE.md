# Decode-time block-sparse attention (`TwoStageBlockSparse`)

An efficient attention for the decode phase of long prompts. Dense attention reads the
whole key-value cache at every step and in every layer; at 128k tokens that read is the
cost of the step. `TwoStageBlockSparse` reads about 5% of it: it picks, per step, layer
and key-value head, the 64-token blocks that hold the attention mass, and attends to
those only. It is implemented as Triton kernels and runs inside the Hugging Face model.

Prefill is dense. Only single-token decode steps are sparse.

## Use

```python
import torch
from sparse_attention_hub.adapters import ModelAdapterHF, Request
from sparse_attention_hub.sparse_attention import TwoStageBlockSparseConfig

adapter = ModelAdapterHF(
    "Qwen/Qwen3-4B-Instruct-2507",
    TwoStageBlockSparseConfig(carry_threshold=0.2, min_cached_tokens=32768),
    model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
    device="cuda",
)
response = adapter.process_request(
    Request(context=long_document, questions="...", answer_prefix=""),
    generation_kwargs={
        "max_new_tokens": 256,
        "graph_decode": True,
        "compile_layers": True,
    },
    request_kwargs={},
)
```

Three settings matter for speed.

- **`graph_decode: True`** in `generation_kwargs`. The default loop of
  `process_request` launches every kernel of every layer from Python at each token,
  which costs more than the attention this method saves, and it prefills the context
  again for every question. With `graph_decode` the context is prefilled once per
  request and moved into a preallocated cache (`StaticCache`); each question is added
  to it and answered there, with the decode step captured in a CUDA graph and replayed.
  It is the same greedy decoding. It also works without sparse attention
  (`sparse_attention_config=None`), where the dense step runs as SDPA with native
  grouped-query attention.
- **`compile_layers: True`** in `generation_kwargs`, with `graph_decode`. The decode
  steps run through `torch.compile`'d decoder layers; the prefill, the questions and
  the default loop do not. The layers are compiled once per model and process: at the
  first request (11 to 39 s) and once more at the first request of another length (18
  to 46 s). After that, requests of any length reuse the compiled code. It pays for a
  benchmark or a long generation, not for one short request.
- **`min_cached_tokens`**. Below about 32k cached tokens the selection costs more than
  it saves; with `min_cached_tokens=32768` such sequences decode densely.

Both options go through any benchmark unchanged:

```python
from benchmark.loft.loft_rag import LoftRag

LoftRag(subsets_to_run=["nq_128k"]).run_benchmark(
    adapter, "results/loft_nq",
    generation_kwargs={
        "max_new_tokens": 4096,
        "graph_decode": True,
        "compile_layers": True,
    },
)
```

## Settings

| Field | Default | Meaning |
|---|---|---|
| `block_size` | 64 | Tokens per block. |
| `prefilter_fraction` | 0.25 | Stage 1: share of blocks kept as candidates, ranked by a cheap score. |
| `cdf_threshold` | 0.95 | Stage 2: keep the candidates that hold this share of the exact attention mass. |
| `floor_fraction`, `max_fraction` | 0.02, 0.15 | Stage 2 keeps at least this share of the candidates and at most this share of all blocks. |
| `sink_tokens`, `local_tokens` | 128, 128 | First and last tokens, always attended. |
| `carry_threshold` | 0.0 | The carried set. A block that any layer attended with at least this probability is a candidate in every layer. **Use 0.2 at 128k and above.** 0 disables it. |
| `carry_decay` | 0.9 | Decay of the carried scores per decode step. |
| `min_cached_tokens` | 0 | Decode densely below this many cached tokens (batch × length). |
| `layer_prefilter_fraction` | `None` | Stage-1 share for particular layers. |

## What was measured

Qwen3-4B-Instruct-2507 on one H200. Speed is the decode step inside the Hugging Face
model against the fastest dense decode measured (SDPA, native GQA, cuDNN), with
compiled decoder layers, in `scripts/hf_graph_decode_bench.py`.

| | 128k × 1 | 128k × 2 | 32k × 8 | 256k × 1 |
|---|---|---|---|---|
| Speedup, `carry_threshold=0.2` | 1.45× | 1.80× | 1.74× | 1.87× |
| Speedup, default | 1.47× | 1.81× | 1.75× | 1.88× |

Quality is sparse minus dense on the same prompts, with a 95% paired-bootstrap interval.

| Benchmark | Dense | Default − dense | `carry_threshold=0.2` − dense |
|---|---|---|---|
| RULER-32k, 250 prompts | 88.6 | −0.7 [−1.8, +0.2] | −0.7 [−1.8, +0.2] |
| RULER-128k, 500 prompts | 74.8 | −3.7 [−5.4, −2.0] | −0.4 [−1.8, +1.0] |
| RULER at 256k, 250 prompts | 59.7 | −1.6 [−3.6, +0.1] | +0.2 [−1.4, +1.7] |
| LOFT RAG 128k, 500 questions | 30.4 | −0.6 [−2.4, +1.2] | +0.0 [−1.4, +1.4] |

At 128k the default loses the answer on RULER's `niah_multikey_3` (thousands of
near-identical lines): stage 1 ranks the right block too low in a few layers. The
carried set repairs most of it at no measurable cost in speed. Four points remain on
that subset.

### A second model family

`microsoft/Phi-4-mini-instruct` (24 query heads on 8 key-value heads), with the settings
above unchanged and nothing tuned on it. The speed (1.30× to 1.55×) and the quality bar
were predicted and written down before the run.

| | Qwen3-4B | Phi-4-mini |
|---|---|---|
| Speedup, one sequence of about 129k cached tokens, `carry_threshold=0.2` | 1.45× | 1.50× |
| RULER-128k, 250 prompts, dense | 75.1 | 49.8 |
| `carry_threshold=0.2` − dense | −0.2 [−1.9, +1.4] | +0.7 [−1.2, +2.6] |
| Default − dense | −4.1 [−6.6, −1.8] | −1.1 [−2.9, +0.7] |

Phi-4-mini is weaker at this length, and two of the ten subsets are at the floor in
every mode. Its decodes end at the model's end-of-turn token (`<|end|>`, named in the
model's generation config and not the tokenizer's eos); the evaluation scripts take
their end tokens from `static_decode.end_token_ids`.

### Through `process_request`

One request of 131,025 prompt tokens and 512 new tokens, `carry_threshold=0.2`
(`scripts/hf_request_bench.py`), Qwen3-4B. The `graph_decode` rows are four fresh
processes on a GPU with no other process. The `compile_layers` rows are four more,
each with three requests per mode at three context lengths: the first request, and the
third as the steady state of a process.

| | Dense | Sparse | Sparse / dense |
|---|---|---|---|
| Default loop, ms per token, lengths seen before | 23.5 | 33.3 | 0.71× |
| Default loop, ms per token, lengths new to the process | 311.7 | 34.0 | |
| Default loop, peak memory | 64.7 GiB | 69.2 GiB | |
| **`graph_decode`, ms per token** | 10.98 | 8.59 | **1.28×** (± 0.02) |
| `graph_decode`, decode start to first replay | 0.4 to 0.8 s | 0.7 s | |
| `graph_decode`, peak memory | 35.3 GiB | 35.3 GiB | |
| **`graph_decode` + `compile_layers`, ms per token**, first request | 7.44 | 5.05 | **1.47×** (± 0.01) |
| `graph_decode` + `compile_layers`, ms per token, steady state | 7.54 | 5.10 | **1.48×** |
| `graph_decode` + `compile_layers`, decode start to first replay, requests 1 / 2 / 3 | 14 to 39 / 22 to 24 / 0.4 s | 16 to 23 / 42 to 46 / 0.3 s | |

- With `graph_decode` alone the sparse step is 1.28× the dense one, and with
  `compile_layers` 1.47× (1.50× on Phi-4-mini at 129k; 1.48× and 1.47× in the steady
  state). Sparse attention saves about 2.4 ms per token in both; compiled layers shorten
  the rest of the step. Both figures were measured against a bar written down before
  the run (1.20, and within 0.03 of the timing harness).
- Against the default loop, `graph_decode` is 2.1× faster per token for dense and 3.9×
  for sparse; with `compile_layers` 3.2× and 6.6×.
- One prefill per request: LOFT nq (110 questions of one context) through
  `LoftRag.run_benchmark` takes 1.5 minutes for dense and 2.2 for sparse, against 40
  with a prefill per question, with the same scores.
- `graph_decode` asks the prefill for one position of logits; the default loop keeps all
  of them, which is the difference in peak memory.
- LOFT nq at 128k through `LoftRag.run_benchmark` with `graph_decode`
  (`scripts/loft_request_eval.py`): subspan exact match on the test split 64.0 dense and
  62.0 with `carry_threshold=0.2`. The three-decode script behind the table above gave
  63.0 and 61.0 on the same questions.
- **The default loop and cuDNN.** With PyTorch 2.11 (cuDNN attention on by default) the
  dense step of the default loop is planned again at every sequence length the process
  has not seen: 312 ms per token at 128k against 23.5 ms once the lengths have been
  seen, and 304 against 16.7 ms at 32k (16.3 ms with
  `torch.backends.cuda.enable_cudnn_sdp(False)`). A static cache has one shape per
  request, so `graph_decode` plans once.

## Where it does not pay

- **Under about 32k cached tokens**: slower than dense. Use `min_cached_tokens`.
- **Models with few key-value heads and many query heads on each.** On Qwen2.5-1.5B
  (2 key-value heads, 6 query heads on each) it is 0.92 to 0.98× dense at one 128k
  sequence and loses about 3 RULER points at 32k.
- **Short generations with `graph_decode`**: moving the cache and capturing the graph
  costs about one second per question at 128k (measured in a fresh process, which also
  pays its one-time costs there). Against the default loop with sequence lengths it has
  seen before, that is repaid after 84 new tokens for dense and 34 for sparse.
- **The default loop of `process_request`** (no CUDA graph): 33 to 34 ms per token
  against 17 ms (32k) and 23.5 ms (128k) for dense. Use `graph_decode`.

## How it works

Per layer and key-value head, at every decode step:

1. **Block scores** from statistics kept up to date as tokens are appended: an upper
   bound on the attention logit from each block's minimum and maximum key (the Quest
   bound), blended with the logit of the block's mean key.
2. **Stage 1** keeps the best `prefilter_fraction` of the blocks, by region of 512
   blocks with quotas, so the kernel runs in parallel over regions.
3. **Exact block mass** of each candidate from an 8-bit copy of the keys.
4. **Stage 2** keeps the blocks that hold `cdf_threshold` of that mass.
5. **Attention** over the kept blocks (split-K flash-decoding), and one kernel that
   folds the new token into the statistics.

The carried set is one score per block shared by the whole model: a layer raises it to
the largest attention probability any head gave the block, and stage 1 of every layer
treats a block at or above `carry_threshold` as a candidate.

The decode itself is `sparse_attention_hub/adapters/static_decode.py`:
`decode_on_static_cache` runs one eager step, records it in a CUDA graph and replays it,
for any attention. `process_request`, the timing scripts and the evaluation scripts all
call it, so what a graph capture needs (a 4D mask, the model's device and stream,
LongRoPE, cache layers that never slide, the model's own end tokens) is handled in one
place.

## Requirements and limits

- CUDA and Triton (installed with PyTorch on Linux). On CPU the attention is dense.
- Unpadded sequences. `process_request` decodes one sequence at a time.
- Models with an attention sink input (`s_aux`) are not supported.
- A head dimension of at least 16 (smaller models raise `NotImplementedError`).
- The 8-bit key copy adds 25% to the key-value cache, plus the per-block statistics.
- `graph_decode` does not support hybrid models, and needs dense attention or a sparse
  attention that sets `supports_static_cache_decode`. It refuses before the prefill.
- `compile_layers` needs `graph_decode`, the model on a GPU, PyTorch 2.6 or later
  (`torch.compiler.set_stance`) and a model with its decoder layers in
  `model.model.layers`. It also refuses before the prefill.
- Tested with transformers 4.57 and 5.23 (the static cache changed between them).
- Phi models are supported. Their LongRoPE picks its frequencies on the host at every
  call; the graphed steps keep the choice of the first step, and a decode that would
  cross the 4,096-token boundary between the two sets runs without a graph. Their
  configs name a sliding window longer than any sequence, which is treated as no
  window. A model whose window is shorter than the sequence is refused, and a model
  with dynamic RoPE decodes without a graph.

## Reproduce and test

```bash
# speed (one dense and one sparse run per fresh process)
python scripts/hf_graph_decode_bench.py --ctx 131072 --batch 1 --compile \
    --modes sdpa_gqa,sparse --sparse "carry_threshold=0.2"
# speed through process_request, and with compiled layers over three requests
python scripts/hf_request_bench.py --ctx 131072 --sparse "carry_threshold=0.2"
python scripts/hf_request_bench.py --ctx 131072 --sparse "carry_threshold=0.2" \
    --compile_layers True --rounds 3 --round_step 500
# quality: dense, carried set and default from one prefill per prompt
python scripts/hf_ruler_multi_eval.py --data ruler128k --subsets niah_multikey_3,qa_1 \
    --ranges 0..24 --sparse "carry_threshold=0.2" --variants "/carry_threshold=0.0"
# a benchmark through Skylight's runner, dense and sparse; every answer is written to
# answers.jsonl as it arrives, and the same command continues an interrupted run
python scripts/benchmark_request_eval.py --benchmark loft_rag --subsets nq_128k \
    --max_new_tokens 4096 --dense --compile_layers True --out results/loft_nq_dense
python scripts/benchmark_request_eval.py --benchmark loft_rag --subsets nq_128k \
    --max_new_tokens 4096 --sparse "carry_threshold=0.2" --compile_layers True \
    --out results/loft_nq_carried
# the two runs prompt by prompt, scored by the benchmark's own post_run_evaluate, with a
# paired-bootstrap interval on the difference
python scripts/benchmark_paired_stats.py --benchmark loft_rag --metric subspan_em \
    --scale 100 --baseline results/loft_nq_dense --other results/loft_nq_carried
# tests (the kernel tests need a GPU)
pytest tests/unit/sparse_attention/efficient_attention/test_two_stage_block_sparse.py \
    tests/unit/adapters/test_graph_decode.py
```
