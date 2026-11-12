"""RULER accuracy of the decode path that scripts/hf_graph_decode_bench.py times.

Per sample: the prompt (chat template around context + question, then the answer prefix,
as scripts/ruler_kernel_eval.py builds it) is prefilled densely once by the HF model; then
each mode generates the answer greedily with hf_graph_decode_bench's _decode: the HF
model's own forward on a static cache, the first step eager and the rest CUDA-graph
replays. "sparse" is TwoStageBlockSparse as the adapter registers it, with the given
config overrides; dense modes are those of hf_graph_decode_bench (sdpa_gqa, dense_flash).

Datasets: "ruler32k" (xAlg-AI/att-hub-ruler-32k) or "ruler128k" (the raw RULER-131072
data Skylight's ruler128k benchmark loads, split by its prepare_dataframe). Only the
parquet files of the requested subsets are downloaded. Every sample is appended to --out
as one JSON line (predictions and references), so an interrupted run keeps its samples
and a rerun with the same --out skips them; the summary at the end scores what is there.
--start skips the first samples of each subset (samples 50-74: --n 75 --start 50).

Example:
    python scripts/hf_ruler_eval.py --data ruler128k --n 25 --out results/ruler128k.jsonl
"""

import ast
import gc
import json
import os
from collections import defaultdict

import fire
import pandas as pd
import torch
from hf_graph_decode_bench import _config, _decode
from huggingface_hub import hf_hub_download
from ruler_kernel_eval import SUBSETS, _prompt_ids, _score

from benchmark.ruler128k.prepare_dataframe import prepare_dataframe
from sparse_attention_hub.adapters import ModelAdapterHF


def _rows(data: str, subset: str, n: int) -> pd.DataFrame:
    if data == "ruler32k":
        from datasets import load_dataset

        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset)
        return rows.select(range(n)).to_pandas()
    path = hf_hub_download(
        "SaylorTwift/RULER-131072-llama-3.2-tokenizer",
        f"data/{subset}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return prepare_dataframe(pd.read_parquet(path).head(n), subset, 131072)


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    data: str = "ruler128k",
    subsets: str = ",".join(SUBSETS),
    n: int = 25,
    modes: str = "sdpa_gqa,sparse",
    sparse: str = "",
    out: str = "results/hf_ruler.jsonl",
    start: int = 0,
) -> None:
    cfg = _config(sparse)
    adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    m, tok = adapter.model.eval(), adapter.tokenizer
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    mode_list = modes.split(",") if isinstance(modes, str) else list(modes)
    subset_list = subsets.split(",") if isinstance(subsets, str) else list(subsets)
    done = set()
    if os.path.exists(out):
        done = {(r["subset"], r["idx"]) for r in map(json.loads, open(out))}
    for subset in subset_list:
        for idx, row in _rows(data, subset, n).iterrows():
            if idx < start or (subset, idx) in done:
                continue
            ids = _prompt_ids(tok, row).to(m.device)
            ctx, max_new = ids.shape[1], int(row["max_new_tokens"])
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            first = pre.logits[:, -1].argmax(-1)
            pkv = pre.past_key_values
            del pre
            preds = {}
            for mode in mode_list:
                # The eager first step's token is in its logits, not in the replays'.
                logits, toks, _ = _decode(
                    adapter, pkv, first, ctx, 1, max_new - 2, mode
                )
                out_ids = [int(first), int(logits.argmax())] + toks[0].tolist()
                cut = next((i for i, t in enumerate(out_ids) if t in eos), None)
                preds[mode] = tok.decode(out_ids[:cut], skip_special_tokens=True)
                gc.collect()
                torch.cuda.empty_cache()
            answer = row["answer"]
            ref = ast.literal_eval(answer) if isinstance(answer, str) else list(answer)
            rec = dict(subset=subset, idx=int(idx), ctx=ctx, ref=ref, preds=preds)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(subset, idx, ctx, json.dumps(preds)[:200], flush=True)
            del pkv
            gc.collect()
            torch.cuda.empty_cache()

    recs = [json.loads(line) for line in open(out)]
    by_subset = defaultdict(list)
    for r in recs:
        by_subset[r["subset"]].append(r)
    table = {
        mode: {
            s: _score(s, [r["preds"][mode] for r in rs], [r["ref"] for r in rs])
            for s, rs in by_subset.items()
        }
        for mode in mode_list
    }
    summary = dict(
        model=model,
        data=data,
        n_samples=len(recs),
        sparse_config=vars(cfg),
        results={
            mode: dict(mean=sum(t.values()) / len(t), scores=t)
            for mode, t in table.items()
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
