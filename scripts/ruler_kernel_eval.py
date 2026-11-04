"""RULER-32k accuracy through the real block-sparse decode kernels (not a simulated mask).

Per sample: the whole prompt (chat template around context + question, then the answer
prefix, as Skylight's HF adapter builds it) is prefilled densely once; then each mode
generates the answer greedily with the decode loop of e2e_decode_bench.py. Scoring is
Skylight's RULER string match. Sparse modes also report the measured density
(selected tokens / valid tokens, averaged over layers, KV heads and steps).

Modes: "dense_sdpa" or "sparse@<cdf>@<floor>@<max_frac>".

Example:
    python scripts/ruler_kernel_eval.py --n 25 --modes "dense_sdpa,sparse@0.95@0.02@0.1"
"""

import ast
import json
import math
from collections import defaultdict
from typing import Dict, List

import fire
import torch
from datasets import load_dataset
from e2e_decode_bench import DecodeLoop, layer_post, layer_pre
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.ruler32k.calculate_metrics import string_match_all, string_match_part

SUBSETS = [
    "niah_single_2",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "fwe",
    "cwe",
    "qa_1",
    "qa_2",
]


def _prompt_ids(tok, row) -> torch.Tensor:
    text = (
        tok.apply_chat_template(
            [{"role": "user", "content": row["context"] + row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        + row["answer_prefix"]
    )
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    subsets: str = ",".join(SUBSETS),
    n: int = 25,
    modes: str = "dense_sdpa,sparse@0.95@0.02@0.1",
    block_size: int = 64,
    sink_tokens: int = 128,
    local_tokens: int = 128,
    out: str = "",
) -> None:
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    subset_list = subsets.split(",") if isinstance(subsets, str) else list(subsets)
    mode_list = modes.split(",") if isinstance(modes, str) else list(modes)
    preds: Dict[str, Dict[str, List[str]]] = {mo: defaultdict(list) for mo in mode_list}
    refs: Dict[str, List[List[str]]] = defaultdict(list)
    density: Dict[str, List[float]] = defaultdict(list)
    for subset in subset_list:
        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", subset, split=subset).select(
            range(n)
        )
        for row in rows:
            ids = _prompt_ids(tok, row).cuda()
            ctx, max_new = ids.shape[1], int(row["max_new_tokens"])
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            first = pre.logits[:, -1].argmax(-1)
            refs[subset].append(
                ast.literal_eval(row["answer"])
                if isinstance(row["answer"], str)
                else list(row["answer"])
            )
            for mo in mode_list:
                name, *knobs = mo.split("@")  # name@cdf@floor@max_frac[@prefilter]
                vals = [float(x) for x in knobs] if knobs else [0.95, 0.02, 0.1]
                cdf, floor, max_frac = vals[:3]
                loop = DecodeLoop(
                    m,
                    pre.past_key_values,
                    ctx,
                    1,
                    max_new + 1,
                    block_size,
                    name,
                    cdf,
                    floor,
                    max_frac,
                    math.ceil(sink_tokens / block_size),
                    math.ceil(local_tokens / block_size),
                    (layer_pre, layer_post),
                )
                loop.track_density = name in ("sparse", "two_stage")
                if len(vals) > 3:
                    loop.prefilter = vals[3]
                loop.reset(first)
                out_ids = [int(first)]
                for _ in range(max_new - 1):
                    if out_ids[-1] == tok.eos_token_id:
                        break
                    loop.step()
                    out_ids.append(int(loop.tok[0, 0]))
                preds[mo][subset].append(tok.decode(out_ids, skip_special_tokens=True))
                if loop.density:
                    density[mo].append(torch.stack(loop.density).mean().item())
                del loop
            del pre
            torch.cuda.empty_cache()
        line = {mo: _score(subset, preds[mo][subset], refs[subset]) for mo in mode_list}
        print(subset, json.dumps(line), flush=True)

    table = {
        mo: {s: _score(s, preds[mo][s], refs[s]) for s in subset_list}
        for mo in mode_list
    }
    summary = {
        mo: dict(
            mean=sum(t.values()) / len(t),
            density=(sum(density[mo]) / len(density[mo])) if density[mo] else 1.0,
            scores=t,
        )
        for mo, t in table.items()
    }
    print(
        json.dumps(
            dict(model=model, n=n, block_size=block_size, results=summary), indent=2
        )
    )
    if out:
        json.dump(
            dict(
                model=model,
                n=n,
                block_size=block_size,
                results=summary,
                predictions={mo: dict(p) for mo, p in preds.items()},
            ),
            open(out, "w"),
            indent=2,
        )


def _score(subset: str, preds: List[str], refs: List[List[str]]) -> float:
    preds = [p.strip() for p in preds]
    fn = string_match_part if subset.startswith("qa") else string_match_all
    return fn(preds, refs[: len(preds)])


if __name__ == "__main__":
    fire.Fire(main)
