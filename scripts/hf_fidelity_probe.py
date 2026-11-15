"""Per-step fidelity of a sparse decode under teacher forcing.

Task scores compare whole answers, and a free-running comparison of two decodes ends at
their first different token. This script measures the step itself: given exactly the
same history, how often is the sparse decode's top token not the dense one's, and how
far apart are the two next-token distributions?

Per prompt: one dense prefill, moved into a static cache. The reference is a dense greedy
decode of --steps tokens that records its tokens and its logits at every step. Then, on
the same cache, each arm decodes while it is fed the reference's tokens, and records its
own logits: "dense" again (what a repeat of the reference differs by, the floor of the
comparison), "sparse" (the --sparse setting) and one arm per --variants entry. For every
step and arm the output has whether the arm's top token is the reference's and
KL(reference || arm) in nats, and for every step the reference's own top probability.

Prompts: --prompts is a "+"-separated list of
  ruler128k:SUBSET:LO..HI        rows of RULER-131072 (the loader and the prompt of
                                 hf_ruler_eval.py)
  infbench:TASK:LO..HI:task      rows of an InfiniteBench task (Skylight's dataset), with
                                 the task's own question
  infbench:TASK:LO..HI:continue  the same contexts with a request to continue the text
InfiniteBench contexts are cut to --max_context tokens before the question is added.
n_valid is the number of steps before the reference ends its answer; later steps are
written too and are not part of the answer.

Example:
    python scripts/hf_fidelity_probe.py --sparse carry_threshold=0.2 \
        --variants "/carry_threshold=0.0" --out results/fidelity.jsonl \
        --prompts "ruler128k:qa_1:200..205+infbench:longbook_sum_eng:0..5:task"
"""

import gc
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import fire
import torch
from datasets import load_dataset
from hf_graph_decode_bench import _attention, _config
from hf_ruler_eval import _rows
from hf_ruler_multi_eval import _static_from
from ruler_kernel_eval import _prompt_ids

from benchmark.infinite_bench.infinite_bench import InfiniteBench
from sparse_attention_hub.adapters import ModelAdapterHF, static_decode

CONTINUE = "\n\nContinue the text above in the same style."
BOOK_INSTRUCTION = "Summarize the book below:"


def _infbench_ids(
    adapter: ModelAdapterHF, row: Any, kind: str, max_context: int
) -> torch.Tensor:
    """The prompt process_request would build: the templated context, cut to max_context
    tokens, then the question and the rest of the template."""
    tok = adapter.tokenizer
    context = str(row["context"])
    if kind == "continue":
        context = context.split(BOOK_INSTRUCTION, 1)[-1]  # the text alone
        question, prefix = CONTINUE, ""
    else:
        question, prefix = str(row["question"]), str(row["answer_prefix"])
    context, questions = adapter._preprocess_context_and_questions(
        context, [question], prefix
    )
    context_ids = tok.encode(
        context, return_tensors="pt", add_special_tokens=tok.chat_template is None
    )[:, :max_context]
    question_ids = tok.encode(
        questions[0], return_tensors="pt", add_special_tokens=False
    )
    return torch.cat([context_ids, question_ids], dim=1)


@torch.no_grad()
def _decode(
    adapter: ModelAdapterHF,
    cache: Any,
    first: torch.Tensor,
    ctx: int,
    steps: int,
    mode: str,
    forced: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """steps decode steps on the static cache, the first eager and the rest replays.
    Step t is fed forced[t - 1] (first at t = 0), or its own top token when forced is
    None. Returns the top token of every step [steps] and its logits [steps, vocab]."""
    m = adapter.model
    if mode == "sparse":
        attention, meta = adapter.enable_sparse_mode(), {}
    else:
        attention, meta = _attention(m, mode), None
    out = static_decode.decode_on_static_cache(
        m,
        cache,
        first,
        ctx,
        steps,
        attention=attention,
        sparse_meta_data=meta,
        forced_tokens=forced,
        record_logits=True,
        opaque_attention=True,
    )
    assert out.used_graph and out.steps == steps
    return out.tokens[0], out.logits


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    prompts: str = "",
    steps: int = 128,
    max_context: int = 120000,
    dense: str = "sdpa_gqa",
    sparse: str = "",
    variants: str = "",
    out: str = "results/fidelity.jsonl",
) -> None:
    cfg = _config(str(sparse))
    base = dict(vars(cfg))
    var_list = [v.strip() for v in str(variants).split("/")]
    names = ["sparse" + ("+" + v if v else "") for v in var_list]
    assert len(set(names)) == len(names), names
    over = []
    for v in var_list:
        keys = [kv.split("=")[0] for kv in filter(None, v.split(","))]
        parsed = _config(v)
        over.append({k: getattr(parsed, k) for k in keys})
    adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    assert adapter.sparse_attention.config is cfg  # the object the decode reads
    m, tok = adapter.model.eval(), adapter.tokenizer
    eos = set(static_decode.end_token_ids(m, tok))
    done = set()
    if os.path.exists(out):
        done = {(r["source"], r["idx"]) for r in map(json.loads, open(out))}

    items: List[Tuple[str, int, str, Any]] = []  # source, idx, kind, row
    for item in str(prompts).split("+"):
        parts = item.split(":")
        lo, hi = (int(x) for x in parts[2].split(".."))
        if parts[0] == "ruler128k":
            rows = _rows("ruler128k", parts[1], hi + 1)
            kind = "answer"
        else:
            assert parts[0] == "infbench", item
            rows = load_dataset(
                InfiniteBench.huggingface_dataset_id, parts[1], split="test"
            ).to_pandas()
            kind = parts[3]
        source = ":".join([parts[0], parts[1], kind])
        items += [(source, idx, kind, rows.iloc[idx]) for idx in range(lo, hi + 1)]

    for source, idx, kind, row in items:
        if (source, idx) in done:
            continue
        if source.startswith("ruler128k"):
            ids = _prompt_ids(tok, row)
        else:
            ids = _infbench_ids(adapter, row, kind, max_context)
        ids = ids.to(m.device)
        ctx = ids.shape[1]
        with torch.no_grad():
            pre = m(ids, use_cache=True, logits_to_keep=1)
        first = pre.logits[:, -1].argmax(-1)
        cache = _static_from(m.config, pre.past_key_values, ctx, steps + 8)
        del pre
        gc.collect()
        torch.cuda.empty_cache()
        ref_tok, ref_logits = _decode(adapter, cache, first, ctx, steps, dense, None)
        ref_lp = torch.log_softmax(ref_logits, -1)
        del ref_logits
        # Step t is fed the token before it; an answer ends where an end token is fed.
        fed = [int(first)] + ref_tok.tolist()[:-1]
        n_valid = next((t for t, x in enumerate(fed) if x in eos), steps)
        rec: Dict[str, Any] = dict(
            source=source,
            idx=idx,
            ctx=ctx,
            n_valid=n_valid,
            ref_top_p=[round(x, 4) for x in ref_lp.exp().max(-1).values.tolist()],
            text=tok.decode(fed[: n_valid + 1], skip_special_tokens=True),
            arms={},
        )
        arms = [("dense", dense, None)] + [
            (name, "sparse", o) for name, o in zip(names, over)
        ]
        line = [source, str(idx), f"ctx {ctx}", f"valid {n_valid}"]
        for name, mode, o in arms:
            for k, v in base.items():  # back to the base, then this variant's knobs
                setattr(cfg, k, v)
            for k, v in (o or {}).items():
                setattr(cfg, k, v)
            gc.collect()
            torch.cuda.empty_cache()
            arm_tok, arm_logits = _decode(
                adapter, cache, first, ctx, steps, mode, ref_tok
            )
            arm_lp = torch.log_softmax(arm_logits, -1)
            kl = (ref_lp.exp() * (ref_lp - arm_lp)).sum(-1)
            agree = arm_tok == ref_tok
            rec["arms"][name] = dict(
                agree=agree.int().tolist(), kl=[round(x, 5) for x in kl.tolist()]
            )
            n = max(n_valid, 1)
            line.append(
                f"{name}: agree {agree[:n].float().mean().item():.3f}"
                f" kl {kl[:n].mean().item():.4f}"
            )
            del arm_logits, arm_lp
        for k, v in base.items():
            setattr(cfg, k, v)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("  ".join(line), flush=True)
        del cache, ref_lp
        gc.collect()
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
