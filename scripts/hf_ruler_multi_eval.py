"""scripts/hf_ruler_eval.py for several sparse settings from one prefill.

Per sample: one dense prefill by the HF model, moved into one StaticCache a layer at a
time (so the peak is one context of K/V and not two), then one greedy decode per mode on
that cache. A decode only writes positions >= ctx, which the next decode masks and then
overwrites, so every decode reads the same prefill. The decode step is the one
scripts/hf_graph_decode_bench.py times: the HF model's own forward, the first step eager
and the rest CUDA-graph replays. On Qwen3-4B it reserves about 39 GiB at 128k and 68 GiB
at 256k.

--sparse is the base setting ('key=value,...' as hf_graph_decode_bench._config).
--variants is a '/'-separated list of override strings applied on top of the base for one
decode each; an empty item is the base itself. Each variant's predictions are stored
under the mode name "sparse" (base) or "sparse+<overrides>". The masker reads its config
on every call and each decode builds new metadata and a new CUDA graph, so switching the
config between decodes is safe. --dense "sdpa_gqa" adds the dense decode ("" skips it).

--data is the dataset name hf_ruler_eval.py takes. --ranges "25..49+75..99" picks sample
indices per subset (inclusive). Every sample is appended to --out as one JSON line
(predictions and references), and a rerun with the same --out skips the samples already
there. Per-sample scores are printed as it runs; it prints no summary.

Example (dense, carry_threshold=0.2 and the default from one prefill):
    python scripts/hf_ruler_multi_eval.py --data ruler128k --subsets vt,fwe \
        --ranges 0..24 --sparse carry_threshold=0.2 --variants "/carry_threshold=0.0" \
        --out results/ruler128k_three.jsonl
"""

import ast
import gc
import json
import os

import fire
import torch
from hf_graph_decode_bench import _attention, _config, _opaque_attention
from hf_ruler_eval import _rows
from ruler_kernel_eval import _prompt_ids, _score
from transformers import StaticCache

from sparse_attention_hub.adapters import ModelAdapterHF


def _static_from(cfg, pkv, ctx, room):
    """A StaticCache of ctx + room tokens holding the prefill cache pkv, which is emptied
    layer by layer as it is copied, so the peak is one cache and not two."""
    cache = StaticCache(config=cfg, max_cache_len=ctx + room)
    for dst, src in zip(cache.layers, pkv.layers):
        dst.lazy_initialization(src.keys[:1, :, :0])
        dst.keys[:, :, :ctx] = src.keys[:1, :, :ctx]
        dst.values[:, :, :ctx] = src.values[:1, :, :ctx]
        src.keys = src.values = None
        torch.cuda.empty_cache()  # the freed layer is a little too small to be reused
    return cache


@torch.no_grad()
def _decode(adapter, cache, first, ctx, batch, new_tokens, mode):
    """hf_graph_decode_bench._decode at batch 1 on a StaticCache that already holds the
    prefill. Returns (first-step logits row 0, tokens, None)."""
    assert batch == 1 and cache.max_cache_len >= ctx + new_tokens + 8
    m = adapter.model
    tok = first.view(1, 1).clone()  # not a view of first: steps write it
    pos = torch.full((1,), ctx, device=m.device, dtype=torch.long)
    slots = torch.arange(cache.max_cache_len, device=m.device)
    mask = torch.zeros(1, 1, 1, slots.numel(), device=m.device, dtype=torch.bool)
    kw = dict(
        input_ids=tok,
        attention_mask=mask,
        position_ids=pos.view(1, 1),
        cache_position=pos,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    if mode == "sparse":
        kw["sparse_meta_data"] = {}
        attention = adapter.enable_sparse_mode()
    else:
        attention = _attention(m, mode)

    def step():
        mask.copy_((slots <= pos).expand_as(mask))
        logits = m(**kw).logits[:, -1].float()
        tok.copy_(logits.argmax(-1, keepdim=True))
        pos.add_(1)
        return logits

    with attention:
        _opaque_attention()  # the adapter registers its function on enable_sparse_mode
        first_logits = step()[0].clone()  # eager: compiles kernels, builds metadata
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        toks = []
        for _ in range(new_tokens):
            graph.replay()
            toks.append(tok.clone())
        torch.cuda.synchronize()
    del graph
    return first_logits, torch.cat(toks, dim=1), None


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    data: str = "ruler128k",
    subsets: str = "fwe",
    ranges: str = "0..24",
    dense: str = "sdpa_gqa",
    sparse: str = "",
    variants: str = "",
    out: str = "results/ruler_multi.jsonl",
) -> None:
    cfg = _config(str(sparse))
    base = dict(vars(cfg))
    var_list = [v.strip() for v in str(variants).split("/")]
    names = ["sparse" + ("+" + v if v else "") for v in var_list]
    assert len(set(names)) == len(names), names
    # Parse every variant now, so a typo fails before the model loads.
    over = []
    for v in var_list:
        keys = [kv.split("=")[0] for kv in filter(None, v.split(","))]
        parsed = _config(v)
        over.append({k: getattr(parsed, k) for k in keys})
    for name, o in zip(names, over):
        print(
            "mode",
            name,
            "=",
            {**{k: base[k] for k in o}, **o} if o else "base",
            flush=True,
        )
    print("base", base, flush=True)

    adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    assert adapter.sparse_attention.config is cfg  # the object the decode reads
    m, tok = adapter.model.eval(), adapter.tokenizer
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    spans = [tuple(int(x) for x in r.split("..")) for r in str(ranges).split("+")]
    n = max(hi for _, hi in spans) + 1
    done = set()
    if os.path.exists(out):
        done = {(r["subset"], r["idx"]) for r in map(json.loads, open(out))}

    def text(first, logits, toks):
        ids = [int(first), int(logits.argmax())] + toks[0].tolist()
        cut = next((i for i, t in enumerate(ids) if t in eos), None)
        return tok.decode(ids[:cut], skip_special_tokens=True)

    # fire turns "a,b" into a tuple
    for subset in subsets.split(",") if isinstance(subsets, str) else list(subsets):
        for idx, row in _rows(data, subset, n).iterrows():
            if not any(lo <= idx <= hi for lo, hi in spans) or (subset, idx) in done:
                continue
            torch.cuda.reset_peak_memory_stats()
            ids = _prompt_ids(tok, row).to(m.device)
            ctx, max_new = ids.shape[1], int(row["max_new_tokens"])
            with torch.no_grad():
                pre = m(ids, use_cache=True, logits_to_keep=1)
            first = pre.logits[:, -1].argmax(-1)
            pkv = _static_from(m.config, pre.past_key_values, ctx, max_new - 2 + 8)
            del pre
            gc.collect()
            torch.cuda.empty_cache()
            preds = {}
            if dense:
                logits, toks, _ = _decode(
                    adapter, pkv, first, ctx, 1, max_new - 2, dense
                )
                preds[dense] = text(first, logits, toks)
            for name, o in zip(names, over):
                for k, v in base.items():  # back to the base, then this variant's knobs
                    setattr(cfg, k, v)
                for k, v in o.items():
                    setattr(cfg, k, v)
                gc.collect()
                torch.cuda.empty_cache()
                logits, toks, _ = _decode(
                    adapter, pkv, first, ctx, 1, max_new - 2, "sparse"
                )
                preds[name] = text(first, logits, toks)
            for k, v in base.items():
                setattr(cfg, k, v)
            answer = row["answer"]
            ref = ast.literal_eval(answer) if isinstance(answer, str) else list(answer)
            rec = dict(subset=subset, idx=int(idx), ctx=ctx, ref=ref, preds=preds)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            sc = {k: _score(subset, [p], [ref]) for k, p in preds.items()}
            peak = "peak_gib alloc %.1f reserved %.1f" % (
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            )
            print(subset, idx, ctx, json.dumps(sc), peak, flush=True)
            del pkv
            gc.collect()
            torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
