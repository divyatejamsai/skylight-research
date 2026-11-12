"""Decode speed of a HF model through ModelAdapterHF: TwoStageBlockSparse vs dense attention.

Unlike scripts/e2e_decode_bench.py (its own decode loop), every step here is the HF model's
own forward on a static KV cache (HF StaticCache), with the whole step captured in a CUDA
graph, which the static-cache path of TwoStageBlockSparse allows. Modes:
  * hf_sdpa:     HF's own SDPA attention (with a mask it repeats K/V per query head);
  * sdpa_gqa:    torch SDPA with native GQA and the length mask (the dense_sdpa mode of
                 e2e_decode_bench);
  * dense_flash: the block-sparse decode kernel over every block (split-K flash-decoding);
  * sparse:      TwoStageBlockSparse as the adapter registers it (enable_sparse_mode), with
                 the given config overrides;
  * no_attention: zeros instead of attention, the step's cost without attention (a floor
                 for timing only).
cuDNN SDPA stays on: it is the fast masked SDPA backend, and the static cache's fixed
shape means it plans once (on a growing dynamic cache it re-plans every step).
With --compile every decoder layer is torch.compile'd around its attention call (as the
e2e loop compiles its layer halves); the attention functions stay eager inside the graph.

Prefill is dense (HF forward, B=1, dynamic cache) and copied to every batch slot; with
--pre_on_cpu the prefill cache waits in host memory between runs, which takes one context
of K/V off the GPU (about 19 GB at 128k tokens for Qwen3-4B) and changes no timed step. The first
decode step runs eagerly and is checked against HF's own next-token logits on the dynamic
cache; the rest are graph replays. With repeats > 1 every mode runs that many times, in
alternating order (to cancel drift), each run with a fresh cache and graph. Reports the
median ms per step and every run's, the per-run speedup against the fastest dense mode
(paired: same repeat), and greedy agreement with it.

Example:
    python scripts/hf_graph_decode_bench.py --ctx 131072 --batch 2 \
        --sparse "prefilter_fraction=0.25"
"""

import contextlib
import gc
import json
import statistics
import time
from typing import Dict, List, Optional

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AttentionInterface, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.kernels.block_sparse_decode import block_sparse_decode_fused
from sparse_attention_hub.sparse_attention import TwoStageBlockSparseConfig

BLOCK = 64


def _seq_len(kw) -> torch.Tensor:
    return (kw["position_ids"][0, -1:] + 1).to(torch.int32)


def sdpa_gqa(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """Decode-only dense attention: SDPA with native GQA over the masked static cache."""
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scaling, enable_gqa=True
    )
    return out.transpose(1, 2), None


def dense_flash(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """Decode-only dense attention: the block-sparse kernel over every live block."""
    B, Hkv, n_max = key.shape[:3]
    seq_len = _seq_len(kw)
    blocks = torch.arange(
        (n_max + BLOCK - 1) // BLOCK, device=key.device, dtype=torch.int32
    )
    count = ((seq_len + BLOCK - 1) // BLOCK).expand(B, Hkv).contiguous()
    out = block_sparse_decode_fused(
        (query[:, :, 0] * scaling).contiguous(),
        key,
        value,
        blocks.expand(B, Hkv, -1).contiguous(),
        count,
        BLOCK,
        seq_len=seq_len,
    )
    return out.unsqueeze(1), None


def no_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """Zeros instead of attention: the step's cost without attention (a timing floor;
    its tokens are meaningless)."""
    return torch.zeros_like(query).transpose(1, 2), None


AttentionInterface.register("sdpa_gqa", sdpa_gqa)
AttentionInterface.register("dense_flash", dense_flash)
AttentionInterface.register("no_attention", no_attention)


def _compile_layers(model) -> None:
    """torch.compile every decoder layer (fused norms/RoPE/SiLU/residuals), with the
    settings scripts/e2e_decode_bench.py uses for its layer halves."""
    torch._dynamo.config.cache_size_limit = 256  # layer_idx specializes per layer
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    for layer in model.model.layers:
        layer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)


def _opaque_attention() -> None:
    """Keep every registered attention function out of the compiled graphs (a graph
    break), so the compiled layers run around it as the e2e loop's layer halves do."""
    mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
    for name, fn in mapping.items():
        if not getattr(fn, "_opaque", False):
            mapping[name] = torch.compiler.disable(fn)
            mapping[name]._opaque = True


@contextlib.contextmanager
def _attention(model, name: str):
    """Switch every module's attention implementation, as ModelAdapterHF does."""
    mods = [
        mod
        for mod in model.modules()
        if hasattr(getattr(mod, "config", None), "_attn_implementation")
    ]
    old = [mod.config._attn_implementation for mod in mods]
    for mod in mods:
        mod.config._attn_implementation = name
    try:
        yield
    finally:
        for mod, impl in zip(mods, old):
            mod.config._attn_implementation = impl


def _config(overrides: str) -> TwoStageBlockSparseConfig:
    """'key=value,...' overrides of TwoStageBlockSparseConfig; a per-layer mapping is
    given as layer_prefilter_fraction=7:0.75|13:0.75."""
    cfg = TwoStageBlockSparseConfig()
    for kv in filter(None, str(overrides).split(",")):
        key, val = kv.split("=")
        old = getattr(cfg, key)
        if key == "layer_prefilter_fraction":
            val = {int(a): float(b) for a, b in (x.split(":") for x in val.split("|"))}
        elif isinstance(old, bool):
            val = val == "True"
        else:
            val = type(old)(val)
        setattr(cfg, key, val)
    return cfg


@torch.no_grad()
def _decode(adapter, pre, first, ctx, batch, new_tokens, mode: str):
    """Eager first step, then graph replays. Returns (first-step logits row 0, tokens, s)."""
    m = adapter.model
    cfg = m.config
    hkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cache = StaticCache(config=cfg, max_cache_len=ctx + new_tokens + 8)
    cache.early_initialization(batch, hkv, hd, m.dtype, m.device)
    # src may wait on the CPU (pre_on_cpu)
    for dst, src in zip(cache.layers, pre.layers):
        dst.keys[:, :, :ctx] = src.keys[:1, :, :ctx].to(dst.keys.device)
        dst.values[:, :, :ctx] = src.values[:1, :, :ctx].to(dst.values.device)
    tok = (
        first.view(1, 1).expand(batch, 1).clone()
    )  # not a view of first: steps write it
    pos = torch.full((1,), ctx, device=m.device, dtype=torch.long)
    # A 4D mask is used as given; HF's eager_mask (registered for the custom attention)
    # copies a host scalar, which CUDA-graph capture forbids. Every mode gets this one.
    slots = torch.arange(cache.max_cache_len, device=m.device)
    mask = torch.zeros(batch, 1, 1, slots.numel(), device=m.device, dtype=torch.bool)
    kw = dict(
        input_ids=tok,
        attention_mask=mask,
        position_ids=pos.view(1, 1).expand(batch, 1),
        cache_position=pos,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    if mode == "sparse":
        kw["sparse_meta_data"] = {}
        attention = adapter.enable_sparse_mode()
    elif mode == "hf_sdpa":
        attention = contextlib.nullcontext()
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(new_tokens):
            graph.replay()
            toks.append(tok.clone())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    del graph, cache
    return first_logits, torch.cat(toks, dim=1), dt


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    ctx: int = 32768,
    batch: int = 8,
    new_tokens: int = 64,
    modes: str = "hf_sdpa,sdpa_gqa,dense_flash,sparse",
    sparse: str = "",
    repeats: int = 1,
    compile: bool = False,
    out: Optional[str] = None,
    pre_on_cpu: bool = False,
) -> None:
    cfg = _config(sparse)
    adapter = ModelAdapterHF(
        model,
        cfg,
        model_kwargs=dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa"),
        device="cuda",
    )
    m = adapter.model.eval()
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    ids = adapter.tokenizer(text[: ctx * 8], return_tensors="pt").input_ids[:, :ctx]
    ids = ids.to(m.device)
    assert ids.shape[1] == ctx, ids.shape
    with torch.no_grad():
        pre = m(ids, use_cache=True, logits_to_keep=1)
        first = pre.logits[:, -1].argmax(-1)
        hf_next = (
            m(first.view(1, 1), past_key_values=pre.past_key_values, use_cache=True)
            .logits[0, -1]
            .float()
        )
    pkv = pre.past_key_values
    del pre
    if pre_on_cpu:  # one context of K/V less on the GPU; no timed step reads this cache
        for layer in pkv.layers:
            layer.keys = layer.keys.cpu()
            layer.values = layer.values.cpu()
        gc.collect()
        torch.cuda.empty_cache()
    if compile:  # after the eager prefill and reference step: decode shapes only
        _compile_layers(m)
    mode_list = modes.split(",") if isinstance(modes, str) else list(modes)
    res: Dict[str, Dict] = {}
    tokens: Dict[str, torch.Tensor] = {}
    runs: Dict[str, List[float]] = {mode: [] for mode in mode_list}
    for r in range(repeats):
        for mode in mode_list if r % 2 == 0 else mode_list[::-1]:
            logits, toks, dt = _decode(
                adapter, pkv, first, ctx, batch, new_tokens, mode
            )
            runs[mode].append(1000 * dt / new_tokens)
            if mode not in res:
                tokens[mode] = toks
                res[mode] = dict(
                    first_step_max_logit_err_vs_hf=(logits - hf_next)
                    .abs()
                    .max()
                    .item(),
                    first_step_top1_matches_hf=bool(
                        logits.argmax() == hf_next.argmax()
                    ),
                )
            print(r, mode, f"{runs[mode][-1]:.3f} ms/step", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    for mode in mode_list:
        res[mode]["step_ms"] = statistics.median(runs[mode])
        res[mode]["step_ms_runs"] = runs[mode]
        res[mode]["tokens_per_s"] = 1000 * batch / res[mode]["step_ms"]
    dense = [mode for mode in res if mode not in ("sparse", "no_attention")]
    if dense:
        best = min(dense, key=lambda mode: res[mode]["step_ms"])
        for mode in res:
            res[mode]["speedup_vs_fastest_dense_runs"] = [
                d / x for d, x in zip(runs[best], runs[mode])
            ]
            res[mode]["speedup_vs_fastest_dense"] = statistics.median(
                res[mode]["speedup_vs_fastest_dense_runs"]
            )
            res[mode]["greedy_agreement_vs_fastest_dense"] = (
                (tokens[mode] == tokens[best]).float().mean().item()
            )
    summary = dict(
        model=model,
        ctx=ctx,
        batch=batch,
        new_tokens=new_tokens,
        repeats=repeats,
        compile=compile,
        fastest_dense=best if dense else None,
        sparse_config=vars(cfg),
        gpu=torch.cuda.get_device_name(),
        results=res,
    )
    print(json.dumps(summary, indent=2))
    if out:
        json.dump(summary, open(out, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(main)
