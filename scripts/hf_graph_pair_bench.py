"""Paired timing of two decode settings in one process, for a GPU that other jobs are
using. scripts/hf_graph_decode_bench.py needs an empty GPU; this gives only the ratio of
two settings' decode times, and holds when both are slowed by the same neighbours.

Both settings (A and B) decode the same prompt with the same model on one static KV
cache. Each has its own CUDA graph of the whole decode step (as hf_graph_decode_bench
captures it, compiled layers included), its own sparse metadata and its own copy of the
cache rows it generates. The two graphs take turns in chunks of CHUNK decode steps along
a TRAJ-step generation, in ABBA order, so a chunk of one setting is always timed right
next to the same chunk of the other and both see the same neighbours. After TRAJ steps
both are put back to the state right after graph capture and the generation is repeated
(REPS times, the order flipped every time).

A setting is either "dense" (the sdpa_gqa mode of hf_graph_decode_bench: torch SDPA with
native GQA over the masked static cache) or a list of TwoStageBlockSparseConfig
overrides ("" is the default configuration). With --batch B the prefill is copied to B
cache rows, as hf_graph_decode_bench does, so every row decodes the same text.

What is put back at a repeat: the token and the position, the metadata of the blocks and
tokens the generation touches, every other state tensor (the carried scores), and the
setting's cache rows after the prompt (which are swapped in and out at every turn). The
check that this is complete: every repeat must generate exactly the tokens of an
uninterrupted run of the same setting, made first. A sparse setting does (its kernels
are bit-repeatable); a dense setting may not, because cuDNN's SDPA is not, and its step
does the same work whatever the tokens.

Run each pair in both capture orders (swap --a and --b), each in a fresh process, and
pool them with scripts/hf_graph_pair_stats.py --pool: what the first-captured and the
second-captured graph differ in then cancels.

Example (PYTHONPATH=.:scripts):
    python scripts/hf_graph_pair_bench.py --a "" --b carry_threshold=0.2 \
        --ctx 131072 --batch 1 --reps 160 --out results/pair_ab.json
"""

import gc
import json
import time
from typing import Any, Dict, List, Tuple

import fire
import hf_graph_decode_bench as hb
import numpy as np
import torch
from datasets import load_dataset
from transformers import StaticCache

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.efficient_attention.implementations.two_stage_block_sparse import (  # noqa: E501
    META_KEY,
)

BLOCK = 64
PER_BLOCK = ("kmin", "kmax", "ksum", "vsum")
PER_TOKEN = ("k8", "ks")
DENSE = "dense"


class Arm:
    """One setting: its graph, its state, and its copy of the generated cache rows."""

    def __init__(
        self,
        name: str,
        spec: str,
        adapter: ModelAdapterHF,
        cache: StaticCache,
        first: torch.Tensor,
        ctx: int,
        traj: int,
        batch: int,
        keep_logits: bool = False,
    ) -> None:
        m = adapter.model
        self.name, self.spec, self.cache, self.ctx = name, spec, cache, ctx
        self.cfg = None if spec == DENSE else hb._config(spec)
        self.rows = traj + 2  # the eager step and the captured step come first
        tok = first.view(1, 1).expand(batch, 1).clone()
        pos = torch.full((1,), ctx, device=m.device, dtype=torch.long)
        slots = torch.arange(cache.max_cache_len, device=m.device)
        mask = torch.zeros(
            batch, 1, 1, slots.numel(), device=m.device, dtype=torch.bool
        )
        meta: Dict[Any, Any] = {}
        # keep_logits (diagnosis only): every step also leaves its logits here
        self.logits = (
            torch.zeros(batch, m.config.vocab_size, device=m.device)
            if keep_logits
            else None
        )
        out = self.logits
        kw = dict(
            input_ids=tok,
            attention_mask=mask,
            position_ids=pos.view(1, 1).expand(batch, 1),
            cache_position=pos,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        if self.cfg is None:
            attention = hb._attention(m, "sdpa_gqa")
        else:
            kw["sparse_meta_data"] = meta
            attention = adapter.enable_sparse_mode()

        def step():
            mask.copy_((slots <= pos).expand_as(mask))
            logits = m(**kw).logits[:, -1].float()
            if out is not None:
                out.copy_(logits)
            tok.copy_(logits.argmax(-1, keepdim=True))
            pos.add_(1)
            return logits

        with attention:
            # after the context is entered: the adapter registers its function there
            hb._opaque_attention()
            if self.cfg is not None:
                adapter.sparse_attention.config = self.cfg
            self.first_logits = step()[0].clone()  # eager: builds the metadata
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                step()
            torch.cuda.synchronize()
        self.tok, self.pos = tok, pos
        # The graph holds raw addresses, not references. Everything the captured step
        # reads or writes must stay alive as long as the graph.
        self.keep = (mask, slots, kw, meta, step)
        self.toks: List[torch.Tensor] = []
        # The state right after capture: what a repeat starts from.
        b0 = ctx // BLOCK
        b1 = (ctx + self.rows) // BLOCK + 2
        snap: List[Tuple[torch.Tensor, tuple]] = [
            (tok, (Ellipsis,)),
            (pos, (Ellipsis,)),
        ]
        for st in meta.get(META_KEY, {}).values():
            for key, v in st.items():
                if not torch.is_tensor(v):
                    continue
                if key in PER_BLOCK:
                    snap.append((v, (slice(None), slice(None), slice(b0, b1))))
                elif key in PER_TOKEN:
                    snap.append(
                        (v, (slice(None), slice(None), slice(ctx, ctx + self.rows)))
                    )
                elif v.numel() <= 1 << 16:
                    snap.append((v, (Ellipsis,)))
                else:
                    raise RuntimeError(
                        f"state tensor {key} {tuple(v.shape)}: not handled"
                    )
        for key, c in meta.items():
            if key == META_KEY:
                continue
            if not isinstance(c, dict):
                raise RuntimeError(f"state {key}: not handled")
            for v in c.values():
                if torch.is_tensor(v):
                    snap.append((v, (Ellipsis,)))
        self.snap = [(t, sl, t[sl].clone()) for t, sl in snap]
        self.state_bytes = sum(c.numel() * c.element_size() for _, _, c in self.snap)
        self.store = [
            (
                layer.keys[:, :, ctx : ctx + self.rows].clone(),
                layer.values[:, :, ctx : ctx + self.rows].clone(),
            )
            for layer in cache.layers
        ]
        self.store0 = [(k.clone(), v.clone()) for k, v in self.store]

    def rewind(self) -> None:
        for t, sl, c in self.snap:
            t[sl].copy_(c)
        for (k, v), (k0, v0) in zip(self.store, self.store0):
            k.copy_(k0)
            v.copy_(v0)
        self.toks = []

    def load(self) -> None:
        a, b = self.ctx, self.ctx + self.rows
        for layer, (k, v) in zip(self.cache.layers, self.store):
            layer.keys[:, :, a:b].copy_(k)
            layer.values[:, :, a:b].copy_(v)

    def save(self) -> None:
        a, b = self.ctx, self.ctx + self.rows
        for layer, (k, v) in zip(self.cache.layers, self.store):
            k.copy_(layer.keys[:, :, a:b])
            v.copy_(layer.values[:, :, a:b])

    def run(self, n: int) -> float:
        """Swap my cache rows in, time n graph replays, swap them out."""
        self.load()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            self.graph.replay()
            self.toks.append(self.tok.clone())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.save()
        return dt

    def tokens(self) -> torch.Tensor:
        return torch.cat(self.toks, dim=1)  # [batch, steps]


def ratio(t: np.ndarray) -> float:
    """B over A: the median over repeats of each chunk's log ratio, averaged along the
    generation."""
    return float(np.exp(np.median(np.log(t[:, :, 1] / t[:, :, 0]), axis=0).mean()))


def first_diff_step(toks: torch.Tensor, ref: torch.Tensor) -> int:
    """First step (1-based) where any batch row differs from the reference; 0 = none."""
    ne = torch.nonzero((toks != ref).any(0)).view(-1)
    return int(ne[0]) + 1 if len(ne) else 0


def main(
    a: str = "",
    b: str = "",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    ctx: int = 131072,
    batch: int = 1,
    traj: int = 128,
    chunk: int = 16,
    reps: int = 60,
    compile: bool = True,
    out: str = "",
    keep_logits: bool = False,
    hf_check: bool = True,
) -> None:
    assert traj % chunk == 0, (traj, chunk)
    specs = [str(a), str(b)]
    adapter = ModelAdapterHF(
        model,
        hb._config(""),
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
            if hf_check
            else None
        )
        pkv = pre.past_key_values
        del pre
        # the prefill cache leaves the GPU before the static one is made
        for layer in pkv.layers:
            layer.keys = layer.keys.cpu()
            layer.values = layer.values.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        mc = m.config
        hd = getattr(mc, "head_dim", None) or mc.hidden_size // mc.num_attention_heads
        cache = StaticCache(config=mc, max_cache_len=ctx + traj + 2 + 8)
        cache.early_initialization(batch, mc.num_key_value_heads, hd, m.dtype, m.device)
        for dst, src in zip(cache.layers, pkv.layers):
            dst.keys[:, :, :ctx] = src.keys[:1, :, :ctx].to(dst.keys.device)
            dst.values[:, :, :ctx] = src.values[:1, :, :ctx].to(dst.values.device)
        del pkv
        gc.collect()
        if compile:
            hb._compile_layers(m)
        print("prefill done", time.strftime("%T"), flush=True)
        arms = [
            Arm(n, s, adapter, cache, first, ctx, traj, batch, keep_logits)
            for n, s in zip("AB", specs)
        ]
        print("graphs captured", time.strftime("%T"), flush=True)
        ref, block_s = [], [[], []]
        for i in (0, 1):  # uninterrupted reference generations
            arms[i].rewind()
            block_s[i].append(arms[i].run(traj))
            ref.append(arms[i].tokens())
        nc = traj // chunk
        t = np.zeros((reps, nc, 2))
        same = [0, 0]
        # per repeat: the first step that differs from the reference, 0 = none
        first_diff: List[List[int]] = [[], []]
        for j in range(reps):
            for arm in arms:
                arm.rewind()
            for i in range(nc):
                for k in (0, 1) if (i + j) % 2 == 0 else (1, 0):
                    t[j, i, k] = arms[k].run(chunk)
            for k in (0, 1):
                fd = first_diff_step(arms[k].tokens(), ref[k])
                first_diff[k].append(fd)
                same[k] += int(fd == 0)
            if j % 10 == 9 or j == reps - 1:
                print(
                    f"repeat {j + 1}: B/A {ratio(t[: j + 1]):.4f}  ms/step A "
                    f"{1000 * np.median(t[: j + 1, :, 0]) / chunk:.2f} B "
                    f"{1000 * np.median(t[: j + 1, :, 1]) / chunk:.2f}  "
                    f"same tokens {same}  first differing step, last repeats: "
                    f"A {first_diff[0][-10:]} B {first_diff[1][-10:]}",
                    time.strftime("%T"),
                    flush=True,
                )
        for i in (1, 0):
            arms[i].rewind()
            block_s[i].append(arms[i].run(traj))
            same[i] += int(torch.equal(arms[i].tokens(), ref[i]))
    summary = dict(
        model=model,
        ctx=ctx,
        batch=batch,
        traj=traj,
        chunk=chunk,
        reps=reps,
        compile=compile,
        a=specs[0],
        b=specs[1],
        config_a=vars(arms[0].cfg) if arms[0].cfg is not None else DENSE,
        config_b=vars(arms[1].cfg) if arms[1].cfg is not None else DENSE,
        gpu=torch.cuda.get_device_name(),
        ratio_b_over_a=ratio(t),
        chunk_s=t.tolist(),
        block_s=block_s,
        repeats_with_reference_tokens=same,
        first_differing_step=first_diff,
        repeats_checked=reps + 1,
        a_b_same_tokens=bool(torch.equal(ref[0], ref[1])),
        batch_rows_same_tokens=[bool((r == r[:1]).all()) for r in ref],
        first_step_max_logit_err_vs_hf=[
            (arm.first_logits - hf_next).abs().max().item() if hf_check else None
            for arm in arms
        ],
        first_step_top1_matches_hf=[
            bool(arm.first_logits.argmax() == hf_next.argmax()) if hf_check else None
            for arm in arms
        ],
        state_bytes=[arm.state_bytes for arm in arms],
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "chunk_s"}, indent=2))
    if out:
        json.dump(summary, open(out, "w"))


if __name__ == "__main__":
    fire.Fire(main)
