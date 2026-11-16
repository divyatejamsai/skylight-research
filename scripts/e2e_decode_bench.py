"""End-to-end decode throughput of a HF model with block-sparse vs dense attention.

One decode loop, three attention backends; everything else (weights, projections, MLP,
static KV cache, CUDA-graph capture of the whole step) is identical:
  * dense_sdpa:   torch SDPA over the static cache with a length mask;
  * dense_flash:  the block-sparse kernel with every valid block (dense flash-decoding);
  * sparse:       Quest-bound scores + TopCdf/floor budget per KV head, then the kernel;
  * two_stage:    Quest-bound prefilter (top 25% per query head) -> block mass from an
                  int8 key copy on the candidates -> TopCdf/floor, then the kernel.
                  Suffixes: _u per-query-head union prefilter; _qm prefilter on the average
                  of the bound and block-mean-key softmaxes; _t add unselected blocks back
                  as block-mean pseudo-tokens (tail estimate); e.g. two_stage_qm_t.

Prefill is dense (HF forward, B=1) and its cache is copied to every batch slot. Before
timing it checks the dense loop against HF's own next-token logits, and reports how often
the sparse loop's greedy tokens agree with dense.

Example:
    python scripts/e2e_decode_bench.py --ctx 65536 --batch 4 --block-size 64
"""

import gc
import json
import math
import time
from typing import Dict, List, Optional

import fire
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from sparse_attention_hub.kernels.block_sparse_decode import TUNE as KTUNE
from sparse_attention_hub.kernels.block_sparse_decode import (
    append_k8,
    append_kv,
    block_min_max,
    block_sparse_decode_fused,
    block_sums,
    quantize_keys,
    select_blocks_fused,
    select_blocks_two_stage,
)


def layer_pre(layer, h, cos, sin, B: int, Hq: int, Hkv: int, D: int):
    """Norm, QKV projections, q/k norms (Qwen3; absent in Qwen2/Llama) and RoPE of one
    decoder layer for one token."""
    attn = layer.self_attn
    x = layer.input_layernorm(h)
    q = attn.q_proj(x).view(B, 1, Hq, D)
    k = attn.k_proj(x).view(B, 1, Hkv, D)
    if hasattr(attn, "q_norm"):
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    v = attn.v_proj(x).view(B, 1, Hkv, D).transpose(1, 2)
    return apply_rotary_pos_emb(q, k, cos, sin) + (v,)


def layer_post(layer, h, o, B: int, width: int):
    """Output projection, residuals and MLP of one decoder layer."""
    h = h + layer.self_attn.o_proj(o.reshape(B, 1, width))
    return h + layer.mlp(layer.post_attention_layernorm(h))


def compiled_layer_fns():
    """torch.compile'd layer halves (fused norms/RoPE/SiLU), shared by every backend."""
    torch._dynamo.config.cache_size_limit = 256  # one graph per layer object
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    opts = dict(mode="max-autotune-no-cudagraphs", dynamic=False)
    return torch.compile(layer_pre, **opts), torch.compile(layer_post, **opts)


class DecodeLoop:
    def __init__(
        self,
        model,
        prefill_cache,
        ctx: int,
        batch: int,
        max_new: int,
        block_size: int,
        mode: str,
        cdf: float,
        floor: float,
        max_frac: float,
        sink_blocks: int,
        local_blocks: int,
        layer_fns,
    ):
        # two_stage[_u][_qm][_t]: union prefilter / quest+mean prefilter / tail estimate.
        flags = set(mode.split("_")[2:]) if mode.startswith("two_stage") else set()
        assert flags <= {"u", "qm", "t"}, mode
        self.prefilter_shared = "u" not in flags
        self.prefilter_mean = "qm" in flags
        self.tail = "t" in flags
        self.tail_rho = 0.6
        self.use_sums = self.prefilter_mean or self.tail
        mode = "two_stage" if mode.startswith("two_stage") else mode
        self.m, self.mode, self.bs = model, mode, block_size
        self.cdf, self.floor, self.max_frac = cdf, floor, max_frac
        self.sink_blocks, self.local_blocks = sink_blocks, local_blocks
        cfg = model.config
        self.B, self.Hq, self.Hkv, self.D = (
            batch,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            getattr(cfg, "head_dim", None)
            or cfg.hidden_size // cfg.num_attention_heads,
        )
        self.ctx, self.nmax = ctx, ctx + max_new
        self.P = (self.nmax + block_size - 1) // block_size
        dev, dt = model.device, model.dtype
        self.k, self.v, self.kmin, self.kmax, self.k8, self.ks = [], [], [], [], [], []
        self.ksum, self.vsum = [], []
        self.prefilter = 0.25
        for layer in prefill_cache.layers:
            # Zeroed: masked positions still enter SDPA's P @ V, and 0 * NaN-garbage = NaN.
            kb = torch.zeros(batch, self.Hkv, self.nmax, self.D, device=dev, dtype=dt)
            vb = torch.zeros_like(kb)
            kb[:, :, :ctx] = layer.keys[:1]
            vb[:, :, :ctx] = layer.values[:1]
            self.k.append(kb)
            self.v.append(vb)
            if mode == "two_stage":
                k8 = torch.zeros(
                    batch, self.Hkv, self.nmax, self.D, device=dev, dtype=torch.int8
                )
                ks = torch.zeros(
                    batch, self.Hkv, self.nmax, device=dev, dtype=torch.float32
                )
                k8[:, :, :ctx], ks[:, :, :ctx] = quantize_keys(kb[:, :, :ctx])
                self.k8.append(k8)
                self.ks.append(ks)
            if mode in ("sparse", "two_stage"):
                mn = torch.empty(batch, self.Hkv, self.P, self.D, device=dev, dtype=dt)
                mx = torch.empty_like(mn)
                pmn, pmx = block_min_max(kb[:, :, :ctx], block_size)
                mn[:, :, : pmn.shape[2]] = pmn
                mx[:, :, : pmx.shape[2]] = pmx
                self.kmin.append(mn)
                self.kmax.append(mx)
            if self.use_sums:
                ksm = torch.zeros(
                    batch, self.Hkv, self.P, self.D, device=dev, dtype=torch.float32
                )
                vsm = torch.zeros_like(ksm)
                pk, pv = block_sums(kb[:, :, :ctx], block_size), block_sums(
                    vb[:, :, :ctx], block_size
                )
                ksm[:, :, : pk.shape[2]] = pk
                vsm[:, :, : pv.shape[2]] = pv
                self.ksum.append(ksm)
                self.vsum.append(vsm)
        self.tok = torch.zeros(batch, 1, device=dev, dtype=torch.long)
        self.pos = torch.zeros(1, device=dev, dtype=torch.long)
        self.seq_len = torch.zeros(1, device=dev, dtype=torch.int32)
        self.positions = torch.arange(self.nmax, device=dev)
        self.blocks_all = (
            torch.arange(self.P, device=dev, dtype=torch.int32)
            .expand(batch, self.Hkv, self.P)
            .contiguous()
        )
        self.logits = None
        self.graph = None
        self.pre, self.post = layer_fns
        self.track_density = False  # eager-only: record selected/valid token fraction
        self.density = []
        # Selection reuse: select every `reuse` steps, reuse the block list in between.
        self.reuse = 1
        self.select_now = True
        self.step_i = 0
        self.sel_idx, self.sel_cnt = {}, {}
        self.graph_reuse = None

    def reset(self, first_token: torch.Tensor) -> None:
        self.step_i = 0
        self.tok.copy_(first_token.view(-1, 1).expand(self.B, 1))
        self.pos.fill_(self.ctx)
        self.seq_len.fill_(self.ctx + 1)
        if self.mode in ("sparse", "two_stage") and self.ctx % self.bs:
            # Re-derive the partial boundary block (graph warm-up may have touched it).
            p, s = self.ctx // self.bs, (self.ctx // self.bs) * self.bs
            for i in range(len(self.k)):
                self.kmin[i][:, :, p] = self.k[i][:, :, s : self.ctx].amin(2)
                self.kmax[i][:, :, p] = self.k[i][:, :, s : self.ctx].amax(2)
                if self.use_sums:
                    self.ksum[i][:, :, p] = self.k[i][:, :, s : self.ctx].float().sum(2)
                    self.vsum[i][:, :, p] = self.v[i][:, :, s : self.ctx].float().sum(2)

    def _attend(self, i: int, q: torch.Tensor, attn) -> torch.Tensor:
        """q: [B, Hq, 1, D] after RoPE; KV for this step is already in the cache."""
        scale = attn.scaling
        if self.mode == "dense_sdpa":
            valid = (self.positions < self.seq_len).view(1, 1, 1, -1)
            return F.scaled_dot_product_attention(
                q, self.k[i], self.v[i], attn_mask=valid, scale=scale, enable_gqa=True
            ).squeeze(2)
        qs = (q.squeeze(2) * scale).contiguous()
        if self.mode == "dense_flash":
            cnt = (
                ((self.seq_len + self.bs - 1) // self.bs)
                .to(torch.int32)
                .expand(self.B, self.Hkv)
                .contiguous()
            )
            return block_sparse_decode_fused(
                qs,
                self.k[i],
                self.v[i],
                self.blocks_all,
                cnt,
                self.bs,
                seq_len=self.seq_len,
            )
        if not self.select_now and i in self.sel_idx:
            return block_sparse_decode_fused(
                qs,
                self.k[i],
                self.v[i],
                self.sel_idx[i],
                self.sel_cnt[i],
                self.bs,
                seq_len=self.seq_len,
            )
        tail = None
        if self.mode == "two_stage":
            sel = select_blocks_two_stage(
                qs,
                self.kmin[i],
                self.kmax[i],
                self.k8[i],
                self.ks[i],
                self.bs,
                self.cdf,
                self.floor,
                self.max_frac,
                self.prefilter,
                self.sink_blocks,
                self.local_blocks,
                seq_len=self.seq_len,
                prefilter_shared=self.prefilter_shared,
                ksum=self.ksum[i] if self.use_sums else None,
                prefilter_mean=self.prefilter_mean,
                return_aux=self.tail,
            )
            idx, cnt = sel[:2]
            if self.tail:
                assert self.reuse <= 1, "the tail estimate needs this step's selection"
                tail = dict(aux=sel[2], vsum=self.vsum[i], rho=self.tail_rho)
        else:
            idx, cnt = select_blocks_fused(
                qs,
                self.kmin[i],
                self.kmax[i],
                self.bs,
                self.cdf,
                self.floor,
                self.max_frac,
                self.sink_blocks,
                self.local_blocks,
                seq_len=self.seq_len,
            )
        if self.reuse > 1:
            idx, cnt = self._with_lookahead(idx, cnt)
            if i in self.sel_idx:
                self.sel_idx[i].copy_(idx)
                self.sel_cnt[i].copy_(cnt)
            else:
                self.sel_idx[i], self.sel_cnt[i] = idx.clone(), cnt.clone()
        if self.track_density:
            self.density.append(
                torch.clamp(
                    cnt.float().mean() * self.bs / self.seq_len.float(), max=1.0
                ).squeeze()
            )
        return block_sparse_decode_fused(
            qs, self.k[i], self.v[i], idx, cnt, self.bs, seq_len=self.seq_len, tail=tail
        )

    def _with_lookahead(self, idx: torch.Tensor, cnt: torch.Tensor):
        """Append the next (still empty) block so tokens decoded before the next
        selection are attended; the kernel masks positions past the current length."""
        assert (
            self.reuse <= self.bs
        ), "a reused selection may span at most one new block"
        cur_p = (self.seq_len + self.bs - 1) // self.bs
        nxt = torch.clamp(cur_p, max=self.P - 1).to(torch.int32).view(1, 1, 1)
        ext = torch.cat([idx, torch.zeros_like(idx[..., :1])], dim=-1)
        ext.scatter_(-1, cnt.long().unsqueeze(-1), nxt.expand(self.B, self.Hkv, 1))
        return ext, cnt + (cur_p < self.P).to(torch.int32)

    @torch.no_grad()
    def step(self, select: Optional[bool] = None) -> None:
        if select is None:
            select = self.step_i % self.reuse == 0
        self.select_now = self.reuse <= 1 or select
        self.step_i += 1
        mm = self.m.model
        B = self.B
        h = mm.embed_tokens(self.tok)
        cos, sin = mm.rotary_emb(h, self.pos.view(1, 1).expand(B, 1))
        for i, layer in enumerate(mm.layers):
            q, k, v = self.pre(layer, h, cos, sin, self.B, self.Hq, self.Hkv, self.D)
            if self.mode in ("sparse", "two_stage"):
                append_kv(
                    k,
                    v,
                    self.k[i],
                    self.v[i],
                    self.pos,
                    self.kmin[i],
                    self.kmax[i],
                    self.bs,
                    self.ksum[i] if self.use_sums else None,
                    self.vsum[i] if self.use_sums else None,
                )
                if self.mode == "two_stage":
                    append_k8(k, self.k8[i], self.ks[i], self.pos)
            else:
                append_kv(k, v, self.k[i], self.v[i], self.pos)
            o = self._attend(i, q, layer.self_attn)
            h = self.post(layer, h, o, self.B, self.Hq * self.D)
        self.logits = self.m.lm_head(mm.norm(h))[:, -1].float()
        self.tok.copy_(self.logits.argmax(-1, keepdim=True))
        self.pos.add_(1)
        self.seq_len.add_(1)

    def capture(self) -> None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self.step(select=True)
                if self.reuse > 1:
                    self.step(select=False)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.step(select=True)
        if self.reuse > 1:
            self.graph_reuse = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph_reuse):
                self.step(select=False)

    def run(self, n: int) -> List[torch.Tensor]:
        toks = []
        for t in range(n):
            if self.reuse <= 1 or t % self.reuse == 0:
                self.graph.replay()
            else:
                self.graph_reuse.replay()
            toks.append(self.tok.clone())
        return toks


def _profile_steps(loop: DecodeLoop, n: int = 8) -> None:
    """GPU time per decode step by kernel, from the torch profiler over graph replays."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        loop.run(n)
        torch.cuda.synchronize()
    rows = [e for e in prof.key_averages() if e.device_time_total > 0]
    tot = sum(e.device_time_total for e in rows) / n
    for e in sorted(rows, key=lambda e: -e.device_time_total)[:25]:
        t = e.device_time_total / n
        print(
            f"  {e.key[:70]:70s} {t / 1000:7.3f} ms {100 * t / tot:5.1f}%"
            f"  x{e.count // n}"
        )
    print(f"  {'kernel time per step':70s} {tot / 1000:7.3f} ms", flush=True)


def main(
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    ctx: int = 65536,
    batch: int = 4,
    new_tokens: int = 64,
    block_size: int = 64,
    cdf: float = 0.95,
    floor: float = 0.02,
    max_frac: float = 0.1,
    sink_tokens: int = 128,
    local_tokens: int = 128,
    modes: str = "dense_sdpa,dense_flash,sparse",
    compile: bool = False,
    tail_rho: float = 0.6,
    profile: bool = False,
    tune: str = "",
    out: str = "",
) -> None:
    # tune: "key=value,..." overrides of kernels.TUNE, e.g. "decode_blocks_per_split=4"
    for kv in filter(None, str(tune).split(",")):
        key, val = kv.split("=")
        assert key in KTUNE, key
        KTUNE[key] = None if val == "None" else int(val)
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    ids = tok(text[: ctx * 8], return_tensors="pt").input_ids[:, :ctx].cuda()
    assert ids.shape[1] == ctx, ids.shape
    with torch.no_grad():
        pre = m(ids, use_cache=True, logits_to_keep=1)
    first = pre.logits[:, -1].argmax(-1)
    with torch.no_grad():
        hf_next = (
            m(first.view(1, 1), past_key_values=pre.past_key_values, use_cache=True)
            .logits[0, -1]
            .float()
        )
    for layer in pre.past_key_values.layers:  # drop the token HF just appended
        layer.keys, layer.values = layer.keys[:, :, :ctx], layer.values[:, :, :ctx]

    mode_list = modes.split(",") if isinstance(modes, str) else list(modes)
    layer_fns = compiled_layer_fns() if compile else (layer_pre, layer_post)
    res: Dict[str, Dict] = {}
    tokens: Dict[str, torch.Tensor] = {}
    for mode in mode_list:
        name, *knobs = mode.split("@")  # name@cdf@floor@max_frac[@prefilter[@reuse]]
        vals = [float(x) for x in knobs] if knobs else [cdf, floor, max_frac]
        m_cdf, m_floor, m_max = vals[:3]
        loop = DecodeLoop(
            m,
            pre.past_key_values,
            ctx,
            batch,
            new_tokens + 4,
            block_size,
            name,
            m_cdf,
            m_floor,
            m_max,
            math.ceil(sink_tokens / block_size),
            math.ceil(local_tokens / block_size),
            layer_fns,
        )
        if len(vals) > 3:
            loop.prefilter = vals[3]
        if len(vals) > 4:
            loop.reuse = int(vals[4])
        loop.tail_rho = tail_rho
        loop.reset(first)
        loop.step()  # eager step: correctness against HF
        ref_err = (loop.logits[0] - hf_next).abs().max().item()
        same_top1 = bool(loop.logits[0].argmax() == hf_next.argmax())
        loop.capture()
        loop.reset(first)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        toks = loop.run(new_tokens)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tokens[mode] = torch.cat(toks, dim=1)
        res[mode] = dict(
            step_ms=1000 * dt / new_tokens,
            tokens_per_s=batch * new_tokens / dt,
            first_step_max_logit_err_vs_hf=ref_err,
            first_step_top1_matches_hf=same_top1,
        )
        print(mode, json.dumps(res[mode]), flush=True)
        if profile:
            _profile_steps(loop)
        del loop
        gc.collect()
        torch.cuda.empty_cache()
    if "dense_sdpa" in tokens:
        for mode in tokens:
            res[mode]["greedy_agreement_vs_dense"] = (
                (tokens[mode] == tokens["dense_sdpa"]).float().mean().item()
            )
            res[mode]["speedup_vs_dense_sdpa"] = (
                res["dense_sdpa"]["step_ms"] / res[mode]["step_ms"]
            )
    summary = dict(
        model=model,
        ctx=ctx,
        batch=batch,
        block_size=block_size,
        cdf=cdf,
        floor=floor,
        compile=compile,
        max_frac=max_frac,
        gpu=torch.cuda.get_device_name(),
        results=res,
    )
    print(json.dumps(summary, indent=2))
    if out:
        json.dump(summary, open(out, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(main)
