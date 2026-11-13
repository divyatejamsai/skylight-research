"""Decode-time quality vs density for SpargeAttn Top-CDF against Quest and oracle top-k.

For each window of a long text: dense prefill of ``context_len`` tokens, then
``decode_len`` teacher-forced decode steps (q_len == 1) with sparse attention in
every layer. Reports, per config:
  * density and attention-output relative error, as logged by ResearchAttention
    (``research_attention_density`` / ``research_attention_output_error``), averaged
    over layers, steps and windows;
  * mean next-token NLL of the decoded tokens minus the dense NLL (delta_nll).

Example:
    python scripts/sparge_decode_eval.py --model Qwen/Qwen2.5-0.5B --context-len 8192
"""

import copy
import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import fire
import torch
from datasets import load_dataset

from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    OracleTopKConfig,
    QuestTopKMaskerConfig,
    SinkMaskerConfig,
    SocketMaskerConfig,
    SpargeTopCdfMaskerConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
    MagicPigConfig,
)

SINK, LOCAL = 128, 128


def _configs(
    heavy_sizes: List[float],
    cdfs: List[float],
    block: int,
    sims: List[float],
    ablate_sim: float,
    var_scales: List[float],
) -> List[Tuple[str, Optional[ResearchAttentionConfig]]]:
    base = [SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL)]
    out: List[Tuple[str, Optional[ResearchAttentionConfig]]] = [("dense", None)]
    for h in heavy_sizes:
        out.append(
            (
                f"oracle_topk h={h}",
                ResearchAttentionConfig(base + [OracleTopKConfig(heavy_size=h)]),
            )
        )
        out.append(
            (
                f"quest p={block} h={h}",
                ResearchAttentionConfig(
                    base + [QuestTopKMaskerConfig(heavy_size=h, page_size=block)]
                ),
            )
        )
    for sim in sims:
        for cdf in cdfs:
            out.append(
                (
                    f"sparge b={block} sim={sim} cdf={cdf}",
                    ResearchAttentionConfig(
                        base
                        + [
                            SpargeTopCdfMaskerConfig(
                                block_size=block, cdf_threshold=cdf, sim_threshold=sim
                            )
                        ]
                    ),
                )
            )
    # One-factor ablations against "sparge sim=<ablate_sim>" at the two highest cdfs.
    ablations = {
        "no_smooth_k": dict(smooth_k=False),
        "drop_crossing": dict(include_crossing=False),
        "unweighted": dict(size_weighted=False),
        "topk0.02": dict(topk_fraction=0.02),
    }
    for tag, kw in ablations.items():
        for cdf in cdfs[-2:]:
            out.append(
                (
                    f"sparge b={block} sim={ablate_sim} cdf={cdf} {tag}",
                    ResearchAttentionConfig(
                        base
                        + [
                            SpargeTopCdfMaskerConfig(
                                block_size=block,
                                cdf_threshold=cdf,
                                sim_threshold=ablate_sim,
                                **kw,
                            )
                        ]
                    ),
                )
            )
    # Variance-corrected block scoring (no similarity gate) at each weight, plus the top-k floor.
    variants = [
        ("var" if lam == 1.0 else f"var{lam}", dict(variance_correction=lam), cdfs)
        for lam in var_scales
    ]
    variants.append(
        ("var+topk0.02", dict(variance_correction=1.0, topk_fraction=0.02), cdfs[-3:])
    )
    # Quest's per-channel upper bound as the block score, under the adaptive TopCdf budget.
    variants.append(("qbound", dict(score="quest_bound"), cdfs))
    variants.append(
        ("qbound+topk0.02", dict(score="quest_bound", topk_fraction=0.02), cdfs[-3:])
    )
    for tag, kw, cdf_list in variants:
        for cdf in cdf_list:
            out.append(
                (
                    f"sparge b={block} sim=-1.0 cdf={cdf} {tag}",
                    ResearchAttentionConfig(
                        base
                        + [
                            SpargeTopCdfMaskerConfig(
                                block_size=block,
                                cdf_threshold=cdf,
                                sim_threshold=-1.0,
                                **kw,
                            )
                        ]
                    ),
                )
            )
    return out


def _factorial(
    heavy_sizes: List[float], cdfs: List[float], block: int
) -> List[Tuple[str, Optional[ResearchAttentionConfig]]]:
    """Block score (mean / Quest bound / exact mass) x budget (TopCdf / fixed top-k / TopCdf + 2% floor).

    Fixed top-k is TopCdf with a vanishing threshold (keeps one block) plus the top-k floor."""
    base = [SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL)]
    out: List[Tuple[str, Optional[ResearchAttentionConfig]]] = []
    for score in ("mean", "quest_bound", "exact"):

        def cfg(**kw):
            return ResearchAttentionConfig(
                base
                + [
                    SpargeTopCdfMaskerConfig(
                        block_size=block, sim_threshold=-1.0, score=score, **kw
                    )
                ]
            )

        for cdf in cdfs:
            out.append((f"F {score} topcdf cdf={cdf}", cfg(cdf_threshold=cdf)))
        for h in heavy_sizes:
            out.append(
                (f"F {score} topk h={h}", cfg(cdf_threshold=1e-6, topk_fraction=h))
            )
        for cdf in cdfs[-3:]:
            out.append(
                (
                    f"F {score} topcdf+floor cdf={cdf}",
                    cfg(cdf_threshold=cdf, topk_fraction=0.02),
                )
            )
    return out


def _kernel_grid() -> List[Tuple[str, Optional[ResearchAttentionConfig]]]:
    """Selection rules the block-sparse decode kernel can run: Quest-bound scores, block
    size 16/32/64, per-query-head vs per-KV-head (shared) selection, TopCdf + 2% floor vs
    fixed top-k."""
    base = [SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL)]
    out: List[Tuple[str, Optional[ResearchAttentionConfig]]] = []
    for b in (16, 32, 64):
        for share in (False, True):
            tag = f"K b={b} {'shared' if share else 'perhead'}"

            def cfg(**kw):
                return ResearchAttentionConfig(
                    base
                    + [
                        SpargeTopCdfMaskerConfig(
                            block_size=b,
                            sim_threshold=-1.0,
                            score="quest_bound",
                            share_kv_groups=share,
                            **kw,
                        )
                    ]
                )

            for cdf in (0.9, 0.95, 0.98):
                out.append(
                    (
                        f"{tag} topcdf+floor cdf={cdf}",
                        cfg(cdf_threshold=cdf, topk_fraction=0.02),
                    )
                )
            for h in (0.025, 0.05, 0.1):
                out.append(
                    (f"{tag} topk h={h}", cfg(cdf_threshold=1e-6, topk_fraction=h))
                )
    return out


def _two_stage_grid() -> List[Tuple[str, Optional[ResearchAttentionConfig]]]:
    """Quest-bound prefilter (top 25% of blocks) then block mass from low-bit keys, then
    TopCdf + 2% floor; per-KV-head selection. Plus the block-64 exact-mass ceiling."""
    base = [SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL)]

    def cfg(b, **kw):
        return ResearchAttentionConfig(
            base
            + [
                SpargeTopCdfMaskerConfig(
                    block_size=b,
                    sim_threshold=-1.0,
                    score="exact",
                    share_kv_groups=True,
                    topk_fraction=0.02,
                    **kw,
                )
            ]
        )

    out: List[Tuple[str, Optional[ResearchAttentionConfig]]] = []
    for b in (64, 32):
        for bits in (16, 8, 4):
            for cdf in (0.9, 0.95):
                out.append(
                    (
                        f"T b={b} int{bits} cdf={cdf}",
                        cfg(
                            b, cdf_threshold=cdf, prefilter_fraction=0.25, key_bits=bits
                        ),
                    )
                )
    for cdf in (0.9, 0.95):
        out.append((f"T b=64 exact-noprefilter cdf={cdf}", cfg(64, cdf_threshold=cdf)))
    # Kernel-cost variants: one prefilter set per KV head, smaller prefilter, per-row cap.
    for pre in (0.25, 0.15):
        for cdf in (0.9, 0.95):
            for cap in (1.0, 0.15):
                out.append(
                    (
                        f"T2 b=64 int8 shared-pre{pre} cap{cap} cdf={cdf}",
                        cfg(
                            64,
                            cdf_threshold=cdf,
                            prefilter_fraction=pre,
                            key_bits=8,
                            prefilter_shared=True,
                            max_fraction=cap,
                        ),
                    )
                )
    out.append(
        (
            "T2 b=64 int8 union-pre0.25 cap0.15 cdf=0.95",
            cfg(
                64,
                cdf_threshold=0.95,
                prefilter_fraction=0.25,
                key_bits=8,
                max_fraction=0.15,
            ),
        )
    )
    return out


def _baselines_grid() -> List[Tuple[str, Optional[ResearchAttentionConfig]]]:
    """Other Skylight methods at comparable density, vAttention sampling stacked on the
    two-stage selector, and two attribution ablations of the two-stage selector (fixed
    top-k instead of TopCdf in stage 2; TopCdf without the floor)."""
    base = [SinkMaskerConfig(sink_size=SINK), LocalMaskerConfig(window_size=LOCAL)]

    def two_stage(**kw):
        cfg = dict(
            block_size=64,
            sim_threshold=-1.0,
            score="exact",
            share_kv_groups=True,
            prefilter_fraction=0.25,
            key_bits=8,
        )
        cfg.update(kw)
        return SpargeTopCdfMaskerConfig(**cfg)

    def sampling():
        return AdaptiveSamplingMaskerConfig(
            base_rate_sampling=0.01,
            epsilon=0.2,
            delta=0.2,
            init_offset=SINK,
            local_offset=LOCAL,
        )

    out: List[Tuple[str, Optional[ResearchAttentionConfig]]] = []
    for h in (0.05, 0.1):
        out.append(
            (
                f"B socket h={h}",
                ResearchAttentionConfig(
                    base + [SocketMaskerConfig(heavy_size=h, K=11, L=55, tau=0.5)]
                ),
            )
        )
    for lsh_l, lsh_k in ((32, 8), (64, 4)):
        out.append(
            (
                f"B magicpig L={lsh_l} K={lsh_k}",
                ResearchAttentionConfig(
                    base + [MagicPigConfig(lsh_l=lsh_l, lsh_k=lsh_k)]
                ),
            )
        )
    for h in (0.02, 0.05):
        out.append(
            (
                f"B vattention-oracle h={h}",
                ResearchAttentionConfig(
                    base + [OracleTopKConfig(heavy_size=h), sampling()]
                ),
            )
        )
    for cdf in (0.9, 0.95):
        out.append(
            (
                f"B vattention+two-stage cdf={cdf}",
                ResearchAttentionConfig(
                    base
                    + [two_stage(cdf_threshold=cdf, topk_fraction=0.02), sampling()]
                ),
            )
        )
    for frac in (0.15, 0.25, 0.35):
        out.append(
            (
                f"A two-stage fixed-topk frac={frac}",
                ResearchAttentionConfig(
                    base + [two_stage(cdf_threshold=1e-6, topk_fraction=frac)]
                ),
            )
        )
    for cdf in (0.9, 0.95):
        out.append(
            (
                f"A two-stage no-floor cdf={cdf}",
                ResearchAttentionConfig(base + [two_stage(cdf_threshold=cdf)]),
            )
        )
    return out


def _decode_nll(
    adapter: ModelAdapterHF,
    cache,
    tokens: torch.Tensor,
    first_logits: torch.Tensor,
    sparse: bool,
) -> List[float]:
    """Teacher-forced decode of ``tokens`` one at a time; returns per-token NLL."""
    nll: List[float] = []
    logits = first_logits
    meta: Dict = {}
    for i in range(tokens.shape[1]):
        nll.append(float(-torch.log_softmax(logits.float(), -1)[0, tokens[0, i]]))
        if i == tokens.shape[1] - 1:
            break
        kwargs = dict(
            input_ids=tokens[:, i : i + 1],
            past_key_values=cache,
            use_cache=True,
            sparse_meta_data=meta,
        )
        if sparse:
            with adapter.enable_sparse_mode():
                out = adapter.model(**kwargs)
        else:
            out = adapter.model(**kwargs)
        cache = out.past_key_values
        logits = out.logits[:, -1]
    return nll


def main(
    model: str = "Qwen/Qwen2.5-0.5B",
    context_len: int = 8192,
    decode_len: int = 64,
    windows: int = 4,
    block: int = 16,
    heavy_sizes: Tuple[float, ...] = (0.02, 0.05, 0.1),
    cdfs: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95),
    sims: Tuple[float, ...] = (-1.0, 0.0, 0.25, 0.5),
    ablate_sim: float = 0.25,
    factorial: bool = False,
    kernel_grid: bool = False,
    two_stage: bool = False,
    baselines: bool = False,
    var_scales: Tuple[float, ...] = (1.0,),
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    dtype: str = "float32",
    filter: str = "",
    shard: int = 0,
    num_shards: int = 1,
    out: str = "sparge_decode_eval.json",
) -> None:
    """``filter``: comma-separated substrings; only matching configs run (dense always does).
    ``shard``/``num_shards``: run every num_shards-th sparse config, for one process per GPU."""
    text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    configs = _configs(
        list(heavy_sizes),
        list(cdfs),
        block,
        list(sims),
        ablate_sim,
        [float(v) for v in var_scales],
    )
    if factorial:
        configs += _factorial(list(heavy_sizes), list(cdfs), block)
    if kernel_grid:
        configs += _kernel_grid()
    if two_stage:
        configs += _two_stage_grid()
    if baselines:
        configs += _baselines_grid()
    if filter:
        # fire turns "a,b" into a tuple.
        parts = filter if isinstance(filter, (tuple, list)) else str(filter).split(",")
        keys = [str(k).strip() for k in parts if str(k).strip()]
        configs = [
            (n, c) for n, c in configs if n == "dense" or any(k in n for k in keys)
        ]
    configs = configs[:1] + configs[1:][shard::num_shards]

    adapters: Dict[str, ModelAdapterHF] = {}
    for name, cfg in configs:
        adapters[name] = ModelAdapterHF(
            model_name=model,
            sparse_attention_config=cfg,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
            device="cpu",
        )
    hf_model = next(iter(adapters.values())).model.to(device).eval()
    tok = next(iter(adapters.values())).tokenizer
    ids = tok(text, return_tensors="pt").input_ids
    span = context_len + decode_len
    starts = [i * (ids.shape[1] - span) // max(1, windows - 1) for i in range(windows)]
    print(f"{ids.shape[1]} tokens; windows at {starts}; device={device}", flush=True)

    logger = MicroMetricLogger()
    results: Dict[str, Dict[str, float]] = {}
    dense_nll: List[float] = []
    per_cfg_nll: Dict[str, List[float]] = {n: [] for n, _ in configs}
    per_cfg_metrics: Dict[str, Dict[str, List[float]]] = {
        n: {"density": [], "error": []} for n, _ in configs
    }

    for w, s in enumerate(starts):
        ctx = ids[:, s : s + context_len].to(device)
        cont = ids[:, s + context_len : s + span].to(device)
        with torch.no_grad():
            pre = hf_model(ctx, use_cache=True, logits_to_keep=1, sparse_meta_data={})
        for name, cfg in configs:
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                if cfg is not None:
                    logger.configure_logging(
                        tmp,
                        [
                            "research_attention_density",
                            "research_attention_output_error",
                        ],
                    )
                with torch.no_grad():
                    nll = _decode_nll(
                        adapters[name],
                        copy.deepcopy(pre.past_key_values),
                        cont,
                        pre.logits[:, -1],
                        cfg is not None,
                    )
                if cfg is not None:
                    logger.flush()
                    with open(os.path.join(tmp, "micro_metrics.jsonl")) as f:
                        for line in f:
                            ev = json.loads(line)
                            key = (
                                "density"
                                if ev["metric"] == "research_attention_density"
                                else "error"
                            )
                            per_cfg_metrics[name][key].append(ev["value"])
            per_cfg_nll[name] += nll
            print(
                f"[w{w}] {name:52s} nll={sum(nll)/len(nll):.4f} ({time.time()-t0:.1f}s)",
                flush=True,
            )

    dense_nll = per_cfg_nll["dense"]
    dense_mean = sum(dense_nll) / len(dense_nll)
    for name, _ in configs:
        nll = per_cfg_nll[name]
        m = per_cfg_metrics[name]
        results[name] = {
            "nll": sum(nll) / len(nll),
            "delta_nll": sum(nll) / len(nll) - dense_mean,
            "density": (sum(m["density"]) / len(m["density"])) if m["density"] else 1.0,
            "output_error": (sum(m["error"]) / len(m["error"])) if m["error"] else 0.0,
            # Tail over (layer, step): a mean can hide heads that a sparse budget starves.
            "output_error_p90": float(torch.tensor(m["error"]).quantile(0.9))
            if m["error"]
            else 0.0,
            "output_error_max": max(m["error"]) if m["error"] else 0.0,
            "n_tokens": len(nll),
        }

    print(f"\n{'config':52s} {'density':>8s} {'out_err':>8s} {'dNLL':>8s}")
    for name, r in results.items():
        print(
            f"{name:52s} {r['density']:8.4f} {r['output_error']:8.4f} {r['delta_nll']:+8.4f}"
        )
    with open(out, "w") as f:
        json.dump(
            {
                "args": dict(
                    model=model,
                    context_len=context_len,
                    decode_len=decode_len,
                    windows=windows,
                    block=block,
                ),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
