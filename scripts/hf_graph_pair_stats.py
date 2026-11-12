"""Read the results of scripts/hf_graph_pair_bench.py.

usage:
    python scripts/hf_graph_pair_stats.py F.json [F.json ...]
        one block per file: the decode time of setting B over setting A
    python scripts/hf_graph_pair_stats.py --pool AB.json BA.json
        the same pair captured in both orders, pooled: (B of the first file) over
        (A of the first file), its inverse, and the capture-order effect

The statistic: for every chunk of the generation, the median over repeats of
log(time B / time A); averaged along the generation; exponentiated. Interval: 10,000
bootstrap resamples of the repeats, percentiles 2.5 to 97.5.
"""

import json
import sys
from typing import Any, Dict, Tuple

import numpy as np


def stat(t: np.ndarray) -> float:
    """Mean over chunks of the median over repeats of log(time B / time A)."""
    return float(np.median(np.log(t[:, :, 1] / t[:, :, 0]), axis=0).mean())


def boot(t: np.ndarray, seed: int, n: int = 10000) -> np.ndarray:
    """The statistic on n resamples of the repeats."""
    rng = np.random.default_rng(seed)
    reps = t.shape[0]
    return np.array([stat(t[rng.integers(0, reps, reps)]) for _ in range(n)])


def load(path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    d = json.load(open(path))
    return d, np.array(d["chunk_s"])


def interval(bs: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def line(path: str, seed: int) -> None:
    d, t = load(path)
    s = stat(t)
    lo, hi = interval(boot(t, seed))
    total = np.log(t[:, :, 1].sum(1) / t[:, :, 0].sum(1))
    ms = 1000 * np.median(t, axis=(0, 1)) / d["chunk"]
    block = [1000 * np.median(x) / d["traj"] for x in d["block_s"]]
    same = d["repeats_with_reference_tokens"]
    err = [round(x, 3) for x in d["first_step_max_logit_err_vs_hf"]]
    print(path)
    if "batch" in d:
        print(
            f"  batch {d['batch']}   all batch rows generate the same tokens: "
            f"{d.get('batch_rows_same_tokens')}"
        )
    print(
        f"  A = {d['a']!r}   B = {d['b']!r}   ctx {d['ctx']}  {d['reps']} repeats x "
        f"{d['traj'] // d['chunk']} chunks of {d['chunk']} steps"
    )
    print(
        f"  B over A  {np.exp(s):.4f} [{np.exp(lo):.4f}, {np.exp(hi):.4f}]   "
        f"(median of whole-generation ratios {np.exp(np.median(total)):.4f}; "
        f"mean {np.exp(total.mean()):.4f})"
    )
    print(
        f"  ms/step in chunks: A {ms[0]:.2f}  B {ms[1]:.2f}   "
        f"uninterrupted: A {block[0]:.2f}  B {block[1]:.2f}"
    )
    print(
        f"  repeats that reproduce the reference tokens: "
        f"A {same[0]}/{d['repeats_checked']}  B {same[1]}/{d['repeats_checked']}   "
        f"A and B same tokens: {d['a_b_same_tokens']}"
    )
    print(
        f"  first step matches HF top-1: {d['first_step_top1_matches_hf']}  "
        f"max logit err {err}   peak reserved {d['peak_reserved_gib']:.1f} GiB"
    )


def pool(path_ab: str, path_ba: str) -> None:
    (da, ta), (db, tb) = load(path_ab), load(path_ba)
    assert da["a"] == db["b"] and da["b"] == db["a"], "not the same pair in both orders"
    sa, sb, ba, bb = stat(ta), stat(tb), boot(ta, 1), boot(tb, 2)
    s = (sa - sb) / 2
    lo, hi = interval((ba - bb) / 2)
    print(
        f"pooled over both capture orders: {da['b']!r} over {da['a']!r}  "
        f"{np.exp(s):.4f} [{np.exp(lo):.4f}, {np.exp(hi):.4f}]"
    )
    print(
        f"  inverse: {da['a']!r} over {da['b']!r}  "
        f"{np.exp(-s):.4f} [{np.exp(-hi):.4f}, {np.exp(-lo):.4f}]"
    )
    # what the two orders share instead of what they differ in
    o = (sa + sb) / 2
    lo, hi = interval((ba + bb) / 2)
    print(
        f"  capture-order effect (second captured over first): "
        f"{np.exp(o):.4f} [{np.exp(lo):.4f}, {np.exp(hi):.4f}]"
    )


if __name__ == "__main__":
    if sys.argv[1] == "--pool":
        pool(sys.argv[2], sys.argv[3])
    else:
        for i, f in enumerate(sys.argv[1:]):
            line(f, i)
