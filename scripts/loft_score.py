"""loft_score.py RESULTS_DIR PREFIX N: score a LOFT RAG 128k run (PREFIX_<name>.jsonl).

Per query: subspan exact match and exact match from benchmark/loft/calculate_metrics.py,
times 100, for dense, the carried set and the default. Then the paired bootstrap of
scripts/ruler_bootstrap.py (10,000 resamples within dataset, seed 0, 95% percentile
interval) on the mean over the five datasets, and the registered endpoints.
It refuses to score unless every dataset has exactly indices 0..N-1.
"""

import json
import sys

import pandas as pd
import torch

from benchmark.loft.calculate_metrics import calculate_metrics, extract_prediction

MODES = ("sdpa_gqa", "sparse", "sparse+carry_threshold=0.0")
LABEL = dict(zip(MODES, ("dense", "carried", "default")))
NAMES = ("nq", "hotpotqa", "musique", "qampari", "quest")
REPS, SEED = 10000, 0

root, prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = {}
for name in NAMES:
    rs = [json.loads(line) for line in open(f"{root}/{prefix}_{name}.jsonl")]
    assert all(r["subset"] == name and tuple(r["preds"]) == MODES for r in rs), name
    assert sorted(r["idx"] for r in rs) == list(range(n)), (name, len(rs))
    rows[name] = sorted(rs, key=lambda r: r["idx"])


def one(name, r, mode, key):
    df = pd.DataFrame(
        {
            "task": [f"{name}_128k"],
            "answers": [r["ref"]],
            "predicted_answer": [r["preds"][mode]],
            "answer_prefix": ["Final Answer: "],
        }
    )
    return 100.0 * calculate_metrics(df)[key]


def table(key):
    return {
        m: torch.tensor([[one(s, r, m, key) for r in rows[s]] for s in NAMES])
        for m in MODES
    }


sub, em = table("subspan_em"), table("em")
parsed = {
    m: [
        sum(bool(extract_prediction(r["preds"][m], "final answer")) for r in rows[s])
        for s in NAMES
    ]
    for m in MODES
}
ctx = [r["ctx"] for s in NAMES for r in rows[s]]
print(
    f"{prefix}: {len(NAMES)} datasets x {n} queries, prompt tokens {min(ctx)}..{max(ctx)}"
)
print("\nsubspan EM (exact match in brackets; then answers parsed out of n)")
print(f"{'dataset':10s}" + "".join(f"{LABEL[m]:>24s}" for m in MODES))
for i, s in enumerate(NAMES):
    cells = [
        f"{sub[m][i].mean():6.1f} ({em[m][i].mean():5.1f}) {parsed[m][i]:3d}/{n}"
        for m in MODES
    ]
    print(f"{s:10s}" + "".join(f"{c:>24s}" for c in cells))
print(
    f"{'mean':10s}"
    + "".join(
        f"{sub[m].mean():6.1f} ({em[m].mean():5.1f})        ".rjust(24) for m in MODES
    )
)

g = torch.Generator().manual_seed(SEED)
idx = torch.randint(0, n, (REPS, len(NAMES), n), generator=g)


def boot(x):
    return torch.gather(x.expand(REPS, -1, -1), 2, idx).float().mean(2).mean(1)


def pair(a, b):
    d = sub[a] - sub[b]
    bs = boot(d)
    lo, hi = float(bs.quantile(0.025)), float(bs.quantile(0.975))
    better, worse = int((d > 0).sum()), int((d < 0).sum())
    print(
        f"{LABEL[a]:8s} - {LABEL[b]:8s} {float(d.mean()):+6.2f} [{lo:+5.1f}, {hi:+5.1f}]"
        f"   queries better {better}, worse {worse}"
    )
    return float(d.mean()), lo, hi


print(f"\npaired differences in subspan EM, {REPS} resamples within dataset")
d1, lo1, hi1 = pair(MODES[1], MODES[0])
d2, lo2, hi2 = pair(MODES[1], MODES[2])
pair(MODES[2], MODES[0])
print("\nendpoints as registered")
ok1 = hi1 >= 0 and d1 >= -1.0
print(
    "1. carried - dense: upper end >= 0 and point >= -1.0:",
    "MET" if ok1 else "NOT MET",
    "(interval wholly above 0)" if lo1 > 0 else "",
)
print(
    "2. carried - default: interval above 0:",
    "yes" if lo2 > 0 else "no",
)
