"""Paired bootstrap CIs for RULER-32k kernel-path results (ruler_kernel_eval.py JSONs).

Re-scores every saved prediction per sample against the dataset references, merges all
given files (same model, same prompts), and reports for each mode the mean score over
subsets with a 95% CI, plus the paired difference to a baseline mode: samples are
resampled jointly across modes, so prompt difficulty cancels out.

Example:
    python scripts/ruler_bootstrap.py results/a.json results/b.json --baseline dense_sdpa
"""

import ast
import json
from collections import defaultdict

import fire
import torch
from datasets import load_dataset
from ruler_kernel_eval import _score


def main(*files, baseline: str = "dense_sdpa", reps: int = 10000, seed: int = 0):
    preds = defaultdict(dict)  # mode -> subset -> list of predictions
    model = None
    for f in files:
        d = json.load(open(f))
        model = model or d["model"]
        assert d["model"] == model, (f, d["model"], model)
        for mode, by_sub in d["predictions"].items():
            for sub, p in by_sub.items():
                preds[mode][sub] = p
    subsets = sorted(set.intersection(*(set(v) for v in preds.values())))
    per = {}  # mode -> [S, n] per-sample scores
    n = min(len(preds[m][s]) for m in preds for s in subsets)
    refs = {}
    for s in subsets:
        rows = load_dataset("xAlg-AI/att-hub-ruler-32k", s, split=s).select(range(n))
        refs[s] = [
            ast.literal_eval(r["answer"])
            if isinstance(r["answer"], str)
            else r["answer"]
            for r in rows
        ]
    for mode in preds:
        per[mode] = torch.tensor(
            [
                [_score(s, [preds[mode][s][i]], [refs[s][i]]) for i in range(n)]
                for s in subsets
            ]
        )
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (reps, len(subsets), n), generator=g)

    def boot(x):  # x [S, n] -> [reps] mean over subsets of resampled subset means
        return torch.gather(x.expand(reps, -1, -1), 2, idx).float().mean(2).mean(1)

    out = {}
    print(f"model {model}  subsets {len(subsets)}  n/subset {n}  reps {reps}")
    base = per.get(baseline)
    for mode, x in sorted(per.items(), key=lambda kv: -kv[1].float().mean()):
        b = boot(x)
        row = dict(
            mean=float(x.float().mean()),
            lo=float(b.quantile(0.025)),
            hi=float(b.quantile(0.975)),
        )
        line = f"{mode:44s} {row['mean']:5.1f} [{row['lo']:5.1f}, {row['hi']:5.1f}]"
        if base is not None and mode != baseline:
            d = boot(x - base)
            row.update(
                diff=float((x - base).float().mean()),
                diff_lo=float(d.quantile(0.025)),
                diff_hi=float(d.quantile(0.975)),
            )
            line += f"   vs {baseline}: {row['diff']:+5.1f} [{row['diff_lo']:+5.1f}, {row['diff_hi']:+5.1f}]"
        out[mode] = row
        print(line)
    return None


if __name__ == "__main__":
    fire.Fire(main)
