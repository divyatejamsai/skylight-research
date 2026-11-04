#!/usr/bin/env python3
"""Table of RULER-32k accuracy per config and subset, with measured density and output error.

Usage:
    python benchmark/scripts/summarize_ruler32k.py results/ruler32k_sparge/Qwen_Qwen3-4B-Instruct-2507
"""

import glob
import json
import os
from collections import defaultdict
from typing import Dict

import fire


def main(model_dir: str, out: str = "") -> None:
    table: Dict[str, Dict[str, float]] = defaultdict(dict)
    stats: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"density": [], "error": []}
    )
    for metrics in sorted(
        glob.glob(os.path.join(model_dir, "*", "ruler32k_*", "metrics.json"))
    ):
        run_dir = os.path.dirname(metrics)
        cfg = os.path.basename(os.path.dirname(run_dir))
        subset = os.path.basename(run_dir).removeprefix("ruler32k_")
        table[cfg][subset] = json.load(open(metrics))["overall_score"]
        mm = os.path.join(run_dir, "micro_metrics.jsonl")
        if os.path.exists(mm):
            for line in open(mm):
                ev = json.loads(line)
                if ev["metric"] == "research_attention_density":
                    stats[cfg]["density"].append(ev["value"])
                elif ev["metric"] == "research_attention_output_error":
                    stats[cfg]["error"].append(ev["value"])

    subsets = sorted({s for row in table.values() for s in row})
    header = (
        f"{'config':40s} {'density':>8s} {'out_err':>8s} "
        + " ".join(f"{s[:12]:>12s}" for s in subsets)
        + f" {'mean':>6s}"
    )
    print(header)
    summary = {}
    for cfg, row in sorted(table.items()):
        d, e = stats[cfg]["density"], stats[cfg]["error"]
        dens = sum(d) / len(d) if d else 1.0
        err = sum(e) / len(e) if e else 0.0
        mean = sum(row.values()) / len(row)
        cells = " ".join(
            f"{row[s]:12.1f}" if s in row else f"{'-':>12s}" for s in subsets
        )
        print(f"{cfg:40s} {dens:8.4f} {err:8.4f} {cells} {mean:6.1f}  (n={len(row)})")
        summary[cfg] = {
            "density": dens,
            "output_error": err,
            "scores": row,
            "mean": mean,
        }
    if out:
        json.dump(summary, open(out, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(main)
