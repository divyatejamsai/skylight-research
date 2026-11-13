"""Plot density vs quality from sparge_decode_eval.py results and print a ranked table.

Example:
    python scripts/plot_sparge_eval.py results/qwen05b_ctx8k_b16.json
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
ABLATIONS = {"no_smooth_k", "drop_crossing", "unweighted", "topk0.02"}


def _family(name: str) -> str:
    """Group configs into one curve per method (the swept knob is h or cdf)."""
    if name.startswith("T2 "):
        return " ".join(name.split()[1:5])
    if name.startswith("T "):
        return " ".join(name.split()[1:3])  # two-stage: "b=<b> int<bits>"
    if name.startswith("K "):
        return " ".join(name.split()[1:4])  # kernel grid: "b=<b> <sharing> <budget>"
    if name.startswith("F "):
        return " ".join(name.split()[1:3])  # factorial: "<score> <budget>"
    if name.startswith("oracle_topk"):
        return "oracle top-k"
    if name.startswith("quest"):
        return "Quest"
    m = re.match(r"sparge b=\d+ sim=(\S+) cdf=\S+(.*)", name)
    if m:
        tag = m.group(2).strip()
        return f"Sparge sim={m.group(1)}" + (f" {tag}" if tag else "")
    return name


def main(path: str, out: str = "", max_density: float = 1.0) -> None:
    with open(path) as f:
        data = json.load(f)
    res: Dict[str, Dict[str, float]] = data["results"]
    args = data["args"]

    print(f"{'config':52s} {'density':>8s} {'out_err':>8s} {'dNLL':>8s}")
    for name, r in sorted(res.items(), key=lambda kv: kv[1]["density"]):
        print(
            f"{name:52s} {r['density']:8.4f} {r['output_error']:8.4f} {r['delta_nll']:+8.4f}"
        )

    curves: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for name, r in res.items():
        fam = _family(name)
        if name == "dense" or name.split()[-1] in ABLATIONS:
            continue  # one-factor ablations are reported in the table, not plotted
        curves[fam].append((r["density"], r["delta_nll"], r["output_error"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for i, (fam, pts) in enumerate(
        sorted(curves.items(), key=lambda kv: (not kv[0].startswith("oracle"), kv[0]))
    ):
        pts = sorted(p for p in pts if p[0] <= max_density)
        xs = [p[0] for p in pts]
        for ax, j in ((axes[0], 1), (axes[1], 2)):
            ax.plot(
                xs,
                [p[j] for p in pts],
                color=COLORS[i % 6],
                marker=MARKERS[i % 6],
                markersize=6,
                linewidth=2,
                label=fam,
            )
    for ax, title in (
        (axes[0], "Δ next-token NLL vs dense (lower is better)"),
        (axes[1], "Attention output relative error (lower is better)"),
    ):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Attention density (fraction of keys attended)")
        ax.grid(True, color="#e4e3df", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhline(0.0, color="#8a8984", linewidth=1)  # dNLL can be negative (noise)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{args.get('model', '')}  ctx={args['context_len']}  decode={args['decode_len']}×{args['windows']} windows  block={args['block']}",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    out = out or path.replace(".json", ".png")
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
