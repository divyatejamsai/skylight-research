"""Report figure: attention error vs density at 32k (Qwen3-4B) for the main selection rules.

Example:
    python scripts/plot_frontier_32k.py --out results/report/frontier_32k.png
"""

import json
import re

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# (label, results file, config-name regex); fixed hue order, never cycled.
SERIES = [
    (
        "Oracle top-k (token level)",
        "results/qwen3_4b_ctx32k_factorial.json",
        r"^oracle_topk h=",
    ),
    (
        "Quest, 16-token blocks",
        "results/qwen3_4b_ctx32k_factorial.json",
        r"^quest p=16 h=",
    ),
    (
        "SpargeAttn port (mean score, TopCdf)",
        "results/qwen3_4b_ctx32k_factorial.json",
        r"^F mean topcdf cdf=",
    ),
    (
        "Bound, 64-token blocks, fixed top-k",
        "results/qwen3_4b_kfocus.json",
        r"^K b=64 shared topk h=",
    ),
    (
        "Two-stage, 64-token blocks (ours)",
        "results/qwen3_4b_twostage.json",
        r"^T b=64 int8 cdf=",
    ),
]


def main(out: str = "results/report/frontier_32k.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, (label, path, pat) in enumerate(SERIES):
        res = json.load(open(path))["results"]
        pts = sorted(
            (v["density"], v["output_error"], v.get("output_error_p90", float("nan")))
            for n, v in res.items()
            if re.match(pat, n)
        )
        for ax, j in ((axes[0], 1), (axes[1], 2)):
            ys = [p[j] for p in pts]
            if all(y == y for y in ys):  # skip series without a p90 tail
                ax.plot(
                    [p[0] for p in pts],
                    ys,
                    color=COLORS[i],
                    marker=MARKERS[i],
                    markersize=7,
                    linewidth=2,
                    label=label,
                )
    for ax, title in (
        (axes[0], "Mean attention-output error (lower is better)"),
        (axes[1], "90th-percentile error over layers and steps"),
    ):
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Attention density (fraction of keys read)")
        ax.set_xlim(0, 0.32)
        ax.set_yscale("log")
        ax.grid(True, color="#e4e3df", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Qwen3-4B-Instruct-2507, 32k context, 64 decode steps x 4 windows, sink 128 + local 128",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    fire.Fire(main)
