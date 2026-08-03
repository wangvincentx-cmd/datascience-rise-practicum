"""Vertical bar chart of what the hit model actually leans on.

Reads data/models/hit_predictor_importance.csv -- held-out permutation
importance from hit_predictor.py, i.e. the drop in out-of-fold ROC-AUC when one
input is shuffled INSIDE the held-out period. That is what an input is worth on
data the model never saw; a training-set importance would only report what the
model chose to lean on, not what generalises.

Only positive-drop inputs are drawn. A negative drop means shuffling the column
made held-out AUC go UP, which is noise, not evidence of a protective feature --
plotting those as short bars would invite reading a rank into random variation.

Usage (from the repo root):
    python src/make_feature_importance_figure.py [--top 6]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, RED, GRAY = "#0072B2", "#C0392B", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
SRC = Path("data/models/hit_predictor_importance.csv")
OUT = Path("figures/prelim_figures"); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})

# Plain English, two lines each so the labels sit upright under the bars.
LABEL = {
    "x_dir_epu":         "direction ×\npolicy\nuncertainty",
    "x_dir_stock_ret6":  "direction ×\nstock return\n(6m)",
    "c_topic":           "topic of\nthe forecast",
    "c_direction":       "direction\npredicted",
    "x_dir_sign":        "direction\nalone (±1)",
    "m_unrate":          "unemployment\nrate",
    "has_ip_growth_12m": "output data\navailable?",
    "has_ip_accel":      "output accel.\navailable?",
    "has_cpi_accel":     "inflation accel.\navailable?",
    "has_epu":           "uncertainty data\navailable?",
}


def draw(top_n):
    d = pd.read_csv(SRC)
    d = d[d["drop"] > 0].sort_values("drop", ascending=False).head(top_n)
    d = d.iloc[::-1]                       # smallest at left, largest at right
    names = [LABEL.get(f, f.replace("_", "\n")) for f in d["feature"]]
    vals, sds = d["drop"].values, d["sd"].values

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    x = range(len(vals))
    colors = [BLUE] * (len(vals) - 1) + [RED]
    ax.bar(x, vals, width=.66, color=colors, zorder=3)
    ax.errorbar(x, vals, yerr=sds, fmt="none", ecolor=GRAY, elinewidth=1.2,
                capsize=3, zorder=4)
    # 4 decimals throughout: at 3, the 3rd and 4th bars both print "0.005"
    # while standing at visibly different heights.
    for xi, v, s in zip(x, vals, sds):
        ax.text(xi, v + s + max(vals) * .022, f"{v:.4f}", ha="center",
                va="bottom", fontsize=12, fontweight="bold", color=INK)

    ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("drop in held-out ROC-AUC\nwhen this input is shuffled",
                  fontsize=11.5)
    ax.grid(axis="x", visible=False)
    ax.axhline(0, color="#cccccc", lw=1)

    ax.text(.985, .95, "error bars: sd over 5 shuffles",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=MUTED)

    fig.tight_layout()
    p = OUT / "figR_feature_importance.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6,
                    help="how many positive-drop inputs to draw")
    draw(ap.parse_args().top)
