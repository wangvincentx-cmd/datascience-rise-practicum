"""
Turn the corpus and Model 1 results into figures for analysis and the writeup.

Two groups:
  EDA (from data/features.csv alone) -- what the 128k-bill corpus says about
    which structural factors move passage. These are descriptive rates, not
    model output, so they are the honest "what did we learn" panel.
  Model diagnostics (refits Model 1 via model.py) -- PR curves, calibration,
    and feature importances on the held-out Congresses.

Design rules (dataviz skill): sequential blue for magnitude, categorical
slots 1/2 for the two models, diverging blue<->red for signed coefficients,
direct labels rather than legend-only identity, recessive grid, no dual axes.
Accuracy is never plotted -- see model.py's docstring for why.

Usage:
  python make_figures.py --features data/features.csv --test-congresses 117,118
  python make_figures.py --eda-only          # skip the model refit (fast)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from model import (CATS, NUMS, _unwrap, feature_names, fit_and_score,
                   load_features, split_by_congress)

# Palette (validated: node scripts/validate_palette.js "#2a78d6,#1baf7a,#e34948" --mode light)
BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def _style(ax, ylabel=None, xlabel=None, title=None, pct=False):
    if title:
        ax.set_title(title, loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if pct:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def _save(fig, path):
    # Reserve a bottom strip so the source-note fig.text (drawn just below the
    # axes) isn't clipped off the canvas.
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_rate_by_congress(df, outdir):
    """Trend over time, one series -> sequential hue, direct-labeled ends."""
    g = df.groupby("congress").agg(rate=("y", "mean"), n=("y", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(g.congress, g.rate, color=BLUE, linewidth=2, marker="o", markersize=6,
            markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for _, r in g.iterrows():
        if r.congress in (g.congress.min(), g.congress.max()):
            ax.annotate(f"{r.rate * 100:.1f}%", (r.congress, r.rate),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", color=INK, fontsize=9, fontweight="bold")
    _style(ax, "share of introduced bills that became law", "Congress",
           "Passage is getting rarer: enactment rate by Congress", pct=True)
    ax.set_ylim(0, g.rate.max() * 1.25)
    ax.set_xticks(g.congress)
    fig.text(0.01, -0.02, f"n = {int(g.n.sum()):,} bills (hr, s, hjres, sjres), 108th–118th Congress. "
             "Source: GPO BILLSTATUS bulk data.", fontsize=8, color=MUTED)
    _save(fig, outdir / "fig1_rate_by_congress.png")


def fig_structural_factors(df, outdir):
    """Emphasis form: each factor's passage rate vs the overall base rate line."""
    base = df.y.mean()
    factors = [
        ("bipartisan", "Bipartisan\ncosponsors", {0: "no", 1: "yes"}),
        ("sponsor_in_majority", "Sponsor in\nmajority party", {0: "no", 1: "yes"}),
        ("has_companion_bill", "Has companion\nbill", {0: "no", 1: "yes"}),
    ]
    chamber = df.groupby("chamber").y.mean()

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)
    for ax, (col, title, labels) in zip(axes, factors):
        g = df.groupby(col).y.mean()
        xs = [labels[v] for v in g.index]
        bars = ax.bar(xs, g.values, color=[SEQ[2], BLUE], width=0.6, zorder=3)
        for b, v in zip(bars, g.values):
            ax.annotate(f"{v * 100:.1f}%", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        color=INK, fontsize=9, fontweight="bold")
        _style(ax, None, None, title, pct=True)
        ax.axhline(base, color=RED, linestyle="--", linewidth=1.2, zorder=4)

    bars = axes[3].bar(chamber.index, chamber.values, color=[SEQ[2], BLUE], width=0.6, zorder=3)
    for b, v in zip(bars, chamber.values):
        axes[3].annotate(f"{v * 100:.1f}%", (b.get_x() + b.get_width() / 2, v),
                         textcoords="offset points", xytext=(0, 4), ha="center",
                         color=INK, fontsize=9, fontweight="bold")
    _style(axes[3], None, None, "Chamber", pct=True)
    axes[3].axhline(base, color=RED, linestyle="--", linewidth=1.2, zorder=4)

    # Label the base-rate line once, on the first panel at its left edge, where
    # the "no" bar (2.3%) sits well below the line so there's clear space.
    axes[0].annotate(f"overall base rate {base * 100:.1f}%", (-0.45, base),
                     textcoords="offset points", xytext=(0, 5), ha="left",
                     color=RED, fontsize=8, fontweight="bold")
    axes[0].set_ylabel("passage rate")
    fig.suptitle("Which introduction-time factors move passage?", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, -0.02, "Dashed red line = overall base rate. Descriptive rates over all "
             "128,778 bills, not model output.", fontsize=8, color=MUTED)
    _save(fig, outdir / "fig2_structural_factors.png")


def fig_policy_area(df, outdir, top_n=15):
    """Compare magnitude across many long-named categories -> horizontal bars,
    sequential shading by magnitude."""
    g = (df.groupby("policy_area").agg(rate=("y", "mean"), n=("y", "size"))
         .query("n >= 200").sort_values("rate", ascending=False))
    g = pd.concat([g.head(top_n // 2 + 1), g.tail(top_n // 2)])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    norm = g.rate / g.rate.max()
    colors = [SEQ[min(int(v * (len(SEQ) - 1)) + 1, len(SEQ) - 1)] for v in norm]
    bars = ax.barh(range(len(g)), g.rate.values, color=colors, height=0.72, zorder=3)
    ax.set_yticks(range(len(g)), [f"{i[:38]}" for i in g.index], fontsize=9)
    ax.invert_yaxis()
    for b, (v, n) in zip(bars, zip(g.rate.values, g.n.values)):
        ax.annotate(f"{v * 100:.1f}%  (n={n:,})", (v, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(5, 0), va="center",
                    color=INK2, fontsize=8)
    ax.axvline(df.y.mean(), color=RED, linestyle="--", linewidth=1.2, zorder=4)
    ax.set_title("Commemorative and procedural bills pass; ambitious policy dies",
                 loc="left", pad=12)
    ax.set_xlabel("passage rate")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.set_xlim(0, g.rate.max() * 1.32)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.02, "Top and bottom policy areas with n >= 200 bills. Dashed line = "
             "overall base rate (3.2%).", fontsize=8, color=MUTED)
    _save(fig, outdir / "fig3_policy_area.png")


def fig_cosponsors(df, outdir):
    """Dose-response: passage rate as a function of original-cosponsor count."""
    bins = [0, 1, 3, 6, 11, 21, 51, 101, 10_000]
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "100+"]
    df = df.assign(bucket=pd.cut(df.n_original_cosponsors, bins=bins, labels=labels,
                                 right=False))
    g = df.groupby("bucket", observed=True).agg(rate=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(g)), g.rate.values, color=BLUE, width=0.66, zorder=3)
    for b, (v, n) in zip(bars, zip(g.rate.values, g.n.values)):
        ax.annotate(f"{v * 100:.1f}%", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color=INK, fontsize=9, fontweight="bold")
        ax.annotate(f"n={n:,}", (b.get_x() + b.get_width() / 2, 0),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    color="white", fontsize=7)
    ax.set_xticks(range(len(g)), g.index)
    ax.axhline(df.y.mean(), color=RED, linestyle="--", linewidth=1.2, zorder=4)
    _style(ax, "passage rate", "original cosponsors at introduction",
           "More original cosponsors, more passage — but it saturates", pct=True)
    fig.text(0.01, -0.02, "Only cosponsors present on the introduction date "
             "(isOriginalCosponsor). Later cosponsors are a leakage feature and excluded.",
             fontsize=8, color=MUTED)
    _save(fig, outdir / "fig4_cosponsors.png")


def fig_pr_curves(test, proba, outdir):
    """Two series (models) -> categorical slots 1 & 2, both direct-labeled
    (aqua is sub-3:1 on the light surface, so the relief rule applies)."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    base = test.y.mean()
    # Anchor labels at low recall, where the two curves are well separated
    # vertically, instead of near the crossover where they collide.
    for name, color, label, anchor in [
            ("gradient_boosting", BLUE, "Gradient boosting", 0.30),
            ("logistic_regression", AQUA, "Logistic regression", 0.15)]:
        p, r, _ = precision_recall_curve(test.y, proba[name])
        ap = average_precision_score(test.y, proba[name])
        ax.plot(r, p, color=color, linewidth=2, zorder=3)
        idx = np.argmin(np.abs(r - anchor))
        ax.annotate(f"{label}  (PR-AUC {ap:.3f})", (r[idx], p[idx]),
                    textcoords="offset points", xytext=(18, 16),
                    color=INK, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=1.2))
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate(f"always-guess-pass baseline ({base * 100:.1f}%)", (0.98, base),
                textcoords="offset points", xytext=(0, 6), ha="right",
                color=MUTED, fontsize=8)
    _style(ax, "precision", "recall", "Both models beat the base rate by ~16x")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.text(0.01, -0.02, f"Held-out 117th–118th Congress: {len(test):,} bills, "
             f"{int(test.y.sum())} became law.", fontsize=8, color=MUTED)
    _save(fig, outdir / "fig5_pr_curves.png")


def fig_calibration(test, proba, outdir):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 0.35], [0, 0.35], color=MUTED, linestyle="--", linewidth=1.2,
            zorder=2)
    ax.annotate("perfect calibration", (0.24, 0.24), rotation=39, color=MUTED,
                fontsize=8, ha="center", va="bottom")
    for name, color, label in [("logistic_regression", AQUA, "Logistic regression"),
                               ("gradient_boosting", BLUE, "Gradient boosting")]:
        frac_pos, mean_pred = calibration_curve(test.y, proba[name], n_bins=10,
                                                strategy="quantile")
        ax.plot(mean_pred, frac_pos, color=color, linewidth=2, marker="o",
                markersize=7, markerfacecolor=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, zorder=3)
        ax.annotate(label, (mean_pred[-1], frac_pos[-1]),
                    textcoords="offset points", xytext=(-8, 10), ha="right",
                    color=INK, fontsize=9, fontweight="bold")
    _style(ax, "observed passage rate", "predicted probability",
           "Predicted probabilities are trustworthy")
    ax.grid(zorder=0)
    fig.text(0.01, -0.02, "10 quantile bins on the held-out Congresses. Points on the "
             "diagonal mean a predicted 12% really is a 12% chance.", fontsize=8, color=MUTED)
    _save(fig, outdir / "fig6_calibration.png")


def fig_importances(fitted, outdir, k=16):
    """Signed logistic coefficients -> diverging blue<->red about zero."""
    pipe = _unwrap(fitted["logistic_regression"])
    names = feature_names(pipe.named_steps["pre"], CATS, NUMS)
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=names)
    # Year tokens act as era dummies (train-only artifact); drop from the display.
    coefs = coefs[~coefs.index.str.match(r"word:(of )?(19|20)\d\d$")]
    top = pd.concat([coefs.sort_values(ascending=False).head(k),
                     coefs.sort_values().head(k)]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = [RED if v < 0 else BLUE for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors, height=0.74, zorder=3)
    ax.set_yticks(range(len(top)),
                  [n.replace("word:", "").replace("policy_area_", "policy: ")
                   .replace("primary_committee_", "cmte: ") for n in top.index],
                  fontsize=9)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_title("What predicts passage: signed logistic coefficients", loc="left", pad=12)
    ax.set_xlabel("← pushes toward DIED        pushes toward BECAME LAW →")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    fig.text(0.01, -0.015, "Model 1, trained on the 108th–116th Congress. Year tokens "
             "(e.g. 'of 2008') omitted: they act as era dummies, not real signal.",
             fontsize=8, color=MUTED)
    _save(fig, outdir / "fig7_importances.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--test-congresses", default="117,118")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--eda-only", action="store_true",
                    help="skip the Model 1 refit; only the corpus figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_features(args.features)

    print("EDA figures:")
    fig_rate_by_congress(df, outdir)
    fig_structural_factors(df, outdir)
    fig_policy_area(df, outdir)
    fig_cosponsors(df, outdir)

    if args.eda_only:
        return
    print("\nRefitting Model 1 for diagnostics...")
    test_congresses = [int(c) for c in args.test_congresses.split(",")]
    train, test = split_by_congress(df, test_congresses)
    fitted, proba, _ = fit_and_score(train, test, CATS, NUMS, "Model 1", verbose=False)

    print("\nModel figures:")
    fig_pr_curves(test, proba, outdir)
    fig_calibration(test, proba, outdir)
    fig_importances(fitted, outdir)


if __name__ == "__main__":
    main()
