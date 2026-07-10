"""
Poster figures for the bill-arm Model 1 (structural features, Section 7).

Re-fits the exact pipelines from model.py (same split, same seeds) on the
held-out 118th Congress and draws:

    figures/fig_bill_pr_curves.png       precision-recall curves (the primary metric)
    figures/fig_bill_calibration.png     reliability curve, quantile bins
    figures/fig_bill_importances.png     strongest logistic coefficients
    figures/fig_bill_pass_rate.png       enactment rate by congress (context)

Usage:  python model_figures.py            # ~2-4 min (re-fits both models)
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from model import (CATS, NUMS, _unwrap, feature_names, fit_and_score,
                   load_features, split_by_congress)

FIGDIR = Path("figures")

# Same reference palette as JeremysShit/model_figures.py (dataviz skill).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # slot 1 -> logistic regression / "became law" pole
AQUA = "#1baf7a"   # slot 2 -> gradient boosting
RED = "#e34948"    # diverging opposite pole -> "died"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 12,
    "legend.frameon": False,
})


def styled_axes(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_pr_curves(test, proba):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    styled_axes(ax)
    base = test.y.mean()
    ax.axhline(base, color=BASELINE, lw=1)
    ax.annotate(f"guessing (base rate {base:.1%})", (0.55, base),
                xytext=(0, 5), textcoords="offset points",
                color=MUTED, fontsize=9)
    for (name, p), color, label in zip(proba.items(), [BLUE, AQUA],
                                       ["Logistic regression", "Gradient boosting"]):
        prec, rec, _ = precision_recall_curve(test.y, p)
        auc = average_precision_score(test.y, p)
        ax.plot(rec, prec, color=color, lw=2, solid_joinstyle="round",
                solid_capstyle="round", label=f"{label} (PR-AUC {auc:.2f})")
        idx = int(np.argmin(np.abs(rec - 0.35)))
        ax.annotate(f"{label}\nPR-AUC {auc:.2f}", (rec[idx], prec[idx]),
                    xytext=(10, 8), textcoords="offset points",
                    color=INK2, fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="recall (share of enacted bills found)",
           ylabel="precision (share of flagged bills that passed)")
    ax.set_title("Predicting which bills become law, from introduction-day\n"
                 "features only — held-out 118th Congress (16,565 bills)")
    ax.legend(loc="upper right", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_bill_pr_curves.png", dpi=200)
    plt.close(fig)


def fig_calibration(test, proba):
    fig, ax = plt.subplots(figsize=(6, 5))
    styled_axes(ax)
    lim = 0.16
    ax.plot([0, lim], [0, lim], color=BASELINE, lw=1)
    ax.annotate("perfectly calibrated", (lim * 0.55, lim * 0.48), color=MUTED,
                fontsize=9, rotation=41, rotation_mode="anchor")
    for (name, p), color, label in zip(proba.items(), [BLUE, AQUA],
                                       ["Logistic regression", "Gradient boosting"]):
        frac_pos, mean_pred = calibration_curve(test.y, p, n_bins=10,
                                                strategy="quantile")
        ax.plot(mean_pred, frac_pos, color=color, lw=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    ax.set(xlim=(0, lim), ylim=(0, lim),
           xlabel="mean predicted P(became law)",
           ylabel="observed share that became law")
    ax.set_title("The models over-predict passage: they learned from Congresses\n"
                 "that passed twice as many bills — reliability curve, 118th")
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_bill_calibration.png", dpi=200)
    plt.close(fig)


def fig_importances(fitted):
    pipe = _unwrap(fitted["logistic_regression"])
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    coefs = pd.Series(clf.coef_[0], index=feature_names(pre, CATS, NUMS))
    # drop raw year tokens; they proxy for congress, not content
    coefs = coefs[~coefs.index.str.contains(r"(?:19|20)\d\d")]
    top = pd.concat([coefs.sort_values().head(10), coefs.sort_values().tail(10)])
    fig, ax = plt.subplots(figsize=(8, 6))
    styled_axes(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(top))
    ax.barh(y, top.values, height=0.55,
            color=np.where(top.values > 0, BLUE, RED), zorder=2)
    ax.axvline(0, color=BASELINE, lw=1, zorder=1)
    ax.set_yticks(y, top.index, fontsize=9)
    ax.set_xlabel("logistic coefficient")
    ax.annotate("pushes toward DIED", (0.02, 1.01), xycoords="axes fraction",
                color=RED, fontsize=9, ha="left")
    ax.annotate("pushes toward BECAME LAW", (0.98, 1.01), xycoords="axes fraction",
                color=BLUE, fontsize=9, ha="right")
    ax.set_title("What a bill's title and structure say about its fate\n"
                 "Strongest logistic-regression features", pad=24)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_bill_importances.png", dpi=200)
    plt.close(fig)


def fig_pass_rate(df):
    rate = df.groupby("congress")["y"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    styled_axes(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(rate))
    ax.bar(x, rate.values * 100, width=0.6, color=BLUE, zorder=2)
    for xi, v in zip(x, rate.values):
        ax.annotate(f"{v * 100:.1f}", (xi, v * 100), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
    ax.set_xticks(x, [f"{c}th" for c in rate.index], fontsize=9)
    ax.set_ylabel("share of introduced bills enacted (%)")
    ax.set_ylim(0, rate.max() * 100 * 1.25)
    ax.set_title("Passing a bill keeps getting harder\n"
                 "Enactment rate by Congress, 2003–2024")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_bill_pass_rate.png", dpi=200)
    plt.close(fig)


def main():
    FIGDIR.mkdir(exist_ok=True)
    df = load_features("data/features.csv")
    train, test = split_by_congress(df, [118])
    fitted, proba, _ = fit_and_score(train, test, CATS, NUMS,
                                     "Model 1 (structural)", verbose=False)
    fig_pr_curves(test, proba)
    fig_calibration(test, proba)
    fig_importances(fitted)
    fig_pass_rate(df)
    print("Wrote 4 figures to", FIGDIR.resolve())


if __name__ == "__main__":
    main()
