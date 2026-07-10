"""
Poster figures for the economy-arm model (the one model that has been trained
so far — election_arm and bill_arm models have no input data yet).

Re-fits the exact pipelines from model.py (same episode split, same seeds) and
draws the evaluation suite:

    figures/fig_model_roc.png            ROC curves, held-out episodes
    figures/fig_model_calibration.png    reliability curves (quantile bins)
    figures/fig_model_proba_dist.png     predicted P(hit) by actual outcome
    figures/fig_model_loeo.png           leave-one-episode-out accuracy
    figures/fig_model_importances.png    logistic coefficients (restyled)

Usage:  python model_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from model import TEST_EPISODES_DEFAULT, build, feature_names, pipeline, FIGDIR

# Reference palette (dataviz skill): series slots in fixed order, ink tokens.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # slot 1 -> logistic regression / "correct" pole
AQUA = "#1baf7a"   # slot 2 -> gradient boosting
RED = "#e34948"    # diverging opposite pole -> "wrong"

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


def fit_models(df):
    test_eps = [e for e in TEST_EPISODES_DEFAULT if e in set(df["episode"])]
    train = df[~df["episode"].isin(test_eps)]
    test = df[df["episode"].isin(test_eps)]
    fits = {}
    for name, model in [("Logistic regression", LogisticRegression(max_iter=2000, C=0.5)),
                        ("Gradient boosting", GradientBoostingClassifier(random_state=0))]:
        pipe = pipeline(model)
        pipe.fit(train, train["hit"])
        fits[name] = (pipe, pipe.predict_proba(test)[:, 1])
    return train, test, fits


def fig_roc(test, fits):
    fig, ax = plt.subplots(figsize=(6, 5))
    styled_axes(ax)
    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1)
    ax.annotate("chance", (0.72, 0.68), color=MUTED, fontsize=9)
    for (name, (_, proba)), color in zip(fits.items(), [BLUE, AQUA]):
        fpr, tpr, _ = roc_curve(test["hit"], proba)
        auc = roc_auc_score(test["hit"], proba)
        ax.plot(fpr, tpr, color=color, lw=2, solid_joinstyle="round",
                solid_capstyle="round", label=f"{name} (AUC {auc:.2f})")
        # direct label at the curve's midpoint
        mid = len(fpr) // 2
        ax.annotate(f"{name}\nAUC {auc:.2f}", (fpr[mid], tpr[mid]),
                    xytext=(8, -4), textcoords="offset points",
                    color=INK2, fontsize=9)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="false positive rate",
           ylabel="true positive rate")
    ax.set_title("Can the model tell right calls from wrong ones?\n"
                 "ROC on held-out post-war episodes (1945–1957)")
    ax.legend(loc="lower right", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_model_roc.png", dpi=200)
    plt.close(fig)


def fig_calibration(test, fits):
    fig, ax = plt.subplots(figsize=(6, 5))
    styled_axes(ax)
    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1)
    ax.annotate("perfectly calibrated", (0.6, 0.55), color=MUTED, fontsize=9,
                rotation=38, rotation_mode="anchor")
    for (name, (_, proba)), color in zip(fits.items(), [BLUE, AQUA]):
        frac_pos, mean_pred = calibration_curve(
            test["hit"], proba, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, color=color, lw=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=name)
        ax.annotate(name, (mean_pred[-1], frac_pos[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    color=INK2, fontsize=9, va="center")
    ax.set(xlim=(0, 1), ylim=(0, 1.02),
           xlabel="mean predicted P(claim was right)",
           ylabel="observed share actually right")
    ax.set_title("Are the model's probabilities honest?\n"
                 "Reliability curve, quantile bins, held-out episodes")
    ax.legend(loc="upper left", fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_model_calibration.png", dpi=200)
    plt.close(fig)


def fig_proba_dist(test, fits):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, (name, (_, proba)) in zip(axes, fits.items()):
        styled_axes(ax)
        right = proba[test["hit"].values == 1]
        wrong = proba[test["hit"].values == 0]
        ax.hist(wrong, bins=bins, color=RED, alpha=0.85, label="claim was wrong",
                edgecolor=SURFACE, linewidth=1)
        ax.hist(right, bins=bins, color=BLUE, alpha=0.65, label="claim was right",
                edgecolor=SURFACE, linewidth=1)
        ax.axvline(0.5, color=BASELINE, lw=1)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("predicted P(claim was right)")
    axes[0].set_ylabel("held-out claims")
    axes[0].legend(fontsize=9, labelcolor=INK2, loc="upper right")
    fig.suptitle("Where the models place right vs. wrong claims",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGDIR / "fig_model_proba_dist.png", dpi=200)
    plt.close(fig)


def fig_loeo(df):
    pipe = pipeline(LogisticRegression(max_iter=2000, C=0.5))
    scores = cross_val_score(pipe, df, df["hit"], groups=df["episode"],
                             cv=LeaveOneGroupOut(), scoring="accuracy")
    eps = sorted(set(df["episode"]))
    order = np.argsort(scores)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(eps) + 1.6))
    styled_axes(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(eps))
    ax.barh(y, scores[order], height=0.55, color=BLUE, zorder=2)
    for yi, sc in zip(y, scores[order]):
        ax.annotate(f"{sc:.2f}", (sc, yi), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    mean = scores.mean()
    ax.axvline(mean, color=BASELINE, lw=1, zorder=1)
    ax.annotate(f"mean {mean:.2f}", (mean, len(eps) - 0.2), xytext=(4, 0),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_yticks(y, [eps[i] for i in order], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("accuracy when this episode is held out entirely")
    ax.set_title("The honest overall number: leave-one-episode-out accuracy\n"
                 "Logistic regression, one bar per held-out episode")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_model_loeo.png", dpi=200)
    plt.close(fig)
    return scores


def fig_importances(fits):
    pipe = fits["Logistic regression"][0]
    coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_names(pipe))
    coefs = coefs[coefs.abs() > 1e-6]
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
    ax.annotate("pushes toward WRONG", (0.02, 1.01), xycoords="axes fraction",
                color=RED, fontsize=9, ha="left")
    ax.annotate("pushes toward RIGHT", (0.98, 1.01), xycoords="axes fraction",
                color=BLUE, fontsize=9, ha="right")
    ax.set_title("What made a newspaper prediction likely to be right?\n"
                 "Strongest logistic-regression features", pad=24)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_model_importances.png", dpi=200)
    plt.close(fig)


def main():
    FIGDIR.mkdir(exist_ok=True)
    df = build(pd.read_csv("claims_scored.csv"))
    train, test, fits = fit_models(df)
    print(f"{len(df)} claims; train {len(train)} / test {len(test)} "
          f"(held-out episodes: {sorted(set(test['episode']))})")
    fig_roc(test, fits)
    fig_calibration(test, fits)
    fig_proba_dist(test, fits)
    scores = fig_loeo(df)
    fig_importances(fits)
    print(f"LOEO accuracy {scores.mean():.3f} ± {scores.std():.3f}")
    print("Wrote 5 figures to", FIGDIR.resolve())


if __name__ == "__main__":
    main()
