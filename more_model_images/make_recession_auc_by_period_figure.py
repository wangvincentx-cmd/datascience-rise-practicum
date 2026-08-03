"""The recession model's AUC per period, tested one period at a time.

Outputs into this folder:
  fig_recession_auc_by_period.png   the chart, in figF1_topic_accuracy's format
  recession_auc_by_period.csv       the same numbers, machine-readable

NOT THE FORWARD-ONLY BACKTEST. fig10 and recession_period_table.png score a
model that only ever trained on years BEFORE the year it predicts, which is why
they start at 1930 -- the 1900s have no prior years to train on. This file asks
the other question: how well does the model rank each period, when that period
is the held-out test set? Every decade is trained on all the OTHER decades and
predicted from a fit that never saw it, so each number is still out-of-sample,
and the 1900s, 1910s and 1920s become scorable.

What that buys and what it costs: leave-one-period-out uses later years to
predict earlier ones, so these AUCs are NOT real-time and must never be quoted
as what a reader in 1908 could have known. They measure whether the press
signal ranks recession onsets within a period at all. The forward-only numbers
are the real-time claim; these are the per-period one. Do not mix them in one
sentence.

Model, features and target are imported from make_broad_vs_attention_figure --
same five d12 press series, same L2 logistic regression at C = 1, same
"NBER recession starts within 12 months" target on expansion months only -- so
the only thing that differs from the forward-only figures is which months train
and which are tested.

Colour follows recession_period_table.png: red ABOVE 0.50, blue below. That is
the reverse of sheets_per_period.png and backtest_table.png in this folder --
see the warning in make_recession_period_table.py, and do not put the two
conventions on one poster.

Run from the repo root:
    python more_model_images/make_recession_auc_by_period_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "src")
sys.path.insert(0, str(HERE))

from make_broad_vs_attention_figure import data, SERIES

CSV = HERE / "recession_auc_by_period.csv"
OUT = HERE / "fig_recession_auc_by_period.png"

PERIODS = [("1900s", 1900, 1909), ("1910s", 1910, 1919), ("1920s", 1920, 1929),
           ("1930s", 1930, 1939), ("1940s", 1940, 1949), ("1950s", 1950, 1959),
           ("1960-63", 1960, 1963)]

# figF1's palette and rcParams, restated rather than imported: make_poster_figures
# reads data/scored/ at import time, and this figure needs none of it.
BLUE = "#0072B2"
RED = "#C0392B"
INK, MUTED = "#1a1a1a", "#6b6b6b"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})


def leave_one_period_out():
    """AUC inside each period, from a model fitted on every other period.

    Standardisation is fitted on the training periods only and applied to the
    held-out one, for the same reason the fit is: a scaler fitted on all years
    has already seen the test period's spread."""
    X, y, yr = data()
    Xf = X[SERIES]
    rows = []
    for lab, lo, hi in PERIODS:
        te = (yr >= lo) & (yr <= hi)
        tr = ~te
        if te.sum() == 0 or y[te].nunique() < 2 or y[tr].nunique() < 2:
            print(f"  [skipping {lab}: one class only]")
            continue
        mu, sd = Xf[tr].mean(), Xf[tr].std().replace(0, 1)
        m = LogisticRegression(C=1.0, max_iter=5000).fit((Xf[tr] - mu) / sd, y[tr])
        p = m.predict_proba((Xf[te] - mu) / sd)[:, 1]
        rows.append({"period": lab, "months": int(te.sum()),
                     "onsets": int(y[te].sum()), "base_rate": float(y[te].mean()),
                     "auc": roc_auc_score(y[te], p)})
    return pd.DataFrame(rows)


def draw(d):
    fig, ax = plt.subplots(figsize=(7.5, 6.45))
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        ax.barh(i, r.auc, color=RED if r.auc > .5 else BLUE, height=.62, zorder=3)
        ax.text(r.auc + .012, i, f"{r.auc:.3f}", va="center", fontsize=13,
                fontweight="bold", color=INK)
    ax.axvline(.5, color="#999999", lw=1.4, ls="--", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.period}\n({r.months} mo, {r.onsets} onsets)"
                        for r in d.itertuples()], fontsize=12)
    # Chronological top to bottom: figF1 sorts by value because its categories
    # have no order, and periods do.
    ax.set_ylim(len(d) - .4, -1.1)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=12)
    ax.set_xlabel("ROC-AUC on the held-out period,\n"
                  "P(recession starts within 12 months)", fontsize=12)
    ax.text(.5, -1.0, "no information", fontsize=11, color=MUTED,
            ha="center", va="bottom")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


def main():
    d = leave_one_period_out()
    d.to_csv(CSV, index=False)
    draw(d)
    print(f"-> {OUT}\n-> {CSV}")
    for r in d.itertuples():
        print(f"  {r.period:<10}{r.auc:>7.3f}   {r.months:>4} months, "
              f"{r.onsets:>3} onsets, base rate {r.base_rate:.2f}")


if __name__ == "__main__":
    main()
