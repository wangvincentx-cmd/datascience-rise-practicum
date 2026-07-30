"""The logistic regression, drawn as a curve against the data it was fitted to.

This is the model as a GRAPH rather than as an equation or a block diagram:
the S-curve that turns the linear index z into a probability, with the actual
out-of-fold outcomes plotted on top of it so a reader can see whether the curve
is telling the truth.

    P(hit) = sigma(z),   z = b_news'x_news + b_econ'x_econ + b_int'(x_econ*d)

Three layers, in order of what they are for:

  the curve      the model. sigma(z), drawn exactly as fitted.
  the dots       reality. Claims are binned by z into equal-count bins and the
                 OBSERVED hit rate in each bin is plotted with a Wilson
                 interval. If the model is honest these sit on the curve.
  the rug        the raw data. Every claim is a tick at y=1 (came true) or
                 y=0 (did not), so the reader sees that the dots summarise
                 14,251 individual binary outcomes rather than appearing from
                 nowhere.

Predictions are OUT-OF-FOLD (leave-one-3-year-block-out, from
data/models/hit_predictions.csv), never in-sample. An in-sample version of this
figure would show a beautiful fit and mean nothing -- the whole point of the
poster's Panel 4 is that in-sample numbers here are worthless.

The figure is deliberately allowed to show the model's known flaw: docs/
RESULTS_MACRO.md records that calibration is "monotone but too extreme at both
ends", and that is visible as dots pulling inside the curve at the tails. A
calibration plot that hid this would be worth less than one that shows it.

    python src/make_logistic_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "figures/poster_v2"
PRED = "data/models/hit_predictions.csv"

# House palette -- see src/make_poster_v2_figures.py.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

W = 13.4
BASE, LABEL, TITLE, NOTE = 17, 18, 25, 14.5

N_BINS = 12          # equal-count bins; 12 keeps ~1,200 claims per point
X_LO, X_HI = 0.0, 1.0    # the axis is the predicted probability itself


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near 0 and 1, where the normal
    approximation gives intervals that run outside [0, 1]."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"], "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": BASE,
    })


def main():
    if not os.path.exists(PRED):
        raise SystemExit(f"{PRED} not found -- run `python src/hit_predictor.py` first.")
    os.makedirs(OUT, exist_ok=True)
    style()

    d = pd.read_csv(PRED)
    d = d[d["hit"].isin([0, 1]) & d["p_hit_oof"].between(1e-6, 1 - 1e-6)]
    p = d["p_hit_oof"].values
    y = d["hit"].values.astype(int)
    z = np.log(p / (1 - p))

    fig, ax = plt.subplots(figsize=(W, 7.4))
    fig.subplots_adjust(left=0.125, right=0.98, top=0.965, bottom=0.235)

    # --- reference lines (recessive: they orient, they do not compete) ------
    ax.axhline(y.mean(), color=ORANGE, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.text(X_LO + 0.008, y.mean() + 0.02,
            f"base rate {y.mean():.3f}  —  {y.mean():.1%} of all forecasts came true",
            color=ORANGE, fontsize=NOTE, ha="left", va="bottom")

    # --- how many forecasts land at each predicted probability --------------
    h, be = np.histogram(p, bins=50, range=(X_LO, X_HI))
    ax.bar(0.5 * (be[:-1] + be[1:]), 0.075 * h / h.max(), bottom=-0.028,
           width=(be[1] - be[0]) * 0.92, color=AXIS, alpha=0.55, lw=0,
           zorder=2)
    ax.text(X_HI - 0.008, 0.055, "how many forecasts the model scored here",
            color=MUTED, fontsize=NOTE - 1.5, ha="right", va="bottom")

    # --- the model ----------------------------------------------------------
    # On a probability axis the fitted logistic IS the 45-degree line: the model
    # saying 0.30 is the model claiming 30% of those forecasts came true. The
    # S-curve only appears when the axis is log-odds.
    ax.plot([X_LO, X_HI], [X_LO, X_HI], color=INK, lw=2.6, zorder=5,
            solid_capstyle="round",
            label="a model whose probabilities were exactly right")

    # --- reality, in equal-count bins ---------------------------------------
    edges = np.quantile(p, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    xs, ys, los, his, ns = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum() < 30:
            continue
        lo, hi = wilson(y[m].sum(), m.sum())
        xs.append(p[m].mean()); ys.append(y[m].mean())
        los.append(lo); his.append(hi); ns.append(m.sum())
    xs, ys = np.array(xs), np.array(ys)

    ax.vlines(xs, los, his, color=BLUE, lw=1.6, alpha=0.55, zorder=6)
    ax.plot(xs, ys, "o", color=BLUE, markersize=9, mec=SURFACE, mew=1.8,
            zorder=7, label=f"what actually happened  —  each dot is "
                            f"{int(np.mean(ns)):,} forecasts, 95% Wilson")

    # --- the honest annotation ----------------------------------------------
    for i, ha, dx, dy in [(int(np.argmax(xs)), "right", -0.03, -0.20),
                          (int(np.argmin(xs)), "left", 0.03, -0.11)]:
        ax.annotate(f"model said {xs[i]:.0%},  truth was {ys[i]:.0%}",
                    xy=(xs[i], ys[i]), xytext=(xs[i] + dx, ys[i] + dy),
                    fontsize=NOTE, color=INK2, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=1.2))

    ax.set_xlim(X_LO - 0.012, X_HI + 0.012)
    ax.set_ylim(-0.03, 1.03)
    # Both labels name the GROUP explicitly. An earlier pair ("what the model
    # predicted" / "how often it actually came true") read as though each dot
    # were one forecast, which made the y-axis nonsense -- a single forecast
    # cannot "often" come true. Every dot is ~1,187 forecasts; say so.
    ax.set_xlabel("what the model predicted for a group of forecasts",
                  fontsize=LABEL, labelpad=10)
    ax.set_ylabel("share of that group\nthat actually came true",
                  fontsize=LABEL, labelpad=10)
    for setter in (ax.set_xticks, ax.set_yticks):
        setter([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    leg = ax.legend(loc="upper left", frameon=False, fontsize=NOTE,
                    handletextpad=0.8, borderaxespad=0.4)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.text(0.125, 0.115,
             "All {:,} forecasts are sorted by the probability the model gave "
             "them, then cut into {} equal-sized groups.\nOut-of-fold only — "
             "every forecast is scored by a model that never saw its 3-year "
             "era.   1900–1963.".format(len(d), N_BINS),
             fontsize=NOTE, color=MUTED, va="top", linespacing=1.5)

    out = os.path.join(OUT, "v2_fig8_logistic_curve.png")
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  bins {len(xs)}   z range {z.min():.2f}..{z.max():.2f}   "
          f"n per bin ~{int(np.mean(ns))}")


if __name__ == "__main__":
    main()
