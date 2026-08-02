"""The recession model against the calendar: what it said, month by month, vs
what the economy actually did.

  fig_timeline_fit.png

The AUC tables say how well the model RANKS months. They cannot show whether it
rose before the 1937 peak and stayed flat before 1960 -- and that is the question
a reader actually asks of a forecasting model. This puts the two things on one
time axis:

  * the line   -- forward-only P(recession begins within 12 months), refit every
                  year on all prior years only, so every point on it was produced
                  by a model that had not seen that year.
  * the amber  -- the months where that WAS true. The line should be high inside
                  amber and low outside it; nothing else about the shape matters.
  * the gray   -- recessions themselves. These months are dropped from both
                  training and scoring (the question is onset, not continuation),
                  which is why the line breaks there rather than being drawn at
                  zero -- a drawn zero would be a prediction the model never made.

Everything comes from make_broad_vs_attention_figure, so this figure, the bar
chart, and the two tables are all one computation. NOTE: importing that module
regenerates broad_vs_attention.png as a side effect -- same numbers, no drift.

Run from the repo root:  python more_model_images/make_timeline_fit_figure.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from truth_data import NBER_RECESSIONS
from make_broad_vs_attention_figure import H, SERIES, forward_only

OUT = Path(__file__).parent

BLUE, ORANGE = "#1967d2", "#c5490b"
INK, MUTED, FAINT = "#202124", "#5f6368", "#9aa0a6"
GRID, RULE = "#e8eaed", "#dadce0"
AMBER, AMBER_INK = "#fbe7c6", "#8a5a00"
RECESSION = "#ebedf0"

START = 1930
LO, HI = pd.Period("1932-01", "M"), pd.Period("1963-12", "M")
FULL = pd.period_range(LO, HI, freq="M")

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": INK, "axes.axisbelow": True})


def spans():
    """(recession, onset-window) spans as timestamp pairs, clipped to the axis.

    The onset window is the H months BEFORE a peak -- the months in which a model
    predicting 'recession within H months' should be firing. It is derived from
    the NBER peak dates alone, exactly as the target is."""
    rec, win = [], []
    for p, t in NBER_RECESSIONS:
        p, t = pd.Period(p, "M"), pd.Period(t, "M")
        if t < LO or p > HI:
            continue
        rec.append((max(p, LO), min(t, HI)))
        w0 = max(p - H, LO)
        if p - 1 >= LO:
            win.append((w0, min(p - 1, HI), p))
    return rec, win


def ts(period):
    return period.to_timestamp()


def main():
    p_five, truth = forward_only(SERIES, START)
    p_attn, _ = forward_only(["attention"], START)
    base = truth.mean()
    rec, win = spans()

    fig, (ax, ax_t) = plt.subplots(
        2, 1, figsize=(12.4, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [6.2, 1], "hspace": 0.14})

    # --- shading first, so every mark sits on top of it ---
    for a in (ax, ax_t):
        for p, t in rec:
            a.axvspan(ts(p), ts(t + 1), color=RECESSION, lw=0, zorder=1)
        for w0, w1, _ in win:
            a.axvspan(ts(w0), ts(w1 + 1), color=AMBER, lw=0, zorder=2)

    # --- the two models, broken wherever a month was not scored ---
    for p, color, lw, label in [(p_five, BLUE, 2.0, "Five press series"),
                                (p_attn, ORANGE, 1.5, "Attention alone")]:
        s = p.reindex(FULL)
        ax.plot(FULL.to_timestamp(), s.values, color=color, lw=lw, zorder=5,
                solid_capstyle="round", label=label)

    ax.axhline(base, color=FAINT, lw=1.2, ls="--", zorder=4)
    ax.text(ts(HI), base, f"  base rate {base:.2f}", color=MUTED, fontsize=9,
            va="center", ha="left")

    # Peak markers, in headroom above the plotted range rather than above the
    # axes -- outside the axes they land on the legend.
    for _, _, peak in win:
        ax.plot([ts(peak)], [1.07], marker="v", ms=7, color=AMBER_INK, zorder=6)
        ax.text(ts(peak), 1.12, str(peak.year), color=AMBER_INK, fontsize=8.5,
                ha="center", va="bottom")

    ax.set_ylim(0, 1.26)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=9.5)
    ax.set_ylabel("P(recession begins\nwithin 12 months)", fontsize=10,
                  color=MUTED, linespacing=1.5)
    ax.grid(axis="y", color=GRID, lw=.8)

    # --- what actually happened, as its own row ---
    ax_t.set_ylim(0, 1)
    ax_t.set_yticks([])
    ax_t.set_xlim(ts(LO), ts(HI + 1))
    for s in ax_t.spines.values():
        s.set_visible(False)
    ax_t.tick_params(axis="x", length=3)
    ax_t.set_xlabel("", fontsize=10)
    ax_t.text(ts(LO) + pd.Timedelta(days=40), .5,
              "what actually happened", fontsize=9.5, color=MUTED,
              va="center", ha="left", zorder=6)

    fig.suptitle("It leaned into three of six downturns — and sat out the "
                 "other three", fontsize=13.5, fontweight="600",
                 x=.008, ha="left", y=1.055)
    fig.text(.008, .985,
             "Forward-only predictions, 1930 start — each year is predicted by a "
             "model trained only on the years before it",
             fontsize=10, color=MUTED, ha="left", va="center")

    # Legend: two lines + the two shadings, drawn as figure text so the plot
    # area keeps its full width.
    fig.add_artist(plt.Line2D([.008, .036], [.918, .918], color=BLUE, lw=2.4))
    fig.text(.042, .918, "Five press series", fontsize=9.5, va="center")
    fig.add_artist(plt.Line2D([.175, .203], [.918, .918], color=ORANGE, lw=2))
    fig.text(.209, .918, "Attention alone", fontsize=9.5, va="center")
    fig.patches.append(plt.Rectangle((.335, .906), .028, .024, color=AMBER,
                                     transform=fig.transFigure, figure=fig))
    fig.text(.369, .918, "recession begins within 12 months (the target)",
             fontsize=9.5, va="center")
    fig.patches.append(plt.Rectangle((.665, .906), .028, .024, color=RECESSION,
                                     transform=fig.transFigure, figure=fig))
    fig.text(.699, .918, "recession under way — months dropped, line breaks",
             fontsize=9.5, va="center")

    # The verdict, computed rather than eyeballed: mean probability inside each
    # amber window against the mean everywhere else. Anything at or below the
    # elsewhere-mean is a peak the model did not lean into.
    means, covered = [], pd.Series(False, index=p_five.index)
    for w0, w1, peak in win:
        s = p_five.reindex(pd.period_range(w0, w1, freq="M")).dropna()
        if s.empty:
            continue
        covered.loc[s.index] = True
        means.append((peak.year, s.mean()))
    elsewhere = p_five[~covered].mean()
    hit = [f"{yy} {m:.2f}" for yy, m in means if m > elsewhere]
    miss = [f"{yy} {m:.2f}" for yy, m in means if m <= elsewhere]

    fig.text(.008, -.10,
             "Read it as: inside amber the line should be HIGH, outside it low. "
             "Mean probability in the twelve months before each peak — "
             f"leaned into: {' · '.join(hit)};   missed: {' · '.join(miss)}   "
             f"(all other months {elsewhere:.2f}).\n"
             "Three of six peaks are anticipated and three are not, which is "
             "what an AUC of 0.64 looks like on a calendar — a real but "
             "unreliable signal, not an alarm you could act on.\n"
             "The curve is a RANKING, not a calibrated probability: it sits well "
             "above the 24% base rate most of the time, so read where it is high "
             "relative to itself, not against 50%.",
             fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.6)

    p = OUT / "fig_timeline_fit.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=.10)
    plt.close(fig)
    print(f"-> {p}")


def per_period_calibration():
    """fig_per_period_calibration.png -- the LEVEL of the predictions per period,
    which the AUC/Brier table cannot show.

    fig_per_period.png reports how well each decade is RANKED. This reports
    something a ranking metric is blind to: on average the model said 44% and the
    answer was 24%. A model can rank a decade almost perfectly and still be
    calling three times as many downturns as happened, and that is exactly what
    1960-1963 does."""
    p_five, truth = forward_only(SERIES, START)
    p_attn, _ = forward_only(["attention"], START)
    years = np.array([m.year for m in p_five.index])

    def bucket(y):
        return ("1930s" if y < 1940 else "1940s" if y < 1950 else
                "1950s" if y < 1960 else "1960-1963")

    g = pd.DataFrame({"period": [bucket(y) for y in years],
                      "five": p_five.values, "attn": p_attn.values,
                      "truth": truth.values})
    agg = g.groupby("period").agg(n=("truth", "size"), actual=("truth", "mean"),
                                  five=("five", "mean"), attn=("attn", "mean"))
    agg = agg.reindex(["1930s", "1940s", "1950s", "1960-1963"])
    allrow = pd.Series({"n": len(g), "actual": g["truth"].mean(),
                        "five": g["five"].mean(), "attn": g["attn"].mean()},
                       name="1930-1963 (all)")
    rows = [(i, r) for i, r in agg.iterrows()] + [(allrow.name, allrow)]
    ys = list(np.arange(len(agg))) + [len(agg) + .55]

    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    for y, (_, r) in zip(ys, rows):
        lo, hi = sorted([r["five"], r["attn"]])
        ax.plot([r["actual"], hi], [y, y], color="#e3e5e8", lw=6, zorder=1,
                solid_capstyle="round")
        ax.plot([r["five"]], [y], "o", ms=11, color=BLUE, mec="white", mew=1.8,
                zorder=5)
        ax.plot([r["attn"]], [y], "o", ms=11, color=ORANGE, mec="white", mew=1.8,
                zorder=4)
        ax.plot([r["actual"]], [y], marker="D", ms=9, color=INK, mec="white",
                mew=1.6, zorder=6)
        ax.text(r["actual"] - .012, y, f"{r['actual']:.2f}", ha="right",
                va="center", fontsize=9.5, color=INK, fontweight="600")
        ax.text(hi + .014, y, f"said {hi:.2f}", ha="left", va="center",
                fontsize=9.5, color=INK, fontweight="600")
        ax.annotate("", xy=(hi - .004, y + .30), xytext=(r["actual"] + .004,
                                                         y + .30),
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1,
                                    shrinkA=0, shrinkB=0), zorder=3)
        ax.text((r["actual"] + hi) / 2, y + .46,
                f"over by {hi - r['actual']:+.2f}", ha="center", va="top",
                fontsize=8.5, color=MUTED)

    ax.axhline((len(agg) - 1 + ys[-1]) / 2, color=RULE, lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{i}\n{int(r['n'])} months" for i, r in rows],
                       fontsize=10)
    ax.set_ylim(max(ys) + .55, -.85)
    ax.set_xlim(0, .72)
    ax.set_xticks([0, .2, .4, .6])
    ax.set_xticklabels(["0", "20%", "40%", "60%"], fontsize=9.5)
    ax.set_xlabel("share of months  ·  predicted vs actual", fontsize=10,
                  color=MUTED)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.tick_params(axis="y", length=0)

    fig.suptitle("Every period is over-called — the model cries recession about "
                 "twice as often as one arrived",
                 fontsize=13, fontweight="600", x=.008, ha="left", y=1.10)
    fig.text(.008, 1.012, "Mean forward-only predicted probability against the "
             "share of months that really were within 12 months of a downturn",
             fontsize=10, color=MUTED, ha="left", va="center")
    fig.add_artist(plt.Line2D([.012], [.945], marker="D", ms=8, color=INK,
                              mec="white", mew=1.4))
    fig.text(.030, .945, "what actually happened", fontsize=9.5, va="center")
    fig.add_artist(plt.Line2D([.262], [.945], marker="o", ms=9, color=BLUE,
                              mec="white", mew=1.4))
    fig.text(.280, .945, "Five press series", fontsize=9.5, va="center")
    fig.add_artist(plt.Line2D([.452], [.945], marker="o", ms=9, color=ORANGE,
                              mec="white", mew=1.4))
    fig.text(.470, .945, "Attention alone", fontsize=9.5, va="center")
    fig.text(.008, -.14,
             "This is the gap a ROC-AUC cannot see: 1960–1963 is ranked at 0.33 "
             "AND over-called five-fold, while the 1950s are ranked at 0.81 and "
             "over-called by a third.\n"
             "The forward-only design is the cause, not a bug — each year is "
             "predicted by a model fitted on earlier years, whose onset rate was "
             "higher, and no recalibration is applied.\n"
             "Use the curve for ranking (which months look riskier than others), "
             "never as a probability to quote.",
             fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.6)

    p = OUT / "fig_per_period_calibration.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=.10)
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
    per_period_calibration()
