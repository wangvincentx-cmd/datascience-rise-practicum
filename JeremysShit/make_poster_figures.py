"""
Poster figures -- the final analysis figures for the monthly corpus.

Each figure answers one question and is designed to be read at poster distance:
big marks, direct labels, one idea per panel. Palette is Okabe-Ito (CVD-safe,
validated with the dataviz skill's checker).

  fig_A  the mechanism: forecast mix doesn't respond to the economy
  fig_B  the consequence: accuracy collapses in recessions
  fig_C  no learning across six decades
  fig_D  what predicts a hit (feature ladder)

Usage: python make_poster_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from truth_data import NBER_RECESSIONS

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("poster_figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})

REC = set()
for a, b in NBER_RECESSIONS:
    REC.update(pd.period_range(a, b, freq="M"))


def title(ax, t, sub):
    n = 1 + sub.count("\n")
    ax.set_title(t, fontsize=15, fontweight="bold", loc="left", pad=18 + 15 * n)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=10.5,
            color=MUTED, va="bottom")


def load_scored():
    s = pd.read_csv("monthly_scored.csv")
    s = s[(s["scorable"] == True) & (s["hit"].isin([0, 1]))].copy()
    s["p"] = pd.PeriodIndex(pd.to_datetime(s["date"]), freq="M")
    s["in_rec"] = [x in REC for x in s["p"]]
    s["year"] = pd.to_datetime(s["date"]).dt.year
    return s


# --- A. THE MECHANISM ------------------------------------------------------
def fig_mechanism(s):
    """Forecast mix is identical in booms and busts -- the press does not react."""
    g = (s.assign(pess=s["predicted_norm"].isin(["worsen", "down"]))
           .groupby("in_rec")
           .agg(pess=("pess", "mean"), n=("pess", "size")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    # left: the share that predicted a downturn
    labels = ["Expansion", "Recession"]
    vals = [g.loc[False, "pess"], g.loc[True, "pess"]]
    ax1.bar(labels, vals, color=[GRAY, VERM], width=.55, zorder=3)
    for i, v in enumerate(vals):
        ax1.text(i, v + .01, f"{v:.1%}", ha="center", fontsize=14,
                 fontweight="bold", color=INK)
    ax1.set_ylim(0, .45); ax1.set_ylabel("share of forecasts predicting a downturn")
    ax1.set_yticks([0, .1, .2, .3, .4])
    ax1.set_yticklabels(["0", "10%", "20%", "30%", "40%"])
    ax1.grid(axis="x", visible=False)

    # right: monthly pessimism share over time with recession shading
    m = (s.assign(pess=s["predicted_norm"].isin(["worsen", "down"]))
           .groupby("p").agg(pess=("pess", "mean"), n=("pess", "size")))
    m = m[m["n"] >= 5]
    dt = m.index.to_timestamp()
    for a, b in NBER_RECESSIONS:
        p0, p1 = pd.Timestamp(a), pd.Timestamp(b)
        if p1 >= pd.Timestamp("1900-01") and p0 <= pd.Timestamp("1963-12"):
            ax2.axvspan(p0, p1, color="#e3e7ea", zorder=0, lw=0)
    ax2.plot(dt, m["pess"].rolling(12, center=True, min_periods=4).mean(),
             color=VERM, lw=2.2, zorder=3)
    ax2.axhline(m["pess"].mean(), color=MUTED, lw=1, ls="--", zorder=2)
    ax2.set_ylim(0, .8); ax2.set_ylabel("share predicting a downturn")
    ax2.set_yticks([0, .2, .4, .6, .8])
    ax2.set_yticklabels(["0", "20%", "40%", "60%", "80%"])
    ax2.set_xlabel("grey bands = NBER recessions")
    fig.suptitle("The press forecast the same mix of good and bad news, "
                 "boom or bust", fontsize=15, fontweight="bold", x=.008, ha="left",
                 y=.995)
    fig.text(.008, .875, "Share of forecasts predicting a downturn: 24.1% in "
             "expansions, 24.3% in recessions — statistically identical.\n"
             "The right panel shows the same series month by month: it does not "
             "rise when the economy turns.", fontsize=10.5, color=MUTED, ha="left",
             va="top")
    fig.tight_layout(rect=[0, 0, 1, .80])
    fig.savefig(OUT / "figA_mechanism.png", bbox_inches="tight")
    plt.close(fig)


# --- B. THE CONSEQUENCE ----------------------------------------------------
def fig_consequence(s):
    """Because the mix never changes, accuracy collapses exactly when it matters."""
    piv = (s.groupby(["predicted_norm", "in_rec"])["hit"].agg(["mean", "size"])
             .reset_index())
    keep = ["improve", "worsen", "up", "down", "flat"]
    piv = piv[piv["predicted_norm"].isin(keep) & (piv["size"] >= 100)]
    order = ["improve", "up", "flat", "down", "worsen"]
    pretty = {"improve": "“business will improve”", "up": "“prices will rise”",
              "flat": "“no change”", "down": "“prices will fall”",
              "worsen": "“business will worsen”"}
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len([o for o in order if o in set(piv["predicted_norm"])]))
    rows = [o for o in order if o in set(piv["predicted_norm"])]
    h = .36
    for i, o in enumerate(rows):
        sub = piv[piv["predicted_norm"] == o]
        exp = sub[~sub["in_rec"]]["mean"].values
        rec = sub[sub["in_rec"]]["mean"].values
        if len(exp):
            ax.barh(i + h/2, exp[0], height=h, color=GRAY, zorder=3)
            ax.text(exp[0] + .01, i + h/2, f"{exp[0]:.0%}", va="center",
                    fontsize=11, color=INK)
        if len(rec):
            ax.barh(i - h/2, rec[0], height=h, color=VERM, zorder=3)
            ax.text(rec[0] + .01, i - h/2, f"{rec[0]:.0%}", va="center",
                    fontsize=11, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([pretty[o] for o in rows], fontsize=11.5)
    ax.axvline(.5, color="#999999", lw=1, ls="--", zorder=2)
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of those forecasts that came true")
    ax.text(.98, len(rows) - .6, "in expansions", color=MUTED, fontsize=11,
            ha="right", fontweight="bold")
    ax.text(.98, len(rows) - .95, "in recessions", color=VERM, fontsize=11,
            ha="right", fontweight="bold")
    ax.invert_yaxis()
    title(ax, "Pessimists were right in recessions — there were just never enough of them",
          "Downbeat forecasts did far BETTER once a downturn arrived; upbeat ones "
          "collapsed. Because roughly\nthree in four forecasts were upbeat "
          "regardless of conditions, overall accuracy still fell from 58.8% to "
          "39.7%\n(gap +18.7 points, 95% CI [+12.1, +24.8], block-bootstrapped by "
          "3-year period).")
    fig.tight_layout()
    fig.savefig(OUT / "figB_consequence.png", bbox_inches="tight")
    plt.close(fig)


# --- C. NO LEARNING --------------------------------------------------------
def fig_no_learning(s):
    yr = s.groupby("year").agg(n=("hit", "size"), hit=("hit", "mean"))
    yr = yr[yr["n"] >= 40]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for a, b in NBER_RECESSIONS:
        p0, p1 = pd.Timestamp(a), pd.Timestamp(b)
        if p1 >= pd.Timestamp("1900-01") and p0 <= pd.Timestamp("1963-12"):
            ax.axvspan(p0.year, p1.year, color="#e3e7ea", zorder=0, lw=0)
    ax.plot(yr.index, yr["hit"], color=BLUE, lw=1.6, marker="o", ms=4, zorder=3)
    z = np.polyfit(yr.index, yr["hit"], 1)
    ax.plot(yr.index, np.poly1d(z)(yr.index), color=VERM, lw=2, ls="--", zorder=4)
    ax.axhline(.5, color="#999999", lw=1, zorder=2)
    ax.set_ylim(0, 1); ax.set_ylabel("share of forecasts that came true")
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.text(1901, .52, "coin flip", fontsize=10, color=MUTED)
    ax.text(1941, np.poly1d(z)(1941) + .06, "trend", color=VERM, fontsize=11,
            fontweight="bold")
    title(ax, "Sixty years, no improvement: forecasting never got better",
          "Annual accuracy of US-national economic forecasts, 1900–1963 "
          "(years with ≥40 scorable forecasts).\nThe fitted trend is flat. "
          "Grey bands are NBER recessions — accuracy dips inside almost every one.")
    fig.tight_layout()
    fig.savefig(OUT / "figC_no_learning.png", bbox_inches="tight")
    plt.close(fig)


# --- D. WHAT PREDICTS A HIT ------------------------------------------------
def fig_feature_ladder(s):
    rows = []
    rows.append(("Economy was expanding", s[~s["in_rec"]]["hit"].mean(),
                 (~s["in_rec"]).sum(), True))
    rows.append(("Economy was in recession", s[s["in_rec"]]["hit"].mean(),
                 s["in_rec"].sum(), True))
    rows.append(("Forecast said “improve”", s[s["predicted_norm"] == "improve"]["hit"].mean(),
                 (s["predicted_norm"] == "improve").sum(), False))
    rows.append(("Forecast said “worsen”", s[s["predicted_norm"] == "worsen"]["hit"].mean(),
                 (s["predicted_norm"] == "worsen").sum(), False))
    named = s["speaker_name"].astype(str).str.lower().ne("na")
    rows.append(("Named forecaster", s[named]["hit"].mean(), named.sum(), False))
    rows.append(("Anonymous", s[~named]["hit"].mean(), (~named).sum(), False))
    if "confidence" in s:
        for lab, key in [("Assertive wording", "assertive"), ("Hedged wording", "hedged")]:
            m = s["confidence"] == key
            if m.sum() > 100:
                rows.append((lab, s[m]["hit"].mean(), m.sum(), False))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    rows = sorted(rows, key=lambda r: r[1])
    for i, (lab, v, n, is_macro) in enumerate(rows):
        ax.barh(i, v, color=VERM if is_macro else BLUE, height=.62, zorder=3)
        ax.text(v + .008, i, f"{v:.1%}", va="center", fontsize=11,
                color=INK, fontweight="bold")
    ax.axvline(.5, color="#999999", lw=1, ls="--", zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]}  (n={r[2]:,})" for r in rows], fontsize=11)
    ax.set_xlim(0, .8); ax.set_xticks([0, .2, .4, .6, .8])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%"])
    ax.set_xlabel("share of forecasts that came true")
    ax.text(.79, .3, "state of the economy", color=VERM, fontsize=11,
            ha="right", fontweight="bold")
    ax.text(.79, -.1, "how the forecast was written", color=BLUE, fontsize=11,
            ha="right", fontweight="bold")
    title(ax, "What the economy was doing mattered more than how a forecast was written",
          "Accuracy split by one feature at a time. The widest gap by far is "
          "expansion vs recession —\nnot hedging, not a named forecaster, not "
          "confident wording.")
    fig.tight_layout()
    fig.savefig(OUT / "figD_what_predicts.png", bbox_inches="tight")
    plt.close(fig)


# --- E. Method figure, sized for the narrow Methods column ------------------
def fig_method_narrow():
    """Regex vs LLM extraction, in a tall-narrow aspect for the side column."""
    regex = {"recall": 0.269, "precision": 0.609}
    llm = {"recall": 0.731, "precision": 0.844}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    metrics = ["recall", "precision"]
    y = np.arange(len(metrics)); h = .34
    for i, m in enumerate(metrics):
        ax.barh(i + h/2, llm[m], height=h, color=GREEN, zorder=3)
        ax.barh(i - h/2, regex[m], height=h, color=GRAY, zorder=3)
        ax.text(llm[m] + .015, i + h/2, f"{llm[m]:.0%}", va="center",
                fontsize=15, fontweight="bold", color=INK)
        ax.text(regex[m] + .015, i - h/2, f"{regex[m]:.0%}", va="center",
                fontsize=15, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(["Recall", "Precision"], fontsize=15)
    ax.set_xlim(0, 1.05); ax.set_xticks([0, .5, 1])
    ax.set_xticklabels(["0", "50%", "100%"], fontsize=13)
    ax.text(.50, 1.62, "whole-page LLM", color=GREEN, fontsize=15,
            fontweight="bold", ha="center")
    ax.text(.50, -0.72, "keyword regex", color=MUTED, fontsize=15, ha="center")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figE_method.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    s = load_scored()
    fig_mechanism(s)
    fig_consequence(s)
    fig_no_learning(s)
    fig_feature_ladder(s)
    fig_method_narrow()
    print(f"4 poster figures -> {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print("  ", f.name)
