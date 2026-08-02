"""
Poster figures -- the final analysis figures for the monthly corpus.

Each figure answers one question and is designed to be read at poster distance:
big marks, direct labels, one idea per panel. Palette is Okabe-Ito (CVD-safe,
validated with the dataviz skill's checker).

Figures carry NO titles or subtitles -- the headline and caption for each panel
live in the poster deck, so the wording can be edited there without re-running
this script. Numbers that used to sit in a subtitle are printed to stdout.

  fig_A  the mechanism: forecast mix doesn't respond to the economy
         (two files: A1 boom-vs-bust bars, A2 the same series over time)
  fig_B  the consequence: accuracy collapses in recessions
  fig_C  no learning across six decades
  fig_D  what predicts a hit (feature ladder)
  fig_L  which papers were most accurate -- and why the ranking is mostly era

Usage: python make_poster_figures.py
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from truth_data import NBER_RECESSIONS

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9AA0A6"
RED = "#C0392B"          # figQ2_ladder_l1's highlight red, for the one bar that
                         # is singled out by subject rather than by value
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("figures/poster_figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})

REC = set()
for a, b in NBER_RECESSIONS:
    REC.update(pd.period_range(a, b, freq="M"))


def load_scored():
    s = pd.read_csv("data/scored/monthly_scored.csv")
    s = s[(s["scorable"] == True) & (s["hit"].isin([0, 1]))].copy()
    s["p"] = pd.PeriodIndex(pd.to_datetime(s["date"]), freq="M")
    s["in_rec"] = [x in REC for x in s["p"]]
    s["year"] = pd.to_datetime(s["date"]).dt.year
    return s


def gap_ci(s, reps=2000, seed=0):
    """Expansion-minus-recession accuracy gap with a block bootstrap.

    Forecasts cluster hard within an era, so resampling individual claims would
    understate the interval. Resample whole 3-year blocks instead -- the same
    grouping the models use for cross-validation. Returned as (exp, rec, lo, hi)
    with the interval in PERCENTAGE POINTS."""
    blk = (s["year"] - 1900) // 3
    blocks = blk.unique()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.choice(blocks, len(blocks), replace=True)
        bs = pd.concat([s[blk == b] for b in pick])
        if bs["in_rec"].nunique() < 2:
            continue
        out.append(bs[~bs["in_rec"]]["hit"].mean() - bs[bs["in_rec"]]["hit"].mean())
    lo, hi = np.percentile(out, [2.5, 97.5]) * 100
    return (s[~s["in_rec"]]["hit"].mean(), s[s["in_rec"]]["hit"].mean(), lo, hi)


# --- A. THE MECHANISM ------------------------------------------------------
def fig_mechanism(s):
    """Forecast mix is identical in booms and busts -- the press does not react.

    Two files. A1 is the boom-vs-bust comparison, A2 is the same quantity month
    by month; they were one figure until the poster wanted them placed apart."""
    pess = s["predicted_norm"].isin(["worsen", "down"])
    g = (s.assign(pess=pess).groupby("in_rec")
          .agg(pess=("pess", "mean"), n=("pess", "size")))

    # A1: the share that predicted a downturn, expansion vs recession
    fig, ax1 = plt.subplots(figsize=(6, 4.6))
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
    fig.tight_layout()
    fig.savefig(OUT / "figA1_mechanism_bars.png", bbox_inches="tight")
    plt.close(fig)

    # A2: monthly pessimism share over time with recession shading
    m = (s.assign(pess=pess).groupby("p")
          .agg(pess=("pess", "mean"), n=("pess", "size")))
    m = m[m["n"] >= 5]
    dt = m.index.to_timestamp()
    fig, ax2 = plt.subplots(figsize=(8, 4.6))
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
    fig.tight_layout()
    fig.savefig(OUT / "figA2_mechanism_timeline.png", bbox_inches="tight")
    plt.close(fig)

    print(f"   figA: pessimistic share {g.loc[False, 'pess']:.1%} in expansions, "
          f"{g.loc[True, 'pess']:.1%} in recessions")


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
    fig.tight_layout()
    fig.savefig(OUT / "figB_consequence.png", bbox_inches="tight")
    plt.close(fig)
    # The caption numbers now live in the poster deck, so print them here.
    e, r, lo, hi = gap_ci(s)
    print(f"   figB: accuracy {e:.1%} in expansions vs {r:.1%} in recessions "
          f"(gap {e - r:+.1%} pts, 95% CI [{lo:+.1f}, {hi:+.1f}] pts)")


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


# --- H. CORPUS SIZE BY DECADE ----------------------------------------------
def fig_corpus_by_decade():
    """How much data the whole study rests on, decade by decade.

    Reads the raw scored file rather than load_scored(), because the point of
    this panel is the unscorable share -- the forecasts that had no resolvable
    direction or horizon -- which load_scored() drops."""
    d = pd.read_csv("data/scored/monthly_scored.csv")
    d["decade"] = pd.to_datetime(d["date"]).dt.year // 10 * 10
    d["scored"] = d["scorable"] == True
    g = d.groupby("decade").agg(n=("scored", "size"), scorable=("scored", "sum"))

    # bars live left of BAR_END; the two number columns sit right of it
    BAR_END, C_TOT, C_SC = 7000, 8100, 9500
    fig, ax = plt.subplots(figsize=(10, 4.4))
    y = np.arange(len(g))
    ax.barh(y, g["scorable"], height=.62, color=BLUE, zorder=3)
    ax.barh(y, g["n"] - g["scorable"], left=g["scorable"] + 60, height=.62,
            color=GRAY, zorder=3)
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(C_TOT, i, f"{int(r['n']):,}", va="center", fontsize=13.5,
                ha="right", fontweight="bold", color=INK)
        ax.text(C_SC, i, f"{int(r['scorable']):,}", va="center", fontsize=13.5,
                ha="right", color=BLUE, fontweight="bold")
    for x, lab, col in [(C_TOT, "extracted", MUTED), (C_SC, "scorable", BLUE)]:
        ax.text(x, -.9, lab, fontsize=11.5, color=col, ha="right", va="center",
                fontweight="bold")
    labels = [f"{int(x)}s" for x in g.index]
    labels[-1] += "\n(to 1963)"
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=13)
    ax.set_ylim(len(g) - .4, -1.4)
    ax.set_xlim(0, C_SC + 200)
    ax.set_xticks([0, 2000, 4000, 6000])
    ax.set_xticklabels(["0", "2,000", "4,000", "6,000"], fontsize=11)
    ax.spines["bottom"].set_bounds(0, BAR_END)
    ax.grid(axis="y", visible=False)
    for gl in ax.get_xgridlines():
        gl.set_visible(gl.get_xdata()[0] <= BAR_END)
    ax.text(g["scorable"].iloc[0] / 2, -.9, "scorable", color=BLUE,
            fontsize=11.5, fontweight="bold", ha="center", va="center")
    ax.text((g["n"].iloc[0] + g["scorable"].iloc[0]) / 2, -.9,
            "no resolvable direction or horizon", color=MUTED, fontsize=11.5,
            ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "figH_corpus_by_decade.png", bbox_inches="tight")
    plt.close(fig)


# --- F. ACCURACY BY TOPIC --------------------------------------------------
PRETTY_TOPIC = {"general_business": "General business", "markets": "Stock markets",
                "prices": "Prices / inflation", "employment": "Jobs / unemployment",
                "other": "Other"}


def fig_topics(s):
    """Accuracy varies far more by SUBJECT than by how a forecast was written.

    Two files. F1 is the hit rate per topic against the coin-flip line. F2 is
    the reason the topic effect does not generalise -- price accuracy is the
    inflation REGIME, not skill. "Prices up" and "prices down" hit rates sum to
    roughly 1 in every decade, which is what a zero-sum regime effect looks
    like."""
    g = (s[s["topic"].isin(PRETTY_TOPIC)]
         .groupby("topic")["hit"].agg(["size", "mean"]))
    g = g[g["size"] >= 150].sort_values("mean")

    # F1: accuracy by topic
    fig, ax1 = plt.subplots(figsize=(7.5, 6.45))
    # Colour marks WHICH topic, not whether it clears the coin flip -- the
    # dashed line at .5 already carries that, and general business is the topic
    # the poster argues about.
    colors = [RED if t == "general_business" else BLUE for t in g.index]
    ax1.barh(range(len(g)), g["mean"], color=colors, height=.62, zorder=3)
    for i, (v, n) in enumerate(zip(g["mean"], g["size"])):
        ax1.text(v + .012, i, f"{v:.1%}", va="center", fontsize=13,
                 fontweight="bold", color=INK)
    ax1.axvline(.5, color="#999999", lw=1.4, ls="--", zorder=4)
    ax1.set_yticks(range(len(g)))
    ax1.set_yticklabels([f"{PRETTY_TOPIC[t]}\n(n={n:,})"
                         for t, n in zip(g.index, g["size"])], fontsize=12)
    ax1.set_xlim(0, .72); ax1.set_xticks([0, .25, .5])
    ax1.set_xticklabels(["0", "25%", "50%"], fontsize=12)
    ax1.set_xlabel("share of forecasts that came true", fontsize=12)
    ax1.text(.5, len(g) - .35, "coin flip", fontsize=11, color=MUTED,
             ha="center", va="bottom")
    ax1.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figF1_topic_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    # F2: price forecasts by decade -- the regime effect
    fig, ax2 = plt.subplots(figsize=(6.5, 4.3))
    pr = s[(s["topic"] == "prices") & (s["predicted_norm"].isin(["up", "down"]))].copy()
    pr["dec"] = (pr["year"] // 10) * 10
    piv = pr.groupby(["dec", "predicted_norm"])["hit"].agg(["mean", "size"])
    decs = sorted(pr["dec"].unique()); w = .38
    for j, (key, col, lab) in enumerate([("up", VERM, "“prices will rise”"),
                                         ("down", BLUE, "“prices will fall”")]):
        vals = [piv.loc[(d, key), "mean"] if (d, key) in piv.index else np.nan
                for d in decs]
        xs = np.arange(len(decs)) + (j - .5) * w
        ax2.bar(xs, vals, width=w, color=col, zorder=3, label=lab)
        # A hit rate of exactly 0 draws no bar and reads as missing data -- but
        # "prices will fall" being right ZERO times after 1948 is the finding,
        # so label those explicitly.
        for xi, v in zip(xs, vals):
            if v == v and v < .02:
                ax2.text(xi, .02, "0%", ha="center", va="bottom", fontsize=10.5,
                         fontweight="bold", color=col)
    ax2.axhline(.5, color="#999999", lw=1.2, ls="--", zorder=2)
    ax2.set_xticks(range(len(decs)))
    ax2.set_xticklabels([f"{d}s" for d in decs], fontsize=12)
    ax2.set_ylim(0, 1); ax2.set_yticks([0, .5, 1])
    ax2.set_yticklabels(["0", "50%", "100%"], fontsize=12)
    ax2.set_ylabel("share that came true", fontsize=12)
    ax2.legend(fontsize=11, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.13))
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figF2_prices_by_decade.png", bbox_inches="tight")
    plt.close(fig)


# --- G. OCTOBER 1929 -------------------------------------------------------
def fig_1929(s):
    """The press stayed bullish straight through the Great Crash.

    Uses ALL national directional claims (not just the scorable ones) so the
    share is a clean picture of what readers were being told that month."""
    raw = pd.read_csv("data/scored/monthly_scored.csv")
    raw["p"] = pd.PeriodIndex(pd.to_datetime(raw["date"]), freq="M")
    nat = raw[(raw["scope"] == "national")
              & (raw["direction"].isin(["improve", "worsen"]))].copy()
    lo, hi = pd.Period("1929-01", freq="M"), pd.Period("1930-06", freq="M")
    w = nat[(nat["p"] >= lo) & (nat["p"] <= hi)].copy()
    w["imp"] = w["direction"].eq("improve").astype(int)
    m = w.groupby("p")["imp"].agg(["size", "mean"])

    band = s[(s["p"] >= pd.Period("1929-08", freq="M"))
             & (s["p"] <= pd.Period("1929-12", freq="M"))]
    hit_band = band["hit"].mean()

    fig, ax = plt.subplots(figsize=(13, 4.4))
    x = np.arange(len(m))
    crash = list(m.index).index(pd.Period("1929-10", freq="M"))
    cols = [VERM if i == crash else GRAY for i in range(len(m))]
    ax.bar(x, m["mean"], color=cols, width=.68, zorder=3)
    ax.axhline(.5, color="#999999", lw=1.2, ls="--", zorder=2)
    ax.text(crash, m["mean"].iloc[crash] + .035, f"{m['mean'].iloc[crash]:.0%}",
            ha="center", fontsize=14, fontweight="bold", color=VERM)
    ax.annotate("Wall Street crash\n24–29 Oct 1929", xy=(crash, .30),
                xytext=(crash + 1.5, .18), fontsize=11.5, color=VERM,
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=VERM, lw=1.6))
    ax.set_xticks(x)
    ax.set_xticklabels([str(p)[:7] for p in m.index], fontsize=10.5, rotation=45,
                       ha="right")
    ax.set_ylim(0, 1.02); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=12)
    ax.set_ylabel("share predicting improvement", fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "figG_1929.png", bbox_inches="tight")
    plt.close(fig)
    print(f"   figG: Aug–Dec 1929 forecasts came true {hit_band:.1%} of the time "
          f"(n={len(band):,})")


# --- L. WHICH PAPERS WERE MOST ACCURATE ------------------------------------
MIN_CLAIMS = 100   # below this a paper's interval is too wide to rank at all


def _wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def pretty_paper(name):
    """'evening star (washington, d.c.) 1854-1972' -> ('Evening Star',
    'Washington, D.C.'). Chronicling America titles all carry the city and the
    run of years; the years are the paper's whole run, not our sample, so they
    are dropped -- the label carries OUR window instead."""
    m = re.match(r"^(.*?)\s*\((.*?)\)", name)
    if not m:
        return name.title(), ""
    return m.group(1).strip().title(), m.group(2).strip().title()


def publisher_table(s, min_claims=MIN_CLAIMS):
    """Observed hit rate per paper, next to the era it published in.

    Two baselines, for two different jobs:

    `exp_period` -- what the chart shows. The arithmetic mean of the corpus-wide
    hit rate in each 3-year period the paper published in, over the periods it
    actually has claims in (unweighted: every period the paper appears in counts
    once). "How hard were the years this paper was forecasting in?"

    The 3-year grid is not cosmetic: it is what lets the baseline skip years the
    paper was absent for. San Antonio Light spans 1923-1937 but has no claims in
    1927-32, the two hardest stretches in the corpus -- baselined on its whole
    span it scores 47.4%, baselined on the periods it actually appears in, 58.8%.

    `exp_cell` -- the stricter control used by paper_spread_test only. Same idea
    but the bucket is period x predicted direction, so it also absorbs the fact
    that upbeat and downbeat forecasts have opposite hit rates in the same year.
    """
    d = s.copy()
    d["period"] = (d["year"] - 1900) // 3
    d["cell"] = d["period"].astype(str) + "|" + d["predicted_norm"].astype(str)
    d["exp"] = d.groupby("cell")["hit"].transform("mean")

    prate = d.groupby("period")["hit"].mean()
    exp_period = (d.groupby(["publisher", "period"]).size().reset_index()
                   .assign(r=lambda t: t["period"].map(prate))
                   .groupby("publisher")["r"].mean())

    g = d.groupby("publisher").agg(n=("hit", "size"), k=("hit", "sum"),
                                   obs=("hit", "mean"), exp_cell=("exp", "mean"),
                                   y0=("year", "min"), y1=("year", "max"))
    g["exp_period"] = exp_period
    g = g[g["n"] >= min_claims].sort_values("obs", ascending=False)
    g[["lo", "hi"]] = [_wilson(int(k), int(n)) for k, n in zip(g["k"], g["n"])]
    return g, d[d["publisher"].isin(g.index)]


def paper_spread_test(b, reps=1000, seed=0):
    """Do papers differ by more than chance, once era and mix are held fixed?

    Shuffles `hit` WITHIN each (period x direction) cell, so the null keeps every
    paper's era and forecast mix and destroys only the paper label. Returns the
    observed between-paper SD of (observed - expected), the null mean, and p."""
    cell = pd.factorize(b["cell"])[0]
    pub = pd.factorize(b["publisher"])[0]
    hit = b["hit"].values.astype(float)
    dev0 = hit - b["exp"].values
    npub = pub.max() + 1
    cnt = np.bincount(pub, minlength=npub)

    def sd(dev):
        return (np.bincount(pub, dev, minlength=npub) / cnt).std()

    base = np.argsort(cell, kind="stable")
    rng = np.random.default_rng(seed)
    null = np.empty(reps)
    for i in range(reps):
        perm = np.lexsort((rng.random(len(cell)), cell))
        hp = np.empty_like(hit)
        hp[base] = hit[perm]
        null[i] = sd(hp - b["exp"].values)
    obs = sd(dev0)
    return obs, null.mean(), float((null >= obs).mean())


def fig_publishers(s):
    """Ranked accuracy by newspaper, against the era each one published in.

    Papers are ordered by raw hit rate because that is the question people ask
    ("which paper was best?"). The grey diamond is the reply: it is the average
    hit rate of the 3-year periods that paper published in, so a bar standing
    well clear of its own diamond is the only thing that looks like a house
    effect."""
    g, b = publisher_table(s)
    fig, ax = plt.subplots(figsize=(15, 7.6))
    x = np.arange(len(g))
    TOP = .88

    ax.bar(x, g["obs"], width=.66, color=BLUE, zorder=3)
    ax.errorbar(x, g["obs"], yerr=[g["obs"] - g["lo"], g["hi"] - g["obs"]],
                fmt="none", ecolor="#5a5a5a", elinewidth=1.4, capsize=4,
                zorder=5)
    ax.plot(x, g["exp_period"], marker="D", ms=9, ls="none", color=GRAY,
            mec="white", mew=1.8, zorder=6)
    for i, (_, r) in enumerate(g.iterrows()):
        # Clear whichever is higher, the whisker or the diamond -- on the weakest
        # papers the diamond sits above the interval.
        ax.text(i, max(r["hi"], r["exp_period"]) + .018, f"{r['obs']:.0%}",
                ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                color=INK)
        # n rides inside the bar so it never competes with the value label
        ax.text(i, .022, f"n={int(r['n']):,}", ha="center", va="bottom",
                rotation=90, fontsize=9.5, color="white", zorder=4)
    ax.axhline(.5, color="#555555", lw=1.2, ls="--", zorder=7)

    labels = []
    for name, r in g.iterrows():
        t, _ = pretty_paper(name)
        labels.append(f"{t}\n{int(r['y0'])}–{int(r['y1'])}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=38, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-.75, len(g) - .25)
    ax.set_ylim(0, TOP)
    ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_yticklabels(["0", "20%", "40%", "60%", "80%"])
    ax.set_ylabel("share of that paper's forecasts that came true")
    ax.grid(axis="x", visible=False)
    # Below the line, where the weakest papers' labels are not.
    ax.text(len(g) - .35, .489, "coin flip", fontsize=10.5, color=MUTED,
            ha="right", va="top")

    # Legend above the tallest whisker, inside the axes.
    bar_key = ax.bar([0], [0], color=BLUE,
                     label="the paper's own hit rate (bar; 95% Wilson interval)")
    # The legend has to say ALL PAPERS -- a reader who thinks the diamond is
    # also this paper's own number cannot read the chart at all.
    dia_key, = ax.plot([], [], marker="D", ms=9, ls="none", color=GRAY,
                       mec="white", mew=1.8,
                       label="all papers' average hit rate, in the years this "
                             "one published")
    ax.legend(handles=[bar_key, dia_key], frameon=False, fontsize=11, ncol=1,
              loc="upper right", handlelength=1.6, borderaxespad=.2)
    fig.tight_layout()
    fig.savefig(OUT / "figL_publishers.png", bbox_inches="tight")
    plt.close(fig)

    obs, null, p = paper_spread_test(b)
    print(f"   figL: {len(g)} papers with >={MIN_CLAIMS} scored forecasts "
          f"({b.shape[0] / len(s):.0%} of the corpus); raw rates "
          f"{g['obs'].min():.1%}–{g['obs'].max():.1%}, field average "
          f"{s['hit'].mean():.1%}")
    print(f"   figL: between-paper SD after removing era and mix {obs:.3f} vs "
          f"{null:.3f} expected by chance (p={p:.3f})")


if __name__ == "__main__":
    s = load_scored()
    fig_mechanism(s)
    fig_consequence(s)
    fig_no_learning(s)
    fig_feature_ladder(s)
    fig_topics(s)
    fig_1929(s)
    fig_method_narrow()
    fig_corpus_by_decade()
    fig_publishers(s)
    print(f"poster figures -> {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print("  ", f.name)
