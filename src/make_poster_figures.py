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


def title(ax, t, sub):
    n = 1 + sub.count("\n")
    ax.set_title(t, fontsize=15, fontweight="bold", loc="left", pad=18 + 15 * n)
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=10.5,
            color=MUTED, va="bottom")


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
    e, r, lo, hi = gap_ci(s)
    title(ax, "Pessimists were right in recessions — there were just never enough of them",
          "Downbeat forecasts did far BETTER once a downturn arrived; upbeat ones "
          "collapsed. Because roughly\nthree in four forecasts were upbeat "
          f"regardless of conditions, overall accuracy still fell from {e:.1%} to "
          f"{r:.1%}\n(gap {e - r:+.1%} points, 95% CI [{lo:+.1f}, {hi:+.1f}], "
          "block-bootstrapped by 3-year period).")
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


# --- F. ACCURACY BY TOPIC --------------------------------------------------
PRETTY_TOPIC = {"general_business": "General business", "markets": "Stock markets",
                "prices": "Prices / inflation", "employment": "Jobs / unemployment",
                "other": "Other"}


def fig_topics(s):
    """Accuracy varies far more by SUBJECT than by how a forecast was written.

    Left: hit rate per topic against the coin-flip line. Right: the reason the
    topic effect does not generalise -- price accuracy is the inflation REGIME,
    not skill. "Prices up" and "prices down" hit rates sum to roughly 1 in every
    decade, which is what a zero-sum regime effect looks like."""
    g = (s[s["topic"].isin(PRETTY_TOPIC)]
         .groupby("topic")["hit"].agg(["size", "mean"]))
    g = g[g["size"] >= 150].sort_values("mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [VERM if v < .5 else BLUE for v in g["mean"]]
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
               bbox_to_anchor=(.5, 1.16))
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Worse than a coin flip on prices, markets and jobs",
                 fontsize=15, fontweight="bold", x=.008, ha="left", y=.995)
    fig.text(.008, .905,
             "Left: the only subject the press beat chance on was vague talk about "
             "“business” in general.\nRight: why that edge does not last — price "
             "accuracy tracks the era’s inflation regime, not skill. The two bars "
             "sum to about 100% in\nevery decade. Out of sample, topic predicts "
             "nothing (AUC 0.495, leave-one-block-out).",
             fontsize=10.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0, 1, .845])
    fig.savefig(OUT / "figF_topics.png", bbox_inches="tight")
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
    title(ax, "In the month of the Great Crash, the press was more bullish than in June",
          "Share of US-national forecasts predicting improvement, month by month. "
          "Nothing in the pipeline was told\nthat 1929 mattered. Forecasts printed "
          f"August–December 1929 came true {hit_band:.1%} of the time "
          f"(n={len(band):,}) — four in five were wrong.")
    fig.tight_layout()
    fig.savefig(OUT / "figG_1929.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    s = load_scored()
    fig_mechanism(s)
    fig_consequence(s)
    fig_no_learning(s)
    fig_feature_ladder(s)
    fig_topics(s)
    fig_1929(s)
    fig_method_narrow()
    print(f"poster figures -> {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print("  ", f.name)
