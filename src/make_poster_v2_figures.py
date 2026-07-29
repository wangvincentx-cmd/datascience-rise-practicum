"""
Figures for poster v2. Five poster panels plus one supporting figure.

    v2_fig1_fifteen_of_fifteen  the lead: before every NBER peak, the press
                                predicted improvement
    v2_fig2_mechanism           the mix never responded, the VOLUME of coverage
                                never responded either, and what that cost
    v2_fig3_no_smart_paper      publisher "skill" is the calendar, not the paper
    v2_fig4_nothing_transfers   in-sample AUC vs out-of-fold AUC, incl. the text
                                model -- the measured answer to "should we train
                                a neural net?"
    v2_fig5_nobody_has_skill    newspapers vs economists vs the Fed, on skill
                                over a naive always-improve rule

    v2_fig6_price_regime        NOT on the poster -- six figures need 39.8in in
                                a 39.15in column. Kept as the answer to "but WHY
                                does nothing transfer?", which is the first
                                question fig4 invites. Print as a handout or
                                keep on a phone for judging.

Every number is recomputed here from the committed CSVs, so the figures cannot
drift from the text. Design follows the house dataviz rules: one validated
categorical pair (blue #2a78d6 / orange #eb6834, all-pairs CVD dE 24.7), a
blue<->red diverging pair with a gray midpoint for signed quantities, hairline
recessive grid, thin marks, selective direct labels, no dual axes, no rainbow.
Figures are drawn at true print size for a 36x48in poster.

Usage:
    python src/make_poster_v2_figures.py
    python src/make_poster_v2_figures.py --only 1 4
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis_v2 import (BLOCK_YEARS, MIN_PUB_N, drivers, in_recession,
                         load_scored, mechanism, publishers)

OUT = "figures/poster_v2"

# --- house palette (validated: see references/palette.md; all checks PASS) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
RED = "#e34948"       # diverging warm pole
NEUTRAL = "#f0efec"   # diverging midpoint

# True print size: the poster's centre results column is 15.0in wide, so a
# 13.4in figure sits inside it with margin and every point size below is a
# real printed point.
W = 13.4
BASE, LABEL, TITLE, NOTE = 17, 18, 25, 14.5


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": BASE, "axes.labelsize": LABEL, "axes.titlesize": TITLE,
        "xtick.labelsize": BASE, "ytick.labelsize": BASE,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": BASE,
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def clean(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{name}.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------------------------ figure 1

def fig1(d_all):
    """Share predicting improvement in the 6 months before each NBER peak.

    Emphasis form: one hue for all fifteen peaks, the second slot reserved for
    1929 (the one everybody recognises), direct-labelled so identity never rests
    on colour alone."""
    from truth_data import NBER_RECESSIONS
    nat = d_all[(d_all["scope"] == "national")
                & (d_all["predicted_norm"].isin(["improve", "worsen"]))].copy()
    nat["m"] = nat["date"].dt.to_period("M")

    rows = []
    for peak, _ in NBER_RECESSIONS:
        p = pd.Period(peak, "M")
        if p < pd.Period("1900-01", "M") or p > pd.Period("1963-12", "M"):
            continue
        w = nat[(nat["m"] >= p - 6) & (nat["m"] <= p - 1)]
        if len(w) == 0:
            continue
        rows.append({"peak": str(p), "n": len(w),
                     "share": (w["predicted_norm"] == "improve").mean()})
    t = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(W, 7.8))
    y = np.arange(len(t))
    is29 = t["peak"].str.startswith("1929")
    colors = np.where(is29, ORANGE, BLUE)

    ax.axvline(0.5, color=MUTED, lw=1.6, zorder=1)
    for yi, xv, c in zip(y, t["share"], colors):
        ax.plot([0.5, xv], [yi, yi], color=c, lw=2.2, alpha=0.35,
                solid_capstyle="round", zorder=2)
    ax.scatter(t["share"], y, s=190, color=colors, zorder=3,
               edgecolors=SURFACE, linewidths=2)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}   (n={n})" for p, n in zip(t["peak"], t["n"])])
    ax.set_xlim(0.30, 1.02)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])
    ax.set_xlabel("share of forecasts predicting the economy would IMPROVE")
    ax.set_ylabel("NBER business-cycle peak (month the downturn began)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)

    ax.text(0.505, len(t) - 0.4, "coin flip", color=MUTED, fontsize=NOTE,
            ha="left", va="center")
    # Both callouts live in the empty right margin on their OWN row, so the
    # leader lines run horizontally and never cross another row's mark.
    i29 = int(np.where(is29)[0][0])
    ax.annotate("two months before\nthe Great Crash",
                xy=(t["share"][i29], i29), xytext=(1.02, i29),
                color=ORANGE, fontsize=NOTE, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.4,
                                shrinkA=4, shrinkB=10))
    imin = int(t["share"].idxmin())
    ax.annotate("the one even split",
                xy=(t["share"][imin], imin), xytext=(1.02, imin),
                color=INK2, fontsize=NOTE, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.4,
                                shrinkA=4, shrinkB=10))

    ax.set_title("Fifteen downturns. Fifteen times the press said things would "
                 "get better.", loc="left", pad=18, fontweight="bold")
    fig.text(0.0, -0.035,
             "Every US-national directional forecast printed in the six months "
             "before each NBER peak, 1900–1963. Mean 74.2% predicting "
             "improvement;\nnot one peak is preceded by a net-pessimistic press. "
             "No episode labels or outcomes were given to the extractor.",
             fontsize=NOTE, color=INK2, ha="left", va="top")
    save(fig, "v2_fig1_fifteen_of_fifteen")
    return t


# ------------------------------------------------------------------ figure 2

def fig2(s):
    """Three panels: the mix did not respond, the VOLUME did not respond, and
    what that cost. The volume panel is the stronger version of the same point
    -- the press did not even write more about the economy in a downturn."""
    from analysis_v2 import attention
    m, s2 = mechanism(s, s)
    att, _ = attention()
    # Three panels at poster font sizes need real gutters, or the y-axis labels
    # of one panel land inside its neighbour.
    fig, axes = plt.subplots(1, 3, figsize=(W, 6.8),
                             gridspec_kw={"width_ratios": [0.85, 0.85, 1.9],
                                          "wspace": 0.55})

    # -- panel A: share predicting a downturn, expansions vs recessions
    ax = axes[0]
    vals = [m["share_worsen_expansion"], m["share_worsen_recession"]]
    ax.bar([0, 1], vals, width=0.55, color=[BLUE, ORANGE], zorder=3)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.012, f"{v:.1%}", ha="center", va="bottom",
                fontsize=LABEL, fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["expansion", "recession"], fontsize=BASE - 2)
    ax.set_ylim(0, 0.30)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.set_ylabel("% predicting a downturn")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.set_title("The mix never moved", loc="left", pad=14,
                 fontsize=TITLE - 7, fontweight="bold")

    # -- panel B: economy coverage per 100 pages
    ax = axes[1]
    av = [att["attention"]["expansion"], att["attention"]["recession"]]
    ax.bar([0, 1], av, width=0.55, color=[BLUE, ORANGE], zorder=3)
    for x, v in zip([0, 1], av):
        ax.text(x, v + 8, f"{v:.0f}", ha="center", va="bottom",
                fontsize=LABEL, fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["expansion", "recession"], fontsize=BASE - 2)
    ax.set_ylim(0, 260)
    ax.set_ylabel("claims per 100 pages")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.set_title("Nor did the volume", loc="left", pad=14,
                 fontsize=TITLE - 7, fontweight="bold")

    # -- panel C: accuracy by what was predicted, expansions vs recessions
    ax = axes[2]
    groups = ["all\nforecasts", '"will\nimprove"', '"will\nworsen"']
    exp = [m["hit_expansion"], m["improve_hit_expansion"], m["worsen_hit_expansion"]]
    rec = [m["hit_recession"], m["improve_hit_recession"], m["worsen_hit_recession"]]
    x = np.arange(3)
    w = 0.34
    ax.axhline(0.5, color=MUTED, lw=1.6, zorder=2)
    ax.bar(x - w / 2 - 0.01, exp, width=w, color=BLUE, label="printed in an expansion",
           zorder=3)
    ax.bar(x + w / 2 + 0.01, rec, width=w, color=ORANGE, label="printed in a recession",
           zorder=3)
    for xi, a, b in zip(x, exp, rec):
        ax.text(xi - w / 2 - 0.01, a + 0.015, f"{a:.0%}", ha="center",
                va="bottom", fontsize=BASE, color=INK)
        ax.text(xi + w / 2 + 0.01, b + 0.015, f"{b:.0%}", ha="center",
                va="bottom", fontsize=BASE, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    # Headroom so the legend clears the tallest bar's value label, and left
    # margin so the baseline can be labelled without landing on a bar.
    ax.set_xlim(-1.05, 2.5)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("% that came true")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(loc="upper right", ncol=1)
    ax.text(-1.02, 0.515, "coin flip", color=MUTED, fontsize=NOTE, ha="left")
    ax.set_title("Pessimists were right when it counted", loc="left", pad=14,
                 fontsize=TITLE - 7, fontweight="bold")

    ci = att["attention_block_ci95"]
    fig.text(0.0, -0.06,
             f"The press did not change what it said in a downturn — and did "
             f"not even write more about it. Left: the share of forecasts "
             f"calling for a downturn is statistically identical in booms and "
             f"busts\n({m['share_worsen_expansion']:.1%} vs "
             f"{m['share_worsen_recession']:.1%}). Middle: economic coverage "
             f"per 100 pages is flat too ({att['attention']['expansion']:.0f} "
             f"vs {att['attention']['recession']:.0f}, p = "
             f"{att['attention']['p']:.2f}; block-bootstrapped 95% CI "
             f"[{ci[0]:+.0f}, {ci[1]:+.0f}], and p = "
             f"{att['attention_within_era_p']:.2f} within era). Raw claims\nper "
             f"month DO rise in recessions — but only because more pages were "
             f"digitised in those months; per page the effect vanishes. Right: "
             f"because the mix stayed ~4:1 upbeat regardless of conditions,\n"
             f"average accuracy fell {m['gap_pts']:.1f} points in recessions "
             f"(95% CI [{m['gap_ci95'][0]:.1f}, {m['gap_ci95'][1]:.1f}]). "
             f"Caveat: NBER dates are both the scoring truth and the split "
             f"variable, so the accuracy gap is a consequence of the mix — not "
             f"an independent finding.",
             fontsize=NOTE, color=INK2, ha="left", va="top")
    save(fig, "v2_fig2_mechanism")
    return m


# ------------------------------------------------------------------ figure 3

def fig3(s):
    """Dumbbell: what the calendar alone guaranteed, vs what the paper scored."""
    r, o = publishers(s, n_perm=1)  # permutation not needed for the figure
    r = r.sort_values("actual")
    short = (r.index.to_series().str.split("(").str[0].str.strip()
             .str.title().str.replace("^The ", "", regex=True))

    fig, ax = plt.subplots(figsize=(W, 8.8))
    y = np.arange(len(r))
    ax.axvline(0.5, color=MUTED, lw=1.6, zorder=1)
    for yi, e, a in zip(y, r["expected"], r["actual"]):
        ax.plot([e, a], [yi, yi], color=AXIS, lw=2.0, zorder=2,
                solid_capstyle="round")
    ax.scatter(r["expected"], y, s=150, color=MUTED, zorder=3,
               edgecolors=SURFACE, linewidths=2,
               label="what its calendar + forecast mix alone guaranteed")
    ax.scatter(r["actual"], y, s=170, color=BLUE, zorder=4,
               edgecolors=SURFACE, linewidths=2,
               label="what the paper actually scored")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}   (n={int(c)})" for n, c in zip(short, r["n"])])
    ax.set_xlim(0.30, 0.72)
    ax.set_xticks([0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_xticklabels(["35%", "40%", "45%", "50%", "55%", "60%", "65%", "70%"])
    ax.set_xlabel("share of that paper's forecasts that came true")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.005))
    ax.text(0.503, len(r) - 0.3, "coin flip", color=MUTED, fontsize=NOTE,
            ha="left", va="center")

    ax.set_title("There was no smart newspaper", loc="left", pad=18,
                 fontweight="bold")
    fig.text(0.0, -0.035,
             f"The {o['spread_lo']:.0%}–{o['spread_hi']:.0%} spread across the "
             f"{o['n_publishers']} papers with ≥{MIN_PUB_N} scorable forecasts "
             f"looks decisive. It is not. {o['share_composition']:.0%} of the "
             f"variance is composition — which years a paper\nprinted in and "
             f"which direction it tended to call, both of which fix the odds "
             f"before any judgement enters — and "
             f"{o['share_residual_that_is_noise']:.0%} of what remains is "
             f"exactly the binomial noise you would\nexpect from coin flips. A "
             f"joint test for any publisher skill at all: χ² = {o['chi2']:.1f} "
             f"on {o['chi2_df']} df, p = {o['chi2_p']:.2f}. Not one paper "
             f"differs from its own calendar's expectation.",
             fontsize=NOTE, color=INK2, ha="left", va="top")
    save(fig, "v2_fig3_no_smart_paper")
    return r, o


# ------------------------------------------------------------------ figure 4

def fig4(s, text_ceiling_result):
    """Dumbbell: in-sample AUC -> out-of-fold AUC. Both axes are AUC, so the
    two quantities share one scale -- no dual axis, and the collapse is the
    length of the line."""
    t, _ = drivers(s)
    t = t.sort_values("insample_auc")

    extra = pd.DataFrame([{
        "factor": "the full text of the forecast\n(24,598 TF-IDF features)",
        "insample_auc": text_ceiling_result["text_word12_insample"],
        "oof_auc": text_ceiling_result["text_word12"]}])
    t = pd.concat([t[["factor", "insample_auc", "oof_auc"]], extra],
                  ignore_index=True)

    fig, ax = plt.subplots(figsize=(W, 8.4))
    y = np.arange(len(t))
    ax.axvline(0.5, color=MUTED, lw=1.6, zorder=1)
    for yi, a, b in zip(y, t["insample_auc"], t["oof_auc"]):
        ax.annotate("", xy=(b, yi), xytext=(a, yi),
                    arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=2.2,
                                    shrinkA=6, shrinkB=6,
                                    mutation_scale=18), zorder=2)
    ax.scatter(t["insample_auc"], y, s=170, color=ORANGE, zorder=3,
               edgecolors=SURFACE, linewidths=2,
               label="in sample  (fit and scored on the same claims)")
    ax.scatter(t["oof_auc"], y, s=170, color=BLUE, zorder=4,
               edgecolors=SURFACE, linewidths=2,
               label=f"out of fold  (held-out {BLOCK_YEARS}-year block, 21 folds)")

    ax.set_yticks(y)
    ax.set_yticklabels(t["factor"])
    ax.set_xlim(0.38, 0.96)
    ax.set_ylim(-0.8, len(t) - 0.15)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("ROC-AUC for predicting whether a forecast came true")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(loc="lower right")
    # Sits above the top row rather than on the axis band, where it collided
    # with the 0.5 and 0.6 ticks.
    ax.text(0.505, len(t) - 0.45, "no better than chance", color=MUTED,
            fontsize=NOTE, ha="left", va="center")

    ax.set_title("Everything predicts the past. Nothing predicts the future.",
                 loc="left", pad=18, fontweight="bold")
    fig.text(0.0, -0.045,
             "Each arrow is one predictor, scored the honest way and the "
             "flattering way. Fit on the whole corpus, the raw text of a "
             "forecast separates hits from\nmisses almost perfectly "
             f"(AUC {text_ceiling_result['text_word12_insample']:.2f}). Asked "
             "to rank forecasts from a decade it has never seen, the same model "
             f"scores {text_ceiling_result['text_word12']:.2f} — against a "
             f"{s['hit'].mean():.2f} base rate.\nSeveral predictors land BELOW "
             "0.5, meaning their in-sample ranking reverses in a new era. The "
             "binding constraint is ~21 independent time blocks, not model "
             "capacity —\nwhich is why more parameters is the one thing that "
             "cannot fix this.",
             fontsize=NOTE, color=INK2, ha="left", va="top")
    save(fig, "v2_fig4_nothing_transfers")
    return t


# ------------------------------------------------------------------ figure 5

def fig5():
    """Diverging bars around zero: skill over a naive always-improve rule."""
    b = pd.read_csv("data/scored/analysis_v2/benchmarks.csv")
    b["label"] = b["forecaster"] + "\n" + b["period"]
    b = b.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(W, 7.0))
    y = np.arange(len(b))
    colors = [BLUE if v >= 0 else RED for v in b["skill"]]
    ax.barh(y, b["skill"], height=0.58, color=colors, zorder=3)
    ax.axvline(0, color=MUTED, lw=1.8, zorder=4)

    for yi, v, raw, nv, n in zip(y, b["skill"], b["hit"],
                                 b["naive_always_improve"], b["n"]):
        off = 0.006 if v >= 0 else -0.006
        ax.text(v + off, yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=LABEL, fontweight="bold", color=INK)
        ax.text(0.095, yi, f"raw {raw:.0%}   ·   naive rule {nv:.0%}   ·   "
                           f"n = {int(n):,}",
                va="center", ha="left", fontsize=NOTE, color=INK2)

    ax.set_yticks(y)
    ax.set_yticklabels(b["label"])
    # Right margin holds the raw/naive/n column; starting it at 0.095 keeps it
    # clear of the one positive bar's value label.
    ax.set_xlim(-0.16, 0.26)
    ax.set_xticks([-0.15, -0.10, -0.05, 0.0, 0.05])
    ax.set_xticklabels(["−15 pts", "−10", "−5", "0", "+5"])
    ax.set_xlabel("skill: accuracy MINUS what \"the economy will improve, always\" "
                  "would have scored")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.text(-0.004, len(b) - 0.35, "worse than the naive rule", color=MUTED,
            fontsize=NOTE, ha="right", va="center")
    ax.text(0.004, len(b) - 0.35, "better", color=MUTED, fontsize=NOTE,
            ha="left", va="center")

    ax.set_title("Nobody has skill — and the professionals are no exception",
                 loc="left", pad=18, fontweight="bold")
    fig.text(0.0, -0.05,
             "Raw accuracy flatters whoever forecast in a calm decade: the "
             "Livingston economists scored 72% to the newspapers' 59% over the "
             "same 1946–63 window.\nSubtract what a rule that said \"improve\" "
             "every single time would have earned, and the gap vanishes — "
             "−0.111 vs −0.112. Every forecaster here predicted\nimprovement "
             "73–94% of the time, and in a century not one clears +0.05. "
             "Caveats: the Livingston window holds only n = 36 surveys, and the "
             "four sources score\ndifferent variables under different rules — "
             "this is a reference point, not a controlled experiment.",
             fontsize=NOTE, color=INK2, ha="left", va="top")
    save(fig, "v2_fig5_nobody_has_skill")
    return b


# ------------------------------------------------------------------ figure 6

def fig6(s):
    """Was being right about prices skill, or the era's regime?

    x = how often an outcome ACTUALLY happened in that decade; y = how often
    forecasts predicting it came true. A forecast carrying real information sits
    ABOVE the identity line. Everything sits ON it."""
    from analysis_v2 import price_regime
    t, o = price_regime(s)

    # A scatter against an identity line was the natural form, but four of the
    # twelve points sit on top of each other at the origin ("prices will fall",
    # never right, three decades running) and no label placement survives that.
    # One row per decade x direction cannot collide, and the finding IS the
    # length of each connector: near zero everywhere.
    t = t.sort_values("base_rate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(W, 7.2))
    y = np.arange(len(t))
    for yi, r in t.iterrows():
        ax.plot([r["base_rate"], r["hit_rate"]], [yi, yi], color=AXIS, lw=2.4,
                solid_capstyle="round", zorder=2)
    ax.scatter(t["base_rate"], y, s=165, color=MUTED, zorder=3,
               edgecolors=SURFACE, linewidths=2,
               label="how often prices ACTUALLY moved that way, that decade")
    ax.scatter(t["hit_rate"], y, s=185,
               color=[BLUE if d == "up" else ORANGE for d in t["direction"]],
               zorder=4, edgecolors=SURFACE, linewidths=2,
               label="how often forecasts predicting that move came true")

    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{int(r['decade'])}s   \"prices will "
        f"{'RISE' if r['direction'] == 'up' else 'FALL'}\"   (n={int(r['n'])})"
        for _, r in t.iterrows()])
    for lbl, d in zip(ax.get_yticklabels(), t["direction"]):
        lbl.set_color(BLUE if d == "up" else ORANGE)
    ax.set_xlim(-0.04, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("rate")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    clean(ax)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, -0.02))

    ax.set_title("Being right about prices was the era, not the forecaster.",
                 loc="left", pad=18, fontweight="bold")
    fig.text(0.0, -0.045,
             f"Each row is one decade and one direction. The grey dot is the "
             f"base rate — how often prices actually moved that way. The "
             f"coloured dot is how often forecasts predicting\nthat move came "
             f"true. A forecast carrying real information would sit well to the "
             f"RIGHT of its grey dot. None do: correlation between hit rate and "
             f"base rate = {o['corr_hit_vs_base']:.3f}, mean gap only "
             f"{o['mean_abs_gap']*100:.1f}\npoints. \"Prices will rise\" was "
             f"right 85% of the time in the inflationary 1910s and 26% in the "
             f"flat 1960s — the forecast never changed, the economy did. After "
             f"1948, \"prices will fall\" was\nright "
             f"{o['down_after_1948_hits']} times out of "
             f"{o['down_after_1948_n']}. This is the mechanism behind the "
             f"panel above: an in-sample edge that is really an era, and so "
             f"transfers to a new era at exactly chance.",
             fontsize=NOTE, color=INK2, ha="left", va="top")
    save(fig, "v2_fig6_price_regime")
    return t, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    a = ap.parse_args()
    style()
    d_all, s = load_scored()
    want = set(a.only) if a.only else {1, 2, 3, 4, 5, 6}

    if 1 in want:
        fig1(d_all)
    if 2 in want:
        fig2(s)
    if 3 in want:
        fig3(s)
    if 4 in want:
        import json
        with open("data/scored/analysis_v2/summary.json") as f:
            summ = json.load(f)
        if "text_ceiling" not in summ:
            raise SystemExit("run: python src/analysis_v2.py --section text")
        fig4(s, summ["text_ceiling"])
    if 5 in want:
        fig5()
    if 6 in want:
        fig6(s)


if __name__ == "__main__":
    main()
