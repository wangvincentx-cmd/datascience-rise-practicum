"""Chart versions of the two recession-model result tables.

  fig_forward_test.png   what sheets_forward_test.png says: forward-only AUC for
                         the five-press-series model vs attention alone, in three
                         training windows, plus the paired difference and its CI.
  fig_per_period.png     what sheets_per_period.png says: AUC and Brier per decade
                         under the 1930 start, for both models.

Both read the same CSVs the tables read (broad_vs_attention_table.csv,
per_period_table.csv), so a chart and its table can never disagree.

Form: paired dot plots ("dumbbells"), not grouped bars. The quantity being read
off these tables is the GAP between two models measured on the same months, and a
dumbbell puts that gap on the page as a length. Bars would spend most of their ink
on the 0-to-0.5 region that AUC does not use, and broad_vs_attention.png already
covers the bar-with-whiskers view of the first table.

Colour: two categorical hues doing identity work only, validated for CVD
separation (protan ΔE 27.7 / normal ΔE 33.5). Reference lines are gray so no
reader mistakes them for a series.

Run from the repo root:  python more_model_images/make_recession_model_graphs.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent

BLUE, ORANGE = "#1967d2", "#c5490b"
INK, MUTED, FAINT = "#202124", "#5f6368", "#9aa0a6"
GRID, RULE = "#e8eaed", "#dadce0"
BAD = "#b3261e"

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.axisbelow": True})

# Paired difference (attention - five series) with its 95% block bootstrap CI,
# carried over from the table so both renderings quote one computation.
PAIRED = {1930: (-0.019, -0.12, +0.12),
          1940: (-0.001, -0.10, +0.15),
          1950: (-0.065, -0.13, -0.01)}

FIVE, ATTN = "Five press series", "Attention alone"


def dumbbell(ax, y, lo_val, hi_val, lo_col, hi_col, fmt="{:.3f}", pad=0.006,
             label=True):
    """One paired row. Labels go OUTSIDE the pair -- the two dots can sit 0.005
    apart (1940) and inside labels would land on top of each other."""
    ax.plot([lo_val, hi_val], [y, y], color=FAINT, lw=2, zorder=2,
            solid_capstyle="round")
    for v, c in [(lo_val, lo_col), (hi_val, hi_col)]:
        ax.plot([v], [y], "o", ms=11, color=c, mec="white", mew=1.8, zorder=4)
    if label:
        # Ink, not the series colour: each number sits against its own dot, so
        # the dot carries identity and the text stays readable next to the red
        # period labels in fig_per_period.
        ax.text(lo_val - pad, y, fmt.format(lo_val), ha="right", va="center",
                fontsize=9.5, color=INK, fontweight="600", zorder=5)
        ax.text(hi_val + pad, y, fmt.format(hi_val), ha="left", va="center",
                fontsize=9.5, color=INK, fontweight="600", zorder=5)


def pair(a_val, a_col, b_val, b_col):
    """(low, high) with each value's own colour attached."""
    return ((a_val, a_col), (b_val, b_col)) if a_val <= b_val else \
           ((b_val, b_col), (a_val, a_col))


def legend_dots(fig, x, y, items, fontsize=10):
    """Legend drawn in figure space so neither panel has to give up plot area."""
    for i, (color, text) in enumerate(items):
        fig.text(x + i * 0.22, y, "●", color=color, fontsize=13,
                 ha="left", va="center")
        fig.text(x + i * 0.22 + 0.016, y, text, color=INK, fontsize=fontsize,
                 ha="left", va="center")


# --- 1. forward-only backtest ----------------------------------------------
def forward_test():
    d = pd.read_csv(OUT / "broad_vs_attention_table.csv")
    starts = sorted(d["start"].unique())
    ys = np.arange(len(starts))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1.75, 1],
                                                "wspace": 0.30})

    for y, s in zip(ys, starts):
        w = d[d["start"] == s].set_index("model")
        (lo, lo_c), (hi, hi_c) = pair(w.loc[FIVE, "auc"], BLUE,
                                      w.loc[ATTN, "auc"], ORANGE)
        dumbbell(ax1, y, lo, hi, lo_c, hi_c)

    ax1.axvline(.5, color=FAINT, lw=1.3, ls="--", zorder=1)
    ax1.text(.5, -0.72, "coin flip", color=MUTED, fontsize=9, ha="center",
             va="center")
    rows = []
    for s in starts:
        w = d[d["start"] == s].iloc[0]
        rows.append(f"{s}–1963\n{w['months']} months · "
                    f"{w['onsets']} onsets")
    ax1.set_yticks(ys); ax1.set_yticklabels(rows, fontsize=10.5)
    ax1.set_ylim(len(starts) - .35, -1.05)
    ax1.set_xlim(.44, .84)
    ax1.set_xticks([.5, .6, .7, .8])
    ax1.set_xticklabels(["0.50", "0.60", "0.70", "0.80"], fontsize=9.5)
    ax1.set_xlabel("forward-only ROC-AUC   (higher is better)", fontsize=10,
                   color=MUTED)
    ax1.grid(axis="x", color=GRID, lw=.8)
    ax1.tick_params(axis="y", length=0)

    # Right: the comparison the table's last two columns exist to make. A CI
    # crossing zero is the finding for the two longer windows, so it is drawn
    # in gray -- only the 1950 window earns a coloured mark.
    for y, s in zip(ys, starts):
        dlt, lo, hi = PAIRED[s]
        sig = not (lo <= 0 <= hi)
        c = BAD if sig else FAINT
        ax2.plot([lo, hi], [y, y], color=c, lw=2.4, solid_capstyle="round",
                 zorder=3, alpha=1 if sig else .75)
        ax2.plot([dlt], [y], "o", ms=10, color=c, mec="white", mew=1.8, zorder=4)
        # Label on the far side from zero, so it never lands on the zero rule.
        ax2.text(lo - .012, y, f"{dlt:+.3f}", ha="right", va="center",
                 fontsize=9.5, color=BAD if sig else INK, fontweight="600")
    ax2.axvline(0, color=FAINT, lw=1.3, ls="--", zorder=1)
    ax2.text(0, -0.72, "no difference", color=MUTED, fontsize=9, ha="center",
             va="center")
    ax2.set_yticks(ys); ax2.set_yticklabels([])
    ax2.set_ylim(len(starts) - .35, -1.05)
    ax2.set_xlim(-.25, .22)
    ax2.set_xticks([-.15, 0, .15])
    ax2.set_xticklabels(["−0.15", "0", "+0.15"], fontsize=9.5)
    ax2.set_xlabel("attention − five series, 95% CI", fontsize=10,
                   color=MUTED)
    ax2.grid(axis="x", color=GRID, lw=.8)
    ax2.tick_params(axis="y", length=0)

    fig.suptitle("One press feature matches five in the two longer windows",
                 fontsize=13.5, fontweight="600", x=.008, ha="left", y=1.10)
    fig.text(.008, 1.015, "Recession onset within 12 months · model refit "
             "every year on all prior years, never on the year it predicts",
             fontsize=10, color=MUTED, ha="left", va="center")
    legend_dots(fig, .008, .945, [(BLUE, FIVE), (ORANGE, ATTN)])
    fig.text(.008, -.16,
             "95% block bootstrap over 3-year blocks. The 1930 and 1940 windows' "
             "intervals cross zero — the two models are\n"
             "indistinguishable there. Only the shortest window (137 months) "
             "separates them, and it favours the five-series model.\n"
             "Eras won (8-year blocks): 2/4, 1/3, 1/2 for five series — too "
             "few blocks to read as a test.",
             fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.6)

    p = OUT / "fig_forward_test.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=.10)
    plt.close(fig)
    print(f"-> {p}")


# --- 2. per-period performance ---------------------------------------------
def per_period():
    d = pd.read_csv(OUT / "per_period_table.csv")
    # The all-period row is a summary, not a fifth period: it goes below a rule
    # so no one reads it as another independent observation.
    is_tot = d["period"].astype(str).str.contains("all")
    rows = d[~is_tot].to_dict("records") + d[is_tot].to_dict("records")
    n_per = int((~is_tot).sum())
    ys = list(np.arange(n_per)) + [n_per + .55]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5),
                                   gridspec_kw={"wspace": 0.28})

    for y, r in zip(ys, rows):
        (lo, lc), (hi, hc) = pair(r["auc_five"], BLUE, r["auc_attn"], ORANGE)
        dumbbell(ax1, y, lo, hi, lc, hc, pad=.014)
        (blo, blc), (bhi, bhc) = pair(r["brier_five"], BLUE,
                                      r["brier_attn"], ORANGE)
        dumbbell(ax2, y, blo, bhi, blc, bhc, pad=.006)
        # What a model that only knows the base rate would score. Brier without
        # this reference is an unreadable number: 0.21 in the 1960s is WORSE
        # than 0.23 in the 1940s once the onset rate is accounted for.
        base = r["base_rate"]
        ax2.plot([base * (1 - base)], [y], marker="|", ms=13, mew=2.2,
                 color=FAINT, zorder=3)

    ax1.axvline(.5, color=FAINT, lw=1.3, ls="--", zorder=1)
    ax1.text(.5, -0.80, "coin flip", color=MUTED, fontsize=9, ha="center",
             va="center")
    labs = [f"{r['period']}\n{r['months']} months · {r['onsets']} onsets"
            for r in rows]
    for ax in (ax1, ax2):
        ax.set_yticks(ys)
        ax.set_ylim(max(ys) + .45, -1.15)
        ax.grid(axis="x", color=GRID, lw=.8)
        ax.tick_params(axis="y", length=0)
        ax.axhline((n_per - 1 + ys[-1]) / 2, color=RULE, lw=1)
    ax1.set_yticklabels(labs, fontsize=10)
    ax2.set_yticklabels([])
    for t, r in zip(ax1.get_yticklabels(), rows):
        if max(r["auc_attn"], r["auc_five"]) < .5:
            t.set_color(BAD)

    ax1.set_xlim(-.01, .95)
    ax1.set_xticks([.1, .3, .5, .7, .9])
    ax1.set_xticklabels(["0.10", "0.30", "0.50", "0.70", "0.90"], fontsize=9.5)
    ax1.set_xlabel("ROC-AUC   (higher is better)", fontsize=10, color=MUTED)
    ax2.set_xlim(.10, .47)
    ax2.set_xticks([.15, .25, .35, .45])
    ax2.set_xticklabels(["0.15", "0.25", "0.35", "0.45"], fontsize=9.5)
    ax2.set_xlabel("Brier score   (LOWER is better)", fontsize=10, color=MUTED)

    fig.suptitle("Both models rank the 1940s and 1950s — and run backwards "
                 "either side of them",
                 fontsize=13.5, fontweight="600", x=.008, ha="left", y=1.09)
    fig.text(.008, 1.010, "Real-time performance by decade, 1930 start — "
             "the longest forward-only run, so every decade is predicted by a "
             "model that never saw it",
             fontsize=10, color=MUTED, ha="left", va="center")
    legend_dots(fig, .008, .950, [(BLUE, FIVE), (ORANGE, ATTN)])
    # Drawn, not typed: the heavy vertical-bar glyph is missing from Arial and
    # renders as a tofu box.
    fig.add_artist(plt.Line2D([.458, .458], [.938, .962], color=FAINT, lw=2.4))
    fig.text(.468, .950, "base-rate forecast (Brier reference)", color=INK,
             fontsize=10, ha="left", va="center")
    fig.text(.008, -.13,
             "Red labels mark decades where both models rank BACKWARDS "
             "(AUC below 0.50). Each decade rests on 1–3 independent "
             "3-year blocks\n"
             "— descriptive, not tests. Accuracy is deliberately not "
             "reported: at a 24% onset rate, always answering 'no onset' "
             "scores 76% while ranking nothing.",
             fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.6)

    p = OUT / "fig_per_period.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=.10)
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    forward_test()
    per_period()
