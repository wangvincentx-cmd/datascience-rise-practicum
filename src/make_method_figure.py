"""
Two Methods-page figures: what the logistic regression does, and what goes in.

  figM_logistic_explainer.png -- one real forecast becoming numbers, and the
      S-curve turning a score into a probability. The two marked points are the
      REAL output of `hit_predictor.predict_new` (hp.DEMO): same date, same
      city, opposite calls. Their x positions are the log-odds of those
      probabilities, so the points sit on the curve by construction rather than
      by eyeballing.

  figN_feature_table.png -- all 41 input columns in their three blocks, with
      human-readable labels beside the code names, so a reader can match the
      poster to the repository.

Block B's 13 factors and their labels are READ FROM macro_context, and block C's
list from hit_predictor, so the table cannot drift from the model it documents.
Only the block-A labels are written out here -- claim_features builds those
column by column and there is no list to import.

Deliberately NOT results figures -- figK_hit_model.png carries the ladder and
the calibration curve. These two only answer "what is the model, and what did
you feed it?".

Figures carry no titles or captions; the poster deck supplies both.

Usage: python src/make_method_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macro_context import FACTORS, PRETTY
from hit_predictor import INTERACT_WITH

# Okabe-Ito. Only BLUE and VERM carry identity (optimistic / pessimistic, the
# same mapping as the scissors figure); everything else is structural ink, so
# the categorical validation is the two-hue palette, which passes all six checks.
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#9AA0A6"
INK, MUTED, HAIR = "#1a1a1a", "#6b6b6b", "#d8d8d8"
OUT = Path("figures/poster_figures"); OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "DejaVu Sans"})

# The real predict_new output -- see hp.DEMO.
P_BULL, P_BEAR = 0.608, 0.234
logit = lambda p: np.log(p / (1 - p))

BLOCKS = [
    ("what the forecast said",  "direction, topic, hedged or flat,\nquoted expert, length, horizon", "10 numbers"),
    ("the economy that week",   "market up or down, how far off its\npeak, output, prices, uncertainty", "26 numbers"),
    ("the two multiplied",      "an upbeat call INTO a rising market\nis a different bet from into a falling one", "5 numbers"),
]


def chip(ax, x, y, w, h, title, body, count):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                fc="#f7f7f6", ec=HAIR, lw=1.2))
    ax.text(x + 1.4, y + h - 2.6, title, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(x + 1.4, y + h - 6.0, body, fontsize=10.2, color=MUTED, va="top", linespacing=1.35)
    ax.text(x + w - 1.4, y + h - 2.6, count, fontsize=10, color=MUTED, ha="right")


def arrow(ax, xy_from, xy_to, color=GRAY, lw=1.6):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def build():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.0, 5.4), gridspec_kw={"width_ratios": [1.30, 1]})

    # Both step headings placed in FIGURE coordinates so they sit on one line --
    # the left panel has no axes to hang a title on, so matching them by eye
    # across two different coordinate systems does not survive a size change.
    fig.text(0.030, 0.935, "1 · every forecast becomes a list of numbers",
             fontsize=14, fontweight="bold", color=INK)
    fig.text(0.600, 0.935, "2 · the numbers become one probability",
             fontsize=14, fontweight="bold", color=INK)

    # ---------------- left: what goes in ----------------
    axL.set_xlim(0, 100); axL.set_ylim(0, 100); axL.axis("off")

    # The forecast itself, in its own words.
    axL.add_patch(FancyBboxPatch((0, 78), 98, 20, boxstyle="round,pad=0.6,rounding_size=2",
                                 fc="white", ec=BLUE, lw=1.8))
    axL.text(3, 95, "“Stock prices will continue their advance\nthrough the coming year.”",
             fontsize=13, color=INK, style="italic", va="top", linespacing=1.5)
    axL.text(3, 80.5, "a real newspaper page, June 1929", fontsize=10.5, color=MUTED)

    arrow(axL, (49, 76.5), (49, 71.5))

    for y, (t, b, c) in zip([46, 24, 2], BLOCKS):
        chip(axL, 0, y, 98, 19, t, b, c)

    # ---------------- right: score -> probability ----------------
    z = np.linspace(-4.4, 4.4, 400)
    axR.plot(z, 1 / (1 + np.exp(-z)), color=INK, lw=2.6, solid_capstyle="round", zorder=3)
    axR.axhline(0.5, color=HAIR, lw=1.2, ls="--", zorder=1)
    axR.text(4.45, 0.555, "coin flip", fontsize=10.5, color=MUTED, ha="right")

    # An S-curve leaves exactly two empty pockets: above it on the left, below it
    # on the right. Both labels are anchored into those, at absolute positions --
    # offsets from the points themselves put one on the 50% line and the other
    # through the y-axis labels.
    for p, color, lbl, tx, ty in [
            (P_BULL, BLUE, "the upbeat call\n61%", 1.30, 0.615),
            (P_BEAR, VERM, "a downbeat call\nthe same week: 23%", -4.45, 0.375)]:
        x = logit(p)
        axR.plot([x, x], [0, p], color=color, lw=1.4, ls=":", zorder=2)
        axR.plot([x], [p], "o", ms=14, color=color, mec="white", mew=2.4, zorder=4)
        axR.text(tx, ty, lbl, fontsize=12, color=color, fontweight="bold",
                 ha="left", va="center", linespacing=1.4, zorder=5)

    axR.set_xlim(-4.6, 4.6); axR.set_ylim(-0.04, 1.04)
    axR.set_xticks([-4, 0, 4])
    axR.set_xticklabels(["← lower", "the score", "higher →"], fontsize=12)
    axR.set_yticks([0, .25, .5, .75, 1])
    axR.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=12)
    axR.set_ylabel("chance the forecast came true", fontsize=12.5, color=INK, labelpad=8)
    axR.tick_params(length=0)
    axR.grid(axis="y", color="#f2f2f2", lw=1)
    axR.set_axisbelow(True)
    for s in ("left", "bottom"):
        axR.spines[s].set_color("#cccccc")

    fig.text(0.785, 0.055,
             "score = every number × its own weight, added up.\n"
             "The weights are learned from 14,251 forecasts whose answers we know.",
             ha="center", va="center", fontsize=11.5, color=MUTED, linespacing=1.6)

    fig.subplots_adjust(left=0.030, right=0.985, top=0.885, bottom=0.175, wspace=0.28)
    p = OUT / "figM_logistic_explainer.png"
    fig.savefig(p, facecolor="white")
    print(f"-> {p}")


# --- figN: every column, in plain English ---------------------------------
GREEN = "#009E73"

# Block A is spelled out because claim_features has no importable label list.
# Blocks B and C are derived from the modules below, so they cannot go stale.
BLOCK_A = [
    ("which way it pointed",      "c_direction",   "improve / worsen / no change"),
    ("what it was about",         "c_topic",       "jobs, business, markets, prices"),
    ("who was speaking",          "c_voice",       "expert, official, journalist…"),
    ("how wide a claim",          "c_scope",       "city, state, national, global"),
    ("hedged or assertive",       "c_confidence",  "“may well” vs “will”"),
    ("a forecaster is quoted",    "c_quoted",      "yes / no"),
    ("that person is named",      "c_named",       "yes / no"),
    ("contains a figure",         "c_has_number",  "yes / no"),
    ("length of the quote",       "c_len",         "words, capped at 80"),
    ("how far ahead it looks",    "c_horizon",     "months"),
]
# The two the held-out permutation importance actually leans on.
TOP = {"x_dir_epu", "x_dir_stock_ret6"}


def _table(ax, color, heading, count, rows, note=None, sub=None):
    """One block: a coloured heading rule, then two-line rows.

    Everything is LEFT-aligned on both lines. Right-aligning the code names put
    them into the long labels of block C, and a right-aligned detail line under a
    left-aligned label reads as belonging to the row below it.

    The subtitle line is reserved in EVERY panel even though only one uses it,
    so the rules and the first rows stay on a common baseline across the three."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(0, 98, heading, fontsize=14, fontweight="bold", color=color, va="top")
    ax.text(100, 98, count, fontsize=10.5, color=MUTED, ha="right", va="top")
    if sub:
        ax.text(0, 92.5, sub, fontsize=10.5, color=MUTED, va="top", style="italic")
    ax.plot([0, 100], [88.5, 88.5], color=color, lw=2.8, solid_capstyle="butt",
            clip_on=False)

    top, step = 83.5, 6.5
    for i, (label, code, detail) in enumerate(rows):
        y = top - i * step
        if i:
            ax.plot([0, 100], [y + 3.6, y + 3.6], color="#ededed", lw=1,
                    solid_capstyle="butt")
        hot = code in TOP
        ax.text(0, y, label, fontsize=11.5, color=INK, va="center",
                fontweight="bold" if hot else "normal")
        sub = f"{code}   {detail}" if detail else code
        ax.text(0, y - 3.0, sub, fontsize=9, color=color if hot else MUTED,
                va="center", family="DejaVu Sans Mono")
    if note:
        ax.text(0, top - len(rows) * step - 0.5, note, fontsize=10.2, color=MUTED,
                va="top", linespacing=1.55, style="italic")


def build_table():
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 8.2),
                             gridspec_kw={"width_ratios": [1, 1, 1]})

    _table(axes[0], BLUE, "A · what the forecast said", "10 columns", BLOCK_A)

    rows_b = [(PRETTY[f], f, "") for f in FACTORS]
    _table(axes[1], VERM, "B · the economy that week", "26 columns", rows_b)

    rows_c = [("which way it pointed", "x_dir_sign", "+1 up · −1 down · 0 flat")]
    short = {"stock_ret6": "stock market return", "stock_drawdown": "market vs its peak",
             "epu": "policy uncertainty", "ip_accel": "output acceleration"}
    rows_c += [(f"direction × {short[f]}", f"x_dir_{f}", "") for f in INTERACT_WITH]
    _table(axes[2], GREEN, "C · direction × economy", "5 columns", rows_c,
           sub="the interaction terms",
           note="One factor per family — output, market level, market\n"
                "risk, policy — so the block cannot be four copies of\n"
                "the same signal.\n\n"
                "Bold = the two largest contributors, by permutation\n"
                "importance measured on held-out periods (shuffling\n"
                "stock market return costs 0.016 of ROC-AUC; policy\n"
                "uncertainty, 0.038).")

    fig.text(0.5, 0.068,
             "Every block-B factor is paired with a “did this series exist yet?” flag, so a missing "
             "factor is marked rather than quietly read as zero.",
             ha="center", fontsize=11, color=MUTED)
    fig.text(0.5, 0.030,
             "41 input columns · one-hot encoding expands the five categorical columns into 20, so "
             "the fitted model carries 56 coefficients and an intercept.",
             ha="center", fontsize=11, color=MUTED)

    fig.subplots_adjust(left=0.028, right=0.972, top=0.945, bottom=0.115, wspace=0.26)
    p = OUT / "figN_feature_table.png"
    fig.savefig(p, facecolor="white")
    print(f"-> {p}")


if __name__ == "__main__":
    build()
    build_table()
