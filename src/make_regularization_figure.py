"""Every regularization setting tried, and what each one actually bought.

FOUR SEPARATE FILES, one question each -- v2_fig10a..d. They are written to be
read in order, and the last one withdraws the answer the first appears to give:

  A  accuracy   pooled out-of-fold ROC-AUC for L2, L1 and elastic net across
                two orders of magnitude of C. Twelve cells beat the incumbent.
  B  sparsity   how many of the 56 coefficients survive. L2 keeps 55 at every
                C; L1 at C=0.005 keeps 12.
  C  the path   the L1 coefficient path. As the penalty tightens, the four
                direction x economy terms are the last things it will delete --
                RESULTS_MACRO.md's central claim arriving by a method whose
                entire job is deleting things.
  D  the catch  pooled AUC gain against WITHIN-ERA gain. Both penalty
                candidates' pooled improvement is entirely cross-era: they are
                level (in fact marginally negative) at ranking forecasts inside
                a single era, which is the only thing this project claims to
                measure. The interaction block, tested the same way, is not.

D is why this figure exists rather than A. An earlier draft of
docs/RESULTS_MODEL_VARIANTS.md recommended switching to C=0.02 on the strength
of A alone; that recommendation was wrong and was withdrawn. Showing A without
D would re-make the same mistake in pictures -- so A's subtitle points at D by
name, which matters more now that they are separate files and can be placed
apart on the poster.

Numbers come from the committed tables (data/models/model_variants_penalty.csv,
model_variants_fold_decomposition.csv) so the figures cannot drift from the
text. The coefficient path in C is the one thing recomputed here -- 7 full-data
L1 fits, cached to data/models/model_variants_l1_path.csv.

    python src/make_regularization_figure.py
    python src/make_regularization_figure.py --refit-path
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hit_predictor as HP
import model_variants as MV

OUT = "figures/poster_v2"
PENALTY = Path("data/models/model_variants_penalty.csv")
DECOMP = Path("data/models/model_variants_fold_decomposition.csv")
PATH_CACHE = Path("data/models/model_variants_l1_path.csv")

# --- house palette ----------------------------------------------------------
# Slots 1-3 of the reference categorical theme, revalidated as a triple against
# this surface: all six checks PASS, worst all-pairs CVD dE 8.4 (protan). The
# 6-8 band obliges a second encoding, so every penalty also carries a distinct
# marker and a direct label -- identity never rests on colour alone here.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # L2
ORANGE = "#eb6834"    # L1
GREEN = "#199e70"     # elastic net

# One panel per file now, at roughly the size a panel occupied in the old 2x2
# sheet -- so the type sizes below, which were tuned against that panel width,
# still read the same.
W, H = 7.6, 6.6
BASE, LABEL, TITLE, NOTE = 17, 18, 25, 14.5

PENS = [("l2", "L2  (ridge)", BLUE, "o", "solid"),
        ("l1", "L1", ORANGE, "s", "solid"),
        ("elasticnet", "elastic net", GREEN, "^", (0, (5, 2.5)))]

INCUMBENT = ("l2", 0.5)     # what hit_predictor.py actually ships

# A subset of C_GRID: all seven labels collide at this width. The lines still
# carry a marker at every fitted C, so nothing is hidden.
XTICKS = [0.005, 0.02, 0.1, 0.5, 1.0]

# Figure C runs on its own, much finer grid, and reaches an order of magnitude
# further left than the evaluated grid. Both are deliberate. C_GRID's coarsest
# step spans the whole event this figure is about -- c_direction_no_change is
# zero at C=0.01 and -0.50 at C=0.02, so on that grid a term's entry point can
# only be bracketed, never drawn. And every term is already alive at C=0.005,
# so the evaluated range cannot show the ORDER in which they enter, which is
# the one thing a coefficient path is for.
PATH_GRID = list(np.exp(np.linspace(np.log(0.0002), np.log(1.0), 30)))
C_XTICKS = [0.001, 0.01, 0.1, 1.0]

# Where each page's text column starts, in figure coords -- titles and captions
# align to this on all four, whatever the axes' own left margin has to be.
NOTE_X = 0.135
# The same point in D's axes coords: (NOTE_X - its left margin) / its width.
D_TITLE_X = -0.246

# The terms figure C is about. INTER_NUM also contains x_dir_sign (direction on
# its own), which is not a direction x ECONOMY product and so is not part of
# the claim being tested here.
KEY_TERMS = {f"x_dir_{f}": n for f, n in [
    ("epu", "× policy uncertainty"), ("stock_ret6", "× stock return"),
    ("stock_drawdown", "× drawdown"), ("ip_accel", "× output accel.")]}

# NOT highlighted, though it is the largest negative coefficient in the model
# and the eye goes to it: c_direction_no_change. It is a big effect on a small
# slice -- 400 of 14,251 claims, hit rate 0.180 against a 0.513 base -- and the
# held-out permutation importance of the whole c_direction feature is 0.005 AUC
# against x_dir_epu's 0.038. Drawing it in ink gave a term that moves the model
# very little the same visual weight as the four that carry it.


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": BASE, "axes.labelsize": LABEL, "axes.titlesize": TITLE,
        "xtick.labelsize": BASE - 2, "ytick.labelsize": BASE - 2,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
        "grid.color": GRID, "grid.linewidth": 1.0,
        "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
    })


def clean(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


def panel_title(ax, letter, text, sub=None, x=0.0):
    """`x` is in axes coords, so a panel with a wide left gutter (D, whose tick
    labels are three-line era counts) can still start its title at the page's
    left edge rather than a quarter of the way in."""
    ax.set_title(f"{letter}   {text}", fontsize=TITLE - 7, pad=40, loc="left",
                 color=INK).set_x(x)
    if sub:
        ax.text(x, 1.015, sub, transform=ax.transAxes, fontsize=NOTE - 2,
                color=MUTED, va="bottom", ha="left")


# --- panel C's data ---------------------------------------------------------
def l1_path(refit=False):
    """Coefficient value at each C, from a full-data L1 fit.

    In-sample on purpose and labelled as such in the caption: the path is a
    DESCRIPTION of what the penalty keeps in what order, not an evaluation of
    it. The evaluation is figure A, which is out-of-fold."""
    if PATH_CACHE.exists() and not refit:
        cached = pd.read_csv(PATH_CACHE, index_col=0)
        # A cache written against a different grid is silently the wrong figure,
        # so the grid itself is the cache key.
        if len(cached.columns) == len(PATH_GRID):
            return cached
        print(f"  [cached path has {len(cached.columns)} C values, "
              f"grid wants {len(PATH_GRID)} -- refitting]")
    d, y, _ = MV.load()
    X, cat, num = MV.design(d)
    cols = {}
    for C in PATH_GRID:
        print(f"  L1 fit at C={C} ...", flush=True)
        pipe = HP.make_pipe(cat, num, clf=MV.clf_for("l1", C))
        pipe.fit(X, y)
        names = [n.split("__", 1)[-1]
                 for n in pipe.named_steps["pre"].get_feature_names_out()]
        cols[C] = pd.Series(pipe.named_steps["clf"].coef_.ravel(), index=names)
    out = pd.DataFrame(cols)
    out.to_csv(PATH_CACHE)
    print(f"  -> {PATH_CACHE}")
    return out


# --- panels -----------------------------------------------------------------
def panel_a(ax, tab):
    ax.grid(True, axis="y", lw=0.9, alpha=0.7)
    ax.set_axisbelow(True)
    for key, label, color, marker, dash in PENS:
        t = tab[tab["penalty"] == key].sort_values("C")
        ax.plot(t["C"], t["roc_auc"], color=color, lw=2.2, ls=dash, marker=marker,
                markersize=8.5, mec=SURFACE, mew=1.6, label=label, zorder=4)

    inc = tab[(tab["penalty"] == INCUMBENT[0]) & (tab["C"] == INCUMBENT[1])].iloc[0]
    ax.axhline(inc["roc_auc"], color=AXIS, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.plot([inc["C"]], [inc["roc_auc"]], "o", markersize=17, mfc="none",
            mec=INK, mew=2.0, zorder=6)
    # Sits above the axis, not on it: as its own file this panel is shorter than
    # it was in the 2x2 sheet, and at the old y the second line landed on the
    # x tick labels.
    ax.annotate("what the model ships:\nL2, C = 0.5", xy=(inc["C"], inc["roc_auc"]),
                xytext=(0.30, 0.6452), fontsize=NOTE - 1, color=INK, ha="center",
                va="top", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=1.2))

    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{c:g}" for c in XTICKS])
    ax.set_ylim(0.6405, 0.6605)
    ax.set_yticks([0.645, 0.650, 0.655, 0.660])
    ax.set_xlabel("C   (larger = weaker penalty)", fontsize=LABEL - 3, labelpad=9)
    ax.set_ylabel("out-of-fold ROC-AUC", fontsize=LABEL - 3, labelpad=9)
    leg = ax.legend(loc="lower left", fontsize=NOTE - 1, handlelength=2.4,
                    labelspacing=0.45, borderaxespad=0.5)
    for t_, (_, _, color, *_) in zip(leg.get_texts(), PENS):
        t_.set_color(color)
    panel_title(ax, "A", "Twelve settings beat the incumbent",
                "on pooled out-of-fold AUC — read figure D before believing it")
    clean(ax)


def panel_b(ax, tab):
    ax.grid(True, axis="y", lw=0.9, alpha=0.7)
    ax.set_axisbelow(True)
    n_coef = int(tab["n_coef"].iloc[0])
    ax.axhline(n_coef, color=AXIS, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.text(MV.C_GRID[0] * 0.9, n_coef + 1.4, f"all {n_coef} coefficients",
            color=MUTED, fontsize=NOTE - 2, va="bottom", ha="left")

    # Direct labels rather than a second legend: the three lines end far apart,
    # which is the panel's whole point. L1 and elastic net land 2 apart at the
    # right edge, so their labels are nudged off each other.
    for (key, label, color, marker, dash), dy in zip(PENS, [0, -2.4, 2.4]):
        t = tab[tab["penalty"] == key].sort_values("C")
        ax.plot(t["C"], t["nonzero"], color=color, lw=2.2, ls=dash, marker=marker,
                markersize=8.5, mec=SURFACE, mew=1.6, zorder=4)
        last = t.iloc[-1]
        ax.text(last["C"] * 1.25, last["nonzero"] + dy, label.split("  ")[0],
                color=color, fontsize=NOTE - 1, va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{c:g}" for c in XTICKS])
    ax.set_xlim(MV.C_GRID[0] * 0.72, MV.C_GRID[-1] * 5.5)
    ax.set_ylim(0, n_coef + 9)
    ax.set_xlabel("C   (larger = weaker penalty)", fontsize=LABEL - 3, labelpad=9)
    ax.set_ylabel("coefficients left non-zero", fontsize=LABEL - 3, labelpad=9)
    panel_title(ax, "B", "Only L1 deletes anything",
                "ridge shrinks all 56 toward zero; it never arrives")
    clean(ax)


def entry_c(path, name):
    """The C at which a coefficient leaves zero for good.

    Read from the right, where every term is alive, and walk left to the first
    C at which this one is zero -- rather than the leftmost non-zero, because a
    coefficient can flicker on, off and on again along a path. Returned as the
    geometric midpoint of the bracketing pair, which is all the grid resolves;
    the figure quotes it to one significant figure for the same reason."""
    if name not in path.index:
        return None
    alive = path.loc[name].abs().values > 1e-8
    cs = np.asarray(path.columns, dtype=float)
    if not alive[-1]:
        return None          # never enters, even at the weakest penalty tried
    i = 0
    for j in range(len(cs) - 1, -1, -1):
        if not alive[j]:
            i = j + 1
            break
    return None if i == 0 else float(np.sqrt(cs[i - 1] * cs[i]))


def panel_c(ax, path):
    ax.grid(True, axis="y", lw=0.9, alpha=0.7)
    ax.set_axisbelow(True)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=2)

    cs = [float(c) for c in path.columns]
    for name, row in path.iterrows():
        if name in KEY_TERMS:
            continue
        ax.plot(cs, row.values, color=MUTED, lw=1.0, alpha=0.5, zorder=3)

    marked = [(n, s, ORANGE) for n, s in KEY_TERMS.items() if n in path.index]

    # Several land within 0.2 of each other at the right edge, so the labels are
    # pulled apart vertically and each keeps a leader to its curve.
    ends = sorted(marked, key=lambda t: path.loc[t[0]].values[-1], reverse=True)
    label_y, floor = {}, None
    for name, _, _ in ends:
        v = path.loc[name].values[-1]
        y = v if floor is None else min(v, floor - 0.26)
        label_y[name], floor = y, y

    for name, nice, color in ends:
        v = path.loc[name].values
        ax.plot(cs, v, color=color, lw=2.4 if color == ORANGE else 2.8,
                zorder=5 if color == ORANGE else 6)
        ax.annotate(nice, xy=(cs[-1], v[-1]), xytext=(cs[-1] * 1.30, label_y[name]),
                    color=color, fontsize=NOTE - 2, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=1.0))

    # --- the deletion line ----------------------------------------------------
    # One line, and it is the only one that can be drawn for the panel as a
    # whole: the C below which EVERY coefficient is zero. Per-term rules were
    # tried and removed -- a full-height rule at one term's entry reads as this
    # boundary, which is an order of magnitude away and means something else.
    # Each term's own entry stays visible as the point where its curve lifts
    # off zero, marked with a ring on the four highlighted ones.
    lo, hi = ax.get_ylim()
    for name, _, color in ends:
        e = entry_c(path, name)
        if e:
            ax.plot([e], [0], "o", mfc=SURFACE, mec=color, mew=2.2,
                    markersize=9, zorder=7)

    e_all = min((c for c in (entry_c(path, n) for n in path.index)
                 if c is not None), default=None)
    x_lo = cs[0] * 0.80
    if e_all:
        ax.axvspan(x_lo, e_all, color=GRID, alpha=0.75, lw=0, zorder=1)
        ax.axvline(e_all, color=INK2, lw=1.8, ls=(0, (5, 3.5)), zorder=4)
        # No leader: the text sits against the rule already, and a leader drawn
        # at the same height runs straight through its own label.
        ax.text(e_all * 1.5, hi * 0.72,
                f"L1 has deleted every\ncoefficient left of here\nC ≈ {e_all:.1g}",
                color=INK2, fontsize=NOTE - 2.5, ha="left", va="center",
                linespacing=1.45)

    n_other = len(path.index) - len(marked)
    ax.text(cs[-1] * 0.45, -0.80, f"the other {n_other} coefficients", color=MUTED,
            fontsize=NOTE - 2, va="center", ha="center")
    ax.set_xscale("log")
    ax.set_xticks(C_XTICKS)
    ax.set_xticklabels([f"{c:g}" for c in C_XTICKS])
    ax.set_xlim(x_lo, cs[-1] * 4.0)
    ax.set_xlabel("C   (smaller = deletes more)", fontsize=LABEL - 3, labelpad=9)
    ax.set_ylabel("L1 coefficient", fontsize=LABEL - 3, labelpad=9)
    panel_title(ax, "C", "What L1 deletes, and in what order",
                "a coefficient is exactly zero until its own curve lifts off —\n"
                "the four direction × economy terms lift off before any other")
    clean(ax)


def panel_d(ax, dec):
    ax.grid(True, axis="x", lw=0.9, alpha=0.7)
    ax.set_axisbelow(True)
    ax.axvline(0, color=AXIS, lw=1.4, zorder=2)

    rows = list(dec.itertuples())[::-1]     # headline last in the CSV -> bottom
    ys = np.arange(len(rows))
    names = {"L2 C=0.02": "L2, C = 0.02",
             "L1 C=0.05": "L1, C = 0.05",
             "interacted (the headline)": "the interaction block\n(the headline)"}

    for i, r in zip(ys, rows):
        head = r.a == "interacted (the headline)"
        c_within = INK if head else INK2
        ax.plot([r.pooled_delta, r.wtd_fold_delta], [i, i], color=AXIS, lw=2.0,
                zorder=3, solid_capstyle="round")
        # Hollow = pooled, filled = within-era. Two MEASURES of one comparison,
        # so they are separated by fill and ink weight rather than by two more
        # hues -- colour in this figure means "penalty" and nothing else.
        ax.plot([r.pooled_delta], [i], "o", color=SURFACE, mec=MUTED, mew=2.4,
                markersize=12, zorder=4)
        ax.plot([r.wtd_fold_delta], [i], "o", color=c_within, markersize=12,
                mec=SURFACE, mew=1.8, zorder=5)
        ax.text(r.pooled_delta, i + 0.26, f"{r.pooled_delta:+.3f}", color=MUTED,
                fontsize=NOTE - 2, ha="center", va="bottom")
        # "-0.000" reads as a rendering bug rather than as a number rounding to
        # zero, which is exactly what the L1 candidate's within-era delta does.
        w = f"{r.wtd_fold_delta:+.3f}".replace("-0.000", "0.000")
        ax.text(r.wtd_fold_delta, i - 0.26, w,
                color=c_within, fontsize=NOTE - 2, ha="center", va="top")

    # Inline key in the panel's empty upper-right, rather than labels hung off
    # the dots: on the two penalty rows the dots are ~0.01 apart and any label
    # touching them lands on its neighbour.
    ax.plot([0.029], [2.46], "o", color=SURFACE, mec=MUTED, mew=2.4,
            markersize=12, zorder=6)
    ax.text(0.034, 2.46, "pooled across eras", color=MUTED, fontsize=NOTE - 2,
            ha="left", va="center")
    ax.plot([0.029], [2.14], "o", color=INK, markersize=12, mec=SURFACE,
            mew=1.8, zorder=6)
    ax.text(0.034, 2.14, "within an era", color=INK, fontsize=NOTE - 2,
            ha="left", va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{names[r.a]}\n{r.folds_won} of {r.n_folds} eras won"
                        for r in rows], fontsize=NOTE - 1)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(INK if r.a == "interacted (the headline)" else INK2)
    ax.set_ylim(-0.62, len(rows) - 0.20)
    ax.set_xlim(-0.026, 0.088)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06])
    ax.set_xlabel("change in ROC-AUC against the incumbent",
                  fontsize=LABEL - 3, labelpad=9)
    panel_title(ax, "D", "Both gains evaporate within an era",
                "the headline's does not — it wins 17 of 22 eras", x=D_TITLE_X)
    clean(ax, keep=("bottom",))


# Each figure now carries its own provenance, since none of them sits under a
# shared caption any more. Two lines common to all four, then the one caveat
# that figure specifically needs.
COMMON = ("The AUC-0.647 hit model of src/hit_predictor.py, 14,251 forecasts 1900–1963.\n"
          "Only the penalty and C change.   docs/RESULTS_MODEL_VARIANTS.md\n")

# name, drawing function, which table it reads, axes margins, its own caveat.
# D gets a wide left gutter: its y labels are two- and three-line era counts.
PAGES = [
    ("a_penalty_auc", panel_a, "tab",
     dict(left=0.145, right=0.965, top=0.795, bottom=0.295),
     "Out-of-fold, leave-one-3-year-era-out over 22 eras. The grid spans 0.012 of\n"
     "AUC; the interval on the headline alone is ±0.07 — read for shape, not a winner."),
    ("b_sparsity", panel_b, "tab",
     dict(left=0.145, right=0.965, top=0.795, bottom=0.295),
     "Survivor counts are a full-data fit at each C — a description of what the\n"
     "penalty keeps, not an evaluation of it. The evaluation is figure A."),
    # C stops well short of the right edge: its four direct labels are drawn
    # past the last fitted C and used to spill into the old sheet's column gap.
    ("c_l1_path", panel_c, "path",
     dict(left=0.145, right=0.735, top=0.795, bottom=0.295),
     "A full-data L1 fit at each C — what the penalty keeps and in what order,\n"
     "not a claim about accuracy. Every term has its own deletion point."),
    ("d_within_era", panel_d, "dec",
     dict(left=0.300, right=0.970, top=0.795, bottom=0.295),
     "Out-of-fold, 22 eras. Pooled = one AUC over all eras' held-out predictions;\n"
     "within-era = the size-weighted mean of the 22 per-era deltas."),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit-path", action="store_true",
                    help="recompute the cached L1 coefficient path")
    a = ap.parse_args()

    for p in (PENALTY, DECOMP):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run `python src/model_variants.py` first.")
    os.makedirs(OUT, exist_ok=True)
    style()

    path = l1_path(a.refit_path)
    path.columns = [float(c) for c in path.columns]
    data = {"tab": pd.read_csv(PENALTY), "dec": pd.read_csv(DECOMP),
            "path": path}
    for name, draw, src, margins, caveat in PAGES:
        fig, ax = plt.subplots(figsize=(W, H))
        fig.subplots_adjust(**margins)
        draw(ax, data[src])
        fig.text(NOTE_X, margins["bottom"] - 0.115, COMMON + caveat,
                 fontsize=NOTE - 2.5, color=MUTED, va="top", ha="left",
                 linespacing=1.6)
        p = f"{OUT}/v2_fig10{name}.png"
        fig.savefig(p, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
