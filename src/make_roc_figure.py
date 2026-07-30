"""The hit model as an ROC curve, one rung of the nested ladder at a time.

This is the model's DISCRIMINATION drawn as a graph: for every threshold, how
many forecasts it correctly flags as likely-to-come-true against how many it
wrongly flags. The area under that curve is the AUC the poster quotes.

    left panel    the four ROC curves, one per rung of hit_predictor's ladder,
                  plus a block-bootstrap uncertainty band on the headline.
    right panel   the same four AUCs as points with 95% block-bootstrap
                  intervals, because a curve makes 0.647 and 0.581 look further
                  apart than the interval says they are.

Everything is OUT-OF-FOLD: leave-one-3-year-block-out, exactly hit_predictor's
evaluation. An in-sample ROC would be a much prettier picture of nothing --
v2_fig4 exists specifically to show how far in-sample AUC on this corpus
overstates the truth.

The uncertainty is a BLOCK bootstrap over the 22 three-year periods, never an
i.i.d. bootstrap over the 14,251 claims. Claims inside one period share one
economy and one set of answers, so resampling claims would report an interval
roughly sqrt(claims-per-block) too narrow -- the effective sample here is 22
blocks and the band should look like it. This is the same reasoning (and the
same helper) as hit_predictor.block_boot_auc.

Out-of-fold predictions for each rung are cached to data/models/ladder_oof.csv,
so re-running the figure does not refit 88 logistic regressions.

    python src/make_roc_figure.py
    python src/make_roc_figure.py --refit      # recompute the cached OOF preds
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hit_predictor as HP
from macro_context import BLOCK_YEARS, FACTORS

OUT = "figures/poster_v2"
CTX = "data/scored/macro_context.csv"
CACHE = Path("data/models/ladder_oof.csv")
CI_CACHE = Path("data/models/ladder_roc_ci.json")

# --- house palette (validated: see references/palette.md) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1 -- the model the poster reports
ORANGE = "#eb6834"    # categorical slot 2 -- the rung directly below it

W = 13.4
BASE, LABEL, TITLE, NOTE = 17, 18, 25, 14.5

BOOT = 1000           # block-bootstrap resamples for the band and the intervals
FPR_GRID = np.linspace(0, 1, 201)

# The rungs drawn, in ladder order. The two baselines are deliberately drawn in
# neutral ink rather than a third and fourth hue: they are the things that do
# NOT work, and they orient the reader without competing for attention. Their
# identity rests on a direct label plus a dash pattern, never on colour alone.
RUNGS = [
    ("claim features only", "claim", MUTED, (0, (1, 2.4)), 2.0),
    ("economy only", "econ", MUTED, (0, (6, 3)), 2.0),
    ("claim + economy (additive)", "additive", ORANGE, "solid", 2.4),
    ("+ direction × economy", "full", BLUE, "solid", 3.2),
]


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
        "grid.color": GRID, "grid.linewidth": 1.0,
        "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
    })


def clean(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.tick_params(length=0)


# --- the predictions --------------------------------------------------------
def compute_oof():
    """Out-of-fold probabilities for every rung, under hit_predictor's own CV.

    Feature blocks and the estimator come from hit_predictor rather than being
    restated here, so this figure cannot describe a model different from the one
    that was actually fitted and saved."""
    d = pd.read_csv(CTX, low_memory=False)
    d = d[d["hit"].isin([0, 1])].reset_index(drop=True)
    y = d["hit"].astype(int).values
    year = pd.to_datetime(d["date"]).dt.year
    groups = ((year // BLOCK_YEARS) * BLOCK_YEARS).values

    X = pd.concat([HP.claim_features(d), HP.macro_block(d[FACTORS]),
                   HP.interaction_block(d, d[FACTORS])], axis=1)

    specs = {
        "claim": (HP.CLAIM_CAT, HP.CLAIM_NUM),
        "econ": ([], HP.MACRO_NUM),
        "additive": (HP.CLAIM_CAT, HP.CLAIM_NUM + HP.MACRO_NUM),
        "full": (HP.CLAIM_CAT, HP.CLAIM_NUM + HP.MACRO_NUM + HP.INTER_NUM),
    }
    out = pd.DataFrame({"date": d["date"], "block": groups, "hit": y})
    for key, (cat, num) in specs.items():
        print(f"  fitting {key} ...", flush=True)
        out[key] = HP.oof_predict(X, y, groups, cat, num)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    print(f"  -> {CACHE}")
    return out


def load_oof(refit=False):
    if refit or not CACHE.exists():
        return compute_oof()
    return pd.read_csv(CACHE)


# --- uncertainty ------------------------------------------------------------
def boot_blocks(y, blocks, reps, seed=HP.SEED):
    """Row indices for `reps` resamples of whole 3-year blocks."""
    rng = np.random.default_rng(seed)
    by_block = {b: np.where(blocks == b)[0] for b in np.unique(blocks)}
    uniq = np.array(list(by_block))
    for _ in range(reps):
        idx = np.concatenate([by_block[b] for b in
                              rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            yield idx


def roc_band(y, p, blocks, reps=BOOT):
    """Pointwise 95% band for one ROC curve, from resampled blocks.

    Pointwise, not simultaneous: read it as "at this false-alarm rate, the hit
    rate is somewhere in here", not as a region containing the whole curve."""
    curves = []
    for idx in boot_blocks(y, blocks, reps):
        f, t, _ = roc_curve(y[idx], p[idx])
        curves.append(np.interp(FPR_GRID, f, t))
    c = np.vstack(curves)
    return np.percentile(c, 2.5, axis=0), np.percentile(c, 97.5, axis=0)


def auc_ci(y, p, blocks, reps=BOOT):
    draws = [roc_auc_score(y[i], p[i]) for i in boot_blocks(y, blocks, reps)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(roc_auc_score(y, p)), float(lo), float(hi)


def stats(d, refit=False):
    """AUCs, intervals and the headline's band -- cached, they take a minute."""
    y = d["hit"].values.astype(int)
    blocks = d["block"].values
    if CI_CACHE.exists() and not refit:
        s = json.loads(CI_CACHE.read_text())
        s["band"] = (np.array(s["band"][0]), np.array(s["band"][1]))
        return s
    s = {"auc": {}, "reps": BOOT, "n": int(len(d)),
         "n_blocks": int(len(np.unique(blocks)))}
    for _, key, *_ in RUNGS:
        s["auc"][key] = auc_ci(y, d[key].values, blocks)
        print(f"  {key:<10}AUC {s['auc'][key][0]:.3f}  "
              f"[{s['auc'][key][1]:.3f}, {s['auc'][key][2]:.3f}]")
    lo, hi = roc_band(y, d["full"].values, blocks)
    s["band"] = (lo.tolist(), hi.tolist())
    CI_CACHE.write_text(json.dumps(s, indent=2))
    s["band"] = (lo, hi)
    return s


# --- the figure -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true",
                    help="recompute the cached out-of-fold predictions and CIs")
    a = ap.parse_args()

    if not os.path.exists(CTX):
        raise SystemExit(f"{CTX} not found -- run `python src/macro_context.py` first.")
    os.makedirs(OUT, exist_ok=True)
    style()

    d = load_oof(a.refit).dropna(subset=[k for _, k, *_ in RUNGS])
    s = stats(d, a.refit)
    y = d["hit"].values.astype(int)

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(W, 7.6), gridspec_kw={"width_ratios": [1, 0.78],
                                             "wspace": 0.62})
    fig.subplots_adjust(left=0.10, right=0.985, top=0.895, bottom=0.215)

    # ---------------------------------------------------------------- panel A
    ax.grid(True, axis="both", lw=0.9, alpha=0.7)
    ax.set_axisbelow(True)

    # Chance first and recessive. It is the reference every curve is read
    # against, so it must be present and must not compete.
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.text(0.86, 0.845, "coin flip", color=MUTED, fontsize=NOTE,
            rotation=45, rotation_mode="anchor", ha="center", va="top")

    lo, hi = s["band"]
    ax.fill_between(FPR_GRID, lo, hi, color=BLUE, alpha=0.15, lw=0, zorder=3)

    # The four curves run nearly parallel through the same corner, so leader
    # lines to each one cross each other and land on top of the band. A legend
    # in the empty triangle below the diagonal carries the same information --
    # name, colour, dash and AUC together -- without a single crossing.
    for label, key, color, dash, lw in RUNGS:
        f, t, _ = roc_curve(y, d[key].values)
        ax.plot(f, t, color=color, lw=lw, ls=dash, zorder=5 if key == "full" else 4,
                solid_capstyle="round",
                label=f"{label}   {s['auc'][key][0]:.3f}")

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="lower right",
                    fontsize=NOTE - 1, handlelength=2.6, handletextpad=0.9,
                    labelspacing=0.55, borderaxespad=0.8)
    for t_, (_, key, color, *_) in zip(leg.get_texts(), RUNGS[::-1]):
        t_.set_color(color if color != MUTED else INK2)

    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.012, 1.012)
    ax.set_aspect("equal")
    ax.set_xlabel("forecasts wrongly flagged as likely to come true",
                  fontsize=LABEL - 3, labelpad=9)
    ax.set_ylabel("forecasts correctly flagged", fontsize=LABEL - 3, labelpad=9)
    for setter in (ax.set_xticks, ax.set_yticks):
        setter([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Every threshold at once", fontsize=TITLE - 5, pad=14,
                 loc="left", color=INK)
    ax.set_anchor("W")   # square plot hugs the left; the slack goes to panel B
    clean(ax)

    # ---------------------------------------------------------------- panel B
    # The curves make 0.647 and 0.581 look decisively apart. Against 22 blocks
    # they are not, and the same figure has to say so or it is arguing one side.
    bx.grid(True, axis="x", lw=0.9, alpha=0.7)
    bx.set_axisbelow(True)
    ys = np.arange(len(RUNGS))
    for i, (label, key, color, dash, lw) in enumerate(RUNGS):
        auc, clo, chi = s["auc"][key]
        c = color if color != MUTED else INK2
        bx.hlines(i, clo, chi, color=c, lw=2.6, alpha=0.45,
                  capstyle="round", zorder=4)
        bx.plot([auc], [i], "o", color=c, markersize=11, mec=SURFACE, mew=2.0,
                zorder=5)
        bx.text(chi + 0.008, i, f"{auc:.3f}", color=c, fontsize=NOTE,
                va="center", ha="left")

    bx.axvline(0.5, color=AXIS, lw=1.4, ls=(0, (5, 4)), zorder=2)
    bx.text(0.5, len(RUNGS) - 0.38, " coin flip", color=MUTED, fontsize=NOTE,
            ha="left", va="center")

    bx.set_yticks(ys)
    bx.set_yticklabels([n.replace(" (additive)", "\n(additive)")
                        for n, *_ in RUNGS], fontsize=NOTE)
    for tick, (_, key, color, *_) in zip(bx.get_yticklabels(), RUNGS):
        tick.set_color(INK if key == "full" else INK2)
    bx.set_ylim(-0.6, len(RUNGS) - 0.4)
    bx.set_xlim(0.42, 0.80)
    bx.set_xticks([0.5, 0.6, 0.7])
    bx.set_xlabel("area under the curve, 95% block bootstrap",
                  fontsize=LABEL - 3, labelpad=9)
    bx.set_title("…with their error bars", fontsize=TITLE - 5, pad=14,
                 loc="left", color=INK)
    clean(bx, keep=("bottom",))

    d_auc = s["auc"]["full"][0] - s["auc"]["additive"][0]
    fig.text(0.10, 0.132,
             "{n:,} forecasts, 1900–1963. Out-of-fold only — every forecast is "
             "scored by a model that never saw its 3-year era.\n"
             "The intervals resample the {b} eras, not the {n:,} forecasts, so "
             "they are wide. Boosting the same inputs scored 0.617.\n"
             "What survives that test is the {dl:+.3f} interaction lift over the "
             "additive rung (paired p = 0.002), not any curve's level."
             .format(n=s["n"], b=s["n_blocks"], dl=d_auc),
             fontsize=NOTE - 1.5, color=MUTED, va="top", linespacing=1.6)

    p = f"{OUT}/v2_fig9_roc.png"
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
