"""How well the hit predictor ranked forecasts, era by era.

  figures/prelim_figures/figS_hit_auc_by_block.png

The ladder reports ONE pooled out-of-fold AUC (0.647). That number is an average
over 64 years, and the project's own rule -- the effective sample is ~21 blocks,
not 14,251 claims -- says the interesting question is how much it moves between
them. This splits the same out-of-fold predictions by their 3-year block and
scores each one on its own.

Every point is honest in the same way the pooled number is: `ladder_oof.csv`
holds leave-one-3-year-period-out predictions, so the claims in a block were
scored by a model fitted without that block.

Two things the chart is careful about:
  * the whiskers resample CLAIMS within a block, which ignores that forecasts in
    one era share wire copy -- so they are the OPTIMISTIC width, drawn thin, and
    the honest uncertainty on any single block is wider.
  * a within-block AUC needs both outcomes present and enough of each; blocks are
    plotted with their claim count below so a 0.99 on 404 claims cannot be read
    as equal evidence to a 0.65 on 1,025.

Run from the repo root:  python src/make_hit_auc_by_block_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

OOF = Path("data/models/ladder_oof.csv")
OUT = Path("figures/prelim_figures"); OUT.mkdir(parents=True, exist_ok=True)

BLUE, BAD = "#0072B2", "#b3261e"
INK, MUTED, FAINT = "#202124", "#5f6368", "#9aa0a6"
GRID, RULE = "#e8eaed", "#dadce0"
SEED, REPS = 0, 2000

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": INK, "axes.axisbelow": True})


def fast_auc(y, s):
    """Mann-Whitney AUC. roc_auc_score in a 2000-rep bootstrap loop is the whole
    runtime of this script; the rank identity is the same number."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def per_block(d):
    rng = np.random.default_rng(SEED)
    rows = []
    for b, g in d.groupby("block"):
        y = g["hit"].to_numpy()
        s = g["full"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        boot = np.array([fast_auc(y[i], s[i]) for i in
                         rng.integers(0, len(y), size=(REPS, len(y)))])
        boot = boot[~np.isnan(boot)]
        yrs = pd.to_datetime(g["date"]).dt.year
        rows.append({"block": int(b), "n": len(g), "hit_rate": y.mean(),
                     "auc": roc_auc_score(y, s),
                     "lo": np.percentile(boot, 2.5),
                     "hi": np.percentile(boot, 97.5),
                     # Real span, not block+2: the corpus starts in 1900 and
                     # stops in 1963, so the first and last blocks are short.
                     "y0": int(yrs.min()), "y1": int(yrs.max())})
    return pd.DataFrame(rows).sort_values("block").reset_index(drop=True)


def main():
    d = pd.read_csv(OOF)
    d = d[d["full"].notna()]
    r = per_block(d)
    pooled = roc_auc_score(d["hit"], d["full"])
    below = r[r["auc"] < .5]

    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(11.6, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [4.4, 1], "hspace": 0.10})

    x = r["block"].to_numpy() + 1          # centre of the 3-year block
    for xi, row in zip(x, r.itertuples()):
        c = BLUE if row.auc >= .5 else BAD
        ax.plot([xi, xi], [.5, row.auc], color=c, lw=2.6, zorder=3,
                solid_capstyle="round")
        ax.plot([xi, xi], [row.lo, row.hi], color=MUTED, lw=1.4, zorder=5,
                solid_capstyle="round")
        ax.plot([xi], [row.auc], "o", ms=8.5, color=c, mec="white", mew=1.6,
                zorder=6)

    ax.axhline(.5, color="#444444", lw=1.2, ls="--", zorder=1)
    ax.text(x[0] - 1.6, .5, "chance", fontsize=9, color=INK, ha="left",
            va="bottom")
    ax.axhline(pooled, color=BLUE, lw=1.1, ls=":", zorder=1, alpha=.9)
    ax.text(x[-1] + 1.4, pooled, f" pooled {pooled:.3f}", fontsize=9,
            color=BLUE, va="center", ha="left")

    for row in r.itertuples():
        if row.auc < .5 or row.auc > .95:
            ax.annotate(f"{row.auc:.2f}", (row.block + 1, row.auc),
                        textcoords="offset points",
                        xytext=(0, -14 if row.auc < .5 else 10),
                        ha="center", fontsize=9, fontweight="600",
                        color=BAD if row.auc < .5 else INK)

    # The top block is not a triumph and should not be read as one: in 1900-01
    # every 'improve' forecast hit and every 'worsen' one missed, so DIRECTION
    # alone separates the era. It is the clearest illustration on the chart of
    # why the effective sample is ~21 blocks and not 14,251 claims.
    top = r.loc[r["auc"].idxmax()]
    ax.annotate(f"{int(top['y0'])}–{int(top['y1'])}: every 'improve' forecast hit and "
                "every 'worsen'\nforecast missed — direction alone separates "
                "this era",
                xy=(top["block"] + 1.6, top["auc"]), xytext=(1906, .93),
                fontsize=8.8, color=MUTED, va="center", ha="left",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=FAINT, lw=1,
                                shrinkA=2, shrinkB=4))

    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=9.5)
    ax.set_ylabel("out-of-fold ROC-AUC\nwithin the block", fontsize=10,
                  color=MUTED, linespacing=1.5)
    ax.grid(axis="y", color=GRID, lw=.8)

    ax_n.bar(x, r["n"], width=2.4, color="#d6d9dd", zorder=3)
    ax_n.set_ylim(0, r["n"].max() * 1.25)
    ax_n.set_yticks([0, 1000])
    ax_n.set_yticklabels(["0", "1,000"], fontsize=9)
    ax_n.set_ylabel("claims\nscored", fontsize=9.5, color=MUTED,
                    linespacing=1.5)
    ax_n.grid(axis="y", color=GRID, lw=.8)
    ax_n.set_xlim(x[0] - 2.4, x[-1] + 2.4)
    ax_n.set_xticks(np.arange(1900, 1965, 10))
    ax_n.set_xticklabels([str(y) for y in range(1900, 1965, 10)], fontsize=9.5)
    ax_n.set_xlabel("3-year block", fontsize=10, color=MUTED)

    fig.suptitle("The pooled 0.647 is an average over eras that range from "
                 "0.21 to 0.99", fontsize=13.5, fontweight="600",
                 x=.008, ha="left", y=1.045)
    fig.text(.008, .975,
             "Each dot is one 3-year block of forecasts, scored by a model "
             "fitted without that block · blue above chance, red below",
             fontsize=10, color=MUTED, ha="left", va="center")
    fig.text(.008, -.15,
             f"{len(below)} of {len(r)} blocks rank BACKWARDS "
             f"({', '.join(f'{b.y0}–{b.y1}' for b in below.itertuples())})"
             " — the model does not merely weaken in those eras, it inverts. "
             "Whiskers are a 95% bootstrap over\n"
             "claims WITHIN a block, so they are the optimistic width: "
             "forecasts in one era share wire copy and one macro reality, and "
             "the true uncertainty on a single block is wider than drawn.\n"
             "Blocks are not equally informative — the bottom panel is the "
             "claim count behind each dot.",
             fontsize=8.6, color=MUTED, ha="left", va="top", linespacing=1.6)

    p = OUT / "figS_hit_auc_by_block.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=.10)
    plt.close(fig)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
