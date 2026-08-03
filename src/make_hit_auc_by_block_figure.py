"""How well the hit predictor ranked forecasts, era by era.

  figures/prelim_figures/figS_hit_auc_by_block.png        (L2, pooled 0.647)
  figures/prelim_figures/figS_hit_auc_by_block_l1.png     (L1, pooled 0.648)

The ladder reports ONE pooled out-of-fold AUC. That number is an average over
64 years, and the project's own rule -- the effective sample is ~21 blocks,
not 14,251 claims -- says the interesting question is how much it moves between
them. This splits the same out-of-fold predictions by their 3-year block and
scores each one on its own.

`--penalty l1` reads the lasso-fit predictions instead of the ridge ones, so the
chart matches the ladder in ladder_l1.json. Every headline number in the chart
is computed from whichever file was loaded -- none of them is written into the
text -- so the two versions cannot silently disagree with their own data.

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

Run from the repo root:
    python src/make_hit_auc_by_block_figure.py
    python src/make_hit_auc_by_block_figure.py --penalty l1
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

OUT = Path("figures/prelim_figures"); OUT.mkdir(parents=True, exist_ok=True)


def paths(penalty):
    """(predictions in, figure out) for one penalty."""
    if penalty == "l2":
        return Path("data/models/ladder_oof.csv"), OUT / "figS_hit_auc_by_block.png"
    return (Path(f"data/models/ladder_oof_{penalty}.csv"),
            OUT / f"figS_hit_auc_by_block_{penalty}.png")

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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--penalty", choices=["l2", "l1"], default="l2",
                    help="which cached out-of-fold predictions to score "
                         "(l1 = the lasso ladder in ladder_l1.json)")
    a = ap.parse_args()
    oof, out_png = paths(a.penalty)
    if not oof.exists():
        raise SystemExit(
            f"{oof} not found -- build it with:\n"
            f"  python -c \"import sys; sys.path.insert(0,'src'); "
            f"import make_roc_figure as M; M.compute_oof('{a.penalty}')\"")

    d = pd.read_csv(oof)
    d = d[d["full"].notna()]
    r = per_block(d)
    pooled = roc_auc_score(d["hit"], d["full"])

    # Near-square canvas with no title or caption block: the panels are the
    # whole image, so the plotting area gets the room the text used to take.
    # The claim-count strip keeps its 4.4:1 share -- it is a reference for
    # reading the dots, not a second chart competing with them.
    fig, (ax, ax_n) = plt.subplots(
        2, 1, figsize=(9.6, 8.4), sharex=True,
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
    # Guarded, not unconditional: the "direction alone" reading was checked
    # against the 1900-01 claims themselves. If a refit moves the top block
    # elsewhere the sentence would be an assertion about data nobody looked at,
    # so it is drawn only where it was verified.
    top = r.loc[r["auc"].idxmax()]
    if int(top["y0"]) == 1900 and top["auc"] >= .95:
        ax.annotate(f"{int(top['y0'])}–{int(top['y1'])}: every 'improve' forecast hit and "
                    "every 'worsen'\nforecast missed — direction alone separates "
                    "this era",
                    xy=(top["block"] + 1.6, top["auc"]), xytext=(1906, .93),
                    fontsize=8.8, color=MUTED, va="center", ha="left",
                    linespacing=1.5,
                    arrowprops=dict(arrowstyle="-", color=FAINT, lw=1,
                                    shrinkA=2, shrinkB=4))

    # Cropped below 0.25 rather than run to zero. Truncating an axis is usually
    # a distortion, but nothing here is measured from zero: the stems are drawn
    # from the 0.5 chance line, so 0.5 is the baseline the eye compares against
    # and an empty 0-0.18 band would only shrink the marks. The floor still
    # clears the lowest whisker (1929-31 at 0.18) and its label.
    ax.set_ylim(.12, 1.05)
    ax.set_yticks([.25, .5, .75, 1])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9.5)
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

    # No title and no caption block: this figure is dropped into a document
    # that carries its own heading and prose. The caveats the caption used to
    # state are still REQUIRED reading, so they are printed to stdout below --
    # dropping them from the PNG must not mean dropping them from the project.
    p = out_png
    fig.savefig(p, bbox_inches="tight", facecolor="white", pad_inches=.10)
    plt.close(fig)

    below = r[r["auc"] < .5]
    pen = "lasso (L1)" if a.penalty == "l1" else "ridge (L2)"
    print(r.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{pen} · pooled out-of-fold AUC {pooled:.3f} · "
          f"blocks range {r['auc'].min():.2f}-{r['auc'].max():.2f}")
    print(f"{len(below)} of {len(r)} blocks rank BACKWARDS "
          f"({', '.join(f'{b.y0}-{b.y1}' for b in below.itertuples())}) -- "
          "the model inverts there, it does not merely weaken.")
    print("Whiskers resample claims WITHIN a block, so they are the OPTIMISTIC "
          "width; true per-block uncertainty is wider.")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
