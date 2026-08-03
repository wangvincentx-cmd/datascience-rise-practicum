"""
Economic-context figures: which conditions at print time predicted a hit.

Two panels, one argument.

  figI  the scissors. Hit rate across terciles of each economic factor, drawn
        separately for optimistic and pessimistic forecasts, with the POOLED
        line on top. The pooled line is flat -- which is exactly why
        model_hit.py's macro-only model sits at chance (AUC 0.505). The two
        component lines are not flat, they are opposed, and they cancel.
  figJ  the ledger. Every factor's correlation with `hit`, optimistic vs
        pessimistic, with block-bootstrap intervals. The honest "which factors
        mattered, and which didn't" chart, including the ones that didn't.

Reads data/scored/macro_context.csv (run src/macro_context.py first).
Palette is Okabe-Ito
(CVD-safe); design follows the dataviz method: one idea per panel, direct
labels, legend for >=2 series, recessive grid, no dual axes.

Figures carry NO titles or subtitles -- the headline and caption for each live
in the poster deck. Numbers that used to sit in a subtitle are printed instead.

Usage (from the repo root): python src/make_macro_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from macro_context import (BLOCK_YEARS, FACTORS, PRETTY, block_boot_corr,
                           wilson)

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("figures/poster_figures"); OUT.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})

# The four headline factors for figI. Chosen for COVERAGE and for being things a
# contemporary could actually observe -- the stock ticker and the newspapers'
# own policy noise. Unemployment is the strongest single factor in the table but
# only exists from 1948 (14% of claims), so it belongs in the ledger, not in the
# headline panel where a reader would assume it spans the century.
HEADLINE = ["stock_ret6", "epu", "stock_drawdown", "ip_accel"]

TERCILES = ["low", "mid", "high"]


def load():
    d = pd.read_csv("data/scored/macro_context.csv", low_memory=False)
    d["optimistic"] = d["predicted_norm"].isin(["improve", "up"])
    d["pessimistic"] = d["predicted_norm"].isin(["worsen", "down"])
    d["block"] = (pd.to_datetime(d["date"]).dt.year // BLOCK_YEARS) * BLOCK_YEARS
    return d


def tercile_rates(d, factor, mask=None):
    """(hit rate, lo, hi, n) per tercile of `factor`.

    Terciles are cut on the FULL sample, not within the subgroup, so the
    optimistic and pessimistic lines are read against the same x-axis -- cutting
    them separately would put different economies at the same tick and make the
    two lines uncomparable."""
    v = d[factor]
    ok = v.notna()
    edges = np.nanpercentile(v[ok], [0, 100 / 3, 200 / 3, 100])
    bins = pd.cut(v, bins=np.unique(edges), labels=TERCILES,
                  include_lowest=True)
    sub = d[ok & (mask if mask is not None else True)]
    b = bins[sub.index]
    out = []
    for t in TERCILES:
        y = sub.loc[b == t, "hit"]
        if len(y) < 25:
            out.append((np.nan, np.nan, np.nan, len(y)))
            continue
        k, n = int(y.sum()), len(y)
        lo, hi = wilson(k, n)
        out.append((k / n, lo, hi, n))
    return out


# --- I. THE SCISSORS -------------------------------------------------------
def fig_scissors(d):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), sharey=True)
    x = np.arange(3)
    for ax, f in zip(axes.ravel(), HEADLINE):
        for mask, color, lbl in [(d["optimistic"], BLUE, "optimistic"),
                                 (d["pessimistic"], VERM, "pessimistic"),
                                 (None, GRAY, "all pooled")]:
            r = tercile_rates(d, f, mask)
            y = [a[0] for a in r]
            lo = [a[0] - a[1] for a in r]
            hi = [a[2] - a[0] for a in r]
            ls = "--" if lbl == "all pooled" else "-"
            ax.errorbar(x, y, yerr=[lo, hi], color=color, lw=2.4, ls=ls,
                        marker="o", ms=9, capsize=4, elinewidth=1.4,
                        zorder=4 if lbl != "all pooled" else 3,
                        label=lbl, mec="white", mew=1.6)
        ax.axhline(.5, color="#bbbbbb", lw=1, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["low", "middle", "high"])
        ax.set_xlim(-.35, 2.35)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_xlabel(PRETTY[f] + "  at print time", fontsize=11.5, color=INK)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of forecasts that came true")

    # Direct labels on the first panel; legend carries the rest.
    ax0 = axes[0, 0]
    ax0.legend(frameon=False, loc="upper left", fontsize=11, ncol=1,
               handlelength=2.4)

    fig.tight_layout()
    fig.savefig(OUT / "figI_macro_scissors.png", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}/figI_macro_scissors.png")


# --- J. THE LEDGER ---------------------------------------------------------
def fig_ledger(d, reps=2000):
    """Every factor, both directions, with block-bootstrap intervals.

    Includes the factors that did nothing. A chart of only the winners is how a
    multiple-comparison artefact gets on a poster."""
    rows = []
    for f in FACTORS:
        entry = {"factor": f}
        for name, mask in [("opt", d["optimistic"]), ("pes", d["pessimistic"])]:
            s = d[mask & d[f].notna()]
            if len(s) < 200 or s["block"].nunique() < 4:
                entry[name] = (np.nan, np.nan, np.nan)
                continue
            r, lo, hi, _ = block_boot_corr(s[f].values.astype(float),
                                           s["hit"].values.astype(int),
                                           s["block"].values, reps=reps)
            entry[name] = (r, lo, hi)
        entry["spread"] = (abs(entry["opt"][0] - entry["pes"][0])
                           if not (pd.isna(entry["opt"][0]) or pd.isna(entry["pes"][0]))
                           else -1)
        rows.append(entry)
    rows.sort(key=lambda e: e["spread"])

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    y = np.arange(len(rows))
    for i, e in enumerate(rows):
        for key, color, off in [("opt", BLUE, .16), ("pes", VERM, -.16)]:
            r, lo, hi = e[key]
            if pd.isna(r):
                continue
            ax.plot([lo, hi], [i + off, i + off], color=color, lw=2.2,
                    solid_capstyle="round", zorder=3)
            ax.plot([r], [i + off], marker="o", ms=10, color=color,
                    mec="white", mew=1.8, zorder=4)
    ax.axvline(0, color="#999999", lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[e["factor"]] for e in rows], fontsize=11.5)
    ax.set_ylim(-1.9, len(rows) - .35)
    ax.set_xlabel("correlation with whether the forecast came true\n"
                  "← higher factor, LESS accurate      "
                  "higher factor, MORE accurate →")
    ax.grid(axis="y", visible=False)

    # Series key on the empty row below the data, so it never collides with a
    # whisker. Colour identifies WHICH forecasts; the axis carries the sign.
    for yk, color, lbl in [(-.95, BLUE, "optimistic forecasts (improve / prices up)"),
                           (-1.5, VERM, "pessimistic forecasts (worsen / prices down)")]:
        ax.plot([-.42], [yk], marker="o", ms=10, color=color, mec="white",
                mew=1.8, clip_on=False, zorder=5)
        ax.text(-.395, yk, lbl, color=color, fontsize=11.5,
                fontweight="bold", ha="left", va="center")
    fig.tight_layout()
    fig.savefig(OUT / "figJ_macro_ledger.png", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}/figJ_macro_ledger.png")


# --- K. THE MODEL ----------------------------------------------------------
def fig_model():
    """Do the hit predictor's probabilities mean anything? The calibration
    curve, nothing else.

    Reads ladder_l1.json -- the full rung (claim + economy + interaction) on
    the same leave-one-3-year-period-out CV as the main model, refitted with an
    L1 (lasso) penalty at C = 0.5. Raw out-of-fold probabilities in decile
    bins, so the curve shows the model as deployed, not after a calibrator has
    been fitted on top of it. Reading a cached file means the chart cannot
    drift out of sync with the fit it claims to show."""
    import json
    p = Path("data/models/ladder_l1.json")
    res = json.loads(p.read_text()) if p.exists() else {}
    if "calibration" not in res:
        print("   [skipping figK: run `python src/make_l1_ladder_figure.py "
              "--refit` first, then recompute the L1 calibration curve]")
        return
    cal = res["calibration"]

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.4, ls="--", zorder=2)
    ax.plot(cal["predicted"], cal["actual"], color=BLUE, lw=2.4, marker="o",
            ms=9, mec="white", mew=1.6, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("probability the model assigned")
    ax.set_ylabel("share that actually came true")
    ax.text(.06, .90, "perfect calibration", color=MUTED, fontsize=10.5,
            rotation=39)

    fig.tight_layout()
    fig.savefig(OUT / "figK_hit_model.png", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}/figK_hit_model.png")
    print(f"   figK: calibration, L1 penalty, C = {res['C']}; "
          f"Brier {res['brier']:.3f}; n = {res['n']:,} forecasts across "
          f"{res['n_blocks']} held-out 3-year periods")


if __name__ == "__main__":
    d = load()
    fig_scissors(d)
    fig_ledger(d)
    fig_model()
