"""Figures for the coverage -> recession-onset model (1900-1963, LOC only).

Outputs into this folder:
  conditional_mapping.png   coverage change -> P(recession starts within 12m), with block-bootstrap band
  backtest_table.png        backtest results as a table: forward-only, and rotating by period
  backtest_table.csv        the same numbers, machine-readable

Run from the repo root:  python more_model_images/make_more_model_figures.py

Target is NBER onset dates only; no outcome information enters any feature.
"""
import sys, warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from truth_data import NBER_RECESSIONS

OUT = Path(__file__).parent
H = 12                       # forecast horizon, months
FULL = pd.period_range("1900-01", "1963-12", freq="M")
FEAT = ["d_attention_12m"]

# palette (light surface)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#ffffff"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
})


def build():
    """Monthly press features + 'recession starts within 12m' target, expansion months only."""
    pi = pd.read_csv("data/scored/press_index.csv")
    pi["month"] = pd.PeriodIndex(pi["month"], freq="M")
    pi = pi.sort_values("month").set_index("month").reindex(FULL)

    att = pi["attention"].rolling(12, min_periods=6).mean()
    X = pd.DataFrame({"d_attention_12m": att - att.shift(12)}, index=FULL)

    rec = pd.Series(0, index=FULL, dtype=int)
    peak = pd.Series(0, index=FULL, dtype=int)
    for p, t in NBER_RECESSIONS:
        p, t = pd.Period(p, "M"), pd.Period(t, "M")
        rec.loc[p:t] = 1
        peak.loc[p] = 1

    y = pd.Series(0, index=FULL, dtype=int)
    for i in range(len(FULL)):
        w = FULL[i + 1:i + 1 + H]
        if len(w) and peak.reindex(w).fillna(0).sum() > 0:
            y.iloc[i] = 1

    keep = (rec == 0) & X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    return X[keep], y[keep], rec


X, y, rec = build()
yr = pd.Series([m.year for m in X.index], index=X.index)


def fit_predict(tr_mask, te_mask):
    mu, sd = X[tr_mask].mean(), X[tr_mask].std()
    m = LogisticRegression(C=1.0, max_iter=2000).fit(((X[tr_mask] - mu) / sd)[FEAT], y[tr_mask])
    return m.predict_proba(((X[te_mask] - mu) / sd)[FEAT])[:, 1]



def _flatten(path):
    """Save as RGB on white. RGBA PNGs can import with a black border in Slides/Keynote."""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "#ffffff")
        bg.paste(im, mask=im.split()[-1])
        bg.save(path)
    else:
        im.convert("RGB").save(path)


# ------------------------------------------------------------------ figure 1
def conditional_mapping():
    d = X["d_attention_12m"]
    Z = (X - X.mean()) / X.std()
    m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[FEAT], y)
    # p10-p90 only: beyond this the curve extrapolates into near-empty territory
    # (the 98th percentile is supported by 2 months of data)
    grid = np.linspace(np.percentile(d, 10), np.percentile(d, 90), 120)
    gz = ((grid - d.mean()) / d.std()).reshape(-1, 1)

    # block bootstrap over 3-year eras: months inside an era are not independent
    eras = (yr.values - 1900) // 3
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(3000):
        pick = rng.choice(np.unique(eras), len(np.unique(eras)), replace=True)
        idx = np.concatenate([np.where(eras == u)[0] for u in pick])
        yy = y.values[idx]
        if yy.sum() < 5 or yy.sum() == len(idx):
            continue
        mb = LogisticRegression(C=1.0, max_iter=2000).fit(Z.values[idx][:, [0]], yy)
        boot.append(mb.predict_proba(gz)[:, 1])
    lo, hi = np.percentile(np.array(boot), [2.5, 97.5], axis=0)
    p = m.predict_proba(gz)[:, 1]

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=220)
    ax.fill_between(grid, lo, hi, color=BLUE, alpha=0.16, lw=0,
                    label="95% CI (3-year block bootstrap)")
    ax.plot(grid, p, color=BLUE, lw=2.4, label="fitted probability", zorder=3)
    ax.axhline(y.mean(), color=MUTED, ls=(0, (4, 4)), lw=1.4, zorder=2)
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    ax.text(grid[0], y.mean() + .022, f"base rate {y.mean():.0%} — guessing",
            color=MUTED, fontsize=9, va="bottom")

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.01, .2)])
    ax.set_xlabel("change in economic coverage over 12 months  (forecasts per 100 pages)",
                  labelpad=20)
    ax.set_ylabel("P(recession starts within 12 months)")
    ax.annotate("shown over the 10th-90th percentile of observed coverage change; "
                "beyond this range the curve extrapolates",
                (0, 1.02), xycoords="axes fraction", color=MUTED, fontsize=8.5,
                annotation_clip=False)
    ax.annotate("<-- coverage falling", (grid[0], -.105), xycoords=("data", "axes fraction"),
                color=MUTED, fontsize=9, annotation_clip=False)
    ax.annotate("coverage rising -->", (grid[-1], -.105), xycoords=("data", "axes fraction"),
                color=MUTED, fontsize=9, ha="right", annotation_clip=False)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "conditional_mapping.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    _flatten(OUT / "conditional_mapping.png")
    print(f"  conditional_mapping.png   p10={np.interp(np.percentile(d,10),grid,p):.3f} "
          f"p90={np.interp(np.percentile(d,90),grid,p):.3f} base={y.mean():.3f}")


# ------------------------------------------------------------------ figure 2
def backtest_table():
    """Backtest results as a table: forward-only (real-time) and rotating by period."""
    fwd = []
    for start in (1930, 1940, 1950):
        P, O = [], []
        for cut in range(start, 1964):
            tr, te = yr < cut, yr == cut
            if tr.sum() < 96 or te.sum() == 0 or y[tr].nunique() < 2:
                continue
            P += list(fit_predict(tr, te)); O += list(y[te])
        P, O = np.array(P), np.array(O)
        fwd.append((f"{start}-1963", len(O), int(O.sum()), roc_auc_score(O, P)))

    rot = []
    for s0 in range(1900, 1964, 8):
        te = (yr >= s0) & (yr <= s0 + 7)
        if y[te].nunique() < 2 or y[~te].nunique() < 2:
            continue
        rot.append((f"{s0}-{s0+7}", int(te.sum()), int(y[te].sum()),
                    roc_auc_score(y[te], fit_predict(~te, te))))

    pd.DataFrame([("forward-only",) + r for r in fwd] + [("rotating",) + r for r in rot],
                 columns=["scheme", "period", "months", "onsets", "auc"]
                 ).to_csv(OUT / "backtest_table.csv", index=False)

    rows = ([("head", "Forward-only \u2014 trained only on earlier years, refit every year", "", "", ""),
             ("cols", "period predicted", "months", "onsets", "AUC")]
            + [("d",) + (a, str(b), str(c), f"{d:.3f}") for a, b, c, d in fwd]
            + [("gap", "", "", "", ""),
               ("head", "Rotating by period \u2014 each era held out, model refit on the rest", "", "", ""),
               ("cols", "era held out", "months", "onsets", "AUC")]
            + [("d",) + (a, str(b), str(c), f"{d:.3f}") for a, b, c, d in rot])

    RH, TOP = 0.30, 0.88
    fig_h = TOP + RH * len(rows) + 1.15
    fig, ax = plt.subplots(figsize=(6.9, fig_h), dpi=220)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    CX = (0.015, 0.60, 0.755, 0.945)
    yv, step = 1 - TOP / fig_h, RH / fig_h

    ax.text(0, 1 - 0.10 / fig_h, "Backtest results", fontsize=13, fontweight="600",
            color=INK, va="top")
    ax.text(0, 1 - 0.28 / fig_h,
            "coverage change -> recession starts within 12 months  \u00b7  AUC 0.50 = coin flip",
            fontsize=8.5, color=MUTED, va="top")

    for kind, c0, c1, c2, c3 in rows:
        if kind == "gap":
            yv -= step * 0.55
            continue
        if kind == "head":
            ax.plot([0, 1], [yv + step * .40] * 2, color="#c3c2b7", lw=1.1)
            ax.text(CX[0], yv - step * .10, c0, fontsize=9.5, fontweight="600",
                    color=INK, va="center")
        elif kind == "cols":
            for x, t, ha in zip(CX, (c0, c1, c2, c3), ("left", "right", "right", "right")):
                ax.text(x, yv - step * .10, t, fontsize=8.5, color=MUTED, va="center", ha=ha)
            ax.plot([0, 1], [yv - step * .52] * 2, color=GRID, lw=1)
        else:
            bad = float(c3) < 0.5
            ax.text(CX[0], yv - step * .10, c0, fontsize=9.5, color=INK2, va="center")
            for x, t in zip(CX[1:3], (c1, c2)):
                ax.text(x, yv - step * .10, t, fontsize=9.5, color=INK2,
                        va="center", ha="right")
            ax.text(CX[3], yv - step * .10, c3, fontsize=9.5,
                    color=RED if bad else INK, fontweight="600", va="center", ha="right")
            if bad:
                ax.text(1.0, yv - step * .10, "  <- runs backwards", fontsize=8,
                        color=RED, va="center", ha="left")
            ax.plot([0, 1], [yv - step * .52] * 2, color=GRID, lw=0.7)
        yv -= step

    med = np.median([r[3] for r in rot])
    ax.text(0, yv - step * .35,
            f"Rotating: median {med:.2f}, {sum(r[3] > .5 for r in rot)} of {len(rot)} eras beat "
            f"coin flip. It trains on future data to predict the past,\nso it tests whether the "
            f"relationship is stable across eras \u2014 not whether it could have been used at the "
            f"time.\nForward-only is the real-time number, and it is the lower one.",
            fontsize=8.3, color=MUTED, va="top", linespacing=1.6)

    fig.savefig(OUT / "backtest_table.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    _flatten(OUT / "backtest_table.png")
    print(f"  backtest_table.png/.csv   forward-only {fwd[0][3]:.3f}/{fwd[1][3]:.3f}/{fwd[2][3]:.3f}"
          f"  rotating median {med:.3f}, {sum(r[3] > .5 for r in rot)}/{len(rot)} above 0.5")


if __name__ == "__main__":
    print(f"eligible expansion months {len(X)}, onsets {int(y.sum())} ({y.mean():.1%})")
    conditional_mapping()
    backtest_table()
    print(f"written to {OUT}/")
