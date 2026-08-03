"""Broad 5-series press index vs attention alone, under the forward-only backtest.

Outputs into this folder:
  broad_vs_attention.png   grouped bars, forward-only AUC by training start, block-bootstrap CIs
  broad_vs_attention.csv   the same numbers, machine-readable

Both models are built from ALL-SCOPE claims, use identical d12 features
(12-month moving average minus the same average 12 months earlier), the same
logistic regression (L2, C=1), and the same forward-only backtest: trained only
on years before the test year, refit every year, never seeing the year it predicts.

Run from the repo root:  python more_model_images/make_broad_vs_attention_figure.py

Target is NBER onset dates only; no outcome information enters any feature.
"""
import json
import sys
import warnings
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
SERIES = ["net_direction", "disagreement", "hedge_rate", "share_expert", "attention"]
STARTS = (1930, 1940, 1950)
N_BOOT = 4000

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"

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
    """All-scope monthly press series + 'recession starts within 12m', expansion months only."""
    claims = pd.DataFrame([json.loads(l) for l
                           in open("data/claims/claims_monthly.jsonl", encoding="utf-8")
                           if l.strip()])
    claims["month"] = pd.to_datetime(claims["date"], errors="coerce").dt.to_period("M")
    claims = claims[claims["month"].notna()]

    # page denominator: the corpus is gitignored, so read it back off the press index
    pi = pd.read_csv("data/scored/press_index.csv")
    pi["month"] = pd.PeriodIndex(pi["month"], freq="M")
    pages = pi.set_index("month")["n_pages"].reindex(FULL)

    claims["_imp"] = claims["direction"].eq("improve").astype(int)
    claims["_wor"] = claims["direction"].eq("worsen").astype(int)
    claims["_hed"] = claims.get("confidence", "").astype(str).eq("hedged").astype(int)
    claims["_exp"] = claims.get("voice", "").astype(str).eq("expert").astype(int)
    g = claims.groupby("month").agg(imp=("_imp", "sum"), wor=("_wor", "sum"),
                                    hed=("_hed", "mean"), exp=("_exp", "mean"),
                                    n=("_imp", "size"))
    directional = (g["imp"] + g["wor"]).replace(0, np.nan)
    net = ((g["imp"] - g["wor"]) / directional).reindex(FULL)
    raw = pd.DataFrame({
        "net_direction": net,
        "disagreement": 1 - net.abs(),
        "hedge_rate": g["hed"].reindex(FULL),
        "share_expert": g["exp"].reindex(FULL),
        "attention": 100.0 * g["n"].reindex(FULL).fillna(0) / pages,
    }, index=FULL)

    X = pd.concat([(lambda s: s - s.shift(12))(raw[c].rolling(12, min_periods=6).mean()).rename(c)
                   for c in SERIES], axis=1)

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
    return X[keep], y[keep]


_DATA = None


def data():
    """Design matrix, target and year index -- built once, on first use.

    Lazy rather than module-level so other figure scripts can import
    forward_only() and get the same model without paying build() at import
    time, and without a second copy of it drifting out of step with this one."""
    global _DATA
    if _DATA is None:
        X, y = build()
        _DATA = (X, y, pd.Series([m.year for m in X.index], index=X.index))
    return _DATA


def forward_only(cols, start):
    """Out-of-fold predictions: train on years < cut, predict cut, march forward."""
    X, y, yr = data()
    Xf = X[cols]
    idx, pred = [], []
    for cut in range(start, 1964):
        tr, te = yr < cut, yr == cut
        if tr.sum() < 96 or te.sum() == 0 or y[tr].nunique() < 2:
            continue
        mu, sd = Xf[tr].mean(), Xf[tr].std().replace(0, 1)
        m = LogisticRegression(C=1.0, max_iter=5000).fit((Xf[tr] - mu) / sd, y[tr])
        pred += list(m.predict_proba((Xf[te] - mu) / sd)[:, 1])
        idx += list(Xf.index[te])
    i = pd.PeriodIndex(idx, freq="M")
    return pd.Series(pred, index=i), y.reindex(i)


def block_ci(p, truth, n_boot=N_BOOT, block_years=3, seed=0):
    """Block bootstrap over 3-year blocks -- forecasts within an era are not independent."""
    years = np.array([m.year for m in p.index])
    blocks = (years - years.min()) // block_years
    uniq, rng, out = np.unique(blocks), np.random.default_rng(seed), []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([np.where(blocks == b)[0] for b in pick])
        t = truth.values[sel]
        if t.min() == t.max():
            continue
        out.append(roc_auc_score(t, p.values[sel]))
    return np.percentile(out, [2.5, 97.5])


MODELS = [("Five press series", SERIES, BLUE),
          ("Attention alone", ["attention"], ORANGE)]


def main():
    rows = []
    for start in STARTS:
        for name, cols, _ in MODELS:
            p, truth = forward_only(cols, start)
            lo, hi = block_ci(p, truth)
            rows.append({"model": name, "n_features": len(cols), "start": start,
                         "months": len(truth), "onsets": int(truth.sum()),
                         "auc": roc_auc_score(truth, p), "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "broad_vs_attention.csv", index=False)

    # paired difference on the shared prediction set, per start year
    diff_txt = []
    for start in STARTS:
        pb, tb = forward_only(SERIES, start)
        pa, _ = forward_only(["attention"], start)
        common = pb.index.intersection(pa.index)
        pb, pa, t = pb[common], pa[common], tb[common]
        years = np.array([m.year for m in common])
        blocks = (years - years.min()) // 3
        uniq, rng, d = np.unique(blocks), np.random.default_rng(0), []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([np.where(blocks == b)[0] for b in pick])
            tt = t.values[sel]
            if tt.min() == tt.max():
                continue
            d.append(roc_auc_score(tt, pa.values[sel]) - roc_auc_score(tt, pb.values[sel]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        diff_txt.append(f"{start}– Δ{np.mean(d):+.3f} [{lo:+.2f}, {hi:+.2f}]")

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=220)
    W = 0.34
    xs = np.arange(len(STARTS))

    for k, (name, cols, color) in enumerate(MODELS):
        sub = res[res["model"] == name].set_index("start").loc[list(STARTS)]
        pos = xs + (k - 0.5) * (W + 0.02)          # 2px-equivalent gap between adjacent bars
        ax.bar(pos, sub["auc"], W, color=color, edgecolor=SURFACE, linewidth=1.4,
               zorder=3, label=f"{name}  ({sub['n_features'].iloc[0]} feature"
                               f"{'s' if len(cols) > 1 else ''})")
        ax.vlines(pos, sub["ci_lo"], sub["ci_hi"], color=INK2, lw=1.3, zorder=4)
        ax.hlines(sub["ci_lo"], pos - 0.05, pos + 0.05, color=INK2, lw=1.3, zorder=4)
        ax.hlines(sub["ci_hi"], pos - 0.05, pos + 0.05, color=INK2, lw=1.3, zorder=4)
        for xp, v in zip(pos, sub["auc"]):
            ax.text(xp, v - 0.022, f"{v:.3f}", ha="center", va="top", fontsize=9,
                    color=SURFACE, fontweight="600", zorder=5)

    ax.axhline(0.5, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(len(STARTS) - 0.42, 0.508, "coin flip", fontsize=8.5, color=MUTED, va="bottom", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}–1963" for s in STARTS])
    ax.set_xlabel("years predicted (model refit each year on all prior years)")
    ax.set_ylabel("forward-only ROC-AUC")
    ax.set_ylim(0.35, 1.0)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.set_title("One press feature matches five in the two longer test windows",
                 fontsize=12.5, fontweight="600", color=INK, loc="left", pad=26)
    ax.text(0, 1.055, "Recession onset within 12 months · forward-only backtest, "
                      "never trained on the year it predicts",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom")

    ax.legend(frameon=False, loc="upper left", fontsize=9, handlelength=1.1,
              handleheight=1.1, borderaxespad=0.6)

    n = res[res["start"] == 1940].iloc[0]
    fig.text(0.012, -0.06,
             "Whiskers: 95% block bootstrap over 3-year blocks — forecasts within an era share "
             "wire copy and one macro reality.\nPaired difference (attention − five series): "
             + ";  ".join(diff_txt)
             + ".\nThe two longer windows are indistinguishable — four series can be dropped "
               "without measurable loss. In the shortest\nwindow (137 months, ~5 independent "
               "blocks) the five-series model is ahead, so the equivalence is not universal.",
             fontsize=8.2, color=MUTED, ha="left", va="top", linespacing=1.55,
             transform=fig.transFigure)

    fig.savefig(OUT / "broad_vs_attention.png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    from PIL import Image
    im = Image.open(OUT / "broad_vs_attention.png")
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "#ffffff")
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        bg.save(OUT / "broad_vs_attention.png")

    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\npaired diffs: " + " | ".join(diff_txt))
    print(f"\nwritten to {OUT}/broad_vs_attention.png (+ .csv)")


if __name__ == "__main__":
    main()
