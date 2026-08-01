"""Two tables for the press-index model, rendered as poster-ready PNGs.

Outputs into this folder:
  lasso_path_table.png / .csv       which of the five press series survive L1 at each C
  broad_vs_attention_table.png/.csv forward-only AUC by training start, with eras won

Both are built from ALL-SCOPE claims with identical d12 features (12-month moving
average minus the same average 12 months earlier) and the same forward-only
backtest as make_broad_vs_attention_figure.py.

Run from the repo root:  python more_model_images/make_sparsity_tables.py

Target is NBER onset dates only; no outcome information enters any feature.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from make_broad_vs_attention_figure import (  # same data, same features, one definition
    X, y, yr, SERIES, STARTS, forward_only, OUT,
    INK, INK2, MUTED, GRID, SURFACE, BLUE, ORANGE,
)

LABEL = {"net_direction": "net direction", "disagreement": "disagreement",
         "hedge_rate": "hedge rate", "share_expert": "expert share",
         "attention": "attention"}
CGRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
ERA = 8


# ---------------------------------------------------------------- table 1
def lasso_path():
    """In-sample L1 fit on all rows -- a DESCRIPTION of what the penalty keeps."""
    Z = (X - X.mean()) / X.std()
    rows = []
    for C in CGRID:
        m = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=20000).fit(Z, y)
        co = dict(zip(SERIES, m.coef_[0]))
        rows.append({"C": C, **co, "nonzero": int(sum(abs(v) > 1e-8 for v in co.values()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- table 2
def eras_won(start):
    """Split this start's out-of-fold months into 8-year eras; who ranks better in each."""
    pb, truth = forward_only(SERIES, start)
    pa, _ = forward_only(["attention"], start)
    common = pb.index.intersection(pa.index)
    pb, pa, t = pb[common], pa[common], truth[common]
    years = np.array([m.year for m in common])
    wins_a = wins_b = ties = 0
    for e0 in range(years.min() // ERA * ERA, years.max() + 1, ERA):
        sel = (years >= e0) & (years < e0 + ERA)
        if sel.sum() < 12 or t.values[sel].min() == t.values[sel].max():
            continue
        aa = roc_auc_score(t.values[sel], pa.values[sel])
        ab = roc_auc_score(t.values[sel], pb.values[sel])
        if abs(aa - ab) < 1e-9:
            ties += 1
        elif aa > ab:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a, wins_b, ties


def comparison():
    rows = []
    for start in STARTS:
        wa, wb, ties = eras_won(start)
        n_era = wa + wb + ties
        for name, cols in (("Five press series", SERIES), ("Attention alone", ["attention"])):
            p, t = forward_only(cols, start)
            rows.append({"start": start, "model": name, "k": len(cols),
                         "months": len(t), "onsets": int(t.sum()),
                         "auc": roc_auc_score(t, p),
                         "eras_won": (wa if cols == ["attention"] else wb), "n_eras": n_era})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rendering
def render(path, title, subtitle, colnames, colx, colalign, body, footnote, width=7.6):
    """body: list of (kind, cells...) where kind is 'row' | 'rule' | 'group'."""
    RH, TOP = 0.30, 0.86
    fig_h = TOP + RH * (len(body) + 1) + 1.0
    fig, ax = plt.subplots(figsize=(width, fig_h), dpi=220)
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    yv, step = 1 - TOP / fig_h, RH / fig_h

    ax.text(0, 1 - 0.10 / fig_h, title, fontsize=13, fontweight="600", color=INK, va="top")
    ax.text(0, 1 - 0.30 / fig_h, subtitle, fontsize=8.5, color=MUTED, va="top")

    for x, t, ha in zip(colx, colnames, colalign):
        ax.text(x, yv - step * .10, t, fontsize=8.5, color=MUTED, va="center", ha=ha)
    ax.plot([0, 1], [yv - step * .52] * 2, color=GRID, lw=1)
    yv -= step

    for kind, *cells in body:
        if kind == "rule":
            ax.plot([0, 1], [yv + step * .40] * 2, color="#c3c2b7", lw=1.1)
            yv -= step * 0.30
            continue
        for x, cell, ha in zip(colx, cells, colalign):
            txt, color, weight = cell
            ax.text(x, yv - step * .10, txt, fontsize=9.5, color=color,
                    fontweight=weight, va="center", ha=ha)
        ax.plot([0, 1], [yv - step * .52] * 2, color=GRID, lw=0.7)
        yv -= step

    ax.text(0, yv - step * .30, footnote, fontsize=8.2, color=MUTED,
            va="top", linespacing=1.6)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "#ffffff")
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        bg.save(path)


def write_combined_csv(lp, cmp):
    """Both tables in one CSV, section-blocked -- File > Import into Google Sheets."""
    lines = []

    def block(title, header, rows, notes):
        lines.append([title])
        lines.append(header)
        lines.extend(rows)
        for n in notes:
            lines.append([n])
        lines.append([])

    block("TABLE 1 - Lasso path: which press series survive at each penalty",
          ["penalty"] + [LABEL[s] for s in SERIES] + ["kept"],
          [[f"C = {r['C']:g}"] + [("" if abs(r[s]) < 1e-8 else f"{r[s]:.3f}") for s in SERIES]
           + [f"{int(r['nonzero'])}/5"] for _, r in lp.iterrows()],
          ["Blank = coefficient driven to exactly zero.",
           "In-sample L1 fit on all 498 expansion months; standardized coefficients.",
           "A description of what the penalty keeps, not an accuracy claim.",
           "Hedge rate and net direction go first, then expert share. At C = 0.02 only attention "
           "retains real weight (0.167); disagreement is nonzero but negligible (0.002).",
           "Disagreement carries the largest coefficient under weak penalty on ALL-SCOPE claims; "
           "it does not when the series are national-only."])

    block("TABLE 2 - Forward-only backtest: five press series vs attention alone",
          ["training start", "model", "features", "months", "onsets", "AUC",
           "eras won", "paired diff vs other model", "95% CI"],
          [[f"{r['start']}-1963", r["model"], r["k"], r["months"], r["onsets"],
            f"{r['auc']:.3f}", f"{r['eras_won']}/{r['n_eras']}",
            (PAIRED[r["start"]][0] if r["model"] == "Attention alone" else ""),
            (PAIRED[r["start"]][1] if r["model"] == "Attention alone" else "")]
           for _, r in cmp.iterrows()],
          ["AUC 0.50 = coin flip. Model refit every year on all prior years; never sees the year "
           "it predicts.",
           "Paired difference = attention minus five series, 95% block bootstrap over 3-year blocks.",
           "The two longer windows are indistinguishable - four series drop out without measurable "
           "loss; the shortest favours five.",
           "Eras won: 8-year blocks inside each window. Only 2-4 fit, so treat the column as "
           "illustrative."])

    path = OUT / "press_model_tables.csv"
    pd.DataFrame(lines).to_csv(path, index=False, header=False)
    return path


PAIRED = {1930: ("-0.019", "[-0.12, +0.12]"),
          1940: ("-0.001", "[-0.10, +0.15]"),
          1950: ("-0.065", "[-0.13, -0.01]")}


def main():
    # ---- table 1
    lp = lasso_path()
    lp.to_csv(OUT / "lasso_path_table.csv", index=False)
    colx = (0.015, 0.30, 0.475, 0.635, 0.80, 0.925, 1.0)
    align = ("left", "right", "right", "right", "right", "right", "right")
    body = []
    for _, r in lp.iterrows():
        cells = [(f"C = {r['C']:g}", INK2, "normal")]
        for s in SERIES:
            v = r[s]
            if abs(v) < 1e-8:
                cells.append(("—", "#c3c2b7", "normal"))
            else:
                last = r["nonzero"] == 1
                cells.append((f"{v:.3f}", ORANGE if s == "attention" else INK2,
                              "600" if (s == "attention" and last) else "normal"))
        cells.append((f"{int(r['nonzero'])}/5", INK, "600"))
        body.append(("row", *cells))
    render(OUT / "lasso_path_table.png",
           "Attention is the last series to hold a substantive coefficient",
           "L1 logistic regression, standardized coefficients · elimination order as the penalty tightens",
           ("penalty", *[LABEL[s] for s in SERIES], "kept"), colx, align, body,
           "In-sample fit on all 498 expansion months — a description of what the penalty keeps, not an "
           "accuracy claim.\nOut-of-sample skill does not improve under L1 (see broad_vs_attention_table); "
           "the result is sparsity, not a gain.\nHedge rate and net direction go first, then expert share. "
           "At C = 0.02 only attention retains real weight (0.167);\ndisagreement is nonzero but negligible "
           "(0.002). Note disagreement carries the largest coefficient under weak\npenalty — on all-scope "
           "claims it competes with attention, which it does not when the series are national-only.",
           width=8.6)

    # ---- table 2
    cmp = comparison()
    cmp.to_csv(OUT / "broad_vs_attention_table.csv", index=False)
    colx = (0.015, 0.46, 0.60, 0.73, 0.87, 1.0)
    align = ("left", "right", "right", "right", "right", "right")
    body = []
    for i, start in enumerate(STARTS):
        if i:
            body.append(("rule",))
        sub = cmp[cmp["start"] == start]
        for _, r in sub.iterrows():
            attn = r["model"] == "Attention alone"
            col = ORANGE if attn else BLUE
            body.append(("row",
                         (f"{start}–1963   {r['model']}", col, "600" if attn else "normal"),
                         (str(r["k"]), INK2, "normal"),
                         (str(r["months"]), INK2, "normal"),
                         (str(r["onsets"]), INK2, "normal"),
                         (f"{r['auc']:.3f}", INK, "600"),
                         (f"{r['eras_won']}/{r['n_eras']}", INK2, "normal")))
    render(OUT / "broad_vs_attention_table.png",
           "One feature matches five out of sample",
           "Forward-only backtest · refit every year on all prior years · AUC 0.50 = coin flip",
           ("training start · model", "features", "months", "onsets", "AUC", "eras won"),
           colx, align, body,
           "Paired block-bootstrap difference (attention − five series): 1930– −0.019 [−0.12, +0.12];\n"
           "1940– −0.001 [−0.10, +0.15];  1950– −0.065 [−0.13, −0.01]. The two longer windows are\n"
           "indistinguishable — four series drop out without measurable loss; the shortest favours five.\n"
           "Eras won: 8-year blocks inside each window. Only 2–4 fit, so treat the column as illustrative.")

    write_combined_csv(lp, cmp)

    print(lp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(cmp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwritten to {OUT}/lasso_path_table.png, {OUT}/broad_vs_attention_table.png (+ .csv)")


if __name__ == "__main__":
    main()
