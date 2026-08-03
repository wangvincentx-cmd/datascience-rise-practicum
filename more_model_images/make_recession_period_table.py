"""Per-period performance of the recession model -- the table behind fig10.

Outputs into this folder:
  recession_period_table.png   spreadsheet-style table, AUC coloured against 0.50
  recession_period_table.csv   the same numbers, machine-readable

One model only: the five-press-series model that fig10 plots, forward-only from a
1930 start, so every decade is scored by a model that never saw it. The numbers
come from make_recession_model_time_figure, which gets them from
make_broad_vs_attention_figure.forward_only -- chart and table are one
computation and cannot disagree.

COLOUR, and a warning about it: red marks AUC ABOVE coin flip, blue marks BELOW,
as requested. That is the reverse of sheets_per_period.png and
backtest_table.png in this same folder, where red marks the below-0.50 rows
because a below-0.50 ranking is the bad outcome. Do not put the two conventions
on one poster -- a reader who has seen either will misread the other.

Run from the repo root:  python more_model_images/make_recession_period_table.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_recession_model_time_figure import predictions, decade_auc
# Imported last so the spreadsheet look (Arial, its own chrome) is the rcParams
# state that survives -- the figure module pulls in the fig8 style.
from make_sheets_style_tables import render_sheet, INK, OUT

ABOVE, BELOW = "#b3261e", "#1967d2"   # red above 0.50, blue below -- see docstring


def brier(s):
    return float(np.mean((s["p_onset"] - s["onset_within_12m"]) ** 2))


def table():
    d = predictions()
    rows = decade_auc(d).drop(columns=["x0", "x1"])
    rows["brier"] = [brier(d[(d.index.year >= lo) & (d.index.year <= hi)])
                     for lo, hi in [(1930, 1939), (1940, 1949),
                                    (1950, 1959), (1960, 1963)]]
    # The all-period row is the honest headline: the decade rows each rest on
    # 1-3 independent 3-year blocks and cannot carry a claim on their own.
    total = pd.DataFrame([{
        "period": f"{d.index.year.min()}-{d.index.year.max()} (all)",
        "months": len(d), "onsets": int(d["onset_within_12m"].sum()),
        "base_rate": d["onset_within_12m"].mean(),
        "auc": roc_auc_score(d["onset_within_12m"], d["p_onset"]),
        "brier": brier(d)}])
    return pd.concat([rows, total], ignore_index=True)


def main():
    t = table()
    t.to_csv(OUT / "recession_period_table.csv", index=False)
    # What a forecaster who only knew the base rate would score: p(1-p).
    flat = float(t.iloc[-1]["base_rate"] * (1 - t.iloc[-1]["base_rate"]))

    cells, colors = [], []
    for _, r in t.iterrows():
        cells.append([str(r["period"]), str(int(r["months"])),
                      str(int(r["onsets"])), f"{r['base_rate']:.3f}",
                      f"{r['auc']:.3f}", f"{r['brier']:.3f}"])
        # Only the AUC cell is coloured; the rest stays ink so the colour means
        # one thing and reads as a judgement on that number alone.
        colors.append([INK, INK, INK, INK,
                       ABOVE if r["auc"] > 0.5 else BELOW, INK])

    render_sheet(
        OUT / "recession_period_table.png",
        ["period", "months", "onsets", "base rate", "AUC", "Brier"],
        cells,
        widths=[1.32, 0.62, 0.62, 0.78, 0.66, 0.66],
        aligns=["l", "r", "r", "r", "r", "r"],
        colors=colors,
        title="Recession model, per-period performance (forward-only, 1930 start)",
        notes=["Red AUC beats coin flip (0.50); blue is below it, meaning the ranking runs backwards -- "
               "the model was most confident before the wrong months.",
               "Forward-only: refit every year on all prior years, never shown the year it predicts. "
               "Target is NBER onset within 12 months.",
               "Brier is the mean squared error of the probability -- LOWER is better; a flat 'always "
               f"the base rate' forecast would score {flat:.3f}.",
               "Decade rows rest on 37-100 months and only 1-3 independent 3-year blocks each: "
               "descriptive, not tests. The all-period row is the claim."])

    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {OUT}/recession_period_table.png (+ .csv)")


if __name__ == "__main__":
    main()
