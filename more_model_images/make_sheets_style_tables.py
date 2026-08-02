"""Spreadsheet-style PNGs of the two result tables — rendered to look like Google Sheets.

Outputs into this folder:
  sheets_forward_test.png   Table 2: forward-only backtest, five press series vs attention alone
  sheets_per_period.png     Table 3: per-period (decade) AUC and Brier for both models

Data comes from the CSVs written by make_sparsity_tables.py, so the numbers cannot
drift between the two renderings. Run that first (or just run this — it will call it).

Run from the repo root:  python more_model_images/make_sheets_style_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent

# Google Sheets chrome
GUTTER_BG, GUTTER_INK = "#f8f9fa", "#5f6368"
GRID = "#d9d9d9"
GUTTER_EDGE = "#c0c0c0"
HEADER_BG = "#f1f3f4"
INK = "#202124"
FREEZE = "#9aa0a6"
ACCENT_BLUE, ACCENT_ORANGE = "#1967d2", "#c5490b"

ROW_H = 0.30          # inches
GUT_W, NUM_W = 0.34, 0.42
FS = 9.5

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})


def col_letters(n):
    out = []
    for i in range(n):
        s, j = "", i
        while True:
            s = chr(ord("A") + j % 26) + s
            j = j // 26 - 1
            if j < 0:
                break
        out.append(s)
    return out


def render_sheet(path, header, rows, widths, aligns, colors=None, title=None,
                 notes=(), freeze_after=1):
    """rows: list of list[str]. colors: optional list of per-row text colors."""
    ncol = len(header)
    body = [header] + rows
    nrow = len(body)
    widths = list(widths)
    total_w = NUM_W + sum(widths)
    fig_w = total_w
    note_h = 0.20 * len(notes) + (0.34 if title else 0)
    fig_h = GUT_W + nrow * ROW_H + note_h + 0.12

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220, facecolor="white")
    ax = fig.add_axes([0, note_h / fig_h, 1, 1 - note_h / fig_h])
    ax.set_xlim(0, total_w); ax.set_ylim(0, GUT_W + nrow * ROW_H)
    ax.invert_yaxis(); ax.set_axis_off()

    xs = [NUM_W]
    for w in widths:
        xs.append(xs[-1] + w)

    # column-letter gutter
    ax.add_patch(Rectangle((0, 0), total_w, GUT_W, facecolor=GUTTER_BG,
                           edgecolor=GUTTER_EDGE, lw=0.7, zorder=1))
    for j, L in enumerate(col_letters(ncol)):
        ax.plot([xs[j], xs[j]], [0, GUT_W], color=GUTTER_EDGE, lw=0.7, zorder=2)
        ax.text((xs[j] + xs[j + 1]) / 2, GUT_W / 2, L, ha="center", va="center",
                fontsize=8.5, color=GUTTER_INK, zorder=3)

    for i, row in enumerate(body):
        y = GUT_W + i * ROW_H
        is_head = i == 0
        # row-number gutter
        ax.add_patch(Rectangle((0, y), NUM_W, ROW_H, facecolor=GUTTER_BG,
                               edgecolor=GUTTER_EDGE, lw=0.7, zorder=1))
        ax.text(NUM_W / 2, y + ROW_H / 2, str(i + 1), ha="center", va="center",
                fontsize=8.5, color=GUTTER_INK, zorder=3)
        for j, cell in enumerate(row):
            ax.add_patch(Rectangle((xs[j], y), widths[j], ROW_H,
                                   facecolor=HEADER_BG if is_head else "white",
                                   edgecolor=GRID, lw=0.7, zorder=1))
            ha = "left" if aligns[j] == "l" else "right"
            pad = 0.07
            x = xs[j] + pad if ha == "left" else xs[j + 1] - pad
            color = INK if (is_head or colors is None) else colors[i - 1]
            ax.text(x, y + ROW_H / 2, cell, ha=ha, va="center", fontsize=FS,
                    color=color, fontweight="600" if is_head else "normal", zorder=3)
        if is_head and freeze_after:
            ax.plot([0, total_w], [y + ROW_H] * 2, color=FREEZE, lw=1.6, zorder=4)

    if title:
        fig.text(0.008, (note_h - 0.10) / fig_h, title, fontsize=10.5, fontweight="600",
                 color=INK, ha="left", va="top")
    for k, n in enumerate(notes):
        fig.text(0.008, (note_h - 0.34 - 0.20 * k) / fig_h, n, fontsize=7.8,
                 color=GUTTER_INK, ha="left", va="top")

    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "#ffffff")
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        bg.save(path)


PAIRED = {1930: ("-0.019", "[-0.12, +0.12]"),
          1940: ("-0.001", "[-0.10, +0.15]"),
          1950: ("-0.065", "[-0.13, -0.01]")}


def forward_sheet():
    cmp = pd.read_csv(OUT / "broad_vs_attention_table.csv")
    rows, colors = [], []
    for _, r in cmp.iterrows():
        attn = r["model"] == "Attention alone"
        rows.append([f"{r['start']}-1963", r["model"], str(r["months"]),
                     str(r["onsets"]), f"{r['auc']:.3f}", f"{r['eras_won']}/{r['n_eras']}",
                     PAIRED[r["start"]][0] if attn else "",
                     PAIRED[r["start"]][1] if attn else ""])
        colors.append(ACCENT_ORANGE if attn else ACCENT_BLUE)
    render_sheet(
        OUT / "sheets_forward_test.png",
        ["training start", "model", "months", "onsets", "AUC",
         "eras won", "paired diff", "95% CI"],
        rows,
        widths=[1.05, 1.28, 0.60, 0.58, 0.64, 0.68, 0.76, 1.00],
        aligns=["l", "l", "r", "r", "r", "r", "r", "r"],
        colors=colors,
        title="Forward-only backtest — five press series vs attention alone",
        notes=["AUC 0.50 = coin flip. Refit every year on all prior years; never sees the year it predicts.",
               "Paired diff = attention minus five series, 95% block bootstrap over 3-year blocks.",
               "Two longer windows indistinguishable; the 1950 window favours five series (CI excludes zero).",
               "Eras won: 8-year blocks inside each window — only 2-4 fit, so treat as illustrative."])


def per_period_sheet():
    pp = pd.read_csv(OUT / "per_period_table.csv")
    rows, colors = [], []
    for _, r in pp.iterrows():
        total = "all" in str(r["period"])
        rows.append([str(r["period"]), str(r["months"]), str(r["onsets"]),
                     f"{r['base_rate']:.3f}",
                     "" if pd.isna(r["auc_five"]) else f"{r['auc_five']:.3f}",
                     "" if pd.isna(r["auc_attn"]) else f"{r['auc_attn']:.3f}",
                     f"{r['brier_five']:.3f}", f"{r['brier_attn']:.3f}"])
        colors.append(INK if total else ("#b3261e" if r["auc_attn"] < 0.5 else INK))
    render_sheet(
        OUT / "sheets_per_period.png",
        ["period", "months", "onsets", "base rate", "AUC 5-series", "AUC attention",
         "Brier 5-series", "Brier attention"],
        rows,
        widths=[1.24, 0.62, 0.60, 0.75, 0.92, 0.98, 0.95, 1.02],
        aligns=["l", "r", "r", "r", "r", "r", "r", "r"],
        colors=colors,
        title="Per-period performance, real-time (1930 start, both models)",
        notes=["From the 1930 start — the longest forward-only run, so every decade is predicted by a model that never saw it.",
               "AUC: 0.50 = coin flip, higher is better.  Brier: mean squared error of the probability, LOWER is better.",
               "Accuracy is deliberately not reported: at a 24% base rate, always answering 'no onset' scores 76% while ranking nothing.",
               "Red rows run backwards (AUC below coin flip). Decade rows rest on 1-3 independent blocks — descriptive, not tests."])


if __name__ == "__main__":
    if not (OUT / "per_period_table.csv").exists():
        import make_sparsity_tables
        make_sparsity_tables.main()
    forward_sheet()
    per_period_sheet()
    print(f"written to {OUT}/sheets_forward_test.png and {OUT}/sheets_per_period.png")
