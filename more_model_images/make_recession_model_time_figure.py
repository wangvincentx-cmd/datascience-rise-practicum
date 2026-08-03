"""The recession model over time: what it said each month, and where it worked.

  figures/prelim_figures/fig10_recession_model_over_time.png
  more_model_images/forward_only_predictions.csv   the plotted months, machine-readable

Third member of the fig8/fig9 family -- same wide layout, same NBER shading, same
reading habit: blue line is the model, grey bands are what the economy did. fig8
plots the press's accuracy over time, fig9 the press's own downturn signal; this
plots a FITTED model's out-of-sample probability, which is the thing fig9's
docstring is careful to say it is not.

Forward-only P(recession starts within 12 months). Every point comes from a model
refit on all prior years only and never shown the year it predicts, so the line
can be read left-to-right as real-time output. The line breaks at each recession
-- recession months are not scored, since "will a recession start?" is only a
question while one is not already running.

One line only -- the five-series model, labelled "recession model". The
attention-alone comparison this file used to draw a second line for lives in
broad_vs_attention.png, which exists to make exactly that comparison; here it was
a second line competing with the series the figure is about.

Untitled by request: the figure carries its caption and no headline, so whatever
it is dropped into supplies the claim. The per-period AUCs run along the bottom
in the caption and print to stdout -- note the 1930s and 1960-63 sit below 0.5,
meaning the ranking is inverted there, which no rule on the chart marks because
the y-axis is probability rather than AUC.

The model comes from make_broad_vs_attention_figure.forward_only, so this chart,
broad_vs_attention.png and per_period_table.csv are one computation rendered
three ways and cannot disagree.

Run from the repo root:  python more_model_images/make_recession_model_time_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The model first, the fig8 style second: both modules set rcParams at import,
# and the later import must be the one whose look this figure borrows.
from make_broad_vs_attention_figure import forward_only, SERIES, H
from make_index_figures import BLUE, INK, MUTED, OUT, shade

START = 1930          # longest forward-only run -- every decade predicted unseen
DECADES = [("1930s", 1930, 1939), ("1940s", 1940, 1949),
           ("1950s", 1950, 1959), ("1960-63", 1960, 1963)]
CSV = Path(__file__).resolve().parent / "forward_only_predictions.csv"
PAD = {"bbox": dict(facecolor="white", edgecolor="none", pad=1.2, alpha=.85)}


def predictions():
    """Out-of-sample P(onset within H months) from the five press series."""
    p, truth = forward_only(SERIES, START)
    d = pd.DataFrame({"p_onset": p, "onset_within_12m": truth}, index=p.index)
    d["dt"] = d.index.to_timestamp()
    return d


def segments(d, col):
    """Split at the recession gaps so no line is drawn across months the model
    never scored -- a continuous line there would imply a prediction that does
    not exist."""
    brk = np.flatnonzero(np.diff([m.ordinal for m in d.index]) > 1) + 1
    return [(g["dt"].values, g[col].values) for g in np.split(d, brk) if len(g)]


def decade_auc(d):
    rows = []
    for lab, lo, hi in DECADES:
        s = d[(d.index.year >= lo) & (d.index.year <= hi)]
        t = s["onset_within_12m"]
        rows.append({"period": lab, "months": len(s), "onsets": int(t.sum()),
                     "base_rate": t.mean(),
                     "auc": roc_auc_score(t, s["p_onset"]),
                     "x0": s["dt"].min(), "x1": s["dt"].max()})
    return pd.DataFrame(rows)


def main():
    d = predictions()
    dec = decade_auc(d)
    base = d["onset_within_12m"].mean()
    auc = roc_auc_score(d["onset_within_12m"], d["p_onset"])

    d.drop(columns="dt").to_csv(CSV, index_label="month")

    # Taller and larger than the fig8/fig9 strip: with nothing above or below it,
    # the axes are the whole image, and 4.6in of height made a line that swings
    # between 0.01 and 0.98 look flatter than it is.
    fig, ax = plt.subplots(figsize=(13, 7.5))
    shade(ax, 0, 1)
    for k, (xs, ys) in enumerate(segments(d, "p_onset")):
        ax.plot(xs, ys, color=BLUE, lw=2.0, zorder=3,
                label="recession model" if k == 0 else None)

    # The base rate, not 0.5: 24% of expansion months are followed by an onset,
    # so 0.24 is what "no information" looks like here and 0.5 is meaningless.
    ax.axhline(base, color="#999999", lw=1, ls="--", zorder=1)
    # White pad behind the label: the line crosses most of the plotting area, so
    # unpadded text lands on it somewhere.
    ax.text(d["dt"].max(), base + .012, f"base rate ({base:.2f})", fontsize=10,
            color=MUTED, ha="right", zorder=5, **PAD)
    ax.set_ylim(0, 1)
    # Lower left is the one corner the line leaves empty (the 1933-37 stretch
    # never drops below 0.17); upper right is where the 1944 and 1953 peaks go.
    ax.set_xlim(d["dt"].min() - pd.Timedelta(days=200),
                d["dt"].max() + pd.Timedelta(days=200))
    # Type scaled up with the canvas -- 10pt on a 13in figure reads as fine print.
    ax.set_ylabel(f"P(recession starts within {H} months)", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower left", fontsize=11, framealpha=.9,
              facecolor="white", edgecolor="none").set_zorder(5)

    # No title, subtitle or caption by request -- the plot is the whole image, so
    # every number that used to sit around it now only prints to stdout. Whatever
    # this is dropped into has to carry the method line and the per-period AUCs.
    p = OUT / "fig10_recession_model_over_time.png"
    fig.tight_layout(pad=0.6)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    print(f"months {len(d)} · onsets {int(d['onset_within_12m'].sum())} "
          f"· base rate {base:.3f}")
    print(f"full window AUC {auc:.3f}")
    print(dec[["period", "months", "onsets", "base_rate", "auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {p}\n-> {CSV}")


if __name__ == "__main__":
    main()
