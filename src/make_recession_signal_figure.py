"""Did the press call recessions before they arrived? Signal vs. NBER reality.

  figures/prelim_figures/fig9_recession_signal.png

Companion to fig8: same wide layout, same NBER shading, same reading habit --
the blue line is what the newspapers said, the grey bands are what the economy
did.

WHAT THE LINE IS NOT: this is not a fitted model's P(recession), and nothing
here is trained on the recession dates. It is the raw monthly share of scorable
US-national forecasts that predicted the economy would WORSEN -- the press's own
downturn signal, straight out of press_index.csv. Reading it as a model output
would claim an out-of-sample validation that was never run.

Because the line is descriptive rather than fitted, the honest comparison is the
one drawn: where the signal sits relative to the bands. It sits slightly LOWER
inside recessions than outside them, and its correlation with a recession twelve
months ahead is about +0.01 -- the direction the project's thesis predicts.

The lead/lag table printed to stdout is the part that carries the claim; the
chart shows the same thing in a form a reader can check by eye.

Run from the repo root:  python src/make_recession_signal_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# shade()/smooth() and the rcParams come from the fig8 module rather than being
# restated, so "similar looking" is enforced by sharing the code, not by two
# copies of a palette drifting apart.
from make_index_figures import BLUE, INK, MUTED, OUT, shade, smooth
from truth_data import recession_months

MIN_CLAIMS = 5   # same floor make_index_figures uses for the index series


def load():
    d = pd.read_csv("data/scored/press_index.csv")
    d["p"] = pd.PeriodIndex(d["month"], freq="M")
    d["dt"] = d["p"].dt.to_timestamp()
    d["rec"] = d["p"].isin(set(recession_months())).astype(int)
    return d[d["n_claims"] >= MIN_CLAIMS].sort_values("p").reset_index(drop=True)


def leadlag(d, ks=range(-12, 13, 3)):
    """corr(share_worsen at t, recession at t+k). k>0 is the FORECASTING
    direction -- press now against the economy later."""
    return {k: d["share_worsen"].corr(d["rec"].shift(-k)) for k in ks}


def main():
    d = load()
    inside = d.loc[d["rec"] == 1, "share_worsen"].mean()
    outside = d.loc[d["rec"] == 0, "share_worsen"].mean()
    ll = leadlag(d)
    ahead12 = ll[12]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade(ax, 0, 1)
    ax.plot(d["dt"], d["share_worsen"], color=BLUE, lw=0.6, alpha=.3, zorder=2)
    ax.plot(d["dt"], smooth(d["share_worsen"]), color=BLUE, lw=2.2, zorder=3)

    # The overall mean, not 0.5: a "downturn share" has no natural midpoint, and
    # the question is whether the line rises ABOVE ITS OWN AVERAGE ahead of the
    # grey bands, not whether it crosses an arbitrary half.
    mean = d["share_worsen"].mean()
    ax.axhline(mean, color="#999999", lw=1, ls="--", zorder=1)
    # Parked over 1927-33, where the smoothed line runs well below the average
    # and the space above the rule is empty. At the right edge it collided with
    # the line itself.
    ax.text(pd.Timestamp("1927-06"), mean + .018,
            f"average ({mean:.2f})", fontsize=8, color=MUTED)

    ax.set_ylim(0, 1)
    ax.set_ylabel("share predicting a downturn")
    ax.set_title("Newspapers were no likelier to call a downturn before a "
                 "recession than at any other time",
                 fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015,
            "Share of scorable US-national forecasts predicting the economy "
            "would worsen, by month (12-month centred mean; thin line = raw).\n"
            f"Grey bands are NBER recessions. The signal averages {inside:.2f} "
            f"inside them against {outside:.2f} outside, and its correlation "
            f"with a recession 12 months ahead is {ahead12:+.2f}.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")

    fig.tight_layout()
    p = OUT / "fig9_recession_signal.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    print(f"months {len(d)} · recession months {int(d['rec'].sum())} "
          f"· base rate {d['rec'].mean():.3f}")
    print(f"share_worsen  inside recessions {inside:.3f} · outside {outside:.3f}")
    print("\ncorr(share_worsen at t, recession at t+k) "
          "-- k>0 is press now vs economy later:")
    for k, c in ll.items():
        print(f"  k={k:+3d}  {c:+.3f}")
    print("\nNOTE: descriptive series, not a fitted model. Nothing here is "
          "trained on the recession dates.")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
