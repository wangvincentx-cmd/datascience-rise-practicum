"""
Monthly press-expectations index figures, 1900-1963 -- the poster centerpiece.

Plots each index series against NBER recession shading, so a viewer can see at a
glance whether the press anticipated, coincided with, or lagged downturns.
Palette is Okabe-Ito (CVD-safe, validated); design follows the dataviz method.

Usage: python make_index_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from truth_data import NBER_RECESSIONS

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("prelim_figures"); OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})


def shade(ax, y0, y1):
    """NBER contractions as grey bands -- the reference the press is judged against."""
    for peak, trough in NBER_RECESSIONS:
        p0, p1 = pd.Timestamp(peak), pd.Timestamp(trough)
        if p1 < pd.Timestamp("1900-01") or p0 > pd.Timestamp("1963-12"):
            continue
        ax.axvspan(p0, p1, color="#dfe3e6", zorder=0, lw=0)


def smooth(s, w=12):
    return s.rolling(w, center=True, min_periods=max(3, w // 3)).mean()


def main():
    d = pd.read_csv("data/press_index.csv")
    d["dt"] = pd.PeriodIndex(d["month"], freq="M").to_timestamp()
    d = d[d["n_claims"] >= 5].sort_values("dt")

    # --- 1. net direction: the headline series -------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade(ax, -1, 1)
    ax.plot(d["dt"], d["net_direction"], color=BLUE, lw=0.6, alpha=.3, zorder=2)
    ax.plot(d["dt"], smooth(d["net_direction"]), color=BLUE, lw=2.2, zorder=3)
    ax.axhline(0, color="#999999", lw=1, zorder=1)
    ax.set_ylim(-1, 1); ax.set_ylabel("net optimism  (improve - worsen)")
    ax.set_title("The American press expected improvement almost every month for 60 years",
                 fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015, "Monthly net direction of newspaper economic forecasts, 1900-1963 "
            "(12-month centred mean; thin line = raw).\nGrey bands are NBER recessions. "
            "The line sits above zero through nearly every downturn.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "fig6_index_net_direction.png", bbox_inches="tight")
    plt.close(fig)

    # --- 2. disagreement + hedging: the uncertainty channel ------------------
    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade(ax, 0, 1)
    ax.plot(d["dt"], smooth(d["disagreement"]), color=VERM, lw=2.2, label="disagreement", zorder=3)
    ax.plot(d["dt"], smooth(d["hedge_rate"]), color=GREEN, lw=2.2, label="hedging rate", zorder=3)
    ax.set_ylim(0, 1); ax.set_ylabel("share")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Forecaster disagreement spikes in the 1940s, not in 1929",
                 fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015, "How divided and how hedged the press was, 1900-1963 "
            "(12-month centred mean). Grey bands are NBER recessions.\nDisagreement is the "
            "press-based uncertainty signal; it is highest around WWII and the postwar scare.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "fig7_index_uncertainty.png", bbox_inches="tight")
    plt.close(fig)

    # --- 3. accuracy over time (now a real series) ---------------------------
    s = pd.read_csv("monthly_scored.csv")
    s = s[(s["scorable"] == True) & (s["hit"].isin([0, 1]))].copy()
    s["dt"] = pd.to_datetime(s["date"], errors="coerce")
    yr = s.groupby(s["dt"].dt.year).agg(n=("hit", "size"), hit=("hit", "mean"))
    yr = yr[yr["n"] >= 20]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    shade(ax, 0, 1)
    ax.plot(pd.to_datetime(yr.index.astype(str)), yr["hit"], color=BLUE, lw=1.8,
            marker="o", ms=3, zorder=3)
    ax.axhline(0.5, color="#999999", lw=1, ls="--", zorder=1)
    ax.text(pd.Timestamp("1961-06"), 0.51, "coin flip", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1); ax.set_ylabel("directional hit rate")
    ax.set_title("Forecast accuracy over six decades: no era was much better than a coin flip",
                 fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015, "Share of scorable US-national forecasts that got the direction right, "
            "by year of publication.\nGrey bands are NBER recessions - accuracy dips inside them.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    fig.tight_layout(); fig.savefig(OUT / "fig8_accuracy_over_time.png", bbox_inches="tight")
    plt.close(fig)

    print("3 index figures ->", OUT)
    for f in ["fig6_index_net_direction.png", "fig7_index_uncertainty.png",
              "fig8_accuracy_over_time.png"]:
        print("  ", f)


if __name__ == "__main__":
    main()
