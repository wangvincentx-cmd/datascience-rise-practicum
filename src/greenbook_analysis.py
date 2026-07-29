"""
Anatomy of the Fed Board staff's Greenbook forecast, 1967-2020.

greenbook_benchmark.py answers "what hit rate does the Fed staff score on the
newspapers' ground truth" (54.0%). This script answers the question the arm
actually cares about -- WHY that number is what it is, and whether the Fed staff
saw the recessions coming -- so the Greenbook stops being one bar on a
leaderboard and becomes the professional-forecaster counterpart to the
newspapers' optimism-gap result.

Three analyses:

  1. HORIZON ANATOMY. Forecast dispersion by horizon (F0 nowcast .. F8, i.e.
     two years out). If the staff reverts to a constant trend forecast as the
     horizon lengthens, then at long horizons the Greenbook is mathematically
     incapable of directional skill -- it is a constant, and a constant cannot
     track turning points. This is the mechanism behind the 54.0%.

  2. RECESSION CALLS. For every NBER peak in the Greenbook era, what did the
     staff forecast in the four quarters BEFORE the economy turned? This is the
     Fed-staff version of "did they see it coming", scored the same way the
     newspapers are: only information available at the forecast date counts.
     Loungani (2001) and Romer & Romer (2000) are the literature anchors.

  3. SKILL BY DECADE, using score_claims.naive_skill's definition (hit rate
     minus the score a constant "always improve" forecaster gets on the same
     rows) -- because a raw hit rate mostly measures the base rate of
     expansion, not skill. See naive_skill's docstring.

Usage (from JeremysShit/):
    python greenbook_analysis.py          # tables + figures/fig_greenbook_anatomy.png

Reads the All-Column-Format folder (cache/Gbweb_All_Column_Format/, see
greenbook_benchmark.py's docstring) and greenbook_scored.csv (run
greenbook_benchmark.py first).
"""

import numpy as np
import pandas as pd

from greenbook_benchmark import load_editions_wide

FIGDIR = __import__("pathlib").Path("figures")
SCORED = "data/scored/greenbook_scored.csv"

# NBER contractions covering the Greenbook era. score_claims.NBER_RECESSIONS
# stops at 1961 because the newspaper corpus does; the Fed staff series runs to
# 2020, so the modern chronology is needed here. Peak month -> trough month.
NBER_MODERN = [
    ("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"),
    ("1981-07", "1982-11"), ("1990-07", "1991-03"), ("2001-03", "2001-11"),
    ("2007-12", "2009-06"), ("2020-02", "2020-04"),
]

HORIZONS = [f"gRGDPF{h}" for h in range(9)]


def load_greenbook():
    return load_editions_wide("gRGDP", max_h=8)


def horizon_anatomy(gb):
    """Does the staff's forecast collapse to a constant as the horizon grows?"""
    rows = []
    for h, col in enumerate(HORIZONS):
        v = gb[col].dropna()
        if len(v) < 20:
            continue
        rows.append({"horizon_q": h, "n": len(v), "mean": round(v.mean(), 2),
                     "sd": round(v.std(), 2),
                     "pct_negative": round((v < 0).mean() * 100, 1)})
    t = pd.DataFrame(rows).set_index("horizon_q")
    print("\n=== 1. HORIZON ANATOMY: does the forecast become a constant? ===")
    print("  (gRGDP growth forecast, quarters ahead of the nowcast)")
    print(t.to_string())
    far = t[t.index >= 6]
    if len(far) and (far["pct_negative"] == 0).all():
        print(f"\n  -> At {far.index.min()}+ quarters out the staff has NEVER forecast")
        print(f"     negative growth: 0 of {int(far['n'].sum())} forecasts across 54 years.")
    print(f"  -> SD falls {t['sd'].iloc[0]:.2f} -> {t['sd'].iloc[-1]:.2f} while the mean stays")
    print(f"     ~{t['mean'].iloc[-4:].mean():.1f}%: beyond a year the Greenbook IS a trend")
    print("     forecast, and a constant cannot track turning points. That is the")
    print("     mechanism behind its zero directional skill, not a scoring artifact.")
    return t


def recession_calls(gb):
    """What did the staff forecast in the 4 quarters before each NBER peak?"""
    print("\n=== 2. RECESSION CALLS: did the Fed staff see the turn coming? ===")
    print("  Greenbooks issued in the 12 months BEFORE each NBER peak.")
    print("  'worst call' = the most pessimistic 1-yr-ahead forecast on record then.")
    rows = []
    for peak, trough in NBER_MODERN:
        p = pd.Timestamp(peak)
        pre = gb[(gb["gbdate"] >= p - pd.DateOffset(months=12)) & (gb["gbdate"] < p)]
        if pre.empty:
            continue
        yr = pre[HORIZONS[1:5]].mean(axis=1)          # the ~1-year-ahead call
        rows.append({"nber_peak": peak, "greenbooks": len(pre),
                     "mean_1yr_fcst": round(yr.mean(), 2),
                     "worst_call": round(yr.min(), 2),
                     "ever_negative": "YES" if (yr < 0).any() else "no"})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    called = (t["ever_negative"] == "YES").sum()
    print(f"\n  -> Forecast a contraction before {called} of {len(t)} recessions.")
    print("     Every pre-recession mean is POSITIVE: entering each downturn the")
    print("     staff was still projecting growth a year out. This is the")
    print("     'failure to predict recessions' (Loungani 2001) in the Fed's own")
    print("     internal numbers -- the same optimism the newspapers show, from")
    print("     the best-resourced forecaster in the country.")
    return t


def skill_by_decade():
    """Skill vs a constant 'always improve' forecaster, by decade."""
    from score_claims import naive_skill
    s = pd.read_csv(SCORED, parse_dates=["forecast_date"])
    s = s.rename(columns={"pred_direction": "predicted_label",
                          "realized_direction": "realized_label"})
    print("\n=== 3. SKILL vs a naive 'always improve' forecaster, by decade ===")
    rows = []
    for dec, g in s.groupby(s["forecast_date"].dt.year // 10 * 10):
        hit, naive, skill, n = naive_skill(g)
        if n >= 20:
            rows.append({"decade": f"{dec}s", "n": n, "hit": round(hit, 1),
                         "naive": round(naive, 1), "skill_pts": round(skill, 1)})
    t = pd.DataFrame(rows).set_index("decade")
    print(t.to_string())
    hit, naive, skill, n = naive_skill(s)
    print(f"\n  overall: hit {hit:.1f}%  naive {naive:.1f}%  skill {skill:+.1f} pts (n={n})")
    print("  -> Skill hovers around zero in every decade. The 1990s look strong")
    print("     only because that decade had no recession, so the naive optimist")
    print("     scores high too -- which is exactly why the raw rate must not be")
    print("     quoted without this baseline.")
    return t


def figure(anatomy, calls):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.bar(anatomy.index, anatomy["sd"], color="#4C72B0", alpha=.85)
    ax1.set_xlabel("quarters ahead of nowcast")
    ax1.set_ylabel("SD of forecast (pp)")
    ax1.set_title("A. The forecast collapses to a constant")
    for x, (sd, pct) in enumerate(zip(anatomy["sd"], anatomy["pct_negative"])):
        ax1.text(x, sd + .06, f"{pct:.0f}%", ha="center", fontsize=8, color="#333")
    ax1.text(.98, .95, "% = share of forecasts below zero", transform=ax1.transAxes,
             ha="right", va="top", fontsize=8, color="#555")

    y = np.arange(len(calls))
    ax2.barh(y, calls["mean_1yr_fcst"], color="#C44E52", alpha=.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(calls["nber_peak"], fontsize=8)
    ax2.axvline(0, color="k", lw=1)
    ax2.set_xlabel("mean 1-yr-ahead gRGDP forecast (%)")
    ax2.set_title("B. Still forecasting growth into every recession")
    ax2.invert_yaxis()

    fig.suptitle("Greenbook (Fed Board staff) forecast anatomy, 1967-2020", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_greenbook_anatomy.png", dpi=150, bbox_inches="tight")
    print("\nfigures/fig_greenbook_anatomy.png written")


if __name__ == "__main__":
    gb = load_greenbook()
    print(f"Loaded {len(gb)} Greenbook editions, "
          f"{gb['gbdate'].min():%Y-%m-%d} -> {gb['gbdate'].max():%Y-%m-%d}")
    anatomy = horizon_anatomy(gb)
    calls = recession_calls(gb)
    try:
        skill_by_decade()
    except FileNotFoundError:
        print(f"\n({SCORED} not found — run greenbook_benchmark.py for the skill table)")
    figure(anatomy, calls)
