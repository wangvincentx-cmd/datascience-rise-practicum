"""
Part 2: does an episode's OVERALL forecaster-disagreement level correlate
with how severe that crisis turned out to be?

A different question from disagreement.py's per-claim local_disagreement
(tested in model.py -- NULL result 2026-07-17, see CHANGELOG: local
disagreement doesn't predict individual claim accuracy). This is
episode-level (n=19, explicitly small-sample/exploratory, not a trained
model) and asks whether CONSENSUS (low disagreement) precedes WORSE
outcomes than genuine disagreement does -- the "false consensus" /
groupthink hypothesis, motivated by a real number already found this
session: 1929 Crash had 64 "improve" vs. only 10 "worsen" claims --
lopsided optimism -- immediately preceding the worst crash in US history.

Severity metric: peak-to-trough %% decline in FRED INDPRO (industrial
production) within each episode's own claim-date span (its earliest/latest
claim date in claims_scored.csv -- an objective, reproducible boundary, not
a hand-picked "official" crisis window). INDPRO only goes back to 1919; the
two earlier episodes (1905 Calm, 1907 Panic) fall back to the NBER
recession chronology already embedded in score_claims.py (fraction of
months in the span that were NBER recession months) -- a DIFFERENT, coarser
scale (0-1 fraction vs. %% decline), so those two are reported in the table
but EXCLUDED from both the correlation and the scatter plot: mixing two
severity scales in one number or one axis would be misleading, not just
imprecise.

Usage: python disagreement_severity.py
Outputs: printed table + Spearman correlation, figures/fig_disagreement_severity.png
"""

from pathlib import Path

import pandas as pd

from disagreement import episode_disagreement_rate
from score_claims import RECESSION_MONTHS, fred

FIGDIR = Path("figures")


def indpro_severity(indpro, start, end):
    """Peak(start)-to-trough %% decline in industrial production within
    [start, end]. None if the window predates INDPRO's 1919 start."""
    p0, p1 = start.to_period("M"), end.to_period("M")
    if p0 not in indpro.index:
        return None
    window = indpro[(indpro.index >= p0) & (indpro.index <= p1)]
    if window.empty:
        return None
    return (window.min() / indpro[p0] - 1) * 100


def nber_severity(start, end):
    """Fraction of months in [start, end] that were NBER recession months --
    coarser fallback for pre-1919 episodes only."""
    months = list(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"))
    if not months:
        return None
    return sum(1 for m in months if m in RECESSION_MONTHS) / len(months)


def main():
    df = pd.read_csv("claims_scored.csv")
    df = df.dropna(subset=["hit"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    disagreement = episode_disagreement_rate(df)
    indpro = fred("INDPRO")

    rows = []
    for ep, group in df.groupby("episode"):
        start, end = group["date"].min(), group["date"].max()
        sev = indpro_severity(indpro, start, end)
        metric = "INDPRO trough decline %"
        if sev is None:
            sev = nber_severity(start, end)
            metric = "NBER recession-month fraction (pre-1919 fallback, different scale)"
        rows.append({"episode": ep, "disagreement": round(disagreement[ep], 3),
                     "severity": round(sev, 3) if sev is not None else None,
                     "severity_metric": metric, "start": start.date(), "end": end.date()})

    out = pd.DataFrame(rows).sort_values("disagreement")
    print(out.to_string(index=False))

    indpro_rows = out[out["severity_metric"] == "INDPRO trough decline %"]
    corr = indpro_rows["disagreement"].corr(indpro_rows["severity"], method="spearman")
    print(f"\nSpearman correlation (disagreement vs. INDPRO trough decline, "
         f"n={len(indpro_rows)} of 19 episodes -- 2 pre-1919 episodes excluded, "
         f"different severity scale): {corr:.3f}")
    print("NOTE: n=17 is a small-sample exploratory analysis, not a trained/"
         "validated model -- read as suggestive, not conclusive, same caution "
         "this project applies everywhere else with small episode counts "
         "(e.g. model.py's LOEO SD ~0.21 across 19 episodes).")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGDIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(indpro_rows["disagreement"], indpro_rows["severity"],
                  color="steelblue", s=80, zorder=3)
        for _, r in indpro_rows.iterrows():
            ax.annotate(r["episode"], (r["disagreement"], r["severity"]),
                       fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("episode disagreement (minority share of improve/worsen claims, "
                     "0 = everyone agreed, 0.5 = perfectly split)")
        ax.set_ylabel("severity: INDPRO peak-to-trough decline within episode window (%)")
        ax.set_title(f"Consensus vs. crisis severity, {len(indpro_rows)} episodes with "
                    f"INDPRO coverage (Spearman r={corr:.2f})\n"
                    f"(1905 Calm, 1907 Panic excluded -- pre-1919, no INDPRO)")
        plt.tight_layout()
        plt.savefig(FIGDIR / "fig_disagreement_severity.png", dpi=200)
        plt.close()
        print("\nWrote figures/fig_disagreement_severity.png")
    except ImportError:
        print("\nmatplotlib missing -- no figure")


if __name__ == "__main__":
    main()
