"""
Did press optimism RISE INTO each crisis peak, or fall as the peak approached?

This is the within-episode, monthly-resolution development of the optimism gap
(model.py's strongest result: a claim's stated `direction` predicts whether it
turns out right, permutation p=0.0099). The episode-level tests elsewhere
(disagreement_severity.py) keep dying at n=19 -- a POWER problem, not
necessarily a real null. The fix is temporal resolution: every claim carries a
date, so within each dense crisis window we can watch optimism move month by
month instead of collapsing the episode to one number.

The behavioral question (Reinhart & Rogoff, *This Time Is Different*): as the
economy neared its peak and the collapse approached, did newspapers grow more
cautious (foresight -- net optimism should FALL toward the peak), or stay
confidently optimistic right up to the edge (complacency -- net optimism stays
high / rises into the peak)? The slope of net optimism against months-to-peak
is the whole answer: negative slope = they saw it coming; flat/positive = they
did not.

Definitions (leakage-safe by construction -- everything is anchored to the
realized business-cycle peak, which is what we are testing foresight against,
and no claim is scored using information after its own date):
  - net_optimism(episode, month) = (n_improve - n_worsen) / (n_improve + n_worsen)
    among that month's general-business predictions (predicted_label in
    {improve, worsen}; price/employment up/down claims are excluded -- "up
    inflation" has no unambiguous optimism sign, documented scope choice).
    +1 = everyone that month predicted improvement, -1 = everyone predicted
    worsening, 0 = evenly split.
  - peak month: the realized industrial-production peak (argmax of FRED INDPRO)
    within the episode's own claim-date span -- an objective, code-derived
    boundary, same INDPRO basis disagreement_severity.py uses. Pre-1919
    episodes (no INDPRO) fall back to the NBER recession peak in the span.
  - months_to_peak(claim month) = (claim month - peak month), negative before
    the peak. The run-up is months_to_peak in [-RUNUP, 0].

Reported: per-episode run-up net optimism + slope; a pooled peak-aligned
optimism curve across crisis episodes with an episode-block-bootstrap CI on the
pooled slope; and crisis-vs-control baseline optimism as a placebo (controls
have no peak to rise into). Only crisis episodes with an identifiable in-span
peak enter the run-up test.

Usage: python optimism_timeline.py
Outputs: optimism_by_month.csv, printed tables + pooled slope CI,
         figures/fig_optimism_timeline.png
"""

from pathlib import Path

import numpy as np
import pandas as pd

from score_claims import NBER_RECESSIONS, fred

FIGDIR = Path("figures")
DIRECTIONS = ("improve", "worsen")
RUNUP = 12          # months before the peak that count as the "run-up"
POST = 6            # months after the peak shown on the aligned curve
N_BOOT = 2000


def optimism_index(df):
    """Per (episode, month) net optimism among general-business improve/worsen
    predictions. Returns a DataFrame with columns
    [episode, kind, period, n_improve, n_worsen, n, net_optimism]."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["predicted_label"].isin(DIRECTIONS)].copy()
    d["period"] = d["date"].dt.to_period("M")
    kind_by_ep = df.drop_duplicates("episode").set_index("episode")["kind"] \
        if "kind" in df.columns else None
    rows = []
    for (ep, period), g in d.groupby(["episode", "period"]):
        n_imp = int((g["predicted_label"] == "improve").sum())
        n_wor = int((g["predicted_label"] == "worsen").sum())
        total = n_imp + n_wor
        rows.append({
            "episode": ep,
            "kind": (kind_by_ep[ep] if kind_by_ep is not None else ""),
            "period": period, "n_improve": n_imp, "n_worsen": n_wor, "n": total,
            "net_optimism": (n_imp - n_wor) / total if total else np.nan})
    return pd.DataFrame(rows).sort_values(["episode", "period"]).reset_index(drop=True)


def episode_peak_month(start, end, indpro):
    """Objective peak month for an episode's [start, end] claim span.
    Returns (Period or None, basis str). INDPRO argmax where covered; NBER
    recession peak within the span otherwise; nearest NBER peak within 18
    months after the span start as a last resort."""
    p0, p1 = pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M")
    window = indpro[(indpro.index >= p0) & (indpro.index <= p1)]
    if not window.empty:
        return window.idxmax(), "INDPRO"
    # Pre-INDPRO fallback: an NBER recession peak from up to a year before the
    # span (a curated crisis window can open at/just after the peak) through its
    # end. Prefer a peak inside the span, else the one nearest the span start.
    cands = [pd.Period(pk, "M") for pk, _ in NBER_RECESSIONS
             if (p0 - 12) <= pd.Period(pk, "M") <= p1]
    if cands:
        inside = [pk for pk in cands if p0 <= pk <= p1]
        pool = inside or cands
        return min(pool, key=lambda pk: abs((pk - p0).n)), "NBER"
    return None, "none"


def months_to_peak(period, peak):
    """Signed month distance (period - peak); negative = before the peak."""
    return (period - peak).n


def weighted_slope(x, y, w):
    """Slope of a weighted least-squares line y ~ x. NaN if <2 distinct x."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if len(np.unique(x)) < 2 or w.sum() == 0:
        return np.nan
    return float(np.polyfit(x, y, 1, w=w)[0])


def _runup_points(idx, peaks):
    """Pooled (months_to_peak, net_optimism, weight) points inside the run-up
    window, tagged by episode, for crisis episodes with a peak."""
    pts = []
    for _, r in idx.iterrows():
        peak = peaks.get(r["episode"])
        if peak is None or pd.isna(r["net_optimism"]):
            continue
        k = months_to_peak(r["period"], peak)
        if -RUNUP <= k <= 0:
            pts.append((r["episode"], k, r["net_optimism"], r["n"]))
    return pd.DataFrame(pts, columns=["episode", "k", "net_optimism", "n"])


def main():
    df = pd.read_csv("claims_scored.csv")
    idx = optimism_index(df)
    idx.to_csv("optimism_by_month.csv", index=False)

    indpro = fred("INDPRO")
    crisis_eps = [e for e in idx["episode"].unique()
                  if (idx.loc[idx["episode"] == e, "kind"] == "crisis").any()]

    # Peak per crisis episode, from that episode's own claim span.
    spans = df.assign(date=pd.to_datetime(df["date"])).groupby("episode")["date"].agg(["min", "max"])
    peaks, peak_basis = {}, {}
    for ep in crisis_eps:
        pk, basis = episode_peak_month(spans.loc[ep, "min"], spans.loc[ep, "max"], indpro)
        peaks[ep], peak_basis[ep] = pk, basis

    # Per-episode run-up net optimism and slope.
    print("=== Per-crisis-episode run-up (last %d months before the IP peak) ===" % RUNUP)
    print("net optimism: +1 all-improve, -1 all-worsen. slope>0 = optimism RISING")
    print("into the peak (complacency); slope<0 = growing caution (foresight).\n")
    per_ep = []
    for ep in sorted(crisis_eps):
        pk = peaks[ep]
        sub = idx[(idx["episode"] == ep) & idx["net_optimism"].notna()].copy()
        if pk is not None:
            sub["k"] = [months_to_peak(p, pk) for p in sub["period"]]
            run = sub[(sub["k"] >= -RUNUP) & (sub["k"] <= 0)]
        else:
            run = sub.iloc[0:0]
        mean_opt = np.average(run["net_optimism"], weights=run["n"]) if len(run) else np.nan
        slope = weighted_slope(run["k"], run["net_optimism"], run["n"]) if len(run) else np.nan
        per_ep.append({"episode": ep, "peak": str(pk) if pk is not None else "-",
                       "peak_basis": peak_basis[ep], "runup_months": len(run),
                       "runup_claims": int(run["n"].sum()) if len(run) else 0,
                       "net_optimism": round(mean_opt, 3) if mean_opt == mean_opt else np.nan,
                       "slope_per_month": round(slope, 4) if slope == slope else np.nan})
    per_ep = pd.DataFrame(per_ep)
    print(per_ep.to_string(index=False))

    # Pooled peak-aligned run-up slope with an episode-block bootstrap CI.
    pts = _runup_points(idx, peaks)
    usable = per_ep[per_ep["slope_per_month"].notna()]
    print(f"\n=== Pooled peak-aligned run-up ({usable.shape[0]} crisis episodes "
          f"with an in-span peak and >=2 run-up months) ===")
    if len(pts) and pts["episode"].nunique() >= 2:
        pooled_slope = weighted_slope(pts["k"], pts["net_optimism"], pts["n"])
        pooled_mean = np.average(pts["net_optimism"], weights=pts["n"])
        rng = np.random.default_rng(0)
        eps = pts["episode"].unique()
        boot = []
        for _ in range(N_BOOT):
            pick = rng.choice(eps, len(eps), replace=True)
            bs = pd.concat([pts[pts["episode"] == e] for e in pick])
            s = weighted_slope(bs["k"], bs["net_optimism"], bs["n"])
            if s == s:
                boot.append(s)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        n_pos = int((usable["slope_per_month"] >= 0).sum())
        print(f"  pooled run-up net optimism:  {pooled_mean:+.3f}  "
              f"(papers were net {'OPTIMISTIC' if pooled_mean > 0 else 'pessimistic'} "
              "on average in the year before the peak)")
        print(f"  pooled slope vs months-to-peak: {pooled_slope:+.4f} per month  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]")
        verdict = ("optimism did NOT fall as the peak approached -> complacency, "
                   "not foresight" if hi >= 0 else
                   "optimism fell significantly toward the peak -> foresight")
        print(f"  reading: {verdict}")
        print(f"  across episodes: {n_pos}/{len(usable)} had a flat-or-rising "
              "run-up slope (optimism sustained into the peak)")
    else:
        print("  not enough peak-aligned run-up points to pool.")

    # Placebo: crisis vs control baseline optimism (controls have no peak).
    base = idx[idx["net_optimism"].notna()].groupby("kind").apply(
        lambda g: np.average(g["net_optimism"], weights=g["n"]), include_groups=False)
    print("\n=== Baseline monthly net optimism, crisis vs control ===")
    for k, v in base.items():
        print(f"  {k:8s}: {v:+.3f}")

    _figure(idx, peaks)
    print("\noptimism_by_month.csv + figures/fig_optimism_timeline.png written")


def _figure(idx, peaks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)

    # Pooled peak-aligned curve: mean net optimism at each months_to_peak.
    rows = []
    for _, r in idx.iterrows():
        pk = peaks.get(r["episode"])
        if pk is None or pd.isna(r["net_optimism"]):
            continue
        k = months_to_peak(r["period"], pk)
        if -RUNUP <= k <= POST:
            rows.append((k, r["net_optimism"], r["n"]))
    aligned = pd.DataFrame(rows, columns=["k", "net_optimism", "n"])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    if len(aligned):
        curve = aligned.groupby("k").apply(
            lambda g: np.average(g["net_optimism"], weights=g["n"]), include_groups=False)
        ax.plot(curve.index, curve.values, "-o", color="goldenrod", lw=2, label="mean net optimism")
        ax.fill_between(curve.index, 0, curve.values, where=(curve.values > 0),
                        color="goldenrod", alpha=.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="crimson", ls="--", lw=1.2, label="business-cycle peak")
    ax.set_xlabel("months relative to the industrial-production peak (0 = peak, negative = before)")
    ax.set_ylabel("net press optimism  (+1 all-improve, -1 all-worsen)")
    ax.set_title("Did the press see it coming? Net optimism into the peak, pooled across crises")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_optimism_timeline.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
