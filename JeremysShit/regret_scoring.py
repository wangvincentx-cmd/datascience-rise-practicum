"""
Not all wrong predictions cost the same. Telling readers the economy will
IMPROVE right before it collapses (an optimistic error) is far costlier than a
false alarm (predicting trouble that never comes). A symmetric hit rate treats
those two the same; this script re-scores the corpus under an ASYMMETRIC loss
so the errors' DIRECTION and the severity of what actually happened both count.

Two things, in increasing order of assumption:

1. Error DIRECTION decomposition (assumption-free, the robust headline). Among
   general-business misses (predicted_label in {improve, worsen}), split into:
     - optimistic error: predicted improve, reality did not improve
     - pessimistic error: predicted worsen, reality did not worsen
   The claim is that misses are disproportionately OPTIMISTIC, and that this
   concentrates in the worst crises. This needs no weights -- it is a count.

2. Severity-weighted REGRET (a documented modeling choice, sensitivity-tested).
   regret = 0 for a hit; for a miss, weight = (w_opt if optimistic else w_pess)
   x episode severity. Severity = magnitude of the episode's peak-to-trough
   INDPRO decline (same objective basis as disagreement_severity.py), min-max
   normalized to [0,1] across the INDPRO-covered episodes. Pre-1919 episodes
   (no INDPRO) are reported but EXCLUDED from the weighted pooled number -- the
   NBER-fraction fallback is a different scale, mixing it in would mislead
   (same rule disagreement_severity.py follows). The weight ratio w_opt:w_pess
   is swept (--weight-ratio) to show the QUALITATIVE result -- errors are
   asymmetrically optimistic -- does not depend on the exact weights; only the
   magnitude of the regret number does.

Usage:
    python regret_scoring.py                 # default 3:1 optimistic:pessimistic cost
    python regret_scoring.py --weight-ratio 5
Outputs: regret_by_episode.csv, printed decomposition + sensitivity table,
         figures/fig_regret.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from disagreement_severity import indpro_severity, nber_severity
from score_claims import fred

FIGDIR = Path("figures")
DIRECTIONS = ("improve", "worsen")


def classify_error(predicted_label, realized_label):
    """'hit' | 'optimistic_error' | 'pessimistic_error' | 'na'.
    Defined only for general-business improve/worsen predictions; a miss is
    tagged by what the PAPER predicted, since that is the direction of the
    reader's exposure (told 'improve' -> optimistic error when it didn't)."""
    if predicted_label not in DIRECTIONS:
        return "na"
    if predicted_label == realized_label:
        return "hit"
    return "optimistic_error" if predicted_label == "improve" else "pessimistic_error"


def regret(error_type, severity, w_opt, w_pess):
    """Asymmetric, severity-scaled loss for one claim. Hits cost nothing."""
    if error_type == "optimistic_error":
        return w_opt * severity
    if error_type == "pessimistic_error":
        return w_pess * severity
    return 0.0


def episode_severity(df):
    """Normalized [0,1] severity per episode from INDPRO peak-to-trough decline.
    Returns (severity Series, basis Series). Pre-1919 episodes get an NBER
    fallback value flagged as a different scale (excluded from normalization)."""
    d = df.assign(date=pd.to_datetime(df["date"]))
    indpro = fred("INDPRO")
    raw, basis = {}, {}
    for ep, g in d.groupby("episode"):
        start, end = g["date"].min(), g["date"].max()
        sev = indpro_severity(indpro, start, end)
        if sev is not None:
            raw[ep], basis[ep] = abs(sev), "INDPRO"        # magnitude of decline
        else:
            raw[ep], basis[ep] = nber_severity(start, end), "NBER_fallback"
    sev = pd.Series(raw, name="severity_raw")
    basis = pd.Series(basis, name="severity_basis")
    ind = sev[basis == "INDPRO"]
    lo, hi = ind.min(), ind.max()
    norm = ((sev - lo) / (hi - lo)).clip(0, 1) if hi > lo else sev * 0
    norm[basis != "INDPRO"] = np.nan            # don't mix scales into the weighted number
    return norm.rename("severity"), basis


def main(args):
    df = pd.read_csv("claims_scored.csv").dropna(subset=["hit"]).copy()
    gb = df[df["predicted_label"].isin(DIRECTIONS)].copy()
    gb["error_type"] = [classify_error(p, r) for p, r in
                        zip(gb["predicted_label"], gb["realized_label"])]

    sev, sev_basis = episode_severity(df)
    gb["severity"] = gb["episode"].map(sev)

    # 1. Assumption-free error-direction decomposition.
    misses = gb[gb["error_type"] != "hit"]
    n_opt = int((misses["error_type"] == "optimistic_error").sum())
    n_pess = int((misses["error_type"] == "pessimistic_error").sum())
    print("=== Error-direction decomposition (general-business misses) ===")
    print(f"  optimistic errors (said improve, it didn't):  {n_opt}")
    print(f"  pessimistic errors (said worsen, it didn't):  {n_pess}")
    if n_opt + n_pess:
        share = n_opt / (n_opt + n_pess)
        print(f"  optimistic share of all misses: {share:.1%}")
        # binomial sign test vs a symmetric 50/50 error direction
        from scipy.stats import binomtest
        p = binomtest(n_opt, n_opt + n_pess, 0.5).pvalue
        print(f"  (binomial test vs 50/50 symmetric errors: p={p:.2e})")

    if "kind" in gb.columns:
        print("\n=== Optimistic-error share, crisis vs control ===")
        for kind, g in misses.groupby("kind"):
            no = (g["error_type"] == "optimistic_error").sum()
            print(f"  {kind:8s}: {no}/{len(g)} misses optimistic = {no/len(g):.1%}")

    # 2. Severity-weighted regret (INDPRO episodes only), default weights.
    reg = gb.copy()
    reg["regret"] = [regret(e, s if s == s else 0.0, args.w_opt, args.w_pess)
                     for e, s in zip(reg["error_type"], reg["severity"])]
    scored = reg[reg["severity"].notna()]          # INDPRO-covered episodes only
    print(f"\n=== Severity-weighted regret (w_opt:w_pess = {args.w_opt:g}:{args.w_pess:g}, "
          f"INDPRO episodes, n={len(scored)}) ===")
    print(f"  mean regret per claim: {scored['regret'].mean():.4f}")
    print(f"  symmetric miss rate (for contrast): {(gb['error_type'] != 'hit').mean():.3f}")

    by_ep = (reg.groupby("episode")
             .agg(n=("error_type", "size"),
                  hit_rate=("error_type", lambda x: (x == "hit").mean()),
                  optimistic_errors=("error_type", lambda x: (x == "optimistic_error").sum()),
                  pessimistic_errors=("error_type", lambda x: (x == "pessimistic_error").sum()),
                  mean_regret=("regret", "mean"))
             .join(sev).join(sev_basis).round(3).sort_values("severity", na_position="last"))
    by_ep.to_csv("regret_by_episode.csv")
    print("\n=== By episode (sorted by severity) ===")
    print(by_ep.to_string())

    # Sensitivity: optimistic share is weight-free; only regret magnitude scales.
    print("\n=== Sensitivity: optimistic-error share is invariant to cost weights ===")
    for ratio in (1, 3, 5, 10):
        r = reg.assign(regret=[regret(e, s if s == s else 0.0, ratio, 1.0)
                               for e, s in zip(reg["error_type"], reg["severity"])])
        m = r[r["severity"].notna()]["regret"].mean()
        print(f"  ratio {ratio:2d}:1 -> mean regret {m:.4f}  "
              "(share optimistic unchanged, only magnitude moves)")

    _figure(by_ep)
    print("\nregret_by_episode.csv + figures/fig_regret.png written")


def _figure(by_ep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    d = by_ep.dropna(subset=["severity"]).sort_values("severity")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(d))
    ax.bar(x, d["optimistic_errors"], color="crimson", alpha=.85, label="optimistic errors")
    ax.bar(x, d["pessimistic_errors"], bottom=d["optimistic_errors"],
           color="steelblue", alpha=.85, label="pessimistic errors")
    ax.set_xticks(x)
    ax.set_xticklabels(d.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("number of misses")
    ax.set_title("Newspaper errors were asymmetrically optimistic, worst in the severest crises\n"
                 "(episodes left->right by increasing severity)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_regret.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight-ratio", type=float, default=3.0,
                    help="cost of an optimistic error relative to a pessimistic one "
                         "(w_opt = ratio, w_pess = 1)")
    a = ap.parse_args()
    a.w_opt, a.w_pess = a.weight_ratio, 1.0
    main(a)
