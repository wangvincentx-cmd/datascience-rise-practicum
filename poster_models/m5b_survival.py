"""MODEL 5b -- Discrete-time survival: not whether it came true, but WHEN.

The problem this solves
-----------------------
score_predictions.py resolves each claim at exactly one point: it takes the
horizon, looks at the economy that many months later, and stamps hit or miss.
Everything in between is discarded. A forecast of recovery that was right by
month 2 and one that scraped in at month 11 both score 1. A forecast that was
right for six months and then overtaken by events scores 0, identically to one
that was never right at all.

Recast as a duration. For each claim, walk h = 1, 2, ... months forward and ask
at each step whether the economy had, by then, moved the way the claim said.
The first h where it had is the event time. Claims whose horizon expires with
no match are RIGHT-CENSORED -- not failures, just unresolved within the window,
which is the correct treatment and the whole reason to use survival machinery
rather than a sequence of logits.

The estimator is a discrete-time hazard model: one row per claim-month at risk,
a logit for "did it come true THIS month, given it had not yet", a flexible
baseline hazard in log(h), and claim features on top. Its coefficients are
hazard ratios -- how much faster a forecast with this feature comes true.
Standard errors are clustered by 3-year block, as everywhere in this folder.

Two things this exposes that `hit` cannot
-----------------------------------------
1. SPEED. Two features can leave the hit rate untouched and still differ in how
   quickly the predicted turn arrived.
2. TRANSIENTLY-RIGHT FORECASTS. "Ever matched within the horizon" is a strictly
   weaker condition than "matched AT the horizon", so the two disagree exactly
   on the claims that were right for a while and then wrong again. That gap is
   reported as its own number; it is a measure of how much the project's
   fixed-horizon scoring rule is doing to the results, and it is not visible
   anywhere else in the pipeline.

Note on interpretation: realized_direction compares the economy at month 0 to
the economy at month h, so "came true at h" means the CUMULATIVE move from
publication had reached the predicted direction by h. That is the right reading
of a forecast, and it is the same rule score_predictions applies -- just
evaluated at every h instead of one.

    python poster_models/m5b_survival.py
    python poster_models/m5b_survival.py --max-h 36
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import _common as C

warnings.filterwarnings("ignore")


class HorizonGrid:
    """realized_direction at every horizon, memoised.

    Naively this is one lookup per claim per month -- around 340,000 calls into
    pandas .asof. Claims cluster heavily on (topic, month), so caching on that
    key plus the horizon collapses it to a few tens of thousands of distinct
    evaluations and turns minutes into seconds."""

    def __init__(self):
        self.truth, self.missing = C.truth_data()
        self._cache = {}

    def at(self, topic, date, h):
        key = (topic, pd.Timestamp(date).to_period("M"), h)
        if key not in self._cache:
            label, ok, basis = self.truth.realized_direction(topic, date, h)
            self._cache[key] = (label if ok else None)
        return self._cache[key]


def build_person_period(df, max_h, grid):
    """One row per claim-month at risk, with the event flag.

    A claim leaves the risk set the month it comes true (event) or the month
    its horizon expires (censored), whichever is first."""
    rows = []
    per_claim = []
    n = len(df)
    for i, (_, r) in enumerate(df.iterrows()):
        if i and i % 2000 == 0:
            print(f"    {i:,}/{n:,} claims expanded ...")
        topic, date = r["topic"], r["date"]
        pred = str(r["predicted_norm"])
        horizon = int(r["horizon_used"]) if np.isfinite(r["horizon_used"]) else 12
        limit = min(max_h, horizon)
        event_h, ever = None, 0
        for h in range(1, limit + 1):
            lab = grid.at(topic, date, h)
            if lab is None:
                # Series coverage ran out mid-window: the claim stops being at
                # risk here rather than being counted as a non-event, which
                # would silently turn missing data into evidence of failure.
                limit = h - 1
                break
            if lab == pred:
                event_h, ever = h, 1
                break
        if limit < 1:
            continue
        stop = event_h if event_h is not None else limit
        for h in range(1, stop + 1):
            rows.append((i, h, 1 if (event_h is not None and h == event_h) else 0))
        per_claim.append({"claim_i": i, "event_h": event_h, "ever": ever,
                          "at_risk_to": limit, "hit_at_horizon": int(r["hit"]),
                          "horizon": horizon})
    pp = pd.DataFrame(rows, columns=["claim_i", "h", "event"])
    return pp, pd.DataFrame(per_claim)


def km_curve(claims, max_h):
    """Kaplan-Meier style: share of forecasts NOT yet come true, by month."""
    surv, at_risk = [], len(claims)
    s = 1.0
    for h in range(1, max_h + 1):
        risk = ((claims["at_risk_to"] >= h) &
                ((claims["event_h"].isna()) | (claims["event_h"] >= h))).sum()
        ev = (claims["event_h"] == h).sum()
        if risk > 0:
            s *= (1 - ev / risk)
        surv.append({"h": h, "n_at_risk": int(risk), "n_events": int(ev),
                     "surv": s, "cum_realized": 1 - s})
    return pd.DataFrame(surv)


def run(args):
    C.header("MODEL 5b: discrete-time survival on time-to-realization",
             "When did the predicted turn actually arrive? Unresolved claims "
             "are censored,\nnot counted as failures.")

    df = C.load_scored(args.scored, args.rigid).reset_index(drop=True)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=0).reset_index(drop=True)
        print(f"\n  NOTE: --sample {args.sample} in use; this is a speed "
              f"option, not a result.")
    blocks = C.time_blocks(df, args.block_years)

    print(f"\n  claims {len(df):,}   max horizon considered {args.max_h} months")
    print("  expanding to claim-month rows ...")
    grid = HorizonGrid()
    pp, claims = build_person_period(df, args.max_h, grid)
    if not len(pp):
        raise SystemExit("No claim-months at risk -- check FRED coverage.")
    print(f"  {len(pp):,} claim-month rows from {len(claims):,} claims "
          f"({len(grid._cache):,} distinct series lookups)")

    ever = claims["ever"].mean()
    at_hz = claims["hit_at_horizon"].mean()
    both = ((claims["ever"] == 1) & (claims["hit_at_horizon"] == 1)).mean()
    transient = ((claims["ever"] == 1) & (claims["hit_at_horizon"] == 0)).mean()
    print(f"\n=== fixed-horizon scoring vs 'ever came true' ===")
    print(f"  came true AT the horizon (the project's `hit`) : {at_hz:.3f}")
    print(f"  came true at SOME point within the horizon     : {ever:.3f}")
    print(f"  both                                           : {both:.3f}")
    print(f"  TRANSIENTLY right -- true at some point, wrong  : {transient:.3f}")
    print(f"    by the time the horizon closed")
    print(f"\n  Reading: {transient:.1%} of forecasts were vindicated at some "
          f"point and then\n  overtaken. Fixed-horizon scoring calls them "
          f"misses. That is a defensible rule --\n  a forecast is a claim "
          f"about a date -- but it is a rule, and this is its cost.")

    # --- survival curve ------------------------------------------------------
    km = km_curve(claims, args.max_h)
    C.save(km, "m5b_survival_curve.csv")
    med = km[km["cum_realized"] >= 0.5]["h"].min()
    print(f"\n=== how fast did forecasts come true? ===")
    print(f"  median time to realization: "
          f"{f'{int(med)} months' if np.isfinite(med) else 'never reached 50% within the window'}")
    print(f"  {'month':>6}{'at risk':>10}{'events':>9}{'cum. realized':>16}")
    for _, r in km.iterrows():
        if r["h"] % max(1, args.max_h // 12) == 0 or r["h"] <= 3:
            print(f"  {int(r['h']):>6}{int(r['n_at_risk']):>10,}"
                  f"{int(r['n_events']):>9,}{r['cum_realized']:>16.3f}")

    # --- discrete-time hazard model -----------------------------------------
    Xc = C.standardize(C.drop_collinear(C.claim_design(df)))
    # The claim's own horizon must not enter: it defines when the claim leaves
    # the risk set, so including it would let the model predict the event from
    # the censoring rule.
    Xc = Xc.drop(columns=[c for c in ["c_horizon"] if c in Xc.columns])

    D = Xc.iloc[pp["claim_i"].values].reset_index(drop=True)
    # Flexible baseline hazard. log(h) plus its square is enough shape for a
    # 24-month window and costs two parameters; month dummies would cost 24 and
    # buy nothing here.
    D.insert(0, "log_h", np.log(pp["h"].values))
    D.insert(1, "log_h_sq", np.log(pp["h"].values) ** 2)
    D = C.drop_collinear(D)
    Dc = sm.add_constant(D, has_constant="add")
    y = pp["event"].values
    grp = blocks[pp["claim_i"].values]

    print(f"\n  fitting discrete-time hazard logit on {len(Dc):,} claim-months ...")
    res = sm.Logit(y, Dc).fit(disp=0, maxiter=300, cov_type="cluster",
                              cov_kwds={"groups": grp})
    tab = C.coef_table(
        Dc.columns, res.params.values, res.bse.values,
        "Discrete-time hazard of coming true (logit, block-clustered SE)",
        note="Odds ratio > 1 = the predicted turn arrived SOONER for forecasts "
             "with this\n  feature. log_h terms are the baseline hazard shape, "
             "not findings.\n  One row per claim-month at risk; claims exit on "
             "realization or horizon expiry.",
        n=len(Dc))
    C.save(tab, "m5b_hazard.csv")

    # --- speed by feature, descriptively ------------------------------------
    print("\n=== median months to realization, by feature ===")
    print("  Among claims that came true at all. A blunt companion to the "
          "hazard model.\n")
    cl = claims.merge(df[["confidence", "voice", "topic", "predicted_norm"]]
                      .reset_index().rename(columns={"index": "claim_i"}),
                      on="claim_i", how="left")
    rows = []
    for col in ["confidence", "voice", "topic", "predicted_norm"]:
        sub = cl[cl["event_h"].notna()]
        g = sub.groupby(sub[col].fillna("na"))["event_h"]
        t = pd.DataFrame({"n": g.size(), "median_months": g.median(),
                          "mean_months": g.mean()})
        t = t[t["n"] >= 25].sort_values("median_months")
        if not len(t):
            continue
        print(f"  by {col}:")
        for k, r in t.iterrows():
            print(f"    {str(k):<20} n={int(r['n']):<6} median "
                  f"{r['median_months']:>4.1f} mo   mean {r['mean_months']:>4.1f}")
        t = t.reset_index()
        t.columns = ["level", "n", "median_months", "mean_months"]
        t.insert(0, "feature", col)
        rows.append(t)
    if rows:
        C.save(pd.concat(rows, ignore_index=True), "m5b_speed_by_feature.csv")

    C.save(claims, "m5b_claim_durations.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--max-h", type=int, default=24,
                    help="longest horizon to walk forward, in months")
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample claims for a fast smoke run (0 = all)")
    run(ap.parse_args())
