"""MODEL 5c -- Zoom out: did the PRESS AS A WHOLE see recessions coming?

The problem this solves
-----------------------
Every other model in this folder is claim-level, and claim-level is where the
result keeps coming up near-null: whether one forecast came true is mostly a
coin flip. But that is not quite the project's title question. "Did American
newspapers see economic downturns coming?" is a question about the press in
aggregate, and aggregation is exactly what kills the per-claim noise -- roughly
forty claims a month, each a noisy read on the same underlying mood.

So: one observation per month, 1900-1963. Outcome, did an NBER contraction
BEGIN within the next h months. Predictors, the monthly press index already
built by src/build_press_index.py. Estimator, a probit -- which is the standard
specification in the recession-prediction literature (Estrella-Mishkin), so the
numbers here are directly comparable to published yield-curve results rather
than being a bespoke metric only this poster uses.

Three things this file is careful about
---------------------------------------
1. OVERLAPPING WINDOWS. "Recession within 12 months" at January and at February
   share eleven months of outcome, so the errors are autocorrelated by
   construction and ordinary standard errors are badly too small. Newey-West
   HAC standard errors with a lag length of h are used throughout; this is
   standard practice in this literature for exactly this reason.
2. NO HINDSIGHT AMONG THE PREDICTORS. The NBER chronology is announced six to
   twenty-one months late, so it is only ever the OUTCOME here, never a lagged
   input. The macro benchmark uses publication-lagged INDPRO, so the comparison
   is between what a contemporary reader could know from the papers and what
   they could know from published statistics.
3. OUT-OF-SAMPLE, EXPANDING WINDOW. In-sample fit on 768 months with a handful
   of recessions overstates everything. The honest number is the AUC from
   forecasts made using only data available before the month being forecast.

The benchmark question is the interesting one: the press index is free and
contemporaneous; published output statistics were slow and revised. If the
press index adds anything to lagged INDPRO, that is a real finding about the
information content of newspapers.

    python poster_models/m5c_press_probit.py
    python poster_models/m5c_press_probit.py --horizons 3 6 12 24
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

import _common as C

warnings.filterwarnings("ignore")

# `share_worsen` is deliberately absent. build_press_index computes
# net_direction as (improve - worsen)/n_directional over claims that are all
# either improve or worsen, so share_improve + share_worsen == 1 exactly and
# net_direction == 1 - 2*share_worsen to machine precision. Including both puts
# an exact linear dependency into the design -- with the constant, no less --
# and the probit still "converges", reporting standard errors in the hundreds
# of millions. Verified on the shipped index: max deviation 2.5e-16.
PRESS_COLS = ["net_direction", "disagreement", "hedge_rate", "share_expert",
              "attention"]


def load_index(path):
    idx = pd.read_csv(path)
    idx["month"] = pd.PeriodIndex(idx["month"], freq="M")
    return idx.set_index("month").sort_index()


def recession_onsets():
    """Months in which an NBER contraction began. Outcome only, never a input."""
    from truth_data import NBER_RECESSIONS
    return {pd.Period(peak, "M") for peak, _ in NBER_RECESSIONS}


def build_panel(idx, horizons, smooth):
    """Monthly panel: press predictors + 'recession starts within h months'."""
    onsets = recession_onsets()
    d = idx.copy()

    # Smoothed levels and 6-month changes. A single month of press mood is
    # noisy (some months carry 11 claims); the change matters as much as the
    # level, since a press that is optimistic but LESS so than last quarter is
    # the interesting signal.
    feats = {}
    for c in PRESS_COLS:
        if c not in d:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        if c == "attention":
            s = np.log1p(s)
        sm_ = s.rolling(smooth, min_periods=max(2, smooth // 2)).mean()
        feats[f"p_{c}"] = sm_
        feats[f"p_{c}_d6"] = sm_ - sm_.shift(6)
    P = pd.DataFrame(feats, index=d.index)

    for h in horizons:
        y = []
        for m in d.index:
            window = pd.period_range(m + 1, m + h, freq="M")
            y.append(int(any(w in onsets for w in window)))
        P[f"y_{h}"] = y
    return P


def add_macro_benchmark(P):
    """Publication-lagged INDPRO growth: what the STATISTICS said at the time.

    Two months of publication lag, matching model_hit.MACRO_LAG_M, so this is
    what a reader could actually have looked up -- not the revised series we
    have now."""
    from model_hit import MACRO_LAG_M
    import model_hit
    missing = C._patch_missing_fred(model_hit)
    if "INDPRO" in missing:
        C.report_fred(["INDPRO"], "No macro benchmark will be fitted.")
        return P, False
    ip = model_hit.load_fred("INDPRO").shift(MACRO_LAG_M)
    g6 = 100 * (ip / ip.shift(6) - 1)
    g12 = 100 * (ip / ip.shift(12) - 1)
    P["m_indpro_g6"] = g6.reindex(P.index)
    P["m_indpro_g12"] = g12.reindex(P.index)
    P["m_indpro_accel"] = P["m_indpro_g6"] - P["m_indpro_g12"]
    return P, True


def fit_probit(P, ycol, cols, h):
    """Probit with Newey-West HAC standard errors.

    maxlags = h because consecutive observations share h-1 months of outcome;
    that is the exact order of the induced moving-average dependence."""
    sub = P[cols + [ycol]].dropna()
    if sub[ycol].nunique() < 2 or len(sub) < 60:
        return None, None, None
    # Guard even after curating PRESS_COLS: derived d6 columns can go dependent
    # on a subsample that dropna() happens to select.
    Xn = C.drop_collinear(sub[cols])
    X = sm.add_constant(Xn, has_constant="add")
    res = sm.Probit(sub[ycol].values, X).fit(
        disp=0, maxiter=200, cov_type="HAC", cov_kwds={"maxlags": h})
    return res, sub, list(Xn.columns)


def expanding_oos(P, ycol, cols, h, min_train, step):
    """Forecast each month using only months that ended before it.

    A gap of h months is left between the end of training and the month being
    forecast. Without it the last training observations' outcomes overlap the
    forecast month's outcome window, and the 'out-of-sample' AUC is inflated by
    an outcome the model has already seen."""
    sub = P[cols + [ycol]].dropna()
    if len(sub) < min_train + 24:
        return np.nan, None
    preds, actual, months = [], [], []
    order = list(range(len(sub)))
    for i in range(min_train, len(sub), step):
        tr = order[:max(0, i - h)]
        if len(tr) < min_train or sub[ycol].iloc[tr].nunique() < 2:
            continue
        Xtr = sm.add_constant(sub[cols].iloc[tr], has_constant="add")
        try:
            m = sm.Probit(sub[ycol].iloc[tr].values, Xtr).fit(disp=0, maxiter=200)
        except Exception:
            continue
        xte = sm.add_constant(sub[cols].iloc[[i]], has_constant="add")
        xte = xte.reindex(columns=Xtr.columns, fill_value=1.0)
        preds.append(float(m.predict(xte)[0]))
        actual.append(int(sub[ycol].iloc[i]))
        months.append(sub.index[i])
    if len(set(actual)) < 2:
        return np.nan, None
    return roc_auc_score(actual, preds), pd.DataFrame(
        {"month": months, "pred": preds, "actual": actual})


def run(args):
    C.header("MODEL 5c: aggregate press-index probit",
             "One row per month, not per claim. Did the press in aggregate "
             "lead NBER\ncontractions? Estrella-Mishkin specification, "
             "HAC standard errors.")

    idx = load_index(args.index)
    P = build_panel(idx, args.horizons, args.smooth)
    P, has_macro = add_macro_benchmark(P)

    press_cols = [c for c in P.columns if c.startswith("p_")]
    macro_cols = [c for c in P.columns if c.startswith("m_")]
    print(f"\n  months {len(P)}   {P.index.min()} -> {P.index.max()}")
    print(f"  press predictors {len(press_cols)}   "
          f"macro benchmark {'yes' if has_macro else 'NO'}")
    for h in args.horizons:
        print(f"  base rate, recession begins within {h:>2}m: "
              f"{P[f'y_{h}'].mean():.3f}")

    rows, all_pred = [], []
    for h in args.horizons:
        ycol = f"y_{h}"
        print(f"\n{'-' * 78}\n  HORIZON: recession begins within {h} months\n"
              f"{'-' * 78}")

        specs = [("press only", press_cols)]
        if has_macro:
            specs += [("macro only", macro_cols),
                      ("press + macro", press_cols + macro_cols)]

        fitted = {}
        for name, cols in specs:
            res, sub, cols = fit_probit(P, ycol, cols, h)
            if res is None:
                print(f"  {name:<16} not estimable at this horizon")
                continue
            fitted[name] = (res, sub, cols)
            auc_in = roc_auc_score(sub[ycol], res.predict(
                sm.add_constant(sub[cols], has_constant="add")))
            auc_oos, pred = expanding_oos(P, ycol, cols, h, args.min_train,
                                          args.step)
            print(f"  {name:<16} pseudo-R2 {res.prsquared:>6.3f}   "
                  f"in-sample AUC {auc_in:.3f}   "
                  f"out-of-sample AUC {auc_oos:.3f}" if np.isfinite(auc_oos)
                  else f"  {name:<16} pseudo-R2 {res.prsquared:>6.3f}   "
                       f"in-sample AUC {auc_in:.3f}   out-of-sample n/a")
            rows.append({"horizon": h, "spec": name, "n": int(res.nobs),
                         "pseudo_r2": res.prsquared, "auc_in": auc_in,
                         "auc_oos": auc_oos, "llf": res.llf})
            if pred is not None:
                pred["horizon"], pred["spec"] = h, name
                all_pred.append(pred)

        if "press only" in fitted:
            res, sub, cols = fitted["press only"]
            C.coef_table(
                ["const"] + cols, res.params.values, res.bse.values,
                f"PRESS-ONLY probit, recession within {h} months",
                note=f"Newey-West HAC SEs, maxlags={h} (overlapping outcome "
                     f"windows).\n  A negative coefficient on p_net_direction "
                     f"means MORE press optimism ->\n  LOWER probability of an "
                     f"imminent contraction, i.e. the press led correctly.",
                n=int(res.nobs)).pipe(C.save, f"m5c_press_probit_h{h}.csv")

        # Does the press add anything to the statistics of the day?
        if "press + macro" in fitted and "macro only" in fitted:
            full, _, _ = fitted["press + macro"]
            base, _, _ = fitted["macro only"]
            lr = 2 * (full.llf - base.llf)
            from scipy import stats as sps
            p = 1 - sps.chi2.cdf(lr, len(press_cols))
            print(f"\n  Does the PRESS add to published statistics?")
            print(f"    likelihood-ratio chi2({len(press_cols)}) = {lr:.2f}, "
                  f"p = {p:.4g}")
            a_full = next(r["auc_oos"] for r in rows
                          if r["horizon"] == h and r["spec"] == "press + macro")
            a_base = next(r["auc_oos"] for r in rows
                          if r["horizon"] == h and r["spec"] == "macro only")
            if np.isfinite(a_full) and np.isfinite(a_base):
                print(f"    out-of-sample AUC {a_base:.3f} -> {a_full:.3f} "
                      f"({a_full - a_base:+.3f})")
            print(f"    Note the LR test is in-sample and the overlapping "
                  f"windows make it\n    anti-conservative; the "
                  f"out-of-sample AUC change is the number to trust.")

    res_tab = pd.DataFrame(rows)
    C.save(res_tab, "m5c_probit_summary.csv")
    if all_pred:
        C.save(pd.concat(all_pred, ignore_index=True), "m5c_oos_predictions.csv")

    print(f"\n{'=' * 78}")
    oos = res_tab["auc_oos"].dropna()
    if len(oos) and (oos < 0.5).all():
        print("  WARNING -- every out-of-sample AUC is BELOW 0.5, including the "
              "macro-only\n  benchmark. That is not 'the press was wrong'; a "
              "specification that were merely\n  uninformative would sit AT "
              "0.5. Consistently-inverted forecasts across all\n  specs point "
              "at the design: with 15 onsets in 64 years and an expanding "
              "window,\n  each refit extrapolates from a handful of past "
              "recessions to a regime it has\n  not seen. Report the in-sample "
              "fit and this instability together; do not\n  quote the OOS AUC "
              "as a measure of press foresight.")
    print("  Read this against the claim-level models. A near-null at the "
          "claim level with\n  real signal here would say the press held "
          "information in AGGREGATE that no\n  individual forecast carried -- "
          "which is a finding about how to read newspapers,\n  not a "
          "contradiction.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(C.DEFAULT_INDEX),
                    help="build_press_index.py CSV")
    ap.add_argument("--horizons", type=int, nargs="+", default=[6, 12],
                    help="months ahead to predict recession onset")
    ap.add_argument("--smooth", type=int, default=3,
                    help="rolling-mean window on press features, in months")
    ap.add_argument("--min-train", type=int, default=240,
                    help="months of history before the first OOS forecast")
    ap.add_argument("--step", type=int, default=3,
                    help="refit every N months in the expanding-window loop")
    run(ap.parse_args())
