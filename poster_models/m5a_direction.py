"""MODEL 5a -- Predict the ERROR DIRECTION, not just right-or-wrong.

The problem this solves
-----------------------
`hit` is a single bit, and collapsing to it destroys the most interesting thing
in the data. A forecast that called improvement during a contraction and one
that called contraction during a boom both score 0, but they are opposite
failures -- and CLAUDE.md's own rule says the metric is "hit rate, ERROR
DIRECTION, Brier, and confidence intervals", not accuracy. Error direction has
no model behind it yet. This is that model.

Every claim is re-expressed on a three-way outcome:

    TOO OPTIMISTIC  the forecast was rosier than what happened
    CORRECT         the forecast matched
    TOO PESSIMISTIC the forecast was gloomier than what happened

and fitted two ways:

  - MULTINOMIAL LOGIT on the three-way outcome, which lets a feature push
    toward optimism and pessimism by different amounts. Hedging might reduce
    over-optimism without touching over-pessimism; a binary hit model cannot
    express that and a single ordered coefficient cannot either.
  - ORDERED LOGIT on the signed error, which imposes that the three categories
    sit on one line. Stricter, one coefficient per feature, easy to read. The
    Brant-style comparison between the two is reported: if the multinomial
    coefficients for optimism and pessimism are near mirror images, the ordered
    model is the honest summary; if they are not, the effect is genuinely
    asymmetric and the ordered model is hiding it.

Two definitions, kept apart on purpose
--------------------------------------
"Improve" and "up" both mean the underlying series rose, but rising output is
good news and rising prices are not. So:

  - SERIES DIRECTION (+1 rose / 0 flat / -1 fell) is defined for every topic and
    is what the multinomial and ordered models use.
  - OPTIMISM (+1 good news / -1 bad news) is defined ONLY for business, markets
    and employment, where welfare direction is unambiguous. Price claims are
    excluded from every optimism number rather than assigned a sign by guess.

Conflating the two would silently encode "inflation is good" and would leak
straight into the headline optimism-bias figure.

    python poster_models/m5a_direction.py
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

import _common as C

warnings.filterwarnings("ignore")

LABELS = {-1: "too pessimistic", 0: "correct", 1: "too optimistic"}


def build_outcomes(df):
    """Signed series direction, signed optimism, and the three-way error."""
    out = pd.DataFrame(index=df.index)
    out["pred_sign"] = [C.signed_direction(p, t) for p, t in
                        zip(df["predicted_norm"], df["topic"])]
    out["real_sign"] = [C.signed_direction(r, t) for r, t in
                        zip(df["realized"], df["topic"])]
    out["pred_optim"] = [C.optimism_sign(p, t) for p, t in
                         zip(df["predicted_norm"], df["topic"])]
    out["real_optim"] = [C.optimism_sign(r, t) for r, t in
                         zip(df["realized"], df["topic"])]
    # Error on the optimism scale where it is defined; sign of the series-
    # direction error elsewhere is NOT interpretable as optimism, so it is left
    # missing rather than filled.
    err = out["pred_optim"] - out["real_optim"]
    out["error_optim"] = np.sign(err)
    out["error_size"] = err
    return out


def descriptive(df, o):
    """Who was wrong in which direction? The table before any model."""
    d = df.copy()
    d["error_optim"] = o["error_optim"].values
    have = d[d["error_optim"].notna()].copy()
    print(f"\n=== error direction, overall ===")
    print(f"  optimism is defined for {len(have):,} of {len(d):,} claims "
          f"(price claims excluded -- see module docstring)")
    vc = have["error_optim"].value_counts(normalize=True).sort_index()
    for k in (-1.0, 0.0, 1.0):
        print(f"    {LABELS[int(k)]:<18} {vc.get(k, 0):.3f}")
    bias = have["error_optim"].mean()
    print(f"  mean signed error {bias:+.3f}   "
          f"({'systematically OVER-optimistic' if bias > 0.02 else 'systematically OVER-pessimistic' if bias < -0.02 else 'no net directional bias'})")

    print("\n=== error direction by feature ===")
    print("  Two forecasts can share a hit rate and fail in opposite "
          "directions.\n")
    rows = []
    for col in ["confidence", "voice", "topic", "predicted_norm"]:
        if col not in have:
            continue
        g = have.groupby(have[col].fillna("na"))
        t = pd.DataFrame({
            "n": g.size(),
            "too_pess": g["error_optim"].apply(lambda s: (s < 0).mean()),
            "correct": g["error_optim"].apply(lambda s: (s == 0).mean()),
            "too_optim": g["error_optim"].apply(lambda s: (s > 0).mean()),
            "net_bias": g["error_optim"].mean(),
        })
        t = t[t["n"] >= 25].sort_values("net_bias", ascending=False)
        if not len(t):
            continue
        print(f"  by {col}:")
        print(f"    {'level':<20}{'n':>7}{'too pess':>10}{'correct':>10}"
              f"{'too optim':>11}{'net':>9}")
        for k, r in t.iterrows():
            print(f"    {str(k):<20}{int(r['n']):>7}{r['too_pess']:>10.3f}"
                  f"{r['correct']:>10.3f}{r['too_optim']:>11.3f}"
                  f"{r['net_bias']:>+9.3f}")
        t = t.reset_index().rename(columns={t.index.name or "index": "level"})
        t.insert(0, "feature", col)
        rows.append(t)
    if rows:
        C.save(pd.concat(rows, ignore_index=True), "m5a_error_by_feature.csv")


def fit_multinomial(y3, X, blocks):
    """MNLogit with 'correct' as the reference outcome.

    Reference is the middle category deliberately: every coefficient then reads
    as 'how much does this feature move a forecast from being right toward
    failing in THIS direction', which is the question."""
    Xc = sm.add_constant(X, has_constant="add")
    codes = pd.Categorical(y3, categories=[0.0, -1.0, 1.0])  # 0 = reference
    res = sm.MNLogit(codes.codes, Xc).fit(disp=0, maxiter=200,
                                          cov_type="cluster",
                                          cov_kwds={"groups": blocks})
    return res, Xc.columns


def run(args):
    C.header("MODEL 5a: error direction",
             "Too optimistic / correct / too pessimistic, instead of a single "
             "hit bit.")

    df = C.load_scored(args.scored, args.rigid)
    o = build_outcomes(df)
    blocks = C.time_blocks(df, args.block_years)

    print(f"\n  claims {len(df):,}   blocks {len(set(blocks))}")
    descriptive(df, o)

    # --- modelling set: claims where optimism is defined ---------------------
    m = o["error_optim"].notna().values
    dfm = df[m].reset_index(drop=True)
    y3 = o.loc[m, "error_optim"].values
    blk = blocks[m]
    X = C.standardize(C.drop_collinear(C.claim_design(dfm)))
    # `direction` dummies would be near-deterministic here: predicting 'worsen'
    # mechanically makes over-pessimism possible and over-optimism impossible
    # when the economy improved. Keeping them would make the model explain the
    # outcome with its own definition.
    X = X[[c for c in X.columns if not c.startswith("c_direction")]]

    print(f"\n  modelling {len(dfm):,} claims with a defined optimism sign; "
          f"{X.shape[1]} features")
    print("  (direction dummies dropped: they define the outcome, "
          "see source comment)")

    res, cols = fit_multinomial(y3, X, blk)
    # MNLogit params: columns are the non-reference outcomes in category order.
    for j, lab in enumerate([-1.0, 1.0]):
        tab = C.coef_table(
            cols, res.params.iloc[:, j].values, res.bse.iloc[:, j].values,
            f"MULTINOMIAL logit: '{LABELS[int(lab)]}' vs 'correct'",
            note="Positive = the feature makes this KIND of error more likely, "
                 "relative to\n  being right. Block-clustered SEs.",
            n=len(dfm))
        C.save(tab, f"m5a_mnlogit_{'pess' if lab < 0 else 'optim'}.csv")

    # --- is the effect symmetric? -------------------------------------------
    b_pess = res.params.iloc[:, 0].values
    b_optim = res.params.iloc[:, 1].values
    print("\n=== is each effect symmetric, or does it push one way only? ===")
    print("  If a feature's two coefficients are mirror images, an ordered "
          "model captures\n  it. If they share a sign, the feature makes "
          "forecasts WRONGER in general\n  rather than shifting them "
          "optimistic or pessimistic.\n")
    print(f"  {'term':<26}{'-> too pess':>13}{'-> too optim':>14}   reading")
    for i, c in enumerate(cols):
        if c == "const":
            continue
        a, b = b_pess[i], b_optim[i]
        if abs(a) < 0.05 and abs(b) < 0.05:
            read = "no effect"
        elif np.sign(a) == np.sign(b):
            read = "both directions (accuracy, not bias)"
        else:
            read = "shifts bias " + ("toward optimism" if b > a else
                                     "toward pessimism")
        print(f"  {c:<26}{a:>+13.3f}{b:>+14.3f}   {read}")

    sym = pd.DataFrame({"term": cols, "to_too_pessimistic": b_pess,
                        "to_too_optimistic": b_optim})
    C.save(sym, "m5a_symmetry.csv")

    # --- ordered logit -------------------------------------------------------
    print("\n  fitting ordered logit on the signed error ...")
    try:
        om = OrderedModel(pd.Series(y3).astype("category"), X,
                          distr="logit").fit(method="bfgs", disp=0)
        k = X.shape[1]
        tab = C.coef_table(
            X.columns, om.params[:k].values, om.bse[:k].values,
            "ORDERED logit on signed error (pessimistic < correct < optimistic)",
            note="One coefficient per feature: positive = shifts the whole "
                 "distribution\n  toward over-optimism. Assumes the two "
                 "thresholds share a slope -- check that\n  against the "
                 "symmetry table above before quoting it.",
            n=len(dfm))
        C.save(tab, "m5a_ordered.csv")
    except Exception as e:
        print(f"  ordered logit failed to converge: {e}")

    # --- how big is the bias, in plain units? -------------------------------
    have = o.loc[m]
    over = float((have["error_optim"] > 0).mean())
    under = float((have["error_optim"] < 0).mean())
    print(f"\n  Headline for the poster: of forecasts that were WRONG, "
          f"{over / (over + under):.1%} were\n  wrong by being too optimistic. "
          f"The press did not just fail to see downturns --\n  it failed in a "
          f"consistent direction.")
    C.save(pd.DataFrame([{
        "n_with_optimism_sign": int(m.sum()), "share_too_optimistic": over,
        "share_too_pessimistic": under,
        "share_correct": float((have["error_optim"] == 0).mean()),
        "mean_signed_error": float(have["error_optim"].mean()),
        "share_of_errors_optimistic": over / (over + under),
    }]), "m5a_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    run(ap.parse_args())
