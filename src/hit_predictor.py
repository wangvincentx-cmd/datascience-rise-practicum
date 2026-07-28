"""
P(this forecast comes true), from the forecast plus the economy it was made into.

Input at prediction time: what a newspaper predicted (direction, topic, hedging,
voice, horizon, ...) and the economic data that was ALREADY PUBLIC when it went
to print. Output: a calibrated probability the forecast turns out correct.
Trained on 14,251 historical forecasts whose answers we now know.

WHAT THIS ADDS OVER model_hit.py. That script answers a research question ("do
claim features add anything over macro?") and prints AUCs. This one is the
model itself: it fits, it is honestly evaluated, it reports WHICH inputs carry
the signal, it is calibrated, and it is SAVED so a new unresolved forecast can
be scored. The substantive difference in the model is the DIRECTION x ECONOMY
interaction block -- macro_context.py shows the economy's effect on accuracy has
opposite sign for optimistic and pessimistic forecasts, so a model with only
additive macro terms is structurally unable to see it, which is most of why the
macro-only baseline sits at chance.

HONEST EVALUATION, and the reasons:
  * LeaveOneGroupOut over 3-year periods. Never a random split: forecasts inside
    one period share one economy and one set of outcomes, so a random split
    trains and tests on the same answers.
  * Reported out-of-fold, pooled across folds -- not an average of per-fold AUCs
    of wildly different size.
  * ROC-AUC *and* PR-AUC *and* Brier, plus a calibration curve. AUC alone hides
    whether the probabilities mean anything.
  * A ladder of nested models, so the marginal contribution of each block is
    visible rather than asserted.
  * A block-permutation test: shuffle `hit` WITHIN period, refit, and see how
    often the full model's AUC is beaten by chance.
  * A block bootstrap over the same periods for the interval around the headline
    AUC, and a PAIRED one for the interaction rung's lift over the additive rung
    -- the permutation test only rules out zero signal, not "no better than the
    model one rung down", which is the claim this module actually makes.
  * Permutation importance computed on HELD-OUT folds, never on training data.

LEAKAGE DISCIPLINE: every economic input comes from macro_context.build_context,
which publication-lags each series (a forecast printed in month M sees only what
was public by M). `hindsight_in_recession` is dropped explicitly and asserted
against -- NBER dates recessions 6-21 months late, so it is an outcome, never a
feature.

Usage:
    python src/macro_context.py                   # -> data/scored/macro_context.csv
    python src/hit_predictor.py                   # train + evaluate + save
    python src/hit_predictor.py --perm 200        # with the permutation test
    python src/hit_predictor.py --predict-demo    # score example new forecasts
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from macro_context import BLOCK_YEARS, FACTORS, PRETTY, build_context

warnings.filterwarnings("ignore")

MODELS = Path("data/models")
MODEL_PATH = MODELS / "hit_predictor.joblib"
NUM_RE = re.compile(r"\d")
SEED = 0


def out_paths(context, exclude=None):
    """Namespace outputs by corpus AND ablation, so a replication or an
    ablation run cannot silently overwrite the primary model's artefacts (both
    did, once)."""
    stem = Path(context).stem
    pre = "" if stem == "macro_context" else f"{stem}_"
    if exclude:
        pre += f"no_{exclude.replace(',', '_')}_"
    MODELS.mkdir(parents=True, exist_ok=True)
    return {"model": MODELS / f"{pre}hit_predictor.joblib",
            "preds": MODELS / f"{pre}hit_predictions.csv",
            "results": MODELS / f"{pre}hit_predictor_results.json",
            "importance": MODELS / f"{pre}hit_predictor_importance.csv"}

# Optimistic vs pessimistic is the axis the economy's effect flips across, so it
# is what the interaction block multiplies by. `flat` gets 0: a "no change"
# forecast is not a directional bet and the sign of the economy's effect on it
# is not defined by this axis.
DIR_SIGN = {"improve": 1.0, "up": 1.0, "worsen": -1.0, "down": -1.0, "flat": 0.0}

# The economic factors that go into the interaction block. Restricted to the
# ones with real coverage across the century -- interacting a 1948+ series with
# direction would produce a term that is zero for 86% of the corpus and whose
# coefficient is estimated off one decade.
INTERACT_WITH = ["stock_ret6", "stock_drawdown", "epu", "ip_accel"]


# --- features --------------------------------------------------------------
def _col(df, name, default):
    """A column as a Series, or a full-length Series of `default` if absent.

    DataFrame.get() returns the bare default when a column is missing, not a
    Series -- which works while training (every column exists in the scored CSV)
    and breaks the moment a NEW forecast arrives without, say, `speaker_name`.
    Scoring unseen forecasts is the whole point of the model, so the missing
    case has to be the normal one."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def claim_features(df):
    """What the forecast itself says -- all knowable at print time."""
    out = pd.DataFrame(index=df.index)
    for c in ["direction", "topic", "voice", "scope", "confidence"]:
        out[f"c_{c}"] = _col(df, c, "na").fillna("na").astype(str)
    out["c_quoted"] = (_col(df, "is_quoted_forecaster", False).astype(str)
                       .isin(["True", "true", "1"]).astype(int))
    out["c_named"] = (_col(df, "speaker_name", "na").astype(str)
                      .str.lower().ne("na").astype(int))
    q = _col(df, "quote", "").astype(str)
    out["c_has_number"] = q.str.contains(NUM_RE).astype(int)
    out["c_len"] = q.str.split().apply(len).clip(0, 80)
    out["c_horizon"] = pd.to_numeric(_col(df, "horizon_used", 12),
                                     errors="coerce").fillna(12)
    return out


def macro_block(ctx):
    """Economic state, standardized names, missing marked not imputed-away.

    Each factor gets a `has_` flag alongside it. A zero-filled NaN without a
    flag would tell the model 'industrial production was flat in 1907' when the
    truth is that the series did not exist yet; the flag lets it learn that
    'unknown' is its own condition."""
    out = pd.DataFrame(index=ctx.index)
    for f in FACTORS:
        out[f"m_{f}"] = ctx[f]
        out[f"has_{f}"] = ctx[f].notna().astype(int)
    return out.fillna(0.0)


def interaction_block(df, ctx):
    """direction x economy -- the term the additive model cannot express.

    Predicting improvement into a rising market is a different bet from making
    the same call into a collapsing one, and predicting a downturn into that
    same rising market is a third thing again. Without this product, a single
    macro coefficient has to serve both groups and lands near zero."""
    sign = df["predicted_norm"].map(DIR_SIGN).fillna(0.0).values
    out = pd.DataFrame(index=ctx.index)
    out["x_dir_sign"] = sign
    for f in INTERACT_WITH:
        out[f"x_dir_{f}"] = sign * ctx[f].fillna(0.0).values
    return out


CLAIM_CAT = ["c_direction", "c_topic", "c_voice", "c_scope", "c_confidence"]
CLAIM_NUM = ["c_quoted", "c_named", "c_has_number", "c_len", "c_horizon"]
MACRO_NUM = [f"m_{f}" for f in FACTORS] + [f"has_{f}" for f in FACTORS]
INTER_NUM = ["x_dir_sign"] + [f"x_dir_{f}" for f in INTERACT_WITH]


def make_pipe(cat, num, clf=None):
    steps = []
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           min_frequency=15), cat))
    if num:
        steps.append(("num", StandardScaler(), num))
    return Pipeline([
        ("pre", ColumnTransformer(steps)),
        ("clf", clf or LogisticRegression(penalty="l2", C=0.5, max_iter=2000,
                                          random_state=SEED))])


# --- evaluation ------------------------------------------------------------
def _inner_group_splits(groups_tr, n_splits=5):
    """Splits for the calibrator, grouped by 3-year period like the outer CV.

    A calibrator fitted on rows the classifier already saw learns the training
    fit's over-confidence, not the deployed model's, and reports a curve that is
    too good. Grouping the inner splits by period (rather than at random) keeps
    the same discipline one level down: the calibration set is periods the
    classifier underneath it did not train on."""
    n = int(min(n_splits, len(np.unique(groups_tr))))
    if n < 2:
        return None
    return list(GroupKFold(n_splits=n).split(np.zeros(len(groups_tr)),
                                             groups=groups_tr))


def oof_predict(X, y, groups, cat, num, clf=None, calibrate=None):
    """Out-of-fold probabilities under LeaveOneGroupOut over 3-year periods.

    `calibrate` ("isotonic" or "sigmoid") wraps the pipeline in a nested,
    period-grouped CalibratedClassifierCV fitted INSIDE the training fold, so
    the held-out period stays untouched by both the classifier and its
    calibrator. Ranking is unchanged by design (both maps are monotone); what
    changes is whether the numbers can be read as odds."""
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        est = make_pipe(cat, num, clf)
        if calibrate:
            inner = _inner_group_splits(groups[tr])
            if inner is not None:
                est = CalibratedClassifierCV(est, method=calibrate, cv=inner)
        est.fit(X.iloc[tr], y[tr])
        oof[te] = est.predict_proba(X.iloc[te])[:, 1]
    return oof


def metrics(y, oof):
    ok = ~np.isnan(oof)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return {}
    return {"auc": roc_auc_score(y[ok], oof[ok]),
            "pr_auc": average_precision_score(y[ok], oof[ok]),
            "brier": brier_score_loss(y[ok], oof[ok]),
            "n": int(ok.sum())}


# --- uncertainty on the AUCs themselves ------------------------------------
# The permutation test asks "is this better than NO signal?". These ask the two
# questions it cannot: how wide is the interval around the headline number, and
# is the interaction rung distinguishable from the additive rung below it. Both
# resample whole 3-year PERIODS for the reason block_boot_corr does -- claims
# inside one period share one economy, so an i.i.d. bootstrap over 14,251 claims
# would report an interval roughly sqrt(claims-per-block) too narrow. The
# effective sample here is ~22 blocks, and the intervals should look like it.
def _block_index(blocks, ok):
    """Row indices of each block, restricted to rows with a prediction."""
    idx = np.where(ok)[0]
    b = np.asarray(blocks)[idx]
    return {u: idx[b == u] for u in np.unique(b)}


def _boot_draws(y, blocks, ok, reps, seed, stat):
    """Resample blocks with replacement; `stat(idx)` per draw, NaNs dropped."""
    rng = np.random.default_rng(seed)
    by_block = _block_index(blocks, ok)
    uniq = np.array(list(by_block))
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_block[b] for b in pick])
        if len(np.unique(y[idx])) < 2:      # a draw with one class has no AUC
            continue
        out.append(stat(idx))
    return np.asarray(out, dtype=float)


def block_boot_auc(y, oof, blocks, reps=2000, seed=SEED):
    """Pooled out-of-fold AUC with a percentile block-bootstrap 95% CI."""
    ok = ~np.isnan(oof)
    obs = roc_auc_score(y[ok], oof[ok])
    draws = _boot_draws(y, blocks, ok, reps, seed,
                        lambda i: roc_auc_score(y[i], oof[i]))
    if draws.size == 0:
        return obs, np.nan, np.nan
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def block_boot_auc_delta(y, oof_a, oof_b, blocks, reps=2000, seed=SEED):
    """AUC(b) - AUC(a) with a PAIRED block-bootstrap CI and two-sided p.

    Paired -- both models are scored on the same resampled blocks -- because the
    two rungs share nearly all their inputs and most of their error. Comparing
    two independently drawn intervals would ignore that correlation and make the
    difference look far less certain than it is.

    The p-value is the bootstrap analogue used by block_boot_corr: recentre the
    resampling distribution on the observed delta and ask how much of it sits at
    least as far from zero in absolute value."""
    ok = ~np.isnan(oof_a) & ~np.isnan(oof_b)
    obs = roc_auc_score(y[ok], oof_b[ok]) - roc_auc_score(y[ok], oof_a[ok])
    draws = _boot_draws(y, blocks, ok, reps, seed,
                        lambda i: (roc_auc_score(y[i], oof_b[i])
                                   - roc_auc_score(y[i], oof_a[i])))
    if draws.size == 0:
        return float(obs), np.nan, np.nan, np.nan
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = float((np.abs(draws - obs) >= abs(obs)).mean())
    return float(obs), float(lo), float(hi), p


LADDER = [
    ("1. base rate (no model)", [], []),
    ("2. claim features only", CLAIM_CAT, CLAIM_NUM),
    ("3. economy only", [], MACRO_NUM),
    ("4. claim + economy (additive)", CLAIM_CAT, CLAIM_NUM + MACRO_NUM),
    ("5. + direction x economy", CLAIM_CAT, CLAIM_NUM + MACRO_NUM + INTER_NUM),
]


def permutation_importance_grouped(X, y, groups, cat, num, base_auc, reps=5):
    """Drop in out-of-fold AUC when one input is shuffled.

    Shuffling happens INSIDE the held-out fold, so this measures what the input
    is worth on data the model never saw -- a training-set importance would just
    report what the model chose to lean on, not what actually generalizes."""
    rng = np.random.default_rng(SEED)
    rows = []
    for col in cat + num:
        drops = []
        for _ in range(reps):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            oof = oof_predict(Xp, y, groups, cat, num)
            m = metrics(y, oof)
            if m:
                drops.append(base_auc - m["auc"])
        if drops:
            rows.append({"feature": col, "drop": float(np.mean(drops)),
                         "sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("drop", ascending=False)


def run(args):
    ctx_path = Path(args.context)
    if not ctx_path.exists():
        raise SystemExit(f"{ctx_path} not found -- run `python src/macro_context.py` first.")
    d = pd.read_csv(ctx_path, low_memory=False)
    d = d[d["hit"].isin([0, 1])].reset_index(drop=True)
    paths = out_paths(args.context, args.exclude)

    # Leakage assertion, not a comment: the hindsight recession flag exists in
    # macro_context.csv for figures and must never reach a feature matrix.
    assert not any(c.startswith("hindsight") for c in MACRO_NUM + INTER_NUM), \
        "hindsight feature leaked into the model"

    y = d["hit"].astype(int).values
    year = pd.to_datetime(d["date"]).dt.year
    groups = ((year // BLOCK_YEARS) * BLOCK_YEARS).values

    cf = claim_features(d)
    mf = macro_block(d[FACTORS])
    xf = interaction_block(d, d[FACTORS])
    X = pd.concat([cf, mf, xf], axis=1)

    # Ablation. EPU in particular deserves this: it is the largest single
    # contributor, it did NOT replicate on the crisis corpus, and it is built
    # FROM newspaper text -- the same instrument as the corpus being scored --
    # so "how much survives without it" is a question a reviewer will ask.
    ladder = list(LADDER)
    if args.exclude:
        drop = {c for c in X.columns
                if any(f in c for f in args.exclude.split(","))}
        ladder = [(name, cat, [c for c in num if c not in drop])
                  for name, cat, num in ladder]
        print(f"  [ablation: dropped {len(drop)} columns matching "
              f"'{args.exclude}' -> {sorted(drop)}]")

    print(f"forecasts: {len(d)}   hit rate {y.mean():.3f}   "
          f"periods: {len(set(groups))}   features: {X.shape[1]}")

    print("\n=== nested ladder (out-of-fold, leave-one-3-year-period-out) ===")
    print(f"  {'model':<34}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}")
    results, best = {}, None
    for name, cat, num in ladder:
        if not cat and not num:
            # The null: predict the training base rate for every held-out claim.
            # Constant predictions have no ROC-AUC (0.5 by definition), so only
            # PR-AUC and Brier are meaningful.
            const = np.full(len(y), y.mean())
            print(f"  {name:<34}{'—':>9}{average_precision_score(y, const):>9.3f}"
                  f"{brier_score_loss(y, const):>8.3f}")
            continue
        oof = oof_predict(X, y, groups, cat, num)
        m = metrics(y, oof)
        results[name] = (m, oof, cat, num)
        print(f"  {name:<34}{m['auc']:>9.3f}{m['pr_auc']:>9.3f}{m['brier']:>8.3f}")
        best = name

    # Non-linear check on the same inputs. If the boosted model is far ahead,
    # the linear one is leaving structure on the table; if it is level, the
    # linear model is an honest summary and stays the primary.
    m_full, _, cat_f, num_f = results[best]
    oof_gb = oof_predict(X, y, groups, cat_f, num_f,
                         clf=HistGradientBoostingClassifier(
                             max_depth=3, max_iter=200, learning_rate=.06,
                             random_state=SEED))
    m_gb = metrics(y, oof_gb)
    print(f"  {'6. gradient boosting (same inputs)':<34}{m_gb['auc']:>9.3f}"
          f"{m_gb['pr_auc']:>9.3f}{m_gb['brier']:>8.3f}")

    auc_econ = results["3. economy only"][0]["auc"]
    auc_add = results["4. claim + economy (additive)"][0]["auc"]
    print(f"\n  interaction block adds {m_full['auc'] - auc_add:+.3f} AUC over "
          f"the additive model,\n  and the economy-only model rises from "
          f"{auc_econ:.3f} once direction is allowed to flip its sign.")

    # HONESTY GUARD, same one model_hit.py carries and for the same reason.
    # On an episode-selected corpus with few blocks, leave-one-block-out drops a
    # whole macro regime per fold, so every model extrapolates to conditions it
    # never trained on and the relationship INVERTS -- producing AUCs well below
    # chance that are an artefact of the CV structure, not a finding. Say so
    # loudly rather than let a reader interpret the ladder.
    n_groups = len(set(groups))
    if m_full["auc"] < 0.5:
        print(f"\n  ** every model is BELOW chance (full model {m_full['auc']:.3f}) "
              f"on {n_groups} blocks.")
        print("     Reading: ARTEFACT, not a result. Leave-one-block-out removes an "
              "entire macro\n     regime per fold; with this few blocks the models "
              "extrapolate and the sign flips.\n     This corpus cannot evaluate "
              "the ladder -- it needs the continuous monthly corpus,\n     which has "
              "many months per regime. Report the descriptive attribution "
              "(macro_context.py)\n     for this corpus instead, and do NOT read "
              "the interaction delta above.")
        return None

    # --- how wide is that number? ---
    # Reported before anything else downstream, because every sentence below
    # should be read against the interval, not the point estimate.
    _, oof_full, _, _ = results[best]
    auc_b, auc_lo, auc_hi = block_boot_auc(y, oof_full, groups, reps=args.boot)
    print(f"\n=== block bootstrap ({args.boot} resamples of {n_groups} periods) ===")
    print(f"  {best:<34}{auc_b:.3f}  95% CI [{auc_lo:.3f}, {auc_hi:.3f}]")
    add_name = "4. claim + economy (additive)"
    dlt, d_lo, d_hi, d_p = block_boot_auc_delta(
        y, results[add_name][1], oof_full, groups, reps=args.boot)
    print(f"  interaction lift over additive{dlt:>+9.3f}  "
          f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}]  p = {d_p:.4f}")
    if d_lo <= 0 <= d_hi:
        print("  ** the lift's interval spans zero -- the interaction rung is NOT\n"
              "     distinguishable from the additive rung on this corpus. Report it "
              "as such.")

    # --- calibration: do the probabilities mean anything? ---
    ok = ~np.isnan(oof_full)
    frac, mean_pred = calibration_curve(y[ok], oof_full[ok], n_bins=10,
                                        strategy="quantile")
    print("\n=== calibration (out-of-fold, decile bins) ===")
    print(f"  {'predicted':>10}{'actual':>10}")
    for p_, a_ in zip(mean_pred, frac):
        print(f"  {p_:>10.3f}{a_:>10.3f}")

    # Isotonic, fitted inside each training fold on held-out periods. Monotone,
    # so ranking (and therefore AUC) is untouched -- the only thing that can move
    # is Brier, i.e. whether the output may be quoted as odds or only as a rank.
    oof_cal = oof_predict(X, y, groups, cat_f, num_f, calibrate="isotonic")
    m_cal = metrics(y, oof_cal)
    ok_c = ~np.isnan(oof_cal)
    frac_c, mean_c = calibration_curve(y[ok_c], oof_cal[ok_c], n_bins=10,
                                       strategy="quantile")
    print(f"\n  isotonic (nested, period-grouped): Brier {m_full['brier']:.3f} "
          f"-> {m_cal['brier']:.3f}   AUC {m_full['auc']:.3f} -> {m_cal['auc']:.3f}")
    print(f"  {'predicted':>10}{'actual':>10}")
    for p_, a_ in zip(mean_c, frac_c):
        print(f"  {p_:>10.3f}{a_:>10.3f}")

    # --- what carries the signal ---
    if args.importance:
        print("\n=== permutation importance (held-out AUC drop when shuffled) ===")
        imp = permutation_importance_grouped(X, y, groups, cat_f, num_f,
                                             m_full["auc"], reps=args.imp_reps)
        for _, r in imp.head(15).iterrows():
            nice = PRETTY.get(r["feature"].replace("m_", "").replace("x_dir_", ""),
                              r["feature"])
            tag = "  [direction x]" if r["feature"].startswith("x_dir_") else ""
            print(f"  {r['feature']:<26}{r['drop']:+.4f} (sd {r['sd']:.4f})"
                  f"  {nice}{tag}")
        imp.to_csv(paths["importance"], index=False)
        print(f"  -> {paths['importance']}")

    # --- is the whole thing better than chance? ---
    if args.perm:
        print(f"\n=== block-permutation test ({args.perm} shuffles) ===")
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(args.perm):
            yp = y.copy()
            for g in set(groups):
                idx = np.where(groups == g)[0]
                yp[idx] = rng.permutation(yp[idx])
            m = metrics(yp, oof_predict(X, yp, groups, cat_f, num_f))
            if m:
                null.append(m["auc"])
        if null:
            p = (1 + sum(a >= m_full["auc"] for a in null)) / (1 + len(null))
            print(f"  observed AUC {m_full['auc']:.3f}   "
                  f"null mean {np.mean(null):.3f}   p = {p:.4f}")

    # --- fit on everything and save ---
    # The ladder above is the honest estimate of how well this generalizes; the
    # deployed model should then use every historical example available.
    final = make_pipe(cat_f, num_f)
    final.fit(X, y)
    try:
        import joblib
        joblib.dump({"pipeline": final, "cat": cat_f, "num": num_f,
                     "trained_n": len(d), "oof_auc": m_full["auc"],
                     "oof_pr_auc": m_full["pr_auc"], "brier": m_full["brier"]},
                    paths["model"])
        print(f"\n-> {paths['model']}  (fit on all {len(d)} forecasts; "
              f"honest out-of-fold AUC {m_full['auc']:.3f})")
    except ImportError:
        print("\n[joblib not installed -- model evaluated but not saved]")

    pd.DataFrame({"date": d["date"], "quote": d.get("quote"),
                  "predicted_norm": d["predicted_norm"], "hit": y,
                  "p_hit_oof": oof_full}).to_csv(paths["preds"], index=False)
    print(f"-> {paths['preds']}  (out-of-fold probability per forecast)")

    # Figures read these numbers rather than repeating the fit -- so a chart can
    # never drift out of sync with the model it is describing.
    ladder = {name: {"auc": m["auc"], "pr_auc": m["pr_auc"], "brier": m["brier"]}
              for name, (m, _, _, _) in results.items()}
    ladder["6. gradient boosting (same inputs)"] = {
        "auc": m_gb["auc"], "pr_auc": m_gb["pr_auc"], "brier": m_gb["brier"]}
    paths["results"].write_text(json.dumps(
        {"n": len(d), "base_rate": float(y.mean()),
         "n_periods": int(len(set(groups))), "ladder": ladder,
         "auc_ci": {"auc": auc_b, "lo": auc_lo, "hi": auc_hi,
                    "reps": args.boot},
         "interaction_lift": {"delta": dlt, "lo": d_lo, "hi": d_hi, "p": d_p},
         "calibration": {"predicted": mean_pred.tolist(),
                         "actual": frac.tolist()},
         "calibration_isotonic": {"predicted": mean_c.tolist(),
                                  "actual": frac_c.tolist(),
                                  "brier": m_cal["brier"],
                                  "auc": m_cal["auc"]}}, indent=2))
    print(f"-> {paths['results']}")
    return final


# --- using it on a new forecast -------------------------------------------
# Factors the model leans on hard enough that losing one makes its output
# meaningless rather than merely noisier (from the held-out permutation
# importance: direction x epu = 0.038, direction x stock return = 0.016, and
# every other input is an order of magnitude smaller).
LOAD_BEARING = ["epu", "stock_ret6"]


def predict_new(forecasts, allow_missing=False):
    """Score unresolved forecasts. `forecasts` is a list of dicts with the
    extractor's fields plus a `date`; the economy is looked up from that date.

    Raises if a load-bearing economic factor could not be loaded. This is
    deliberate: `macro_block` fills missing factors with 0.0 and flips a `has_`
    flag, which is right during TRAINING (the flag lets the model learn that
    pre-1919 is its own regime) but silently catastrophic at PREDICT time
    against a model trained WITH that factor. It is not a subtle degradation --
    when EPU failed to load once, the same 1929 forecast scored 0.269 instead of
    0.608, i.e. the answer inverted. A loud failure beats a confident wrong
    number. Pass allow_missing=True only if you genuinely want the degraded
    estimate and will label it as such."""
    import joblib
    bundle = joblib.load(MODEL_PATH)
    df = pd.DataFrame(forecasts)
    df["horizon_used"] = df.get("horizon_months", 12)
    if "predicted_norm" not in df:
        df["predicted_norm"] = df["direction"]
    ctx = build_context(df["date"])

    missing = [f for f in LOAD_BEARING if f in ctx and ctx[f].isna().all()]
    if missing and not allow_missing:
        raise RuntimeError(
            f"cannot score: load-bearing factor(s) {missing} unavailable for "
            f"every row, but the model was trained with them. Predictions would "
            f"be confidently wrong, not merely noisier. Fix the data source "
            f"(see the '[epu unavailable: ...]' message above for the cause), "
            f"or pass allow_missing=True to accept a degraded estimate.")
    if missing:
        print(f"  WARNING: scoring WITHOUT {missing} -- degraded, do not report "
              f"these as the model's probabilities")

    X = pd.concat([claim_features(df), macro_block(ctx),
                   interaction_block(df, ctx)], axis=1)
    return bundle["pipeline"].predict_proba(X)[:, 1]


# Deliberately an UNFLATTERING demo. In June 1929 the model gives the bull 0.61
# and the bear 0.23 -- and the bear was right within four months. It is reading
# the pre-crash market's momentum, which is exactly what it was trained to do
# and exactly why it cannot call turning points: the conditions that make an
# upbeat forecast usually right are the same conditions that precede a crash.
# Anyone demoing this should show these three, not cherry-picked wins.
DEMO = [
    {"date": "1929-06-15", "direction": "improve", "topic": "markets",
     "voice": "expert", "scope": "national", "confidence": "assertive",
     "quote": "Stock prices will continue their advance through the coming year.",
     "horizon_months": 12},
    {"date": "1929-06-15", "direction": "worsen", "topic": "markets",
     "voice": "expert", "scope": "national", "confidence": "assertive",
     "quote": "The market is dangerously overextended and must break.",
     "horizon_months": 12},
    {"date": "1933-03-01", "direction": "improve", "topic": "general_business",
     "voice": "official", "scope": "national", "confidence": "assertive",
     "quote": "Business will recover substantially within the year.",
     "horizon_months": 12},
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default="data/scored/macro_context.csv")
    ap.add_argument("--perm", type=int, default=0,
                    help="block-permutation shuffles (0 = skip; 200 is a real test)")
    ap.add_argument("--importance", action="store_true", default=True)
    ap.add_argument("--no-importance", dest="importance", action="store_false")
    ap.add_argument("--imp-reps", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000,
                    help="block-bootstrap resamples for the AUC intervals")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated factor substrings to ablate, e.g. 'epu'")
    ap.add_argument("--predict-demo", action="store_true",
                    help="score the built-in example forecasts with the saved model")
    args = ap.parse_args()

    if args.predict_demo:
        for f, p in zip(DEMO, predict_new(DEMO)):
            print(f"  P(comes true) = {p:.3f}   {f['date']}  "
                  f"[{f['direction']}/{f['topic']}]  \"{f['quote'][:60]}...\"")
    else:
        run(args)
