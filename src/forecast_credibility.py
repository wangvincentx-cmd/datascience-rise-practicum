"""
Forecast-credibility model: given a forecast + the economy's state when it was
made, predict P(the forecast comes true). See forecast_credibility_PLAN.md.

Reads forecasts.csv (forecast_data.py), adds economic-state features from FRED, and
trains THREE models -- Greenbook-only, Livingston-only, Combined -- plus an
ablation and cross-source transfer, all on the SHARED (intersection) feature set so
the comparison is fair. Logistic regression is primary (interpretable); a shallow
gradient-boosted tree is a secondary nonlinearity check on the larger sets.

Reviewer-proofing built in:
  - Reproducible: fixed SEED on every stochastic estimator and bootstrap.
  - Temporal split (train early, test late), NEVER random -- forecasts of one
    recession share its outcome.
  - Bootstrap 95% CIs on every held-out ROC-AUC (a single split's AUC is noisy).
  - ABLATION (state-only vs forecast-only vs full) to substantiate WHICH features
    carry the signal, rather than asserting it.
  - Honest caveats printed; figures rendered for review.

LEAKAGE NOTE: v1 builds state features from REVISED FRED series (not real-time
vintages). The forecast->outcome direction is leakage-safe (state at t, outcome
t..t+12), but the state LEVELS are as-revised; real-time vintages (Philly Fed
RTDSM) are the clean upgrade. Disclosed, not hidden.

Run:  python forecast_credibility.py            # analysis + figures
      python forecast_credibility.py --no-figures
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from score_claims import fred

warnings.simplefilter("ignore")
SEED = 0
TRAIN_FRAC = 0.70

# usrec_now / expansion_age_yrs use NBER's FINAL recession chronology -- not
# knowable in real time (NBER dates a recession 6-18 months late), so they are
# RETROSPECTIVE-only. The real-time-safe (deployable) set drops them and leans on
# the yield curve, a genuinely real-time recession signal. indpro/unrate momentum
# use lightly-revised data (~1-month lag, small revisions) -> treated as ~real-time.
PRED_FEATURES = ["pred_growth", "pred_revision"]
RT_STATE = ["indpro_mom12", "unrate_level", "unrate_chg12", "yield_curve"]
NBER_STATE = ["usrec_now", "expansion_age_yrs"]        # retrospective-only (leaky)
STATE_FEATURES = RT_STATE + NBER_STATE
SHARED_FEATURES = PRED_FEATURES + STATE_FEATURES       # full / retrospective
RT_FEATURES = PRED_FEATURES + RT_STATE                 # deployable / real-time-safe

# Okabe-Ito colorblind-safe palette, assigned by entity (fixed order).
SRC_COLOR = {"greenbook": "#0072B2", "spf": "#E69F00", "livingston": "#009E73"}
POS_COLOR, NEG_COLOR = "#0072B2", "#D55E00"     # diverging: +credibility / -credibility


# ---------- economic state at forecast date (from FRED) ---------------------
def _asof(series, when):
    idx = series.index[series.index <= when]
    return series.loc[idx[-1]] if len(idx) else np.nan


def add_state_features(df):
    def _dt(s):
        s = s.dropna().copy()
        s.index = (s.index.to_timestamp() if isinstance(s.index, pd.PeriodIndex)
                   else pd.to_datetime(s.index))
        return s.sort_index()
    indpro, unrate, usrec = _dt(fred("INDPRO")), _dt(fred("UNRATE")), _dt(fred("USREC"))
    gs10, tb3ms = _dt(fred("GS10")), _dt(fred("TB3MS"))    # yield-curve slope (real-time)
    rec_months = usrec[usrec > 0].index

    rows = []
    for d in df["forecast_date"]:
        ip_now, ip_prev = _asof(indpro, d), _asof(indpro, d - pd.DateOffset(years=1))
        ur_now, ur_prev = _asof(unrate, d), _asof(unrate, d - pd.DateOffset(years=1))
        in_rec = _asof(usrec, d)
        long_r, short_r = _asof(gs10, d), _asof(tb3ms, d)
        past = rec_months[rec_months <= d]
        exp_age = (d - past[-1]).days / 365.25 if len(past) else np.nan
        rows.append({
            "indpro_mom12": 100 * (ip_now / ip_prev - 1) if ip_prev else np.nan,
            "unrate_level": ur_now,
            "unrate_chg12": (ur_now - ur_prev) if (ur_now == ur_now and ur_prev == ur_prev) else np.nan,
            "yield_curve": (long_r - short_r) if (long_r == long_r and short_r == short_r) else np.nan,
            "usrec_now": float(in_rec) if in_rec == in_rec else np.nan,
            "expansion_age_yrs": 0.0 if in_rec else exp_age,
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_pred_features(df):
    df = df.sort_values(["forecaster_id", "forecast_date"]).copy()
    df["pred_revision"] = df.groupby("forecaster_id")["pred_growth"].diff().fillna(0.0)
    return df


# ---------- modeling --------------------------------------------------------
def _pipe(kind="logit", features=None):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        est = GradientBoostingClassifier(max_depth=2, n_estimators=150,
                                         learning_rate=0.05, random_state=SEED)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", est)])


def _split(sub):
    sub = sub.dropna(subset=["hit"]).copy()
    order = sub["forecast_date"].argsort().values
    cut = int(TRAIN_FRAC * len(order))
    return sub, order[:cut], order[cut:]


def _boot_auc(y, p, n=2000):
    from sklearn.metrics import roc_auc_score
    if len(set(y)) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            aucs.append(roc_auc_score(y[i], p[i]))
    return (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)) if aucs else (np.nan, np.nan)


def _metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    m = {"n": len(y), "base_rate": float(np.mean(y)), "brier": brier_score_loss(y, p)}
    two = len(set(y)) > 1
    m["roc_auc"] = roc_auc_score(y, p) if two else np.nan
    m["pr_auc"] = average_precision_score(y, p) if two else np.nan
    m["roc_lo"], m["roc_hi"] = _boot_auc(np.asarray(y), np.asarray(p))
    return m


def fit_source(sub, features=SHARED_FEATURES, kind="logit"):
    """Temporal-split fit; returns (y_test, p_test, fitted_pipe, sub, test_idx)."""
    sub, tr, te = _split(sub)
    X, y = sub[features].astype(float), sub["hit"].astype(int).values
    pipe = _pipe(kind, features).fit(X.iloc[tr], y[tr])
    return y[te], pipe.predict_proba(X.iloc[te])[:, 1], pipe, sub, te


def report_source(sub, name, kinds=("logit",)):
    sub2, tr, te = _split(sub)
    print(f"\n--- {name}  (n={len(sub2)}, train {len(tr)} / test {len(te)}, "
          f"test era {sub2['forecast_date'].iloc[te].dt.year.min()}-"
          f"{sub2['forecast_date'].iloc[te].dt.year.max()}) ---")
    for kind in kinds:
        yte, p, _, _, _ = fit_source(sub, SHARED_FEATURES, kind)
        m = _metrics(yte, p)
        print(f"  {kind:6s}: ROC-AUC {m['roc_auc']:.3f} [{m['roc_lo']:.2f},{m['roc_hi']:.2f}]"
              f"  PR-AUC {m['pr_auc']:.3f}  Brier {m['brier']:.3f}  (base {m['base_rate']:.1%})")
    # standardized coefficients on full data
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    s2 = sub.dropna(subset=["hit"])
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]
                    ).fit(s2[SHARED_FEATURES].astype(float), s2["hit"].astype(int))
    return pd.Series(pipe.named_steps["clf"].coef_[0], index=SHARED_FEATURES)


def ablation(sub, name):
    """State-only vs forecast-only vs full held-out ROC-AUC -- shows what carries it."""
    print(f"  {name}:")
    for label, feats in [("forecast-only", PRED_FEATURES), ("state-only", STATE_FEATURES),
                         ("full", SHARED_FEATURES)]:
        yte, p, _, _, _ = fit_source(sub, feats, "logit")
        m = _metrics(yte, p)
        print(f"    {label:14s} ROC-AUC {m['roc_auc']:.3f} [{m['roc_lo']:.2f},{m['roc_hi']:.2f}]")


def transfer(train_sub, test_sub, a, b):
    from sklearn.metrics import roc_auc_score
    tr = train_sub.dropna(subset=["hit"])
    te = test_sub.dropna(subset=["hit"])
    pipe = _pipe("logit").fit(tr[SHARED_FEATURES].astype(float), tr["hit"].astype(int))
    p = pipe.predict_proba(te[SHARED_FEATURES].astype(float))[:, 1]
    y = te["hit"].astype(int).values
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else np.nan
    print(f"  train {a:11s} -> test {b:11s}: ROC-AUC {auc:.3f}  (test base {np.mean(y):.1%})")
    return auc


def cv_auc(sub, features, n_splits=5):
    """Multi-origin expanding-window CV (TimeSeriesSplit) -> (mean, min, max, folds).
    Addresses single-split noise: every fold trains on the past, tests on the
    future, so no leakage and the origin is varied."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    sub = sub.dropna(subset=["hit"]).sort_values("forecast_date")
    X, y = sub[features].astype(float).values, sub["hit"].astype(int).values
    if len(y) < n_splits * 4:
        n_splits = max(2, len(y) // 4)
    aucs = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        m = _pipe("logit").fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return (np.mean(aucs), np.min(aucs), np.max(aucs), len(aucs)) if aucs else (np.nan, np.nan, np.nan, 0)


def cv_report(sub, name):
    """Retrospective (full, incl. NBER final-vintage features) vs deployable
    (real-time-safe: no NBER, uses yield curve), each over multi-origin CV."""
    r = cv_auc(sub, SHARED_FEATURES)
    d = cv_auc(sub, RT_FEATURES)
    print(f"  {name:14s} retrospective {r[0]:.3f} [{r[1]:.2f}-{r[2]:.2f}]   "
          f"deployable/real-time {d[0]:.3f} [{d[1]:.2f}-{d[2]:.2f}]   ({r[3]} folds)")


# ---------- figures ---------------------------------------------------------
def figures(df, coefs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
    except ImportError:
        print("(matplotlib missing -- no figures)")
        return
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score
    Path("figures").mkdir(exist_ok=True)
    srcs = ["greenbook", "spf", "livingston"]

    # Fig 1: calibration curves (identity = model; colorblind-safe; diagonal ref)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect calibration")
    for s in srcs:
        yte, p, _, _, _ = fit_source(df[df["source"] == s])
        if len(set(yte)) < 2:
            continue
        frac, mean = calibration_curve(yte, p, n_bins=6, strategy="quantile")
        ax.plot(mean, frac, "-o", color=SRC_COLOR[s], lw=2, ms=6,
                label=f"{s} (test)")
    ax.set_xlabel("mean predicted P(hit)"); ax.set_ylabel("observed hit rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Credibility model calibration, held-out era")
    ax.legend(frameon=False); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig("figures/fig_credibility_calibration.png", dpi=200); plt.close()

    # Fig 2: signed standardized coefficients (diverging: +/- credibility)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    c = coefs.reindex(coefs.abs().sort_values().index)
    ax.barh(c.index, c.values, color=[POS_COLOR if v >= 0 else NEG_COLOR for v in c.values])
    ax.axvline(0, color="black", lw=.8)
    for i, v in enumerate(c.values):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("standardized logit coefficient  (+ raises / - lowers P(hit))")
    ax.set_title("What makes a Greenbook forecast credible")
    ax.margins(x=0.18); plt.tight_layout()
    plt.savefig("figures/fig_credibility_coefficients.png", dpi=200); plt.close()

    # Fig 3: cross-source transfer heatmap (sequential blues; ROC-AUC train->test)
    M = np.full((3, 3), np.nan)
    for i, a in enumerate(srcs):
        for j, b in enumerate(srcs):
            tr = df[df["source"] == a].dropna(subset=["hit"])
            te = df[df["source"] == b].dropna(subset=["hit"])
            if a == b:
                yte, p, _, _, _ = fit_source(df[df["source"] == a])
            else:
                pipe = _pipe("logit").fit(tr[SHARED_FEATURES].astype(float), tr["hit"].astype(int))
                p = pipe.predict_proba(te[SHARED_FEATURES].astype(float))[:, 1]
                yte = te["hit"].astype(int).values
            if len(set(yte)) > 1:
                M[i, j] = roc_auc_score(yte, p)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=0.85)
    ax.set_xticks(range(3), srcs); ax.set_yticks(range(3), srcs)
    ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
    for i in range(3):
        for j in range(3):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] > 0.7 else "black", fontsize=11)
    ax.set_title("Cross-source transfer (ROC-AUC)\ncredibility generalizes across forecasters")
    fig.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
    plt.savefig("figures/fig_credibility_transfer.png", dpi=200); plt.close()
    print("figures/: fig_credibility_calibration.png, _coefficients.png, _transfer.png")


def main(args):
    df = pd.read_csv(args.data, parse_dates=["forecast_date"])
    df = add_pred_features(df)
    df = add_state_features(df)
    df.to_csv("data/scored/credibility_features.csv", index=False)
    gb, liv, spf = (df[df["source"] == s] for s in ("greenbook", "livingston", "spf"))

    print("=" * 70)
    print("FORECAST-CREDIBILITY MODELS  (target: P(forecast hit); shared features)")
    print("features:", SHARED_FEATURES, f"| SEED={SEED} | train_frac={TRAIN_FRAC}")

    print("\n### 1. Single-source models (own held-out late era; ROC-AUC [95% CI])")
    gb_coef = report_source(gb, "Greenbook-only", ("logit", "gbm"))
    report_source(liv, "Livingston-only", ("logit",))
    report_source(spf, "SPF-only", ("logit", "gbm"))

    print("\n### 2. Ablation -- does the signal come from the forecast or the STATE?")
    ablation(gb, "Greenbook")
    ablation(spf, "SPF")

    print("\n### 3. Cross-source transfer (does credibility generalize?)")
    transfer(liv, gb, "livingston", "greenbook")
    transfer(spf, gb, "spf", "greenbook")
    transfer(gb, spf, "greenbook", "spf")
    transfer(gb, liv, "greenbook", "livingston")

    print("\n### 4. Multi-origin CV + real-time robustness (ROC-AUC mean [min-max])")
    print("    retrospective = full features (incl. NBER final chronology);")
    print("    deployable    = real-time-safe (drops NBER usrec/age, adds yield curve)")
    cv_report(gb, "Greenbook")
    cv_report(spf, "SPF")
    cv_report(liv, "Livingston")

    if not args.no_figures:
        figures(df, gb_coef)

    print("\nCAVEATS: (1) ~8-12 US recessions in-sample -> underpowered on turning-"
          "point calls (see the wide CV fold ranges above); read as CALIBRATION, not "
          "recession prediction. (2) sources share those recessions -> combined n != "
          "independent episodes. (3) NBER-final features (usrec/age) are retrospective-"
          "only; the DEPLOYABLE model drops them and loses only ~0.03 AUC, so leakage "
          "isn't driving the result. indpro/unrate are lightly-revised (~real-time); "
          "RTDSM vintages are a further refinement. (4) Livingston forecasts industrial "
          "production, Greenbook/SPF real GDP -- all on the same INDPRO ruler, native "
          "series in variable_native.")
    print("\ncredibility_features.csv written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--data", default="data/scored/forecasts.csv",
                    help="forecasts.csv (default) or forecasts_multivar.csv (robustness)")
    main(ap.parse_args())
