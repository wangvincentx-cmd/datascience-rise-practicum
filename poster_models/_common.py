"""Shared plumbing for the poster_models suite.

Why this folder exists at all, given that `src/` is deliberately flat: these are
not pipeline stages. They are seven alternative *statistical treatments* of one
already-built artefact (`data/scored/monthly_scored.csv`), each answering the
same question a different way. Grouping them keeps the pipeline unambiguous --
nothing in here produces data anything else consumes.

The cost of the folder is one sys.path line, paid once, here. Every module
imports from this file rather than repeating it.

Design rules inherited from the project and enforced throughout:
  - no hindsight: macro features are publication-lagged, NBER status is never
    an input, only ever an outcome
  - never split randomly: every CV split and every standard error is grouped by
    time block, because the effective sample is ~21 blocks, not 14,251 claims
  - unscorable stays unscored
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# truth_data.CACHE is Path("cache") -- relative to the WORKING DIRECTORY, not to
# the repo. Run a model from inside poster_models/ and load_fred looks for
# poster_models/cache/, finds nothing, creates the directory and tries to
# re-download every series. RISE_FRED_DIR is the project's own supported
# override (it is how the scorer runs inside the ProQuest TDM sandbox), so
# pinning it here makes these scripts runnable from anywhere without touching
# src/. An explicit setting from the caller always wins.
os.environ.setdefault("RISE_FRED_DIR", str(ROOT / "cache"))

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

DEFAULT_SCORED = ROOT / "data" / "scored" / "monthly_scored.csv"
DEFAULT_INDEX = ROOT / "data" / "scored" / "press_index.csv"

# Width of a time block, in years. Three years is the project's standing choice
# (model_hit.py): far coarser than the autocorrelation length of the business
# cycle -- so claims in one block share a macro regime and cannot leak across
# folds -- while still giving ~21 blocks over 1900-1963.
BLOCK_YEARS = 3


def add_common_args(ap):
    """Arguments every model in this folder accepts."""
    ap.add_argument("--scored", default=str(DEFAULT_SCORED),
                    help="score_predictions.py CSV (default: monthly corpus)")
    ap.add_argument("--rigid", action="store_true",
                    help="real-horizon claims only (drops defaulted windows)")
    ap.add_argument("--block-years", type=int, default=BLOCK_YEARS,
                    help="width of a CV / clustering time block, in years")
    return ap


def load_scored(path=None, rigid=False):
    """Scorable, graded claims with a `block` column attached.

    Mirrors model_hit.run()'s filtering exactly so every number in this folder
    is computed on the same rows as the poster's headline model."""
    df = pd.read_csv(path or DEFAULT_SCORED, low_memory=False)
    df = df[df["scorable"] == True].copy()
    if rigid and "horizon_basis" in df:
        df = df[df["horizon_basis"] != "default"].copy()
    df = df[df["hit"].isin([0, 1])].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].reset_index(drop=True)
    return df


def time_blocks(df, years=BLOCK_YEARS):
    """Group labels for CV and clustering. Never a random split -- see CLAUDE.md."""
    yr = df["date"].dt.year
    return ((yr // years) * years).astype(int).astype(str).values


def month_index(df):
    """Calendar month of publication, as a string. The finest honest grouping."""
    return df["date"].dt.to_period("M").astype(str).values


def dummies(series, prefix, min_count=25, drop_first=True):
    """One-hot a categorical, pooling rare levels away first.

    Rare levels are the classic source of separation: a level appearing in one
    month is perfectly explained by that month's intercept, and in a bootstrap
    resample it may vanish or become fully separated, which sends the estimated
    coefficient and its standard error to absurd values.

    Two-step pooling. Levels below `min_count` are collected into 'other'; then,
    if 'other' is ITSELF below `min_count`, those rows are folded into the modal
    level so they are absorbed by the reference category instead of getting a
    dummy of their own. Without the second step, `direction` (which has 10 rows
    across 'unclear' and 'up') produces a ten-observation dummy whose bootstrap
    standard error came out at 70 log-odds -- a number that says nothing except
    that the level should not have been there."""
    s = series.fillna("na").astype(str)
    counts = s.value_counts()
    keep = set(counts[counts >= min_count].index)
    s = s.where(s.isin(keep), "other")
    if (s == "other").sum() < min_count:
        s = s.replace("other", counts.idxmax())
    return pd.get_dummies(s, prefix=prefix, drop_first=drop_first, dtype=float)


def claim_design(df, min_count=25):
    """Numeric design matrix of print-time claim features, no intercept.

    Deliberately the same information as model_hit.claim_features, re-expressed
    as plain floats so statsmodels can consume it. Kept separate from the
    sklearn path rather than shared, because the two need different encodings
    (reference-level dropping vs one-hot-everything) and silently reusing one
    for the other is how collinearity bugs get in."""
    X = pd.DataFrame(index=df.index)
    for c in ["direction", "topic", "voice"]:
        if c in df.columns:
            X = pd.concat([X, dummies(df[c], f"c_{c}", min_count)], axis=1)
    X["c_hedged"] = (df.get("confidence", "").astype(str) == "hedged").astype(float)
    X["c_quoted"] = (df.get("is_quoted_forecaster", False).astype(str)
                     .isin(["True", "true", "1"]).astype(float))
    X["c_named"] = (df.get("speaker_name", "na").astype(str)
                    .str.lower().ne("na").astype(float))
    q = (df["quote"].astype(str) if "quote" in df.columns
         else pd.Series([""] * len(df), index=df.index))
    X["c_has_number"] = q.str.contains(r"\d").astype(float)
    X["c_len"] = q.str.split().apply(len).clip(0, 80).astype(float)
    X["c_horizon"] = pd.to_numeric(df.get("horizon_used"), errors="coerce").fillna(12.0)
    return X


FRED_SERIES = ["INDPRO", "CPIAUCNS", "UNRATE"]  # + STOCK_SERIES, resolved lazily

MACRO_MISSING_NOTE = (
    "Run  python poster_models/fetch_fred.py  once with network access to fill "
    "it;\n  cache/ is gitignored, so a fresh clone starts with nothing.")


def fred_status():
    """Which FRED series are cached on disk. (cache/ is gitignored.)"""
    from truth_data import STOCK_SERIES
    want = FRED_SERIES + [STOCK_SERIES]
    cache = Path(os.environ["RISE_FRED_DIR"])
    return {s: (cache / f"fred_{s}.csv").exists() for s in want}


def _absent_series():
    """A stand-in for a series that is not cached.

    One NaN value indexed a century in the future. Every coverage guard in
    truth_data and model_hit is of the form `if p < series.index.min(): give
    up`, so a far-future index makes them all give up cleanly -- returning the
    'outside coverage' answer they already know how to return. An EMPTY series
    would not do: .asof() raises IndexError on it, and .index.min() is NaT,
    which compares False against everything and slips past the guards.

    The point is to reuse the audited missing-coverage path rather than write a
    second, parallel definition of what a macro feature is."""
    return pd.Series([np.nan], index=pd.PeriodIndex(["2100-01"], freq="M"))


def _patch_missing_fred(module):
    """Point `module.load_fred` at the sentinel for series that aren't cached."""
    import truth_data
    have = fred_status()
    real = truth_data.load_fred

    def loader(sid):
        if have.get(sid, True):
            return real(sid)
        return _absent_series()

    module.load_fred = loader
    return [s for s, ok in have.items() if not ok]


def report_fred(missing, context):
    if not missing:
        return
    print(f"\n  !! FRED cache incomplete: missing {', '.join(missing)}.")
    print(f"     {context}")
    print(f"     {MACRO_MISSING_NOTE}")


def macro_design(df, required=True):
    """Publication-lagged macro state. Delegates to the audited implementation.

    Degrades rather than dies when only some series are cached: the missing
    ones come through as NaN, which model_hit.macro_features already handles by
    zero-filling and setting its `m_has_*` indicator columns to 0. The model
    therefore knows the difference between 'unemployment was 5%' and 'nobody
    measured unemployment yet', which is the same distinction it has to make
    for 1900 anyway.

    `required=True` still fails hard when NOTHING is cached -- a model whose
    question is 'what does wording add OVER the economy' must not quietly
    answer a different question."""
    import model_hit
    missing = _patch_missing_fred(model_hit)
    if len(missing) == len(fred_status()):
        if required:
            raise SystemExit(
                f"\nERROR: no FRED series cached, so there is no macro state to "
                f"control for.\n  {MACRO_MISSING_NOTE}")
        print(f"  NOTE: no FRED data cached; proceeding WITHOUT macro controls.")
        return None
    report_fred(missing, "Those macro features will be all-zero with their "
                         "m_has_* flag set to 0.")
    m = model_hit.macro_features(df["date"])
    m.index = df.index
    return m


def truth_data():
    """TruthData built from whatever FRED series are cached.

    Topics whose series is absent come back unscorable with a reason, which is
    the project's standing rule -- unscorable means unscored, never guessed --
    rather than being silently scored against the wrong series."""
    import truth_data as td
    have = fred_status()
    from truth_data import STOCK_SERIES
    kw = {}
    for name, sid in [("indpro", "INDPRO"), ("cpi", "CPIAUCNS"),
                      ("unrate", "UNRATE"), ("stocks", STOCK_SERIES)]:
        if not have.get(sid, True):
            kw[name] = _absent_series()
    missing = [s for s, ok in have.items() if not ok]
    report_fred(missing, "Claims scored against those series will drop out of "
                         "this model as uncovered.")
    return td.TruthData(**kw), missing


def standardize(X, cols=None):
    """Z-score continuous columns so logit coefficients are comparable in size.

    Binary 0/1 columns are left alone: a 'one standard deviation of hedged'
    is not a thing anyone can interpret."""
    X = X.copy()
    cols = cols or [c for c in X.columns if X[c].nunique() > 2]
    for c in cols:
        sd = X[c].std()
        if sd and np.isfinite(sd):
            X[c] = (X[c] - X[c].mean()) / sd
    return X


def drop_collinear(X, tol=1e-8):
    """Drop columns that are exact linear combinations of earlier ones.

    Fixed-effects and residualized designs both routinely produce a rank-
    deficient X; statsmodels will happily return NaN standard errors rather
    than complain, so the check is done here instead of being discovered later
    in a results table."""
    keep, seen = [], np.zeros((len(X), 0))
    for c in X.columns:
        v = X[c].values.astype(float).reshape(-1, 1)
        if seen.shape[1]:
            resid = v - seen @ np.linalg.lstsq(seen, v, rcond=None)[0]
        else:
            resid = v
        if float((resid ** 2).sum()) > tol * max(1.0, float((v ** 2).sum())):
            keep.append(c)
            seen = np.hstack([seen, v])
    return X[keep]


def signed_direction(label, topic):
    """Map a direction label to +1/0/-1 on ONE consistent scale: did the
    underlying series go UP?

    Not a good/bad scale. 'improve' for business and 'up' for prices both mean
    the series rose, but rising prices are not good news; conflating the two is
    the easy mistake here. Goodness is handled separately, and only for the
    topics where it is defined (see optimism_sign)."""
    s = str(label).lower()
    if s in ("improve", "up"):
        return 1.0
    if s in ("worsen", "down"):
        return -1.0
    if s in ("flat", "no_change", "stable"):
        return 0.0
    return np.nan


def optimism_sign(label, topic):
    """+1 if the label is GOOD news, -1 if bad, 0 if neutral, NaN if undefined.

    Defined only for topics with an unambiguous welfare direction: business,
    markets and employment (unemployment up = bad). Price direction is left
    undefined -- whether inflation is good news depends on the decade, and
    guessing would quietly bake an assumption into the optimism-bias number."""
    s = str(label).lower()
    if topic in ("general_business", "markets", "industry", "other"):
        return {"improve": 1.0, "worsen": -1.0, "flat": 0.0,
                "no_change": 0.0, "stable": 0.0}.get(s, np.nan)
    if topic == "employment":
        # 'up' here means the unemployment rate rose.
        return {"up": -1.0, "down": 1.0, "flat": 0.0,
                "no_change": 0.0, "stable": 0.0}.get(s, np.nan)
    return np.nan


def stars(p):
    if not np.isfinite(p):
        return "   "
    return "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "


def coef_table(names, coefs, ses, title, note=None, n=None, odds=True):
    """Print, and return, a coefficient table with 95% CIs.

    Every model in this folder reports on this one format, so a reader can put
    two of them side by side. Confidence intervals are printed always -- the
    project's rules forbid reporting a point estimate on its own."""
    from scipy import stats as sps
    coefs = np.asarray(coefs, dtype=float)
    ses = np.asarray(ses, dtype=float)
    z = np.divide(coefs, ses, out=np.full_like(coefs, np.nan), where=ses > 0)
    p = 2 * (1 - sps.norm.cdf(np.abs(z)))
    tab = pd.DataFrame({
        "term": list(names), "coef": coefs, "se": ses, "z": z, "p": p,
        "lo95": coefs - 1.96 * ses, "hi95": coefs + 1.96 * ses,
        "odds_ratio": np.exp(coefs),
    })
    print(f"\n=== {title} ===")
    if n is not None:
        print(f"  n = {n}")
    def num(v, w, plus=False):
        """Fixed-width formatting that degrades to scientific notation.

        A blown-up standard error is a signal, not a rendering problem -- but
        printed as %.3f it runs into the next column and silently destroys the
        table's alignment, which is how a collinear design goes unnoticed."""
        if not np.isfinite(v):
            return f"{'--':>{w}}"
        s = f"{v:+.3f}" if plus else f"{v:.3f}"
        if len(s) > w:
            s = f"{v:+.1e}" if plus else f"{v:.1e}"
        return f"{s:>{w}}"

    def compact(v):
        if not np.isfinite(v):
            return "--"
        return f"{v:+.3f}" if abs(v) < 1e4 else f"{v:+.1e}"

    # `odds=False` for models whose coefficients are already on the probability
    # scale (DML). exp() of a probability-scale effect is not an odds ratio and
    # printing one invites exactly the misreading the units note warns against.
    if not odds:
        tab = tab.drop(columns=["odds_ratio"])
    head = f"  {'term':<26}{'coef':>10}{'se':>10}{'95% CI':>24}"
    print(head + (f"{'OR':>9}  " if odds else "  "))
    for _, r in tab.iterrows():
        ci = f"[{compact(r['lo95'])}, {compact(r['hi95'])}]"
        line = (f"  {r['term']:<26}{num(r['coef'], 10, True)}"
                f"{num(r['se'], 10)}{ci:>24}")
        if odds:
            orv = r["odds_ratio"]
            line += (f"{orv:>9.2f}" if np.isfinite(orv) and abs(orv) < 1e5
                     else f"{orv:>9.1e}")
        print(f"{line} {stars(r['p'])}")
    if note:
        print(f"\n  {note}")
    return tab


def save(tab, name):
    p = OUT / name
    tab.to_csv(p, index=False)
    print(f"  -> {p.relative_to(ROOT)}")
    return p


def header(title, subtitle=""):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")
    if subtitle:
        print(subtitle)
