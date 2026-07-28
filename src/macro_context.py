"""
The economy each forecast was made INTO -- attached to every scored claim, and
attributed factor by factor.

Scoring itself is unchanged and stays deterministic (`truth_data.py` decides what
happened, no LLM, no macro data anywhere near it). This module runs AFTER scoring
and adds a second column block: what the economy was visibly doing at the moment
each claim went to print. That turns "the press was right 51% of the time" into a
question with a shape -- right when output was accelerating? when prices were
running? when the market had already broken? -- and produces the per-factor
answer that `model_hit.py` currently compresses into a single macro-only AUC.

WHY THIS IS NOT ALREADY ANSWERED. model_hit.py reports macro-only AUC 0.505 on
the monthly corpus and calls it a genuine null. That is one number over eleven
features pooled: it cannot say WHICH factor was flat, and it cannot see a factor
that helps one kind of forecast and hurts another (an "improve" call and a
"worsen" call made in the same falling economy have opposite fates, and pooled
they cancel exactly). Both are reported here.

LEAKAGE DISCIPLINE, non-negotiable and the reason for most of this file:
  * every series is shifted forward by a publication lag before it is read, so a
    claim printed in month M only sees data that would actually have been public
    by M (LAGS below, per series -- not one uniform number)
  * stock prices take NO lag: they were on the same day's ticker
  * NBER recession status is NEVER a real-time factor. It is dated 6-21 months
    after the fact. It is computed here ONLY as `hindsight_*`, is excluded from
    FACTORS, and must never enter a model. It exists so figures can contrast
    "what was knowable" against "what we now know".
  * significance uses a BLOCK bootstrap over 3-year periods, never i.i.d.
    resampling of claims: forecasts inside one period share one macro reality, so
    treating them as independent would shrink every interval by ~3x.

Outputs:
    macro_context.csv   scored claims + one column per factor (+ coverage flags)
    printed attribution: per-factor n, correlation with `hit` + block-bootstrap
                         CI, tercile hit rates, and Benjamini-Hochberg q-values
                         across the whole factor family
    stratified attribution by topic and by forecast direction

Usage:
    python src/macro_context.py                          # monthly corpus
    python src/macro_context.py --scored data/scored/claims_v2_scored.csv \
        --out data/scored/crisis_context.csv
    python src/macro_context.py --lag-scale 2.0          # slower-news sensitivity
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from truth_data import STOCK_SERIES, load_fred, recession_months

warnings.filterwarnings("ignore")

# Months of publication lag per series: a statistic describing month M was not
# public until roughly M+lag, so a forecast printed in M may only see through
# M-lag. These are deliberately conservative for the modern era and, if anything,
# still generous for 1900-1963, when releases were slower -- so `--lag-scale 2`
# is the honest sensitivity check, not a fudge. Stocks: 0, they were in the same
# paper. EPU: 0, it is built FROM contemporaneous newspapers.
#
# The lag is applied with Series.shift(), which moves by POSITION -- so it equals
# "N months" only while the monthly index is gap-free. It is, through 1970, for
# every series used here, and test_offline.py asserts it. Extending the corpus
# past 1970 must re-check: CPIAUCNS and UNRATE are each missing 2025-10 in the
# current FRED vintage, and a closed gap silently overstates the lag.
LAGS = {"INDPRO": 2, "CPIAUCNS": 2, "UNRATE": 2, "STOCK": 0, "EPU": 0}

# 3-year blocks: the same grouping model_hit.py cross-validates over, and coarser
# than the autocorrelation length of every series here.
BLOCK_YEARS = 3


# --- factor construction ---------------------------------------------------
def _pct_change(series, p, months):
    """Percent change of `series` over the `months` ending at period `p`."""
    p0 = p - months
    if p0 < series.index.min() or p > series.index.max():
        return np.nan
    v0, v1 = series.asof(p0), series.asof(p)
    if pd.isna(v0) or pd.isna(v1) or v0 == 0:
        return np.nan
    return 100.0 * (v1 - v0) / v0


def _window(series, p, months):
    """The `months` of `series` ending at `p`, or an empty Series."""
    p0 = p - months
    if p0 < series.index.min() or p > series.index.max():
        return pd.Series(dtype=float)
    return series[(series.index >= p0) & (series.index <= p)]


def build_context(dates, lag_scale=1.0):
    """One row of publication-lagged economic state per claim date.

    Everything here is what a reader of that month's paper could have known.
    Missing (before a series exists) stays NaN -- never imputed, because "we do
    not know what industrial production was doing in 1907" is the truth, and a
    zero would silently become "it was doing nothing"."""
    L = {k: int(round(v * lag_scale)) for k, v in LAGS.items()}
    indpro = load_fred("INDPRO").shift(L["INDPRO"])
    cpi = load_fred("CPIAUCNS").shift(L["CPIAUCNS"])
    unrate = load_fred("UNRATE").shift(L["UNRATE"])
    stocks = load_fred(STOCK_SERIES).shift(L["STOCK"])
    stk_ret = stocks.pct_change()
    epu = _epu_or_none()
    rec = recession_months()

    rows = []
    for d in dates:
        p = pd.Timestamp(d).to_period("M")
        r = {}

        # --- real activity ---
        ip6, ip12 = _pct_change(indpro, p, 6), _pct_change(indpro, p, 12)
        r["ip_growth_6m"] = ip6
        r["ip_growth_12m"] = ip12
        # acceleration: is the expansion picking up or rolling over? The level of
        # growth and its second derivative are different signals and a forecaster
        # arguably reads the second one.
        r["ip_accel"] = ip6 - ip12 if not (pd.isna(ip6) or pd.isna(ip12)) else np.nan
        # output-gap proxy: output relative to its own trailing 3-year average.
        # Not a real potential-output estimate (none exists for 1919), but it
        # separates "growing from a hole" from "growing at a peak".
        w = _window(indpro, p, 36)
        v = indpro.asof(p) if indpro.index.min() <= p <= indpro.index.max() else np.nan
        r["ip_gap"] = (100.0 * (v - w.mean()) / w.mean()
                       if len(w) >= 12 and not pd.isna(v) and w.mean() else np.nan)

        # --- prices ---
        cp12, cp6 = _pct_change(cpi, p, 12), _pct_change(cpi, p, 6)
        r["cpi_yoy"] = cp12
        r["cpi_accel"] = cp6 * 2 - cp12 if not (pd.isna(cp6) or pd.isna(cp12)) else np.nan

        # --- labour (1948+ only; most of the corpus predates it) ---
        ur = unrate.asof(p) if unrate.index.min() <= p <= unrate.index.max() else np.nan
        r["unrate"] = ur
        r["unrate_d6"] = (ur - unrate.asof(p - 6)
                          if not pd.isna(ur) and (p - 6) >= unrate.index.min() else np.nan)

        # --- markets: the one thing a 1929 forecaster was certainly watching ---
        r["stock_ret6"] = _pct_change(stocks, p, 6)
        r["stock_ret12"] = _pct_change(stocks, p, 12)
        rw = _window(stk_ret, p, 6)
        r["stock_vol6"] = float(rw.std() * 100) if len(rw) >= 3 else np.nan
        pw = _window(stocks, p, 24)
        r["stock_drawdown"] = (100.0 * (pw.iloc[-1] - pw.max()) / pw.max()
                               if len(pw) >= 6 and pw.max() else np.nan)

        # --- policy uncertainty (newspaper-derived; see caveat in attribute()) ---
        r["epu"] = float(epu.asof(p)) if epu is not None and p >= epu.index.min() \
            and p <= epu.index.max() else np.nan

        # --- HINDSIGHT ONLY. Not knowable at print time; never a model feature. ---
        r["hindsight_in_recession"] = int(p in rec)

        rows.append(r)
    return pd.DataFrame(rows)


EPU_CSV = Path("cache") / "epu_monthly.csv"


def _epu_or_none():
    """Historical policy-uncertainty index, or None if it cannot be loaded.

    Reads a plain CSV cache if present, and only falls back to the third-party
    .xlsx otherwise -- converting it to CSV on the way so the next caller never
    needs the spreadsheet again.

    WHY THE CSV CACHE. The .xlsx path needs `openpyxl`, an optional pandas
    dependency that is easy to have in one interpreter and not another: this
    notebook's Jupyter kernel lacked it while the CLI python had it, and EPU
    silently vanished from a model that depends on it more than any other
    feature. A CSV needs nothing beyond pandas, and it is the same trick
    `truth_data.load_fred` uses (RISE_FRED_DIR) to stay runnable inside the
    walled ProQuest environment."""
    if EPU_CSV.exists():
        s = pd.read_csv(EPU_CSV)
        return pd.Series(s["epu"].values,
                         index=pd.PeriodIndex(s["month"], freq="M"),
                         name="epu").sort_index()
    try:
        from tier2_analysis import epu_series
        s = epu_series()
    except Exception as e:
        # Print the MESSAGE, not just the exception class. A bare "[epu
        # unavailable: ImportError]" once cost an hour of forensics -- the
        # message would have named the missing piece immediately.
        print(f"  [epu unavailable: {type(e).__name__}: {e}]")
        print("   other factors unaffected, but any MODEL trained with epu "
              "will be wrong on these rows -- see hit_predictor.predict_new")
        return None
    try:
        EPU_CSV.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame({"month": s.index.astype(str), "epu": s.values}).to_csv(
            EPU_CSV, index=False)
        print(f"  [cached EPU to {EPU_CSV} -- openpyxl no longer needed]")
    except Exception:
        pass  # caching is a convenience; never fail the run over it
    return s


FACTORS = ["ip_growth_6m", "ip_growth_12m", "ip_accel", "ip_gap",
           "cpi_yoy", "cpi_accel", "unrate", "unrate_d6",
           "stock_ret6", "stock_ret12", "stock_vol6", "stock_drawdown", "epu"]

PRETTY = {
    "ip_growth_6m": "industrial output, 6m growth",
    "ip_growth_12m": "industrial output, 12m growth",
    "ip_accel": "output acceleration (6m − 12m)",
    "ip_gap": "output vs 3-year average",
    "cpi_yoy": "inflation, 12m",
    "cpi_accel": "inflation acceleration",
    "unrate": "unemployment rate",
    "unrate_d6": "unemployment, 6m change",
    "stock_ret6": "stock market, 6m return",
    "stock_ret12": "stock market, 12m return",
    "stock_vol6": "stock market volatility",
    # Drawdown is <= 0, so a HIGH tercile means the market is AT its peak. Named
    # for what the high end means, so a reader of the figure is not inverted.
    "stock_drawdown": "stock market vs its 2-year peak",
    "epu": "economic policy uncertainty",
}


# --- attribution -----------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval -- correct near 0 and 1 where the normal
    approximation is not, and these tercile cells get small."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # Clip: at k=0 or k=n the bound is analytically 0 or 1, but floating point
    # lands a hair outside, and these are drawn as error bars on a proportion
    # axis -- a whisker below 0% would be a visible nonsense.
    return (float(np.clip(c - h, 0.0, 1.0)), float(np.clip(c + h, 0.0, 1.0)))


def block_boot_corr(x, y, blocks, reps=2000, seed=0):
    """Correlation between a factor and `hit`, with a BLOCK-bootstrap CI.

    Resamples whole 3-year periods, not individual claims. Claims in one period
    share one economy and largely one set of outcomes, so an i.i.d. bootstrap
    would report an interval roughly sqrt(claims-per-block) too narrow and turn
    noise into findings."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    obs = np.corrcoef(x, y)[0, 1]
    out = []
    idx_by_block = {b: np.where(blocks == b)[0] for b in uniq}
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_block[b] for b in pick])
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            continue
        out.append(np.corrcoef(xb, yb)[0, 1])
    if not out:
        return obs, np.nan, np.nan, np.nan
    lo, hi = np.percentile(out, [2.5, 97.5])
    # Two-sided bootstrap p: how far the null (r = 0) sits inside the
    # resampling distribution, centred on the observed estimate.
    centered = np.array(out) - obs
    p = float((np.abs(centered) >= abs(obs)).mean())
    return obs, lo, hi, p


def bh_qvalues(pvals):
    """Benjamini-Hochberg. Thirteen factors x several strata is a lot of tests;
    without this, one 'significant' factor is the expected result of noise."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    idx = np.argsort(p[ok])
    n = ok.sum()
    ranked = p[ok][idx]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(n)
    tmp[idx] = np.clip(adj, 0, 1)
    q[ok] = tmp
    return q


def attribute(df, ctx, label="ALL CLAIMS", min_n=200, reps=2000):
    """Per-factor table: coverage, correlation with `hit`, terciles, BH q."""
    y = df["hit"].astype(int).values
    blocks = ((pd.to_datetime(df["date"]).dt.year // BLOCK_YEARS)
              * BLOCK_YEARS).values

    rows = []
    for f in FACTORS:
        v = ctx[f].values
        m = ~pd.isna(v)
        if m.sum() < min_n or len(np.unique(blocks[m])) < 4:
            rows.append({"factor": f, "n": int(m.sum()), "r": np.nan,
                         "lo": np.nan, "hi": np.nan, "p": np.nan,
                         "note": "too little coverage"})
            continue
        r, lo, hi, p = block_boot_corr(v[m].astype(float), y[m], blocks[m],
                                       reps=reps)
        row = {"factor": f, "n": int(m.sum()), "r": r, "lo": lo, "hi": hi,
               "p": p, "note": ""}
        # terciles of the factor, hit rate in each
        try:
            t = pd.qcut(v[m].astype(float), 3, labels=["low", "mid", "high"],
                        duplicates="drop")
            for name, sub in pd.Series(y[m]).groupby(t, observed=True):
                k, n = int(sub.sum()), int(sub.size)
                clo, chi = wilson(k, n)
                row[f"{name}_hit"] = k / n
                row[f"{name}_n"] = n
                row[f"{name}_lo"], row[f"{name}_hi"] = clo, chi
        except (ValueError, IndexError):
            pass
        rows.append(row)

    out = pd.DataFrame(rows)
    out["q"] = bh_qvalues(out["p"].values)

    print(f"\n=== {label}  (n={len(df)}, hit rate {y.mean():.3f}) ===")
    print(f"  {'factor':<34}{'n':>6}  {'corr with hit':>22}   "
          f"{'low':>12} {'mid':>12} {'high':>12}   {'q':>6}")
    for _, r in out.sort_values("r", key=lambda s: s.abs(),
                                ascending=False, na_position="last").iterrows():
        if r["note"]:
            print(f"  {PRETTY[r['factor']]:<34}{r['n']:>6}  {r['note']}")
            continue
        ci = f"{r['r']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]"
        cells = []
        for t in ["low", "mid", "high"]:
            cells.append(f"{r[f'{t}_hit']:.3f}" if f"{t}_hit" in r
                         and not pd.isna(r.get(f"{t}_hit")) else "     .")
        star = "  *" if r["q"] < 0.05 else ""
        print(f"  {PRETTY[r['factor']]:<34}{r['n']:>6}  {ci:>22}   "
              f"{cells[0]:>12} {cells[1]:>12} {cells[2]:>12}   "
              f"{r['q']:>6.3f}{star}")
    return out


def run(args):
    df = pd.read_csv(args.scored, low_memory=False)
    df = df[(df["scorable"] == True) & (df["hit"].isin([0, 1]))].reset_index(drop=True)
    print(f"scored claims: {len(df)}   hit rate {df['hit'].mean():.3f}")

    print("\nbuilding publication-lagged economic context "
          f"(lag scale {args.lag_scale}x) ...")
    ctx = build_context(df["date"], lag_scale=args.lag_scale)
    for f in FACTORS:
        ctx[f"has_{f}"] = ctx[f].notna().astype(int)

    print("\ncoverage (a factor can only speak for the claims it covers):")
    for f in FACTORS:
        n = int(ctx[f].notna().sum())
        print(f"  {PRETTY[f]:<34}{n:>7}  ({n/len(ctx):.0%})")

    out = pd.concat([df.reset_index(drop=True), ctx], axis=1)
    out.to_csv(args.out, index=False)
    print(f"\n-> {args.out}  ({len(out)} rows, {len(FACTORS)} factors)")

    reps = args.reps
    tables = {"all": attribute(df, ctx, "ALL CLAIMS", reps=reps)}

    # The pooled test is the one that can hide the effect. An optimistic and a
    # pessimistic forecast made in the same collapsing economy have OPPOSITE
    # fates, so any factor that genuinely predicts accuracy through the economy
    # cancels when they are averaged together. Split them.
    opt = df["predicted_norm"].isin(["improve", "up"])
    pes = df["predicted_norm"].isin(["worsen", "down"])
    for name, mask in [("OPTIMISTIC forecasts (improve / prices up)", opt),
                       ("PESSIMISTIC forecasts (worsen / prices down)", pes)]:
        if mask.sum() >= 300:
            tables[name] = attribute(df[mask.values].reset_index(drop=True),
                                     ctx[mask.values].reset_index(drop=True),
                                     name, reps=reps)

    if not args.by_topic:
        return tables
    for topic, sub in df.groupby("topic"):
        if len(sub) >= 500:
            tables[topic] = attribute(sub.reset_index(drop=True),
                                      ctx.loc[sub.index].reset_index(drop=True),
                                      f"TOPIC = {topic}", reps=reps)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", default="data/scored/monthly_scored.csv",
                    help="score_predictions.py output")
    ap.add_argument("--out", default="data/scored/macro_context.csv")
    ap.add_argument("--lag-scale", type=float, default=1.0,
                    help="multiply every publication lag (sensitivity test)")
    ap.add_argument("--reps", type=int, default=2000,
                    help="block-bootstrap resamples")
    ap.add_argument("--by-topic", action="store_true",
                    help="also attribute within each topic")
    run(ap.parse_args())
