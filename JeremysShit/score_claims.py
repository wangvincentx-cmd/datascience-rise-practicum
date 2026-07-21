"""
Score graded newspaper claims against what actually happened, and run the
newspapers-vs-Livingston head-to-head.

Ground truth (all free, cached in cache/):
  - NBER business-cycle chronology (embedded below) — general business direction
  - FRED CPIAUCNS (monthly CPI, 1913- )               — price claims
  - FRED INDPRO   (industrial production, 1919- )     — general business, when available
  - FRED UNRATE   (unemployment rate, 1948- )         — employment claims

Scoring rule per claim: take its date and horizon (6 or 12 months), compute the
realized direction of its topic's series over that window, and compare with the
predicted direction. Correct = hit. Brier score uses confidence as a crude
probability (assertive=0.9, hedged=0.7).

Also computes `composite_score` (project spec Step 4b): the mean of accuracy
(`hit`), punctuality (did the claim commit to a specific timeframe, via
`resolve_horizon`'s basis), and specificity (`resolve_specificity`, rule-based
from the quote text -- no added LLM grading pass). See the comment above
`resolve_specificity` for the exact formula. Only defined where `hit` is (i.e.
scorable predictions).

Coding choices to sensitivity-test (documented for the methods section):
  - "no_change" bands: CPI +/-1.5%, INDPRO +/-2%, UNRATE +/-0.3 pt
  - When only the NBER chronology is available (pre-1919): window ends in
    recession -> realized "worsen"; ends in expansion -> "improve"
  - Price claims before 1913 and employment claims before 1948 are UNSCORED
    (no reliable monthly series), not guessed.

Usage:
    python score_claims.py                  # needs claims_graded.csv
    python score_claims.py --claims claims_raw.csv --heuristic
        (--heuristic scores WITHOUT LLM grades, using crude keyword direction —
         lets the pipeline run end-to-end before an API key exists)

Outputs: claims_scored.csv, results_by_episode.csv, publisher_leaderboard.csv,
figures/*.png, and a printed head-to-head table.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path("cache")
FIGDIR = Path("figures")
# FRED silently hangs on non-browser user agents — a browser-like UA is required.
FRED_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# NBER US business cycle contractions (peak month -> trough month), 1902-1961.
NBER_RECESSIONS = [
    ("1902-09", "1904-08"), ("1907-05", "1908-06"), ("1910-01", "1912-01"),
    ("1913-01", "1914-12"), ("1918-08", "1919-03"), ("1920-01", "1921-07"),
    ("1923-05", "1924-07"), ("1926-10", "1927-11"), ("1929-08", "1933-03"),
    ("1937-05", "1938-06"), ("1945-02", "1945-10"), ("1948-11", "1949-10"),
    ("1953-07", "1954-05"), ("1957-08", "1958-04"), ("1960-04", "1961-02"),
]
RECESSION_MONTHS = set()
for peak, trough in NBER_RECESSIONS:
    for m in pd.period_range(peak, trough, freq="M"):
        RECESSION_MONTHS.add(m)

# No-change thresholds: how big a move counts as the economy actually
# changing direction, vs. normal noise. tier3_robustness.py's band_sensitivity()
# showed the reported hit rate swings 12+ points across plausible choices here
# (see fig_band_sensitivity.png) -- so these need real justification, not a
# round number picked by feel, or the "justification" is just outcome-shopping
# with extra steps.
#   UNRATE: 0.5pt, the Sahm Rule threshold (Sahm 2019; a 0.5-point rise in the
#   3-month avg unemployment rate vs. its 12-month low is the standard,
#   externally-validated recession-onset signal -- FRED publishes it as
#   SAHMREALTIME/SAHMCURRENT). Was 0.3pt with no citation; this is an adapted
#   use (Sahm's rule is specifically about a rise from a 12-month low, this
#   project scores any 12-month move in either direction), so treat this as
#   "anchored to a real standard," not a literal transplant of the rule itself.
#   CPI: 1.17%, INDPRO: 2.33% -- human-calibrated (calibrate_bands.py,
#   calibration_sample.csv, 80 historical windows sampled independent of any
#   claim/outcome, judged 2026-07-16). No external standard like Sahm's Rule
#   exists for these (checked: BLS's published CPI standard error measures
#   measurement precision, not economic significance, and would set an
#   absurdly low ~0.14pt band; no inflation/production regime-shift rule was
#   found in the literature). CAVEAT: this was a single JOINT judgment
#   (Vincent + Jeremy graded together), not independently double-coded, so
#   there's no kappa check the way the grading rubric has -- weaker evidence
#   than that, disclose accordingly if reported. Was 1.5%/2.0% with no
#   citation at all before this.
BANDS = {"CPI": 1.17, "INDPRO": 2.33, "UNRATE": 0.5}
CONF_P = {"assertive": 0.9, "hedged": 0.7}


def fred(series_id):
    """Monthly FRED series as a Period-indexed Series (downloaded once, cached)."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"fred_{series_id}.csv"
    if not f.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        for attempt in range(3):
            try:
                r = requests.get(url, headers=FRED_HEADERS, timeout=60)
                r.raise_for_status()
                f.write_bytes(r.content)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
    df = pd.read_csv(f)
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index(pd.to_datetime(df["date"]).dt.to_period("M"))["value"].dropna()
    return s


def realized_direction(topic, price_dir, unemp_dir, start, months, cpi, indpro, unrate):
    """Return (realized_label, scorable, basis) for one claim window."""
    p0, p1 = start.to_period("M"), (start + pd.DateOffset(months=months)).to_period("M")

    if topic == "prices":
        if p0 in cpi.index and p1 in cpi.index:
            chg = (cpi[p1] / cpi[p0] - 1) * 100
            lab = "up" if chg > BANDS["CPI"] else "down" if chg < -BANDS["CPI"] else "stable"
            return lab, True, "CPI"
        return "", False, "no CPI before 1913"

    if topic == "employment":
        if p0 in unrate.index and p1 in unrate.index:
            chg = unrate[p1] - unrate[p0]
            lab = "up" if chg > BANDS["UNRATE"] else "down" if chg < -BANDS["UNRATE"] else "stable"
            return lab, True, "UNRATE"
        return "", False, "no UNRATE before 1948"

    # general_business / markets / other -> economy direction
    if p0 in indpro.index and p1 in indpro.index:
        chg = (indpro[p1] / indpro[p0] - 1) * 100
        lab = ("improve" if chg > BANDS["INDPRO"]
               else "worsen" if chg < -BANDS["INDPRO"] else "no_change")
        return lab, True, "INDPRO"
    lab = "worsen" if p1 in RECESSION_MONTHS else "improve"
    return lab, True, "NBER"


def predicted_label(row):
    """The claim's prediction in the same label space as the realized direction."""
    topic = row["topic"]
    if topic == "prices":
        return row.get("price_direction", "")
    if topic == "employment":
        d = row.get("unemployment_direction", "")
        if d in ("up", "down", "stable"):
            return d
        return {"improve": "down", "worsen": "up", "no_change": "stable"}.get(row["direction"], "")
    return row["direction"]


HEURISTIC_WORSE = re.compile(r"\b(panic|depression|hard times|collapse|slump|worse|"
                             r"fall|decline|unemploy)", re.I)
HEURISTIC_BETTER = re.compile(r"\b(prosperity|recover\w*|revival|improve\w*|better|"
                              r"bright|confiden\w*|boom)", re.I)

# --- Horizon inference (project spec Step 4) -------------------------------
# The grader labels each claim's horizon 6, 12, or "vague". A "vague" claim
# ("prosperity is right around the corner", "recovery will take years") states
# no number, and scoring it over a blanket 12-month window can wrongly flip a
# hit to a miss. Instead of defaulting, read the TIME LANGUAGE in the quote and
# map it to a horizon. This only changes the OUTCOME WINDOW a vague claim is
# checked over -- it is a documented modeling ASSUMPTION, so it is
# sensitivity-testable via --horizon-scale (e.g. 0.5 / 2.0 to widen/narrow the
# inferred windows and confirm the headline result is not an artifact of it).
# Claims with a numeric horizon from the grader are never touched.
SHORT_HORIZON = re.compile(
    r"\b(soon|shortly|immediat\w+|at once|right away|near future|coming months|"
    r"next few months|before long|within (?:a few )?months|"
    r"this (?:spring|summer|fall|autumn|winter)|by (?:spring|summer|fall|winter))\b",
    re.I)
LONG_HORIZON = re.compile(
    r"\b(long[- ]?run|long[- ]?term|for years|coming years|years to come|"
    r"eventually|ultimately|in (?:the )?time|decade|for some time|"
    r"permanent\w*|lasting)\b", re.I)
HORIZON_SHORT_M, HORIZON_DEFAULT_M, HORIZON_LONG_M = 6, 12, 24


def resolve_horizon(row, scale=1.0):
    """Return (months, basis). Numeric grader horizons are authoritative;
    vague ones are inferred from time-language, else the neutral 12-mo default."""
    h = str(row.get("horizon_months", "")).strip()
    if h in ("6", "12", "24"):
        return int(h), "stated"
    q = str(row.get("quote", ""))
    if LONG_HORIZON.search(q):
        m, why = HORIZON_LONG_M, "inferred_long"
    elif SHORT_HORIZON.search(q):
        m, why = HORIZON_SHORT_M, "inferred_short"
    else:
        m, why = HORIZON_DEFAULT_M, "default_12"
    return max(1, round(m * scale)), why


# --- Composite claim score (project spec Step 4b) --------------------------
# A single 0-1 score combining three dimensions of "how good was this
# prediction," not just whether it happened to be right:
#   accuracy     - `hit` (already scored above: did the direction match reality).
#   punctuality  - did the paper commit to a specific, falsifiable timeframe,
#                  rather than something vague enough to need a default? Reuses
#                  `resolve_horizon`'s `basis` (spec called this "punctuality";
#                  see CHANGELOG) rather than adding a second horizon concept:
#                  "stated" (the grader read an explicit 6/12/24-month horizon
#                  off the sentence) = 1.0; "inferred_short"/"inferred_long"
#                  (no number, but time-language let us infer one) = 0.5;
#                  "default_12" (no time information at all) = 0.0.
#   specificity  - rule-based, computed straight from the quote text and the
#                  grader's speaker_name field -- deliberately NOT a new LLM
#                  grading pass (would mean re-grading the full corpus at real
#                  API cost for a claim that arguably follows from the text
#                  already on hand). Mean of three independent 0/1 signals:
#                    - named_forecaster: attributed to an actual person, not
#                      anonymous ("Roger Babson predicts..." vs "it is expected")
#                    - numeric_magnitude: a number attached to the prediction
#                      (a percent, dollar figure, or count), not just a bare
#                      direction
#                    - concrete_time_reference: a specific calendar point named
#                      (a year or month) -- distinct from punctuality's horizon
#                      DURATION; this is about naming a point, e.g. "by the
#                      fall of 1930" is more specific than "before long."
# Composite = the unweighted mean of the three. Only defined where accuracy is
# (i.e. `hit` is not NaN) -- a composite score without ground truth on the
# accuracy leg isn't really scoring the prediction, just its packaging.
PUNCTUALITY_BY_BASIS = {"stated": 1.0, "inferred_short": 0.5, "inferred_long": 0.5,
                        "default_12": 0.0}
NUMERIC_MAGNITUDE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|per\s?cent)|\$\s?[\d,]+", re.I)
CONCRETE_TIME_REF = re.compile(r"\b(?:19\d\d|20[0-2]\d)\b|\b(?:january|february|march|april|may|"
                               r"june|july|august|september|october|november|december)\b", re.I)


def resolve_specificity(row):
    """Return (score in [0,1], detail dict of the three 0/1 components)."""
    speaker = str(row.get("speaker_name", "")).strip().lower()
    named_forecaster = speaker not in ("", "na", "none", "unclear", "nan")
    quote = str(row.get("quote", ""))
    numeric_magnitude = bool(NUMERIC_MAGNITUDE.search(quote))
    concrete_time_reference = bool(CONCRETE_TIME_REF.search(quote))
    parts = {"named_forecaster": named_forecaster, "numeric_magnitude": numeric_magnitude,
             "concrete_time_reference": concrete_time_reference}
    return sum(parts.values()) / len(parts), parts


def composite_score(hit, horizon_basis, specificity):
    if hit != hit:  # NaN: unscorable, no accuracy leg to anchor the composite
        return np.nan
    return (hit + PUNCTUALITY_BY_BASIS.get(horizon_basis, 0.0) + specificity) / 3


def heuristic_grade(df):
    """Keyword stand-in for LLM grades so the pipeline runs before an API key exists."""
    df = df.copy()
    df["topic"] = "general_business"
    df["price_direction"] = "na"
    df["unemployment_direction"] = "na"
    df["horizon_months"] = "12"
    df["confidence"] = "hedged"
    df["voice"] = "unclear"
    df["speaker_name"] = "na"
    b = df["quote"].str.count(HEURISTIC_BETTER)
    w = df["quote"].str.count(HEURISTIC_WORSE)
    df["direction"] = np.select([b > w, w > b], ["improve", "worsen"], default="unclear")
    df["is_prediction"] = np.where(df["direction"] != "unclear", "yes", "no")
    return df


def score(args):
    df = pd.read_csv(args.claims)
    if args.heuristic:
        print("HEURISTIC MODE: crude keyword grading (replace with grade_claims.py output)")
        df = heuristic_grade(df)
    missing = [c for c in ["is_prediction", "direction", "topic"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.claims} lacks grade columns {missing}. "
                         "Run grade_claims.py first, or pass --heuristic.")

    cpi, indpro, unrate = fred("CPIAUCNS"), fred("INDPRO"), fred("UNRATE")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    preds = df[(df["is_prediction"] == "yes") & df["date"].notna()].copy()
    preds = preds[preds["direction"].isin(["improve", "worsen", "no_change"]) |
                  preds["topic"].isin(["prices", "employment"])]

    out = []
    for _, r in preds.iterrows():
        months, horizon_basis = resolve_horizon(r, args.horizon_scale)
        realized, scorable, basis = realized_direction(
            r["topic"], r.get("price_direction", ""), r.get("unemployment_direction", ""),
            r["date"], months, cpi, indpro, unrate)
        pred = predicted_label(r)
        hit = int(pred == realized) if (scorable and pred) else np.nan
        p = CONF_P.get(str(r.get("confidence", "hedged")), 0.7)
        specificity, specificity_parts = resolve_specificity(r)
        out.append({**r, "months": months, "horizon_basis": horizon_basis,
                    "predicted_label": pred, "realized_label": realized, "basis": basis,
                    "hit": hit, "brier": (p - hit) ** 2 if hit == hit else np.nan,
                    "specificity": round(specificity, 3), **specificity_parts,
                    "composite_score": composite_score(hit, horizon_basis, specificity)})
    scored = pd.DataFrame(out)
    scored.to_csv("claims_scored.csv", index=False)

    s = scored.dropna(subset=["hit"])
    print(f"\n{len(scored)} predictions, {len(s)} scorable "
          f"({len(scored) - len(s)} unscorable: pre-1913 prices / pre-1948 employment)")

    print("\n=== Composite score (accuracy + punctuality + specificity, mean of 3) ===")
    print(f"  mean composite: {s['composite_score'].mean():.3f}  "
          f"(accuracy leg {s['hit'].mean():.3f}, punctuality leg "
          f"{s['horizon_basis'].map(PUNCTUALITY_BY_BASIS).mean():.3f}, "
          f"specificity leg {s['specificity'].mean():.3f})")

    by_ep = s.groupby("episode").agg(
        n=("hit", "size"), hit_rate=("hit", "mean"), brier=("brier", "mean"),
        composite_score=("composite_score", "mean"),
        share_predicting_improve=("predicted_label",
                                  lambda x: (x == "improve").mean())).round(3)
    by_ep.to_csv("results_by_episode.csv")
    print("\n=== Hit rate by episode ===")
    print(by_ep.to_string())

    lb = (s.groupby("publisher").agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                     brier=("brier", "mean"))
          .query(f"n >= {args.min_claims}").sort_values("hit_rate", ascending=False).round(3))
    lb.to_csv("publisher_leaderboard.csv")
    print(f"\n=== Publisher leaderboard (>= {args.min_claims} scored claims) ===")
    print(lb.head(15).to_string() if len(lb) else "  (none reach the threshold yet — scale up the scrape)")

    tier1(s)
    head_to_head(s)
    figures(s, by_ep, lb)


def tier1(s):
    """Calibration, voice, crisis-vs-control, and named-forecaster analyses."""
    print("\n=== Calibration: were confident claims more accurate? ===")
    cal = s.groupby("confidence").agg(n=("hit", "size"), hit_rate=("hit", "mean")).round(3)
    print(cal.to_string())
    if {"assertive", "hedged"} <= set(cal.index) and \
            cal.loc["assertive", "hit_rate"] < cal.loc["hedged", "hit_rate"]:
        print("  -> assertive claims were LESS accurate than hedged ones: overconfidence.")

    print("\n=== Whose predictions to trust: hit rate by voice ===")
    print(s.groupby("voice").agg(n=("hit", "size"), hit_rate=("hit", "mean"))
          .round(3).to_string())

    if "kind" in s.columns:
        print("\n=== Crisis vs. control (placebo) windows ===")
        k = s.groupby("kind").agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                  share_predicting_improve=(
                                      "predicted_label", lambda x: (x == "improve").mean())).round(3)
        print(k.to_string())
        if "control" not in set(s["kind"].dropna()):
            print("  (no control-window claims yet — rerun newspaper_scraper.py; "
                  "cached crisis pages make the rerun fast)")

    if "speaker_name" in s.columns:
        named = s[~s["speaker_name"].fillna("na").astype(str).str.strip().str.lower()
                  .isin(["", "na", "none", "unclear", "nan"])]
        if len(named):
            cols = [c for c in ["speaker_name", "publisher", "state", "date", "episode",
                                "quote", "predicted_label", "realized_label",
                                "confidence", "hit"] if c in named.columns]
            (named.sort_values(["hit", "confidence"], ascending=[False, True])[cols]
             .to_csv("famous_calls.csv", index=False))
            print(f"\nNamed-forecaster claims -> famous_calls.csv ({len(named)} rows, "
                  "best calls first — mine it for the poster's best/worst-calls sidebar)")


def livingston_directional(band_scale=1.0):
    """Livingston 12-month median forecasts as directional calls, 1946-1963.

    band_scale multiplies the no-change bands (tier3_robustness.py's
    sensitivity test); 1.0 reproduces the headline numbers."""
    xl = pd.ExcelFile("medians.xlsx")
    rows = []
    bounds = {"CPI": (-8, 99), "IP": (-18, 40)}
    for v, band in [("CPI", 1.5 * band_scale), ("IP", 2.0 * band_scale),
                    ("UNPR", 0.3 * band_scale)]:
        d = xl.parse(v).sort_values("Date").reset_index(drop=True)
        bp, f12 = d[f"{v}_BP"], d[f"{v}_12M"]
        nxt = bp.shift(-2)
        if v == "UNPR":
            pred_chg, act_chg = f12 - bp, nxt - bp
        else:
            pred_chg = (f12 - bp) / bp * 100
            act_chg = (nxt - bp) / bp * 100
            g6 = (bp.shift(-1) / bp - 1) * 100          # rebase artifact filter,
            lo, hi = bounds[v]                           # same idea as the notebook
            bad = ((g6 < lo) | (g6 > hi)).fillna(False)
            act_chg = act_chg.mask(bad | bad.shift(-1).fillna(False))
        lab = lambda x: np.select([x > band, x < -band], ["improve", "worsen"], "no_change")
        sub = pd.DataFrame({"date": pd.to_datetime(d["Date"]), "variable": v,
                            "pred": lab(pred_chg), "act": lab(act_chg),
                            "valid": pred_chg.notna() & act_chg.notna()})
        if v == "UNPR":  # rising unemployment = economy worsening: flip labels
            flip = {"improve": "worsen", "worsen": "improve", "no_change": "no_change"}
            sub["pred"] = sub["pred"].map(flip); sub["act"] = sub["act"].map(flip)
        rows.append(sub[sub["valid"]])
    liv = pd.concat(rows)
    return liv[(liv["date"] >= "1946-01-01") & (liv["date"] < "1964-01-01")]


def head_to_head(s):
    print("\n=== HEAD-TO-HEAD 1946-1963: newspapers vs. Livingston economists ===")
    news = s[s["date"] >= "1946-01-01"]
    try:
        liv = livingston_directional()
        liv_hit = (liv["pred"] == liv["act"]).mean()
        print(f"  Livingston directional hit rate: {liv_hit:.1%}  (n={len(liv)} forecasts)")
    except Exception as e:
        print(f"  Livingston side unavailable here ({e}) — run in the notebook instead.")
        liv_hit = None
    if len(news):
        print(f"  Newspaper directional hit rate:  {news['hit'].mean():.1%}  (n={len(news)} claims)")
        if len(news) < 10:
            print("  (fewer than 10 scored 1946-63 claims — CI not meaningful yet)")
        elif liv_hit is not None:
            rng = np.random.default_rng(0)
            boots = [rng.choice(news["hit"], len(news)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  Newspaper 95% CI: [{lo:.1%}, {hi:.1%}] — "
                  f"{'overlaps' if lo <= liv_hit <= hi else 'excludes'} the Livingston rate")
    else:
        print("  No scored newspaper claims in 1946-63 yet — scale up episodes 6-7.")


def figures(s, by_ep, lb):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping figures)")
        return
    FIGDIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    by_ep["hit_rate"].plot(kind="bar", ax=ax, color="steelblue", alpha=.85)
    ax.axhline(0.5, color="crimson", ls="--", lw=1, label="coin flip")
    ax.set_ylabel("directional hit rate"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Were newspaper economic predictions right? Hit rate by crisis episode")
    plt.tight_layout(); plt.savefig(FIGDIR / "fig_hit_by_episode.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    share = s.groupby("episode").agg(
        predicted_improve=("predicted_label", lambda x: (x == "improve").mean()),
        actually_improved=("realized_label", lambda x: (x == "improve").mean()))
    share.plot(kind="bar", ax=ax, color=["goldenrod", "seagreen"], alpha=.85)
    ax.set_ylabel("share"); ax.set_ylim(0, 1)
    ax.set_title("The optimism gap: share predicting improvement vs. share that improved")
    plt.tight_layout(); plt.savefig(FIGDIR / "fig_optimism_gap.png", dpi=200); plt.close()

    if len(lb):
        fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(lb))))
        lb["hit_rate"].sort_values().plot(kind="barh", ax=ax, color="steelblue", alpha=.85)
        ax.axvline(0.5, color="crimson", ls="--", lw=1)
        ax.set_xlabel("hit rate"); ax.set_title("Publisher leaderboard (scored claims)")
        plt.tight_layout(); plt.savefig(FIGDIR / "fig_leaderboard.png", dpi=200); plt.close()

    # Tier-1 figures: calibration, voice, crisis vs. control
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in [(axes[0], "confidence", "By stated confidence"),
                           (axes[1], "voice", "By whose voice")]:
        g = s.groupby(col).agg(n=("hit", "size"), hit_rate=("hit", "mean"))
        g["hit_rate"].plot(kind="bar", ax=ax, color="steelblue", alpha=.85, rot=20)
        for i, (n, hr) in enumerate(zip(g["n"], g["hit_rate"])):
            ax.text(i, hr + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.axhline(0.5, color="crimson", ls="--", lw=1)
        ax.set_ylim(0, 1); ax.set_title(title); ax.set_ylabel("hit rate")
    ax = axes[2]
    if "kind" in s.columns and s["kind"].nunique() > 1:
        g = s.groupby("kind").agg(n=("hit", "size"), hit_rate=("hit", "mean"))
        g["hit_rate"].plot(kind="bar", ax=ax, color=["seagreen", "crimson"], alpha=.8, rot=0)
        for i, (n, hr) in enumerate(zip(g["n"], g["hit_rate"])):
            ax.text(i, hr + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylim(0, 1); ax.set_title("Calm (control) vs. crisis windows")
    else:
        ax.axis("off"); ax.set_title("(control windows not scraped yet)")
    fig.suptitle("Who and when to trust: prediction accuracy by confidence, voice, and regime")
    plt.tight_layout(); plt.savefig(FIGDIR / "fig_calibration_voice_control.png", dpi=200)
    plt.close()
    print(f"\nFigures written to {FIGDIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="claims_graded.csv")
    ap.add_argument("--min-claims", type=int, default=10)
    ap.add_argument("--heuristic", action="store_true",
                    help="score ungraded claims with keyword rules (pipeline test only)")
    ap.add_argument("--horizon-scale", type=float, default=1.0,
                    help="multiply INFERRED (vague-claim) horizons for sensitivity "
                         "testing, e.g. 0.5 or 2.0; stated 6/12-mo horizons unaffected")
    args = ap.parse_args()
    score(args)
