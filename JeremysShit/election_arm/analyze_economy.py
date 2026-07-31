"""
Score economy-arm claims against the NBER recession chronology.

Scoring rule: each claim predicts the economy's state (recession or expansion)
at claim_date + horizon_months. The actual state comes from the NBER monthly
chronology (recession = month after peak through trough month, per NBER
convention). Hit = predicted state matches actual state.

Brier scores: each claim's confidence is mapped from its hedged flag
(firm -> 0.90, hedged -> 0.70; a documented assumption, tune in CONFIDENCE).
Brier = (p - outcome)^2, lower is better, and it punishes confident misses,
which is the overconfidence result.

Which claims get scored is a choice, so it is an argument:

  --set verified  (default)  data/verified/  -- gpt-4o-mini's second pass, the
                             analysis set. Precision 0.409 -> 0.700 on the gold
                             pages. Falls back to raw with a warning if the
                             verification phase has not run yet.
  --set raw                  data/predictions/ -- every candidate the extractor
                             proposed. Run both to report the filter's effect.

Outputs: printed tables + data/scored_economy.csv (or _raw.csv)

Usage:  python analyze_economy.py
        python analyze_economy.py --set raw
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CONFIDENCE = {False: 0.90, True: 0.70}   # firm vs hedged -> P(predicted state)
SCHEMA_VERSION = 2                       # must track extract_gpt.SCHEMA_VERSION

# pred_<source>_economy_<shard>.jsonl, and nothing else in the folder. The bare
# glob pred_*_economy_*.jsonl also matches pred_proquest_economy_1990.export.jsonl
# and ....jsonl.dropped.jsonl, which are a stripped copy of the same records and
# the verifier's rejects -- loading either double-counts claims or re-admits
# exactly what the filter threw out.
PRED_RE = re.compile(r"^pred_(?P<source>[a-z0-9]+)_economy_(?P<shard>[^.]+)\.jsonl$")
YEAR_RE = re.compile(r"^\d{4}$")


def load_epu():
    """Monthly historical EPU (1900-2014) if data/epu_monthly.csv exists, else None.

    Baker-Bloom-Davis newspaper-based policy-uncertainty index; exported from the
    economy arm's tier2_analysis.py. On that arm, EPU-at-claim-time was the #2
    predictor of claim correctness after the claim text itself.
    """
    p = Path("data/epu_monthly.csv")
    if not p.exists():
        return None
    epu = pd.read_csv(p)
    epu["month"] = pd.PeriodIndex(epu["month"], freq="M")
    return epu.set_index("month")["epu"]


def load_recessions():
    rec = pd.read_csv("data/nber_recessions.csv")
    periods = []
    for _, row in rec.iterrows():
        peak = pd.Period(row["peak"], freq="M")
        trough = pd.Period(row["trough"], freq="M")
        # NBER: recession runs from the month AFTER the peak through the trough
        periods.append((peak + 1, trough))
    return periods


def state_at(month, recessions):
    """'recession' or 'expansion' for a pd.Period month."""
    for start, end in recessions:
        if start <= month <= end:
            return "recession"
    return "expansion"


def pred_files(directory):
    """The claim files in `directory` that are units of the current corpus.

    Skips, loudly, the ProQuest per-window files from the scrapped "periods"
    design (pred_proquest_economy_gulf_1990.jsonl and friends). Those cover
    articles the 1900-2010 corpus also covers, at schema v1, so pooling them
    double-counts the overlap and mixes two vocabularies. LOC/NYT window files
    are a different corpus and are still loaded.
    """
    kept, periods = [], []
    for path in sorted(Path(directory).glob("*.jsonl")):
        match = PRED_RE.match(path.name)
        if not match:
            continue
        if (match.group("source") == "proquest"
                and not YEAR_RE.match(match.group("shard"))):
            periods.append(path)
            continue
        kept.append(path)
    if periods:
        print(f"skipping {len(periods)} ProQuest per-window file(s) from the "
              f"scrapped periods design;\n  the 1900-2010 corpus covers those "
              f"articles at schema v{SCHEMA_VERSION}:")
        for path in periods:
            print(f"    {path}")
    return kept


def load_claims(which):
    """Claims from the verified or the raw set. Returns (df, label)."""
    directory = Path("data/verified" if which == "verified" else "data/predictions")
    files = pred_files(directory) if directory.is_dir() else []
    if which == "verified" and not files:
        print(f"*** no claim files in {directory}: the verification phase has "
              f"not run.\n*** Falling back to the RAW extractor output, which "
              f"has precision ~0.41 on\n*** the gold pages. Numbers below are "
              f"unfiltered candidates, not the analysis set.")
        which = "raw"
        directory = Path("data/predictions")
        files = pred_files(directory) if directory.is_dir() else []

    rows, stale = [], 0
    for path in files:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("no_predictions"):
                    continue
                # v1 records use a different label vocabulary the scorer below
                # cannot read (predicted_state_at_horizon does not exist there).
                if r.get("schema_version") != SCHEMA_VERSION:
                    stale += 1
                    continue
                rows.append(r)
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} economy claims ({which}) from {len(files)} file(s) "
          f"in {directory}")
    if stale:
        print(f"  skipped {stale} record(s) at an older schema version "
              f"(run migrate_v2.py if you need them)")
    return df, which


def score(df, recessions):
    df = df.copy()
    df["claim_month"] = pd.PeriodIndex(pd.to_datetime(df["date"], errors="coerce"),
                                       freq="M")
    df["horizon_months"] = (pd.to_numeric(df["horizon_months"], errors="coerce")
                            .fillna(6).clip(1, 24).astype(int))
    df["target_month"] = df["claim_month"] + df["horizon_months"]
    df["actual_state"] = df["target_month"].map(lambda m: state_at(m, recessions))
    df["hit"] = df["predicted_state_at_horizon"] == df["actual_state"]
    df["hedged"] = df["hedged"].fillna(False).astype(bool)
    df["confidence"] = df["hedged"].map(CONFIDENCE)
    df["brier"] = (df["confidence"] - df["hit"].astype(int)) ** 2
    epu = load_epu()
    if epu is not None:
        df["epu"] = df["claim_month"].map(epu)
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="which", choices=["verified", "raw"],
                    default="verified",
                    help="verified = data/verified (default); raw = every "
                         "extractor candidate")
    args = ap.parse_args()

    df, which = load_claims(args.which)
    if df.empty:
        raise SystemExit("No economy claims found. Run the downloaders and "
                         "extractor for the economy arm first.")
    recessions = load_recessions()
    df = score(df, recessions)
    df = df.dropna(subset=["claim_month"])
    print(f"scored {len(df)} claims\n")

    # The corpus is now the whole dataset, so the continuous series leads: claim
    # COUNT per decade is only interpretable alongside how many articles were
    # read, which corpus_progress.py reports.
    decades = df["claim_month"].dt.year // 10 * 10
    if decades.nunique() > 2:
        print("--- By decade (the continuous 1900-2010 series) ---")
        print(df.groupby(decades)[["hit", "brier"]].mean()
              .join(df.groupby(decades).size().rename("count")))

    # Crisis-vs-placebo survives the scrapping of the per-window DATASETS,
    # because window_kind is derived from each article's date, not from which
    # query found it -- and it is now a fair comparison for the first time: the
    # denominator is articles read, not articles a forecast-catcher query
    # returned. But most corpus claims sit outside every window, where
    # window_kind is null and pandas' groupby drops them silently, so say what
    # fraction the table covers before showing it.
    in_window = df["window_kind"].notna().sum()
    print(f"\nnote: {in_window:,} of {len(df):,} claims fall inside a configured "
          f"window.\nThe two tables below cover ONLY those; every other table "
          f"covers all {len(df):,}.")

    print("\n--- Crisis vs placebo (the base-rate control; window subset) ---")
    if in_window:
        print(df.groupby("window_kind")[["hit", "brier"]].agg(["mean", "count"]))
        print("\n--- By window (window subset) ---")
        print(df.groupby("window")[["hit", "brier"]].mean()
              .join(df.groupby("window").size().rename("count")))
    else:
        print("  no claims fall inside a configured window")

    print("\n--- By voice (whose prediction was it) ---")
    print(df.groupby("voice")[["hit", "brier"]].mean()
          .join(df.groupby("voice").size().rename("count"))
          .sort_values("hit", ascending=False))

    print("\n--- Hedged vs firm (overconfidence check via Brier) ---")
    print(df.groupby("hedged")[["hit", "brier"]].agg(["mean", "count"]))

    print("\n--- By data source ---")
    print(df.groupby("source")[["hit", "brier"]].agg(["mean", "count"]))

    if "epu" in df.columns and df["epu"].notna().any():
        print("\n--- Accuracy by policy uncertainty at claim time (EPU terciles) ---")
        d = df.dropna(subset=["epu"]).copy()
        d["epu_tercile"] = pd.qcut(d["epu"], 3, labels=["low", "mid", "high"])
        print(d.groupby("epu_tercile", observed=True)[["hit", "brier"]]
              .agg(["mean", "count"]))

    print("\n--- Optimism at turning points ---")
    crisis = df[df["window_kind"] == "crisis"]
    if len(crisis):
        optimists = crisis[crisis["predicted_state_at_horizon"] == "expansion"]
        print(f"share of crisis-window claims predicting expansion: "
              f"{len(optimists) / len(crisis):.2%}")
        if len(optimists):
            print(f"...and their hit rate: {optimists['hit'].mean():.2%}")

    # Separate files per set: the verified table is the analysis set, and
    # overwriting it with the raw run (or the reverse) would silently swap which
    # one every downstream figure and model.py was fitted on.
    out = Path("data/scored_economy.csv" if which == "verified"
               else "data/scored_economy_raw.csv")
    df.to_csv(out, index=False)
    print(f"\nfull scored table -> {out}")
    if which == "verified":
        print("compare against the unfiltered candidates with: "
              "python analyze_economy.py --set raw")


if __name__ == "__main__":
    main()
