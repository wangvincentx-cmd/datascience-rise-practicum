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

Outputs: printed tables + data/scored_economy.csv

Usage:  python analyze_economy.py
"""

import glob
import json
from pathlib import Path

import pandas as pd

CONFIDENCE = {False: 0.90, True: 0.70}   # firm vs hedged -> P(predicted state)


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


def load_claims():
    rows = []
    for path in glob.glob("data/predictions/pred_*_economy_*.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("no_predictions"):
                    continue
                rows.append(r)
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} economy claims from "
          f"{df['window'].nunique() if len(df) else 0} windows")
    return df


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
    df = load_claims()
    if df.empty:
        raise SystemExit("No economy claims found. Run the downloaders and "
                         "extractor for the economy arm first.")
    recessions = load_recessions()
    df = score(df, recessions)
    df = df.dropna(subset=["claim_month"])
    print(f"scored {len(df)} claims\n")

    print("--- Crisis vs placebo (the base-rate control) ---")
    print(df.groupby("window_kind")[["hit", "brier"]].agg(["mean", "count"]))

    print("\n--- By window ---")
    print(df.groupby("window")[["hit", "brier"]].mean()
          .join(df.groupby("window").size().rename("count")))

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

    df.to_csv("data/scored_economy.csv", index=False)
    print("\nfull scored table -> data/scored_economy.csv")


if __name__ == "__main__":
    main()
