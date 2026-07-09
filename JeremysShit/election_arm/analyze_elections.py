"""
Score extracted predictions against ground truth and produce the core tables.

Scores national-scope claims out of the box. State-scope claims need a
state-level results file (data/state_results.csv with columns
cycle,state,winner_party); add one and set SCORE_STATES = True.
Good sources: Wikipedia per-election pages, ICPSR study 8611, Dave Leip's atlas.

Usage:
  python analyze.py
"""

import glob
import json

import pandas as pd

SCORE_STATES = False

PARTY_ALIASES = {
    "republican": "Republican", "gop": "Republican",
    "democratic": "Democratic", "democrat": "Democratic",
    "progressive": "Progressive", "bull moose": "Progressive",
}


def load_predictions():
    rows = []
    for path in glob.glob("data/predictions/pred_*_elections_*.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("no_predictions"):
                    continue
                rows.append(r)
    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} claims from {df['cycle'].nunique()} cycles")
    return df


def normalize_winner(name):
    if not isinstance(name, str):
        return None
    return PARTY_ALIASES.get(name.strip().lower(), name.strip().title())


def score(df, truth):
    df = df.copy()
    df["predicted_norm"] = df["predicted_winner"].map(normalize_winner)
    truth = truth.set_index("cycle")

    def is_correct(row):
        if row["scope"] != "national":
            return None
        t = truth.loc[row["cycle"]]
        return row["predicted_norm"] in (t["winner_candidate"], t["winner_party"])

    df["correct"] = df.apply(is_correct, axis=1)
    return df


def main():
    df = load_predictions()
    truth = pd.read_csv("data/ground_truth_elections.csv")
    df = score(df, truth)

    national = df[df["scope"] == "national"].dropna(subset=["correct"])
    print(f"\nscored {len(national)} national claims")

    print("\n--- Accuracy by source type (polls vs editorial vs odds) ---")
    print(national.groupby("source_type")["correct"]
          .agg(["mean", "count"]).sort_values("mean", ascending=False))

    if "source" in national.columns and national["source"].notna().any():
        print("\n--- Accuracy by data source (LOC papers vs NYT) ---")
        print(national.groupby("source")["correct"].agg(["mean", "count"]))

    print("\n--- Accuracy by cycle ---")
    print(national.groupby("cycle")["correct"].agg(["mean", "count"]))

    print("\n--- Top publishers by accuracy (min 10 claims) ---")
    pub = national.groupby("newspaper_title")["correct"].agg(["mean", "count"])
    print(pub[pub["count"] >= 10].sort_values("mean", ascending=False).head(20))

    print("\n--- Hedged vs firm claims ---")
    print(national.groupby("hedged")["correct"].agg(["mean", "count"]))

    df.to_csv("data/scored_claims.csv", index=False)
    print("\nfull scored table -> data/scored_claims.csv")
    print("Next step for the ML part: use scored_claims.csv as training data.")
    print("Features: cycle, source_type, hedged, publisher_state, days_before_election,")
    print("plus TF-IDF of claim_text. Target: correct. Then logistic regression and")
    print("gradient boosting with cycle-based train/test splits (train on early cycles,")
    print("test on held-out later ones) to avoid leakage.")


if __name__ == "__main__":
    main()
