"""
Human validation of the LLM's grades. Two modes.

MODE 1 - sample: draws a random sample of extracted claims (default 20%,
capped at 200) into data/validation_sample.csv with blank grader columns.
Two team members INDEPENDENTLY fill in their columns without looking at each
other's answers or at the LLM's labels (they are hidden in a separate file).

  python validate_kappa.py sample --arm economy
  python validate_kappa.py sample --arm elections

MODE 2 - kappa: computes Cohen's kappa between grader A and grader B, and
each grader vs the LLM. Report these numbers in the paper. Rule of thumb:
kappa above 0.6 is substantial agreement, above 0.8 near-perfect.

  python validate_kappa.py kappa --arm economy

What the graders label (fill each column with exactly these values):
  economy arm  : grader col = predicted state -> "recession" / "expansion" / "not_a_prediction"
  elections arm: grader col = predicted winner (candidate or party) -> or "not_a_prediction"
"""

import argparse
import glob
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

SAMPLE_FRAC = 0.20
SAMPLE_CAP = 200
LABEL_COL = {"economy": "predicted_state_at_horizon", "elections": "predicted_winner"}


def load_claims(arm):
    rows = []
    for path in glob.glob(f"data/predictions/pred_*_{arm}_*.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("no_predictions"):
                    rows.append(r)
    return pd.DataFrame(rows)


def mode_sample(arm):
    df = load_claims(arm)
    if df.empty:
        raise SystemExit(f"No {arm} claims found yet.")
    n = min(max(int(len(df) * SAMPLE_FRAC), 10), SAMPLE_CAP, len(df))
    random.seed(42)   # fixed seed so the sample is reproducible for the paper
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)
    sample["sample_id"] = range(1, n + 1)

    # Graders see the claim text and context but NOT the LLM's label.
    blind = sample[["sample_id", "claim_text", "date", "newspaper_title",
                    "window"]].copy()
    blind["grader_A"] = ""
    blind["grader_B"] = ""
    blind.to_csv("data/validation_sample.csv", index=False)

    # LLM labels stored separately, joined at kappa time.
    key = sample[["sample_id", LABEL_COL[arm]]].rename(
        columns={LABEL_COL[arm]: "llm_label"})
    key.to_csv("data/validation_llm_key.csv", index=False)

    print(f"wrote data/validation_sample.csv ({n} claims) for double coding.")
    print("Each grader fills their column INDEPENDENTLY. LLM labels are in "
          "data/validation_llm_key.csv - do not open it until both are done.")


def mode_kappa(arm):
    sample = pd.read_csv("data/validation_sample.csv")
    key = pd.read_csv("data/validation_llm_key.csv")
    df = sample.merge(key, on="sample_id")
    for col in ("grader_A", "grader_B"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise SystemExit(f"{col} has blank rows. Both graders must finish first.")
    for col in ("grader_A", "grader_B", "llm_label"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    ab = cohen_kappa_score(df["grader_A"], df["grader_B"])
    a_llm = cohen_kappa_score(df["grader_A"], df["llm_label"])
    b_llm = cohen_kappa_score(df["grader_B"], df["llm_label"])
    print(f"n = {len(df)} double-coded claims ({arm} arm)")
    print(f"Cohen's kappa, grader A vs grader B : {ab:.3f}")
    print(f"Cohen's kappa, grader A vs LLM      : {a_llm:.3f}")
    print(f"Cohen's kappa, grader B vs LLM      : {b_llm:.3f}")
    print("Guide: >0.6 substantial, >0.8 near-perfect. Report all three.")

    disagree = df[df["grader_A"] != df["llm_label"]]
    if len(disagree):
        out = Path("data/validation_disagreements.csv")
        disagree.to_csv(out, index=False)
        print(f"{len(disagree)} grader-A-vs-LLM disagreements -> {out} "
              f"(read these; they tell you HOW the LLM errs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "kappa"])
    ap.add_argument("--arm", choices=["economy", "elections"], required=True)
    args = ap.parse_args()
    (mode_sample if args.mode == "sample" else mode_kappa)(args.arm)


if __name__ == "__main__":
    main()
