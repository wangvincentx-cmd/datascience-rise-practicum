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

Add --source to validate one source in isolation, e.g. the GPT extraction:
  python validate_kappa.py sample --arm economy --source proquest

PROQUEST runs INSIDE the TDM Studio VM. The exported pred_*.export.jsonl has
claim_text stripped, so graders can only read the claims on the un-stripped
in-VM files (this script skips the .export.jsonl copies automatically). Run all
three steps in the VM; only the printed kappa NUMBERS leave (do not export
validation_sample.csv or validation_disagreements.csv -- they contain claim_text).

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

SAMPLE_FRAC = 0.20
SAMPLE_CAP = 200
LABEL_COL = {"economy": "predicted_state_at_horizon", "elections": "predicted_winner"}


def cohen_kappa(a, b):
    """Cohen's kappa for two label sequences (no sklearn dependency, so this
    runs in the locked TDM Studio VM with just pandas)."""
    from collections import Counter
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[lbl] / n) * (cb[lbl] / n) for lbl in set(a) | set(b))
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def load_claims(arm, source=None):
    """Load extracted claims for an arm, optionally only one source (e.g.
    'proquest'). Skips *.export.jsonl (the text-stripped export copies) so
    validation runs on the in-VM files that still carry claim_text."""
    rows = []
    pattern = f"data/predictions/pred_{source or '*'}_{arm}_*.jsonl"
    for path in glob.glob(pattern):
        if path.endswith(".export.jsonl"):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("no_predictions"):
                    rows.append(r)
    return pd.DataFrame(rows)


def mode_sample(arm, source=None):
    df = load_claims(arm, source)
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

    ab = cohen_kappa(df["grader_A"], df["grader_B"])
    a_llm = cohen_kappa(df["grader_A"], df["llm_label"])
    b_llm = cohen_kappa(df["grader_B"], df["llm_label"])
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
    ap.add_argument("--source", choices=["loc", "nyt", "proquest"],
                    help="validate only one source (e.g. proquest); default all")
    args = ap.parse_args()
    if args.mode == "sample":
        mode_sample(args.arm, args.source)
    else:
        mode_kappa(args.arm)


if __name__ == "__main__":
    main()
