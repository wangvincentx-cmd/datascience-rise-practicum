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

What the graders label (fill each column with EXACTLY one of these):
  economy arm  : which way the claim says conditions will go --
                 improve | worsen | no_change | unclear | not_a_prediction
  elections arm: predicted winner (candidate or party), or not_a_prediction

Read the quote and answer as if the LLM's label did not exist. "not_a_prediction"
is the important one: it is how a false positive gets recorded, and on a weak
extractor that is the error worth measuring. Use "unclear" only when the text IS
a forecast but its direction cannot be read.

NOTE the economy vocabulary changed with schema v2. It used to be
recession/expansion -- the OUTCOME vocabulary. Under v2 that is derived from
`direction` by a fixed table in adapt_proquest_claims.py, so grading it would
have been grading a lookup table rather than the model's actual judgement.
Grade `direction`, which is what the LLM decides.
"""

import argparse
import glob
import json
import random
from pathlib import Path

import pandas as pd

SAMPLE_FRAC = 0.20
SAMPLE_CAP = 200
# The label humans double-code. For the economy arm this is `direction` -- the
# thing the LLM actually decides. (v1 validated `predicted_state_at_horizon`,
# but under schema v2 that is DERIVED from direction by adapt_proquest_claims.py,
# so grading it would be grading a lookup table, not the model.) v1 files that
# still carry the old column are handled in load_claims.
LABEL_COL = {"economy": "direction", "elections": "predicted_winner"}
V1_LABEL_COL = {"economy": "predicted_direction"}
# v1's direction vocabulary differed by one word. Without this, a mixed sample
# would carry both "stable" and "no_change" for the same judgement and the
# graders' single answer could only ever match one of them.
V1_VALUE_MAP = {"economy": {"stable": "no_change"}}
# What a grader is allowed to write. Free text typed by two people is the most
# likely way this measurement quietly breaks: one "Improve " or "worsens" and
# that row counts as a disagreement forever. Checked before kappa is computed.
GRADER_VALUES = {"economy": ["improve", "worsen", "no_change", "unclear",
                             "not_a_prediction"]}
NOT_A_PREDICTION = "not_a_prediction"


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
    validation runs on the in-VM files that still carry the claim text.

    Normalizes the two schema generations onto v2's names, so a mixed
    data/predictions/ still yields one coherent sample: v1's `claim_text`
    becomes `quote`, v1's `predicted_direction` becomes `direction`."""
    rows = []
    pattern = f"data/predictions/pred_{source or '*'}_{arm}_*.jsonl"
    v1_label = V1_LABEL_COL.get(arm)
    for path in glob.glob(pattern):
        if path.endswith(".export.jsonl"):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("no_predictions"):
                    continue
                r.setdefault("quote", r.get("claim_text"))
                if v1_label and not r.get(LABEL_COL[arm]):
                    r[LABEL_COL[arm]] = r.get(v1_label)
                vmap = V1_VALUE_MAP.get(arm, {})
                val = str(r.get(LABEL_COL[arm], "")).lower()
                if val in vmap:
                    r[LABEL_COL[arm]] = vmap[val]
                r["schema"] = "v2" if r.get("schema_version") == 2 else "v1"
                rows.append(r)
    return pd.DataFrame(rows)


def mode_sample(arm, source=None):
    df = load_claims(arm, source)
    if df.empty:
        raise SystemExit(f"No {arm} claims found yet.")

    # A sample spanning both schemas measures two different prompts at once and
    # the resulting kappa describes neither. Say so loudly -- the cost here is
    # two people's grading time, which is the expensive input.
    mix = df["schema"].value_counts().to_dict()
    if len(mix) > 1:
        print(f"WARNING: this sample would span BOTH schemas ({mix}).")
        print("  v1 and v2 came from different prompts, so one kappa over both")
        print("  describes neither. Migrate and re-extract first, or pass")
        print("  --source to isolate one, before spending grading time.\n")
    else:
        print(f"all claims are schema {list(mix)[0]} ({sum(mix.values())} total)")

    n = min(max(int(len(df) * SAMPLE_FRAC), 10), SAMPLE_CAP, len(df))
    random.seed(42)   # fixed seed so the sample is reproducible for the paper
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)
    sample["sample_id"] = range(1, n + 1)

    # Graders see the claim text and context but NOT the LLM's label.
    blind = sample[["sample_id", "quote", "date", "newspaper_title",
                    "window"]].copy()
    blind["grader_A"] = ""
    blind["grader_B"] = ""
    blind.to_csv("data/validation_sample.csv", index=False)

    # LLM labels stored separately, joined at kappa time.
    key = sample[["sample_id", LABEL_COL[arm]]].rename(
        columns={LABEL_COL[arm]: "llm_label"})
    key.to_csv("data/validation_llm_key.csv", index=False)

    print(f"\nwrote data/validation_sample.csv ({n} claims) for double coding.")
    print("Each grader fills their column INDEPENDENTLY. LLM labels are in "
          "data/validation_llm_key.csv - do not open it until both are done.")
    allowed = GRADER_VALUES.get(arm)
    if allowed:
        print(f"\nFill grader_A / grader_B with EXACTLY one of:\n  "
              + " | ".join(allowed))
        print(f"  '{NOT_A_PREDICTION}' = the quote is not a forecast at all "
              f"(this is how a false positive gets recorded).")
        print("  'unclear' = it IS a forecast but the direction cannot be read.")
    print("\nNeither CSV may leave the VM -- both contain article text.")


def mode_kappa(arm):
    sample = pd.read_csv("data/validation_sample.csv")
    key = pd.read_csv("data/validation_llm_key.csv")
    df = sample.merge(key, on="sample_id")
    for col in ("grader_A", "grader_B"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise SystemExit(f"{col} has blank rows. Both graders must finish first.")
    for col in ("grader_A", "grader_B", "llm_label"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Catch typos BEFORE reporting a number. "worsens" or "no change" reads as a
    # disagreement on every row it appears in, which would silently drag kappa
    # down and get published as a finding about the model.
    allowed = GRADER_VALUES.get(arm)
    if allowed:
        for col in ("grader_A", "grader_B"):
            bad = sorted(set(df[col]) - set(allowed))
            if bad:
                counts = df[col].value_counts()
                raise SystemExit(
                    f"\n*** {col} has values outside the allowed set: "
                    f"{', '.join(repr(b) for b in bad)}\n"
                    f"*** Rows affected: "
                    f"{sum(counts.get(b, 0) for b in bad)} of {len(df)}\n"
                    f"*** Allowed: {' | '.join(allowed)}\n"
                    f"*** Fix them in data/validation_sample.csv and re-run; "
                    f"otherwise every one of those rows counts as a\n"
                    f"*** disagreement and the kappa you report is wrong.\n")

    ab = cohen_kappa(df["grader_A"], df["grader_B"])
    a_llm = cohen_kappa(df["grader_A"], df["llm_label"])
    b_llm = cohen_kappa(df["grader_B"], df["llm_label"])
    print(f"n = {len(df)} double-coded claims ({arm} arm)")
    print(f"Cohen's kappa, grader A vs grader B : {ab:.3f}")
    print(f"Cohen's kappa, grader A vs LLM      : {a_llm:.3f}")
    print(f"Cohen's kappa, grader B vs LLM      : {b_llm:.3f}")
    print("Guide: >0.6 substantial, >0.8 near-perfect. Report all three.")

    # The extractor's false-positive rate, straight from the graders: how often
    # a "claim" is not a forecast at all. SCORING.md wants this number, and
    # kappa alone hides it -- two graders can agree perfectly that the model
    # keeps inventing forecasts.
    if allowed and NOT_A_PREDICTION in allowed:
        for col in ("grader_A", "grader_B"):
            fp = (df[col] == NOT_A_PREDICTION).sum()
            print(f"  {col}: {fp}/{len(df)} ({fp / len(df):.1%}) marked "
                  f"'{NOT_A_PREDICTION}' -- extractor false positives")
        both = ((df["grader_A"] == NOT_A_PREDICTION)
                & (df["grader_B"] == NOT_A_PREDICTION)).sum()
        print(f"  both graders agreed on {both} of those "
              f"({both / len(df):.1%} of the sample)")

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
