"""
Human validation sample for the Narrative Economics LLM pass
(narratives.py --batch -> claims_narratives_llm_full.csv).

Same "kappa must be MEASURED, not manufactured" discipline as everywhere else
in this project (see handgrade_newspapers/kappa.py, election_arm/
validate_kappa.py): draw a blind sample, hide the LLM's label in a separate
file, have a human fill in human_narrative INDEPENDENTLY without looking at
it, THEN compute kappa. Never edit the human's answer to match the LLM, and
never fabricate this step -- an agent/script cannot fill human_narrative
itself without making the kappa measure the LLM against itself.

Usage (from JeremysShit/):
    python narratives_kappa_sample.py sample     # draws sample, writes blind template
    # -- a human fills in narratives_validation_sample.csv's human_narrative column --
    python narratives_kappa_sample.py kappa      # computes kappa once it's filled in
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grade_claims import cohens_kappa  # noqa: E402

SAMPLE_MIN, SAMPLE_MAX = 40, 80
IN_PATH = "claims_narratives_llm_full.csv"
SAMPLE_PATH = "narratives_validation_sample.csv"
KEY_PATH = "narratives_validation_llm_key.csv"

NARRATIVE_LABELS = ("new_era", "sound_fundamentals", "temporary_setback",
                    "panic_fear", "recovery_normalcy", "none")


def mode_sample():
    df = pd.read_csv(IN_PATH)
    n = min(SAMPLE_MAX, max(SAMPLE_MIN, len(df)))
    sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    blind = sample[["claim_id", "date", "episode", "quote"]].copy()
    blind["human_narrative"] = ""
    blind.to_csv(SAMPLE_PATH, index=False)

    sample[["claim_id", "narrative_llm"]].to_csv(KEY_PATH, index=False)

    print(f"wrote {SAMPLE_PATH} ({len(sample)} claims) for blind human coding.")
    print(f"Fill human_narrative with exactly one of: {', '.join(NARRATIVE_LABELS)}")
    print(f"Do NOT open {KEY_PATH} until coding is done.")


def mode_kappa():
    sample = pd.read_csv(SAMPLE_PATH)
    key = pd.read_csv(KEY_PATH)
    df = sample.merge(key, on="claim_id")
    blank = df["human_narrative"].isna() | (df["human_narrative"].astype(str).str.strip() == "")
    if blank.any():
        raise SystemExit(f"{blank.sum()} blank human_narrative row(s) -- finish coding first.")

    pairs = list(zip(df["human_narrative"].astype(str).str.strip().str.lower(),
                     df["narrative_llm"].astype(str).str.strip().str.lower()))
    kappa = cohens_kappa(pairs)
    agree = sum(1 for a, b in pairs if a == b) / len(pairs)
    print(f"human vs LLM narrative label: kappa = {kappa:+.2f}   "
         f"raw agreement {agree:.0%}  (n={len(pairs)})")
    print("Benchmark: >=0.6 substantial agreement, >=0.8 near-perfect.")

    disagree = df[df["human_narrative"].str.strip().str.lower()
                 != df["narrative_llm"].str.strip().str.lower()]
    if len(disagree):
        out = Path("narratives_validation_disagreements.csv")
        disagree.to_csv(out, index=False)
        print(f"{len(disagree)} disagreements -> {out} (read these; they show how the LLM errs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sample", "kappa"])
    args = ap.parse_args()
    (mode_sample if args.mode == "sample" else mode_kappa)()
