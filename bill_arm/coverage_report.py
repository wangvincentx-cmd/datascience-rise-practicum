"""
Section 8.1 decision gate. Print the coverage rate before spending any more
time on the press experiment: how many bills got national coverage, and how
many of those carry a non-neutral press prediction (pass or fail). If the
covered-with-prediction subset is under a few hundred bills, the press
experiment is underpowered -- widen the newspaper source or reframe press as
a topic-salience proxy instead of a prediction signal. Do not skip this gate;
join_dataset.py and Model 2 assume it passed.

Input:  data/press_features_{congress}.csv, one or more (from extract_press.py)
Usage:
  python coverage_report.py --congress 108 --congress 109 ... --congress 118
  python coverage_report.py                       # all data/press_features_*.csv
"""

import argparse
from pathlib import Path

import pandas as pd

UNDERPOWERED_THRESHOLD = 300


def load_all(congresses):
    if congresses:
        paths = [Path(f"data/press_features_{c}.csv") for c in congresses]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Missing files: {missing}")
    else:
        paths = sorted(Path(".").glob("data/press_features_*.csv"))
        if not paths:
            raise SystemExit("No data/press_features_*.csv found. Run extract_press.py first.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def compute_counts(df):
    total = len(df)
    covered = df[df["has_national_coverage"] == True]  # noqa: E712
    n_covered = len(covered)
    non_neutral = covered[covered["press_predicts_pass"].isin(["pass", "fail"])]
    n_non_neutral = len(non_neutral)
    return {
        "total": total, "n_covered": n_covered, "n_non_neutral": n_non_neutral,
        "gate_pass": n_non_neutral >= UNDERPOWERED_THRESHOLD,
    }


def report(df):
    total = len(df)
    covered = df[df["has_national_coverage"] == True]  # noqa: E712
    n_covered = len(covered)
    non_neutral = covered[covered["press_predicts_pass"].isin(["pass", "fail"])]
    n_non_neutral = len(non_neutral)

    print(f"total bills:                         {total}")
    print(f"bills with national coverage:        {n_covered} "
         f"({n_covered / total:.4%} of all bills)" if total else "n/a")
    print(f"  of which pass/fail (non-neutral):  {n_non_neutral} "
         f"({n_non_neutral / n_covered:.2%} of covered)" if n_covered else "  n/a")
    print(f"  of which neutral tone:             "
         f"{len(covered[covered['press_predicts_pass'] == 'neutral'])}")
    print(f"  of which mixed signal:             "
         f"{len(covered[covered['press_predicts_pass'] == 'mixed'])}")

    if n_covered:
        print("\npress_confidence among non-neutral coverage:")
        print(non_neutral["press_confidence"].value_counts(dropna=False).to_string())

    print()
    if n_non_neutral < UNDERPOWERED_THRESHOLD:
        print(f"GATE: FAIL. Only {n_non_neutral} bills have national coverage with a "
             f"non-neutral press prediction (< {UNDERPOWERED_THRESHOLD}). The press "
             f"experiment (Section 9, research questions 2-3) is underpowered as "
             f"specified. Options: widen the newspaper source beyond NYT, relax the "
             f"prediction window, or reframe press coverage as a topic-salience "
             f"proxy (has_national_coverage as a feature) rather than a directional "
             f"prediction signal. Bring this to the team before proceeding to "
             f"join_dataset.py / Model 2.")
    else:
        print(f"GATE: PASS. {n_non_neutral} bills have national coverage with a "
             f"non-neutral press prediction (>= {UNDERPOWERED_THRESHOLD}). Proceed "
             f"to join_dataset.py and the Model 1 vs Model 2 comparison.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, action="append")
    args = ap.parse_args()
    df = load_all(args.congress)
    report(df)


if __name__ == "__main__":
    main()
