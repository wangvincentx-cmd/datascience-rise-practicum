"""
Merge structural features (build_features.py) with bill-level press features
(extract_press.py) into the modeling dataset for the core experiment
(Section 9). Only run this after coverage_report.py's gate passes.

Input:  data/features.csv
        data/press_features_{congress}.csv, one or more
Output: data/modeling.csv -- every bill from the joined congresses, tagged
        with has_national_coverage etc. Model 2 (in model.py) filters this
        to has_national_coverage == True itself; this script does not drop
        uncovered bills, so the file still shows the true coverage rate.

Usage:
  python join_dataset.py --congress 108 --congress 109 ... --congress 118
  python join_dataset.py                       # all available press_features files
"""

import argparse
from pathlib import Path

import pandas as pd


def load_press_features(congresses=None):
    if congresses:
        paths = [Path(f"data/press_features_{c}.csv") for c in congresses]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"Missing press feature files: {missing}. "
                             f"Run extract_press.py for those congresses first.")
    else:
        paths = sorted(Path(".").glob("data/press_features_*.csv"))
        if not paths:
            raise SystemExit("No data/press_features_*.csv found. Run extract_press.py first.")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def join(features_path, congresses=None):
    features = pd.read_csv(features_path)
    press = load_press_features(congresses)
    joined_congresses = sorted(press["congress"].unique())
    features = features[features["congress"].isin(joined_congresses)]

    # Merge keys must match dtype AND formatting exactly, or pandas silently
    # matches zero rows instead of raising. bill_type just needs a consistent
    # str cast; "number" additionally needs numeric normalization first,
    # since a bare str cast turns a float64 1.0 into "1.0" which won't match
    # an int64 1 cast to "1" -- an easy way for int64-vs-float64 inference
    # quirks between the two CSVs to cause silent data loss.
    for col in ["bill_type"]:
        features[col] = features[col].astype(str)
        press[col] = press[col].astype(str)
    features["number"] = pd.to_numeric(features["number"]).astype(int).astype(str)
    press["number"] = pd.to_numeric(press["number"]).astype(int).astype(str)

    merged = features.merge(press, on=["congress", "bill_type", "number"], how="inner",
                            validate="one_to_one")
    return merged, joined_congresses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, action="append",
                    help="repeatable; defaults to all data/press_features_*.csv found")
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out", default="data/modeling.csv")
    args = ap.parse_args()

    merged, joined_congresses = join(args.features, args.congress)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    n_covered = int(merged["has_national_coverage"].sum())
    print(f"joined congresses: {joined_congresses}")
    print(f"{len(merged)} bills -> {args.out}")
    print(f"of which has_national_coverage: {n_covered} ({n_covered / len(merged):.4%})")


if __name__ == "__main__":
    main()
