"""
Build the introduction-time structural feature table from data/bills/*.jsonl.

One row per bill. See Section 7 of the project spec for the schema. Every
feature here must be knowable on the bill's introduced_date -- nothing from
download_bills.py is later-dated, so no extra leakage filtering is needed at
this stage, but sponsor_is_committee_chair is left unset (no committee
leadership data source wired up; documented as a known gap, not fabricated).

Input:  data/bills/{congress}.jsonl (one or more files)
Output: data/features.csv

Usage:
  python build_features.py --congress 108 --congress 109 ... --out data/features.csv
  python build_features.py                      # all data/bills/*.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd

CHAMBER_BY_TYPE = {"hr": "House", "hjres": "House", "s": "Senate", "sjres": "Senate"}


def load_majority_table(path="data/majority_by_congress.csv"):
    df = pd.read_csv(path)
    table = {}
    for _, row in df.iterrows():
        table[int(row["congress"])] = {
            "start_year": int(row["start_year"]),
            "end_year": int(row["end_year"]),
            "House": row["house_majority_party"],
            "Senate": row["senate_majority_party"],
        }
    return table


def intro_month_in_session(introduced_date, start_year):
    """1-24: month index within the two-year Congress, Jan of start_year = 1."""
    if not introduced_date or start_year is None:
        return None
    year, month = int(introduced_date[:4]), int(introduced_date[5:7])
    idx = (year - start_year) * 12 + month
    return min(max(idx, 1), 24)


def bipartisan(cosponsors):
    parties = {c.get("party") for c in cosponsors if c.get("party")}
    return len(parties & {"D", "R"}) == 2


def frac_cosponsors_majority(cosponsors, majority_party):
    if not cosponsors or not majority_party:
        return 0.0
    n_majority = sum(1 for c in cosponsors if c.get("party") == majority_party)
    return n_majority / len(cosponsors)


def row_from_record(rec, majority_table):
    congress = rec["congress"]
    bill_type = rec["bill_type"]
    chamber = CHAMBER_BY_TYPE.get(bill_type)
    maj = majority_table.get(congress, {})
    majority_party = maj.get(chamber)
    sponsor_party = rec.get("sponsor_party")
    cosponsors = rec.get("original_cosponsors") or []
    title = rec.get("title") or ""

    return {
        "congress": congress,
        "chamber": chamber,
        "bill_type": bill_type,
        "number": rec.get("number"),
        "sponsor_party": sponsor_party,
        "sponsor_state": rec.get("sponsor_state"),
        "sponsor_in_majority": (sponsor_party == majority_party
                                if sponsor_party and majority_party else None),
        "sponsor_is_committee_chair": None,  # not implemented: needs committee leadership data
        "n_original_cosponsors": len(cosponsors),
        "bipartisan": bipartisan(cosponsors),
        "frac_cosponsors_majority": frac_cosponsors_majority(cosponsors, majority_party),
        "policy_area": rec.get("policy_area"),
        "primary_committee": rec.get("primary_committee"),
        "intro_month_in_session": intro_month_in_session(
            rec.get("introduced_date"), maj.get("start_year")),
        "title_length": len(title.split()),
        "has_companion_bill": bool(rec.get("has_companion_bill")),
        "title_text": title,
        "introduced_text": rec.get("introduced_text") or "",
        "introduced_date": rec.get("introduced_date"),
        "became_law": bool(rec.get("became_law")),
    }


def build_features(bill_files, majority_table_path="data/majority_by_congress.csv"):
    majority_table = load_majority_table(majority_table_path)
    rows = []
    for path in bill_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append(row_from_record(rec, majority_table))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, action="append",
                    help="repeatable; defaults to all data/bills/*.jsonl")
    ap.add_argument("--bills-dir", default="data/bills")
    ap.add_argument("--majority-table", default="data/majority_by_congress.csv")
    ap.add_argument("--out", default="data/features.csv")
    args = ap.parse_args()

    bills_dir = Path(args.bills_dir)
    if args.congress:
        files = [bills_dir / f"{c}.jsonl" for c in args.congress]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"Missing files: {missing}")
    else:
        files = sorted(bills_dir.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No .jsonl files in {bills_dir}. Run download_bills.py first.")

    df = build_features(files, args.majority_table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} bills -> {args.out}")
    print(f"base rate (became_law): {df['became_law'].mean():.4f}")
    print(f"by congress:\n{df.groupby('congress')['became_law'].agg(['count', 'mean'])}")


if __name__ == "__main__":
    main()
