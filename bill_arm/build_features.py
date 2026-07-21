"""
Build the introduction-time structural feature table from data/bills/*.jsonl.

One row per bill. See Section 7 of the project spec for the schema. Every
feature here must be knowable on the bill's introduced_date -- nothing from
download_bills.py is later-dated, so no extra leakage filtering is needed at
this stage.

sponsor_is_committee_chair is joined from data/committee_chairs.csv on
(congress, chamber, primary_committee), comparing the sponsor's last name
(case-insensitive) to the chair's last name. That table only covers the 10
highest-bill-volume (chamber, committee) pairs (~60% of bills, see
data/committee_chairs.csv and CHANGELOG); bills referred to any other
committee -- or matching congress/chamber/committee but not the chair's
surname -- get 0. This is an introduction-time-legal feature: who chaired a
committee on a given date is public record at that date, unlike e.g.
recession_flag.

Macroeconomic climate columns (unemployment_rate, recession_flag,
gdp_growth_yoy, cpi_inflation_yoy, consumer_sentiment, initial_claims) are
joined in from data/macro_daily.csv (see build_macro_features.py), which is
already lag-adjusted to what was publicly known as of each calendar day --
so the join here is a plain lookup by introduced_date, no leakage handling
needed at this stage.

Input:  data/bills/{congress}.jsonl (one or more files)
        data/macro_daily.csv (optional; run build_macro_features.py first)
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
        "sponsor_last_name": rec.get("sponsor_last_name"),
        "sponsor_in_majority": (sponsor_party == majority_party
                                if sponsor_party and majority_party else None),
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


# download_bills_bulk.py's XML parser drops "the" from this committee's name
# for the 118th Congress only ("Education and Workforce Committee" instead of
# "Education and the Workforce Committee"); normalize so the chair-table join
# (which uses the canonical name for every Congress) still matches.
COMMITTEE_NAME_ALIASES = {
    "Education and Workforce Committee": "Education and the Workforce Committee",
}


def add_committee_chair_feature(df, chairs_csv="data/committee_chairs.csv"):
    """sponsor_is_committee_chair: was the bill's sponsor the chair of
    primary_committee at introduction time? Joined from data/committee_chairs.csv
    (10 highest-volume (chamber, committee) pairs only, see that file's
    header comment / CHANGELOG -- everything else defaults to 0, both bills
    referred elsewhere and true mismatches, so this column undercounts."""
    if not Path(chairs_csv).exists():
        df["sponsor_is_committee_chair"] = 0
        return df
    chairs = pd.read_csv(chairs_csv)
    chairs["chair_last_name"] = (chairs["chair_name"]
                                 .str.split(",").str[0].str.strip().str.upper())
    chair_lookup = {
        (row.congress, row.chamber, row.committee): row.chair_last_name
        for row in chairs.itertuples()
    }
    committee_norm = df["primary_committee"].replace(COMMITTEE_NAME_ALIASES)
    sponsor_last = df["sponsor_last_name"].fillna("").str.upper()

    def is_chair(congress, chamber, committee, sponsor_ln):
        chair_ln = chair_lookup.get((congress, chamber, committee))
        return int(chair_ln is not None and sponsor_ln != "" and sponsor_ln == chair_ln)

    df["sponsor_is_committee_chair"] = [
        is_chair(c, ch, comm, sp)
        for c, ch, comm, sp in zip(df["congress"], df["chamber"], committee_norm, sponsor_last)
    ]
    return df


def add_macro_features(df, macro_csv="data/macro_daily.csv"):
    if not Path(macro_csv).exists():
        return df
    macro = pd.read_csv(macro_csv, parse_dates=["date"])
    df = df.assign(_intro_date=pd.to_datetime(df["introduced_date"]))
    df = df.merge(macro, left_on="_intro_date", right_on="date", how="left")
    return df.drop(columns=["date", "_intro_date"])


def build_features(bill_files, majority_table_path="data/majority_by_congress.csv",
                   macro_csv="data/macro_daily.csv",
                   chairs_csv="data/committee_chairs.csv"):
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
    df = pd.DataFrame(rows)
    df = add_committee_chair_feature(df, chairs_csv)
    return add_macro_features(df, macro_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, action="append",
                    help="repeatable; defaults to all data/bills/*.jsonl")
    ap.add_argument("--bills-dir", default="data/bills")
    ap.add_argument("--majority-table", default="data/majority_by_congress.csv")
    ap.add_argument("--macro-csv", default="data/macro_daily.csv",
                    help="from build_macro_features.py; skipped if missing")
    ap.add_argument("--chairs-csv", default="data/committee_chairs.csv",
                    help="from committee-chair research; skipped if missing (all 0s)")
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

    df = build_features(files, args.majority_table, args.macro_csv, args.chairs_csv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} bills -> {args.out}")
    print(f"base rate (became_law): {df['became_law'].mean():.4f}")
    print(f"by congress:\n{df.groupby('congress')['became_law'].agg(['count', 'mean'])}")


if __name__ == "__main__":
    main()
