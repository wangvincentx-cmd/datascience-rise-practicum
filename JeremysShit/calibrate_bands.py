"""
Human calibration for the CPI/INDPRO no-change bands in score_claims.py.

Why this exists: UNRATE's band is now anchored to the Sahm Rule (a real,
externally-validated threshold). No comparably clean external standard
exists for CPI or industrial production (checked -- BLS publishes a
measurement-error standard error for CPI, but that answers "is this specific
reading reliable," not "is this a real 12-month change in direction," which
is a much bigger threshold; no Sahm-Rule equivalent for inflation/production
regime shifts was found in the literature). The remaining legitimate path is
the same one the project already uses for the grading rubric: human
double-coding, kappa-checked, same as validation_sample.csv.

What this is NOT: a way to tune the band to make the newspaper hit rate look
better. Windows are sampled from the FULL historical CPI/INDPRO series
(1913-2010), not filtered to claim dates or outcomes, specifically so the
calibration can't be influenced by which claims it would flip. Judges should
have no idea which (if any) newspaper claim a window is connected to -- there
isn't one; these are raw historical data windows, nothing else.

The task: for each window, judge whether the reported 12-month change would
count as a REAL, meaningful move in that direction, or whether it's
essentially flat / within normal noise for that series. Same judgment a
reasonably economically-literate reader would make, not a statistical test.

Usage:
    python calibrate_bands.py                          # build calibration_sample.csv
    python calibrate_bands.py --n-per-bin 8             # more windows per magnitude bin
    python calibrate_bands.py --analyze calibration_sample_filled.csv   # after humans judge

Two people should fill in human1_judgment / human2_judgment independently
(values: real_change / no_change / unsure), the same double-coding process
used for validation_sample.csv. Then run --analyze.
"""

import argparse
import csv
import random

import numpy as np
import pandas as pd

from grade_claims import cohens_kappa
from score_claims import fred

SERIES = ["CPIAUCNS", "INDPRO"]
N_BINS = 8
SEED = 0


def twelve_month_changes(series_id):
    s = fred(series_id)
    rows = []
    for p in s.index:
        fwd = p + 12
        if fwd not in s.index:
            continue
        pct = (s[fwd] / s[p] - 1) * 100
        rows.append({"series": series_id, "window_start": str(p),
                     "window_end": str(fwd), "pct_change": round(pct, 2)})
    return pd.DataFrame(rows)


def build_sample(n_per_bin):
    rng = random.Random(SEED)
    all_rows = []
    for series_id in SERIES:
        df = twelve_month_changes(series_id)
        df["abs_change"] = df["pct_change"].abs()
        # Quantile bins on this series' OWN distribution -- CPI and INDPRO
        # have very different natural scales (INDPRO swings much harder),
        # so a fixed cutoff would over- or under-sample one of them.
        df["bin"] = pd.qcut(df["abs_change"], N_BINS, duplicates="drop")
        picked = []
        for _, grp in df.groupby("bin", observed=True):
            idx = list(grp.index)
            rng.shuffle(idx)
            picked.extend(idx[:n_per_bin])
        all_rows.append(df.loc[picked])
    sample = pd.concat(all_rows, ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle order
    sample.insert(0, "window_id", range(1, len(sample) + 1))
    sample = sample.drop(columns=["bin", "abs_change"])
    sample["human1_judgment"] = ""
    sample["human2_judgment"] = ""

    with open("calibration_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample.columns))
        w.writeheader()
        for _, r in sample.iterrows():
            w.writerow(r.to_dict())

    print(f"Wrote calibration_sample.csv ({len(sample)} windows: "
          f"{(sample['series'] == 'CPIAUCNS').sum()} CPI, "
          f"{(sample['series'] == 'INDPRO').sum()} INDPRO)")
    print("\nInstructions for the two judges (fill independently, don't compare notes):")
    print("  For each row: given the series and the 12-month percent change shown,")
    print("  would a reasonably economically-literate reader call this a REAL, ")
    print("  meaningful move in that direction, or essentially flat / normal noise?")
    print('  Write "real_change", "no_change", or "unsure" in your column.')
    print("  Do NOT look up what was happening in the economy at these dates --")
    print("  judge the number on its own, the same way score_claims.py has to.")


def analyze(path):
    df = pd.read_csv(path)
    h1 = df["human1_judgment"].fillna("").astype(str).str.strip()
    h2 = df["human2_judgment"].fillna("").astype(str).str.strip()
    independent = (h2 != "").any()

    if independent:
        filled = df[(h1 != "") & (h2 != "")]
        if filled.empty:
            raise SystemExit(f"No filled rows in {path} -- fill human1_judgment / "
                             "human2_judgment first.")
        print(f"{len(filled)}/{len(df)} rows judged by both humans\n")
        pairs = list(zip(filled["human1_judgment"].str.strip().str.lower(),
                         filled["human2_judgment"].str.strip().str.lower()))
        print(f"Inter-rater kappa (human1 vs human2): {cohens_kappa(pairs):+.2f}  "
              f"(n={len(pairs)})")
        print("Target: kappa >= 0.7, same bar as the grading rubric. Below that, "
              "the band isn't well-defined enough to derive a number from yet -- "
              "discuss disagreements and re-judge before trusting the threshold below.\n")
        consensus = filled[filled["human1_judgment"].str.strip().str.lower() ==
                           filled["human2_judgment"].str.strip().str.lower()]
        consensus_col = "human1_judgment"
    else:
        # Only human1_judgment is filled -- graded together as a joint
        # consensus, not independently. No kappa is computable from a single
        # column (kappa needs two independent judgments to measure agreement
        # against); reporting one would be fabricating a number, so this is
        # disclosed instead of silently skipped or faked.
        filled = df[h1 != ""]
        if filled.empty:
            raise SystemExit(f"No filled rows in {path} -- fill human1_judgment first.")
        print(f"{len(filled)}/{len(df)} rows judged (SINGLE joint-consensus column "
              f"-- human2_judgment is empty, so no inter-rater kappa is computable).")
        print("This is weaker evidence than independent double-coding: there's no "
              "check that the judgment is reproducible by a second, uninfluenced "
              "rater. Treat the threshold below as a working number, not a "
              "kappa-validated one like the grading rubric's.\n")
        consensus = filled
        consensus_col = "human1_judgment"

    for series_id in SERIES:
        s = consensus[consensus["series"] == series_id].copy()
        if s.empty:
            print(f"{series_id}: no agreed-upon rows, skipping")
            continue
        s["abs_change"] = s["pct_change"].abs()
        s["judgment"] = s[consensus_col].str.strip().str.lower()
        s = s.sort_values("abs_change")
        real = s[s["judgment"] == "real_change"]["abs_change"]
        no_ch = s[s["judgment"] == "no_change"]["abs_change"]
        n_label = "agreed rows" if independent else "judged rows"
        print(f"=== {series_id} (n={len(s)} {n_label}) ===")
        if no_ch.empty or real.empty:
            print("  All judgments are the same category -- widen the "
                 "sample range (more --n-per-bin, or check the bins) before "
                 "trusting a threshold from this.")
            continue
        print(f"  largest move judged 'no_change':  {no_ch.max():.2f}%")
        print(f"  smallest move judged 'real_change': {real.min():.2f}%")
        if no_ch.max() < real.min():
            threshold = (no_ch.max() + real.min()) / 2
            print(f"  CLEAN separation -> suggested band: {threshold:.2f}%")
        else:
            overlap = s[(s["abs_change"] >= real.min()) & (s["abs_change"] <= no_ch.max())]
            print(f"  OVERLAP zone ({real.min():.2f}%-{no_ch.max():.2f}%, "
                 f"n={len(overlap)}): judgment isn't monotonic in magnitude here.")
            threshold = overlap["abs_change"].median()
            print(f"  suggested band (median of overlap zone): {threshold:.2f}% "
                 f"-- treat as rough, not precise, given the overlap")
        print(f"  current score_claims.py BANDS value: "
             f"{1.5 if series_id == 'CPIAUCNS' else 2.0}%\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-bin", type=int, default=5,
                    help="windows sampled per magnitude bin per series (default 5, "
                         "-> ~40 windows/series with N_BINS=8)")
    ap.add_argument("--analyze", metavar="FILLED_CSV",
                    help="compute kappa + suggested band from a filled-in sample")
    args = ap.parse_args()

    if args.analyze:
        analyze(args.analyze)
    else:
        build_sample(args.n_per_bin)


if __name__ == "__main__":
    main()
