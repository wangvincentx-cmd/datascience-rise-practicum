"""
Real input -> output predictor for bill_arm, mirroring
JeremysShit/predict_claim.py's approach for the economy arm.

factor_analysis.py already fits and validates calibrated classifiers
(logistic regression + gradient boosting) on 128,778 bills, 108th-118th
Congress -- it was kept specifically because that fitting/calibration
machinery is reusable, even though the "run this to predict a bill's fate"
CLI was deliberately removed when the passage-prediction project was
dropped (see CHANGELOG, 2026-07-17). This script is that machinery, packaged
back into an actual input -> output tool: give it a hypothetical new bill's
introduction-time characteristics, get back a predicted probability it
becomes law.

Trained on ALL 128,778 bills (not a held-out Congress split) -- the split
exists in factor_analysis.py/make_figures.py to honestly measure
generalization to an unseen Congress; a deployed predictor should use every
historical example available.

KNOWN LIMIT, state this out loud in any demo: ~3.2% of bills become law, so
predicted probabilities are almost always low in absolute terms -- the
model is useful for RELATIVE ranking (which bills look more likely than
others), not for treating any single number as "this bill will/won't pass."
Also: this reflects the 108th-118th Congresses' (2003-2024) political era
specifically, not a universal law of Congress -- see factor_analysis.py's
module docstring.

Usage:
  python predict_bill.py --chamber House --bill-type hr --sponsor-party D \\
      --sponsor-state CA --policy-area Health --committee "Energy and Commerce" \\
      --cosponsors 15 --bipartisan --title "A bill to expand rural health access"

  python predict_bill.py --interactive
"""

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import factor_analysis as fa

CHAMBER_OPTIONS = ["House", "Senate"]
BILL_TYPE_OPTIONS = ["hr", "s", "hjres", "sjres"]  # confirmed present in data/features.csv;
# other Congress.gov types (hres, sres, hconres, sconres) will still work via
# the pipeline's handle_unknown="ignore" one-hot encoding, just untested here.
PARTY_OPTIONS = ["D", "R", "I", "ID", "L", "unknown"]


def fit_deployed_models(features_path="data/features.csv"):
    """Fits on ALL bills -- see module docstring for why this differs from
    factor_analysis.py's held-out-Congress evaluation split."""
    df = fa.load_features(features_path)
    fitted = {}

    n_pos, n_neg = df.y.sum(), len(df) - df.y.sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    logit = Pipeline([
        ("pre", fa.build_preprocessor(fa.CATS, fa.NUMS)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=fa.LOGIT_C)),
    ])
    fitted["logistic_regression"] = fa.fit_calibrated(logit, df)

    xgb = Pipeline([
        ("pre", fa.build_preprocessor(fa.CATS, fa.NUMS)),
        ("clf", XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                              random_state=0, **fa.XGB_PARAMS)),
    ])
    fitted["gradient_boosting"] = fa.fit_calibrated(xgb, df)
    return fitted


def predict(fitted, *, chamber, bill_type, sponsor_party, sponsor_state,
           policy_area, primary_committee, n_original_cosponsors, title,
           bipartisan=False, has_companion_bill=False, sponsor_in_majority=False,
           intro_month_in_session=6, frac_cosponsors_majority=0.5):
    row = pd.DataFrame([{
        "chamber": chamber, "bill_type": bill_type, "sponsor_party": sponsor_party,
        "sponsor_state": sponsor_state, "policy_area": policy_area,
        "primary_committee": primary_committee,
        "n_original_cosponsors": n_original_cosponsors,
        "bipartisan": int(bool(bipartisan)),
        "frac_cosponsors_majority": frac_cosponsors_majority,
        "intro_month_in_session": intro_month_in_session,
        "title_length": len(title.split()),
        "has_companion_bill": int(bool(has_companion_bill)),
        "sponsor_in_majority": int(bool(sponsor_in_majority)),
        "combined_text": title,
    }])
    return {name: float(pipe.predict_proba(row)[0, 1]) for name, pipe in fitted.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chamber", choices=CHAMBER_OPTIONS)
    ap.add_argument("--bill-type", default="hr",
                    help=f"typically one of {BILL_TYPE_OPTIONS}, but any Congress.gov "
                        "type is accepted")
    ap.add_argument("--sponsor-party", choices=PARTY_OPTIONS)
    ap.add_argument("--sponsor-state", help="two-letter state code, e.g. CA")
    ap.add_argument("--policy-area", help="Congress.gov policy area, e.g. Health")
    ap.add_argument("--committee", dest="primary_committee",
                    help="referred committee name, e.g. 'Energy and Commerce'")
    ap.add_argument("--cosponsors", dest="n_original_cosponsors", type=int, default=0)
    ap.add_argument("--bipartisan", action="store_true",
                    help="original cosponsors include both parties")
    ap.add_argument("--companion-bill", dest="has_companion_bill", action="store_true")
    ap.add_argument("--sponsor-in-majority", action="store_true",
                    help="sponsor's party controls this chamber")
    ap.add_argument("--intro-month", dest="intro_month_in_session", type=int, default=6,
                    help="1-24, month within the 2-year Congress")
    ap.add_argument("--title", default="", help="bill title -- feeds the TF-IDF text features")
    ap.add_argument("--interactive", action="store_true",
                    help="prompt for each field instead of using flags -- for live demos")
    ap.add_argument("--features", default="data/features.csv")
    args = ap.parse_args()

    print("Fitting on all 128,778 bills, 108th-118th Congress (this is "
         "factor_analysis.py's validated pipeline)...")
    fitted = fit_deployed_models(args.features)

    if args.interactive:
        print("\n--- Predict whether a bill becomes law ---")
        chamber = input(f"chamber {CHAMBER_OPTIONS}: ").strip()
        bill_type = input(f"bill type (e.g. {BILL_TYPE_OPTIONS}): ").strip()
        sponsor_party = input(f"sponsor party {PARTY_OPTIONS}: ").strip()
        sponsor_state = input("sponsor state (2-letter code): ").strip()
        policy_area = input("policy area (e.g. Health, Taxation): ").strip()
        primary_committee = input("referred committee: ").strip()
        cosponsors = int(input("number of original cosponsors [0]: ").strip() or 0)
        bipartisan = input("bipartisan cosponsors? [y/N]: ").strip().lower() == "y"
        title = input("bill title: ").strip()
        result = predict(fitted, chamber=chamber, bill_type=bill_type,
                         sponsor_party=sponsor_party, sponsor_state=sponsor_state,
                         policy_area=policy_area, primary_committee=primary_committee,
                         n_original_cosponsors=cosponsors, bipartisan=bipartisan,
                         title=title)
    else:
        for field, val in [("--chamber", args.chamber), ("--sponsor-party", args.sponsor_party),
                           ("--sponsor-state", args.sponsor_state),
                           ("--policy-area", args.policy_area),
                           ("--committee", args.primary_committee)]:
            if not val:
                raise SystemExit(f"{field} is required (or use --interactive)")
        result = predict(fitted, chamber=args.chamber, bill_type=args.bill_type,
                         sponsor_party=args.sponsor_party, sponsor_state=args.sponsor_state,
                         policy_area=args.policy_area, primary_committee=args.primary_committee,
                         n_original_cosponsors=args.n_original_cosponsors,
                         bipartisan=args.bipartisan, has_companion_bill=args.has_companion_bill,
                         sponsor_in_majority=args.sponsor_in_majority,
                         intro_month_in_session=args.intro_month_in_session, title=args.title)

    print("\nPredicted probability this bill becomes law:")
    for name, p in result.items():
        print(f"  {name:20s} {p:.2%}")
    print("\n(base rate across all 128,778 bills: 3.24% became law -- compare the "
         "prediction above against THIS, not against 50%. A prediction of e.g. 8% "
         "is a strong positive signal here, not a low number.)")


if __name__ == "__main__":
    main()
