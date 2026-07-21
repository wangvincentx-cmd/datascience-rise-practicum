"""
A real input -> output predictor, not another factor-analysis pass.

model.py already trains and VALIDATES a real classifier (LOEO accuracy
0.593 with the grouped-CV-tuned LOGIT_C, permutation-test p=0.0099 on the
1,628-claim corpus -- it demonstrably beats chance, driven mainly by
`direction` and `epu`). This script packages that exact model as something
you can actually use: give it a new, unresolved economic prediction's
characteristics, get back a probability it will turn out correct.

Trained on ALL 1,628 scored claims (not the held-out train/test split
model.py uses for honest evaluation) -- the split exists to measure how
well this approach generalizes; a deployed predictor should use every
historical example available. The 0.593 LOEO accuracy / p=0.0099 result
from model.py is the honest estimate of how good this is; this script is
the same model, put to use. (Corpus grew from 843 -> 1,644 claims
2026-07-19 via a targeted LOC rescrape of under-recall episodes, then
1,644 -> 1,628 after removing 69 exact-duplicate claims found during a
data-cleaning pass -- LOEO accuracy went DOWN slightly overall, 0.624 ->
0.593, not up; more data was not a free win here, see CHANGELOG. Still
solidly beats chance either way.)

KNOWN LIMIT, state this out loud in any demo: the EPU index
(tier2_analysis.epu_series()) only covers 1900-2014. For a claim genuinely
made today, either pass --epu with a real current value looked up at
https://www.policyuncertainty.com/us_monthly.html, or accept the fallback
(the historical median), which is a real degradation, not a live estimate.

Usage:
  python predict_claim.py --topic general_business --voice expert \\
      --confidence assertive --direction improve --region northeast \\
      --quote "The economy will improve substantially by next year."

  python predict_claim.py --interactive
"""

import argparse

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import model as m

VOICE_OPTIONS = ["expert", "official", "journalist", "layperson", "unclear"]
CONFIDENCE_OPTIONS = ["assertive", "hedged"]
DIRECTION_OPTIONS = ["improve", "worsen", "no_change"]
TOPIC_OPTIONS = ["general_business", "employment", "markets", "prices", "other"]
REGION_OPTIONS = sorted(set(m.STATE_TO_REGION.values())) + ["unknown"]
FIN_CENTER_OPTIONS = ["financial-center state", "political hub (DC)", "elsewhere"]


def fit_deployed_models(claims_path="claims_scored.csv"):
    """Fits on ALL scored claims -- see module docstring for why this
    differs from model.py's held-out evaluation split."""
    df = m.build(pd.read_csv(claims_path))
    fitted = {}
    for name, clf in [("logistic_regression", LogisticRegression(max_iter=2000, C=m.LOGIT_C)),
                      ("gradient_boosting", GradientBoostingClassifier(random_state=0))]:
        pipe = m.pipeline(clf)
        pipe.fit(df, df["hit"])
        fitted[name] = pipe
    return fitted, df["epu"].median()


def predict(fitted, epu_fallback, *, topic, voice, confidence, direction,
           region="unknown", fin_center="elsewhere", quote="", year=2026,
           months=12, epu=None, kind="crisis"):
    row = pd.DataFrame([{
        "kind": kind, "topic": topic, "voice": voice, "confidence": confidence,
        "direction": direction, "region": region, "fin_center": fin_center,
        "year": year, "months": months,
        "epu": epu if epu is not None else epu_fallback,
        "quote": quote,
    }])
    return {name: float(pipe.predict_proba(row)[0, 1]) for name, pipe in fitted.items()}


def _validate_choice(value, options, field):
    if value not in options:
        raise SystemExit(f"--{field} must be one of {options}, got {value!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", choices=TOPIC_OPTIONS)
    ap.add_argument("--voice", choices=VOICE_OPTIONS)
    ap.add_argument("--confidence", choices=CONFIDENCE_OPTIONS)
    ap.add_argument("--direction", choices=DIRECTION_OPTIONS)
    ap.add_argument("--region", choices=REGION_OPTIONS, default="unknown")
    ap.add_argument("--fin-center", choices=FIN_CENTER_OPTIONS, default="elsewhere")
    ap.add_argument("--quote", default="",
                    help="the claim's own wording -- feeds the TF-IDF text features")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--months", type=int, default=12, help="claim's stated time horizon")
    ap.add_argument("--epu", type=float, default=None,
                    help="real EPU value at claim time; falls back to the historical "
                        "median (index only covers 1900-2014) if not given -- see "
                        "https://www.policyuncertainty.com/us_monthly.html for a "
                        "current real value")
    ap.add_argument("--interactive", action="store_true",
                    help="prompt for each field instead of using flags -- for live demos")
    ap.add_argument("--claims", default="claims_scored.csv")
    args = ap.parse_args()

    print("Fitting on all 1,628 scored claims (this is the model.py of this repo -- "
         "LOEO-validated accuracy 0.593, permutation-test p=0.0099)...")
    fitted, epu_fallback = fit_deployed_models(args.claims)

    if args.interactive:
        print("\n--- Predict whether an economic claim will turn out correct ---")
        topic = input(f"topic {TOPIC_OPTIONS}: ").strip()
        voice = input(f"voice {VOICE_OPTIONS}: ").strip()
        confidence = input(f"confidence {CONFIDENCE_OPTIONS}: ").strip()
        direction = input(f"direction {DIRECTION_OPTIONS}: ").strip()
        region = input(f"region {REGION_OPTIONS} [unknown]: ").strip() or "unknown"
        quote = input("the claim's actual wording (optional): ").strip()
        epu_in = input(f"EPU value at claim time [blank = historical median "
                       f"{epu_fallback:.1f}]: ").strip()
        epu = float(epu_in) if epu_in else None
    else:
        for field, val, opts in [("topic", args.topic, TOPIC_OPTIONS),
                                 ("voice", args.voice, VOICE_OPTIONS),
                                 ("confidence", args.confidence, CONFIDENCE_OPTIONS),
                                 ("direction", args.direction, DIRECTION_OPTIONS)]:
            if val is None:
                raise SystemExit(f"--{field} is required (or use --interactive). "
                                f"Options: {opts}")
        topic, voice, confidence, direction = (args.topic, args.voice,
                                               args.confidence, args.direction)
        region, quote, epu = args.region, args.quote, args.epu

    result = predict(fitted, epu_fallback, topic=topic, voice=voice,
                     confidence=confidence, direction=direction, region=region,
                     fin_center=args.fin_center, quote=quote, year=args.year,
                     months=args.months, epu=epu)

    print(f"\nPredicted probability this claim turns out CORRECT:")
    for name, p in result.items():
        print(f"  {name:20s} {p:.1%}")
    print(f"\n(baseline/prevalence in the historical data: "
         f"{pd.read_csv(args.claims).dropna(subset=['hit'])['hit'].astype(float).mean():.1%} "
         f"of all claims were correct -- compare the prediction above against this, "
         f"not against 50%)")


if __name__ == "__main__":
    main()
