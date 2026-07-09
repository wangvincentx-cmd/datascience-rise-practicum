"""
Benchmark newspaper claims against the Livingston Survey, the oldest
continuous survey of professional economists' expectations, started in 1946
by columnist Joseph Livingston and run by the Philadelphia Fed since 1990.
Surveys run every June and December with 6- and 12-month-ahead forecasts.

ONE MANUAL DOWNLOAD REQUIRED (the Fed blocks scripted pulls):
  1. Go to philadelphiafed.org -> Surveys & Data -> Livingston Survey
     -> Historical Data -> medians file.
  2. Download the MEDIAN forecasts file. You want industrial production (IP):
     base value and the 6-month and 12-month forecasts per survey date.
  3. Save a CSV at data/livingston_medians.csv with these columns
     (rename/reshape from the Fed file as needed; their documentation PDF
     defines each column):
        survey_date, ip_base, ip_6m_forecast, ip_12m_forecast
     Dates as YYYY-MM. One row per survey (June and December each year).

What this script computes: for each survey, the economists' implied direction
call (will industrial production be higher or lower in 6 months), scored the
SAME way the newspaper claims are scored - predicted economic state at the
horizon vs the NBER chronology. A forecast of falling IP counts as predicting
recession; rising IP as expansion. That gives an apples-to-apples hit rate:
newspapers vs the professionals, 1946 onward.

Usage:  python livingston_benchmark.py
Then compare its hit rate to the 1946+ rows of data/scored_economy.csv.
"""

from pathlib import Path

import pandas as pd

from analyze_economy import load_recessions, state_at

HORIZON_MONTHS = 6


def main():
    path = Path("data/livingston_medians.csv")
    if not path.exists():
        raise SystemExit(
            "Missing data/livingston_medians.csv.\n"
            "Download the Livingston median forecasts from the Philadelphia "
            "Fed (see this file's docstring for the exact columns), then rerun.")

    liv = pd.read_csv(path)
    required = {"survey_date", "ip_base", "ip_6m_forecast"}
    missing = required - set(liv.columns)
    if missing:
        raise SystemExit(f"livingston_medians.csv is missing columns: {missing}")

    liv["survey_month"] = pd.PeriodIndex(pd.to_datetime(liv["survey_date"]),
                                         freq="M")
    liv["target_month"] = liv["survey_month"] + HORIZON_MONTHS
    liv["predicted_state"] = (liv["ip_6m_forecast"] < liv["ip_base"]).map(
        {True: "recession", False: "expansion"})

    recessions = load_recessions()
    liv["actual_state"] = liv["target_month"].map(lambda m: state_at(m, recessions))
    liv["hit"] = liv["predicted_state"] == liv["actual_state"]

    print(f"Livingston surveys scored: {len(liv)} "
          f"({liv['survey_month'].min()} to {liv['survey_month'].max()})")
    print(f"Economists' 6-month direction hit rate: {liv['hit'].mean():.2%}")

    by_decade = liv.groupby(liv["survey_month"].dt.year // 10 * 10)["hit"]
    print("\nBy decade:")
    print(by_decade.agg(["mean", "count"]))

    # Turning-point subset: surveys within 6 months before an NBER peak
    peaks = [start - 1 for start, _ in recessions]   # peak = month before recession start
    def near_peak(m):
        return any(0 <= (p - m).n <= HORIZON_MONTHS for p in peaks)
    turning = liv[liv["survey_month"].map(near_peak)]
    if len(turning):
        print(f"\nSurveys taken within 6 months BEFORE a business-cycle peak: "
              f"{len(turning)}, hit rate {turning['hit'].mean():.2%}")
        print("(this is the 'did the experts see the turning point' number)")

    liv.to_csv("data/scored_livingston.csv", index=False)
    print("\nscored table -> data/scored_livingston.csv")
    print("Compare against 1946+ newspaper claims in data/scored_economy.csv "
          "for the headline chart: newspapers vs the professionals.")


if __name__ == "__main__":
    main()
