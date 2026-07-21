# Congressional Bill Factor Analysis

Ingests U.S. Congressional bill metadata (108th-118th Congresses, 2003-2024)
and analyzes which introduction-time structural factors (sponsor party,
majority status, cosponsors, committee, chamber, policy area, text, and the
macroeconomic/political climate at introduction) are associated with a bill
becoming law.

**Not a predictive deployment tool.** The bill-passage *prediction* model and
the NYT press-coverage pipeline that fed its Model 2 comparison were dropped
2026-07-17 (see CHANGELOG) -- the project was scoped too tightly to the 118th
Congress's specific political/economic climate to generalize, and the press
pipeline's coverage was too sparse (~0.4-0.6% of bills) to reach a
well-powered sample even at full-Congress scale. What's kept here is the
data-ingestion and factor-analysis machinery: it fits classifiers internally
(logistic regression, gradient boosting) as the mechanism for reading off
feature importances/calibration, same reasoning as e.g. permutation
importance in any factor-analysis workflow -- the deliverable is "which
factors matter and how much," not a bill-by-bill passage forecast.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CONGRESS_API_KEY=your_key      # https://api.congress.gov/sign-up (or DEMO_KEY, ~40 req/hr)
```

Run `python test_offline.py` first. It runs the real parsing, feature-
derivation, and factor-analysis fitting functions against mocked API
responses -- no network or API keys needed.

## Pipeline and build order

```
download_bills_bulk.py   govinfo BILLSTATUS bulk ingester -> data/bills/{congress}.jsonl  (preferred)
download_bills.py        Congress.gov API ingester         -> data/bills/{congress}.jsonl  (spot-checks, bill text)
build_macro_features.py  FRED macro indicators (no key)    -> data/macro_daily.csv
build_features.py        structural feature table          -> data/features.csv  (joins macro_daily.csv if present)
factor_analysis.py       shared fitting/importance/calibration utilities (imported by the figure scripts below, not run directly)
make_figures.py, model_figures.py, make_ablation_figure.py,
_ablation_figdata.py, ablation_macro.py, make_timelevel_figure.py,
time_level_economy.py    figures -> figures/*.png
test_offline.py          mock-response tests, run this first
data/majority_by_congress.csv   per-Congress chamber majorities (hardcoded, 108th-118th)
```

Bill ingestion has two paths that emit the identical JSONL schema:

- **`download_bills_bulk.py` (preferred):** downloads GPO's BILLSTATUS
  bulk-data zips from govinfo.gov -- no API key, no rate limit, whole
  108th-118th corpus in minutes. Verified field-for-field identical to the
  API path on live data. Does not include bill text (title only).
- **`download_bills.py`:** the Congress.gov API, 4 calls per bill,
  ~5,000 calls/hour with a registered key (~12h per Congress). Use it for
  spot-checks and for `--fetch-text` (introduced bill text), which the bulk
  files don't carry.

Example commands, smallest first:

```
python download_bills_bulk.py --congress 118 --bill-types sjres   # tiny test
python download_bills_bulk.py --congress 108 ... --congress 118    # full corpus
python download_bills.py --congress 118 --limit 25                 # API spot-check
python build_macro_features.py                                     # once; covers 2002-2025
python build_features.py --congress 118
python make_figures.py       # or any of the other figure scripts
```

Scope is the 108th-118th Congresses (2003-2024): modern enough for reliable
Congress.gov metadata. Split by Congress for train/test in any fitting done
for factor analysis, never randomly -- bills within a Congress share the
same political environment, and a random split leaks that context into the
held-out set.

## The target and why accuracy isn't reported

`became_law`: True if `latestAction` text contains "Became Public Law" /
"Became Private Law", or the bill's `laws` array is non-empty. About 3-4% of
introduced bills become law. A model that always predicts "dies" scores
~96% accuracy and has zero value, so accuracy is never reported here.
PR-AUC on the passed class is the primary metric when fitting is used for
factor analysis, alongside ROC-AUC, precision/recall on the passed class,
and a Brier score / calibration curve. Fitting uses class-weighted logistic
regression and XGBoost with `scale_pos_weight`, and calibrates probabilities
via `CalibratedClassifierCV` where the training set has enough positive
examples (falls back to an uncalibrated fit otherwise -- see
`factor_analysis.fit_calibrated`).

## Leakage rules

Only introduction-time information is a legal feature: sponsor identity and
party, chamber, bill type, policy area, referred committee, *original*
cosponsors only (`isOriginalCosponsor == True`), introduced date, title and
introduced text. Later cosponsors, later actions, committee/floor votes, and
anything dated after introduction are forbidden.

Macroeconomic indicators (`build_macro_features.py`) are joined by
introduced_date but must respect the same rule: government statistics are
revised and released with a lag after the period they describe, so each
FRED series is shifted forward by its typical publication lag before the
join -- see the script's docstring for the per-series lag and the USREC
(NBER recession) caveat specifically.

## Known limits

1. `sponsor_is_committee_chair` is not implemented (no committee-leadership
   data source wired up) and is always null in the feature table.
2. `recession_flag` (NBER, via FRED's USREC) is announced 6-21 months after
   the fact historically, not in real time. It's included as a coarse,
   backward-looking macro signal, not something a real-time predictor could
   have used at introduction -- disclose this if it shows up as an important
   feature.
3. Findings from fitted-classifier factor analysis (feature importances,
   calibration) reflect the 108th-118th Congresses' specific political era
   and should not be read as universal claims about what makes a bill pass.

## Prior art

- Nay, J.J. (2017), *"Predicting and Understanding Law-Making with Word
  Vectors and an Ensemble Model"*, PLOS ONE -- ~70,000 bills, 2001-2015
  (~2,513 became law), same train-on-past/test-on-future discipline used
  here.
- GovTrack's long-running bill prognosis model.
- Various 2025 student/commercial tools predicting passage from the
  Congress.gov API with imbalance-aware ensembles.
