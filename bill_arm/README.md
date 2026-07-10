# Did the Press See It Coming? Predicting Bill Passage

Predicts whether a newly introduced U.S. Congressional bill becomes law,
using only features known at introduction time, and tests whether national
newspaper coverage adds predictive signal beyond the bill's structural
features (sponsor party, majority status, cosponsors, committee, chamber,
policy area, text).

## Research questions

1. Which structural factors most affect whether a bill becomes law. This
   reproduces well-trodden ground (Nay 2017, PLOS ONE; GovTrack's bill
   prognosis model) and is the baseline, not the contribution.
2. Does national newspaper coverage improve prediction of passage over
   structural features alone -- did the press see it coming better than the
   sponsor's party and cosponsor count already predict? This is the novel
   contribution.
3. Among bills the press did cover, was the direction of the press's
   prediction (will pass / will fail) accurate?

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CONGRESS_API_KEY=your_key      # https://api.congress.gov/sign-up (or DEMO_KEY, ~40 req/hr)
export NYT_API_KEY=your_key           # https://developer.nytimes.com/, enable Article Search API
export GEMINI_API_KEY=your_key        # https://aistudio.google.com/apikey (free tier)
```

`link_coverage.py` and `extract_press.py` also auto-load a `bill_arm/.env` file
(via `python-dotenv`) if present, as an alternative to `export`. `.env` is
git-ignored -- never commit it.

Run `python test_offline.py` first. It runs the real parsing, aggregation,
and modeling functions against mocked API responses -- no network or API
keys needed. It must pass before spending any API budget.

## Pipeline and build order

```
download_bills_bulk.py  govinfo BILLSTATUS bulk ingester -> data/bills/{congress}.jsonl  (preferred)
download_bills.py    Congress.gov API ingester     -> data/bills/{congress}.jsonl  (spot-checks, bill text)
build_macro_features.py  FRED macro indicators (no key) -> data/macro_daily.csv
build_features.py    structural feature table      -> data/features.csv  (joins macro_daily.csv if present)
model.py              Model 1 (and Model 2, see below), split by Congress, metrics
link_coverage.py      NYT search per bill           -> data/press_raw/{congress}.jsonl
extract_press.py      LLM article-match + prediction -> data/press_labeled/{congress}.jsonl
                                                       data/press_features_{congress}.csv
coverage_report.py    Section 8.1 decision gate
join_dataset.py        structural + press merge      -> data/modeling.csv
test_offline.py        mock-response tests, run this first
data/majority_by_congress.csv   per-Congress chamber majorities (hardcoded, 108th-118th)
```

Build in stages, cheapest and most certain first:

1. `download_bills.py` -> `build_macro_features.py` -> `build_features.py` ->
   `model.py` (Model 1). Get an honest structural baseline (now including
   the macroeconomic climate at introduction) with imbalance-aware metrics
   before touching the press pipeline.
2. `link_coverage.py` -> `extract_press.py` -> `coverage_report.py`. Stop at
   the decision gate and read the covered-subset size before spending more
   budget. If it's under ~300 covered bills with a non-neutral prediction,
   the press experiment is underpowered as specified -- see the gate's
   printed guidance.
3. Only if the gate passes: `join_dataset.py`, then
   `python model.py --modeling-csv data/modeling.csv` for the Model 1 vs.
   Model 2 comparison and research question 3.

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
python model.py --features data/features.csv --test-congresses 118

python link_coverage.py --congress 118 --limit 25
python extract_press.py --congress 118 --limit 20
python coverage_report.py --congress 118

python join_dataset.py --congress 118
python model.py --features data/features.csv \
                 --modeling-csv data/modeling.csv --test-congresses 118
```

For the real study, scope is the 108th-118th Congresses (2003-2024): modern
enough for reliable Congress.gov metadata, recent enough to have matching
NYT coverage. Split by Congress for train/test, never randomly -- bills
within a Congress share the same political environment, and a random split
leaks that context into the test set.

## The target and why accuracy isn't reported

`became_law`: True if `latestAction` text contains "Became Public Law" /
"Became Private Law", or the bill's `laws` array is non-empty. About 3-4% of
introduced bills become law. A model that always predicts "dies" scores
~96% accuracy and has zero value, so accuracy is never reported here.
PR-AUC on the passed class is the primary metric, alongside ROC-AUC,
precision/recall on the passed class, and a Brier score / calibration curve.
Both models use class-weighted logistic regression and XGBoost with
`scale_pos_weight`, and calibrate probabilities via `CalibratedClassifierCV`
where the training set has enough positive examples (falls back to an
uncalibrated fit otherwise -- see `model.fit_calibrated`).

## Leakage rules

Only introduction-time information is a legal feature: sponsor identity and
party, chamber, bill type, policy area, referred committee, *original*
cosponsors only (`isOriginalCosponsor == True`), introduced date, title and
introduced text. Later cosponsors, later actions, committee/floor votes, and
anything dated after introduction are forbidden. Press coverage is windowed
to `[introduced_date - 7 days, introduced_date + 180 days]` (capped at the
bill's final action if that came sooner) for the same reason.

Macroeconomic indicators (`build_macro_features.py`) are joined by
introduced_date but must respect the same rule: government statistics are
revised and released with a lag after the period they describe, so each
FRED series is shifted forward by its typical publication lag before the
join -- see the script's docstring for the per-series lag and the USREC
(NBER recession) caveat specifically.

## Known limits

1. The NYT Article Search API returns headline, abstract, lead paragraph,
   and snippet only, never full article text, so press recall is limited.
2. Most bills get no national coverage; the press experiment runs on a
   small, salient subset, not the full bill population.
   `coverage_report.py` reports that subset's size -- read it before
   trusting anything downstream of it.
3. Article-to-bill linking is done by an LLM call and is fuzzy. Before
   trusting the press experiment, manually read ~20 of the linker's
   `about_this_bill: true` calls from `data/press_labeled/{congress}.jsonl`
   against the source article to check it isn't attaching unrelated
   coverage; report that manual-check accuracy in the writeup.
4. The structural model (Model 1) reproduces existing work; the press test
   (Model 2, research questions 2-3) is the novel piece.
5. `sponsor_is_committee_chair` is not implemented (no committee-leadership
   data source wired up) and is always null in the feature table.
6. `recession_flag` (NBER, via FRED's USREC) is announced 6-21 months after
   the fact historically, not in real time. It's included as a coarse,
   backward-looking macro signal, not something a real-time predictor could
   have used at introduction -- disclose this if it shows up as an important
   feature.

## Prior art

- Nay, J.J. (2017), *"Predicting and Understanding Law-Making with Word
  Vectors and an Ensemble Model"*, PLOS ONE -- ~70,000 bills, 2001-2015
  (~2,513 became law), same train-on-past/test-on-future discipline used
  here.
- GovTrack's long-running bill prognosis model.
- Various 2025 student/commercial tools predicting passage from the
  Congress.gov API with imbalance-aware ensembles.
