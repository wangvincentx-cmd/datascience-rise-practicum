# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo holds multiple independent BU RISE research pipelines, each predicting
whether someone "saw something coming" before it happened, using only
information available at the time of the prediction. They share no code but
share a discipline: strict information-time leakage control and imbalance-aware
evaluation (the target event is always rare).

- **`bill_arm/`** — the active, primary project. Predicts whether a newly
  introduced U.S. Congressional bill becomes law, and whether national press
  coverage adds signal beyond structural features. Has its own venv, tests,
  and detailed README. Work here unless told otherwise.
- **`JeremysShit/`** — the "economy arm": newspaper predictions of economic
  crises (1900s LOC archives) scored against NBER/FRED and the Livingston
  Survey of economists. See `JeremysShit/README.md`.
- **`JeremysShit/election_arm/`** — a second arm sharing the economy arm's
  design: presidential election forecasts from newspapers (1896-2008). See
  `JeremysShit/election_arm/VINCENT_README.md`.

Unless a task explicitly references the economy or election arm, assume work
is scoped to `bill_arm/`.

## Environment

The repo-root `.venv` is broken (stale interpreter path, no packages) — do not
use it. Each subproject manages its own virtualenv:

```
cd bill_arm
python -m venv .venv && source .venv/bin/activate   # if not already created
pip install -r requirements.txt
```

`bill_arm/requirements.txt`: requests, pandas, scikit-learn, xgboost, openai.

API keys (as env vars, only needed past the offline-test stage):
```
CONGRESS_API_KEY    # api.congress.gov (DEMO_KEY works for spot-checks, ~40 req/hr)
NYT_API_KEY         # developer.nytimes.com, Article Search API
DEEPSEEK_API_KEY    # platform.deepseek.com, used by extract_press.py's LLM linker
```

## Commands (bill_arm)

Always run the offline test suite first — it exercises the real pipeline
functions against mocked API responses, no network or keys required:

```
python test_offline.py
```

Full pipeline, in dependency order (see `bill_arm/README.md` for the build-stage
rationale — get the structural baseline solid before spending press/LLM budget):

```
python download_bills_bulk.py --congress 118 --bill-types sjres   # small smoke test
python download_bills_bulk.py --congress 108 ... --congress 118   # full corpus, no key/rate-limit
python build_macro_features.py                                    # once, covers 2002-2025
python build_features.py --congress 118
python model.py --features data/features.csv --test-congresses 118

python link_coverage.py --congress 118 --limit 25
python extract_press.py --congress 118 --limit 20
python coverage_report.py --congress 118          # decision gate: read covered-subset size before continuing

python join_dataset.py --congress 118
python model.py --features data/features.csv --modeling-csv data/modeling.csv --test-congresses 118
```

`download_bills.py` (Congress.gov API) is the alternative ingester, used only
for spot-checks or `--fetch-text` (bulk files carry title only, not introduced
bill text).

## Architecture (bill_arm)

Data flows through a fixed pipeline, each stage a standalone script writing
CSV/JSONL that the next stage reads:

```
download_bills_bulk.py / download_bills.py -> data/bills/{congress}.jsonl
build_macro_features.py                    -> data/macro_daily.csv
build_features.py (joins macro_daily.csv)  -> data/features.csv
link_coverage.py                           -> data/press_raw/{congress}.jsonl
extract_press.py                           -> data/press_labeled/{congress}.jsonl, data/press_features_{congress}.csv
coverage_report.py                          (decision gate, no output file)
join_dataset.py                            -> data/modeling.csv
model.py                                    (Model 1 on features.csv; Model 2 on modeling.csv)
```

`model.py` trains two things depending on invocation: **Model 1** (structural
features only — the reproduction baseline, research question 1) when given
just `--features`, and **Model 2** (structural + press features, trained and
evaluated on the press-covered subset only) when `--modeling-csv` is also
passed. Both use class-weighted logistic regression and XGBoost with
`scale_pos_weight`, calibrated via `CalibratedClassifierCV` where enough
positive examples exist.

**The leakage rule governs every stage.** Only introduction-time information
is a legal feature: sponsor identity/party, chamber, bill type, policy area,
referred committee, *original* cosponsors only (`isOriginalCosponsor == True`),
introduced date/title/text. Press coverage is windowed to
`[introduced_date - 7d, introduced_date + 180d]`. Macro indicators in
`build_macro_features.py` are each shifted forward by their real-world
publication lag (documented per-series in that file's docstring) before
joining, so a bill only "sees" macro data that would actually have been
public by its introduction date — `recession_flag` (NBER/USREC) in particular
is announced 6-21 months after the fact and should be treated as
backward-looking, not real-time, if it shows up as an important feature.

**Splits are always by Congress, never random** (`split_by_congress` in
`model.py`): bills within one Congress share a political environment
(chamber control, what's salient that cycle), so a random split leaks that
context into the test set. Train on earlier Congresses, test on the most
recent one or two.

**Accuracy is never reported as a metric** — ~3-4% of introduced bills become
law, so a trivial "always dies" classifier scores ~96% and is meaningless.
PR-AUC on the passed class is primary, alongside ROC-AUC, precision/recall on
the passed class, and Brier score / calibration curve.

The press experiment (`link_coverage.py` -> `extract_press.py` ->
`coverage_report.py`) only makes sense to continue past the decision gate: if
the covered subset is under ~300 bills with a non-neutral press prediction,
it's underpowered as specified. `extract_press.py`'s article-to-bill linking
is done by an LLM call and is fuzzy — before trusting its output, manually spot
check ~20 `about_this_bill: true` calls from `data/press_labeled/{congress}.jsonl`
against the source article.

`data/majority_by_congress.csv` is a hardcoded lookup (108th-118th chamber
majorities) used to derive `sponsor_in_majority`.
