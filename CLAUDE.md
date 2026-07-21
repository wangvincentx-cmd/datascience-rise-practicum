# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This repo holds multiple independent BU RISE research pipelines, each predicting
whether someone "saw something coming" before it happened, using only
information available at the time of the prediction. They share no code but
share a discipline: strict information-time leakage control and imbalance-aware
evaluation (the target event is always rare).

- **`bill_arm/`** — deprioritized 2026-07-17. Originally predicted whether a
  newly introduced U.S. Congressional bill becomes law, and whether national
  press coverage added signal beyond structural features; the passage-
  prediction model and press-coverage pipeline were dropped (too tied to the
  118th Congress's specific climate; press coverage too sparse to power even
  at full-Congress scale — see CHANGELOG). What remains is bill ingestion and
  structural/macro **factor analysis** (not prediction) — see
  `bill_arm/README.md`.
- **`JeremysShit/`** — the "economy arm": newspaper predictions of economic
  crises (1900s LOC archives) scored against NBER/FRED and the Livingston
  Survey of economists. See `JeremysShit/README.md`. Active direction as of
  2026-07-17: a new model predicting the *state* of the economy from a
  newspaper-article corpus (as opposed to the existing claim-extraction
  approach), using the NYT article corpus in `JeremysShit/election_arm/data/raw/`.
- **`JeremysShit/election_arm/`** — a second arm sharing the economy arm's
  design: presidential election forecasts from newspapers (1896-2008). See
  `JeremysShit/election_arm/VINCENT_README.md`.

No single arm is the default scope anymore — confirm which one a task means
if it's not obvious from context.

## Environment

The repo-root `.venv` is broken (stale interpreter path, no packages) — do not
use it. Each subproject manages its own virtualenv:

```
cd bill_arm
python -m venv .venv && source .venv/bin/activate   # if not already created
pip install -r requirements.txt
```

`bill_arm/requirements.txt`: requests, pandas, scikit-learn, xgboost.

API keys (as env vars, only needed past the offline-test stage):
```
CONGRESS_API_KEY    # api.congress.gov (DEMO_KEY works for spot-checks, ~40 req/hr)
```

## Commands (bill_arm)

Always run the offline test suite first — it exercises the real pipeline
functions against mocked API responses, no network or keys required:

```
python test_offline.py
```

Ingestion + factor-table build (see `bill_arm/README.md` for detail):

```
python download_bills_bulk.py --congress 118 --bill-types sjres   # small smoke test
python download_bills_bulk.py --congress 108 ... --congress 118   # full corpus, no key/rate-limit
python build_macro_features.py                                    # once, covers 2002-2025
python build_features.py --congress 118
python make_figures.py                                             # or the other figure scripts
```

`download_bills.py` (Congress.gov API) is the alternative ingester, used only
for spot-checks or `--fetch-text` (bulk files carry title only, not introduced
bill text).

## Architecture (bill_arm)

```
download_bills_bulk.py / download_bills.py -> data/bills/{congress}.jsonl
build_macro_features.py                    -> data/macro_daily.csv
build_features.py (joins macro_daily.csv)  -> data/features.csv
factor_analysis.py                          (shared fitting/importance utilities,
                                              imported by the figure scripts, not run directly)
```

**The leakage rule governs every stage.** Only introduction-time information
is a legal feature: sponsor identity/party, chamber, bill type, policy area,
referred committee, *original* cosponsors only (`isOriginalCosponsor == True`),
introduced date/title/text. Macro indicators in `build_macro_features.py` are
each shifted forward by their real-world publication lag (documented
per-series in that file's docstring) before joining, so a bill only "sees"
macro data that would actually have been public by its introduction date —
`recession_flag` (NBER/USREC) in particular is announced 6-21 months after
the fact and should be treated as backward-looking, not real-time, if it
shows up as an important feature.

**Splits are always by Congress, never random** (`split_by_congress` in
`factor_analysis.py`): bills within one Congress share a political
environment (chamber control, what's salient that cycle), so a random split
leaks that context into the test set. Train on earlier Congresses, test on
the most recent one or two.

**Accuracy is never reported as a metric** — ~3-4% of introduced bills become
law, so a trivial "always dies" classifier scores ~96% and is meaningless.
PR-AUC on the passed class is primary, alongside ROC-AUC, precision/recall on
the passed class, and Brier score / calibration curve.

`data/majority_by_congress.csv` is a hardcoded lookup (108th-118th chamber
majorities) used to derive `sponsor_in_majority`.
