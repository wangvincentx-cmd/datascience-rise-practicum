# Project Log — bill_arm (Did the Press See It Coming? Predicting Bill Passage)

Living status doc for the bill-passage research project. Not a code diff log —
see `git log` for that. This tracks goals, methods, what's done, what's next,
and what needs a human. Update this alongside code changes, don't let it drift.

## Goals

1. Reproduce the standard structural baseline for predicting whether a newly
   introduced U.S. Congressional bill becomes law (Nay 2017; GovTrack), using
   only information known at introduction time.
2. Test the novel question: does national newspaper coverage add predictive
   signal beyond structural features alone — did the press see it coming
   better than sponsor party / cosponsor count already predict?
3. Among bills the press did cover, was the *direction* of the press's
   prediction (will pass / will fail) accurate?

## Methods (why things are built the way they are)

- **Scope**: 108th-118th Congresses (2003-2024). Split by Congress, never
  randomly — a random split leaks the political environment of a Congress
  (chamber control, what's salient that cycle) into the test set.
- **Leakage discipline**: only introduction-time info is a legal feature
  (sponsor/party, chamber, bill type, policy area, committee, *original*
  cosponsors only, intro date/title/text). Press features are windowed to
  `[intro - 7d, intro + 180d]`. Macro/FRED indicators are shifted forward by
  each series' real publication lag before joining, so a bill only "sees"
  data that would have actually been public by then.
- **Metric**: accuracy is never reported (~3-4% base rate makes it
  meaningless). PR-AUC on the passed class is primary, plus ROC-AUC,
  precision/recall, and Brier/calibration.
- **Models**: class-weighted logistic regression + XGBoost with
  `scale_pos_weight`, calibrated via `CalibratedClassifierCV` where positives
  are sufficient. Model 1 = structural only. Model 2 = structural + press,
  trained/evaluated on the press-covered subset only (fair comparison).

## Done so far

- [x] Bill ingestion for all of 108th-118th via `download_bills_bulk.py`
      (govinfo BILLSTATUS bulk, no key/rate-limit) — `data/bills/108.jsonl`
      through `118.jsonl`, ~2.5M bills' worth of records on disk.
- [x] `download_bills.py` (Congress.gov API path) built and available for
      spot-checks / `--fetch-text`, not yet needed for the main run.
- [x] Structural feature table (`build_features.py`) → `data/features.csv`.
- [x] Macroeconomic climate features (`build_macro_features.py`): FRED
      unemployment, recession dating, GDP growth, CPI inflation, consumer
      sentiment, initial claims — lag-adjusted and joined into the feature
      table and into Model 1/2's feature set.
- [x] `test_offline.py` — full mocked pipeline test suite, passing.
- [x] Model 1 (structural baseline) implemented and refit-able via
      `model.py` / `make_figures.py`.
- [x] Report figures generated in `figures/`: rate-by-Congress, structural
      factors, policy area, cosponsors (EDA), PR curves, calibration,
      feature importances (Model 1 diagnostics on held-out Congresses).
- [x] `join_dataset.py`, `link_coverage.py`, `extract_press.py`,
      `coverage_report.py` are all written, and covered by
      `test_offline.py`'s mocks — but **not yet run against live APIs**
      (see Not done).

## Not done / next up

- [ ] Run `link_coverage.py` against the real NYT Article Search API for the
      108th-118th corpus (needs `NYT_API_KEY`).
- [ ] Run `extract_press.py` (LLM article-to-bill linking + prediction —
      needs `DEEPSEEK_API_KEY`).
- [ ] Run `coverage_report.py` and read the decision gate: if the covered
      subset is under ~300 bills with a non-neutral prediction, the press
      experiment is underpowered as specified and needs a scope change
      before continuing.
- [ ] If the gate passes: `join_dataset.py` → Model 2 run
      (`model.py --modeling-csv data/modeling.csv`) → research questions 2
      and 3 answered.
- [ ] Figures for the press experiment (Model 2 vs. Model 1 PR-AUC
      comparison) — `make_figures.py` currently only covers Model 1.

## Known limits / needs improvement

- `sponsor_is_committee_chair` is not implemented — no committee-leadership
  data source wired up, always null in the feature table. Would need a new
  data source to fix.
- `recession_flag` (NBER/USREC) is announced 6-21 months after the fact
  historically — it's a backward-looking macro signal, not real-time. Flag
  this if it shows up as an important feature in the writeup.
- NYT Article Search API returns headline/abstract/lead paragraph/snippet
  only, never full text — press recall will be structurally limited.
- Most bills get no national coverage at all; the press experiment will run
  on a small, salient subset, not the full bill population — this is a
  scope limitation to disclose, not a bug.
- `extract_press.py`'s article-to-bill linking is an LLM call and is fuzzy —
  needs a manual accuracy check (see below) before the press result can be
  trusted.

## Needs to be done by you (not automatable / requires a decision)

- [ ] Get and export `NYT_API_KEY` and `DEEPSEEK_API_KEY` before the press
      pipeline can run for real (`CONGRESS_API_KEY` only needed if you use
      the API ingester instead of the bulk one).
- [ ] After `extract_press.py` runs for real: manually read ~20 of the
      linker's `about_this_bill: true` calls from
      `data/press_labeled/{congress}.jsonl` against the source article, and
      report that manual-check accuracy in the writeup — this can't be
      automated, it's a human judgment call on whether the LLM is attaching
      unrelated coverage.
- [ ] Decide how to handle the `coverage_report.py` gate if the covered
      subset comes back underpowered (expand date range? lower the bar?
      accept a smaller-scope claim?).
- [ ] Sign off on whether `recession_flag` and other macro features should
      stay in the final feature set given the backward-looking-signal
      caveat, or be dropped/flagged in the writeup.
