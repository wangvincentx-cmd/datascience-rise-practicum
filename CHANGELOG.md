# Project Log — RISE research arms

Living status doc. Not a code diff log — see `git log` for that. This tracks
goals, methods, what's done, what's next, and what needs a human. Update this
alongside code changes, don't let it drift.

Two arms are tracked here: **bill_arm** (primary) and the **economy arm**
(`JeremysShit/`). The election arm has its own README and is not tracked here.

---

# bill_arm — Did the Press See It Coming? Predicting Bill Passage

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

---

# Economy arm (`JeremysShit/`) — Did newspapers see economic crises coming?

## Goals

1. Score 1900s-era newspaper economic predictions against what actually
   happened (NBER/FRED), across 7 crisis windows and 3 calm control windows.
2. Benchmark newspapers against professional economists (Livingston Survey)
   and households (Michigan sentiment).
3. Identify which factors predicted a claim being *right* (publisher, voice,
   confidence, policy uncertainty at time of printing).

## Done so far

- [x] `newspaper_scraper.py` — LOC Chronicling America scraper. Produced
      `claims_raw.csv`: **1,324 claims, 218 publishers, 1905-1958**, across
      10 episodes (7 crisis, 3 control). Corpus-transparency log in
      `search_log.csv`.
- [x] Ground truth wired up: NBER chronology, FRED (CPIAUCNS, INDPRO,
      UNRATE), historical EPU, Livingston medians.
- [x] `score_claims.py`, `model.py`, `tier2_analysis.py` written; figures
      generated in `figures/`.
- [x] `test_offline.py` — 25 checks, passing.
- [x] `grade_claims.py` (LLM claim grading) made to actually work against a
      live provider. Three blocking bugs fixed: TLS verification failure on
      macOS (now uses `certifi`), Groq's Cloudflare edge rejecting the default
      `Python-urllib` User-Agent with HTTP 403 / CF 1010, and no rate limiting
      or resume. Added `--sleep` (with `Retry-After`-aware 429 backoff),
      `--overwrite`, and resume-from-existing-output. Verified end to end on
      6 real claims via Groq / `llama-3.3-70b-versatile`.
- [x] Fixed: a failed grading run silently overwrote `validation_sample.csv`
      with zero rows, which would have destroyed hand-graded work.
- [x] Fixed three throughput bugs that stalled the first real run dead (1 claim
      in 5 min):
      (a) **No `max_tokens` — the root cause.** Providers reserve a request's
      *maximum possible* output against the tokens/min budget, not what it
      actually uses. Unset, Groq reserved thousands of tokens per call against a
      12k/min ceiling while we genuinely used ~600. Now capped at 300 (the graded
      JSON is ~100), which is worth roughly 10x throughput.
      (b) Exponential backoff slept 10/20/40/80s to wait out a tokens/min window
      that resets in ~200ms — now honours `Retry-After` with a 15s fallback.
      (c) `DAILY_CAP_SECONDS` was 300, so a *transient* 6-minute throttle was
      misread as the daily cap and the run quit. Raised to 1800; the genuine cap
      comes back as hours (~2.5h rolling reset).
      Also `-u`/`flush=True` — the log was silently buffering and looked empty
      during the stall, which hid all of the above.
- [x] Resolved the `1105dda` merge conflicts, which had left `<<<<<<<` markers
      in both `grade_claims.py` (unrunnable — syntax error) and `.gitignore`.
      That commit carried a parallel, independent fix of the same UA/resume
      bugs; kept the HEAD version since it is a superset (adds the `certifi`
      SSL fix, `--sleep`, `--overwrite`).
- [x] `voice` (who is speaking) reworked into a 5-bucket taxonomy —
      `journalist / expert / official / layperson / unclear` — in the LLM rubric
      (`grade_claims.py`) and the second-vendor schema (`tier3_robustness.py`).
      Verified the live model returns only in-taxonomy values. **Decision
      (2026-07-14): `voice` is NOT hand-graded** — dropped from the grading file
      and the kappa report to keep the human task to five columns.
- [x] `handgrade_newspapers/` — blind, episode-stratified 80-claim sample
      (`handgrade_BLANK.csv`, five `human_*` columns), grading instructions
      (`README.md`), and `kappa.py` computing human-vs-human *and* human-vs-LLM
      Cohen's kappa on is_prediction, topic, direction, confidence. Sampled from
      `claims_raw.csv` rather than the LLM's output, so it can catch the LLM
      *wrongly rejecting* a real prediction — the built-in
      `validation_sample.csv` cannot.
- [x] Repo-root `.gitignore` added (there was none; `.env` and API keys were
      one `git add .` from being committed). `claims_graded.csv` and
      `grade_run.log` are ignored while a run is in flight — a half-finished
      grading file is a partial corpus, not a result. Commit the finished one
      deliberately: `git add -f JeremysShit/claims_graded.csv`.

## Not done / next up

- [ ] **Run `grade_claims.py` for real.** IN PROGRESS (2026-07-14), ~176/1,324
      done. **Groq's free tier cannot finish this in one day**: the good model
      (`llama-3.3-70b-versatile`) allows **1,000 requests/day** and we need
      1,324. Plan: run to the cap today (~claim 1,070), rerun tomorrow, resume
      handles the rest (~20 min). Alternatives rejected — `llama-3.1-8b-instant`
      has 14,400 req/day but is far weaker at this nuanced call, and we already
      know the LLM errs toward too-loose.
- [ ] Decide the **conditional rule** before the final regrade: does *"business
      will lack confidence until this uncertainty is dissipated"* count as a
      prediction? Recommend NO — `score_claims.py` cannot verify whether the
      condition was met, so scoring it is unfair to the paper. Needs Vincent's
      sign-off; it goes in `RUBRIC_PROMPT`.
- [ ] Regenerate every downstream artifact afterwards. `claims_scored.csv`,
      `publisher_leaderboard.csv`, `results_by_episode.csv`,
      `model_predictions.csv` and all 8 figures currently derive from
      `--heuristic` keyword labels, not LLM grades.
- [ ] Rewrite `RUBRIC_PROMPT` to close the three gaps found by adjudication
      (quoted forecasts count / ads never count / conditionals — pending
      Vincent's decision), then regrade. Do this AFTER the in-flight run
      finishes. Resume makes the regrade cheap.
- [ ] Compute kappa; if `direction` < 0.6, tighten `RUBRIC_PROMPT` and
      regrade before proceeding to scoring.
- [ ] Spec Step 4: composite 0-1 claim score (accuracy + punctuality +
      specificity). Does not exist in any form; current scoring is binary
      `hit` + Brier. `specificity` has no field at all, and the "soon" /
      "long-term" → time-range inference is unimplemented.
- [ ] Spec Steps 3 and 5: polls. No poll data exists anywhere in the repo.
- [ ] Spec's second model (predict economic *state* from press) — a separate
      model from the existing "was this claim right?" model. Different unit of
      observation (time period, not claim). At episode level this yields only
      10 training rows; needs a month-level corpus expansion to be viable.
- [ ] NYT run for the 6 post-1963 windows in `election_arm/data/windows_economy.csv`
      to extend coverage toward 2010.
- [ ] **ProQuest `covid_2020` window IN PROGRESS.** Added `covid_2020`
      (crisis, 2020-01-01..2020-07-31, NBER Feb-Apr 2020) to
      `election_arm/data/windows_economy.csv`; dataset `covid2020` built in the
      TDM dashboard and parsed in-VM (`tdm_parse.py`). `extract_gpt.py` running
      against the in-VM proxy. Note: this makes crisis:placebo 7:3 (was 6:3) and
      it's a pandemic-driven, not financial-system, crisis — flag as a possible
      outlier. Finish the run, then strip+export label-only preds to the Mac.
- [ ] Fixed `extract_gpt.py` quitting after ~112 articles: a *transient*
      per-minute token limit (`token/minute rate reached`) was matched by the
      old `_is_rate_limit()` and misread as the permanent daily cost cap, so the
      whole run stopped. Split into `_is_daily_cap()` (day rate / cost/day →
      stop cleanly) vs `_is_transient_rate_limit()` (per-minute → wait `TPM_WAIT`
      65s and retry, bounded by MAX_RETRIES). Same bug class as the
      `grade_claims.py` throttle fix. VM copy must be re-synced for the fix to
      take effect.

## Known limits / needs improvement

- **All current numbers are provisional.** `claims_scored.csv` has
  `topic=general_business`, `voice=unclear`, `is_prediction=yes` on all 403
  rows — those are `--heuristic` fallback defaults, not findings.
- **`voice` is an unvalidated feature.** The LLM labels it and `model.py` can
  use it, but it is not hand-graded, so there is no kappa for it. If it shows up
  as an important predictor of accuracy ("experts beat journalists"), that claim
  rests on unaudited LLM labels — disclose it, or hand-grade voice after all.
- **The rubric has three concrete gaps, found by adjudicating all 80 hand-graded
  claims against an independent pass (`handgrade_newspapers/adjudication_claude.csv`,
  82% agreement, 14 disagreements — the disagreements are patterned, not random):**
  1. **Quoted forecasts.** Vincent rejected the four most explicit claims in the
     sample — incl. #867 (*"the downward trend will continue until the last
     quarter of '50 when production will be 15 per cent below the '48 average"*:
     direction + dated horizon + magnitude) and #1048 (Slichter, named economist,
     dated horizon) — apparently because they are reported in a third party's
     voice. That rule would gut the corpus of its best content; `voice` exists
     precisely so quoted experts can be counted *and* distinguished. The rubric
     must say quoted forecasts count.
  2. **Advertisements.** #972 (champagne price list), #861 (Standard Oil motor-oil
     ad), #1210 (lumber-yard promo) were graded as predictions because they use
     words like "outlook"/"prosperity". The rubric says ads → no; it needs to say
     so louder.
  3. **Conditionals.** #845, #223 ("*Until this uncertainty is dissipated,
     business will lack confidence*") are genuinely ambiguous and currently
     unruled. `score_claims.py` cannot verify whether the "if" was satisfied, so
     scoring them as forecasts is systematically unfair to the paper. NEEDS A
     DECISION (recommend: conditional → not a prediction).
  Net: Vincent's 27.5% prediction rate is too strict (driven by gap 1); the LLM's
  ~63% is too loose (swallowing ads and OCR garbage). Both err, in opposite
  directions. Fix `RUBRIC_PROMPT` and regrade — but only after the current run
  finishes, so the corpus isn't split across two rubrics.
- `publisher_leaderboard.csv` has only 4 publishers above the n≥10 threshold.
  Underpowered for "which paper predicted best" until the corpus grows.
- **Sampling skew**: 378 of 1,324 claims (29%) are from Washington DC, 261
  from the Evening Star alone. `tier2_analysis.py` codes DC as a "financial
  center" — that bucket is mostly one government town's paper, which likely
  drives `fig_geography.png`. Needs a coding decision.
- One search term (`"business outlook"`) produced 606 of 1,324 claims (46%).
  `search_log.csv` shows we sampled rather than cherry-picked, but says
  nothing about what the 12 terms *missed*.
- The retrospective-vs-prediction boundary is the main leakage risk. Claim #1
  (`"Panic Is Nearly Over"`) was graded `is_prediction: yes, direction:
  improve`. Misfiled retrospectives add harmless noise to the current model
  but would *manufacture* signal in the state-prediction model.
- **`horizon_months` is mostly `vague`** — ~70% of predictions in the first 100
  real graded claims (44 vague / 14 six-month / 4 twelve-month). `score_claims.py`
  needs a 6 or 12 to pick an outcome window, so a large share of claims may be
  unscorable as things stand. This is precisely the gap spec Step 4 describes
  ("if the paper gives no number, read 'soon' / 'long-term' and map it to a
  range per sub-genre") — that inference is still unimplemented. Confirm the
  final distribution once the run lands, then decide: implement the mapping,
  or default vague→12 and disclose it.
- `JeremysShit/` has no venv of its own (`test_offline.py` needs pandas; run
  it with `bill_arm/.venv` for now).
- `famous_calls.csv` contains a row from publisher `"test paper a"` — test
  fixture data leaked into a real output file.

## Needs to be done by you (not automatable / requires a decision)

- [ ] **Hand-grade the 80-claim sample.** Vincent DONE (80/80, 2026-07-14: 22
      predictions / 58 not). **Jeremy still needs to grade** — his copy must be
      made from `handgrade_BLANK.csv`, never from Vincent's file, or the
      human-vs-human kappa is worthless. Five `human_*` columns per row (see
      `handgrade_newspapers/README.md`), ~2-3 hours. This is the credibility
      floor: without it the method is "we asked an AI and believed it."
- [ ] Rotate the Groq API key — it was pasted in plaintext into a chat
      transcript.
- [ ] Decide whether DC counts as a financial center or a government town,
      and re-run `tier2_analysis.py` accordingly. Defend it in methods.
- [ ] Decide what "polls" means: Gallup pre-1960 microdata is not freely
      downloadable. Livingston (economists, 1946+) and UMCSENT (households,
      1952+) are already wired up — are they an acceptable substitute?
- [ ] Preregister the `no_change` band choices in `score_claims.py` (CPI
      ±1.5%, INDPRO ±2%, UNRATE ±0.3pt) *before* seeing results, then run the
      sensitivity analysis the docstring already flags.
- [ ] Hand-code publisher metadata (circulation, political lean, urban/rural)
      for the top ~30 publishers from the LOC US Newspaper Directory. This is
      the only human task that adds *features* to the model rather than
      auditing it — it enables "did partisan papers predict worse?"
