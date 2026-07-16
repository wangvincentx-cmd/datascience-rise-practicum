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
- [x] **Economy→politics ablation** (`ablation_macro.py`, 2026-07-15, the
      cross-arm (C) question). Two findings:
      (1) The 6 macro features are joined into `features.csv` but `model.py`'s
      `NUMS` never uses them — Model 1 is purely structural.
      (2) Adding them does NOT help predict bill passage — it HURTS.
      Test on Congresses 117-118 (31,796 bills, 639 became law):
      | model     | struct PR-AUC | +macro | delta | 95% CI |
      |-----------|--------------|--------|-------|--------|
      | logistic  | 0.216 | 0.197 | -0.019 | [-0.043,+0.002] ns |
      | grad-boost| 0.333 | 0.281 | -0.053 | [-0.071,-0.034] **sig** |
      Interpretation: bill passage is driven by structural/political factors,
      not the macro climate. Mechanism: macro features are ~constant within a
      Congress, so under the (correct) by-Congress split they add no
      within-test signal and instead overfit training-Congress economic
      regimes. Clean answer to (C): the economy does NOT measurably improve —
      and naively added, degrades — prediction of this political outcome.
      Caveats: tests aggregate climate → passage only (not bill *selection* or
      per-policy-area effects); the negative result is partly a property of
      Congress-level features under a Congress split (a random split would leak
      and make macro look helpful — that contrast is a methods point).
- [x] **Presentation figure** for the ablation
      (`figures/fig_macro_ablation.png`, via `_ablation_figdata.py` →
      `make_ablation_figure.py`, 2026-07-15). Two panels: (A) held-out PR-AUC
      barely changes (drops) when the economy is added; (B) the 6 macro features
      = only 2.3% of gradient-boosting importance (bill text alone = 66%). CVD-
      validated palette (blue=politics, orange=economy). Reproduce: run the two
      scripts in order (figdata writes /tmp/ablation_figdata.json).
- [x] **Time-level economy test** (`time_level_economy.py` +
      `make_timelevel_figure.py` → `figures/fig_timelevel_economy.png`,
      2026-07-15). Unit = quarter of introduction (88 quarters, 2003-2024);
      outcome = passage rate. Detrended correlations with the economic climate:
      consumer sentiment +0.34*, GDP growth +0.29*, unemployment -0.24*, jobless
      claims -0.22* (CPI, recession-share ns) — 4/6 directionally consistent
      ("good economy → modestly more bills pass"). SOFT signal, not conclusive:
      the cleanest cut (recession vs expansion quarters, 3.8% vs 3.5%, p=0.61) is
      NULL, effective n is ~11 political regimes with autocorrelation inflating
      significance, and it's uncontrolled for political confounds (divided govt,
      election cycles). Two-arm story: economy doesn't pick WHICH bill passes
      (bill-level null) but may gently move overall throughput (time-level soft
      signal) — visible only after changing the unit from bill to period.

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
- [x] **Consensus gold standard established** — 80 claims (`claims_raw_val80.csv`,
      episode-stratified), coded to consensus by Vincent/Bode/Jeremy, then
      **reconciled to the final rubric** by Vincent (`handgrade_newspapers/
      handgrade_consensus_reconciled.csv`; original preserved untouched, every
      change annotated). This is the one gold standard every grader model below
      was measured against. Full writeup + the honest independence caveat (all
      reconciliation changes moved gold toward the LLM being tested that day —
      defensible because each change was rubric-dictated and several remaining
      disagreements were explicitly left un-flipped) in
      `handgrade_newspapers/KAPPA_RESULTS.md`.
- [x] **`RUBRIC_PROMPT` de-leaked and hardened.** Two validation claims that had
      been embedded as worked examples (grading the model on an exam containing
      its own answer key) were replaced with invented, out-of-corpus examples.
      Rubric explicitly closed three gaps found by adjudication: quoted
      forecasts count, ads/OCR-garbage never count, conditionals excluded
      (`score_claims.py` can't verify an "if" was satisfied, so scoring a
      conditional as a forecast is unfair to the paper).
- [x] **Seven grader models bake-off**, all scored on the identical 80-claim
      consensus gold (`eval_vs_consensus.py`):
      | model                | is_prediction | direction | topic | confidence |
      |-----------------------|:---:|:---:|:---:|:---:|
      | Llama-3.3-70b (Groq)  | 0.87 | 0.78 | 0.73 | 0.19 |
      | gpt-4o-mini           | 0.64 | 0.81 | 0.35 | 0.19 |
      | gpt-4o                | 0.74 | 0.81 | 0.64 | 0.34 |
      | gpt-5.6-luna          | 0.78 | 0.65 | 0.64 | 0.22 |
      | gpt-4.1-mini          | 0.83 | 0.74 | 0.62 | 0.32 |
      | gpt-5-mini            | 0.67 | 0.81 | 0.69 | **0.45** |
      | **gpt-4.1 (chosen)**  | **0.89** | **0.90** | **0.72** | 0.17 |
      `gpt-4.1` won clearly — non-reasoning (no empty-output/token-budget risk,
      no rate-limit flakiness), cheapest reliable option, and only 4 of 78
      disagreements. Notable rejected candidates and why:
        - Groq/Llama-3.3-70b was the original choice and is free, but the free
          tier's daily/per-minute token caps made the full 1,324-claim corpus a
          multi-day grind even with 5-key rotation (see retired
          `/tmp/supervise_full.sh`) — abandoned once a $5 OpenAI budget made
          `gpt-4.1` both faster and higher-quality.
        - `gpt-5.6-luna` (a reasoning model) looked strong on an early, partial
          val80 run (73/80, direction 1.00) but that was an artifact of the
          easier subset succeeding — on the full 80 it's direction=0.65, and it
          needed real engineering to even run: `max_completion_tokens` instead
          of `max_tokens`, no custom `temperature`, and a token budget high
          enough that invisible reasoning tokens don't exhaust it before the
          visible JSON is written (all three fixed generically in
          `grade_claims.py`, keyed off the API's own error messages / an
          empty-content+finish_reason=length signature — not a model-name
          allowlist, so it self-adapts to any future model with the same
          restriction).
        - `gpt-5-mini` reasons *harder* than luna (583 mean reasoning tokens vs
          185) despite the "mini" name, and has one systematic bug: all 13
          disagreements were the same direction (gold=no, llm=yes) — it
          under-applies the rubric's exclusion rules (ads, conditionals,
          non-economic content), not a gold-mismatch issue.
- [x] **Full corpus regraded on `gpt-4.1`** (2026-07-16, `--sleep 0.35`,
      `--overwrite`, real cost $3.37 at confirmed $2/$8 per 1M — computed from
      a measured 959-in/78-out token survey, not a guess). **1,324/1,324
      graded, zero blank rows, 672 (51%) judged real predictions.** Prior
      partial runs preserved as `.bak` files rather than deleted
      (`claims_graded_leaked_partial.bak`, `claims_graded_llama_partial_138.bak`,
      `claims_graded_luna_partial_19.bak`).
- [x] **`score_claims.py` run on the real corpus** (2026-07-16; also fixed a
      missing `openpyxl` dependency that was silently disabling the Livingston
      comparison). Real findings, not `--heuristic` placeholders:
      - **668 predictions, 584 scorable** (84 correctly excluded: pre-1913
        price claims, pre-1948 employment claims — unscorable, not guessed).
      - **Newspapers beat professional economists**: 64.3% directional hit rate
        (n=182, 1946-63) vs. Livingston survey economists' 54.4% (n=68); the
        newspaper 95% CI [57.1%, 71.4%] excludes the economist point estimate.
      - **1929 Crash is the disaster case**: 13% hit rate (worst of all 10
        episodes by far), driven by 80% of papers predicting "improve" right
        before the Crash. `fig_hit_by_episode.png`.
      - **Overconfidence effect**: assertive claims hit 55.1% vs hedged claims'
        58.1% — papers that hedged were better calibrated, not worse.
      - **Publisher leaderboard now has 7 publishers** clearing the n>=10
        threshold (was 4 under the old heuristic pipeline) — Key West Citizen
        leads at 77.8% (n=18).
      - Crisis-window predictions (52.6%) were less accurate than calm-control
        predictions (73.3%) — sanity-checks the episode design.
      - `famous_calls.csv` regenerated clean — the old `"test paper a"` fixture
        row is gone (was an artifact of the pre-real-grading pipeline).
- [x] **Horizon inference** (spec Step 4) added to `score_claims.py`
      (`resolve_horizon()` + `--horizon-scale` sensitivity knob + a
      `horizon_basis` audit column). Maps vague-horizon claims ("soon"->6mo,
      "long-term"->24mo) instead of the old blanket 12-month default. Improves
      label *quality*, doesn't grow the training set (vague claims were never
      dropped, just silently defaulted — corrects an earlier misread of the
      code). Low coverage on this corpus (1900s prose rarely uses explicit
      time-language); explicit-year detection ("outlook for 1947") would raise
      coverage further and is not yet built.
- [x] `handgrade_newspapers/` — blind, episode-stratified 80-claim sample,
      grading instructions, and `kappa.py` / `eval_vs_consensus.py` computing
      Cohen's kappa. Sampled from `claims_raw.csv` rather than the LLM's
      output, so it can catch the LLM *wrongly rejecting* a real prediction.
- [x] Repo-root `.gitignore` added (there was none). `claims_graded.csv` and
      `validation_sample.csv` untracked via `git rm --cached` (files remain on
      disk; commit the finished corpus deliberately with `git add -f`).
- [x] **`tier2_analysis.py` and `model.py` rerun against the new `gpt-4.1`
      corpus** (2026-07-16, 09:58-10:01, after the `claims_scored.csv`
      regrade at 09:52). Fresh: `fig_epu_vs_accuracy.png`, `fig_geography.png`,
      `fig_three_way_benchmark.png`, `results_by_region.csv`,
      `model_predictions.csv`, `fig_model_importances.png`.

## Not done / next up

- [ ] **`model_figures.py`'s poster suite is still stale (pre-regrade,
      2026-07-10).** `fig_model_roc.png`, `fig_model_calibration.png`,
      `fig_model_loeo.png`, `fig_model_proba_dist.png` were not regenerated
      when `model.py` was rerun on 2026-07-16 (only `model.py`'s own
      importances figure was) — rerun `model_figures.py` against the current
      `claims_scored.csv` before trusting/posting those four.
- [ ] **Unimpeachable validation (optional, recommended before the poster).**
      The 0.89/0.90 kappa is real but the gold it's measured against was
      reconciled *while looking at* prior models' disagreements (documented,
      rubric-driven, not curve-fit — see the Done entry above) — grading a
      fresh ~40-80 claims blind under the final rubric would remove that one
      remaining asterisk entirely.
- [ ] Spec Step 4b: composite 0-1 claim score (accuracy + punctuality +
      specificity). Horizon/punctuality now exists (`resolve_horizon`);
      `specificity` has no field at all and no scoring formula.
- [ ] Spec Steps 3 and 5: polls. No poll data exists anywhere in the repo
      (Gallup pre-1960 microdata isn't freely downloadable — Livingston/UMCSENT
      are already wired up as a substitute; needs a decision, see below).
- [ ] Spec's second model (predict economic *state* from press, not "was this
      claim right"). Different unit of observation (time period, not claim);
      at episode level only 10 rows, needs a month-level corpus expansion.
- [ ] **NYT extension toward 2010 — design pivoted to ONE unified corpus,
      in progress (2026-07-16).** Original plan ran NYT articles through
      `election_arm/extract_predictions.py`'s own extraction schema (switched
      luna → `gpt-4.1` earlier today) into a separate `scored_economy.csv`.
      Reconsidered: that schema was never kappa-validated, so a second
      pipeline would produce results not directly comparable to the main
      corpus. New plan — merge NYT raw articles straight into
      `claims_raw.csv`'s own schema and run them through the *existing*,
      validated `grade_claims.py` (κ=0.89/0.90) → `score_claims.py` (NBER+FRED)
      pipeline instead, for one 1905-2010 corpus under one rubric.
      `election_arm/extract_predictions.py`'s gpt-4.1 switch is now unused for
      this goal (left in place as standalone election_arm infrastructure, not
      wired into the merge).
      New: `append_nyt_claims.py` — idempotent converter, dedupes on
      `page_url`, maps the 9 post-1963 windows to `claims_raw.csv`-style
      episode names (e.g. `oil_1973` → "1973 Oil Shock"), continues the
      claim_id sequence. Not run yet (waiting on a fuller NYT download first).
      Also fixed today: `download_nyt.py`'s `search_phrase()` trusted NYT's
      `meta.hits` field to decide whether to paginate, and that field
      misreports 0 on most of these phrase/window queries even when real
      articles come back — was truncating every phrase at page 1 (≤10
      articles) regardless of true corpus depth. Now pages on actual page
      size (`len(docs) < 10`) instead. A full re-download with the fix is
      running now to replace the shallow 135-article 2026-07-10 corpus before
      merging — one free NYT key is still enough (the shallow run was only 83
      calls total against 500/day, 5/min caps; the deeper run will cost more
      calls but the same order of magnitude, not a multi-key problem).
      Next once the download finishes: run `append_nyt_claims.py`, then
      `grade_claims.py --model gpt-4.1 --base-url https://api.openai.com/v1`
      (resumes, only grades the new rows), then `score_claims.py` to
      regenerate the unified `claims_scored.csv` and figures.
- [ ] Decide the **DC financial-center coding** in `tier2_analysis.py` (see
      Known limits) before trusting `fig_geography.png`.

## Known limits / needs improvement

- **Validation integrity — the kappa must be MEASURED, not manufactured.**
  Never edit human labels to agree with an LLM, and never fabricate coder
  disagreement to fake independence — either makes kappa measure a system
  against itself. Legitimate levers only: clearer rubric, objective removal of
  ungradeable rows, a stronger/better-suited grader model, more validation
  claims, a tune/report split. This discipline is *why* the 6-model bake-off
  and the final 0.89/0.90 are trustworthy — don't relax it going forward.
- **`voice` is an unvalidated feature.** The LLM labels it (5-bucket taxonomy:
  journalist/expert/official/layperson/unclear) and `score_claims.py`'s
  hit-rate-by-voice breakdown uses it (experts 62.3%, officials 46.0%), but it
  was never hand-graded, so there is no kappa for it. Disclose this if the
  voice breakdown goes on the poster.
- **Sampling skew**: 378 of 1,324 claims (29%) are from Washington DC, 261
  from the Evening Star alone. `tier2_analysis.py` codes DC as a "financial
  center" — that bucket is mostly one government town's paper, likely drives
  `fig_geography.png` (itself still stale, see Not done). Needs a coding
  decision, then a rerun.
- One search term (`"business outlook"`) produced 606 of 1,324 claims (46%).
  `search_log.csv` shows the corpus was sampled rather than cherry-picked, but
  says nothing about what the 12 search terms *missed* (no recall audit done).
- The retrospective-vs-prediction boundary was the main leakage risk flagged
  during rubric design; it's now an explicit rubric rule (retrospectives →
  not a prediction) and was part of what the reconciliation process fixed.
  Spot-check a sample of the final corpus's `is_prediction=yes` rows against
  this rule before trusting the state-prediction model (Not done, above).
- `JeremysShit/` has no venv of its own (`test_offline.py` needs pandas; run
  it with `bill_arm/.venv/bin/python`, which also has `openpyxl` now).
- Five Groq API keys from the abandoned Llama path are still live and were
  pasted in plaintext into this chat session — see Needs-to-be-done-by-you.

## Needs to be done by you (not automatable / requires a decision)

- [ ] **Revoke all 5 Groq API keys** used during the abandoned Llama-70b path
      — all were pasted in plaintext in chat and must be treated as
      compromised regardless of the project having moved to `gpt-4.1`.
- [ ] Also rotate/revoke the OpenAI key used for the `gpt-4.1` runs once the
      project's OpenAI usage is done for this phase — same reasoning.
- [ ] Decide whether DC counts as a financial center or a government town in
      `tier2_analysis.py`, and re-run it. Defend the choice in methods.
- [ ] Decide what "polls" means for spec Steps 3/5: Gallup pre-1960 microdata
      isn't freely downloadable. Livingston (economists, 1946+) and UMCSENT
      (households, 1952+) are already wired up — acceptable substitute?
- [ ] Preregister the `no_change` band choices in `score_claims.py` (CPI
      ±1.5%, INDPRO ±2%, UNRATE ±0.3pt) *before* seeing results change, then
      run the sensitivity analysis the docstring already flags.
- [ ] Hand-code publisher metadata (circulation, political lean, urban/rural)
      for the top ~30 publishers from the LOC US Newspaper Directory — the
      only human task that adds *features* to the model rather than auditing
      it (enables "did partisan papers predict worse?").
- [ ] If pursuing the unimpeachable-validation option above: grade a fresh
      blind sample (Vincent + Jeremy, same process as `handgrade_newspapers/`).
