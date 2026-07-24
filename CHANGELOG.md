# Project Log — RISE research arms

Living status doc. Not a code diff log — see `git log` for that. This tracks
goals, methods, what's done, what's next, and what needs a human. Update this
alongside code changes, don't let it drift.

Two arms are tracked here: **bill_arm** (primary) and the **economy arm**
(`JeremysShit/`). The election arm has its own README and is not tracked here
(status log: `JeremysShit/election_arm/README.md` — most recent entry
2026-07-22, a `test_offline.py` mock fix).

---

# bill_arm — Congressional Bill Factor Analysis (deprioritized 2026-07-17)

**Pivoted away from bill-passage prediction and the NYT press-coverage
experiment on 2026-07-17.** Reasons: (1) any prediction model trained on
Congress-108-118 political/economic factors would mostly reflect the 118th
Congress's specific climate and wasn't judged to generalize meaningfully;
(2) the press-coverage side (does NYT/press predict a bill's fate) turned
out to be far too sparse to power even at full-Congress scale — a
1,000-bill random sample found only 6 confirmed press hits (0.60%, 95% CI
0.22%-1.30%), implying ~29,000-275,000 bills searched to reach the
project's 300-bill decision-gate threshold, well beyond one full Congress
(16,565 bills). Explored several alternative press sources (The Hill/
Politico/Roll Call via ProQuest TDM Studio, NewsData.io, GDELT, Bing News)
before deciding to drop the press angle rather than keep chasing volume.

**What was deleted**: `model.py` (the passage-prediction Model 1/Model 2
harness), `link_coverage.py`, `extract_press.py`, `coverage_report.py`,
`join_dataset.py` (the whole NYT press pipeline), and their data
(`data/press_raw/`, `data/press_labeled/`, `data/modeling.csv`,
`data/press_features_*.csv`, `data/press_search_log.csv`,
`data/pre_window_fix_backup/` — ~20MB) and API keys (`bill_arm/.env`, all
entries were pipeline-specific).

**What was kept** (still useful for factor analysis, not prediction):
bill ingestion (`download_bills.py`, `download_bills_bulk.py`), the
structural feature table (`build_features.py`, `build_macro_features.py`),
and a new `factor_analysis.py` — model.py's fitting/importance/calibration
functions minus the passage-prediction framing and press-experiment code,
still imported by `make_figures.py`, `model_figures.py`,
`_ablation_figdata.py`, `ablation_macro.py` to produce factor-importance
figures. `test_offline.py` trimmed to match (removed sections testing the
deleted modules; kept ingestion/feature/fitting-utility tests). All
remaining tests pass; `make_figures.py`'s import chain verified working
against real `data/features.csv` post-refactor. README.md rewritten to
match the new scope; `requirements.txt` dropped `openai`/`python-dotenv`
(nothing left imports them); root `CLAUDE.md` updated to stop describing
deleted commands/architecture and to drop bill_arm as the assumed default
scope for unlabeled tasks.

**Findings worth keeping from the structural/macro factor analysis** (see
git history for full CHANGELOG detail pre-pivot):
- Macro/economic climate features do NOT help predict which bill passes,
  and HURT PR-AUC when added (`ablation_macro.py`, test on Congresses
  117-118, re-run 2026-07-18 after the hyperparameter tuning below:
  logistic -0.0785 **sig**, gradient-boosting -0.0727 sig — both models
  significant now; pre-tuning this was logistic -0.019 ns, GB -0.053 sig,
  so tuning made this finding STRONGER, not weaker). Bill passage is driven
  by structural/political factors, not the economy, at the bill level.
- BUT at the time-period level (quarter of introduction, 88 quarters
  2003-2024), overall passage *rate* shows a soft positive correlation with
  a good economy (`time_level_economy.py`: consumer sentiment +0.34*, GDP
  growth +0.29*, unemployment -0.24*) — not conclusive (recession-vs-
  expansion cut itself is null, p=0.61; effective n ~11 political regimes),
  but a real bill-level/time-level contrast: the economy doesn't pick WHICH
  bill passes, but may gently move overall legislative throughput.
- Figures for both live in `figures/` (`fig_macro_ablation.png`,
  `fig_timelevel_economy.png`) alongside the earlier structural-baseline
  diagnostics (rate-by-Congress, cosponsors, policy area, PR curves,
  calibration, feature importances).

- [x] **`predict_bill.py` — packaged `factor_analysis.py` as an actual
      input->output predictor**, mirroring `JeremysShit/predict_claim.py`
      for this arm (2026-07-18, user asked for more predictive-modeling
      work; this was the lowest-risk option since the underlying model
      already works, no new feature hunting). CLI + `--interactive` mode:
      give it a hypothetical new bill's chamber/type/sponsor
      party+state/policy area/committee/cosponsors, get back a predicted
      probability it becomes law. Fit on ALL 128,778 bills (not the
      held-out-Congress split — that split is for honestly measuring
      generalization; a deployed predictor should use every historical
      example). Sanity-checked before trusting it: adding bipartisan
      support + sponsor-in-majority + a companion bill + more cosponsors
      moves predicted probability up substantially (0.44%->1.63% logistic,
      0.68%->4.99% gradient boosting) — correct direction, matches the
      known real factors from the earlier ablation work, not random output.
      Documented limit: ~3.2% base rate means predictions are almost always
      low in absolute terms — useful for relative ranking between bills,
      not as a single "will/won't pass" number; also reflects the
      108th-118th Congresses' specific era, not a universal law of
      Congress. `python -m py_compile` clean; bill_arm's `test_offline.py`
      (15 sections) still all pass.

- [x] **Recall diagnostic (2026-07-18, user asked to try other improvement
      methods beyond tuning).** `predict_bill.py`'s gradient-boosting model
      only catches 33% of bills that actually become law (Congress 118 held
      out). Checked WHY: missed bills (184 of 274) have a real, explainable
      pattern vs. caught ones — short titles (median 6 words vs. 27), few
      cosponsors (median 1 vs. 6), less bipartisan (37.5% vs. 58.9%), and
      cluster in specific committees (Homeland Security/Governmental
      Affairs, Natural Resources, Judiciary, Veterans' Affairs) — i.e.
      small, narrow, technical bills (land transfers, program extensions,
      minor corrections) that quietly pass without the "big legislation"
      signals (broad cosponsorship, bipartisan coalition, long descriptive
      title) the model has learned to rely on.
      Checked whether this is just a bad threshold choice, not a real
      ceiling: the precision-recall tradeoff is genuinely steep (recall 0.3
      -> precision 0.582; recall 0.8 -> precision 0.068, i.e. 93% false
      alarms to catch 80% of eventual laws) — confirms the missed bills
      really do carry weak signal in this feature set, not that a
      different cutoff would fix it for free. **Conclusion: this looks
      like a real ceiling given current features, not a tunable gap** —
      catching quiet technical bills would likely need information this
      dataset doesn't have (committee-specific dynamics, end-of-session
      omnibus bundling, whip counts), not more hyperparameter search.
      Reporting this honestly rather than forcing a fix.

- [x] **`factor_analysis.py` hyperparameter tuning — real, adopted** (2026-07-18,
      direct analog to the economy arm's `LOGIT_C` win, but checked rather
      than assumed since bill_arm's feature-to-sample ratio is much
      healthier). `GridSearchCV`/`cross_val_score` with `GroupKFold(5)`
      grouped by Congress, `scoring="average_precision"` (PR-AUC):
      - Logistic `C`: old default 1.0 -> 0.3017 CV PR-AUC; tuned **0.1 ->
        0.3136**. Unlike the economy arm, NOT monotonic all the way down
        (C=0.01 was worse at 0.2915) -- a real interior optimum here, not
        "more regularization is always better."
      - XGBoost: old `n_estimators=300, max_depth=4, learning_rate=0.05`
        -> mean 0.3775 (per-fold [0.353, 0.339, 0.357, 0.426, 0.414});
        tuned **`n_estimators=500, max_depth=6, learning_rate=0.1`
        -> mean 0.3921** (per-fold [0.371, 0.352, 0.377, 0.440, 0.421]) --
        beats the old params in EVERY fold, not just on average, which is
        why this was trusted (a mean-only comparison can hide a lucky
        fold).
      Both adopted as documented constants (`LOGIT_C`, `XGB_PARAMS`) in
      `factor_analysis.py`, wired into `fit_and_score()` and
      `predict_bill.py`'s duplicate model-construction code.
      **Confirmed on the actual held-out-Congress-118 evaluation** (not
      just the tuning CV): PR-AUC 0.280->**0.336** (logistic),
      0.393->**0.400** (gradient boosting); ROC-AUC 0.856->0.880 and
      0.897->0.900; recall on "became law" improved for BOTH models
      (29%->38% logistic, 33%->38% GB) without hurting precision — a
      partial, real answer to the recall gap diagnosed above, though the
      broader "quiet technical bills" ceiling described there still holds.
      `python -m py_compile` clean; all 15 `test_offline.py` sections pass.

- [ ] **IN PROGRESS: `sponsor_is_committee_chair`, second attempt after the
      first stalled** (2026-07-18). First attempt (unscoped: all 70
      chamber/committee pairs, all 11 Congresses) ran ~9h with no output and
      no clean exit — `TaskStop` couldn't even find it, so it had already
      died/orphaned rather than being genuinely slow. Relaunched with a hard
      scope cut: only the **top-10 highest-bill-volume (chamber, committee)
      pairs** (~60% of all 128,778 bills — Ways and Means, Judiciary x2
      (House+Senate), Energy and Commerce, Finance, HELP, Education and the
      Workforce, Natural Resources, Transportation and Infrastructure,
      Financial Services), a ~3-4-fetch budget per committee, explicit
      instruction to record gaps and move on rather than perfect one
      committee. **This run completed the data-building half cleanly**:
      `data/committee_chairs.csv` (110 rows = 10 committees x 11 Congresses,
      no gaps, sourced from Wikipedia chair-history tables, one source URL
      per row) and `build_features.py` now joins it in
      (`add_committee_chair_feature()`, matches on sponsor's last name,
      case-insensitive; also fixed a pre-existing name inconsistency where
      `download_bills_bulk.py`'s XML parser drops "the" from "Education and
      the Workforce Committee" for the 118th Congress only). Committees
      outside the top 10 (~40% of bills) get 0 by design, not a bug — stated
      as a limitation, not hidden.
      **Both agent attempts at the validation step stalled** (2026-07-19):
      the second agent finished the data-build cleanly but its own
      backgrounded CV check went quiet for 3+ hours with no process running
      and no report — same silent-death pattern as the first agent's 9h
      hang, just later in the pipeline. Ran `_chair_feature_cv_check.py`
      directly instead of delegating further. **`GroupKFold(5)` result**:
      without feature mean PR-AUC 0.3926 (per-fold [0.3714, 0.3566, 0.3766,
      0.4373, 0.4214]); with feature mean **0.3953** (per-fold [0.3737,
      0.3548, 0.3830, 0.4429, 0.4220]) — **+0.0027, positive in 4/5 folds**
      (one small loss in fold 2). Much smaller than this session's real,
      adopted wins (`LOGIT_C` +0.0119, XGBoost tuning +0.0146, both 5/5
      folds) and comparable in size to the economy arm's rejected GB-tuning
      gain (+0.005, called noise 2026-07-18) — looks marginal, not
      confidently real. Confirmed with `bootstrap_pr_auc_delta()` (the same
      utility already in `factor_analysis.py`, used to validate other
      decisions) on pooled out-of-fold predictions: **mean delta +0.00224,
      95% CI [-0.00052, +0.00510] — crosses zero.**
      **VERDICT: REJECTED, not adopted.** Consistent with this session's
      standard (same call made on the economy arm's GB tuning): a gain
      whose CI/noise band includes zero doesn't clear the bar, regardless
      of which direction the point estimate points. `sponsor_is_committee_chair`
      stays OUT of `factor_analysis.py`'s `NUMS` — still computed by
      `build_features.py` and present in `data/features.csv` for future
      exploration (same "computed but not load-bearing" pattern as
      `political_lean`/`urban_rural`/`local_disagreement`), just not part
      of the reported model. Both scratch validation scripts
      (`_chair_feature_cv_check.py`, `_chair_bootstrap_check.py`) deleted
      per their own "delete after use" header, consistent with how prior
      scratch checks in this project were cleaned up. `data/committee_chairs.csv`
      (110 rows, top-10 committees x 108th-118th Congress, Wikipedia-sourced)
      stays — real, cited data, useful if this angle is revisited with a
      fuller committee list later.

## Known limits
- `recession_flag` (NBER/USREC) is announced 6-21 months after the fact
  historically — backward-looking, not real-time; flag if it shows up as
  an important feature.
- Any factor-analysis finding here reflects the 108th-118th Congresses'
  specific political era, not a universal claim about bill passage.

## Needs to be done by you

- Nothing pressing. If a future direction reopens the political/economic
  factor angle (e.g. combining with the economy arm's `political_climate.csv`
  work), start from `factor_analysis.py` + `build_features.py`.


# Economy arm (`JeremysShit/`) — Did newspapers see economic crises coming?

## Goals

1. Score 1900s-era newspaper economic predictions against what actually
   happened (NBER/FRED), across 7 crisis windows and 3 calm control windows.
2. Benchmark newspapers against professional economists (Livingston Survey)
   and households (Michigan sentiment).
3. Identify which factors predicted a claim being *right* (publisher, voice,
   confidence, policy uncertainty at time of printing).

## Done so far

- [x] **Three new analyses built to turn the optimism gap into a thesis
      (2026-07-22, user: "expand the project," ideas #1/#2/#3). All standalone
      scripts reading `claims_scored.csv`, same convention as
      `disagreement_severity.py`; 20 new offline tests, suite now 54/54.**
      - **`regret_scoring.py` — the standout, well-powered positive result.**
        Asymmetric error scoring: a miss is tagged by the direction the PAPER
        predicted (told readers "improve," it didn't = optimistic error).
        Assumption-free decomposition: **66.1% of CRISIS misses were optimistic
        vs only 31.8% of CONTROL misses** — papers erred optimistic when it
        mattered, not in calm times (clean crisis-vs-control placebo split).
        Overall optimistic-vs-pessimistic error split is lopsided,
        **binomial p=4.47e-10**. **1929 Crash: 155 optimistic errors vs 1
        pessimistic** (hit rate 0.103) — the archetype; 1920 Depression 99 vs
        12. Severity-weighted regret (INDPRO peak-to-trough magnitude,
        min-max normalized, pre-1919 excluded per the disagreement_severity
        cross-scale rule) concentrates in the severest crises. The qualitative
        asymmetry is **weight-invariant** (share optimistic doesn't move with
        the w_opt:w_pess ratio; only the regret magnitude does — swept 1:1..10:1).
        Outputs `regret_by_episode.csv`, `figures/fig_regret.png`.
      - **`hedging_lexicon.py` — an HONEST CORRECTION that OVERTURNS the prior
        overconfidence finding.** Replaces the LLM `confidence` label (this
        project's weakest grade, kappa=0.19) with an objective, reproducible
        Hyland-2005 hedge/booster lexical score (no rater, so no kappa needed).
        On the objective measure, **assertive claims are NOT less accurate**
        (assertive 0.516 vs hedged 0.485) — so the earlier "assertive < hedged
        -> overconfidence" result (which rested on the 0.19-kappa label) **does
        not survive**. The objective class barely agrees with the LLM confidence
        label (**kappa=0.27, 62.5%**), and continuous hedge_score vs hit is
        r=-0.012 (null). Read: the overconfidence story was a label artifact;
        confident and hedged papers were about equally accurate. Report the
        objective measure and retire the LLM-confidence calibration claim.
        Outputs `claims_hedging.csv`, `figures/fig_hedging.png`.
      - **`optimism_timeline.py` — monthly within-episode optimism index; the
        peak-alignment idea works but is POWER-LIMITED by event-windowed
        sampling, disclosed honestly.** net_optimism per (episode, month) among
        improve/worsen predictions; peak = INDPRO argmax in the claim span (NBER
        peak pre-1919); test = slope of optimism vs months-to-peak (negative =
        foresight, flat/positive = complacency). Robust sub-results: within
        crisis windows papers were strongly net optimistic (**+0.475, ~1.5:1
        improve:worsen**), statistically indistinguishable from calm controls
        (+0.446) — persistent optimism regardless of regime; 1929 = +1.0 net
        optimism right before the peak. BUT claims cluster AT/after each peak
        (the scrape was windowed on the crisis), so only 4 episodes have >=2
        run-up months: **pooled run-up slope +0.141/mo, 95% CI [-0.10,+1.15]
        crosses zero** — directionally complacency, not significant. The
        error-direction result in `regret_scoring.py` delivers the same
        "optimism into crises" story far more convincingly; lead with that,
        treat the timeline as supporting + a documented sampling limitation.
        Outputs `optimism_by_month.csv`, `figures/fig_optimism_timeline.png`.
      Net for the poster: **#2 is a new centerpiece** (asymmetric, expensive
      optimism, huge significance, clean placebo); **#3 is a credibility win**
      (we checked our own weak finding with an objective measure and it didn't
      hold); **#1 is supporting** (persistent optimism) with an honest
      power caveat. None required new scraping, LLM spend, or ProQuest.

- [x] **SPF professional-forecaster benchmark added (2026-07-22, idea #4).**
      `spf_benchmark.py` scores the Survey of Professional Forecasters -- which
      disagreement.py already frames itself against but the project never
      actually scored -- on the IDENTICAL ground truth the newspapers use
      (score_claims.realized_direction, INDPRO/NBER, 12-mo window), over the
      overlapping 1968+ era. Data: Philly Fed median_rgdp_growth.xlsx (free,
      quarterly, 1968Q4-2026), auto-downloaded + cached. SPF's ~1-year call =
      mean of DRGDP3..DRGDP6 (the four quarters AFTER the survey quarter --
      leakage-safe), banded to improve/worsen/no_change (--band, default 1.0
      annualized %). Needed a robust .xlsx loader: Philly Fed files carry a
      date-only docProps `modified` field that crashes openpyxl (TypeError:
      expected datetime); `read_xlsx_robust()` strips those elements in-memory
      before pandas reads -- reusable for any Philly Fed file (incl. Livingston).
      **CLEAN STANDALONE RESULT: SPF directional hit 54.1% (95% CI [47.6,60.6],
      n=231), prediction mix improve 94% / no_change 6% / WORSEN 0%** -- the
      professionals essentially NEVER forecast a contraction a year out
      (Loungani 2001, "failure to predict recessions"); they score decently by
      siding with the economy's usual upward drift, not by calling turning
      points. Nice convergence for the poster: **SPF 54.1% ~= Livingston 54.4%
      ~= Michigan ~55%** -- three independent benchmarks, different eras, all
      hover just above a coin flip on turning-point-inclusive directional calls.
      **HONEST CONFOUND, disclosed in the output and here:** the raw press-vs-SPF
      hit-rate gap (newspapers 30.8% vs SPF 54.1%, 1968+) is NOT a clean skill
      comparison -- the post-1968 newspaper corpus is NYT crisis-windowed, so it
      over-samples downturns and skews pessimistic (worsen share 58%), while SPF
      is a continuous quarterly survey. The sampling-robust contrast is the
      PREDICTION MIX (SPF ~0% worsen vs the crisis-sampled press), not the hit
      rate -- same NYT under-sampling limit flagged elsewhere. 8 new offline
      tests (direction_label, survey_date, score_spf with an injected realized
      fn so no FRED needed), suite now 62/62. Outputs `spf_scored.csv`,
      `figures/fig_spf_benchmark.png`.

- [x] **Narrative Economics (Shiller) scaffold built; LEXICAL preview run,
      authoritative LLM pass NOT yet run (2026-07-22, idea #5, the stretch
      goal).** `narratives.py` codes each claim into one of six perennial
      economic narratives (new_era / sound_fundamentals / temporary_setback /
      panic_fear / recovery_normalcy / none), connecting the optimism-gap
      result to Shiller's *Narrative Economics* (2019) over a far longer span
      than the SPF-era narrative literature can reach. Two passes, same
      "objective-first, then LLM" discipline as hedging->grade_claims:
      (1) `classify_narrative` -- deterministic lexical screen, runs with no
      API, offline-tested (8 new tests, suite now 70/70); (2)
      `grade_narratives_llm` (--llm) -- the AUTHORITATIVE gpt-4.1 pass reusing
      grade_claims.call_llm, flag-gated, NOT run (needs OPENAI_API_KEY + a small
      spend + a ~40-80-claim human kappa check before it can be trusted, same
      bar as every other LLM field here).
      **The lexical preview is deliberately reported as INCONCLUSIVE**: 78% of
      claims fall to "none" and it captures only 4% complacent narratives in the
      1929 Crash window -- i.e. a keyword screen cannot see narrative FRAMING
      (1929's optimism is not phrased with literal "new era" strings), which is
      exactly why the LLM pass is required, not optional. The preview's
      "complacent 58.4% vs other 48.3% hit" is on tiny n (new_era n=4) and must
      not be reported. Outputs `claims_narratives.csv` (lexical, preview only),
      `figures/fig_narratives.png`. **Next step: run --batch gpt-4.1 over the
      full corpus + kappa-validate; this is the only one of ideas #1-#5 that
      still needs API spend.**

- [x] **Partisan-publisher analysis (2026-07-22, user: put politics back in).**
      `partisan_analysis.py` -- does a paper's POLITICS colour its economic
      forecasts? Uses publisher_metadata.csv (hand-researched lean) +
      political_climate.csv (president's party by year), both already on disk.
      9 new offline tests, suite now 79/79. Outputs `partisan_by_lean.csv`,
      `figures/fig_partisan.png`.
      **REAL DATA-QUALITY BUG FOUND: model.py's publisher-lean join is broken.**
      claims_scored.csv publisher names carry a "(location) dates" suffix that
      publisher_metadata.csv's clean names lack, so model.py's exact-string
      `.map()` matched only **258 of 1,628** predictions -- `political_lean` was
      "unknown" for 84% of rows. This partly explains why the political features
      read as null in model.py (mostly MISSING, not merely weak). This script's
      `short_name()` strips the suffix first, recovering **861 leaned claims**.
      FOLLOW-UP worth doing: re-run model.py with the fixed join before treating
      "political features don't predict claim accuracy" as settled.
      FINDINGS (with the fixed join):
      - A (by lean): Socialist/left papers were the LEAST optimistic (net
        optimism +0.30, e.g. the Milwaukee Leader), vs Republican +0.64,
        Democratic +0.56, Independent +0.32 (~NYT+Evening Star). Directionally
        sensible (a Socialist daily is not cheerleading capitalism) but
        era-confounded -- each small partisan paper wrote in a narrow window.
      - B (partisan alignment -- the sharp test: rosier under your OWN party's
        president?): **NULL.** aligned net optimism +0.593 vs opposed +0.577
        (diff +0.016), share-improve 0.80 vs 0.79, **Fisher exact p=1.000**. No
        evidence papers forecast a better economy under their own party. (Aligned
        hit rate looked higher, 0.64 vs 0.51, but n is tiny and era-confounded --
        not claimed.)
      Honest read: a clean, reportable NULL on the classic partisan-perceptual-
      bias hypothesis (Bartels 2002 / Gerber-Huber 2010) at newspaper level, plus
      a genuine bug fix. Politics is now IN the paper as a tested null, not absent.

- [x] **Extended abstract drafted for the poster (2026-07-22)** --
      `JeremysShit/EXTENDED_ABSTRACT.md`, pulling the spine together (optimism
      gap / asymmetric regret as centerpiece, the three-benchmark
      turning-point-blindness convergence, the hedging correction, narratives
      as in-progress). Numbers cited are the current committed ones; author
      line + a couple of framing choices are marked as placeholders to confirm.

- [x] **Pre-submission review found the Result-1 headline p-value was
      anti-conservative; fixed with an episode-level test (2026-07-22).** The
      claim-level binomial test (p=4.5e-10 on 66% vs 32% optimistic-miss share,
      crisis vs control) treats every claim as an independent draw, which they
      aren't -- claims inside one episode share wire-service copy and one
      macro reality. Added an episode-level check to `regret_scoring.py`
      (Mann-Whitney U, one optimistic-share number per episode, 13 crisis vs 6
      control): median 45.9% vs 25.6%, **p=0.15 -- does NOT clear significance**
      at n=19 episodes. Directionally consistent, not yet confirmed; the
      asymmetry claim in the abstract is now hedged accordingly (still leads
      with 1929 as a real, large single-episode result). Also added a Wilson
      95% CI column to `results_by_episode.csv` (`score_claims.py`) since
      several episodes have n<20 (1987 n=14, 1990 n=10, 1995 n=15, 2008 n=17).
      Two smaller citation fixes: abstract said "six-model" bake-off, table has
      seven; hedging_lexicon.py/abstract cited the original Llama-3.3 run's
      confidence kappa (0.19) instead of gpt-4.1's own (0.17) -- the model
      actually used on the scored corpus. **Next step: more crisis episodes
      (post-1963 ProQuest expansion) before the episode-level test can reach
      the power to confirm or reject Result 1.**

- [x] **Report scope cleanup (2026-07-22, user: "remove anything not worth
      reporting").** bill_arm's factor analysis confirmed OUT of scope for
      this report -- unrelated dataset/question (Congressional bills vs.
      newspaper economic predictions), no shared narrative, don't cite it here.
      Within the economy arm: deleted 11 superseded/backup CSVs no longer read
      by anything (`claims_*_pre_dedup_backup.csv`, `claims_*_pre_expansion_backup.csv`,
      `claims_graded_expanded.csv`, `claims_raw_expanded.csv`,
      `claims_graded_newprompt.csv`, `claims_graded_v2.csv`,
      `search_log_baseline_2026-07-16.csv`) -- all git-tracked, recoverable
      from history if ever needed. Kept (user's call, not a unilateral one):
      the confirmed-null analyses (disagreement, geography, political climate)
      stay in the repo with one Limitations-section line disclosing they were
      tested and ruled out, rather than being deleted or silently dropped --
      the point being the shipped feature set reads as filtered, not
      incomplete. Added the `model_interactions.py`/`validate_interactions.py`
      finding (7/8 pairs significant, p=0.005) to `EXTENDED_ABSTRACT.md` as
      Result 4 (was previously validated but never written into the abstract);
      Narrative Economics renumbered to Result 5. Also kept, not deleted: the
      seven `claims_graded_val80_*.csv` bake-off files -- they're the raw
      evidence behind the headline kappa=0.89/0.90 claim, not clutter.

(`narratives_kappa_sample.py` and the full-corpus `narratives.py --batch` run
it depends on are both done -- see the "Done" entries further down, dated
2026-07-22, for the finished state: batch pass completed, 80-claim blind
sample drawn. This still blocks Result 5 in `EXTENDED_ABSTRACT.md` from being
reportable until a human fills in `human_narrative` and kappa is computed.)

- [x] **Result 1's episode-level test upgraded from one nonparametric check to
      three converging methods (2026-07-22, user: "give our project substantial
      claims... actual substance").** Mann-Whitney on episode medians (added
      earlier today) is valid but low-power -- it collapses each episode to one
      number regardless of how many claims it has, so a 174-miss episode
      (1929) counts the same as a 3-miss one. Added to `regret_scoring.py`:
      (1) `cluster_permutation_test` -- an EXACT randomization test (episode
      is the resampling unit, so it's valid regardless of cluster count, no
      asymptotics; the test STATISTIC is pooled claim-level data, so it keeps
      the power Mann-Whitney throws away). All C(19,6)=27,132 possible
      crisis/control label assignments enumerated exactly, no Monte Carlo
      approximation needed at this scale. (2) `cluster_bootstrap_ci` -- a
      percentile bootstrap (resample episodes with replacement within each
      kind, 10,000 draws) on the GAP'S SIZE, since "not significant" and "no
      effect" are different claims and a p-value alone doesn't convey
      magnitude.
      **Real result, run against the actual corpus:** exact cluster-permutation
      gap +34.3pp (crisis 66.1% vs control 31.8% optimistic-miss share), exact
      one-sided p=0.161 -- close to but consistent with the earlier
      Mann-Whitney's p=0.146, not a fluke of test choice. Bootstrap 95% CI on
      the gap: [-1.1pp, +63.9pp] -- barely touches zero; most of the
      distribution is a large, real-looking effect. **Net: three different
      valid tests now converge on the same honest conclusion (directionally
      large, not yet significant at n=19 episodes), which is itself the useful
      finding -- the non-significance is a genuine power limit, not an
      artifact of which test was picked.** `EXTENDED_ABSTRACT.md` Result 1 and
      its Limitations entry updated with the exact numbers. `test_offline.py`
      still 70/70 (no existing test touched this function).

- [x] **Checked EPU (policy uncertainty) as a possible added finding; declined
      to report it (2026-07-22).** `tier2_analysis.py` shows hit rate falling
      monotonically with EPU tercile (58.0% low / 44.6% mid / 43.9% high,
      point-biserial r=-0.049) -- looked promising, but claims share the same
      episode-clustering problem just fixed for Result 1 (EPU moves slowly, so
      claims in one episode mostly share a tercile), and `model.py` already
      found EPU non-dominant once `direction` is in the model -- likely just
      re-detecting the crisis/control split, not new information. Not added to
      the abstract without the same cluster-aware validation Result 1 got.
      Side effect of running the script: `results_by_region.csv` and 4 figures
      (`fig_epu_vs_accuracy`, `fig_geography`, `fig_regret`,
      `fig_three_way_benchmark`) were STALE (roughly half the current claim
      count -- last generated before this session's corpus updates) and got
      refreshed to match the current `claims_scored.csv`; real numbers moved
      (e.g. midwest n 153->280, hit_rate 0.529->0.511) but no reported claim
      depended on the stale version.

- [ ] **Free power expansion for the optimism-asymmetry test: scraping 4 more
      LOC-era crisis episodes, IN PROGRESS (started 2026-07-22, user: "do the
      scrape and expansion").** Result 1's episode-level test is honest but
      underpowered at 13 crisis vs 6 control episodes (see entry above); LOC's
      free full-text archive (no API key, no cost) covers more NBER recessions
      than the 7 already scraped. Added 4 to `newspaper_scraper.py`'s
      `EPISODES`: **1910 Recession** (NBER 1910-01..1912-01, window
      1910-06-01..1911-03-31), **1913 Recession** (NBER 1913-01..1914-12,
      window 1913-06-01..1914-03-31, kept short of Aug 1914 on purpose --
      same reasoning the project already uses to skip the 1918-19 war
      recession, wartime claims are a different domain), **1923 Recession**
      (NBER 1923-05..1924-07, window 1923-09-01..1924-05-31), **1926
      Recession** (NBER 1926-10..1927-11, window 1927-01-01..1927-09-30).
      Search terms matched to the existing episodes' vocabulary for that era.
      **Two overwrite hazards found and fixed BEFORE running anything**:
      `newspaper_scraper.py`'s `run()` opens both `--out` and a hardcoded
      `search_log.csv` in write mode -- running the new episodes with
      defaults would have silently DESTROYED the entire existing 4,730-row
      corpus and the real search log. Added `--log-out` (defaults to the old
      hardcoded `search_log.csv` for backward compatibility) and ran this
      scrape with both `--out claims_raw_newloc.csv --log-out
      search_log_newloc.csv` -- isolated from the real files. Also had to set
      `SSL_CERT_FILE` to certifi's bundle to get `urllib` working in this
      venv at all (system default cert path was missing/stale).
      Wrote `append_loc_claims.py` (dedupe on `page_url`, continue the
      `claim_id` sequence, append-only, idempotent -- same pattern as
      `append_nyt_claims.py` but simpler since the source is already in
      `claims_raw.csv`'s exact schema) for the merge step once scraping
      finishes.
      **First background run did NOT actually cover all 4 episodes.**
      Checked after the fact: `claims_raw_newloc.csv` has 115 real claims,
      all of them **1910 Recession** -- zero for 1913/1923/1926. `search_log_
      newloc.csv` came back completely empty (not even a header row), which
      is the signature of the process being killed mid-run rather than those
      three episodes genuinely returning 0 hits (a real 0-hit term still
      writes a log row). So the run silently died partway through, not a
      clean completion.
      **Re-launched 2026-07-22, scoped to only the 3 missing episodes**
      (`--episodes "1913 Recession" "1923 Recession" "1926 Recession" --out
      claims_raw_newloc2.csv --log-out search_log_newloc2.csv`), backgrounded
      again, isolated from both the real corpus and the first run's output.
      **Completed cleanly this time** (verified `search_log_newloc2.csv` has
      real rows, not empty like the first run's failure signature): **351
      candidate claims from 245 pages** -- 1913 Recession 153, 1923 Recession
      125, 1926 Recession 73. Survived LOC's usual flakiness
      (`IncompleteRead`/`HTTP 520` on a few terms) via the scraper's existing
      retry logic, same as the original full-corpus run.
      **Combined with the first run's 115-claim 1910 Recession batch: 466
      candidate claims across all 4 new episodes, still NOT merged, graded,
      or rescored.** Remaining steps: `append_loc_claims.py` for both
      `claims_raw_newloc.csv` and `claims_raw_newloc2.csv`, `grade_claims.py
      --model gpt-4.1` (new rows only, small cost), `score_claims.py`, then
      rerun `regret_scoring.py`'s cluster-permutation test to see whether the
      added episodes move Result 1's p=0.161 -- report the real number
      either way, not just if it clears 0.05.

- [x] **Data cleaning pass (2026-07-19, user: "do some data cleaning...
      outliers... missing data").** Investigated before acting — this
      corpus doesn't have classic sensor-noise numeric outliers, so a
      generic "cap/remove outliers" pass would have been wrong here:
      - **`epu`/`year`/`months`: no anomalies.** EPU range 31.9-312.7 is
        real (the max is Sept 2001, i.e. the actual post-9/11 uncertainty
        spike — genuine signal, not an error, would be wrong to cap).
        `year` matches each episode's real date range. `months` is already
        deliberately snapped to {6,12,24} by `score_claims.py`'s
        `resolve_horizon()`, a pre-existing documented design choice, not a
        bug.
      - **`state` missingness (1,320/4,194 rows) is 100% structural, not
        random**: every missing row is NYT-sourced (the NYT API has no
        per-article state field the way LOC does — confirmed 0 missing
        among the 2,874 LOC-sourced rows). Already handled correctly by
        `build()`'s existing `fillna("")` -> `region="unknown"` default.
        Not imputed further — mapping "the new york times" to NY state
        would overstate the "financial center" narrative for what's a
        national paper, not a regional one; the existing "unknown" bucket
        is the more honest choice.
      - **Short quotes (225 rows < 40 chars) are mostly bare NYT headlines
        with no abstract/snippet attached — but SHORT DID NOT MEAN BAD.**
        220/225 were already correctly graded `is_prediction=no` (headline
        fragments like "BUSINESS DIGEST", "NEWS SUMMARY" aren't
        forecasts) and never reach `claims_scored.csv` in the first place
        (`score_claims.py` only scores `is_prediction=yes` rows). The other
        5 ("Greenspan Sees Chance Of Recession", "Budget Office Sees Rise
        in Deficit", etc.) are short but genuine, information-dense
        forecasts — deliberately did NOT drop these on a length heuristic,
        that would have deleted real signal for looking like an outlier.
      - **REAL finding, acted on: 69 exact duplicate `(episode, quote)`
        pairs** (out of 4,194) — not independent claims. Two patterns: (1)
        a Brownsville Herald promotional blurb ("Citrus groves are
        beginning to bloom...") reprinted verbatim across 9 daily editions
        in Feb 1930 — boosterism copy that slipped past the ad-junk regex;
        (2) the same NYT article double-indexed under two URL variants
        (http vs. https / canonical vs. legacy path) by the Article Search
        API. Removed (keep-first), re-ran `score_claims.py` and `model.py`
        on the cleaned corpus. Originals backed up first
        (`claims_{raw,graded,scored}_pre_dedup_backup.csv`).
        **`claims_raw.csv`/`claims_graded.csv`: 4,194 -> 4,125.
        `claims_scored.csv`: 1,644 -> 1,628.** Retune confirmed `LOGIT_C`
        unchanged (C=0.1 only +0.0026 better, within noise). Final numbers
        on the cleaned corpus: **LOEO accuracy 0.593 ± 0.209** (barely
        moved from 0.598); permutation test still significant for both
        models (logistic p=0.0099, GB p=0.0099 — beats every one of 100
        shuffles either way). Figures, `predict_claim.py` docstring/output,
        and this entry all updated to these final numbers.
- [x] **Corpus grown and merged in — claims_raw/graded/scored ~1.8x bigger**
      (2026-07-19, user: "remove any duplicates and merge and rerun"). The
      recall-audit-driven LOC rescrape (`claims_raw_expanded.csv`, 2,536
      candidate claims from the 7 under-recall crisis episodes — see bill_arm
      section's "levers" work, though this rescrape targeted this arm) and its
      OpenAI-batch grading (`claims_graded_expanded.csv`) were sitting
      unmerged into the main pipeline. Deduped against the existing corpus by
      exact `(page_url, quote)` match (1,550 of 2,536 were genuinely new —
      986 were re-fetches of pages already in `claims_raw.csv` from the
      original 30-page-cap run) and appended with fresh `claim_id`s
      continuing from the existing max, preserving every downstream
      `claim_id` reference. Originals backed up first
      (`claims_{raw,graded,scored}_pre_expansion_backup.csv`).
      **`claims_raw.csv`/`claims_graded.csv`: 2,644 -> 4,194 rows
      (is_prediction=yes: 929 -> 1,653). `claims_scored.csv` (after rerunning
      `score_claims.py`): 927 -> 1,644 scored claims, same 19 episodes.**
      `famous_calls.csv`, `publisher_leaderboard.csv`,
      `results_by_episode.csv`, `results_by_region.csv` all regenerated by
      that same `score_claims.py` run.
      **`model.py` rerun on the bigger corpus: LOEO accuracy 0.598 (was
      0.624 on the 843-claim corpus) — went DOWN.** Re-ran the same
      `LeaveOneGroupOut` grid search used to originally pick `LOGIT_C` (8
      values, 1.0 down to 0.005) to check whether 0.05 was now stale for
      ~1.8x more data — **it isn't: C=0.05 is still the best value on the
      new corpus too (0.5981, next best 0.2 at 0.5818), confirmed by a full
      regrid, not assumed.** So the 0.624 -> 0.598 drop is real, not a
      stale-tuning artifact — more data did not improve LOEO accuracy here.
      Per-episode breakdown shows 1929 Crash is the extreme outlier (0.05
      accuracy, i.e. almost perfectly anti-correlated with truth on that
      fold) dragging the mean down; worth a closer look before the poster
      if time allows, not investigated further yet.
      **Permutation test (100 shuffles) confirms both models still beat
      chance decisively despite the lower raw accuracy**: logistic
      regression 0.598 vs. null mean 0.504 (SD 0.021, max across 100
      shuffles 0.563) — p=0.0099, exceeds every shuffle; gradient boosting
      0.583 vs. null mean 0.502 (SD 0.022, max 0.559) — p=0.0099, same.
      Both `model_figures.py`'s 5 figures and `predict_claim.py`'s
      docstring/output text updated to the new 1,644-claim / 0.598 /
      p=0.0099 numbers (was 843 / 0.624 / p=0.0196). All 34 offline tests
      still pass; all touched files `py_compile` clean.
      **Honest summary of this whole exercise**: growing the corpus ~1.8x
      did NOT improve the headline LOEO accuracy (0.624 -> 0.598) — the
      earlier "sample size is the ceiling" hypothesis from 2026-07-18 was
      wrong, or at least incomplete. The model is still clearly real
      (permutation p=0.0099 either way), just not bigger-corpus-improvable
      the way the tuning wins were. Worth stating this directly if asked
      "did more data help" — the honest answer is no, and that itself is a
      finding (this corpus's signal ceiling isn't sample-size-bound).
- [x] **New direction (2026-07-17): forecaster-disagreement model.** After
      bill_arm's prediction model was scrapped (see that section) and
      "predict economic state from newspaper articles" was rejected (already
      have this — the real Baker-Bloom-Davis EPU index via
      `tier2_analysis.epu_series()` — and the raw-article corpus is
      episode-curated, not continuous, so a from-scratch version would be
      confounded), landed on: does forecaster DISAGREEMENT predict how hard
      a claim is to call correctly, and does false consensus precede worse
      crises? Novel because existing disagreement-as-leading-indicator
      research is built entirely on the Survey of Professional Forecasters
      (1968-present) — this 1905-2009 dataset can test whether it holds
      across regime changes the SPF can't reach. Checked against real data
      before building: every one of 19 episodes has measurable two-sided
      disagreement (e.g. 1929 Crash: 64 improve vs. 10 worsen; 1973 Oil
      Shock: 25 vs. 39).
      New `disagreement.py`: `add_disagreement_features()` computes a
      per-claim, BACKWARD-LOOKING, leakage-safe `local_disagreement`
      (minority share of improve/worsen among other claims in the same
      episode dated on/before this claim, within a 3-month window; first-
      claim-in-episode edge case imputed to the episode's overall rate, a
      documented assumption same as `score_claims.resolve_horizon()`'s
      default-12-month fallback). `episode_disagreement_rate()` is the same
      logic aggregated to one number per episode, for the planned Part 2
      (episode-level disagreement vs. NBER/FRED crisis severity — not yet
      built). 9 new offline tests added to `test_offline.py` (hand-computed
      expected values on a tiny synthetic episode, backward-only-window
      behavior, no_change-claims-still-get-a-value-but-don't-count-toward-it,
      first-claim imputation) — all pass, 34/34 total.
      Wired into `model.py`'s `NUM` list, held to the same permutation-test
      validation gate `political_lean`/`urban_rural` went through on
      2026-07-16 (LOEO accuracy WITH vs. WITHOUT in `NUM`, both models,
      50-shuffle permutation test each). **RESULT (2026-07-17): NULL —
      logistic regression 0.583->0.580 (unchanged, within noise), gradient
      boosting 0.573->0.552 (measurably WORSE).** Local disagreement does
      not help predict individual claim accuracy once EPU/direction/voice
      are already in the model, and mildly hurts gradient boosting.
      **Reverted from `NUM`** — `add_disagreement_features` is still called
      in `build()` so the column is computed and available to explore
      (same pattern as political_lean/urban_rural), just not load-bearing
      for the reported model. Verified: `model.py` re-run after the revert
      reproduces the exact 0.583 baseline LOEO accuracy.
      This is a real, honest negative finding, not nothing — but it does
      NOT settle Part 2 (below), a different question at a different level
      of analysis (episode-level consensus vs. crisis severity, not
      per-claim local disagreement vs. accuracy). User's call: build Part 2
      anyway and present Part 1's null honestly alongside it (not hide it).

- [x] **Part 2 built and run: `disagreement_severity.py`** (2026-07-17).
      Episode-level (n=19) test: does overall consensus (low disagreement)
      precede WORSE crises than genuine disagreement does? Severity =
      peak-to-trough %% decline in FRED INDPRO within each episode's own
      claim-date span (objective boundary, not hand-picked); pre-1919
      episodes (1905 Calm, 1907 Panic, INDPRO starts 1919) fall back to
      NBER recession-month fraction, reported in the table but EXCLUDED
      from the correlation/plot -- different scale, would mislead if mixed
      in. **RESULT: Spearman r=0.118, p=0.651, n=17.** Right direction
      (weakly consistent with "more disagreement -> milder outcomes") but
      nowhere near significant -- 1929 Crash (disagreement 0.135, -24.4%
      decline) and 1920 Depression (0.153, -29.9%) fit the hypothesis, but
      1937 Recession (0.293, -26.3%) doesn't, and the overall relationship
      across all 17 is statistical noise at this sample size. **Both parts
      of this idea are honest null/inconclusive results, not confirmed
      findings** -- report them that way; the real contribution is the
      rigor (leakage-safe backward-looking feature construction, the
      permutation-test gate, disclosing p=0.651 instead of only showing the
      directionally-nice cherry-picked examples), not a "we found it"
      story. `figures/fig_disagreement_severity.png` written. All 34
      offline tests still pass after this addition.

- [x] **Publisher track-record persistence — quick-checked, also null**
      (2026-07-17, ad-hoc check, not a formal script). Does a publisher's
      past accuracy (>=3 prior claims in strictly earlier episodes) predict
      its accuracy on a new claim? n=418 claims with a real prior track
      record: hit rate above-median track record 41.5% vs. below-median
      43.5% — indistinguishable, correlation -0.11. Forecasting skill does
      not appear to persist for an institution across different economic
      regimes in this data. Third null in a row on this 843-claim corpus
      (after both disagreement parts) — flagged to the user as a pattern
      (thin subgroup samples, not bad luck) rather than continuing to slice
      the same data a fourth way.

- [x] **`predict_claim.py` — packaged the validated model as an actual
      input->output predictor**, not another factor-analysis pass
      (2026-07-17, user's explicit ask: "data science is not just
      analysis"). Reuses `model.py`'s exact pipeline/features, fit on ALL
      843 claims (not the held-out split — that split is for honestly
      measuring generalization, a deployed predictor should use every
      historical example). CLI + `--interactive` mode: give it a claim's
      voice/confidence/direction/topic/region/EPU, get back a predicted
      probability it's correct. Sanity-checked against known real patterns
      before trusting it: officials score lower than experts/journalists
      (40.0% vs 52.9%, matches the real 36.5%/53.2% split), "improve" scores
      higher than "worsen" (matches the real 58.7%/33.3% split) — not
      random output. Documented limit: `epu_series()` only covers
      1900-2014, so predicting on a claim about TODAY's economy needs a
      real `--epu` value looked up manually
      (policyuncertainty.com/us_monthly.html), not the historical-median
      fallback, or the prediction is degraded, not live. All 34 offline
      tests still pass; `python -m py_compile` clean.

- [x] **Improved model accuracy for real: `LOGIT_C` tuned via grouped CV**
      (2026-07-18, user asked "how can we improve the AI accuracy"). Tried
      three things:
      (1) **Soft-voting ensemble of logistic + gradient boosting — null.**
      LOEO accuracy 0.581, no better than logistic alone (0.583); GB's
      weaker 0.573 just dilutes the average. Killed the job partway through
      its permutation-test run once the headline number made the verdict
      clear — not worth the remaining compute.
      (2) **Hyperparameter tuning — real improvement.** `GridSearchCV` with
      `LeaveOneGroupOut` (grouped by episode, not random — same leakage
      rule as everywhere else) over logistic regression's `C`: monotonic
      trend across the whole grid (stronger regularization = better, all
      the way down), best at **C=0.05: LOEO 0.624 vs. the old default
      C=0.5's 0.583**. Makes sense — TF-IDF gives up to 500 text features
      on only 843 examples, so the old default was underregularized.
      Gradient boosting's own grid search found only a noise-level gain
      (0.573->0.578) — not adopted.
      (3) Confirmed C=0.05 isn't a fluke with the same permutation-test
      gate as everything else: real accuracy 0.624 vs. null mean 0.491
      (SD 0.025, max 0.533), p=0.0196 — exceeds every one of 50 shuffles.
      Adopted as `model.LOGIT_C` (documented constant, not a magic number)
      and wired into every place `C=0.5` was hardcoded: `model.py`,
      `model_figures.py` (figures regenerated), `predict_claim.py`
      (docstring/output text updated from the stale 0.583 to 0.624).
      Re-verified `predict_claim.py`'s sanity checks still hold with the
      new setting (officials 27.1% vs experts 34.3%, same direction as
      before). All 34 offline tests still pass.
      Also identified (not yet acted on): of 2,644 graded newspaper quotes,
      only 929 (35%) are actual forward-looking predictions — the other
      65% are factual/descriptive reporting. This is a real structural
      fact about the corpus (pipeline isn't broken), and it caps how much
      more scored data exists without fresh scraping — the highest-ceiling
      remaining lever (growing claims_raw.csv itself) needs real time, not
      attempted given the 2-week clock.

- [x] **Gradient boosting tuning — validated and REJECTED, not adopted**
      (2026-07-18). The n_estimators=200 result found earlier (0.573->0.578)
      was checked against the same permutation-test gate as everything
      else: the null distribution's spread (SD ~0.025-0.026) is 5x bigger
      than the gain itself (+0.005) — nowhere near distinguishable from
      noise, unlike `LOGIT_C`'s +0.041 jump (several SDs beyond the null).
      Left gradient boosting at its sklearn defaults. Consistent with this
      project's standard: report what was tested, don't adopt what
      doesn't clear the bar just because it was tried.

- [x] **Recall audit of the 12 LOC search terms** (2026-07-18, closes the gap
      flagged in "Known limits": "says nothing about what the 12 search terms
      *missed*"). Compared `search_log.csv`'s `total_hits` (LOC's own count of
      matching pages) against `pages_taken` (capped at the scraper's default
      `--pages-per-term 30` = 300 articles/term). Result: **18 of 32
      episode/term searches were fully exhausted** (every available hit
      fetched), but **9 hit the page cap having fetched under 50% of what was
      available** — worst cases: "reconversion" (1945 Reconversion) fetched
      only 4.8% of 6,214 available hits, "unemployment will" (same episode)
      7.5% of 4,005, "financial panic" (1907 Panic) 10.3% of 2,908. **20,331
      hits across the corpus were never fetched, purely from the page cap**,
      concentrated in the highest-volume crisis episodes (1945 Reconversion,
      1907 Panic, 1957 Recession, 1929 Crash, 1920 Depression, 1937
      Recession, 1948 Recession). Scope note, stated honestly: this measures
      under-fetching of terms that WERE searched, not phrasing the 12 terms
      never tried at all — a true blind-spot audit would need a different
      method (e.g. sampling random articles from these episodes and checking
      for forecast language absent from the 12 terms), not attempted here.
      **Follow-up action, completed 2026-07-18**: reran
      `newspaper_scraper.py --pages-per-term 100` on the 7 under-recall
      crisis episodes, writing to a NEW file (`claims_raw_expanded.csv`, NOT
      `claims_raw.csv` — the scraper overwrites its output on every run, so
      reusing the original filename would have destroyed the existing,
      already-graded 3,253-row corpus). `search_log.csv` from before this
      run preserved as `search_log_baseline_2026-07-16.csv` (the scraper
      also overwrites this file every run). Ran to completion despite
      persistent LOC server flakiness along the way (`IncompleteRead`/
      `HTTP 520` errors on several high-volume terms) — the script's
      existing per-term retry/skip logic absorbed it. **Result: 2,536
      candidate claims from 1,859 pages**, all now graded (see the
      `grade_claims.py` batch-grading entry below) — this DID end up
      growing the trained model's usable data, once the user supplied
      `OPENAI_API_KEY` via `bill_arm/.env` to unblock grading.
- [x] **Spot-check: sampled 30 `is_prediction=yes` rows from
      `claims_graded.csv` against the retrospective-vs-prediction rubric
      rule** (2026-07-18, closes the "spot-check a sample... before trusting
      the state-prediction model" item in "Not done / next up"). Manually
      read each quote. **Result: ~1/30 (3%) is a clear rubric violation** —
      claim_id 117 ("There is not a vacant store on Main street...") is a
      present-state rebuttal used rhetorically, not a forecast, and should
      not have been graded `is_prediction=yes`. **2 more (7%) are soft/
      debatable borderline cases** — claim_id 182 ("at no time in history
      has the outlook been better") and claim_id 520 ("absence of dealers
      concerned over the uncertain business outlook") are retrospective/
      present-state sentiment rather than an explicit forward claim, though
      both use the corpus's established "business outlook" convention that
      the rubric does treat as forecast-bearing elsewhere. **27/30 (90%) are
      clean, unambiguous forward-looking predictions.** Also noticed (a
      separate, minor data-quality issue, not a rubric violation): 2/30
      quotes (claim_id 1045, 864) have unrelated OCR text glued on after the
      real claim sentence (ad copy, garbled headlines) — the `JUNK` regex in
      `newspaper_scraper.py` doesn't catch these specific patterns. **Verdict:
      the ~3% (1/30) violation rate is small enough not to justify a full
      re-grading pass**, but worth disclosing exactly like this if the
      state-prediction model's accuracy goes on the poster — the label noise
      floor is not zero.
- [x] **`grade_claims.py` — added OpenAI Batch API grading + Groq multi-key
      rotation** (2026-07-18, user's explicit cost plan: "use the ChatGPT
      key for grading until it runs out of credits [$11.69 in the account],
      then start rotating Groq keys"). New `--batch` mode submits the whole
      job as one JSONL file to `/v1/batches` (50% cheaper than one call per
      claim, since it doesn't need `--sleep` throttling either) and polls
      until done; new `--auto` mode runs the OpenAI batch phase first, then
      — only if that phase doesn't fully complete (job failure, expiry, or
      per-request errors, which is how an empty account shows up) —
      automatically falls back to a new `KeyRotator` class that cycles
      through all 5 Groq keys in `bill_arm/.env` (rotating on
      `DailyCapReached` instead of stopping the whole run, only raising once
      every key is capped). `bill_arm/.env` is a human notes file, not valid
      shell syntax (`KEY: value`, not `KEY=value`), so `load_labeled_keys()`
      parses it directly by regex rather than relying on `source`. Smoke-
      tested on 3 real claims via `--batch` before any real spend (correct
      is_prediction/direction/voice output, matches the existing rubric
      shape) — confirms the pipeline is correct end to end, not just that it
      runs.
      **First real run failed, root-caused, fixed** (2026-07-18): submitting
      all 2,415 claims as one batch failed immediately with
      `token_limit_exceeded` — "Enqueued token limit reached for gpt-4.1 ...
      Limit: 900,000 enqueued tokens" (an org-level cap on tokens across ALL
      in_progress batches, unrelated to the $11.69 balance; `--auto`'s
      generic "batch didn't complete -> fall back" logic correctly caught
      this as a failure, but wrongly treated it the same as an exhausted
      account and started burning the weaker Groq model on it). Confirmed
      by querying the failed batch object directly (`errors.data[0].code`)
      rather than guessing. **Fix**: `run_batch()` now submits in
      sequential chunks (`--batch-chunk-size`, default 600 claims, well
      under the 900k-token cap) instead of one giant job, so only one chunk
      is ever in_progress at a time; a chunk that still hits
      `token_limit_exceeded` (e.g. another job sharing the org's quota) gets
      one 60s-delayed retry before the run gives up on the OpenAI phase and
      hands off to Groq for real. Killed the bad run after only 22 claims
      had gone through Groq (cheap to discard, `claims_graded_expanded.csv`
      resume logic picked up cleanly). **Relaunched and completed clean**:
      all 5 chunks (4x600 + 1x26 = 2,426 claims, plus the 23 already on disk
      = **2,449 total graded**) finished via OpenAI batch alone, no Groq
      fallback needed. Usage summed across the 5 batch jobs: ~2.32M input +
      ~188k output tokens — at gpt-4.1 batch pricing this is a low-single-
      -digit-dollar spend, comfortably inside the $11.69 balance (exact
      dollar figure not pulled from a billing endpoint, this is a token-
      count-based estimate). Output: `claims_graded_expanded.csv`.
      **Top-up pass completed 2026-07-18** once the LOC scraper (see below)
      finished: 87 more claims, 1 chunk, no issues -> **2,536 claims fully
      graded, 0 remaining**. Corpus-level result: **1,232/2,536 (48.6%)
      judged real forward-looking predictions** — notably higher than the
      original corpus's ~35% rate, consistent with this expansion
      deliberately targeting the highest-volume crisis episodes the old
      page cap had truncated (see "Recall audit" entry above). **Still not
      done**: merge-or-hold-out decision for whether these 2,536 claims
      (`claims_graded_expanded.csv`) join `model.py`'s training data or stay
      a separate held-out set — not yet made.
- [x] **Resumed the stalled NYT downloads** (2026-07-18, `NYT_API_KEY`
      supplied via `bill_arm/.env`). `gulf_1990` grew 422->**456** articles
      (34 new, from the "economic downturn" phrase); `oil_1973` and
      `volcker_1980` were already fully exhausted for the current 15-phrase
      list (0 new each) — these two windows were NOT actually incomplete,
      just never rerun after the phrase list was broadened 2026-07-16 (see
      that entry). New articles land in `election_arm/data/raw/
      nyt_economy_gulf_1990.jsonl`; still need `append_nyt_claims.py` +
      grading to reach `model.py`.
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
- [x] **NYT post-1963 corpus merged and scored end-to-end** (2026-07-16, later
      session). Ran the logged next step: `append_nyt_claims.py` merged the 156
      NYT articles into `claims_raw.csv` (now **1,480 rows**, claim_ids
      1325-1480), then `grade_claims.py --model gpt-4.1` graded ONLY the 156 new
      rows (30 judged predictions: gfc_2008=10, oil_1973=6, dotcom_2001=5,
      volcker_1980=4, gulf_1990=2, crash_1987=1, calm_1965=1, calm_1995=1), then
      `score_claims.py` regenerated the unified 1905-2010 `claims_scored.csv`
      (**698 predictions**, was 668) and its figures (`fig_leaderboard`,
      `fig_hit_by_episode`, `fig_optimism_gap`, `fig_calibration_voice_control`
      at 13:30). Results: NYT now on the publisher leaderboard (n=30, 43.3% hit,
      the lowest-scoring publisher there); the newspapers-vs-Livingston
      head-to-head now spans 1946-2008 (newspaper n 182->212 = +30 NYT), still
      beating economists 61.3% vs 54.4% (CI [54.7,67.5] excludes Livingston).
      DATA-RECOVERY NOTE: the final gpt-4.1 `claims_graded.csv` (1,324 rows) had
      been DELETED after the 12:53 run (not on disk/staged/stashed; only
      `.bak` partials + the 668-row `claims_scored.csv` survived). It was
      faithfully RECONSTRUCTED (668 predictions pulled from `claims_scored.csv`
      + 656 non-predictions marked `is_prediction=no` with blank grade fields,
      per rubric) before grading the NYT rows — byte-consistent with the grade
      that produced the existing figures, so no LOC regrade was spent. If a
      cleaner artifact is wanted, a full `--overwrite` regrade of all 1,480 on
      gpt-4.1 costs ~$3.7.

- [x] **NYT under-sampling fixed; full 1905-2010 corpus merged, graded, and
      scored** (2026-07-16 evening). The 30-NYT-claim corpus was badly
      under-sampled (9 narrow exact-quote phrases against a headline/lead-only
      index) and confounded era with source. Fix chain: broadened
      `ECONOMY_PHRASES` to 16 LOC-aligned terms; fixed a real bug in
      `download_nyt.py`'s `search_phrase()` where it trusted NYT's `meta.hits`
      field (frequently misreports 0 even when real docs come back) instead of
      actual page size, silently truncating every phrase to page 1; added
      `--include-pre-1963` so `--arm economy` defaults to the 9 post-1963
      windows instead of wastefully re-hitting the 10 LOC already covers.
      Re-downloaded all 9 windows to real depth (was 12-881, now 120-881,
      every window has substance). Merged via `append_nyt_claims.py` with a
      flat cap of 150 claims/episode (matched to the LOC episodes' own
      87-212-claim scale — no clean crisis/control 1:1 pairing exists to match
      against instead), fixed seed for reproducibility: 1,164 new claims
      appended to `claims_raw.csv` (1,480 -> **2,644 rows**), every window
      landing 97-145 claims, well balanced. Graded with `gpt-4.1` (2,644/2,644,
      929 judged real predictions). `score_claims.py` regenerated
      `claims_scored.csv` (**843 scorable predictions across 19 episodes**,
      was 668) and all figures; `tier2_analysis.py`, `model.py`,
      `model_figures.py` rerun against the unified corpus (LOEO accuracy
      0.520 +/- 0.166 across all 19 episodes). Full 1905-2010 span now has
      real per-episode depth instead of a thin post-1963 tail.
      SECURITY: `CONTINUE_HERE_nyt.md` had two live NYT API keys pasted in
      plaintext (one matching `bill_arm/.env`'s live key), committed in
      `1e201bc`. Stripped from the working copy, but still in git history —
      **user needs to rotate both keys at developer.nytimes.com**; a
      working-tree edit doesn't undo the history exposure.
      Caveat carried forward: NYT returns headline/lead only (no body text),
      so recall is capped regardless of term count; full-text depth would need
      library ProQuest access, not more NYT calls.
- [x] **`tier3_robustness.py` run for the first time ever** (2026-07-16 — the
      file existed since 07-10 but had never been executed). Results REVISE
      two headline claims downward:
      (1) **Newspapers-vs-Livingston "newspapers win" claim does NOT survive
      a proper time-series test.** Diebold-Mariano (Newey-West HAC, HLN
      small-sample correction) on the 1946-63 head-to-head: DM=-1.033,
      p=0.349 — no significant difference. Only 6 matched half-year buckets,
      12 Livingston forecasts — that's the real effective n, not the raw
      claim count. The "64.3% vs 54.4%, newspapers beat economists" framing
      used earlier in this project should be retired; the honest headline is
      "no detectable difference at this sample size."
      (2) **The `no_change` bands are NOT robust.** Newspaper hit rate swings
      43.4% (2x bands) -> 52.9% (1x) -> 55.8% (0.5x bands) — a 12.4-point
      spread, well past the script's own 5-point "needs disclosure"
      threshold. Any reported hit rate needs this sensitivity attached, not
      a single number.
      (3) **Livingston era rankings are NOT stable** to +/-1yr boundary
      shifts (Great Moderation and Vietnam/stagflation swap best/worst
      ranking). Era-specific claims about economist accuracy need this
      caveat.
      This addresses the "Preregister the no_change bands" and general
      robustness items previously in Needs-to-be-done-by-you — the bands
      were never preregistered (can't be, retroactively), but they are now
      at least sensitivity-tested and the result is honestly reported as
      fragile rather than asserted as clean.
      Deliberately did NOT pick a "better" single band width after seeing
      these results — that would just be a new undisclosed post-hoc choice,
      the same problem in a different spot. Instead added figure output to
      `band_sensitivity()` and `era_shift_robustness()`
      (`figures/fig_band_sensitivity.png`, `figures/fig_era_stability.png`)
      so the fragility is a citable, includable artifact rather than console
      text someone has to remember to mention. Bonus finding visible only in
      the figure: crisis vs. control hit rate actually FLIPS ordering at 2x
      bands (control drops below crisis) — another reason a single point
      estimate would mislead.
- [x] **Justified fixes for two of the three fragility sources** (2026-07-16,
      following up on the robustness run above). Deliberately partial —
      implemented what has a real external citation, disclosed what doesn't,
      rather than inventing shaky justifications to look complete.
      `UNRATE` band: 0.3pt (uncited) -> **0.5pt, the Sahm Rule threshold**
      (Sahm 2019, FRED's SAHMREALTIME series) — a real, externally-validated
      recession-signal threshold, adapted here (Sahm's rule is about a rise
      from a 12-month low specifically; this project scores any 12-month
      move either direction, so it's "anchored to," not a literal copy of,
      the original rule. Effect on the band-sensitivity spread: negligible
      (55.8%->55.6% at 0.5x bands, etc.) — confirms CPI/INDPRO, still
      uncited round numbers, are what actually drive the 12.4-point spread,
      not UNRATE. `CPI`/`INDPRO` deliberately left as-is: a self-computed
      "historical volatility" band was considered and rejected because it
      would mix real business-cycle swings into what's supposed to be a
      noise floor, making it a worse-hidden problem, not a fix. Documented
      as a disclosed limitation in `score_claims.py` instead.
      Era boundaries (`tier3_robustness.py`'s `ERAS`): replaced round decade
      numbers (1965/2000/2012) with actual regime-change anchors —
      1973 (Oct 1973 OPEC oil embargo), 1982 (NBER trough of the Volcker
      recession), 2007 (NBER peak / start of Great Recession, also roughly
      where Bernanke's own 2004 "Great Moderation" speech dates that era's
      end), 2014 (Yellen becomes Fed Chair). "Terror / fin. crisis" renamed
      to "Financial crisis & recovery" since its span no longer includes
      2001 at all. Result: rankings are now LOCALLY stable (all of -3y/-2y/-1y
      agree with each other; +0y/+1y agree with each other) — a real
      improvement over the old boundaries, which flipped at nearly every
      single-year shift — but still not stable across the full +/-3y range,
      so "flag the unstable eras" still stands, just less severely.
      KNOWN INCONSISTENCY: `BU_RISE_forecast_analysis_FIXED.ipynb`'s section 3
      still uses the old uncited 1965/2000/2012 boundaries — now diverges
      from `tier3_robustness.py`. Needs reconciling before treating the
      notebook and script as consistent (not done here — out of scope for a
      .ipynb edit in this pass).
- [x] **`calibrate_bands.py` built — human-calibration protocol for CPI/INDPRO
      bands** (2026-07-16), the path identified above since no external
      citation exists for these two (unlike UNRATE's Sahm Rule). Searched
      first for a real external anchor: BLS does publish a CPI standard
      error (median 0.07% on the 12-month change), but that measures
      *sampling/measurement precision*, not *economic significance* — using
      it would set the band absurdly low (~0.14pt) and answer a different
      question than "did this count as a real 12-month directional move."
      Searched for a Sahm-Rule equivalent for inflation/production regime
      shifts specifically; none found in the literature. So: built the same
      human double-coding protocol the grading rubric already uses.
      `calibrate_bands.py` samples 80 windows (40 CPI, 40 INDPRO) from the
      FULL historical FRED series via quantile-stratified bins (per-series,
      since CPI and INDPRO have very different natural volatility scales),
      seeded and shuffled, deliberately NOT filtered to claim dates or
      outcomes so the calibration can't be tuned toward a preferred result.
      Wrote `calibration_sample.csv` (ready to fill). `--analyze` mode
      computes inter-rater kappa (same 0.7 bar as the grading validation)
      and derives a suggested band per series from where judgment flips
      from "no_change" to "real_change" — reports an honest overlap zone
      instead of false precision if judgments aren't cleanly separated by
      magnitude. Waiting on two humans (Needs-to-be-done-by-you) — nothing
      else can proceed on this until that's filled in.
- [x] **CPI/INDPRO bands calibrated and applied** (2026-07-16). Vincent and
      Jeremy graded `calibration_sample.csv` TOGETHER as a joint consensus,
      not independently (`human1_judgment` filled for all 80 rows,
      `human2_judgment` empty) — `calibrate_bands.py --analyze` adapted to
      handle this (found and fixed a real bug in the process: pandas reads
      empty CSV cells as `NaN`, and `.astype(str)` turns that into the
      literal string `"nan"`, which broke the original empty-column check).
      **This is weaker evidence than independent double-coding** — no kappa
      is computable from one column, so there's no check the judgment is
      reproducible by an uninfluenced second rater. Disclosed as such in
      `score_claims.py`'s comment, not presented as kappa-validated.
      Both series showed CLEAN separation (no ambiguous overlap zone):
      **CPI 1.5% -> 1.17%**, **INDPRO 2.0% -> 2.33%**. Applied to `BANDS`,
      full pipeline rerun (`score_claims.py`, `tier2_analysis.py`,
      `tier3_robustness.py`, `model.py`, `model_figures.py`).
      REAL, MATERIAL EFFECT — not a rounding change:
      (1) Newspapers-vs-Livingston 1946-63 flips sign under the naive
      bootstrap-CI method: was 64.3% vs 54.4% ("newspapers win"), now
      **47.6% vs 54.4%, CI [42.9%,52.2%] excludes Livingston** ("newspapers
      lose, significantly, by the same naive method that showed the
      opposite before"). Illustrates exactly why the band-sensitivity
      disclosure matters — a "significant" finding flipped direction from
      a defensible band re-calibration.
      (2) The proper Diebold-Mariano test (item 8) is UNCHANGED in
      conclusion under the new bands: still "no significant difference"
      (p=0.407, was p=0.349) — the rigorous test is stable across both old
      and new bands; the naive bootstrap CI is not. That contrast is itself
      the headline methods point: use DM, not the bootstrap CI, for this
      comparison.
      (3) Band-sensitivity spread got slightly WORSE, not better (14.7pt,
      was 12.4pt) — calibration fixes what the 1x band-width value IS
      justified to be, it does NOT make the result robust to still-somewhat-
      arbitrary 0.5x/2x perturbations. Don't oversell this as "fixed
      fragility"; it's "fixed which single number is defensible."
      (4) NYT drops from lowest-but-one to outright LOWEST publisher on the
      leaderboard (45.2% -> 39.4%). Three-way benchmark newspaper rate also
      drops (64.3% -> 59.3%, still above Livingston's 54.4% and Michigan's
      55.2%, unchanged since Michigan/Livingston don't depend on these
      bands).
      All figures regenerated under the calibrated bands.
- [x] **DC recoded as a political hub, not a financial center** (2026-07-16,
      per user decision — "unimpeachable validation" deprioritized, not
      pursuing). `tier2_analysis.py`'s `geography_analysis` now splits three
      ways instead of two: financial-center states (NY/IL/MA/PA only, DC
      removed), `political hub (DC)`, and elsewhere. Result changes materially
      now that DC's volume isn't inflating the financial-center bucket: true
      financial centers n=23, 65.2% hit rate; DC n=163, 55.8%; elsewhere n=657,
      51.8%. `fig_geography.png` and `results_by_region.csv` regenerated.
- [x] **"Michigan" labeled explicitly as a national survey, not a state one**
      (2026-07-16). `tier2_analysis.py`'s docstring, printed table labels, and
      `fig_three_way_benchmark.png` now say "US households, Michigan SRC
      survey" instead of bare "Michigan households" — UMCSENT is a
      nationally-representative survey administered by the University of
      Michigan Survey Research Center, not a Michigan-state-specific one;
      the old label repeatedly caused this to be misread as geographically
      restricted. Also fixed a stale docstring comment still listing DC as a
      financial-center state after the recoding above. Figure regenerated.
- [x] **`model.py` feature set fixed — added 3 already-available, unused
      features** (2026-07-16): `direction` (the claim's own stated
      prediction — was never fed in as a feature at all), `months` (resolved
      horizon, likewise unused), and replaced sparse raw `state` with
      `region`/`fin_center` (same derivation as `tier2_analysis.py`, less
      noisy than 50+ mostly-thin state categories). REAL improvement, not
      noise: gradient boosting AUC 0.557 -> **0.646**, held-out accuracy
      0.533 -> **0.607** (now clears the 0.574 majority baseline, wasn't
      before); logistic regression AUC 0.448 (below chance!) -> **0.543**.
      LOEO accuracy 0.507-0.533 -> **0.586** (still wide, ±0.215).
      Permutation importance shows why: `direction` dominates by a wide
      margin (+0.154 AUC drop when shuffled, ~3x the next feature) —
      consistent with the optimism-gap finding, a claim's own predicted
      direction carries real information about whether it's right.
      `region`/`fin_center` contributed ~nothing (~0.000) — didn't hurt,
      but the improvement is almost entirely from `direction`, not the
      geography cleanup.
- [x] **`permutation_test()` built in `model.py`** (`--permutation-test
      --n-perm N`, 2026-07-16) — the test that was still missing: shuffle
      `hit` labels globally (unrestricted, breaking any real feature-target
      relationship while leaving the LeaveOneGroupOut CV machinery and base
      rate exactly as they really are), rerun the identical LOEO procedure
      many times, compare the REAL LOEO accuracy to that null distribution.
      Answers "does this model beat chance," which `permutation_importance`
      (already used elsewhere in the file) does NOT — that only ranks
      columns against each other, it doesn't test the model as a whole.
      Timed first: one 19-fold LOEO pass costs ~9-10s for EITHER model
      (logistic regression is slower than gradient boosting here, ~10.4s vs
      ~8.8s, likely from lbfgs convergence struggling on the sparse
      TF-IDF+OneHot feature space — matches the ConvergenceWarning spam
      already visible in normal runs). `n_perm=200` for both models would be
      over an hour; running at `n_perm=100` (~32 min) instead, backgrounded
      — still gives ~0.01 p-value resolution, enough for a clear
      significant/not-significant call.
      RESULT (2026-07-16): both models beat chance decisively. Logistic
      regression LOEO 0.586 vs. null mean 0.493 (SD 0.024, max across all
      100 shuffles 0.539) -- p=0.0099, ~3.9 SD above the null. Gradient
      boosting LOEO 0.571 vs. null mean 0.496 (SD 0.025, max 0.556) --
      p=0.0099, ~3.0 SD above the null. The real value exceeded EVERY
      permuted run for both models -- as clean a significant result as
      n_perm=100 can show. Null distribution centers almost exactly on the
      ~0.50 base rate, confirming the test is well-calibrated.
      SCOPE, stated honestly: since `direction` dominates permutation
      importance (+0.154 vs ~0.05 for the next feature), this significant
      result is mostly attributable to ONE feature, not broad predictive
      structure across the metadata. Correct framing: "whether a claim
      predicts improve vs. worsen carries real, statistically verified
      information about whether it turns out right" -- a rigorous
      confirmation of the optimism-gap finding, not a general "claim
      correctness is predictable from newspaper metadata" claim. This is
      the strongest, cleanest positive result in the project so far.
- [x] **Gallup substitution accepted by user** (2026-07-16) — Livingston
      (economists) + UMCSENT (households) stay as the permanent stand-ins
      for the originally-planned Gallup poll data (not freely downloadable
      pre-1960). No further action needed on this.
- [x] **Publisher metadata received and wired in** (2026-07-16). User
      hand-researched the top 30 publishers (political lean, urban/rural,
      circulation, per-row sourcing — loc.gov essays, encyclopedias,
      explicit "UNKNOWN" where unverified) -> `publisher_metadata.csv`
      (renamed from a spreadsheet-export filename). Joined into `model.py`.
- [x] **Political-climate proxy built for the full 1905-2010 span**
      (2026-07-16) — `data/political_climate.csv`, 59th-111th Congress,
      president/Senate/House majority party per Congress, sourced from
      Wikipedia's "Party divisions of United States Congresses" table
      (verified against official House/Senate history page framing before
      use, not from memory). Deliberately NOT linked to `bill_arm` (only
      2003-2024, ~7 years of overlap out of 105 — would leave 93% of the
      corpus missing the feature or force stitching two construction
      methods together, recreating the NYT-only era-confound problem).
      TESTED, HONEST NEGATIVE/AMBIGUOUS RESULT: added `political_lean`,
      `urban_rural`, `unified_government`, `president_party` to `model.py`.
      A single train/test split suggested a big drop (GB accuracy
      0.607->0.549) — but isolating each addition under the more robust
      LOEO CV showed all four combinations (neither/publisher-only/
      political-only/both) landing within 0.548-0.584, indistinguishable
      from each other given LOEO's own ~0.20 SD. Same lesson as the DM-vs-
      bootstrap finding: a single split gave a misleadingly large signal a
      proper CV estimate didn't support. Kept the simpler, permutation-
      confirmed feature set as the shipped model rather than add complexity
      the evidence doesn't clearly support — the new columns are still
      computed and available in `build()`'s output for exploration
      (`df.groupby("political_lean")["hit"].mean()` etc.), just not
      load-bearing for the reported result.
- [x] **Composite 0-1 claim score built** (spec Step 4b, 2026-07-21) —
      `score_claims.py`'s `composite_score()`: unweighted mean of accuracy
      (`hit`), punctuality (`resolve_horizon`'s basis: stated=1.0,
      inferred=0.5, defaulted=0.0), and a new `resolve_specificity()`
      (named forecaster / numeric magnitude / concrete year-or-month
      reference in the quote, each 0/1, averaged). Deliberately rule-based
      off the existing quote text rather than a new LLM grading field — adding
      a fourth rubric field would mean re-grading the full ~4,100-claim corpus
      at real API cost for something derivable from what's already graded.
      Only defined where `hit` is (unscorable claims have no accuracy leg to
      anchor a composite to). Wired into `claims_scored.csv` and
      `results_by_episode.csv`; ran clean on the real corpus (mean composite
      0.292: accuracy 0.489, punctuality 0.209, specificity 0.177 — low
      punctuality/specificity legs mostly reflect that most claims are vague
      on both horizon and magnitude, not a scoring bug). `test_offline.py`
      still 34/34.
- [x] **`BU_RISE_forecast_analysis_FIXED.ipynb` reconciled with
      `tier3_robustness.py`'s anchored era boundaries** (2026-07-21) — the
      known inconsistency flagged above. Section 3's `ERAS` dict updated to
      the same 1973/1982/2007/2014 regime-change anchors (OPEC embargo,
      Volcker-recession NBER trough, Great-Recession NBER peak, Yellen
      confirmed as Fed Chair) and "Terror / fin. crisis" renamed to
      "Financial crisis & recovery" to match. Notebook re-executed end-to-end
      (`jupyter nbconvert --execute --inplace`, installed `nbconvert`/
      `ipykernel` into `bill_arm/.venv` for this — not added to any
      requirements file, dev-only) so every cached output/figure reflects the
      new boundaries rather than leaving stale results under new code.
- [x] **`model.py`'s GradientBoostingClassifier hyperparameters grid-searched
      — CONFIRMED NEGATIVE, kept at sklearn defaults** (2026-07-21). Same
      method as `LOGIT_C`'s own tuning: GridSearchCV via LeaveOneGroupOut
      (grouped by episode), scoring="accuracy", on the full 1,428-claim
      scorable corpus. 13 candidates across n_estimators 50-500, max_depth
      1-4, learning_rate 0.02-0.1. Best candidate (n_estimators=50, same
      depth/lr as default) edged the default on mean LOEO accuracy (0.6191 vs
      0.6172) but only won 15/19 folds — doesn't clear this project's own bar
      for adopting a tuning change (win in EVERY fold, per `LOGIT_C`'s and
      bill_arm's `XGB_PARAMS`' precedent). No candidate cleared that bar.
      Defaults are already near-optimal for this corpus; documented in a code
      comment so this isn't re-attempted without new data. Consistent with
      this arm's already-known lesson that GB tuning here doesn't survive
      rigorous validation (see the DM-vs-bootstrap and disagreement-feature
      findings above).
      Also added the equivalent tuning-readiness comment to
      `election_arm/model.py` (the shared elections/economy script,
      currently untuned) and calibration + PR-AUC/Brier reporting there
      (`CalibratedClassifierCV`, ported from `bill_arm/factor_analysis.py`,
      smoke-tested against synthetic data since `data/scored_claims.csv`/
      `data/scored_economy.csv` don't exist in this working tree yet —
      real tuning there is blocked on `analyze_elections.py`/
      `analyze_economy.py` actually being run against real data, not a
      fabricated result against synthetic rows.
- [x] **`model_interactions.py` built — feature-INTERACTION analysis, a
      question `model.py`'s marginal importances can't answer** (2026-07-21).
      None of this project's models previously looked past marginal
      importance (coefficients / gain / single-column permutation
      importance) at whether two features only matter IN COMBINATION.
      Fits a GradientBoostingClassifier on the structural (non-text) columns
      only — region, fin_center, political_lean, urban_rural,
      unified_government, president_party, confidence, voice, direction,
      year, epu, months, local_disagreement — on the same episode-based
      train split `model.py` uses, then computes SHAP interaction values
      (`shap.TreeExplainer`) on the held-out test claims only. Text (TF-IDF)
      features deliberately excluded: interaction values are O(features²)
      per sample, and 500 mostly-single-claim word features would be both
      too slow and too sparse to read.
      Single-split result at the time (since validated under LOEO CV + a
      permutation test — see the `validate_interactions.py` entry below):
      the strongest interactions were `direction x year`, `direction x epu`,
      and `president_party x year` / `unified_government x year|epu` — i.e.
      political-climate features dropped from the shipped model for a null
      MARGINAL effect (see "Political-climate proxy" entry above) showing up
      with real interaction strength.
- [x] **`validate_interactions.py` built — validated the interaction lead
      above under LOEO CV + a permutation test** (2026-07-22), closing the
      "Next step if pursued" this arm's own bar demanded. Method: each of the
      8 flagged pairs recomputed pooled OUT-OF-FOLD across all 19
      LeaveOneGroupOut(episode) folds (fit on 18 episodes, SHAP interaction
      values on the 19th only), then a permutation test (same machinery as
      `model.py`'s `permutation_test()`: shuffle `hit` globally, rerun the
      identical LOEO/SHAP pass, 200 shuffles, compare real pooled statistic
      to the null distribution).
      **Result: 7 of 8 pairs SIGNIFICANT at p=0.0050** (the maximum possible
      with 200 shuffles — real value beat every single null shuffle):
      `direction_worsen x year` (real 0.126 vs null mean 0.004),
      `direction_improve x epu` (0.098 vs 0.008), `direction_improve x year`
      (0.157 vs 0.005), `president_party_R x year` (0.032 vs 0.001),
      `president_party_D x year` (0.021 vs 0.001), `unified_government_False
      x year` (0.019 vs 0.001), `unified_government_True x year` (0.021 vs
      0.001). One pair did NOT clear the bar: `unified_government_False x
      epu` (real 0.006 vs null mean 0.002, p=0.080) — report as not
      significant, don't fold it into the "political climate interacts with
      time" claim.
      **Interpretation, and what this does NOT settle**: this confirms the
      interaction is a real (non-null) signal in the fitted model, not a
      single-split fluke — but it does not by itself separate a genuine
      political-climate-conditional effect from political-climate being a
      proxy for a time trend (`year` and `epu` are themselves correlated
      across 120 years of history, same caveat as before). Still just an
      interaction lead worth reporting with that caveat attached, not grounds
      to reopen `unified_government`/`president_party` as MARGINAL features
      in `model.py`'s shipped `CAT`/`NUM` (that decision was about marginal
      effect, which this doesn't change).
- [x] **Episode-level robustness check on the optimism-gap result — a real
      correction, not a footnote** (2026-07-22, `regret_scoring.py`). The
      headline claim-level test (66% vs 32% optimistic misses, binomial
      p=4.5e-10) treats every claim as an independent draw, which they are
      not: claims inside one episode share wire-service copy and one macro
      reality. Re-tested at the EPISODE level (13 crisis vs 6 control, the
      defensible independent unit): crisis episodes still lean far more
      optimistic (median optimistic-error share 45.9% vs 25.6%) but **does
      NOT clear significance** (Mann-Whitney U, one-sided, p=0.15, n=19).
      **1929 Crash alone remains a real, large result** (155 optimistic
      errors to 1, 10.6% hit rate) — but the century-wide asymmetry is now
      reported as directionally consistent and underpowered, not confirmed.
      More crisis episodes (the pending post-1963 ProQuest expansion) would
      be the natural way to actually power this test.
- [x] **Wilson CI added to `results_by_episode.csv`** (`score_claims.py`,
      2026-07-22) — several episodes have n<20 (1990 Recession n=10, 1987
      Crash n=14) where a bare point estimate invites over-reading noise as
      signal; the normal (Wald) interval is unreliable at these n/extreme-
      rate combinations, Wilson is not.
- [x] **`confidence` kappa citation corrected**: 0.17, not 0.19**
      (`hedging_lexicon.py`, 2026-07-22). 0.19 was the original Llama-3.3
      validation run in `KAPPA_RESULTS.md`; the corpus is actually graded by
      gpt-4.1 (the bake-off winner), whose own kappa on this field is 0.17 —
      an even weaker number, strengthening (not weakening) the case for the
      objective hedging-lexicon replacement. Does not change the "assertive
      claims are NOT less accurate" finding, only the citation backing it.
- [x] **`EXTENDED_ABSTRACT.md` revised for the two corrections above, plus a
      new Result 4** (2026-07-22): the `validate_interactions.py` SHAP-
      interaction lead (7/8 pairs significant, see entry above) is now
      Result 4 ("a lead, not a settled effect"); Narrative Economics
      renumbered to Result 5. Added a "ruled out, not hidden" limitations
      bullet (forecaster disagreement, publisher geography, political-climate
      features — all tested, none cleared the permutation-test bar) and an
      "episode count limits power" bullet tying directly to the Wilson-CI
      and Mann-Whitney results above. Conclusion softened from "systematically
      optimistic" to "leaned optimistic... clear and large in specific
      episodes, directionally consistent century-wide, not yet statistically
      confirmed at the episode level" — this is a more defensible claim, not
      a weaker paper.
- [x] **`election_arm/extract_predictions.py` — OpenAI Batch API support
      added** (2026-07-22). Ported the submit/poll/fetch pattern from
      `grade_claims.py`'s `run_batch` (50% cheaper, one job instead of one
      call per page); adapted for extraction's JSON-ARRAY-of-claims response
      shape (grade_claims.py's version returns one graded field per claim).
      New `--window all` merges every downloaded raw file for a source/arm
      into one job (chunked, default 150 pages/chunk — kept small since a
      12k-char OCR-page prompt is much heavier per-request than
      grade_claims.py's short grading prompts) instead of one small,
      latency-dominated job per window (each pays ~5-10 min of OpenAI
      validating/finalizing overhead regardless of size, so 28 tiny per-
      cycle jobs would cost hours vs. ~7 chunks for the whole corpus).
      `--model` now overridable (default still gpt-4.1, the bake-off
      winner); user supplied a paid OpenAI key earmarked for gpt-4.1-mini to
      cut cost further — used for this run, flagged inline that kappa
      validation matters more with an unvalidated cheaper model. Smoke-
      tested on 10 real NYT pages before the full run (correct claim
      parsing + metadata merge, matches existing schema).
      **Found and fixed a real, pre-existing bug while testing**:
      `election_arm/test_offline.py`'s extraction mock (`FakeClient`) mimicked
      the OLD Anthropic-SDK call shape (`messages.create(...)`); the code
      migrated to calling OpenAI's REST `chat/completions` directly on
      2026-07-16 and the stale mock was silently falling through to a REAL
      network call with a bogus key (HTTP 401) instead of testing anything —
      "run test_offline.py first, no network needed" was quietly false.
      Replaced with a `requests.post`-level mock. All 21 checks pass again.
      **Full election corpus (1,209 NYT articles, 28 cycles, 1900-2008)
      extracted** via `--window all --batch --model gpt-4.1-mini` — all 7
      chunks completed, 0 failures -> `data/predictions/pred_nyt_elections_
      all.jsonl` (119 real claims, 1,113 pages with no prediction found).
      **Stages 3/4/6 run for the first time ever on real (not synthetic)
      elections data (2026-07-22):**
      - Stage 3 (`validate_kappa.py sample --arm elections`): 23-claim blind
        sample drawn -> `data/validation_sample.csv` + hidden
        `data/validation_llm_key.csv`. **Still needs TWO independent human
        graders** before kappa can be computed — not automatable.
      - Stage 4 (`analyze_elections.py`): only 24 of 119 claims are
        national-scope + scorable (the rest are state-scope, unscored per
        `SCORE_STATES = False` — needs a state-results file, see Known
        limits). **Overall national-claim accuracy 58.3% (14/24)** — hedged
        claims notably MORE accurate than firm ones (71.4% vs 40%, n=14/10),
        opposite the naive "hedging = uncertainty = worse" prior, though at
        this n a couple of flipped claims would erase the gap entirely.
        -> `data/scored_claims.csv`.
      - Stage 6 (`model.py --arm elections`, default 2/3-1/3 cycle split:
        train 1904-1944, test 1948-2000): **both models score BELOW the
        0.571 majority-class baseline** (logistic accuracy 0.429, PR-AUC
        0.337; gradient boosting accuracy 0.143, PR-AUC 0.369) — at 17
        train / 7 test claims this is not a real test of anything, just the
        expected shape of a first pass on a corpus two orders of magnitude
        smaller than the economy arm's. One feature is worth noting anyway:
        `word:dewey` is among the strongest predictors of a WRONG call — the
        model landed on the famous 1948 "Dewey Defeats Truman" miscall
        without being told to, which at least confirms the pipeline surfaces
        real historical signal, not noise, even though the aggregate metrics
        aren't meaningful yet. **Needs the LOC pre-1963 side (never
        downloaded for elections, NYT-only so far) and more claims per cycle
        before this model means anything — flag this prominently if these
        numbers ever leave this repo.**
- [x] **`narratives.py --batch` added** (2026-07-22) — the full-corpus
      authoritative LLM narrative pass the module's own docstring called for
      but never had ("for the full corpus use grade_claims.py's --batch
      path" — that path didn't exist for narratives specifically). Reuses
      `grade_claims.py`'s generic `poll_batch`/`fetch_file` (no dependency on
      its grading-specific prompt), with its own request-building for the
      single-field `{"narrative": ...}` response. Smoke-tested on 5 real
      claims (plausible, LLM disagrees with the crude lexical screen on 4/5
      — expected, the lexical pass is a known-crude preview). **Full
      1,628-claim corpus completed** via `--batch --model gpt-4.1-mini
      --batch-chunk-size 800` (3 chunks, 0 failures) -> `claims_narratives_llm_
      full.csv`. **Result, NOT YET kappa-validated (see next entry) — do not
      quote on the poster before that**: lexical-vs-LLM raw agreement only
      18.2% (confirms the lexical screen was a crude preview, as documented).
      Narrative prevalence is far richer than the lexical pass suggested (only
      7.6% "none" vs. the lexical screen's 78%). Accuracy varies sharply and
      **in the opposite direction Shiller's hypothesis would suggest**:
      `panic_fear` claims have the LOWEST hit rate (34.0%, n=285) of any
      narrative, `new_era` the HIGHEST (64.7%, n=51); complacent narratives
      overall hit 52.1% vs 46.3% for everything else. Complacent-narrative
      share by episode also runs opposite to the classic "complacency spikes
      right before a crash" story: several CALM controls (1965: 71.4%, 1925:
      61.8%, 1955: 55.6%) show MORE complacent framing than most crisis
      episodes, including 1929 Crash itself (38.6%) and 2008 GFC (35.3%).
      Flagging this prominently because it's surprising and could easily be
      an LLM-labeling artifact rather than a real pattern — exactly what the
      kappa check below needs to settle before anyone treats it as a finding.
- [x] **`narratives_kappa_sample.py` written AND run** (2026-07-22) — the
      blind human-validation sample/kappa script the Narrative strand still
      needed, same discipline as `election_arm/validate_kappa.py` (blind
      sample + separate LLM-answer-key file, human fills `human_narrative`
      independently, kappa computed only after). `sample` mode run against
      the completed batch pass above -> `narratives_validation_sample.csv`
      (80 claims, blind) + `narratives_validation_llm_key.csv` (hidden LLM
      answers). **`kappa` mode now run** (2026-07-22) — human_narrative
      filled by two people who discussed disagreements and combined their
      opinion into one consensus label per claim (same "consensus gold
      standard" method as the original 80-claim `KAPPA_RESULTS.md` bake-off
      validation, not two independently-graded files), which is why this is
      a human-vs-LLM comparison only, not also a human-vs-human one.
      **Result: kappa = 0.58, raw agreement 68% (n=80)** — Landis-Koch
      "moderate," just under the 0.6 "substantial" bar this project uses
      elsewhere. 26 disagreements -> `narratives_validation_disagreements.csv`.
      Confusion is concentrated at CLOSE, related labels, not random or a
      polarity flip: `sound_fundamentals` vs `recovery_normalcy` (8 of 26,
      the single biggest confusion), `temporary_setback` vs `panic_fear` (5),
      `recovery_normalcy` vs `temporary_setback` (3), `none` vs
      `recovery_normalcy` (3) — only 1 of 26 is a full optimistic/pessimistic
      flip (`sound_fundamentals` vs `panic_fear`). **This matters directly
      for the "Full-corpus LLM pass" result above**: roughly 18 of the 26
      disagreements cross the complacent/non-complacent boundary (COMPLACENT
      = new_era + sound_fundamentals + temporary_setback), so the
      panic_fear-least-accurate / calm-controls-more-complacent-than-crises
      finding is validated to a MODERATE degree — real signal, not random —
      but should be reported with that kappa attached, not as a clean,
      settled result. A finer-grained taxonomy (e.g. splitting
      `sound_fundamentals` from `recovery_normalcy` more explicitly in the
      prompt) is the natural next step if this strand gets more investment.
- [x] **`bill_arm/.env` given canonical `OPENAI_API_KEY=...` line**
      (2026-07-22). The file's existing Groq/OpenAI/Pinecone/GDELT entries
      are human-readable `Label: value` notes (by design, for
      `grade_claims.py`'s regex-based multi-key loader), not valid
      `KEY=value` — added a proper line alongside them so standard
      `os.environ`/`python-dotenv` loading also works, without touching the
      label lines other scripts already parse directly.
- [x] **Stale "Known limits" note on the LOC recall-audit fixed** — it said
      "follow-up scrape in progress," but the follow-up scrape was actually
      completed 2026-07-18 and merged into the main corpus 2026-07-19 (see
      "Corpus grown and merged in" above); no scrape is outstanding.
- [x] **Cleanup: removed now-redundant intermediate/backup CSVs** —
      `claims_{raw,graded}_expanded.csv`, `claims_{raw,graded,scored}_pre_
      {dedup,expansion}_backup.csv`, `claims_graded_{newprompt,v2}.csv`,
      `search_log_baseline_2026-07-16.csv`. All were safety copies or
      intermediate outputs from the 2026-07-18/19 recall-audit rescrape and
      dedup-merge, which is complete and already reflected in
      `claims_raw.csv`/`claims_graded.csv`/`claims_scored.csv` (verified row
      counts match the documented final numbers above). **Currently staged
      for deletion but NOT committed** — flagging for a look before this
      lands in a commit, since git history still has these files' last
      committed contents (`94a289c`) if anything here turns out to still be
      needed.
- [x] **Spot-check: does the regex candidate-sentence extractor
      (`newspaper_scraper.py`'s `extract_claims`, FUTURE_MARKERS + JUNK
      filter within ±2 sentences of a search-phrase hit) actually have good
      recall, or does whole-page LLM extraction find something different?**
      (2026-07-22). Took one real cached page with 5 regex-found candidates
      (`sn84026749/1930-07-02`, "business outlook" hits), fed the SAME raw
      24k-char OCR text to gpt-4.1-mini with an election-arm-style whole-
      page extraction prompt (find every forward-looking economic/market
      statement, not anchored to any search phrase). **Result: 7 claims
      found, ZERO exact-sentence overlap with the regex's 5** — e.g. "worst
      of the decline in freight traffic is over," "we expect the rally to
      carry further," "Many sound stocks... should eventually sell much
      higher" were all present on the page and clearly forward-looking, but
      never within ±2 sentences of a "business outlook" hit, so the regex
      window never saw them. This is a sharper problem than the already-
      documented "phrase lists bound recall": even on a page the phrase
      search DID correctly find, the regex sentence-window step drops
      predictions that use different language than the trigger phrase.
      **n=1 page — anecdotal, not yet a corpus-wide recall estimate.** Next
      step if pursued: run the same whole-page-LLM-vs-regex comparison
      across a real sample (10-20 pages, multiple episodes) to get an actual
      recall-gap number instead of one example.
      User has H200 GPU access and wants to explore running this
      (or a full corpus re-extraction) via a local model (e.g. Llama-70B)
      instead of paid API calls. Noted as a good fit but not set up here:
      `grade_claims.py` already takes an arbitrary OpenAI-compatible
      `--base-url` (how it already hits Groq/DeepSeek), so a local vLLM/TGI
      server would plug in with no code changes — just needs the endpoint
      stood up on their end.

## Not done / next up

- [ ] **Re-extract the ProQuest corpus with `extract_llm.py` (in progress,
      2026-07-24).** The 9-window ProQuest export on branch
      `proquest-tdm-integration` (commit `9da000e`) was labelled by the crude
      `extract_gpt.py` (single-shot prompt, 12k-char hard truncation, no
      hallucination guard). Replacing it with `extract_llm.py` (chunked, strict
      schema, quote-grounding guard) run against the same in-VM GPT proxy — the
      raw parsed text (`data/raw/proquest_economy_{window}.jsonl`) is already in
      the VM, so no dashboard rebuild is needed, only the extraction step.
      `JeremysShit/PASTE_INTO_VM.txt` is a generated paste-helper (base64 blob +
      per-window commands) for the VM terminal; NOT meant to be committed. Two
      ProQuest-specific gotchas baked into that helper: (a) the export must drop
      `extract_llm.py`'s verbatim-text field **`quote`** (the old
      `strip_for_export.py` only drops `claim_text`), or the export leaks full
      text; (b) `extract_llm.py`'s 429 handler was patched to recognise
      ProQuest's daily-cap wording (`day rate`/`cost/day`) so it stops cleanly
      instead of crashing on the quota reset. Also now covers `covid_2020`,
      added since the last export. **4o-mini quality-gating step:** ProQuest is
      forced onto `gpt_4o_mini` (only in-VM proxy model; can't reach the OpenAI
      API from the locked VM), weaker than the `gpt-4.1` used on loc/nyt. To
      quantify the gap rather than hand-wave it, the gold-eval harness
      (`grade_claims.py` + `gold_extraction/{eval_extraction,gold_pages,
      gold_claims}`) gets transferred into the VM (`GOLD_TRANSFER_TO_VM.txt`,
      another generated base64 paste-helper, not for commit), 4o-mini is run
      over the same 16 gold pages via `extract_llm.py`, and scored with
      `eval_extraction.py` for precision/recall/F1 comparable to `RESULTS.md`.
      Only the metrics JSON leaves the VM. Reporting rule stands: never pool
      4o-mini ProQuest labels with 4.1 loc/nyt labels — always split by
      `source`. **4o-mini gold result (2026-07-24): extraction F1 0.475**
      (prec 0.42, rec 0.54) — bottom tier of the extraction bake-off, tied with
      Llama-3.3-70B (0.522)/flash-lite (0.48), far below gemini-3.5-flash's 0.79
      (which can't run in the VM). Note this is the EXTRACTION task; gpt-4.1's
      0.89/0.90 is the GRADING kappa, a different job — don't conflate. Labels
      on found claims are fine (`direction` kappa 0.83); the weakness is
      precision. FP audit (`--verbose`): all 38 FPs fall into categories the
      prompt already forbids at length (present-conditions, elections, policy
      advocacy, conditionals, ads) — so a prompt tweak can't help and a verify
      pass would be ~F1-neutral from rec 0.54 while doubling calls against the
      daily cap. Decision: no re-prompt, no verify; lean on the free
      `scope==national` downstream filter (most FPs are industry/regional/
      non-economic and never enter national NBER scoring anyway).
- [ ] **Schema-adapter needed before the reparsed corpus can score
      (2026-07-24).** `extract_llm.py`'s output is NOT drop-in for
      `analyze_economy.py`: the scorer keys on `predicted_state_at_horizon`
      (recession/expansion) and `hedged` (bool) and needs `window`/
      `window_kind`/`source`, but `extract_llm.py` emits `direction`
      (improve/worsen/no_change/unclear) + `confidence` (assertive/hedged) and
      injects none of the window metadata. Fed straight in, every `hit` is NaN.
      Fix: `election_arm/adapt_llm_economy.py` (Mac, label-only) maps
      worsen→recession, improve/no_change→expansion, unclear→drop,
      confidence→hedged, injects window/window_kind (from `windows_economy.csv`)/
      source=proquest, and applies the `scope==national` filter. One judgment
      call: `no_change→expansion` (a "no recession coming" call is a real
      expansion prediction, wanted esp. in placebo windows) — one-line switch to
      drop instead. Pipeline is now extract → strip → **adapt** →
      `analyze_economy.py`. **`adapt_llm_economy.py` written + unit-tested
      2026-07-24** (synthetic export lines through the real mapping: worsen→
      recession, improve/no_change→expansion, unclear+non-national dropped,
      no_predictions preserved). Lives on branch `proquest-tdm-integration`.
- [ ] **ProQuest scope decision (2026-07-24): finish `covid_2020` only; gate
      the other 8 on a proxy-model check.** Reasoning: post-1963 full text is a
      walled-garden monopoly (LOC ends 1963; NewsBank/Newspapers.com wall it the
      same as ProQuest; GDELT is 2015+ metadata) and the credible post-1963
      numbers already exist via the **NYT+gpt-4.1** corpus (done). ProQuest's
      only edge is depth (full body vs NYT headline+lead) + publisher breadth,
      but it's locked to gpt-4o-mini (F1 0.475) by the no-internet VM. So: (1)
      finish `covid_2020` (already parsed, 4,209 XMLs) end-to-end as a pipeline
      smoke-test AND a standalone probe; (2) run the proxy `/models` endpoint —
      if it serves anything above `gpt_4o_mini` (some TDM deployments have full
      `gpt-4o`), the ~1hr-each dashboard rebuild of the other 8 windows becomes
      worth it; if mini-only, don't rebuild — lean on the NYT-4.1 corpus and use
      ProQuest only as a narrow robustness/diversity note. Caveat: covid alone
      scores its own hit-rate but can't power crisis-vs-placebo (needs the calm
      windows rebuilt too).
- [ ] **Proxy `/models` check done: `gpt-4o` (full) is available, and it does
      NOT clear the bar (2026-07-24).** gpt-4o gold: F1 0.537 (prec 0.465, rec
      0.635) vs mini's 0.475 (prec 0.42, rec 0.54). The +0.06 is all recall —
      precision barely moved and it's the *identical 38 false positives*, proving
      the precision ceiling is GPT-family over-extraction on these pages (ignores
      the exclusion list), NOT a model-size gap; gemini-flash hits prec 1.00/0.84
      / F1 0.79 but can't run in the locked VM. 4o's `direction` kappa is also
      WORSE (0.70 vs mini's 0.83), and direction is what maps to the scored
      recession/expansion call. At ~10-15x mini's per-token cost (and the daily
      cap is a *cost* cap, so ~10-15x faster exhaustion over ~30k articles), 4o
      is not worth it. **Decision: no full 9-window rebuild on either model;
      finish `covid_2020` on mini as a pipeline test + single-window robustness
      anecdote; the NYT+gpt-4.1 corpus remains the post-1963 story.** Open cheap
      check: re-eval mini on the `scope==national` subset (the claims that
      actually enter scoring) to see if effective precision is better than the
      all-claims 0.46.
- [ ] **Pivot away from ProQuest to the Fed Greenbook as a 4th benchmark
      (2026-07-24).** Both VM-available OpenAI models cap at ~0.5 F1 / ~0.46
      precision on ProQuest extraction (gemini, the 0.79 winner, can't run in the
      locked VM), and the post-1963 *newspaper* problem is unfixable either way —
      NYT is headline-only (recall-capped), ProQuest is deep-but-noisy. Rather
      than keep fighting newspaper copyright, added the **Philadelphia Fed
      Greenbook/Tealbook Data Set** (Fed Board staff real-time forecasts, monthly
      by FOMC meeting, **Jan 1966–Dec 2020**) as forecaster #4 next to SPF/
      Livingston/Michigan. It's structured numbers (no extraction, no LLM, no
      precision problem), scored on the identical INDPRO/NBER rule via
      `greenbook_benchmark.py` (mirrors `spf_benchmark.py`, reuses its
      `read_xlsx_robust` Philly-Fed loader). Fit is BENCHMARK-only: it has none
      of `model.py`'s newspaper features (publisher/voice/hedging/text), so it
      does NOT feed the claim-accuracy model. Coverage note: the 5-yr FOMC
      confidentiality lag means it never reaches 2021–2026, but every post-1963
      crisis window (oil_1973 … gfc_2008, **covid_2020**) sits inside 1966–2020,
      so it costs nothing. `greenbook_benchmark.py` scaffolded + syntax-checked;
      PENDING a one-click manual download of the JS-gated Row-Format xlsx to
      `cache/greenbook_row_format.xlsx`, then `--inspect` to map the real
      sheet/date/forecast-column names (the only 3 constants left to finalize).
      UPDATE 2026-07-24: user downloaded the **All-Column-Format** variant (a
      folder of per-variable xlsx, columns = editions, rows = target quarters),
      not the single-workbook Row Format the scaffold assumed — loader needs
      adapting. Inspected the real layout: `gRGDP` has **490 editions
      (1967-03-29 → 2020-12-04, covid included)**; exact modeling-row counts are
      **1,804** (edition × horizon, +1..+4q ~1yr band) / **2,735** (all horizons)
      / **480** (one call per edition). Folder also carries gIP (INDPRO-scored,
      matches newspapers), UNEMP, gPCPI, +11 more variables.
- [ ] **Forecast-credibility model planned (2026-07-24, `forecast_credibility_
      PLAN.md`).** "Given a forecast + the economy's state when it was made, P(it
      comes true)." Small tabular (~1.8-2.7k rows × ~14 features) → logistic
      regression primary + shallow trees secondary; NO neural net (too small,
      needs interpretability). User's design: **3 models — Greenbook-only,
      Livingston-only, Combined — compared**, with a `source` tag throughout.
      Guardrails locked to keep the comparison fair: shared (intersection)
      feature set for the head-to-head (source-native extras — Livingston
      dispersion, Greenbook gap — reported separately); report full-era AND the
      1967-2020 overlap; add cross-source transfer (train A→test B ⇒ is
      credibility a general property?); source-stratified reporting so Livingston
      (~few hundred rows) isn't drowned by Greenbook volume. Same discipline as
      model.py: split by era not random, PR-AUC/ROC/Brier/calibration, real-time
      state features only. Build order: adapt greenbook loader → greenbook_scored
      .csv → forecast_credibility.py.
      SOURCES LOCKED (user, 2026-07-24): **US-only, three forecasters —
      Greenbook + Livingston + SPF individual microdata.** SPF adds forecaster
      IDENTITY (individual responses, `Individual_RGDP.xlsx` 683 KB confirmed
      curl-able), multi-variable, dispersion, and the **Anxious Index**
      (P(GDP decline) — a ready-made recession-probability). EXCLUDED by user:
      Fed SEP (too recent) and ALL international (IMF/OECD) — keeps US scope at
      the cost of turning-point power (accepted; frame as calibration).
      Provenance discipline REQUIRED since 3 different forecaster types are being
      merged: every row carries core harmonized fields + a `source`/`forecaster_id`
      tag + never-dropped provenance columns (`source_file`, `variable_native`,
      `horizon_native`); source-native extras namespaced (`livingston_dispersion`,
      `greenbook_output_gap`, `spf_forecaster_track_record`) and kept out of the
      shared-feature head-to-head; a variable crosswalk + a per-build merge audit
      (rows × source × variable × era). Reported caveat: sources share the same
      ~8–12 US recessions, so combined n ≠ independent episodes.
- [ ] **Scale the regex-vs-LLM recall spot-check to a real sample** (see
      "Spot-check" entry above) — 10-20 pages across multiple episodes,
      to get an actual recall-gap estimate instead of the n=1 anecdote.
      If the gap holds up, whole-page LLM re-extraction over the ~4,300
      already-cached pages (no new downloads needed) is the natural
      follow-up, ideally via a local model on the user's H200s rather than
      paid API calls.
- [ ] Spec's second model (predict economic *state* from press, not "was this
      claim right"). Different unit of observation (time period, not claim);
      at episode level only 10 rows, needs a month-level corpus expansion.
      Deliberately deferred (2026-07-21, user call): treated as a separate
      future project rather than folded into this pass, since it needs new
      corpus engineering and a new unit-of-analysis design, not just
      execution of an already-spec'd plan.
- [ ] **Election arm kappa validation (Stage 3) — the only remaining step.**
      Stages 2/4/6 are done on real data (see "Full election corpus...
      extracted" above): 23-claim blind sample sitting in
      `data/validation_sample.csv`, needs TWO independent human graders to
      fill in `grader_A`/`grader_B` before `validate_kappa.py kappa --arm
      elections` can run — cannot be automated (see "Validation integrity"
      below). Separately, the election arm's LOC (pre-1963) side has never
      been downloaded — NYT-only so far, only 24/119 claims are national-
      scope+scorable, and `model.py`'s first real run scored below its own
      majority-class baseline at this n; more claims per cycle (LOC download
      + a state-results file for `SCORE_STATES`) matters more than kappa
      here for getting a meaningful model number.
(Narrative Economics kappa validation — done, see "kappa mode now run"
above: kappa=0.58, moderate. Consider a finer-grained taxonomy for
`sound_fundamentals` vs `recovery_normalcy` specifically, the single biggest
confusion pair, if this strand gets more investment.)
- [ ] **Review the staged backup-CSV deletion before it's committed** (see
      "Cleanup" above) — currently sitting in the index, not yet committed.
- [ ] **A new LOC subset scrape is in progress elsewhere (confirmed with
      user — expected, not a stray process), first seen 2026-07-22 ~14:54,
      not yet merged.** `append_loc_claims.py` — a safer version of the
      2026-07-18 recall-audit merge workflow: run `newspaper_scraper.py`
      with `--out`/`--log-out` pointed at separate files instead of
      overwriting `claims_raw.csv`/`search_log.csv` directly, then merge
      with dedup-on-`page_url` + continued `claim_id` sequence (idempotent,
      safe to rerun). Two subset files in flight,
      `claims_raw_newloc.csv` (115 claims, appears done) and
      `claims_raw_newloc2.csv` (actively growing row-by-row as of this
      writing — exact count not tracked here, check the file directly for
      current size). `claims_raw.csv`/`claims_graded.csv`/
      `claims_scored.csv` remain untouched — don't run `append_loc_claims.py`
      on either subset
      until both settle; merging mid-scrape would lock in a partial corpus.
- (`results_by_region.csv` and `fig_epu_vs_accuracy.png`/`fig_geography.png`/
  `fig_three_way_benchmark.png` changed this pass too, but only as a
  mechanical regeneration from the already-documented `score_claims.py`/
  `regret_scoring.py` reruns above — not new findings, no action needed.)

## Known limits / needs improvement

- **`election_arm/data/search_log.csv` gains a few duplicate rows every time
  `test_offline.py` runs** — the download-side offline tests exercise the
  real `search_pages`/`search_phrase` functions against mocked HTTP
  responses, and those functions write to the real search log as a side
  effect ("no network needed" is true, "no filesystem writes" isn't). Only
  ever adds identical mock rows (`elections/1948/"will be elected"` and
  `economy/gfc_2008/"recession likely"`), so it doesn't corrupt the real
  corpus-transparency exhibit, just adds cosmetic noise. Not fixed — low
  priority, but if this file goes in front of a reviewer, strip test rows
  first (`timestamp` values line up with whenever tests were run).

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
- **Sampling skew**: DC is still ~29% of the LOC-era corpus (261 claims from
  the Evening Star alone) — now coded as its own `political hub (DC)` bucket
  rather than folded into financial centers (see Done), but the underlying
  publisher concentration itself is unchanged.
- One search term (`"business outlook"`) produced 606 of 1,324 claims (46%,
  original pre-expansion corpus). **Recall audit done 2026-07-18** (see
  Done): confirms the corpus was sampled, not cherry-picked, but also found
  9/32 term/episode searches fetched under 50% of LOC's own available hits
  (page-cap truncation, not missed phrasing) — 20,331 hits never fetched.
  **Follow-up scrape completed 2026-07-18 and merged 2026-07-19** (see Done,
  "Corpus grown and merged in"): `--pages-per-term 100` rerun on the 7
  under-recall crisis episodes added 1,550 genuinely-new claims after
  dedup, growing `claims_raw.csv`/`claims_graded.csv` 2,644 -> 4,194 rows.
  This note previously said "in progress" — stale; no further scrape is
  outstanding.
- The retrospective-vs-prediction boundary was the main leakage risk flagged
  during rubric design; it's now an explicit rubric rule (retrospectives →
  not a prediction) and was part of what the reconciliation process fixed.
  **Spot-checked 2026-07-18** (see Done): 30-row hand sample found 1/30 (3%)
  clear violation, 2/30 (7%) soft/debatable, 27/30 clean — small non-zero
  label noise, disclose if the state-prediction model's accuracy is
  presented.
- `JeremysShit/` has no venv of its own (`test_offline.py` needs pandas; run
  it with `bill_arm/.venv/bin/python`, which also has `openpyxl` now).
- Groq (5 keys), NYT (now 6 keys), Pinecone, GDELT, and OpenAI keys were at
  various points pasted in plaintext in chat and now live in `bill_arm/.env`
  (gitignored, confirmed untracked — but still plaintext on disk and were
  previously exposed in chat history, so technically compromised regardless).
  User re-supplied them 2026-07-18 specifically to unblock NYT downloads and
  claim grading; still not rotated. Real exposure still stands if this
  becomes relevant again later.

## Needs to be done by you (not automatable / requires a decision)

- **Rotate both exposed NYT API keys at developer.nytimes.com.** They were
  pasted in plaintext into `CONTINUE_HERE_nyt.md`, committed in `1e201bc`,
  and later stripped from the working copy — but a working-tree edit doesn't
  undo the git-history exposure, and one of the two matched `bill_arm/.env`'s
  live key. Not yet confirmed done as of 2026-07-22.
