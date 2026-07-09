# Who Saw It Coming? — BU RISE Project (merged, two arms)

American predictions 1900–2010, two arms sharing one design:
- **Economy arm (this folder):** newspaper predictions around 7 crises + 3 calm controls
  (LOC, 1905–1958), scored vs. NBER/FRED, benchmarked against Livingston Survey economists
- **Election arm (`election_arm/`):** Vincent's pipeline — presidential forecasts 1896–2008
  (LOC + NYT API) scored against actual winners; see `election_arm/README.md` for the
  drop-in checklist and shared schema

Merged design: **[project_plan_v3_merged.md](project_plan_v3_merged.md)**
(economy-arm details: [project_plan_v2.md](project_plan_v2.md))

## The pipeline

```
1. SCRAPE      newspaper_scraper.py      LOC Chronicling America -> claims_raw.csv + search_log.csv
2. GRADE       grade_claims.py           LLM (DeepSeek) applies the rubric -> claims_graded.csv
                                         + validation_sample.csv for human double-coding (Cohen's kappa)
3. SCORE       score_claims.py           vs. NBER dates + FRED CPI/INDPRO/UNRATE
                                         -> hit rates, Brier scores, publisher leaderboard,
                                            calibration (assertive vs. hedged), voice analysis,
                                            crisis-vs-control comparison, famous_calls.csv,
                                            head-to-head vs. Livingston, figures/
4. MODEL       model.py                  which factors predict a claim being right?
                                         EPISODE-grouped split (no leakage) + permutation
                                         importances + EPU-at-claim-time feature
4b. TIER 2     tier2_analysis.py         policy uncertainty (historical EPU 1900-2014) vs
                                         accuracy; geography (regions, financial centers);
                                         Michigan household sentiment as third benchmark
5. SURVEY ARM  BU_RISE_forecast_analysis_FIXED.ipynb   Livingston 1946-2026 analysis
                                         (verified runs end-to-end locally, 2026-07-09)

TESTS          test_offline.py           21 offline checks, mock API responses through
                                         the real functions — run after any change
```

## Quick start

```powershell
python newspaper_scraper.py --pages-per-term 30        # ~30 min, cached & resumable
python score_claims.py --claims claims_raw.csv --heuristic   # end-to-end test, no API key
$env:DEEPSEEK_API_KEY = "sk-..."
python grade_claims.py --limit 20                      # cheap test of the LLM rubric
python grade_claims.py                                 # grade everything (~cents)
python score_claims.py                                 # real results + figures
```

Then: two people independently fill the `human_*` columns of `validation_sample.csv`
and run `python grade_claims.py --kappa validation_sample_filled.csv` (target κ ≥ 0.7).

## Files

| File | What it is |
|---|---|
| `newspaper_scraper.py` | 10-window scraper: 7 crisis episodes + 3 calm placebo controls (loc.gov JSON API, polite + cached) |
| `newspaper_scraper_starter.py` | Minimal first version, kept for reference |
| `grade_claims.py` | LLM rubric grading + human-validation kappa tooling |
| `score_claims.py` | Ground-truth scoring, leaderboard, Livingston head-to-head |
| `BU_RISE_forecast_analysis_FIXED.ipynb` | Survey-arm analysis notebook (Colab or local) |
| `medians.xlsx`, `Dispersion2.xlsx`, `MedianGrowthRate.xlsx` | Livingston Survey data (Philadelphia Fed) |
| `claims_raw.csv`, `search_log.csv` | Scraper output |
| `claims_graded.csv`, `validation_sample.csv` | Grading output |
| `claims_scored.csv`, `results_by_episode.csv`, `publisher_leaderboard.csv`, `famous_calls.csv`, `figures/` | Scoring output |
| `model.py` | Accuracy-factors model (episode-grouped split) |
| `test_offline.py` | Offline test harness (21 checks, no network) |
| `election_arm/` | Vincent's election pipeline lands here (schema contract in its README) |
| `cache/` | Every downloaded page/series (safe to delete; will re-download) |

## Notes & gotchas

- The legacy `chroniclingamerica.loc.gov` API is retired — everything uses the
  loc.gov JSON API. LOC full text ends at **1963**; the Livingston arm carries 1963–2010.
- FRED CSV downloads hang unless the request sends a browser-like User-Agent
  (handled inside `score_claims.py`).
- CPI/IP/GDP index rebasing artifacts in the Livingston data are detected and
  dropped in the notebook (data-quality cell lists every dropped survey).
- Claims before 1913 (prices) / 1948 (employment) are marked unscorable, not guessed.
