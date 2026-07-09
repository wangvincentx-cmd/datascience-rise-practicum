# Who Saw It Coming? Newspapers, Polls, and a Century of American Economic Predictions (1900–2010)

## BU RISE — Full Project Plan v2.1 (updated 2026-07-09)

> **Status update (v2.1):** the whole pipeline is now implemented and tested —
> `newspaper_scraper.py` (all 7 episodes), `grade_claims.py` (DeepSeek/LLM rubric grading
> + Cohen's κ tooling), `score_claims.py` (NBER/FRED ground truth, hit rate + Brier,
> publisher leaderboard, Livingston head-to-head, auto-generated figures). The Livingston
> notebook was verified to run end-to-end locally and its growth-file robustness check was
> fixed (`G_BP_To_12M`, full 1946– coverage). See README.md for the run order.
> Preliminary: Livingston directional hit rate 1946–63 = **54.4%** (n=68) — the newspaper
> number to beat.

This plan merges the team's newspaper/poll scraping idea with the already-working
Livingston Survey pipeline (`BU_RISE_forecast_analysis_FIXED.ipynb`). Two arms, one question:

- **Newspaper arm (1900–1963):** scraped predictions from Library of Congress *Chronicling America*
- **Survey/poll arm (1946–2010):** Livingston Survey median forecasts (already downloaded & running)
- **Head-to-head window (1946–1963):** the only period where both exist — the direct newspapers-vs-polls comparison
- Newspapers cannot cover 1963–2010 (LOC full text ends 1963; later archives are paywalled and
  un-scrapable). The survey arm carries the story from 1963 to 2010.

---

## 1. Research Questions

1. **Newspapers vs. experts:** In 1946–1963, were newspaper economic predictions more or less
   accurate than the Livingston Survey economists? (directional hit rate + Brier score, identical
   scoring for both)
2. **Publisher ranking:** Which newspapers made the most accurate predictions? (min. 15 graded
   claims per publisher to qualify)
3. **Crisis type:** Did prediction accuracy depend on the kind of instability — banking panic
   (1907), deflationary bust (1920–21, 1929–33), policy recession (1937), war transition
   (1945–46), ordinary recessions (1948, 1957)?
4. **The long arc (1946–2010):** Did expert forecast accuracy improve over time, and did accuracy
   at turning points ever improve? (this is the existing Livingston analysis, kept intact)
5. **Optimism bias:** Were newspapers systematically more optimistic than what happened?
   (signed direction bias by episode)

**Falsifiable hypothesis:** newspapers and experts perform similarly in normal times, but at
turning points *both* fail in the optimistic direction — professionalized forecasting (polls,
surveys, economists) improved calibration in calm eras without improving crisis prediction.

---

## 2. Episodes (the "things to modify/measure" across 1900–2010)

Scraping the entire century is infeasible and unnecessary. Sample **prediction windows** just
before/inside each episode, score claims against what happened 6–12 months later.

| # | Episode | Claim window | Outcome to predict (ground truth) |
|---|---------|--------------|------------------------------------|
| 1 | Panic of 1907 | Oct 1907 – Jun 1908 | Recovery timing (NBER trough Jun 1908) |
| 2 | Depression of 1920–21 | Jun 1920 – Mar 1921 | Deflation & rebound (CPI −11% in 1921; trough Jul 1921) |
| 3 | Crash → Great Depression | Nov 1929 – Dec 1930 | "Prosperity around the corner" vs. 1930–33 collapse |
| 4 | Recession of 1937–38 | Sep 1937 – Mar 1938 | Depth/duration (trough Jun 1938) |
| 5 | Postwar reconversion | Aug 1945 – Jun 1946 | Predicted new depression that never came — famous mass failure |
| 6 | 1948–49 recession | Jul 1948 – Jun 1949 | First head-to-head vs. Livingston |
| 7 | 1957–58 recession | Aug 1957 – Jun 1958 | Second head-to-head vs. Livingston |
| 8–12 | 1973–75 oil, 1979–82 Volcker, 1990–91, 2001, 2007–09 | — | Survey arm only (Livingston, already computed) |
| C1–C3 | **Calm controls:** 1905–06, 1925–26, 1955–56 | same terms, mid-expansion | Placebo windows: base-rate accuracy/optimism, so crisis failure isn't confused with "newspapers are always wrong" |

Additional analyses baked into the pipeline (v2.1): **calibration** (assertive vs. hedged hit
rate — overconfidence test), **voice** (editorial vs. quoted expert vs. official — whom to
trust), **crisis vs. control** comparison, and **named-forecaster extraction**
(`famous_calls.csv` — best/worst calls sidebar for the poster).

Ground truth sources (all free):
- **NBER business cycle dates** (peak/trough months, back to 1854) — the primary yardstick
- **CPI** (BLS, monthly from 1913); 1900–1912 via NBER macrohistory / Rees index
- **Industrial production** (Fed G.17 from 1919; Miron–Romer index before)
- **Unemployment** annual historical estimates (Weir/Romer series) — pre-1948 is an estimate; state this as a limitation
- **Dow Jones Industrial Average** (daily from 1896) for market-direction claims

---

## 3. Data Pipeline — Newspaper Arm

### 3.1 Source & API (verified working 2026-07-08)

The legacy `chroniclingamerica.loc.gov` API is retired; use the **loc.gov JSON API**:

```
Search:   https://www.loc.gov/collections/chronicling-america/
          ?qs=<phrase>&ops=PHRASE&searchType=advanced&dl=page
          &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&fo=json&c=<page_size>
Full OCR: fetch the result's resource URL with &fo=json, then download
          response["resource"]["fulltext_file"]  (plain text of the whole page)
```

Verified example: `"financial panic"`, 1907-10-01 → 1907-12-31 = **885 pages**, each with
publisher (`partof_title`), date, state, page URL, and retrievable full OCR text.

Etiquette: no API key needed; sleep ≥1 s between requests, exponential backoff on 429/5xx,
set a descriptive User-Agent. Budget ≈ 1.5 s/page → 500 pages ≈ 12 min. Cache every download
to disk so nothing is fetched twice.

### 3.2 Search terms (per episode, 2–4 phrases)

Prediction-bearing phrases, not just topic words: `business outlook`, `prosperity will`,
`business depression`, `hard times ahead`, `return of prosperity`, `financial panic`,
`business revival`, plus episode-specific ones (e.g., 1929–30: `prosperity is just around`,
`business conditions will improve`). Keep a `search_terms.csv` log — the term list is a
methods-section artifact.

### 3.3 Claim extraction (scrape → candidate sentences)

1. Search each term/episode window; collect page hits (dedupe by page URL).
2. Cap per episode: **300 pages max**, sampled across states/publishers if over.
3. Download full OCR text; split into sentences (OCR is dirty — split on `.` + length filters).
4. Keep sentences containing the search term **±2 sentences** that also contain a
   future marker: `will | expect | predict | forecast | outlook | ahead | coming |
   by (spring|summer|fall|winter|next year) | months`.
5. Write `claims_raw.csv`: `episode, publisher, state, date, page_url, quote`.

Target: **60–100 candidate claims per episode** after filtering (≈400–700 total).

### 3.4 Grading rubric (the "scoring rubric" — make it a poster artifact)

Each claim gets:

| Field | Values |
|---|---|
| `is_prediction` | yes / no (kills ads, retrospectives, OCR garbage) |
| `topic` | general business / prices / employment / markets |
| `direction` | better / worse / no-change |
| `horizon` | ≤6 mo / 6–12 mo / vague |
| `confidence` | hedged ("may", "likely") / assertive ("will", "is certain") |
| `voice` | editorial / quoted expert / quoted official |

**Grading protocol (credibility requirement):** two team members independently grade a shared
20% sample → report **Cohen's κ** (target ≥ 0.7); resolve disagreements by discussion; then
split the rest. *Optional accelerator:* an LLM grades everything first and humans verify the
20% sample against it — but the human double-coding is what makes it defensible at a symposium.

### 3.5 Scoring

- A claim is **correct** if its direction matches the realized change of its topic's ground-truth
  series over its horizon (no-change = realized change within ±0.5 SD of that era's typical move).
- **Hit rate** per publisher/episode; **Brier score** using confidence as a crude probability
  (assertive = 0.9, hedged = 0.7). RMSE stays on the Livingston side only — scraped claims are
  directional, not numeric, so RMSE is the wrong tool there (document this reasoning in methods).
- **Head-to-head (1946–63):** convert Livingston numeric forecasts into the *same* directional
  calls (does the median forecast direction match realized direction?) so both sources are scored
  on an identical metric. Bootstrap CIs on hit-rate differences.

---

## 4. Survey/Poll Arm (already built — keep, don't rebuild)

`BU_RISE_forecast_analysis_FIXED.ipynb` already computes Livingston forecast errors, era
aggregation, naive-baseline skill scores, bias, and shock recovery for 1946–2026 (trim displays
to 2010 if the team wants a clean 1900–2010 frame). Open items carried over from the passoff:

1. **Fix CPI/IP index-rebasing artifacts first** (fake giant "actual changes" around rebases,
   e.g. CPI ~1988) — either drop flagged rows or switch to `MedianGrowthRate.xlsx`. This is the
   top data-quality task and it predates the newspaper work.
2. Verify `Dispersion2.xlsx` sheet structure on first real run of the dispersion cell.
3. Optional: SPF (1968–) cross-check — cut first if time runs short.

---

## 5. Analysis & Poster Figures

| Fig | Content | Research Q |
|---|---|---|
| 1 (headline) | Timeline 1900–2010: newspaper hit-rate per episode (points) + Livingston rolling error (line), NBER recessions shaded | all |
| 2 | Pipeline diagram: LOC API → claims → rubric → scoring (with a real 1907 quote as the example) | methods |
| 3 | Newspapers vs. Livingston, 1946–63: directional hit rate with bootstrap CIs | Q1 |
| 4 | Publisher leaderboard (≥15 claims), hit rate ± CI | Q2 |
| 5 | Direction-bias chart: share of "better" calls vs. realized outcome per episode — the optimism gap (1929–30 and 1945–46 are the stars: wrong in *opposite* directions) | Q3, Q5 |
| 6 | Era MAE / skill score from Livingston arm (existing figures 3–4) | Q4 |
| 7 | Rubric card + Cohen's κ + example graded claims (judges love seeing the instrument) | methods |

Headline finding candidates (falsifiable either way): "1945 newspapers predicted a depression
that never came; 1929 newspapers predicted a recovery that never came" — accuracy failure is
symmetric around optimism only sometimes.

---

## 6. Timeline (program week 2 of 6 starts now)

| Week | Newspaper arm | Survey arm | Deliverable |
|---|---|---|---|
| 2 | Scraper built & cached corpus for episodes 1, 3, 5 (starter script exists: `newspaper_scraper_starter.py`) | Fix rebasing artifacts | `claims_raw.csv` for 3 episodes |
| 3 | Remaining episodes scraped; rubric finalized; double-code κ sample | Verify dispersion cell | Graded claims for 3 episodes, κ reported |
| 4 | Finish grading; scoring vs. ground truth; head-to-head calc | Freeze Livingston figures | Figs 3, 4, 5 drafts |
| 5 | Robustness: does publisher ranking survive dropping any one episode? Sensitivity of "no-change" band | Rolling-error fig trimmed to 2010 | Figs 1, 6, 7; robustness notes |
| 6 | — | — | Poster assembly + practice talk |

**Fallback ladder (pre-committed):** behind after Week 3 → drop to 4 episodes (1907, 1929,
1945, 1957). Behind after Week 4 → drop publisher leaderboard (Q2), keep episode-level results.
The Livingston arm alone is still a complete poster — the newspaper arm can shrink without
killing the project.

---

## 7. Limitations (state honestly on the poster)

- Chronicling America ends 1963 and under-represents big metros (no NYT/WSJ) — findings describe
  the *regional* press; frame as a feature ("what ordinary Americans read")
- OCR errors → some claims lost/garbled; search recall is imperfect and term-dependent
- Keyword search finds prediction-flavored language, not all predictions (selection bias —
  mitigated by fixed term list applied identically to every episode)
- Pre-1948 unemployment figures are scholarly estimates
- Polls proper (Gallup 1936–) rarely made economic *predictions*; the Livingston Survey (an
  expert survey) is the fair "poll" comparator, and this substitution must be stated explicitly
- Confidence→probability mapping for Brier scores is a rubric choice; show hit rate alongside
