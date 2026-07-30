# Project handoff — "Did Anyone See It Coming?" (BU RISE economy arm)

**Written 2026-07-28. Self-contained on purpose.**

You are probably reading this in the Claude Desktop app, which has **no access to
this repository's filesystem**. Everything you need is therefore inline below —
every number here was verified against the committed data on 2026-07-28. Do not
assume you can open the files named in it; ask the user to paste anything you
need to see.

If a number in a conversation disagrees with a number here, **this file wins
unless the user says the analysis has been re-run.** Several figures in older
documents are stale, and the specific stale values are listed in §8.

---

## 1. What the project is

One research question: **did the American press see economic downturns coming?**

An LLM machine-reads Library of Congress newspaper pages (1900–1963), extracts
every forward-looking economic forecast, and a deterministic scorer marks each
one right or wrong against NBER recession dates and Federal Reserve (FRED) data.

**The rule the whole project rests on:** the language model decides *what was
predicted*; real economic data decides *whether it came true*. The model is never
asked whether a forecast was right — that would be hindsight grading itself.
Keep these separate in any new analysis or writing.

Team: Vincent Wang, Jeremy Liu, Bode Bosell (interns); Eugene Pinsky, Indrajit
Kalita (mentors). Deliverables are a symposium poster and a 300-word abstract.

---

## 2. CRITICAL — there are TWO corpora, and they are different instruments

This is the single most common mistake made about this project. A previous
session conflated them and produced a summary that was materially wrong.

| | **Corpus M — continuous** | **Corpus C — episodic** |
|---|---|---|
| what | every month 1900-01 → 1963-12, no gaps | 19 hand-picked windows (13 crisis, 6 calm control) |
| pages | 15,721 | — |
| claims extracted | 30,765 | 4,125 |
| scorable | **14,251** US-national | 1,428 |
| extractor | gpt-oss-120b, low effort | gpt-4.1 |
| quality | precision 0.615 / **recall 0.462** / F1 0.527 | κ = 0.89 (is_prediction), 0.90 (direction) |
| sampling | direction-neutral, outcome-blind | windows chosen *because* of what the economy did |

**Corpus M carries the headline findings.** It is outcome-neutral by
construction, which is the objection Corpus C is vulnerable to.

**Corpus C carries the professional benchmarks** (SPF, Livingston, Greenbook,
Michigan) and the fine-grained validation work.

**Never compare them claim-for-claim.** Different extractors, different error
profiles, different sampling. Aggregate monthly series tolerate Corpus M's weaker
extractor because per-claim error partly averages out over ~40 claims/month;
individual claim comparisons across corpora do not.

---

## 3. How the corpus was built (6 steps)

1. **Sample.** All 768 months, using a **fixed direction-neutral query set** —
   five phrases: `business conditions`, `business outlook`, `trade conditions`,
   `the business situation`, `financial outlook`. None implies a direction.
   *Why it matters:* the earlier version searched `return of prosperity`,
   `business revival`, `worst is over` — terms that only match optimistic copy —
   while the headline finding was "forecasts leaned optimistic." That is
   circular. Those terms also matched bank New Year **advertisements**.
2. **Extract.** The LLM reads the **whole page**, not keyword windows. A
   hallucination guard drops any quote not traceable to the page (0.3% of
   returns).
3. **Grade.** Fixed rubric: is it a prediction, topic, direction, hedging, voice,
   scope. Ads, OCR garbage, retrospectives and refusals-to-forecast are excluded
   by explicit rule.
4. **Scope gate.** Only `national` claims are scored. The split is national
   16,439 / regional 6,486 / industry 5,567 / foreign 2,217 — so **43% would
   otherwise be graded against the wrong yardstick.**
5. **Score.** Deterministic, no LLM. NBER chronology + FRED series with
   human-calibrated no-change bands. Proven by **33 known-answer tests**.
6. **Index.** 761 months yield ≥1 scorable national claim → a continuous monthly
   series of press expectations.

Total compute cost for the whole project: **$24.97**.

---

## 4. The findings (all verified 2026-07-28)

### 4.1 Method result — how you read the page decides what you find
Measured against a hand-built gold standard (16 pages annotated exhaustively
before any model ran, 52 forecasts, 44 documented boundary cases):

| extractor | precision | recall |
|---|---|---|
| keyword regex (the standard approach) | 0.61 | **0.27** |
| **gpt-oss-120b low effort — WHAT ACTUALLY RAN** | **0.615** | **0.462** |
| gpt-oss-120b default effort (not used) | 0.536 | 0.712 |
| gemini-3.5-flash 8k (not used — $403 vs $30 budget) | 0.84 | 0.73 |

Keyword search also fails *non-randomly*: it retrieves advertisements, and in one
documented case extracts a bullish 1907 forecast from a column headlined
*"Prophecies Gone Wrong"* — a report that forecasters were **wrong** — which then
scores against the Panic and manufactures a fake "nobody saw it coming" signal.

### 4.2 THE headline — the forecast mix never responded to the economy
- Downturn forecasts: **24.1%** in expansions vs **24.3%** in recessions —
  statistically identical.
- **~72% of all forecasts were upbeat**, in booms and busts alike.

This is the free-standing empirical fact. Everything else follows from it.

### 4.3 The consequence — accuracy collapses when it matters
- **58.8%** in expansions → **39.7%** in recessions.
- Gap **19.1 percentage points**, 95% CI [12.9, 24.4], block-bootstrapped by
  3-year period.
- Present as a *consequence* of 4.2, not an independent finding: NBER dates are
  both the scoring ground truth and the split variable.

### 4.4 Pessimists were right when it counted
| forecast | in expansions | in recessions |
|---|---|---|
| "business will improve" | 76% | **37%** |
| "business will worsen" | 21% | **61%** |

Not everyone got worse — pessimists got *better*. But the mix stayed ~3:1 upbeat
regardless, so the average fell anyway. **The press had the right analysis and
published it too rarely.**

### 4.5 Sixty years, no improvement
- Annual accuracy trends **+0.002 per decade (p = 0.87)**; 1900s 53.7% → 1960s 49.9%.
- **No publisher better**: 44.2%–56.4% across seven papers with ≥200 scorable
  forecasts.

### 4.6 Overall accuracy and breakdowns
Overall hit rate **0.513** (n = 14,251).

By what was predicted: improve 0.597 (n=8,962) · prices up 0.510 (n=1,359) ·
worsen 0.364 (n=2,577) · prices down 0.267 (n=868) · no-change 0.194 (n=485).

By topic: general business 56.1% (n=8,713) · other 47.9% (n=870) · markets 44.9%
(n=2,219) · prices 41.5% (n=2,275) · jobs 32.8% (n=174).

**Markets, prices and employment are all significantly worse than a coin flip** —
and the press published 10,400 of them. The only above-chance topic is vague
optimism about "business," and only because that is right most years by default.
Price accuracy tracks the era's inflation *regime*, not skill: after 1948
"prices will fall" was right **0 times out of 93**. Topic scores AUC 0.495 —
chance — out of fold.

### 4.7 October 1929 (the vivid case)
Share of national forecasts predicting improvement:
1929-06 76.5% · 07 93.3% · 08 86.4% · 09 64.3% · **10 (Crash) 86.1%** · 11 78.4%
· 12 76.2% · 1930-01 79.2%.

**In the month of the Great Crash the press was more bullish than in June.**
Forecasts printed Aug–Dec 1929 came true **20.8%** of the time (n=168). Found
with no episode labels and no outcome information.

### 4.8 The hit-predictor model (logistic regression)
Out-of-fold, leave-one-3-year-period-out over 21 blocks:

| model | ROC-AUC |
|---|---|
| how the forecast is written | 0.561 |
| the economy at print time | 0.538 |
| both, added together | 0.581 |
| **both, allowed to interact** | **0.647** |
| gradient boosting, same inputs | 0.617 |

Block-permutation test (200 shuffles of `hit` within each 3-year period):
observed **0.647**, null mean 0.478, **p = 0.005**.

**The mechanism — this is the actual finding, not the number.** An earlier
document reported a "macro null" (economy carries no information). That was a
**cancellation, not an absence**: optimistic and pessimistic forecasts respond to
identical conditions with **opposite sign** — 6-month stock return correlates
**+0.216** with optimistic forecasts coming true and **−0.283** with pessimistic
ones. One coefficient cannot serve both, so pooling cancels to nothing.

Permutation importance — the entire gain is two terms: direction × policy
uncertainty (0.038), direction × stock return (0.016). Topic 0.007, direction
0.005, **every macro main effect ≈ 0 or negative.** Industrial output growth and
inflation do nothing in every stratum — that part of the original null holds. The
signal is in *financial-market and policy-uncertainty* conditions, not real
activity.

### 4.9 Would a bigger model help? Measured, not assumed.
Full forecast text, **24,598 TF-IDF features**, linear model: **AUC 0.911 in
sample, 0.541 out of fold.** Several single predictors land *below* 0.5 out of
fold, meaning their in-sample ranking reverses in a new era.

**The binding constraint is ~21 independent time blocks, not model capacity.**
More parameters is the one intervention that provably cannot fix this. The
defensible use of a GPU here is the *extractor*, not the predictor.

### 4.10 Professional benchmarks (Corpus C) — nobody is better
Scored on identical ground truth:

| forecaster | hit rate |
|---|---|
| Federal Reserve Board staff (Greenbook, n=480) | **54.0%** |
| Survey of Professional Forecasters | 54.1% |
| Livingston economists | 54.4% |
| Michigan households | ≈55% |

They converge because **none of them forecast contractions.** SPF issues a
downturn call in ~0% of surveys; Greenbook's mix is 92% improve / 7% no-change /
1% worsen. The Greenbook predicted "worsen" **6 times in 490 forecasts and got 0
of 6 right.**

**The Greenbook mechanism (strongest supporting result).** Across 490 editions,
forecast dispersion collapses from a **4.01**-point SD at the nowcast to **0.90**
at 8 quarters, while the mean holds near **+2.8%**. Share of below-zero calls
falls 12.9% → **exactly 0.0%**. **At 6+ quarters out the staff never once
forecast negative growth — 0 of 545 forecasts across 54 years.** Beyond a year
the Greenbook *is* a trend forecast, and a constant cannot call a turning point
by construction.

Pre-recession record — all 8 NBER peaks 1969–2020 have a **positive** mean
1-year-ahead forecast. Only 1969, 1980, 1981 ever produced a single negative
call. **1990, 2001, 2007 (GFC — worst call still +1.45%) and 2020 (COVID) were
missed entirely.**

### 4.11 A finding that REVERSED (report this — it is a strength)
On the small crisis corpus (n=232), forecasts that swam against press consensus
looked *more* accurate: 52% vs 43%. On the continuous corpus (n=1,967) it
**reverses: 38.7% vs 53.3%** — contrarians were **worse**.

Small-sample noise from an outcome-selected sample. **Do not put the contrarian
claim on the poster.** This is the strongest argument for why the continuous
corpus was worth building.

---

## 5. Deliverables status

**Abstract** — `docs/Liu, Jeremy, ABSTRACT.docx`, 300 words exactly, Times New
Roman, title 14pt, body 12pt, superscripted affiliations, single-spaced,
4 references (uncounted). Emailed to rise@bu.edu. Authors: Vincent Wang¹˒²,
Jeremy Liu¹˒³, Bode Bosell¹˒⁴, Indrajit Kalita¹, Eugene Pinsky¹.

**Poster** — `RISE_Poster_2026.pptx`. Known gaps as of this writing: the
Discussion/Conclusions box contains only Limitations (no conclusions); the
strongest figure (`figA_mechanism`, the 24.1/24.3 bars) is not on it; 1929 is not
on it; the professional benchmark is not on it; figures have no takeaway
captions. 16 finished figures exist in `figures/poster_figures/`; the poster uses
about five.

---

## 6. Open / unresolved — needs a human decision

1. **The disagreement measure contradicts itself.** Two measures in the repo give
   opposite answers about whether forecaster disagreement predicts accuracy:
   dividing by *all* claims gives a null (53.7/48.6/51.4); dividing by
   *directional* claims gives a clean monotone relationship (56.6/49.9/47.3).
   The project's own poster draft says **"do not print either way yet."**
2. **Extraction F1 is documented inconsistently** — `RESULTS_MACRO.md` says
   ≈0.61, `RESULTS_MONTHLY.md` says 0.527. **0.527 is correct** (low reasoning
   effort ran; confirmed three ways, including that spend was $7.47 vs $15.48
   projected at default effort).
3. **The gold standard is model-adjudicated, not human.** A two-person recode of
   ~40 claims is the outstanding step before publication.
4. **Never measured, and it matters most:** recall/precision **split by forecast
   direction**. The headline (4.2) is a *composition* statistic, so it is
   sensitive to whether the extractor finds optimistic and pessimistic forecasts
   at equal rates. The gold standard supports this test (30 improve / 22 worsen)
   and it would cost ~$0.02. **This is the highest-value remaining analysis.**
5. **ProQuest.** The team dropped it 2026-07-24 (both VM-available models capped
   at ~0.5 F1 / **0.46 precision**). One intern wants to revive it. Note the
   asymmetry: Corpus M's error is *low recall* (misses real forecasts — if random,
   the mix stays unbiased); ProQuest's error is *low precision* (invents false
   ones — and newspaper junk skews optimistic, which biases the headline
   directly). "Both are ~0.5 F1" hides that.

---

## 7. Coverage limits (state these, don't hide them)

- **Effective sample is ~21 time blocks, not 14,251 claims.** Forecasts cluster
  hard within eras; all inference uses grouped CV and block bootstraps.
- **Newspaper coverage ends 1963** — that is where LOC full text ends
  (copyright). Post-1963 evidence comes from professional forecaster panels, not
  newspapers.
- **Coverage thins after 1940** (~600 claims/year pre-1940 → ~200–290 after), so
  the late index is noisier. Figures filter to n ≥ 5.
- **Band sensitivity** — hit rates swing 12–15 points across plausible no-change
  bands.
- **Our index does not correlate with the published historical EPU index**
  (all |r| < 0.08). Different constructs; the null is reported, not hidden.

---

## 8. GUARDRAILS — claims that are WRONG. Do not make them.

| ❌ Wrong claim | ✅ Reality |
|---|---|
| "Our extractor recovers **73%**" | 73% is gemini-3.5-flash, **never used**. What ran recovers **46.2%**. |
| "The model achieved **0.588**" | Stale. Current interaction model is **0.647**. |
| "Macro conditions score **0.505**" | Stale. Current figure is **0.538**, and the "null" was a cancellation. |
| "Adding both feature sets gives 0.65" | Adding gives **0.581**. Only the **interaction** gives 0.647. |
| "Newspapers beat the economists" | **Reversed and retired.** Diebold–Mariano finds no significant difference. |
| "Our model (0.65) beats forecasters (54%)" | **Category error.** AUC vs hit rate, different tasks, and the model was trained on outcomes with hindsight. |
| "Contrarian forecasts were more accurate" | **Reversed** on 8× the sample: 38.7% vs 53.3%. |
| "Greenbook's 54% shows modest skill" | It is **exactly** the naive always-improve baseline — **zero skill**. |
| "gpt-4o-mini was the extractor" | It was used for **ProQuest** quality-gating only. Never touched Corpus M. |
| "We have 4,125 claims" | That is Corpus C only. Corpus M has **30,765 / 14,251 scorable**. |
| Quoting any raw hit rate as skill | Compare to the naive "always improve" baseline on the **same rows**; different eras have different base rates. |

**The base-rate trap, stated once more because it bit this project:** the economy
expands most of the time, so a broken clock that always says "improve" scores the
base rate for free. Greenbook 54.0% vs naive 54.0% = zero skill. In the 1946–63
window the naive baseline is **79.2%** — so a comparison showing newspapers at
58% "beating" economists at 54% was comparing two forecasters who were *both*
far below a broken clock.

---

## 9. Where things live (the repo was reorganised 2026-07-28)

```
src/        44 scripts, flat (score_predictions.py, model_hit.py, extract_llm.py,
            greenbook_benchmark.py, build_press_index.py, scrape_monthly.py …)
data/       corpus/ claims/ scored/ reference/ proquest/
            → data/scored/monthly_scored.csv is the main results file
            → data/scored/press_index.csv is the 761-month index
docs/       POSTER.md, POSTER_V2.md, RESULTS_MONTHLY.md, RESULTS_MACRO.md,
            EXTENDED_ABSTRACT.md, SCORING.md, this file
figures/    poster_figures/ (16), poster_v2/, prelim_figures/, v1_episode/
validation/ gold_extraction/ (gold standard + 15-model bake-off), handgrade_newspapers/
tests/      test_offline.py (90 checks), test_scoring.py (33), test_forecasts.py
```

There is **no `JeremysShit/` directory of code any more** — it was dissolved into
the structure above. If a document refers to `JeremysShit/foo.py`, the file is now
`src/foo.py`.

Reproduce the headline numbers:
```
python src/score_predictions.py --claims data/claims/claims_monthly.jsonl \
    --out data/scored/monthly_scored.csv
python src/build_press_index.py --claims data/claims/claims_monthly.jsonl \
    --pages data/corpus/monthly/pages_monthly.jsonl --out data/scored/press_index.csv
python src/model_hit.py --scored data/scored/monthly_scored.csv --perm 0
python tests/test_scoring.py && python tests/test_offline.py
```

---

## 10. The one-paragraph summary

Across sixty years and 14,251 scored forecasts, the American press told readers
the economy would improve — in booms and busts alike, in the same proportion
(24.1% vs 24.3% downturn forecasts), with no improvement over six decades and no
paper better than any other. When downturns came, three in four forecasts pointed
the wrong way. This is not a journalism failing: the Survey of Professional
Forecasters, the Livingston economists and the Federal Reserve Board's own
Greenbook staff all score ~54%, and the Greenbook's forecast mechanically
collapses to a trend line beyond a year out, missing the four most recent
recessions entirely. The methodological contribution travels furthest —
whole-page LLM reading recovers 46% of forecasts against keyword search's 27%,
meaning the standard approach in this literature undercounts by more than half,
and non-randomly.
