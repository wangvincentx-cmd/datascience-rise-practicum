# Poster v2 — the plan

**Compiled 2026-07-28.** Every number cited here was re-derived from committed
data by `src/analysis_v2.py`; nothing is copied forward from the v1 write-ups.
Outputs land in `data/scored/analysis_v2/` and `figures/poster_v2/`.

Decisions made with the author before drafting:

| decision | choice |
|---|---|
| lead finding | **"15 out of 15"** — the press predicted improvement before every downturn |
| H200 cluster | **report the measured negative result; train nothing new** |
| template | slide 1 of `2026 Poster Template.pptx` — narrow left column, tall 15″ centre results column |
| caveats | **printed inside each panel**, not exiled to a footnote |

---

## 1. What changed since v1

### 1.1 New results

**A. Fifteen downturns, fifteen misses.** In the six months before *every one* of
the fifteen NBER business-cycle peaks between 1900 and 1963, a majority of
US-national forecasts predicted the economy would improve. Mean 74.2%; the
single most doubtful case (Feb 1945) is an exact even split; not one peak is
preceded by a net-pessimistic press. This generalises the v1 "October 1929"
panel from one anecdote into a rule, on the same continuous corpus, with no
episode labels anywhere in the pipeline.
→ `v2_fig1_fifteen_of_fifteen.png`

**B. There was no smart newspaper — now a formal null.** v1 reported a 44–56%
spread across seven papers and called it "all near chance" by eyeball. Decomposed
properly across the 19 papers with ≥100 scorable forecasts (spread 36.6–66.0%):

- **64%** of the between-paper variance is *composition* — which years a paper
  printed in and which direction it tended to call, both of which fix the odds
  before any judgement enters;
- **95%** of what remains is exactly the binomial noise coin flips would produce;
- joint test for any publisher skill at all: **χ² = 14.9 on 19 df, p = 0.73**;
- permutation null holding (year × direction) fixed: p = 0.08 on the SD.

Not one paper differs from its own calendar's expectation.
→ `v2_fig3_no_smart_paper.png`

**C. Everything predicts the past; nothing predicts the future.** Each candidate
factor scored twice — in sample, and out of fold under leave-one-3-year-block-out:

| factor | in-sample AUC | out-of-fold AUC |
|---|---|---|
| full text of the forecast (24,598 TF-IDF features) | **0.911** | **0.541** |
| what was predicted | 0.621 | 0.514 |
| subject of the forecast | 0.564 | 0.500 |
| newspaper | 0.527 | 0.427 |
| era printed | 0.523 | 0.411 |
| hedging / voice / named forecaster / horizon | 0.506–0.511 | 0.409–0.433 |

Several land *below* 0.5 — their in-sample ranking reverses in an unseen era.
→ `v2_fig4_nothing_transfers.png`

**D. The corrected benchmark.** See §2.2 — the v1 claim was wrong and its
replacement is stronger.
→ `v2_fig5_nobody_has_skill.png`

**E. The press did not even write MORE about the economy in a downturn.**
Economic coverage per 100 pages: **197.5 in expansions vs 196.4 in recessions**
(p = 0.83; block-bootstrapped 95% CI [−17.7, +17.6]; p = 0.96 with era means
removed). This is the stronger form of finding (A)/Panel 2: not only was the
*direction* of forecasts unresponsive, the *volume* of economic coverage was too.

> **The naive version of this metric produces a false positive, and the poster
> says so.** Raw claims per month DO rise sharply in recessions (36.9 → 47.3,
> p < 1e-11) — but so do *pages sampled* per month (18.8 → 24.4, p < 1e-17),
> because LOC digitisation density varies by era and recession months cluster in
> the densely digitised early decades. Per page, the effect is exactly zero.

Folded into Panel 2 as a third sub-panel at no cost in poster height.

**F. Price forecasting was the era, not the forecaster.** For each decade and
direction, the hit rate of price forecasts equals the base rate of the outcome
they predicted: **correlation 0.994, mean gap 3.6 points.** "Prices will rise"
was right 85% of the time in the inflationary 1910s and 26% in the flat 1960s —
the forecast never changed, the economy did.
→ `v2_fig6_price_regime.png` — **not on the poster** (see §2.1), kept as the
answer to "but *why* does nothing transfer?"

> Corrects two v1 claims in the process. Up/down hit rates do **not** sum to 1
> in every decade (0.86 / 0.75 / 0.75 / 0.90 / 0.57 / 0.26) — prices are scored
> three ways and "flat" absorbs the rest, taking 50% of 1950s outcomes and 79%
> of 1960s. And "prices will fall" after 1948 was right **9 times out of 139**,
> not 0 out of 93.

### 1.2 Corrections to v1

| v1 said | v2 says | why |
|---|---|---|
| Newspapers ≈ Livingston economists, 0.56 vs 0.545, n = 314 / 68 | **Does not reproduce.** Raw rates are 59.0% vs 72.2% (n = 1,384 / 36), p = 0.008 — economists better | v1 read the numbers off a chart whose input file had been deleted; recovered from commit `daf1724` to `data/reference/scored_livingston.csv` |
| Disagreement has no relationship with accuracy (0.537/0.486/0.514) | **Monotone: 0.574 / 0.501 / 0.460** | v1 used the `model_hit.py` measure, which divides by *all* claims that month. The `build_press_index.py` measure — divide by *directional* claims — is the coherent one and is cleanly monotone |
| Monthly corpus extracted at F1 ≈ 0.61 | **F1 = 0.527** | 0.612 is the *default-effort* row. Both `CLAUDE.md` and `RESULTS_MONTHLY.md` record the run as `--reasoning-effort low`, which measures 0.527. The run log was not retained; 0.527 is the conservative reading |
| "768/768 months, no gaps" | **768 months sampled; 761 yield a scorable US-national claim** | two different quantities, stated separately |

### 1.3 Dropped from v1

- **October 1929 as its own panel** — absorbed into finding A, which is strictly
  stronger. The 1929 point stays, annotated on fig 1.
- **The contrarian finding** — already reversed in v1 and correctly suppressed;
  it survives only as one line in Discussion, because *reporting a reversal* is
  itself a methods contribution.
- **Accuracy-by-topic as a headline** — the 23-point spread is real in sample but
  transfers at 0.500. It moves into fig 4 as one row rather than a panel.

---

## 2. What goes on the poster

Template slide 1: 36 × 48 in portrait. Left column stacks Introduction over
Methods; the 15″-wide centre column carries all five figures; Discussion,
References and Acknowledgements run down the right.

### 2.1 Panel order (centre column, top to bottom)

| # | figure | height | the one sentence it must land |
|---|---|---|---|
| 1 | `v2_fig1_fifteen_of_fifteen` | 8.09″ | Fifteen downturns. Fifteen times the press said things would get better. |
| 2 | `v2_fig2_mechanism` | 5.17″ | Neither what the press said nor how much it said responded to the economy — and that, not bad judgement, is what cost accuracy. |
| 3 | `v2_fig3_no_smart_paper` | 6.78″ | The paper that looks smartest is the one whose calendar handed it the best odds. |
| 4 | `v2_fig4_nothing_transfers` | 7.38″ | A model can memorise which forecasts came true; none can predict them in a decade it has not seen. |
| 5 | `v2_fig5_nobody_has_skill` | 6.60″ | Newspapers, Livingston economists and the Fed's own staff all score at or below a rule that says "improve, always." |

**Total 34.02″ in a 39.15″ column → 1.28″ between panels.** Adding
`v2_fig6_price_regime` (5.77″) would need 39.80″ and leave no gaps at all, so it
is deliberately held back as a handout rather than crammed in. If it is wanted
on the board, the cheapest swap is to drop Panel 3 — but Panel 3 is a formal
null and the price figure is an illustration, so that trade is not recommended.

### 2.2 The benchmark panel, restated

Raw accuracy flatters whoever forecast in a calm decade. Subtract what a naive
"the economy will improve, always" rule would have earned on the *same* claims:

| forecaster | period | n | raw | naive | **skill** | % improve |
|---|---|---|---|---|---|---|
| Newspapers (this project) | 1900–1963 | 8,557 | .571 | .611 | **−.041** | 81% |
| Newspapers (this project) | 1946–1963 | 1,384 | .590 | .702 | **−.112** | 73% |
| Livingston economists | 1946–1963 | 36 | .722 | .833 | **−.111** | 78% |
| Livingston economists | 1946–2026 | 161 | .839 | .888 | **−.050** | 86% |
| Fed Greenbook (internal staff) | 1967–2020 | 490 | .540 | .549 | **−.009** | 90% |
| Survey of Prof. Forecasters | 1968–2026 | 231 | .541 | .498 | **+.043** | 94% |

The 13-point raw gap between newspapers and economists in the matched window
collapses to 0.001 on skill. In a century, nobody clears +0.05, and every
forecaster on the list predicted improvement 73–94% of the time.

**Caveats printed with the panel:** the matched Livingston window holds only
n = 36 surveys, and the four sources score different variables under different
rules. This is a reference point, not a controlled experiment.

### 2.3 Left column

**Introduction** — the question, why it has never been answered at scale, and
the rule the project rests on: *the language model decides what was predicted;
real economic data decides whether it came true.* The model is never asked
whether a forecast was right.

**Methods** — the pipeline in four steps, the extraction bake-off, and the
validation evidence:

| extractor | precision | recall | F1 |
|---|---|---|---|
| keyword regex (the standard approach) | .609 | **.269** | .373 |
| whole-page LLM reading (gemini-3.5-flash) | .844 | **.731** | .784 |
| as shipped on the monthly corpus (gpt-oss-120b, low effort) | .615 | .462 | .527 |

Plus: 16 gold pages annotated before any model ran (52 forecasts, 44 boundary
cases), 33 known-answer scorer tests, 90 offline pipeline tests, $24.97 total
compute.

### 2.4 Right column

**Discussion** — four moves: (i) optimism is structural, not stupidity, because
the economy really does expand most of the time; (ii) the accuracy gap is a
consequence of the mix, and NBER dates are both the truth and the split
variable, so we say so; (iii) the professionals are no better, which is what
stops this reading as "old newspapers were dumb"; (iv) two findings that
reversed under more data, reported as evidence the method is working.

**The H200 question, answered on the poster.** Fitting the full text of a
forecast separates hits from misses at AUC 0.911 in sample and 0.541 out of
fold. The binding constraint is ~21 independent time blocks, not model capacity
— so more parameters is the one intervention that cannot help. We did not train
one, and the measurement is why. *(If the cluster is to be used, the defensible
target is the extractor, not the predictor — see §4.)*

---

## 3. Statistical discipline (unchanged, and stated on the poster)

- Split by time block, never randomly. Effective sample ≈ **21 three-year
  blocks**, not 14,251 claims.
- Accuracy is never the metric on its own — hit rate, skill vs naive, error
  direction and block-bootstrapped intervals.
- Unscorable means unscored: 16,439 US-national claims yield 14,251 scorable;
  the rest carry a recorded reason and are left out, never guessed.
- No hindsight in labels: no episode name, outcome or recession flag ever
  reaches an extraction or grading prompt. `in_recession` splits results that
  are *already* scored.

---

## 3a. Extending past 1963 — investigated, and the answer is no

`data/proquest/` holds **37,277 records (29,267 with a forecast) covering
1965–2009**, and extending "fifteen out of fifteen" to the modern recessions
was the single largest upside on the table. It does not work, for two
independent reasons. Both were checked, not assumed.

**Reason 1 — the windows begin AT each crisis, not before it.** The corpus was
sampled in windows around known events, and every window starts at or after the
NBER peak it surrounds:

| window | starts | nearest peak | offset | claims in the 6 months BEFORE that peak |
|---|---|---|---|---|
| oil_1973 | 1973-10 | 1973-11 | −1 mo | **75** |
| volcker_1980 | 1980-01 | 1980-01 | 0 | 0 |
| gulf_1990 | 1990-07 | 1990-07 | 0 | 0 |
| dotcom_2001 | 2001-03 | 2001-03 | 0 | 0 |
| gfc_2008 | 2008-09 | 2007-12 | **+9 mo** | 0 |
| crash_1987 · calm_1965 · calm_1995 · calm_2005 | — | — | — | 0 |

**75 of 29,267 claims fall in the run-up to any modern peak, and all 75 are from
a single month.** The question "did they see it coming?" is a question about the
lead-up, and the lead-up was never collected.

**Reason 2 — the schema cannot reach our scorer.** ProQuest records carry
`predicted_direction`, `horizon_months`, `voice`, `hedged`, `date`,
`newspaper_title`. They are missing **`topic`** — which is what routes a claim to
its ground-truth series (general business → NBER/INDPRO, prices → CPI,
employment → UNRATE, markets → stock index) — and also `quote`, `scope`,
`price_direction`, `unemployment_direction`. Without `topic` nothing can be
scored by `score_predictions.py`; without `quote` the claim features and the
text model in Panel 4 cannot be built either.

There is a partial workaround — the ProQuest extraction asked a
recession-vs-expansion question directly, so every claim is a general-business
claim by construction and could be routed that way. It does not help, because
Reason 1 is fatal on its own.

**Third, softer problem: it is a different instrument.** 68% of ProQuest claims
predict "worsen" against 18% in the LOC corpus. That reflects crisis-window
retrieval, a different extractor, a different prompt, and different papers
(Wall Street Journal, NYT, LA Times vs local dailies) — so the two corpora
cannot be compared claim-for-claim in either direction. Note that even the
"calm" windows run 40–54% worsen, which is implausible for 1965/1995/2005 and
suggests the retrieval itself is tilted toward pessimistic copy.

**What this is worth:** it is a precise spec for a future scrape. To extend the
finding, a continuous monthly sample 1964–2010 is needed — the same
direction-neutral query design as the LOC corpus — with `topic`, `scope` and
`quote` retained. That is a scraping and extraction job, not an analysis job.

## 4. Open items, in priority order

1. **The gold standard is model-adjudicated.** A two-person human recode of ~40
   claims remains the outstanding step before any publication claim.
2. **Extraction quality is the binding limitation, not sample size.** F1 0.527 on
   the shipped corpus vs 0.784 for the model the budget could not afford. This is
   the one place more compute genuinely buys accuracy — fine-tuning an open model
   on the gold standard is the natural H200 job, and would strengthen every
   downstream number. Deliberately out of scope for v2.
3. **Coverage thins after 1940** (~600 claims/yr pre-1940 → 200–290 after), so
   late-period months are noisier. Figures filter to n ≥ 5 and every series
   carries its claim count.
4. **The scope filter is not neutral.** The 13,935 discarded regional / industry /
   foreign claims are *more* optimistic than the national ones kept (78.7% vs
   73.1% improve), so true press-wide optimism is somewhat higher than the index
   shows. One sentence in Limitations.
5. **The disagreement result is partly mechanical.** Low disagreement means
   near-unanimous optimism, and optimism usually pays. Within "improve" claims
   alone the effect shrinks from 11.4 to 6.8 points but persists (.641/.570/.573).
   Print with that caveat or not at all.

---

## 5. Reproduce

```
python src/analysis_v2.py --section all          # every v2 number
python src/make_poster_v2_figures.py             # the five figures
python src/build_poster_v2.py                    # -> RISE_Poster_2026_v2.pptx
python tests/test_offline.py && python tests/test_scoring.py && python tests/test_forecasts.py
```
