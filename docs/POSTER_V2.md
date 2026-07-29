# Did Anyone See It Coming?
### Machine-reading sixty years of American newspaper economic forecasts, 1900–1963

*Poster draft v2. Layout and figure placement are built by `src/build_poster_v2.py`
into `2026 Poster Template.pptx`. Every number is re-derived from committed data
by `src/analysis_v2.py` — see `POSTER_V2_OUTLINE.md` for what changed from v1.*

---

## INTRODUCTION

For sixty years American newspapers told readers what the economy would do next.
Were they right?

Nobody has scored the record at scale, because reading a century of newspapers by
hand is infeasible — and the standard shortcut, keyword search, throws away most
of the evidence before the analysis begins.

**We built a pipeline that reads whole newspaper pages and grades what it finds
against real economic data.**

```
15,721 newspaper pages       every month, 1900-01 … 1963-12 · 1,103 publishers
        ↓ a language model reads each full page and returns structured forecasts
30,765 extracted forecasts   direction · topic · horizon · hedging · speaker · scope
        ↓ a deterministic scorer — NBER dates + Federal Reserve series, no model judgement
14,251 scorable forecasts    each marked hit, miss, or honestly unscorable
```

**The rule the whole project rests on:** the language model decides *what was
predicted*; real economic data decides *whether it came true*. The model is never
asked whether a forecast was right — that would be hindsight grading itself.
Correctness is a lookup against NBER business-cycle dates and Federal Reserve
series, computed the same way every time.

**Overall directional accuracy: 51.3%** (n = 14,251). A coin flip. The rest of
this poster is about *why*, and about who else does any better.

---

## METHODS

**1 · Collect.** Library of Congress *Chronicling America*, a fixed
direction-neutral query set, sampled in all 768 months so the corpus is
continuous rather than selected around famous crashes. 761 of those months yield
at least one scorable US-national forecast.

**2 · Extract.** A language model reads each full page and returns structured
forecasts. Whole-page reading is the methodological contribution — measured
against a gold standard of **16 pages annotated before any model ran** (52
forecasts, plus 44 documented boundary cases: advertisements, serialised fiction,
"Twenty Years Ago" reprints, refusals to forecast):

| extractor | precision | recall | F1 |
|---|---|---|---|
| keyword regex — the standard approach | .609 | **.269** | .373 |
| whole-page LLM reading (gemini-3.5-flash) | .844 | **.731** | .784 |
| as shipped on this corpus (gpt-oss-120b, low effort) | .615 | .462 | .527 |

**Keyword search loses roughly three forecasts in four.** It also actively
distorts: loaded search terms retrieve *advertising* ("return of prosperity"
appears on one gold page only inside a bank's New Year ad), and a keyword scrape
pulls a bullish 1907 forecast out of a passage headlined *"Prophecies Gone
Wrong"* — a report that forecasters had been wrong — which then scores against
the Panic and manufactures a "nobody saw it coming" signal out of nothing.

**3 · Score.** Deterministic, no model involved. 43% of extracted claims are not
about the US economy (regional, industry, foreign); without a scope field they
would be graded against national statistics. Claims with no direction, foreign
scope, or a date outside a series' coverage are marked **unscorable with a
recorded reason and left out — never guessed.**

**4 · Analyse.** Forecasts within an era share wire copy and one macroeconomic
reality, so the effective sample is **≈ 21 three-year blocks, not 14,251
claims.** Every out-of-fold number uses leave-one-block-out cross-validation and
every interval is block-bootstrapped.

**Validation:** 33 known-answer scorer tests against a synthetic economy · 90
offline pipeline tests · **total compute cost $24.97.**

---

## RESULTS

### Panel 1 — Fifteen downturns. Fifteen times the press said things would get better.

In the six months before **every one** of the fifteen NBER business-cycle peaks
between 1900 and 1963, a majority of US-national forecasts predicted the economy
would improve. Mean **74.2%**. The single most doubtful case — February 1945 — is
an exact even split. Not one downturn is preceded by a net-pessimistic press.

Found with no episode labels and no outcome information anywhere in the pipeline.
August 1929, two months before the Great Crash: **74% still predicting
improvement.**

### Panel 2 — The mechanism, and what it cost

> Share of forecasts predicting a downturn:
> **18.2% in expansions · 17.9% in recessions**
>
> Economic coverage per 100 pages:
> **197.5 in expansions · 196.4 in recessions**

Statistically identical, on both counts. Across sixty years the forecast mix
simply does not respond when the economy turns — and neither does the sheer
*volume* of economic coverage (p = 0.83; block-bootstrapped 95% CI
[−17.7, +17.6]; p = 0.96 with era means removed). The press did not change what
it said in a downturn, and did not even write more about it.

*Worth stating out loud:* the naive version of that second metric says the
opposite. Raw claims per month DO jump in recessions (36.9 → 47.3, p < 1e-11) —
but so do pages sampled per month (18.8 → 24.4), because digitisation density
varies by era. Per page the effect is exactly zero. It is a clean example of why
the denominator has to be chosen before the answer is known.

The consequence: accuracy falls from **58.8% in expansions to 39.7% in
recessions**, a gap of **19.1 points** (95% CI [12.1, 24.8], block-bootstrapped
by 3-year period).

And it is *not* that everyone got worse:

| | expansions | recessions |
|---|---|---|
| "business will improve" | 75.6% | **36.6%** |
| "business will worsen" | 21.0% | **60.5%** |

**Pessimists were right when it counted — there were simply never enough of
them.** Because the mix stayed ~4:1 upbeat regardless of conditions, the average
fell anyway.

*Stated with the panel:* NBER dates are both the scoring ground truth and the
split variable, so the accuracy gap is a **consequence** of the mix, not an
independent finding. The free-standing empirical fact is the mix.

### Panel 3 — There was no smart newspaper

Across the 19 papers with ≥100 scorable forecasts, accuracy runs from **36.6% to
66.0%** — a spread that looks decisive. It is not.

- **64%** of the between-paper variance is *composition*: which years a paper
  printed in, and which direction it tended to call. Both fix the odds before any
  editorial judgement enters.
- **95%** of what remains is exactly the binomial noise coin flips would produce.
- Joint test for any publisher skill at all: **χ² = 14.9 on 19 df, p = 0.73.**

Not one paper differs significantly from its own calendar's expectation. The
paper that looks smartest is the one whose calendar handed it the best odds.

### Panel 4 — Everything predicts the past. Nothing predicts the future.

Every candidate predictor, scored the flattering way and the honest way:

| predictor | in sample | out of fold |
|---|---|---|
| **the full text of the forecast** (24,598 TF-IDF features) | **0.911** | **0.541** |
| what was predicted | 0.621 | 0.514 |
| subject of the forecast | 0.564 | 0.500 |
| newspaper | 0.527 | 0.427 |
| era printed | 0.523 | 0.411 |
| hedging · voice · named forecaster · horizon | .506–.511 | .409–.433 |

Fit on the whole corpus, the raw text of a forecast separates hits from misses
almost perfectly. Asked to rank forecasts from a decade it has never seen, the
same model scores 0.541 against a 0.513 base rate. Several predictors land
*below* 0.5 — their in-sample ranking **reverses** in a new era.

**We do not claim to predict which forecasts come true.** Saying so plainly is
the result: forecast accuracy in this corpus is close to irreducible.

### Panel 5 — Nobody has skill, and the professionals are no exception

Raw accuracy flatters whoever happened to forecast in a calm decade. Subtract
what a naive rule — *"the economy will improve, always"* — would have earned on
the same claims:

| forecaster | period | n | raw | naive | **skill** | % improve |
|---|---|---|---|---|---|---|
| Newspapers (this project) | 1900–1963 | 8,557 | .571 | .611 | **−.041** | 81% |
| Newspapers (this project) | 1946–1963 | 1,384 | .590 | .702 | **−.112** | 73% |
| Livingston economists | 1946–1963 | 36 | .722 | .833 | **−.111** | 78% |
| Livingston economists | 1946–2026 | 161 | .839 | .888 | **−.050** | 86% |
| Fed Greenbook (internal staff) | 1967–2020 | 490 | .540 | .549 | **−.009** | 90% |
| Survey of Prof. Forecasters | 1968–2026 | 231 | .541 | .498 | **+.043** | 94% |

Over the matched 1946–63 window the economists beat the newspapers by 13 raw
points. On skill the gap is **0.001**. In a century, nobody clears +0.05 — and
every forecaster on this list predicted improvement 73–94% of the time.

*Caveats printed with the panel:* the matched Livingston window holds only n = 36
surveys, and the four sources score different variables under different rules.
A reference point, not a controlled experiment.

---

## DISCUSSION / CONCLUSIONS

**Optimism was structural, not stupidity.** The economy really does expand most
of the time, so a permanent bull is right more often than not — and the naive
rule beats almost everyone in the table above. What the record shows is not that
forecasters were foolish but that the press published a *fixed* ratio of optimism
that carried no information about conditions. The signal was never in the
direction of the forecast; there was no signal.

**The professionals are no better.** This is what stops the finding reading as
"old newspapers were dumb." The Fed's own staff, with modern national accounts,
scored 54% and had essentially zero skill over the naive rule; the Greenbook
predicted "worsen" 6 times in 490 forecasts and got 0 of 6 right. Sixty years of
newspapers and sixty years of professional macroeconomics land in the same place.

**Sixty years, no improvement.** Annual accuracy trends **+0.002 per decade**
(p = 0.87) across 1900–1963. Two world wars, the Depression, and the founding of
modern macroeconomics produced no measurable gain in the press's ability to call
the economy's direction.

**Should we train a neural network?** We measured instead of guessing. The full
text of a forecast, given 24,598 features and a linear model, reaches **AUC 0.911
in sample and 0.541 out of fold**. The binding constraint is not model
capacity — it is that sixty years of forecasts contain only **~21 independent
time blocks**. More parameters is the one intervention that provably cannot fix
that; a larger model would memorise harder and transfer no better. *(The
defensible use of a GPU cluster here is the extractor, not the predictor — see
Limitations.)*

**Two findings reversed under more data, and we report both.** On a smaller
crisis-only corpus (n = 232), forecasts that swam against the press consensus
looked *more* accurate (52% vs 43%). On the full continuous corpus (n = 1,967) it
reverses: **38.7% vs 53.3%** — contrarians were worse. Separately, a benchmark
comparison this project previously reported did not survive recovering its
deleted source data, and is corrected in Panel 5. Both are the strongest
available argument for building a continuous corpus rather than sampling around
famous crashes: it overturns things a selected sample suggests.

### Limitations, stated plainly

- **Extraction quality is the binding limitation.** This corpus was extracted at
  **F1 = 0.527** to fit a $30 budget; the best extractor tested reaches 0.784 but
  would have cost $404. Aggregate monthly series tolerate this — per-claim error
  partly averages out over ~40 claims/month — but the two are different
  instruments. This, not sample size, is where more compute would genuinely buy
  accuracy.
- **The gold standard is model-adjudicated, not human.** A two-person recode of
  ~40 claims is the outstanding step before publication.
- **Coverage thins after 1940** (~600 claims/yr pre-1940 → 200–290 after,
  copyright-driven), so the late index is noisier. Every series carries its claim
  count; figures filter to n ≥ 5.
- **The scope filter is not neutral.** The 13,935 discarded regional, industry and
  foreign claims are *more* optimistic than the national ones kept (78.7% vs
  73.1% improve), so true press-wide optimism is somewhat higher than we report.
- **Effective sample ≈ 21 time blocks, not 14,251 claims.** All inference uses
  grouped CV and block bootstraps.

### The headline

**Across sixty years and 14,251 scored forecasts, the American press told readers
the economy would improve — in booms and in busts alike, in the same proportion,
with no improvement over six decades and no paper doing better than any other.
Before all fifteen downturns of the era, it said things would get better.**

Forecasting the economy's direction was, and remained, close to a coin flip. The
contribution is that this is now *measured* rather than asserted — and the
pipeline that measures it is validated, reproducible, and cost $24.97.

---

## REFERENCES

1. Library of Congress, *Chronicling America: Historic American Newspapers.*
   chroniclingamerica.loc.gov (public API).
2. National Bureau of Economic Research, *US Business Cycle Expansions and
   Contractions.* nber.org/research/business-cycle-dating
3. Federal Reserve Bank of St. Louis, *FRED* — INDPRO, CPIAUCNS, UNRATE, and the
   NBER historical common-stock price index.
4. Federal Reserve Board, *Greenbook Forecast Data Sets*, 1967–2020.
5. Federal Reserve Bank of Philadelphia, *Survey of Professional Forecasters* and
   the *Livingston Survey.*
6. Baker, S., Bloom, N., & Davis, S. (2016). Measuring Economic Policy
   Uncertainty. *Quarterly Journal of Economics*, 131(4), 1593–1636.

## ACKNOWLEDGEMENTS

Boston University RISE Practicum. Data from the Library of Congress, the National
Bureau of Economic Research, and the Federal Reserve — all public and free to
use. Total compute cost $24.97.
