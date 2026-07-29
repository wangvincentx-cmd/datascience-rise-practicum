# Poster data inventory — everything we can put on it

**Compiled 2026-07-28.** Every number below was re-derived from the committed
data (`monthly_scored.csv`, `data/press_index.csv`, `gold_extraction/*.json`,
`greenbook_scored.csv`, `spf_scored.csv`) rather than copied from the write-ups.
Numbers that did **not** reproduce are flagged in §9.

Tiers: **A** = verified, robust, poster-ready · **B** = solid, needs a stated
caveat · **C** = do not print / fix first.

---

## 1. Scale and provenance (tier A) — the credibility strip

| number | value |
|---|---|
| newspaper pages read | **15,721** |
| period | 1900-01 → 1963-12, every month (768 months) |
| months with ≥1 national claim | 761 |
| pages yielding ≥1 forecast | 11,736 (74.7%) |
| distinct publishers | **1,103** |
| forecasts extracted | **30,765** |
| US-national forecasts | 16,439 |
| scorable against ground truth | **14,251** |
| median forecasts/month | 21 |
| months with <5 claims (filtered from figures) | 59 |
| scorer known-answer tests | **33 passed** (re-run 2026-07-28) |
| offline pipeline checks | **90 passed** (re-run 2026-07-28) |
| extraction cost, monthly corpus | **$7.47** |
| total project compute | **$24.97** of a $30 cap |
| extractor bake-off cost | $0.86 for 21 configurations |
| pages truncated at token cap | 5 of 15,721 (0.03%) |

Ground truth: NBER business-cycle chronology + FRED (INDPRO, CPI, UNRATE,
stock index). No API key required for LOC *Chronicling America*.

---

## 2. Methods panel — keyword search loses three forecasts in four (tier A)

Gold standard: **16 pages, 52 forecasts**, annotated before any model ran, plus
**44 documented boundary cases** (32 traps to refuse, 12 awkward real forecasts).

| extractor | precision | recall | F1 |
|---|---|---|---|
| keyword regex (standard approach) | 0.609 | **0.269** | 0.373 |
| gemini-3.5-flash 8k (whole-page LLM) | 0.844 | **0.731** | 0.784 |
| gemini-3.5-flash 1-window | **1.000** | 0.654 | **0.791** |
| gpt-oss-120b → gpt-oss verify | 0.618 | 0.654 | 0.636 |
| gpt-oss-120b (as-shipped, default effort) | 0.536 | 0.712 | 0.612 |
| gpt-oss-120b **low effort** | 0.615 | 0.462 | **0.527** |
| Llama-3.1-8B (floor) | 0.121 | 0.462 | 0.191 |

**21 configurations tested.** Cost-quality frontier at 23,000 pages: regex $0 ·
gpt-oss-120b self-verify **$27** (F1 0.636) · gemini-flash-lite verify $44 ·
gemini-3.5-flash 8k **$404** (F1 0.784). Ten times the price buys +0.12 F1 —
a real judgement call, and a genuine methods contribution on its own.

**Three concrete ways keyword search distorts the answer** (all from gold pages):
1. Loaded terms retrieve **advertising** — "return of prosperity" appears on one
   gold page only inside a bank's New Year ad.
2. It extracts a *bullish* 1907 forecast from a passage headlined **"Prophecies
   Gone Wrong"** — a report that forecasters were wrong — which then scores
   against the Panic and manufactures a "nobody saw it coming" signal.
3. **43% of extracted claims are not about the US economy** (regional 6,486 /
   industry 5,567 / foreign 2,217). Without a scope field they get graded
   against national statistics.

---

## 3. The index itself (tier A) — 761 months of press expectations

Net optimism = share improve − share worsen, national scope only.

| decade | net optimism | disagreement | recession months |
|---|---|---|---|
| 1900s | +0.48 | 0.50 | 32% |
| 1910s | +0.53 | 0.46 | 48% |
| 1920s | +0.54 | 0.45 | 44% |
| 1930s | +0.49 | 0.50 | 44% |
| **1940s** | **+0.16** | **0.71** | 18% |
| 1950s | +0.32 | 0.59 | 17% |
| 1960s | +0.38 | 0.50 | 24% |

**The 1940s spike is the instrument validating itself** — wartime and
postwar-depression-scare uncertainty, found with no episode labels and no
outcome information.

**Extremes worth annotating on the time-series figure:**
- Most pessimistic months: 1941-04 (−0.60), 1900-11 (−0.59), 1950-12 (−0.54),
  1945-01 (−0.40), 1919-08 (−0.37), 1948-12 (−0.35)
- Maximum-disagreement months (net exactly 0, press evenly split): 1907-02,
  1920-02, 1920-08, 1916-11, 1918-08
- Unanimous-optimism months: 1902-08, 1906-05, 1909-05, 1913-08, 1922-04

---

## 4. The core result (tier A)

**Overall directional hit rate: 0.513** (n = 14,251).

**4a. The mechanism — the forecast mix never responded to the economy**

| | expansions | recessions |
|---|---|---|
| share predicting a downturn | **24.1%** | **24.3%** |

Roughly **72% of all forecasts were upbeat**, in booms and busts alike. This
survives every robustness check I ran: within-decade comparison (mean difference
−0.004, p = 0.911), claim-level block bootstrap by 3-year period (95% CI
[−0.024, +0.061], spans zero), and index-level t-test after removing era means.

> The raw pooled comparison looks like the press was *more* optimistic in
> recessions (0.466 vs 0.398, p = 0.016) — this is **Simpson's paradox**. Early
> decades had both more recession months and higher baseline optimism
> (corr = +0.885). Use the within-era version; the pooled one is wrong.

**4b. The consequence — accuracy collapses when it matters**

| | expansions | recessions |
|---|---|---|
| overall accuracy | **58.8%** | **39.7%** |

Gap **+19.1 points**, block-bootstrapped 95% CI **[+12.9, +24.4]**.

| | expansions | recessions |
|---|---|---|
| "business will improve" | 75.6% | **36.6%** |
| "business will worsen" | 21.0% | **60.5%** |

**Pessimists were right when it counted — there were never enough of them.**

**4c. What was predicted**

| prediction | n | hit rate |
|---|---|---|
| improve | 8,962 | **0.597** |
| prices up | 1,359 | 0.510 |
| worsen | 2,577 | 0.364 |
| prices down | 868 | 0.267 |
| no change / flat | 485 | **0.194** |

"Business will hold steady" scored **0.045** (n = 156). The economy is never flat.

**4d. Sixty years, no improvement**

Annual accuracy trend **+0.0008 per decade** over 64 years — flat.
1900s 53.7% → 1910s 48.9% → 1920s 51.7% → 1930s 49.5% → 1940s 55.0% →
1950s 50.8% → 1960s 49.9%.

**And no publisher was better.** Seven papers with ≥200 scorable forecasts:

| publisher | n | hit rate |
|---|---|---|
| new-york tribune | 534 | 0.442 |
| the washington times | 1,040 | 0.465 |
| the washington herald | 212 | 0.500 |
| evening star (DC) | 3,728 | 0.516 |
| springfield weekly republican | 311 | 0.543 |
| the washington daily news | 259 | 0.556 |
| the birmingham age-herald | 298 | 0.564 |

---

## 5. NEW — accuracy by topic (tier A descriptive / tier B predictive)

**Not currently on the poster. The largest effect in the dataset.**

| topic | extracted | scorable | hit rate | vs 50% |
|---|---|---|---|---|
| general_business | 14,267 | 8,713 | **0.561** | p = 2e-30 |
| other | 2,965 | 870 | 0.479 | — |
| markets | 3,533 | 2,219 | **0.449** | p = 2e-06 |
| prices | 5,229 | 2,275 | **0.415** | p = 5e-16 |
| employment | 1,663 | 174 | **0.328** | p = 6e-06 |
| industry | 3,015 | **0** | — | no ground-truth series |

A 23-point spread — against 2 points for hedging and 1.4 for a named forecaster.
**Markets, prices and employment forecasts are all significantly worse than a
coin flip.** The press published 10,400 of them. The only above-chance category
is vague general-business optimism, and only because "business will improve" is
right most years by default.

**5a. Price forecasting was regime, not skill**

| decade | "prices up" hit | "prices down" hit | sum |
|---|---|---|---|
| 1910s | **0.852** | 0.007 | 0.86 |
| 1920s | 0.269 | 0.476 | 0.75 |
| 1930s | 0.311 | 0.443 | 0.75 |
| 1940s | **0.825** | 0.070 | 0.90 |
| 1950s | 0.574 | **0.000** | 0.57 |
| 1960s | 0.256 | **0.000** | 0.26 |

The two columns sum to ~1 in every decade — the signature of a **zero-sum
regime effect**. Being right about prices meant your standing bias happened to
match the era's inflation regime. After 1948, "prices will fall" was right
**0 times out of 93**.

**5b. Topic separates in-sample and predicts nothing out-of-sample**

Leave-one-3-year-block-out AUC (same CV as `model_hit.py`): topic alone
**0.495** — chance. topic + direction 0.558. The topic advantage does not
transfer across eras, for exactly the reason in 5a. *This strengthens the
"close to irreducible" conclusion rather than complicating it.*

**5c. Employment is a coverage story**

Only **174 of 1,663** employment claims (10.5%) are scorable — 721 die on "no
UNRATE before 1948." The most policy-relevant topic is the one we can barely
measure. Limitations line, not a result.

**5d. The recession gap is mostly one topic.** Excluding general_business, the
expansion→recession gap falls from +19.1 to **+9.4** points. Markets claims
barely move (47.2% → 41.7%).

---

## 6. NEW — October 1929 (tier A) — the single strongest panel available

National directional claims, share predicting improvement:

| month | n | improve share |
|---|---|---|
| 1929-06 | 17 | 76.5% |
| 1929-07 | 15 | 93.3% |
| 1929-08 | 22 | 86.4% |
| 1929-09 | 42 | 64.3% |
| **1929-10 (the Crash)** | 36 | **86.1%** |
| 1929-11 | 37 | 78.4% |
| 1929-12 | 42 | 76.2% |
| 1930-01 | 48 | 79.2% |

**In the month of the Great Crash the press was more bullish than it had been in
June.** Sep–Dec 1929 national claims: 119 improve / 38 worsen / 3 no-change.

**Forecasts printed Aug–Dec 1929 scored 0.208 (n = 168).** Four in five were
wrong. The crisis-corpus episode file puts the 1929 Crash window at **0.106
(n = 188)** with 82.4% predicting improvement.

Episode table (crisis corpus, `results_by_episode.csv`) also gives: 1907 Panic
0.860 (n=129, but naive rate 0.915 — *below* naive), 1920 Depression 0.475
(n=242), 1937 0.472, 1973 Oil Shock 0.470 with **+0.44 skill vs naive** (the
one genuine win), 2001 Dot-com 0.182, 2008 GFC 0.471 (n=17, too thin).

---

## 7. Benchmarks (tier A/B) — external validity, the best framing available

This is what stops the poster reading as "old newspapers were dumb."

**7a. Era-matched comparison, 1946–63 — CORRECTED 2026-07-28.**

> ⚠️ **The numbers previously printed here were wrong and have been removed.**
> They were read off `figures/v1_episode/fig_three_way_benchmark.png` rather
> than from data, because the figure's input had been deleted in commit
> `daf1724`. That file is now restored to
> `data/reference/scored_livingston.csv` (161 rows), and it does **not**
> reproduce the claim "newspapers ≈ economists, 0.56 vs 0.545, n = 314 / 68."
> Do not use that figure.

Recomputed from the restored file by `src/analysis_v2.py`:

| forecaster | window | n | raw hit | naive "always improve" | **skill** |
|---|---|---|---|---|---|
| Newspapers (this project) | 1946–63 | 1,384 | 0.590 | 0.702 | **−0.112** |
| Livingston survey economists | 1946–63 | 36 | 0.722 | 0.833 | **−0.111** |

**On raw accuracy the economists beat the newspapers by 13 points (p = 0.008).
On skill over the naive rule the gap is 0.001.** The entire apparent advantage
is the base rate of the window each forecaster happened to be scored in — which
is why *skill*, not accuracy, is the only comparable quantity across sources.
Newspaper rows are restricted to general-business directional calls, the closest
match to Livingston's industrial-production question.

Caveats to print with it: n = 36 in the matched window, and the two sources
score different variables under different rules.

**7b. Modern professional forecasters (secondary, cross-era)**

| forecaster | period | n | hit rate | skill vs always-"improve" |
|---|---|---|---|---|
| Newspapers (this project) | 1900–1963 | 14,251 | 0.513 | — |
| **Fed Greenbook** (internal staff) | 1967–2020 | 490 | **0.540** | **−0.009** |
| **SPF** (Survey of Prof. Forecasters) | 1968–2026 | 231 | **0.541** | +0.043 |

**The Fed's own staff, with modern national accounts, scored 54% — and had
essentially zero skill over a naive always-improve rule.**

And the optimism bias is not a newspaper artifact:
- Greenbook predicted "worsen" **6 times in 490 forecasts (1.2%)** — and got
  **0 of 6** right.
- SPF predicted "improve" in **218 of 231** forecasts (94%).

**Caveat to state on 7b:** different eras, different variables (qualitative
direction vs real GDP growth), different horizons. A reference point, not a
controlled comparison — which is exactly why 7a should lead.

---

## 8. The nulls (tier A) — these are results, not failures

- **Macro conditions at print time carry no information**: out-of-fold AUC
  **0.505**, every macro feature |r| < 0.10 with `hit`. Verified directly.
- **How a forecast is written**: AUC **0.561** — beats base rate (0.513) and
  macro, but barely. Claim + macro + derived 0.588; gradient boosting 0.573.
- **No expert advantage**: expert 0.523 / journalist 0.504 / official 0.499 /
  layperson 0.504.
- **Hedging doesn't help**: assertive 0.520 vs hedged 0.500.
- **Named forecaster doesn't help**: 0.520 vs 0.506.
- **Conditional forecasts did *worse***: 0.482 vs 0.521 (n = 2,917). An explicit
  "if" clause was not protective.
- **Horizon barely matters**: 6mo 0.493, 12mo 0.516; shortest (0–3mo) worst at
  0.414.
- **The contrarian finding REVERSED.** Crisis corpus (n=232): contrarians 52%
  vs 43%. Full corpus (n=1,967): **38.7% vs 53.3%**. Small-sample noise from an
  outcome-selected sample. The single best argument for building the continuous
  corpus.
- **The index does not lead recessions** (new): corr(12m-smoothed net optimism,
  recession 12 months ahead) = **+0.14** — wrong sign and negligible. Event
  study around all 15 NBER peaks is flat (mean z between −0.22 and +0.13).
- **Coverage volume doesn't respond either** (new): 197.5 claims/100 pages in
  expansions vs 196.4 in recessions, p = 0.80. The press didn't even write
  *more* about the economy in downturns. This is a stronger version of Panel 2.
- **No correlation with the published historical EPU index** (all |r| < 0.08) —
  we measure forecast direction, not policy uncertainty.
- **Effective sample is ~21 time blocks, not 14,251 claims.** All inference uses
  grouped CV and block bootstraps.

---

## 9. Fix before printing (tier C)

**Items 1–3 were resolved on 2026-07-28 by `src/analysis_v2.py`.** See
`POSTER_V2_OUTLINE.md` §1.2.

1. ~~The disagreement null is definition-dependent.~~ **RESOLVED — use the
   `build_press_index.py` measure** (divide by *directional* claims). Dividing
   by *all* claims, as `model_hit.py:165` does, is contaminated by how many
   non-directional claims a month happened to contain. On the correct measure
   the relationship is cleanly monotone — **0.574 / 0.501 / 0.460** — survives
   dropping the n ≥ 5 filter (0.574/0.500/0.459), holds within expansions
   (0.693/0.595/0.482), and is not a recession proxy (corr = −0.019). The v1
   "no relationship" claim was an artefact of the wrong denominator.
   *Caveat that must be printed with it:* low disagreement means near-unanimous
   optimism, and optimism usually pays, so the effect is partly mechanical.
   Within "improve" claims alone it shrinks from 11.4 to 6.8 points but persists
   (0.641/0.570/0.573).
2. ~~Stated extraction quality may be too high.~~ **RESOLVED — report F1 =
   0.527.** 0.612 is the default-effort row; both `CLAUDE.md` and
   `RESULTS_MONTHLY.md` record the shipped run as `--reasoning-effort low`,
   which measures 0.527 (`result_oss120b_low.json`). The run log was not
   retained, so 0.527 is the conservative reading. Stop citing 0.61.
3. ~~768 vs 761.~~ **RESOLVED — they are different quantities.** 768 months were
   *sampled*; **761** yield at least one scorable US-national claim. State both,
   never one as the other.
4. **The recession accuracy gap is partly mechanical.** NBER dates are both the
   scoring ground truth and the split variable; in recessions 65% of claims are
   "improve" and 68.3% of all misses are "improve" claims. Lead with the *mix*
   (Panel 2), present the gap as its consequence. Don't let +19.1 read as an
   independent finding.
5. **The scope filter is not neutral.** The 13,935 discarded regional/industry/
   foreign claims are *more* optimistic than the national ones kept (78.7% vs
   73.1% improve). True press-wide optimism is somewhat higher than the index
   shows — worth one sentence, since the index is a headline deliverable.
6. **Do not claim** contrarians were more accurate, or that the model usefully
   predicts which forecasts come true (AUC 0.561).

---

## 10. Quotable material

`famous_calls.csv` — 344 named, dated forecasts, hit rate 0.488; 51 from the
1929 Crash window, 52 from the 1920 Depression. Best poster candidates are
assertive, named, and wrong:

- **Roger W. Babson, 1920-11-20** — *"Ten Reasons Why Business Outlook In United
  States Favorable To Continued Prosperity"* → predicted improve, economy
  worsened, in the middle of the 1920–21 depression.
- **Judge Gary (US Steel), 1920-10-29** — *"Business Outlook Unusually Bright,
  Says Gary, Steel Head"* → improve / worsened.
- **Sawyer, 1929-12-18** — *"Outlook Good. Sawyer was optimistic over the
  business outlook for next year."* → improve / worsened.
- **President Roosevelt, 1907-11-09** — *"WILL NOT CALL EXTRA SESSION … Thinks
  It Unnecessary as Panic Is Nearly Over"* → improve, and this one **hit**.

Pair a wrong-and-confident 1929 quote with the 86.1% October figure.

---

## 11. Existing figure assets

`poster_figures/`: figA_mechanism · figB_consequence · figC_no_learning ·
figD_what_predicts · figE_method
`prelim_figures/`: fig1_extraction_gap · fig2_hit_by_episode ·
fig3_optimism_asymmetry · fig4_scope_gate · fig5/8_accuracy_over_time ·
fig6_index_net_direction · fig7_index_uncertainty
`figures/` (27 more, from earlier arms): fig_greenbook_benchmark ·
fig_spf_benchmark · fig_three_way_benchmark · fig_leaderboard ·
fig_hit_by_episode · fig_epu_vs_accuracy · fig_era_stability ·
fig_model_calibration · fig_regret · fig_partisan · fig_geography …

**Figures that do not yet exist and are worth making:**
1. Accuracy-by-topic bar chart with a 50% reference line (§5) — the biggest
   effect in the data and currently unplotted.
2. The 1929 month-by-month optimism strip (§6), annotated with a wrong-and-
   confident quote from §10.
3. Price "up" vs "down" hit rate by decade (§5a) — the zero-sum regime pattern.

`fig_three_way_benchmark.png` already covers §7a and is the strongest benchmark
asset in the repo, but its input data must be restored from git first.
