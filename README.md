# A Historical Analysis of Newspaper and Economic Forecasts Incorporating Machine Learning

**BU RISE Practicum — Data Science.**
Bode Bosell, Jeremy Liu, Vincent Wang (equal contribution) · mentor Eugene Pinsky,
Boston University.

Poster: [`RISE_Poster_2026_v2.pptx`](RISE_Poster_2026_v2.pptx) (36 × 48 in).

## The question

Newspapers have told readers what the economy would do next for over a hundred
years, and nobody has ever graded those predictions against what actually
happened. Modern forecasters do badly at it — Campbell (2024) found them
overconfident: 53% stated confidence, 23% correct. The historical press had never
been tested the same way.

**When the press said the economy would improve or worsen, was it right?**

```
15,721 newspaper pages       Library of Congress, sampled from all 768 months 1900-1963
30,765 forecasts extracted   direction · topic · horizon · hedging · speaker · scope
14,251 scored                against NBER business-cycle dates and Federal Reserve series
```

On top of the scored record we fit two models: one that predicts **whether a
given forecast will come true**, and one that predicts **whether a recession
starts in the next 12 months** from the press signal alone.

## What we found

**The press was always optimistic.** About 72% of forecasts were upbeat, and the
mix barely moved between expansions and recessions. Accuracy dropped sharply
right before downturns — newspapers generally got *more* optimistic just as a
recession was about to start.

**Sixty years, no improvement.** Decade accuracy is flat, 53.7% in the 1900s →
49.9% in the 1960s, and every decade falls inside 44.2–56.4%. No publisher beat
the rest.

**Professionals were not much better.** Survey of Professional Forecasters 54.1%,
Livingston economists 54.4%, Federal Reserve Greenbook 54.0% — all in the same
neighbourhood as the newspapers.

**Forecast model — significant, small.** Letting newspaper labels and economic
data interact (economy × direction terms) reached out-of-fold ROC-AUC **0.648**.
A block permutation test puts the null at 0.477, so **p = 0.048**. Brier 0.240
against a 0.250 base rate. A gradient-boosted classifier on the same inputs
scored 0.617, so linearity was not the binding constraint. *Direction × economic
policy uncertainty* was by far the most important feature.

**Recession model — not significant, but not empty.** Forward-testing AUC
**0.641**, p = 0.187: with only 6 recessions in the window that result is
reachable by chance. Brier 0.233, 0.012 *worse* than the base rate, driven by the
1930s (0.425). Per-decade, AUC clears 0.5 in 4 of 7 decades. What did hold up:
**how much economic coverage there is** correlates with recession risk, and so
does **how much the papers disagree** — but *which way* they leaned (optimistic
vs pessimistic) does not.

Full write-up: [`docs/POSTER.md`](docs/POSTER.md) ·
findings [`docs/RESULTS_MONTHLY.md`](docs/RESULTS_MONTHLY.md) ·
method [`docs/SCORING.md`](docs/SCORING.md) ·
forecast-model variants [`docs/RESULTS_MODEL_VARIANTS.md`](docs/RESULTS_MODEL_VARIANTS.md)

The recession model has no prose write-up yet — its results live as committed
tables and figure scripts in [`more_model_images/`](more_model_images/).

## How correctness is decided

The language model decides **what was predicted**. Real economic data decides
**whether it came true**. The two never mix — asking a model with hindsight to
grade forecasts would measure nothing.

The scorer is deterministic, so it is *proven* rather than merely measured:
33 known-answer tests against a synthetic economy. Claims that cannot be fairly
scored — no clear direction, foreign or regional scope, or a date outside a
series' coverage — are marked with a reason and left unscored, never guessed.

## Extraction quality

The extractor is `gpt-oss-120b` at low reasoning effort, chosen for cost. Against
a hand-built gold standard of 16 pages / 52 forecasts it scores **precision
0.615, recall 0.462, F1 0.527**.

Keyword search — the standard shortcut — gets similar precision (0.61) but
recovers only **27%** of the forecasts on a page, and misses whole categories of
phrasing entirely. Better models exist: the best tested in the bake-off reached
recall 0.73, at roughly 40× the cost. See
[`validation/gold_extraction/RESULTS.md`](validation/gold_extraction/RESULTS.md).

## The two models

**1 · Will this forecast come true?** L1-penalized logistic regression, C = 0.5,
on three kinds of feature:

| block | features |
|---|---|
| newspaper labels | direction (improve/worsen/no change), topic (business, prices, market, employment, other), voice (official, journalist, layperson, expert), hedged, quoted forecaster, named speaker |
| economy at print time | industrial production (6m, 12m, acceleration), CPI year-over-year, unemployment (6m), stocks (6m return, 6m volatility) |
| economy × direction | forecast sign (+1 improve / 0 unclear / −1 worsen) × stock 6m return, stock vs 2-year peak, economic policy uncertainty, output acceleration (6m − 12m) |

```
z = b0 + b_news' x_news + b_econ' x_econ + b_int' (x_econ * d)
```

Economic features are taken as of the month the forecast went to print. Industrial
production, CPI and unemployment are lagged 2 months to prevent leakage.

**2 · Will a recession start in the next 12 months?** L2-penalized logistic
regression, C = 1, on five press features, each computed as the trailing 12-month
average minus the 12 months before that:

- `net_direction` — (improve − worsen) / directional claims
- `disagreement` — 1 − |net_direction|
- `hedge_rate` — hedged claims / all claims
- `expert_rate` — expert claims / all claims
- `attention` — forecasts / pages sampled

Tested two ways: leave-one-decade-out, and a forward-testing model trained on
1900–1930 that refits after every year.

## Quick start

```bash
pip install pandas numpy scikit-learn matplotlib requests openpyxl python-pptx truststore

# everything runs from the repo root
python tests/test_offline.py      # leakage / lag / hindsight guards
python tests/test_scoring.py      # 33 known-answer scorer proofs
python tests/test_forecasts.py    # 33 checks

# rebuild the analysis from committed data (no network, no API key)
python src/score_predictions.py --claims data/claims/claims_monthly.jsonl \
    --out data/scored/monthly_scored.csv
python src/make_poster_figures.py
python src/build_poster.py        # -> RISE_Poster_2026.pptx

# the economy-at-print-time layer and the forecast model
python src/macro_context.py       # -> data/scored/macro_context.csv + attribution
python src/hit_predictor.py       # -> data/models/hit_predictor.joblib
python src/make_macro_figures.py  # -> figures/poster_figures/figI, figJ, figK

# recession-model figures (read committed result CSVs)
python more_model_images/make_recession_model_graphs.py
```

Rebuilding the *corpus* from scratch (a ~24 h scrape plus a paid extraction pass)
is documented in [`CLAUDE.md`](CLAUDE.md) and
[`docs/MONTHLY_PIPELINE.md`](docs/MONTHLY_PIPELINE.md).

## Repository

| path | what it holds |
|---|---|
| `src/` | all Python — collection, extraction, scoring, analysis, figures |
| `tests/` | three suites, ~180 checks |
| `data/claims/` | extracted forecasts (the expensive artifact) |
| `data/scored/` | scored claims + the monthly press-expectations index |
| `data/models/` | the fitted forecast model and its held-out predictions |
| `data/corpus/` | raw newspaper pages — **gitignored**, 738 MB, regenerable |
| `validation/` | the gold standard, its eval harness, and human hand-grading |
| `figures/` | poster figures and exploratory plots |
| `more_model_images/` | recession-model result tables, CSVs and figure scripts |
| `docs/` | poster text, results, methods, operational runbooks |
| `notebooks/` | Livingston survey analysis, and `hit_model.ipynb` — the forecast model end to end |

**Data sources**, all public: Library of Congress *Chronicling America*; NBER
business-cycle chronology; Federal Reserve FRED (INDPRO, CPIAUCNS, UNRATE,
historical common-stock index); Economic Policy Uncertainty index; Philadelphia
Fed Livingston Survey.

Total compute cost to build everything: **≈ $25**.

## Limitations

- 14,251 scorable claims, and no substantial newspaper coverage past 1963.
- Library of Congress OCR is noisy — advertisements and unrelated copy leak in.
- Dataset options were constrained: the NYT API returns titles only, and ProQuest
  supports GPT-4o-mini only.
- The extractor was chosen on budget (F1 0.527), not on quality.

## Future work

- Extend past 1963 with ProQuest or a comparable database.
- Re-extract with a stronger model.
- Include regional and international claims instead of national only.
- Have the recession model also predict whether an ongoing recession continues.
- With more and cleaner data, test genuinely higher-capacity models.

## Acknowledgements

Thanks to our mentors, Eugene Pinsky and Indrajit Kalita, for their expertise and
guidance throughout, and to BU RISE Data Science for the opportunity.
