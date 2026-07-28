# Did Anyone See It Coming?

**Machine-reading a century of American newspaper economic forecasts, 1900–1963.**
BU RISE Practicum — Data Science.

For sixty years American newspapers told readers what the economy would do next.
This project extracts those forecasts at scale and scores every one against what
the economy actually did.

```
15,721 newspaper pages      Library of Congress, sampled from all 768 months 1900–1963
30,765 forecasts extracted  direction, topic, horizon, hedging, speaker, scope
14,251 scored               against NBER business-cycle dates and Federal Reserve series
```

## What we found

- **The forecast mix never responded to the economy.** Downbeat forecasts were
  24.1% of the total in expansions and 24.3% in recessions — identical. About
  72% of all forecasts were upbeat, in booms and busts alike.
- **So accuracy collapsed when it mattered:** 58.8% in expansions → 39.7% in
  recessions (gap +18.7 pts, 95% CI [+12.1, +24.8], block-bootstrapped).
  Pessimists were right when it counted; there were never enough of them.
- **Sixty years, no improvement.** Annual accuracy is flat (53.7% in the 1900s →
  49.9% in the 1960s), and no publisher beat the rest (seven papers with 200+
  scored forecasts all fall between 44% and 56%).
- **Keyword search would have lost the story.** The standard approach recovers
  27% of the forecasts on a page; whole-page LLM reading recovers 73% at higher
  precision — measured against a hand-built gold standard.

Full write-up: [`docs/POSTER.md`](docs/POSTER.md) ·
findings [`docs/RESULTS_MONTHLY.md`](docs/RESULTS_MONTHLY.md) ·
method [`docs/SCORING.md`](docs/SCORING.md)

## How correctness is decided

The language model decides **what was predicted**. Real economic data decides
**whether it came true**. The two never mix — asking a model with hindsight to
grade forecasts would measure nothing.

The scorer is deterministic, so it is *proven* rather than merely measured:
33 known-answer tests against a synthetic economy. Claims that cannot be fairly
scored — no clear direction, foreign or regional scope, or a date outside a
series' coverage — are marked with a reason and left unscored, never guessed.

## Quick start

```bash
pip install pandas numpy scikit-learn matplotlib requests openpyxl python-pptx truststore

# everything runs from the repo root
python tests/test_offline.py      # 90 checks
python tests/test_scoring.py      # 33 known-answer scorer proofs
python tests/test_forecasts.py    # 33 checks

# rebuild the analysis from committed data (no network, no API key)
python src/score_predictions.py --claims data/claims/claims_monthly.jsonl \
    --out data/scored/monthly_scored.csv
python src/make_poster_figures.py
python src/build_poster.py        # -> RISE_Poster_2026.pptx
```

Rebuilding the *corpus* from scratch (a ~24 h scrape plus a paid extraction pass)
is documented in [`CLAUDE.md`](CLAUDE.md) and
[`docs/MONTHLY_PIPELINE.md`](docs/MONTHLY_PIPELINE.md).

## Repository

| path | what it holds |
|---|---|
| `src/` | all Python — collection, extraction, scoring, analysis, figures |
| `tests/` | three suites, 156 checks total |
| `data/claims/` | extracted forecasts (the expensive artifact) |
| `data/scored/` | scored claims + the monthly press-expectations index |
| `data/corpus/` | raw newspaper pages — **gitignored**, 738 MB, regenerable |
| `validation/` | the gold standard, its eval harness, and human hand-grading |
| `figures/` | poster figures and exploratory plots |
| `docs/` | poster text, results, methods, operational runbooks |

**Data sources**, all public: Library of Congress *Chronicling America*; NBER
business-cycle chronology; Federal Reserve FRED (INDPRO, CPIAUCNS, UNRATE,
historical common-stock index); Philadelphia Fed Livingston Survey.

Total compute cost to build everything: **≈ $25**.
