# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this project is

One research project: **did American newspapers see economic downturns coming?**
It machine-reads Library of Congress newspaper pages (1900–1963), extracts every
forward-looking economic forecast, and scores each one against what the economy
actually did.

**The rule the whole project rests on:** the language model decides *what was
predicted*; real economic data (NBER business-cycle dates, FRED series) decides
*whether it came true*. The model is never asked whether a forecast was right —
that would be hindsight grading itself. Keep these separate in any new code.

Deliverables live in `docs/`: `POSTER.md` (the poster text), `RESULTS_MONTHLY.md`
(findings), `SCORING.md` (how correctness is decided and why it is trustworthy).

## Layout

```
src/          all Python, flat on purpose — see "Why src/ is flat" below
tests/        three suites: test_offline (129), test_scoring (33), test_forecasts (33)
data/
  corpus/     raw pages (GITIGNORED, huge, regenerable by re-scraping)
  claims/     extractor output — the expensive artifact, committed
  scored/     scorer output + the monthly press index, committed
  reference/  Livingston/SPF spreadsheets, publisher metadata
  proquest/   37k claims 1965-2009 from a ProQuest run (extends past LOC's 1963 end)
  v1_outputs/ frozen outputs of the original episode-based pipeline
validation/   gold_extraction/ (the gold standard + eval harness), handgrade_newspapers/
figures/      poster_figures/ (final), prelim_figures/, v1_episode/
docs/         poster text, results, methods, operational runbooks
notebooks/    Livingston survey analysis
cache/        raw API responses (GITIGNORED)
```

## Pipeline (run from the repo root)

```
# 1. collect  — ~24 h, free, resumable; parallelisable across machines
python src/scrape_monthly.py --stage both
python src/combine_shards.py --in-dir data/corpus/monthly     # if sharded

# 2. extract  — the step that costs money; ALWAYS cost-test 100 pages first
python src/extract_llm.py --pages data/corpus/monthly/pages_monthly.jsonl \
    --out data/claims/claims_monthly.jsonl --model openai/gpt-oss-120b \
    --base-url https://api.deepinfra.com/v1/openai --api-key-env DEEPINFRA_API_KEY \
    --chunk-chars 40000 --reasoning-effort low --workers 12

# 3. score    — deterministic, no LLM, no network beyond cached FRED
python src/score_predictions.py --claims data/claims/claims_monthly.jsonl \
    --out data/scored/monthly_scored.csv

# 4. analyse
python src/build_press_index.py --claims data/claims/claims_monthly.jsonl \
    --pages data/corpus/monthly/pages_monthly.jsonl --out data/scored/press_index.csv
python src/model_hit.py --scored data/scored/monthly_scored.csv
python src/make_poster_figures.py && python src/build_poster.py
```

Always run the three suites after a change:
```
python tests/test_offline.py && python tests/test_scoring.py && python tests/test_forecasts.py
```

## Why src/ is flat

35 of 37 modules are directly runnable entry points, and they import each other
by bare name (`from truth_data import ...`). Nesting them in stage subfolders
would require a sys.path bootstrap in every file. Flat keeps every script
runnable and every import working; the pipeline stages are documented above
instead of encoded in directory names.

Scripts assume the **repo root is the working directory** — data paths inside
them are root-relative (`data/claims/...`). The test files chdir to the root
themselves, so they run from anywhere.

## Rules that are not negotiable

- **No hindsight in labels.** Never pass an episode name, outcome, or recession
  flag to an extraction or grading prompt. This bug existed once (`"1929 Crash"`
  in the grader prompt) and it silently invalidates everything downstream.
- **Accuracy is never the metric.** Report hit rate, error direction, Brier, and
  confidence intervals. The target is rare and clustered.
- **Split by time block or episode, never randomly.** Forecasts within an era
  share wire copy and one macro reality; the effective sample is ~21 blocks, not
  14,251 claims. Use grouped CV and block bootstraps.
- **Unscorable means unscored.** Claims with no direction, foreign/regional
  scope, or a date outside a series' coverage are marked with a reason and left
  out — never guessed. The scored fraction is a number we report, not maximise.
- **Cost-test before any full extraction run.** Measure 100 pages, project, and
  only proceed under an explicit cap. Estimates have been wrong before.

## Environment

No repo-level venv. Requires: `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`requests`, `openpyxl`, `python-pptx`, and `truststore` (needed on machines
behind TLS-inspecting proxies, or every HTTPS call fails cert verification).

API keys live in `.env` (gitignored): `DEEPINFRA_API_KEY`, `GEMINI_API_KEY`.
loc.gov and FRED need no key.
