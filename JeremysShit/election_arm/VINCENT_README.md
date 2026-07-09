# Who Saw It Coming? American Predictions and Their Accuracy, 1900-2010

The project pulls a century of predictions out of American newspapers, both
election forecasts and economic forecasts, grades each one with an LLM,
validates the grading with human double-coding, checks every claim against
what actually happened, and measures who was right, when, and why.

Two arms share one pipeline:

- ELECTIONS: claims about who will win, scored against actual winners,
  every presidential cycle 1896-2008.
- ECONOMY: claims about recession or recovery, scored against the NBER
  business-cycle chronology, inside 13 crisis windows and 6 calm placebo
  windows from 1905 to 2008, benchmarked against professional economists
  (the Livingston Survey, 1946 onward).

Data sources: Library of Congress Chronicling America (full OCR, many
papers, through ~1963) and the NYT Article Search API (headline + lead only,
one paper, through 2008). No scraping of paywalled archives.

## Setup (everyone does this once)

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...        # console.anthropic.com
    export NYT_API_KEY=...              # free at developer.nytimes.com, enable Article Search API

Put your real email in HEADERS at the top of download_loc.py (LOC asks for a
contact). Then run the offline tests, which need no network and no keys:

    python test_offline.py

All 21 lines must say PASS before anyone spends API budget.

## Pipeline

STAGE 1 DOWNLOAD (resume-safe; kill and rerun anytime)

    python download_loc.py --arm elections --cycle 1948 --max-pages 2    # tiny test
    python download_loc.py --arm economy  --window crash_1929 --max-pages 2
    python download_nyt.py --arm economy  --window gfc_2008 --max-pages 2

    python download_loc.py --arm elections        # full: 1896-1960
    python download_loc.py --arm economy          # full: pre-1963 windows
    python download_nyt.py --arm elections        # full: 1900-2008 (multi-day; 500 req/day cap)
    python download_nyt.py --arm economy          # full: all windows incl. post-1963

Every query's total hits and pages fetched are logged to data/search_log.csv.
That file is the "we sampled the corpus, we didn't cherry-pick" exhibit; keep
it and cite it.

After the first real LOC run, confirm OCR text is nonempty:

    python -c "import json,glob; f=glob.glob('data/raw/loc_*.jsonl')[0]; r=[json.loads(l) for l in open(f)]; print(f, len(r), 'pages,', sum(1 for x in r if x['ocr_text']), 'with text')"

STAGE 2 EXTRACT (one Claude call per page; test with --limit first)

    python extract_predictions.py --source loc --arm economy --window crash_1929 --limit 20
    python extract_predictions.py --source nyt --arm elections --window 1980

Uses claude-haiku-4-5 and truncates pages to 12k chars for cost. Check the
console usage page after a --limit 20 run and extrapolate before full runs.

STAGE 3 VALIDATE (the answer to "how do you know the LLM's labels are right")

    python validate_kappa.py sample --arm economy

This writes data/validation_sample.csv. TWO team members fill grader_A and
grader_B INDEPENDENTLY, without looking at each other's answers or opening
data/validation_llm_key.csv. Then:

    python validate_kappa.py kappa --arm economy

Report all three kappas (A vs B, A vs LLM, B vs LLM) in the paper. Above 0.6
is substantial agreement, above 0.8 near-perfect. Read
data/validation_disagreements.csv; it shows how the LLM errs. Repeat for the
elections arm.

STAGE 4 SCORE

    python analyze_elections.py     # -> data/scored_claims.csv + tables
    python analyze_economy.py       # -> data/scored_economy.csv + tables

Elections tables: accuracy by source type (polls vs editorials vs betting
odds), by publisher, by cycle, hedged vs firm, LOC vs NYT. Economy tables:
crisis vs placebo hit rates (the base-rate control), by window, by voice, by
source, Brier scores for the overconfidence result, and the
optimism-at-turning-points number.

Economy scoring rule: each claim predicts recession or expansion at
claim date + horizon; the actual state comes from data/nber_recessions.csv
(recession = month after peak through trough, NBER convention). Brier
confidence mapping is firm=0.90, hedged=0.70; a documented assumption, tune
CONFIDENCE in analyze_economy.py and note it in methods.

STAGE 5 EXPERT BENCHMARK (one manual download)

Download the Livingston Survey median forecasts from the Philadelphia Fed
(Surveys & Data -> Livingston Survey -> Historical Data) and save
data/livingston_medians.csv with columns
survey_date,ip_base,ip_6m_forecast,ip_12m_forecast (see the docstring in
livingston_benchmark.py). Then:

    python livingston_benchmark.py

That yields the economists' 6-month direction hit rate, scored on the same
NBER rule as the newspaper claims, plus the did-the-experts-see-the-peak
subset. Compare against 1946+ rows of scored_economy.csv for the headline
chart: newspapers vs the professionals.

STAGE 6 MODEL (which factors made predictions accurate)

    python model.py --arm elections --test-from 1980
    python model.py --arm economy   --test-windows gfc_2008,dotcom_2001,calm_2005

NEVER change the split to random. Claims inside one cycle or one crisis share
the outcome; a random split leaks it and fakes high accuracy. Hold out whole
cycles or whole windows. The logistic-regression coefficient printout is the
feature-importance answer to the core research question.

## Editing the study design

The crisis and placebo windows live in data/windows_economy.csv; add or edit
rows there, no code changes needed. Same for data/nber_recessions.csv and
data/ground_truth_elections.csv. Search phrases are lists at the top of the
two downloaders. The 1987 window is a deliberate negative case: a crash with
no recession within 12 months, which catches models and newspapers that
equate panic with recession.

## Known limits (say these in the paper, they are features not bugs)

1. NYT returns headline + lead only, never full text. Post-1963 recall is
   structurally lower than pre-1963 LOC full-OCR recall. Never compare
   accuracy across that boundary without saying so.
2. Scientific polls barely exist before 1936; the polls-vs-papers split is
   thin early. Betting odds partially fill that role pre-1936.
3. The phrase lists bound recall: forecasts worded differently are missed.
   search_log.csv documents exactly what was searched.
4. Brier confidence is mapped from the hedged flag, not elicited
   probabilities. Coarse, but transparent and uniform across a century.
5. State-level election claims stay unscored until someone adds a
   state-results file.

## Suggested division of labor (three people)

Person A: downloads. Run both downloaders across all windows and cycles
(NYT takes multiple days due to the 500/day cap), monitor search_log.csv,
verify OCR nonempty, keep data/raw/ synced to the shared drive.

Person B: extraction + validation. Run extract_predictions.py per window,
watch cost, own the kappa protocol (both graders, disagreement review),
tune the prompts if disagreements reveal systematic LLM errors.

Person C: scoring + benchmark + model. Own analyze_*.py outputs, do the
Livingston download and benchmark, run the models, build the figures
(crisis vs placebo bars, newspapers vs Livingston, calibration plot from
Brier components, feature importances).

Everyone: rerun test_offline.py after touching any script.

## Files

    download_loc.py           LOC downloader, both arms, search logging
    download_nyt.py           NYT downloader, both arms, search logging
    extract_predictions.py    LLM extraction, election + economy schemas
    validate_kappa.py         double-coding sample + Cohen's kappa
    analyze_elections.py      election scoring + tables
    analyze_economy.py        NBER scoring + Brier + crisis/placebo tables
    livingston_benchmark.py   economists' hit rate on the same rule
    model.py                  logistic regression + gradient boosting, both arms
    test_offline.py           21 offline checks of all parsing + scoring
    data/ground_truth_elections.csv   winners 1896-2008
    data/nber_recessions.csv          NBER peaks/troughs 1899-2009
    data/windows_economy.csv          13 crisis + 6 placebo windows (editable)
