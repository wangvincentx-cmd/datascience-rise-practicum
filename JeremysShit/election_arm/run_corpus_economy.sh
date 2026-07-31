#!/usr/bin/env bash
# Batch-run the ProQuest economy pipeline over the year-sharded CORPUS -- one
# ProQuest dataset spanning 1900-2010 (~146k articles), parsed by
# `tdm_parse.py --corpus`. Runs INSIDE the TDM Studio workbench.
#
# THE CORPUS IS NOW THE ONLY DATASET. The older layout -- one keyword query per
# crisis/placebo window, driven by the deleted run_all_economy.sh -- is scrapped:
# it let the query decide how many articles each window held, so "crisis windows
# contain more forecasts" was partly a fact about the query. The corpus gives a
# real denominator (articles read per year) and a continuous series. Window
# labels still ride along on each article (derived from its date, see
# tdm_parse.py) so a crisis-vs-placebo cut is still available as a SUBSET of the
# corpus -- but nothing here prioritises or filters by them any more.
#
# TWO PHASES, IN THIS ORDER.
#
#   1. EXTRACT every year shard, oldest first, until all ~146k pages have been
#      read by gpt-4o-mini.
#   2. VERIFY -- gpt-4o-mini's second pass over the candidate claims -- only
#      ONCE EVERY PAGE IS EXTRACTED.
#
# Verification deliberately waits for the whole corpus. It is a precision filter
# whose prompt and keep-rate are a property of the run, and running it against a
# growing pile of claims would apply it to early years under one set of
# conditions and late years under another; the keep-rate would then be
# uninterpretable as a single number. It is also the cheaper pass (one call per
# page, a quote plus ~400 chars of context, versus extraction's whole-document
# calls), so it is the one that should wait.
#
# THE SHAPE OF THIS RUN. Extraction is ~1.2-1.5 LLM calls per article
# (extract_gpt.py chunks at 8000 chars). Against a daily cost cap that stopped an
# earlier run at ~4.3k articles, expect WEEKS of once-a-day runs, then several
# more days of verification. Everything here resumes off its own output, so the
# operating procedure is simply to run this same command once a day until
# `corpus_progress.py` shows 0 pages left and every year verified. The phase
# switch is automatic -- you do not have to notice when extraction finishes.
#
# Usage (from /home/ec2-user/SageMaker/election_arm):
#   bash run_corpus_economy.sh              # extract; verify once extraction is done
#   bash run_corpus_economy.sh --extract    # extraction pass only
#   bash run_corpus_economy.sh --verify     # verification pass only
#   bash run_corpus_economy.sh 1973 1974    # just these years
#   bash run_corpus_economy.sh --export     # also bundle the tarball (see below)

set -u

PY="${PY:-/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python3.12}"
if [ ! -x "$PY" ]; then
    echo "Interpreter not found: $PY"
    echo "Run:  python vm_doctor.py   and export the PY= line it prints."
    exit 1
fi

PHASE=auto
DO_EXPORT=0
FORCE=0
YEARS_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --export)         DO_EXPORT=1 ;;
        --extract)        PHASE=extract ;;
        --verify)         PHASE=verify ;;
        --force)          FORCE=1 ;;
        -*)               echo "unknown flag: $1"; exit 1 ;;
        *)                YEARS_ARG="$YEARS_ARG $1" ;;
    esac
    shift
done

if [ ! -f tdm_parse.py ] || [ ! -f gpt_sample.txt ]; then
    echo "Run this from the election_arm folder (tdm_parse.py and gpt_sample.txt must be here)."
    exit 1
fi

# Never two at once: concurrent runs fight over the rate-limited proxy and
# double-process articles into duplicate claims. Cheap to check, expensive to undo.
if ps -ef | grep -E "run_corpus_economy|extract_gpt|verify_gpt" \
          | grep -v grep | grep -qv "$$"; then
    echo "*** A pipeline process is already running. Not starting a second one:"
    ps -ef | grep -E "run_corpus_economy|extract_gpt|verify_gpt" | grep -v grep
    exit 1
fi

# Chronological over every shard on disk. There is no prioritised ordering any
# more: with the per-window datasets scrapped, the deliverable is the whole
# 1900-2010 series, and every year is equally part of it. `nodate` is not a year
# shard and is never extracted (tdm_parse.py parks undated articles there).
if [ -n "$YEARS_ARG" ]; then
    YEARS="$YEARS_ARG"
else
    YEARS=$(ls data/raw 2>/dev/null \
            | sed -n 's/^proquest_economy_\([0-9][0-9][0-9][0-9]\)\.jsonl$/\1/p' \
            | sort)
fi

if [ -z "${YEARS// /}" ]; then
    echo "No corpus year shards in data/raw. Parse the dataset first:"
    echo "  $PY tdm_parse.py --arm economy --corpus --dataset-dir <folder>"
    exit 1
fi

echo "=== starting state ==="
$PY corpus_progress.py
echo

STOPPED=0

# ---------------------------------------------------------------------------
# Phase 1: extraction
# ---------------------------------------------------------------------------
if [ "$PHASE" = auto ] || [ "$PHASE" = extract ]; then
    echo "########## PHASE 1: EXTRACT ##########"
    for y in $YEARS; do
        echo
        echo "=============================================================="
        echo "  extract $y"
        echo "=============================================================="
        if [ ! -f "data/raw/proquest_economy_${y}.jsonl" ]; then
            echo "  SKIP: no shard at data/raw/proquest_economy_${y}.jsonl"
            continue
        fi

        $PY extract_gpt.py --source proquest --window "$y"
        rc=$?
        if [ $rc -eq 2 ]; then
            echo "  DAILY RATE LIMIT hit on $y during EXTRACT -- stopping the batch."
            echo "  Re-run this same command after the quota resets to resume here."
            STOPPED=1
            break
        elif [ $rc -ne 0 ]; then
            echo "  extract failed on $y (rc=$rc); skipping to next year"
            continue
        fi

        $PY strip_for_export.py "data/predictions/pred_proquest_economy_${y}.jsonl"
    done
fi

# ---------------------------------------------------------------------------
# The gate between the phases: is every parsed page extracted?
#
# `corpus_progress.py --left` counts done pages exactly the way extract_gpt.py
# does -- a `no_predictions` record counts as done (the call happened), and a
# record at an older schema version does NOT (extract_gpt will redo it). Asking
# it, rather than tracking success in this loop, means a run that was interrupted
# yesterday is judged on what is actually on disk.
# ---------------------------------------------------------------------------
LEFT=$($PY corpus_progress.py --left 2>/dev/null)
case "$LEFT" in
    ''|*[!0-9]*) echo; echo "*** could not read pages-left from corpus_progress.py"
                 echo "*** not starting verification blind."; exit 1 ;;
esac

DO_VERIFY=0
if [ "$PHASE" = auto ]; then
    if [ "$LEFT" -eq 0 ]; then
        DO_VERIFY=1
    else
        echo
        echo "########## extraction is NOT finished: $LEFT page(s) left ##########"
        echo "Verification runs once every page has been read, not before -- see the"
        echo "note at the top of this script. Run this same command again after the"
        echo "quota resets. corpus_progress.py estimates how many runs remain."
    fi
elif [ "$PHASE" = verify ]; then
    if [ "$LEFT" -eq 0 ] || [ $FORCE -eq 1 ]; then
        DO_VERIFY=1
        [ "$LEFT" -gt 0 ] && echo "*** --force: verifying with $LEFT page(s) still unextracted." \
                          && echo "*** The keep-rate from this run describes a partial corpus."
    else
        echo "Refusing to verify: $LEFT page(s) are still unextracted."
        echo "Finish extraction first (bash run_corpus_economy.sh), or pass --force"
        echo "if you deliberately want a partial-corpus keep-rate."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Phase 2: verification (gpt-4o-mini's second pass)
#
# The gold eval put raw gpt-4o-mini at precision 0.409; the verifier lifts it to
# 0.700 (F1 0.458 -> 0.512) by dropping candidates that are not forecasts.
# Output goes to data/verified/, NEVER data/predictions/: analyze_economy.py
# globs pred_*_economy_*.jsonl there and would otherwise double-count every claim
# and re-admit the rejects. verify_gpt.py refuses that path.
#
# This phase is itself resumable and quota-capped -- pages already judged are
# skipped -- so it too may take several daily runs over a corpus this size.
# ---------------------------------------------------------------------------
if [ $DO_VERIFY -eq 1 ] && [ "${SKIP_VERIFY:-0}" = "1" ]; then
    echo
    echo "SKIP_VERIFY=1 -- extraction is complete but leaving the corpus unverified"
elif [ $DO_VERIFY -eq 1 ]; then
    echo
    echo "########## PHASE 2: VERIFY (all pages extracted) ##########"
    for y in $YEARS; do
        claims="data/predictions/pred_proquest_economy_${y}.jsonl"
        if [ ! -f "$claims" ]; then
            continue
        fi
        echo
        echo "=============================================================="
        echo "  verify $y"
        echo "=============================================================="
        $PY verify_gpt.py \
            --claims "$claims" \
            --pages  "data/raw/proquest_economy_${y}.jsonl" \
            --out    "data/verified/pred_proquest_economy_${y}.jsonl"
        rc=$?
        if [ $rc -eq 2 ]; then
            echo "  DAILY RATE LIMIT hit on $y during VERIFY -- stopping the batch."
            echo "  Extraction is DONE and saved; only verification resumes."
            STOPPED=1
            break
        elif [ $rc -ne 0 ]; then
            echo "  verify failed on $y (rc=$rc); the unverified claims are still"
            echo "  on disk, so re-running just redoes verification"
            continue
        fi
        $PY strip_for_export.py "data/verified/pred_proquest_economy_${y}.jsonl"
    done
fi

echo
echo "=== ending state ==="
$PY corpus_progress.py
[ $STOPPED -eq 1 ] && echo && echo "Quota stopped this run. Run the same command again tomorrow."

# Bundling is OPT-IN. The export allowance is ~15 MB on a ROLLING 7-DAY window,
# and this corpus takes weeks: a tarball built automatically on every daily run
# would spend that allowance on partial data and leave none for the finished set.
# Export deliberately, when a chunk of years is actually complete.
if [ $DO_EXPORT -eq 0 ]; then
    echo
    echo "(no tarball built -- re-run with --export when you want one;"
    echo " the 15 MB export cap is a rolling 7-day budget, so spend it deliberately)"
    exit 0
fi

echo
echo "=== bundling stripped exports into one file to Export ==="
if ls data/verified/*.export.jsonl data/predictions/*.export.jsonl >/dev/null 2>&1; then
    rm -rf export_staging proquest_exports.tar.gz
    mkdir -p export_staging/verified export_staging/unverified
    cp data/verified/*.export.jsonl      export_staging/verified/    2>/dev/null
    cp data/predictions/*.export.jsonl   export_staging/unverified/  2>/dev/null

    # The check that matters, BEFORE anything is archived: article text may not
    # leave the VM. A failure to read the files is itself a failure.
    echo "  checking no article text is present..."
    n_files=$(find export_staging -name '*.jsonl' | wc -l | tr -d ' ')
    if [ "$n_files" -eq 0 ]; then
        echo "    *** no files staged -- nothing to export, and nothing checked."
        exit 1
    fi
    if grep -l '"quote"' export_staging/*/*.jsonl 2>/dev/null | head -1 | grep -q .; then
        echo "    *** FOUND a \"quote\" field. DO NOT EXPORT."
        echo "    *** Re-run strip_for_export.py on the offending file(s):"
        grep -l '"quote"' export_staging/*/*.jsonl | sed 's/^/        /'
        exit 1
    fi
    echo "    OK: checked $n_files file(s), no \`quote\` field in any record."

    tar -czf proquest_exports.tar.gz -C export_staging .
    rm -rf export_staging
    echo
    echo "Export THIS one file (set data_to_export to it):"
    echo "  /home/ec2-user/SageMaker/election_arm/proquest_exports.tar.gz"
    ls -lh proquest_exports.tar.gz
    echo
    echo "  If that size is near 15 MB, export a subset of years instead --"
    echo "  the cap is shared and rolling."
else
    echo "  no stripped export files yet"
fi
