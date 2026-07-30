#!/usr/bin/env bash
# Batch-run the ProQuest economy pipeline over a year-sharded CORPUS -- one
# ProQuest dataset spanning many years (e.g. a single 1900-2010 query), parsed by
# `tdm_parse.py --corpus`. Runs INSIDE the TDM Studio workbench.
#
# This is the sibling of run_all_economy.sh, which handles the older layout of
# one keyword-query dataset per window. Same stages per unit of work
# (extract -> verify -> strip), but the unit is a year, not a window, and there
# are ~110 of them instead of 9.
#
# THE SHAPE OF THIS RUN. A ~146k-article corpus is roughly 30x the keyword
# windows. Extraction is ~1.2-1.5 LLM calls per article (extract_gpt.py chunks at
# 8000 chars), and verification adds one call per surviving claim. Against a
# daily cost cap that stopped an earlier run at ~4.3k articles, expect this to
# take WEEKS of once-a-day runs, not an afternoon. Everything here is built for
# that: each stage resumes off its own shard's output, so the correct operating
# procedure is simply to run this same command once a day until
# `corpus_progress.py` shows 0 left.
#
# Usage (from /home/ec2-user/SageMaker/election_arm):
#   bash run_corpus_economy.sh              # window years first, then the rest
#   bash run_corpus_economy.sh --chrono     # strict chronological order
#   bash run_corpus_economy.sh 1973 1974    # just these years
#   bash run_corpus_economy.sh --export     # also bundle the tarball (see below)

set -u

PY="${PY:-/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python3.12}"
if [ ! -x "$PY" ]; then
    echo "Interpreter not found: $PY"
    echo "Run:  python vm_doctor.py   and export the PY= line it prints."
    exit 1
fi

ORDER=windows-first
DO_EXPORT=0
YEARS_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --export)        DO_EXPORT=1 ;;
        --chrono)        ORDER=chrono ;;
        --windows-first) ORDER=windows-first ;;
        -*)              echo "unknown flag: $1"; exit 1 ;;
        *)               YEARS_ARG="$YEARS_ARG $1" ;;
    esac
    shift
done

if [ ! -f tdm_parse.py ] || [ ! -f gpt_sample.txt ]; then
    echo "Run this from the election_arm folder (tdm_parse.py and gpt_sample.txt must be here)."
    exit 1
fi

# Never two at once: concurrent runs fight over the rate-limited proxy and
# double-process articles into duplicate claims. Cheap to check, expensive to undo.
if ps -ef | grep -E "run_corpus_economy|run_all_economy|extract_gpt" \
          | grep -v grep | grep -qv "$$"; then
    echo "*** A pipeline process is already running. Not starting a second one:"
    ps -ef | grep -E "run_corpus_economy|run_all_economy|extract_gpt" | grep -v grep
    exit 1
fi

# Ordering is a scheduling decision, not a cosmetic one. At a few thousand pages
# a day the whole corpus takes weeks, so by default the years that a configured
# crisis/placebo window touches go FIRST -- that makes the crisis-vs-placebo
# result available in days instead of at the very end, while the remaining years
# fill in the continuous 1900-2010 series behind it. --chrono opts out.
if [ -n "$YEARS_ARG" ]; then
    YEARS="$YEARS_ARG"
else
    YEARS=$($PY - "$ORDER" <<'PYEOF'
import csv, re, sys
from pathlib import Path

mode = sys.argv[1]
shard_re = re.compile(r"^proquest_economy_(\d{4})\.jsonl$")
years = sorted(m.group(1)
               for m in (shard_re.match(p.name)
                         for p in Path("data/raw").glob("proquest_economy_*.jsonl"))
               if m)
if mode == "windows-first":
    window_years = set()
    with open("data/windows_economy.csv") as f:
        for row in csv.DictReader(f):
            for y in range(int(row["start_date"][:4]), int(row["end_date"][:4]) + 1):
                window_years.add(str(y))
    years = ([y for y in years if y in window_years]
             + [y for y in years if y not in window_years])
print(" ".join(years))
PYEOF
)
fi

if [ -z "${YEARS// /}" ]; then
    echo "No corpus year shards in data/raw. Parse the dataset first:"
    echo "  $PY tdm_parse.py --arm economy --corpus --dataset-dir <folder>"
    exit 1
fi

echo "=== starting state ==="
$PY corpus_progress.py
echo
echo "order: $ORDER"

STOPPED=0
for y in $YEARS; do
    echo
    echo "=============================================================="
    echo "  year $y"
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

    # Second pass. The gold eval put raw gpt-4o-mini at precision 0.409; the
    # verifier lifts it to 0.700 (F1 0.458 -> 0.512) by dropping candidates that
    # are not forecasts. Output goes to data/verified/, NEVER data/predictions/:
    # analyze_economy.py globs pred_*_economy_*.jsonl there and would otherwise
    # double-count every claim and re-admit the rejects. verify_gpt.py refuses
    # that path.
    if [ "${SKIP_VERIFY:-0}" = "1" ]; then
        echo "  SKIP_VERIFY=1 -- leaving $y unverified"
    else
        $PY verify_gpt.py \
            --claims "data/predictions/pred_proquest_economy_${y}.jsonl" \
            --pages  "data/raw/proquest_economy_${y}.jsonl" \
            --out    "data/verified/pred_proquest_economy_${y}.jsonl"
        rc=$?
        if [ $rc -eq 2 ]; then
            echo "  DAILY RATE LIMIT hit on $y during VERIFY -- stopping the batch."
            echo "  Extraction for $y is DONE and saved; only verification resumes."
            STOPPED=1
            break
        elif [ $rc -ne 0 ]; then
            echo "  verify failed on $y (rc=$rc); the unverified claims are still"
            echo "  on disk, so re-running just redoes verification"
            continue
        fi
        $PY strip_for_export.py "data/verified/pred_proquest_economy_${y}.jsonl"
    fi

    $PY strip_for_export.py "data/predictions/pred_proquest_economy_${y}.jsonl"
done

echo
echo "=== ending state ==="
$PY corpus_progress.py
[ $STOPPED -eq 1 ] && echo && echo "Quota stopped this run. Run the same command again tomorrow."

# Bundling is OPT-IN here, unlike run_all_economy.sh. The export allowance is
# ~15 MB on a ROLLING 7-DAY window, and this corpus takes weeks: a tarball built
# automatically on every daily run would spend that allowance on partial data and
# leave none for the finished set. Export deliberately, when a chunk of years is
# actually complete.
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
