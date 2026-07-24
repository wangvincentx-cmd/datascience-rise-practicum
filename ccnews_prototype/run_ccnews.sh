#!/bin/bash -l
# SGE array job: one task = one CC-News WARC file for a given month.
#
# Each task downloads its WARC to node-local scratch, extracts article text,
# appends to a per-task JSONL, then deletes the raw WARC. Merge afterwards.
#
# Submit a small test — first 20 WARCs of Jan 2022:
#   mkdir -p logs
#   qsub -v YEAR=2022,MONTH=01 -t 1-20 run_ccnews.sh
#
# NOTE: pass YEAR/MONTH with `-v` (SGE does NOT inherit your shell vars).

#$ -N ccnews
#$ -cwd                       # run in the submit directory (relative paths work)
#$ -j y                       # merge stdout/stderr
#$ -o logs/                   # this dir MUST exist at submit time: mkdir -p logs
#$ -l h_rt=02:00:00
#$ -l mem_per_core=4G

set -uo pipefail
# Loud failure: print the line number and command if anything errors.
trap 'echo "ERROR: task ${SGE_TASK_ID:-?} failed at line $LINENO: $BASH_COMMAND" >&2' ERR
set -e

# ---- preflight: show exactly what this task sees ----
echo "=== ccnews task ${SGE_TASK_ID:-?} on $(hostname) at $(date) ==="
echo "pwd=$(pwd)  YEAR=${YEAR:-UNSET}  MONTH=${MONTH:-UNSET}  TMPDIR=${TMPDIR:-UNSET}"

YEAR="${YEAR:?set YEAR via: qsub -v YEAR=2022,MONTH=01 ...}"
MONTH="${MONTH:?set MONTH via: qsub -v YEAR=2022,MONTH=01 ...}"

# ---- environment: adjust to `module avail python3` on your cluster ----
module load python3/3.10.12
python --version
python -c "import warcio, trafilatura; print('imports OK: trafilatura', trafilatura.__version__)"

BASE="https://data.commoncrawl.org"
PROJ_OUT="ccnews_out/${YEAR}-${MONTH}"
mkdir -p "$PROJ_OUT"

# Cache the month's WARC listing once.
PATHS="$PROJ_OUT/warc.paths"
if [[ ! -s "$PATHS" ]]; then
    echo "downloading warc.paths for ${YEAR}/${MONTH}"
    curl -sfL "$BASE/crawl-data/CC-NEWS/${YEAR}/${MONTH}/warc.paths.gz" | gunzip > "$PATHS"
fi
echo "month has $(wc -l < "$PATHS") WARC files"

WARC_PATH="$(sed -n "${SGE_TASK_ID}p" "$PATHS")"
if [[ -z "$WARC_PATH" ]]; then
    echo "no WARC at line $SGE_TASK_ID — nothing to do for this task"
    exit 0
fi

LOCAL="${TMPDIR:-/tmp}/$(basename "$WARC_PATH")"
OUT="$PROJ_OUT/articles.$(printf '%05d' "$SGE_TASK_ID").jsonl"

echo "task $SGE_TASK_ID -> $WARC_PATH"
curl -sfL "$BASE/$WARC_PATH" -o "$LOCAL"
echo "downloaded $(du -h "$LOCAL" | cut -f1) -> extracting"

# Add --domains domains.txt here once you want to filter to specific outlets.
python extract_articles.py "$LOCAL" "$OUT"

rm -f "$LOCAL"
echo "task $SGE_TASK_ID DONE -> $(wc -l < "$OUT") articles in $OUT"
