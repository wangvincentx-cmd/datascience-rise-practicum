#!/bin/bash -l
# SGE array job: one task = one CC-News WARC file for a given month.
#
# Each task downloads its WARC to $TMPDIR (node-local scratch), extracts
# article text, appends to a per-task JSONL, then deletes the raw WARC.
# Merge the per-task files afterwards (see README).
#
# Submit a small test — first 20 WARCs of Jan 2022:
#   YEAR=2022 MONTH=01 qsub -t 1-20 run_ccnews.sh
#
# -t sets $SGE_TASK_ID; each task processes the SGE_TASK_ID-th path line.

#$ -N ccnews
#$ -j y                       # merge stdout/stderr
#$ -o logs/                   # make this dir first: mkdir -p logs
#$ -l h_rt=02:00:00           # walltime per task; bump if WARCs are slow
#$ -l mem_per_core=4G

set -euo pipefail

YEAR="${YEAR:?set YEAR, e.g. 2022}"
MONTH="${MONTH:?set MONTH, e.g. 01}"

# --- environment: adjust the module name to what `module avail python3` shows ---
module load python3/3.10.12
# One-time, before submitting:  pip install --user warcio trafilatura

BASE="https://data.commoncrawl.org"
PROJ_OUT="ccnews_out/${YEAR}-${MONTH}"          # lives in your project/scratch space
mkdir -p "$PROJ_OUT"

# Cache the month's file listing once (task 1 races are harmless — same content).
PATHS="$PROJ_OUT/warc.paths"
if [[ ! -s "$PATHS" ]]; then
    curl -sfL "$BASE/crawl-data/CC-NEWS/${YEAR}/${MONTH}/warc.paths.gz" \
        | gunzip > "$PATHS"
fi

# Pick this task's WARC path (1-indexed line).
WARC_PATH="$(sed -n "${SGE_TASK_ID}p" "$PATHS")"
if [[ -z "$WARC_PATH" ]]; then
    echo "no path at line $SGE_TASK_ID (month has $(wc -l < "$PATHS") files) — done"
    exit 0
fi

LOCAL="$TMPDIR/$(basename "$WARC_PATH")"
OUT="$PROJ_OUT/articles.$(printf '%05d' "$SGE_TASK_ID").jsonl"

echo "task $SGE_TASK_ID -> $WARC_PATH"
curl -sfL "$BASE/$WARC_PATH" -o "$LOCAL"

# Add --domains domains.txt here once you want to filter to specific outlets.
python extract_articles.py "$LOCAL" "$OUT"

rm -f "$LOCAL"      # keep node scratch clean
echo "task $SGE_TASK_ID done -> $OUT"
