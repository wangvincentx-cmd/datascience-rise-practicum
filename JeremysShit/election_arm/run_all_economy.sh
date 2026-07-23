#!/usr/bin/env bash
# Batch-run the ProQuest economy pipeline over every window whose dataset exists.
# Runs INSIDE the TDM Studio workbench, from the election_arm folder.
#
# For each window it does:  tdm_parse -> extract_gpt (in-VM GPT proxy) -> strip.
# Windows whose ProQuest dataset folder is not built yet are skipped, so you can
# run this repeatedly as datasets finish. Every stage is resume-safe, so if the
# run stops (e.g. ProQuest's daily LLM limit), just run it again to continue.
#
# Dataset naming convention: the ProQuest dataset for window W must be the folder
# named W with underscores removed, e.g.
#   gfc_2008  -> /home/ec2-user/SageMaker/data/gfc2008
#   calm_1965 -> /home/ec2-user/SageMaker/data/calm1965
#
# Usage (from /home/ec2-user/SageMaker/election_arm):
#   bash run_all_economy.sh                      # all 9 post-1963 economy windows
#   bash run_all_economy.sh gfc_2008 oil_1973    # just these

set -u

# One interpreter for the whole pipeline: the sample env has openai (extract_gpt)
# and lxml (tdm_parse). Verify once: $PY -c "import lxml, openai".
PY=/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python
GPT_PY="$PY"
DATA=/home/ec2-user/SageMaker/data

DEFAULT_WINDOWS="oil_1973 volcker_1980 crash_1987 gulf_1990 dotcom_2001 gfc_2008 calm_1965 calm_1995 calm_2005"
WINDOWS="${*:-$DEFAULT_WINDOWS}"

if [ ! -f tdm_parse.py ] || [ ! -f gpt_sample.txt ]; then
    echo "Run this from the election_arm folder (tdm_parse.py and gpt_sample.txt must be here)."
    exit 1
fi

for w in $WINDOWS; do
    folder="$DATA/${w//_/}"                       # strip underscores: gfc_2008 -> gfc2008
    echo
    echo "=============================================================="
    echo "  $w   (dataset: $folder)"
    echo "=============================================================="
    if [ ! -d "$folder" ]; then
        echo "  SKIP: no dataset folder yet at $folder"
        continue
    fi
    $PY tdm_parse.py --arm economy --window "$w" --dataset-dir "$folder" \
        || { echo "  parse failed, skipping $w"; continue; }
    $GPT_PY extract_gpt.py --source proquest --window "$w"
    rc=$?
    if [ $rc -eq 2 ]; then
        echo "  DAILY RATE LIMIT hit on $w -- stopping the batch."
        echo "  Re-run this same command after the quota resets to resume here."
        break
    elif [ $rc -ne 0 ]; then
        echo "  extract failed on $w (rc=$rc); skipping to next window"
        continue
    fi
    $PY strip_for_export.py "data/predictions/pred_proquest_economy_${w}.jsonl"
done

echo
echo "=== bundling stripped exports into one file to Export ==="
if cd data/predictions 2>/dev/null && ls pred_proquest_economy_*.export.jsonl >/dev/null 2>&1; then
    tar -czf proquest_exports.tar.gz pred_proquest_economy_*.export.jsonl
    echo "Export THIS one file (set data_to_export to it):"
    echo "  /home/ec2-user/SageMaker/election_arm/data/predictions/proquest_exports.tar.gz"
    ls -lh proquest_exports.tar.gz
else
    echo "  no stripped export files yet"
fi
