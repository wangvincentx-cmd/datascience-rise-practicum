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
# and lxml (tdm_parse).
#
# Honours an exported $PY, so `python vm_doctor.py` -> `export PY=...` just works
# and nothing here has to be edited. The fallback is the binary that is actually
# present in the current image: note it is python3.12, NOT python -- hardcoding
# `bin/python` is what produced a run of "No module named openai" against an
# interpreter that existed but was the wrong one. If this fallback ever goes
# stale again, run vm_doctor.py rather than guessing.
PY="${PY:-/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python3.12}"
GPT_PY="$PY"
if [ ! -x "$PY" ]; then
    echo "Interpreter not found: $PY"
    echo "Run:  python vm_doctor.py   and export the PY= line it prints."
    exit 1
fi
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
        echo "  DAILY RATE LIMIT hit on $w during EXTRACT -- stopping the batch."
        echo "  Re-run this same command after the quota resets to resume here."
        break
    elif [ $rc -ne 0 ]; then
        echo "  extract failed on $w (rc=$rc); skipping to next window"
        continue
    fi

    # Second pass. The gold eval put raw gpt-4o-mini at precision 0.409; the
    # verifier lifts it to 0.700 (F1 0.458 -> 0.512) by dropping candidates that
    # are not forecasts. It is cheap -- a quote plus ~400 chars of context, one
    # call per page, against extraction's whole-document calls.
    #
    # Output goes to data/verified/, NEVER data/predictions/: analyze_economy.py
    # globs pred_*_economy_*.jsonl there and would otherwise load the verified
    # copy AND its .dropped.jsonl alongside the original, double-counting every
    # claim and re-admitting the rejects. verify_gpt.py refuses that path.
    if [ "${SKIP_VERIFY:-0}" = "1" ]; then
        echo "  SKIP_VERIFY=1 -- leaving $w unverified"
    else
        $GPT_PY verify_gpt.py \
            --claims "data/predictions/pred_proquest_economy_${w}.jsonl" \
            --pages  "data/raw/proquest_economy_${w}.jsonl" \
            --out    "data/verified/pred_proquest_economy_${w}.jsonl"
        rc=$?
        if [ $rc -eq 2 ]; then
            echo "  DAILY RATE LIMIT hit on $w during VERIFY -- stopping the batch."
            echo "  Extraction for $w is DONE and saved; only verification resumes."
            break
        elif [ $rc -ne 0 ]; then
            echo "  verify failed on $w (rc=$rc); the unverified claims are still"
            echo "  on disk, so re-running just redoes verification"
            continue
        fi
        $PY strip_for_export.py "data/verified/pred_proquest_economy_${w}.jsonl"
    fi

    # Both sets are exported: the verified one is the analysis set, and the
    # unverified one is what makes the filter's effect measurable on the Mac
    # without spending quota again. Label-only JSONL compresses ~28x, so both
    # together are well under the 15 MB export cap.
    $PY strip_for_export.py "data/predictions/pred_proquest_economy_${w}.jsonl"
done

echo
echo "=== bundling stripped exports into one file to Export ==="
# Bundled from the repo root with paths intact, so the tarball unpacks on the
# Mac as verified/ and unverified/ and the two cannot be confused. Only
# *.export.jsonl is matched -- the un-stripped originals hold article text and
# must never enter this archive.
if ls data/verified/*.export.jsonl data/predictions/*.export.jsonl >/dev/null 2>&1; then
    # Staged into a directory rather than using tar --transform, which is GNU-only
    # and cannot be tested on a Mac before shipping. Copying is portable and the
    # layout is visible on disk before it is archived.
    rm -rf export_staging proquest_exports.tar.gz
    mkdir -p export_staging/verified export_staging/unverified
    cp data/verified/*.export.jsonl      export_staging/verified/    2>/dev/null
    cp data/predictions/*.export.jsonl   export_staging/unverified/  2>/dev/null

    # The check that matters, BEFORE anything is archived: article text may not
    # leave the VM. Written so that a failure to read the files is itself a
    # failure -- an earlier version printed "OK" when the archive did not exist,
    # which is exactly the wrong way for a licence check to fail.
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
    echo "  contents:"
    tar -tzf proquest_exports.tar.gz | grep -v '/$' | sed 's/^/    /'
else
    echo "  no stripped export files yet"
fi
