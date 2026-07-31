#!/usr/bin/env bash
# Run any pipeline script under the right interpreter, with no setup.
#
#   bash pyrun.sh corpus_progress.py
#   bash pyrun.sh corpus_progress.py --by-year --rate 1700
#   bash pyrun.sh extract_gpt.py --source proquest --window 1930 --limit 5
#
# WHY THIS EXISTS. `export PY=...` dies with the shell, and on SageMaker
# everything outside /home/ec2-user/SageMaker is wiped when the instance
# restarts, so neither the export nor a ~/.bashrc edit reliably survives a new
# terminal. Every fresh terminal then hits `$PY foo.py` -> "command not found",
# which reads like a broken pipeline and is really just an unset variable.
# run_corpus_economy.sh already resolves the interpreter on its own; this gives
# the ad-hoc commands the same immunity.
#
# Resolution order, most specific first:
#   1. $PY, if it is exported and executable (so an override still wins)
#   2. the known path in the current image
#   3. any sample-* conda env holding a python3.12 -- covers the version suffix
#      changing between workbenches, which is what stales the hardcoded path
#
# Note python3.12, NOT python: `bin/python` exists in some envs but is the wrong
# interpreter, and its failure mode is a misleading "No module named openai".

set -u

KNOWN=/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python3.12

pick_interpreter() {
    if [ -n "${PY:-}" ] && [ -x "${PY:-}" ]; then
        echo "$PY"; return
    fi
    if [ -x "$KNOWN" ]; then
        echo "$KNOWN"; return
    fi
    # Newest first, and skip the `-r` envs: those are R, not Python.
    for candidate in $(ls -t /home/ec2-user/SageMaker/.conda/envs/*/bin/python3.12 \
                       2>/dev/null | grep -v -- '-r/bin/'); do
        if [ -x "$candidate" ]; then
            echo "$candidate"; return
        fi
    done
}

if [ $# -eq 0 ]; then
    echo "usage: bash pyrun.sh <script.py> [args...]"
    echo "   or: bash pyrun.sh --which     # just print the interpreter"
    exit 1
fi

INTERP=$(pick_interpreter)
if [ -z "${INTERP:-}" ]; then
    echo "No usable interpreter found."
    echo "Looked for \$PY, then $KNOWN, then any"
    echo "/home/ec2-user/SageMaker/.conda/envs/*/bin/python3.12."
    echo "Run:  python vm_doctor.py   to find the right one."
    exit 1
fi

if [ "$1" = "--which" ]; then
    echo "$INTERP"
    exit 0
fi

# exec so the script's exit code reaches the caller unchanged -- extract_gpt.py's
# exit 2 (daily cap) has to stay distinguishable from a real failure.
exec "$INTERP" "$@"
