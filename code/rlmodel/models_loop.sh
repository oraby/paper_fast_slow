#!/usr/bin/env bash
# Loop over a pre-defined set of (drift, bias, noise) tuples and run
# rlmodel.model_runner once per tuple. Every extra CLI argument passed to
# this script is forwarded verbatim to the runner.
#
# Usage (run from the project ROOT — one level above ``code/``):
#
#   ./code/rlmodel/models_loop --fit-mode mle --mle-backend GPU \
#                               --mle-gpu-memory-gb 16 --mle-device-id 2
#
# The runner is invoked as ``python -m code.rlmodel.model_runner``, so
# Python resolves the package from the current working directory. Do NOT
# ``cd`` into ``code/rlmodel/`` before running — the import path would
# break.
#
# To pick which combinations to run, edit the ``combos`` array below. Each
# entry uses ``|`` as the field separator so values can contain spaces,
# parentheses, and dashes (e.g. ``NoiseGain-RewardRate Decay Q (Offset)``).

set -uo pipefail

combos=(
    "Classic|None_|Normal(0, 1)"
    "Classic|Q-Val (Offset)|Normal(0, 1)"
    # "Decay Q (Offset)|None_|Normal(0, 1)"
    "NoiseGain-RewardRate|None_|Normal(0, 1)"
    "NoiseGain-RewardRate|Q-Val (Offset)|Normal(0, 1)"
    # "NoiseGain-RewardRate Decay Q (Offset)|None_|Normal(0, 1)"
)

PY=${PYTHON:-python}

failed=()
total=${#combos[@]}
i=0
for combo in "${combos[@]}"; do
    i=$((i + 1))
    IFS='|' read -r drift bias noise <<< "$combo"
    echo
    echo "=================================================================="
    echo "[$i/$total] drift='$drift'  bias='$bias'  noise='$noise'"
    echo "=================================================================="
    if "$PY" -m code.rlmodel.model_runner \
        --drift "$drift" \
        --bias "$bias" \
        --noise "$noise" \
        "$@"; then
        echo "[$i/$total] OK"
    else
        rc=$?
        echo "[$i/$total] FAILED (exit $rc)"
        failed+=("$combo")
    fi
done

echo
if (( ${#failed[@]} == 0 )); then
    echo "All $total combinations completed successfully."
else
    echo "${#failed[@]}/${total} combinations failed:"
    for c in "${failed[@]}"; do
        echo "  - $c"
    done
    exit 1
fi
