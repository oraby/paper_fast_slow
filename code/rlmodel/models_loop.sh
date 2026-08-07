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
# To pick which combinations to run, edit the ``combos`` array below.
# Each entry uses ``|`` as the field separator so values can contain
# spaces, parentheses, and dashes (e.g. ``RewardRate Decay Q (Offset)``).
# An OPTIONAL fourth ``|``-separated field carries extra CLI args to append
# for that combo (e.g. ``--asym --scale-bound``) — this lets a single canonical
# model name pair with multiple asym opt-ins side-by-side.

set -uo pipefail

combos=(
    # --- symmetric baselines ---
    #"Classic|None_|Normal(0, 1)|"
    #"Classic|Q-Val (Offset)|Normal(0, 1)|"
    #"RewardRate|None_|Normal(0, 1)|"
    "RewardRate|Q-Val (Offset)|Normal(0, 1)|"
    # --- Decay-Q ---
    #"Decay Q (Offset)|None_|Normal(0, 1)|"
    #"RewardRate Decay Q (Offset)|None_|Normal(0, 1)|"
    # --- reward rate on the DRIFT (--use-drift-rr) ---
    # Overrides the noise / threshold channel: mu *= g(r_t), sigma and bound
    # flat. Resolves to the DriftGain-* drift, so these land in their own
    # pickles. --scale-bound is orthogonal here (it only picks the fitted
    # scale axis), hence the fixed-noise / fixed-threshold pairs below.
    #"RewardRate|None_|Normal(0, 1)|--use-drift-rr"
    #"RewardRate|Q-Val (Offset)|Normal(0, 1)|--use-drift-rr"
    #"RewardRate|None_|Normal(0, 1)|--use-drift-rr --scale-bound"
    #"RewardRate|Q-Val (Offset)|Normal(0, 1)|--use-drift-rr --scale-bound"
    # Flipped polarity (high reward rate => faster, like the other channels).
    # These two back the RR (Drift 1+r) bars of model_analysis' reward-rate
    # channel figure (aggregate.DRIFT_RR_SPECS).
    #"RewardRate|None_|Normal(0, 1)|--use-drift-rr --drift-rr-map 1+r"
    #"RewardRate|Q-Val (Offset)|Normal(0, 1)|--use-drift-rr --drift-rr-map 1+r"
)

PY=${PYTHON:-python}

failed=()
total=${#combos[@]}
i=0
for combo in "${combos[@]}"; do
    i=$((i + 1))
    IFS='|' read -r drift bias noise extra <<< "$combo"
    extra=${extra:-}
    echo
    echo "=================================================================="
    echo "[$i/$total] drift='$drift'  bias='$bias'  noise='$noise'  extra='$extra'"
    echo "=================================================================="
    # Word-split ``extra`` so flags like "--asym-q --asym-rr" become
    # two separate argv entries to argparse.
    if "$PY" -m code.rlmodel.model_runner \
        --drift "$drift" \
        --bias "$bias" \
        --noise "$noise" \
        $extra \
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
