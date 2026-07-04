#!/usr/bin/env bash
# Chi-square RL-model fit as a Slurm job ARRAY: one job per subject.
#
# Each array task fits ONE subject (picked from slurm/subjects.txt by
# SLURM_ARRAY_TASK_ID) via `model_runner --fit-mode chisq --only-subject`.
# Chi2 fits are CPU-bound, so each task takes a whole node exclusively on the
# `normal` partition. The runner's per-subject reload-merge save lets all tasks
# write into the same on-disk pickle without clobbering each other.
#
# Usage (submit from the project ROOT — one level above `code/`):
#
#   mkdir -p code/rlmodel/slurm/logs   # once, if the dir is missing
#   sbatch code/rlmodel/slurm/fit_chisq.sbatch \
#       --drift "RewardRate" --bias "Q-Val (Offset)" --noise "Normal(0, 1)" --asym
#
# All args after the script name are forwarded VERBATIM to the runner (drift/
# bias/noise plus any extras like --asym, --scale-bound, ...). The script only
# injects --fit-mode, --num-cpus, and --only-subject. An optional
#   --conda-env NAME   (default: py314)
# selects the conda env to activate and is consumed here (not forwarded).
#
# The `--array` range below must equal the number of subjects in subjects.txt
# minus one (regenerate with `python -m code.rlmodel.slurm.gen_subjects`).
#
#SBATCH --job-name=rl_chisq
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --array=0-19
#SBATCH --output=code/rlmodel/slurm/logs/rl_chisq_%A_%a.out

set -eo pipefail

# --- parse the optional --conda-env out of the forwarded args ---------------
CONDA_ENV="py314"
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)   CONDA_ENV="$2"; shift 2 ;;
        --conda-env=*) CONDA_ENV="${1#*=}"; shift ;;
        *)             args+=("$1"); shift ;;
    esac
done

# --- move to the project root (contains code/ and data/) --------------------
# Submit from the project root, or export RL_PROJECT_ROOT to point at it.
cd "${RL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"

# --- activate conda ---------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- pick this task's subject ----------------------------------------------
SUBJECTS_FP="code/rlmodel/slurm/subjects.txt"
subject=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FP")
if [[ -z "$subject" ]]; then
    echo "No subject at array index ${SLURM_ARRAY_TASK_ID} in ${SUBJECTS_FP}; nothing to do."
    exit 0
fi

echo "host=$(hostname) cwd=$(pwd) conda_env=${CONDA_ENV} cpus=${SLURM_CPUS_ON_NODE}"
echo "Fitting (chisq) subject: ${subject}"

# Script-injected flags come first; the forwarded "${args[@]}" comes last so a
# user-supplied flag overrides a script default.
python -m code.rlmodel.model_runner \
    --fit-mode chisq \
    --num-cpus "$SLURM_CPUS_ON_NODE" \
    --only-subject "$subject" \
    "${args[@]}"
