#!/usr/bin/env bash
# MLE RL-model fit as a Slurm job ARRAY: one job per subject.
#
# Each array task fits ONE subject (picked from slurm/subjects.txt by
# SLURM_ARRAY_TASK_ID) via `model_runner --fit-mode mle --only-subject`. MLE
# uses the CuPy GPU backend, so each task grabs one GPU on the `gpu` partition
# (>=16 GB is satisfied by the partition's GPUs). With --gres=gpu:1 Slurm sets
# CUDA_VISIBLE_DEVICES to the allocated GPU and CuPy indexes relative to that
# mask, so we do NOT pass --mle-device-id (CuPy uses the single visible GPU).
# --mle-no-progress silences the tqdm bar so it doesn't bloat the log file.
#
# PREFERRED entry point is the Python launcher, which computes the array size
# dynamically and routes logs into a per-model directory:
#
#   python -m code.rlmodel.slurm.launch mle \
#       --drift "RewardRate" --bias "Q-Val (Offset)" --mle-gpu-memory-gb 16
#
# Direct submit also works (submit from the project ROOT — one level above
# `code/`); the `#SBATCH --array` / `--output` defaults below apply unless the
# launcher overrides them on the command line:
#
#   mkdir -p code/rlmodel/slurm/logs   # once, if the dir is missing
#   sbatch code/rlmodel/slurm/fit_mle.sbatch \
#       --drift "RewardRate" --bias "Q-Val (Offset)" --mle-gpu-memory-gb 16
#
# All args after the script name are forwarded VERBATIM to the runner (drift/
# bias/noise plus any extras like --asym, --mle-gpu-memory-gb, --mle-conditions,
# --mle-mle-weight, ...). The script only injects --fit-mode, --mle-backend GPU,
# --mle-no-progress, and --only-subject. An optional
#   --conda-env NAME   (default: py314)
# selects the conda env to activate and is consumed here (not forwarded).
#
# The `--array` range below is a fallback for direct sbatch use; it must equal
# the number of subjects in subjects.txt minus one (regenerate with
# `python -m code.rlmodel.slurm.gen_subjects`). The launcher overrides it.
#
#SBATCH --job-name=rl_mle
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-19
#SBATCH --output=code/rlmodel/slurm/logs/rl_mle_%A_%a.out

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

echo "host=$(hostname) cwd=$(pwd) conda_env=${CONDA_ENV} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Fitting (mle/GPU) subject: ${subject}"

# Script-injected flags come first; the forwarded "${args[@]}" comes last so a
# user-supplied flag overrides a script default.
python -m code.rlmodel.model_runner \
    --fit-mode mle \
    --mle-backend GPU \
    --mle-no-progress \
    --only-subject "$subject" \
    "${args[@]}"
