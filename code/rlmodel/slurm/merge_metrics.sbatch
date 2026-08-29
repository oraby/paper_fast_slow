#!/usr/bin/env bash
# Merge one metrics work dir's shards into the pickle model_analysis.ipynb
# reads. Submitted by launch_metrics.py with a dependency on the evaluation
# arrays, so the notebook-ready file appears without a manual step; also fine to
# run by hand (or as `python code/rlmodel/metrics_runner.py --mode merge`,
# which needs no Slurm at all).
#
# RL_WORK_DIR (required) and RL_CACHE_DIR (optional) come in via --export.
# Extra args are forwarded to the runner (--allow-missing, --dry-run, ...).
#
#SBATCH --job-name=rl_metrics_merge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=code/rlmodel/slurm/logs/rl_metrics_merge_%j.out

set -eo pipefail

CONDA_ENV="py314"
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)   CONDA_ENV="$2"; shift 2 ;;
        --conda-env=*) CONDA_ENV="${1#*=}"; shift ;;
        *)             args+=("$1"); shift ;;
    esac
done

cd "${RL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# See activate_conda.sh: `conda` itself is not callable in a batch job.
source code/rlmodel/slurm/activate_conda.sh

if [[ -z "${RL_WORK_DIR}" ]]; then
    echo "RL_WORK_DIR is not set; pass it via sbatch --export." >&2
    exit 1
fi

cache_args=()
if [[ -n "${RL_CACHE_DIR}" ]]; then
    cache_args=(--cache-dir "${RL_CACHE_DIR}")
fi

echo "host=$(hostname) cwd=$(pwd) work_dir=${RL_WORK_DIR}"

python code/rlmodel/metrics_runner.py \
    --mode merge \
    --work-dir "${RL_WORK_DIR}" \
    "${cache_args[@]}" \
    "${args[@]}"
