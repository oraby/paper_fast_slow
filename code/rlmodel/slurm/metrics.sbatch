#!/usr/bin/env bash
# One metrics evaluation as ONE Slurm array task: a single (model, subject,
# iteration) combination from a work dir built by `metrics_runner --mode
# prepare`.
#
# Deliberately NOT `--exclusive` (unlike fit_chisq.sbatch): one evaluation is
# ~2.5s of single-threaded work, so thousands of 1-cpu tasks backfill onto
# shared nodes far faster than whole-node reservations schedule. Each task
# writes its own `shards/<index>.pkl`, so nothing is shared and nothing can
# race.
#
# PREFERRED entry point is the Python launcher, which builds the work dir,
# computes the array size, splits it around the cluster's MaxArraySize and
# chains the merge job:
#
#   python code/rlmodel/slurm/launch_metrics.py --figure fig1l --num-evaluations 100
#
# Direct submit also works (from the project ROOT — one level above `code/`);
# RL_WORK_DIR is required, RL_TASK_OFFSET defaults to 0:
#
#   sbatch --array=0-999 --export=ALL,RL_WORK_DIR=data/RLModel/metrics/work/fig1l \
#          code/rlmodel/slurm/metrics.sbatch
#
# All args after the script name are forwarded VERBATIM to the runner
# (--min-num-trials, --overwrite, --no-save-sim, ...). The script only injects
# --mode run, --work-dir and --task-id. An optional
#   --conda-env NAME   (default: py314)
# selects the conda env to activate and is consumed here (not forwarded).
#
#SBATCH --job-name=rl_metrics
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=code/rlmodel/slurm/logs/rl_metrics_%A_%a.out

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
cd "${RL_PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"

# --- activate conda ---------------------------------------------------------
# NOT `source "$(conda info --base)/..."`: `conda` is a shell function from an
# interactive rc file and does not exist in a batch job. See the helper.
source code/rlmodel/slurm/activate_conda.sh

if [[ -z "${RL_WORK_DIR}" ]]; then
    echo "RL_WORK_DIR is not set; pass it via sbatch --export." >&2
    exit 1
fi

# Slurm caps array indices at MaxArraySize, so a big worklist is submitted as
# several arrays that each cover a window of it. The launcher exports the
# window's start. RL_ITEMS_PER_TASK (default 1, i.e. one combination per job)
# is raised when the cluster caps how many jobs may be QUEUED; then each task
# walks that many consecutive work items instead.
items=${RL_ITEMS_PER_TASK:-1}
first=$(( ${RL_TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID * items ))
last=$(( first + items - 1 ))

echo "host=$(hostname) cwd=$(pwd) conda_env=${CONDA_ENV}"
echo "work_dir=${RL_WORK_DIR} items=${first}-${last}"

# Script-injected flags come first; the forwarded "${args[@]}" comes last so a
# user-supplied flag overrides a script default. Invoked as a script, not with
# `-m`: see metrics_runner's module docstring (stdlib `code` shadowing).
# --task-range tolerates a tail that runs past the end of the worklist, which
# the last task of a run does whenever items doesn't divide the worklist.
python code/rlmodel/metrics_runner.py \
    --mode run \
    --work-dir "${RL_WORK_DIR}" \
    --task-range "${first}-${last}" \
    "${args[@]}"
