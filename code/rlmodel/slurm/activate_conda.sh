#!/usr/bin/env bash
# Activate a conda env inside a Slurm job. Sourced, not executed.
#
#   CONDA_ENV=py314
#   source code/rlmodel/slurm/activate_conda.sh
#
# Why this is not just `source "$(conda info --base)/etc/profile.d/conda.sh"`:
# on a compute node that fails with "conda: command not found", because
#
#   - `conda` is a shell FUNCTION installed by `conda init` into an interactive
#     rc file, and shell functions are not inherited by a batch job; and
#   - `condabin/` is frequently absent from the PATH the job inherits.
#
# So the base is found without ever calling `conda`, in descending order of
# trust:
#   1. RL_CONDA_BASE          -- explicit override (also what launch_metrics.py
#                                exports, derived from the submitting shell)
#   2. `conda` on PATH        -- the easy case, when it really is there
#   3. CONDA_EXE              -- a plain env var, so it DOES survive into the
#                                job whenever the submitter had conda active
#   4. the usual install dirs
_rl_conda_base() {
    if [[ -n "${RL_CONDA_BASE}" && -f "${RL_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "${RL_CONDA_BASE}"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null)"
        if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
            echo "${base}"
            return 0
        fi
    fi
    if [[ -n "${CONDA_EXE}" && -x "${CONDA_EXE}" ]]; then
        local base
        base="$(dirname "$(dirname "${CONDA_EXE}")")"
        if [[ -f "${base}/etc/profile.d/conda.sh" ]]; then
            echo "${base}"
            return 0
        fi
    fi
    local candidate
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
                     "$HOME/mambaforge" /opt/conda /usr/local/conda; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

_RL_CONDA_BASE="$(_rl_conda_base)" || {
    echo "Could not find a conda installation on $(hostname)." >&2
    echo "  RL_CONDA_BASE='${RL_CONDA_BASE}'  CONDA_EXE='${CONDA_EXE}'" >&2
    echo "  PATH=${PATH}" >&2
    echo "Set RL_CONDA_BASE to the directory holding etc/profile.d/conda.sh" >&2
    echo "(e.g. export RL_CONDA_BASE=\$HOME/miniconda3 before submitting)." >&2
    exit 1
}

# shellcheck source=/dev/null
source "${_RL_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}" || {
    echo "conda activate '${CONDA_ENV}' failed (base ${_RL_CONDA_BASE})." >&2
    echo "Available envs:" >&2
    conda env list >&2
    echo "Pass a different one with: --conda-env NAME" >&2
    exit 1
}
echo "conda: base=${_RL_CONDA_BASE} env=${CONDA_ENV} python=$(command -v python)"
