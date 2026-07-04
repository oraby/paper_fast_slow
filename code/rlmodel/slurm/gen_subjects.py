"""Generate the static per-subject list used by the Slurm array jobs.

Writes ``slurm/subjects.txt`` (one subject name per line) matching EXACTLY the
subject set ``model_runner`` itself would fit: it reuses
``model_runner.loadDF(min_valid_trials=0)`` and lists ``sorted(Name.unique())``
so a job-array index maps 1:1 to a subject the runner accepts via
``--only-subject``.

Run ONCE from the project root (one level above ``code/``), e.g.::

    conda run -n py314 python -m code.rlmodel.slurm.gen_subjects

Commit the resulting ``subjects.txt`` and update the ``#SBATCH --array=0-<N-1>``
line in ``fit_chisq.sbatch`` / ``fit_mle.sbatch`` to ``N = len(subjects.txt)``.
Regenerate only if the dataset's subject set changes.
"""
import pathlib

from ..model_runner import loadDF

# Files live alongside this script (code/rlmodel/slurm/).
_HERE = pathlib.Path(__file__).resolve().parent
SUBJECTS_FP = _HERE / "subjects.txt"
LOGS_DIR = _HERE / "logs"


def main():
    # Same subject set the runner fits: no min-trial pruning here (the runner
    # calls loadDF(min_valid_trials=0)), and the fit order is Name.unique().
    df = loadDF(min_valid_trials=0)
    subjects = sorted(df["Name"].unique())

    # Force LF: subjects.txt is read by ``sed`` on the Linux cluster, so a
    # Windows-generated CRLF would leave a trailing "\r" in each subject name.
    SUBJECTS_FP.write_text(
        "\n".join(subjects) + "\n", encoding="utf-8", newline="\n")

    # The sbatch --output directory must exist before Slurm starts the job
    # (Slurm won't create it); ensure it does as part of setup.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Wrote {len(subjects)} subjects to {SUBJECTS_FP}")
    print("Set the sbatch --array to 0-{} (N-1).".format(len(subjects) - 1))
    for s in subjects:
        print(f"  {s}")


if __name__ == "__main__":
    main()
