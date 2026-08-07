"""Submit a per-subject Slurm array for one RL-model fit.

Thin wrapper around ``fit_chisq.sbatch`` / ``fit_mle.sbatch`` that fixes the two
things a static ``.sbatch`` file can't do:

1. **Dynamic array size.** ``#SBATCH`` directives are read by Slurm *before* any
   shell expansion, so the number of subjects can't be computed inside the file.
   This launcher reads ``slurm/subjects.txt``, works out the array range (or the
   exact index list for a restricted subset), and passes it as
   ``sbatch --array=…`` on the command line, which overrides the file directive.

2. **Per-model log directory.** Logs go to ``slurm/logs/<save-name>/`` where
   ``<save-name>`` is the model's on-disk pickle filename (from
   ``fit.evolveFP``), so each model+fit-mode gets its own log folder instead of
   everything landing in one flat dir. Passed as ``sbatch --output=…`` (also
   overrides the file directive).

Usage (run from anywhere; the launcher submits with cwd = project root):

    python -m code.rlmodel.slurm.launch mle \
        --drift "RewardRate" --bias "Q-Val (Offset)" --noise "Normal(0, 1)" --asym

    # restrict the fan-out (e.g. quick test on one/few subjects):
    python -m code.rlmodel.slurm.launch chisq \
        --drift Classic --bias None_ --only-subject Avgat1 --only-subject WF10

    # see the sbatch command without submitting:
    python -m code.rlmodel.slurm.launch mle --drift ... --bias ... --dry-run

The first positional (``chisq`` | ``mle``) selects the fit mode and hence the
script; every other flag is forwarded VERBATIM to the runner (``--noise``,
``--asym``, ``--scale-bound``, ``--mle-gpu-memory-gb``, ``--mle-mle-weight``,
``--conda-env``, …). The launcher only interprets ``--only-subject`` (to size /
select the array) and ``--dry-run``; ``--only-subject`` is NOT forwarded (the
.sbatch script injects one per array task).
"""
import argparse
import pathlib
import shlex
import subprocess
import sys

from ..model import fit
from ..model.bias import BIAS_FN_DICT
from ..model.drift import user_facing_drift_keys
from ..model.state_updates import DEFAULT_RR_DRIFT_MAP, RR_DRIFT_MAPS
from ..model.initvals import DT, T_dur
from ..model.noise import NOISE_FN_DICT
from ..model_runner import _expand_asym_shorthand, _resolve_drift_alias_args

_SLURM_DIR = pathlib.Path(__file__).resolve().parent
# code/rlmodel/slurm -> parents: [0]=rlmodel, [1]=code, [2]=paper_fast_slow.
_PROJECT_ROOT = _SLURM_DIR.parents[2]
_SUBJECTS_FP = _SLURM_DIR / "subjects.txt"


def _read_subjects():
    if not _SUBJECTS_FP.exists():
        sys.exit(f"subjects.txt not found at {_SUBJECTS_FP}. Generate it first: "
                 f"python -m code.rlmodel.slurm.gen_subjects")
    subjects = [ln.strip() for ln in
                _SUBJECTS_FP.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    if not subjects:
        sys.exit(f"{_SUBJECTS_FP} is empty; re-run gen_subjects.")
    return subjects


def _array_spec(subjects, only_subject):
    """Return the ``sbatch --array`` spec indexing into subjects.txt.

    Full set -> ``0-(N-1)``. A restricted subset -> a comma list of the selected
    subjects' 0-based line indices, so the .sbatch script still reads the
    canonical subjects.txt by SLURM_ARRAY_TASK_ID (no per-run subject file).
    """
    if not only_subject:
        return f"0-{len(subjects) - 1}"
    missing = [s for s in only_subject if s not in subjects]
    if missing:
        sys.exit(f"--only-subject not in subjects.txt: {missing}\n"
                 f"available: {subjects}")
    idxs = sorted({subjects.index(s) for s in only_subject})
    return ",".join(str(i) for i in idxs)


def _save_name(passthrough, fit_mode):
    """Reproduce the runner's on-disk pickle filename for these flags.

    Reuses the SAME resolution the runner applies (drift-alias then asym
    shorthand) and ``fit.evolveFP`` so the log dir tracks the fit's save path
    exactly. Parses only the filename-affecting flags out of ``passthrough``;
    unknown flags are ignored (they don't change the name).
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--drift", required=True, choices=user_facing_drift_keys())
    p.add_argument("--bias", required=True, choices=list(BIAS_FN_DICT.keys()))
    p.add_argument("--noise", default="Normal(0, 1)",
                   choices=list(NOISE_FN_DICT.keys()))
    p.add_argument("--loss-no-dir", action="store_true", default=False)
    p.add_argument("--asym", action="store_true", default=False)
    p.add_argument("--asym-q", action="store_true", default=False)
    p.add_argument("--asym-rr", action="store_true", default=False)
    p.add_argument("--scale-bound", action="store_true", default=False)
    # --use-drift-rr / --drift-rr-map don't add a filename SUFFIX, but they do
    # change which DRIFT_FN_DICT key --drift resolves to (DriftGain-*), and the
    # drift name IS part of the pickle name — so they must be parsed here or
    # the logs land in the wrong directory.
    p.add_argument("--use-drift-rr", action="store_true", default=False)
    p.add_argument("--drift-rr-map", type=str, default=DEFAULT_RR_DRIFT_MAP,
                   choices=list(RR_DRIFT_MAPS))
    p.add_argument("--mle-mle-weight", type=float, default=1.0)
    p.add_argument("--mle-chi2-weight", type=float, default=0.0)
    ns, _ = p.parse_known_args(passthrough)
    _resolve_drift_alias_args(ns)          # RewardRate alias -> canonical key
    if ns.asym or ns.asym_q or ns.asym_rr:
        _expand_asym_shorthand(ns)         # --asym -> --asym-q / --asym-rr
    return fit.evolveFP(
        ns.drift, ns.bias, ns.noise, t_dur=T_dur, dt=DT,
        is_loss_no_dir=ns.loss_no_dir, fit_mode=fit_mode,
        uses_asym_q=ns.asym_q, uses_asym_rr=ns.asym_rr,
        uses_scaled_bound=ns.scale_bound,
        mle_mle_weight=ns.mle_mle_weight,
        mle_chi2_weight=ns.mle_chi2_weight).name


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="launch", allow_abbrev=False,
        description="Submit a per-subject Slurm array for one RL-model fit.")
    parser.add_argument("fit_mode", choices=["chisq", "mle"],
                        help="Fit mode; selects fit_<mode>.sbatch and the "
                             "--fit-mode passed to the runner.")
    parser.add_argument("--only-subject", action="append", default=None,
                        metavar="NAME",
                        help="Restrict the array to these subjects (repeatable; "
                             "e.g. for a quick test). Default: every subject in "
                             "subjects.txt. NOT forwarded to the runner.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch command and exit without "
                             "submitting.")
    args, passthrough = parser.parse_known_args(argv)

    subjects = _read_subjects()
    array_spec = _array_spec(subjects, args.only_subject)
    save_name = _save_name(passthrough, args.fit_mode)

    log_dir = _SLURM_DIR / "logs" / save_name
    output_pattern = str(log_dir / "%A_%a.out")

    script = _SLURM_DIR / f"fit_{args.fit_mode}.sbatch"
    cmd = ["sbatch",
           f"--array={array_spec}",
           f"--output={output_pattern}",
           str(script),
           *passthrough]

    n = len(subjects) if not args.only_subject else len(args.only_subject)
    print(f"fit_mode   : {args.fit_mode}")
    print(f"save name  : {save_name}")
    print(f"log dir    : {log_dir}")
    print(f"subjects   : {n} ({'all' if not args.only_subject else array_spec})")
    print(f"array spec : {array_spec}")
    print(f"submit cwd : {_PROJECT_ROOT}")
    print(f"sbatch cmd : {shlex.join(cmd)}")

    if args.dry_run:
        print("--dry-run: not submitting.")
        return
    # Slurm resolves --output before the job body runs and won't create the
    # directory, so make it exist now (right before submitting).
    log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
