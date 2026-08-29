"""Build the metrics work dirs and submit them as per-combination Slurm arrays.

The counterpart of ``launch.py`` for the *evaluation* side: where that fans a
model fit out over subjects, this fans ``model_analysis.ipynb``'s
``NUM_EVALUATIONS`` collection out over every (model × subject × iteration)
combination — one single-CPU, non-exclusive array task each, one output file
each (see ``model/metrics_shards.py``).

It exists because three things cannot live in a static ``.sbatch``:

1. **Dynamic array size.** ``#SBATCH`` directives are read before any shell
   expansion, so the number of combinations can't be computed in the file. This
   reads the work dir's ``worklist.tsv`` and passes ``sbatch --array=…``.
2. **MaxArraySize.** Clusters cap array indices (commonly 1001), and a
   four-figure run at N=100 is ~17k combinations. The worklist is therefore
   submitted as several arrays, each covering a window of it and carrying its
   window start in ``RL_TASK_OFFSET``.
3. **The dependent merge.** The array ids aren't known until submission, so the
   ``--dependency=afterany:…`` merge job has to be built here.

Usage (run from anywhere; jobs are submitted with cwd = project root):

    # everything: prepare + submit + chained merge
    python code/rlmodel/slurm/launch_metrics.py --num-evaluations 100

    # one figure, capped at 200 simultaneous tasks
    python code/rlmodel/slurm/launch_metrics.py --figure fig1l \
        --num-evaluations 100 --max-concurrent 200

    # see what would be submitted
    python code/rlmodel/slurm/launch_metrics.py --figure fig1l --dry-run

    # resubmit only the combinations a previous run left missing (the array
    # spec is printed by the merge step)
    python code/rlmodel/slurm/launch_metrics.py --figure fig1l \
        --skip-prepare --array 17,220-233

Anything after ``--`` is forwarded verbatim to ``metrics_runner --mode run``
(e.g. ``-- --overwrite --conda-env py312``).

A plain script rather than ``python -m code.rlmodel.slurm.launch_metrics`` (how
``launch.py`` is run) for the reason spelled out in ``metrics_runner``: ``-m``
from the project root lets the local ``code/`` package shadow the stdlib
``code`` module, which this import chain needs.
"""
import argparse
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys

_SLURM_DIR = pathlib.Path(__file__).resolve().parent
# code/rlmodel/slurm -> parents: [0]=rlmodel, [1]=code, [2]=paper_fast_slow.
_PROJECT_ROOT = _SLURM_DIR.parents[2]
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from paper_fast_slow.code.rlmodel.metrics_runner import (   # noqa: E402
    DEFAULT_CACHE_DIR, DEFAULT_DF_FP, DEFAULT_RESULT_DIR, DEFAULT_WORK_ROOT,
    FIGURES, do_prepare)
from paper_fast_slow.code.rlmodel.model import metrics_shards    # noqa: E402

_RUN_SCRIPT = _SLURM_DIR / "metrics.sbatch"
_MERGE_SCRIPT = _SLURM_DIR / "merge_metrics.sbatch"

# Slurm's own default when `scontrol show config` can't be read (no Slurm on
# this host, e.g. a --dry-run from a laptop).
DEFAULT_MAX_ARRAY_SIZE = 1001


def max_array_size(override=None):
    """The cluster's ``MaxArraySize``, or a safe default."""
    if override:
        return override
    try:
        out = subprocess.run(["scontrol", "show", "config"], check=True,
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print(f"Could not read MaxArraySize from scontrol; assuming "
              f"{DEFAULT_MAX_ARRAY_SIZE}.")
        return DEFAULT_MAX_ARRAY_SIZE
    match = re.search(r"MaxArraySize\s*=\s*(\d+)", out)
    return int(match.group(1)) if match else DEFAULT_MAX_ARRAY_SIZE


def _check_exportable(**paths):
    """Reject paths Slurm's ``--export`` can't carry.

    ``--export`` is a comma-separated NAME=VALUE list, so a comma anywhere in a
    path silently truncates the variable and the job fails with a confusing
    "work dir not found" much later. Fail here instead, where the cause is
    obvious.
    """
    for name, value in paths.items():
        value = str(value)
        if "," in value or value != value.strip():
            sys.exit(f"{name}={value!r} cannot be passed through "
                     f"sbatch --export (commas / leading-trailing whitespace). "
                     f"Move it somewhere without them, or pass an explicit "
                     f"--work-root / --cache-dir.")


def _job_id(sbatch_stdout):
    """The job id out of sbatch's ``Submitted batch job 12345``."""
    match = re.search(r"(\d+)", sbatch_stdout or "")
    return match.group(1) if match else None


def conda_base():
    """The conda install root of the shell doing the submitting, or ``None``.

    Exported to the jobs as ``RL_CONDA_BASE`` because a compute node usually
    cannot find conda by itself: ``conda`` is a shell function from an
    interactive rc file, so it does not exist in a batch job (see
    ``activate_conda.sh``). The submitting shell always knows, so it tells.
    """
    exe = os.environ.get("CONDA_EXE")
    if exe and pathlib.Path(exe).exists():
        base = pathlib.Path(exe).resolve().parent.parent
        if (base / "etc" / "profile.d" / "conda.sh").exists():
            return base
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        # An activated env is <base>/envs/<name>; the base env is <base>.
        for candidate in (pathlib.Path(prefix).parent.parent,
                          pathlib.Path(prefix)):
            if (candidate / "etc" / "profile.d" / "conda.sh").exists():
                return candidate
    return None


class SubmitFailed(Exception):
    """sbatch rejected a submission — carries what it actually said."""


def _submit(cmd, *, dry_run):
    print(f"  sbatch: {shlex.join(str(c) for c in cmd)}")
    if dry_run:
        return None
    res = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True,
                         text=True)
    if res.returncode != 0:
        # Never let sbatch's diagnosis get swallowed: "AssocMaxSubmitJobLimit",
        # "Invalid job array specification" etc. is the whole story, and it is
        # on stderr, which capture_output hides.
        raise SubmitFailed(
            f"sbatch exited {res.returncode}\n"
            f"  stdout: {res.stdout.strip() or '(empty)'}\n"
            f"  stderr: {res.stderr.strip() or '(empty)'}")
    print(f"    {res.stdout.strip()}")
    return _job_id(res.stdout)


def _array_windows(num_items, cap, explicit=None, items_per_task=1):
    """``[(work_item_offset, array_spec), …]`` covering the work to submit.

    Two caps have to be respected at once:

    - ``MaxArraySize`` bounds the array *index*, so a worklist longer than that
      becomes several arrays, each indexed from 0 and carrying its window's
      start in ``RL_TASK_OFFSET``.
    - ``items_per_task`` (> 1) makes each array task walk that many consecutive
      work items, which is the lever for a cluster that caps how many jobs a
      user may have *queued* — 15,900 combinations at 10 per task is 1,590
      array tasks, not 15,900.

    The .sbatch does the arithmetic: ``start = RL_TASK_OFFSET +
    SLURM_ARRAY_TASK_ID * RL_ITEMS_PER_TASK``.
    """
    if explicit:
        indices = []
        for part in explicit.split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                indices.extend(range(int(start), int(end) + 1))
            else:
                indices.append(int(part))
        indices = sorted(set(indices))
        contiguous = indices == list(range(indices[0], indices[-1] + 1))
        if items_per_task > 1 and contiguous:
            # A contiguous resubmission is just a smaller worklist: same
            # arithmetic, shifted to start at indices[0]. The last task may
            # walk a little past the end, which is a no-op (those items either
            # already have their file or are out of range and skipped).
            return _dense_windows(indices[0], len(indices), cap, items_per_task)
        if items_per_task > 1:
            print(f"  note: --items-per-task {items_per_task} ignored — an "
                  f"array task walks CONSECUTIVE work items and "
                  f"--array {explicit} is not contiguous.")
        # Scattered indices, one per task. Windowed the same way so the spec
        # keeps working when it spans the cap.
        windows = []
        for start in range(0, num_items, cap):
            chunk = [i - start for i in indices if start <= i < start + cap]
            if chunk:
                windows.append((start, metrics_shards.compress_indices(chunk)))
        return windows

    return _dense_windows(0, num_items, cap, items_per_task)


def _dense_windows(first_item, count, cap, items_per_task):
    """Windows covering ``count`` consecutive work items from ``first_item``."""
    num_tasks = math.ceil(count / items_per_task)
    return [(first_item + task_offset * items_per_task,
             f"0-{min(cap, num_tasks - task_offset) - 1}")
            for task_offset in range(0, num_tasks, cap)]


def _report_partial_submission(exc, name, windows, submitted, num_items, args):
    """Explain a mid-way sbatch rejection and how to pick up where it stopped.

    Whatever was already submitted keeps running and its results are kept — a
    finished combination has its own file and is skipped on a resubmission — so
    the fix is never "start over", it is "submit the rest".
    """
    first_unsent = windows[submitted][0] if submitted < len(windows) else num_items
    print(f"\nsbatch rejected window {submitted + 1} of {len(windows)}:\n{exc}",
          file=sys.stderr)
    print(
        f"\n{submitted} array job(s) for {name!r} were accepted and are still "
        f"running; work items 0-{first_unsent - 1} are covered.\n"
        f"Nothing is lost — finished combinations are skipped on a "
        f"resubmission.\n"
        f"\nThe usual cause is a per-user cap on QUEUED jobs (not running "
        f"ones), so\n--max-concurrent does not help. Check yours with:\n"
        f"    sacctmgr show assoc user=$USER format=user,maxjobs,maxsubmitjobs\n"
        f"    scontrol show config | grep -i -E 'maxarray|maxsubmit'\n"
        f"\nTwo ways forward:\n"
        f"  1. Pack more work into each job (fewer jobs, same total work):\n"
        f"       python code/rlmodel/slurm/launch_metrics.py --figure {name} "
        f"--skip-prepare \\\n"
        f"           --items-per-task 20 --array {first_unsent}-{num_items - 1}\n"
        f"  2. Or just submit the rest later, once the queue has drained:\n"
        f"       python code/rlmodel/slurm/launch_metrics.py --figure {name} "
        f"--skip-prepare \\\n"
        f"           --array {first_unsent}-{num_items - 1}\n"
        f"\nThe merge job was NOT submitted. Run it yourself when the array "
        f"finishes:\n"
        f"    python code/rlmodel/metrics_runner.py --mode merge --work-dir "
        f"{windows and args.work_root / name}",
        file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="launch_metrics", allow_abbrev=False,
        description="Submit model_analysis's metrics collection as one Slurm "
                    "array task per (model, subject, iteration).")
    parser.add_argument("--figure", action="append", choices=sorted(FIGURES),
                        metavar="NAME",
                        help=f"Figure preset (repeatable). Choices: "
                             f"{', '.join(sorted(FIGURES))}. Default: all.")
    parser.add_argument("--num-evaluations", type=int, default=100)
    parser.add_argument("--n-psych-fits", type=int, default=None,
                        help="Psychometric multi-starts per fit; default is "
                             "aggregate.N_PSYCH_FITS.")
    parser.add_argument("--only-subject", action="append", default=None,
                        metavar="NAME")
    parser.add_argument("--work-root", type=pathlib.Path,
                        default=DEFAULT_WORK_ROOT)
    parser.add_argument("--result-dir", type=pathlib.Path,
                        default=DEFAULT_RESULT_DIR)
    parser.add_argument("--cache-dir", type=pathlib.Path,
                        default=DEFAULT_CACHE_DIR)
    parser.add_argument("--df-fp", type=pathlib.Path, default=DEFAULT_DF_FP)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild work dirs that already have a manifest.")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Use the existing work dirs as-is (resubmission).")
    parser.add_argument("--array", type=str, default=None, metavar="SPEC",
                        help="Submit only these worklist indices, e.g. "
                             "'17,220-233' — what the merge step prints for "
                             "the items it found missing.")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Cap simultaneously RUNNING tasks (sbatch's "
                             "--array=…%%N). Worth setting for a big run. Does "
                             "not reduce how many jobs are queued — see "
                             "--items-per-task for that.")
    parser.add_argument("--items-per-task", type=int, default=1, metavar="K",
                        help="Consecutive work items each array task walks "
                             "(default 1 — one job per combination). Raise it "
                             "when the cluster caps how many jobs you may have "
                             "QUEUED: K=20 turns 15,900 combinations into 795 "
                             "array tasks. Total work is unchanged; each job "
                             "just runs K x longer, so raise --time to match.")
    parser.add_argument("--max-array-size", type=int, default=None,
                        help="Override the MaxArraySize read from scontrol.")
    parser.add_argument("--conda-base", type=pathlib.Path, default=None,
                        metavar="DIR",
                        help="Conda install root (the directory holding "
                             "etc/profile.d/conda.sh). Auto-detected from the "
                             "submitting shell and exported to the jobs, which "
                             "cannot find conda by themselves. Set this only if "
                             "detection fails.")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't submit the dependent merge job.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the sbatch commands without submitting "
                             "(the work dirs are still built unless "
                             "--skip-prepare).")
    args, passthrough = parser.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    args.figure = args.figure or sorted(FIGURES)

    if args.skip_prepare:
        work_dirs = {name: args.work_root / name for name in args.figure}
        for name, work_dir in work_dirs.items():
            if not metrics_shards.manifest_path(work_dir).exists():
                sys.exit(f"--skip-prepare but {metrics_shards.manifest_path(work_dir)}"
                         f" does not exist; drop the flag to build it.")
    else:
        from paper_fast_slow.code.rlmodel.model.aggregate import N_PSYCH_FITS
        prep = argparse.Namespace(
            figure=args.figure, num_evaluations=args.num_evaluations,
            n_psych_fits=args.n_psych_fits or N_PSYCH_FITS,
            only_subject=args.only_subject, work_root=args.work_root,
            result_dir=args.result_dir, df_fp=args.df_fp, force=args.force)
        work_dirs = do_prepare(prep)

    cap = max_array_size(args.max_array_size)
    base = args.conda_base or conda_base()
    conda_export = f",RL_CONDA_BASE={base}" if base else ""
    if base:
        print(f"conda base : {base} (exported as RL_CONDA_BASE)")
    else:
        print("conda base : not detected — the jobs will search for one "
              "themselves. Pass --conda-base if they can't find it.")

    for name in args.figure:
        work_dir = pathlib.Path(work_dirs[name])
        _check_exportable(RL_PROJECT_ROOT=_PROJECT_ROOT, RL_WORK_DIR=work_dir,
                          RL_CACHE_DIR=args.cache_dir,
                          **({"RL_CONDA_BASE": base} if base else {}))
        num_items = metrics_shards.read_manifest(work_dir)["num_items"]
        windows = _array_windows(num_items, cap, args.array, args.items_per_task)

        log_dir = _SLURM_DIR / "logs" / f"metrics_{name}"
        print(f"\n=== {name} ===")
        print(f"work dir   : {work_dir}")
        print(f"log dir    : {log_dir}")
        print(f"work items : {num_items} ({args.items_per_task} per task, "
              f"MaxArraySize {cap} -> {len(windows)} array job(s))")
        if not args.dry_run:
            # Slurm resolves --output before the job body runs and won't create
            # the directory, so make it exist now.
            log_dir.mkdir(parents=True, exist_ok=True)

        array_ids, submitted_windows = [], 0
        try:
            for offset, spec in windows:
                if args.max_concurrent:
                    spec = f"{spec}%{args.max_concurrent}"
                export = (f"ALL,RL_PROJECT_ROOT={_PROJECT_ROOT},"
                          f"RL_WORK_DIR={work_dir},RL_TASK_OFFSET={offset},"
                          f"RL_ITEMS_PER_TASK={args.items_per_task}"
                          f"{conda_export}")
                cmd = ["sbatch",
                       f"--array={spec}",
                       f"--output={log_dir / '%A_%a.out'}",
                       f"--export={export}",
                       str(_RUN_SCRIPT),
                       *passthrough]
                job_id = _submit(cmd, dry_run=args.dry_run)
                submitted_windows += 1
                if job_id:
                    array_ids.append(job_id)
        except SubmitFailed as exc:
            _report_partial_submission(exc, name, windows, submitted_windows,
                                       num_items, args)
            return 1

        if args.no_merge:
            continue
        export = (f"ALL,RL_PROJECT_ROOT={_PROJECT_ROOT},"
                  f"RL_WORK_DIR={work_dir},RL_CACHE_DIR={args.cache_dir}"
                  f"{conda_export}")
        cmd = ["sbatch"]
        if array_ids:
            # afterany, not afterok: a handful of failed combinations shouldn't
            # block the merge, which reports and lets you resubmit just those.
            cmd.append(f"--dependency=afterany:{':'.join(array_ids)}")
        cmd += [f"--output={log_dir / 'merge_%j.out'}",
                f"--export={export}",
                str(_MERGE_SCRIPT)]
        _submit(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("\n--dry-run: nothing was submitted.")


if __name__ == "__main__":
    main()
