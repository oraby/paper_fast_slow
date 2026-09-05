"""CLI for the sharded metrics collection (see ``model/metrics_shards.py``).

Three modes, run as a plain script (paths below are from the project root, but
any cwd works)::

    # 1. once: build the work dir(s) for the figures you want
    python code/rlmodel/metrics_runner.py --mode prepare \
        --figure fig1l --num-evaluations 100

    # 2. per Slurm array task: evaluate ONE combination
    python code/rlmodel/metrics_runner.py --mode run \
        --work-dir data/RLModel/metrics/work/fig1l --task-id 42

    # 3. once, after the array: shards -> the pickle the notebook reads
    python code/rlmodel/metrics_runner.py --mode merge \
        --work-dir data/RLModel/metrics/work/fig1l

``code/rlmodel/slurm/launch_metrics.py`` chains all three for you. Running the
whole thing locally instead is ``--mode run --all [--num-cpus N]``.

Every path defaults to an absolute one derived from this file, so any mode runs
from any working directory — unlike the notebook-relative
``aggregate.DEFAULT_METRICS_CACHE_DIR``.

A plain script and NOT ``python -m code.rlmodel.metrics_runner`` (which is how
``model_runner`` is launched): ``-m`` prepends the cwd to ``sys.path``, so the
project's ``code/`` package shadows the stdlib ``code`` module, and this
module's import chain reaches IPython, which imports ``pdb``, which needs the
real one. Running it as a script puts ``code/rlmodel/`` on ``sys.path`` instead
and the clash disappears — the same reason ``golden_fig1l.py`` is a script.
The fit pickles still need that ``code`` package to exist while they are read;
:func:`code_package_alias` supplies it for exactly that long.
"""
import os

# Each task owns exactly ONE cpu on a shared node; BLAS must not spawn a thread
# per core underneath it. Set before numpy is imported anywhere below.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse          # noqa: E402
import contextlib        # noqa: E402
import pathlib           # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402

import matplotlib        # noqa: E402
matplotlib.use("Agg")    # no display on a compute node


def _bootstrap():
    """Make the deep package imports work as a script (the same trick as
    ``golden_fig1l.py`` and the notebooks' first cell).

    The prefix comes from the checkout's own directory name, so this runs from
    a clone under any name or location.
    """
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]           # the checkout, whatever it is called
    if str(repo_root.parent) not in sys.path:
        sys.path.insert(0, str(repo_root.parent))
    return repo_root, f"{repo_root.name}.code.rlmodel"


_PROJECT_ROOT, _RLMODEL_PKG = _bootstrap()

import importlib                                                  # noqa: E402
compare = importlib.import_module(f"{_RLMODEL_PKG}.model.compare")
metrics_shards = importlib.import_module(
    f"{_RLMODEL_PKG}.model.metrics_shards")
_aggregate = importlib.import_module(f"{_RLMODEL_PKG}.model.aggregate")
DRIFT_RR_SPECS = _aggregate.DRIFT_RR_SPECS
FIG1L_SPECS = _aggregate.FIG1L_SPECS
MIN_NUM_TRIALS = _aggregate.MIN_NUM_TRIALS
MLE_WEIGHT_SPECS = _aggregate.MLE_WEIGHT_SPECS
N_PSYCH_FITS = _aggregate.N_PSYCH_FITS
SCALE_BOUND_SPECS = _aggregate.SCALE_BOUND_SPECS


DEFAULT_RESULT_DIR = _PROJECT_ROOT / "data" / "RLModel"
DEFAULT_CACHE_DIR = DEFAULT_RESULT_DIR / "metrics"
DEFAULT_WORK_ROOT = DEFAULT_CACHE_DIR / "work"
DEFAULT_DF_FP = _PROJECT_ROOT / "data" / "behavior" / "df_behavior.pkl"

# The figure presets, keyed by the cache_name the notebook passes to
# load_or_collect_metrics — so a merged file lands exactly where cell 18/26/28/30
# looks for it.
FIGURES = {
    "fig1l": FIG1L_SPECS,
    "scale_bound": SCALE_BOUND_SPECS,
    "mle_weights": MLE_WEIGHT_SPECS,
    "drift_rr": DRIFT_RR_SPECS,
}


def _parse_range(text):
    """``"12"`` → ``[12]``; ``"12-15"`` → ``[12, 13, 14, 15]``."""
    if "-" not in text:
        return [int(text)]
    start, end = text.split("-", 1)
    return list(range(int(start), int(end) + 1))


# Derived from the checkout, never hardcoded -- a clone under a different
# directory name aliases its own package, not this one.
_ALIAS_PREFIX = f"{_PROJECT_ROOT.name}.code."


@contextlib.contextmanager
def code_package_alias():
    """Make the project's ``code`` package importable under that bare name.

    The fit pickles were written by ``python -m code.rlmodel.model_runner``, so
    they name their classes ``code.rlmodel.model.…`` and will not unpickle
    unless a ``code`` package answers to that. The notebooks get this for free
    (their first cell registers ``code`` in ``sys.modules``); a script has to
    ask for it — and only around the read, because the alias shadows the stdlib
    ``code`` module for as long as it is installed.

    Every ``paper_fast_slow.code.*`` module already imported is aliased to the
    *same* object, so unpickled classes are the ones this process is already
    using rather than a second, unrelated copy of each.
    """
    root = _ALIAS_PREFIX.rstrip(".")          # "<checkout>.code"
    imported = {name: module for name, module in sys.modules.items()
                if name == root or name.startswith(root + ".")}
    originals = {}
    for name, module in imported.items():
        alias = "code" + name[len(_ALIAS_PREFIX) - 1:]
        originals[alias] = sys.modules.get(alias)
        sys.modules[alias] = module
    before = set(sys.modules)
    try:
        yield
    finally:
        # Drop whatever pickle imported fresh under the alias, then restore.
        for name in set(sys.modules) - before:
            if name == "code" or name.startswith("code."):
                sys.modules.pop(name, None)
        for alias, original in originals.items():
            if original is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = original


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
def do_prepare(args):
    """Build one work dir per requested figure. Returns ``{name: work_dir}``."""
    work_root = pathlib.Path(args.work_root)
    print("Loading behavior dataframe...")
    df_behavior = compare.prepare_behavior_df(df_fp=str(args.df_fp))
    print(f"Discovering fits in {args.result_dir}...")
    with code_package_alias():
        fits = compare.discover_fits(result_dir=str(args.result_dir))
    if not fits:
        sys.exit(
            f"discover_fits() found no readable fits in {args.result_dir}.\n"
            "discover_fits silently skips pickles it cannot read, so check the\n"
            "skip reasons above. A pandas mismatch means the wrong "
            "environment:\n"
            "the saved fits embed a subject_df written by a newer pandas, so\n"
            "run this in the conda environment the notebooks use, not the uv "
            "venv.")

    built = {}
    for name in args.figure:
        work_dir = work_root / name
        if metrics_shards.manifest_path(work_dir).exists() and not args.force:
            sys.exit(f"{metrics_shards.manifest_path(work_dir)} already exists. "
                     f"Pass --force to rebuild it (this invalidates the "
                     f"existing shards, whose indices are worklist lines).")
        print(f"\n=== {name} ===")
        metrics_shards.build_workdir(
            fits, FIGURES[name], df_behavior, work_dir, cache_name=name,
            num_evaluations=args.num_evaluations,
            n_psych_fits=args.n_psych_fits, result_dir=str(args.result_dir),
            subjects=args.only_subject)
        built[name] = work_dir
    return built


def do_run(args):
    """Evaluate one work item, a range of them, or all of them."""
    work_dir = pathlib.Path(args.work_dir)
    # The items' own indices, not 0..n-1: an index is a worklist LINE.
    known = [item.index for item in metrics_shards.read_worklist(work_dir)]
    if args.all:
        indices = known
    elif args.task_range:
        # Clamped to what exists: a Slurm task covering a fixed-size block of
        # items runs past the end of the worklist whenever the block size
        # doesn't divide it, and that tail is not an error.
        wanted = set(_parse_range(args.task_range))
        indices = [i for i in known if i in wanted]
        if not indices:
            print(f"No work items in range {args.task_range}; nothing to do.")
            return
    else:
        indices = [args.task_id]

    kwargs = dict(min_num_trials=args.min_num_trials,
                  save_sim=not args.no_save_sim, overwrite=args.overwrite)
    started = time.time()
    if args.num_cpus and args.num_cpus > 1 and len(indices) > 1:
        # Local convenience only (a laptop / an interactive node). On Slurm each
        # task is a separate single-CPU job and this branch is never taken.
        import multiprocessing
        import functools
        run = functools.partial(_run_one, str(work_dir), kwargs)
        with multiprocessing.Pool(args.num_cpus) as pool:
            results = list(pool.imap_unordered(run, indices, chunksize=1))
    else:
        results = [metrics_shards.run_work_item(work_dir, i, **kwargs)
                   for i in indices]

    counts = {status: results.count(status) for status in set(results)}
    print(f"{len(indices)} item(s) in {time.time() - started:.1f}s: "
          + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))


def _run_one(work_dir, kwargs, index):
    """Module-level so the local --num-cpus pool can pickle it."""
    return metrics_shards.run_work_item(work_dir, index, verbose=False, **kwargs)


def do_merge(args):
    """Shards → ``metrics_<name>.pkl`` (+ the sim-cache sidecar)."""
    for work_dir in args.work_dir:
        work_dir = pathlib.Path(work_dir)
        print(f"\n=== {work_dir} ===")
        manifest = metrics_shards.read_manifest(work_dir)
        result = metrics_shards.merge_shards(work_dir)
        if result.missing and not args.allow_missing:
            sys.exit(
                f"{len(result.missing)} of {manifest['num_items']} work item(s) "
                f"have no shard, so the frame would be incomplete. Resubmit "
                f"them, or pass --allow-missing to write what is there anyway.")
        if args.dry_run:
            print(f"--dry-run: would write metrics_{manifest['cache_name']}.pkl "
                  f"({len(result.metrics_df)} rows) to {args.cache_dir}")
            continue
        if result.missing:
            # The file records the REQUESTED num_evaluations, so the notebook's
            # cache check will accept it as complete. Say so out loud.
            print(f"WARNING: writing with {len(result.missing)} item(s) missing. "
                  f"The file still claims {manifest['num_evaluations']} "
                  f"evaluation(s), so the notebook will treat it as complete — "
                  f"some (model, subject) pairs will simply average over fewer "
                  f"seeds.")
        metrics_shards.write_metrics_cache(result, manifest, args.cache_dir)


# --------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        prog="metrics_runner", allow_abbrev=False,
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True,
                        choices=["prepare", "run", "merge"])

    prep = parser.add_argument_group("prepare")
    prep.add_argument("--figure", action="append", choices=sorted(FIGURES),
                      metavar="NAME",
                      help=f"Figure preset to collect (repeatable). "
                           f"Choices: {', '.join(sorted(FIGURES))}. "
                           f"Default: all of them.")
    prep.add_argument("--num-evaluations", type=int, default=100,
                      help="Seeds 0..N-1 per (model, subject). Default 100.")
    prep.add_argument("--n-psych-fits", type=int, default=N_PSYCH_FITS,
                      help=f"Psychometric multi-starts per fit — the dominant "
                           f"cost of one evaluation. Default {N_PSYCH_FITS}; "
                           f"lower only for exploratory passes, it changes the "
                           f"fitted values.")
    prep.add_argument("--only-subject", action="append", default=None,
                      metavar="NAME",
                      help="Restrict the fan-out to these subjects (repeatable).")
    prep.add_argument("--work-root", type=pathlib.Path,
                      default=DEFAULT_WORK_ROOT,
                      help=f"Parent of the per-figure work dirs. "
                           f"Default {DEFAULT_WORK_ROOT}")
    prep.add_argument("--result-dir", type=pathlib.Path,
                      default=DEFAULT_RESULT_DIR,
                      help=f"Where the fit pickles live. Default {DEFAULT_RESULT_DIR}")
    prep.add_argument("--df-fp", type=pathlib.Path, default=DEFAULT_DF_FP,
                      help=f"Behavior dataframe. Default {DEFAULT_DF_FP}")
    prep.add_argument("--force", action="store_true",
                      help="Rebuild a work dir that already has a manifest.")

    run = parser.add_argument_group("run")
    run.add_argument("--work-dir", action="append", default=None,
                     help="The work dir built by --mode prepare. Repeatable "
                          "for --mode merge; exactly one for --mode run.")
    run.add_argument("--task-id", type=int, default=None,
                     help="Index of the single work item to evaluate.")
    run.add_argument("--task-range", type=str, default=None, metavar="A-B",
                     help="Inclusive range of work items to evaluate serially.")
    run.add_argument("--all", action="store_true",
                     help="Evaluate every work item (a local full run).")
    run.add_argument("--min-num-trials", type=int, default=MIN_NUM_TRIALS,
                     help=f"Subjects simulating fewer valid trials than this "
                          f"contribute no metrics. Default {MIN_NUM_TRIALS}.")
    run.add_argument("--no-save-sim", action="store_true",
                     help="Don't keep the seed-0 simulation frames (the "
                          "per-subject figure panels then have no data).")
    run.add_argument("--overwrite", action="store_true",
                     help="Re-evaluate items that already have a shard. Off by "
                          "default, which is what makes a blanket resubmission "
                          "only redo what is missing.")
    run.add_argument("--num-cpus", type=int, default=None,
                     help="Local-run only: process the selected items in a "
                          "process pool. Not used on Slurm (one cpu per task).")

    merge = parser.add_argument_group("merge")
    merge.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE_DIR,
                       help=f"Where metrics_<name>.pkl is written — the "
                            f"directory the notebook reads. "
                            f"Default {DEFAULT_CACHE_DIR}")
    merge.add_argument("--allow-missing", action="store_true",
                       help="Write the frame even though some work items have "
                            "no shard.")
    merge.add_argument("--dry-run", action="store_true",
                       help="Report what would be written, write nothing.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.mode == "prepare":
        args.figure = args.figure or sorted(FIGURES)
        do_prepare(args)
        return
    if not args.work_dir:
        sys.exit(f"--mode {args.mode} needs --work-dir.")
    if args.mode == "merge":
        do_merge(args)
        return

    if len(args.work_dir) != 1:
        sys.exit("--mode run takes exactly one --work-dir.")
    selectors = (args.task_id is not None, args.task_range is not None,
                 bool(args.all))
    if sum(selectors) != 1:
        sys.exit("--mode run needs exactly one of --task-id / --task-range / "
                 "--all.")
    args.work_dir = args.work_dir[0]
    do_run(args)


if __name__ == "__main__":
    main()
