"""Run the paper's figure notebooks headless, through papermill.

    uv run python code/run_notebooks.py                     # run all, write nothing
    uv run python code/run_notebooks.py --save-figs          # regenerate results/
    uv run python code/run_notebooks.py --save-figs --paper-figures-only
    uv run python code/run_notebooks.py --only behavior opto
    uv run python code/run_notebooks.py --list
    uv run python code/run_notebooks.py --only widefield --param MAP=standard

Every figure notebook has one cell tagged ``parameters`` declaring

    SAVE_FIGS = False           # write figures under results/
    SAVE_DATA = False           # rewrite cached intermediate data under data/
    PAPER_FIGURES_ONLY = False  # skip per-subject / per-session figures

and papermill injects the values chosen here right after it. With no flags a
run **writes nothing** under ``results/`` or ``data/`` -- it only proves the
notebooks execute -- so saving is always a deliberate choice. ``SAVE_DATA``
rewrites cached intermediates that later notebooks and the download archive
depend on; leave it off unless that is the point.

Each notebook runs with its own directory as the working directory, because
their data paths (``../data/...``) and their package bootstrap are relative to
it. Executed copies, outputs included, are written under
``runs/<timestamp>/`` at the repository root (git-ignored).

A failing notebook does not stop the others; the summary at the end lists
every result, and the exit status is non-zero if anything failed.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent

#: The figure notebooks, in the order they are run, each with the parameters
#: that run sets on top of the runner's flags. Interactive viewers, diagnostics
#: and data_downloader.ipynb are deliberately absent.
#:
#: widefield.ipynb runs twice, once per reference map (its ``MAP`` parameter):
#: its default, the redefined MFC/LFC map, draws 3E-F, 4B, S5B and S7A-B; the
#: standard Allen map draws 3C, S5D and S5E, which no other run writes.
RUNS = [
    ("behavior.ipynb", {}),
    ("tracking.ipynb", {}),
    ("opto.ipynb", {}),
    ("widefield.ipynb", {}),
    ("widefield.ipynb", {"MAP": "standard"}),
    ("2p_0_preprocessing.ipynb", {}),
    ("2p_1_population.ipynb", {}),
    ("2p_2_sequences.ipynb", {}),
    ("2p_3_choice.ipynb", {}),
    ("2p_4_feedback.ipynb", {}),
    ("rlmodel/rlmodel_analysis.ipynb", {}),
    ("rlmodel/rlmodel_to_behavior.ipynb", {}),
    ("rlmodel/rlmodel_neural_correlate.ipynb", {}),
    ("rlmodel/rlmodel_compare.ipynb", {}),
]

#: Each figure notebook once, in run order.
NOTEBOOKS = list(dict.fromkeys(relpath for relpath, _ in RUNS))

#: Seconds to wait for a cell's remaining output once it has finished. The
#: default, 4, is too short for cells that draw many figures.
IOPUB_TIMEOUT = 180

#: What every notebook's parameters cell declares, and the runner's flags.
STANDARD_PARAMETERS = {
    "SAVE_FIGS": "save_figs",
    "SAVE_DATA": "save_data",
    "PAPER_FIGURES_ONLY": "paper_figures_only",
}


def notebookName(relpath: str) -> str:
    """``rlmodel/rlmodel_analysis.ipynb`` -> ``rlmodel_analysis``."""
    return Path(relpath).stem


def selectNotebooks(only=None, notebooks=NOTEBOOKS):
    """The notebooks to run, in run order; ``only`` names a subset by stem."""
    if not only:
        return list(notebooks)
    by_name = {notebookName(n): n for n in notebooks}
    unknown = [name for name in only if notebookName(name) not in by_name]
    if unknown:
        raise SystemExit(f"Unknown notebook(s): {', '.join(unknown)}. "
                         f"Choose from: {', '.join(by_name)}")
    wanted = {notebookName(name) for name in only}
    return [n for n in notebooks if notebookName(n) in wanted]


def selectRuns(only=None, runs=RUNS):
    """The runs of the notebooks ``only`` names (all of them if None), in run order."""
    chosen = set(selectNotebooks(only, list(dict.fromkeys(n for n, _ in runs))))
    return [(relpath, fixed) for relpath, fixed in runs if relpath in chosen]


def executedName(relpath, params, declared) -> str:
    """Where a run's executed copy goes, under the run directory: the notebook's
    own path, with any notebook-specific parameter that differs from its
    default appended (``widefield_standard.ipynb``), so two runs of one
    notebook do not overwrite each other."""
    changed = [str(value) for name, value in params.items()
               if name not in STANDARD_PARAMETERS and value != declared.get(name)]
    if not changed:
        return relpath
    path = Path(relpath)
    return (path.parent / f"{path.stem}_{'_'.join(changed)}{path.suffix}").as_posix()


def plannedRuns(runs, declared: dict, flags: dict, extra: dict):
    """``[(relpath, params, executed_name)]``: what each run injects, and where
    its executed copy goes.

    A run's own parameters sit on top of the runner's flags, and ``--param``
    on top of both, so ``--param MAP=standard`` makes both widefield runs draw
    the standard map. Runs that end up identical run once.
    """
    planned, seen = [], set()
    for relpath, fixed in runs:
        params = buildParameters(declared[relpath], flags,
                                 {**fixed, **extra})
        key = (relpath, tuple(sorted(params.items())))
        if key in seen:
            continue
        seen.add(key)
        planned.append((relpath, params, executedName(relpath, params, declared[relpath])))
    return planned


def parametersCell(nb: dict) -> dict:
    """The single cell tagged ``parameters``."""
    tagged = [c for c in nb["cells"]
              if "parameters" in c.get("metadata", {}).get("tags", [])]
    if len(tagged) != 1:
        raise ValueError(f"expected exactly one 'parameters' cell, found {len(tagged)}")
    return tagged[0]


def declaredParameters(nb: dict) -> dict:
    """``{name: default}`` for every plain assignment in the parameters cell."""
    source = "".join(parametersCell(nb)["source"])
    declared = {}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            declared[node.targets[0].id] = ast.literal_eval(node.value)
    return declared


def parseExtra(pairs):
    """``["MAP=standard"]`` -> ``{"MAP": "standard"}``."""
    extra = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name.isidentifier():
            raise SystemExit(f"--param expects NAME=VALUE, got {pair!r}")
        try:
            extra[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            extra[name] = value            # a bare string
    return extra


def buildParameters(declared: dict, flags: dict, extra: dict) -> dict:
    """What to inject into one notebook.

    The standard flags always go in -- every figure notebook declares them.
    An extra ``--param`` goes in only where the notebook declares that name, so
    a notebook-specific switch cannot leak into the others.
    """
    missing = [name for name in STANDARD_PARAMETERS if name not in declared]
    if missing:
        raise ValueError(f"parameters cell does not declare {', '.join(missing)}")
    params = {name: bool(flags[attr]) for name, attr in STANDARD_PARAMETERS.items()}
    params.update({name: value for name, value in extra.items() if name in declared})
    return params


def unusedExtra(extra: dict, declared_by_notebook: dict) -> list:
    """Extra parameters no selected notebook declares -- almost always a typo."""
    known = set().union(*declared_by_notebook.values()) if declared_by_notebook else set()
    return sorted(name for name in extra if name not in known)


def _runOne(relpath, run_dir, params, kernel, timeout, executed_name=None):
    import papermill as pm

    src = CODE_DIR / relpath
    dst = run_dir / (executed_name or relpath)
    dst.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        pm.execute_notebook(str(src), str(dst), parameters=params, cwd=str(src.parent),
                            kernel_name=kernel, execution_timeout=timeout,
                            iopub_timeout=IOPUB_TIMEOUT, progress_bar=False, log_output=False)
        return "ok", "", time.time() - started
    except pm.PapermillExecutionError as err:
        return "FAILED", f"cell {err.cell_index}: {err.ename}: {err.evalue}"[:300], \
               time.time() - started
    except Exception as err:    # a dead kernel, a timeout: the next notebook still runs
        return "FAILED", f"{type(err).__name__}: {err}"[:300], time.time() - started


def _argParser():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Without --save-figs or --save-data nothing is written.")
    parser.add_argument("--save-figs", action="store_true",
                        help="write figures under results/")
    parser.add_argument("--save-data", action="store_true",
                        help="rewrite cached intermediate data under data/ (careful)")
    parser.add_argument("--paper-figures-only", action="store_true",
                        help="skip per-subject / per-session figures")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="run just these notebooks, by name without .ipynb")
    parser.add_argument("--param", action="append", metavar="NAME=VALUE",
                        help="set a notebook-specific parameter, e.g. MAP=standard")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name")
    parser.add_argument("--timeout", type=int, default=None,
                        help="per-cell timeout in seconds (default: none)")
    parser.add_argument("--out", type=Path, default=None,
                        help="where executed notebooks go (default: runs/<timestamp>)")
    parser.add_argument("--list", action="store_true",
                        help="list the notebooks and their parameters, then exit")
    return parser


def main(argv=None):
    args = _argParser().parse_args(argv)
    runs = selectRuns(args.only)
    extra = parseExtra(args.param)
    declared = {n: declaredParameters(json.load(open(CODE_DIR / n, encoding="utf-8")))
                for n, _ in runs}

    typos = unusedExtra(extra, declared)
    if typos:
        raise SystemExit(f"No selected notebook declares: {', '.join(typos)}")

    flags = {"save_figs": args.save_figs, "save_data": args.save_data,
             "paper_figures_only": args.paper_figures_only}
    planned = plannedRuns(runs, declared, flags, extra)

    if args.list:
        # The values each run would use, not the notebook's defaults.
        for relpath, params, name in planned:
            print(f"{name:36s} {declared[relpath] | params}")
        return 0
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out or (REPO_DIR / "runs" / stamp)
    print(f"Running {len(planned)} notebook run(s) -> {run_dir}")
    print(f"  SAVE_FIGS={args.save_figs} SAVE_DATA={args.save_data} "
          f"PAPER_FIGURES_ONLY={args.paper_figures_only}"
          + (f" extra={extra}" if extra else ""))

    results = []
    for relpath, params, name in planned:
        print(f"-> {name} ...", flush=True)
        status, detail, seconds = _runOne(relpath, run_dir, params, args.kernel,
                                          args.timeout, executed_name=name)
        results.append((name, status, detail, seconds))
        print(f"   {status} in {seconds / 60:.1f} min {detail}", flush=True)

    print("\nSummary")
    for name, status, detail, seconds in results:
        print(f"  {status:6s} {seconds / 60:6.1f} min  {name}  {detail}")
    failed = [r for r in results if r[1] != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
