"""Sharded metrics collection — one Slurm task per (model × subject × iteration).

``aggregate.collect_metrics`` evaluates ``specs × subjects × num_evaluations``
in one serial loop. At ``num_evaluations=100`` that is hours per figure, which
is what this module exists to escape: the same work is enumerated up front and
run as thousands of independent single-CPU jobs, each writing **its own** result
file. Nothing is appended to a shared file, so there is no locking and no race
to get wrong, and a job that dies takes exactly one shard with it.

Three phases, one per public entry point:

``build_workdir``  (once, on the login node)
    Resolve the specs, extract the handful of fitted parameters each simulation
    actually needs, slice the behavior dataframe per subject, and enumerate the
    combinations into ``worklist.tsv``. This is what makes a task cheap: a
    naive one would re-run ``prepare_behavior_df`` and unpickle a 29–339 MB fit
    file for a few floats — tens of seconds of startup against ~2.5 s of work,
    times thousands of jobs.

``run_work_item``  (once per array task)
    One line of the worklist → one ``shards/<index>.pkl``.

``merge_shards``  (once, after the array)
    Every shard → the exact frame ``collect_metrics`` would have returned, ready
    for :func:`write_metrics_cache` to store in the on-disk format
    ``aggregate.load_or_collect_metrics`` reads. The notebook then cache-hits
    and never simulates.

Work directory layout (``data/RLModel/metrics/work/<cache_name>/``)::

    manifest.pkl          specs + per (spec, subject) params/flags/loss
    behavior/b###.pkl     the prepared behavior frame, sliced per subject
    worklist.tsv          "<spec_idx>\\t<subject>\\t<iteration>"; line N == task N
    shards/######.pkl     one metric row (or a below-min-trials marker)
    simshards/######.pkl  seed-0 only: the frame the per-subject figures want
    errors/######.pkl     what went wrong, for tasks that raised

Determinism
-----------
The DDM trajectory is seeded from the iteration (``logic.makeOneRun``), but the
psychometric fit's multi-starts draw from the *unseeded* global ``np.random``
(``psychofit.mle_fit_psycho``). Independent jobs would therefore each draw
arbitrary start points and no two runs of the pipeline would agree. Every task
seeds the global RNG from its own coordinates instead (see :func:`task_seed`),
so a shard's contents depend only on which work item it is — not on which node,
process or order it ran in.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
import pickle
import zlib

import numpy as np
import pandas as pd

from . import compare
from .aggregate import (MIN_NUM_TRIALS, N_PSYCH_FITS, _source_fit_paths,
                        _spec_key, frame_from_rows, resolve_spec,
                        save_sim_cache, subject_metrics)
from .mle_reeval import fitted_params_from_result


MANIFEST_NAME = "manifest.pkl"
WORKLIST_NAME = "worklist.tsv"
SHARD_DIRNAME = "shards"
SIM_SHARD_DIRNAME = "simshards"
BEHAVIOR_DIRNAME = "behavior"
ERROR_DIRNAME = "errors"

# Rough per-evaluation cost, for the wall-time estimate ``prepare`` prints.
SECONDS_PER_EVALUATION = 2.5


@dataclass(frozen=True)
class WorkItem:
    """One unit of work == one array task == one shard."""
    index: int
    spec_idx: int
    subject: str
    iteration: int


@dataclass
class MergeResult:
    metrics_df: pd.DataFrame
    sim_cache: dict
    missing: list          # indices with no shard — the array task never finished
    skipped: list          # (index, reason) — below min_num_trials
    errors: list           # (index, message) — the simulation raised


# --------------------------------------------------------------------------
# Paths + atomic IO
# --------------------------------------------------------------------------
def manifest_path(work_dir):
    return pathlib.Path(work_dir) / MANIFEST_NAME


def worklist_path(work_dir):
    return pathlib.Path(work_dir) / WORKLIST_NAME


def shard_path(work_dir, index):
    return pathlib.Path(work_dir) / SHARD_DIRNAME / f"{index:06d}.pkl"


def sim_shard_path(work_dir, index):
    return pathlib.Path(work_dir) / SIM_SHARD_DIRNAME / f"{index:06d}.pkl"


def error_path(work_dir, index):
    """Where a failed task records why.

    Deliberately NOT under ``shards/``: the shard-exists check is what makes a
    blanket resubmission skip finished work, and a failure must stay retryable.
    """
    return pathlib.Path(work_dir) / ERROR_DIRNAME / f"{index:06d}.pkl"


def _atomic_dump(obj, fp):
    """Pickle ``obj`` to ``fp`` via a pid-suffixed temp + ``replace``.

    Same trick as ``fit._merge_save_evolve``: a task killed mid-write (Slurm
    timeout, preemption) must never leave a truncated shard that ``merge_shards``
    would then read as real data.
    """
    fp = pathlib.Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    tmp = fp.parent / f"{fp.name}.tmp.{os.getpid()}"
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    tmp.replace(fp)
    return fp


# --------------------------------------------------------------------------
# Phase 1 — build the work directory
# --------------------------------------------------------------------------
def build_workdir(fits, specs, df_behavior, work_dir, *, cache_name,
                  num_evaluations, n_psych_fits=N_PSYCH_FITS,
                  result_dir=compare.DEFAULT_RESULT_DIR, subjects=None,
                  verbose=True):
    """Write the manifest, per-subject behavior slices and worklist.

    ``subjects`` restricts the fan-out to those names (for a quick test);
    ``None`` means every subject each spec resolves to.

    Returns the manifest dict. The heavy inputs (``fits`` — up to ~1.9 GB of
    unpickled fit payloads — and the full ``df_behavior``) are consumed here and
    never reach a task: only the fitted params, the include flags and the
    subject's own rows are written out.
    """
    work_dir = pathlib.Path(work_dir)
    behavior_dir = work_dir / BEHAVIOR_DIRNAME
    behavior_dir.mkdir(parents=True, exist_ok=True)

    specs = tuple(specs)
    contexts, items = {}, []
    for spec_idx, spec in enumerate(specs):
        resolved = resolve_spec(fits, spec)
        chosen = sorted(resolved) if subjects is None else [
            s for s in sorted(resolved) if s in set(subjects)]
        if subjects is not None and not chosen:
            raise KeyError(
                f"Spec {spec.label!r} resolves to none of the requested "
                f"subjects {sorted(set(subjects))}. It has: {sorted(resolved)}")
        for subject in chosen:
            col_fit = resolved[subject]
            include_Q, include_RewardRate = compare._include_flags(col_fit.payload)
            if include_Q is None or include_RewardRate is None:
                raise KeyError(
                    f"Fit {col_fit.filename!r} / subject {subject!r} is missing "
                    f"the include_Q / include_RewardRate flags; without them the "
                    f"simulation would silently run the wrong model.")
            contexts[(spec_idx, subject)] = {
                "fid": col_fit.fid,
                "params": fitted_params_from_result(col_fit.payload),
                "include_Q": include_Q,
                "include_RewardRate": include_RewardRate,
                # The fit's own loss, which sim_cache carries for the figure
                # titles. Read here so no task needs the payload.
                "loss": float(getattr(col_fit.payload.get("OptimRes"), "fun",
                                      np.nan)),
                "filename": col_fit.filename,
            }
            for iteration in range(num_evaluations):
                items.append((spec_idx, subject, iteration))

    if not items:
        raise ValueError("No (spec, subject) combination to evaluate.")

    # One behavior file per subject, keyed by an index so no subject name ever
    # has to survive a round trip through a filesystem.
    all_subjects = sorted({subject for _, subject in contexts})
    behavior_files = {}
    for idx, subject in enumerate(all_subjects):
        name = f"b{idx:03d}.pkl"
        subject_df = df_behavior[df_behavior.Name == subject]
        with (behavior_dir / name).open("wb") as f:
            pickle.dump(subject_df, f)
        behavior_files[subject] = name

    manifest = {
        "cache_name": cache_name,
        "specs": specs,
        "spec_key": _spec_key(specs),
        "num_evaluations": num_evaluations,
        "n_psych_fits": n_psych_fits,
        "source_files": [p.name for p in _source_fit_paths(fits, specs,
                                                           result_dir)],
        "contexts": contexts,
        "behavior_files": behavior_files,
        "num_items": len(items),
    }
    _atomic_dump(manifest, manifest_path(work_dir))

    lines = [f"{spec_idx}\t{subject}\t{iteration}"
             for spec_idx, subject, iteration in items]
    worklist_path(work_dir).write_text("\n".join(lines) + "\n",
                                       encoding="utf-8", newline="\n")
    if verbose:
        print(f"Work dir  : {work_dir}")
        print(f"Specs     : {len(specs)} × {len(all_subjects)} subject(s) × "
              f"{num_evaluations} evaluation(s)")
        print(f"Tasks     : {len(items)}  (~{len(items) * SECONDS_PER_EVALUATION / 3600:.1f} "
              f"CPU-hours at {SECONDS_PER_EVALUATION:g}s each)")
    return manifest


def read_manifest(work_dir):
    with manifest_path(work_dir).open("rb") as f:
        return pickle.load(f)


def read_worklist(work_dir):
    """``[WorkItem, …]`` — a work item's ``index`` is its line in ``worklist.tsv``.

    That line number is the Slurm array index, so it is the identity of the
    item; blank lines are skipped without renumbering the ones after them.
    Address items by ``WorkItem.index``, not by position in this list.
    """
    text = worklist_path(work_dir).read_text(encoding="utf-8")
    items = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        spec_idx, subject, iteration = line.split("\t")
        items.append(WorkItem(index=index, spec_idx=int(spec_idx),
                              subject=subject, iteration=int(iteration)))
    return items


# --------------------------------------------------------------------------
# Phase 2 — run one work item
# --------------------------------------------------------------------------
def task_seed(spec_label, subject, iteration):
    """A stable 32-bit seed for one work item.

    ``crc32`` and not ``hash()``: Python randomizes string hashing per process,
    so ``hash()`` would give a different psychometric multi-start on every job
    and the pipeline would not be reproducible.
    """
    key = repr((spec_label, subject, int(iteration))).encode("utf-8")
    return zlib.crc32(key) & 0xFFFF_FFFF


# Per-process caches so a ``--all`` / ``--num-cpus`` run re-reads neither the
# manifest, the worklist nor a subject's frame. A single Slurm task touches each
# once anyway, so this costs nothing there.
_MANIFEST_CACHE = {}
_WORKLIST_CACHE = {}
_BEHAVIOR_CACHE = {}


def _cached_manifest(work_dir):
    key = str(pathlib.Path(work_dir).resolve())
    if key not in _MANIFEST_CACHE:
        _MANIFEST_CACHE[key] = read_manifest(work_dir)
    return _MANIFEST_CACHE[key]


def _cached_worklist(work_dir):
    """``{index: WorkItem}`` — keyed by the array index, not list position."""
    key = str(pathlib.Path(work_dir).resolve())
    if key not in _WORKLIST_CACHE:
        _WORKLIST_CACHE[key] = {item.index: item
                                for item in read_worklist(work_dir)}
    return _WORKLIST_CACHE[key]


def _cached_behavior(work_dir, manifest, subject):
    key = (str(pathlib.Path(work_dir).resolve()), subject)
    if key not in _BEHAVIOR_CACHE:
        fp = (pathlib.Path(work_dir) / BEHAVIOR_DIRNAME /
              manifest["behavior_files"][subject])
        with fp.open("rb") as f:
            _BEHAVIOR_CACHE[key] = pickle.load(f)
    return _BEHAVIOR_CACHE[key]


def clear_caches():
    """Drop the per-process caches (tests, long-lived hosts)."""
    _MANIFEST_CACHE.clear()
    _WORKLIST_CACHE.clear()
    _BEHAVIOR_CACHE.clear()


def run_work_item(work_dir, index, *, min_num_trials=MIN_NUM_TRIALS,
                  save_sim=True, overwrite=False, verbose=True):
    """Evaluate work item ``index`` and write its shard.

    Returns ``"done"``, ``"skipped"`` (below ``min_num_trials``) or ``"cached"``
    (a shard already existed and ``overwrite`` is false — which is what makes
    resubmitting the whole array after a partial failure cheap). A skip writes a
    shard too, so it is cached like any other outcome: changing
    ``min_num_trials`` afterwards needs ``overwrite=True`` to take effect.

    A simulation error is recorded in the shard *and* re-raised, so the Slurm
    task exits non-zero and shows up in ``sacct`` instead of failing silently.
    """
    work_dir = pathlib.Path(work_dir)
    out_fp = shard_path(work_dir, index)
    if out_fp.exists() and not overwrite:
        if verbose:
            print(f"[{index}] shard exists, skipping")
        return "cached"

    manifest = _cached_manifest(work_dir)
    items = _cached_worklist(work_dir)
    if index not in items:
        raise IndexError(f"Work item {index} out of range; the worklist at "
                         f"{worklist_path(work_dir)} holds {len(items)} item(s).")
    item = items[index]
    spec = manifest["specs"][item.spec_idx]
    ctx = manifest["contexts"][(item.spec_idx, item.subject)]
    seed = item.iteration

    # Before anything that touches the global RNG (the psychometric fit does).
    np.random.seed(task_seed(spec.label, item.subject, item.iteration))

    subject_df = _cached_behavior(work_dir, manifest, item.subject)
    sim_df, bound, _bias_kwargs, error = compare._compute_sim_from_params(
        item.subject, ctx["fid"], ctx["params"], subject_df,
        include_Q=ctx["include_Q"], include_RewardRate=ctx["include_RewardRate"],
        seed=seed)
    if error is not None:
        _atomic_dump({"error": error, "index": index, "Name": item.subject,
                      "SpecLabel": spec.label},
                     error_path(work_dir, index))
        raise RuntimeError(
            f"[{index}] {spec.label} / {item.subject} / seed {seed}: {error}")

    if len(sim_df) < min_num_trials:
        # The serial loop breaks out of the iteration loop here, so such a
        # subject contributes no rows and no sim_cache entry. Independent tasks
        # can't break, so each records the skip and merge_shards drops them.
        _atomic_dump({"skipped": "below_min_trials", "index": index,
                      "Name": item.subject, "NumTrials": len(sim_df)}, out_fp)
        if verbose:
            print(f"[{index}] {spec.label} / {item.subject}: "
                  f"{len(sim_df)} trials < {min_num_trials}, skipped")
        return "skipped"

    if item.iteration == 0 and save_sim:
        _atomic_dump((ctx["loss"], sim_df, bound, ctx["include_Q"],
                      ctx["include_RewardRate"]),
                     sim_shard_path(work_dir, index))

    row = subject_metrics(item.subject, sim_df,
                          n_psych_fits=manifest["n_psych_fits"])
    row.update(SpecLabel=spec.label, ModelKey=spec.model_key,
               ColumnLabel=spec.column_label, Iteration=item.iteration,
               Seed=seed)
    _atomic_dump(row, out_fp)
    if verbose:
        print(f"[{index}] {spec.label} / {item.subject} / seed {seed}: done "
              f"({len(sim_df)} trials)")
    return "done"


# --------------------------------------------------------------------------
# Phase 3 — merge
# --------------------------------------------------------------------------
def merge_shards(work_dir, *, verbose=True):
    """Every shard → the frame ``collect_metrics`` would have returned."""
    work_dir = pathlib.Path(work_dir)
    manifest = read_manifest(work_dir)
    items = read_worklist(work_dir)

    rows, missing, skipped, errors, sim_cache = [], [], [], [], {}
    for item in items:
        fp = shard_path(work_dir, item.index)
        if not fp.exists():
            missing.append(item.index)
            err_fp = error_path(work_dir, item.index)
            if err_fp.exists():
                try:
                    with err_fp.open("rb") as f:
                        errors.append((item.index, pickle.load(f)["error"]))
                except Exception as exc:  # noqa: BLE001
                    errors.append((item.index, f"unreadable error file: {exc!r}"))
            continue
        try:
            with fp.open("rb") as f:
                shard = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 — one bad shard isn't fatal
            missing.append(item.index)
            errors.append((item.index, f"unreadable shard: {exc!r}"))
            continue
        if "skipped" in shard:
            skipped.append((item.index, shard["skipped"]))
            continue
        rows.append(shard)

        sim_fp = sim_shard_path(work_dir, item.index)
        if item.iteration == 0 and sim_fp.exists():
            spec = manifest["specs"][item.spec_idx]
            with sim_fp.open("rb") as f:
                sim_cache.setdefault(spec.label, {})[item.subject] = pickle.load(f)

    if not rows:
        detail = (f"{len(missing)} missing, {len(skipped)} below "
                  f"min_num_trials, {len(errors)} recorded an error")
        hint = ""
        if len(missing) == len(items) and not errors:
            # Nothing ran at all: not one task got as far as writing a result,
            # and none reported a failure from inside Python either. So the
            # array is still queued, or the jobs are dying before the work —
            # bad conda env, wrong cwd, OOM, time limit. The job log says which.
            hint = ("\nNot one task produced output AND none reported an "
                    "error, so the work never started. Check, in order:\n"
                    "  squeue -u $USER                   # still PENDING?\n"
                    "  sacct -X -j <jobid> --format=JobID,State,ExitCode,"
                    "Elapsed\n"
                    "  ls code/rlmodel/slurm/logs/metrics_*/  # then read one\n"
                    "A task that ran at all prints 'host=… work_dir=…' as its "
                    "first line.")
        raise ValueError(
            f"No shard in {work_dir / SHARD_DIRNAME} produced metrics "
            f"({detail}).{hint}")

    metrics_df = frame_from_rows(rows).sort_values(
        ["SpecLabel", "Name", "Iteration"], kind="stable").reset_index(drop=True)

    if verbose:
        print(f"Merged {len(rows)} row(s) from {len(items)} work item(s): "
              f"{len(skipped)} below min_num_trials, {len(missing)} missing "
              f"(of which {len(errors)} recorded an error).")
        for index, message in errors[:10]:
            print(f"  error  [{index}]: {message}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more error(s)")
        if missing:
            print(f"  resubmit the missing items with "
                  f"--array={compress_indices(missing)}")
    return MergeResult(metrics_df=metrics_df, sim_cache=sim_cache,
                       missing=missing, skipped=skipped, errors=errors)


def compress_indices(indices):
    """``[0,1,2,5,7,8]`` → ``"0-2,5,7-8"`` — a Slurm ``--array`` spec."""
    indices = sorted(set(int(i) for i in indices))
    if not indices:
        return ""
    parts, start, prev = [], indices[0], indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = value
    parts.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(parts)


def write_metrics_cache(result, manifest, cache_dir, *, verbose=True):
    """Store a merged result in the format ``load_or_collect_metrics`` reads.

    The payload keys mirror that function's writer exactly — ``spec_key``,
    ``num_evaluations``, ``n_psych_fits`` and ``source_files`` are what its
    ``_cache_invalid_reason`` check consults, so a merged file is accepted as a
    hit by the notebook with no code change. The seed-0 frames go to the
    ``simcache_<name>.pkl`` sidecar that ``load_or_collect_metrics`` restores.
    """
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = manifest["cache_name"]
    cache_fp = cache_dir / f"metrics_{cache_name}.pkl"
    _atomic_dump({"metrics_df": result.metrics_df,
                  "spec_key": manifest["spec_key"],
                  "num_evaluations": manifest["num_evaluations"],
                  "n_psych_fits": manifest["n_psych_fits"],
                  "source_files": manifest["source_files"]}, cache_fp)
    written = [cache_fp]
    if verbose:
        print(f"Wrote {cache_fp}")
    if result.sim_cache:
        written.append(save_sim_cache(result.sim_cache, cache_name, cache_dir))
        if verbose:
            print(f"Wrote {written[-1]}")
    return written
