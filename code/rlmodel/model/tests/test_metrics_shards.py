"""Tests for the sharded cluster collection (``rlmodel/model/metrics_shards.py``).

The contract that matters: running every work item and merging the shards must
produce exactly what ``aggregate.collect_metrics`` would have produced serially,
in a file ``load_or_collect_metrics`` accepts as a cache hit. Everything here
therefore compares against the serial path rather than against hand-written
expectations.

Same synthetic approach as ``test_aggregate.py`` — the forward simulation and
the metric computation are monkeypatched, so no behavior data and no real fit
pickle is read (those embed a ``subject_df`` written by a newer pandas and do
not unpickle under this environment).
"""
from __future__ import annotations

import pickle
import types

import numpy as np
import pandas as pd
import pytest

from ... import metrics_runner
from ...slurm import launch_metrics
from .. import aggregate, metrics_shards
from ..aggregate import (EvalSpec, collect_metrics, load_or_collect_metrics,
                         model_key)
from ..compare import ColumnFit, ModelEntry
from ..metrics_shards import (build_workdir, compress_indices, merge_shards,
                              read_worklist, run_work_item, task_seed,
                              write_metrics_cache)
from ..mle_reeval import parse_fit_filename


_FILENAME = ("chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_"
             "Normal(0, 1)_4.8s_dt0.005.pkl")
_SPECS = (
    EvalSpec("RR+Q", model_key("RewardRate", "Q-Val (Offset)"),
             "Chi²-Noise", "green"),
    EvalSpec("RR+Q (b)", model_key("RewardRate", "Q-Val (Offset)"),
             "Chi²-Noise", "blue"),
)
_SUBJECTS = ("S1", "S2")
_TRIALS = 3_000
# S2 simulates fewer trials than this, so it is the "too small" subject.
_MIN_TRIALS = 2_500


def _payload(fun=100.0):
    return {"params_names": ["DRIFT_COEF"],
            "OptimRes": types.SimpleNamespace(x=np.array([1.0]), fun=fun),
            "include_Q": True, "include_RewardRate": True}


def _fits(subjects=_SUBJECTS):
    """A minimal ``{model_key: ModelEntry}`` index shaped like discover_fits'."""
    fid = parse_fit_filename(_FILENAME)
    entry = ModelEntry(model_key=fid.model_key, label=fid.model_label,
                       fid_example=fid)
    for s in subjects:
        entry.subjects[s] = [ColumnFit(fid=fid, payload=_payload(),
                                       filename=_FILENAME,
                                       column_label="Chi²-Noise",
                                       order_rank=100.0)]
    return {fid.model_key: entry}


def _behavior(subjects=_SUBJECTS):
    return pd.DataFrame({"Name": [s for s in subjects for _ in range(5)],
                         "DV": 0.0})


def _fake_sim_df(subject, seed, n_trials=_TRIALS):
    return pd.DataFrame({"Name": [subject] * n_trials, "seed": seed})


@pytest.fixture
def stub_sim(monkeypatch):
    """Replace the forward pass + metric computation; record every call.

    The fake metric is ``seed``-dependent so a row can be traced back to the
    trajectory that produced it, and it also samples the global ``np.random`` —
    which is what lets the determinism test see whether each task seeded it.
    """
    calls = []

    def fake_compute_sim_from_params(subject, fid, params, df_behavior,
                                     include_Q, include_RewardRate, seed=0):
        calls.append({"subject": subject, "seed": seed})
        n_trials = _TRIALS if subject != "S2" else 100
        return _fake_sim_df(subject, seed, n_trials), 1.0, {}, None

    def fake_subject_metrics(subject, fitted_df, *, n_psych_fits=None):
        seed = int(fitted_df["seed"].iloc[0])
        return {"Name": subject, "NumTrials": len(fitted_df),
                "R2_Psych": 0.5 + seed + (10 if subject == "S2" else 0),
                "RewardRateCorr": 0.25,
                # Depends on the unseeded global RNG, like psychofit's
                # multi-starts do.
                "Jitter": float(np.random.rand())}

    monkeypatch.setattr(aggregate.compare, "_compute_sim_from_params",
                        fake_compute_sim_from_params)
    monkeypatch.setattr(metrics_shards.compare, "_compute_sim_from_params",
                        fake_compute_sim_from_params)
    monkeypatch.setattr(aggregate, "subject_metrics", fake_subject_metrics)
    monkeypatch.setattr(metrics_shards, "subject_metrics", fake_subject_metrics)
    return calls


@pytest.fixture(autouse=True)
def _clear_caches():
    metrics_shards.clear_caches()
    yield
    metrics_shards.clear_caches()


def _build(tmp_path, *, num_evaluations=3, specs=_SPECS, subjects=None):
    return build_workdir(_fits(), specs, _behavior(), tmp_path / "work",
                         cache_name="fig1l", num_evaluations=num_evaluations,
                         result_dir=str(tmp_path), subjects=subjects,
                         verbose=False)


def _run_all(work_dir, **kwargs):
    kwargs.setdefault("min_num_trials", _MIN_TRIALS)
    statuses = []
    for item in read_worklist(work_dir):
        statuses.append(run_work_item(work_dir, item.index, verbose=False,
                                      **kwargs))
    return statuses


# --------------------------------------------------------------------------
# Work-dir construction
# --------------------------------------------------------------------------
def test_worklist_enumerates_every_combination(tmp_path, stub_sim):
    manifest = _build(tmp_path, num_evaluations=3)
    items = read_worklist(tmp_path / "work")

    assert len(items) == len(_SPECS) * len(_SUBJECTS) * 3 == manifest["num_items"]
    assert [i.index for i in items] == list(range(len(items)))
    assert {(i.spec_idx, i.subject, i.iteration) for i in items} == {
        (s, subj, it) for s in range(len(_SPECS))
        for subj in _SUBJECTS for it in range(3)}


def test_build_workdir_keeps_payloads_out_of_the_manifest(tmp_path, stub_sim):
    """A task must never need the 29-339 MB fit pickle — only the params."""
    manifest = _build(tmp_path)
    ctx = manifest["contexts"][(0, "S1")]

    assert ctx["params"] == {"DRIFT_COEF": 1.0}
    assert ctx["loss"] == 100.0
    assert ctx["include_Q"] is True
    assert "OptimRes" not in ctx and "payload" not in ctx


def test_build_workdir_can_restrict_subjects(tmp_path, stub_sim):
    manifest = _build(tmp_path, num_evaluations=2, subjects=["S1"])
    items = read_worklist(tmp_path / "work")

    assert {i.subject for i in items} == {"S1"}
    assert manifest["num_items"] == len(_SPECS) * 1 * 2


def test_build_workdir_rejects_unknown_subject(tmp_path, stub_sim):
    with pytest.raises(KeyError, match="resolves to none"):
        _build(tmp_path, subjects=["nobody"])


def test_work_item_index_is_the_worklist_line_number(tmp_path, stub_sim):
    """The index IS the Slurm array index, so it must survive a blank line
    rather than renumbering everything after it."""
    _build(tmp_path, num_evaluations=2)
    fp = metrics_shards.worklist_path(tmp_path / "work")
    lines = fp.read_text(encoding="utf-8").splitlines()
    lines.insert(2, "")
    fp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    metrics_shards.clear_caches()

    items = read_worklist(tmp_path / "work")
    assert [i.index for i in items] == [0, 1, 3, 4, 5, 6, 7, 8]

    # Item 6 must be the one on LINE 6, not the sixth entry in the list.
    run_work_item(tmp_path / "work", 6, min_num_trials=_MIN_TRIALS,
                  verbose=False)
    with metrics_shards.shard_path(tmp_path / "work", 6).open("rb") as f:
        shard = pickle.load(f)
    spec_idx, subject, iteration = lines[6].split("\t")
    assert shard["Name"] == subject
    assert shard["Iteration"] == int(iteration)
    assert shard["SpecLabel"] == _SPECS[int(spec_idx)].label


# --------------------------------------------------------------------------
# Equivalence with the serial path
# --------------------------------------------------------------------------
def test_merged_frame_matches_collect_metrics(tmp_path, stub_sim):
    """The whole point: sharded == serial."""
    _build(tmp_path, num_evaluations=3)
    _run_all(tmp_path / "work")
    merged = merge_shards(tmp_path / "work", verbose=False).metrics_df

    serial = collect_metrics(_fits(), _SPECS, _behavior(), num_evaluations=3,
                             min_num_trials=_MIN_TRIALS, verbose=False)
    key = ["SpecLabel", "Name", "Iteration"]
    serial = serial.sort_values(key, kind="stable").reset_index(drop=True)

    assert list(merged.columns) == list(serial.columns)
    # Jitter is RNG-derived: equal columns, not equal values (the serial path
    # doesn't seed per evaluation).
    compare_cols = [c for c in serial.columns if c != "Jitter"]
    pd.testing.assert_frame_equal(merged[compare_cols], serial[compare_cols])


def test_below_min_trials_subject_contributes_no_rows(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=3)
    statuses = _run_all(tmp_path / "work")
    result = merge_shards(tmp_path / "work", verbose=False)

    # S2 simulates 100 trials: every one of its items is skipped, and none of
    # them reaches the frame -- matching the serial loop, which breaks.
    assert statuses.count("skipped") == len(_SPECS) * 3
    assert set(result.metrics_df.Name) == {"S1"}
    assert len(result.skipped) == len(_SPECS) * 3
    assert result.missing == []


def test_below_min_trials_subject_has_no_sim_cache_entry(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=2)
    _run_all(tmp_path / "work")
    sim_cache = merge_shards(tmp_path / "work", verbose=False).sim_cache

    assert sorted(sim_cache) == ["RR+Q", "RR+Q (b)"]
    for subjects in sim_cache.values():
        assert list(subjects) == ["S1"]


# --------------------------------------------------------------------------
# sim cache
# --------------------------------------------------------------------------
def test_sim_shards_are_seed_zero_only(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=3)
    _run_all(tmp_path / "work")

    for item in read_worklist(tmp_path / "work"):
        fp = metrics_shards.sim_shard_path(tmp_path / "work", item.index)
        expected = item.iteration == 0 and item.subject == "S1"
        assert fp.exists() is expected, (item, fp)


def test_sim_cache_has_the_serial_tuple_shape(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=2)
    _run_all(tmp_path / "work")
    sim_cache = merge_shards(tmp_path / "work", verbose=False).sim_cache

    loss, sim_df, bound, include_Q, include_RewardRate = sim_cache["RR+Q"]["S1"]
    assert (loss, bound, include_Q, include_RewardRate) == (100.0, 1.0, True, True)
    assert len(sim_df) == _TRIALS
    assert (sim_df["seed"] == 0).all()


def test_no_save_sim_writes_no_sim_shards(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=2)
    _run_all(tmp_path / "work", save_sim=False)

    assert merge_shards(tmp_path / "work", verbose=False).sim_cache == {}


# --------------------------------------------------------------------------
# Determinism + idempotence
# --------------------------------------------------------------------------
def test_task_seed_is_stable_and_distinct():
    a = task_seed("RR+Q", "S1", 0)
    assert a == task_seed("RR+Q", "S1", 0)
    assert a != task_seed("RR+Q", "S1", 1)
    assert a != task_seed("RR+Q", "S2", 0)
    assert a != task_seed("other", "S1", 0)


def test_reruns_reproduce_the_same_metrics(tmp_path, stub_sim):
    """The RNG-derived metric must not depend on run order or process state."""
    _build(tmp_path, num_evaluations=2)
    _run_all(tmp_path / "work")
    first = merge_shards(tmp_path / "work", verbose=False).metrics_df

    np.random.seed(999)          # a different global state than the first pass
    _run_all(tmp_path / "work", overwrite=True)
    second = merge_shards(tmp_path / "work", verbose=False).metrics_df

    pd.testing.assert_frame_equal(first, second)
    assert first.Jitter.nunique() > 1   # the metric really is RNG-derived


def test_existing_shard_is_not_recomputed(tmp_path, stub_sim):
    """What makes a blanket resubmission only redo what is actually missing."""
    _build(tmp_path, num_evaluations=1)
    _run_all(tmp_path / "work")
    n_calls = len(stub_sim)

    statuses = _run_all(tmp_path / "work")

    # Skips write a shard too, so a below-min-trials decision is cached like any
    # other -- nothing at all re-simulates.
    assert set(statuses) == {"cached"}
    assert len(stub_sim) == n_calls


def test_overwrite_recomputes(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=1)
    _run_all(tmp_path / "work")
    n_calls = len(stub_sim)

    _run_all(tmp_path / "work", overwrite=True)

    assert len(stub_sim) == 2 * n_calls


# --------------------------------------------------------------------------
# Failure reporting
# --------------------------------------------------------------------------
def test_missing_shards_are_reported_not_silently_dropped(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=3)
    items = read_worklist(tmp_path / "work")
    for item in items:
        if not (item.spec_idx == 0 and item.iteration == 2):
            run_work_item(tmp_path / "work", item.index,
                          min_num_trials=_MIN_TRIALS, verbose=False)

    result = merge_shards(tmp_path / "work", verbose=False)

    expected = sorted(i.index for i in items
                      if i.spec_idx == 0 and i.iteration == 2)
    assert result.missing == expected


def test_simulation_error_is_recorded_and_raised(tmp_path, stub_sim, monkeypatch):
    _build(tmp_path, num_evaluations=1)
    good = metrics_shards.compare._compute_sim_from_params

    def boom(subject, fid, params, df_behavior, **kw):
        if subject == "S1":
            return None, 1.0, {}, "ValueError: nope"
        return good(subject, fid, params, df_behavior, **kw)
    monkeypatch.setattr(metrics_shards.compare, "_compute_sim_from_params", boom)

    with pytest.raises(RuntimeError, match="nope"):
        run_work_item(tmp_path / "work", 0, verbose=False)

    # The failure is retryable: it must NOT leave a shard that a resubmission
    # would treat as finished work.
    assert not metrics_shards.shard_path(tmp_path / "work", 0).exists()
    assert metrics_shards.error_path(tmp_path / "work", 0).exists()

    monkeypatch.setattr(metrics_shards.compare, "_compute_sim_from_params", good)
    _run_all(tmp_path / "work")
    result = merge_shards(tmp_path / "work", verbose=False)

    # Index 0 was retried and now has a row; the error file is history.
    assert result.missing == [] and result.errors == []


def test_error_index_is_reported_while_it_is_still_missing(tmp_path, stub_sim,
                                                           monkeypatch):
    _build(tmp_path, num_evaluations=2)
    good = metrics_shards.compare._compute_sim_from_params

    def boom(subject, fid, params, df_behavior, **kw):
        if subject == "S1" and kw.get("seed") == 0:
            return None, 1.0, {}, "ValueError: nope"
        return good(subject, fid, params, df_behavior, **kw)
    monkeypatch.setattr(metrics_shards.compare, "_compute_sim_from_params", boom)

    for item in read_worklist(tmp_path / "work"):
        try:
            run_work_item(tmp_path / "work", item.index,
                          min_num_trials=_MIN_TRIALS, verbose=False)
        except RuntimeError:
            pass

    result = merge_shards(tmp_path / "work", verbose=False)
    assert sorted(result.missing) == [0, 4]     # the two seed-0 S1 items
    assert [msg for _, msg in result.errors] == ["ValueError: nope"] * 2


def test_merging_nothing_points_at_the_job_logs(tmp_path, stub_sim):
    """Zero shards and zero errors means the array never ran — a Slurm problem,
    not a data one, so the message has to send you to the job log."""
    _build(tmp_path, num_evaluations=2)

    with pytest.raises(ValueError, match="the work never started"):
        merge_shards(tmp_path / "work", verbose=False)


def test_run_work_item_rejects_an_out_of_range_index(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=1)
    with pytest.raises(IndexError, match="out of range"):
        run_work_item(tmp_path / "work", 999, verbose=False)


@pytest.mark.parametrize("indices, expected", [
    ([], ""),
    ([4], "4"),
    ([0, 1, 2, 5, 7, 8], "0-2,5,7-8"),
    ([3, 1, 2, 1], "1-3"),
])
def test_compress_indices(indices, expected):
    assert compress_indices(indices) == expected


# --------------------------------------------------------------------------
# The merged file is what the notebook reads
# --------------------------------------------------------------------------
def test_written_cache_is_a_hit_for_load_or_collect_metrics(tmp_path, stub_sim):
    manifest = _build(tmp_path, num_evaluations=3)
    _run_all(tmp_path / "work")
    result = merge_shards(tmp_path / "work", verbose=False)
    cache_dir = tmp_path / "metrics"
    write_metrics_cache(result, manifest, cache_dir, verbose=False)

    sim_cache = {}
    loaded = load_or_collect_metrics(
        _fits(), _SPECS, _behavior(), cache_name="fig1l", num_evaluations=3,
        cache_dir=str(cache_dir), result_dir=str(tmp_path), sim_cache=sim_cache,
        verbose=False)

    # Served from disk: no simulation ran while loading it.
    n_calls = len(stub_sim)
    pd.testing.assert_frame_equal(loaded, result.metrics_df)
    assert len(stub_sim) == n_calls
    # ...and the seed-0 frames came back too, which they never did on a cache
    # hit before the sidecar existed.
    assert sorted(sim_cache) == ["RR+Q", "RR+Q (b)"]
    assert len(sim_cache["RR+Q"]["S1"][1]) == _TRIALS


def test_written_cache_serves_a_smaller_num_evaluations(tmp_path, stub_sim):
    manifest = _build(tmp_path, num_evaluations=3)
    _run_all(tmp_path / "work")
    result = merge_shards(tmp_path / "work", verbose=False)
    cache_dir = tmp_path / "metrics"
    write_metrics_cache(result, manifest, cache_dir, verbose=False)

    loaded = load_or_collect_metrics(
        _fits(), _SPECS, _behavior(), cache_name="fig1l", num_evaluations=2,
        cache_dir=str(cache_dir), result_dir=str(tmp_path), verbose=False)

    assert sorted(loaded.Iteration.unique()) == [0, 1, 2]


def test_written_cache_is_rejected_when_the_specs_change(tmp_path, stub_sim):
    manifest = _build(tmp_path, num_evaluations=1)
    _run_all(tmp_path / "work")
    result = merge_shards(tmp_path / "work", verbose=False)
    cache_dir = tmp_path / "metrics"
    write_metrics_cache(result, manifest, cache_dir, verbose=False)

    other = (_SPECS[0],)   # a different spec set under the same cache_name
    n_calls = len(stub_sim)
    load_or_collect_metrics(
        _fits(), other, _behavior(), cache_name="fig1l", num_evaluations=1,
        cache_dir=str(cache_dir), result_dir=str(tmp_path),
        min_num_trials=_MIN_TRIALS, verbose=False)

    assert len(stub_sim) > n_calls   # it recollected instead of trusting it


def test_sim_cache_sidecar_round_trips(tmp_path):
    sim_cache = {"RR+Q": {"S1": (1.0, pd.DataFrame({"a": [1, 2]}), 2.0,
                                 True, False)}}
    aggregate.save_sim_cache(sim_cache, "fig1l", tmp_path)
    restored = aggregate.load_sim_cache("fig1l", tmp_path)

    assert sorted(restored) == ["RR+Q"]
    pd.testing.assert_frame_equal(restored["RR+Q"]["S1"][1],
                                  sim_cache["RR+Q"]["S1"][1])


def test_load_sim_cache_tolerates_a_missing_or_broken_sidecar(tmp_path):
    assert aggregate.load_sim_cache("nope", tmp_path) == {}
    aggregate.sim_cache_path("broken", tmp_path).write_bytes(b"not a pickle")
    assert aggregate.load_sim_cache("broken", tmp_path) == {}


# --------------------------------------------------------------------------
# The CLI — what the .sbatch actually invokes
# --------------------------------------------------------------------------
def test_cli_run_then_merge_round_trip(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=2)
    work_dir, cache_dir = str(tmp_path / "work"), tmp_path / "metrics"

    metrics_runner.main(["--mode", "run", "--work-dir", work_dir, "--all",
                         "--min-num-trials", str(_MIN_TRIALS)])
    metrics_runner.main(["--mode", "merge", "--work-dir", work_dir,
                         "--cache-dir", str(cache_dir)])

    assert (cache_dir / "metrics_fig1l.pkl").exists()
    assert (cache_dir / "simcache_fig1l.pkl").exists()


def test_cli_run_task_range_runs_only_that_range(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=2)
    work_dir = str(tmp_path / "work")

    metrics_runner.main(["--mode", "run", "--work-dir", work_dir,
                         "--task-range", "0-1",
                         "--min-num-trials", str(_MIN_TRIALS)])

    done = [i for i in range(8)
            if metrics_shards.shard_path(tmp_path / "work", i).exists()]
    assert done == [0, 1]


def test_cli_run_task_range_tolerates_a_tail_past_the_worklist(tmp_path,
                                                               stub_sim):
    """A fixed-size block of items overshoots the end of the worklist whenever
    the block size doesn't divide it — the last Slurm task of every run."""
    _build(tmp_path, num_evaluations=2)     # 8 items, indices 0-7
    work_dir = str(tmp_path / "work")

    metrics_runner.main(["--mode", "run", "--work-dir", work_dir,
                         "--task-range", "6-15",
                         "--min-num-trials", str(_MIN_TRIALS)])

    done = [i for i in range(8)
            if metrics_shards.shard_path(tmp_path / "work", i).exists()]
    assert done == [6, 7]


def test_cli_run_task_range_entirely_past_the_end_is_a_no_op(tmp_path,
                                                             stub_sim):
    _build(tmp_path, num_evaluations=2)
    metrics_runner.main(["--mode", "run", "--work-dir", str(tmp_path / "work"),
                         "--task-range", "100-109"])

    assert stub_sim == []


def test_cli_merge_refuses_an_incomplete_frame(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=2)
    work_dir, cache_dir = str(tmp_path / "work"), tmp_path / "metrics"
    metrics_runner.main(["--mode", "run", "--work-dir", work_dir,
                         "--task-range", "0-3",
                         "--min-num-trials", str(_MIN_TRIALS)])

    with pytest.raises(SystemExit, match="no shard"):
        metrics_runner.main(["--mode", "merge", "--work-dir", work_dir,
                             "--cache-dir", str(cache_dir)])
    assert not (cache_dir / "metrics_fig1l.pkl").exists()

    metrics_runner.main(["--mode", "merge", "--work-dir", work_dir,
                         "--cache-dir", str(cache_dir), "--allow-missing"])
    assert (cache_dir / "metrics_fig1l.pkl").exists()


def test_cli_run_needs_exactly_one_selector(tmp_path, stub_sim):
    _build(tmp_path, num_evaluations=1)
    work_dir = str(tmp_path / "work")
    with pytest.raises(SystemExit, match="exactly one"):
        metrics_runner.main(["--mode", "run", "--work-dir", work_dir])
    with pytest.raises(SystemExit, match="exactly one"):
        metrics_runner.main(["--mode", "run", "--work-dir", work_dir,
                             "--task-id", "0", "--all"])


def test_figure_presets_cover_the_notebook_cache_names():
    """The keys are the cache_name slugs model_analysis.ipynb passes to
    load_or_collect_metrics — a merged file has to land where it looks."""
    assert sorted(metrics_runner.FIGURES) == [
        "drift_rr", "fig1l", "mle_weights", "scale_bound"]
    assert metrics_runner.FIGURES["fig1l"] is aggregate.FIG1L_SPECS


# --------------------------------------------------------------------------
# Array windowing (the MaxArraySize workaround)
# --------------------------------------------------------------------------
def test_array_windows_cover_every_index_once():
    windows = launch_metrics._array_windows(2_500, 1_000)

    assert windows == [(0, "0-999"), (1_000, "0-999"), (2_000, "0-499")]
    covered = sorted(offset + i for offset, spec in windows
                     for i in range(int(spec.split("-")[1]) + 1))
    assert covered == list(range(2_500))


def test_array_windows_fit_in_one_job_when_under_the_cap():
    assert launch_metrics._array_windows(50, 1_000) == [(0, "0-49")]


def test_array_windows_translate_an_explicit_resubmission_spec():
    """The indices merge prints are worklist indices, so each has to be shifted
    into its window before it can be an --array index."""
    windows = launch_metrics._array_windows(2_500, 1_000, "3,1005-1007,2400")

    assert windows == [(0, "3"), (1_000, "5-7"), (2_000, "400")]


def test_items_per_task_shrinks_the_number_of_array_tasks():
    """The lever for a cluster that caps QUEUED jobs: same work, fewer jobs."""
    windows = launch_metrics._array_windows(15_900, 1_000, items_per_task=20)

    # 15900/20 = 795 tasks, so it now fits in a single array job.
    assert windows == [(0, "0-794")]
    # ...and task i covers items [20i, 20i+20).
    assert windows[0][0] == 0


def test_items_per_task_still_windows_around_max_array_size():
    windows = launch_metrics._array_windows(15_900, 1_000, items_per_task=5)

    # 3180 tasks -> 4 arrays; each window starts where the previous ended.
    assert [offset for offset, _ in windows] == [0, 5_000, 10_000, 15_000]
    assert [spec for _, spec in windows] == ["0-999", "0-999", "0-999", "0-179"]


def test_items_per_task_covers_a_ragged_tail():
    """The last task overshoots the worklist; that tail is a no-op, not a gap."""
    windows = launch_metrics._array_windows(25, 1_000, items_per_task=10)

    assert windows == [(0, "0-2")]     # 3 tasks: items 0-9, 10-19, 20-29


def test_items_per_task_applies_to_a_contiguous_resubmission():
    windows = launch_metrics._array_windows(15_900, 1_000, "9009-15899",
                                            items_per_task=20)

    assert windows == [(9_009, "0-344")]   # 6891 items / 20 = 345 tasks


def test_items_per_task_is_ignored_for_a_scattered_resubmission(capsys):
    """An array task walks CONSECUTIVE items, so it can't pack arbitrary ones."""
    windows = launch_metrics._array_windows(2_500, 1_000, "3,1005,2400",
                                            items_per_task=20)

    assert windows == [(0, "3"), (1_000, "5"), (2_000, "400")]
    assert "ignored" in capsys.readouterr().out


def _fake_conda_install(tmp_path):
    base = tmp_path / "miniconda3"
    (base / "etc" / "profile.d").mkdir(parents=True)
    (base / "etc" / "profile.d" / "conda.sh").write_text("")
    (base / "bin").mkdir()
    (base / "bin" / "conda").write_text("")
    return base


def test_conda_base_found_from_conda_exe(tmp_path, monkeypatch):
    """A compute node can't call conda, so the submitting shell has to tell it
    where conda lives — CONDA_EXE is a plain var and does survive into a job."""
    base = _fake_conda_install(tmp_path)
    monkeypatch.setenv("CONDA_EXE", str(base / "bin" / "conda"))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    assert launch_metrics.conda_base() == base


def test_conda_base_found_from_an_activated_env_prefix(tmp_path, monkeypatch):
    base = _fake_conda_install(tmp_path)
    (base / "envs" / "py314").mkdir(parents=True)
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(base / "envs" / "py314"))

    assert launch_metrics.conda_base() == base


def test_conda_base_is_none_when_undetectable(tmp_path, monkeypatch):
    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "not-conda"))

    assert launch_metrics.conda_base() is None


def test_max_array_size_falls_back_without_slurm(monkeypatch):
    def no_slurm(*a, **kw):
        raise OSError("scontrol not found")
    monkeypatch.setattr(launch_metrics.subprocess, "run", no_slurm)

    assert launch_metrics.max_array_size() == launch_metrics.DEFAULT_MAX_ARRAY_SIZE
    assert launch_metrics.max_array_size(7) == 7
