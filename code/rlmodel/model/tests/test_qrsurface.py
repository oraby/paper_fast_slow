"""Tests for Figure 7D's surface builder.

The three things worth pinning: the binning does not wrap (the bug that drew
reward-rate-0 trials at 1.0), the nudge is off unless asked for and touches
only the side the update wrote, and the parallel path gives the same grid
whatever the worker count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .. import logic, qrsurface, state_updates
from ..bias import _biasQVal
from ..drift import _noiseGainRewardRate
from ..noise import _noiseNormal


# --- binning ------------------------------------------------------------------

def test_bin_index_never_wraps_onto_the_last_bin():
    edges = qrsurface.RR_EDGES
    # digitize(..., right=True) - 1 gives -1 here, which indexes the last bin.
    assert qrsurface.binIndex(0.0, edges) == 0
    assert qrsurface.binIndex(-0.5, edges) == 0
    assert qrsurface.binIndex(1.0, edges) == edges.size - 2
    assert qrsurface.binIndex(2.0, edges) == edges.size - 2


def test_bin_index_spans_exactly_the_bins():
    values = np.linspace(-1, 1, 501)
    idx = qrsurface.binIndex(values, qrsurface.Q_EDGES)
    assert idx.min() == 0
    assert idx.max() == qrsurface.Q_EDGES.size - 2
    assert qrsurface.SHAPE == (10, 20, 3)


def test_bin_centres_are_the_middle_not_the_edge():
    centres = qrsurface.binCentres(qrsurface.RR_EDGES)
    assert centres[0] == pytest.approx(0.05)
    assert centres[-1] == pytest.approx(0.95)
    assert len(centres) == qrsurface.SHAPE[0]


def test_q_relative_rotates_onto_the_stimulus_side():
    q = np.array([0.4, 0.4, -0.4, -0.4, 0.4])
    dv = np.array([1.0, -1.0, 1.0, -1.0, 0.0])
    got = qrsurface.qRelative(q, dv)
    # positive when the bias favours the stimulus side, negative against it,
    # and the raw value when there is no stimulus side.
    assert got == pytest.approx([0.4, -0.4, -0.4, 0.4, 0.4])


# --- the nudge ----------------------------------------------------------------

def test_nudge_touches_only_the_side_the_update_wrote():
    rng = np.random.default_rng(0)
    q_left = np.array([0.5, 0.5])
    q_right = np.array([0.5, 0.5])
    chose_left = np.array([1.0, 0.0])
    left, right = state_updates.nudge_q_values(q_left, q_right, chose_left, 0.1, rng)
    assert left[0] != q_left[0] and left[1] == q_left[1]
    assert right[1] != q_right[1] and right[0] == q_right[0]


def test_nudge_stays_inside_the_unit_interval():
    rng = np.random.default_rng(0)
    q = np.full(500, 0.99)
    left, _ = state_updates.nudge_q_values(q, q, np.ones(500), 0.5, rng)
    assert left.max() <= 1.0 and left.min() >= 0.0
    rate = state_updates.nudge_reward_rate(np.full(500, 0.01), 0.5,
                                           np.random.default_rng(0))
    assert rate.max() <= 1.0 and rate.min() >= 0.0


def test_a_zero_nudge_leaves_the_simulation_exactly_as_fitted(tiny_fit):
    subject, fit = tiny_fit
    plain = qrsurface.runSubject(subject, fit, resample_count=1, nudge_sd=0.0)
    again = qrsurface.runSubject(subject, fit, resample_count=1, nudge_sd=0.0)
    for a, b in zip(plain[:3], again[:3]):
        assert np.array_equal(a, b)
    nudged = qrsurface.runSubject(subject, fit, resample_count=1, nudge_sd=0.05)
    assert not np.array_equal(plain[1], nudged[1]), "a nudge should change the surface"


def test_makeOneRun_defaults_to_no_nudge(monkeypatch):
    called = {}
    original = state_updates.nudge_q_values

    def spy(*args, **kwargs):
        called["yes"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(state_updates, "nudge_q_values", spy)
    import inspect
    assert inspect.signature(logic.makeOneRun).parameters["latent_nudge_sd"].default == 0.0
    assert not called


# --- aggregation and parallelism ----------------------------------------------

def test_balanced_weights_subjects_not_trials():
    counts_a = np.zeros(qrsurface.SHAPE); counts_a[0, 0, 0] = 100
    total_a = np.zeros(qrsurface.SHAPE); total_a[0, 0, 0] = 100.0     # mean 1.0
    counts_b = np.zeros(qrsurface.SHAPE); counts_b[0, 0, 0] = 2
    total_b = np.zeros(qrsurface.SHAPE); total_b[0, 0, 0] = 6.0       # mean 3.0
    grids = qrsurface.combine([(counts_a, total_a, total_a),
                               (counts_b, total_b, total_b * 3)])
    # 102 trials averaging 1.04 against two subjects averaging 2.0
    assert grids["pooled"][0, 0, 0] == pytest.approx(106 / 102)
    assert grids["balanced"][0, 0, 0] == pytest.approx(2.0)
    assert grids["subjects_per_facet"][0, 0, 0] == 2


def test_the_grid_does_not_depend_on_the_worker_count(tmp_path, tiny_fit_file):
    one = qrsurface.simulate(tiny_fit_file, resample_count=1, nudge_sd=0.01,
                             workers=1)
    two = qrsurface.simulate(tiny_fit_file, resample_count=1, nudge_sd=0.01,
                             workers=2)
    assert one["subjects"] == two["subjects"]
    assert one["trials"] == two["trials"]
    for key in ("pooled", "balanced", "counts"):
        assert np.array_equal(np.nan_to_num(one[key], nan=-999),
                              np.nan_to_num(two[key], nan=-999)), key


# --- a small synthetic fit ------------------------------------------------------

def _subject_frame(name, sessions=2, trials=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sess in range(sessions):
        dv = rng.uniform(-1, 1, size=trials)
        rows.append(pd.DataFrame({
            "DVabs": np.abs(dv), "DV": dv.astype(np.float32),
            "DVstr": np.where(np.abs(dv) > 0.5, "Easy", "Hard"),
            "valid": True, "calcStimulusTime": rng.uniform(0.2, 2.0, trials),
            "ChoiceCorrect": rng.integers(0, 2, trials).astype(float),
            "ChoiceLeft": (dv > 0).astype(float), "Name": name,
            "Date": "2020-01-0%d" % (sess + 1), "SessionNum": sess + 1,
            "TrialNumber": np.arange(1, trials + 1),
            "SessId": f"{name}_{sess}", "SimRT": np.nan,
            "SimStartingPoint": np.nan, "SimChoiceCorrect": np.nan,
            "SimChoiceLeft": np.nan, "Q_L": 0.5, "Q_R": 0.5, "Q_val": 0.0,
            "RewardRate": 0.5}))
    return pd.concat(rows, ignore_index=True)


def _fit_payload(name, seed=0):
    fixed = {"df": _subject_frame(name, seed=seed),
             "biasFn": _biasQVal, "biasFn_df_cols": ["Q_val"],
             "biasFn_kwargs": ["BIAS_COEF", "Q_VAL_OFFSET"],
             "driftFn": _noiseGainRewardRate, "driftFn_df_cols": ["RewardRate"],
             "driftFn_kwargs": [], "noiseFn": _noiseNormal,
             "noiseFn_df_cols": [], "noiseFn_kwargs": [],
             "include_Q": True, "include_RewardRate": True,
             "dt": 0.01, "t_dur": 1.0, "return_df": True, "is_loss_no_dir": False}
    params = {"BOUND": 1.0, "DRIFT_COEF": 4.0, "NOISE_SIGMA": 1.0,
              "Q_VAL_OFFSET": 0.0, "NON_DECISION_TIME": 0.1, "BIAS_COEF": 0.5,
              "ALPHA": 0.8, "BETA": 0.4}
    return {"fixed_params_names": np.asarray(list(fixed)),
            "fixed_params_vals": np.asarray(list(fixed.values()), dtype=object),
            "params_names": np.asarray(list(params)),
            "OptimRes": {"x": np.asarray(list(params.values()), dtype=float)}}


@pytest.fixture
def tiny_fit():
    return "mouse_a", _fit_payload("mouse_a")


@pytest.fixture
def tiny_fit_file(tmp_path):
    """A two-subject fit on disk, so the parallel path can load it per worker."""
    from ..fitio import saveFit
    path = tmp_path / "tiny_fit_1.0s_dt0.01.pkl"
    saveFit({"mouse_a": _fit_payload("mouse_a", seed=1),
             "mouse_b": _fit_payload("mouse_b", seed=2)}, path)
    return path
