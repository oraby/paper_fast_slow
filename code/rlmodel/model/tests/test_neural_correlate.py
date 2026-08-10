"""Tests for the neural_correlate engine (``rlmodel/model/neural_correlate.py``).

Pure-logic + headless (Agg) coverage: the per-(neuron, trial) table build
(np.nanmax windowing, long_trace_id, MFC/LFC mapping, identity-key join,
require_valid, per-subject max-loss exclusion), the Pearson correlation, the
parameter-availability gating, and the save-mode plot path builder. No real
pickles, behavior data, or MLE simulation are needed.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors

from .. import neural_correlate as nc


# A real parseable MLE fit filename (drift/bias/noise with spaces + parens).
_FIT_FNAME = ("mle_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)"
              "_4.8s_dt0.005.pkl")


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------
def _trace(length, window_slice, peak_pos, peak_val):
    """A trace of ``length`` whose max inside ``window_slice`` is ``peak_val``
    at ``peak_pos`` (kept below ``peak_val`` outside so the window governs)."""
    arr = np.full(length, -5.0, dtype=float)
    arr[window_slice] = 0.0
    arr[peak_pos] = peak_val
    return arr


def _twop_row(name, date, sess, trial, region, short, neuronal,
              s=2, e=6, dv=0.5, quantile=1):
    return {
        "Name": name, "Date": date, "SessionNum": sess, "TrialNumber": trial,
        "BrainRegion": int(region), "ShortName": short,
        "trace_start_idx": s, "trace_end_idx": e,
        "traces_sets": {"neuronal": neuronal},
        # sampling start = 2, movement start = 7 (matches the length-10 trace).
        "epochs_ranges": [(0, 1), (2, 6), (7, 9)],
        "epochs_names": ["-0.1s Sampling", "Sampling", "Movement to Lateral Port"],
        "DV": dv, "quantile_idx": quantile,   # DVabs derived downstream
    }


def _make_2p(region=nc.BrainRegion.M2_Bi, short="S1", name="rat",
             date="2025-01-01", sess=1, n_trials=4, activities=None,
             dvs=None, quantiles=None):
    """One session, two neurons (n1, n2), ``n_trials`` trials. ``activities`` is
    a dict {trace_id: [per-trial window-max]}. Window is indices 2..6. ``dvs`` /
    ``quantiles`` give per-trial DV and quantile_idx (fast=1, slow=3)."""
    if activities is None:
        activities = {"n1": [1.0, 2.0, 3.0, 4.0],
                      "n2": [4.0, 3.0, 2.0, 1.0]}
    if dvs is None:
        dvs = [0.5] * n_trials
    if quantiles is None:
        quantiles = [1] * n_trials
    rows = []
    for t in range(n_trials):
        neuronal = {tid: _trace(10, slice(2, 7), 4, activities[tid][t])
                    for tid in activities}
        rows.append(_twop_row(name, date, sess, t + 1, region, short, neuronal,
                              dv=dvs[t], quantile=quantiles[t]))
    return pd.DataFrame(rows)


def _make_mle_pt(name="rat", date="2025-01-01", sess=1, n_trials=4,
                 q_val=None, loglik=None, valid=None, reward_rate=None):
    q_val = q_val if q_val is not None else [0.1, 0.2, 0.3, 0.4]
    loglik = loglik if loglik is not None else [-1.0, -2.0, -3.0, -1.5]
    valid = valid if valid is not None else [True] * n_trials
    reward_rate = reward_rate if reward_rate is not None else [0.5] * n_trials
    return pd.DataFrame({
        "Name": [name] * n_trials,
        "Date": [pd.Timestamp(date)] * n_trials,
        "SessionNum": [sess] * n_trials,
        "TrialNumber": list(range(1, n_trials + 1)),
        "Q_L": q_val,
        "Q_R": [0.0] * n_trials,
        "Q_val": q_val,
        "RewardRate": reward_rate,
        "mle_loglik": loglik,
        "mle_valid": valid,
        "model_label": ["m"] * n_trials,
        "model_name": ["m"] * n_trials,
    })


PARAM_KEYS = ["Q_L", "Q_R", "Q_val", "RewardRate"]


def _raw_mle_df(name, n_trials=4):
    """A payload-style ``mle_df`` (the raw ``mle_*`` column names as stored in a
    fit pickle) for one subject."""
    return pd.DataFrame({
        "Name": [name] * n_trials,
        "Date": [pd.Timestamp("2025-01-01")] * n_trials,
        "SessionNum": [1] * n_trials,
        "TrialNumber": list(range(1, n_trials + 1)),
        "mle_Q_left_before": np.linspace(0.1, 0.4, n_trials),
        "mle_Q_right_before": [0.0] * n_trials,
        "mle_Q_rel_before": np.linspace(0.1, 0.4, n_trials),
        "mle_reward_rate_before": [0.5] * n_trials,
        "mle_loglik": [-1.0, -2.0, -3.0, -1.5][:n_trials],
        "mle_valid_for_loss": [True] * n_trials,
    })


def _write_fit_pickle(path, subjects, *, include_Q=True, include_RewardRate=True,
                      with_mle_df=True):
    """Write a minimal ``{subject: payload}`` fit pickle to ``path``."""
    data = {}
    for subj in subjects:
        payload = {"include_Q": include_Q,
                   "include_RewardRate": include_RewardRate,
                   "params_names": np.array(["DRIFT_COEF"]),
                   "params_init": np.array([1.0]),
                   "OptimRes": None}
        if with_mle_df:
            payload["mle_df"] = _raw_mle_df(subj)
        data[subj] = payload
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


# --------------------------------------------------------------------------
# available_params (principle 6)
# --------------------------------------------------------------------------
def test_available_params_gates_by_include_flags():
    assert [s.key for s in nc.available_params(True, True)] == \
        ["Q_L", "Q_R", "Q_val", "RewardRate"]
    assert [s.key for s in nc.available_params(True, False)] == \
        ["Q_L", "Q_R", "Q_val"]
    assert [s.key for s in nc.available_params(False, True)] == ["RewardRate"]
    assert nc.available_params(False, False) == []


# --------------------------------------------------------------------------
# build_neuron_trial_table
# --------------------------------------------------------------------------
def test_build_table_max_activity_and_identity():
    df_2p = _make_2p()
    mle_pt = _make_mle_pt()
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)

    # 2 neurons x 4 trials = 8 rows
    assert len(table) == 8
    assert set(table.trace_id.unique()) == {"n1", "n2"}
    # long name + region mapping (M2_Bi -> MFC)
    assert set(table.long_trace_id.unique()) == {"S1_n1", "S1_n2"}
    assert (table.BrainRegion == "MFC").all()

    # np.nanmax over the window == the per-trial peak we planted
    n1 = table[table.trace_id == "n1"].sort_values("TrialNumber")
    assert list(n1.max_activity) == [1.0, 2.0, 3.0, 4.0]
    # latents broadcast onto each neuron's trial
    assert list(n1.Q_val) == [0.1, 0.2, 0.3, 0.4]
    # full trace kept as an object of the original (length-10) trace
    assert len(n1.iloc[0].trace) == 10


def test_build_table_region_lfc_mapping():
    df_2p = _make_2p(region=nc.BrainRegion.ALM_Bi)
    table = nc.build_neuron_trial_table(df_2p, _make_mle_pt(), PARAM_KEYS)
    assert (table.BrainRegion == "LFC").all()


def test_build_table_inner_join_drops_unmatched_trials():
    # 2p has 4 trials; MLE only covers trials 1..2 -> only those survive.
    df_2p = _make_2p(n_trials=4)
    mle_pt = _make_mle_pt(n_trials=2, q_val=[0.1, 0.2], loglik=[-1.0, -2.0],
                          valid=[True, True])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    assert set(table.TrialNumber.unique()) == {1, 2}
    assert len(table) == 4  # 2 neurons x 2 trials


def test_build_table_require_valid_drops_invalid_mle_trials():
    df_2p = _make_2p(n_trials=4)
    mle_pt = _make_mle_pt(valid=[True, False, True, True])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS,
                                        require_valid=True)
    assert set(table.TrialNumber.unique()) == {1, 3, 4}


def test_build_table_only_available_param_columns():
    df_2p = _make_2p()
    mle_pt = _make_mle_pt()
    table = nc.build_neuron_trial_table(df_2p, mle_pt, ["RewardRate"])
    for absent in ("Q_L", "Q_R", "Q_val"):
        assert absent not in table.columns
    assert "RewardRate" in table.columns


# --------------------------------------------------------------------------
# per-subject max-loss exclusion (principle 7)
# --------------------------------------------------------------------------
def test_exclude_max_loss_drops_worst_trial_per_subject():
    df_2p = _make_2p(n_trials=4)
    # trial 3 has the worst (most negative) loglik = highest loss.
    mle_pt = _make_mle_pt(loglik=[-1.0, -2.0, -9.0, -1.5])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS,
                                        exclude_max_loss=True)
    assert 3 not in set(table.TrialNumber.unique())
    assert set(table.TrialNumber.unique()) == {1, 2, 4}


def test_exclude_max_loss_ignores_invalid_when_finding_max():
    df_2p = _make_2p(n_trials=4)
    # trial 2 is invalid AND has the worst loglik; the max-loss among *valid*
    # trials is trial 3 (-4.0), so trial 3 is the one excluded, not 2.
    mle_pt = _make_mle_pt(loglik=[-1.0, -99.0, -4.0, -1.5],
                          valid=[True, False, True, True])
    excluded = nc._max_loss_excluded_keys(mle_pt)
    trials = {k[3] for k in excluded}
    assert trials == {3}


# --------------------------------------------------------------------------
# compute_neuron_correlations (principle 8)
# --------------------------------------------------------------------------
def test_correlation_perfect_linear():
    # neuron n1: activity == Q_val (r = +1); neuron n2: activity == -Q_val (-1)
    df_2p = _make_2p(activities={"n1": [1.0, 2.0, 3.0, 4.0],
                                 "n2": [4.0, 3.0, 2.0, 1.0]})
    mle_pt = _make_mle_pt(q_val=[1.0, 2.0, 3.0, 4.0])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)

    assert set(corr.long_trace_id) == {"S1_n1", "S1_n2"}
    r_n1 = corr.loc[corr.long_trace_id == "S1_n1", "Q_val_r"].iloc[0]
    r_n2 = corr.loc[corr.long_trace_id == "S1_n2", "Q_val_r"].iloc[0]
    assert np.isclose(r_n1, 1.0)
    assert np.isclose(r_n2, -1.0)
    # carries identity + n_trials
    assert corr.loc[corr.long_trace_id == "S1_n1", "BrainRegion"].iloc[0] == "MFC"
    assert corr.loc[corr.long_trace_id == "S1_n1", "n_trials"].iloc[0] == 4


def test_correlation_constant_param_is_nan():
    df_2p = _make_2p()
    mle_pt = _make_mle_pt()
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)
    # Q_R is constant 0.0 -> undefined correlation
    assert corr["Q_R_r"].isna().all()


# --------------------------------------------------------------------------
# save-mode plot path builder (principle 9)
# --------------------------------------------------------------------------
def test_plot_save_mode_writes_signed_named_files(tmp_path):
    df_2p = _make_2p(activities={"n1": [1.0, 2.0, 3.0, 4.0],
                                 "n2": [4.0, 3.0, 2.0, 1.0]})
    mle_pt = _make_mle_pt(q_val=[1.0, 2.0, 3.0, 4.0])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)

    out_dir = nc.plot_neuron_results(
        table, corr, "Q_val", mode="save",
        save_root=str(tmp_path), model_name="MyModel", ext="svg")

    files = sorted(p.name for p in out_dir.glob("*.svg"))
    assert len(files) == 2
    # signed correlation prefixes: +1.000 and -1.000
    assert any(f.startswith("+1.000_S1_n1") for f in files)
    assert any(f.startswith("-1.000_S1_n2") for f in files)
    # region appended because long name ("S1_n1") lacks "MFC"
    assert all(f.endswith("_MFC.svg") for f in files)
    # written under {save_root}/{model_name}/traces/{param}/
    assert out_dir.parts[-3:] == ("MyModel", "traces", "Q_val")


def test_region_bars_runs_headless(tmp_path):
    # Two sessions so the session-mean ± SEM aggregation has >1 session.
    df_2p = pd.concat([
        _make_2p(short="S1", sess=1,
                 activities={"n1": [1.0, 2.0, 3.0, 4.0],
                             "n2": [4.0, 3.0, 2.0, 1.0]}),
        _make_2p(short="S2", sess=2,
                 activities={"n3": [1.0, 2.0, 3.0, 4.0],
                             "n4": [2.0, 1.0, 4.0, 3.0]}),
    ], ignore_index=True)
    mle_pt = pd.concat([
        _make_mle_pt(sess=1, q_val=[1.0, 2.0, 3.0, 4.0]),
        _make_mle_pt(sess=2, q_val=[1.0, 2.0, 3.0, 4.0]),
    ], ignore_index=True)
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)

    summary = nc.plot_region_bars(
        corr, table, min_abs_corr=0.5, n_shuffles=50, by_region=False,
        seed=0, save=True, save_root=str(tmp_path), model_name="MyModel")

    # one row per (region × parameter); combined -> 1 region × 4 params
    assert set(summary.param) == set(PARAM_KEYS)
    assert (summary.BrainRegion == "MFC & LFC").all()
    # observed n1/n3 are perfectly Q_val-tuned -> observed% far above shuffle%
    q = summary[summary.param == "Q_val"].iloc[0]
    assert q.observed_pct > q.shuffle_pct
    assert 0.0 <= q.p_vs_shuffle <= 1.0
    saved = tmp_path / "MyModel" / "tuning_bars"
    assert (saved / "bars_combined.svg").exists()
    assert (saved / "summary_combined.csv").exists()


def test_shuffle_null_tuned_shapes_and_assessable():
    df_2p = _make_2p(activities={"n1": [1.0, 2.0, 3.0, 4.0],
                                 "n2": [4.0, 3.0, 2.0, 1.0]})
    mle_pt = _make_mle_pt(q_val=[1.0, 2.0, 3.0, 4.0])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    rng = np.random.default_rng(0)
    meta, null, assess = nc._shuffle_null_tuned(
        table, ["Q_val", "Q_R"], min_abs_corr=0.5, n_shuffles=30, rng=rng)
    assert len(meta) == 2                     # two neurons
    assert null["Q_val"].shape == (2, 30)     # (neurons, shuffles)
    assert assess["Q_val"].all()              # Q_val varies -> assessable
    assert not assess["Q_R"].any()            # Q_R constant -> not assessable


# --------------------------------------------------------------------------
# Factor bars: group params into one "modulation" axis (OR, dedup by neuron)
# --------------------------------------------------------------------------
def test_default_factors_trims_and_drops_absent():
    # Full corr (Q + RR + DV) -> all three factors.
    df_2p = _make_2p(dvs=[-1.0, -0.3, 0.3, 1.0])
    table = nc.build_neuron_trial_table(df_2p, _make_mle_pt(), PARAM_KEYS)
    corr_full = nc.compute_neuron_correlations(table, PARAM_KEYS + ["DV"])
    assert [f.name for f in nc.default_factors(corr_full)] == \
        ["Q", "RewardRate", "DV"]
    # No DV column -> DV factor dropped; Q keeps only present params.
    corr_noDV = nc.compute_neuron_correlations(table, PARAM_KEYS)
    facs = nc.default_factors(corr_noDV)
    assert [f.name for f in facs] == ["Q", "RewardRate"]
    assert nc.PARAM_BY_KEY  # sanity
    q = next(f for f in facs if f.name == "Q")
    assert q.param_keys == ("Q_L", "Q_R", "Q_val")


def test_factor_masks_or_counts_each_neuron_once():
    # n1 is perfectly linear with q_val (== Q_L == Q_val in the fixture), so it
    # is tuned to BOTH Q_L and Q_val; the Q factor must still count it ONCE.
    # n2 (alternating) has |r| ~ 0.45 < 0.5 -> not tuned.
    df_2p = _make_2p(activities={"n1": [1.0, 2.0, 3.0, 4.0],
                                 "n2": [2.0, 1.0, 2.0, 1.0]})
    table = nc.build_neuron_trial_table(df_2p, _make_mle_pt(q_val=[1, 2, 3, 4]),
                                        PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)
    assessable, tuned = nc._factor_masks(corr, ("Q_L", "Q_R", "Q_val"), 0.5)
    assert int(tuned.sum()) == 1                     # n1 once, not twice
    # 2 neurons, 1 modulated -> 50% for the single session
    pct = nc._session_factor_percent(corr, ("Q_L", "Q_R", "Q_val"), 0.5)
    assert pct.iloc[0] == 50.0


def test_plot_factor_bars_runs_headless(tmp_path):
    df_2p = pd.concat([
        _make_2p(short="S1", sess=1, dvs=[-1.0, -0.3, 0.3, 1.0],
                 activities={"n1": [1.0, 2.0, 3.0, 4.0],
                             "n2": [2.0, 1.0, 2.0, 1.0]}),
        _make_2p(short="S2", sess=2, dvs=[-1.0, -0.3, 0.3, 1.0],
                 activities={"n3": [1.0, 2.0, 3.0, 4.0],
                             "n4": [2.0, 1.0, 2.0, 1.0]}),
    ], ignore_index=True)
    mle_pt = pd.concat([_make_mle_pt(sess=1, q_val=[1, 2, 3, 4]),
                        _make_mle_pt(sess=2, q_val=[1, 2, 3, 4])],
                       ignore_index=True)
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS + ["DV"])

    summary = nc.plot_factor_bars(
        corr, table, min_abs_corr=0.5, n_shuffles=30, by_region=False,
        seed=0, save=True, save_root=str(tmp_path), model_name="MyModel")

    assert set(summary.factor) == {"Q", "RewardRate", "DV"}
    assert (summary.BrainRegion == "MFC & LFC").all()
    q = summary[summary.factor == "Q"].iloc[0]
    assert q.observed_pct > 0 and 0.0 <= q.p_vs_shuffle <= 1.0
    assert q.params == "Q_L+Q_R+Q_val"
    saved = tmp_path / "MyModel" / "factor_bars"
    assert (saved / "bars_Q_RewardRate_DV_combined.svg").exists()
    assert (saved / "summary_Q_RewardRate_DV_combined.csv").exists()


def test_plot_factor_bars_fastslow_dv(tmp_path):
    # 6 trials, fast=first tercile (q=1), slow=last (q=3), >=3 trials each so the
    # within-subset DV correlation is defined.
    q = [1, 1, 1, 3, 3, 3]
    dvs = [-1.0, -0.3, 0.6, -0.8, 0.2, 1.0]
    df_2p = pd.concat([
        _make_2p(short="S1", sess=1, n_trials=6, quantiles=q, dvs=dvs,
                 activities={"n1": [1., 2., 3., 4., 5., 6.],
                             "n2": [2., 1., 2., 1., 2., 1.]}),
        _make_2p(short="S2", sess=2, n_trials=6, quantiles=q, dvs=dvs,
                 activities={"n3": [6., 5., 4., 3., 2., 1.],
                             "n4": [1., 2., 1., 2., 1., 2.]}),
    ], ignore_index=True)
    mle_pt = pd.concat([
        _make_mle_pt(sess=1, n_trials=6, q_val=[1, 2, 3, 4, 5, 6],
                     loglik=[-1.] * 6, valid=[True] * 6),
        _make_mle_pt(sess=2, n_trials=6, q_val=[1, 2, 3, 4, 5, 6],
                     loglik=[-1.] * 6, valid=[True] * 6),
    ], ignore_index=True)
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr_all = nc.compute_neuron_correlations(table, PARAM_KEYS + ["DV"])

    summary = nc.plot_factor_bars_fastslow_dv(
        corr_all, table, min_abs_corr=0.5, n_shuffles=30, by_region=False,
        seed=0, save=True, save_root=str(tmp_path), model_name="MyModel")

    # Q + Reward-Rate + DV split into fast / slow = four bars.
    assert list(summary.factor) == ["Q", "RewardRate", "DV_fast", "DV_slow"]
    saved = tmp_path / "MyModel" / "factor_bars"
    assert (saved / "bars_fastslowDV_combined.svg").exists()
    assert (saved / "summary_fastslowDV_combined.csv").exists()


# --------------------------------------------------------------------------
# load_mle_per_trial: subject filter + reuse-stored-df (no recompute)
# --------------------------------------------------------------------------
def test_load_mle_reuse_and_subject_filter(tmp_path):
    fp = _write_fit_pickle(tmp_path / _FIT_FNAME, ["subjA", "subjB", "subjC"])
    # Only subjA/subjB have imaging -> load just those; reuse stored mle_df so
    # df_behavior is not needed.
    mle_pt, keys = nc.load_mle_per_trial(
        fp, subjects=["subjA", "subjB"], reuse_fit_df=True, verbose=False)

    assert keys == ["Q_L", "Q_R", "Q_val", "RewardRate"]
    assert set(mle_pt.Name.unique()) == {"subjA", "subjB"}  # subjC excluded
    # raw mle_* columns renamed to the friendly latent columns
    for col in ("Q_L", "Q_R", "Q_val", "RewardRate", "mle_valid", "model_name"):
        assert col in mle_pt.columns
    assert mle_pt["mle_valid"].all()
    # before-trial relative Q surfaced as Q_val
    a = mle_pt[mle_pt.Name == "subjA"].sort_values("TrialNumber")
    assert np.isclose(a.Q_val.iloc[0], 0.1)


def test_load_mle_reuse_missing_df_errors(tmp_path):
    fp = _write_fit_pickle(tmp_path / _FIT_FNAME, ["subjA"], with_mle_df=False)
    try:
        nc.load_mle_per_trial(fp, subjects=["subjA"], reuse_fit_df=True,
                              verbose=False)
    except KeyError as exc:
        assert "reuse_fit_df=False" in str(exc)
    else:
        raise AssertionError("expected KeyError when stored mle_df is absent")


def test_load_mle_recompute_requires_behavior(tmp_path):
    fp = _write_fit_pickle(tmp_path / _FIT_FNAME, ["subjA"])
    try:
        nc.load_mle_per_trial(fp, df_behavior=None, subjects=["subjA"],
                              reuse_fit_df=False, verbose=False)
    except ValueError as exc:
        assert "df_behavior" in str(exc)
    else:
        raise AssertionError("expected ValueError without df_behavior")


def test_subjects_in_2p_lists_names():
    df_2p = pd.concat([_make_2p(name="ratA", short="A"),
                       _make_2p(name="ratB", short="B")], ignore_index=True)
    assert nc.subjects_in_2p(df_2p) == ["ratA", "ratB"]


# --------------------------------------------------------------------------
# epoch info + per-value-range trace plot + save-subset threshold
# --------------------------------------------------------------------------
def test_build_table_stores_epochs():
    table = nc.build_neuron_trial_table(_make_2p(), _make_mle_pt(), PARAM_KEYS)
    assert table.epochs_ranges.iloc[0] == ((0, 1), (2, 6), (7, 9))
    assert list(table.epochs_names.iloc[0])[1] == "Sampling"


def test_param_gradient_light_to_dark():
    cols = nc._param_gradient("purple", 4)
    assert len(cols) == 4
    # monotonic light -> dark: total brightness (sum of RGB) strictly decreases
    brightness = [sum(c) for c in cols]
    assert all(a > b for a, b in zip(brightness, brightness[1:]))
    assert brightness[0] > brightness[-1] + 0.5   # a clearly visible spread


def test_range_colors_styles_signed_magnitude_and_dash():
    # Symmetric ranges: negatives dashed, and mirror ranges share a shade
    # (colour encodes |midpoint|, not the ordered position).
    spec = nc.PARAM_BY_KEY["DV"]
    ranges = [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)]
    colors, styles = nc._range_colors_styles(spec, ranges)
    assert styles == ["--", "--", "-", "-"]           # negative side dashed
    # extremes (|mid|=0.75) match; inner (|mid|=0.25) match; extreme != inner
    assert colors[0] == colors[3] and colors[1] == colors[2]
    assert colors[0] != colors[1]
    # extreme is darker (lower total brightness) than the inner range
    assert sum(colors[0]) < sum(colors[1])


def test_range_colors_styles_unsigned_all_solid():
    spec = nc.PARAM_BY_KEY["DVabs"]
    ranges = [(0.0, 0.35), (0.35, 0.65), (0.65, 1.0)]
    colors, styles = nc._range_colors_styles(spec, ranges)
    assert styles == ["-", "-", "-"]
    # plain light -> dark gradient over the ordered ranges
    assert sum(colors[0]) > sum(colors[-1])


def test_plot_results_save_min_abs_corr_filters(tmp_path):
    df_2p = _make_2p(activities={"n1": [1.0, 2.0, 3.0, 4.0],
                                 "n2": [4.0, 3.0, 2.0, 1.0]})
    mle_pt = _make_mle_pt(q_val=[1.0, 2.0, 3.0, 4.0])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)
    # both neurons are |r|=1 for Q_val
    out = nc.plot_neuron_results(table, corr, "Q_val", mode="save",
                                 min_abs_corr=0.5, save_root=str(tmp_path),
                                 model_name="M")
    assert len(list(out.glob("*.svg"))) == 2
    # a threshold above 1 excludes everyone
    out2 = nc.plot_neuron_results(table, corr, "Q_val", mode="save",
                                  min_abs_corr=1.5, save_root=str(tmp_path),
                                  model_name="M2")
    assert len(list(out2.glob("*.svg"))) == 0


def test_param_traces_save_runs_and_filters(tmp_path):
    df_2p = _make_2p(activities={"n1": [1.0, 2.0, 3.0, 4.0],
                                 "n2": [4.0, 3.0, 2.0, 1.0]})
    mle_pt = _make_mle_pt(q_val=[1.0, 2.0, 3.0, 4.0])
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS)
    edges = [0.5, 2.5, 4.5]               # 2 bins: Q_val {1,2} vs {3,4}
    out = nc.plot_neuron_param_traces(
        table, corr, "Q_val", edges, mode="save", min_abs_corr=0.5,
        save_root=str(tmp_path), model_name="M")
    files = sorted(p.name for p in out.glob("*.svg"))
    assert len(files) == 2
    assert any(f.startswith("+1.000_S1_n1") for f in files)
    assert out.parts[-3:] == ("M", "param_traces", "Q_val")


# --------------------------------------------------------------------------
# Fast vs slow drift analysis (DV / DVabs)
# --------------------------------------------------------------------------
# 8 trials: 4 fast (quantile 1), 4 slow (quantile 3); DV = [-1,-.5,.5,1] each.
# n1: activity tracks DV in fast (r=+1, tuned) but not slow (|r|=.32);
# n2: mirror (r=-1 fast, |r|=.32 slow). Activities kept positive so nanmax works.
_FS_DVS = [-1.0, -0.5, 0.5, 1.0, -1.0, -0.5, 0.5, 1.0]
_FS_Q = [1, 1, 1, 1, 3, 3, 3, 3]
_FS_ACT = {"n1": [1.0, 1.5, 2.5, 3.0, 1.0, 3.0, 1.0, 3.0],
           "n2": [3.0, 2.5, 1.5, 1.0, 3.0, 1.0, 3.0, 1.0]}


def _make_fastslow():
    df_2p = _make_2p(n_trials=8, activities=_FS_ACT, dvs=_FS_DVS, quantiles=_FS_Q)
    mle_pt = _make_mle_pt(n_trials=8, q_val=list(range(8)),
                          loglik=[-1.0] * 8, valid=[True] * 8)
    table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    return table


def test_build_table_carries_drift_and_quantile():
    table = _make_fastslow()
    for col in ("DV", "DVabs", "quantile_idx"):
        assert col in table.columns
    # DVabs == |DV|; quantile_idx preserved
    assert np.allclose(table.DVabs, table.DV.abs())
    assert set(table.quantile_idx.unique()) == {1, 3}


def test_load_2p_activity_derives_dvabs(tmp_path):
    import pickle
    df = _make_2p(n_trials=2)           # has DV, no DVabs
    df["epoch"] = "Sampling"
    fp = tmp_path / "twop.pkl"
    with open(fp, "wb") as f:
        pickle.dump(df, f)
    loaded = nc.load_2p_activity(fp)
    assert "DVabs" in loaded.columns
    assert np.allclose(loaded.DVabs, loaded.DV.abs())


def test_split_fast_slow_and_drift_corr():
    table = _make_fastslow()
    fast, slow = nc.split_fast_slow(table)
    assert (fast.quantile_idx == 1).all() and (slow.quantile_idx == 3).all()
    corr_fast, corr_slow = nc.drift_correlations(table)
    assert {"DV_r", "DVabs_r"} <= set(corr_fast.columns)
    # n1: r=+1 fast, sub-threshold slow
    n1_fast = corr_fast.loc[corr_fast.long_trace_id == "S1_n1", "DV_r"].iloc[0]
    n1_slow = corr_slow.loc[corr_slow.long_trace_id == "S1_n1", "DV_r"].iloc[0]
    assert np.isclose(n1_fast, 1.0)
    assert abs(n1_slow) < 0.5


def test_plot_fast_slow_bars_contrast():
    table = _make_fastslow()
    corr_fast, corr_slow = nc.drift_correlations(table)
    summary = nc.plot_fast_slow_bars(corr_fast, corr_slow, "DV",
                                     min_abs_corr=0.5, by_region=True)
    assert set(summary.speed) == {"fast", "slow"}
    assert (summary.BrainRegion == "MFC").all()          # one region
    fast_pct = summary[summary.speed == "fast"].pct.iloc[0]
    slow_pct = summary[summary.speed == "slow"].pct.iloc[0]
    assert fast_pct == 100.0 and slow_pct == 0.0         # tuned only in fast
    # combined -> still fast + slow rows
    comb = nc.plot_fast_slow_bars(corr_fast, corr_slow, "DVabs",
                                  min_abs_corr=0.5, by_region=False)
    assert (comb.BrainRegion == "MFC & LFC").all() and len(comb) == 2


def test_fastslow_scatter_and_traces_save(tmp_path):
    table = _make_fastslow()
    corr_fast, corr_slow = nc.drift_correlations(table)
    sc = nc.plot_neuron_fastslow_scatter(
        table, corr_fast, corr_slow, "DV", mode="save", min_abs_corr=0.5,
        save_root=str(tmp_path), model_name="M")
    assert len(list(sc.glob("*.svg"))) == 2               # n1, n2 both max|r|=1
    assert sc.parts[-3:] == ("M", "drift_scatter", "DV")
    tr = nc.plot_neuron_fastslow_traces(
        table, corr_fast, corr_slow, "DV", [-1.001, 0.0, 1.001], mode="save",
        min_abs_corr=0.5, save_root=str(tmp_path), model_name="M")
    assert len(list(tr.glob("*.svg"))) == 2
    assert tr.parts[-3:] == ("M", "drift_traces", "DV")


# --------------------------------------------------------------------------
# Significance helpers: paired permutation + hierarchical bootstrap
# --------------------------------------------------------------------------
def _corr_frame(rvals_by_session, rcol="DV_r", region="MFC"):
    """A minimal per-neuron correlation frame: ``rvals_by_session`` maps a
    session name to a list of that session's neurons' r values."""
    rows = []
    for sess, rs in rvals_by_session.items():
        for i, r in enumerate(rs):
            rows.append({"long_trace_id": f"{sess}_{i}", "ShortName": sess,
                         "BrainRegion": region, rcol: r})
    return pd.DataFrame(rows)


def test_paired_perm_fastslow_equal_vs_strong():
    rng = np.random.default_rng(0)
    # Equal per-session fast/slow % -> zero paired difference -> p == 1.
    same = {f"S{i}": [0.9] for i in range(6)}
    obs, p = nc._paired_perm_fastslow(_corr_frame(same), _corr_frame(same),
                                      "DV_r", 0.5, 2000, rng)
    assert obs == 0.0 and p == 1.0
    # Fast fully tuned, slow untuned in every session -> large diff, small p.
    fast = {f"S{i}": [0.9] for i in range(6)}
    slow = {f"S{i}": [0.0] for i in range(6)}
    obs2, p2 = nc._paired_perm_fastslow(_corr_frame(fast), _corr_frame(slow),
                                        "DV_r", 0.5, 2000, rng)
    assert obs2 == 100.0 and p2 < 0.05


def test_paired_perm_fastslow_needs_two_sessions():
    one = {"S1": [0.9]}
    obs, p = nc._paired_perm_fastslow(_corr_frame(one), _corr_frame(one),
                                      "DV_r", 0.5, 100, np.random.default_rng(0))
    assert np.isnan(obs) and np.isnan(p)


def test_hier_bootstrap_region_diff_gap_vs_identical():
    rng = np.random.default_rng(0)
    tuned = [np.array([[0.9], [0.8]]) for _ in range(5)]   # 100% each session
    untuned = [np.array([[0.0], [0.1]]) for _ in range(5)]  # 0% each session
    res = nc._hier_bootstrap_region_diff(tuned, untuned, 0.5, 500, rng)
    assert res["diff"] > 0 and res["p"] < 0.05
    assert np.isclose(res["stat_a"], 100.0) and np.isclose(res["stat_b"], 0.0)
    # Identical regions -> zero observed difference -> p == 1.
    same = nc._hier_bootstrap_region_diff(tuned, [m.copy() for m in tuned],
                                          0.5, 500, rng)
    assert same["diff"] == 0.0 and same["p"] == 1.0


def test_sigstar_thresholds():
    assert nc._sigstar(0.0005) == "***"
    assert nc._sigstar(0.005) == "**"
    assert nc._sigstar(0.03) == "*"
    assert nc._sigstar(0.2) == "n.s."
    assert nc._sigstar(np.nan) == ""


# --------------------------------------------------------------------------
# Fast/slow bars: per-bar vs-chance test + run_stats gating
# --------------------------------------------------------------------------
def _make_fastslow_two_sessions():
    """Two MFC sessions (S1, S2), 8 trials each, fast (q=1) / slow (q=3)."""
    df_2p = pd.concat([
        _make_2p(short="S1", sess=1, n_trials=8, activities=_FS_ACT,
                 dvs=_FS_DVS, quantiles=_FS_Q),
        _make_2p(short="S2", sess=2, n_trials=8,
                 activities={"n3": _FS_ACT["n1"], "n4": _FS_ACT["n2"]},
                 dvs=_FS_DVS, quantiles=_FS_Q),
    ], ignore_index=True)
    mle_pt = pd.concat([
        _make_mle_pt(sess=1, n_trials=8, q_val=list(range(8)),
                     loglik=[-1.0] * 8, valid=[True] * 8),
        _make_mle_pt(sess=2, n_trials=8, q_val=list(range(8)),
                     loglik=[-1.0] * 8, valid=[True] * 8),
    ], ignore_index=True)
    return nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)


def test_plot_fast_slow_bars_stats_columns():
    table = _make_fastslow_two_sessions()
    corr_fast, corr_slow = nc.drift_correlations(table)
    summary = nc.plot_fast_slow_bars(
        corr_fast, corr_slow, "DV", table=table, min_abs_corr=0.5,
        n_perm=200, n_boot=200, by_region=True, run_stats=True)
    # New significance columns present, and every finite p is a probability.
    for col in ("p_vs_chance", "p_fast_vs_slow"):
        assert col in summary.columns
        finite = summary[col].dropna()
        assert ((finite >= 0.0) & (finite <= 1.0)).all()


def _make_two_region_fastslow():
    """Two sessions in MFC (M2_Bi) + two in LFC (ALM_Bi), fast/slow trials, so the
    MFC-vs-LFC hierarchical bootstrap + cross-region brackets are exercised."""
    frames, mles = [], []
    for ri, region in enumerate((nc.BrainRegion.M2_Bi, nc.BrainRegion.ALM_Bi)):
        for si in range(2):
            sess = ri * 2 + si + 1
            frames.append(_make_2p(
                region=region, short=f"R{ri}S{si}", sess=sess, n_trials=8,
                activities={f"n{sess}a": _FS_ACT["n1"], f"n{sess}b": _FS_ACT["n2"]},
                dvs=_FS_DVS, quantiles=_FS_Q))
            mles.append(_make_mle_pt(sess=sess, n_trials=8, q_val=list(range(8)),
                                     loglik=[-1.0] * 8, valid=[True] * 8))
    table = nc.build_neuron_trial_table(pd.concat(frames, ignore_index=True),
                                        pd.concat(mles, ignore_index=True),
                                        PARAM_KEYS)
    return table


def test_two_region_cross_bootstrap_runs():
    table = _make_two_region_fastslow()
    corr_all = nc.compute_neuron_correlations(table, PARAM_KEYS + ["DV"])
    corr_fast, corr_slow = nc.drift_correlations(table)

    # Summary figure: one figure per region, per-bar vs-chance stars + DV
    # fast-vs-slow bracket (no cross-region comparison anymore).
    fsdv = nc.plot_factor_bars_fastslow_dv(
        corr_all, table, min_abs_corr=0.5, n_shuffles=30,
        by_region=True, run_stats=True)
    assert {"p_vs_shuffle", "p_vs_shuffle_holm",
            "p_dv_fast_vs_slow"} <= set(fsdv.columns)
    # Cross-region columns are gone now that MFC/LFC live in separate figures.
    assert "p_mfc_vs_lfc" not in fsdv.columns
    finite = fsdv["p_vs_shuffle_holm"].dropna()
    assert len(finite) and ((finite >= 0.0) & (finite <= 1.0)).all()
    # Both regions present as separate row groups.
    assert set(fsdv.BrainRegion) == {"MFC", "LFC"}
    # DV fast-vs-slow paired p present on the DV bars.
    dv_fs = fsdv.loc[fsdv.factor == "DV_slow", "p_dv_fast_vs_slow"].dropna()
    assert len(dv_fs) and ((dv_fs >= 0.0) & (dv_fs <= 1.0)).all()

    # Fast/slow bars: cross-region rows appended (spanning MFC vs LFC per speed).
    fs = nc.plot_fast_slow_bars(
        corr_fast, corr_slow, "DV", table=table, min_abs_corr=0.5,
        n_perm=200, n_boot=200, by_region=True, run_stats=True)
    cross = fs[fs.BrainRegion.astype(str).str.contains("vs")]
    assert set(cross.speed) == {"fast", "slow"}
    assert cross["p_cross_region"].notna().all()


def test_run_stats_false_skips_tests(tmp_path):
    """run_stats=False returns the same-shaped summaries with no stars/shuffle."""
    table = _make_fastslow_two_sessions()
    corr = nc.compute_neuron_correlations(table, PARAM_KEYS + ["DV"])
    corr_fast, corr_slow = nc.drift_correlations(table)

    reg = nc.plot_region_bars(corr, table, min_abs_corr=0.5, by_region=False,
                              run_stats=False)
    assert len(reg) and reg["p_vs_shuffle"].isna().all()

    fac = nc.plot_factor_bars(corr, table, min_abs_corr=0.5, by_region=False,
                              run_stats=False)
    assert len(fac) and fac["p_vs_shuffle"].isna().all()

    fsdv = nc.plot_factor_bars_fastslow_dv(corr, table, min_abs_corr=0.5,
                                           by_region=True, run_stats=False)
    assert list(fsdv.factor.unique()) == ["Q", "RewardRate", "DV_fast", "DV_slow"]

    fs = nc.plot_fast_slow_bars(corr_fast, corr_slow, "DV", table=table,
                                min_abs_corr=0.5, by_region=True, run_stats=False)
    assert len(fs) and fs["p_vs_chance"].isna().all()


# --------------------------------------------------------------------------
# Reward-rate bins: correlated neurons across reward-rate levels
# --------------------------------------------------------------------------
# A session is one 4-trial block per reward-rate level, DV = [-1,-.5,.5,1] within
# each block. A neuron's block activity either tracks DV (r=+1) or does not
# (|r| = .32, below a 0.5 threshold).
_RR_DV_BLOCK = [-1.0, -0.5, 0.5, 1.0]
_TUNED_BLOCK = [1.0, 1.5, 2.5, 3.0]
_FLAT_BLOCK = [1.0, 3.0, 1.0, 3.0]
_RR_EDGES = [0.0, 0.5, 1.0]        # low [0,0.5) vs high [0.5,1.0]


def _make_rr_table(tuned_per_session=(1, 1, 1, 2), rates=(0.1, 0.9),
                   region=nc.BrainRegion.M2_Bi, tag=""):
    """Neuron-trial table with one 4-trial block per reward rate in ``rates``.

    One session per entry of ``tuned_per_session``; the entry is how many of that
    session's 2 neurons track DV inside the **lowest** reward-rate block (0, 1 or
    2). Every neuron is untuned at the other rates, so the % of DV-correlated
    neurons is high in the low bin and 0 elsewhere.
    """
    n_levels = len(rates)
    n_trials = 4 * n_levels
    dvs = _RR_DV_BLOCK * n_levels
    reward_rate = [r for r in rates for _ in range(4)]
    frames, mles = [], []
    for si, n_tuned in enumerate(tuned_per_session):
        acts = {}
        for j in range(2):
            blocks = [(_TUNED_BLOCK if (li == 0 and j < n_tuned) else _FLAT_BLOCK)
                      for li in range(n_levels)]
            acts[f"n{j}"] = [v for blk in blocks for v in blk]
        frames.append(_make_2p(region=region, short=f"{tag}S{si}", sess=si + 1,
                               n_trials=n_trials, activities=acts, dvs=dvs,
                               quantiles=[1] * n_trials))
        mles.append(_make_mle_pt(sess=si + 1, n_trials=n_trials,
                                 q_val=list(range(n_trials)),
                                 loglik=[-1.0] * n_trials,
                                 valid=[True] * n_trials,
                                 reward_rate=reward_rate))
    return nc.build_neuron_trial_table(pd.concat(frames, ignore_index=True),
                                       pd.concat(mles, ignore_index=True),
                                       PARAM_KEYS)


def test_make_value_bins_from_edges():
    bins = nc.make_value_bins(np.arange(0.0, 1.01, 0.5))
    assert [(b.lo, b.hi) for b in bins] == [(0.0, 0.5), (0.5, 1.0)]
    # only the LAST bin is closed on the right (so the max value is kept)
    assert [b.closed_right for b in bins] == [False, True]
    assert [b.label for b in bins] == ["[0, 0.5)", "[0.5, 1]"]


def test_make_value_bins_from_tuples_allows_gap():
    bins = nc.make_value_bins([(0.0, 0.3), (0.7, 1.0)])
    assert [(b.lo, b.hi) for b in bins] == [(0.0, 0.3), (0.7, 1.0)]
    assert [b.closed_right for b in bins] == [False, True]


def test_make_value_bins_rejects_bad_specs():
    for bad in ([], [0.0], [1.0, 0.0], [(0.5, 0.5)]):
        try:
            nc.make_value_bins(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_split_by_value_bins_membership_and_gap():
    table = _make_rr_table()
    bins = nc.make_value_bins(_RR_EDGES)
    low, high = nc.split_by_value_bins(table, "RewardRate", bins)
    assert set(low.RewardRate.unique()) == {0.1}
    assert set(high.RewardRate.unique()) == {0.9}
    assert len(low) + len(high) == len(table)      # contiguous bins keep it all
    # An explicit gap drops the trials that fall in it.
    gap = nc.split_by_value_bins(table, "RewardRate",
                                 nc.make_value_bins([(0.0, 0.05), (0.5, 1.0)]))
    assert len(gap[0]) == 0 and len(gap[1]) == len(high)


def test_split_by_value_bins_missing_column():
    table = _make_rr_table()
    try:
        nc.split_by_value_bins(table, "NoSuchLatent", nc.make_value_bins(_RR_EDGES))
    except KeyError as exc:
        assert "RewardRate" in str(exc)
    else:
        raise AssertionError("expected KeyError for an absent column")


def test_reward_rate_correlations_tuned_only_in_low_bin():
    table = _make_rr_table(tuned_per_session=(2,))       # both neurons tuned low
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    assert [b.label for b in bins] == ["[0, 0.5)", "[0.5, 1]"]
    assert [len(t) for t in bin_tables] == [8, 8]        # 2 neurons x 4 trials
    low, high = corr_bins
    assert {"DV_r", "DVabs_r"} <= set(low.columns)
    assert np.allclose(low.DV_r.abs(), 1.0)              # perfect inside low RR
    assert (high.DV_r.abs() < 0.5).all()                 # sub-threshold in high


def _make_skewed_rr_table(session_rates, tag=""):
    """One session per entry of ``session_rates`` (a list of per-trial reward
    rates), so sessions can be given deliberately different RR distributions."""
    frames, mles = [], []
    rng = np.random.default_rng(0)
    for si, rates in enumerate(session_rates):
        n = len(rates)
        acts = {f"n{j}": list(rng.normal(5.0, 1.0, n)) for j in range(3)}
        frames.append(_make_2p(short=f"{tag}S{si}", sess=si + 1, n_trials=n,
                               activities=acts,
                               dvs=list(rng.choice([-1., -.5, .5, 1.], n)),
                               quantiles=[1] * n))
        mles.append(_make_mle_pt(sess=si + 1, n_trials=n, q_val=list(range(n)),
                                 loglik=[-1.0] * n, valid=[True] * n,
                                 reward_rate=list(rates)))
    return nc.build_neuron_trial_table(pd.concat(frames, ignore_index=True),
                                       pd.concat(mles, ignore_index=True),
                                       PARAM_KEYS, verbose=False)


def test_session_quantile_split_is_balanced_within_every_session():
    """Each session is cut at ITS OWN quantiles, so every session contributes
    an equal number of trials to every level — the whole point of the split."""
    # Session 0 lives at low reward rates, session 1 at high ones. A fixed cut
    # point would give each session all-or-nothing; quantiles must not.
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 30),
                                   np.linspace(0.60, 0.99, 30)])
    bins, tables = nc.split_trials(table, "RewardRate", 3,
                                   by="session_quantile")
    assert [b.label for b in bins] == ["Q1/3 (low)", "Q2/3", "Q3/3 (high)"]
    for t in tables:
        per_sess = t.drop_duplicates(subset=["ShortName", "TrialNumber"]) \
                    .groupby("ShortName").size()
        assert list(per_sess) == [10, 10]        # both sessions, every level
    # Every trial is used exactly once.
    assert sum(len(t) for t in tables) == len(table)

    # A fixed-value split on the same data is all-or-nothing, as expected.
    _vb, vtables = nc.split_trials(table, "RewardRate", [0.0, 0.5, 1.0])
    v_per_sess = [t.drop_duplicates(subset=["ShortName", "TrialNumber"])
                   .ShortName.nunique() for t in vtables]
    assert v_per_sess == [1, 1]                  # each bin sees only 1 session


def test_session_quantile_split_keeps_every_session_paired():
    """The failure the split exists to fix: no session drops out."""
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 30),
                                   np.linspace(0.60, 0.99, 30),
                                   np.linspace(0.30, 0.95, 30)])
    cov = nc.session_bin_coverage(table, "RewardRate", 3, by="session_quantile")
    assert cov.attrs["n_complete"] == 3 and cov.complete.all()
    occ = nc.bin_occupancy(table, "RewardRate", 3, by="session_quantile")
    assert list(occ.n_trials) == [30, 30, 30]        # 3 sessions x 10
    assert list(occ.n_sessions) == [3, 3, 3]
    assert occ.attrs["dropped_trials"] == 0
    # Fixed cut points on the same data lose sessions.
    cov_v = nc.session_bin_coverage(table, "RewardRate", [0.0, 0.5, 1.0])
    assert cov_v.attrs["n_complete"] < 3


def test_bin_occupancy_reports_session_time_confound():
    """A latent that rises through the session makes its quantile split an
    early-vs-late split too; `mean_session_time` is what exposes that."""
    # RewardRate increases monotonically with trial number in every session.
    rising = _make_skewed_rr_table([np.linspace(0.1, 0.9, 30),
                                    np.linspace(0.2, 0.95, 30)])
    occ = nc.bin_occupancy(rising, "RewardRate", 3, by="session_quantile")
    t = occ.mean_session_time.to_numpy()
    assert t[0] < 0.25 and t[-1] > 0.75          # Q1 is early, Q3 is late
    assert (np.diff(t) > 0).all()

    # A latent that is shuffled in time gives time-matched levels (~0.5).
    rng = np.random.default_rng(0)
    shuffled_rates = [rng.permutation(np.linspace(0.1, 0.9, 30)) for _ in range(2)]
    mixed = _make_skewed_rr_table(shuffled_rates, tag="M")
    occ2 = nc.bin_occupancy(mixed, "RewardRate", 3, by="session_quantile")
    assert (np.abs(occ2.mean_session_time - 0.5) < 0.2).all()


def test_session_time_fraction_spans_zero_to_one():
    table = _make_rr_table(tuned_per_session=(1, 1))
    pos = nc._session_time_fraction(table)
    assert np.isclose(pos.min(), 0.0) and np.isclose(pos.max(), 1.0)
    # Computed per session, so every session spans the full range.
    per_sess = table.assign(_p=pos).groupby("ShortName")["_p"].agg(["min", "max"])
    assert (per_sess["min"] == 0.0).all() and (per_sess["max"] == 1.0).all()


def test_session_quantile_split_handles_ties_and_remainders():
    # 10 trials / 3 quantiles -> sizes 4,3,3 (remainder round-robin); heavy ties.
    table = _make_skewed_rr_table([[0.5] * 4 + [0.8] * 3 + [0.9] * 3])
    bins, tables = nc.split_trials(table, "RewardRate", 3,
                                   by="session_quantile")
    sizes = [t.drop_duplicates(subset=["ShortName", "TrialNumber"]).shape[0]
             for t in tables]
    assert sorted(sizes) == [3, 3, 4] and sum(sizes) == 10
    # Ties never leave a level empty (which value-binning on ties would).
    assert all(s > 0 for s in sizes)


def test_session_quantile_split_warns_on_too_few_trials():
    table = _make_skewed_rr_table([np.linspace(0.1, 0.9, 30), [0.5, 0.6]])
    with pytest.warns(UserWarning, match="fewer than 3 usable trials"):
        _bins, tables = nc.split_trials(table, "RewardRate", 3,
                                        by="session_quantile")
    # The 2-trial session is unassigned everywhere, the good one is intact.
    for t in tables:
        assert set(t.ShortName.unique()) == {"S0"}


def test_quantile_range_split_tracks_each_sessions_own_distribution():
    """Extremes in QUANTILE space: every session contributes to both bands even
    though their value ranges do not overlap at all."""
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 20),
                                   np.linspace(0.60, 0.99, 20)])
    bands = [(0.0, 0.25), (0.75, 1.0)]
    bins, tables = nc.split_trials(table, "RewardRate", bands,
                                   by="session_quantile_range")
    assert [b.label for b in bins] == ["0-25% (low)", "75-100% (high)"]
    for t in tables:
        per_sess = (t.drop_duplicates(subset=["ShortName", "TrialNumber"])
                     .groupby("ShortName").size())
        assert list(per_sess) == [5, 5]          # both sessions, both bands
    # The crowded middle is dropped, as an extremes split should.
    kept = sum(t.drop_duplicates(subset=["ShortName", "TrialNumber"]).shape[0]
               for t in tables)
    assert kept == 20 and kept < 40

    # Fixed value bands over the same quantiles are all-or-nothing here.
    _vb, vtables = nc.split_trials(table, "RewardRate", [(0.0, 0.25), (0.75, 1.0)])
    assert [t.ShortName.nunique() for t in vtables] == [1, 1]


def test_quantile_range_split_keeps_tied_values_whole():
    """The reason this criterion exists: a run of tied values is never cut in
    half to make the bands equal-sized."""
    # Half the session sits at exactly 0.1, so the 25% cut lands inside the tie.
    vals = [0.1] * 5 + [0.5, 0.6, 0.7, 0.8, 0.9]
    table = _make_skewed_rr_table([vals])
    _bins, tables = nc.split_trials(table, "RewardRate",
                                    [(0.0, 0.25), (0.75, 1.0)],
                                    by="session_quantile_range")
    low, high = [t.drop_duplicates(subset=["ShortName", "TrialNumber"])
                 for t in tables]
    # ALL 5 tied trials are in the low band — 50% of the session, not 25%.
    assert sorted(low.RewardRate.round(3)) == [0.1] * 5
    assert sorted(high.RewardRate.round(3)) == [0.7, 0.8, 0.9]
    # Unequal by design: the tie was kept whole rather than trimmed to match.
    assert len(low) != len(high)

    # The equal-count criterion does the opposite on the same data — trials with
    # IDENTICAL reward rates end up in different levels.
    _qb, qtables = nc.split_trials(table, "RewardRate", 4,
                                   by="session_quantile")
    tied_in = [((t.RewardRate.round(3) == 0.1).any()) for t in qtables]
    assert sum(tied_in) > 1


def test_quantile_range_split_unassigns_sessions_with_no_spread():
    """A session whose values barely move would have its low and high band land
    on the same trials; those are dropped, not double-counted."""
    table = _make_skewed_rr_table([np.linspace(0.1, 0.9, 20), [0.5] * 10])
    with pytest.warns(UserWarning, match="more than one band"):
        _bins, tables = nc.split_trials(table, "RewardRate",
                                        [(0.0, 0.25), (0.75, 1.0)],
                                        by="session_quantile_range")
    for t in tables:
        assert set(t.ShortName.unique()) == {"S0"}     # flat session excluded
    # ... and it is reported as uncovered rather than vanishing quietly.
    cov = nc.session_bin_coverage(table, "RewardRate",
                                  [(0.0, 0.25), (0.75, 1.0)],
                                  by="session_quantile_range")
    assert cov.attrs["n_complete"] == 1


def test_make_quantile_ranges_validates_and_sorts():
    # Sorted by q_lo, whatever order they are given in.
    assert nc.make_quantile_ranges([(0.75, 1.0), (0.0, 0.25)]) == \
        [(0.0, 0.25), (0.75, 1.0)]
    # Percentages are a likely slip, and are caught by name.
    for bad, msg in ([[(0, 25), (75, 100)], "FRACTIONS"],
                     [[(0.0, 0.5), (0.5, 1.0)], "touch or overlap"],
                     [[(0.0, 0.6), (0.4, 1.0)], "touch or overlap"],
                     [[(0.5, 0.5)], "0 <= q_lo < q_hi <= 1"],
                     [[], "empty"]):
        try:
            nc.make_quantile_ranges(bad)
        except ValueError as exc:
            assert msg in str(exc), f"{bad} -> {exc}"
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_plot_session_value_hist_draws_quantile_bands():
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 20),
                                   np.linspace(0.60, 0.99, 20)])
    fig, per_sess = nc.plot_session_value_hist(
        table, "RewardRate", bins=np.arange(0, 1.1, 0.1),
        quantile_ranges=[(0.0, 0.25), (0.75, 1.0)])
    assert len(fig.axes) >= 2
    assert list(per_sess.n_per_quantile_band.iloc[0]) == [5, 5]
    assert per_sess.usable_in_all_bands.all()
    # The bands sit at DIFFERENT values per session — the drift a fixed cut has.
    lo0 = per_sess.quantile_band_edges.iloc[0][0]
    lo1 = per_sess.quantile_band_edges.iloc[1][0]
    assert lo0 != lo1


def test_split_trials_rejects_unknown_criterion():
    table = _make_rr_table(tuned_per_session=(1,))
    try:
        nc.split_trials(table, "RewardRate", 3, by="nonsense")
    except ValueError as exc:
        assert "session_quantile" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown criterion")


def test_make_value_bins_rejects_overlapping_tuples():
    try:
        nc.make_value_bins([(0.0, 0.6), (0.4, 1.0)])
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError for overlapping bins")


def test_reward_rate_bars_with_session_quantiles(tmp_path):
    """End-to-end on the quantile criterion: equal trials per level per session,
    so no session is dropped and the paired test uses all of them."""
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 30),
                                   np.linspace(0.60, 0.99, 30),
                                   np.linspace(0.30, 0.95, 30),
                                   np.linspace(0.20, 0.80, 30)])
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(
        table, 3, by="session_quantile")
    assert [len(t) for t in bin_tables] == [120, 120, 120]   # 4 sess x 10 x 3 neu
    summary = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        n_perm=100, by_region=False, run_stats=True, save=True,
        save_root=str(tmp_path), model_name="M", ext="svg")
    bars = summary[summary.scope == "bin"]
    assert len(bars) == 3
    assert (bars.n_sessions == 4).all()          # nobody dropped
    assert (bars.n_sessions_dropped == 0).all()
    assert (bars.test_across_bins == "RM-ANOVA").all()
    # Equal exposure per level: 4 sessions x 10 trials = 40 trials, and
    # 40 x 3 neurons = 120 (neuron x trial) rows.
    assert (bars.n_trials == 40).all()
    assert (bars.n_rows == 120).all()
    assert (tmp_path / "M" / "reward_rate_bars" / "DV"
            / "bars_combined.svg").exists()


def test_reward_rate_summary_has_per_session_rows():
    """Each bar is backed by per-session rows carrying that session's own trial
    and neuron counts, so an outlying session is not averaged away invisibly."""
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 30),
                                   np.linspace(0.60, 0.99, 30),
                                   np.linspace(0.30, 0.95, 30)])
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(
        table, 3, by="session_quantile")
    summary = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        by_region=False, run_stats=False)

    assert set(summary.scope) == {"bin", "session"}      # no post-hocs w/o stats
    sess_rows = summary[summary.scope == "session"]
    # 3 sessions x 3 levels
    assert len(sess_rows) == 9
    assert set(sess_rows.ShortName) == {"S0", "S1", "S2"}
    assert (sess_rows.n_trials == 10).all()              # equal split per session
    assert (sess_rows.n_neurons == 3).all()
    assert (sess_rows.n_rows == 30).all()                # 3 neurons x 10 trials
    assert (sess_rows.trials_per_neuron == 10).all()
    assert sess_rows.paired.all()

    # The bar is the mean of its sessions' percentages.
    for _, bar in summary[summary.scope == "bin"].iterrows():
        mine = sess_rows[sess_rows["bin"] == bar["bin"]]
        assert np.isclose(bar.pct, mine.pct.mean())
        assert bar.n_trials == mine.n_trials.sum()       # trials, not rows
        assert bar.n_rows == mine.n_rows.sum()


def test_summary_separates_trials_from_neuron_trial_rows():
    """n_trials counts distinct trials; n_rows counts (neuron x trial) rows."""
    table = _make_rr_table(tuned_per_session=(1, 1))     # 2 neurons per session
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    summary = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        by_region=False, run_stats=False)
    bar = summary[summary.scope == "bin"].iloc[0]
    assert bar.n_rows == bar.n_trials * 2               # 2 neurons
    assert bar.n_trials == 8                            # 2 sessions x 4 trials


def test_plot_session_value_hist(tmp_path):
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 30),
                                   np.linspace(0.60, 0.99, 30)])
    fig, per_sess = nc.plot_session_value_hist(
        table, "RewardRate", bins=np.arange(0, 1.1, 0.1), n_quantiles=3,
        save=True, save_root=str(tmp_path), model_name="M", ext="svg")
    assert list(per_sess.ShortName) == ["S0", "S1"]
    assert (per_sess.n_trials == 30).all()               # one row per TRIAL
    # The two sessions really are shifted apart — the point of the plot.
    assert per_sess.p50.iloc[0] < 0.5 < per_sess.p50.iloc[1]
    # Percentiles are named p0..p100, never "median" (that shadows a DF method).
    assert {"p0", "p25", "p50", "p75", "p100"} <= set(per_sess.columns)
    assert "median" not in per_sess.columns
    # Each session gets its OWN cut points (n_quantiles - 1 of them).
    assert all(len(c) == 2 for c in per_sess.quantile_cuts)
    assert per_sess.quantile_cuts.iloc[0] != per_sess.quantile_cuts.iloc[1]
    saved = tmp_path / "M" / "reward_rate_bars"
    assert (saved / "session_RewardRate_hist.svg").exists()
    assert (saved / "session_RewardRate_stats.csv").exists()


def test_plot_session_value_hist_fixed_bin_overlay():
    """In value mode the plot reports each session's trials per FIXED level and
    flags the levels it cannot contribute to."""
    table = _make_skewed_rr_table([np.linspace(0.05, 0.45, 30),   # low only
                                   np.linspace(0.10, 0.95, 30)])  # spans both
    _fig, per_sess = nc.plot_session_value_hist(
        table, "RewardRate", bins=np.arange(0, 1.1, 0.1),
        value_bins=[(0.0, 0.6), (0.8, 1.0)])
    counts = list(per_sess.n_per_fixed_bin)
    assert counts[0][1] == 0                     # S0 never reaches [0.8, 1]
    assert not per_sess.usable_in_all_fixed.iloc[0]
    assert counts[1][0] > 0 and counts[1][1] > 0
    assert per_sess.usable_in_all_fixed.iloc[1]
    # Matches what the split itself would do.
    cov = nc.session_bin_coverage(table, "RewardRate", [(0.0, 0.6), (0.8, 1.0)])
    assert cov.attrs["n_complete"] == 1


def _two_condition_corr(r_a, r_b, sessions=None, region="MFC"):
    """A pair of per-neuron correlation frames with prescribed r values.

    ``region`` may be one label for every neuron or one label per neuron.
    """
    n = len(r_a)
    sessions = sessions or [f"S{i % 2}" for i in range(n)]
    regions = [region] * n if isinstance(region, str) else list(region)
    mk = lambda rs: pd.DataFrame({
        "long_trace_id": [f"n{i}" for i in range(n)],
        "ShortName": sessions, "BrainRegion": regions, "DV_r": rs})
    return mk(r_a), mk(r_b)


def test_select_tuned_neurons_any_and_all():
    a, b = _two_condition_corr([0.8, 0.1, 0.4, np.nan],
                               [0.1, 0.2, 0.5, 0.9])
    # |r| >= 0.3 in EITHER frame: n0 (a), n2 (both), n3 (b).
    assert nc.select_tuned_neurons([a, b], "DV", min_abs_corr=0.3) == \
        ["n0", "n2", "n3"]
    # ... in BOTH frames: only n2.
    assert nc.select_tuned_neurons([a, b], "DV", min_abs_corr=0.3,
                                   how="all") == ["n2"]
    # Sign is irrelevant to selection: it is |r|.
    neg, pos = _two_condition_corr([-0.9, 0.0], [0.0, 0.0])
    assert nc.select_tuned_neurons([neg, pos], "DV", min_abs_corr=0.5) == ["n0"]


def test_paired_neuron_r_pairs_and_drops_undefined():
    a, b = _two_condition_corr([0.8, 0.1, 0.4, np.nan],
                               [0.1, 0.2, np.nan, 0.9])
    paired = nc.paired_neuron_r(a, b, "DV")
    # n2 (undefined in b) and n3 (undefined in a) cannot be paired.
    assert list(paired.long_trace_id) == ["n0", "n1"]
    assert paired.attrs["n_dropped_undefined"] == 2
    # Restricting to a selected set, and reporting what of it was unusable.
    sel = nc.paired_neuron_r(a, b, "DV", neurons=["n0", "n2"])
    assert list(sel.long_trace_id) == ["n0"]
    assert sel.attrs["n_requested"] == 2 and sel.attrs["n_missing"] == 1
    assert list(sel.columns) == ["long_trace_id", "ShortName", "BrainRegion",
                                 "r_a", "r_b"]


def test_plot_paired_neuron_r_uses_abs_and_tests(tmp_path):
    # Every neuron's |r| halves from condition a to b -> a clear paired effect.
    r_a = [0.8, 0.6, -0.7, 0.5, -0.9, 0.65]
    r_b = [0.4, 0.3, -0.35, 0.25, -0.45, 0.32]
    a, b = _two_condition_corr(r_a, r_b, sessions=list("AABBCC"))
    paired = nc.paired_neuron_r(a, b, "DV")
    fig, stats = nc.plot_paired_neuron_r(
        paired, ("fast", "slow"), "DV", save=True, save_root=str(tmp_path),
        model_name="M", tag="fastslow", ext="svg")

    assert stats["n_neurons"] == 6
    # |r| means, not signed: signed would average to ~0 with mixed signs.
    assert np.isclose(stats["mean_a"], np.mean(np.abs(r_a)))
    assert np.isclose(stats["mean_b"], np.mean(np.abs(r_b)))
    assert stats["mean_a"] > stats["mean_b"]
    assert stats["neuron_test"]["test"] == "paired t-test"
    assert stats["neuron_test"]["p"] < 0.01
    # The session-level test is reported alongside (3 sessions here).
    assert stats["session_test"]["n_sessions"] == 3
    saved = tmp_path / "M" / "paired_neuron_r"
    assert (saved / "fastslow.svg").exists() and (saved / "fastslow.csv").exists()

    # Signed mode really does keep the sign (and here washes out).
    _fig2, signed = nc.plot_paired_neuron_r(paired, ("fast", "slow"), "DV",
                                            use_abs=False)
    assert np.isclose(signed["mean_a"], np.mean(r_a))
    assert abs(signed["mean_a"]) < stats["mean_a"]


def test_plot_paired_neuron_r_same_neurons_across_two_comparisons():
    """The workflow: pick neurons on fast/slow, then reuse that SAME set for the
    reward-rate comparison."""
    fast, slow = _two_condition_corr([0.8, 0.1, 0.55, 0.2],
                                     [0.2, 0.15, 0.6, 0.25])
    low, high = _two_condition_corr([0.5, 0.4, 0.3, 0.45],
                                    [0.2, 0.35, 0.25, 0.4])
    sel = nc.select_tuned_neurons([fast, slow], "DV", min_abs_corr=0.3)
    assert sel == ["n0", "n2"]
    p_fs = nc.paired_neuron_r(fast, slow, "DV", neurons=sel)
    p_rr = nc.paired_neuron_r(low, high, "DV", neurons=sel)
    # Same neurons in both comparisons — the point of the design.
    assert list(p_fs.long_trace_id) == list(p_rr.long_trace_id) == sel
    _f1, s1 = nc.plot_paired_neuron_r(p_fs, ("fast", "slow"), "DV")
    _f2, s2 = nc.plot_paired_neuron_r(p_rr, ("low RR", "high RR"), "DV")
    assert s1["n_neurons"] == s2["n_neurons"] == 2


def test_plot_paired_neuron_r_draws_bars_and_grey_lines():
    """The panel is condition-coloured bars ± SEM, one *neutral* connecting line
    per neuron (no direction colouring), and a black line joining the means."""
    a, b = _two_condition_corr([0.8, 0.6, 0.7, 0.5], [0.4, 0.7, 0.3, 0.6])
    paired = nc.paired_neuron_r(a, b, "DV")
    fig, stats = nc.plot_paired_neuron_r(paired, ("fast", "slow"), "DV",
                                         bar_colors=nc.FAST_SLOW_COLORS)
    ax = fig.axes[0]

    bars = [p for p in ax.patches if p.get_height() != 0]
    assert len(bars) == 2
    assert np.isclose(bars[0].get_height(), stats["mean_a"])
    assert np.isclose(bars[1].get_height(), stats["mean_b"])
    assert all(bar.get_y() == 0 for bar in bars)     # bars start at the axis
    # Each bar wears its own condition's colour, not a neutral grey.
    assert [mcolors.to_hex(bar.get_facecolor()) for bar in bars] == \
        [mcolors.to_hex(c) for c in nc.FAST_SLOW_COLORS]

    # One line per neuron, all the same colour despite 2 rising and 2 falling.
    lines = [ln for ln in ax.lines if ln.get_label() == "_neuron"]
    assert len(lines) == 4
    assert len({ln.get_color() for ln in lines}) == 1
    # The means are joined in black, with no error bars of their own.
    mean_line, = [ln for ln in ax.lines if ln.get_label() == "_mean"]
    assert mcolors.to_hex(mean_line.get_color()) == "#000000"
    assert list(mean_line.get_ydata()) == [stats["mean_a"], stats["mean_b"]]
    # 0 is in view, since the bars are drawn from it.
    assert ax.get_ylim()[0] == 0.0


def test_reward_rate_colors_match_the_bar_figure():
    """Two levels get the extremes of the same gradient the reward-rate bars
    use, so the paired plot's colours line up with them at any resolution."""
    pair = nc.reward_rate_colors()
    assert len(pair) == 2
    for n in (2, 3, 5):
        grad = nc.reward_rate_colors(n)
        assert len(grad) == n
        assert (grad[0], grad[-1]) == tuple(pair)    # ends are resolution-free
    assert nc.FAST_SLOW_COLORS == ("tomato", "goldenrod")


def test_plot_paired_neuron_r_by_region_panels_and_stats(tmp_path):
    # MFC halves from a to b; LFC is flat. Pooling would blur the two.
    r_a = [0.8, 0.6, 0.7, 0.4, 0.5, 0.45]
    r_b = [0.4, 0.3, 0.35, 0.4, 0.5, 0.45]
    a, b = _two_condition_corr(r_a, r_b, sessions=list("AABBCC"),
                               region=["MFC"] * 3 + ["LFC"] * 3)
    paired = nc.paired_neuron_r(a, b, "DV")
    fig, stats = nc.plot_paired_neuron_r(
        paired, ("fast", "slow"), "DV", by_region=True, save=True,
        save_root=str(tmp_path), model_name="M", tag="fs", ext="svg")

    assert len(fig.axes) == 2
    assert set(stats) == {"LFC", "MFC"}              # keyed by region
    assert stats["MFC"]["n_neurons"] == stats["LFC"]["n_neurons"] == 3
    # Each panel is tested on its OWN neurons: MFC drops, LFC does not move.
    assert stats["MFC"]["mean_a"] > stats["MFC"]["mean_b"]
    assert np.isclose(stats["LFC"]["mean_a"], stats["LFC"]["mean_b"])
    assert stats["MFC"]["neuron_test"]["p"] < stats["LFC"]["neuron_test"]["p"]

    saved = tmp_path / "M" / "paired_neuron_r"
    assert (saved / "fs_by_region.svg").exists()     # distinct from the pooled
    rows = pd.read_csv(saved / "fs_by_region.csv")
    assert list(rows.BrainRegion) == ["LFC", "MFC"]

    # Pooling the same data is still a single panel with a flat dict.
    _f2, pooled = nc.plot_paired_neuron_r(paired, ("fast", "slow"), "DV")
    assert pooled["n_neurons"] == 6 and pooled["region"] == "MFC & LFC"


def test_paired_neuron_r_empty_selection_raises():
    a, b = _two_condition_corr([0.1], [0.2])
    empty = nc.paired_neuron_r(a, b, "DV", neurons=["nope"])
    try:
        nc.plot_paired_neuron_r(empty, ("x", "y"), "DV")
    except ValueError as exc:
        assert "no pairable neurons" in str(exc)
    else:
        raise AssertionError("expected ValueError on an empty selection")


def test_quantile_bin_edges_balances_occupancy():
    # Reward rates 0.1 (x4) and 0.9 (x4) per session -> the median splits them.
    table = _make_rr_table(tuned_per_session=(1, 1))
    edges = nc.quantile_bin_edges(table, "RewardRate", 2)
    assert len(edges) == 3
    assert edges[0] <= 0.1 and edges[-1] >= 0.9
    occ = nc.bin_occupancy(table, "RewardRate", edges)
    assert list(occ.n_trials) == [8, 8]            # 2 sessions x 4 trials each
    assert np.allclose(occ.pct_of_trials, 50.0)


def test_quantile_bin_edges_rejects_over_fine_split():
    table = _make_rr_table(tuned_per_session=(1,))   # only 2 distinct rates
    try:
        nc.quantile_bin_edges(table, "RewardRate", 5)
    except ValueError as exc:
        assert "too concentrated" in str(exc)
    else:
        raise AssertionError("expected ValueError for duplicate quantile edges")


def test_bin_occupancy_counts_and_dropped():
    table = _make_rr_table(tuned_per_session=(1, 1), rates=(0.1, 0.5, 0.9))
    occ = nc.bin_occupancy(table, "RewardRate", [(0.0, 0.3), (0.7, 1.0)])
    assert list(occ["bin"]) == ["[0, 0.3)", "[0.7, 1]"]
    assert list(occ.n_trials) == [8, 8]             # the 0.5 block is excluded
    # 2 sessions x 2 neurons = 4 distinct long_trace_ids in each bin.
    assert list(occ.n_neurons) == [4, 4] and list(occ.n_sessions) == [2, 2]
    assert occ.attrs["dropped_trials"] == 8         # the skipped middle block


def test_build_table_names_sessions_lost_in_the_join(capsys):
    """A session with no matching model trial is named + warned, not silent."""
    df_2p = pd.concat([_make_2p(short="Kept", sess=1),
                       _make_2p(short="Gone", sess=2)], ignore_index=True)
    mle_pt = _make_mle_pt(sess=1)          # only session 1 has model trials
    with pytest.warns(UserWarning, match="matched no model trial"):
        table = nc.build_neuron_trial_table(df_2p, mle_pt, PARAM_KEYS)
    assert set(table.ShortName.unique()) == {"Kept"}
    out = capsys.readouterr().out
    assert "1 session(s) LOST ENTIRELY" in out and "Gone" in out
    assert "1/2 sessions survive" in out


def test_build_table_join_report_is_silenceable(capsys):
    nc.build_neuron_trial_table(_make_2p(), _make_mle_pt(), PARAM_KEYS,
                                verbose=False)
    assert capsys.readouterr().out == ""


def test_session_bin_coverage_needs_enough_trials_to_correlate():
    """A bin holding 1-2 trials looks populated but yields no correlation, so
    the session is not usable there — `complete` must reflect that."""
    # 8 trials/session: rates put 1 trial in the low bin, 7 in the high bin.
    rates = [0.1] + [0.9] * 7
    frames, mles = [], []
    for si in range(2):
        acts = {f"n{j}": list(np.linspace(1.0, 8.0, 8)) for j in range(2)}
        frames.append(_make_2p(short=f"S{si}", sess=si + 1, n_trials=8,
                               activities=acts,
                               dvs=[-1.0, -0.5, 0.5, 1.0] * 2,
                               quantiles=[1] * 8))
        mles.append(_make_mle_pt(sess=si + 1, n_trials=8, q_val=list(range(8)),
                                 loglik=[-1.0] * 8, valid=[True] * 8,
                                 reward_rate=rates))
    table = nc.build_neuron_trial_table(pd.concat(frames, ignore_index=True),
                                        pd.concat(mles, ignore_index=True),
                                        PARAM_KEYS, verbose=False)

    cov = nc.session_bin_coverage(table, "RewardRate", _RR_EDGES)
    assert (cov["n_[0, 0.5)"] == 1).all()      # the bin IS visited...
    assert (cov.bins_covered == 2).all()
    assert (cov.bins_usable == 1).all()        # ... but 1 trial cannot correlate
    assert not cov.complete.any()
    assert cov.attrs["n_complete"] == 0 and cov.attrs["min_trials"] == 3
    # And that is what the correlations do: undefined r in the sparse bin.
    _bins, _tabs, corr = nc.reward_rate_correlations(table, _RR_EDGES)
    assert corr[0].DV_r.isna().all() and corr[1].DV_r.notna().any()


def test_session_attrition_accounts_for_every_session():
    """Every recorded session lands in exactly one bucket, and the three loss
    stages are told apart."""
    # A: spans both bins, plenty of trials      -> kept
    # B: high reward rate only                  -> too few trials (low bin)
    # C: in the 2p df but with no model trials  -> no model trials
    a = _make_rr_table(tuned_per_session=(1,), rates=(0.1, 0.9), tag="A")
    b = _make_rr_table(tuned_per_session=(1,), rates=(0.9, 0.9), tag="B")
    table = pd.concat([a, b], ignore_index=True)
    # The raw 2p df additionally holds a session the join dropped.
    _acts = {"n0": list(range(1, 9)), "n1": list(range(8, 0, -1))}
    df_2p = pd.concat([
        _make_2p(short=s, sess=i + 1, n_trials=8, activities=_acts)
        for i, s in enumerate(["AS0", "BS0", "CS0"])
    ], ignore_index=True)

    bins, _bt, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    att = nc.session_attrition(df_2p, table, "RewardRate", _RR_EDGES, "DV",
                               corr_by_bin=corr_bins)

    assert len(att) == 3                       # every recorded session appears
    by = dict(zip(att.ShortName, att.outcome))
    assert by["AS0"] == "kept"
    assert by["BS0"] == "too few trials"
    assert by["CS0"] == "no model trials"
    detail = dict(zip(att.ShortName, att.detail))
    assert "[0, 0.5)" in detail["BS0"]         # names the offending bin
    assert "join" in detail["CS0"]
    # Trial counts trace the loss: the dropped session kept none.
    assert att.loc[att.ShortName == "CS0", "n_trials_kept"].iloc[0] == 0
    assert att.loc[att.ShortName == "AS0", "n_trials_kept"].iloc[0] == 8


def test_session_attrition_flags_undefined_correlations():
    """A session with enough trials but no DEFINED r is its own category."""
    table = _make_rr_table(tuned_per_session=(1,), rates=(0.1, 0.9), tag="A")
    df_2p = _make_2p(short="AS0", sess=1, n_trials=8,
                     activities={"n0": list(range(1, 9)),
                                 "n1": list(range(8, 0, -1))})
    bins, _bt, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    # Blank the low bin's correlations: trials are there, r is not.
    corr_bins[0]["DV_r"] = np.nan
    att = nc.session_attrition(df_2p, table, "RewardRate", _RR_EDGES, "DV",
                               corr_by_bin=corr_bins, common_neurons=False)
    row = att.iloc[0]
    assert row.outcome == "no assessable neuron"
    assert "[0, 0.5)" in row.detail
    assert row["n_[0, 0.5)"] == 4              # the trials were never the issue


def test_session_bin_coverage_flags_one_sided_sessions():
    """A session whose reward rate never enters a bin is not a paired session."""
    # S0 spans both rates; S1 sits entirely at the high rate.
    both = _make_rr_table(tuned_per_session=(1,), rates=(0.1, 0.9), tag="A")
    high = _make_rr_table(tuned_per_session=(1,), rates=(0.9, 0.9), tag="B")
    table = pd.concat([both, high], ignore_index=True)

    cov = nc.session_bin_coverage(table, "RewardRate", _RR_EDGES)
    assert list(cov.ShortName) == ["AS0", "BS0"]
    spanning = cov[cov.ShortName == "AS0"].iloc[0]
    one_sided = cov[cov.ShortName == "BS0"].iloc[0]
    assert spanning.complete and spanning.bins_covered == 2
    assert not one_sided.complete and one_sided.bins_covered == 1
    assert one_sided["n_[0, 0.5)"] == 0 and one_sided["n_[0.5, 1]"] == 8
    assert one_sided.min_RewardRate == 0.9
    # Only the spanning session can be paired across bins.
    assert cov.attrs["n_complete"] == 1

    # ... and that is exactly the count the paired test ends up with.
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    pct = nc._bin_pct_frame(corr_bins, bins, "DV_r", 0.5)
    assert nc.paired_bins_test(pct)["n_sessions"] == 1


def _pct_frame(data):
    """sessions x bins % frame from a {bin_label: [per-session %]} dict."""
    return pd.DataFrame(data, index=[f"S{i}" for i in range(len(next(iter(
        data.values()))))])


def test_paired_bins_test_two_bins_is_paired_ttest():
    res = nc.paired_bins_test(_pct_frame({"low": [50.0, 50.0, 50.0, 100.0],
                                          "high": [0.0, 0.0, 0.0, 0.0]}))
    assert res["test"] == "paired t-test"
    assert res["n_sessions"] == 4 and res["df1"] == 3
    assert np.isclose(res["stat"], 5.0)                  # 62.5 / (25/2)
    assert res["p"] < 0.05
    assert "paired t(3)" in res["label"]


def test_paired_bins_test_three_bins_is_rm_anova():
    res = nc.paired_bins_test(_pct_frame({"lo": [10.0, 12.0, 8.0, 11.0],
                                          "mid": [30.0, 33.0, 28.0, 31.0],
                                          "hi": [60.0, 62.0, 58.0, 61.0]}))
    assert res["test"] == "RM-ANOVA"
    assert res["df1"] == 2 and res["df2"] == 6          # (k-1), (k-1)(n-1)
    assert res["p"] < 0.001
    assert "RM-ANOVA F(2,6)" in res["label"]


def test_paired_bins_test_degenerate_cases():
    # No between-bin variation anywhere -> p = 1 (not a 0/0 NaN).
    flat = nc.paired_bins_test(_pct_frame({"low": [40.0, 40.0, 20.0],
                                           "high": [40.0, 40.0, 20.0]}))
    assert flat["p"] == 1.0 and flat["stat"] == 0.0
    # Fewer than two complete sessions -> no test.
    one = nc.paired_bins_test(_pct_frame({"low": [50.0], "high": [0.0]}))
    assert one["test"] == "none" and np.isnan(one["p"])
    # Sessions missing a bin drop out (complete cases only).
    partial = nc.paired_bins_test(pd.DataFrame({"low": [50.0, 60.0, 70.0],
                                                "high": [0.0, np.nan, 10.0]}))
    assert partial["n_sessions"] == 2


def test_pairwise_bin_tests_holm_corrects():
    out = nc.pairwise_bin_tests(_pct_frame({"lo": [10.0, 12.0, 8.0, 11.0],
                                            "mid": [30.0, 33.0, 28.0, 31.0],
                                            "hi": [60.0, 62.0, 58.0, 61.0]}))
    assert len(out) == 3                                  # 3 choose 2
    assert set(out.bin_a) == {"lo", "mid"}
    assert (out.p_holm >= out.p).all()
    assert (out.n_sessions == 4).all()


def test_common_assessable_intersects_bins():
    # A neuron whose correlation is undefined in one bin is dropped everywhere.
    a = pd.DataFrame({"long_trace_id": ["x", "y"], "DV_r": [0.9, 0.4]})
    b = pd.DataFrame({"long_trace_id": ["x", "y"], "DV_r": [0.2, np.nan]})
    assert nc._common_assessable([a, b], "DV_r") == {"x"}


def test_plot_reward_rate_bars_contrast_and_stats(tmp_path):
    table = _make_rr_table(tuned_per_session=(1, 1, 1, 2))
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    summary = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        n_perm=200, by_region=True, run_stats=True, save=True,
        save_root=str(tmp_path), model_name="M", ext="svg")

    bars = summary[summary.scope == "bin"]
    assert list(bars["bin"]) == ["[0, 0.5)", "[0.5, 1]"]
    assert (bars.BrainRegion == "MFC").all()
    # per-session % low = [50, 50, 50, 100] -> 62.5; high = 0 everywhere.
    assert np.isclose(bars.pct.iloc[0], 62.5)
    assert bars.pct.iloc[1] == 0.0
    # 4 sessions x 4 trials = 16 trials per level; x2 neurons = 32 rows.
    assert (bars.n_sessions == 4).all()
    assert (bars.n_trials == 16).all() and (bars.n_rows == 32).all()
    # The across-bin test is the session-paired t-test, significant here.
    assert (bars.test_across_bins == "paired t-test").all()
    assert bars.p_across_bins.iloc[0] < 0.05
    assert 0.0 <= bars.p_across_bins_holm.iloc[0] <= 1.0
    finite = bars.p_vs_chance_holm.dropna()
    assert len(finite) and ((finite >= 0.0) & (finite <= 1.0)).all()
    # Raw p reported next to the corrected one, and Holm never lowers a p.
    raw = bars.p_vs_chance.dropna()
    assert len(raw) == len(finite)
    assert (bars.p_vs_chance_holm >= bars.p_vs_chance - 1e-12).all()
    # In a 2-bar family the LARGER raw p is multiplied by 1 -> untouched.
    worst = bars.p_vs_chance.idxmax()
    assert np.isclose(bars.loc[worst, "p_vs_chance_holm"],
                      bars.loc[worst, "p_vs_chance"])

    saved = tmp_path / "M" / "reward_rate_bars" / "DV"
    assert (saved / "bars_by_region.svg").exists()
    assert (saved / "summary_by_region.csv").exists()


def test_plot_reward_rate_bars_three_bins_anova_and_posthocs():
    """>2 bins -> RM-ANOVA + Holm-corrected pairwise rows appended to the summary."""
    table = _make_rr_table(tuned_per_session=(1, 1, 1, 2),
                           rates=(0.1, 0.5, 0.9))
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(
        table, [0.0, 0.3, 0.7, 1.0])
    assert [len(t) for t in bin_tables] == [32, 32, 32]     # all three populated
    summary = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        n_perm=100, by_region=False, run_stats=True)

    labels = [b.label for b in bins]
    bars = summary[summary.scope == "bin"]
    assert len(bars) == 3
    # low bin: per-session % = [50, 50, 50, 100] -> 62.5; the other two are 0.
    assert np.isclose(bars.pct.iloc[0], 62.5)
    assert (bars.pct.iloc[1:] == 0.0).all()
    assert (bars.test_across_bins == "RM-ANOVA").all()
    assert bars.df1.iloc[0] == 2 and bars.df2.iloc[0] == 6
    assert bars.p_across_bins.iloc[0] < 0.05
    # Post-hoc rows: 3 pairwise comparisons, Holm-corrected.
    posthoc = summary[summary.scope == "comparison"]
    assert len(posthoc) == 3
    assert (posthoc.test_across_bins == "paired t-test (post-hoc)").all()
    assert (posthoc.p_across_bins_holm >= posthoc.p_across_bins).all()
    # The two untuned bins are identical -> that pair is the non-significant one.
    same = posthoc[posthoc["bin"] == f"{labels[1]} vs {labels[2]}"]
    assert same.p_across_bins.iloc[0] == 1.0


def test_plot_reward_rate_bars_empty_bin_warns_under_common_neurons():
    """A bin no neuron can be correlated within leaves the common-neuron
    intersection empty; that is reported rather than silently blank."""
    table = _make_rr_table(tuned_per_session=(1, 1))
    # Only rates 0.1 / 0.9 exist, so the middle bin here catches no trial.
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(
        table, [0.0, 0.5, 0.8, 1.0])
    assert len(bin_tables[1]) == 0
    with pytest.warns(UserWarning, match="no neuron has a defined correlation"):
        summary = nc.plot_reward_rate_bars(
            corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
            by_region=False, run_stats=False)
    bars = summary[summary.scope == "bin"]
    assert len(bars) == 3 and bars.pct.isna().all()
    # Without the restriction the two populated bins still plot.
    loose = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        by_region=False, run_stats=False, common_neurons=False)
    loose_bars = loose[loose.scope == "bin"]
    assert np.isclose(loose_bars.pct.iloc[0], 50.0)
    assert np.isnan(loose_bars.pct.iloc[1])


def test_reward_rate_bars_reports_dropped_sessions(capsys):
    """A session that cannot be paired is named + warned about, never silent."""
    both = _make_rr_table(tuned_per_session=(1, 1), rates=(0.1, 0.9), tag="A")
    high = _make_rr_table(tuned_per_session=(1,), rates=(0.9, 0.9), tag="B")
    table = pd.concat([both, high], ignore_index=True)
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)

    with pytest.warns(UserWarning, match="sessions dropped from the paired"):
        summary = nc.plot_reward_rate_bars(
            corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
            by_region=False, run_stats=False, common_neurons=False)

    out = capsys.readouterr().out
    assert "BS0" in out and "EXCLUDED from the paired test" in out
    bars = summary[summary.scope == "bin"]
    # 3 sessions contribute neurons; only the 2 spanning ones can be paired.
    assert (bars.n_sessions_region == 3).all()
    assert (bars.n_sessions_dropped == 1).all()
    assert (bars.n_paired_sessions == 2).all()


def test_reward_rate_bars_zero_pairable_sessions_is_not_an_error(capsys):
    """No session spans the bins -> bars still draw, test reports 'none'."""
    table = _make_rr_table(tuned_per_session=(1, 1), rates=(0.9, 0.9))
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    with pytest.warns(UserWarning):
        summary = nc.plot_reward_rate_bars(
            corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
            by_region=False, run_stats=False, common_neurons=False)
    bars = summary[summary.scope == "bin"]
    assert bars.test_across_bins.iloc[0] == "none"
    assert bars.n_paired_sessions.iloc[0] == 0
    assert (bars.n_sessions_dropped == 2).all()
    # The populated bin still reports a real percentage.
    assert np.isnan(bars.pct.iloc[0]) and np.isfinite(bars.pct.iloc[1])
    assert "EXCLUDED from the paired test" in capsys.readouterr().out


def test_plot_reward_rate_bars_run_stats_false_and_pooled():
    table = _make_rr_table(tuned_per_session=(1, 2))
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    summary = nc.plot_reward_rate_bars(
        corr_bins, bins, "DVabs", bin_tables=bin_tables, min_abs_corr=0.5,
        by_region=False, run_stats=False)
    assert (summary.BrainRegion == "MFC & LFC").all()
    assert summary.p_vs_chance_holm.isna().all()
    assert (summary.param == "DVabs").all()


def test_reward_rate_bars_common_neurons_uses_one_population():
    """A neuron assessable in only one bin is excluded from both bars."""
    table = _make_rr_table(tuned_per_session=(2, 2))
    bins, bin_tables, corr_bins = nc.reward_rate_correlations(table, _RR_EDGES)
    # Blank out one neuron's high-bin correlation -> it is no longer assessable
    # in every bin, so common_neurons must drop it from the low bar too.
    dropped = corr_bins[1].long_trace_id.iloc[0]
    corr_bins[1].loc[corr_bins[1].long_trace_id == dropped, "DV_r"] = np.nan
    keep = nc._common_assessable(corr_bins, "DV_r")
    assert dropped not in keep

    with_common = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        by_region=False, run_stats=False, common_neurons=True)
    without = nc.plot_reward_rate_bars(
        corr_bins, bins, "DV", bin_tables=bin_tables, min_abs_corr=0.5,
        by_region=False, run_stats=False, common_neurons=False)
    # Same low-bin % (the dropped neuron was tuned like the rest) but a smaller
    # neuron count behind it.
    assert with_common.n_neurons.iloc[0] < without.n_neurons.iloc[0]
