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
import matplotlib
matplotlib.use("Agg")

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
                 q_val=None, loglik=None, valid=None):
    q_val = q_val if q_val is not None else [0.1, 0.2, 0.3, 0.4]
    loglik = loglik if loglik is not None else [-1.0, -2.0, -3.0, -1.5]
    valid = valid if valid is not None else [True] * n_trials
    return pd.DataFrame({
        "Name": [name] * n_trials,
        "Date": [pd.Timestamp(date)] * n_trials,
        "SessionNum": [sess] * n_trials,
        "TrialNumber": list(range(1, n_trials + 1)),
        "Q_L": q_val,
        "Q_R": [0.0] * n_trials,
        "Q_val": q_val,
        "RewardRate": [0.5] * n_trials,
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
