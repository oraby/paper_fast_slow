"""Tests for fast_slow_qr — the (Q-relative, reward-rate) fast/slow figures.

Pure logic (the DV-relative Q transform, the fast/slow split, the slow-fraction
binning) plus headless renders, all on synthetic frames. The data-assembly path
(build_fast_slow_qr) monkeypatches compare._compute_mle, since the real fit
pickles embed a subject_df that doesn't unpickle in this environment (same
constraint as test_model_compare / test_aggregate).
"""
from __future__ import annotations

import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import pytest

from .. import fast_slow_qr as fsqr
from .. import compare
from ..fast_slow_qr import (add_q_relative, label_fast_slow, label_speeds,
                            fastness_grid, build_fast_slow_qr,
                            plot_subject_qr_scatter, plot_all_subjects_qr_scatter,
                            plot_pooled_fastness_heatmap, CHI2_W05_SPEC)


# --------------------------------------------------------------------------
# Q-relative transform
# --------------------------------------------------------------------------
def test_q_relative_sign_follows_dv_vs_q_agreement():
    # (DV sign, q sign) -> Q_relative sign: agree -> +, disagree -> -.
    df = pd.DataFrame({
        "DV": [0.5, 0.5, -0.5, -0.5, 0.0],
        "mle_Q_rel_before": [0.8, -0.8, 0.8, -0.8, 0.3]})
    out = add_q_relative(df)["Q_relative"].to_numpy()
    assert out[0] == pytest.approx(0.8)    # DV+, q+ -> favours rewarded -> +
    assert out[1] == pytest.approx(-0.8)   # DV+, q- -> favours wrong    -> -
    assert out[2] == pytest.approx(-0.8)   # DV-, q+ -> favours wrong    -> -
    assert out[3] == pytest.approx(0.8)    # DV-, q- -> favours rewarded -> +
    assert out[4] == pytest.approx(0.3)    # DV 0 -> raw signed value


def test_q_relative_equals_sign_dv_times_q():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"DV": rng.uniform(-1, 1, 200),
                       "mle_Q_rel_before": rng.uniform(-1, 1, 200)})
    out = add_q_relative(df)["Q_relative"].to_numpy()
    expect = np.sign(df.DV.to_numpy()) * df.mle_Q_rel_before.to_numpy()
    assert out == pytest.approx(expect)


# --------------------------------------------------------------------------
# Fast/slow labelling
# --------------------------------------------------------------------------
def _trials(n_per_dvstr=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for dvstr, dv in (("Easy", 0.9), ("Med", 0.5), ("Hard", 0.2)):
        for _ in range(n_per_dvstr):
            rows.append({"DVstr": dvstr, "DV": dv,
                         "calcStimulusTime": float(rng.gamma(2, 0.2) + 0.1),
                         "mle_Q_rel_before": rng.uniform(-1, 1),
                         "RewardRate": rng.uniform(0, 1), "valid": True})
    return pd.DataFrame(rows)


def test_label_fast_slow_drops_typical_and_tags_the_rest():
    labelled = label_fast_slow(_trials())
    assert set(labelled.Speed) == {"Fast", "Slow"}
    # Two of three RT-thirds kept per difficulty -> 2/3 of trials.
    assert len(labelled) == pytest.approx(90 * 2 / 3, abs=3)


def test_label_fast_slow_fast_is_faster_than_slow_within_difficulty():
    labelled = label_fast_slow(_trials(n_per_dvstr=60))
    for _dvstr, g in labelled.groupby("DVstr"):
        fast = g[g.Speed == "Fast"].calcStimulusTime
        slow = g[g.Speed == "Slow"].calcStimulusTime
        assert fast.max() <= slow.min()


# --------------------------------------------------------------------------
# Slow-fraction binning
# --------------------------------------------------------------------------
def test_fastness_grid_slow_fraction_and_orientation():
    # One bin all-slow, one bin half/half, one bin all-fast, rest empty.
    df = pd.DataFrame({
        "RewardRate": [0.05, 0.05, 0.55, 0.55, 0.95],
        "Q_relative": [-0.9, -0.9, 0.0, 0.0, 0.9],
        "Speed": ["Slow", "Slow", "Slow", "Fast", "Fast"]})
    grid, counts = fastness_grid(df, n_bins_r=10, n_bins_q=10)
    assert grid.shape == (10, 10)                      # (q, r)
    assert grid[0, 0] == pytest.approx(1.0)            # low R, low Q: all slow
    assert grid[5, 5] == pytest.approx(0.5)            # mid: half slow
    assert grid[9, 9] == pytest.approx(0.0)            # high R, high Q: all fast
    assert np.isnan(grid[2, 2])                        # empty bin -> NaN
    assert counts[0, 0] == 2


def test_fastness_grid_respects_bin_counts():
    df = pd.DataFrame({"RewardRate": [0.5], "Q_relative": [0.0],
                       "Speed": ["Fast"]})
    grid, _ = fastness_grid(df, n_bins_r=5, n_bins_q=8)
    assert grid.shape == (8, 5)


def test_fastness_grid_clips_out_of_range_into_edge_bins():
    df = pd.DataFrame({"RewardRate": [1.0, 0.0], "Q_relative": [1.0, -1.0],
                       "Speed": ["Slow", "Fast"]})
    grid, counts = fastness_grid(df, n_bins_r=4, n_bins_q=4)
    assert counts.sum() == 2                           # no trial dropped
    assert grid[-1, -1] == 1.0 and grid[0, 0] == 0.0


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _labelled(seed=0):
    df = _trials(n_per_dvstr=40, seed=seed)
    df = add_q_relative(df)
    return label_fast_slow(df)


def test_scatter_is_one_interleaved_collection_of_all_points():
    labelled = _labelled()
    ax = plot_subject_qr_scatter(labelled, "S1", point_size=3)
    colls = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(colls) == 1                              # single scatter artist
    assert colls[0].get_offsets().shape[0] == len(labelled)
    # Interleaved, not fast-block-then-slow-block: the per-point colours change
    # sign back and forth rather than being sorted by category.
    rgba = colls[0].get_facecolors()
    is_red = rgba[:, 0:3].sum(axis=1) < 1.5            # red ~ (1,0,0)
    flips = np.abs(np.diff(is_red.astype(int))).sum()
    assert flips > 5
    plt.close("all")


def test_scatter_shuffle_is_deterministic_in_seed():
    labelled = _labelled()
    a = plot_subject_qr_scatter(labelled, "S1", seed=7).collections[0]
    b = plot_subject_qr_scatter(labelled, "S1", seed=7).collections[0]
    assert np.array_equal(a.get_offsets(), b.get_offsets())
    plt.close("all")


def test_scatter_x_axis_is_reversed():
    ax = plot_subject_qr_scatter(_labelled(), "S1")
    assert ax.get_xlim() == (1.0, 0.0)     # reward-rate 1 left, 0 right
    plt.close("all")


def test_scatter_q_ylim_parameter_sets_y_range():
    ax = plot_subject_qr_scatter(_labelled(), "S1", q_ylim=(-0.4, 0.6))
    assert ax.get_ylim() == (-0.4, 0.6)
    plt.close("all")


def test_scatter_default_y_range_is_full():
    ax = plot_subject_qr_scatter(_labelled(), "S1")
    assert ax.get_ylim() == (-1.0, 1.0)
    plt.close("all")


def test_scatter_saves_svg(tmp_path):
    plot_subject_qr_scatter(_labelled(), "S1", save_prefix=str(tmp_path))
    assert (tmp_path / "FastSlowQR" / "S1.svg").exists()
    plt.close("all")


def test_all_subjects_scatter_pools_every_subject_into_one_collection():
    subject_dfs = {"S1": _labelled(0), "S2": _labelled(1), "S3": _labelled(2)}
    total = sum(len(d) for d in subject_dfs.values())
    ax = plot_all_subjects_qr_scatter(subject_dfs, point_size=3)
    colls = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(colls) == 1                              # single scatter artist
    assert colls[0].get_offsets().shape[0] == total     # every subject's points
    plt.close("all")


def test_all_subjects_scatter_shares_reversed_axis_and_ylim():
    subject_dfs = {"S1": _labelled(0), "S2": _labelled(1)}
    ax = plot_all_subjects_qr_scatter(subject_dfs, q_ylim=(-0.3, 0.7))
    assert ax.get_xlim() == (1.0, 0.0)                  # reward-rate 1 left
    assert ax.get_ylim() == (-0.3, 0.7)
    plt.close("all")


def test_all_subjects_scatter_saves_svg(tmp_path):
    subject_dfs = {"S1": _labelled(0), "S2": _labelled(1)}
    plot_all_subjects_qr_scatter(subject_dfs, save_prefix=str(tmp_path))
    assert (tmp_path / "FastSlowQR" / "all_subjects.svg").exists()
    plt.close("all")


def test_heatmap_renders_with_autumn_0_1_range():
    subject_dfs = {"S1": _labelled(0), "S2": _labelled(1)}
    ax = plot_pooled_fastness_heatmap(subject_dfs, n_bins_r=12, n_bins_q=15)
    im = ax.images[0]
    assert im.get_clim() == (0.0, 1.0)
    assert im.get_cmap().name == "autumn"
    assert im.get_array().shape == (15, 12)
    plt.close("all")


def test_heatmap_x_axis_is_reversed():
    subject_dfs = {"S1": _labelled(0), "S2": _labelled(1)}
    ax = plot_pooled_fastness_heatmap(subject_dfs, n_bins_r=12, n_bins_q=15)
    assert ax.get_xlim() == (1.0, 0.0)     # reward-rate 1 left, 0 right
    plt.close("all")


# --------------------------------------------------------------------------
# Three-way speed labelling
# --------------------------------------------------------------------------
def test_label_speeds_keeps_all_three_thirds():
    labelled = label_speeds(_trials(n_per_dvstr=30))
    assert set(labelled.Speed) == {"Fast", "Typical", "Slow"}
    assert len(labelled) == 90                         # nothing dropped
    for _dvstr, g in labelled.groupby("DVstr"):
        fast = g[g.Speed == "Fast"].calcStimulusTime
        typ = g[g.Speed == "Typical"].calcStimulusTime
        slow = g[g.Speed == "Slow"].calcStimulusTime
        assert fast.max() <= typ.min() and typ.max() <= slow.min()


# --------------------------------------------------------------------------
# Stacked speed histograms
# --------------------------------------------------------------------------
def _labelled3(seed=0):
    df = _trials(n_per_dvstr=40, seed=seed)
    df = add_q_relative(df)
    df["RewardRate"] = np.random.default_rng(seed).uniform(0, 1, len(df))
    return label_speeds(df)


def test_speed_fraction_hist_sums_to_one_per_nonempty_bin():
    pooled = pd.concat([_labelled3(0), _labelled3(1)], ignore_index=True)
    edges, fractions, totals = fsqr.speed_fraction_hist(
        pooled, "RewardRate", n_bins=8, value_range=(0.0, 1.0))
    assert len(edges) == 9
    stacked = np.vstack([fractions[s] for s in ("Fast", "Typical", "Slow")])
    nonempty = totals > 0
    assert np.allclose(stacked[:, nonempty].sum(axis=0), 1.0)
    assert np.isnan(stacked[:, ~nonempty]).all()       # empty bins -> NaN


def test_speed_fraction_hist_counts_match_totals():
    df = pd.DataFrame({
        "RewardRate": [0.1, 0.1, 0.1, 0.9],
        "Speed": ["Fast", "Typical", "Slow", "Fast"]})
    _edges, fractions, totals = fsqr.speed_fraction_hist(
        df, "RewardRate", n_bins=2, value_range=(0.0, 1.0))
    assert totals[0] == 3 and totals[1] == 1
    assert fractions["Fast"][0] == pytest.approx(1 / 3)
    assert fractions["Fast"][1] == pytest.approx(1.0)


def test_stacked_hist_three_series_reach_full_height():
    pooled = pd.concat([_labelled3(0), _labelled3(1)], ignore_index=True)
    ax = fsqr.plot_speed_stacked_hist(
        pooled, "RewardRate", n_bins=8, value_range=(0.0, 1.0))
    bars = list(ax.containers)                          # one BarContainer per speed
    assert len(bars) == 3
    # The summed bar heights per bin reach 1 wherever there are trials.
    heights = np.vstack([[p.get_height() for p in bc] for bc in bars])
    totals = fsqr.speed_fraction_hist(
        pooled, "RewardRate", n_bins=8, value_range=(0.0, 1.0))[2]
    assert np.allclose(heights.sum(axis=0)[totals > 0], 1.0)
    plt.close("all")


def test_stacked_hist_reverse_x_flips_reward_rate_axis():
    pooled = _labelled3(0)
    ax = fsqr.plot_speed_stacked_hist(
        pooled, "RewardRate", n_bins=6, value_range=(0.0, 1.0), reverse_x=True)
    assert ax.get_xlim() == (1.0, 0.0)
    plt.close("all")


def test_speed_histograms_reversed_R_natural_Q_and_saves(tmp_path):
    subject_dfs = {"S1": _labelled3(0), "S2": _labelled3(1)}
    fig, (ax_r, ax_q) = fsqr.plot_speed_histograms(
        subject_dfs, n_bins_r=10, n_bins_q=12, save_prefix=str(tmp_path))
    assert ax_r.get_xlim() == (1.0, 0.0)               # reward-rate reversed
    assert ax_q.get_xlim() == (-1.0, 1.0)              # Q-relative natural
    assert (tmp_path / "FastSlowQR_speed_histograms.svg").exists()
    plt.close("all")


def test_build_include_typical_keeps_middle_third():
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    out = build_fast_slow_qr(fits, min_num_trials=2500, include_typical=True,
                             verbose=False)
    assert set(out["Big"].Speed) == {"Fast", "Typical", "Slow"}


def test_build_bias_scaled_uses_mle_z_rotated_to_dv():
    """bias_scaled=True builds Q_relative from mle_z (= sign(DV) * mle_z)."""
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    out = build_fast_slow_qr(fits, min_num_trials=2500, include_typical=True,
                             bias_scaled=True, verbose=False)
    df = out["Big"]
    # Q_relative must equal sign(DV) * mle_z (DV != 0 in this synthetic data).
    expect = np.sign(df["DV"].to_numpy()) * df["mle_z"].to_numpy()
    assert df["Q_relative"].to_numpy() == pytest.approx(expect)
    # And it must NOT equal the raw-q version, so the flag really switched source.
    raw = build_fast_slow_qr(fits, min_num_trials=2500, include_typical=True,
                             bias_scaled=False, verbose=False)["Big"]
    assert not np.allclose(df["Q_relative"], raw["Q_relative"])


def test_build_bias_scaled_default_is_raw_q():
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    out = build_fast_slow_qr(fits, min_num_trials=2500, verbose=False)["Big"]
    expect = np.sign(out["DV"].to_numpy()) * out["mle_Q_rel_before"].to_numpy()
    assert out["Q_relative"].to_numpy() == pytest.approx(expect)


def test_build_bias_scaled_errors_without_mle_z():
    """A fit missing mle_z gives a clear error under bias_scaled."""
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    stored = fits[next(iter(fits))].subjects["Big"][0].payload["mle_df"]
    stored.drop(columns=["mle_z"], inplace=True)
    with pytest.raises(KeyError, match="mle_z"):
        build_fast_slow_qr(fits, min_num_trials=2500, bias_scaled=True,
                           verbose=False)


# --------------------------------------------------------------------------
# Data assembly (compare._compute_mle stubbed)
# --------------------------------------------------------------------------
def _subject_mle_df(name, n):
    """A stored per-trial mle_df like the fit saves: behavior cols + latents."""
    rng = np.random.default_rng(abs(hash(name)) % 1000)
    rows = []
    for i in range(n):
        dvstr, dv = (("Easy", 0.9), ("Med", 0.5), ("Hard", 0.2))[i % 3]
        q = rng.uniform(-1, 1)
        rows.append({"Name": name, "DVstr": dvstr, "DV": dv, "valid": True,
                     "calcStimulusTime": float(rng.gamma(2, 0.2) + 0.1),
                     "mle_Q_rel_before": q,
                     # z = clip(BIAS_COEF*q + offset, -1, 1); a plausible stand-in.
                     "mle_z": float(np.clip(0.6 * q + 0.05, -1, 1)),
                     "mle_reward_rate_before": rng.uniform(0, 1)})
    return pd.DataFrame(rows)


def _fits_with_subjects(subjects, *, store_mle_df=False):
    from ..mle_reeval import parse_fit_filename
    from ..compare import ColumnFit, ModelEntry
    fname = ("mle_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_"
             "4.8s_dt0.005_mleW1_chi2W0.5.pkl")
    fid = parse_fit_filename(fname)
    entry = ModelEntry(model_key=fid.model_key, label=fid.model_label,
                       fid_example=fid)
    for s, n in (subjects.items() if isinstance(subjects, dict)
                 else ((s, 3000) for s in subjects)):
        payload = {"params_names": ["DRIFT_COEF"],
                   "OptimRes": types.SimpleNamespace(x=np.array([1.0]), fun=1.0),
                   "include_Q": True, "include_RewardRate": True}
        if store_mle_df:
            payload["mle_df"] = _subject_mle_df(s, n)
        entry.subjects[s] = [ColumnFit(fid=fid, payload=payload, filename=fname,
                                       column_label="MLE=1, Chi²=0.5",
                                       order_rank=1.5)]
    return {fid.model_key: entry}


@pytest.fixture
def stub_mle(monkeypatch):
    def fake(subject, fid, payload, df_behavior, include_Q, include_RewardRate,
             terminal_c, lapse_override):
        sub = df_behavior[df_behavior.Name == subject].copy()
        sub["mle_Q_rel_before"] = np.linspace(-0.9, 0.9, len(sub))
        sub["mle_reward_rate_before"] = np.linspace(0.1, 0.9, len(sub))
        return sub, -1.0, 0.0, None
    monkeypatch.setattr(compare, "_compute_mle", fake)


def _behavior(subject_trials):
    rng = np.random.default_rng(0)
    rows = []
    for name, n in subject_trials.items():
        for i in range(n):
            dvstr, dv = (("Easy", 0.9), ("Med", 0.5), ("Hard", 0.2))[i % 3]
            rows.append({"Name": name, "DVstr": dvstr, "DV": dv, "valid": True,
                         "calcStimulusTime": float(rng.gamma(2, 0.2) + 0.1)})
    return pd.DataFrame(rows)


# --- reuse path (default): read the mle_df stored in the fit -------------
def test_reuse_needs_no_behavior_df_and_reads_stored_mle_df():
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    out = build_fast_slow_qr(fits, df_behavior=None, min_num_trials=2500,
                             reuse_fit_df=True, verbose=False)
    df = out["Big"]
    assert {"Q_relative", "RewardRate", "Speed"} <= set(df.columns)
    assert set(df.Speed) == {"Fast", "Slow"}
    assert df.RewardRate.between(0, 1).all()


def test_reuse_filters_subjects_below_min_trials(capsys):
    fits = _fits_with_subjects({"Big": 3000, "Small": 300}, store_mle_df=True)
    out = build_fast_slow_qr(fits, min_num_trials=2500, verbose=True)
    assert set(out) == {"Big"}
    assert "skip Small" in capsys.readouterr().out


def test_reuse_does_not_call_compute_mle(monkeypatch):
    """Reuse must not recompute — the whole point of the flag."""
    def boom(*a, **k):
        raise AssertionError("_compute_mle should not be called when reusing")
    monkeypatch.setattr(compare, "_compute_mle", boom)
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    out = build_fast_slow_qr(fits, verbose=False)
    assert set(out) == {"Big"}


def test_reuse_missing_stored_mle_df_raises_actionably():
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=False)
    with pytest.raises(KeyError, match="reuse_fit_df=False"):
        build_fast_slow_qr(fits, verbose=False)


# --- recompute path: re-evaluate params under MLE (needs df_behavior) -----
def test_recompute_requires_behavior_df():
    fits = _fits_with_subjects({"Big": 3000}, store_mle_df=True)
    with pytest.raises(ValueError, match="needs df_behavior"):
        build_fast_slow_qr(fits, df_behavior=None, reuse_fit_df=False)


def test_recompute_filters_and_builds(stub_mle):
    fits = _fits_with_subjects(["Big", "Small"])
    df_behavior = _behavior({"Big": 3000, "Small": 300})
    out = build_fast_slow_qr(fits, df_behavior, min_num_trials=2500,
                             reuse_fit_df=False, verbose=False)
    assert set(out) == {"Big"}
    df = out["Big"]
    assert {"Q_relative", "RewardRate", "Speed"} <= set(df.columns)
    assert set(df.Speed) == {"Fast", "Slow"}


def test_reuse_and_recompute_agree_when_latents_match(monkeypatch):
    """Reuse and recompute are the same downstream computation; on identical
    latents they must yield identical frames — reuse just skips the re-eval."""
    stored = _subject_mle_df("Big", 3000)
    fits = _fits_with_subjects(["Big"])
    fits[next(iter(fits))].subjects["Big"][0].payload["mle_df"] = stored
    reused = build_fast_slow_qr(fits, reuse_fit_df=True, verbose=False)

    monkeypatch.setattr(compare, "_compute_mle",
                        lambda *a, **k: (stored.copy(), -1.0, 0.0, None))
    recomputed = build_fast_slow_qr(fits, df_behavior="unused",
                                    reuse_fit_df=False, verbose=False)

    pd.testing.assert_frame_equal(
        reused["Big"].reset_index(drop=True),
        recomputed["Big"].reset_index(drop=True))


def test_chi2_w05_spec_targets_the_weighted_mle_fit():
    assert CHI2_W05_SPEC.column_label == "MLE=1, Chi²=0.5"
    assert "Q-Val (Offset)" in CHI2_W05_SPEC.model_key
