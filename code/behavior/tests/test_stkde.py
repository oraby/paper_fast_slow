'''Tests for the raw-seconds sampling-time KDE figures (mice / humans / both).

The session-collection pass is the part with real logic -- RDK performance
filter, sampling-time cap, last-N sessions, subject acceptance -- so it gets the
bulk of the tests; the plotting entry points get smoke + save-path coverage and
one check that the combined figure draws exactly the requested mouse groups.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..stkde import (COMBINED_MOUSE_GROUPS, MOUSE_GROUPS, MOUSE_LABELS,
                     collectFMHFSessions, humanSubjectGroups,
                     humanSubjectPrintStr, mouseSubjectGroups,
                     plotFMHFSubjects, plotHumanSubjects, plotMiceVsHumansST,
                     plotMouseHumanSTKde, plotSeriesGroups, plotSTGroup,
                     poolST, zBinsAndLim, zScorePooled)
from ..stkde import MOUSE_COLORS
from ...common.definitions import ExperimentType, MouseState

N_TRIALS = 60
HUMAN_COLORS = {"Accuracy Competition": "C0",
                "Maximizing Correct-Outcome Competition": "C1"}


def _sessDf(name, date, sess_num, exp_type, state, st_mean=1.0, easy_perf=1.0,
            n_trials=N_TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    # Half easy (|DV| >= .8) trials, whose accuracy drives the RDK filter.
    dv = np.tile([1.0, 0.1], n_trials//2)
    correct = np.where(np.abs(dv) >= .8,
                       rng.random(n_trials) < easy_perf, True).astype(float)
    return pd.DataFrame({
        "Name": name, "Date": date, "SessionNum": sess_num,
        "GUI_ExperimentType": int(exp_type), "GUI_MouseState": float(state),
        "DV": dv, "ChoiceCorrect": correct,
        "calcStimulusTime": np.abs(rng.normal(st_mean, .2, n_trials)),
        "GUI_MinSampleMin": 0.3, "GUI_MinSampleMax": 0.3, "MinSample": 0.3})


def _miceDf(n_sess=2, rdk_easy_perf=1.0):
    dfs = []
    for i in range(n_sess):
        dfs += [
            _sessDf("m1", f"2024-01-{i+1:02d}", 1, ExperimentType.LightIntensity,
                    MouseState.FreelyMoving, st_mean=0.6, seed=i),
            _sessDf("m1", f"2024-02-{i+1:02d}", 1, ExperimentType.RDK,
                    MouseState.HeadFixed, st_mean=2.0,
                    easy_perf=rdk_easy_perf, seed=10 + i),
            _sessDf("m1", f"2024-03-{i+1:02d}", 1, ExperimentType.LightIntensity,
                    MouseState.HeadFixed, st_mean=1.5, seed=20 + i),
            _sessDf("m1", f"2024-04-{i+1:02d}", 1, ExperimentType.RDK,
                    MouseState.FreelyMoving, st_mean=1.8,
                    easy_perf=rdk_easy_perf, seed=30 + i)]
    return pd.concat(dfs, ignore_index=True)


def _humansDf():
    rng = np.random.default_rng(2)
    dfs = []
    for name in ["S1", "S2"]:
        for sess_type, st_mean, sess_num in [("ReactionTime", 1.6, 1),
                                             ("Competition", 0.7, 2)]:
            dfs.append(pd.DataFrame({
                "Name": name, "Date": "2024-02-07", "SessionNum": sess_num,
                "session_type": sess_type,
                "calcStimulusTime": np.abs(rng.normal(st_mean, .3, N_TRIALS))}))
    return pd.concat(dfs, ignore_index=True)


def test_collect_keys_groups_by_experiment_and_state():
    accepted = collectFMHFSessions(_miceDf(), min_sess_per_state=1,
                                   verbose=False)

    assert list(accepted) == ["m1"]
    groups = {k: v for k, v in accepted["m1"].items()
              if k != "PrintStr" and v is not None}
    assert set(groups) == set(MOUSE_GROUPS)
    assert len(groups["LightIntensityFreelyMoving"]) == 2*N_TRIALS
    assert "m1" in accepted["m1"]["PrintStr"]


def test_collect_drops_rdk_sessions_below_the_easy_perf_cut_off():
    df = _miceDf(rdk_easy_perf=0.5)     # 50% on easy trials -> below 75
    accepted = collectFMHFSessions(df, min_sess_per_state=1, verbose=False)
    # Every RDK session is dropped, so those groups fall under the minimum and
    # the subject is rejected outright.
    assert accepted == {}

    kept = collectFMHFSessions(df, min_sess_per_state=1, min_rdk_easy_perf=0,
                               verbose=False)
    assert kept["m1"]["RDKHeadFixed"] is not None


def test_collect_drops_trials_above_the_sampling_time_cap():
    df = _miceDf()
    df.loc[df.index[:5], "calcStimulusTime"] = 9.9
    accepted = collectFMHFSessions(df, min_sess_per_state=1, verbose=False)
    st = accepted["m1"]["LightIntensityFreelyMoving"].calcStimulusTime
    assert (st <= 4.9).all()
    assert len(st) == 2*N_TRIALS - 5


def test_collect_keeps_only_the_last_sessions():
    accepted = collectFMHFSessions(_miceDf(n_sess=12), min_sess_per_state=1,
                                   max_sess=10, verbose=False)
    kept = accepted["m1"]["LightIntensityFreelyMoving"]
    assert len(kept) == 10*N_TRIALS
    # The last 10 dates survive, the first two are dropped.
    assert "2024-01-01" not in set(kept.Date)
    assert "2024-01-12" in set(kept.Date)


def test_collect_rejects_a_subject_with_too_few_sessions_in_a_group():
    assert collectFMHFSessions(_miceDf(n_sess=2), min_sess_per_state=3,
                               verbose=False) == {}


def test_plot_st_group_labels_with_mean_sem_and_n():
    fig, ax = plt.subplots()
    st = pd.Series([1.0, 1.0, 2.0, 2.0])
    plotSTGroup(st, ax=ax, color="C0", label="Grp")
    label = ax.get_legend_handles_labels()[1][0]
    assert label.startswith("Grp - RT: 1.50s ±0.29s SEM")
    assert "(n=4 trials)" in label
    plt.close(fig)


def test_mouse_subject_groups_returns_only_the_requested_groups():
    accepted = collectFMHFSessions(_miceDf(), min_sess_per_state=1,
                                   verbose=False)
    group_st = mouseSubjectGroups(accepted["m1"], groups=COMBINED_MOUSE_GROUPS)
    assert list(group_st) == list(COMBINED_MOUSE_GROUPS) != list(MOUSE_GROUPS)

    fig, ax = plt.subplots()
    n_plotted = plotSeriesGroups(group_st, ax=ax, colors=MOUSE_COLORS,
                                 labels=MOUSE_LABELS)
    labels = ax.get_legend_handles_labels()[1]

    assert n_plotted == len(COMBINED_MOUSE_GROUPS)
    assert [l.split(" - RT")[0] for l in labels] == [
        MOUSE_LABELS[group] for group in COMBINED_MOUSE_GROUPS]
    plt.close(fig)


def test_human_subject_groups_are_raw_seconds_per_experiment_type():
    groups = humanSubjectGroups(_humansDf(), "S1")
    assert list(groups) == ["Accuracy Competition",
                            "Maximizing Correct-Outcome Competition"]
    assert groups["Accuracy Competition"].mean() == pytest.approx(1.6, abs=0.2)
    assert len(groups["Accuracy Competition"]) == N_TRIALS
    assert humanSubjectGroups(_humansDf(), "nobody") == {}


def test_human_subject_groups_respect_the_sampling_time_cap():
    df = _humansDf()
    df.loc[df.index[:3], "calcStimulusTime"] = 9.9
    assert len(humanSubjectGroups(df, "S1")["Accuracy Competition"]) == N_TRIALS - 3
    assert len(humanSubjectGroups(df, "S1", max_st=None)[
                                        "Accuracy Competition"]) == N_TRIALS


def test_human_print_str_lists_each_session():
    print_str = humanSubjectPrintStr(_humansDf(), "S1")
    assert print_str.startswith("S1\n")
    assert print_str.count("2024-02-07/") == 2      # one line per session


def test_plot_human_subjects_makes_and_saves_one_figure_per_subject(tmp_path):
    figs = plotHumanSubjects(_humansDf(), colors=HUMAN_COLORS,
                             save_prefix=str(tmp_path), save_figs=True)
    assert sorted(figs) == ["S1", "S2"]
    assert len(figs["S1"].axes[0].get_legend_handles_labels()[1]) == 2
    saved = sorted(fp.name for fp in (tmp_path/"humans_st").iterdir())
    assert saved == ["stimulus_time_S1.svg", "stimulus_time_S2.svg"]
    plt.close("all")


def test_plot_fmhf_subjects_saves_under_the_mouse_subdir(tmp_path):
    figs = plotFMHFSubjects(_miceDf(), min_sess_per_state=1, verbose=False,
                            save_prefix=str(tmp_path), save_figs=True)
    assert list(figs) == ["m1"]
    saved = [fp.name for fp in (tmp_path/"fm_hf").iterdir()]
    assert saved == ["stimulus_time_m1.svg"]
    plt.close("all")


def test_combined_figure_holds_both_species_curves(tmp_path):
    fig = plotMouseHumanSTKde(_miceDf(), _humansDf(), mouse_subject="m1",
                              human_subject="S1", human_colors=HUMAN_COLORS,
                              save_prefix=str(tmp_path), save_figs=True)
    labels = fig.axes[0].get_legend_handles_labels()[1]

    assert len(labels) == len(COMBINED_MOUSE_GROUPS) + 2
    assert [l.split(" - RT")[0] for l in labels] == (
        [f"Mouse m1: {MOUSE_LABELS[group]}" for group in COMBINED_MOUSE_GROUPS]
        + ["Human S1: Accuracy Competition",
           "Human S1: Maximizing Correct-Outcome Competition"])
    assert "s ±" in labels[0]                    # raw seconds by default
    assert fig.axes[0].get_xlabel() == "Stimulus Time (s)"
    assert (tmp_path/"fm_hf"/"stimulus_time_mouse_m1_human_S1.svg").exists()
    plt.close("all")


def test_zscore_pooled_normalizes_the_pool_not_each_series():
    groups = {"a": pd.Series([1.0, 2.0, 3.0]), "b": pd.Series([10.0, 11.0])}
    zscored = zScorePooled(groups)

    pooled = pd.concat(list(zscored.values()))
    assert pooled.mean() == pytest.approx(0, abs=1e-12)
    assert pooled.std(ddof=0) == pytest.approx(1, abs=1e-12)
    # Each series keeps its offset within the pool -- neither is centred itself.
    assert zscored["a"].mean() < -0.5 < 0.5 < zscored["b"].mean()
    # ... and their spacing is just the raw one, rescaled.
    raw_gap = groups["b"].mean() - groups["a"].mean()
    assert (zscored["b"].mean() - zscored["a"].mean()) == pytest.approx(
        raw_gap / pd.concat(list(groups.values())).std(ddof=0))


def test_z_bins_cover_every_value_at_the_fixed_width():
    groups = {"a": pd.Series([-1.23, 0.0]), "b": pd.Series([2.71])}
    bins, xlim = zBinsAndLim(groups)
    assert xlim[0] <= -1.23 and xlim[1] >= 2.71
    assert np.allclose(np.diff(bins), 0.1)
    assert bins[0] == pytest.approx(xlim[0]) and bins[-1] == pytest.approx(xlim[1])


def test_combined_figure_zscores_each_species_against_its_own_pool(tmp_path):
    fig = plotMouseHumanSTKde(_miceDf(), _humansDf(), mouse_subject="m1",
                              human_subject="S1", human_colors=HUMAN_COLORS,
                              zscore=True, save_prefix=str(tmp_path),
                              save_figs=True)
    ax = fig.axes[0]
    labels = ax.get_legend_handles_labels()[1]
    n_mouse = len(COMBINED_MOUSE_GROUPS)

    # The mean of each species' pool is 0, so its curves' means straddle it
    # (weighted by trial count); the units suffix is dropped from the legend.
    means = [float(l.split("RT: ")[1].split(" ±")[0]) for l in labels]
    assert min(means[:n_mouse]) < 0 < max(means[:n_mouse])
    assert min(means[n_mouse:]) < 0 < max(means[n_mouse:])
    assert "s ±" not in labels[0]
    assert ax.get_xlabel() == "Within-Species Z-Scored Stimulus Time"
    assert "Z-Scored" in ax.get_title()
    assert (tmp_path/"fm_hf" /
            "stimulus_time_mouse_m1_human_S1_zscored.svg").exists()
    plt.close("all")


def test_combined_figure_zscore_pools_are_independent():
    '''Rescaling one species must not move the other's curves.'''
    df_users = _humansDf()
    fig = plotMouseHumanSTKde(_miceDf(), df_users, mouse_subject="m1",
                              human_subject="S1", human_colors=HUMAN_COLORS,
                              zscore=True)
    human_means = [float(l.split("RT: ")[1].split(" ±")[0])
                   for l in fig.axes[0].get_legend_handles_labels()[1]
                   ][len(COMBINED_MOUSE_GROUPS):]

    # Halved, not inflated -- scaling up would push trials past the 4.9s cap
    # and drop them, changing more than just the mouse pool's scale.
    slow_mice = _miceDf()
    slow_mice["calcStimulusTime"] *= 0.5
    fig2 = plotMouseHumanSTKde(slow_mice, df_users, mouse_subject="m1",
                               human_subject="S1", human_colors=HUMAN_COLORS,
                               zscore=True)
    human_means2 = [float(l.split("RT: ")[1].split(" ±")[0])
                    for l in fig2.axes[0].get_legend_handles_labels()[1]
                    ][len(COMBINED_MOUSE_GROUPS):]

    assert human_means == human_means2
    plt.close("all")


def test_combined_figure_complains_about_a_missing_subject():
    with pytest.raises(AssertionError):
        plotMouseHumanSTKde(_miceDf(), _humansDf(), mouse_subject="nobody",
                            human_subject="S1", human_colors=HUMAN_COLORS)
    with pytest.raises(AssertionError):
        plotMouseHumanSTKde(_miceDf(), _humansDf(), mouse_subject="m1",
                            human_subject="nobody", human_colors=HUMAN_COLORS)
    plt.close("all")


def test_fmhf_zscore_pools_each_subject_against_its_own_trials(tmp_path):
    figs = plotFMHFSubjects(_miceDf(), min_sess_per_state=1, verbose=False,
                            zscore=True, save_prefix=str(tmp_path),
                            save_figs=True)
    ax = figs["m1"].axes[0]
    labels = ax.get_legend_handles_labels()[1]
    means = [float(l.split("RT: ")[1].split(" ±")[0]) for l in labels]

    # One pool for the subject: its groups straddle 0 rather than each sitting
    # at 0, and the seconds unit is dropped.
    assert min(means) < 0 < max(means)
    assert "s ±" not in labels[0]
    assert ax.get_xlabel() == "Within-Subject Z-Scored Stimulus Time"
    assert "Z-Scored" in ax.get_title()
    saved = [fp.name for fp in (tmp_path/"fm_hf").iterdir()]
    assert saved == ["stimulus_time_m1_zscored.svg"]
    plt.close("all")


def test_fmhf_zscore_matches_a_hand_computed_pool():
    figs = plotFMHFSubjects(_miceDf(), min_sess_per_state=1, verbose=False,
                            zscore=True)
    accepted = collectFMHFSessions(_miceDf(), min_sess_per_state=1,
                                   verbose=False)
    raw = mouseSubjectGroups(accepted["m1"])
    pooled = pd.concat(list(raw.values()))
    expected = [((st.mean() - pooled.mean())/pooled.std(ddof=0))
                for st in raw.values()]

    means = [float(l.split("RT: ")[1].split(" ±")[0])
             for l in figs["m1"].axes[0].get_legend_handles_labels()[1]]
    assert means == [pytest.approx(v, abs=0.005) for v in expected]
    plt.close("all")


def test_fmhf_raw_and_zscored_save_side_by_side(tmp_path):
    for zscore in (False, True):
        plotFMHFSubjects(_miceDf(), min_sess_per_state=1, verbose=False,
                         zscore=zscore, save_prefix=str(tmp_path),
                         save_figs=True)
    saved = sorted(fp.name for fp in (tmp_path/"fm_hf").iterdir())
    assert saved == ["stimulus_time_m1.svg", "stimulus_time_m1_zscored.svg"]
    plt.close("all")


def test_pool_st_drops_nulls_and_caps():
    df = pd.DataFrame({"calcStimulusTime": [1.0, np.nan, 9.9, 2.0]})
    assert list(poolST(df)) == [1.0, 2.0]
    assert list(poolST(df, max_st=None)) == [1.0, 9.9, 2.0]


def test_mice_vs_humans_pools_every_subject_into_one_curve(tmp_path):
    mice = _miceDf()
    humans = _humansDf()
    colors = {"Mice - RDK": "C0", "Humans - Speed Context": "C1"}
    fig = plotMiceVsHumansST(mice, humans, colors=colors,
                             save_prefix=str(tmp_path), save_figs=True)
    ax = fig.axes[0]
    labels = ax.get_legend_handles_labels()[1]

    assert len(labels) == 2
    assert labels[0].startswith("Mice - RDK (n=1 subjects)")
    assert labels[1].startswith("Humans - Speed Context (n=2 subjects)")
    # The human curve is the Competition sessions of both subjects pooled.
    assert f"(n={2*N_TRIALS:,} trials)" in labels[1]
    assert "s ±" in labels[0]                    # raw seconds, never z-scored
    assert ax.get_xlabel() == "Stimulus Time (s)"
    assert (tmp_path/"stimulus_time_mice_vs_humans.svg").exists()
    plt.close("all")


def test_mice_vs_humans_picks_the_requested_human_session_type():
    humans = _humansDf()
    colors = {"Mice - RDK": "C0", "Humans - Speed Context": "C1"}
    fig = plotMiceVsHumansST(_miceDf(), humans, colors=colors,
                             human_session_type="ReactionTime")
    human_mean = float(fig.axes[0].get_legend_handles_labels()[1][1]
                       .split("RT: ")[1].split("s ±")[0])
    expected = humans[humans.session_type == "ReactionTime"].calcStimulusTime
    assert human_mean == pytest.approx(expected.mean(), abs=0.005)
    plt.close("all")


def test_mice_vs_humans_complains_when_a_side_is_empty():
    colors = {"Mice - RDK": "C0", "Humans - Speed Context": "C1"}
    with pytest.raises(AssertionError):
        plotMiceVsHumansST(_miceDf(), _humansDf(), colors=colors,
                           human_session_type="NoSuchSession")
    plt.close("all")
