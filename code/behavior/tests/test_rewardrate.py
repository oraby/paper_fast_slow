'''Tests for the reward-rate helpers (Figures 1H, S3B and the per-timeout view).

- ``calcAvgRewardRate`` -- the share of correct choices over the previous
  trials of the same session, never the current one;
- ``prepareMiceRewardRate`` / ``keepWellSampledRewardRates`` -- the trial
  selection the behaviour notebook used to do inline, kept identical;
- ``plotRewardRateByTimeout`` -- S3B once per error timeout. It goes through
  the per-timeout branch of ``_plotLoopResults``, which used to raise a
  TypeError on its first use (a numeric timeout added to a string label).
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import rewardrate as rr


def mice(seed=0):
    """Three mice, three sessions each; m1 runs 0 s timeouts, m2 and m3 4 s,
    and m3's short last session 6 s."""
    rng = np.random.default_rng(seed)
    rows = []
    plan = {"m1": [(0., 80)] * 3, "m2": [(4., 80)] * 3,
            "m3": [(4., 80), (4., 80), (6., 12)]}
    for name, sessions in plan.items():
        for sess_num, (timeout, num_trials) in enumerate(sessions, 1):
            for trial in range(1, num_trials + 1):
                rows.append(dict(Name=name, Date=20200101, SessionNum=sess_num,
                                 TrialNumber=trial,
                                 ChoiceCorrect=float(rng.random() < .75),
                                 calcStimulusTime=rng.gamma(4, .25),
                                 DV=rng.choice([-.5, -.2, .2, .5]),
                                 GUI_TimeOutIncorrectChoice=timeout))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def test_reward_rate_is_the_mean_of_the_previous_five_trials():
    df = rr.calcAvgRewardRate(mice())
    sess = df[(df.Name == "m1") & (df.SessionNum == 1)].reset_index(drop=True)
    assert sess.RewardRate5.iloc[:5].isnull().all()
    for idx in (5, 17, 79):
        assert sess.RewardRate5[idx] == pytest.approx(
            sess.ChoiceCorrect[idx - 5:idx].mean())


def test_reward_rate_restarts_with_every_session():
    df = rr.calcAvgRewardRate(mice())
    firsts = df.groupby(rr.SESSION_COLS).head(5)
    assert firsts.RewardRate5.isnull().all()


def test_a_session_with_a_missing_coherence_is_dropped():
    df = mice()
    df.loc[(df.Name == "m2") & (df.SessionNum == 2) &
           (df.TrialNumber == 40), "DV"] = np.nan
    out = rr.prepareMiceRewardRate(df)
    assert not ((out.Name == "m2") & (out.SessionNum == 2)).any()
    assert ((out.Name == "m2") & (out.SessionNum == 1)).any()


def test_the_selection_matches_what_the_notebook_did_inline():
    df = mice()
    # The behaviour notebook's reward-rate cell, before it moved here.
    inline = df.copy().sort_values(by=["Name", "Date", "SessionNum", "TrialNumber"])
    inline = rr.calcAvgRewardRate(inline)
    inline = inline.groupby(["Name", "Date", "SessionNum"]).filter(
        lambda sess_df: sess_df.DV.isnull().sum() == 0)
    num_subjects_in_1st = inline[inline["RewardRate5"] == 1].Name.nunique()
    inline = inline.groupby("RewardRate5").filter(
        lambda rr_df: rr_df.Name.nunique() >= .75 * num_subjects_in_1st)

    moved = rr.keepWellSampledRewardRates(rr.prepareMiceRewardRate(df))
    pd.testing.assert_frame_equal(moved, inline)


def test_rare_reward_rates_are_dropped():
    df = rr.calcAvgRewardRate(mice())
    df = df[~((df.RewardRate5 == 0) & (df.Name != "m1"))]
    kept = rr.keepWellSampledRewardRates(df)
    assert 0 not in set(kept.RewardRate5)
    assert kept.RewardRate5.notnull().all()


def test_one_timeout_can_be_drawn_on_a_shared_axis():
    df = rr.addSessId(rr.prepareMiceRewardRate(mice()))
    fig, ax = plt.subplots()
    rr.loopRewardRateAnalysis(df, plot_subjects=False, plot_singles_on_all=True,
                              ax_all=ax, ax_all_used_timeout_time=4.,
                              min_num_trials=0, run_stats=False)
    assert ax.get_legend_handles_labels()[1][0].startswith("4.0")


def test_one_panel_per_timeout_above_the_share():
    df = rr.keepWellSampledRewardRates(rr.prepareMiceRewardRate(mice()))
    fig, share = rr.plotRewardRateByTimeout(df, min_trial_share=.05)
    titles = [ax.get_title() for ax in fig.axes]
    assert share[6.] < .05
    assert len(titles) == 2
    assert titles[0].startswith("Error timeout 0 s\n(1 mice")
    assert titles[1].startswith("Error timeout 4 s\n(2 mice")


def test_the_panels_carry_no_per_point_labels():
    df = rr.keepWellSampledRewardRates(rr.prepareMiceRewardRate(mice()))
    fig, _ = rr.plotRewardRateByTimeout(df)
    assert all(not ax.texts for ax in fig.axes)


def test_the_pooled_figure_keeps_its_per_point_labels():
    df = rr.addSessId(rr.prepareMiceRewardRate(mice()))
    fig, ax = plt.subplots()
    rr.loopRewardRateAnalysis(df, plot_subjects=False, plot_singles_on_all=False,
                              ax_all=ax, ax_all_used_timeout_time="All",
                              min_num_trials=0, run_stats=False)
    assert ax.texts


def test_saving_writes_one_svg(tmp_path):
    df = rr.keepWellSampledRewardRates(rr.prepareMiceRewardRate(mice()))
    rr.plotRewardRateByTimeout(df, save_figs=True, save_prefix=str(tmp_path))
    assert [p.name for p in tmp_path.iterdir()] == ["reward_rate5_by_timeout.svg"]
