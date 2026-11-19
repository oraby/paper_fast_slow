'''Tests for the per-subject model-fit figures (Figures 2E, 2F and S4A).

The drawing itself (``plotPlots``, ``loopRewardRateAnalysis``) is stubbed: what
is under test is which subjects get drawn and where the files land, since the
paper's example and the full loop share these paths.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from .. import subjectfits as sf


MODELS = ["Classic", "Q-Val", "RewardRate", "RewardRate\n+ Q-Val"]
SUBJECTS = ["Avgat1", "Avgat3", "vgat-40"]


def _frame(subject):
    return pd.DataFrame({"Name": [subject] * 3, "Date": [20200101] * 3,
                         "SessionNum": [1] * 3})


@pytest.fixture
def sim_cache():
    return {model: {subject: (1.5, _frame(subject), 1.0, True, False)
                    for subject in SUBJECTS}
            for model in MODELS}


@pytest.fixture
def behavior():
    rng = np.random.default_rng(0)
    return pd.concat([_frame(s).assign(x=rng.random(3)) for s in SUBJECTS],
                     ignore_index=True)


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    drawn = {"fits": [], "rr": []}

    def createFig(is_small_fig_mode):
        fig, ax = plt.subplots()
        return fig, [ax]

    def plotPlots(df, axs, **kwargs):
        drawn["fits"].append(df.Name.iloc[0])

    def loopRewardRateAnalysis(df, descrp, **kwargs):
        drawn["rr"].append((descrp, tuple(sorted(df.Name.unique()))))

    monkeypatch.setattr(sf, "createFig", createFig)
    monkeypatch.setattr(sf, "plotPlots", plotPlots)
    monkeypatch.setattr(sf, "loopRewardRateAnalysis", loopRewardRateAnalysis)
    monkeypatch.setattr(plt, "show", lambda: None)
    yield drawn
    plt.close("all")


def test_the_example_is_avgat3():
    assert sf.EXAMPLE_SUBJECT == "Avgat3"


def test_the_example_draws_one_subject_every_model(sim_cache, stubs, tmp_path):
    saved = sf.plotSubjectFits(sim_cache, save_prefix=str(tmp_path),
                               subjects=["Avgat3"], save_figs=True,
                               plt_show=True)
    assert stubs["fits"] == ["Avgat3"] * len(MODELS)
    names = sorted(p.relative_to(tmp_path).as_posix() for p in saved)
    assert names == sorted(f"figs/Avgat3/{sf.safe_filename(m)}.svg"
                           for m in MODELS)
    assert all(p.exists() for p in saved)


def test_the_published_file_names(sim_cache, tmp_path):
    # results/RLModel/figs/{subject}/ holds exactly these four names.
    saved = sf.plotSubjectFits(sim_cache, save_prefix=str(tmp_path),
                               subjects=["Avgat3"], save_figs=True)
    assert sorted(p.name for p in saved) == [
        "Classic.svg", "Q-Val.svg", "RewardRate + Q-Val.svg", "RewardRate.svg"]


def test_the_loop_draws_every_subject(sim_cache, stubs):
    sf.plotSubjectFits(sim_cache, plt_show=True)
    assert sorted(stubs["fits"]) == sorted(SUBJECTS * len(MODELS))


def test_nothing_is_saved_without_save_figs(sim_cache, tmp_path):
    assert sf.plotSubjectFits(sim_cache, save_prefix=str(tmp_path)) == []
    assert not any(tmp_path.iterdir())


def test_the_reward_rate_example_is_one_subject(sim_cache, behavior, stubs,
                                                tmp_path):
    saved = sf.plotRewardRates(sim_cache, behavior, min_num_trials=0,
                               min_trials_per_sess_rr=0, subjects=["Avgat3"],
                               plot_pooled=False, save_prefix=str(tmp_path),
                               save_figs=True)
    assert [p.relative_to(tmp_path).as_posix() for p in saved] == [
        "RewardRate/subject_Avgat3.svg"]
    # The mouse, then each model -- all on Avgat3's trials only.
    assert stubs["rr"] == ([("Behavior", ("Avgat3",))] +
                           [(m, ("Avgat3",)) for m in MODELS])


def test_the_reward_rate_loop_adds_the_pooled_panel(sim_cache, behavior,
                                                    tmp_path):
    saved = sf.plotRewardRates(sim_cache, behavior, min_num_trials=0,
                               min_trials_per_sess_rr=0,
                               save_prefix=str(tmp_path), save_figs=True)
    assert sorted(p.name for p in saved) == sorted(
        ["all_subjects.svg"] + [f"subject_{s}.svg" for s in SUBJECTS])


def test_subjects_below_the_trial_floor_are_dropped(sim_cache, behavior,
                                                    stubs):
    behavior = pd.concat([behavior, _frame("Avgat3")], ignore_index=True)
    sf.plotRewardRates(sim_cache, behavior, min_num_trials=4,
                       min_trials_per_sess_rr=0, plot_pooled=False)
    assert {names for _, names in stubs["rr"]} == {("Avgat3",)}
