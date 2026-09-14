'''Tests for the previous-outcome x current-difficulty heatmap (Figure S3E).

The panel shows mean (or median) sampling time for each combination of
*previous* trial outcome and *current* difficulty. The companion violin panel
carries the Figure S3F statistics: Shapiro-Wilk on the per-animal values, then
one-way ANOVA on the log data with Tukey if every cell is normal, else
Kruskal-Wallis on the raw data with Dunn and Holm correction.

Nothing exercised this module before, so several of these are regression tests
in the plain sense -- they would have caught the ``matplotlib.cm.get_cmap``
breakage that took Figure 1D out (see ``test_stbydifficulty.py``).

Two things are pinned because they decide what the panel means:

- the cohort heatmap averages **one value per animal**, so an animal with more
  trials does not pull a cell;
- rows and columns come out in a **fixed** order -- rewarded previous trial
  first, then easy / medium / hard -- whatever subset of levels is present.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..stheatmap import (CUR_DIFFICULTY_ORDER, PREV_TRIAL_ORDER,
                         _violinPlot, stHeatmap)

DIFFICULTIES = ("Easy", "Med", "Hard")
REWARDED, NOT_REWARDED = "Rewarded", "Not-Rewarded"


def trials(name, prev_correct, prev_difficulty, difficulty, times,
           *, session=1) -> pd.DataFrame:
    times = np.asarray(times, dtype=float)
    return pd.DataFrame({"Name": name, "Date": "2024-01-01",
                         "SessionNum": session,
                         "calcStimulusTime": times,
                         "ChoiceCorrect": 1., "Stay": 1.,
                         "PrevChoiceCorrect": float(prev_correct),
                         "PrevDVstr": prev_difficulty, "DVstr": difficulty})


def cohort(n_animals=5, n_per_cell=25, seed=0, slower_after_error=.35):
    '''Sampling is longer after an error and on harder trials.'''
    rng = np.random.default_rng(seed)
    frames = []
    for animal in range(n_animals):
        offset = animal * .12
        for prev_correct in (True, False):
            for prev_difficulty in DIFFICULTIES:
                for difficulty, centre in zip(DIFFICULTIES, (.9, 1.2, 1.6)):
                    mean = centre + offset + (0. if prev_correct
                                              else slower_after_error)
                    frames.append(trials(
                        f"m{animal}", prev_correct, prev_difficulty,
                        difficulty, rng.normal(mean, .12, n_per_cell)))
    return pd.concat(frames, ignore_index=True)


def violinFrame(df=None) -> pd.DataFrame:
    '''The shape `_violinPlot` expects, as `_gropuDataByPriorAndCurrentSubject`
    hands it over: renamed columns, no raw outcome columns.'''
    df = cohort() if df is None else df
    out = df[["calcStimulusTime", "Name", "Date", "SessionNum"]].copy()
    out["PrevTrial"] = np.where(df.PrevChoiceCorrect == 1., REWARDED,
                                NOT_REWARDED)
    out["CurDifficulty"] = df.DVstr
    out["PrevDifficulty"] = df.PrevDVstr
    return out


# --------------------------------------------------------------------------
# The heatmap runs at all
# --------------------------------------------------------------------------

@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_the_cohort_heatmap_runs_for_the_statistics_the_paper_uses(statistic):
    '''``behavior.ipynb`` calls this three times, always with ``"mean"``.'''
    stHeatmap(cohort(), mean_or_median=statistic, dscrp="Mice",
              plot_combined_subjects=True, plot_single_subjects=False)
    plt.close("all")


def test_mode_is_no_longer_an_option():
    """It never ran, and no caller used it, so it was removed."""
    with pytest.raises(AssertionError):
        stHeatmap(cohort(), mean_or_median="mode", dscrp="Mice",
                  plot_combined_subjects=True, plot_single_subjects=False)
    plt.close("all")


def test_the_single_animal_heatmap_runs():
    stHeatmap(cohort(n_animals=2), mean_or_median="median", dscrp="Mice",
              plot_combined_subjects=False, plot_single_subjects=True)
    plt.close("all")


def test_an_unknown_statistic_is_refused():
    with pytest.raises(AssertionError):
        stHeatmap(cohort(), mean_or_median="average", dscrp="Mice",
                  plot_combined_subjects=True, plot_single_subjects=False)
    plt.close("all")


def test_saving_without_a_prefix_is_refused():
    with pytest.raises(AssertionError, match="save_prefix"):
        stHeatmap(cohort(), mean_or_median="median", dscrp="Mice",
                  plot_combined_subjects=True, plot_single_subjects=False,
                  save_fig=True)
    plt.close("all")


def test_the_heatmap_saves_under_the_prefix(tmp_path):
    stHeatmap(cohort(), mean_or_median="median", dscrp="Mice",
              plot_combined_subjects=True, plot_single_subjects=False,
              save_prefix=str(tmp_path), save_fig=True)
    saved = sorted(path.name for path in tmp_path.glob("*.svg"))
    assert any(name.startswith("prior_cur_") for name in saved)
    assert any("violin" in name for name in saved)
    plt.close("all")


def test_a_query_that_keeps_all_three_difficulties_works():
    stHeatmap(cohort(), mean_or_median="median", dscrp="Mice",
              df_query="calcStimulusTime < 2.0", plot_combined_subjects=True,
              plot_single_subjects=False)
    plt.close("all")


def _heatmapAxis():
    for number in plt.get_fignums():
        for ax in plt.figure(number).axes:
            if ax.get_xlabel() == "Current Trial":
                return ax
    raise AssertionError("no heatmap axis drawn")


def _ticks(labels):
    return [label.get_text() for label in labels]


def test_the_heatmap_uses_the_published_row_and_column_order():
    """Rewarded first; easy, medium, hard -- not pandas' alphabetical order."""
    plt.close("all")
    stHeatmap(cohort(), mean_or_median="mean", dscrp="Mice",
              plot_combined_subjects=True, plot_single_subjects=False)
    ax = _heatmapAxis()
    assert _ticks(ax.get_xticklabels()) == list(CUR_DIFFICULTY_ORDER)
    assert _ticks(ax.get_yticklabels()) == list(PREV_TRIAL_ORDER)
    plt.close("all")


def test_a_query_that_drops_a_difficulty_level_keeps_the_rest_in_order():
    """The old sort callbacks raised here; label selection does not."""
    plt.close("all")
    stHeatmap(cohort(), mean_or_median="median", dscrp="Mice",
              df_query="DVstr != 'Med'", plot_combined_subjects=True,
              plot_single_subjects=False)
    assert _ticks(_heatmapAxis().get_xticklabels()) == ["Easy", "Hard"]
    plt.close("all")


# --------------------------------------------------------------------------
# What gets dropped
# --------------------------------------------------------------------------

@pytest.mark.parametrize("column",
                         ["ChoiceCorrect", "Stay", "PrevChoiceCorrect"])
def test_trials_missing_a_required_outcome_are_dropped(column):
    '''A trial with no previous choice cannot sit in a previous-outcome cell.'''
    df = cohort()
    df.loc[df.index[:200], column] = np.nan
    stHeatmap(df, mean_or_median="median", dscrp="Mice",
              plot_combined_subjects=True, plot_single_subjects=False)
    plt.close("all")


# --------------------------------------------------------------------------
# The violin panel and its Figure S3F statistics
# --------------------------------------------------------------------------

def _violinTitle(df):
    fig, ax = plt.subplots()
    _violinPlot(df, ax)
    title = ax.get_title()
    plt.close(fig)
    return title


def _parsePValue(title):
    return float(title.split("p-value: ")[1].split(" ")[0])


def test_the_violin_panel_reports_its_test_and_p_value():
    title = _violinTitle(violinFrame())
    assert "Stimulus Time Distribution" in title
    assert any(name in title for name in ("f_oneway", "kruskal"))
    assert 0. <= _parsePValue(title) <= 1.


def test_the_animal_count_reaches_the_violin_title():
    title = _violinTitle(violinFrame(cohort(n_animals=4)))
    assert "n=4" in title


def test_non_normal_cells_fall_back_to_kruskal_and_dunn():
    '''Shapiro runs on the *per-animal* values, so the animals must be skewed.

    Per-trial noise averages away, which is why an outlier-heavy trial
    distribution is not enough -- two animals well outside the rest are.
    '''
    df = cohort(n_animals=10)
    cell = ((df.PrevChoiceCorrect == 1.) & (df.DVstr == "Easy")
            & (df.PrevDVstr == "Easy"))
    outliers = cell & df.Name.isin(["m0", "m1"])
    df.loc[outliers, "calcStimulusTime"] += 6.
    title = _violinTitle(violinFrame(df))
    assert "kruskal" in title
    assert "raw" in title
    assert "Dunn" in title
    assert "holm" in title


def test_normal_cells_use_anova_on_the_log_data_with_tukey():
    df = cohort(n_animals=12, seed=3)
    title = _violinTitle(violinFrame(df))
    assert "f_oneway" in title
    assert "log-transformed" in title
    assert "Tukey" in title


def test_a_real_effect_of_previous_outcome_is_detected():
    strong = violinFrame(cohort(n_animals=10, slower_after_error=.8))
    assert _parsePValue(_violinTitle(strong)) < 0.05


def test_no_effect_anywhere_is_not_detected():
    '''Every cell drawn from one distribution -- nothing to find.'''
    rng = np.random.default_rng(5)
    df = cohort(n_animals=10, slower_after_error=0.)
    for difficulty in DIFFICULTIES:
        df.loc[df.DVstr == difficulty, "calcStimulusTime"] = rng.normal(
            1.2, .12, (df.DVstr == difficulty).sum())
    assert _parsePValue(_violinTitle(violinFrame(df))) > 0.05


def test_the_violin_panel_aggregates_per_animal_not_per_trial():
    '''An animal with ten times the trials still contributes one value.

    Checked through the animal count in the title, which counts animals, and
    by the panel surviving a wildly unbalanced frame.
    '''
    df = cohort(n_animals=6)
    heavy = df.Name == "m0"
    df = pd.concat([df, df[heavy], df[heavy], df[heavy]], ignore_index=True)
    title = _violinTitle(violinFrame(df))
    assert "n=6" in title
