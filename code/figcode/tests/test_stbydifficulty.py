'''Tests for the sampling-time-by-difficulty panels (Figures 1D, 1G, S2E-F, S3A).

Figure 1D is the histogram of sampling times split by coherence, with the
fast/typical/slow bands drawn over it. Figure 1G is the mean sampling time
against difficulty, fast tertile versus slow. Both come from this module, and
neither had a test.

**Figure 1D could not be regenerated at all** before these tests were written:
:func:`_handleDifficultyDf` called ``matplotlib.cm.get_cmap``, removed in
matplotlib 3.9, and the locked environment has 3.11. The call is on the
``stDistOnly`` path, so it raised ``AttributeError`` on real data.
:func:`test_the_histogram_path_runs` is the regression test for that, and
:func:`test_bars_are_coloured_along_the_autumn_ramp` pins the replacement as
the same colormap: 141 of the 143 distinct fills in the published
``Mice_All Mice.svg`` lie exactly on the ``autumn`` ramp (the other two are the
white background and the grey band lines).

Two conventions are worth stating because both are easy to "simplify" wrongly:

- the tertile band edges are **each animal's own tertiles, averaged across
  animals** -- not the tertiles of the pooled trials;
- Figure 1G aggregates **per animal** for the cohort panel but **per session**
  for a single animal, and the cohort panel reads the per-animal z-scored
  column rather than raw seconds.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.container import ErrorbarContainer

from ..stbydifficulty import (DifficultyClr, _handleDifficultyDf,
                              _rtVsDifficulty, _setupAxes, stDistOnly,
                              stVsDiffOnly)

DIFFICULTIES = ("Easy", "Med", "Hard")
FAST, TYPICAL, SLOW = 1, 2, 3
ST = "calcStimulusTime"
ZST = "transformedCalcStimulusTime"


def trials(name, difficulty, times, *, session=1, correct=1., quantile=FAST):
    times = np.asarray(times, dtype=float)
    return pd.DataFrame({"Name": name, "Date": "2024-01-01",
                         "SessionNum": session, "DVstr": difficulty,
                         "DV": {"Easy": .8, "Med": .3, "Hard": .05}[difficulty],
                         ST: times, "ChoiceCorrect": correct,
                         "quantile_idx": quantile})


def cohort(n_animals=4, n_per_cell=30, seed=0) -> pd.DataFrame:
    '''Slower sampling on harder trials, with each animal on its own scale.'''
    rng = np.random.default_rng(seed)
    frames = []
    for animal in range(n_animals):
        scale = 1. + animal * .4
        for difficulty, centre in zip(DIFFICULTIES, (.8, 1.1, 1.5)):
            for quantile, shift in ((FAST, -.3), (TYPICAL, 0.), (SLOW, .4)):
                for session in (1, 2):
                    frames.append(trials(
                        f"m{animal}", difficulty,
                        rng.normal((centre + shift) * scale, .15, n_per_cell),
                        session=session, quantile=quantile,
                        correct=float(rng.integers(0, 2))))
    df = pd.concat(frames, ignore_index=True)
    return df[df[ST] > 0].reset_index(drop=True)


# --------------------------------------------------------------------------
# Figure 1D: the histogram path that could not run
# --------------------------------------------------------------------------

def test_the_histogram_path_runs():
    '''Regression for the removed ``matplotlib.cm.get_cmap``.

    This raised ``AttributeError: module 'matplotlib.cm' has no attribute
    'get_cmap'`` under matplotlib 3.9+, which is every environment this repo
    now supports.
    '''
    stDistOnly(cohort(), plot_combined_subjects=True,
               plot_single_subjects=False, descrp="Mice")
    plt.close("all")


def _histogramAxis(df=None, difficulty="Easy"):
    fig, ax = plt.subplots()
    ax.set_xlim(-1.3, 1.7)
    df = cohort() if df is None else df
    df = df.copy()
    df[ZST] = (df[ST] - df[ST].mean()) / df[ST].std(ddof=0)
    _handleDifficultyDf(difficulty, df[df.DVstr == difficulty], ax,
                        DifficultyClr.Easy, CDF=False, many_animals=True,
                        plot_median=False)
    return fig, ax


#: Histogram bin width in `_handleDifficultyDf`. The fast/typical/slow spans
#: are patches too, and red/orange/yellow all sit on the autumn ramp, so
#: filtering on colour alone would let them pass unnoticed.
BIN_STEP = 0.04


def _bars(ax):
    '''The histogram bars only -- not the three band spans behind them.'''
    return [patch for patch in ax.patches
            if patch.get_height() > 0
            and patch.get_width() == pytest.approx(BIN_STEP)]


def _barColours(ax):
    return [patch.get_facecolor() for patch in _bars(ax)]


def test_bars_are_coloured_along_the_autumn_ramp():
    '''``autumn`` is red=1, blue=0, green rising -- the published palette.'''
    fig, ax = _histogramAxis()
    colours = _barColours(ax)
    assert colours
    for red, green, blue, _alpha in colours:
        assert red == pytest.approx(1.)
        assert blue == pytest.approx(0.)
        assert 0. <= green <= 1.
    plt.close(fig)


def test_bar_colour_tracks_the_cumulative_distribution():
    '''Colour is indexed by the CDF, so it rises left to right.'''
    fig, ax = _histogramAxis()
    bars = sorted(((patch.get_x(), patch.get_facecolor()[1])
                   for patch in _bars(ax)), key=lambda pair: pair[0])
    greens = [green for _x, green in bars]
    assert greens == sorted(greens)
    assert greens[-1] > greens[0]
    plt.close(fig)


def test_the_band_spans_are_not_mistaken_for_histogram_bars():
    '''The spans are red/orange/yellow, all of which lie on the autumn ramp.

    Without a width filter a colour-only check would pass whether or not the
    bars themselves were coloured at all.
    '''
    fig, ax = _histogramAxis()
    spans = [patch for patch in ax.patches if patch.get_alpha() == 0.1]
    assert len(spans) == 3
    assert not any(patch.get_width() == pytest.approx(BIN_STEP)
                   for patch in spans)
    plt.close(fig)


def test_the_colormap_has_the_published_1024_entry_resolution():
    '''``resampled(1024)`` must keep the LUT the published figure used.'''
    cmap = matplotlib.colormaps["autumn"].resampled(1024)
    assert cmap.N == 1024
    for index in (0, 1, 511, 1022, 1023):
        np.testing.assert_allclose(cmap(index),
                                   (1., index / 1023, 0., 1.), atol=1e-12)


# --------------------------------------------------------------------------
# The fast / typical / slow bands
# --------------------------------------------------------------------------

def test_three_bands_and_two_edges_are_drawn():
    fig, ax = _histogramAxis()
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    assert len(dashed) == 2                       # the two tertile edges
    assert len({tuple(patch.get_facecolor()) for patch in ax.patches
                if patch.get_alpha() == 0.1}) == 3
    plt.close(fig)


def test_band_edges_average_each_animal_s_own_tertiles():
    '''Not the tertiles of the pooled trials -- the two differ here.

    Animal A sits at 0 with a tail up to 3; animal B at 0 with a tail down to
    -3. Each animal's own 1/3 quantile is 0.0 and -1.0, averaging to -0.5,
    while the pooled 1/3 quantile is 0.0.
    '''
    frame = pd.concat([trials("A", "Easy", [0.] * 6 + [3.] * 3),
                       trials("B", "Easy", [-3.] * 3 + [0.] * 6)],
                      ignore_index=True)
    frame[ZST] = frame[ST]

    per_animal = frame.groupby("Name")[ZST].quantile(1 / 3).mean()
    pooled = frame[ZST].quantile(1 / 3)
    assert per_animal == pytest.approx(-0.5)
    assert pooled == pytest.approx(0.0)

    fig, ax = plt.subplots()
    ax.set_xlim(-4, 4)
    _handleDifficultyDf("Easy", frame, ax, DifficultyClr.Easy, CDF=False,
                        many_animals=True, plot_median=False)
    lower_edge = min(line.get_xdata()[0] for line in ax.lines
                     if line.get_linestyle() == "--")
    assert lower_edge == pytest.approx(per_animal)
    assert lower_edge != pytest.approx(pooled)
    plt.close(fig)


def test_an_empty_difficulty_is_refused_rather_than_drawn_blank():
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    with pytest.raises(AssertionError, match="No data"):
        _handleDifficultyDf("Easy", cohort().iloc[:0], ax, DifficultyClr.Easy,
                            CDF=False, many_animals=True)
    plt.close(fig)


# --------------------------------------------------------------------------
# Axis furniture and the count strings
# --------------------------------------------------------------------------

def test_the_title_counts_subjects_sessions_and_trials():
    fig, ax = plt.subplots()
    _setupAxes(ax, many_animals=True, is_CDF=False, subject_name="Easy",
               num_trials=23803, num_sessions=262, num_subjects=20)
    assert ax.get_title() == "Easy (20 subjects / 262 sessions / 23,803 trials)"
    plt.close(fig)


def test_omitted_counts_are_left_out_of_the_title():
    fig, ax = plt.subplots()
    _setupAxes(ax, many_animals=False, is_CDF=False, subject_name="m0",
               num_sessions=12)
    assert ax.get_title() == "m0 (12 sessions)"
    plt.close(fig)


def test_a_bare_title_carries_no_bracket():
    fig, ax = plt.subplots()
    _setupAxes(ax, many_animals=False, is_CDF=False, subject_name="m0")
    assert ax.get_title() == "m0"
    plt.close(fig)


def test_the_axis_label_says_normalized_only_for_the_cohort():
    fig, (cohort_ax, single_ax) = plt.subplots(1, 2)
    _setupAxes(cohort_ax, many_animals=True, is_CDF=False, subject_name="All")
    _setupAxes(single_ax, many_animals=False, is_CDF=False, subject_name="m0")
    assert cohort_ax.get_xlabel() == "Stimulus Time (normalized)"
    assert single_ax.get_xlabel() == "Stimulus Time (s)"
    assert cohort_ax.get_xlim() == (-1.3, 1.7)
    assert single_ax.get_xlim() == (0.3, 3)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 1G: sampling time against difficulty
# --------------------------------------------------------------------------

def _difficultyAxis(df, **kwargs):
    fig, ax = plt.subplots()
    _rtVsDifficulty(df, "All Mice", ax=ax, **kwargs)
    return fig, ax


def _series(ax):
    return [container for container in ax.containers
            if isinstance(container, ErrorbarContainer)]


def test_difficulty_runs_hard_to_easy_along_x():
    df = cohort()
    df[ZST] = df[ST]
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False)
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "Hard", "Med", "Easy"]
    line = _series(ax)[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [1, 2, 3])
    plt.close(fig)


def test_the_cohort_panel_averages_animals_not_trials():
    '''One value per animal, so an animal with more trials counts once.'''
    lopsided = pd.concat([trials("A", "Easy", [1.] * 200),
                          trials("B", "Easy", [3.] * 20)], ignore_index=True)
    lopsided[ZST] = lopsided[ST]
    fig, ax = _difficultyAxis(lopsided, many_animals=True,
                              sep_corr_incorr=False)
    point, = _series(ax)[0].lines[0].get_ydata()
    assert point == pytest.approx(2.)             # (1 + 3) / 2, not 1.18
    assert lopsided[ST].mean() == pytest.approx(1.1818, abs=1e-4)
    plt.close(fig)


def test_a_single_animal_panel_averages_sessions():
    one_animal = pd.concat([trials("A", "Easy", [1.] * 100, session=1),
                            trials("A", "Easy", [3.] * 10, session=2)],
                           ignore_index=True)
    fig, ax = _difficultyAxis(one_animal, many_animals=False,
                              sep_corr_incorr=False)
    point, = _series(ax)[0].lines[0].get_ydata()
    assert point == pytest.approx(2.)             # (1 + 3) / 2 across sessions
    plt.close(fig)


def test_the_cohort_panel_reads_the_z_scored_column():
    '''Raw seconds are per-animal; the cohort panel must not mix scales.'''
    df = cohort()
    df[ZST] = df[ST] * -1                          # a marker, not a z-score
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False)
    assert all(value < 0 for value in _series(ax)[0].lines[0].get_ydata())
    plt.close(fig)


def test_the_cohort_panel_refuses_a_different_sampling_time_column():
    df = cohort()
    with pytest.raises(AssertionError, match="calcStimulusTime"):
        _difficultyAxis(df, many_animals=True, sep_corr_incorr=False,
                        col_rt="somethingElse")
    plt.close("all")


def test_correct_and_incorrect_are_drawn_as_separate_series():
    df = cohort()
    df[ZST] = df[ST]
    df.loc[df.index[::2], "ChoiceCorrect"] = 0.
    df.loc[df.index[1::2], "ChoiceCorrect"] = 1.
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=True)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(_series(ax)) == 2
    assert any(label.startswith("Correct") for label in labels)
    assert any(label.startswith("Incorrect") for label in labels)
    plt.close(fig)


def test_incorrect_trials_can_be_suppressed():
    df = cohort()
    df[ZST] = df[ST]
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=True,
                              plot_incorr=False)
    assert len(_series(ax)) == 1
    plt.close(fig)


def test_a_missing_difficulty_level_is_skipped_not_zero_filled():
    df = cohort()
    df = df[df.DVstr != "Med"]
    df[ZST] = df[ST]
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False)
    line = _series(ax)[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [1, 3])   # Hard and Easy only
    plt.close(fig)


# --------------------------------------------------------------------------
# The fitted slope and its angle
# --------------------------------------------------------------------------

def _slopeAnnotation(ax):
    for text in ax.texts:
        if text.get_text().startswith("Slope="):
            body = text.get_text()
            slope = float(body.split("Slope=")[1].split(" ")[0])
            angle = float(body.split("θ=")[1].rstrip("°"))
            return slope, angle
    return None


def test_the_reported_slope_is_negated_so_slower_on_hard_reads_positive():
    '''x runs Hard→Easy, so a real effect has a negative regression slope.

    The annotation flips the sign, which is why Figure 1G quotes positive
    slopes for animals that sample longer on hard trials.
    '''
    df = pd.concat([trials("A", "Hard", [1.5] * 20),
                    trials("A", "Med", [1.1] * 20),
                    trials("A", "Easy", [0.7] * 20)], ignore_index=True)
    df[ZST] = df[ST]
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False,
                              linfit=True)
    slope, angle = _slopeAnnotation(ax)
    assert slope == pytest.approx(0.4)
    assert angle == pytest.approx(np.rad2deg(np.arctan(0.4)), abs=.01)
    plt.close(fig)


def test_a_flat_animal_reports_no_slope():
    df = pd.concat([trials("A", difficulty, [1.] * 20)
                    for difficulty in DIFFICULTIES], ignore_index=True)
    df[ZST] = df[ST]
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False,
                              linfit=True)
    slope, angle = _slopeAnnotation(ax)
    assert slope == pytest.approx(0., abs=1e-9)
    assert angle == pytest.approx(0., abs=1e-9)
    plt.close(fig)


def test_a_steeper_animal_reports_a_larger_angle():
    def _angle(step):
        df = pd.concat([trials("A", difficulty, [1. + index * step] * 20)
                        for index, difficulty in enumerate(("Easy", "Med",
                                                            "Hard"))],
                       ignore_index=True)
        df[ZST] = df[ST]
        fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False,
                                  linfit=True)
        _slope, angle = _slopeAnnotation(ax)
        plt.close(fig)
        return angle

    assert _angle(.1) < _angle(.5) < _angle(2.)
    assert _angle(2.) < 90.


def test_no_slope_is_annotated_unless_asked_for():
    df = cohort()
    df[ZST] = df[ST]
    fig, ax = _difficultyAxis(df, many_animals=True, sep_corr_incorr=False)
    assert _slopeAnnotation(ax) is None
    plt.close(fig)


# --------------------------------------------------------------------------
# The two public entry points
# --------------------------------------------------------------------------

def test_the_histogram_saves_one_file_per_difficulty_panel(tmp_path):
    stDistOnly(cohort(), plot_combined_subjects=True,
               plot_single_subjects=False, descrp="Mice",
               save_prefix=f"{tmp_path}/", save_figs=True)
    assert (tmp_path / "st_only" / "Mice_All Mice.svg").exists()
    plt.close("all")


def test_the_histogram_can_also_save_per_animal(tmp_path):
    stDistOnly(cohort(n_animals=2), plot_combined_subjects=False,
               plot_single_subjects=True, descrp="Mice",
               save_prefix=f"{tmp_path}/", save_figs=True)
    saved = sorted(path.name for path in (tmp_path / "st_only").iterdir())
    assert saved == ["Mice_m0.svg", "Mice_m1.svg"]
    plt.close("all")


def test_the_fast_slow_panel_runs_and_saves(tmp_path):
    stVsDiffOnly(cohort(), plot_combined_subjects=True,
                 plot_single_subjects=False, save_prefix=f"{tmp_path}/",
                 save_figs=True, all_dscrp="All Mice")
    saved = list((tmp_path / "st_vs_diff_fast_slow").iterdir())
    assert len(saved) == 1
    plt.close("all")


def test_trials_without_a_sampling_time_never_reach_the_panel():
    df = cohort()
    df.loc[df.index[:50], ST] = np.nan
    stDistOnly(df, plot_combined_subjects=True, plot_single_subjects=False,
               descrp="Mice")
    plt.close("all")
