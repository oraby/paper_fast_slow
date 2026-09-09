'''Tests for the per-predictor variance-explained figure (Figure 2A).

These pin the properties the published figure and its Methods paragraph rest
on: the leave-one-out share is a fraction of the *log-likelihood improvement*
over an intercept-only model, an irrelevant predictor scores ~0, collinear
predictors are attributed to neither (which is why the published predictor set
is the one it is), and the shares only sum to ~100% when the design is close to
orthogonal.

The row-filtering rules are pinned too, because they set the trial and animal
counts quoted in the paper: a trial needs every column in ``REQUIRED_COLS``
non-null -- ``RewardRate5`` included, even though the reward rate is not a
default predictor -- and an animal needs *more* than ``MIN_TRIALS`` trials,
counted after whole sessions with a null DV are dropped.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..varexplained import (MIN_TRIALS, PREDICTORS, PREDICTOR_COLORS,
                            REQUIRED_COLS, loopSubjects, plotVarianceExplained,
                            prepareSubject, totalVarianceExplained,
                            varianceExplained)

N_TRIALS = 400


def _preparedDf(seed=0, n=N_TRIALS, betas=(1.0, 0.0), noise=0.3):
    '''A frame shaped like ``prepareSubject`` output, with a known generator.

    ``RT`` is exp() of a linear combination of two independent regressors, so
    that ``varianceExplained``'s internal log() recovers the linear model.
    '''
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    log_rt = betas[0] * x1 + betas[1] * x2 + rng.normal(0, noise, n)
    return pd.DataFrame({"RT": np.exp(log_rt), "x1": x1, "x2": x2})


def _subjectRows(name, n, seed, sessions=2, dv_nulls=0, rr_nulls=0):
    '''Raw per-trial rows for one animal, as ``loopSubjects`` expects them.'''
    rng = np.random.default_rng(seed)
    per_sess = n // sessions
    rows = []
    for s in range(sessions):
        choice_left = rng.integers(0, 2, per_sess).astype(float)
        correct = rng.integers(0, 2, per_sess).astype(float)
        dv = rng.uniform(-1, 1, per_sess)
        frame = pd.DataFrame({
            "Name": name,
            "Date": f"2024-01-{s + 1:02d}",
            "SessionNum": 1,
            "TrialNumber": np.arange(1, per_sess + 1),
            "DV": dv,
            "ChoiceLeft": choice_left,
            "ChoiceCorrect": correct,
            "calcStimulusTime": np.exp(0.8 * np.abs(dv) + rng.normal(0, .3,
                                                                    per_sess)),
            "PrevChoiceCorrect": np.roll(correct, 1),
            "Stay": rng.integers(0, 2, per_sess).astype(float),
            "PrevOutcomeCount": rng.integers(-3, 4, per_sess).astype(float),
        })
        rows.append(frame)
    df = pd.concat(rows, ignore_index=True)
    if dv_nulls:
        df.loc[df.index[:dv_nulls], "DV"] = np.nan
    if rr_nulls:
        df.loc[df.index[:rr_nulls], "calcStimulusTime"] = np.nan
    return df


# --------------------------------------------------------------------------
# varianceExplained -- the leave-one-out arithmetic
# --------------------------------------------------------------------------

def test_irrelevant_predictor_scores_about_zero():
    '''x2 does not enter the generator, so dropping it costs nothing.'''
    fit = varianceExplained(_preparedDf(betas=(1.0, 0.0)),
                            predictors=["x1", "x2"])
    assert fit.var_explained["x1"] > 95
    assert abs(fit.var_explained["x2"]) < 5


def test_shares_are_additive_when_effects_are_weak():
    '''The regime the published fit is in: R^2 = 0.006-0.081 per animal.

    Orthogonal regressors that explain little of the variance give shares that
    add to ~100%, which is what the Methods report (101.9%).
    '''
    fit = varianceExplained(_preparedDf(betas=(1.0, 1.0), noise=8, n=4000),
                            predictors=["x1", "x2"])
    assert 95 < sum(fit.var_explained.values()) < 105


def test_shares_super_add_when_effects_are_strong():
    '''The sum is not, on its own, an orthogonality diagnostic.

    Log-likelihood improvement is logarithmic in the residual sum of squares,
    so it is only near-additive while the predictors explain little. With the
    *same perfectly orthogonal* regressors and a strong effect, two predictors
    sum to ~160%. A near-100% total therefore says the effects are weak at
    least as much as it says the design is orthogonal -- the condition number
    is the direct collinearity check.
    '''
    fit = varianceExplained(_preparedDf(betas=(1.0, 1.0), noise=0.3, n=4000),
                            predictors=["x1", "x2"])
    assert sum(fit.var_explained.values()) > 140


def test_collinear_predictors_are_attributed_to_neither():
    '''The property that dictates the published predictor set.

    With a duplicated regressor, dropping either one leaves the information
    intact, so leave-one-out credits neither -- and the shares no longer sum
    to 100%. This is why the Methods use one signed streak count rather than
    separate previous-outcome and reward-rate terms.
    '''
    df = _preparedDf(betas=(1.0, 0.0))
    df["x1_copy"] = df.x1
    fit = varianceExplained(df, predictors=["x1", "x1_copy"])
    assert abs(fit.var_explained["x1"]) < 5
    assert abs(fit.var_explained["x1_copy"]) < 5
    assert sum(fit.var_explained.values()) < 50


def test_share_is_relative_to_improvement_not_raw_loglik():
    '''A share of 100% means the predictor carries the whole improvement.

    Guards against regressing to ``1 - cur_llf/full_llf``, which would give a
    number near zero for the same fit because the raw log-likelihoods are
    large and similar.
    '''
    fit = varianceExplained(_preparedDf(betas=(1.0, 0.0)),
                            predictors=["x1", "x2"])
    assert fit.full_llf > fit.null_llf
    naive = 100 * (1 - (fit.full_llf - 1e-9) / fit.full_llf)
    assert fit.var_explained["x1"] > 10 * abs(naive)


def test_reports_condition_number_and_trial_count():
    '''Both are quoted in the Methods, so they must survive extraction.'''
    df = _preparedDf()
    fit = varianceExplained(df, predictors=["x1", "x2"])
    assert fit.n_trials == len(df)
    # Two standardised, independent regressors plus an intercept: well
    # conditioned, nowhere near the collinearity danger zone.
    assert 1 < fit.cond_no < 10


def test_does_not_mutate_the_caller_frame():
    '''RT is logged internally; the caller's copy must stay in seconds.'''
    df = _preparedDf()
    before = df.RT.copy()
    varianceExplained(df, predictors=["x1", "x2"])
    pd.testing.assert_series_equal(df.RT, before)


# --------------------------------------------------------------------------
# prepareSubject -- the derived regressors and the row filter
# --------------------------------------------------------------------------

def test_difficulty_regressor_increases_from_easy_to_hard():
    '''DVabs is 1 - |DV|, so a coherent stimulus scores low.'''
    df = _subjectRows("M1", 20, seed=3)
    df["RewardRate5"] = 0.5
    out = prepareSubject(df)
    np.testing.assert_allclose(out.DVabs, 1 - df.DV.abs().values)
    assert out.DVabs.max() <= 1


def test_win_stay_and_lose_stay_partition_stay():
    '''Every stay trial is either a win-stay or a lose-stay, never both.'''
    df = _subjectRows("M1", 40, seed=4)
    df["RewardRate5"] = 0.5
    out = prepareSubject(df)
    assert ((out.WinStay + out.LoseStay) == out.Stay).all()
    assert not ((out.WinStay == 1) & (out.LoseStay == 1)).any()


def test_drops_trials_missing_any_required_column():
    df = _subjectRows("M1", 40, seed=5)
    df["RewardRate5"] = 0.5
    df.loc[df.index[:3], "Stay"] = np.nan
    assert len(prepareSubject(df)) == len(df) - 3


def test_null_reward_rate_drops_the_trial_though_it_is_not_a_predictor():
    '''Pins the surprising rule that sets the published trial count.

    RewardRate5 is not in PREDICTORS, but a trial missing it is still dropped
    -- which is what the published figure did.
    '''
    assert "RewardRate5" in REQUIRED_COLS
    assert "RewardRate5" not in PREDICTORS
    df = _subjectRows("M1", 40, seed=6)
    df["RewardRate5"] = 0.5
    df.loc[df.index[:4], "RewardRate5"] = np.nan
    assert len(prepareSubject(df)) == len(df) - 4


# --------------------------------------------------------------------------
# loopSubjects -- the animal / session filters
# --------------------------------------------------------------------------

def test_animal_needs_strictly_more_than_min_trials():
    '''The cut is ">", not ">=" -- an animal exactly at the bound is out.'''
    keep = _subjectRows("Keep", 60, seed=7)
    drop = _subjectRows("Drop", 40, seed=8)
    out = loopSubjects(pd.concat([keep, drop], ignore_index=True),
                       predictors=["DVabs", "ChoiceLeft"], min_trials=40)
    assert set(out.Name) == {"Keep"}


def test_sessions_with_any_null_dv_are_dropped_whole():
    '''One bad DV removes its session, not just its trial.'''
    clean = loopSubjects(_subjectRows("M1", 80, seed=9, sessions=2),
                         predictors=["DVabs", "ChoiceLeft"], min_trials=0)
    holed = loopSubjects(_subjectRows("M1", 80, seed=9, sessions=2,
                                      dv_nulls=1),
                         predictors=["DVabs", "ChoiceLeft"], min_trials=0)
    # Two equal sessions, so losing one whole session halves the fitted rows.
    # A per-trial drop would have cost 1 row, not 35.
    assert holed.iloc[0].n_trials == clean.iloc[0].n_trials // 2


def test_reward_rate_warm_up_costs_the_first_trials_of_each_session():
    '''RewardRate5 is a 5-trial rolling mean, so it is null until trial 6.

    Those trials are then dropped by the REQUIRED_COLS filter, which is why a
    40-trial session contributes 35 rows.
    '''
    out = loopSubjects(_subjectRows("M1", 80, seed=9, sessions=2),
                       predictors=["DVabs", "ChoiceLeft"], min_trials=0)
    assert out.iloc[0].n_trials == 2 * (40 - 5)


def test_returns_one_row_per_animal_with_diagnostics():
    df = pd.concat([_subjectRows("M1", 80, seed=10),
                    _subjectRows("M2", 80, seed=11)], ignore_index=True)
    out = loopSubjects(df, predictors=["DVabs", "ChoiceLeft"], min_trials=0)
    assert list(out.Name) == ["M1", "M2"]
    assert {"cond_no", "n_trials", "DVabs", "ChoiceLeft"} <= set(out.columns)


def test_total_variance_explained_sums_the_across_animal_means():
    var_exp_df = pd.DataFrame({"Name": ["M1", "M2"],
                               "DVabs": [20.0, 30.0], "Stay": [10.0, 40.0]})
    assert totalVarianceExplained(var_exp_df,
                                  predictors=["DVabs", "Stay"]) == 50.0


# --------------------------------------------------------------------------
# plotVarianceExplained
# --------------------------------------------------------------------------

def test_plot_draws_one_bar_per_predictor_labelled_with_mean_and_sem():
    var_exp_df = pd.DataFrame({"Name": ["M1", "M2", "M3"],
                               "DVabs": [10.0, 20.0, 30.0],
                               "Stay": [5.0, 5.0, 5.0]})
    fig = plotVarianceExplained(var_exp_df, predictors=["DVabs", "Stay"])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("20.00%" in lbl and "n=3" in lbl for lbl in labels)
    # Zero variance across animals -> zero SEM.
    assert any("5.00% ±0.00%" in lbl for lbl in labels)
    plt.close(fig)


def test_plot_marks_animals_whose_fit_returned_nan():
    '''NaN animals are excluded from the bar but shown, not hidden.'''
    var_exp_df = pd.DataFrame({"Name": ["M1", "M2", "M3"],
                               "DVabs": [10.0, 20.0, np.nan]})
    fig = plotVarianceExplained(var_exp_df, predictors=["DVabs"])
    ax = fig.axes[0]
    scatters = [c for c in ax.collections
                if isinstance(c, matplotlib.collections.PathCollection)]
    assert len(scatters) == 1
    assert len(scatters[0].get_offsets()) == 1     # one NaN animal marked
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("n=2" in lbl for lbl in labels)     # bar averages the other two
    plt.close(fig)


def test_plot_saves_under_the_prefix(tmp_path):
    var_exp_df = pd.DataFrame({"Name": ["M1", "M2"], "DVabs": [10.0, 20.0]})
    fig = plotVarianceExplained(var_exp_df, predictors=["DVabs"],
                                save_prefix=tmp_path / "behavior",
                                save_figs=True)
    assert (tmp_path / "behavior"
            / "model_OLS_var_explained_w_filter.svg").exists()
    plt.close(fig)


def test_plot_refuses_to_save_without_a_prefix():
    var_exp_df = pd.DataFrame({"Name": ["M1"], "DVabs": [10.0]})
    with pytest.raises(ValueError, match="save_prefix"):
        plotVarianceExplained(var_exp_df, predictors=["DVabs"], save_figs=True)


def test_every_predictor_has_a_valid_matplotlib_colour():
    '''Guards the inline map's "rose", which is not a colour name.'''
    for name, colour in PREDICTOR_COLORS.items():
        assert matplotlib.colors.is_color_like(colour), f"{name}={colour!r}"
