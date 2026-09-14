'''Tests for the psychometric binning and fit (Figures 1E, 1F, S2G, S2J).

Two pieces of this repo's own logic sit on top of the vendored ``psychofit``
submodule (upstream ``cortex-lab/psychofit``, which carries its own tests and
is not re-tested here):

- **binning** -- ``_getGroups`` cuts trials into coherence bins that differ
  between mice and humans, and mirrors them around zero unless the two sides
  are combined;
- **model choice** -- ``_psychFitBasic`` picks ``erf_psycho_2gammas`` (four
  parameters, independent lapse rates per side) when the sides are kept apart
  and ``erf_psycho`` (three) when they are pooled, then trims the shared
  start/bounds arrays to match.

Curves here are synthesised with ``psychofit``'s own functions, so the tests
assert that the wrapper recovers the parameters it was given rather than
re-deriving upstream's threshold convention.
'''
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from .. import psychometric as psych
from ..psychometric import _getGroups, _psychFitBasic, getGroupsDVstr, psychofit

#: Signed coherences, densely enough sampled for a four-parameter fit.
STIMS = np.array([-1., -.5, -.25, -.12, -.06, .06, .12, .25, .5, 1.])
TRIALS_PER_POINT = 400.


@pytest.fixture(autouse=True)
def seededRestarts():
    '''``mle_fit_psycho`` draws its extra start points from ``np.random``.

    Without a fixed seed the recovered parameters move from run to run, so a
    test that passes on its own fails once another module has advanced the
    global state. Worth knowing before trusting a single fit: the published
    panels use ``nfits=100``, which makes the spread small but not zero.
    '''
    np.random.seed(0)


def curve(pars, stims=STIMS, model="erf_psycho_2gammas"):
    return getattr(psychofit, model)(pars, stims)


def fit(pars, *, combine_sides=False, stims=STIMS, nfits=50):
    model = "erf_psycho" if combine_sides else "erf_psycho_2gammas"
    return _psychFitBasic(stims, np.full_like(stims, TRIALS_PER_POINT),
                          curve(pars, stims, model),
                          combine_sides=combine_sides, nfits=nfits)


# --------------------------------------------------------------------------
# Coherence binning
# --------------------------------------------------------------------------

def _edges(groups):
    return [(interval.left, interval.right) for interval in groups.groups]


def test_mice_and_humans_get_different_bin_edges():
    '''Humans see far fewer, coarser coherence levels than mice.'''
    df = pd.DataFrame({"DV": np.linspace(-1, 1, 101)})
    mice = _edges(_getGroups(df, combine_sides=True, is_human_subject=False))
    humans = _edges(_getGroups(df, combine_sides=True, is_human_subject=True))
    assert len(mice) == 8
    assert len(humans) == 6
    assert mice != humans


def test_separating_the_sides_mirrors_the_bins_around_zero():
    df = pd.DataFrame({"DV": np.linspace(-1, 1, 101)})
    combined = _edges(_getGroups(df, combine_sides=True, is_human_subject=False))
    signed = _edges(_getGroups(df, combine_sides=False, is_human_subject=False))
    assert len(signed) == 2 * len(combined)
    assert sorted(signed) == sorted([(-b, -a) for a, b in signed])


def test_combining_the_sides_bins_on_absolute_coherence():
    df = pd.DataFrame({"DV": [-.5, .5]})
    groups = _getGroups(df, combine_sides=True, is_human_subject=False)
    assert sum(len(idx) for idx in groups.groups.values()) == 2
    assert len(groups.groups[[k for k, v in groups.groups.items() if len(v)][0]]) == 2


def test_a_zero_coherence_trial_is_treated_inconsistently_by_the_two_modes():
    '''A hazard, pinned rather than fixed.

    ``pd.cut`` makes right-closed bins, so with the sides separated a DV of
    exactly 0 lands in ``(-0.01, 0.0]`` and is counted as a *left* stimulus;
    with the sides combined the lowest bin starts at 0 and the trial is
    dropped altogether. The mice used here have no true 0% coherence level,
    so neither behaviour shows up in the published panels.
    '''
    df = pd.DataFrame({"DV": [0.0]})
    signed = _getGroups(df, combine_sides=False, is_human_subject=False)
    combined = _getGroups(df, combine_sides=True, is_human_subject=False)
    assert sum(len(idx) for idx in signed.groups.values()) == 1
    assert sum(len(idx) for idx in combined.groups.values()) == 0


def test_coherences_beyond_the_last_edge_are_dropped():
    df = pd.DataFrame({"DV": [1.5, -1.5, .5]})
    groups = _getGroups(df, combine_sides=True, is_human_subject=False)
    assert sum(len(idx) for idx in groups.groups.values()) == 1


def test_an_empty_coherence_bin_is_kept_not_dropped():
    '''Pinned because pandas 3 flips the ``observed`` default.

    Keeping empty bins is harmless: ``_fitPsych`` appends NaN for them, and
    ``psychofit.mle_fit_psycho`` drops non-finite proportions before fitting.
    The fit is the same either way; the explicit ``observed=False`` only keeps
    today's iteration unchanged.
    '''
    df = pd.DataFrame({"DV": [.5, .9]})       # nothing below 0.32
    groups = _getGroups(df, combine_sides=True, is_human_subject=False)
    assert len(groups.groups) == 8
    assert sum(1 for idx in groups.groups.values() if len(idx) == 0) == 6


def test_binning_raises_no_pandas_deprecation_warning():
    df = pd.DataFrame({"DV": np.linspace(-1, 1, 51)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        _getGroups(df, combine_sides=False, is_human_subject=False)
        _getGroups(df, combine_sides=True, is_human_subject=True)


def test_the_string_grouping_splits_by_side_unless_combined():
    df = pd.DataFrame({"DV": [-.5, .5, -.5, .5], "DVstr": ["50%"] * 4})
    assert len(getGroupsDVstr(df, combine_sides=True).groups) == 1
    assert len(getGroupsDVstr(df, combine_sides=False).groups) == 2


def test_the_string_grouping_needs_the_column():
    with pytest.raises(AssertionError, match="DVstr"):
        getGroupsDVstr(pd.DataFrame({"DV": [.5]}), combine_sides=True)


# --------------------------------------------------------------------------
# Which model the wrapper picks
# --------------------------------------------------------------------------

def test_separate_sides_fit_four_parameters_and_combined_sides_three():
    '''The extra parameter is the second lapse rate.'''
    assert len(fit((0., .4, .05, .05))[0]) == 4
    assert len(fit((0., .4, .05), combine_sides=True,
                   stims=np.abs(STIMS))[0]) == 3


def test_the_returned_callable_evaluates_the_fitted_curve():
    pars, fitted = fit((0., .4, .05, .05))
    np.testing.assert_allclose(fitted(STIMS),
                               psychofit.erf_psycho_2gammas(pars, STIMS))


def test_the_fit_reproduces_the_data_it_was_given():
    pars, fitted = fit((.1, .4, .04, .06))
    np.testing.assert_allclose(fitted(STIMS), curve((.1, .4, .04, .06)),
                               atol=.01)


# --------------------------------------------------------------------------
# Parameter recovery
# --------------------------------------------------------------------------

def test_bias_is_recovered():
    for bias in (-.3, 0., .3):
        pars, _ = fit((bias, .4, .05, .05))
        assert pars[0] == pytest.approx(bias, abs=.02)


def test_a_steeper_subject_gets_a_smaller_threshold():
    steep, _ = fit((0., .15, .05, .05))
    shallow, _ = fit((0., .9, .05, .05))
    assert steep[1] < shallow[1]


def test_threshold_is_recovered():
    for threshold in (.2, .5):
        pars, _ = fit((0., threshold, .05, .05))
        assert pars[1] == pytest.approx(threshold, rel=.15)


def test_the_two_lapse_rates_are_fitted_independently():
    '''What ``erf_psycho_2gammas`` buys over ``erf_psycho``.'''
    pars, _ = fit((0., .4, .02, .18))
    assert pars[2] == pytest.approx(.02, abs=.03)
    assert pars[3] == pytest.approx(.18, abs=.03)
    assert pars[3] > pars[2]


def test_the_bounds_are_respected():
    pars, _ = fit((.9, .4, .05, .05))
    assert psych._parmin[0] <= pars[0] <= psych._parmax[0]
    assert psych._parmin[1] <= pars[1] <= psych._parmax[1]
    for lapse in pars[2:]:
        assert 0. <= lapse <= 1.


def test_custom_bounds_override_the_module_defaults():
    '''``parmin``/``parmax`` are trimmed to the model's parameter count.'''
    pars, _ = _psychFitBasic(
        STIMS, np.full_like(STIMS, TRIALS_PER_POINT),
        curve((.4, .4, .05, .05)), combine_sides=False, nfits=50,
        parstart=np.array([0., .5, .1, .1]),
        parmin=np.array([-.1, .01, 0., 0.]),
        parmax=np.array([.1, 2., .3, .3]))
    assert -.1 <= pars[0] <= .1     # clamped well away from the true 0.4
    assert pars[0] < .4
