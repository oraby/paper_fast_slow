"""Regression tests for the registry entries the paper does not use.

Three defects kept these variants from being fitted at all under chisq:

1. ``Decaying Q-Val`` noise: ``util.decayingQ`` took ``np.arange`` of the
   noise array's ``(trials, steps)`` shape and indexed its scalar rate
   parameter, so every chisq evaluation raised.
2. The same noise drew from the global ``np.random`` rather than the seeded
   generator ``logic.makeOneRun`` installs, so two evaluations of the same
   parameters gave different losses.
3. ``μ, σ (Corr/Incorr)`` bias called ``_biasMeanDir`` without ``BIAS_COEF``,
   which raised.

Plus a sweep that every registry combination completes a dry run under both
fitting modes, which is what caught them.
"""
from __future__ import annotations

import contextlib
import io
import itertools

import numpy as np
import pandas as pd
import pytest

from .. import bias, fit, noise
from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..initvals import InitVals
from ..mle import SUPPORTED_MLE_BIASES, _decaying_q_noise_array
from ..noise import NOISE_FN_DICT
from ..util import decayingQ


def test_decaying_q_matches_the_mle_form():
    q = np.array([-0.8, -0.1, 0.0, 0.3, 0.9])
    rate, coef, n_t = 1.7, 2.5, 40
    chisq = decayingQ((len(q), n_t), q, rate, coef, dt=0.01)
    mle = _decaying_q_noise_array(
        q, {"Q_VAL_DECAY_RATE": rate, "Q_VAL_COEF": coef}, n_t)
    assert chisq.shape == (len(q), n_t)
    np.testing.assert_allclose(chisq, mle)


def test_decaying_q_does_not_modify_the_q_values():
    q = np.array([-0.5, 0.5])
    decayingQ((2, 10), q, 1.0, 3.0, dt=0.01)
    np.testing.assert_array_equal(q, [-0.5, 0.5])


def test_decaying_q_noise_uses_the_seeded_generator(monkeypatch):
    q = np.array([0.2, -0.4, 0.7])
    draws = []
    for _ in range(2):
        monkeypatch.setattr(noise, "rnd_default_rng", np.random.default_rng(3))
        draws.append(noise._noiseQval((3, 25), 0.01, q, 1.0, 2.0))
    np.testing.assert_array_equal(draws[0], draws[1])


def test_mean_corr_incorr_bias_flips_on_incorrect_side_and_scales_once(monkeypatch):
    dv = np.array([0.5, -0.5, 0.2, -0.2])
    monkeypatch.setattr(bias, "rnd_default_rng", np.random.default_rng(7))
    got = bias._biasMeanCorrIncorr(4, BIAS_COEF=2.0, BIAS_MU=0.3,
                                   BIAS_SIGMA=0.1, DV=dv)
    raw = np.random.default_rng(7).normal(0.3, 0.1, size=4)
    raw[dv < 0] = -raw[dv < 0]
    np.testing.assert_allclose(got, raw * 2.0)


def _session_df(seed=0, n_sess=2, n_trials=30):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sess):
        for t in range(1, n_trials + 1):
            dv = float(rng.choice([-1, -.5, -.2, .2, .5, 1]))
            left = float(rng.random() < .5)
            rows.append(dict(
                Name="S1", Date=pd.Timestamp(f"2026-01-0{s + 1}"),
                SessionNum=1, TrialNumber=t, SessId=f"S1_{s}", DV=dv,
                DVstr=str(dv), valid=True,
                calcStimulusTime=float(rng.uniform(.1, .9)), ChoiceLeft=left,
                ChoiceCorrect=float(left == (dv > 0))))
    return pd.DataFrame(rows)


def _dry_run(b, d, n, mode, asym):
    kw = dict(bounds_and_defaults=InitVals().toDict(), dt=0.01, t_dur=1.0,
              biasFn=BIAS_FN_DICT[b], driftFn=DRIFT_FN_DICT[d],
              noiseFn=NOISE_FN_DICT[n], is_loss_no_dir=False, num_cpus=1,
              evolvs_res={}, fit_mode=mode, dry_run=True, bias_fn_str=b,
              drift_fn_str=d, uses_asym_q=asym, uses_asym_rr=asym,
              scale_bound=d.startswith("Bound-"))
    with contextlib.redirect_stdout(io.StringIO()):
        return fit.simulateDDM(_session_df(), **kw)


_COMBOS = list(itertools.product(BIAS_FN_DICT, DRIFT_FN_DICT, NOISE_FN_DICT))


@pytest.mark.parametrize("b,d,n", _COMBOS)
def test_every_combination_dry_runs_under_chisq_deterministically(b, d, n):
    first = float(_dry_run(b, d, n, "chisq", asym=True)["S1"])
    again = float(_dry_run(b, d, n, "chisq", asym=True)["S1"])
    assert np.isfinite(first)
    assert first == again


@pytest.mark.parametrize(
    "b,d,n", [c for c in _COMBOS if c[0] in SUPPORTED_MLE_BIASES])
def test_every_mle_supported_combination_dry_runs_under_mle(b, d, n):
    payload = _dry_run(b, d, n, "mle", asym=True)["S1"]
    assert np.isfinite(payload["neg_loglik"])


def test_mle_refuses_the_simulation_only_biases():
    unsupported = set(BIAS_FN_DICT) - SUPPORTED_MLE_BIASES
    assert unsupported == {"Fixed (Dir)", "Fixed (Corr/Incorr)",
                           "μ, σ (Dir)", "μ, σ (Corr/Incorr)"}
    with pytest.raises(NotImplementedError):
        _dry_run("Fixed (Dir)", "Classic", "Normal(0, 1)", "mle", asym=False)
