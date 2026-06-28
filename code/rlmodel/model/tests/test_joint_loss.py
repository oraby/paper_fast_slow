"""Tests for the joint MLE + Chi² loss (``--mle-mle-weight`` /
``--mle-chi2-weight``).

The DE objective becomes
``w_mle·(MLE_negloglik/N_mle) + w_chi2·(Chi2/N_chi2)`` — the teacher-forced
per-trial MLE term (GPU-vectorized population path) plus the generative
Ratcliff-quantile Chi² term (per-candidate CPU simulation), each divided by its
own valid-trial count so a single weight transfers across subjects. The default
``(1.0, 0.0)`` is pure MLE: the Chi² simulation is never run.

Coverage:
- the combiner arithmetic + per-trial normalization (stubbed terms);
- a zero weight skips its term entirely (no GPU call / no Chi² simulation);
- the shared ``_candidate_to_makeOneRun_kwargs`` sentinel logic;
- end-to-end dispatch + diagnostics through ``simulateDDM`` dry-run, including
  the gate that keeps pure MLE byte-clean;
- config validation and ``evolveFP`` filename invariance.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from .. import fit
from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..initvals import InitVals
from ..mle import MLEModelConfig, validate_mle_config
from ..noise import NOISE_FN_DICT


def _df():
    """8 valid trials (both directions) + 1 no-choice padding row."""
    spec = [
        # trial, choice_left, reward, rt,   dv,   valid
        (1, 1.0, 1.0, 0.18,  0.7,  True),
        (2, 1.0, 1.0, 0.22,  0.5,  True),
        (3, 0.0, 0.0, 0.26, -0.6,  True),
        (4, 0.0, 1.0, 0.30, -0.4,  True),
        (5, 1.0, 0.0, 0.20,  0.3,  True),
        (6, 0.0, 0.0, 0.24, -0.5,  True),
        (7, 1.0, 1.0, 0.28,  0.6,  True),
        (8, 0.0, 1.0, 0.32, -0.7,  True),
        (9, np.nan, np.nan, np.nan, 0.0, False),
    ]
    rows = [dict(Name="S1", Date=pd.Timestamp("2026-01-01"), SessionNum=1,
                 TrialNumber=t, SessId="S1_2026-01-01_1", DV=dv, DVstr=str(dv),
                 valid=v, calcStimulusTime=rt, ChoiceLeft=cl, ChoiceCorrect=rew)
            for (t, cl, rew, rt, dv, v) in spec]
    return pd.DataFrame(rows)


def _config(w_mle, w_chi2):
    return MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_", noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False, dt=0.01, t_dur=0.5,
        mle_mle_weight=w_mle, mle_chi2_weight=w_chi2)


# ---------------------------------------------------------------------------
# Combiner arithmetic + normalization (MLE / Chi² terms stubbed)
# ---------------------------------------------------------------------------

def _call_wrapper(monkeypatch, w_mle, w_chi2, mle_val, chi2_val,
                  n_mle, n_chi2, n_cand=4):
    """Drive ``_jointVectorizedObjectiveWrapper`` with both terms stubbed to
    constants so only the weighting/÷N arithmetic is exercised."""
    monkeypatch.setattr(fit, "objective_from_population",
                        lambda xm, names, df, cfg: np.full(xm.shape[1], mle_val))
    monkeypatch.setattr(fit, "_candidate_to_makeOneRun_kwargs",
                        lambda *a, **k: {})
    monkeypatch.setattr(fit, "makeOneRun", lambda **k: chi2_val)
    x = np.zeros((3, n_cand))
    return fit._jointVectorizedObjectiveWrapper(
        x, np.array(["A", "B", "C"]), object(), _config(w_mle, w_chi2),
        None, None, None, None, None, None, None, None, None, None,
        n_mle, n_chi2)


def test_combiner_weighted_sum(monkeypatch):
    out = _call_wrapper(monkeypatch, 2.0, 3.0, mle_val=8.0, chi2_val=6.0,
                        n_mle=4, n_chi2=2)
    # 2*(8/4) + 3*(6/2) = 4 + 9 = 13, for every candidate.
    assert out.shape == (4,)
    assert np.allclose(out, 13.0)


def test_combiner_equal_weights(monkeypatch):
    out = _call_wrapper(monkeypatch, 1.0, 1.0, 8.0, 6.0, n_mle=8, n_chi2=6)
    assert np.allclose(out, 2.0)  # (8/8) + (6/6)


def test_trial_count_invariance(monkeypatch):
    """Each term ÷ its own valid-trial count → the per-trial value is invariant
    when the raw term and the count scale together (the whole point of ÷N)."""
    small = _call_wrapper(monkeypatch, 0.0, 1.0, 0.0, chi2_val=6.0,
                          n_mle=1, n_chi2=6)
    big = _call_wrapper(monkeypatch, 0.0, 1.0, 0.0, chi2_val=60.0,
                        n_mle=1, n_chi2=60)
    assert np.allclose(small, 1.0)
    assert np.allclose(small, big)


def test_chi2_skipped_when_weight_zero(monkeypatch):
    def _boom(**k):
        raise AssertionError("Chi² must not run when chi2-weight == 0")
    monkeypatch.setattr(fit, "makeOneRun", _boom)
    monkeypatch.setattr(fit, "objective_from_population",
                        lambda xm, *a: np.full(xm.shape[1], 5.0))
    out = fit._jointVectorizedObjectiveWrapper(
        np.zeros((2, 3)), np.array(["A", "B"]), object(), _config(1.0, 0.0),
        None, None, None, None, None, None, None, None, None, None, 5, 1)
    assert np.allclose(out, 1.0)  # 5/5, Chi² never evaluated


def test_mle_skipped_when_weight_zero(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("MLE must not run when mle-weight == 0")
    monkeypatch.setattr(fit, "objective_from_population", _boom)
    monkeypatch.setattr(fit, "_candidate_to_makeOneRun_kwargs",
                        lambda *a, **k: {})
    monkeypatch.setattr(fit, "makeOneRun", lambda **k: 4.0)
    out = fit._jointVectorizedObjectiveWrapper(
        np.zeros((2, 3)), np.array(["A", "B"]), object(), _config(0.0, 1.0),
        None, None, None, None, None, None, None, None, None, None, 1, 4)
    assert np.allclose(out, 1.0)  # 4/4, MLE never evaluated


def test_non_finite_chi2_is_finitized(monkeypatch):
    """A degenerate simulation (nan/inf Chi²) must not poison DE selection."""
    monkeypatch.setattr(fit, "objective_from_population",
                        lambda xm, *a: np.zeros(xm.shape[1]))
    monkeypatch.setattr(fit, "_candidate_to_makeOneRun_kwargs",
                        lambda *a, **k: {})
    monkeypatch.setattr(fit, "makeOneRun", lambda **k: np.nan)
    out = fit._jointVectorizedObjectiveWrapper(
        np.zeros((2, 2)), np.array(["A", "B"]), object(), _config(1.0, 1.0),
        None, None, None, None, None, None, None, None, None, None, 1, 1)
    assert np.all(np.isfinite(out))
    assert np.all(out > 1e6)


# ---------------------------------------------------------------------------
# Shared candidate→kwargs mapping (the refactor reused by chisq + joint)
# ---------------------------------------------------------------------------

def test_candidate_to_makeOneRun_kwargs_sentinels():
    empty = np.array([], dtype=int)
    kw = fit._candidate_to_makeOneRun_kwargs(
        np.array([0.7]), np.array(["ALPHA"], dtype=object),
        np.array(["include_Q", "include_RewardRate"], dtype=object),
        np.array([True, False], dtype=object),
        logicFn_x_idxs=np.array([0]), logicFn_fix_idxs=np.array([0, 1]),
        biasFn_x_idxs=empty, biasFn_fix_idxs=empty,
        driftFn_x_idxs=empty, driftFn_fix_idxs=empty,
        noiseFn_x_idxs=empty, noiseFn_fix_idxs=empty)
    assert kw["ALPHA"] == 0.7            # include_Q True → kept
    assert np.isnan(kw["BETA"])          # include_RewardRate False → NaN sentinel
    assert kw["ALPHA_UNREWARDED"] is None
    assert kw["BETA_UNREWARDED"] is None
    # Nested dispatch kwargs are always present (and empty here).
    assert kw["driftFn_kwargs"] == {} and kw["biasFn_kwargs"] == {}


def test_candidate_to_makeOneRun_kwargs_alpha_nan_when_no_q():
    empty = np.array([], dtype=int)
    kw = fit._candidate_to_makeOneRun_kwargs(
        np.array([0.7]), np.array(["ALPHA"], dtype=object),
        np.array(["include_Q", "include_RewardRate"], dtype=object),
        np.array([False, True], dtype=object),
        np.array([0]), np.array([0, 1]),
        empty, empty, empty, empty, empty, empty)
    assert np.isnan(kw["ALPHA"])         # include_Q False → NaN placeholder


# ---------------------------------------------------------------------------
# End-to-end dispatch + diagnostics via simulateDDM dry-run
# ---------------------------------------------------------------------------

def _dry_run(monkeypatch, **weight_kwargs):
    """simulateDDM MLE dry-run on the synthetic subject. num_cpus=1 ⇒
    _running_locally, so no df-dump disk writes (same as test_mle_smoke)."""
    return fit.simulateDDM(
        _df(), bounds_and_defaults=InitVals().toDict(), dt=0.01, t_dur=0.5,
        biasFn=BIAS_FN_DICT["None_"], driftFn=DRIFT_FN_DICT["Classic"],
        noiseFn=NOISE_FN_DICT["Normal(0, 1)"], is_loss_no_dir=False,
        num_cpus=1, evolvs_res={}, fit_mode="mle", dry_run=True,
        mle_array_backend="numpy", **weight_kwargs)


def test_simulateDDM_joint_dry_run_reports_components(monkeypatch):
    # Stub the Chi² simulation to a constant — its numerics are covered by the
    # chisq tests; here we verify the dispatch + diagnostics wiring.
    monkeypatch.setattr(fit, "makeOneRun", lambda **k: 12.0)
    payload = _dry_run(monkeypatch, mle_mle_weight=1.0, mle_chi2_weight=1.0)["S1"]
    assert payload["joint_mode"] is True
    assert payload["mle_mle_weight"] == 1.0
    assert payload["mle_chi2_weight"] == 1.0
    assert payload["n_mle"] >= 1 and payload["n_chi2"] >= 1
    assert np.isfinite(payload["mle_loss_pt"])
    # Chi² term is the stubbed value ÷ valid-trial count.
    assert payload["chi2_loss_pt"] == pytest.approx(12.0 / payload["n_chi2"])


def test_simulateDDM_pure_mle_dry_run_skips_chi2(monkeypatch):
    # Default mle_chi2_weight=0.0 ⇒ the joint path is never taken and the Chi²
    # simulation is never invoked (the stub would raise if it were).
    def _boom(**k):
        raise AssertionError("Chi² must not run when chi2-weight == 0")
    monkeypatch.setattr(fit, "makeOneRun", _boom)
    payload = _dry_run(monkeypatch)["S1"]
    assert "joint_mode" not in payload
    assert "chi2_loss_pt" not in payload
    assert np.isfinite(payload["neg_loglik"])


# ---------------------------------------------------------------------------
# Validation + filename invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w_mle,w_chi2,needle", [
    (1.0, -1.0, "mle_chi2_weight"),
    (1.0, float("nan"), "mle_chi2_weight"),
    (1.0, float("inf"), "mle_chi2_weight"),
    (-2.0, 0.5, "mle_mle_weight"),
])
def test_validate_rejects_bad_weights(w_mle, w_chi2, needle):
    with pytest.raises(ValueError, match=needle):
        validate_mle_config(_config(w_mle, w_chi2))


def test_default_weights_validate():
    # Pure-MLE default (1.0, 0.0) must pass validation untouched.
    validate_mle_config(_config(1.0, 0.0))


def test_weights_excluded_from_evolveFP():
    params = inspect.signature(fit.evolveFP).parameters
    assert "mle_mle_weight" not in params
    assert "mle_chi2_weight" not in params
    # The joint weights don't enter the saved-fit path, so an A/B run overwrites
    # the same pickle (by design).
    path = fit.evolveFP(
        drift_fn_str="Classic", bias_fn_str="None_", noise_fn_str="Normal(0, 1)",
        t_dur=3, dt=0.005, is_loss_no_dir=False, fit_mode="mle")
    assert str(path).endswith("_3s_dt0.005.pkl")
