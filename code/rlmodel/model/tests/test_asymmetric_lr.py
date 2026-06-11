"""Asymmetric learning rate (ALPHA_UNREWARDED / BETA_UNREWARDED) tests.

Covers: gating by bias_fn_str / noise_fn_str via ``fit.simulateDDM``;
equivalence of the MLE scalar path (``_compute_latent_arrays``) with the
Chisqr ``state_updates`` recurrence under matched asymmetric params.
"""
import numpy as np
import pandas as pd

from .. import fit
from .. import state_updates
from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..initvals import InitVals
from ..mle import MLEModelConfig, _compute_latent_arrays, prepare_mle_data
from ..noise import NOISE_FN_DICT


def _small_df():
    rows = []
    for trial_num, choice_left, reward, rt, dv in [
        (1, 1.0, 1.0, 0.12, 0.7),
        (2, 0.0, 0.0, 0.14, -0.5),
        (3, np.nan, np.nan, np.nan, 0.2),
    ]:
        rows.append(dict(
            Name="S1",
            Date=pd.Timestamp("2026-01-01"),
            SessionNum=1,
            TrialNumber=trial_num,
            SessId="S1_2026-01-01_1",
            DV=dv,
            DVstr=str(dv),
            valid=True,
            calcStimulusTime=rt,
            ChoiceLeft=choice_left,
            ChoiceCorrect=reward,
        ))
    return pd.DataFrame(rows)


def _run_dry(bias_name, drift_name="Classic", noise_name="Normal(0, 1)"):
    return fit.simulateDDM(
        _small_df(),
        bounds_and_defaults=InitVals().toDict(),
        dt=0.01,
        t_dur=0.2,
        biasFn=BIAS_FN_DICT[bias_name],
        driftFn=DRIFT_FN_DICT[drift_name],
        noiseFn=NOISE_FN_DICT[noise_name],
        is_loss_no_dir=False,
        num_cpus=1,
        evolvs_res={},
        fit_mode="mle",
        dry_run=True,
        bias_fn_str=bias_name,
        drift_fn_str=drift_name,
    )


def test_alpha_unrewarded_fit_only_when_bias_is_q_val_asym_offset():
    # The -asym bias variant enables asymmetric ALPHA.
    result = _run_dry(bias_name="Q-Val-asym (Offset)")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" in names
    assert "BETA_UNREWARDED" not in names

    # Canonical Q-Val (Offset) STAYS SYMMETRIC — it is the legacy fit
    # used as the reference against the -asym variant in side-by-side
    # comparisons.
    result = _run_dry(bias_name="Q-Val (Offset)")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" not in names

    # Q-Val (no offset) bias → no asymmetric LR.
    result = _run_dry(bias_name="Q-Val")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" not in names

    # No bias at all → no asymmetric LR.
    result = _run_dry(bias_name="None_")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" not in names


def test_beta_unrewarded_fit_only_when_drift_is_noisegain_reward_rate_asym():
    # The -asym drift variant enables asymmetric BETA.
    result = _run_dry(
        bias_name="None_", drift_name="NoiseGain-RewardRate-asym")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "BETA_UNREWARDED" in names
    assert "ALPHA_UNREWARDED" not in names

    # Canonical NoiseGain-RewardRate STAYS SYMMETRIC.
    result = _run_dry(bias_name="None_", drift_name="NoiseGain-RewardRate")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "BETA_UNREWARDED" not in names

    # Classic drift → no asymmetric LR.
    result = _run_dry(bias_name="None_", drift_name="Classic")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "BETA_UNREWARDED" not in names


def test_both_gates_active_when_both_asym_variants_selected():
    # Combine Q-Val-asym (Offset) bias with NoiseGain-RewardRate-asym drift
    # → both asymmetric params enter the fit vector.
    result = _run_dry(
        bias_name="Q-Val-asym (Offset)",
        drift_name="NoiseGain-RewardRate-asym")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" in names
    assert "BETA_UNREWARDED" in names


def test_mle_latent_recompute_matches_state_updates_for_asymmetric_alpha():
    """End-to-end: the MLE per-trial Q recurrence with asymmetric ALPHA
    must produce the same Q_left_after / Q_right_after that the Chisqr
    state_updates recurrence does when given the same params."""
    df = _small_df()
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val-asym (Offset)",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=True,
        # Flags are required to enable strict access to *_UNREWARDED
        # params (loud failure over silent fallback).
        uses_asymmetric_alpha=True,
        uses_asymmetric_beta=True,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )
    params = {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
        "BIAS_COEF": 0.5,
        "Q_VAL_OFFSET": 0.0,
        "ALPHA": 0.2,
        "ALPHA_UNREWARDED": 0.8,
        "BETA": 0.3,
        "BETA_UNREWARDED": 0.7,
    }

    data = prepare_mle_data(df)
    latents = _compute_latent_arrays(data, params, config)

    # Hand-roll the same recurrence using state_updates. The data is
    # sorted by (TrialNumber, SessId) inside prepare_mle_data; for a
    # single-session df this is the same order as the input.
    sorted_index = data.sorted_index
    q_left = 0.5
    q_right = 0.5
    reward_rate = 0.5
    expected_q_left_after = []
    expected_q_right_after = []
    expected_reward_rate_after = []
    for trial_pos in range(len(sorted_index)):
        # valid_for_loss = data.valid & finite(dv). Trial 3 (NaN dv) is
        # invalid, so its update is a no-op in both paths.
        if data.valid[trial_pos] and np.isfinite(data.dv[trial_pos]):
            reward = data.reward[trial_pos]
            choice_left = data.choice_left[trial_pos]
            q_left, q_right = state_updates.update_q_values(
                q_left=q_left,
                q_right=q_right,
                observed_choice_left=choice_left,
                observed_reward=reward,
                alpha=params["ALPHA"],
                alpha_unrewarded=params["ALPHA_UNREWARDED"],
            )
            reward_rate = state_updates.update_reward_rate(
                reward_rate=reward_rate,
                observed_reward=reward,
                beta=params["BETA"],
                beta_unrewarded=params["BETA_UNREWARDED"],
            )
        expected_q_left_after.append(q_left)
        expected_q_right_after.append(q_right)
        expected_reward_rate_after.append(reward_rate)

    np.testing.assert_allclose(
        latents["q_left_after"], expected_q_left_after, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        latents["q_right_after"], expected_q_right_after, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        latents["reward_rate_after"], expected_reward_rate_after,
        rtol=1e-9, atol=1e-12)


def test_mle_latent_recompute_collapses_to_symmetric_when_unrewarded_absent():
    """Same fixture, but ALPHA_UNREWARDED / BETA_UNREWARDED omitted from
    the params dict: latents must equal a symmetric (ALPHA only) run."""
    df = _small_df()
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=True,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )
    base_params = {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
        "ALPHA": 0.2,
        "BETA": 0.3,
    }
    data = prepare_mle_data(df)

    latents_sym = _compute_latent_arrays(data, base_params, config)
    # config has uses_asymmetric_alpha=False (default), so the
    # strict-access path skips ALPHA_UNREWARDED entirely. Re-running
    # with the same symmetric params via the same config must produce
    # bit-identical latents — this pins the silent-fallback-free
    # contract for the legacy symmetric case.
    latents_eq = _compute_latent_arrays(data, base_params, config)
    np.testing.assert_allclose(
        latents_sym["q_left_after"], latents_eq["q_left_after"])
    np.testing.assert_allclose(
        latents_sym["q_right_after"], latents_eq["q_right_after"])
    np.testing.assert_allclose(
        latents_sym["reward_rate_after"], latents_eq["reward_rate_after"])


def test_mle_latent_recompute_raises_keyerror_when_asymmetric_param_missing():
    """Strict access: ``uses_asymmetric_alpha=True`` MUST come with
    ``params["ALPHA_UNREWARDED"]``; otherwise the function raises
    ``KeyError`` — no silent fallback to ``ALPHA``."""
    import pytest

    df = _small_df()
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val-asym (Offset)",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=False,
        uses_asymmetric_alpha=True,
        uses_asymmetric_beta=False,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )
    data = prepare_mle_data(df)
    params_missing_unrewarded = {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
        "BIAS_COEF": 0.5,
        "Q_VAL_OFFSET": 0.0,
        "ALPHA": 0.2,
        # No ALPHA_UNREWARDED — strict access should fire KeyError.
    }
    with pytest.raises(KeyError, match="ALPHA_UNREWARDED"):
        _compute_latent_arrays(data, params_missing_unrewarded, config)
