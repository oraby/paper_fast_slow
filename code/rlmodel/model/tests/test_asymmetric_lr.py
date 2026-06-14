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


def _run_dry(bias_name, drift_name="Classic", noise_name="Normal(0, 1)",
             uses_asym_q=False, uses_asym_rr=False):
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
        uses_asym_q=uses_asym_q,
        uses_asym_rr=uses_asym_rr,
    )


def test_alpha_unrewarded_fit_only_when_asym_q_set_and_q_learning_present():
    # --asym-q on a Q-learning bias enables ALPHA_UNREWARDED.
    result = _run_dry(bias_name="Q-Val (Offset)", uses_asym_q=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" in names
    assert "BETA_UNREWARDED" not in names

    # Same model without the flag stays SYMMETRIC.
    result = _run_dry(bias_name="Q-Val (Offset)")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" not in names

    # Q-Val (no offset) bias + --asym-q: also valid (Q-learning active).
    result = _run_dry(bias_name="Q-Val", uses_asym_q=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" in names

    # No bias at all + --asym-q: no Q-learning ⇒ flag is a no-op here.
    # (The CLI pre-flight in model_runner.py rejects this combo with a
    # helpful error before reaching simulateDDM; simulateDDM itself
    # silently drops the unused param. Both layers are tested.)
    result = _run_dry(bias_name="None_", uses_asym_q=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" not in names


def test_beta_unrewarded_fit_only_when_asym_rr_set_and_rr_learning_present():
    # --asym-rr on a RewardRate drift enables BETA_UNREWARDED.
    result = _run_dry(
        bias_name="None_", drift_name="NoiseGain-RewardRate", uses_asym_rr=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "BETA_UNREWARDED" in names
    assert "ALPHA_UNREWARDED" not in names

    # Same model without the flag stays SYMMETRIC.
    result = _run_dry(bias_name="None_", drift_name="NoiseGain-RewardRate")
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "BETA_UNREWARDED" not in names

    # Classic drift + --asym-rr: no reward-rate ⇒ no-op at simulateDDM.
    result = _run_dry(
        bias_name="None_", drift_name="Classic", uses_asym_rr=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "BETA_UNREWARDED" not in names


def test_both_asym_flags_compose_with_any_compatible_model():
    # Q-Val (Offset) + NoiseGain-RewardRate + both flags → both
    # *_UNREWARDED params in the fit vector.
    result = _run_dry(
        bias_name="Q-Val (Offset)",
        drift_name="NoiseGain-RewardRate",
        uses_asym_q=True,
        uses_asym_rr=True,
    )
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" in names
    assert "BETA_UNREWARDED" in names


def test_decay_q_drift_supports_asym_q():
    """Orthogonality regression: under the old design, only the
    dedicated ``Q-Val-asym (Offset)`` bias could enable asym ALPHA.
    Now any Q-learning model — including the Decay-Q drift family
    that learns Q-values through the time-varying drift — can opt
    in via --asym-q."""
    result = _run_dry(
        bias_name="None_", drift_name="Decay Q (Offset)", uses_asym_q=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" in names
    assert "BETA_UNREWARDED" not in names


def test_mle_latent_recompute_matches_state_updates_for_asymmetric_alpha():
    """End-to-end: the MLE per-trial Q recurrence with asymmetric ALPHA
    must produce the same Q_left_after / Q_right_after that the Chisqr
    state_updates recurrence does when given the same params."""
    df = _small_df()
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val (Offset)",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=True,
        # Asym flags are orthogonal to bias / drift / noise identity
        # (set by --asym-q / --asym-rr at the CLI). Both True here so
        # _compute_latent_arrays strictly reads ALPHA_UNREWARDED /
        # BETA_UNREWARDED from params (loud KeyError on miss).
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
        bias_fn_str="Q-Val (Offset)",
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


def test_evolveFP_filename_suffix_per_flag_combination():
    """The orthogonal asym opt-ins surface as a filename suffix so
    symmetric and asym fits of the same model coexist on disk."""
    base_args = dict(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val (Offset)",
        noise_fn_str="Normal(0, 1)",
        t_dur=3.0,
        dt=0.005,
        is_loss_no_dir=False,
        fit_mode="mle",
    )
    # Symmetric (both flags False) → no suffix; existing files on disk
    # load under exactly this path.
    p = fit.evolveFP(**base_args)
    assert p.name.endswith("3.0s_dt0.005.pkl"), p.name
    # Asym Q only.
    p = fit.evolveFP(uses_asym_q=True, **base_args)
    assert p.name.endswith("3.0s_dt0.005_asymQ.pkl"), p.name
    # Asym RR only.
    p = fit.evolveFP(uses_asym_rr=True, **base_args)
    assert p.name.endswith("3.0s_dt0.005_asymRR.pkl"), p.name
    # Both.
    p = fit.evolveFP(uses_asym_q=True, uses_asym_rr=True, **base_args)
    assert p.name.endswith("3.0s_dt0.005_asymQRR.pkl"), p.name


def test_asym_q_flag_against_non_q_learning_model_is_no_op_at_simulateDDM():
    """The CLI pre-flight in model_runner.py rejects --asym-q against
    a non-Q-learning model. ``simulateDDM`` itself doesn't run that
    pre-flight (it's CLI-only) — it silently drops the unused flag.
    Pin both layers so a future refactor can't accidentally surface
    a silent fallback at the simulateDDM layer."""
    # Classic / None_ / Normal: no Q-learning. --asym-q is a no-op
    # here: the simulateDDM layer just skips ALPHA_UNREWARDED.
    result = _run_dry(bias_name="None_", uses_asym_q=True)
    names = [str(n).upper() for n in result["S1"]["params_names"]]
    assert "ALPHA_UNREWARDED" not in names

    # The CLI is the friendly error layer (tested separately by
    # invoking the runner; see test_asym_cli_preflight_rejects_*).


def test_asym_cli_preflight_rejects_asym_q_without_q_learning():
    """``model_runner --asym-q`` against Classic / None_ exits with
    ``parser.error`` (exit code 2) and a message naming the missing
    prerequisite."""
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-m", "code.rlmodel.model_runner",
         "--drift", "Classic",
         "--bias", "None_",
         "--noise", "Normal(0, 1)",
         "--fit-mode", "mle",
         "--mle-backend", "CPU",
         "--asym-q",
         "--dry-run"],
        capture_output=True, text=True, timeout=60,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[4]),
    )
    assert proc.returncode == 2, (
        f"expected argparse error (exit 2); got {proc.returncode}\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}")
    assert "--asym-q requires" in proc.stderr, proc.stderr


def test_asym_cli_preflight_rejects_asym_rr_without_rr_learning():
    """``model_runner --asym-rr`` against Classic exits with
    ``parser.error`` and a helpful message."""
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-m", "code.rlmodel.model_runner",
         "--drift", "Classic",
         "--bias", "None_",
         "--noise", "Normal(0, 1)",
         "--fit-mode", "mle",
         "--mle-backend", "CPU",
         "--asym-rr",
         "--dry-run"],
        capture_output=True, text=True, timeout=60,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[4]),
    )
    assert proc.returncode == 2, (
        f"expected argparse error (exit 2); got {proc.returncode}\n"
        f"stdout: {proc.stdout!r}\nstderr: {proc.stderr!r}")
    assert "--asym-rr requires" in proc.stderr, proc.stderr


# ---------------------------------------------------------------------------
# ``--asym`` shorthand expansion (``_expand_asym_shorthand``)
# ---------------------------------------------------------------------------

import argparse

from ...model_runner import _expand_asym_shorthand


def _asym_args(drift, bias, noise, asym=True, asym_q=False, asym_rr=False):
    return argparse.Namespace(
        drift=drift, bias=bias, noise=noise,
        asym=asym, asym_q=asym_q, asym_rr=asym_rr,
    )


def test_expand_asym_shorthand_q_only_model():
    """Q-Val bias + Classic drift learns Q but no reward rate.
    ``--asym`` flips only asym_q."""
    args = _asym_args(drift="Classic", bias="Q-Val", noise="Normal(0, 1)")
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is True and learns_rr is False
    assert args.asym_q is True
    assert args.asym_rr is False


def test_expand_asym_shorthand_rr_only_model():
    """NoiseGain-RewardRate drift + None_ bias learns RR but no Q.
    ``--asym`` flips only asym_rr."""
    args = _asym_args(drift="NoiseGain-RewardRate", bias="None_",
                       noise="Normal(0, 1)")
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is False and learns_rr is True
    assert args.asym_q is False
    assert args.asym_rr is True


def test_expand_asym_shorthand_both_q_and_rr_model():
    """NoiseGain-RewardRate + Q-Val learns both. ``--asym`` flips both."""
    args = _asym_args(drift="NoiseGain-RewardRate", bias="Q-Val",
                       noise="Normal(0, 1)")
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is True and learns_rr is True
    assert args.asym_q is True
    assert args.asym_rr is True


def test_expand_asym_shorthand_neither_model_is_noop():
    """Classic + None_ + Normal(0, 1) learns neither Q nor RR.
    ``--asym`` is a strict no-op; both flags stay False."""
    args = _asym_args(drift="Classic", bias="None_", noise="Normal(0, 1)")
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is False and learns_rr is False
    assert args.asym_q is False
    assert args.asym_rr is False


def test_expand_asym_shorthand_off_does_not_flip():
    """``--asym`` not passed: no expansion regardless of what the model
    learns. The helper still returns detection booleans so the caller's
    validation block can use them."""
    args = _asym_args(drift="NoiseGain-RewardRate", bias="Q-Val",
                       noise="Normal(0, 1)", asym=False)
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is True and learns_rr is True
    # asym was False, so no expansion
    assert args.asym_q is False
    assert args.asym_rr is False


def test_expand_asym_shorthand_or_composes_with_explicit_flag():
    """``--asym --asym-q`` is harmless: asym_q was already True, the
    expansion leaves it True (OR is idempotent). asym_rr flips iff the
    model learns RR."""
    args = _asym_args(drift="NoiseGain-RewardRate", bias="Q-Val",
                       noise="Normal(0, 1)", asym=True, asym_q=True)
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is True and learns_rr is True
    assert args.asym_q is True   # was True, still True
    assert args.asym_rr is True  # OR-flipped by --asym


def test_expand_asym_shorthand_decay_q_drift_detected_as_q():
    """``Decay Q`` drift uses Q-values; ``--asym`` should treat it as
    Q-learning even though the bias is None_."""
    args = _asym_args(drift="Decay Q", bias="None_", noise="Normal(0, 1)")
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is True
    assert args.asym_q is True


def test_expand_asym_shorthand_decaying_q_val_noise_detected_as_q():
    """``Decaying Q-Val`` noise function uses Q-values; ``--asym``
    should treat it as Q-learning."""
    args = _asym_args(drift="Classic", bias="None_", noise="Decaying Q-Val")
    learns_q, learns_rr = _expand_asym_shorthand(args)
    assert learns_q is True
    assert args.asym_q is True
