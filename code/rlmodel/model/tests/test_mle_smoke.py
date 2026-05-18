import numpy as np
import pandas as pd

from .. import fit
from ..bias import BIAS_FN_DICT
from ..drift import DRIFT_FN_DICT
from ..initvals import InitVals
from ..mle import MLEModelConfig, objective_from_vector, result_payload
from ..mle import evaluate_neg_loglik
from ..mle_likelihood import LOGLIK_FLOOR, trial_choice_rt_loglik
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


def _params_for(include_q, include_reward_rate, decay_q=False, q_bias=False):
    params = {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
    }
    if include_q:
        params["ALPHA"] = 0.3
    if include_reward_rate:
        params["BETA"] = 0.3
    if q_bias:
        params["BIAS_COEF"] = 0.5
        params["Q_VAL_OFFSET"] = 0.0
    if decay_q:
        params["Q_VAL_DECAY_RATE"] = 1.0
        params["Q_VAL_COEF"] = 0.5
        params.setdefault("Q_VAL_OFFSET", 0.0)
    return params


def _payload_for_variant(drift_name, bias_name, noise_name, include_q,
                         include_reward_rate, decay_q=False, q_bias=False):
    params = _params_for(include_q, include_reward_rate, decay_q, q_bias)
    params_names = np.asarray(list(params.keys()))
    params_init = np.asarray(list(params.values()), dtype=float)
    params_bounds = np.asarray([(0.0, 2.0)] * len(params_names), dtype=float)
    config = MLEModelConfig(
        drift_fn_str=drift_name,
        bias_fn_str=bias_name,
        noise_fn_str=noise_name,
        include_Q=include_q,
        include_RewardRate=include_reward_rate,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )
    return result_payload(
        optim_res=None,
        params_names=params_names,
        params_init=params_init,
        params_bounds=params_bounds,
        subject_df=_small_df(),
        model_config=config,
    )


def test_mle_smoke_named_variants_have_finite_loss_and_latents():
    variants = [
        ("Classic", "None_", "Normal(0, 1)", False, False, False, False),
        ("Classic", "Q-Val", "Normal(0, 1)", True, False, False, True),
        ("NoiseGain-RewardRate", "None_", "Normal(0, 1)", False, True, False, False),
        ("NoiseGain-RewardRate", "Q-Val", "Normal(0, 1)", True, True, False, True),
        ("Decay Q", "None_", "Normal(0, 1)", True, False, True, False),
        ("Classic", "None_", "Decaying Q-Val", True, False, True, False),
    ]

    for drift_name, bias_name, noise_name, include_q, include_rr, decay_q, q_bias in variants:
        payload = _payload_for_variant(
            drift_name, bias_name, noise_name, include_q, include_rr, decay_q, q_bias)
        assert payload["fit_mode"] == "mle"
        assert payload["mle_observation_model"] == "choice_rt"
        assert np.isfinite(payload["neg_loglik"])
        assert np.isfinite(payload["aic"])
        assert np.isfinite(payload["bic"])
        assert payload["n_trials_loss"] == 3
        for col in [
            "mle_Q_left_before",
            "mle_Q_right_before",
            "mle_z",
            "mle_mu",
            "mle_sigma",
            "mle_loglik",
            "mle_valid_for_loss",
            "mle_survival_at_tmax",
            "mle_Q_left_after",
            "mle_Q_right_after",
        ]:
            assert col in payload["mle_df"].columns


def test_fit_dispatcher_mle_dry_run_returns_mle_payload():
    result = fit.simulateDDM(
        _small_df(),
        bounds_and_defaults=InitVals().toDict(),
        dt=0.01,
        t_dur=0.2,
        biasFn=BIAS_FN_DICT["None_"],
        driftFn=DRIFT_FN_DICT["Classic"],
        noiseFn=NOISE_FN_DICT["Normal(0, 1)"],
        is_loss_no_dir=False,
        num_cpus=1,
        evolvs_res={},
        fit_mode="mle",
        dry_run=True,
    )

    payload = result["S1"]
    assert payload["fit_mode"] == "mle"
    assert payload["mle_df"] is not None
    assert np.isfinite(payload["neg_loglik"])


def test_trial_likelihood_invalid_solver_inputs_are_floored():
    for sigma in [0.0, np.nan]:
        like = trial_choice_rt_loglik(
            observed_choice_left=1,
            observed_rt=0.2,
            z=0.0,
            mu=1.0,
            sigma=sigma,
            bound=1.0,
            non_decision_time=0.05,
            dt=0.01,
            dx=0.05,
            tmax=1.0,
        )
        assert np.isfinite(like.loglik)
        assert like.choice_prob_or_density == LOGLIK_FLOOR


def test_mle_objective_returns_finite_penalty_for_invalid_candidate():
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val (Offset)",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=False,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )
    names = np.array([
        "DRIFT_COEF",
        "NOISE_SIGMA",
        "BIAS_COEF",
        "Q_VAL_OFFSET",
        "BOUND",
        "ALPHA",
        "NON_DECISION_TIME",
    ])
    x = np.array([1.0, np.nan, 0.5, 0.0, 1.0, 0.3, 0.02])

    value = objective_from_vector(x, names, _small_df(), config)

    assert np.isfinite(value)
    assert value > 0


def test_mle_skips_likelihood_for_missing_dv_without_nan_propagation():
    df = _small_df()
    df.loc[0, "DV"] = np.nan
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val (Offset)",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=False,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )
    params = {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BIAS_COEF": 0.5,
        "Q_VAL_OFFSET": 0.0,
        "BOUND": 1.0,
        "ALPHA": 0.3,
        "NON_DECISION_TIME": 0.02,
    }

    result = evaluate_neg_loglik(params, df, config, return_df=True)

    assert np.isfinite(result.neg_loglik)
    assert result.n_trials_loss == 2
    assert not result.mle_df.loc[0, "mle_valid_for_loss"]
    assert np.isnan(result.mle_df.loc[0, "mle_loglik"])
