from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import state_updates
from .mle_likelihood import trial_choice_rt_loglik


SUPPORTED_MLE_BIASES = {"None_", "Q-Val", "Q-Val (Offset)"}


@dataclass(frozen=True)
class MLEModelConfig:
    drift_fn_str: str
    bias_fn_str: str
    noise_fn_str: str
    include_Q: bool
    include_RewardRate: bool
    dt: float
    t_dur: float
    dx: float = 0.02
    diffusion_backend: str = "auto"

    @property
    def uses_q_bias(self):
        return self.bias_fn_str in {"Q-Val", "Q-Val (Offset)"}

    @property
    def uses_decay_q_drift(self):
        return "Decay Q" in self.drift_fn_str

    @property
    def uses_decay_q_noise(self):
        return self.noise_fn_str == "Decaying Q-Val"


@dataclass
class MLEEvalResult:
    neg_loglik: float
    loglik: float
    n_trials_loss: int
    n_trials_total: int
    mle_df: pd.DataFrame | None = None


def validate_mle_config(model_config):
    if model_config.bias_fn_str not in SUPPORTED_MLE_BIASES:
        raise NotImplementedError(
            "MLE currently supports only bias functions: "
            f"{sorted(SUPPORTED_MLE_BIASES)}. Got {model_config.bias_fn_str!r}.")


def params_from_vector(x, params_names):
    return {str(k).upper(): float(v) for k, v in zip(params_names, x)}


def neg_loglik(params, df, model_config):
    return evaluate_neg_loglik(params, df, model_config, return_df=False).neg_loglik


def evaluate_neg_loglik(params, df, model_config, return_df=False):
    validate_mle_config(model_config)
    params = {str(k).upper(): float(v) for k, v in params.items()}

    total_loglik = 0.0
    n_trials_loss = 0
    rows = [] if return_df else None
    state_by_sess = {}

    sort_cols = [col for col in ["SessId", "TrialNumber"] if col in df.columns]
    iter_df = df.sort_values(sort_cols) if sort_cols else df

    for row_idx, trial in iter_df.iterrows():
        sess_id = trial["SessId"]
        if sess_id not in state_by_sess:
            state_by_sess[sess_id] = state_updates.initialize_latent_state(
                include_Q=model_config.include_Q,
                include_RewardRate=model_config.include_RewardRate,
            )
        state = state_by_sess[sess_id]

        q_rel_before = float(state_updates.compute_q_value(
            state.q_left, state.q_right))
        reward_rate_before = state.reward_rate
        z = _compute_z(state, params, model_config, q_rel_before)
        sigma = state_updates.compute_trial_sigma(
            _param(params, "NOISE_SIGMA"),
            reward_rate_before,
            model_config.include_RewardRate,
        )
        mu = _compute_mu(
            float(trial["DV"]), params, model_config, q_rel_before, sigma)
        bound = _param(params, "BOUND", 1.0)
        non_decision_time = _param(params, "NON_DECISION_TIME", 0.0)

        is_valid = bool(trial["valid"])
        choice_left = trial["ChoiceLeft"]
        reward = trial["ChoiceCorrect"]
        no_choice = is_valid and pd.isna(choice_left)
        contributes_likelihood = is_valid
        trial_like = None
        if contributes_likelihood:
            trial_like = trial_choice_rt_loglik(
                observed_choice_left=choice_left,
                observed_rt=trial["calcStimulusTime"],
                z=z,
                mu=mu,
                sigma=sigma,
                bound=bound,
                non_decision_time=non_decision_time,
                dt=model_config.dt,
                dx=model_config.dx,
                tmax=model_config.t_dur,
                diffusion_backend=model_config.diffusion_backend,
                no_choice=no_choice,
            )
            total_loglik += trial_like.loglik
            n_trials_loss += 1

        q_left_after = state.q_left
        q_right_after = state.q_right
        reward_rate_after = state.reward_rate
        if is_valid:
            if model_config.include_Q:
                q_left_after, q_right_after = state_updates.update_q_values(
                    state.q_left,
                    state.q_right,
                    None if pd.isna(choice_left) else choice_left,
                    None if pd.isna(reward) else reward,
                    _param(params, "ALPHA"),
                )
                q_left_after = float(q_left_after)
                q_right_after = float(q_right_after)
            if model_config.include_RewardRate:
                reward_rate_after = state_updates.update_reward_rate(
                    state.reward_rate,
                    None if pd.isna(reward) else reward,
                    _param(params, "BETA"),
                )
                reward_rate_after = float(reward_rate_after)
            state_by_sess[sess_id] = state_updates.LatentState(
                include_Q=state.include_Q,
                include_RewardRate=state.include_RewardRate,
                q_left=q_left_after,
                q_right=q_right_after,
                reward_rate=reward_rate_after,
            )

        if rows is not None:
            rows.append(_row_record(
                row_idx=row_idx,
                state=state,
                q_rel_before=q_rel_before,
                reward_rate_before=reward_rate_before,
                z=z,
                mu=mu,
                sigma=sigma,
                trial_like=trial_like,
                valid_for_loss=contributes_likelihood,
                q_left_after=q_left_after,
                q_right_after=q_right_after,
                reward_rate_after=reward_rate_after,
            ))

    mle_df = None
    if rows is not None:
        latents = pd.DataFrame(rows).set_index("_row_index")
        mle_df = df.copy().join(latents, how="left")

    return MLEEvalResult(
        neg_loglik=-float(total_loglik),
        loglik=float(total_loglik),
        n_trials_loss=n_trials_loss,
        n_trials_total=int(len(df)),
        mle_df=mle_df,
    )


def objective_from_vector(x, params_names, df, model_config):
    return neg_loglik(params_from_vector(x, params_names), df, model_config)


def result_payload(optim_res, params_names, params_init, params_bounds,
                   subject_df, model_config):
    if optim_res is None:
        x = np.asarray(params_init, dtype=float)
    else:
        x = np.asarray(optim_res.x, dtype=float)
    params = params_from_vector(x, params_names)
    eval_res = evaluate_neg_loglik(params, subject_df, model_config, return_df=True)
    k = len(params_names)
    n = max(eval_res.n_trials_loss, 1)
    return dict(
        fit_mode="mle",
        mle_observation_model="choice_rt",
        subject_df=subject_df,
        mle_df=eval_res.mle_df,
        dt=model_config.dt,
        dx=model_config.dx,
        t_dur=model_config.t_dur,
        noise_dt_scaling="sqrt_dt",
        include_Q=model_config.include_Q,
        include_RewardRate=model_config.include_RewardRate,
        params_names=np.asarray(params_names),
        params_init=np.asarray(params_init, dtype=float),
        params_bounds=np.asarray(params_bounds, dtype=float),
        OptimRes=optim_res,
        loglik=eval_res.loglik,
        neg_loglik=eval_res.neg_loglik,
        n_trials_loss=eval_res.n_trials_loss,
        n_trials_total=eval_res.n_trials_total,
        aic=2 * k - 2 * eval_res.loglik,
        bic=k * np.log(n) - 2 * eval_res.loglik,
        model_config=model_config,
    )


def make_objective(params_names, df, model_config):
    return partial(objective_from_vector, params_names=params_names,
                   df=df, model_config=model_config)


def _param(params, name, default=None):
    if name in params:
        return params[name]
    if default is not None:
        return default
    raise KeyError(f"missing MLE parameter {name!r}")


def _compute_z(state, params, model_config, q_rel_before):
    if not model_config.uses_q_bias:
        return 0.0
    return float(state_updates.compute_starting_point_z(
        state.q_left,
        state.q_right,
        delta=_param(params, "BIAS_COEF"),
        offset=_param(params, "Q_VAL_OFFSET", 0.0),
        include_Q=True,
    ))


def _compute_mu(coherence, params, model_config, q_rel_before, sigma):
    base_mu = _param(params, "DRIFT_COEF") * coherence
    if not model_config.uses_decay_q_drift and not model_config.uses_decay_q_noise:
        return base_mu

    n_t = int(round(model_config.t_dur / model_config.dt))
    mu = np.full(n_t, base_mu, dtype=float)
    if model_config.uses_decay_q_drift:
        indices = np.arange(n_t)
        decay_form = (1 - indices / n_t) ** _param(params, "Q_VAL_DECAY_RATE")
        q_for_drift = np.clip(q_rel_before + _param(params, "Q_VAL_OFFSET", 0.0),
                              -1, 1)
        mu += q_for_drift * decay_form * _param(params, "Q_VAL_COEF")
    if model_config.uses_decay_q_noise:
        mu += sigma * _decaying_q_noise(q_rel_before, params, n_t) / model_config.dt
    return mu


def _decaying_q_noise(q_rel_before, params, n_t):
    indices = np.arange(n_t)
    decay = 1 - (
        _param(params, "Q_VAL_DECAY_RATE") * np.log(indices + 1)
        / np.log(n_t + 1)
    )
    q_abs = abs(q_rel_before) * _param(params, "Q_VAL_COEF")
    decayed = np.maximum(q_abs - decay, 0)
    if q_rel_before < 0:
        decayed = -decayed
    return decayed


def _row_record(row_idx, state, q_rel_before, reward_rate_before, z, mu, sigma,
                trial_like, valid_for_loss, q_left_after, q_right_after,
                reward_rate_after):
    if trial_like is None:
        decision_time = np.nan
        choice_prob_or_density = np.nan
        loglik = np.nan
        survival = np.nan
        upper_prob = np.nan
        lower_prob = np.nan
    else:
        decision_time = trial_like.decision_time
        choice_prob_or_density = trial_like.choice_prob_or_density
        loglik = trial_like.loglik
        survival = trial_like.survival_at_tmax
        upper_prob = trial_like.upper_hit_prob_tmax
        lower_prob = trial_like.lower_hit_prob_tmax

    if np.isscalar(mu):
        mu_value = float(mu)
    else:
        mu_value = float(np.asarray(mu)[0])

    return {
        "_row_index": row_idx,
        "mle_Q_left_before": state.q_left,
        "mle_Q_right_before": state.q_right,
        "mle_Q_rel_before": q_rel_before,
        "mle_reward_rate_before": reward_rate_before,
        "mle_z": z,
        "mle_mu": mu_value,
        "mle_sigma": sigma,
        "mle_decision_time_observed": decision_time,
        "mle_choice_prob_or_density": choice_prob_or_density,
        "mle_loglik": loglik,
        "mle_valid_for_loss": valid_for_loss,
        "mle_survival_at_tmax": survival,
        "mle_upper_hit_prob_tmax": upper_prob,
        "mle_lower_hit_prob_tmax": lower_prob,
        "mle_Q_left_after": q_left_after,
        "mle_Q_right_after": q_right_after,
        "mle_reward_rate_after": reward_rate_after,
    }
