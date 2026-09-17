"""Posterior predictive simulation from MLE-fitted parameters.

MLE fitting uses observed-history propagation; this module is the inverse
operation: take fitted parameters and generate synthetic behavior. It is
deliberately separate from the MLE objective — a generative posterior
predictive check, not part of the optimization.

Default mode (``use_observed_history_for_inputs=False``): full chisq
simulation path with the fitted parameters. Simulated choices propagate Q/R
state forward — a complete synthetic session. Output dataframe has the same
``Sim*`` / ``Q_*`` / ``RewardRate`` columns as a chisq fit, so existing
non-MLE behavioral plots work unchanged.

Debug mode (``use_observed_history_for_inputs=True``): per-trial sampling
from the first-passage density computed under observed-history latents. Only
suitable for trial-level diagnostics; not a synthetic session.
"""
from __future__ import annotations

import pathlib
import pickle
from typing import Optional, Union

import numpy as np
import pandas as pd

from . import state_updates
from .bias import BIAS_FN_DICT
from .drift import DRIFT_FN_DICT
from .first_passage import first_passage_density
from .logic import makeOneRun
from .mle import (MLEModelConfig, _compute_mu, _compute_z, _param,
                  drift_scale_for_config, validate_mle_config)
from .noise import NOISE_FN_DICT
from .util import biasFnColsAndKwargs, driftFnColsAndKwargs, noiseFnColsAndKwargs


__all__ = [
    "fitted_params_from_pickle",
    "simulate_from_fitted_params",
    "simulate_from_result_pickle",
]


# Keys that go directly to makeOneRun's top-level kwargs (not into the per-fn
# *_kwargs dicts). Matches the convention used by fit.py.
_LOGIC_TOP_LEVEL_KEYS = {
    "NON_DECISION_TIME", "BOUND", "DRIFT_COEF", "NOISE_SIGMA", "ALPHA", "BETA",
}


def fitted_params_from_pickle(result_pickle: dict) -> dict[str, float]:
    """Extract fitted parameters from an MLE result pickle.

    Returns a dict keyed by UPPERCASE parameter name, suitable for passing
    to ``simulate_from_fitted_params``.
    """
    optim_res = result_pickle["OptimRes"]
    names = list(result_pickle["params_names"])
    if optim_res is None:
        x = np.asarray(result_pickle["params_init"], dtype=float)
    else:
        x = np.asarray(optim_res.x, dtype=float)
    return {str(name).upper(): float(val) for name, val in zip(names, x)}


def simulate_from_result_pickle(
    result_pickle: dict,
    df: Optional[pd.DataFrame] = None,
    *,
    n_repeats: int = 1,
    seed: Optional[int] = None,
    use_observed_history_for_inputs: bool = False,
) -> dict:
    """Convenience wrapper: extract params/config from a saved MLE pickle.

    If ``df`` is ``None``, falls back to ``result_pickle["subject_df"]``.
    """
    fitted_params = fitted_params_from_pickle(result_pickle)
    model_config = result_pickle["model_config"]
    if df is None:
        df = result_pickle["subject_df"]
    return simulate_from_fitted_params(
        df, fitted_params, model_config,
        n_repeats=n_repeats, seed=seed,
        use_observed_history_for_inputs=use_observed_history_for_inputs,
    )


def simulate_from_fitted_params(
    df: pd.DataFrame,
    fitted_params: dict[str, float],
    model_config: MLEModelConfig,
    *,
    n_repeats: int = 1,
    seed: Optional[int] = None,
    use_observed_history_for_inputs: bool = False,
) -> dict:
    """Run posterior predictive simulation with MLE-fitted parameters.

    Parameters
    ----------
    df : DataFrame
        Behavioral dataframe (same shape as the MLE input).
    fitted_params : dict
        Parameters by UPPERCASE name (e.g. ``DRIFT_COEF``, ``ALPHA``).
    model_config : MLEModelConfig
        Carries drift / bias / noise function names plus dt, dx, t_dur.
    n_repeats : int, default 1
        Number of independent simulation repeats. Each gets a different seed.
        Output is a long-format DataFrame with a ``Repeat`` column.
    seed : int, optional
        Base RNG seed. If ``None``, ``np.random.SeedSequence().entropy`` is
        used. Repeat ``r`` uses ``seed + r``.
    use_observed_history_for_inputs : bool, default False
        - ``False`` (default): full chisq simulation path. Simulated choices
          propagate Q/R state — a complete synthetic session.
        - ``True``: per-trial sampling under observed-history latents. Debug
          mode — does NOT propagate simulated choices.

    Returns
    -------
    dict
        Chisq-compatible result dict with keys ``sim_df`` (the simulated
        DataFrame), ``fitted_params``, ``model_config``, ``n_repeats``,
        ``seed``, ``mode``, plus the function dicts used.
    """
    validate_mle_config(model_config)
    fitted_params = {str(k).upper(): float(v) for k, v in fitted_params.items()}

    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31))

    if use_observed_history_for_inputs:
        sim_df = _simulate_observed_history(
            df, fitted_params, model_config, n_repeats=n_repeats, seed=seed)
        mode = "observed_history"
    else:
        sim_df = _simulate_chisq_path(
            df, fitted_params, model_config, n_repeats=n_repeats, seed=seed)
        mode = "simulated_history"

    return dict(
        sim_df=sim_df,
        fitted_params=fitted_params,
        model_config=model_config,
        n_repeats=n_repeats,
        seed=seed,
        mode=mode,
        fit_mode="posterior_predictive",
        noise_dt_scaling="sqrt_dt",
        drift_fn_str=model_config.drift_fn_str,
        bias_fn_str=model_config.bias_fn_str,
        noise_fn_str=model_config.noise_fn_str,
        include_Q=model_config.include_Q,
        include_RewardRate=model_config.include_RewardRate,
        dt=model_config.dt,
        t_dur=model_config.t_dur,
    )


# --- chisq-path simulation -------------------------------------------------


def _split_kwargs(fitted_params, driftFn, biasFn, noiseFn):
    """Partition fitted_params into makeOneRun's top-level kwargs vs the
    function-specific kwargs dicts."""
    _, drift_extra = driftFnColsAndKwargs(driftFn)
    _, bias_extra = biasFnColsAndKwargs(biasFn)
    _, noise_extra = noiseFnColsAndKwargs(noiseFn)

    # _fnColsAndKargs returns lowercase param names from the function
    # signature; but the chisq-path drift/bias/noise functions declare their
    # extra params in UPPERCASE already (e.g. Q_VAL_OFFSET). The lookup
    # below is case-tolerant.
    def _grab(extra_names):
        out = {}
        for name in extra_names:
            up = name.upper()
            if up in fitted_params:
                out[name] = fitted_params[up]
        return out

    drift_kwargs = _grab(drift_extra)
    bias_kwargs = _grab(bias_extra)
    noise_kwargs = _grab(noise_extra)
    logic_kwargs = {k: fitted_params[k] for k in _LOGIC_TOP_LEVEL_KEYS
                    if k in fitted_params}
    return logic_kwargs, drift_kwargs, bias_kwargs, noise_kwargs


def _simulate_one_pass(df, fitted_params, model_config, seed):
    driftFn = DRIFT_FN_DICT[model_config.drift_fn_str]
    biasFn = BIAS_FN_DICT[model_config.bias_fn_str]
    noiseFn = NOISE_FN_DICT[model_config.noise_fn_str]

    bias_df_cols, _ = biasFnColsAndKwargs(biasFn)
    drift_df_cols, _ = driftFnColsAndKwargs(driftFn)
    noise_df_cols, _ = noiseFnColsAndKwargs(noiseFn)

    logic_kwargs, drift_kwargs, bias_kwargs, noise_kwargs = _split_kwargs(
        fitted_params, driftFn, biasFn, noiseFn)

    # makeOneRun requires ALPHA/BETA to be set if the corresponding state is
    # in scope, otherwise they default to NaN and assertions trip.
    if model_config.include_Q and "ALPHA" not in logic_kwargs:
        raise KeyError("include_Q is True but ALPHA missing from fitted_params")
    if model_config.include_RewardRate and "BETA" not in logic_kwargs:
        raise KeyError(
            "include_RewardRate is True but BETA missing from fitted_params")

    _loss, sim_df = makeOneRun(
        df,
        include_Q=model_config.include_Q,
        include_RewardRate=model_config.include_RewardRate,
        biasFn=biasFn,
        driftFn=driftFn,
        noiseFn=noiseFn,
        dt=model_config.dt,
        t_dur=model_config.t_dur,
        biasFn_df_cols=bias_df_cols,
        biasFn_kwargs=bias_kwargs,
        driftFn_df_cols=drift_df_cols,
        driftFn_kwargs=drift_kwargs,
        noiseFn_df_cols=noise_df_cols,
        noiseFn_kwargs=noise_kwargs,
        return_df=True,
        seed=seed,
        skip_loss=True,
        **logic_kwargs,
    )
    return _loss, sim_df


def _simulate_chisq_path(df, fitted_params, model_config, *, n_repeats, seed):
    pieces = []
    for r in range(n_repeats):
        loss, sim_df = _simulate_one_pass(
            df, fitted_params, model_config, seed=seed + r)
        sim_df = sim_df.copy()
        sim_df["RepeatIdx"] = r
        sim_df["SessId"] = sim_df.apply(
            lambda x: f"{x['Name']}_{x['Date']}_{x['SessionNum']}_{r}]",
            axis=1)
        sim_df["Seed"] = seed + r
        sim_df["Loss"] = loss
        sim_df["driftFn"] = model_config.drift_fn_str
        sim_df["biasFn"] = model_config.bias_fn_str
        sim_df["noiseFn"] = model_config.noise_fn_str
        pieces.append(sim_df)
    return pd.concat(pieces, ignore_index=True)


# --- observed-history sampling (debug mode) --------------------------------


def _sample_one_trial(fpr, rng, non_decision_time, tmax):
    """Inverse-CDF sample of (rt, choice_left) from a FirstPassageResult."""
    dt = float(fpr.metadata["dt"])
    f_upper = np.asarray(fpr.f_upper)
    f_lower = np.asarray(fpr.f_lower)
    p_upper = float(f_upper.sum()) * dt
    p_lower = float(f_lower.sum()) * dt
    p_no_choice = max(1.0 - p_upper - p_lower, 0.0)

    u = rng.random()
    if u < p_upper:
        cdf = np.cumsum(f_upper) * dt / p_upper
        idx = int(np.searchsorted(cdf, rng.random()))
        idx = min(idx, len(f_upper) - 1)
        decision_time = float(fpr.times[idx])
        return decision_time + non_decision_time, 1
    if u < p_upper + p_lower:
        cdf = np.cumsum(f_lower) * dt / p_lower
        idx = int(np.searchsorted(cdf, rng.random()))
        idx = min(idx, len(f_lower) - 1)
        decision_time = float(fpr.times[idx])
        return decision_time + non_decision_time, 0
    # No-choice
    return tmax + non_decision_time, np.nan


def _simulate_observed_history(df, fitted_params, model_config, *,
                               n_repeats, seed):
    """Per-trial sampling under observed-history latents (debug mode)."""
    sort_cols = [col for col in ["SessId", "TrialNumber"] if col in df.columns]
    iter_df = df.sort_values(sort_cols) if sort_cols else df

    pieces = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        state_by_sess = {}
        sim_rt = np.full(len(iter_df), np.nan)
        sim_choice_left = np.full(len(iter_df), np.nan)
        sim_starting_point = np.full(len(iter_df), np.nan)

        for i, (_row_idx, trial) in enumerate(iter_df.iterrows()):
            sess_id = trial["SessId"]
            if sess_id not in state_by_sess:
                state_by_sess[sess_id] = state_updates.initialize_latent_state(
                    include_Q=model_config.include_Q,
                    include_RewardRate=model_config.include_RewardRate,
                )
            state = state_by_sess[sess_id]

            q_rel_before = float(state_updates.compute_q_value(
                state.q_left, state.q_right))
            sigma = state_updates.compute_trial_sigma(
                _param(fitted_params, "NOISE_SIGMA"),
                state.reward_rate,
                model_config.include_RewardRate,
                rr_channel=model_config.sigma_rr_channel,
            )
            z = _compute_z(state, fitted_params, model_config, q_rel_before)
            mu = _compute_mu(
                float(trial["DV"]), fitted_params,
                drift_scale=drift_scale_for_config(
                    state.reward_rate, model_config))
            bound = _param(fitted_params, "BOUND", 1.0)
            non_decision_time = _param(fitted_params, "NON_DECISION_TIME", 0.0)

            fpr = first_passage_density(
                z, mu, sigma, bound,
                model_config.dt, model_config.dx, model_config.t_dur)
            rt, choice_left = _sample_one_trial(
                fpr, rng, non_decision_time, model_config.t_dur)
            sim_rt[i] = rt
            sim_choice_left[i] = choice_left
            sim_starting_point[i] = z * bound

            # Propagate Q/R from OBSERVED choice/reward (teacher-forced).
            if bool(trial["valid"]):
                choice_left_obs = trial["ChoiceLeft"]
                reward_obs = trial["ChoiceCorrect"]
                q_left_after = state.q_left
                q_right_after = state.q_right
                reward_rate_after = state.reward_rate
                if model_config.include_Q:
                    q_left_after, q_right_after = state_updates.update_q_values(
                        state.q_left, state.q_right,
                        None if pd.isna(choice_left_obs) else choice_left_obs,
                        None if pd.isna(reward_obs) else reward_obs,
                        _param(fitted_params, "ALPHA"),
                    )
                    q_left_after = float(q_left_after)
                    q_right_after = float(q_right_after)
                if model_config.include_RewardRate:
                    reward_rate_after = state_updates.update_reward_rate(
                        state.reward_rate,
                        None if pd.isna(reward_obs) else reward_obs,
                        _param(fitted_params, "BETA"),
                    )
                    reward_rate_after = float(reward_rate_after)
                state_by_sess[sess_id] = state_updates.LatentState(
                    include_Q=state.include_Q,
                    include_RewardRate=state.include_RewardRate,
                    q_left=q_left_after,
                    q_right=q_right_after,
                    reward_rate=reward_rate_after,
                )

        sim_df = iter_df.copy()
        sim_df["SimRT"] = sim_rt
        sim_df["SimChoiceLeft"] = sim_choice_left
        sim_df["SimStartingPoint"] = sim_starting_point
        # Reconstruct SimChoiceCorrect from DV sign and SimChoiceLeft.
        dv = sim_df["DV"].to_numpy()
        correct = np.full(len(sim_df), np.nan)
        not_nan = ~np.isnan(sim_choice_left)
        correct[not_nan & (dv > 0) & (sim_choice_left == 1)] = 1.0
        correct[not_nan & (dv < 0) & (sim_choice_left == 0)] = 1.0
        correct[not_nan & (dv > 0) & (sim_choice_left == 0)] = 0.0
        correct[not_nan & (dv < 0) & (sim_choice_left == 1)] = 0.0
        sim_df["SimChoiceCorrect"] = correct
        sim_df["RepeatIdx"] = r
        pieces.append(sim_df)

    return pd.concat(pieces, ignore_index=True)
