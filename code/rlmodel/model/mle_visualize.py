"""Visualization and one-trial debug utilities for the MLE path.

Three public functions:

- ``visualize_first_passage_result``: stacked plot of f_upper, f_lower, survival
  over time, with optional observed RT/choice markers.
- ``visualize_probability_flow``: heatmap of the accumulator distribution over
  (time, x) for a logged result from the reference backend.
- ``debug_one_trial_mle_flow``: end-to-end debug helper. Reconstructs Q/R from
  observed history up to a given trial, computes z/mu/sigma, calls the logged
  reference backend, computes the trial likelihood, and produces both plots.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import state_updates
from .diffusion import FirstPassageResult
from .first_passage import first_passage_density
from .mle import (
    _compute_mu,
    _compute_z,
    _param,
    drift_scale_for_config,
    validate_mle_config,
)
from .mle_likelihood import trial_choice_rt_loglik


__all__ = [
    "DebugOneTrialResult",
    "debug_one_trial_mle_flow",
    "visualize_first_passage_result",
    "visualize_probability_flow",
]


@dataclass
class DebugOneTrialResult:
    """Structured return value from ``debug_one_trial_mle_flow``."""
    trial_index: object
    state_before: state_updates.LatentState
    q_rel_before: float
    z: float
    mu: float | np.ndarray
    sigma: float
    bound: float
    non_decision_time: float
    first_passage_result: FirstPassageResult
    trial_likelihood: object  # mle_likelihood.TrialLikelihood
    figure_paths: dict[str, pathlib.Path]


def visualize_first_passage_result(
    result: FirstPassageResult,
    observed_choice_left: Optional[int] = None,
    observed_rt: Optional[float] = None,
    non_decision_time: Optional[float] = None,
    save_path: Optional[Union[str, pathlib.Path]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot upper / lower first-passage densities and the survival mass.

    Parameters
    ----------
    result : FirstPassageResult
        Output of ``first_passage_density``.
    observed_choice_left : 1 or 0, optional
        If provided, the corresponding subplot is annotated. ``1`` maps to the
        upper bound (matching the simulation convention).
    observed_rt : float, optional
        Vertical dashed line at the observed RT. If ``non_decision_time`` is
        also provided, a second line is drawn at the decision time
        (``RT - T0``) on the density plots.
    non_decision_time : float, optional
        Used to position the decision-time marker.
    save_path : str or Path, optional
        If given, the figure is written to disk and not shown.
    title : str, optional
        Figure suptitle.
    """
    times = np.asarray(result.times)
    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    ax_u, ax_l, ax_s = axes

    ax_u.plot(times, np.asarray(result.f_upper), color="C0", lw=1.5)
    ax_u.set_ylabel("$f_{upper}(t)$")
    ax_u.set_title("Upper bound (ChoiceLeft = 1)")

    ax_l.plot(times, np.asarray(result.f_lower), color="C3", lw=1.5)
    ax_l.set_ylabel("$f_{lower}(t)$")
    ax_l.set_title("Lower bound (ChoiceLeft = 0)")

    ax_s.plot(times, np.asarray(result.survival), color="0.3", lw=1.5)
    ax_s.set_ylabel("Survival $S(t)$")
    ax_s.set_xlabel("Time (s)")
    ax_s.set_ylim(0, 1.02)

    if observed_rt is not None and not np.isnan(observed_rt):
        for ax in axes:
            ax.axvline(observed_rt, color="black", ls="--", lw=1,
                       label="observed RT")
        if non_decision_time is not None:
            decision_time = observed_rt - non_decision_time
            if decision_time > 0:
                for ax in (ax_u, ax_l):
                    ax.axvline(decision_time, color="0.5", ls=":", lw=1,
                               label="decision time")

    if observed_choice_left is not None and not pd.isna(observed_choice_left):
        target_ax = ax_u if int(observed_choice_left) == 1 else ax_l
        target_ax.set_title(target_ax.get_title() + "  ← observed")

    for ax in axes:
        # Only show legend if at least one labeled artist exists.
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=8)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def visualize_probability_flow(
    logged_result: FirstPassageResult,
    save_path: Optional[Union[str, pathlib.Path]] = None,
    log_scale: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of the accumulator distribution p(x, t) over time.

    Requires a logged result (``p_by_t`` and ``x_grid`` populated) — i.e., one
    from the reference backend. Absorbing bounds at ±bound are marked.
    """
    if logged_result.p_by_t is None or logged_result.x_grid is None:
        raise ValueError(
            "visualize_probability_flow requires a logged FirstPassageResult; "
            "use backend='reference' to populate p_by_t and x_grid.")

    times = np.asarray(logged_result.times)
    x_grid = np.asarray(logged_result.x_grid)
    p_by_t = np.asarray(logged_result.p_by_t)
    bound = float(logged_result.metadata["bound"])

    fig, ax = plt.subplots(figsize=(8, 4))

    plot_p = p_by_t.T  # shape (n_x, n_t) so x runs vertical
    if log_scale:
        floor = 1e-12
        plot_p = np.log10(np.maximum(plot_p, floor))
        cbar_label = "log10 p(x, t)"
    else:
        cbar_label = "p(x, t)"

    extent = (float(times[0]), float(times[-1]), float(x_grid[0]),
              float(x_grid[-1]))
    im = ax.imshow(plot_p, aspect="auto", origin="lower", extent=extent,
                   cmap="viridis")
    ax.axhline(+bound, color="white", ls="--", lw=1)
    ax.axhline(-bound, color="white", ls="--", lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accumulator state $x$")
    fig.colorbar(im, ax=ax, label=cbar_label)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def debug_one_trial_mle_flow(
    df: pd.DataFrame,
    trial_index,
    params: dict,
    model_config,
    save_dir: Optional[Union[str, pathlib.Path]] = None,
) -> DebugOneTrialResult:
    """Reconstruct, evaluate, and visualize the MLE flow for one trial.

    Walks the dataframe in (SessId, TrialNumber) order, propagating Q/R from
    observed history exactly as ``evaluate_neg_loglik`` does, until reaching
    the row identified by ``trial_index``. Then computes z, mu, sigma, calls
    the *reference* (logged) first-passage backend, computes the trial
    likelihood, and produces both visualizations.

    Parameters
    ----------
    df : pandas DataFrame
        Behavioral dataframe (same shape consumed by ``evaluate_neg_loglik``).
    trial_index : object
        Label-based index into ``df`` (i.e., a value from ``df.index``).
    params : dict
        MLE parameters by UPPERCASE name (e.g., ``DRIFT_COEF``, ``BIAS_COEF``,
        ``NOISE_SIGMA``, ``ALPHA``, ``BETA``, ``NON_DECISION_TIME``).
    model_config : MLEModelConfig
    save_dir : str or Path, optional
        If given, figures are saved as ``trial_<idx>_densities.png`` and
        ``trial_<idx>_flow.png`` in this directory; otherwise figures are
        returned in memory.

    Returns
    -------
    DebugOneTrialResult
    """
    validate_mle_config(model_config)
    params = {str(k).upper(): float(v) for k, v in params.items()}

    sort_cols = [col for col in ["SessId", "TrialNumber"] if col in df.columns]
    iter_df = df.sort_values(sort_cols) if sort_cols else df

    if trial_index not in iter_df.index:
        raise KeyError(
            f"trial_index {trial_index!r} not found in df.index")

    state_by_sess = {}
    state_before = None
    trial = None
    for row_idx, row in iter_df.iterrows():
        sess_id = row["SessId"]
        if sess_id not in state_by_sess:
            state_by_sess[sess_id] = state_updates.initialize_latent_state(
                include_Q=model_config.include_Q,
                include_RewardRate=model_config.include_RewardRate,
            )
        if row_idx == trial_index:
            state_before = state_by_sess[sess_id]
            trial = row
            break

        # Advance state using observed choice/reward for this trial.
        is_valid = bool(row["valid"])
        if is_valid:
            state = state_by_sess[sess_id]
            choice_left = row["ChoiceLeft"]
            reward = row["ChoiceCorrect"]
            q_left_after = state.q_left
            q_right_after = state.q_right
            reward_rate_after = state.reward_rate
            if model_config.include_Q:
                q_left_after, q_right_after = state_updates.update_q_values(
                    state.q_left, state.q_right,
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
    if state_before is None or trial is None:
        raise KeyError(
            f"trial_index {trial_index!r} was not reached during iteration")

    q_rel_before = float(state_updates.compute_q_value(
        state_before.q_left, state_before.q_right))
    z = _compute_z(state_before, params, model_config, q_rel_before)
    sigma = state_updates.compute_trial_sigma(
        _param(params, "NOISE_SIGMA"),
        state_before.reward_rate,
        model_config.include_RewardRate,
        rr_channel=model_config.sigma_rr_channel,
    )
    mu = _compute_mu(
        float(trial["DV"]), params,
        drift_scale=drift_scale_for_config(
            state_before.reward_rate, model_config))
    bound = _param(params, "BOUND", 1.0)
    non_decision_time = _param(params, "NON_DECISION_TIME", 0.0)

    # Use the reference backend so p_by_t/x_grid are populated for the
    # probability-flow plot.
    fpr = first_passage_density(
        z, mu, sigma, bound,
        model_config.dt, model_config.dx, model_config.t_dur,
        backend="reference")

    is_valid = bool(trial["valid"])
    choice_left = trial["ChoiceLeft"]
    reward = trial["ChoiceCorrect"]
    no_choice = is_valid and pd.isna(choice_left)
    trial_like = trial_choice_rt_loglik(
        observed_choice_left=choice_left if not pd.isna(choice_left) else 1,
        observed_rt=trial["calcStimulusTime"],
        z=z, mu=mu, sigma=sigma, bound=bound,
        non_decision_time=non_decision_time,
        dt=model_config.dt, dx=model_config.dx, tmax=model_config.t_dur,
        diffusion_backend="reference",
        no_choice=no_choice,
    )

    figure_paths: dict[str, pathlib.Path] = {}
    densities_path = None
    flow_path = None
    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        densities_path = save_dir / f"trial_{trial_index}_densities.png"
        flow_path = save_dir / f"trial_{trial_index}_flow.png"

    visualize_first_passage_result(
        fpr,
        observed_choice_left=None if pd.isna(choice_left) else int(choice_left),
        observed_rt=trial["calcStimulusTime"],
        non_decision_time=non_decision_time,
        save_path=densities_path,
        title=f"Trial {trial_index} | z={z:.3f}, mu={_summarize_mu(mu):.3f}, "
              f"sigma={sigma:.3f}, loglik={trial_like.loglik:.3f}",
    )
    if densities_path is not None:
        figure_paths["densities"] = densities_path

    visualize_probability_flow(
        fpr,
        save_path=flow_path,
        title=f"Trial {trial_index} probability flow",
    )
    if flow_path is not None:
        figure_paths["flow"] = flow_path

    return DebugOneTrialResult(
        trial_index=trial_index,
        state_before=state_before,
        q_rel_before=q_rel_before,
        z=z,
        mu=mu,
        sigma=sigma,
        bound=bound,
        non_decision_time=non_decision_time,
        first_passage_result=fpr,
        trial_likelihood=trial_like,
        figure_paths=figure_paths,
    )


def _summarize_mu(mu) -> float:
    """Return mu as a scalar for titles. For time-varying mu, returns mean."""
    if np.isscalar(mu):
        return float(mu)
    arr = np.asarray(mu)
    return float(arr.mean())
