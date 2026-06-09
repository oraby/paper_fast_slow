"""Single-trial DDM viewer helpers for MLE notebooks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..first_passage import first_passage_density
from ..array_backend import resolve_array_backend
from ..mle import _compute_mu, evaluate_neg_loglik
from .data import MLEModelResult, fitted_params_from_result


@dataclass(frozen=True)
class SliderSpec:
    name: str
    value: float
    min: float
    max: float
    step: float


@dataclass(frozen=True)
class DDMFrameData:
    times: np.ndarray
    upper_density: np.ndarray
    lower_density: np.ndarray
    survival: np.ndarray
    x_grid: np.ndarray
    p_at_tmax: np.ndarray
    current_state_mass: np.ndarray
    bound: float
    dt: float
    tmax: float
    terminal_c: float
    current_step: int
    current_time: float
    current_survival_ratio: float
    terminal_no_decision_mass: float
    terminal_upper_mass: float
    terminal_lower_mass: float
    observed_choice_left: float | None
    observed_rt: float | None
    non_decision_time: float
    trial_index: object
    requested_backend: str
    actual_backend: str
    backend_warning: str | None
    # Contamination / lapse-mixture fields. Defaults preserve the legacy
    # (no-mixture) interpretation so tests and callers that don't care
    # about lapse keep working unchanged. ``lapse_baseline_density`` is
    # ``λ/(2·tmax)`` — the per-time-unit floor on any (choice, RT) point.
    # ``terminal_no_decision_likelihood`` is the post-mixture value used
    # as the no-choice trial's per-trial likelihood:
    # ``(1-λ)·terminal_no_decision_mass + λ/(2·tmax)``.
    lapse_rate: float = 0.0
    lapse_baseline_density: float = 0.0
    terminal_no_decision_likelihood: float = 0.0
    # Per-trial diagnostics surfaced in the figure title: the trial's
    # stimulus DV, the likelihood ``L = mle_choice_prob_or_density`` and
    # its log ``logL = mle_loglik``. NaN-defaulted so legacy fixtures
    # without these columns still construct a frame.
    dv: float = float("nan")
    choice_prob_or_density: float = float("nan")
    loglik: float = float("nan")


@dataclass(frozen=True)
class DDMTrialBuffer:
    times: np.ndarray
    upper_density: np.ndarray
    lower_density: np.ndarray
    survival: np.ndarray
    x_grid: np.ndarray
    p_by_t: np.ndarray
    p_at_tmax: np.ndarray
    bound: float
    dt: float
    tmax: float
    terminal_c: float
    terminal_no_decision_mass: float
    terminal_upper_mass: float
    terminal_lower_mass: float
    observed_choice_left: float | None
    observed_rt: float | None
    non_decision_time: float
    trial_index: object
    requested_backend: str
    actual_backend: str
    backend_warning: str | None
    # See DDMFrameData for the semantics of these three fields.
    lapse_rate: float = 0.0
    lapse_baseline_density: float = 0.0
    terminal_no_decision_likelihood: float = 0.0
    # Per-trial diagnostics — see DDMFrameData.
    dv: float = float("nan")
    choice_prob_or_density: float = float("nan")
    loglik: float = float("nan")

    def frame_at_step(self, current_step: int | None = None) -> DDMFrameData:
        n_steps = len(self.times)
        if current_step is None:
            current_step = n_steps - 1
        current_step = int(np.clip(current_step, 0, n_steps - 1))
        survival0 = max(float(self.survival[0]), 1e-12)
        survival_ratio = float(self.survival[current_step]) / survival0
        return DDMFrameData(
            times=self.times,
            upper_density=self.upper_density,
            lower_density=self.lower_density,
            survival=self.survival,
            x_grid=self.x_grid,
            p_at_tmax=self.p_at_tmax,
            current_state_mass=self.p_by_t[current_step],
            bound=self.bound,
            dt=self.dt,
            tmax=self.tmax,
            terminal_c=self.terminal_c,
            current_step=current_step,
            current_time=float(self.times[current_step]),
            current_survival_ratio=survival_ratio,
            terminal_no_decision_mass=self.terminal_no_decision_mass,
            terminal_upper_mass=self.terminal_upper_mass,
            terminal_lower_mass=self.terminal_lower_mass,
            observed_choice_left=self.observed_choice_left,
            observed_rt=self.observed_rt,
            non_decision_time=self.non_decision_time,
            trial_index=self.trial_index,
            requested_backend=self.requested_backend,
            actual_backend=self.actual_backend,
            backend_warning=self.backend_warning,
            lapse_rate=self.lapse_rate,
            lapse_baseline_density=self.lapse_baseline_density,
            terminal_no_decision_likelihood=self.terminal_no_decision_likelihood,
            dv=self.dv,
            choice_prob_or_density=self.choice_prob_or_density,
            loglik=self.loglik,
        )


def parameter_slider_specs(result: dict) -> list[SliderSpec]:
    """Return slider specs from fitted params and saved bounds."""
    names = [str(name).upper() for name in result["params_names"]]
    params = fitted_params_from_result(result)
    bounds = np.asarray(result.get("params_bounds", []), dtype=float)
    specs = []
    for i, name in enumerate(names):
        value = float(params[name])
        if bounds.shape == (len(names), 2):
            low, high = map(float, bounds[i])
        else:
            delta = max(abs(value), 1.0)
            low, high = value - delta, value + delta
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            delta = max(abs(value), 1.0)
            low, high = value - delta, value + delta
        step = abs(high - low) / 200.0
        specs.append(SliderSpec(name=name, value=value, min=low, max=high, step=step))
    return specs


def build_ddm_trial_buffer(
    result: dict,
    *,
    params: dict[str, float] | None = None,
    trial_index=None,
    tmax: float | None = None,
    observed_choice_left: float | None = None,
    observed_rt: float | None = None,
    mle_array_backend: str = "numpy",
    mle_device_id: int | None = None,
    terminal_c: float | None = None,
) -> DDMTrialBuffer:
    """Compute and cache all timestep data for one trial."""
    params = fitted_params_from_result(result) if params is None else {
        str(k).upper(): float(v) for k, v in params.items()}
    model_config = result["model_config"]
    backend = resolve_array_backend(
        mle_array_backend, mle_device_id, cupy_fallback="numpy")
    config_updates = {
        "mle_array_backend": backend.actual_backend,
        "mle_device_id": backend.device_id,
        "mle_cupy_fallback": "numpy",
    }
    if tmax is not None:
        config_updates["t_dur"] = float(tmax)
    if terminal_c is not None:
        config_updates["mle_terminal_c"] = float(terminal_c)
    model_config = replace(model_config, **config_updates)
    eval_res = evaluate_neg_loglik(
        params, result["subject_df"], model_config, return_df=True)
    mle_df = eval_res.mle_df
    assert mle_df is not None
    if trial_index is None:
        valid = mle_df[mle_df["mle_valid_for_loss"]].index
        if len(valid) == 0:
            raise ValueError("No valid MLE trials available for DDM viewer")
        trial_index = valid[0]
    row = mle_df.loc[trial_index]

    bound = float(params.get("BOUND", 1.0))
    non_decision_time = float(params.get("NON_DECISION_TIME", 0.0))
    sigma = float(row["mle_sigma"])
    q_rel = float(row["mle_Q_rel_before"])
    mu = _compute_mu(float(row["DV"]), params, model_config, q_rel, sigma)
    z = float(row["mle_z"])
    fpr = first_passage_density(
        z, mu, sigma, bound,
        model_config.dt, model_config.dx, model_config.t_dur,
        backend="reference",
    )
    if observed_choice_left is None:
        choice = row.get("ChoiceLeft", np.nan)
        observed_choice_left = None if pd.isna(choice) else float(choice)
    if observed_rt is None:
        rt = row.get("calcStimulusTime", np.nan)
        observed_rt = None if pd.isna(rt) else float(rt)

    x_grid = np.asarray(fpr.x_grid)
    p_by_t = np.asarray(fpr.p_by_t)
    p_at_tmax = p_by_t[-1]
    terminal_upper, terminal_lower, terminal_no_decision = _terminal_masses(
        x_grid, p_at_tmax, bound, float(model_config.mle_terminal_c))

    # Lapse / contamination mixture. The lapse parameter is fit by DE for
    # MLE results, so it lives in the fitted params dict (and is also
    # exposed per-trial in mle_df via _build_mle_df). Pre-compute the
    # baseline density and the post-mixture no-choice likelihood once so
    # the plotter doesn't repeat the math each frame.
    tmax_val = float(model_config.t_dur)
    lapse_rate = float(params.get("LAPSE_RATE", 0.0))
    lapse_baseline_density = (
        lapse_rate / (2.0 * tmax_val) if tmax_val > 0.0 else 0.0)
    terminal_no_decision_likelihood = (
        (1.0 - lapse_rate) * terminal_no_decision + lapse_baseline_density)

    # Per-trial diagnostics for the figure title. ``DV`` is the trial's
    # stimulus value, ``mle_choice_prob_or_density`` is the per-trial
    # likelihood L, and ``mle_loglik`` is log(L). ``float(NaN)`` is NaN,
    # so missing columns/values surface cleanly as ``nan`` in the title.
    dv_value = float(row["DV"])
    choice_prob_value = float(row.get("mle_choice_prob_or_density", np.nan))
    loglik_value = float(row.get("mle_loglik", np.nan))

    return DDMTrialBuffer(
        times=np.asarray(fpr.times),
        upper_density=np.asarray(fpr.f_upper),
        lower_density=np.asarray(fpr.f_lower),
        survival=np.asarray(fpr.survival),
        x_grid=x_grid,
        p_by_t=p_by_t,
        p_at_tmax=p_at_tmax,
        bound=bound,
        dt=float(model_config.dt),
        tmax=tmax_val,
        terminal_c=float(model_config.mle_terminal_c),
        terminal_no_decision_mass=terminal_no_decision,
        terminal_upper_mass=terminal_upper,
        terminal_lower_mass=terminal_lower,
        observed_choice_left=observed_choice_left,
        observed_rt=observed_rt,
        non_decision_time=non_decision_time,
        trial_index=trial_index,
        requested_backend=backend.requested_backend,
        actual_backend=backend.actual_backend,
        backend_warning=backend.warning,
        lapse_rate=lapse_rate,
        lapse_baseline_density=lapse_baseline_density,
        terminal_no_decision_likelihood=terminal_no_decision_likelihood,
        dv=dv_value,
        choice_prob_or_density=choice_prob_value,
        loglik=loglik_value,
    )


def build_ddm_frame_data(
    result: dict,
    *,
    params: dict[str, float] | None = None,
    trial_index=None,
    tmax: float | None = None,
    current_step: int | None = None,
    observed_choice_left: float | None = None,
    observed_rt: float | None = None,
    mle_array_backend: str = "numpy",
    mle_device_id: int | None = None,
    terminal_c: float | None = None,
) -> DDMFrameData:
    """Compute one single-trial DDM frame from existing MLE functions."""
    return build_ddm_trial_buffer(
        result,
        params=params,
        trial_index=trial_index,
        tmax=tmax,
        observed_choice_left=observed_choice_left,
        observed_rt=observed_rt,
        mle_array_backend=mle_array_backend,
        mle_device_id=mle_device_id,
        terminal_c=terminal_c,
    ).frame_at_step(current_step)


def plot_ddm_frame(frame: DDMFrameData, ax=None):
    """Plot one DDM frame and return the Matplotlib figure."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8))
    else:
        fig = ax.figure
    ax.clear()
    t = frame.times
    upper = frame.upper_density
    lower = frame.lower_density
    scale = _density_scale(upper, lower, frame.bound)
    internal_scale = _internal_density_scale(
        frame.current_state_mass,
        frame.tmax,
        frame.dt,
        frame.current_survival_ratio,
    )

    ax.axhline(frame.bound, color="black", lw=1.2)
    ax.axhline(-frame.bound, color="black", lw=1.2)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(frame.current_time, color="0.2", alpha=0.35, lw=2)
    ax.fill_between(t, frame.bound, frame.bound + upper * scale,
                    color="C0", alpha=0.45, label="upper density")
    ax.fill_between(t, -frame.bound, -frame.bound - lower * scale,
                    color="C3", alpha=0.45, label="lower density")

    # Lapse baseline overlay: a dashed reference at the mixture term
    # ``λ/(2·T_max)`` on both density panels. Anything below this level
    # is dominated by the lapse contribution rather than the DDM — moving
    # the LAPSE_RATE slider up/down raises/lowers this line. Kept thin
    # and dashed so it doesn't compete with the density fills visually.
    if frame.lapse_rate > 0.0 and frame.lapse_baseline_density > 0.0:
        baseline_height = frame.lapse_baseline_density * scale
        ax.hlines(
            y=frame.bound + baseline_height,
            xmin=0, xmax=frame.tmax,
            color="C0", lw=1.0, ls="--", alpha=0.55,
            label=f"upper lapse floor (λ/(2·T)={frame.lapse_baseline_density:.3g})",
        )
        ax.hlines(
            y=-frame.bound - baseline_height,
            xmin=0, xmax=frame.tmax,
            color="C3", lw=1.0, ls="--", alpha=0.55,
            label="lower lapse floor",
        )
    ax.plot(
        frame.current_time + frame.current_state_mass * internal_scale,
        frame.x_grid,
        color="C2",
        lw=1.8,
        label="state density",
    )
    ax.fill_betweenx(
        frame.x_grid,
        frame.current_time,
        frame.current_time + frame.current_state_mass * internal_scale,
        color="C2",
        alpha=0.22,
    )

    terminal_half_height = max(frame.terminal_c * frame.bound, 0.04 * frame.bound)
    terminal_bar_width = _terminal_no_decision_bar_width(
        frame.terminal_no_decision_mass, frame.tmax, frame.dt)
    ax.vlines(
        frame.tmax,
        -terminal_half_height,
        terminal_half_height,
        color="0.2",
        lw=2.0,
        alpha=0.7,
        label="terminal C window",
    )
    ax.hlines(
        y=0,
        xmin=frame.tmax,
        xmax=frame.tmax + terminal_bar_width,
        color="0.2",
        lw=8,
        alpha=0.65,
        label="terminal no-decision mass",
    )
    # Annotate the terminal no-decision bar with the raw DDM mass and,
    # when the lapse mixture is active, the post-mixture no-choice
    # likelihood `(1-λ)·mass + λ/(2·T_max)`. The post-mixture value is
    # what actually goes into the loss for no-choice trials, so this is
    # the more useful number when comparing against ``mle_loglik``.
    if frame.lapse_rate > 0.0:
        terminal_label = (
            f"mass={frame.terminal_no_decision_mass * 100:.1f}% → "
            f"L={frame.terminal_no_decision_likelihood:.3g}")
    else:
        terminal_label = f"{frame.terminal_no_decision_mass * 100:.1f}%"
    ax.text(
        frame.tmax + terminal_bar_width + frame.dt,
        0,
        terminal_label,
        va="center",
        ha="left",
        fontsize=8,
        color="0.2",
    )
    if frame.observed_rt is not None and frame.observed_choice_left is not None:
        decision_time = frame.observed_rt - frame.non_decision_time
        choice = frame.observed_choice_left
        # Anchor the dashed RT marker at the bound the observed choice
        # actually crossed (upper bound for choice_left=1, lower for =0)
        # and extend outward to the edge of the visible y-range. Matches
        # the ylim below: ``[-bound*1.65, bound*1.65]``.
        if decision_time >= 0 and choice in (0.0, 1.0):
            if choice == 1.0:
                color = "C0"
                y_low, y_high = frame.bound, frame.bound * 1.65
            else:
                color = "C3"
                y_low, y_high = -frame.bound * 1.65, -frame.bound
            ax.vlines(
                decision_time, y_low, y_high,
                color=color, ls="--", lw=1.5,
                label="selected RT")

    terminal_pad = _terminal_no_decision_bar_width(1.0, frame.tmax, frame.dt)
    ax.set_xlim(0, frame.tmax + terminal_pad + frame.dt * 8)
    ax.set_ylim(-frame.bound * 1.65, frame.bound * 1.65)
    ax.set_xlabel("decision time (s)")
    ax.set_ylabel("DDM state / density offset")
    # Title: top line keeps the trial / step / time identifiers and adds
    # the trial's stimulus ``DV``; bottom line surfaces the per-trial
    # likelihood ``loss(L)`` and log-likelihood ``log loss(logL)`` so the
    # user can see — at a glance — what number the MLE objective sees for
    # this trial as they sweep sliders.
    ax.set_title(
        f"trial={frame.trial_index} | DV={frame.dv:.3g} | "
        f"step={frame.current_step} | t={frame.current_time:.4f}s\n"
        f"loss(L)={frame.choice_prob_or_density:.4g} | "
        f"log loss(logL)={frame.loglik:.4g}",
        fontsize=10,
    )
    # Legend lives outside the axes on the right so it never overlaps the
    # density fills or terminal annotation. ``bbox_to_anchor=(1.02, 1)`` in
    # axes-fraction coords puts the legend's upper-left corner just to the
    # right of the axes; ``tight_layout(rect=...)`` reserves the matching
    # strip on the figure so the legend isn't clipped.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=8, borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return fig


def save_ddm_tiff_stack(frames: list[DDMFrameData], path: str | Path) -> Path:
    """Save DDM frames as a multi-page TIFF stack."""
    from PIL import Image
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        fig = plot_ddm_frame(frame)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buffer = buffer.reshape((height, width, 3))
        images.append(Image.fromarray(buffer))
        plt.close(fig)
    if not images:
        raise ValueError("No DDM frames to save")
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


def show_mle_ddm_viewer(
    results: list[MLEModelResult],
    *,
    mle_array_backend: str = "numpy",
    mle_device_id: int | None = None,
):
    """Display the single-trial MLE DDM viewer in a notebook."""
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    if not results:
        raise ValueError("No MLE results loaded")
    # Nest results as ``subject -> model_name -> MLEModelResult`` so the UI
    # can offer a Subject dropdown and a per-subject Model dropdown. This
    # also lets us preserve the SessId / TrialNumber selection when the
    # user swaps between models for the same subject (the underlying
    # subject_df — and therefore the SessId / TrialNumber namespace — is
    # the same across that subject's model fits).
    by_subject: dict[str, dict[str, MLEModelResult]] = {}
    for item in results:
        by_subject.setdefault(item.subject, {})[item.model_name] = item
    subject_dropdown = widgets.Dropdown(
        options=sorted(by_subject), description="Subject")
    model_dropdown = widgets.Dropdown(description="Model")
    # Two-level trial selector: pick a session (SessId), then a trial within
    # that session (TrialNumber). The TrialNumber dropdown stores
    # ``(label, dataframe_index)`` option tuples so its ``.value`` is still
    # the row index needed by ``build_ddm_trial_buffer``.
    sess_id_dropdown = widgets.Dropdown(description="SessId")
    trial_number_dropdown = widgets.Dropdown(description="TrialNumber")
    realtime = widgets.Checkbox(value=False, description="Real-time update")
    run_button = widgets.Button(description="Run", button_style="primary")
    save_button = widgets.Button(description="Save TIFF")
    save_path = widgets.Text(value="ddm_stack.tiff", description="Path")
    choice = widgets.Dropdown(
        options=[("observed", "observed"), ("left/upper", 1.0),
                 ("right/lower", 0.0), ("no choice", "none")],
        description="Choice")
    rt = widgets.FloatText(value=np.nan, description="RT")
    tmax = widgets.FloatText(description="Tmax")
    terminal_c = widgets.FloatSlider(
        description="C",
        min=0.0,
        max=1.0,
        step=0.01,
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="240px"),
    )
    time_slider = widgets.IntSlider(description="step", min=0, max=1, value=1)
    slider_box = widgets.VBox()
    status = widgets.HTML()
    progress = widgets.HTML(value="Idle")
    fig, ax = plt.subplots(figsize=(9, 4.8))

    state = SimpleNamespace(
        sliders={}, trial_buffer=None, buffer_key=None, valid_df=None)

    def selected_item():
        subject = subject_dropdown.value
        model_name = model_dropdown.value
        return by_subject[subject][model_name]

    def _current_trial_label():
        """Return the label of the currently-displayed TrialNumber option."""
        value = trial_number_dropdown.value
        if value is None:
            return None
        for label, idx in trial_number_dropdown.options:
            if idx == value:
                return label
        return None

    def set_trial_number_options(*_args):
        """Repopulate TrialNumber dropdown for the currently selected SessId.

        When called from a model swap within the same subject, the previous
        TrialNumber label is preserved if it still exists in the new
        model's valid trials — otherwise the first available trial is
        selected. This keeps the user pinned to the same trial when they
        only want to compare model fits.
        """
        valid_df = state.valid_df
        sess_id = sess_id_dropdown.value
        if valid_df is None or sess_id is None or "SessId" not in valid_df.columns:
            trial_number_dropdown.options = []
            return
        prev_label = _current_trial_label()
        rows = valid_df[valid_df["SessId"] == sess_id]
        if "TrialNumber" in rows.columns:
            rows = rows.sort_values("TrialNumber")
            options = []
            for idx, trial_no in zip(rows.index, rows["TrialNumber"]):
                label = (
                    str(int(trial_no)) if pd.notna(trial_no) and float(trial_no) == int(trial_no)
                    else (f"{float(trial_no):g}" if pd.notna(trial_no) else str(idx)))
                options.append((label, idx))
        else:
            options = [(str(idx), idx) for idx in rows.index]
        trial_number_dropdown.options = options
        if options:
            target_idx = None
            if prev_label is not None:
                for label, idx in options:
                    if label == prev_label:
                        target_idx = idx
                        break
            trial_number_dropdown.value = (
                target_idx if target_idx is not None else options[0][1])

    def set_subject_models(*_args):
        """Repopulate the Model dropdown for the currently selected Subject.

        Preserves the previously-chosen model_name when it also exists for
        the new subject. Always ensures ``set_model_controls`` runs once at
        the end so the figure refreshes — even if the model name is
        preserved (and therefore no ``model_dropdown.value`` change fires
        the model observer).
        """
        subject = subject_dropdown.value
        if subject is None or subject not in by_subject:
            model_dropdown.options = []
            return
        models = sorted(by_subject[subject])
        prev_model = model_dropdown.value
        model_dropdown.options = models
        if not models:
            return
        # Always explicitly assign .value: when the dropdown was empty
        # (.value=None) some ipywidgets versions don't auto-promote to the
        # first new option, which would leave .value=None and break
        # selected_item(). When the new value matches the prior one this
        # is a no-op; otherwise the model observer fires (idempotently
        # before the explicit set_model_controls() below).
        target_model = prev_model if prev_model in models else models[0]
        model_dropdown.value = target_model
        set_model_controls()

    def set_model_controls(*_args):
        item = selected_item()
        cfg = item.result["model_config"]
        tmax.value = float(cfg.t_dur)
        terminal_c.value = float(cfg.mle_terminal_c)
        df = item.result["mle_df"]
        if df is not None:
            valid_df = df[df["mle_valid_for_loss"]]
        else:
            valid_df = None
        state.valid_df = valid_df
        if valid_df is not None and len(valid_df) and "SessId" in valid_df.columns:
            sess_ids = sorted(valid_df["SessId"].dropna().unique().tolist())
        else:
            sess_ids = []
        prev_sess_id = sess_id_dropdown.value
        sess_id_dropdown.options = sess_ids
        if sess_ids:
            # Preserve the previous SessId when the new model fits the same
            # subject (the common case for a model swap); otherwise fall
            # back to the first session.
            target_sess = prev_sess_id if prev_sess_id in sess_ids else sess_ids[0]
            sess_id_dropdown.value = target_sess
            # Setting .value to the SAME thing doesn't fire the observer,
            # so call the cascade explicitly to ensure the TrialNumber
            # options reflect the new model's valid trials.
            set_trial_number_options()
        else:
            trial_number_dropdown.options = []
        state.sliders = {}
        children = []
        for spec in parameter_slider_specs(item.result):
            slider = widgets.FloatSlider(
                description=spec.name,
                value=spec.value,
                min=spec.min,
                max=spec.max,
                step=spec.step,
                continuous_update=False,
                readout_format=".4g",
                layout=widgets.Layout(width="420px"),
            )
            state.sliders[spec.name] = slider
            children.append(slider)
            slider.observe(lambda change: maybe_update(), names="value")
        slider_box.children = children
        maybe_update(force=True)

    def params_from_sliders():
        return {name: slider.value for name, slider in state.sliders.items()}

    def buffer_key(selected_choice_key, selected_rt):
        params = params_from_sliders()
        params_key = tuple(
            (name, float(params[name]))
            for name in sorted(params)
        )
        rt_key = None if selected_rt is None else float(selected_rt)
        return (
            model_dropdown.value,
            sess_id_dropdown.value,
            trial_number_dropdown.value,
            params_key,
            float(tmax.value),
            float(terminal_c.value),
            selected_choice_key,
            rt_key,
            str(mle_array_backend),
            None if mle_device_id is None else int(mle_device_id),
        )

    def trial_buffer_for_current_controls():
        item = selected_item()
        selected_choice = choice.value
        if selected_choice == "observed":
            selected_choice_key = "observed"
            selected_choice = None
        elif selected_choice == "none":
            selected_choice_key = "none"
            selected_choice = np.nan
        else:
            selected_choice_key = float(selected_choice)
        selected_rt = None if np.isnan(rt.value) else float(rt.value)
        key = buffer_key(selected_choice_key, selected_rt)
        if state.buffer_key == key and state.trial_buffer is not None:
            return state.trial_buffer
        progress.value = "Computing trial..."
        state.trial_buffer = build_ddm_trial_buffer(
            item.result,
            params=params_from_sliders(),
            trial_index=trial_number_dropdown.value,
            tmax=float(tmax.value),
            observed_choice_left=selected_choice,
            observed_rt=selected_rt,
            mle_array_backend=mle_array_backend,
            mle_device_id=mle_device_id,
            terminal_c=float(terminal_c.value),
        )
        state.buffer_key = key
        return state.trial_buffer

    def frame_for_current_step(step=None):
        buffer = trial_buffer_for_current_controls()
        return buffer.frame_at_step(time_slider.value if step is None else step)

    def redraw(step=None):
        progress.value = "Rendering..."
        try:
            frame = frame_for_current_step(step)
            time_slider.max = max(len(frame.times) - 1, 0)
            if time_slider.value > time_slider.max:
                time_slider.value = time_slider.max
            plot_ddm_frame(frame, ax=ax)
            fig.canvas.draw_idle()
            warning = f"; {frame.backend_warning}" if frame.backend_warning else ""
            status.value = (
                f"t={frame.current_time:.4f}s, trial={frame.trial_index}, "
                f"C={frame.terminal_c:.2f}, λ={frame.lapse_rate:.4g}, "
                f"backend={frame.actual_backend}{warning}")
        finally:
            progress.value = "Idle"

    def maybe_update(*_args, force=False):
        if force or realtime.value:
            redraw()

    def on_run(_button):
        redraw()

    def on_save(_button):
        progress.value = "Saving TIFF..."
        try:
            buffer = trial_buffer_for_current_controls()
            frames = [
                buffer.frame_at_step(i)
                for i in range(len(buffer.times))
            ]
            saved = save_ddm_tiff_stack(frames, save_path.value)
            status.value = f"saved {saved}"
        finally:
            progress.value = "Idle"

    subject_dropdown.observe(set_subject_models, names="value")
    model_dropdown.observe(set_model_controls, names="value")
    # SessId only cascades into TrialNumber options; the resulting
    # trial_number_dropdown value change triggers maybe_update by itself,
    # so we don't observe sess_id for maybe_update directly.
    sess_id_dropdown.observe(set_trial_number_options, names="value")
    time_slider.observe(lambda change: redraw(), names="value")
    for widget in (trial_number_dropdown, tmax, terminal_c, choice, rt):
        widget.observe(lambda change: maybe_update(), names="value")
    run_button.on_click(on_run)
    save_button.on_click(on_save)
    set_subject_models()
    display(widgets.VBox([
        widgets.HBox([
            subject_dropdown, model_dropdown,
            sess_id_dropdown, trial_number_dropdown,
            realtime, run_button,
        ]),
        slider_box,
        widgets.HBox([choice, rt, terminal_c, tmax]),
        time_slider,
        widgets.HBox([save_path, save_button]),
        widgets.HBox([widgets.HTML("Update status:"), progress, status]),
    ]))
    return {"figure": fig, "controls": state}


def _terminal_masses(x_grid, p_at_tmax, bound, terminal_c):
    threshold = float(terminal_c) * float(bound)
    upper = p_at_tmax[x_grid > threshold].sum()
    lower = p_at_tmax[x_grid < -threshold].sum()
    no_decision = p_at_tmax[(x_grid <= threshold) & (x_grid >= -threshold)].sum()
    return float(upper), float(lower), float(no_decision)


def _density_scale(upper, lower, bound):
    max_density = max(float(np.nanmax(upper, initial=0.0)),
                      float(np.nanmax(lower, initial=0.0)),
                      1e-12)
    return 0.45 * float(bound) / max_density


def _internal_density_scale(state_mass, tmax, dt, survival_ratio):
    """Scale current state mass relative to the first step's surviving mass."""
    max_mass = max(float(np.nanmax(state_mass, initial=0.0)), 1e-12)
    visible_width = max(float(tmax) * 0.12, float(dt) * 5.0)
    return visible_width * max(float(survival_ratio), 0.0) / max_mass


def _terminal_no_decision_bar_width(no_decision_mass, tmax, dt):
    max_width = max(float(tmax) * 0.16, float(dt) * 8.0)
    mass = float(np.clip(no_decision_mass, 0.0, 1.0))
    return max_width * mass
