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
from .data import MLEModelResult, fitted_params_from_result, result_key


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
        tmax=float(model_config.t_dur),
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
    ax.text(
        frame.tmax + terminal_bar_width + frame.dt,
        0,
        f"{frame.terminal_no_decision_mass * 100:.1f}%",
        va="center",
        ha="left",
        fontsize=8,
        color="0.2",
    )
    if frame.observed_rt is not None:
        decision_time = frame.observed_rt - frame.non_decision_time
        if decision_time >= 0:
            color = "C0" if frame.observed_choice_left == 1 else (
                "C3" if frame.observed_choice_left == 0 else "0.25")
            ax.axvline(decision_time, color=color, ls="--", lw=1.5,
                       label="selected RT")

    terminal_pad = _terminal_no_decision_bar_width(1.0, frame.tmax, frame.dt)
    ax.set_xlim(0, frame.tmax + terminal_pad + frame.dt * 8)
    ax.set_ylim(-frame.bound * 1.65, frame.bound * 1.65)
    ax.set_xlabel("decision time (s)")
    ax.set_ylabel("DDM state / density offset")
    ax.set_title(
        f"trial={frame.trial_index} | step={frame.current_step} | "
        f"t={frame.current_time:.4f}s")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
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
    by_key = {result_key(item): item for item in results}
    model_dropdown = widgets.Dropdown(options=sorted(by_key), description="Model")
    trial_dropdown = widgets.Dropdown(description="Trial")
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
        max=0.999,
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

    state = SimpleNamespace(sliders={}, trial_buffer=None, buffer_key=None)

    def selected_item():
        return by_key[model_dropdown.value]

    def set_model_controls(*_args):
        item = selected_item()
        cfg = item.result["model_config"]
        tmax.value = float(cfg.t_dur)
        terminal_c.value = float(cfg.mle_terminal_c)
        df = item.result["mle_df"]
        valid_idx = list(df.index[df["mle_valid_for_loss"]]) if df is not None else []
        trial_dropdown.options = valid_idx
        if valid_idx:
            trial_dropdown.value = valid_idx[0]
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
            trial_dropdown.value,
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
            trial_index=trial_dropdown.value,
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
                f"C={frame.terminal_c:.2f}, backend={frame.actual_backend}{warning}")
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

    model_dropdown.observe(set_model_controls, names="value")
    time_slider.observe(lambda change: redraw(), names="value")
    for widget in (trial_dropdown, tmax, terminal_c, choice, rt):
        widget.observe(lambda change: maybe_update(), names="value")
    run_button.on_click(on_run)
    save_button.on_click(on_save)
    set_model_controls()
    display(widgets.VBox([
        widgets.HBox([model_dropdown, trial_dropdown, tmax, realtime, run_button]),
        slider_box,
        widgets.HBox([choice, rt, terminal_c]),
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
