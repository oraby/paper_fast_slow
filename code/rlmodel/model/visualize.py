from .logic import makeOneRun
from .drift import DRIFT_FN_DICT
from .noise import NOISE_FN_DICT
from .bias import BIAS_FN_DICT
from .plotter import runAndPlot, createFig
from .util import (initDF, assignQuantiles, driftFnColsAndKwargs,
                   biasFnColsAndKwargs, noiseFnColsAndKwargs, PsychometricPlot)
from .initvals import InitVals, MLE_TERMINAL_C
from .mle import MLEModelConfig, evaluate_neg_loglik
# from ._readparams import readParams
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
from inspect import signature
import pathlib
import os


# Sliders that are NOT a kwarg of any drift/bias/noise/logic function but
# DO feed the MLE objective. The widget loop's auto-disable check
# (`widget.disabled = widget.description not in all_kwargs_names`) would
# otherwise hide them, so we exempt them and let the user always edit them.
# The value is consumed by ``_mle_params_from_widgets`` only when "Run MLE"
# is pressed, so the chisq simulator path remains unaffected.
_MLE_ONLY_SLIDER_NAMES = {"LAPSE_RATE", "MLE_TERMINAL_C"}


def createWidget(init_vals : InitVals, gui_cache : InitVals, df, t_dur, dt,
                 is_small_fig_mode, subjects_defaults=None, save_figs=False,
                 save_ovewrite=True):

    drop_downs_labels = ["Subject", "DV", "Drift Fn", "Bias Fn",
                         "Psychometric", "Noise Fn"]

    checkboxes_labels = {"Real-time": gui_cache.get("Real-time", True)}
    chi2_reset_label = "Reset to Chi\u00b2"
    buttons_labels = ["Reset to MLE defaults", chi2_reset_label, "Update", "Run MLE"]

    slider_widgets = {}
    for label, (min_val, max_val, default_val) in init_vals.items():
        val = gui_cache.get(label, default_val)
        val = min(max(val, min_val), max_val)
        slider_widgets[label] = widgets.FloatSlider(min=min_val,
                                                    max=max_val,
                                                    value=val,
                                                    description=label,
                                                    continuous_update=False,
                                                    step=(max_val-min_val)/100)

    # MLE_TERMINAL_C is a config knob (not a DE-fittable parameter) so it
    # lives outside the InitVals dataclass and the auto-creation loop above
    # doesn't see it. Add it manually here using the centralized
    # min/max/default from initvals.MLE_TERMINAL_C.
    _mtc_val = gui_cache.get("MLE_TERMINAL_C", MLE_TERMINAL_C.Default)
    _mtc_val = min(max(_mtc_val, MLE_TERMINAL_C.Min), MLE_TERMINAL_C.Max)
    slider_widgets["MLE_TERMINAL_C"] = widgets.FloatSlider(
        min=MLE_TERMINAL_C.Min,
        max=MLE_TERMINAL_C.Max,
        value=_mtc_val,
        description="MLE_TERMINAL_C",
        continuous_update=False,
        step=(MLE_TERMINAL_C.Max - MLE_TERMINAL_C.Min) / 100)

    drop_down_widgets = {}
    for label in drop_downs_labels:
        equals_fn = lambda x, y: x == y
        default_val_idx = 0
        if label == "DV":
            dv_values = sorted(list(df["DV"].unique()))
            dv_values_abs = sorted(list(np.unique(np.abs(dv_values))))
            # Add twice, once absolute and once pos/neg
            options_str = ["All"]
            options_str += [f"±{v:.2f}" for v in dv_values_abs]
            options_str += [f"{'+' if v > 0 else '-'}{v:.2f}" for v in dv_values]
            # Each entry is a list of values
            values = [list(dv_values)] + [[dv, -dv] for dv in dv_values_abs] + [[dv] for dv in dv_values]
            # equals_fn = lambda x, y: all(x == y)
        elif label == "Subject":
            options_str = sorted(df["Name"].unique())
            values = options_str
        elif label == "Drift Fn":
            options_str = list(DRIFT_FN_DICT.keys())
            values =      list(DRIFT_FN_DICT.values())
            default_val_idx = options_str.index("NoiseGain-RewardRate")
            # print("options_str:", options_str)
            # print("values:", values)
        elif label == "Bias Fn":
            options_str = list(BIAS_FN_DICT.keys())
            values =      list(BIAS_FN_DICT.values())
            default_val_idx = options_str.index("Q-Val (Offset)")
        elif label == "Psychometric":
            options_str = ["None", "All", "Slow/Fast"]
            values = [PsychometricPlot._None, PsychometricPlot.All, PsychometricPlot.SlowFast]
            default_val_idx = 2
        elif label == "Noise Fn":
            options_str = list(NOISE_FN_DICT.keys())
            values =      list(NOISE_FN_DICT.values())
        else:
            raise ValueError(f"Unknown label: {label}")

        options = zip(options_str, values)
        try:
            from inspect import isfunction
            if isfunction(values[0]):
                cached_value = gui_cache.get(label, values[default_val_idx])
                print("cached_value:", cached_value)
                val_idx = [v.__name__ for v in values].index(cached_value.__name__)
            else:
                cached_value = gui_cache.get(label, default_val_idx)
                val_idx = [str(v) for v in values].index(str(cached_value))
        except ValueError:
            print(f"Couldn't find {cached_value} in {values}")
            val_idx = default_val_idx
        # print(f"Creating {label} with options: {options_str}")
        # print(f"Values: {values}")
        drop_down_widgets[label]  = widgets.Dropdown(options=options,
                                                     value=values[val_idx],
                                                     description=label,
                                                     equals=equals_fn)

    # TODO: Add callbacks to the checkboxes
    checkbox_widgets = {label:widgets.Checkbox(value=val, description=label)
                        for label, val in checkboxes_labels.items()}
    button_widgets_li = [widgets.Button(description=label) if label != "Update" else widgets.ToggleButton(description=label,
                                                                                                          value=True)
                         for label in buttons_labels]

    html_math_widgets = {label:widgets.HTMLMath(value=label)
                         for label in ["DriftEq", "BiasEq",]}

    # Make a copy before popping
    all_widgets = {**slider_widgets, **drop_down_widgets, **checkbox_widgets, **html_math_widgets,
                   **{label:widget for label, widget in zip(buttons_labels, button_widgets_li)}}
    # print("All widgets:", all_widgets)
    # Make three columns: parameters, conditions, and buttons/settings
    first_col = [drop_down_widgets.pop("Drift Fn"),
                 slider_widgets.pop("DRIFT_COEF"),
                 slider_widgets.pop("NOISE_SIGMA"),
                 slider_widgets.pop("BOUND"),
                 slider_widgets.pop("NON_DECISION_TIME"),
                 slider_widgets.pop("BETA"),
                 #slider_widgets.pop("Drift RR Coef"),
                 ]
    second_col = [drop_down_widgets.pop("Bias Fn"),
                  slider_widgets.pop("BIAS_COEF"),
                  slider_widgets.pop("BIAS_FIXED"),
                  slider_widgets.pop("BIAS_MU"),
                  slider_widgets.pop("BIAS_SIGMA"),
                  slider_widgets.pop("ALPHA"),]
    third_col = [drop_down_widgets.pop("Subject"),
                 drop_down_widgets.pop("DV"),
                 drop_down_widgets.pop("Psychometric"),
                 drop_down_widgets.pop("Noise Fn"),
                 slider_widgets.pop("Q_VAL_DECAY_RATE"),
                 slider_widgets.pop("Q_VAL_COEF"),
                 slider_widgets.pop("Q_VAL_OFFSET"),]
    # LAPSE_RATE and MLE_TERMINAL_C are MLE-only knobs (contamination /
    # lapse mixture and terminal-time no-decision-band threshold). The
    # chisq simulation path doesn't read either; their values only enter
    # the MLE path when "Run MLE" is executed. Placed next to the MLE
    # buttons so their scope is visually obvious.
    fourth_col = ([checkbox_widgets.pop("Real-time")] + button_widgets_li +
                  [slider_widgets.pop("LAPSE_RATE"),
                   slider_widgets.pop("MLE_TERMINAL_C"),
                   html_math_widgets.pop("DriftEq"),
                   html_math_widgets.pop("BiasEq")])
    # Make sure we didn't leave any widgets
    assert not len(drop_down_widgets), f"Left drop downs: {drop_down_widgets}"
    assert not len(slider_widgets), f"Left sliders: {slider_widgets}"
    assert not len(checkbox_widgets), f"Left checkboxes: {checkbox_widgets}"
    assert not len(html_math_widgets), f"Left html math: {html_math_widgets}"
    layout = widgets.HBox([widgets.VBox(first_col),
                           widgets.VBox(second_col),
                           widgets.VBox(third_col),
                           widgets.VBox(fourth_col)])


    include_Q, include_RewardRate = False, False
    checkbox_last_val = all_widgets["Real-time"].value
    fig = None
    last_subject = None
    last_driftFn = None
    last_biasFn = None
    last_noiseFn = None
    last_loss = None
    last_mle_loss = None
    last_mle_loss_source = None
    last_mle_loss_key = None
    def updateGUI(force_update=False, run_mle=False):
        nonlocal all_widgets, include_Q, include_RewardRate, checkbox_last_val, fig, last_loss
        nonlocal last_subject, last_driftFn, last_biasFn, last_noiseFn
        nonlocal last_mle_loss, last_mle_loss_source, last_mle_loss_key
        # print("Updating GUI:", "Update:", all_widgets["Real-time"].value)

        cur_subject = all_widgets["Subject"].value
        df = all_df[all_df.Name == cur_subject]
        # print("0")
        if ((last_subject != cur_subject) or (last_driftFn != all_widgets["Drift Fn"].value) or
            (last_biasFn != all_widgets["Bias Fn"].value) or
            (last_noiseFn != all_widgets["Noise Fn"].value)) and subjects_defaults is not None:
            _try_apply_fit_defaults(
                all_widgets, subjects_defaults, t_dur, cur_subject,
                preferred_modes=("mle", "chisq"), required=False)

        last_subject = cur_subject
        last_driftFn = all_widgets["Drift Fn"].value
        last_biasFn = all_widgets["Bias Fn"].value
        last_noiseFn = all_widgets["Noise Fn"].value
        # We can't do the DV filtering here, because we need all subsequent
        # trials to build Q values abd RewardRate. So we will rather do
        # the filtering on the results

        # Continue with the update
        # If Drift Fn is "Classic" then disable RewardRate's BETA
        biasFn = all_widgets["Bias Fn"].value
        driftFm = all_widgets["Drift Fn"].value
        noiseFn = all_widgets["Noise Fn"].value
        print(f"Selected Drift Fn: {driftFm.__name__}, Bias Fn: {biasFn.__name__}, Noise Fn: {noiseFn.__name__}")
        # Get string associated with each value
        biasFn_str = [k for k, v in BIAS_FN_DICT.items() if v == biasFn][0]
        driftFn_str = [k for k, v in DRIFT_FN_DICT.items() if v == driftFm][0]
        noiseFn_str = [k for k, v in NOISE_FN_DICT.items() if v == noiseFn][0]
        if "RepeatIdx" in df.columns:
            df = df[df.RepeatIdx == 1]
        if "biasFn" in df.columns:
            df = df[df.biasFn == biasFn_str]
        if "driftFn" in df.columns:
            df = df[df.driftFn == driftFn_str]
        if "noiseFn" in df.columns:
            df = df[df.noiseFn == noiseFn_str]

        biasFn_df_cols, biasFn_kwargs_li = biasFnColsAndKwargs(biasFn)
        driftFn_df_cols, driftFn_kwargs_li = driftFnColsAndKwargs(driftFm)
        noiseFn_df_cols, noiseFn_kwargs_li = noiseFnColsAndKwargs(noiseFn)
        logic_kwargs = signature(makeOneRun).parameters


        all_kwargs_names = set(biasFn_kwargs_li) | set(driftFn_kwargs_li) | \
                           set(noiseFn_kwargs_li) | set(logic_kwargs)
        # print("All kwargs names:", all_kwargs_names)
        for widget in all_widgets.values():
            if isinstance(widget, (widgets.FloatSlider, widgets.Dropdown,
                                   widgets.Checkbox)):
                gui_cache[widget.description] = widget.value
            # Skip non-slider widgets
            if not isinstance(widget, widgets.FloatSlider):
                continue
            # MLE-only sliders never appear in any drift/bias/noise/logic
            # function signature, so the all_kwargs_names check would
            # disable them. Keep them user-editable; they're consumed by
            # the MLE objective path, not by the chisq simulator.
            if widget.description in _MLE_ONLY_SLIDER_NAMES:
                widget.disabled = False
                continue
            widget.disabled = widget.description not in all_kwargs_names

        include_Q = "Q_val" in biasFn_df_cols or "Q_val" in driftFn_df_cols or "Q_val" in noiseFn_df_cols
        include_RewardRate = "RewardRate" in driftFn_df_cols or "RewardRate" in noiseFn_df_cols or "RewardRate" in biasFn_df_cols

        # The logic function always receive an ALPHA and BETA even if not used
        if not include_Q:
            all_widgets["ALPHA"].disabled = True
        if not include_RewardRate:
            all_widgets["BETA"].disabled = True

        # Now we should have update the GUI, but dont continue unless the
        # real-time checkbox is checked or the update button is pressed
        is_realtime = all_widgets["Real-time"].value
        checkbox_changed = is_realtime != checkbox_last_val
        checkbox_last_val = is_realtime
        update_toggl_btn = all_widgets["Update"]
        if is_realtime and checkbox_changed and update_toggl_btn.button_style != 'warning':
            is_realtime = False # No need to update the button
        if not update_toggl_btn.value and not is_realtime and not force_update:
            # Update the "Update" button color to denote that it needs to be clicked
            if not checkbox_changed:
                update_toggl_btn.button_style = 'warning'
            return
        update_toggl_btn.button_style = "" #'success'
        update_toggl_btn.disabled = True

        dvs_filter = all_widgets["DV"].value
        psych_plot = all_widgets["Psychometric"].value


        updatePlots_kwargs_li = signature(runAndPlot).parameters
        biasFn_kwargs, driftFn_kwargs, noiseFn_kwargs = {}, {}, {}
        updatePlots_kwargs = {}
        for widget in all_widgets.values():
            if not isinstance(widget, (widgets.FloatSlider, widgets.Dropdown)) \
               or widget.disabled and widget.description not in ["ALPHA", "BETA"]:
                continue
            if widget.description in biasFn_kwargs_li:
                biasFn_kwargs[widget.description] = widget.value
            if widget.description in driftFn_kwargs_li:
                driftFn_kwargs[widget.description] = widget.value
            if widget.description in noiseFn_kwargs_li:
                noiseFn_kwargs[widget.description] = widget.value
            if widget.description in updatePlots_kwargs_li:
                updatePlots_kwargs[widget.description] = widget.value

        plot_bias_dir = "CorrIncorr" in biasFn.__name__
        mle_loss_key = _mle_loss_key(
            subject=cur_subject,
            driftFn_str=driftFn_str,
            biasFn_str=biasFn_str,
            noiseFn_str=noiseFn_str,
            t_dur=t_dur,
            dt=dt,
            params=_mle_params_from_widgets(all_widgets),
        )
        fit_entry = _get_subject_fit_entry(
            subjects_defaults, t_dur, noiseFn.__name__, biasFn.__name__,
            driftFm.__name__, cur_subject)
        stored_mle_loss = _stored_mle_loss_from_fit_entry(fit_entry)
        if stored_mle_loss is None:
            stored_mle_loss = _stored_mle_loss_from_df(df)
        if run_mle:
            run_mle_btn = all_widgets["Run MLE"]
            old_desc = run_mle_btn.description
            run_mle_btn.disabled = True
            run_mle_btn.description = "Running MLE..."
            try:
                last_mle_loss = _evaluate_mle_loss_for_gui(
                    df=df,
                    params=_mle_params_from_widgets(all_widgets),
                    driftFn_str=driftFn_str,
                    biasFn_str=biasFn_str,
                    noiseFn_str=noiseFn_str,
                    include_Q=include_Q,
                    include_RewardRate=include_RewardRate,
                    dt=dt,
                    t_dur=t_dur,
                    fit_entry=fit_entry,
                    terminal_c_override=float(
                        all_widgets["MLE_TERMINAL_C"].value),
                )
                last_mle_loss_source = "current"
            except Exception as exc:
                print(f"MLE loss failed: {type(exc).__name__}: {exc}")
                last_mle_loss = np.nan
                last_mle_loss_source = "error"
            finally:
                last_mle_loss_key = mle_loss_key
                run_mle_btn.disabled = False
                run_mle_btn.description = old_desc
        elif last_mle_loss_key == mle_loss_key:
            pass
        else:
            last_mle_loss = stored_mle_loss
            last_mle_loss_source = (
                "fit" if _is_finite_number(stored_mle_loss) else None)
            last_mle_loss_key = None

        last_loss, _ = runAndPlot(df, biasFn=biasFn, driftFn=driftFm, noiseFn=noiseFn,
                               dvs_filter=dvs_filter, fig=fig, axs=axs,
                               plot_bias_dir=plot_bias_dir, psych_plot=psych_plot,
                               t_dur=t_dur, dt=dt, include_Q=include_Q,
                               include_RewardRate=include_RewardRate,
                               biasFn_kwargs=biasFn_kwargs, biasFn_df_cols=biasFn_df_cols,
                               driftFn_kwargs=driftFn_kwargs, driftFn_df_cols=driftFn_df_cols,
                               noiseFn_kwargs=noiseFn_kwargs, noiseFn_df_cols=noiseFn_df_cols,
                               is_small_fig_mode=is_small_fig_mode,
                               mle_loss=last_mle_loss,
                               mle_loss_source=last_mle_loss_source,
                               **updatePlots_kwargs)
        # Copied from pyddm.plot.model_gui_jupyter
        # Set the "update" button back to False, but don't trigger a redraw
        changes_tmp = update_toggl_btn._trait_notifiers['value']['change']
        update_toggl_btn._trait_notifiers['value']['change'] = []
        update_toggl_btn.disabled = False
        update_toggl_btn.value = False
        update_toggl_btn._trait_notifiers['value']['change'] = changes_tmp
        # plt.show()
        fig.canvas.draw_idle()


    fig, axs = createFig(is_small_fig_mode=is_small_fig_mode)
    # def figOnclick(event):
    #     if "_handleFigClickEvent" in globals():
    #         _handleFigClickEvent(event)
    # fig.canvas.mpl_connect('button_press_event', figOnclick)

    # all_df = initDF(df, include_Q=True, include_RewardRate=True)
    all_df = df
    all_df["GUI_TimeOutIncorrectChoice"] = 0
    # updateGUI(all_widgets["Update"])


    # Run the display
    all_widgets_wo_btns = {k:v for k,v in all_widgets.items()
                           if not isinstance(v, widgets.Button)}
    def outHandler(*args, **kwargs):
        # print("Args:", args)
        # print("Kwargs:", kwargs)
        updateGUI()
        # return updateGUI()

    all_widgets["Reset to MLE defaults"].on_click(
        lambda _button: _reset_defaults_and_update(
            all_widgets, subjects_defaults, t_dur, "mle", updateGUI))
    all_widgets[chi2_reset_label].on_click(
        lambda _button: _reset_defaults_and_update(
            all_widgets, subjects_defaults, t_dur, "chisq", updateGUI))
    all_widgets["Run MLE"].on_click(
        lambda _button: updateGUI(force_update=True, run_mle=True))
    out = widgets.interactive_output(outHandler, all_widgets_wo_btns)

    display_widget = display(layout, out)
    if save_figs and subjects_defaults is not None:
        real_time_cur_val = all_widgets["Real-time"].value
        all_widgets["Real-time"].value = False
        _NOISE_FN_NAMES = {v.__name__:v for v in NOISE_FN_DICT.values()}
        _BIAS_FN_NAMES = {v.__name__:v for v in BIAS_FN_DICT.values()}
        _DRIFT_FN_NAMES = {v.__name__:v for v in DRIFT_FN_DICT.values()}
        # [t_dur, noiseFn, biasFn, driftFm, subject]
        tmp_t_dur = t_dur
        for cur_t_dur, t_dur_dict in subjects_defaults.items():
            t_dur = cur_t_dur
            for noiseFn, noiseFn_dict in t_dur_dict.items():
                all_widgets["Noise Fn"].value = _NOISE_FN_NAMES[noiseFn]
                for biasFn, biasFn_dict in noiseFn_dict.items():
                    all_widgets["Bias Fn"].value = _BIAS_FN_NAMES[biasFn]
                    for driftFn, driftFn_dict in biasFn_dict.items():
                        all_widgets["Drift Fn"].value = _DRIFT_FN_NAMES[driftFn]
                        # Now switch to real time to update the plots
                        all_widgets["Real-time"].value = True
                        fn = f"{noiseFn}{biasFn}{driftFn}_maxdur_{t_dur}s"
                        for name in driftFn_dict:
                            save_fp = pathlib.Path(f"figs/{name}/{fn}")
                            assert save_fp.parent.parent.exists(), (
                                "Save folder doesn't exist: "
                                f"{save_fp.parent.parent}")
                            # CHeck if file exists before overwriting:
                            if not save_ovewrite and os.path.exists(f"{save_fp}.png"):
                                print("Skipping already existing:", save_fp)
                                continue
                            all_widgets["Subject"].value = name
                            _makeSaveFigTitle(fig, name, last_loss, driftFn,
                                              biasFn)
                            print("Saving:", save_fp)
                            save_fp.parent.mkdir(exist_ok=True)
                            # fig.savefig(f"{save_fp}.png", bbox_inches="tight")
                            small_fig_str = "_small" if is_small_fig_mode else ""
                            fig.savefig(f"{save_fp}_{small_fig_str}.svg", bbox_inches="tight")
                        # Disable agaub so we can update other values
                        all_widgets["Real-time"].value = False
        t_dur = tmp_t_dur
        all_widgets["Real-time"].value = real_time_cur_val
    return display_widget


def _mle_params_from_widgets(all_widgets):
    param_names = {
        "DRIFT_COEF",
        "NOISE_SIGMA",
        "BOUND",
        "NON_DECISION_TIME",
        "BETA",
        "BIAS_COEF",
        "BIAS_FIXED",
        "BIAS_MU",
        "BIAS_SIGMA",
        "ALPHA",
        "Q_VAL_DECAY_RATE",
        "Q_VAL_COEF",
        "Q_VAL_OFFSET",
        # MLE-only contamination / lapse mixture. Read here so "Run MLE"
        # propagates the slider value into ``evaluate_neg_loglik``; not
        # touched by the chisq simulation path.
        "LAPSE_RATE",
    }
    return {
        name: float(widget.value)
        for name, widget in all_widgets.items()
        if name in param_names and hasattr(widget, "value")
    }


def _reset_defaults_and_update(all_widgets, subjects_defaults, t_dur, mode,
                               update_fn):
    subject = all_widgets["Subject"].value
    _apply_fit_defaults(
        all_widgets, subjects_defaults, t_dur, subject, mode=mode)
    update_fn(force_update=True)


def _try_apply_fit_defaults(all_widgets, subjects_defaults, t_dur, subject,
                            preferred_modes, required=False):
    for mode in preferred_modes:
        if _apply_fit_defaults(
                all_widgets, subjects_defaults, t_dur, subject, mode=mode,
                required=False):
            return mode
    if required:
        raise KeyError(
            f"No defaults found for subject={subject!r}, "
            f"modes={tuple(preferred_modes)!r}")
    return None


def _apply_fit_defaults(all_widgets, subjects_defaults, t_dur, subject, mode,
                        required=True):
    entry = _get_subject_fit_entry(
        subjects_defaults,
        t_dur,
        all_widgets["Noise Fn"].value.__name__,
        all_widgets["Bias Fn"].value.__name__,
        all_widgets["Drift Fn"].value.__name__,
        subject,
    )
    fit_entry = _fit_entry_for_mode(entry, mode)
    if fit_entry is None:
        if required:
            raise KeyError(
                f"No {mode} defaults found for subject={subject!r}")
        return False
    params = _fit_entry_params(fit_entry)
    print(f"Setting {mode.upper()} defaults for:", subject)
    for val_name, val in params.items():
        if val_name not in all_widgets:
            continue
        widget = all_widgets[val_name]
        changes_tmp = widget._trait_notifiers["value"]["change"]
        widget._trait_notifiers["value"]["change"] = []
        widget.value = val
        widget._trait_notifiers["value"]["change"] = changes_tmp
    # MLE-only config knobs that live on the saved ``model_config`` rather
    # than in ``params``. Restore them too so loading an MLE fit also
    # restores the terminal_c / etc. it was fitted under. Chisq pickles
    # don't carry a ``model_config``, so this is a no-op there.
    fit_config = _fit_entry_mle_config(fit_entry)
    if fit_config is not None and "MLE_TERMINAL_C" in all_widgets:
        terminal_c = getattr(fit_config, "mle_terminal_c", None)
        if terminal_c is not None:
            widget = all_widgets["MLE_TERMINAL_C"]
            terminal_c = min(
                max(float(terminal_c), MLE_TERMINAL_C.Min),
                MLE_TERMINAL_C.Max)
            changes_tmp = widget._trait_notifiers["value"]["change"]
            widget._trait_notifiers["value"]["change"] = []
            widget.value = terminal_c
            widget._trait_notifiers["value"]["change"] = changes_tmp
    return True


def _get_subject_fit_entry(subjects_defaults, t_dur, noiseFn, biasFn, driftFn,
                           subject):
    if subjects_defaults is None:
        return None
    try:
        return subjects_defaults[t_dur][noiseFn][biasFn][driftFn][subject]
    except KeyError:
        return None


def _mle_loss_key(subject, driftFn_str, biasFn_str, noiseFn_str, t_dur, dt,
                  params):
    params_key = tuple((key, float(params[key])) for key in sorted(params))
    return (
        subject, driftFn_str, biasFn_str, noiseFn_str,
        float(t_dur), float(dt), params_key)


def _stored_mle_loss_from_df(df):
    for col in ("mle_loss", "mle_neg_loglik", "mle_fit_neg_loglik"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
            values = [float(value) for value in values if np.isfinite(value)]
            if values:
                return values[0]
    return None


def _stored_mle_loss_from_fit_entry(fit_entry):
    fit_entry = _fit_entry_for_mode(fit_entry, "mle")
    if fit_entry is None:
        return None
    result = _fit_entry_result(fit_entry)
    for key in ("neg_loglik", "mle_loss", "mle_neg_loglik"):
        value = result.get(key)
        if _is_finite_number(value):
            return float(value)
    return None


def _evaluate_mle_loss_for_gui(df, params, driftFn_str, biasFn_str, noiseFn_str,
                               include_Q, include_RewardRate, dt, t_dur,
                               fit_entry=None, terminal_c_override=None):
    fit_config = _fit_entry_mle_config(fit_entry)
    if terminal_c_override is not None:
        # GUI slider value wins over both the saved fit-config and the
        # df-stored column, so the user can sweep mle_terminal_c
        # interactively without reloading.
        terminal_c = float(terminal_c_override)
    elif fit_config is not None:
        terminal_c = getattr(
            fit_config, "mle_terminal_c", MLE_TERMINAL_C.Default)
    else:
        terminal_c = _scalar_column_value(
            df, "mle_terminal_c", default=MLE_TERMINAL_C.Default)
    dx = (
        _scalar_column_value(df, "mle_dx", default=0.02)
        if fit_config is None
        else getattr(fit_config, "dx", 0.02))
    config = MLEModelConfig(
        drift_fn_str=driftFn_str,
        bias_fn_str=biasFn_str,
        noise_fn_str=noiseFn_str,
        include_Q=include_Q,
        include_RewardRate=include_RewardRate,
        dt=float(dt),
        t_dur=float(t_dur),
        dx=float(dx),
        mle_array_backend="numpy",
        mle_cupy_fallback="numpy",
        mle_terminal_c=float(terminal_c),
    )
    return evaluate_neg_loglik(params, df, config, return_df=False).neg_loglik


def _fit_entry_mle_config(fit_entry):
    fit_entry = _fit_entry_for_mode(fit_entry, "mle")
    if fit_entry is None:
        return None
    return _fit_entry_result(fit_entry).get("model_config")


def _fit_entry_for_mode(entry, mode):
    return _fit_entries_by_mode(entry).get(mode)


def _fit_entries_by_mode(entry):
    if entry is None:
        return {}
    if isinstance(entry, dict):
        if "mle" in entry or "chisq" in entry:
            return {
                mode: fit_entry for mode, fit_entry in entry.items()
                if mode in {"mle", "chisq"}}
        if _is_params_dict(entry):
            return {"mle": entry, "chisq": entry}
        inferred = _infer_fit_entry_mode(entry)
        return {} if inferred is None else {inferred: entry}

    entries = {}
    if isinstance(entry, (tuple, list)):
        # Older notebook cells stored at most two fit entries as positional
        # values. Preserve that shape by treating the order as MLE, then chisq
        # unless the payload itself makes the mode clear.
        for idx, fit_entry in enumerate(entry):
            inferred = _infer_fit_entry_mode(fit_entry)
            if inferred is None and _is_fit_entry_payload(fit_entry):
                inferred = ("mle", "chisq")[idx] if idx < 2 else None
            if inferred is not None and inferred not in entries:
                entries[inferred] = fit_entry
    return entries


def _infer_fit_entry_mode(fit_entry):
    result = _fit_entry_result(fit_entry)
    mode = result.get("fit_mode")
    if mode in {"mle", "chisq"}:
        return mode
    if any(key in result for key in ("neg_loglik", "mle_df",
                                     "mle_observation_model")):
        return "mle"
    if "OptimRes" in result:
        return "chisq"
    return None


def _fit_entry_params(fit_entry):
    if isinstance(fit_entry, dict) and "params" in fit_entry:
        return _coerce_params_dict(fit_entry["params"])
    if _is_params_dict(fit_entry):
        return _coerce_params_dict(fit_entry)
    result = _fit_entry_result(fit_entry)
    if "params_names" in result and "OptimRes" in result:
        optim_res = result["OptimRes"]
        return {
            str(name): val
            for name, val in zip(result["params_names"], optim_res.x)}
    if isinstance(fit_entry, (tuple, list)) and fit_entry:
        for item in fit_entry:
            if _is_params_dict(item):
                return _coerce_params_dict(item)
    print("Couldn't find params in fit entry:", fit_entry)
    raise KeyError("Fit entry does not contain parameter defaults")


def _fit_entry_result(fit_entry):
    if isinstance(fit_entry, dict):
        result = fit_entry.get("result")
        return result if isinstance(result, dict) else fit_entry
    if isinstance(fit_entry, (tuple, list)):
        for item in fit_entry:
            if isinstance(item, dict) and (
                    "params_names" in item or "OptimRes" in item
                    or "neg_loglik" in item or "fit_mode" in item):
                return item
    return {}


def _is_fit_entry_payload(value):
    return isinstance(value, dict) and (
        _is_params_dict(value)
        or "params" in value
        or "result" in value
        or "params_names" in value
        or "OptimRes" in value)


def _is_params_dict(value):
    if not isinstance(value, dict) or not value:
        return False
    metadata_keys = {
        "params", "result", "params_names", "OptimRes", "neg_loglik",
        "mle_loss", "mle_neg_loglik", "fit_mode", "model_config", "mle_df",
        "mle_observation_model",
    }
    keys = {str(key) for key in value}
    if keys & metadata_keys:
        return False
    widget_param_names = {
        "DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME", "BETA",
        "BIAS_COEF", "BIAS_FIXED", "BIAS_MU", "BIAS_SIGMA", "ALPHA",
        "Q_VAL_DECAY_RATE", "Q_VAL_COEF", "Q_VAL_OFFSET",
    }
    return bool(keys & widget_param_names)


def _coerce_params_dict(params):
    return {str(name): val for name, val in params.items()}


def _scalar_column_value(df, col, default):
    if col not in df.columns:
        return default
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    values = [float(value) for value in values if np.isfinite(value)]
    return values[0] if values else default


def _is_finite_number(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False

def _makeSaveFigTitle(fig, name, loss, driftFn_str, biasFn_str):
    model_name = _makeModelName(driftFn_str, biasFn_str)
    fig_title = f"{name} - {model_name} - Chi-Square Loss={loss:,.2f}"
    fig.suptitle(fig_title, y=1.05)
    return fig_title

def _makeModelName(driftFn_str, biasFn_str):
    if driftFn_str == "_driftClassic":
        if biasFn_str == "_biasNone":
            model_name = "Classic DDM (No Bias, z=0)"
        elif biasFn_str == "_biasQVal":
            model_name = "Classic DDM + Init Q-Value"
        else:
            model_name = f"Classic DDM + {biasFn_str}"
    elif driftFn_str == "_decayQ_nondectime_Q_True":
        if biasFn_str == "_biasNone":
            model_name = "Classic DDM + Decaying Q"
        else:
            model_name = f"Classic DDM + Decaying Q + {biasFn_str}"
    elif driftFn_str == "_noiseGainRewardRate":
        if biasFn_str == "_biasNone":
            model_name = "Noise*RewardRate (No Bias, z=0)"
        elif biasFn_str == "_biasQVal":
            model_name = "Noise*RewardRate + Init Q-Value"
    elif driftFn_str == "_noiseGainDecayingQ_nondectime_Q_True":
        if biasFn_str == "_biasNone":
            model_name = "Noise*RewardRate + Decaying Q"
        else:
            model_name = f"Noise*RewardRate + Decaying Q + {biasFn_str}"
    else:
        model_name = f"{driftFn_str} + {biasFn_str}"
    return model_name
