from .logic import makeOneRun
from .drift import DRIFT_FN_DICT
from .noise import NOISE_FN_DICT
from .bias import BIAS_FN_DICT
from .plotter import runAndPlot, createFig
from .util import (initDF, assignQuantiles, driftFnColsAndKwargs, 
                   biasFnColsAndKwargs, noiseFnColsAndKwargs, PsychometricPlot)
from .initvals import InitVals
# from ._readparams import readParams
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
from inspect import signature
import pathlib
import os


def createWidget(init_vals : InitVals, gui_cache : InitVals, df, t_dur, dt,
                 is_small_fig_mode, subjects_defaults=None, save_figs=False,
                 save_ovewrite=True):

    drop_downs_labels = ["Subject", "DV", "Drift Fn", "Bias Fn",
                         "Psychometric", "Noise Fn"]

    checkboxes_labels = {"Real-time": gui_cache.get("Real-time", True)}
    buttons_labels = ["Reset to defaults", "Update"]

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
            default_val_idx = options_str.index("NoiseGain-RewardRate Decay Q")
            # print("options_str:", options_str)
            # print("values:", values)
        elif label == "Bias Fn":
            # BIASN_FN_FIXED, BIAS_FN_MEAN_SIGMA, BIAS_FN_QVAL = \
            #        "Fixed",       "Mean+Sigma",    "Q-Value"
            # options_str = [BIASN_FN_FIXED, BIAS_FN_MEAN_SIGMA, BIAS_FN_QVAL]
            # values =      [_biasFixed,     _biasMean,          _biasQVal]
            options_str = list(BIAS_FN_DICT.keys())
            values =      list(BIAS_FN_DICT.values())
            default_val_idx = options_str.index("Q-Val")
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
    fourth_col = ([checkbox_widgets.pop("Real-time")] + button_widgets_li +
                  [html_math_widgets.pop("DriftEq"),
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
    last_loss = None
    def updateGUI():
        nonlocal all_widgets, include_Q, include_RewardRate, checkbox_last_val, fig, last_loss
        nonlocal last_subject, last_driftFn, last_biasFn
        # print("Updating GUI:", "Update:", all_widgets["Real-time"].value)

        cur_subject = all_widgets["Subject"].value
        cached_subject_df = None
        df = all_df[all_df.Name == cur_subject]
        # print("0")
        if ((last_subject != cur_subject) or (last_driftFn != all_widgets["Drift Fn"].value) or
            (last_biasFn != all_widgets["Bias Fn"].value)) and subjects_defaults is not None:
        # if True: 
            # [t_dur, noiseFn, biasFn, driftFm, subject]
            # print("1")
            cur_t_dur_dict = subjects_defaults.get(t_dur, {})
            if cur_t_dur_dict:
                # print("2:", all_widgets["Noise Fn"].value.__name__)
                # print("cur_t_dur_dict:", cur_t_dur_dict.keys())
                cur_noiseFn_dict = cur_t_dur_dict.get(all_widgets["Noise Fn"].value.__name__)
                if cur_noiseFn_dict:
                    # print("3")
                    cur_biasFn_dict = cur_noiseFn_dict.get(all_widgets["Bias Fn"].value.__name__)
                    if cur_biasFn_dict:
                        # print("4", all_widgets["Drift Fn"].value.__name__)
                        # print("cur_biasFn_dict:", cur_biasFn_dict.keys())
                        cur_driftFn_dict = cur_biasFn_dict.get(all_widgets["Drift Fn"].value.__name__)
                        if cur_driftFn_dict:
                            # print("5")
                            cur_subject_dict, cached_subject_df = cur_driftFn_dict.get(cur_subject)
                            if cur_subject_dict is not None:
                                print("Setting defaults for:", cur_subject)
                                for val_name, val in cur_subject_dict.items():
                                    changes_tmp = all_widgets[val_name]._trait_notifiers['value']['change']
                                    all_widgets[val_name]._trait_notifiers['value']['change'] = []
                                    all_widgets[val_name].value = val
                                    all_widgets[val_name]._trait_notifiers['value']['change'] = changes_tmp

        last_subject = cur_subject
        last_driftFn = all_widgets["Drift Fn"].value
        last_biasFn = all_widgets["Bias Fn"].value
        # We can't do the DV filtering here, because we need all subsequent
        # trials to build Q values abd RewardRate. So we will rather do
        # the filtering on the results

        # Continue with the update
        # If Drift Fn is "Classic" then disable RewardRate's BETA
        biasFn = all_widgets["Bias Fn"].value
        driftFm = all_widgets["Drift Fn"].value
        noiseFn = all_widgets["Noise Fn"].value

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
        if not update_toggl_btn.value and not is_realtime:
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

        last_loss, _ = runAndPlot(df, biasFn=biasFn, driftFn=driftFm, noiseFn=noiseFn,
                               dvs_filter=dvs_filter, fig=fig, axs=axs,
                               plot_bias_dir=plot_bias_dir, psych_plot=psych_plot,
                               t_dur=t_dur, dt=dt, include_Q=include_Q,
                               include_RewardRate=include_RewardRate,
                               biasFn_kwargs=biasFn_kwargs, biasFn_df_cols=biasFn_df_cols,
                               driftFn_kwargs=driftFn_kwargs, driftFn_df_cols=driftFn_df_cols,
                               noiseFn_kwargs=noiseFn_kwargs, noiseFn_df_cols=noiseFn_df_cols,
                               cached_subject_df=cached_subject_df,
                               is_small_fig_mode=is_small_fig_mode,
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

    all_df = initDF(df, include_Q=True, include_RewardRate=True)
    # updateGUI(all_widgets["Update"])


    # Run the display
    all_widgets_wo_btns = {k:v for k,v in all_widgets.items()
                           if not isinstance(v, widgets.Button)}
    def outHandler(*args, **kwargs):
        # print("Args:", args)
        # print("Kwargs:", kwargs)
        updateGUI()
        # return updateGUI()

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

def _makeSaveFigTitle(fig, name, loss, driftFn_str, biasFn_str):
    model_name = _makeModelName(driftFn_str, biasFn_str)
    fig_title = f"{name} - {model_name} - Loss={loss:,.2f}"
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