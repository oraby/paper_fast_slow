from .logic import makeOneRun
from .drift import (
    DRIFT_FN_DICT, _REWARDRATE_ALIAS_FOR_INTERNAL, resolve_drift_alias,
    user_facing_drift_keys)
from .noise import NOISE_FN_DICT
from .bias import BIAS_FN_DICT
from .plotter import runAndPlot, createFig
from .util import (initDF, assignQuantiles, driftFnColsAndKwargs,
                   biasFnColsAndKwargs, noiseFnColsAndKwargs, PsychometricPlot)
from .initvals import InitVals, MLE_TERMINAL_C
from .mle import MLEModelConfig, evaluate_neg_loglik
# from ._readparams import readParams
import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
from inspect import signature
import pathlib
import os
import pickle
import re


# Sliders that are NOT a kwarg of any drift/bias/noise/logic function but
# DO feed the MLE objective. The widget loop's auto-disable check
# (`widget.disabled = widget.description not in all_kwargs_names`) would
# otherwise hide them, so we exempt them and let the user always edit them.
# The value is consumed by ``_mle_params_from_widgets`` only when "Run MLE"
# is pressed, so the chisq simulator path remains unaffected.
_MLE_ONLY_SLIDER_NAMES = {"LAPSE_RATE", "MLE_TERMINAL_C"}


def _include_flags_for_current_model(all_widgets):
    """Whether the active model learns Q-values / RewardRate.

    Source of truth is the Drift/Bias/Noise dropdowns (the *Fn columns
    each fn declares it needs) — NOT the asym checkboxes' ``disabled``
    flag. updateGUI sets that disabled flag late in the same callback,
    so any caller earlier in updateGUI (the auto-apply variant-suffix
    lookup at the top of updateGUI) would otherwise read a stale value
    from the previous model.
    """
    biasFn  = BIAS_FN_DICT[all_widgets["Bias Fn"].value]
    driftFn = DRIFT_FN_DICT[_resolved_drift_fn_value(all_widgets)]
    noiseFn = NOISE_FN_DICT[all_widgets["Noise Fn"].value]
    biasFn_df_cols,  _ = biasFnColsAndKwargs(biasFn)
    driftFn_df_cols, _ = driftFnColsAndKwargs(driftFn)
    noiseFn_df_cols, _ = noiseFnColsAndKwargs(noiseFn)
    include_Q = (
        "Q_val" in biasFn_df_cols
        or "Q_val" in driftFn_df_cols
        or "Q_val" in noiseFn_df_cols)
    include_RewardRate = (
        "RewardRate" in driftFn_df_cols
        or "RewardRate" in noiseFn_df_cols
        or "RewardRate" in biasFn_df_cols)
    return include_Q, include_RewardRate


def _asym_mode_suffix(all_widgets):
    """Return the asym suffix that matches the current checkbox state.

    Matches ``fit.evolveFP``'s filename convention so the GUI can look up
    ``subjects_defaults[...][subject]["mle_asymQ"]`` etc. when the
    relevant checkbox is ticked. Returns ``""`` when neither checkbox is
    set so symmetric fits load as before.

    A ticked checkbox is treated as effectively off when the active
    model doesn't learn the matching quantity (no Q-learning →
    ``Asymmetric Q-update`` contributes no suffix). Otherwise a stale
    tick left over from a previous model would steer the preferred-mode
    chain (and the Reset-button gating that consults its first element)
    at a saved-fit key — ``mle_asymQ`` — that can't exist for the new
    model, hiding the symmetric ``mle`` fit that DOES exist.
    """
    include_Q, include_RewardRate = _include_flags_for_current_model(
        all_widgets)
    asym_q = (
        include_Q
        and "Asymmetric Q-update" in all_widgets
        and bool(all_widgets["Asymmetric Q-update"].value))
    asym_rr = (
        include_RewardRate
        and "Asymmetric RR-update" in all_widgets
        and bool(all_widgets["Asymmetric RR-update"].value))
    if asym_q and asym_rr:
        return "_asymQRR"
    if asym_q:
        return "_asymQ"
    if asym_rr:
        return "_asymRR"
    return ""


def _scaled_bound_suffix(all_widgets):
    """Return ``_scaledB`` when the Scale-How dropdown is set to "Bound".

    Mirrors the filename convention written by ``fit.evolveFP`` for
    ``--scale-bound`` fits.
    """
    return ("_scaledB"
            if ("Scale-How" in all_widgets
                and all_widgets["Scale-How"].value == "Bound")
            else "")


def _weight_mode_suffix(all_widgets):
    """Return the joint-loss weight suffix selected in the "Joint Wt" dropdown
    (e.g. ``_mleW1_chi2W0.5``), or ``""`` for the pure / "None" option.

    A fourth orthogonal variant axis alongside asym / Scale-How: the dropdown
    value IS the suffix ``fit.evolveFP`` appends to joint MLE+Chi² fits, so the
    GUI looks up the exact weight variant. Appended LAST (after scaledB),
    matching evolveFP's suffix order and the notebook's ``fit_key``.
    """
    if "Joint Wt" in all_widgets:
        return str(all_widgets["Joint Wt"].value)
    return ""


def _variant_suffix(all_widgets):
    """The composed model-variant filename suffix — asym, then scaledB, then
    joint weights — in ``fit.evolveFP`` order.

    Single source of truth so the two consumers can't drift apart:
    ``_preferred_modes_for`` (which builds the saved-fit mode key for the
    title / Reset / auto-apply lookups) and ``updateGUI``'s variant-change
    detector (which decides whether to re-pull a fit's params into the
    sliders). They were independently concatenating these parts, and when the
    "Joint Wt" axis was added to only the former, changing the dropdown updated
    the title but never re-triggered the auto-apply — the figure stayed stale
    until a manual Reset.
    """
    return (_asym_mode_suffix(all_widgets) + _scaled_bound_suffix(all_widgets)
            + _weight_mode_suffix(all_widgets))


# Joint-loss weight suffix as written by fit.evolveFP (``_mleW{m}_chi2W{c}``).
_WEIGHT_SUFFIX_IN_KEY_RE = re.compile(
    r"(_mleW[-+0-9.eE]+_chi2W[-+0-9.eE]+)")


def _discover_weight_suffixes(subjects_defaults):
    """``[(label, suffix)]`` joint-weight options for the "Joint Wt" dropdown,
    discovered from the loaded fit cache. Always leads with the pure
    ``("None", "")`` option; each joint variant present anywhere in the cache
    adds one entry (``_mleW1_chi2W0.5`` → label ``mleW=1 chi2W=0.5``)."""
    suffixes = set()
    stack = [subjects_defaults]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        # A fit-entry leaf — match nothing, don't descend into the payload.
        if "result" in node or "params" in node:
            continue
        for key, val in node.items():
            m = _WEIGHT_SUFFIX_IN_KEY_RE.search(str(key))
            if m:
                suffixes.add(m.group(1))
            else:
                stack.append(val)
    options = [("None", "")]
    for suffix in sorted(suffixes):
        label = (suffix.lstrip("_").replace("_", " ")
                 .replace("mleW", "mleW=").replace("chi2W", "chi2W="))
        options.append((label, suffix))
    return options


def _resolved_drift_fn_value(all_widgets):
    """Read the Drift Fn dropdown's value and resolve any
    ``RewardRate*`` alias to the canonical ``DRIFT_FN_DICT`` key.

    The dropdown shows the alias; all downstream code (``DRIFT_FN_DICT``
    lookups, ``_get_subject_fit_entry`` saved-fit keys, the
    ``startswith("Bound-RewardRate")`` per-trial-bound check) expects
    the resolved internal name. The Scale-How dropdown is the source
    of truth for which implementation the alias maps to.
    """
    raw = all_widgets["Drift Fn"].value
    scale_bound_active = (
        "Scale-How" in all_widgets
        and all_widgets["Scale-How"].value == "Bound")
    return resolve_drift_alias(raw, scale_bound_active)


def _preferred_modes_for(base_mode, all_widgets):
    """Build the strict saved-fit mode key for the current checkbox state.

    Each ticked checkbox / dropdown contributes a suffix to the key we look
    up: ``mle_asymQ`` / ``mle_asymRR`` / ``mle_asymQRR`` for asym + Q/RR,
    ``_scaledB`` for the BOUND-fitted axis, and ``_mleW{m}_chi2W{c}`` for the
    "Joint Wt" joint-loss variant. Suffix ordering matches ``fit.evolveFP``:
    asym, then scaledB, then joint weights.

    Returns a SINGLE-ELEMENT tuple by design — strict matching only.
    Earlier versions returned a fallback chain (``mle_asymQ`` → ``mle``
    → ``chisq``) so the GUI could silently load *something* even when
    the exact variant hadn't been fit, but that hid surprises:
    ticking Asym-Q on a model whose ``mle_asymQ`` didn't exist would
    silently load the symmetric ``mle`` and the user wouldn't realize
    they were looking at the wrong variant. Strict policy now: if the
    exact variant has no saved fit the auto-apply / title / Reset paths
    leave the GUI in its previous state (sliders untouched, title
    showing "not run"), making the absence visible.

    The tuple shape (rather than a bare string) is kept for API
    compatibility with the chain-walking callers
    (``_try_apply_fit_defaults`` / ``_stored_mle_loss_from_fit_entry``),
    which iterate over the returned value — strict matching is just the
    1-element case of "walk the chain".
    """
    return (f"{base_mode}{_variant_suffix(all_widgets)}",)


def createWidget(init_vals : InitVals, gui_cache : InitVals, df, t_dur, dt,
                 is_small_fig_mode, subjects_defaults=None, save_figs=False,
                 save_ovewrite=True):

    # One-shot GUI cache migration: the Drift Fn dropdown used to expose
    # the NoiseGain-/Bound- implementation names directly; both are now
    # collapsed into a single ``RewardRate`` alias and Scale-How decides
    # which family dispatches. Map deprecated cached values to the alias
    # and pre-set Scale-How so users keep their previous selection
    # across the change. Idempotent on already-migrated caches.
    if gui_cache is not None:
        cached_drift = gui_cache.get("Drift Fn", None)
        if cached_drift in _REWARDRATE_ALIAS_FOR_INTERNAL:
            gui_cache["Drift Fn"] = _REWARDRATE_ALIAS_FOR_INTERNAL[cached_drift]
            gui_cache["Scale-How"] = (
                "Bound" if cached_drift.startswith("Bound-RewardRate")
                else "Noise")

    drop_downs_labels = ["Scale-How", "Joint Wt", "Subject", "DV", "Drift Fn",
                         "Bias Fn", "Psychometric", "Noise Fn"]
    # Joint-loss weight variants discovered from the loaded fit cache (the
    # "Joint Wt" dropdown), leading with the pure ("None", "") option.
    weight_suffix_options = _discover_weight_suffixes(subjects_defaults)

    # The two asym checkboxes are orthogonal to the bias / drift / noise
    # dropdown selection — they enable ALPHA_UNREWARDED / BETA_UNREWARDED
    # fitting on top of whatever Q-learning / RewardRate-learning the
    # selected model already does. updateGUI grays them out (via
    # widget.disabled) when the active model doesn't actually learn
    # Q-values or a reward rate.
    checkboxes_labels = {
        "Real-time": gui_cache.get("Real-time", True),
        "Asymmetric Q-update":  gui_cache.get("Asymmetric Q-update", False),
        "Asymmetric RR-update": gui_cache.get("Asymmetric RR-update", False),
    }
    # NOTE: ``Scale Bound`` was a checkbox in earlier work; the same
    # capability is now driven by the ``Scale-How`` dropdown (top of
    # first column). See the dropdown handler below.
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
            # User-facing names: the NoiseGain-/Bound- pairs are collapsed
            # to a single ``RewardRate`` alias. The Scale-How dropdown
            # decides which implementation actually dispatches; the alias
            # is resolved at the top of updateGUI before any downstream
            # code touches driftFn_str.
            options_str = user_facing_drift_keys()
            values = options_str
            default_val_idx = options_str.index("RewardRate")
        elif label == "Bias Fn":
            options_str = list(BIAS_FN_DICT.keys())
            values = options_str
            default_val_idx = options_str.index("Q-Val (Offset)")
        elif label == "Psychometric":
            options_str = ["None", "All", "Slow/Fast"]
            values = [PsychometricPlot._None, PsychometricPlot.All, PsychometricPlot.SlowFast]
            default_val_idx = 2
        elif label == "Noise Fn":
            options_str = list(NOISE_FN_DICT.keys())
            values = options_str
        elif label == "Scale-How":
            # Two-option dropdown for the scale-axis swap. ``Noise`` is
            # the legacy default — NOISE_SIGMA is fitted, BOUND is
            # frozen at ``InitVals._BOUND_FIXED``. ``Bound`` flips the
            # gate via the InitVal override in fit.simulateDDM (and
            # implies absolute-bias semantics in the MLE eval).
            options_str = ["Noise", "Bound"]
            values = options_str
            default_val_idx = 0
        elif label == "Joint Wt":
            # Joint MLE+Chi² weight variant, a fourth orthogonal model axis
            # (like asym / Scale-How). ``None`` (value "") is the pure fit;
            # other entries are the ``_mleW{m}_chi2W{c}`` suffixes discovered
            # in the loaded cache. The selected suffix flows into the saved-fit
            # mode key via ``_weight_mode_suffix`` / ``_preferred_modes_for``.
            options_str = [label_text for label_text, _suffix
                           in weight_suffix_options]
            values = [suffix for _label_text, suffix in weight_suffix_options]
            default_val_idx = 0
        else:
            raise ValueError(f"Unknown label: {label}")

        options = zip(options_str, values)
        # Default to the per-label default. Only consult the cache if the
        # label actually has an entry. The previous fallback of
        # ``default_val_idx`` (an int) into ``values.index(str(...))``
        # always missed when the cache was empty and printed a spurious
        # "Couldn't find N in [...]" for every widget on first load.
        val_idx = default_val_idx
        cached_value = gui_cache.get(label, None) if gui_cache is not None else None
        if cached_value is not None:
            try:
                from inspect import isfunction
                if isfunction(values[0]):
                    val_idx = [v.__name__ for v in values].index(
                        cached_value.__name__)
                else:
                    val_idx = [str(v) for v in values].index(str(cached_value))
            except (ValueError, AttributeError):
                # The cached value is incompatible with the current option
                # set (e.g. the widget setup switched from fn objects to
                # registry-key strings — old caches reset to default here).
                print(f"Couldn't find cached {label}={cached_value!r} in "
                      f"options; falling back to default.")
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
    # *_UNREWARDED sliders sit directly under their symmetric siblings,
    # gated by the matching asym checkbox. Asym is now orthogonal to the
    # bias / drift / noise selection — any Q-learning model can use
    # ALPHA_UNREWARDED, any RewardRate model can use BETA_UNREWARDED.
    # First column heads with the Scale-How dropdown so the mode-switch
    # visually drives the four scale-pair sliders directly below it:
    # NOISE_SIGMA + _NOISE_FIXED (the noise axis) and BOUND + _BOUND_FIXED
    # (the bound axis). Exactly one of each pair is enabled at a time
    # — see the gating block in updateGUI.
    first_col = [drop_down_widgets.pop("Scale-How"),
                 drop_down_widgets.pop("Drift Fn"),
                 slider_widgets.pop("DRIFT_COEF"),
                 slider_widgets.pop("NOISE_SIGMA"),
                 slider_widgets.pop("_NOISE_FIXED"),
                 slider_widgets.pop("BOUND"),
                 slider_widgets.pop("_BOUND_FIXED"),
                 slider_widgets.pop("NON_DECISION_TIME"),
                 slider_widgets.pop("BETA"),
                 checkbox_widgets.pop("Asymmetric RR-update"),
                 slider_widgets.pop("BETA_UNREWARDED"),
                 #slider_widgets.pop("Drift RR Coef"),
                 ]
    second_col = [drop_down_widgets.pop("Bias Fn"),
                  slider_widgets.pop("BIAS_COEF"),
                  slider_widgets.pop("BIAS_FIXED"),
                  slider_widgets.pop("BIAS_MU"),
                  slider_widgets.pop("BIAS_SIGMA"),
                  slider_widgets.pop("ALPHA"),
                  checkbox_widgets.pop("Asymmetric Q-update"),
                  slider_widgets.pop("ALPHA_UNREWARDED"),]
    third_col = [drop_down_widgets.pop("Subject"),
                 drop_down_widgets.pop("Joint Wt"),
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
    last_variant_suffix = None   # composed asym + Scale-How + Joint-Wt suffix
    last_loss = None
    last_mle_loss = None
    last_mle_loss_source = None
    last_mle_loss_key = None
    def updateGUI(force_update=False, run_mle=False):
        nonlocal all_widgets, include_Q, include_RewardRate, checkbox_last_val, fig, last_loss
        nonlocal last_subject, last_driftFn, last_biasFn, last_noiseFn
        nonlocal last_variant_suffix
        nonlocal last_mle_loss, last_mle_loss_source, last_mle_loss_key
        # print("Updating GUI:", "Update:", all_widgets["Real-time"].value)

        cur_subject = all_widgets["Subject"].value
        df = all_df[all_df.Name == cur_subject]
        # print("0")
        # Treat an asym-checkbox or Scale-How toggle as a "load defaults"
        # trigger — flipping any of them should pull the matching
        # ``mle_asymQ`` / ``mle_asymRR`` / ``mle_asymQRR`` / ``mle_scaledB``
        # / composed-suffix fit's params straight into the sliders if
        # that fit exists for the subject. Strict on the variant suffix,
        # with one cross-mode fallback: try the exact MLE variant first,
        # then the matching chisq variant, then nothing. The chisq
        # cross-mode fallback is intentional ONLY on this auto-apply
        # path — the Reset-MLE button (label promises MLE) and the
        # title's "MLE Loss:" (label says MLE) stay strict-MLE-or-
        # nothing. Here we'd rather show *some* fit's params than leave
        # the sliders stale from the previous subject.
        # Same composition the saved-fit mode key uses (see _variant_suffix):
        # this MUST include the "Joint Wt" weight suffix, else changing the
        # dropdown wouldn't register as a variant change and the auto-apply
        # below wouldn't re-pull the fit's params — the title would update but
        # the figure would stay stale until a manual Reset.
        cur_variant_suffix = _variant_suffix(all_widgets)
        cur_driftFn_resolved = _resolved_drift_fn_value(all_widgets)
        if ((last_subject != cur_subject) or (last_driftFn != cur_driftFn_resolved) or
            (last_biasFn != all_widgets["Bias Fn"].value) or
            (last_noiseFn != all_widgets["Noise Fn"].value) or
            (last_variant_suffix != cur_variant_suffix)) and subjects_defaults is not None:
            _try_apply_fit_defaults(
                all_widgets, subjects_defaults, t_dur, cur_subject,
                preferred_modes=(
                    _preferred_modes_for("mle", all_widgets)
                    + _preferred_modes_for("chisq", all_widgets)),
                required=False, quiet=True)

        last_subject = cur_subject
        last_driftFn = cur_driftFn_resolved
        last_biasFn = all_widgets["Bias Fn"].value
        last_variant_suffix = cur_variant_suffix
        last_noiseFn = all_widgets["Noise Fn"].value
        # We can't do the DV filtering here, because we need all subsequent
        # trials to build Q values abd RewardRate. So we will rather do
        # the filtering on the results

        # Continue with the update
        # If Drift Fn is "Classic" then disable RewardRate's BETA
        # Widget values are now the registry KEY strings (see the widget
        # setup above) — they uniquely identify the model variant, which
        # the underlying fn objects don't because the -asym aliases share
        # them. ``biasFn`` etc. (the actual callables) are looked up
        # explicitly when needed below.
        biasFn_str = all_widgets["Bias Fn"].value
        driftFn_str = _resolved_drift_fn_value(all_widgets)
        noiseFn_str = all_widgets["Noise Fn"].value
        biasFn = BIAS_FN_DICT[biasFn_str]
        driftFm = DRIFT_FN_DICT[driftFn_str]
        noiseFn = NOISE_FN_DICT[noiseFn_str]
        print(f"Selected Drift Fn: {driftFn_str}, Bias Fn: {biasFn_str}, "
              f"Noise Fn: {noiseFn_str}")
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
        # Scale-How dropdown drives the 4-slider scale-pair gating.
        # Exactly two of (NOISE_SIGMA, _NOISE_FIXED, BOUND, _BOUND_FIXED)
        # are enabled at a time: the fittable slider for the active
        # axis + the frozen counterpart for the inactive axis. The
        # kwarg-translation block right after the kwarg-collecting
        # loop (below, search "Scale-How translates") maps the
        # active-axis pair to the canonical BOUND / NOISE_SIGMA names
        # that runAndPlot consumes.
        scale_how = all_widgets["Scale-How"].value
        if scale_how == "Noise":
            all_widgets["NOISE_SIGMA"].disabled  = False
            all_widgets["_BOUND_FIXED"].disabled = False
            all_widgets["BOUND"].disabled        = True
            all_widgets["_NOISE_FIXED"].disabled = True
        else:  # "Bound"
            all_widgets["NOISE_SIGMA"].disabled  = True
            all_widgets["_BOUND_FIXED"].disabled = True
            all_widgets["BOUND"].disabled        = False
            all_widgets["_NOISE_FIXED"].disabled = False
        # Asymmetric-LR gating is now orthogonal to the model identity:
        # the checkbox is the source of truth. The checkbox itself is
        # disabled when the active model wouldn't learn the matching
        # quantity (no Q-learning → no asym Q, no reward-rate → no asym
        # RR); the *_UNREWARDED slider follows both the checkbox state
        # and the include_* flag.
        asym_q_cb  = all_widgets["Asymmetric Q-update"]
        asym_rr_cb = all_widgets["Asymmetric RR-update"]
        asym_q_cb.disabled  = not include_Q
        asym_rr_cb.disabled = not include_RewardRate
        if not (include_Q and asym_q_cb.value):
            all_widgets["ALPHA_UNREWARDED"].disabled = True
        if not (include_RewardRate and asym_rr_cb.value):
            all_widgets["BETA_UNREWARDED"].disabled = True

        # Gray out the Reset buttons when the EXACT variant for the
        # current (Asym-Q × Asym-RR × Scale-How) checkbox state has no
        # saved fit. ``_preferred_modes_for`` is strict — it returns
        # only the exact variant — so the button is enabled iff that
        # specific saved fit exists for the subject. Ticking Asym-Q
        # without a saved ``mle_asymQ`` leaves the button disabled even
        # when the symmetric ``mle`` exists, making the absence visible
        # rather than silently loading the wrong variant. Re-evaluated
        # on every updateGUI pass.
        mle_chain = _preferred_modes_for("mle", all_widgets)
        chisq_chain = _preferred_modes_for("chisq", all_widgets)
        all_widgets["Reset to MLE defaults"].disabled = (
            not _any_fit_available_for_chain(
                subjects_defaults, t_dur, all_widgets, cur_subject,
                mle_chain))
        all_widgets[chi2_reset_label].disabled = (
            not _any_fit_available_for_chain(
                subjects_defaults, t_dur, all_widgets, cur_subject,
                chisq_chain))

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

        # Scale-How translates the slider in the active column into the
        # canonical BOUND / NOISE_SIGMA kwarg names runAndPlot consumes.
        # Without this fixup the disabled slider in the inactive column
        # would be skipped by the kwarg-collecting loop above (because
        # the loop filters out widget.disabled) and runAndPlot would
        # error with "missing 1 required positional argument: BOUND".
        if scale_how == "Noise":
            updatePlots_kwargs["BOUND"]       = all_widgets["_BOUND_FIXED"].value
            updatePlots_kwargs["NOISE_SIGMA"] = all_widgets["NOISE_SIGMA"].value
        else:  # "Bound"
            updatePlots_kwargs["BOUND"]       = all_widgets["BOUND"].value
            updatePlots_kwargs["NOISE_SIGMA"] = all_widgets["_NOISE_FIXED"].value

        # Registry-key form of the previous ``"CorrIncorr" in biasFn.__name__``
        # check — matches "Fixed (Corr/Incorr)" and "μ, σ (Corr/Incorr)".
        plot_bias_dir = "Corr/Incorr" in biasFn_str
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
            subjects_defaults, t_dur, noiseFn_str, biasFn_str,
            driftFn_str, cur_subject)
        # Use the same strict variant key the apply-defaults path uses
        # so the title's "MLE Loss: …" reflects the EXACT variant the
        # current (Scale-How × Asym-Q × Asym-RR) selection refers to.
        # No fallback: if no saved fit for the exact variant, this
        # returns None and the title shows "not run" rather than the
        # loss of a different variant.
        stored_mle_loss = _stored_mle_loss_from_fit_entry(
            fit_entry, _preferred_modes_for("mle", all_widgets))
        if stored_mle_loss is None:
            stored_mle_loss = _stored_mle_loss_from_df(df)
        # Joint MLE+Chi² breakdown for the saved variant (None for non-joint
        # fits) → second title line reconstructing how the total was built.
        joint_info = _joint_breakdown_from_fit_entry(
            fit_entry, _preferred_modes_for("mle", all_widgets))
        if run_mle:
            run_mle_btn = all_widgets["Run MLE"]
            old_desc = run_mle_btn.description
            run_mle_btn.disabled = True
            run_mle_btn.description = "Running MLE..."
            try:
                # NB: no ``except Exception`` here on purpose. A silent
                # swallow used to map any failure to ``np.nan`` +
                # ``last_mle_loss_source = "error"``, which surfaced in
                # the loss-title as a bland ``"not run (error)"`` and
                # hid the actual traceback (broadcast errors, missing
                # params, dtype mismatches, etc.). Letting the
                # exception propagate gives the user the real stack
                # trace in the notebook cell output; the ``finally``
                # block below still restores the button so the GUI
                # doesn't end up stuck in "Running MLE…".
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
                    uses_asym_q=bool(
                        all_widgets["Asymmetric Q-update"].value),
                    uses_asym_rr=bool(
                        all_widgets["Asymmetric RR-update"].value),
                    scale_bound=(
                        all_widgets["Scale-How"].value == "Bound"),
                )
                last_mle_loss_source = "current"
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
                               joint_info=joint_info,
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

    # The Reset buttons consult the asym checkboxes so the variant the
    # user is sweeping in the GUI (Q-update / RR-update) gets its own
    # saved defaults loaded. Strict matching: if the exact variant has
    # no saved fit the click is a no-op (sliders unchanged) — the
    # button gate above grays it out in that case anyway, so this is
    # mostly belt-and-suspenders.
    all_widgets["Reset to MLE defaults"].on_click(
        lambda _button: _reset_defaults_and_update(
            all_widgets, subjects_defaults, t_dur,
            _preferred_modes_for("mle", all_widgets), updateGUI))
    all_widgets[chi2_reset_label].on_click(
        lambda _button: _reset_defaults_and_update(
            all_widgets, subjects_defaults, t_dur,
            _preferred_modes_for("chisq", all_widgets), updateGUI))
    all_widgets["Run MLE"].on_click(
        lambda _button: updateGUI(force_update=True, run_mle=True))
    out = widgets.interactive_output(outHandler, all_widgets_wo_btns)

    display_widget = display(layout, out)
    if save_figs and subjects_defaults is not None:
        real_time_cur_val = all_widgets["Real-time"].value
        all_widgets["Real-time"].value = False
        # subjects_defaults is now keyed by registry KEY strings — the same
        # strings the widgets accept — so no reverse map is needed.
        # [t_dur, noiseFn, biasFn, driftFn, subject]
        tmp_t_dur = t_dur
        for cur_t_dur, t_dur_dict in subjects_defaults.items():
            t_dur = cur_t_dur
            for noiseFn, noiseFn_dict in t_dur_dict.items():
                all_widgets["Noise Fn"].value = noiseFn
                for biasFn, biasFn_dict in noiseFn_dict.items():
                    all_widgets["Bias Fn"].value = biasFn
                    for driftFn, driftFn_dict in biasFn_dict.items():
                        # subjects_defaults is keyed by INTERNAL drift
                        # names (the saved-fit filename uses those);
                        # the dropdown now shows RewardRate aliases.
                        # Translate before assigning + pre-set Scale-How
                        # so the figure folder layout still works.
                        if driftFn in _REWARDRATE_ALIAS_FOR_INTERNAL:
                            all_widgets["Scale-How"].value = (
                                "Bound" if driftFn.startswith("Bound-RewardRate")
                                else "Noise")
                            all_widgets["Drift Fn"].value = (
                                _REWARDRATE_ALIAS_FOR_INTERNAL[driftFn])
                        else:
                            all_widgets["Drift Fn"].value = driftFn
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
        # Asymmetric-LR opt-ins. ``_compute_latent_arrays`` reads
        # these via strict access whenever ``uses_asymmetric_*`` is
        # True on the model_config (set when the GUI checkbox is
        # ticked) — KeyError on miss. Pass them through
        # unconditionally; when asym is off, the model_config flag
        # is False and these values are ignored.
        "ALPHA_UNREWARDED",
        "BETA_UNREWARDED",
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


def _reset_defaults_and_update(all_widgets, subjects_defaults, t_dur,
                               preferred_modes, update_fn):
    """Apply the first available fit from ``preferred_modes``.

    Under the strict-matching policy ``_preferred_modes_for`` returns
    a 1-element tuple (e.g. ``("mle_asymQ",)``) so this collapses to
    "apply the exact variant if it exists; otherwise no-op". The
    chain-walking signature is kept for API stability with the
    underlying ``_try_apply_fit_defaults`` helper. ``required=False``
    so a missing variant leaves the sliders alone instead of raising.
    """
    subject = all_widgets["Subject"].value
    _try_apply_fit_defaults(
        all_widgets, subjects_defaults, t_dur, subject,
        preferred_modes=preferred_modes, required=False)
    update_fn(force_update=True)


def _try_apply_fit_defaults(all_widgets, subjects_defaults, t_dur, subject,
                            preferred_modes, required=False, quiet=False):
    for mode in preferred_modes:
        if _apply_fit_defaults(
                all_widgets, subjects_defaults, t_dur, subject, mode=mode,
                required=False, quiet=quiet):
            return mode
    if required:
        raise KeyError(
            f"No defaults found for subject={subject!r}, "
            f"modes={tuple(preferred_modes)!r}")
    return None


def _apply_fit_defaults(all_widgets, subjects_defaults, t_dur, subject, mode,
                        required=True, quiet=False):
    """Apply a saved fit's params to the GUI sliders.

    ``quiet=True`` suppresses the "Setting X defaults" log line AND
    short-circuits when the (subject, mode, model) tuple matches the
    last successful apply. ipywidgets fires updateGUI multiple times
    per user action (one per traitlet update inside the dropdown
    change), so the auto-apply path passes ``quiet=True`` to avoid
    spamming stdout + re-running the per-widget assign/restore dance
    when nothing has actually changed. Explicit Reset button presses
    pass the default ``quiet=False`` so the user gets feedback on
    every click, even if they're clicking the same button.
    """
    entry = _get_subject_fit_entry(
        subjects_defaults,
        t_dur,
        all_widgets["Noise Fn"].value,
        all_widgets["Bias Fn"].value,
        _resolved_drift_fn_value(all_widgets),
        subject,
    )
    fit_entry = _fit_entry_for_mode(entry, mode)
    if fit_entry is None:
        if required:
            raise KeyError(
                f"No {mode} defaults found for subject={subject!r}")
        return False
    if quiet:
        log_key = (
            subject, mode,
            all_widgets["Noise Fn"].value,
            all_widgets["Bias Fn"].value,
            _resolved_drift_fn_value(all_widgets),
        )
        if getattr(_apply_fit_defaults, "_last_log_key", None) == log_key:
            return True
        _apply_fit_defaults._last_log_key = log_key
    params = _fit_entry_params(fit_entry)
    # Pull per-fit metadata from the saved entry so the user can see
    # WHICH fit (which finish time, which condition-balanced loss) is
    # being applied — important when ``--only-subject`` leaves a pickle
    # with mixed ``mle_condition_columns`` across subjects (each re-fit
    # subject overwrites its own dict entry, others stay untouched).
    suffix_parts = []
    if isinstance(fit_entry, dict):
        fit_result = fit_entry.get("result", {})
        finish_time = fit_result.get("fit_finish_time")
        if finish_time:
            try:
                # Saved as ``datetime.now().isoformat(timespec="seconds")``;
                # render as ``Wed Jun 17 15:23:45 2026`` for readability.
                finish_time_display = datetime.datetime.fromisoformat(
                    finish_time).ctime()
            except (TypeError, ValueError):
                finish_time_display = str(finish_time)
            suffix_parts.append(f"fit saved {finish_time_display}")
    fit_model_config = _fit_entry_result(fit_entry).get("model_config")
    cond_cols = getattr(fit_model_config, "mle_condition_columns", ())
    if cond_cols:
        suffix_parts.append(
            f"mle_conditions={','.join(cond_cols)}")
    suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
    print(f"Setting {mode.upper()} defaults for: {subject}{suffix}")
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


def _any_fit_available_for_chain(subjects_defaults, t_dur, all_widgets,
                                  subject, preferred_modes):
    """True iff any mode in ``preferred_modes`` has a saved fit for the
    current (subject, Drift Fn, Bias Fn, Noise Fn) selection.

    Used by ``updateGUI`` to gate the "Reset to MLE defaults" /
    "Reset to Chi²" buttons — they gray out when no corresponding
    saved fit exists for the current variant, so the user gets a
    visual cue about which combinations have actually been fit.
    """
    entry = _get_subject_fit_entry(
        subjects_defaults, t_dur,
        all_widgets["Noise Fn"].value,
        all_widgets["Bias Fn"].value,
        _resolved_drift_fn_value(all_widgets),
        subject,
    )
    if entry is None:
        return False
    return any(_fit_entry_for_mode(entry, mode) is not None
               for mode in preferred_modes)


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


def _stored_mle_loss_from_fit_entry(fit_entry, preferred_modes=("mle",)):
    """Look up a saved fit's neg_loglik for the given mode key(s).

    Under the strict-matching policy ``_preferred_modes_for`` passes a
    1-element tuple here (e.g. ``("mle_scaledB",)``) so the title
    reflects the EXACT variant the current Scale-How / Asym-Q / Asym-RR
    selection refers to — no silent fallback to a different mode.

    The function still iterates over ``preferred_modes`` so legacy
    callers (and tests) that pass a multi-element chain continue to
    work — strict matching is enforced upstream at the call site, not
    inside this walker.
    """
    for mode in preferred_modes:
        mode_entry = _fit_entry_for_mode(fit_entry, mode)
        if mode_entry is None:
            continue
        result = _fit_entry_result(mode_entry)
        for key in ("neg_loglik", "mle_loss", "mle_neg_loglik"):
            value = result.get(key)
            if _is_finite_number(value):
                return float(value)
    return None


def _joint_breakdown_from_fit_entry(fit_entry, preferred_modes=("mle",)):
    """Pull the joint MLE+Chi² breakdown (weights, part losses, references and
    their obtain times) from a saved fit entry for the loss-title's second
    line. Returns ``None`` unless the matched mode is a joint fit
    (``joint_mode``). Mirrors ``_stored_mle_loss_from_fit_entry``'s strict
    mode walk, so the breakdown reflects the EXACT variant the GUI selected."""
    for mode in preferred_modes:
        mode_entry = _fit_entry_for_mode(fit_entry, mode)
        if mode_entry is None:
            continue
        result = _fit_entry_result(mode_entry)
        if not result.get("joint_mode"):
            continue
        return {k: result.get(k) for k in (
            "total_loss", "mle_mle_weight", "mle_chi2_weight",
            "mle_part_loss", "chi2_part_loss", "ref_mle", "ref_chi2",
            "ref_mle_time", "ref_chi2_time")}
    return None


# Joint-loss weight suffix written by fit.evolveFP (``_mleW{m}_chi2W{c}``).
_FIT_WEIGHT_SUFFIX_RE = re.compile(
    r"_mleW(?P<m>[-+0-9.eE]+)_chi2W(?P<c>[-+0-9.eE]+)$")


def _parse_fit_filename(stem):
    """``(fit_mode, mle_weight, chi2_weight)`` from a result-file stem; weights
    default to ``(1.0, 0.0)`` when the joint suffix is absent (pure MLE/chisq)."""
    fit_mode = stem.split("_", 1)[0]
    m = _FIT_WEIGHT_SUFFIX_RE.search(stem)
    if m:
        return fit_mode, float(m.group("m")), float(m.group("c"))
    return fit_mode, 1.0, 0.0


def discover_saved_fits(subject, results_dir="data/RLModel"):
    """List the saved fits available on disk for ``subject``.

    Scans ``results_dir`` for ``mle_*`` / ``chisq_*`` ``{subject: payload}``
    pickles and returns one dict per fit that has an entry for the subject:
    ``{label, fit_mode, model_file, drift_fn, bias_fn, noise_fn, mle_weight,
    chi2_weight, save_time, path}``. Model identity / weights come from the
    saved ``model_config`` when present (MLE), with the filename as fallback.

    Drives a 'saved fit' selector in the GUI: the user picks among the
    (model x weight) combinations that actually exist on disk rather than
    hand-configuring widgets and hoping a matching file was fit. Sorted by
    fit_mode, model, then weights for a stable dropdown order."""
    fits = []
    results_path = pathlib.Path(results_dir)
    if not results_path.exists():
        return fits
    for fp in sorted(results_path.glob("*.pkl")):
        if not (fp.name.startswith("mle_") or fp.name.startswith("chisq_")):
            continue
        try:
            with open(fp, "rb") as f:
                data = pickle.load(f)
        except Exception:  # noqa: BLE001 — skip any unreadable pickle
            continue
        if not isinstance(data, dict) or subject not in data:
            continue
        payload = data[subject]
        cfg = payload.get("model_config") if isinstance(payload, dict) else None
        fit_mode, w_mle, w_chi2 = _parse_fit_filename(fp.stem)
        if cfg is not None:
            w_mle = float(getattr(cfg, "mle_mle_weight", w_mle))
            w_chi2 = float(getattr(cfg, "mle_chi2_weight", w_chi2))
        drift = getattr(cfg, "drift_fn_str", None)
        bias = getattr(cfg, "bias_fn_str", None)
        noise = getattr(cfg, "noise_fn_str", None)
        save_time = (payload.get("fit_finish_time")
                     if isinstance(payload, dict) else None)
        model_desc = (f"{drift}/{bias}/{noise}" if drift is not None
                      else fp.stem)
        if w_chi2 and w_chi2 > 0.0:
            label = f"{fit_mode} (mleW={w_mle:g}, chi2W={w_chi2:g}) | {model_desc}"
        else:
            label = f"{fit_mode} | {model_desc}"
        if save_time:
            label += f"  [{save_time}]"
        fits.append(dict(
            label=label, fit_mode=fit_mode, model_file=fp.stem,
            drift_fn=drift, bias_fn=bias, noise_fn=noise,
            mle_weight=w_mle, chi2_weight=w_chi2, save_time=save_time,
            path=str(fp)))
    fits.sort(key=lambda d: (d["fit_mode"], d["model_file"],
                             d["mle_weight"], d["chi2_weight"]))
    return fits


def _evaluate_mle_loss_for_gui(df, params, driftFn_str, biasFn_str, noiseFn_str,
                               include_Q, include_RewardRate, dt, t_dur,
                               fit_entry=None, terminal_c_override=None,
                               uses_asym_q=False, uses_asym_rr=False,
                               scale_bound=False):
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
    # Choice/RT-weight loss settings come from the saved fit's config so
    # the GUI's "Run MLE" loss matches the objective the fit was scored
    # under. Old pickles (no such config / fields) fall back to the
    # MLEModelConfig dataclass defaults — marginal, (1, 1) — i.e. the
    # legacy joint loss, so they render exactly as before.
    mle_choice_weight = getattr(fit_config, "mle_choice_weight", 1.0)
    mle_rt_weight = getattr(fit_config, "mle_rt_weight", 1.0)
    mle_choice_norm = getattr(fit_config, "mle_choice_norm", "marginal")
    # Mirror fit.py's gating: the explicit asym flags (sourced from
    # the GUI checkboxes by the caller) are the only signal. Combined
    # with include_Q / include_RewardRate so a checkbox ticked against
    # an incompatible model surfaces as a no-op here — the GUI
    # createWidget loop disables the checkbox itself in that case.
    uses_asymmetric_alpha = include_Q and uses_asym_q
    uses_asymmetric_beta  = include_RewardRate and uses_asym_rr
    # Mirror fit.simulateDDM's flag derivation: Bound-RewardRate drift
    # family triggers the per-trial bound rescaling; --scale-bound
    # checkbox triggers the absolute-bias / fitted-BOUND semantic.
    uses_per_trial_bound = driftFn_str.startswith("Bound-RewardRate")
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
        mle_choice_weight=float(mle_choice_weight),
        mle_rt_weight=float(mle_rt_weight),
        mle_choice_norm=str(mle_choice_norm),
        uses_asymmetric_alpha=uses_asymmetric_alpha,
        uses_asymmetric_beta=uses_asymmetric_beta,
        uses_per_trial_bound=uses_per_trial_bound,
        uses_scaled_bound=bool(scale_bound),
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
        # The notebook keys variants as ``mle_asymQ`` / ``mle_asymRR`` /
        # ``mle_asymQRR`` / ``mle_scaledB`` (and the composed
        # ``mle_asymQ_scaledB`` etc.), plus the bare ``mle`` / ``chisq``
        # symmetric fits — and the same set for chisq. Recognize any
        # key with one of those base mode prefixes followed by an
        # optional suffix; ``_preferred_modes_for`` decides which
        # specific variant the GUI is asking for.
        recognized = {k for k in entry.keys()
                      if k in ("mle", "chisq")
                      or k.startswith("mle_")
                      or k.startswith("chisq_")}
        if recognized:
            return {mode: entry[mode] for mode in recognized}
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
    """Build a human-readable model name from registry key strings.

    Inputs are the BIAS_FN_DICT / DRIFT_FN_DICT keys ("Classic",
    "NoiseGain-RewardRate-asym", "Q-Val (Offset)", "None_", …), not
    function ``__name__``s. The earlier ``__name__``-based implementation
    is dead since the GUI was switched to registry-key widget values.
    """
    if biasFn_str == "None_":
        bias_label = "No Bias, z=0"
    elif biasFn_str == "Q-Val":
        bias_label = "Init Q-Value"
    elif biasFn_str == "Q-Val (Offset)":
        bias_label = "Init Q-Value (Offset)"
    elif biasFn_str == "Q-Val-asym (Offset)":
        bias_label = "Init Q-Value (Offset, asym α)"
    else:
        bias_label = biasFn_str

    if driftFn_str == "Classic":
        drift_label = "Classic DDM"
    elif driftFn_str.startswith("Decay Q"):
        drift_label = f"Classic DDM + Decaying Q ({driftFn_str})"
    elif driftFn_str == "NoiseGain-RewardRate":
        drift_label = "Noise*RewardRate"
    elif driftFn_str == "NoiseGain-RewardRate-asym":
        drift_label = "Noise*RewardRate (asym β)"
    elif driftFn_str.startswith("NoiseGain-RewardRate"):
        drift_label = f"Noise*RewardRate + Decaying Q ({driftFn_str})"
    else:
        drift_label = driftFn_str
    return f"{drift_label} + {bias_label}"
