import numpy as np
import pandas as pd
import json
from pathlib import Path

from .. import plotter, visualize


def test_loss_title_includes_chi_square_and_mle_loss():
    title = plotter._loss_title(
        "S1", chi_square_loss=12.345, mle_loss=67.89,
        mle_loss_source="fit")

    assert "Chi-Square Loss: 12.35" in title
    assert "MLE Loss: 67.89 (fit)" in title


def test_loss_title_marks_missing_mle_loss_as_not_run():
    title = plotter._loss_title("S1", chi_square_loss=12.345)

    assert "Chi-Square Loss: 12.35" in title
    assert "MLE Loss: not run" in title


def test_stored_mle_loss_from_df_reads_attached_fit_loss():
    df = pd.DataFrame({"mle_fit_neg_loglik": [np.nan, 123.4, 123.4]})

    assert visualize._stored_mle_loss_from_df(df) == 123.4


def test_stored_mle_loss_from_fit_entry_reads_mle_result():
    fit_entry = {"mle": {"result": {"neg_loglik": 456.7}}}

    assert visualize._stored_mle_loss_from_fit_entry(fit_entry) == 456.7


def test_stored_mle_loss_from_fit_entry_walks_preferred_modes_chain():
    """Regression: the title's "MLE Loss: …" must reflect the SAME
    variant the GUI just applied. Before this fix
    ``_stored_mle_loss_from_fit_entry`` hardcoded ``mode="mle"``, so a
    user with Scale-How=Bound (preferred mode ``mle_scaledB``) re-fit
    their subject under the scaled variant — ``fit_finish_time``
    updated correctly via ``_apply_fit_defaults`` (which DOES walk
    the preferred-mode chain) but the title silently displayed the
    pre-existing symmetric ``mle`` entry's old loss. The two paths
    were inconsistent. Now both consult the same chain.
    """
    fit_entry = {
        "mle":         {"result": {"neg_loglik": 1000.0}},   # legacy
        "mle_scaledB": {"result": {"neg_loglik": 250.0}},    # the new variant
    }
    # Scale-How=Bound preferred chain: walk scaled first, then fall
    # back to symmetric mle.
    bound_chain = ("mle_scaledB", "mle", "chisq")
    assert visualize._stored_mle_loss_from_fit_entry(
        fit_entry, bound_chain) == 250.0
    # Scale-How=Noise (default) preferred chain: just mle.
    noise_chain = ("mle", "chisq")
    assert visualize._stored_mle_loss_from_fit_entry(
        fit_entry, noise_chain) == 1000.0
    # Backwards-compat: the legacy single-arg call still works and
    # behaves like the old ``("mle",)`` default.
    assert visualize._stored_mle_loss_from_fit_entry(fit_entry) == 1000.0


def test_stored_mle_loss_from_fit_entry_falls_back_when_preferred_missing():
    """A user with Asym-Q checked may not have a saved ``mle_asymQ``
    fit yet; the chain falls back to symmetric ``mle`` so the title
    still shows something useful instead of None."""
    fit_entry = {"mle": {"result": {"neg_loglik": 100.0}}}
    asym_q_chain = ("mle_asymQ", "mle", "chisq")
    assert visualize._stored_mle_loss_from_fit_entry(
        fit_entry, asym_q_chain) == 100.0
    # When NOTHING in the chain has a saved loss → None.
    asym_only_chain = ("mle_asymQ", "mle_asymRR")
    assert visualize._stored_mle_loss_from_fit_entry(
        fit_entry, asym_only_chain) is None


def test_asym_mode_suffix_ignores_asym_q_when_active_model_lacks_q_value():
    """Regression: after selecting a Q-learning model and checking
    ``Asymmetric Q-update``, switching to a model with no Q-learning
    (``Classic``) leaves the checkbox checked-but-disabled — its
    ``value`` stays True even though it's irrelevant. The variant-suffix
    lookup must IGNORE that stale tick, otherwise the preferred-mode
    chain leads with ``mle_asymQ`` which the new model can't possibly
    have on disk, and the Reset-MLE button (gated on the chain's first
    element) stays disabled even when the symmetric ``mle`` fit exists.
    """
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    # User-facing scenario: Asym-Q ticked, but active model is Classic
    # (no Q-learning) → suffix should drop _asymQ.
    classic_widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("Classic"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(True),   # stale tick from previous model
        "Asymmetric RR-update": _W(False),
    }
    assert visualize._asym_mode_suffix(classic_widgets) == ""

    # Sanity: the SAME tick on an actually-Q-learning model keeps _asymQ.
    q_widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("NoiseGain-RewardRate Decay Q"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(True),
        "Asymmetric RR-update": _W(False),
    }
    assert visualize._asym_mode_suffix(q_widgets) == "_asymQ"


def test_asym_mode_suffix_ignores_asym_rr_when_active_model_lacks_reward_rate():
    """Symmetric to the Q-value regression: a stale Asym-RR tick on a
    model that doesn't learn a reward rate (``Classic``) must drop the
    ``_asymRR`` suffix."""
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    classic_widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("Classic"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(False),
        "Asymmetric RR-update": _W(True),   # stale tick
    }
    assert visualize._asym_mode_suffix(classic_widgets) == ""

    # And on a RewardRate-learning model the tick survives.
    rr_widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("NoiseGain-RewardRate"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(False),
        "Asymmetric RR-update": _W(True),
    }
    assert visualize._asym_mode_suffix(rr_widgets) == "_asymRR"


def test_preferred_modes_chain_drops_asym_for_q_less_model():
    """End-to-end regression: when both asym ticks are stale (carried
    over from a Q+RR model) but the active model is ``Classic``, the
    preferred-mode chain reduces to the plain symmetric ``mle``.
    Without this fix the chain would lead with a composed ``mle_asymQRR``
    key that can't exist for Classic, and the Reset-MLE button gate
    would stay disabled even when ``mle`` IS on disk for the subject.
    """
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    classic_widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("Classic"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(True),
        "Asymmetric RR-update": _W(True),
    }
    chain = visualize._preferred_modes_for("mle", classic_widgets)
    # Strict matching: a single-element tuple with the exact variant —
    # no spurious _asym* prefix, no chisq last-resort fallback.
    assert chain == ("mle",), chain


def test_preferred_modes_for_is_strict_single_element():
    """The strict-matching policy: ``_preferred_modes_for`` returns a
    1-element tuple of the exact variant key for the current widget
    state. No fallback chain — auto-apply, title, and Reset gating
    paths all see strict-or-nothing.

    Previously the function returned multi-element chains like
    ``("mle_asymQ_scaledB", "mle_asymQ", "mle_scaledB", "mle",
    "chisq")`` so callers could silently load *something* even when the
    exact variant hadn't been fit. That hid surprises (the user would
    see a fit's params but it was the symmetric variant, not the asym
    one they were toggling). New policy: strict, with sliders / title
    holding their previous state when the exact match is missing.
    """
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    base_widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("NoiseGain-RewardRate Decay Q"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(False),
        "Asymmetric RR-update": _W(False),
    }
    # Plain symmetric noise-scaled.
    assert visualize._preferred_modes_for("mle",   base_widgets) == ("mle",)
    assert visualize._preferred_modes_for("chisq", base_widgets) == ("chisq",)

    # Asym-Q only on a Q-learning model.
    base_widgets["Asymmetric Q-update"].value = True
    assert visualize._preferred_modes_for("mle", base_widgets) == ("mle_asymQ",)

    # Asym-Q + Asym-RR + Scale-Bound composed suffix, all on a model
    # that supports both.
    base_widgets["Asymmetric RR-update"].value = True
    base_widgets["Scale-How"].value = "Bound"
    assert visualize._preferred_modes_for("mle", base_widgets) == (
        "mle_asymQRR_scaledB",)

    # Switching to a Q-less model with stale ticks collapses the
    # composed suffix back to the bare scaledB key — Asym-Q / Asym-RR
    # become irrelevant per ``_asym_mode_suffix``'s gating.
    base_widgets["Drift Fn"].value = "Classic"
    assert visualize._preferred_modes_for("mle", base_widgets) == (
        "mle_scaledB",)


def test_preferred_modes_for_no_mle_to_chisq_fallback():
    """Regression pin: ``_preferred_modes_for`` itself stays strict —
    asking for ``base_mode="mle"`` never returns ``chisq``. The
    cross-mode mle→chisq fallback lives only at the auto-apply callsite
    in updateGUI (see ``test_auto_apply_chain_includes_chisq_fallback``)
    so the Reset-MLE button gating / title's "MLE Loss:" reader (which
    both also call ``_preferred_modes_for("mle", …)``) stay strictly
    MLE-or-nothing.
    """
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("Classic"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(False),
        "Asymmetric RR-update": _W(False),
    }
    assert "chisq" not in visualize._preferred_modes_for("mle", widgets)


def test_auto_apply_chain_includes_chisq_fallback():
    """The auto-apply path (the subject/Drift Fn/checkbox-change
    trigger in updateGUI) augments the strict MLE chain with the strict
    chisq chain — so when the exact MLE variant has no saved fit but
    the matching chisq variant does, the sliders still get loaded
    (chisq fits tend to be more abundant than MLE re-fits). The chain
    matches ``mle_<suffix> → chisq_<suffix>`` strictly on the suffix.

    The Reset-MLE button / title's MLE Loss intentionally do NOT
    include this cross-mode fallback because their UI labels promise
    MLE; loading a chisq fit under those labels would be misleading.
    """
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False

    # Plain symmetric noise-scaled: mle → chisq.
    widgets = {
        "Bias Fn":              _W("None_"),
        "Drift Fn":             _W("Classic"),
        "Noise Fn":             _W("Normal(0, 1)"),
        "Scale-How":            _W("Noise"),
        "Asymmetric Q-update":  _W(False),
        "Asymmetric RR-update": _W(False),
    }
    auto_apply_chain = (
        visualize._preferred_modes_for("mle", widgets)
        + visualize._preferred_modes_for("chisq", widgets))
    assert auto_apply_chain == ("mle", "chisq")

    # Composed variant on a model that supports both flags: the chisq
    # fallback also gets the same suffix — strict on the suffix, not
    # bare ``chisq``.
    widgets["Drift Fn"].value = "NoiseGain-RewardRate Decay Q"
    widgets["Asymmetric Q-update"].value = True
    widgets["Asymmetric RR-update"].value = True
    widgets["Scale-How"].value = "Bound"
    auto_apply_chain = (
        visualize._preferred_modes_for("mle", widgets)
        + visualize._preferred_modes_for("chisq", widgets))
    assert auto_apply_chain == (
        "mle_asymQRR_scaledB", "chisq_asymQRR_scaledB")


def test_stored_mle_loss_from_positional_fit_entry_tuple():
    fit_entry = (
        {"params": {"DRIFT_COEF": 1.0}, "result": {"neg_loglik": 456.7}},
        {"params": {"DRIFT_COEF": 2.0}, "result": {"OptimRes": object()}},
    )

    assert visualize._stored_mle_loss_from_fit_entry(fit_entry) == 456.7


def test_get_subject_fit_entry_supports_separate_fit_modes():
    subjects_defaults = {
        3.0: {
            "_noiseNormal": {
                "_biasQVal": {
                    "_driftClassic": {
                        "S1": {
                            "mle": {"params": {"DRIFT_COEF": 1.0}},
                            "chisq": {"params": {"DRIFT_COEF": 2.0}},
                        }
                    }
                }
            }
        }
    }

    entry = visualize._get_subject_fit_entry(
        subjects_defaults, 3.0, "_noiseNormal", "_biasQVal",
        "_driftClassic", "S1")

    assert entry["mle"]["params"]["DRIFT_COEF"] == 1.0
    assert entry["chisq"]["params"]["DRIFT_COEF"] == 2.0


def test_fit_entry_for_mode_supports_positional_tuple_without_mode_names():
    mle_entry = {"params": {"DRIFT_COEF": 1.0}, "result": {"neg_loglik": 10.0}}
    chisq_entry = {"params": {"DRIFT_COEF": 2.0}, "result": {"OptimRes": object()}}
    entry = (mle_entry, chisq_entry)

    assert visualize._fit_entry_for_mode(entry, "mle") is mle_entry
    assert visualize._fit_entry_for_mode(entry, "chisq") is chisq_entry
    assert visualize._fit_entry_params(
        visualize._fit_entry_for_mode(entry, "chisq")
    )["DRIFT_COEF"] == 2.0


def test_fit_entry_for_mode_recognizes_scaledB_and_composed_variants():
    """Regression: ``_fit_entries_by_mode`` used to recognize only
    ``mle`` / ``chisq`` and ``mle_asym* / chisq_asym*``. ``--scale-bound``
    fits are keyed as ``mle_scaledB`` / ``chisq_scaledB`` (and the
    composed ``mle_asymQ_scaledB`` family); those were silently dropped
    from the recognized set, so the Reset-to-Defaults button stayed
    grayed out even when the saved fit existed on disk.
    """
    mle_scaled = {"params": {"BOUND": 1.5}}
    mle_asymQ_scaled = {"params": {"BOUND": 2.0}}
    chisq_scaled = {"params": {"BOUND": 0.8}}
    entry = {
        "mle": {"params": {"BOUND": 1.0}},
        "mle_asymQ": {"params": {"BOUND": 1.1}},
        "mle_scaledB": mle_scaled,
        "mle_asymQ_scaledB": mle_asymQ_scaled,
        "chisq_scaledB": chisq_scaled,
    }

    assert visualize._fit_entry_for_mode(entry, "mle_scaledB") is mle_scaled
    assert visualize._fit_entry_for_mode(
        entry, "mle_asymQ_scaledB") is mle_asymQ_scaled
    assert visualize._fit_entry_for_mode(
        entry, "chisq_scaledB") is chisq_scaled
    # Existing modes still resolve correctly.
    assert visualize._fit_entry_for_mode(entry, "mle") is entry["mle"]
    assert visualize._fit_entry_for_mode(entry, "mle_asymQ") is entry["mle_asymQ"]


def test_fit_entry_params_supports_plain_numpy_scalar_param_dict():
    fit_entry = {
        np.str_("DRIFT_COEF"): np.float64(1.47),
        np.str_("BOUND"): np.float64(1.0),
        np.str_("ALPHA"): np.float64(0.16),
    }

    params = visualize._fit_entry_params(fit_entry)

    assert params == {
        "DRIFT_COEF": np.float64(1.47),
        "BOUND": np.float64(1.0),
        "ALPHA": np.float64(0.16),
    }


def test_fit_entry_for_mode_supports_positional_plain_param_dicts():
    mle_params = {np.str_("DRIFT_COEF"): np.float64(1.0)}
    chisq_params = {np.str_("DRIFT_COEF"): np.float64(2.0)}
    entry = (mle_params, chisq_params)

    assert visualize._fit_entry_params(
        visualize._fit_entry_for_mode(entry, "mle")
    ) == {"DRIFT_COEF": np.float64(1.0)}
    assert visualize._fit_entry_params(
        visualize._fit_entry_for_mode(entry, "chisq")
    ) == {"DRIFT_COEF": np.float64(2.0)}


def test_fit_entry_for_mode_ignores_non_fit_tuple_payloads():
    params = {np.str_("DRIFT_COEF"): np.float64(1.0)}
    entry = (params, pd.DataFrame({"Name": ["S1"]}))

    assert visualize._fit_entry_params(
        visualize._fit_entry_for_mode(entry, "mle")
    ) == {"DRIFT_COEF": np.float64(1.0)}
    assert visualize._fit_entry_for_mode(entry, "chisq") is None


def test_apply_fit_defaults_print_renders_asctime_and_mle_conditions(capsys):
    """``_apply_fit_defaults`` prints a single line per defaults-load so
    the user can tell WHICH fit got applied. The line surfaces:
    1. The fit's finish time, rendered as ``asctime`` (e.g.
       ``Wed Jun 17 15:23:45 2026``) — much easier to scan than the
       saved ISO ``2026-06-17T15:23:45``.
    2. ``mle_condition_columns`` if the fit was condition-balanced —
       crucial when a pickle has mixed conditions across subjects
       (``--only-subject`` leaves the other subjects' fits untouched
       while the re-fit subject uses the new conditions).
    """
    from types import SimpleNamespace

    class _W:
        # Minimal widget stub with the attributes _apply_fit_defaults touches.
        def __init__(self, value=None):
            self.value = value
            self.disabled = False
            self._trait_notifiers = {"value": {"change": []}}

    all_widgets = {
        "Noise Fn": _W("Normal(0, 1)"),
        "Bias Fn": _W("None_"),
        "Drift Fn": _W("Classic"),
        # Plus a slider whose value the apply loop will try to set —
        # the print is what we assert on, not the slider update.
        "DRIFT_COEF": _W(0.0),
    }
    # Fake model_config that carries mle_condition_columns.
    saved_model_config = SimpleNamespace(
        mle_condition_columns=("ChoiceCorrect", "ChoiceLeft"),
        mle_terminal_c=0.5,  # accessed by the apply-defaults follow-up
        dx=0.02,
    )
    # Shape matches what ``assignGUISubjectsDefaults`` produces:
    # ``{<mode>: {"params": ..., "result": {... model_config ...}}}``.
    subjects_defaults = {
        3.0: {"Normal(0, 1)": {"None_": {"Classic": {"S1": {
            "mle": {
                "params": {"DRIFT_COEF": 1.23},
                "result": {
                    "model_config": saved_model_config,
                    "neg_loglik": 100.0,
                    # ``fit_finish_time`` is stored inside the result
                    # dict by ``mle.result_payload``; the GUI's apply
                    # path reads it from there. See visualize.py.
                    "fit_finish_time": "2026-06-17T15:23:45",
                },
            },
        }}}}}}

    # Reset the dedupe key so the print actually fires.
    if hasattr(visualize._apply_fit_defaults, "_last_log_key"):
        del visualize._apply_fit_defaults._last_log_key

    visualize._apply_fit_defaults(
        all_widgets, subjects_defaults, t_dur=3.0, subject="S1",
        mode="mle", required=True, quiet=False)

    out = capsys.readouterr().out
    # asctime form has the weekday and the year at the end.
    assert "Wed Jun 17 15:23:45 2026" in out, out
    # The ISO form should NOT appear (we reformatted it).
    assert "2026-06-17T15:23:45" not in out, out
    # The conditions are surfaced verbatim.
    assert "mle_conditions=ChoiceCorrect,ChoiceLeft" in out, out
    # Subject name is still there.
    assert "S1" in out, out


def test_apply_fit_defaults_print_omits_conditions_when_unweighted(capsys):
    """When the saved fit was the legacy unweighted sum
    (``mle_condition_columns=()``), the print line carries only the
    timestamp — no spurious empty ``mle_conditions=`` suffix that
    would clutter the typical case."""
    from types import SimpleNamespace

    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False
            self._trait_notifiers = {"value": {"change": []}}

    all_widgets = {
        "Noise Fn": _W("Normal(0, 1)"),
        "Bias Fn": _W("None_"),
        "Drift Fn": _W("Classic"),
        "DRIFT_COEF": _W(0.0),
    }
    saved_model_config = SimpleNamespace(
        mle_condition_columns=(), mle_terminal_c=0.5, dx=0.02)
    subjects_defaults = {
        3.0: {"Normal(0, 1)": {"None_": {"Classic": {"S2": {
            "mle": {
                "params": {"DRIFT_COEF": 1.0},
                "result": {
                    "model_config": saved_model_config,
                    "neg_loglik": 100.0,
                    "fit_finish_time": "2026-06-17T15:23:45",
                },
            },
        }}}}}}
    if hasattr(visualize._apply_fit_defaults, "_last_log_key"):
        del visualize._apply_fit_defaults._last_log_key

    visualize._apply_fit_defaults(
        all_widgets, subjects_defaults, t_dur=3.0, subject="S2",
        mode="mle", required=True, quiet=False)

    out = capsys.readouterr().out
    assert "Wed Jun 17 15:23:45 2026" in out, out
    assert "mle_conditions" not in out, out


def test_mle_params_from_widgets_includes_asym_unrewarded_params():
    """``_compute_latent_arrays`` reads ``ALPHA_UNREWARDED`` /
    ``BETA_UNREWARDED`` via strict access whenever the matching
    ``uses_asymmetric_*`` flag is True. The GUI's ``Run MLE`` button
    builds the params dict from this whitelist; if the unrewarded
    sliders aren't in the dict the MLE call fails with KeyError —
    which used to be swallowed and rendered as ``"not run (error)"``.
    Pin both keys here so a future refactor doesn't drop them again.
    """
    class _Slider:
        def __init__(self, value):
            self.value = value

    widgets = {
        "DRIFT_COEF": _Slider(1.0),
        "NOISE_SIGMA": _Slider(1.0),
        "BOUND": _Slider(1.0),
        "NON_DECISION_TIME": _Slider(0.02),
        "ALPHA": _Slider(0.3),
        "BETA": _Slider(0.4),
        "ALPHA_UNREWARDED": _Slider(0.11),
        "BETA_UNREWARDED": _Slider(0.22),
        "BIAS_COEF": _Slider(0.5),
        "Q_VAL_OFFSET": _Slider(0.0),
        "LAPSE_RATE": _Slider(0.0),
        "Drift Fn": _Slider("RewardRate"),  # non-param widget, ignored
    }

    params = visualize._mle_params_from_widgets(widgets)

    assert params["ALPHA_UNREWARDED"] == 0.11
    assert params["BETA_UNREWARDED"] == 0.22
    # Non-param widgets stay out.
    assert "Drift Fn" not in params


def test_mle_loss_key_changes_with_params():
    base = dict(
        subject="S1",
        driftFn_str="Classic",
        biasFn_str="Q-Val (Offset)",
        noiseFn_str="Normal(0, 1)",
        t_dur=3.0,
        dt=0.005,
    )

    key1 = visualize._mle_loss_key(**base, params={"DRIFT_COEF": 1.0})
    key2 = visualize._mle_loss_key(**base, params={"DRIFT_COEF": 1.1})

    assert key1 != key2


def test_interactive_notebook_loads_fit_files_without_posterior_pickles():
    nb_path = Path(__file__).parents[2] / "model_interactive.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"])

    assert "loadFitResults" in source
    assert 'FIT_MODES = ("mle", "chisq")' in source
    assert 'glob(f"{mode}_*.pkl")' in source
    assert "pp_" not in source
    assert "df=all_df" in source
