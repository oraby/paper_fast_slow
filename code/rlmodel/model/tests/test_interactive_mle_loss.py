import numpy as np
import pandas as pd
import json
from pathlib import Path

from .. import plotter, visualize


def test_loss_title_includes_chi_square_and_mle_loss():
    title = plotter._loss_title(
        "S1", num_trials=1234, chi_square_loss=12.345, mle_loss=67.89,
        mle_loss_source="fit")

    assert "Chi-Square Loss: 12.35" in title
    assert "MLE Loss: 67.89 (fit)" in title
    assert "1,234 Trials" in title


def test_loss_title_marks_missing_mle_loss_as_not_run():
    title = plotter._loss_title("S1", num_trials=1234, chi_square_loss=12.345)

    assert "Chi-Square Loss: 12.35" in title
    assert "MLE Loss: not run" in title


def test_loss_title_appends_joint_breakdown():
    info = dict(total_loss=3.0, mle_mle_weight=1.0, mle_chi2_weight=0.5,
                mle_part_loss=1.0, chi2_part_loss=4.0, ref_mle=120.0,
                ref_chi2=55.0, ref_mle_time="2026-06-01T10:00:00",
                ref_chi2_time="2026-06-02T09:00:00")
    title = plotter._loss_title("S1", num_trials=10, chi_square_loss=1.0,
                                mle_loss=2.0, mle_loss_source="fit",
                                joint_info=info)
    assert "\n" in title  # breakdown is a second title line
    assert "Joint total 3.00" in title
    assert "MLE_part 1.00" in title
    assert "Chi" in title and "_part 4.00" in title  # Chi²_part
    assert "ref_MLE 120.00 @ 2026-06-01T10:00:00" in title
    assert "ref_Chi" in title and "55.00 @ 2026-06-02T09:00:00" in title


def test_loss_title_no_joint_line_without_info():
    title = plotter._loss_title("S1", num_trials=10, chi_square_loss=1.0)
    assert "\n" not in title
    assert "Joint total" not in title


def test_joint_breakdown_from_fit_entry_reads_joint_fields():
    fit_entry = {"mle": {"result": {
        "joint_mode": True, "total_loss": 3.0, "mle_mle_weight": 1.0,
        "mle_chi2_weight": 0.5, "ref_mle": 120.0, "ref_chi2": 55.0,
        "ref_mle_time": "t1", "ref_chi2_time": "t2",
        "mle_part_loss": 1.0, "chi2_part_loss": 4.0}}}
    info = visualize._joint_breakdown_from_fit_entry(fit_entry, ("mle",))
    assert info["total_loss"] == 3.0
    assert info["ref_mle"] == 120.0
    assert info["ref_chi2_time"] == "t2"


def test_joint_breakdown_none_for_non_joint_fit():
    fit_entry = {"mle": {"result": {"neg_loglik": 100.0}}}  # no joint_mode
    assert visualize._joint_breakdown_from_fit_entry(fit_entry, ("mle",)) is None


def test_parse_fit_filename():
    assert visualize._parse_fit_filename("mle_x_3s_dt0.005") == ("mle", 1.0, 0.0)
    assert visualize._parse_fit_filename(
        "mle_x_3s_dt0.005_mleW2_chi2W0.25") == ("mle", 2.0, 0.25)
    assert visualize._parse_fit_filename("chisq_x_3s_dt0.005") == ("chisq", 1.0, 0.0)


def test_discover_saved_fits(tmp_path):
    import pickle
    from ..mle import MLEModelConfig

    def _cfg(**ov):
        base = dict(drift_fn_str="Classic", bias_fn_str="None_",
                    noise_fn_str="Normal(0, 1)", include_Q=False,
                    include_RewardRate=False, dt=0.005, t_dur=3.0)
        base.update(ov)
        return MLEModelConfig(**base)

    def _w(name, obj):
        with open(tmp_path / name, "wb") as f:
            pickle.dump(obj, f)

    _w("mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl",
       {"S1": dict(model_config=_cfg(), fit_finish_time="t-mle")})
    _w("mle_Classic_biasNone__Normal(0, 1)_3s_dt0.005_mleW1_chi2W0.5.pkl",
       {"S1": dict(model_config=_cfg(mle_mle_weight=1.0, mle_chi2_weight=0.5),
                   fit_finish_time="t-joint")})
    _w("chisq_Classic_biasNone__Normal(0, 1)_3s_dt0.005.pkl",
       {"S2": dict(fit_finish_time="t-chisq")})  # only S2 → excluded for S1

    fits = visualize.discover_saved_fits("S1", tmp_path)
    assert len(fits) == 2  # the two S1 mle files; the S2-only chisq is excluded
    joint = [f for f in fits if f["chi2_weight"] == 0.5][0]
    assert joint["fit_mode"] == "mle"
    assert joint["drift_fn"] == "Classic"
    assert joint["mle_weight"] == 1.0
    assert joint["save_time"] == "t-joint"
    assert "chi2W=0.5" in joint["label"]
    # A subject with no saved fits anywhere → empty list.
    assert visualize.discover_saved_fits("NOPE", tmp_path) == []


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


def test_preferred_modes_for_includes_joint_weight_suffix():
    class _W:
        def __init__(self, value=None):
            self.value = value
            self.disabled = False
    widgets_ = {
        "Bias Fn": _W("None_"),
        "Drift Fn": _W("Classic"),
        "Noise Fn": _W("Normal(0, 1)"),
        "Scale-How": _W("Noise"),
        "Asymmetric Q-update": _W(False),
        "Asymmetric RR-update": _W(False),
        "Joint Wt": _W(""),   # "None" / pure
    }
    # Pure (Joint Wt = None) leaves the key unchanged → back-compatible.
    assert visualize._preferred_modes_for("mle", widgets_) == ("mle",)
    # Selecting a joint weight variant appends its suffix LAST.
    widgets_["Joint Wt"].value = "_mleW1_chi2W0.5"
    assert visualize._preferred_modes_for("mle", widgets_) == (
        "mle_mleW1_chi2W0.5",)
    # Composes after scaledB (evolveFP order: asym, scaledB, then weights).
    widgets_["Scale-How"].value = "Bound"
    assert visualize._preferred_modes_for("mle", widgets_) == (
        "mle_scaledB_mleW1_chi2W0.5",)


def test_weight_mode_suffix():
    class _W:
        def __init__(self, value):
            self.value = value
    assert visualize._weight_mode_suffix(
        {"Joint Wt": _W("_mleW1_chi2W0.5")}) == "_mleW1_chi2W0.5"
    assert visualize._weight_mode_suffix({"Joint Wt": _W("")}) == ""
    assert visualize._weight_mode_suffix({}) == ""  # absent → pure


def test_discover_weight_suffixes():
    # Cache shape: t_dur -> noise -> bias -> drift -> subject -> {fit_key: entry}
    cache = {3.0: {"Normal(0, 1)": {"None_": {"Classic": {"S1": {
        "mle": {"result": {}, "params": {}},
        "chisq": {"result": {}, "params": {}},
        "mle_mleW1_chi2W0.5": {"result": {}, "params": {}},
        "mle_asymQ_mleW1_chi2W0.25": {"result": {}, "params": {}},
    }}}}}}
    opts = visualize._discover_weight_suffixes(cache)
    assert opts[0] == ("None", "")  # pure option leads
    label_for = {suffix: label for label, suffix in opts}
    assert "_mleW1_chi2W0.5" in label_for
    assert "_mleW1_chi2W0.25" in label_for
    # Pure fit_keys (mle / chisq) contribute no weight suffix.
    assert len([s for _l, s in opts if s]) == 2
    assert label_for["_mleW1_chi2W0.5"] == "mleW=1 chi2W=0.5"


def test_discover_weight_suffixes_empty():
    assert visualize._discover_weight_suffixes({}) == [("None", "")]
    assert visualize._discover_weight_suffixes(None) == [("None", "")]


def test_joint_wt_dropdown_constructs_and_feeds_suffix():
    """The actual 'Joint Wt' dropdown construction (the one createWidget path
    not covered by the pure-logic tests): discovered options build a valid
    Dropdown whose value flows back through _weight_mode_suffix."""
    import ipywidgets as widgets
    opts = visualize._discover_weight_suffixes(
        {3.0: {"Normal(0, 1)": {"None_": {"Classic": {"S1": {
            "mle": {"result": {}, "params": {}},
            "mle_mleW1_chi2W0.5": {"result": {}, "params": {}}}}}}}})
    dd = widgets.Dropdown(
        options=list(zip([l for l, _ in opts], [s for _, s in opts])),
        value=opts[0][1], description="Joint Wt")
    assert dd.value == ""  # the pure "None" option leads
    assert visualize._weight_mode_suffix({"Joint Wt": dd}) == ""
    dd.value = "_mleW1_chi2W0.5"
    assert visualize._weight_mode_suffix({"Joint Wt": dd}) == "_mleW1_chi2W0.5"


def test_every_declared_dropdown_is_placed_in_a_column():
    """Source-level invariant: every label in createWidget's
    ``drop_downs_labels`` is ``drop_down_widgets.pop(...)``-ed into exactly one
    column. createWidget asserts ``not len(drop_down_widgets)`` after building
    the layout, so a declared-but-unplaced dropdown is a hard runtime crash
    (this is exactly how the "Joint Wt" dropdown first slipped through). The GUI
    is too heavy to instantiate headlessly, so pin the invariant via the AST
    instead of running createWidget."""
    import ast

    src = Path(visualize.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "createWidget")

    declared = None
    placed = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "drop_downs_labels"
                        for t in node.targets)
                and isinstance(node.value, ast.List)):
            declared = [e.value for e in node.value.elts
                        if isinstance(e, ast.Constant)]
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pop"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "drop_down_widgets"
                and node.args and isinstance(node.args[0], ast.Constant)):
            placed.append(node.args[0].value)

    assert declared is not None, "drop_downs_labels list literal not found"
    # Every declared dropdown is placed exactly once; nothing extra is popped.
    assert sorted(placed) == sorted(declared), (
        f"declared={sorted(declared)} vs placed={sorted(placed)}")
    assert len(placed) == len(set(placed)), f"duplicate pops: {placed}"


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


def test_variant_suffix_drives_both_mode_key_and_auto_apply_detector():
    """Regression pin for the "Joint Wt changes title but not figure" bug.

    The saved-fit mode key (``_preferred_modes_for``) and updateGUI's
    variant-change detector (which gates the auto-apply that re-pulls a fit's
    params) must both derive their suffix from the SAME ``_variant_suffix``.
    The bug: the weight axis was added to the mode key but not the detector,
    so changing "Joint Wt" updated the title (mode-key path) without firing the
    auto-apply (detector path) — figure stayed stale until a manual Reset.

    Pin the contract directly: (1) the mode key is exactly
    ``base + _variant_suffix``, and (2) flipping ONLY "Joint Wt" changes
    ``_variant_suffix``, so the detector registers it as a variant change.
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
        "Joint Wt":             _W(""),
    }
    # (1) Mode key is base + the shared suffix — no independent concatenation.
    assert visualize._preferred_modes_for("mle", widgets) == (
        "mle" + visualize._variant_suffix(widgets),)
    assert visualize._preferred_modes_for("chisq", widgets) == (
        "chisq" + visualize._variant_suffix(widgets),)

    # (2) Flipping ONLY the Joint Wt dropdown moves the suffix the detector
    # compares against — so updateGUI will re-fire the auto-apply.
    before = visualize._variant_suffix(widgets)
    widgets["Joint Wt"].value = "_mleW1_chi2W0.5"
    after = visualize._variant_suffix(widgets)
    assert before != after
    assert after.endswith("_mleW1_chi2W0.5")
    # The weight axis composes AFTER asym + scaledB, matching fit.evolveFP.
    # (asym only fires on a model that actually learns Q — Classic doesn't.)
    widgets["Drift Fn"].value = "NoiseGain-RewardRate Decay Q"
    widgets["Asymmetric Q-update"].value = True
    widgets["Scale-How"].value = "Bound"
    assert visualize._variant_suffix(widgets) == "_asymQ_scaledB_mleW1_chi2W0.5"


def test_asym_both_excluded_from_auto_observed_set():
    """'Asym: Both' must NOT be in all_widgets_wo_btns.

    interactive_output observes all_widgets_wo_btns. If 'Asym: Both' were in
    that set, setting the two individual checkboxes from its callback would
    trigger 3 outHandler calls instead of 1, and every updateGUI-internal sync
    of 'Asym: Both' (which sets its .value) would trigger another re-render.
    Pin this at the source level via the AST so the exclusion can't be
    accidentally removed.
    """
    import ast

    src = Path(visualize.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "createWidget")

    # Find the all_widgets_wo_btns assignment — it must reference "Asym: Both"
    # in a negative test (i.e. the string literal must appear in the node that
    # builds all_widgets_wo_btns, as an exclusion).
    found_exclusion = False
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "all_widgets_wo_btns"
                        for t in node.targets)):
            src_seg = ast.unparse(node)
            assert "Asym: Both" in src_seg, (
                "all_widgets_wo_btns assignment must explicitly exclude 'Asym: Both'")
            found_exclusion = True
    assert found_exclusion, "all_widgets_wo_btns assignment not found in createWidget"


def test_asym_both_suppresses_intermediate_outhandler_calls():
    """When 'Asym: Both' is toggled, outHandler is called exactly once (from
    the RR-update write); the Q-update write is suppressed by the counter.

    Also verifies the reverse sync: updateGUI-style direct writes to
    'Asym: Both' (with the counter held) don't re-enter the callback.
    """

    calls = []

    # Minimal widget stubs
    class _W:
        def __init__(self, value=False, disabled=False):
            self.value = value
            self.disabled = disabled
            self._obs = []

        def observe(self, fn, names='value'):
            self._obs.append(fn)

        def _fire(self, new_val):
            old = self.value
            self.value = new_val
            if old != new_val:
                for fn in self._obs:
                    fn({'new': new_val, 'old': old, 'owner': self})

    q_cb  = _W(False)
    rr_cb = _W(False)
    asym_all_cb = _W(False)

    _asym_suppress = [0]

    def outHandler():
        if _asym_suppress[0] > 0:
            return
        calls.append('update')

    # Wire individual boxes to outHandler (simulating interactive_output)
    q_cb.observe(lambda _: outHandler(), names='value')
    rr_cb.observe(lambda _: outHandler(), names='value')

    all_widgets = {
        'Asymmetric Q-update':  q_cb,
        'Asymmetric RR-update': rr_cb,
        'Asym: Both':           asym_all_cb,
    }

    def _on_asym_all_change(change):
        if _asym_suppress[0] > 0:
            return
        new_val = change['new']
        _asym_suppress[0] += 1
        all_widgets['Asymmetric Q-update']._fire(new_val)
        _asym_suppress[0] -= 1
        all_widgets['Asymmetric RR-update']._fire(new_val)

    asym_all_cb.observe(_on_asym_all_change, names='value')

    # Toggle "Asym: Both" ON → exactly one updateGUI call
    asym_all_cb._fire(True)
    assert calls == ['update'], f"Expected 1 call, got {calls}"
    assert q_cb.value is True
    assert rr_cb.value is True

    # Simulate updateGUI syncing "Asym: Both" back (suppress prevents cascade)
    calls.clear()
    _asym_suppress[0] += 1
    asym_all_cb._fire(True)   # already True — no _obs fires (same value guard)
    _asym_suppress[0] -= 1
    assert calls == []  # no extra update triggered

    # Toggle OFF → exactly one call
    calls.clear()
    asym_all_cb._fire(False)
    assert calls == ['update']
    assert q_cb.value is False
    assert rr_cb.value is False


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
