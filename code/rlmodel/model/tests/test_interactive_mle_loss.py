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
