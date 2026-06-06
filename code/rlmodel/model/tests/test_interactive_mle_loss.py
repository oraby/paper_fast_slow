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
