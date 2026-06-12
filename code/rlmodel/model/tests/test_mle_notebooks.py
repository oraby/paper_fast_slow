import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..mle import MLEModelConfig
from ..mle_notebooks import ddm_viewer as ddm_viewer_module
from ..mle_notebooks.data import (
    flatten_mle_results,
    load_mle_population_results,
)
from ..mle_notebooks.ddm_viewer import (
    build_ddm_frame_data,
    build_ddm_trial_buffer,
    parameter_slider_specs,
)
from ..mle_notebooks.histograms import (
    HistogramFilterState,
    compute_histogram_layers,
)


def _small_df():
    rows = []
    for trial_num, choice_left, reward, rt, dv in [
        (1, 1.0, 1.0, 0.12, 0.7),
        (2, 0.0, 0.0, 0.14, -0.5),
        (3, np.nan, np.nan, np.nan, 0.2),
    ]:
        rows.append(dict(
            Name="S1",
            Date=pd.Timestamp("2026-01-01"),
            SessionNum=1,
            TrialNumber=trial_num,
            SessId="S1_2026-01-01_1",
            DV=dv,
            DVstr=str(dv),
            valid=True,
            calcStimulusTime=rt,
            ChoiceLeft=choice_left,
            ChoiceCorrect=reward,
        ))
    return pd.DataFrame(rows)


def _result_dict():
    return dict(
        fit_mode="mle",
        subject_df=_small_df(),
        mle_df=pd.DataFrame({
            "mle_Q_rel_before": [0.0, 0.2, -0.2],
            "mle_Q_left_before": [0.5, 0.6, 0.4],
            "mle_Q_right_before": [0.5, 0.4, 0.6],
            "mle_reward_rate_before": [0.5, 0.55, 0.45],
            "mle_valid_for_loss": [True, True, True],
        }),
        params_names=np.array([
            "DRIFT_COEF", "NOISE_SIGMA", "BOUND", "NON_DECISION_TIME"]),
        params_init=np.array([1.0, 1.0, 1.0, 0.02]),
        params_bounds=np.array([[0.1, 5.0], [0.1, 5.0], [0.5, 2.0], [0.0, 0.5]]),
        OptimRes=None,
        model_config=MLEModelConfig(
            drift_fn_str="Classic",
            bias_fn_str="None_",
            noise_fn_str="Normal(0, 1)",
            include_Q=False,
            include_RewardRate=False,
            dt=0.01,
            t_dur=0.2,
            dx=0.1,
            mle_terminal_c=0.99,
        ),
    )


def test_load_mle_population_results_and_flatten(tmp_path):
    result_dir = tmp_path / "RLModel"
    result_dir.mkdir()
    with (result_dir / "mle_test_model.pkl").open("wb") as f:
        pickle.dump({"S1": _result_dict()}, f)

    loaded = load_mle_population_results(result_dir, attach_posterior=False)
    flat = flatten_mle_results(loaded)

    assert len(loaded) == 1
    assert loaded[0].subject == "S1"
    assert loaded[0].model_name == "test_model"
    assert flat.shape[0] == 3
    assert {"mle_model_name", "mle_subject"}.issubset(flat.columns)
    # Asymmetric-LR flag columns surface alongside mle_terminal_c so
    # the population explorer can filter cross-fit by asym variant.
    # The test fixture's MLEModelConfig leaves them at the dataclass
    # default (False) — they still appear as a column, not as NaN.
    assert {"mle_uses_asymmetric_alpha",
            "mle_uses_asymmetric_beta"}.issubset(flat.columns)
    assert (flat["mle_uses_asymmetric_alpha"] == False).all()
    assert (flat["mle_uses_asymmetric_beta"] == False).all()


def test_histogram_filter_stack_query_and_undo():
    df = pd.DataFrame({
        "mle_Q_rel_before": [0.1, 0.2, 0.8, 0.9],
        "mle_Q_left_before": [0.1, 0.6, 0.7, 0.8],
    })
    state = HistogramFilterState(df)
    state.add_bin_filter("mle_Q_rel_before", 0.0, 0.5)
    state.set_query("mle_Q_left_before", ">= 0.5")

    selected = state.filtered_df()
    assert selected.index.tolist() == [1]

    removed = state.undo()
    assert removed is not None
    assert removed.kind == "query"
    assert state.filtered_df().index.tolist() == [0, 1]

    layers = compute_histogram_layers(
        df, state, {"Q": "mle_Q_rel_before"}, bins=[0.0, 0.5, 1.0])
    assert layers["Q"]["total"].tolist() == [2, 2]
    assert layers["Q"]["filtered"].tolist() == [2, 0]


def test_parameter_slider_specs_from_result():
    specs = parameter_slider_specs(_result_dict())
    by_name = {spec.name: spec for spec in specs}

    assert by_name["DRIFT_COEF"].value == 1.0
    assert by_name["DRIFT_COEF"].min == 0.1
    assert by_name["DRIFT_COEF"].max == 5.0
    assert by_name["NON_DECISION_TIME"].step > 0


def test_build_ddm_frame_data_single_trial():
    result = _result_dict()
    frame = build_ddm_frame_data(result, trial_index=0, current_step=3)

    assert frame.current_step == 3
    assert np.isclose(frame.current_time, frame.times[3])
    assert frame.bound == 1.0
    assert frame.requested_backend == "numpy"
    assert frame.actual_backend == "numpy"
    assert frame.terminal_c == 0.99
    assert np.isclose(
        frame.current_survival_ratio, frame.survival[3] / frame.survival[0])
    assert frame.current_state_mass.shape == frame.x_grid.shape
    assert frame.upper_density.shape == frame.times.shape
    assert frame.lower_density.shape == frame.times.shape
    np.testing.assert_allclose(
        frame.terminal_upper_mass
        + frame.terminal_lower_mass
        + frame.terminal_no_decision_mass,
        frame.survival[-1],
        rtol=1e-10,
        atol=1e-12,
    )


def test_ddm_trial_buffer_reuses_full_trial_for_step_frames():
    buffer = build_ddm_trial_buffer(_result_dict(), trial_index=0)
    early = buffer.frame_at_step(1)
    late = buffer.frame_at_step(5)

    assert early.current_step == 1
    assert late.current_step == 5
    assert late.current_survival_ratio <= early.current_survival_ratio
    assert early.upper_density is buffer.upper_density
    assert late.upper_density is buffer.upper_density
    assert early.current_state_mass.shape == buffer.x_grid.shape
    assert late.current_state_mass.shape == buffer.x_grid.shape
    assert not np.array_equal(early.current_state_mass, late.current_state_mass)


def test_build_ddm_frame_data_terminal_c_override():
    frame = build_ddm_frame_data(
        _result_dict(), trial_index=0, current_step=3, terminal_c=0.5)

    assert frame.terminal_c == 0.5
    np.testing.assert_allclose(
        frame.terminal_upper_mass
        + frame.terminal_lower_mass
        + frame.terminal_no_decision_mass,
        frame.survival[-1],
        rtol=1e-10,
        atol=1e-12,
    )


def test_terminal_no_decision_bar_width_encodes_mass():
    zero = ddm_viewer_module._terminal_no_decision_bar_width(0.0, 1.0, 0.01)
    half = ddm_viewer_module._terminal_no_decision_bar_width(0.5, 1.0, 0.01)
    full = ddm_viewer_module._terminal_no_decision_bar_width(1.0, 1.0, 0.01)

    assert zero == 0.0
    assert half == full / 2.0
    assert full > 0.0


def test_build_ddm_frame_data_cupy_request_can_fallback_to_numpy(monkeypatch):
    def fake_resolver(requested_backend, device_id, cupy_fallback):
        assert requested_backend == "cupy"
        assert device_id is None
        assert cupy_fallback == "numpy"
        return SimpleNamespace(
            requested_backend="cupy",
            actual_backend="numpy",
            device_id=None,
            warning="CuPy unavailable; using NumPy.",
        )

    monkeypatch.setattr(
        ddm_viewer_module, "resolve_array_backend", fake_resolver)

    frame = build_ddm_frame_data(
        _result_dict(), trial_index=0, current_step=1, mle_array_backend="cupy")

    assert frame.requested_backend == "cupy"
    assert frame.actual_backend == "numpy"
    assert "CuPy unavailable" in frame.backend_warning


def test_mle_population_explorer_notebook_smoke():
    nb_path = Path(__file__).parents[1] / "mle_notebooks" / "mle_population_explorer.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    first_code = next(cell for cell in nb["cells"] if cell["cell_type"] == "code")
    source = "".join(first_code["source"])
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in nb["cells"])

    assert nb["nbformat"] == 4
    assert "root_parent_level = 3" in source
    assert "PKG = %pwd" in source
    assert "ddm_backend_dropdown" in all_source
    assert "mle_array_backend=ddm_mle_backend" in all_source
