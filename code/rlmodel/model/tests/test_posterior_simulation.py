"""Smoke tests for posterior_simulate.

Verifies that:
- ``simulate_from_fitted_params`` runs end-to-end on a tiny synthetic df.
- Output dataframe has chisq-compatible Sim* columns.
- Q_* / RewardRate columns appear iff the variant uses them.
- ``fitted_params_from_pickle`` correctly extracts UPPERCASE-keyed params.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ..mle import MLEModelConfig
from ..posterior_simulate import (
    fitted_params_from_pickle,
    simulate_from_fitted_params,
    simulate_from_result_pickle,
)
from ..util import initDF


N_TRIALS = 12


def _make_synthetic_df(n_trials=N_TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Name": ["test_subj"] * n_trials,
        "Date": [pd.Timestamp("2024-01-01")] * n_trials,
        "SessionNum": [1] * n_trials,
        "TrialNumber": list(range(1, n_trials + 1)),
        "DV": rng.uniform(-0.5, 0.5, n_trials).astype(np.float32),
        "DVstr": ["test"] * n_trials,
        "valid": [True] * n_trials,
        "calcStimulusTime": rng.uniform(0.3, 1.0, n_trials).astype(np.float32),
        "ChoiceCorrect": rng.choice([0.0, 1.0], n_trials).astype(np.float32),
        "ChoiceLeft": rng.choice([0.0, 1.0], n_trials).astype(np.float32),
    })
    return df


def _common_params():
    return {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.1,
        "BIAS_COEF": 0.5,
        "BIAS_FIXED": 0.0,
        "ALPHA": 0.3,
        "BETA": 0.3,
        "Q_VAL_OFFSET": 0.0,
        "Q_VAL_COEF": 5.0,
        "Q_VAL_DECAY_RATE": 1.0,
        "BIAS_MU": 0.0,
        "BIAS_SIGMA": 0.1,
    }


SIM_COLS = ["SimRT", "SimStartingPoint", "SimChoiceCorrect", "SimChoiceLeft"]


@pytest.mark.parametrize("drift,bias,include_Q,include_RR", [
    ("Classic", "None_", False, False),
    ("Classic", "Q-Val", True, False),
    ("NoiseGain-RewardRate", "None_", False, True),
    ("NoiseGain-RewardRate", "Q-Val", True, True),
])
def test_sim_from_fitted_params_chisq_compatible_columns(
        drift, bias, include_Q, include_RR):
    df = initDF(_make_synthetic_df(), include_Q=include_Q,
                include_RewardRate=include_RR)
    model_config = MLEModelConfig(
        drift_fn_str=drift, bias_fn_str=bias, noise_fn_str="Normal(0, 1)",
        include_Q=include_Q, include_RewardRate=include_RR,
        dt=0.01, t_dur=1.0, dx=0.05,
    )
    result = simulate_from_fitted_params(
        df, _common_params(), model_config, n_repeats=1, seed=0)
    sim_df = result["sim_df"]

    for col in SIM_COLS:
        assert col in sim_df.columns, (
            f"missing {col} for drift={drift}, bias={bias}")
    if include_Q:
        for col in ("Q_L", "Q_R", "Q_val"):
            assert col in sim_df.columns, f"missing {col} (include_Q={include_Q})"
    if include_RR:
        assert "RewardRate" in sim_df.columns, (
            "missing RewardRate (include_RewardRate=True)")
    assert result["mode"] == "simulated_history"
    assert result["noise_dt_scaling"] == "sqrt_dt"
    assert len(sim_df) == len(df)


def test_n_repeats_produces_stacked_rows():
    df = initDF(_make_synthetic_df(), include_Q=False, include_RewardRate=False)
    model_config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=1.0, dx=0.05,
    )
    result = simulate_from_fitted_params(
        df, _common_params(), model_config, n_repeats=3, seed=42)
    sim_df = result["sim_df"]

    assert len(sim_df) == 3 * len(df)
    assert "Repeat" in sim_df.columns
    assert set(sim_df["Repeat"].unique()) == {0, 1, 2}


def test_observed_history_mode_runs_and_has_sim_columns():
    df = initDF(_make_synthetic_df(n_trials=6),
                include_Q=True, include_RewardRate=False)
    model_config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="Q-Val",
        noise_fn_str="Normal(0, 1)",
        include_Q=True, include_RewardRate=False,
        dt=0.01, t_dur=1.0, dx=0.05,
    )
    result = simulate_from_fitted_params(
        df, _common_params(), model_config, n_repeats=1, seed=0,
        use_observed_history_for_inputs=True,
    )
    sim_df = result["sim_df"]
    for col in SIM_COLS:
        assert col in sim_df.columns
    assert result["mode"] == "observed_history"


def test_fitted_params_from_pickle_uppercases_keys():
    class _MockOptim:
        def __init__(self, x):
            self.x = np.asarray(x, dtype=float)

    pickle_dict = {
        "OptimRes": _MockOptim([0.3, 1.5]),
        "params_names": np.array(["alpha", "drift_coef"]),
        "params_init": np.array([0.1, 1.0]),
    }
    params = fitted_params_from_pickle(pickle_dict)
    assert params == {"ALPHA": 0.3, "DRIFT_COEF": 1.5}


def test_fitted_params_from_pickle_dry_run_uses_init():
    pickle_dict = {
        "OptimRes": None,
        "params_names": np.array(["alpha", "drift_coef"]),
        "params_init": np.array([0.2, 0.9]),
    }
    params = fitted_params_from_pickle(pickle_dict)
    assert params == {"ALPHA": 0.2, "DRIFT_COEF": 0.9}


def test_simulate_from_result_pickle_round_trip():
    df = initDF(_make_synthetic_df(), include_Q=False, include_RewardRate=False)
    model_config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=1.0, dx=0.05,
    )

    class _MockOptim:
        def __init__(self, x):
            self.x = np.asarray(x, dtype=float)

    pickle_dict = {
        "OptimRes": _MockOptim([1.0, 1.0, 1.0, 0.1]),
        "params_names": np.array(
            ["drift_coef", "noise_sigma", "bound", "non_decision_time"]),
        "params_init": np.array([1.0, 1.0, 1.0, 0.1]),
        "subject_df": df,
        "model_config": model_config,
        "fit_mode": "mle",
    }
    result = simulate_from_result_pickle(pickle_dict, n_repeats=1, seed=0)
    assert "SimRT" in result["sim_df"].columns
    assert result["mode"] == "simulated_history"
