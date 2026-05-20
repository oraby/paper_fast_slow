import numpy as np

from ..mle import MLEModelConfig, estimate_population_settings, evaluate_neg_loglik
from ..mle_batch import estimate_flat_trial_capacity_for_memory
from .test_mle_smoke import _params_for, _small_df


def _config(drift_name, bias_name, noise_name, include_q, include_rr,
            *, batched, backend="numpy"):
    return MLEModelConfig(
        drift_fn_str=drift_name,
        bias_fn_str=bias_name,
        noise_fn_str=noise_name,
        include_Q=include_q,
        include_RewardRate=include_rr,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
        mle_array_backend=backend,
        mle_use_batched_likelihood=batched,
    )


def test_batched_numpy_matches_rowwise_for_constant_mu_q_bias():
    params = _params_for(
        include_q=True,
        include_reward_rate=False,
        decay_q=False,
        q_bias=True,
    )
    rowwise = evaluate_neg_loglik(
        params,
        _small_df(),
        _config("Classic", "Q-Val", "Normal(0, 1)", True, False,
                batched=False),
        return_df=True,
    )
    batched = evaluate_neg_loglik(
        params,
        _small_df(),
        _config("Classic", "Q-Val", "Normal(0, 1)", True, False,
                batched=True),
        return_df=True,
    )

    assert batched.backend_info["actual_backend"] == "numpy"
    assert batched.backend_info["likelihood_evaluator"] == "batched"
    np.testing.assert_allclose(
        batched.neg_loglik, rowwise.neg_loglik, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        batched.mle_df.mle_loglik,
        rowwise.mle_df.mle_loglik,
        rtol=1e-8,
        atol=1e-10,
    )


def test_batched_numpy_matches_rowwise_for_time_varying_mu():
    params = _params_for(
        include_q=True,
        include_reward_rate=False,
        decay_q=True,
        q_bias=False,
    )
    rowwise = evaluate_neg_loglik(
        params,
        _small_df(),
        _config("Decay Q", "None_", "Normal(0, 1)", True, False,
                batched=False),
        return_df=True,
    )
    batched = evaluate_neg_loglik(
        params,
        _small_df(),
        _config("Decay Q", "None_", "Normal(0, 1)", True, False,
                batched=True),
        return_df=True,
    )

    np.testing.assert_allclose(
        batched.neg_loglik, rowwise.neg_loglik, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        batched.mle_df.mle_loglik,
        rowwise.mle_df.mle_loglik,
        rtol=1e-8,
        atol=1e-10,
    )


def test_optional_cupy_backend_is_skipped_without_cuda():
    pytest = __import__("pytest")
    try:
        import cupy  # noqa: F401
    except Exception:
        pytest.skip("CuPy is not installed or CUDA is unavailable")

    params = _params_for(False, False)
    result = evaluate_neg_loglik(
        params,
        _small_df(),
        _config("Classic", "None_", "Normal(0, 1)", False, False,
                batched=True, backend="cupy"),
        return_df=False,
    )
    assert np.isfinite(result.neg_loglik)


def test_gpu_memory_budget_estimates_flat_trial_capacity():
    flat_capacity, estimate = estimate_flat_trial_capacity_for_memory(
        memory_gb=0.25,
        bound=1.0,
        dx=0.1,
        tmax=0.2,
        dt=0.01,
    )

    assert flat_capacity > 0
    assert estimate["requested_memory_gb"] == 0.25
    assert estimate["estimated_total_bytes"] <= int(0.25 * (1024 ** 3))


def test_memory_budget_scales_population_candidates_below_ceiling():
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False,
        include_RewardRate=False,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
        mle_gpu_memory_gb=0.01,
    )

    info = estimate_population_settings(
        config, n_trials=25, n_params=4, bound=1.0)

    assert info["actual_candidates"] <= info["target_candidates"]
    assert info["actual_candidates"] == info["scipy_popsize"] * 4
    assert info["target_candidates"] >= 1
