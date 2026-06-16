import numpy as np

from ..mle import MLEModelConfig, estimate_population_settings, evaluate_neg_loglik
from ..mle_batch import (
    batched_choice_rt_loglik,
    estimate_flat_trial_capacity_for_memory,
    _prepare_mu,
)
from .test_mle_smoke import _params_for, _small_df


def _config(drift_name, bias_name, noise_name, include_q, include_rr,
            *, batched, backend="numpy"):
    # mle_terminal_c=0.99 keeps the batched path's no-choice likelihood
    # equal to the full survival mass (every bin center on the dx=0.1
    # bound=1.0 grid satisfies |x| <= 0.95 < 0.99), which matches the
    # rowwise reference path (which doesn't implement terminal_c).
    # Without this, the new default terminal_c=0.0 would floor no-choice
    # likelihoods in batched but not in rowwise — see
    # rlmodel/model/mle_terminal_c_plan.md.
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
        mle_terminal_c=0.99,
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


def test_prepare_mu_keeps_constant_mu_1d():
    mu = np.array([0.2, 0.4, 0.4])

    prepared, is_constant = _prepare_mu(mu, n_trials=3, n_t=5)

    assert is_constant
    assert prepared.shape == (3,)
    np.testing.assert_allclose(prepared, mu)


def test_constant_mu_fast_path_matches_explicit_time_matrix():
    observed_choice_left = np.array([1.0, 0.0, np.nan, 1.0])
    observed_rt = np.array([0.08, 0.10, np.nan, 0.16])
    no_choice = np.array([False, False, True, False])
    valid_for_loss = np.array([True, True, True, True])
    z = np.array([0.0, 0.1, -0.1, 0.0])
    mu = np.array([0.2, 0.2, -0.1, -0.1])
    sigma = np.array([1.0, 1.0, 1.2, 1.2])
    non_decision_time = np.array([0.02, 0.02, 0.02, 0.02])

    constant = batched_choice_rt_loglik(
        observed_choice_left, observed_rt, no_choice, valid_for_loss,
        z, mu, sigma, 1.0, non_decision_time, 0.01, 0.1, 0.2,
    )
    time_matrix = batched_choice_rt_loglik(
        observed_choice_left, observed_rt, no_choice, valid_for_loss,
        z, np.repeat(mu[:, None], 20, axis=1), sigma, 1.0,
        non_decision_time, 0.01, 0.1, 0.2,
    )

    assert constant.metadata["mu_is_constant"]
    assert not time_matrix.metadata["mu_is_constant"]
    assert constant.metadata["bucket_count"] == 2 * 20
    assert constant.metadata["kernel_cache_count"] == 2
    assert constant.metadata["kernel_cache_hits"] == 2 * 20
    assert time_matrix.metadata["kernel_cache_count"] == 0
    assert time_matrix.metadata["kernel_cache_hits"] == 0
    np.testing.assert_allclose(
        constant.loglik, time_matrix.loglik, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        constant.choice_prob_or_density,
        time_matrix.choice_prob_or_density,
        rtol=1e-10,
        atol=1e-12,
    )


def test_constant_mu_bucket_building_does_not_use_numpy_unique(monkeypatch):
    def fail_unique(*_args, **_kwargs):
        raise AssertionError("constant-mu bucketing should use the array backend")

    monkeypatch.setattr(np, "unique", fail_unique)
    result = batched_choice_rt_loglik(
        observed_choice_left=np.array([1.0, 0.0, 1.0]),
        observed_rt=np.array([0.08, 0.10, 0.16]),
        no_choice=np.array([False, False, False]),
        valid_for_loss=np.array([True, True, True]),
        z=np.array([0.0, 0.1, 0.0]),
        mu_values=np.array([0.2, 0.2, -0.1]),
        sigma=np.array([1.0, 1.0, 1.2]),
        bound=1.0,
        non_decision_time=np.array([0.02, 0.02, 0.02]),
        dt=0.01,
        dx=0.1,
        tmax=0.2,
    )

    assert result.metadata["mu_is_constant"]
    assert result.metadata["kernel_cache_count"] == 2
    assert np.all(np.isfinite(result.loglik))


def test_constant_mu_bucket_lexsort_uses_stacked_keys(monkeypatch):
    original_lexsort = np.lexsort

    def strict_lexsort(keys, *args, **kwargs):
        assert not isinstance(keys, tuple)
        assert getattr(keys, "ndim", None) == 2
        return original_lexsort(keys, *args, **kwargs)

    monkeypatch.setattr(np, "lexsort", strict_lexsort)
    result = batched_choice_rt_loglik(
        observed_choice_left=np.array([1.0, 0.0, 1.0]),
        observed_rt=np.array([0.08, 0.10, 0.16]),
        no_choice=np.array([False, False, False]),
        valid_for_loss=np.array([True, True, True]),
        z=np.array([0.0, 0.1, 0.0]),
        mu_values=np.array([0.2, 0.2, -0.1]),
        sigma=np.array([1.0, 1.0, 1.2]),
        bound=1.0,
        non_decision_time=np.array([0.02, 0.02, 0.02]),
        dt=0.01,
        dx=0.1,
        tmax=0.2,
    )

    assert result.metadata["kernel_cache_count"] == 2
    assert np.all(np.isfinite(result.loglik))


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


def _tiny_memory_config(min_pop):
    """Squeeze the memory budget down so the population floor binds.

    ``mle_gpu_memory_gb=1e-6`` gives a flat-trial capacity of 1, so
    ``target_candidates`` is also 1 and the floor is the only thing
    determining ``actual_candidates``.
    """
    return MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=0.2, dx=0.1,
        mle_gpu_memory_gb=1e-6,
        mle_min_population_candidates=min_pop,
    )


def test_estimate_population_settings_honors_mle_model_config_floor():
    """``MLEModelConfig.mle_min_population_candidates`` overrides the
    default floor; the CLI threads ``--mle-min-population`` here."""
    # Lower floor (8) -> actual_candidates rises only to 8.
    info_low = estimate_population_settings(
        _tiny_memory_config(min_pop=8), n_trials=1000, n_params=4, bound=1.0)
    assert info_low["actual_candidates"] == 8
    assert info_low["scipy_popsize"] == 2  # ceil(8 / 4)

    # Higher floor (128) -> actual_candidates rises to 128 even though
    # the memory budget would have picked a much smaller batch.
    info_high = estimate_population_settings(
        _tiny_memory_config(min_pop=128), n_trials=1000, n_params=4, bound=1.0)
    assert info_high["actual_candidates"] == 128
    assert info_high["scipy_popsize"] == 32  # ceil(128 / 4)


def test_estimate_population_settings_default_floor_matches_module_constant():
    """When no override is passed, the floor falls back to
    ``MIN_POPULATION_CANDIDATES`` (64)."""
    from ..mle import MIN_POPULATION_CANDIDATES

    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=0.2, dx=0.1,
        mle_gpu_memory_gb=1e-6,
    )
    info = estimate_population_settings(
        config, n_trials=1000, n_params=4, bound=1.0)
    assert info["actual_candidates"] == MIN_POPULATION_CANDIDATES


def test_validate_mle_config_rejects_nonpositive_min_population():
    """Floor must be a positive int; zero / negative is a config bug."""
    import pytest

    from ..mle import validate_mle_config

    config = MLEModelConfig(
        drift_fn_str="Classic", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)",
        include_Q=False, include_RewardRate=False,
        dt=0.01, t_dur=0.2, dx=0.1,
        mle_min_population_candidates=0,
    )
    with pytest.raises(ValueError, match="mle_min_population_candidates"):
        validate_mle_config(config)
