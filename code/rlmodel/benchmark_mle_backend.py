"""Manual benchmark for MLE likelihood backends.

Example:
    python -m rlmodel.benchmark_mle_backend --trials 512
"""
import argparse
import time

import numpy as np
import pandas as pd

from .model.array_backend import resolve_array_backend
from .model.mle import MLEModelConfig, evaluate_neg_loglik, prepare_mle_data


def _synthetic_df(n_trials):
    rng = np.random.default_rng(123)
    return pd.DataFrame({
        "Name": "S1",
        "Date": pd.Timestamp("2026-01-01"),
        "SessionNum": 1,
        "TrialNumber": np.arange(1, n_trials + 1),
        "SessId": "S1_2026-01-01_1",
        "DV": rng.normal(0.0, 0.7, size=n_trials),
        "DVstr": "synthetic",
        "valid": True,
        "calcStimulusTime": rng.uniform(0.08, 0.35, size=n_trials),
        "ChoiceLeft": rng.integers(0, 2, size=n_trials).astype(float),
        "ChoiceCorrect": rng.integers(0, 2, size=n_trials).astype(float),
    })


def _config(args, backend, batched):
    return MLEModelConfig(
        drift_fn_str=args.drift,
        bias_fn_str=args.bias,
        noise_fn_str=args.noise,
        include_Q=args.bias != "None_" or "Q" in args.drift or "Q" in args.noise,
        include_RewardRate="RewardRate" in args.drift or "RewardRate" in args.noise,
        dt=args.dt,
        t_dur=args.t_dur,
        dx=args.dx,
        mle_array_backend=backend,
        mle_device_id=args.device_id,
        mle_cupy_fallback=args.cupy_fallback,
        mle_batch_size=args.batch_size,
        mle_gpu_memory_gb=args.gpu_memory_gb,
        mle_use_batched_likelihood=batched,
    )


def _params(config):
    params = {
        "DRIFT_COEF": 1.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
    }
    if config.include_Q:
        params.update({
            "ALPHA": 0.3,
            "BIAS_COEF": 0.5,
            "Q_VAL_OFFSET": 0.0,
            "Q_VAL_COEF": 0.5,
            "Q_VAL_DECAY_RATE": 1.0,
        })
    if config.include_RewardRate:
        params["BETA"] = 0.3
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--cupy-fallback", choices=["numpy", "error"],
                        default="error")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gpu-memory-gb", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dx", type=float, default=0.1)
    parser.add_argument("--t-dur", type=float, default=0.5)
    parser.add_argument("--drift", default="Classic")
    parser.add_argument("--bias", default="None_")
    parser.add_argument("--noise", default="Normal(0, 1)")
    args = parser.parse_args()

    df = _synthetic_df(args.trials)
    prep_start = time.perf_counter()
    prepared = prepare_mle_data(df)
    prep_elapsed = time.perf_counter() - prep_start

    scenarios = [
        ("rowwise_reference", "numpy", False),
        ("batched_numpy", "numpy", True),
    ]
    try:
        resolve_array_backend("cupy", args.device_id, args.cupy_fallback)
        scenarios.append(("batched_cupy", "cupy", True))
    except RuntimeError as exc:
        print(f"batched_cupy: skipped ({exc})")

    for label, backend, batched in scenarios:
        config = _config(args, backend, batched)
        params = _params(config)
        start = time.perf_counter()
        result = evaluate_neg_loglik(params, prepared, config, return_df=False)
        total_elapsed = time.perf_counter() - start
        info = result.backend_info
        print(
            f"{label}: backend={info['actual_backend']} "
            f"evaluator={info['likelihood_evaluator']} "
            f"trials={result.n_trials_loss} neg_loglik={result.neg_loglik:.6g} "
            f"prep_seconds={prep_elapsed:.4f} objective_seconds={total_elapsed:.4f} "
            f"batch_size={info.get('batch_size')}"
        )


if __name__ == "__main__":
    main()
