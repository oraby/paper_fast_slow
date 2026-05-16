"""CLI entry point for posterior predictive simulation from MLE fits.

Usage (invoked as a module, mirroring model_runner.py)::

    python -m code.rlmodel.run_posterior_predictive \
        --mle-result /path/to/mle_result.pkl \
        --output /path/to/sim_result.pkl \
        --n-repeats 1 \
        --seed 0

Saves a chisq-compatible result dict (sim_df has the same Sim*/Q_*/RewardRate
columns as a chisq output) so existing non-MLE behavioral plots can be reused
unchanged.
"""
from __future__ import annotations

import argparse
import pathlib
import pickle

from .model.posterior_simulate import (
    save_posterior_result,
    simulate_from_result_pickle,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run posterior predictive simulation from an MLE result.")
    p.add_argument("--mle-result", type=pathlib.Path, required=True,
                   help="Path to a saved MLE result pickle "
                        "(from --fit-mode mle).")
    p.add_argument("--output", type=pathlib.Path, required=True,
                   help="Where to save the posterior simulation pickle.")
    p.add_argument("--n-repeats", type=int, default=1,
                   help="Number of independent simulation repeats "
                        "(each gets a distinct seed). Default 1.")
    p.add_argument("--seed", type=int, default=None,
                   help="Base RNG seed. Default: random.")
    p.add_argument("--observed-history", action="store_true",
                   help="Per-trial sampling under observed-history latents "
                        "(debug mode, not a synthetic session).")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.mle_result.exists():
        raise FileNotFoundError(f"MLE result not found: {args.mle_result}")

    with open(args.mle_result, "rb") as f:
        mle_pickle = pickle.load(f)

    if mle_pickle.get("fit_mode") != "mle":
        raise ValueError(
            f"Expected a fit_mode='mle' pickle; got "
            f"fit_mode={mle_pickle.get('fit_mode')!r}. "
            "Posterior predictive simulation is only defined for MLE fits.")

    result = simulate_from_result_pickle(
        mle_pickle,
        n_repeats=args.n_repeats,
        seed=args.seed,
        use_observed_history_for_inputs=args.observed_history,
    )
    save_posterior_result(result, args.output)
    print(f"Saved posterior simulation to {args.output}")
    print(f"  mode={result['mode']}, n_repeats={result['n_repeats']}, "
          f"seed={result['seed']}, n_rows={len(result['sim_df'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
