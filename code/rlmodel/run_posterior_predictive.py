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
import pandas as pd

from .model.posterior_simulate import simulate_from_result_pickle


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
    p.add_argument("--population-df", type=bool, default=True,
                   help="Whether the loaded MLE result is expected to have a "
                        "population-level (True) or subject-level (False) "
                        "DataFrame. ")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.mle_result.exists():
        raise FileNotFoundError(f"MLE result not found: {args.mle_result}")

    with open(args.mle_result, "rb") as f:
        mle_pickle = pickle.load(f)
    if args.population_df:
        df_li = []
        for subject_id, mle_subject_pickle in mle_pickle.items():
            print(f"Simulating posterior predictive for subject {subject_id}...")
            result_df = _iterate_subjects(mle_subject_pickle, args)

            df_li.append(result_df)
            # Do an intermediate save of the reults
            if len(df_li) > 1:
                result_df = pd.concat(df_li, ignore_index=True).reset_index(
                                                                      drop=True)
            result_df.to_pickle(args.output)
            print(f"Saved intermediate posterior simulation to {args.output}")
        print("Finished all subjects.")
    else:
        result_df = _iterate_subjects(mle_pickle, args)
        result_df.to_pickle(args.output)
        print(f"Saved posterior simulation to {args.output}")

    return 0

def _iterate_subjects(mle_subject_pickle, args):
    """Helper to iterate over subjects in a population-level MLE result."""
    if mle_subject_pickle.get("fit_mode") != "mle":
        raise ValueError(
            f"Expected a fit_mode='mle' pickle; got "
            f"fit_mode={mle_subject_pickle.get('fit_mode')!r}. "
            "Posterior predictive simulation is only defined for MLE fits.")

    result_dict = simulate_from_result_pickle(
        mle_subject_pickle,
        n_repeats=args.n_repeats,
        seed=args.seed,
        use_observed_history_for_inputs=args.observed_history,
    )
    sim_df = result_dict.pop("sim_df")
    assign_keys = ["drift_fn_str", "bias_fn_str", "noise_fn_str", "include_Q",
                   "include_RewardRate"]
    for key in assign_keys:
        assert key not in sim_df.columns, f"Expected key {key!r} not to be in sim_df columns"
        sim_df[key] = result_dict[key]
    return sim_df

if __name__ == "__main__":
    raise SystemExit(main())
