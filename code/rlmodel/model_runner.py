from .model.initvals import NUM_CPUS, InitVal, InitVals, MLE_TERMINAL_C, DT, T_dur
from .model import fit
from .model.mle import MIN_POPULATION_CANDIDATES
from .model.drift import (
    DRIFT_FN_DICT, is_rewardrate_alias, resolve_drift_alias,
    user_facing_drift_keys)
from .model.state_updates import DEFAULT_RR_DRIFT_MAP, RR_DRIFT_MAPS
from .model.bias import BIAS_FN_DICT
from .model.noise import NOISE_FN_DICT
import numpy as np
import pandas as pd
import argparse
import pathlib
import pickle


DF_FP = "data/behavior/df_behavior.pkl"


def _concatExtraDFs(df, extra_dfs):
    """Concatenate supplemental behavior dataframe(s) onto ``df``.

    ``extra_dfs`` is a single item or an iterable of items, each either an
    already-loaded ``pd.DataFrame`` or a path to a pickle that
    ``pd.read_pickle`` can open (e.g. ``data/behavior/df_human_subjects.pkl``).
    Falsy (``None`` / ``[]``) is a no-op returning ``df`` unchanged.

    Every extra must carry **at least** every column of ``df``: a missing
    column would silently concat as all-NaN and only surface hours into a
    fit (or, for ``valid`` / ``ChoiceLeft``, quietly change what gets
    fitted), so it raises ``ValueError`` naming the missing columns.
    Columns the extra has in addition are kept (NaN for the main df's
    rows) and reported, since ``_reduceDFSize`` may later drop them.

    Called at the TOP of ``loadDF`` so the supplemental trials go through
    the identical EarlyWithdrawal handling, ``calcStimulusTime > T_dur``
    nullification, ``min_valid_trials`` subject filtering and
    Name/Date/SessionNum/TrialNumber sort as the main dataframe.
    """
    if extra_dfs is None:
        return df
    if isinstance(extra_dfs, (pd.DataFrame, str, pathlib.Path)):
        extra_dfs = [extra_dfs]
    extra_dfs = list(extra_dfs)
    if not extra_dfs:
        return df

    main_cols = set(df.columns)
    loaded = []
    for extra in extra_dfs:
        if isinstance(extra, pd.DataFrame):
            label, df_extra = "<DataFrame>", extra
        else:
            label = str(extra)
            if not pathlib.Path(extra).exists():
                raise ValueError(f"--extra-df file not found: {label}")
            df_extra = pd.read_pickle(extra)
            if not isinstance(df_extra, pd.DataFrame):
                raise ValueError(
                    f"--extra-df {label}: pickle holds a "
                    f"{type(df_extra).__name__}, expected a DataFrame")
        missing = sorted(main_cols - set(df_extra.columns))
        if missing:
            raise ValueError(
                f"--extra-df {label}: missing {len(missing)} column(s) "
                f"present in {DF_FP}: {missing}")
        added = sorted(set(df_extra.columns) - main_cols)
        overlap = sorted(set(df_extra["Name"].unique())
                         & set(df["Name"].unique()))
        print(f"Extra df {label}: +{len(df_extra):,} trials, "
              f"{df_extra['Name'].nunique()} subject(s)"
              + (f", extra columns (NaN for the main df): {added}"
                 if added else ""))
        if overlap:
            print(f"  WARNING: subject name(s) already in {DF_FP}: {overlap} "
                  f"— their trials will be merged into one fit per name")
        loaded.append(df_extra)

    df = pd.concat([df] + loaded, ignore_index=True)
    print(f"Combined dataframe: {len(df):,} trials, "
          f"{df['Name'].nunique()} subjects")
    return df


def loadDF(min_valid_trials=0, accepts_subjects=[], df_fp=DF_FP,
           extra_dfs=None):
    df_behavior = pd.read_pickle(df_fp)
    # Supplemental dataframe(s) join BEFORE any cleaning so they get the
    # exact same treatment as the main one (see _concatExtraDFs).
    df_behavior = _concatExtraDFs(df_behavior, extra_dfs)
    if "EarlyWithdrawal" in df_behavior.columns:
        df_ewd_mask = df_behavior.EarlyWithdrawal == 1
        # df_ewd_index = df_behavior[df_ewd_mask].index
        # df_ewd = df_behavior[df_ewd_mask]
        # Mark those trials where we don't know the chosen direction as invalid
        unknown_choice_ewd_mask = df_ewd_mask & df_behavior.ChoiceLeft.isnull()
        df_behavior.loc[unknown_choice_ewd_mask, "valid"] = False # Don't contribute to calculations
        # Assign them as random decisions directions
        df_behavior.loc[unknown_choice_ewd_mask, "ChoiceLeft"] = \
                        np.random.choice([0, 1],
                                         size=unknown_choice_ewd_mask.sum(),
                                         p=[0.5, 0.5])
        # Earlywithdrawl trials are never rewarded
        df_behavior.loc[df_ewd_mask, "ChoiceCorrect"] = 0
        # print(df_ewd.ChoiceCorrect.isnull().sum(), df_ewd.ChoiceLeft.notnull().sum())

    else:
        print("TODO: EarlyWithdrawal trials are not included, results may differ")

    nullify_mask = df_behavior.calcStimulusTime > T_dur
    print(f"Nullifying: {nullify_mask.sum():,}/{len(df_behavior):,} trials "
          f"with calcStimulusTime > {T_dur}s (of which "
          f"{df_behavior[nullify_mask].valid.sum():,} are valid trials)")
    #df_behavior.loc[nullify_mask, "ChoiceCorrect"] = np.nan # Treat as no choice
    #df_behavior.loc[nullify_mask, "calcStimulusTime"] = np.nan # Treat as no decision time
    null_stim_time_or_no_choice = df_behavior.calcStimulusTime.isnull() | \
                                  df_behavior.ChoiceLeft.isnull()
    nullify_mask |= null_stim_time_or_no_choice
    df_behavior.loc[nullify_mask, "valid"] = False # Don't contribute to calculations
    # Consider no choice as incorrect
    df_behavior.loc[null_stim_time_or_no_choice, "ChoiceCorrect"] = 0
    accepted_subjects_li = []
    for name, subject_df in df_behavior.groupby("Name"):
        subject_df_valid = subject_df[subject_df.valid]
        if len(subject_df_valid) < min_valid_trials and name not in accepts_subjects:
            print(f"Removing: {name} with {len(subject_df):,} trials")
            continue
        accepted_subjects_li.append(name)
    df_behavior = df_behavior[df_behavior.Name.isin(accepted_subjects_li)]
    # Make sure it"s correctly sorted
    df_behavior = df_behavior.sort_values(
                               by=["Name", "Date", "SessionNum", "TrialNumber"])
    return df_behavior


def _parse_init_val_overrides(specs):
    """Parse repeated ``--init-val NAME=MIN,MAX[,DEFAULT]`` into a dict.

    Returns ``{NAME_UPPERCASE: InitVal(Min, Max, Default)}``. Two-value form
    fills DEFAULT with the midpoint of (MIN, MAX). Raises ValueError with a
    pointed message on malformed entries so argparse can surface it.
    """
    overrides = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(
                f"--init-val expects NAME=MIN,MAX[,DEFAULT]; got {spec!r}")
        name, rhs = spec.split("=", 1)
        parts = [p.strip() for p in rhs.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError(
                f"--init-val NAME=MIN,MAX[,DEFAULT] expects 2 or 3 comma-"
                f"separated floats; got {len(parts)} in {spec!r}")
        try:
            nums = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(
                f"--init-val {spec!r}: could not parse as floats ({exc})"
            ) from None
        mn, mx = nums[0], nums[1]
        default = nums[2] if len(nums) == 3 else 0.5 * (mn + mx)
        overrides[name.strip().upper()] = InitVal(mn, mx, default)
    return overrides


def _parse_mle_conditions(raw):
    """Split a comma-separated ``--mle-conditions`` string into a tuple
    of column names. Empty / ``None`` → empty tuple (= unweighted
    legacy loss). Whitespace and trailing commas are tolerated.
    """
    if not raw:
        return ()
    return tuple(col.strip() for col in raw.split(",") if col.strip())


def _select_only_subjects(df, only_subject):
    """Restrict ``df`` to the named subjects for ``--only-subject``.

    No-op when ``only_subject`` is falsy (None / empty). Raises
    ``ValueError`` (the caller maps it to ``parser.error``) listing the
    available subject names when any requested name is absent, so a typo
    fails fast at startup instead of silently fitting nothing.
    """
    if not only_subject:
        return df
    available = set(df["Name"].unique())
    missing = [s for s in only_subject if s not in available]
    if missing:
        raise ValueError(
            f"--only-subject names not in dataset: {missing}; "
            f"available: {sorted(available)}")
    return df[df["Name"].isin(only_subject)]


def _order_by_only_subjects(df, only_subject):
    """Reorder ``df`` so subjects appear in the order given on the command
    line (``--only-subject S2 --only-subject S1`` ⇒ S2 fit before S1).

    ``simulateDDM`` derives its processing order from
    ``df.Name.unique()`` (first appearance), but ``_extendTrials`` sorts
    by ``Name`` alphabetically, so without this the subjects would be fit
    alphabetically regardless of the CLI order. A **stable** sort keyed by
    CLI position rearranges the per-subject blocks while preserving each
    subject's within-block Date/Session/Trial order, which the RL state
    propagation depends on. No-op when ``only_subject`` is falsy. Must run
    AFTER ``_extendTrials`` (whose alphabetical sort would otherwise undo
    it). Assumes ``df`` was already filtered to the named subjects.
    """
    if not only_subject:
        return df
    order = {name: i for i, name in enumerate(only_subject)}
    return df.sort_values(
        by="Name", key=lambda s: s.map(order), kind="stable")


def _resolve_drift_alias_args(args):
    """Resolve ``--drift RewardRate*`` into the canonical
    ``DRIFT_FN_DICT`` key based on ``--use-drift-rr`` / ``--drift-rr-map``
    and ``--scale-bound``. Mutates ``args.drift`` in place. No-op for
    non-alias drifts (e.g. ``Classic``, ``Decay Q``).

    ``--use-drift-rr`` takes precedence over ``--scale-bound`` when
    choosing the reward-rate channel; see ``drift.resolve_drift_alias``.
    Both new attributes are read with ``getattr`` defaults so namespaces
    built by other callers (and the alias unit tests) stay valid.

    Must run before any code that touches ``args.drift`` — in
    particular before ``_expand_asym_shorthand``'s column-based
    detection, so the substring scan sees the canonical name.
    """
    args.drift = resolve_drift_alias(
        args.drift, args.scale_bound,
        use_drift_rr=getattr(args, "use_drift_rr", False),
        drift_rr_map=getattr(args, "drift_rr_map", DEFAULT_RR_DRIFT_MAP))


def _expand_asym_shorthand(args):
    """Resolve ``--asym`` into the canonical ``--asym-q`` / ``--asym-rr``
    flags based on whether the chosen drift / bias / noise functions
    actually learn Q-values or a reward rate. Also returns the two
    detection booleans so the caller can reuse them for downstream
    validation without re-extracting the column sets.

    The column-dependency derivation matches the
    ``include_Q`` / ``include_RewardRate`` logic in
    ``fit.simulateDDM``, so a model that the fitter ignores Q on is
    likewise a no-op for the shorthand. OR-folds with explicit
    ``--asym-q`` / ``--asym-rr``; passing both is harmless.

    Mutates ``args.asym_q`` / ``args.asym_rr`` in place. Returns
    ``(learns_q, learns_rr)``.
    """
    from .model.util import (
        biasFnColsAndKwargs, driftFnColsAndKwargs, noiseFnColsAndKwargs)
    bias_cols, _ = biasFnColsAndKwargs(BIAS_FN_DICT[args.bias])
    drift_cols, _ = driftFnColsAndKwargs(DRIFT_FN_DICT[args.drift])
    noise_cols, _ = noiseFnColsAndKwargs(NOISE_FN_DICT[args.noise])
    learns_q = ("Q_val" in bias_cols or "Q_val" in drift_cols
                or "Q_val" in noise_cols)
    learns_rr = ("RewardRate" in bias_cols or "RewardRate" in drift_cols
                 or "RewardRate" in noise_cols)
    if args.asym:
        args.asym_q = args.asym_q or learns_q
        args.asym_rr = args.asym_rr or learns_rr
    return learns_q, learns_rr


def runModel(df, bias_fn_str, drift_fn_str, noise_fn_str, is_loss_no_dir,
             fit_mode, evolve_res : dict = None, num_cpus=None,
             dry_run=False, mle_array_backend="numpy", mle_device_id=None,
             mle_cupy_fallback="error", mle_gpu_memory_gb=None,
             mle_show_progress=False, mle_terminal_c=MLE_TERMINAL_C.Default,
             mle_min_population_candidates=None,
             mle_condition_columns=(),
             mle_choice_weight=1.0, mle_rt_weight=1.0,
             mle_choice_norm="conditional",
             mle_mle_weight=1.0, mle_chi2_weight=0.0,
             init_val_overrides=None,
             uses_asym_q=False, uses_asym_rr=False,
             scale_bound=False):
    biasFn = BIAS_FN_DICT[bias_fn_str]
    driftFn = DRIFT_FN_DICT[drift_fn_str]
    noiseFn = NOISE_FN_DICT[noise_fn_str]
    if evolve_res is None:
        evolve_res = {}

    init_vals = InitVals()
    if init_val_overrides:
        for name, iv in init_val_overrides.items():
            init_vals.override(name, iv)
            print(f"InitVals override: {name} -> (min={iv.Min}, "
                  f"max={iv.Max}, default={iv.Default})")
    init_vals_dict = init_vals.toDict()

    evolve_res_res = fit.simulateDDM(df,
                                     driftFn=driftFn,
                                     biasFn=biasFn,
                                     noiseFn=noiseFn,
                                     bounds_and_defaults=init_vals_dict,
                                     dt=DT, t_dur=T_dur,
                                     is_loss_no_dir=is_loss_no_dir,
                                     num_cpus=num_cpus,
                                     evolvs_res=evolve_res,
                                     fit_mode=fit_mode,
                                     dry_run=dry_run,
                                     mle_array_backend=mle_array_backend,
                                     mle_device_id=mle_device_id,
                                     mle_cupy_fallback=mle_cupy_fallback,
                                     mle_gpu_memory_gb=mle_gpu_memory_gb,
                                     mle_show_progress=mle_show_progress,
                                     mle_terminal_c=mle_terminal_c,
                                     mle_min_population_candidates=mle_min_population_candidates,
                                     mle_condition_columns=mle_condition_columns,
                                     mle_choice_weight=mle_choice_weight,
                                     mle_rt_weight=mle_rt_weight,
                                     mle_choice_norm=mle_choice_norm,
                                     mle_mle_weight=mle_mle_weight,
                                     mle_chi2_weight=mle_chi2_weight,
                                     bias_fn_str=bias_fn_str,
                                     drift_fn_str=drift_fn_str,
                                     uses_asym_q=uses_asym_q,
                                     uses_asym_rr=uses_asym_rr,
                                     scale_bound=scale_bound)
    evolve_res.update(evolve_res_res)
    return evolve_res

def main():
    # Add command line arguments to read drift_fn_str, bias_fn_str,
    # noise_fn_str, num_cpus, dry_run
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", type=str, required=True,
                        choices=user_facing_drift_keys())
    parser.add_argument("--bias", type=str, required=True,
                        choices=BIAS_FN_DICT.keys())
    parser.add_argument("--noise", type=str, #required=True,
                        choices=NOISE_FN_DICT.keys(), default="Normal(0, 1)")
    parser.add_argument("--fit-mode", type=str, required=True,
                        choices=["chisq", "mle"],
                        help="Fitting mode: 'chisq' (existing simulation + "
                             "Chi-square loss) or 'mle' (trial-by-trial "
                             "maximum likelihood). Required, no default.")
    parser.add_argument("--loss-no-dir", action="store_true", default=False,
                        help="Calculate Loss without direction")
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mle-backend", type=str, default=None,
                        choices=["CPU", "GPU"],
                        help="MLE compute backend (required when --fit-mode mle). "
                             "GPU uses CuPy/CUDA and hard-fails if CUDA is "
                             "unavailable — no silent fallback. CPU uses NumPy. "
                             "When GPU is selected, the runner also enforces "
                             "single-process / vectorized=True / "
                             "updating='deferred' in scipy DE (see fit.py).")
    parser.add_argument("--mle-device-id", type=int, default=None,
                        help="CUDA device id to use when --mle-backend GPU.")
    parser.add_argument("--mle-gpu-memory-gb", type=float, default=None,
                        help="Memory budget used to size each vectorized MLE "
                             "DE population.")
    parser.add_argument("--mle-progress", action="store_true",
                        help="Show a transient tqdm progress bar for each "
                             "vectorized MLE diffusion solve. Enabled "
                             "automatically for --mle-backend GPU.")
    parser.add_argument("--mle-no-progress", action="store_true",
                        help="Force-disable the MLE tqdm progress bar even under "
                             "--mle-backend GPU (which otherwise auto-enables "
                             "it) and --mle-progress. Use for batch / Slurm runs "
                             "whose stdout is redirected to a log file, where "
                             "tqdm's in-place \\r updates don't overwrite and "
                             "instead accumulate, bloating the log.")
    parser.add_argument(
        "--init-val", action="append", default=[],
        metavar="NAME=MIN,MAX[,DEFAULT]",
        help="Override the (min, max, default) tuple for a fittable "
             "parameter from model/initvals.py. Repeatable. NAME is matched "
             "case-insensitively. If DEFAULT is omitted, the midpoint of "
             "(MIN, MAX) is used. Example: "
             "--init-val NON_DECISION_TIME=0.0,0.4,0.1 "
             "--init-val DRIFT_COEF=0,5")
    parser.add_argument(
        "--mle-terminal-c", type=float,
        default=MLE_TERMINAL_C.Default,
        help=(
            "Terminal-time no-decision band fraction C in "
            f"[{MLE_TERMINAL_C.Min}, {MLE_TERMINAL_C.Max}]. At t=T_max, "
            "residual mass with |x| > C*B is reassigned to the closest "
            "choice; |x| <= C*B remains no-decision mass used as the "
            f"no-choice likelihood. C={MLE_TERMINAL_C.Default} (default) "
            f"forces all residual mass to a choice; "
            f"C={MLE_TERMINAL_C.Max} reproduces the legacy survival-only "
            "behavior. Only honored by the batched MLE path (which is "
            "the default)."))
    parser.add_argument(
        "--mle-min-population", type=int,
        default=MIN_POPULATION_CANDIDATES,
        help=(
            "Floor on the DE population size (actual candidates = "
            "scipy_popsize * n_params). Default "
            f"({MIN_POPULATION_CANDIDATES}) keeps the population diverse "
            "enough for DE to explore the parameter space even when the "
            "memory budget would otherwise pick a smaller batch. Lower "
            "this on tight-memory GPUs to fit the budget; raise it for "
            "harder loss landscapes that need more candidates per "
            "generation."))
    parser.add_argument(
        "--mle-conditions", type=str, default=None,
        help=(
            "Comma-separated df column names that define condition "
            "groups for a sample-balanced MLE loss. Each trial's "
            "per-trial loglik is reweighted by "
            "``total_valid / (num_groups * group_size)`` so each "
            "condition contributes equally regardless of size. "
            "Inspired by logic.calcLoss's with-direction route — "
            "e.g. ``--mle-conditions=ChoiceCorrect,ChoiceLeft``. "
            "Omit (or pass empty) for the legacy unweighted sum. "
            "Does NOT affect the saved-fit filename, so you can A/B "
            "test against an existing fit by overwriting the same "
            "pickle on disk."))
    parser.add_argument(
        "--mle-choice-weight", type=float, default=1.0,
        help=(
            "Weight on the CHOICE component of the per-trial MLE loss "
            "(w_choice in `w_choice*log P(c) + w_rt*log p(rt|c)`). "
            "Default 1.0. Raise above --mle-rt-weight to prioritize "
            "reproducing choice/side-bias phenotypes over RT shape."))
    parser.add_argument(
        "--mle-rt-weight", type=float, default=1.0,
        help=(
            "Weight on the RT (reaction-time-given-choice) component of "
            "the per-trial MLE loss (w_rt). Default 1.0. With "
            "--mle-choice-norm marginal, weights (1, 1) reproduce the "
            "legacy joint loss exactly."))
    parser.add_argument(
        "--mle-choice-norm", choices=["conditional", "marginal"],
        default="conditional",
        help=(
            "How the choice component's probability is normalized. "
            "'conditional' (DEFAULT) divides by P(L)+P(R), conditioning "
            "on a decision being made so survival/no-hit mass does not "
            "leak into the choice term (isolates side-bias from overall "
            "decisiveness). 'marginal' uses the raw bound-hit prob; with "
            "weights (1, 1) it reproduces today's exact loss. NOTE: "
            "because the default is 'conditional', a run with no weight "
            "flags already changes the objective vs the legacy loss, and "
            "since this does NOT affect the filename it overwrites an old "
            "fit at the same path (losses not comparable). Use "
            "'--mle-choice-norm marginal --mle-choice-weight 1 "
            "--mle-rt-weight 1' to reproduce the exact legacy fit."))
    parser.add_argument(
        "--mle-mle-weight", type=float, default=1.0,
        help=(
            "Outer weight on the MLE term of the joint loss "
            "`w_mle*(MLE/N) + w_chi2*(Chi2/N)`. Default 1.0. Each term is "
            "divided by its valid-trial count so the weights transfer across "
            "subjects regardless of trial count."))
    parser.add_argument(
        "--mle-chi2-weight", type=float, default=0.0,
        help=(
            "Outer weight on the generative Ratcliff-quantile Chi² term of the "
            "joint loss. Default 0.0 ⇒ pure MLE (the Chi² simulation is never "
            "run; the fit is byte-identical to today). >0 switches to the "
            "joint MLE+Chi² objective and REQUIRES --fit-mode mle. NOTE: joint "
            "mode is no longer pure MLE (AIC/BIC/standard errors don't apply) "
            "and does NOT affect the filename, so it overwrites any existing "
            "fit at the same path."))
    parser.add_argument(
        "--asym-q", action="store_true", default=False,
        help=(
            "Fit a separate ALPHA_UNREWARDED rate for Q-value updates on "
            "unrewarded / no-choice trials. Requires the selected model to "
            "actually learn Q-values (Q-Val bias or Decay-Q drift); the "
            "runner fails fast at startup otherwise."))
    parser.add_argument(
        "--asym-rr", action="store_true", default=False,
        help=(
            "Fit a separate BETA_UNREWARDED rate for reward-rate updates on "
            "unrewarded trials. Requires the model to actually learn a "
            "reward rate (RewardRate drift family)."))
    parser.add_argument(
        "--asym", action="store_true", default=False,
        help=(
            "Shorthand: enable --asym-q if the model learns Q-values "
            "(Q-Val bias / Decay-Q drift / Decaying Q-Val noise) AND/OR "
            "--asym-rr if it learns a reward rate (RewardRate drift "
            "family). No-op for models that learn neither. Composes "
            "with explicit --asym-q / --asym-rr via OR — passing both "
            "is harmless."))
    parser.add_argument(
        "--scale-bound", action="store_true", default=False,
        help=(
            "Swap which of (BOUND, NOISE_SIGMA) is the fitted scale axis. "
            "Default behavior fits NOISE_SIGMA with BOUND frozen at 1.0; "
            "--scale-bound fits BOUND in [0.3, 5.0] with NOISE_SIGMA frozen "
            "at 1.0 (see initvals.py: ``InitVals.BOUND`` / ``_NOISE_FIXED``). "
            "Implies absolute-bias semantics: the bias contribution is "
            "interpreted in absolute DDM-state units (clipped to +/-BOUND) "
            "rather than fraction-of-bound. Saved-fit filename gains the "
            "_scaledB suffix so symmetric and scale-bound fits coexist. "
            "Note: per-candidate BOUND values disable the population-batch "
            "code path, so this is slower than the symmetric default."))
    parser.add_argument(
        "--use-drift-rr", action="store_true", default=False,
        help=(
            "Route the learned reward rate to the DRIFT instead of the noise "
            "or the threshold: the per-step update becomes "
            "d += DV*V*g(r_t)*dt + S*sqrt(dt)*eps, with sigma and the bound "
            "both left flat. OVERRIDES the default reward-rate behavior "
            "(NoiseGain's sigma *= r_t without --scale-bound, "
            "Bound-RewardRate's b_t = BOUND*(2-r_t) with it). Requires an "
            "R-learning model, i.e. --drift RewardRate[ Decay Q[ (Offset)]]. "
            "Composes freely with --scale-bound, which then only picks which "
            "of (BOUND, NOISE_SIGMA) is the fitted scale axis. Saved fits are "
            "named after the resolved DriftGain-* drift, so they never "
            "collide with the noise / bound variants."))
    parser.add_argument(
        "--drift-rr-map", type=str, default=DEFAULT_RR_DRIFT_MAP,
        choices=list(RR_DRIFT_MAPS),
        help=(
            "The r -> drift-gain mapping g(r) for --use-drift-rr (ignored "
            "without it). '2-r' (default) gives d += DV*(2V - r*V): a HIGH "
            "reward rate weakens the drift, so decisions get slower and less "
            "accurate — note that is the opposite SPEED direction from the "
            "noise and bound channels, which both speed up at high r. '1+r' "
            "gives d += DV*(V + r*V), flipping it so a high reward rate is "
            "faster and more accurate. Both keep g(r) in [1, 2]."))
    parser.add_argument(
        "--extra-df", type=str, default=[], action="append", metavar="PATH",
        help=(
            f"Path to a supplemental behavior dataframe pickle to concatenate "
            f"onto {DF_FP} (repeatable: --extra-df A.pkl --extra-df B.pkl). "
            f"E.g. data/behavior/df_human_subjects.pkl. The extra trials are "
            f"merged BEFORE any cleaning, so they go through the same "
            f"EarlyWithdrawal / calcStimulusTime nullification / sorting as "
            f"the main dataframe, and their subjects are fit like any other. "
            f"Each extra must carry every column the main dataframe has (the "
            f"runner fails fast at startup otherwise); columns it has in "
            f"addition are kept as NaN for the main rows. NOTE: this does NOT "
            f"affect the saved-fit filename — new subjects land in the same "
            f"pickle alongside the existing ones."))
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--load-evolve", action="store_true")
    parser.add_argument("--only-subject", type=str, default=None,
                        action='append',
                        help=(
                            "Fit ONLY the named subject(s) (repeatable: "
                            "--only-subject S1 --only-subject S2). The "
                            "per-subject save reload-merges into the on-disk "
                            "pickle, so other subjects already saved at that "
                            "path are preserved without --load-evolve. The "
                            "named subjects are always (re)fit."))
    args = parser.parse_args()
    if args.test:
        runTest()
        return

    # --mle-backend is required when fit-mode is mle. We delay the check until
    # after parse_args so chisq users aren't forced to pass an unused flag.
    if args.fit_mode == "mle" and args.mle_backend is None:
        parser.error("--mle-backend {CPU,GPU} is required when --fit-mode mle")
    if not (MLE_TERMINAL_C.Min <= args.mle_terminal_c <= MLE_TERMINAL_C.Max):
        parser.error(
            f"--mle-terminal-c must satisfy "
            f"{MLE_TERMINAL_C.Min} <= C <= {MLE_TERMINAL_C.Max}; "
            f"got {args.mle_terminal_c}")
    if args.mle_min_population < 1:
        parser.error(
            f"--mle-min-population must be a positive integer; "
            f"got {args.mle_min_population}")
    for _wname, _wval in (("--mle-mle-weight", args.mle_mle_weight),
                          ("--mle-chi2-weight", args.mle_chi2_weight)):
        if not np.isfinite(_wval) or _wval < 0.0:
            parser.error(f"{_wname} must be a finite, non-negative number; "
                         f"got {_wval}")
    if args.mle_mle_weight == 0.0 and args.mle_chi2_weight == 0.0:
        parser.error("--mle-mle-weight and --mle-chi2-weight cannot both be 0 "
                     "(the loss would be identically zero).")
    if args.mle_chi2_weight > 0.0 and args.fit_mode != "mle":
        parser.error("--mle-chi2-weight > 0 requires --fit-mode mle (the joint "
                     "MLE+Chi² objective runs in the MLE driver).")
    # Pre-flight on --use-drift-rr / --drift-rr-map. Runs BEFORE the alias
    # resolution below, while args.drift is still the user-facing name, so
    # the error message quotes what the user actually typed.
    if args.use_drift_rr and not is_rewardrate_alias(args.drift):
        parser.error(
            f"--use-drift-rr routes the learned reward rate to the drift, so "
            f"it requires an R-learning model; got --drift {args.drift!r}. "
            f"Use one of: 'RewardRate', 'RewardRate Decay Q', "
            f"'RewardRate Decay Q (Offset)'.")
    if args.drift_rr_map != DEFAULT_RR_DRIFT_MAP and not args.use_drift_rr:
        parser.error(
            f"--drift-rr-map {args.drift_rr_map!r} only applies to the drift "
            f"reward-rate channel; add --use-drift-rr (without it the reward "
            f"rate modulates the noise or the threshold and the mapping is "
            f"unused).")
    # Resolve the RewardRate drift alias into the canonical DRIFT_FN_DICT
    # key. Must run before _expand_asym_shorthand so its column-based
    # detection sees the resolved name.
    _resolve_drift_alias_args(args)
    if args.use_drift_rr:
        print(f"--use-drift-rr: reward rate modulates the DRIFT "
              f"(g(r) = {args.drift_rr_map}); noise and threshold stay flat. "
              f"Resolved drift: {args.drift!r}")
    # Pre-flight on --asym / --asym-q / --asym-rr: --asym is a
    # shorthand that expands to the canonical flags based on what the
    # model actually learns; the explicit flags require the underlying
    # learning quantity to exist. Both branches share the same
    # column-based detection inside _expand_asym_shorthand, which
    # matches fit.simulateDDM's include_Q / include_RewardRate.
    if args.asym_q or args.asym_rr or args.asym:
        learns_q, learns_rr = _expand_asym_shorthand(args)
        if args.asym_q and not learns_q:
            parser.error(
                f"--asym-q requires a model that learns Q-values "
                f"(Q-Val bias or Decay-Q drift); got "
                f"bias={args.bias!r}, drift={args.drift!r}, "
                f"noise={args.noise!r}.")
        if args.asym_rr and not learns_rr:
            parser.error(
                f"--asym-rr requires a model that learns a reward rate "
                f"(RewardRate drift family); got "
                f"bias={args.bias!r}, drift={args.drift!r}, "
                f"noise={args.noise!r}.")
    try:
        init_val_overrides = _parse_init_val_overrides(args.init_val)
    except ValueError as exc:
        parser.error(str(exc))
    mle_condition_columns = _parse_mle_conditions(args.mle_conditions)
    if mle_condition_columns:
        print(f"--mle-conditions: balancing MLE loss across condition "
              f"groups defined by {list(mle_condition_columns)}")
    if args.fit_mode == "mle":
        print(f"MLE choice/RT loss: choice_weight={args.mle_choice_weight}, "
              f"rt_weight={args.mle_rt_weight}, "
              f"choice_norm={args.mle_choice_norm!r}"
              + ("" if (args.mle_choice_norm == "marginal"
                        and args.mle_choice_weight == 1.0
                        and args.mle_rt_weight == 1.0)
                 else "  (NOTE: differs from the legacy joint loss; "
                      "filename unchanged so this overwrites any existing "
                      "fit at the same path)"))
        if args.mle_chi2_weight > 0.0:
            print(f"Joint MLE+Chi² loss ENABLED: mle_weight="
                  f"{args.mle_mle_weight}, chi2_weight={args.mle_chi2_weight}. "
                  f"Each term is normalized by its reference loss "
                  f"(loss/ref): ref_mle = the pure-MLE (--mle-chi2-weight 0) "
                  f"fit, ref_chi2 = the --fit-mode chisq fit — run those first. "
                  f"NOTE: no longer pure MLE (AIC/BIC don't apply); the joint "
                  f"weights are encoded in the saved filename.")

    # Translate the user-facing CPU/GPU knob into the two internal flags that
    # mle.MLEModelConfig + array_backend.resolve_array_backend understand:
    #   CPU → numpy (fallback irrelevant; we never try cupy)
    #   GPU → cupy with cupy_fallback="error" (no silent fallback). This pair
    #         is what MLEModelConfig.requires_gpu detects to trigger the
    #         pre-flight assert_gpu_backend probe in fit._processSubject.
    if args.mle_backend == "GPU":
        mle_array_backend = "cupy"
        mle_cupy_fallback = "error"
    else:
        # CPU mode for MLE, or any value for chisq (unused on the chisq path).
        mle_array_backend = "numpy"
        mle_cupy_fallback = "error"
    mle_show_progress = bool((args.mle_progress or args.mle_backend == "GPU")
                             and not args.mle_no_progress)

    try:
        df_behavior = loadDF(min_valid_trials=0, extra_dfs=args.extra_df)
    except ValueError as exc:
        parser.error(str(exc))
    # --only-subject: restrict to the named subjects before the per-subject
    # extend/reduce work. Validation lists available names on a typo.
    try:
        df_behavior = _select_only_subjects(df_behavior, args.only_subject)
    except ValueError as exc:
        parser.error(str(exc))
    if args.only_subject:
        print(f"--only-subject: (re)fitting only {list(args.only_subject)}")
    df_behavior = _extendTrials(df_behavior)
    df_behavior = _reduceDFSize(df_behavior)
    # Reorder to the CLI order AFTER _extendTrials (which sorts by Name
    # alphabetically) so subjects are processed in the order specified.
    df_behavior = _order_by_only_subjects(df_behavior, args.only_subject)

    evolve_res = None
    if args.load_evolve:
        load_evolve_fp = fit.evolveFP(args.drift, args.bias, args.noise,
                                      t_dur=T_dur, dt=DT,
                                      is_loss_no_dir=args.loss_no_dir,
                                      fit_mode=args.fit_mode,
                                      uses_asym_q=args.asym_q,
                                      uses_asym_rr=args.asym_rr,
                                      uses_scaled_bound=args.scale_bound,
                                      mle_mle_weight=args.mle_mle_weight,
                                      mle_chi2_weight=args.mle_chi2_weight)
        assert load_evolve_fp.exists(), f"File not found: {load_evolve_fp}"
        with open(load_evolve_fp, "rb") as f:
            evolve_res = pickle.load(f)
    # --only-subject always (re)fits the named subjects: drop them from any
    # loaded results so simulateDDM doesn't skip them (the fit set is
    # df.Name.unique() minus evolve_res keys). The df was already filtered
    # to them above, so this single runModel call fits exactly the named
    # subjects; simulateDDM's reload-merge save preserves everyone else on
    # disk (even across parallel processes), so --load-evolve isn't required.
    if args.only_subject and evolve_res:
        for subject in args.only_subject:
            evolve_res.pop(subject, None)
    runModel(df_behavior, bias_fn_str=args.bias, drift_fn_str=args.drift,
             noise_fn_str=args.noise, num_cpus=args.num_cpus,
             dry_run=args.dry_run, evolve_res=evolve_res,
             is_loss_no_dir=args.loss_no_dir,
             fit_mode=args.fit_mode,
             mle_array_backend=mle_array_backend,
             mle_device_id=args.mle_device_id,
             mle_cupy_fallback=mle_cupy_fallback,
             mle_gpu_memory_gb=args.mle_gpu_memory_gb,
             mle_show_progress=mle_show_progress,
             mle_terminal_c=args.mle_terminal_c,
             mle_min_population_candidates=args.mle_min_population,
             mle_condition_columns=mle_condition_columns,
             mle_choice_weight=args.mle_choice_weight,
             mle_rt_weight=args.mle_rt_weight,
             mle_choice_norm=args.mle_choice_norm,
             mle_mle_weight=args.mle_mle_weight,
             mle_chi2_weight=args.mle_chi2_weight,
             init_val_overrides=init_val_overrides,
             uses_asym_q=args.asym_q,
             uses_asym_rr=args.asym_rr,
             scale_bound=args.scale_bound)



def _extendTrials(df):
    # df = df.copy()
    sess_li = []
    subjects = df.Name.unique()
    for subject in subjects:
        df_subj = df[df.Name == subject]
        max_trial = df_subj.TrialNumber.max()
        for sess_key, sess_df in df_subj.groupby(["Date", "SessionNum"]):
            sess_li.append(sess_df)
            sess_max_trial = sess_df.TrialNumber.max()
            remaining_trials = max_trial - sess_max_trial
            if remaining_trials == 0:
                continue
            # Duplicate the last trial to fill the remaining trials
            # print("Subject:", subject, "Session:", sess_key, sess_max_trial, "Remaining trials:", remaining_trials)
            # assert len(sess_df) >= remaining_trials, "Didn't implement the case where there are more remaining trials than the session has"
            remaining_cpy = sess_df.iloc[-1].copy()
            remaining_cpy["valid"] = False
            li = []
            for i in range(remaining_trials):
                remaining_cpy["TrialNumber"] = sess_max_trial + i + 1
                li.append(remaining_cpy.copy())
            remaining_df = pd.DataFrame(li)
            sess_li.append(remaining_df)

    df = pd.concat(sess_li).reset_index(drop=True)
    df = df.sort_values(["Name", "Date", "SessionNum", "TrialNumber"])
    DEBUG = False
    if DEBUG:
        for subject in subjects:
            df_subj = df[df.Name == subject]
            max_trial = df_subj.TrialNumber.max()
            print("Subject:", subject, "Max Trial:", max_trial)
            for sess_key, sess_df in df_subj.groupby(["Date", "SessionNum"]):
                print("   Subject:", subject, "Session:", sess_key, "Trials:", len(sess_df),
                    "Sess Max Trial:", sess_df.TrialNumber.max())
                sess_df = sess_df[sess_df.valid]
                print("   Max valid Trials:", sess_df.TrialNumber.max())

    return df

def _reduceDFSize(df, debuug=False):
    ''' Reduce the dataframe size by converting dtypes. This helps having faster
    multiprocessing as less data is being pickled/unpickled, as well as plotting.
    '''
    ##
    # [print("col: ", col, "dtype: ", df_behavior[col].dtype) for col in df_behavior.columns]
    drop_cols = []
    df["Name"] = df["Name"].astype("string")
    df["DVstr"] = df["DVstr"].astype("string")
    df["Date"] =  pd.to_datetime(df["Date"])
    for col in df.columns:
        if df[col].dtype == "object":
            drop_cols.append(col)
        elif df[col].dtype == np.float64:
            # print("col: ", col,)
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype != np.float32:
            if debuug:
                print("col: ", col, "dtype: ", df[col].dtype)
    if debuug:
        print("Dropping: ", drop_cols)
    df = df.drop(columns=drop_cols)

    return df


def runTest():
    df_behavior = loadDF()
    noise_fn_str = "Normal(0, 1)"
    for drift_fn_str in DRIFT_FN_DICT:
        for bias_fn_str in BIAS_FN_DICT:
            print("Drift:", drift_fn_str, "Bias:", bias_fn_str)
            runModel(df_behavior, drift_fn_str=drift_fn_str,
                     bias_fn_str=bias_fn_str, noise_fn_str=noise_fn_str,
                     num_cpus=1, dry_run=True, is_loss_no_dir=True,
                     fit_mode="chisq")
            print()


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        # Save SVG with text as text, not paths
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
    except ImportError:
        pass
    # Enable loading from relative packes
    if False and "PKG" not in globals():
        import importlib, importlib.util, sys, pathlib # https://stackoverflow.com/a/50395128/11996983
        # PKG = %pwd
        PKG = pathlib.Path(".").resolve()
        root_parent_level = 2
        root = PKG
        full_pkg = f"{root.name}"
        for _ in range(root_parent_level):
          root = root.parent
        full_pkg = f"{root.name}.{full_pkg}"
        MODULE_PATH = f"{root}{pathlib.os.path.sep}__init__.py"
        MODULE_NAME = f"{root.name}"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        __package__ = full_pkg
    main()
