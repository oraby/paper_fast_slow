from .bias import BIAS_FN_DICT
from .drift import DRIFT_FN_DICT
from .noise import NOISE_FN_DICT
from .logic import makeOneRun
from .array_backend import assert_gpu_backend, resolve_array_backend
from .initvals import MLE_TERMINAL_C
from .mle import (
    MIN_POPULATION_CANDIDATES,
    MLEModelConfig,
    estimate_population_settings,
    objective_from_population,
    prepare_mle_data,
    result_payload,
)
from .util import initDF, driftFnColsAndKwargs, biasFnColsAndKwargs, noiseFnColsAndKwargs
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from functools import partial
import inspect
import datetime
import multiprocessing
import os
import pathlib
import pickle



# Discover makeOneRun's full kwarg surface — needed by the downstream
# index-bookkeeping in simulateDDM, which subsets BOTH the fit-param
# vector AND the fixed_params dict against this set. Discovery (not
# fit-eligibility) is the job here; flag-driven fittability lives in
# the ``_PARAM_FIT_GATES`` declarative table inside ``simulateDDM``.
_NON_FITTABLE_MAKEONERUN_PARAMS = {"seed", "skip_loss"}
_makeOneRun_params_names = inspect.signature(makeOneRun).parameters.keys()
_makeOneRun_params_names = np.asanyarray([
    p for p in _makeOneRun_params_names
    if p not in _NON_FITTABLE_MAKEONERUN_PARAMS])

def _candidate_to_makeOneRun_kwargs(x, x_params_names,
                                    fixed_params_names, fixed_params_vals,
                                    logicFn_x_idxs, logicFn_fix_idxs,
                                    biasFn_x_idxs,  biasFn_fix_idxs,
                                    driftFn_x_idxs, driftFn_fix_idxs,
                                    noiseFn_x_idxs, noiseFn_fix_idxs):
    """Map one candidate parameter vector ``x`` into a ready-to-splat
    ``makeOneRun`` kwargs dict.

    Splits ``x`` (fit params) and ``fixed_params_vals`` (fixed params, incl.
    ``df``/``dt``/``t_dur``/``is_loss_no_dir``/``biasFn``/… appended in
    ``simulateDDM``) across the logic / drift / noise / bias dispatch tables via
    the precomputed index arrays, then applies the frozen-rate sentinels.

    Shared by the chisq DE driver (``_makeOneRunWrapper``) and the joint MLE+Chi²
    objective (``_jointVectorizedObjectiveWrapper``). MLE-only fit params (e.g.
    ``LAPSE_RATE``) are not in any dispatch index set, so they're harmlessly
    ignored here — Chi² is computed from the same superset candidate vector.
    """
    x_params_names = np.asanyarray(x_params_names)
    logicFn_kwargs = {k: v for k, v in zip(x_params_names[logicFn_x_idxs],
                                            x[logicFn_x_idxs])}
    logicFn_kwargs.update({k: v for k, v in zip(fixed_params_names[logicFn_fix_idxs],
                                                fixed_params_vals[logicFn_fix_idxs])})
    driftFn_kwargs = {k: v for k, v in zip(x_params_names[driftFn_x_idxs],
                                            x[driftFn_x_idxs])}
    driftFn_kwargs.update({k: v for k, v in zip(fixed_params_names[driftFn_fix_idxs],
                                                fixed_params_vals[driftFn_fix_idxs])})
    noiseFn_kwargs = {k: v for k, v in zip(x_params_names[noiseFn_x_idxs],
                                            x[noiseFn_x_idxs])}
    noiseFn_kwargs.update({k: v for k, v in zip(fixed_params_names[noiseFn_fix_idxs],
                                                fixed_params_vals[noiseFn_fix_idxs])})
    biasFn_kwargs  = {k: v for k, v in zip(x_params_names[biasFn_x_idxs],
                                            x[biasFn_x_idxs])}
    biasFn_kwargs.update({k: v for k, v in zip(fixed_params_names[biasFn_fix_idxs],
                                               fixed_params_vals[biasFn_fix_idxs])})

    include_Q = logicFn_kwargs["include_Q"]
    include_RewardRate = logicFn_kwargs["include_RewardRate"]
    # Frozen-rate sentinels: ALPHA / BETA stay NaN (makeOneRun asserts
    # ``~np.isnan(ALPHA) if include_Q`` — NaN here means "Q-learning is
    # off, this is a placeholder"). The asymmetric *_UNREWARDED params
    # propagate as None, which state_updates interprets as "fall back to
    # the symmetric rate" — no sentinel value special-case.
    if not include_Q:
        logicFn_kwargs["ALPHA"] = np.nan
    if not include_RewardRate:
        logicFn_kwargs["BETA"] = np.nan
    logicFn_kwargs.setdefault("ALPHA_UNREWARDED", None)
    logicFn_kwargs.setdefault("BETA_UNREWARDED", None)
    return {**logicFn_kwargs, "driftFn_kwargs": driftFn_kwargs,
            "noiseFn_kwargs": noiseFn_kwargs, "biasFn_kwargs": biasFn_kwargs}


def _trace_population(trace, x_matrix, losses):
    """Append each candidate's ``(param values, loss)`` to the DE trace.

    ``trace`` is a list (in-process MLE/joint path) or a
    ``multiprocessing.Manager().list()`` proxy (pooled chisq path); both support
    ``.append``. ``None`` disables tracing. Params are stored as plain lists so
    the row pickles cheaply across the pool boundary."""
    if trace is None:
        return
    xm = np.asarray(x_matrix, dtype=float)
    if xm.ndim == 1:
        xm = xm[:, None]
    losses = np.asarray(losses, dtype=float)
    for s in range(xm.shape[1]):
        trace.append((xm[:, s].tolist(), float(losses[s])))


def _build_candidate_loss_df(trace, params_names, model_config=None):
    """Assemble the per-candidate DE loss trace into a DataFrame: ``loss`` + one
    column per fit param, plus (MLE) the weights / conditions the fit ran with
    (constant per fit, so the frame is self-describing). ``None`` if empty."""
    rows = list(trace) if trace is not None else []
    if not rows:
        return None
    names = [str(n) for n in params_names]
    xs = np.array([np.asarray(x, dtype=float) for (x, _loss) in rows])
    df = pd.DataFrame(xs, columns=names)
    df.insert(0, "loss", [float(loss) for (_x, loss) in rows])
    if model_config is not None:
        df["mle_choice_weight"] = float(getattr(model_config, "mle_choice_weight", np.nan))
        df["mle_rt_weight"] = float(getattr(model_config, "mle_rt_weight", np.nan))
        df["mle_choice_norm"] = str(getattr(model_config, "mle_choice_norm", ""))
        df["mle_mle_weight"] = float(getattr(model_config, "mle_mle_weight", np.nan))
        df["mle_chi2_weight"] = float(getattr(model_config, "mle_chi2_weight", np.nan))
        df["mle_condition_columns"] = str(
            tuple(getattr(model_config, "mle_condition_columns", ())))
    return df


def _makeOneRunWrapper(x, x_params_names, fixed_params_names, fixed_params_vals,
                       logicFn_x_idxs, logicFn_fix_idxs,
                       biasFn_x_idxs,  biasFn_fix_idxs,
                       driftFn_x_idxs, driftFn_fix_idxs,
                       noiseFn_x_idxs, noiseFn_fix_idxs,
                       trace=None,
                       ):
    assert len(x_params_names) == len(x)
    loss = makeOneRun(**_candidate_to_makeOneRun_kwargs(
        x, x_params_names, fixed_params_names, fixed_params_vals,
        logicFn_x_idxs, logicFn_fix_idxs, biasFn_x_idxs, biasFn_fix_idxs,
        driftFn_x_idxs, driftFn_fix_idxs, noiseFn_x_idxs, noiseFn_fix_idxs))
    if trace is not None:
        trace.append((np.asarray(x, dtype=float).tolist(), float(loss)))
    return loss


def _mleVectorizedObjectiveWrapper(x_matrix, x_params_names, subject_df,
                                   model_config, trace=None):
    """Vectorized DE objective.

    SciPy calls this once per generation with ``x_matrix.shape == (n_params, S)``
    and expects ``(S,)`` losses back. By doing the popsize loop inside our
    process we keep a single CuPy context across the whole generation, avoid
    the multiprocessing.Pool boundary, and pay one parameter-vector transfer
    per generation instead of one per candidate.
    """
    losses = objective_from_population(
        x_matrix, x_params_names, subject_df, model_config)
    _trace_population(trace, x_matrix, losses)
    return losses


def _jointVectorizedObjectiveWrapper(
        x_matrix, x_params_names, prepared_subject, model_config,
        fixed_params_names, fixed_params_vals,
        logicFn_x_idxs, logicFn_fix_idxs, biasFn_x_idxs, biasFn_fix_idxs,
        driftFn_x_idxs, driftFn_fix_idxs, noiseFn_x_idxs, noiseFn_fix_idxs,
        ref_mle, ref_chi2, trace=None):
    """Joint MLE + Chi² DE objective:
    ``w_mle·(MLE/ref_mle) + w_chi2·(Chi2/ref_chi2)``.

    The MLE term stays on the GPU-vectorized whole-population path
    (``objective_from_population`` — one solver call per generation). The Chi²
    term is the existing generative Ratcliff-quantile loss, computed per
    candidate on CPU via ``makeOneRun`` (numpy; independent of the GPU MLE
    call). Each term is divided by a fixed per-subject **reference loss** — the
    component's standalone-best loss (``ref_mle`` = pure-MLE chi2_weight=0;
    ``ref_chi2`` = ``--fit-mode chisq``) — so each lands near 1 at its own
    optimum and the two are commensurable despite different raw scales (both
    references are guaranteed positive at load time). SciPy passes ``x_matrix``
    as ``(n_params, S)`` and expects ``(S,)`` back.

    The Chi² simulation's ``df`` is taken from ``fixed_params_vals[0]`` (set to
    the subject df in ``_processSubject``) via the dispatch index arrays — the
    same wiring the chisq driver uses — so no separate df arg is needed.

    A zero weight skips its term entirely (no GPU call / no simulation). Since
    ``_processSubject`` only routes here when ``mle_chi2_weight > 0``, the Chi²
    loop always runs; ``mle_mle_weight == 0`` is supported (Chi²-only via this
    driver) but the pure vectorized-MLE path is preferred when Chi² is off.
    """
    x_matrix = np.asarray(x_matrix, dtype=float)
    if x_matrix.ndim == 1:
        x_matrix = x_matrix[:, None]
    x_params_names = np.asanyarray(x_params_names)
    n_cand = x_matrix.shape[1]
    w_mle = float(model_config.mle_mle_weight)
    w_chi2 = float(model_config.mle_chi2_weight)
    ref_mle = float(ref_mle)
    ref_chi2 = float(ref_chi2)

    if w_mle > 0.0:
        mle_negll = np.asarray(
            objective_from_population(
                x_matrix, x_params_names, prepared_subject, model_config),
            dtype=float)
        mle_pt = mle_negll / ref_mle
    else:
        mle_pt = np.zeros(n_cand, dtype=float)

    if w_chi2 > 0.0:
        chi2_pt = np.empty(n_cand, dtype=float)
        for s in range(n_cand):
            kwargs = _candidate_to_makeOneRun_kwargs(
                x_matrix[:, s], x_params_names,
                fixed_params_names, fixed_params_vals,
                logicFn_x_idxs, logicFn_fix_idxs, biasFn_x_idxs, biasFn_fix_idxs,
                driftFn_x_idxs, driftFn_fix_idxs, noiseFn_x_idxs, noiseFn_fix_idxs)
            chi2_raw = float(makeOneRun(**kwargs))
            if not np.isfinite(chi2_raw):
                # Keep DE selection finite; a degenerate sim ranks as worst.
                chi2_raw = 1e12
            chi2_pt[s] = chi2_raw / ref_chi2
    else:
        chi2_pt = np.zeros(n_cand, dtype=float)

    total = w_mle * mle_pt + w_chi2 * chi2_pt
    _trace_population(trace, x_matrix, total)
    return total


# Cache result if we are running in parallel
_last_df_FP = None
_last_df = None
_running_locally = False
def _processSubject(subject_df, fixed_params_names, fixed_params_vals,
                    fit_params_names, fit_params_bounds, fit_params_init,
                    logicFn_x_idxs, logicFn_fix_idxs,
                    driftFn_x_idxs, driftFn_fix_idxs,
                    noiseFn_x_idxs, noiseFn_fix_idxs,
                    biasFn_x_idxs, biasFn_fix_idxs,
                    include_Q, include_RewardRate, dt, t_dur,
                    is_loss_no_dir, workers, evolve_dump_FP, dry_run,
                    fit_mode, model_config=None,
                    ref_mle_FP=None, ref_chi2_FP=None):

    if not _running_locally:
        assert isinstance(subject_df, (str, pathlib.Path))
        global _last_df_FP, _last_df
        if subject_df != _last_df_FP:
            _tmp_df = pd.read_pickle(subject_df)
            _last_df_FP = subject_df
            _last_df = _tmp_df
        else:
            _tmp_df = _last_df
        subject_df = _tmp_df

    subject = subject_df.Name.iloc[0]
    # print(f"Subject: {subject}")
    # Crash early if we built the wrong path
    evolve_dump_FP_subject = _evolveFPSubject(evolve_dump_FP, subject)
    if not dry_run:
        assert evolve_dump_FP_subject.parent.exists(), f"{evolve_dump_FP_subject.parent} does not exist"

    fixed_params_vals[0] = subject_df
    if fit_mode == "mle":
        assert model_config is not None, (
            "fit_mode='mle' requires a non-None model_config; "
            "simulateDDM constructs one — this should be unreachable.")
        # Prepare ONCE with the condition columns so
        # ``PreparedMLEData.trial_weights`` is cached upfront.
        # Downstream callers (``objective_from_population``,
        # ``result_payload``'s ``evaluate_neg_loglik``) short-circuit
        # on an already-prepared instance and would otherwise inherit
        # ``trial_weights=None`` — silently running the unweighted
        # objective even when ``--mle-conditions`` was passed.
        prepared_subject = prepare_mle_data(
            subject_df, model_config.mle_condition_columns)

        # Pre-flight: when the user asked for --mle-backend GPU, resolve and
        # probe the backend before scipy DE starts. This catches
        # CUDA-unavailable / wrong-device-id / cupy-fell-back-silently cases
        # *immediately* instead of after a few generations of confusing slow
        # behavior.
        if model_config.requires_gpu:
            backend = resolve_array_backend(
                model_config.mle_array_backend,
                model_config.mle_device_id,
                model_config.mle_cupy_fallback,
            )
            probe = assert_gpu_backend(backend)
            print(
                f"MLE GPU backend confirmed: backend={backend.actual_backend}, "
                f"device_id={backend.device_id}, "
                f"probe_type={type(probe).__module__}.{type(probe).__name__}")
            num_workers = 1
        else:
            num_workers = workers
            print(f"Using CPU backend with {num_workers} workers for MLE.")


        bound_for_population = 1.0
        bound_idxs = np.flatnonzero(
            np.asarray([str(name).upper() for name in fit_params_names])
            == "BOUND")
        if len(bound_idxs):
            bound_for_population = float(fit_params_bounds[bound_idxs[0]][1])
        population_info = estimate_population_settings(
            model_config,
            n_trials=prepared_subject.n_trials,
            n_params=len(fit_params_names),
            bound=bound_for_population,
        )
        print("MLE population info:", population_info)

        # Joint MLE+Chi² mode: a non-zero --mle-chi2-weight augments the MLE
        # objective with the generative Ratcliff-quantile Chi² loss. Each
        # component is normalized by a fixed per-subject REFERENCE loss — the
        # component's standalone-best loss (pure-MLE chi2_weight=0 for ref_mle,
        # --fit-mode chisq for ref_chi2) — so the two terms are commensurable.
        # References are loaded once here; a missing one fails loudly.
        # mle_chi2_weight==0 (default) keeps the byte-identical pure-MLE driver.
        joint_mode = float(model_config.mle_chi2_weight) > 0.0
        ref_mle = ref_chi2 = 1.0
        ref_mle_time = ref_chi2_time = None
        if joint_mode:
            ref_mle, ref_mle_time = _load_reference_loss(ref_mle_FP, subject)
            ref_chi2, ref_chi2_time = _load_reference_loss(ref_chi2_FP, subject)

        def _add_joint_diag(payload, x_vec):
            """Attach the MLE / Chi² breakdown + references at ``x_vec`` so the
            joint total ``w_mle·(mle/ref_mle) + w_chi2·(chi2/ref_chi2)`` is
            reconstructable. Reuses the pure-MLE ``neg_loglik`` already in
            ``payload`` for the MLE component; recomputes Chi² at ``x_vec`` via
            the generative simulation (``makeOneRun``)."""
            w_mle = float(model_config.mle_mle_weight)
            w_chi2 = float(model_config.mle_chi2_weight)
            mle_raw = float(payload["neg_loglik"])
            chi2_raw = float(makeOneRun(**_candidate_to_makeOneRun_kwargs(
                np.asarray(x_vec, dtype=float), fit_params_names,
                fixed_params_names, fixed_params_vals,
                logicFn_x_idxs, logicFn_fix_idxs, biasFn_x_idxs, biasFn_fix_idxs,
                driftFn_x_idxs, driftFn_fix_idxs, noiseFn_x_idxs,
                noiseFn_fix_idxs)))
            mle_part = mle_raw / ref_mle
            chi2_part = chi2_raw / ref_chi2
            payload.update(
                joint_mode=True,
                mle_mle_weight=w_mle, mle_chi2_weight=w_chi2,
                mle_raw_loss=mle_raw, chi2_raw_loss=chi2_raw,
                ref_mle=ref_mle, ref_chi2=ref_chi2,
                ref_mle_time=ref_mle_time, ref_chi2_time=ref_chi2_time,
                mle_part_loss=mle_part, chi2_part_loss=chi2_part,
                total_loss=w_mle * mle_part + w_chi2 * chi2_part)
            return payload

        if dry_run:
            payload = result_payload(
                optim_res=None,
                params_names=fit_params_names,
                params_init=fit_params_init,
                params_bounds=fit_params_bounds,
                subject_df=prepared_subject,
                model_config=model_config,
                population_info=population_info,
            )
            if joint_mode:
                _add_joint_diag(payload, fit_params_init)
                print("MLE+Chi² joint dry-run: "
                      f"mle_part={payload['mle_part_loss']:.6g} "
                      f"(ref_mle={payload['ref_mle']:.6g}), "
                      f"chi2_part={payload['chi2_part_loss']:.6g} "
                      f"(ref_chi2={payload['ref_chi2']:.6g}), "
                      f"total={payload['total_loss']:.6g}, "
                      f"weights=({payload['mle_mle_weight']}, "
                      f"{payload['mle_chi2_weight']})")
            print("MLE backend info:", payload["mle_backend_info"])
            return payload

        # MLE path: scipy DE with vectorized=True collapses the per-generation
        # popsize round-trips into one objective call. ``workers=1`` and
        # ``updating="deferred"`` are the right defaults for BOTH backends:
        # - GPU: one CuPy context per process, no Pool sharing the device,
        #   no synchronous per-candidate evaluations.
        # - CPU: vectorized=True already calls the objective once per
        #   generation, so a worker pool would only add IPC/pickle overhead.
        # The upstream warning in ``simulateDDM`` rejects --num-cpus != 1
        # for MLE for the same reason.
        # Per-candidate DE loss trace (in-process: MLE/joint is single-process
        # vectorized). Appended to by the objective wrapper each generation.
        candidate_trace = []
        if joint_mode:
            # Joint objective: MLE term stays on the GPU-vectorized population
            # path; Chi² is looped per candidate on CPU and combined. The Chi²
            # histogram term is non-smooth, so polish stays False (as below).
            objective = _jointVectorizedObjectiveWrapper
            objective_args = (
                fit_params_names, prepared_subject, model_config,
                fixed_params_names, fixed_params_vals,
                logicFn_x_idxs, logicFn_fix_idxs, biasFn_x_idxs, biasFn_fix_idxs,
                driftFn_x_idxs, driftFn_fix_idxs, noiseFn_x_idxs,
                noiseFn_fix_idxs, ref_mle, ref_chi2, candidate_trace)
        else:
            objective = _mleVectorizedObjectiveWrapper
            objective_args = (fit_params_names, prepared_subject, model_config,
                              candidate_trace)
        res = differential_evolution(
            objective,
            bounds=fit_params_bounds,
            args=objective_args,
            x0=fit_params_init,
            disp=True,
            workers=num_workers,
            polish=False,
            popsize=population_info["scipy_popsize"],
            updating="deferred",
            vectorized=True,
            #mutation=(0.5, 1.5),
        )
        dict_res = result_payload(
            optim_res=res,
            params_names=fit_params_names,
            params_init=fit_params_init,
            params_bounds=fit_params_bounds,
            subject_df=prepared_subject,
            model_config=model_config,
            population_info=population_info,
        )
        if joint_mode:
            _add_joint_diag(dict_res, res.x)
            print("MLE+Chi² joint fit: "
                  f"mle_part={dict_res['mle_part_loss']:.6g} "
                  f"(ref_mle={dict_res['ref_mle']:.6g}), "
                  f"chi2_part={dict_res['chi2_part_loss']:.6g} "
                  f"(ref_chi2={dict_res['ref_chi2']:.6g}), "
                  f"total={dict_res['total_loss']:.6g}, "
                  f"de_fun={float(res.fun):.6g}")
        dict_res["candidate_losses_df"] = _build_candidate_loss_df(
            candidate_trace, fit_params_names, model_config)
        print("MLE backend info:", dict_res["mle_backend_info"])
        with open(evolve_dump_FP_subject, 'wb') as f:
            pickle.dump(dict_res, f)
        return dict_res

    if dry_run:
        loss = _makeOneRunWrapper(x=np.array(fit_params_init),
                                  x_params_names=fit_params_names,
                                  fixed_params_names=fixed_params_names,
                                  fixed_params_vals=fixed_params_vals,
                                  logicFn_x_idxs=logicFn_x_idxs,
                                  logicFn_fix_idxs=logicFn_fix_idxs,
                                  biasFn_x_idxs=biasFn_x_idxs,
                                  biasFn_fix_idxs=biasFn_fix_idxs,
                                  driftFn_x_idxs=driftFn_x_idxs,
                                  driftFn_fix_idxs=driftFn_fix_idxs,
                                  noiseFn_x_idxs=noiseFn_x_idxs,
                                  noiseFn_fix_idxs=noiseFn_fix_idxs,
                                  )
        return loss

    # Per-candidate DE loss trace. The chisq DE may parallelize candidate evals
    # across a process pool (workers=_pool.map), so when pooled we accumulate
    # through a Manager().list() proxy that workers can append to across the
    # process boundary; single-process runs use a plain in-process list.
    if callable(workers):
        trace_manager = multiprocessing.Manager()
        candidate_trace = trace_manager.list()
    else:
        trace_manager = None
        candidate_trace = []
    res = differential_evolution(_makeOneRunWrapper,
                                 bounds=fit_params_bounds,
                                 args=(fit_params_names,
                                       fixed_params_names,
                                       fixed_params_vals,
                                       logicFn_x_idxs, logicFn_fix_idxs,
                                       biasFn_x_idxs, biasFn_fix_idxs,
                                       driftFn_x_idxs, driftFn_fix_idxs,
                                       noiseFn_x_idxs, noiseFn_fix_idxs,
                                       candidate_trace,
                                       ),
                                 x0=fit_params_init,
                                 disp=True,
                                #  maxiter=500,
                                 workers=workers,
                                #  polish = False,
                                polish = True,
                                popsize=100,
                                mutation=(0.5, 1.5),
                                )
    candidate_trace = list(candidate_trace)  # drain proxy → plain list
    if trace_manager is not None:
        trace_manager.shutdown()

    dict_res = dict(OptimRes=res,
                    fixed_params_names=fixed_params_names,
                    fixed_params_vals=fixed_params_vals,
                    params_names=fit_params_names,
                    params_init=fit_params_init,
                    #model_class=model_class,
                    dt=dt,
                    t_dur=t_dur,
                    subject_df=subject_df,
                    include_Q=include_Q,
                    include_RewardRate=include_RewardRate,
                    is_loss_no_dir=is_loss_no_dir,
                    fit_mode=fit_mode,
                    noise_dt_scaling="sqrt_dt",
                    fit_finish_time=datetime.datetime.now().isoformat(
                        timespec="seconds"),
                    candidate_losses_df=_build_candidate_loss_df(
                        candidate_trace, fit_params_names, model_config=None),
                    )

    with open(evolve_dump_FP_subject, 'wb') as f:
        pickle.dump(dict_res, f)

    return dict_res

def _evolveFPSubject(evolveFP : pathlib.Path, subject):
    # Add the subject before .pkl and save in the evolv_res_dump/ folder
    fp_str = str(evolveFP).replace("\\", "/")
    save_dir = "data/RLModel/"
    assert f"{save_dir}" in fp_str
    evolve_subj_FP = fp_str.replace(f"{save_dir}", f"{save_dir}/subject/")
    evolve_subj_FP = evolve_subj_FP.replace(".pkl", f"_{subject}.pkl")
    print("evolve_subj_FP:", evolve_subj_FP)
    return pathlib.Path(evolve_subj_FP)

def _weight_suffix(mle_mle_weight=1.0, mle_chi2_weight=0.0):
    """Filename suffix encoding the joint-loss weights.

    Only joint fits (``mle_chi2_weight > 0``) get a suffix; pure MLE
    (chi2 weight 0) and chisq keep the canonical name so they remain the
    normalization references and existing pickles still resolve. ``{:g}``
    drops trailing zeros so the suffix round-trips cleanly (1.0 -> "1").
    """
    if float(mle_chi2_weight) > 0.0:
        return (f"_mleW{float(mle_mle_weight):g}"
                f"_chi2W{float(mle_chi2_weight):g}")
    return ""


def evolveFP(drift_fn_str, bias_fn_str, noise_fn_str, t_dur, dt,
            is_loss_no_dir, fit_mode,
            uses_asym_q=False, uses_asym_rr=False,
            uses_scaled_bound=False,
            mle_mle_weight=1.0, mle_chi2_weight=0.0):
    """Build the saved-fit pickle path.

    The optional ``_asymQ`` / ``_asymRR`` / ``_asymQRR`` suffix carries
    the asymmetric-LR opt-ins orthogonally to the bias / drift / noise
    model identity. ``_scaledB`` further marks fits where BOUND is the
    fitted axis (NOISE_SIGMA frozen) — see ``--scale-bound``. A trailing
    ``_mleW{m}_chi2W{c}`` marks joint MLE+Chi² fits so weight variants
    don't overwrite each other (and the pure-MLE/chisq references
    survive). Suffix ordering: asym, then scaledB, then weights.
    Symmetric / fixed-bound / pure-MLE / chisq fits keep the original
    filename format unchanged.
    """
    loss_no_dir_str = "" if not is_loss_no_dir else "_loss_no_dir"
    if uses_asym_q and uses_asym_rr:
        asym_suffix = "_asymQRR"
    elif uses_asym_q:
        asym_suffix = "_asymQ"
    elif uses_asym_rr:
        asym_suffix = "_asymRR"
    else:
        asym_suffix = ""
    scaled_bound_suffix = "_scaledB" if uses_scaled_bound else ""
    weight_suffix = _weight_suffix(mle_mle_weight, mle_chi2_weight)
    main_str = (f"data/RLModel/{fit_mode}_{drift_fn_str}_"
                f"bias{bias_fn_str}_{noise_fn_str}"
                f"{loss_no_dir_str}_{t_dur}s_dt{dt}"
                f"{asym_suffix}{scaled_bound_suffix}{weight_suffix}.pkl")
    return pathlib.Path(main_str)


def _load_reference_loss(ref_FP, subject):
    """Load a subject's reference loss for joint-loss normalization.

    Returns ``(loss, fit_finish_time)`` from the saved merged result file
    ``ref_FP`` (a ``{subject: payload}`` dict), using ``OptimRes.fun`` — the
    fit's minimized loss. Raises with an actionable message if the file or the
    subject entry is missing, or the reference isn't a positive finite number:
    the ``loss / reference`` normalization needs a positive reference, else the
    ratio would invert the optimization.
    """
    if ref_FP is None:
        raise ValueError("Joint-loss normalization requires a reference path; "
                         "got None (this should be set in simulateDDM).")
    ref_path = pathlib.Path(ref_FP)
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Joint-loss normalization needs a reference fit at {ref_path}, "
            f"which does not exist. Run that fit first: pure MLE with "
            f"--mle-chi2-weight 0 (for ref_mle), and --fit-mode chisq "
            f"(for ref_chi2), before the joint fit.")
    with open(ref_path, "rb") as f:
        ref_dict = pickle.load(f)
    if subject not in ref_dict:
        raise KeyError(
            f"Reference fit {ref_path} has no entry for subject {subject!r}; "
            f"available subjects: {sorted(ref_dict)}.")
    payload = ref_dict[subject]
    optim = payload.get("OptimRes") if isinstance(payload, dict) else None
    loss = getattr(optim, "fun", None)
    if loss is None or not np.isfinite(loss) or float(loss) <= 0.0:
        raise ValueError(
            f"Reference fit {ref_path} subject {subject!r} has a non-positive "
            f"or non-finite reference loss ({loss!r}); reference normalization "
            f"requires a positive reference loss.")
    fit_time = payload.get("fit_finish_time") if isinstance(payload, dict) else None
    return float(loss), fit_time


def _merge_save_evolve(evolve_dump_FP, subject, dict_res):
    """Reload the on-disk evolve pickle, set this subject's entry, and
    atomically replace the file.

    Lets parallel processes fitting DIFFERENT subjects merge into the
    shared pickle instead of the last writer clobbering earlier ones
    with the stale snapshot it loaded at startup: the reload happens
    immediately before each write, so a process that finishes later
    picks up the entries other processes already saved. Writing to a
    pid-suffixed temp file + ``Path.replace`` keeps the pickle from ever
    being observed half-written.

    NOTE: this narrows but does not fully close the inter-process race
    (a TOCTOU window remains between the reload and the replace). It is
    safe for the intended workload — a few long-running processes each
    fitting a different subject — because they touch different keys.
    """
    dump_path = pathlib.Path(evolve_dump_FP)
    merged = {}
    if dump_path.exists():
        try:
            with open(dump_path, "rb") as f:
                merged = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            merged = {}  # tolerate a concurrent half-write; re-add below
    merged[subject] = dict_res
    tmp_path = dump_path.with_suffix(dump_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        pickle.dump(merged, f)
    tmp_path.replace(dump_path)  # atomic on the same filesystem
    return merged


_pool = None # Ruse pool between runs
def simulateDDM(df, bounds_and_defaults, dt, t_dur, biasFn, driftFn, noiseFn,
                is_loss_no_dir, num_cpus, evolvs_res : dict, fit_mode,
                dry_run=False, mle_array_backend="numpy",
                mle_device_id=None, mle_cupy_fallback="error",
                mle_gpu_memory_gb=None, mle_show_progress=False,
                mle_terminal_c=MLE_TERMINAL_C.Default,
                mle_min_population_candidates=None,
                mle_condition_columns=(),
                mle_choice_weight=1.0, mle_rt_weight=1.0,
                mle_choice_norm="conditional",
                mle_mle_weight=1.0, mle_chi2_weight=0.0,
                bias_fn_str=None, drift_fn_str=None,
                uses_asym_q=False, uses_asym_rr=False,
                scale_bound=False):
    global _pool
    if fit_mode != "chisq":
        if fit_mode != "mle":
            raise ValueError(
                f"Unknown fit_mode: {fit_mode!r}. Expected 'chisq' or 'mle'.")
    # print("fixed params names:", scipy_params["fixed_params_names"])
    # Strip down our df to the minimum in case it gets copied to the parallel processes

    biasFn_df_cols, biasFn_kwargs_li = biasFnColsAndKwargs(biasFn)
    driftFn_df_cols, driftFn_kwargs_li = driftFnColsAndKwargs(driftFn)
    noiseFn_df_cols, noiseFn_kwargs_li = noiseFnColsAndKwargs(noiseFn)

    print("bias_df_cols:", biasFn_df_cols)
    print("bias_kwars_li:", biasFn_kwargs_li)
    print("drft_df_cols:", driftFn_df_cols)
    print("drift_kwars_li:", driftFn_kwargs_li)
    print("noise_df_cols:", noiseFn_df_cols)
    print("noise_kwars_li:", noiseFn_kwargs_li)


    include_Q = "Q_val" in biasFn_df_cols or "Q_val" in driftFn_df_cols or "Q_val" in noiseFn_df_cols
    include_RewardRate = "RewardRate" in driftFn_df_cols or "RewardRate" in noiseFn_df_cols or "RewardRate" in biasFn_df_cols
    # Bound-RewardRate detection: the new drift family is observationally
    # equivalent to a per-trial-bound model via the path-D rescaling
    # (mu/r_t, sigma/r_t, z/r_t) — see scale_bound_equivalence.ipynb and
    # mle.py:_compute_latent_population_equal_sessions. Triggered on
    # ``drift_fn_str`` prefix; flows into MLEModelConfig below.
    uses_per_trial_bound = bool(
        drift_fn_str is not None
        and drift_fn_str.startswith("Bound-RewardRate"))
    # Soft-limit: Bound-RewardRate is mathematically equivalent to
    # NoiseGain-RewardRate when BOUND is frozen at 1.0 (the default
    # without --scale-bound). Running them together wastes compute and
    # emits a misleadingly-named pickle.
    #
    # CLI/GUI users no longer reach this branch — both surfaces only
    # expose ``RewardRate`` and ``drift.resolve_drift_alias`` routes
    # the alias to the right family based on --scale-bound (or
    # Scale-How in the GUI). The check stays as protection for direct
    # programmatic callers of ``simulateDDM`` that bypass the alias
    # layer. If you genuinely want to run Bound-RewardRate without
    # --scale-bound from code, comment out the following block.
    if uses_per_trial_bound and not scale_bound:
        raise ValueError(
            f"Drift {drift_fn_str!r} uses per-trial bound scaling "
            f"(BOUND * r_t), but --scale-bound is OFF so BOUND is "
            f"frozen at 1.0. This collapses to NoiseGain-RewardRate "
            f"semantics with extra compute. Either add --scale-bound "
            f"to actually fit BOUND in [0.3, 5.0], or use a "
            f"NoiseGain-RewardRate drift instead. Soft-limit — see "
            f"fit.py near this line to override.")
    # ``--scale-bound``: swap which of (BOUND, NOISE_SIGMA) is fit.
    # InitVal override applied to the bounds_and_defaults dict (which is
    # InitVals.toDict() — owned by the caller; mutating it is the same
    # mechanism InitVals.override uses internally).
    #
    # ``InitVals.BOUND`` is the fittable bound (default range
    # (0.3, 5.0)); ``InitVals._BOUND_FIXED`` is its frozen counterpart
    # (default (1, 1, 1)). Same pairing for NOISE_SIGMA /
    # _NOISE_FIXED. Exactly one of the pair is fittable at a time —
    # the other is overridden in-place onto the canonical fit-vector
    # key (``BOUND`` / ``NOISE_SIGMA``) below.
    bounds_and_defaults = dict(bounds_and_defaults)
    if scale_bound:
        # Scale-How=Bound: BOUND uses its dataclass default (fittable).
        # NOISE_SIGMA is overridden to the frozen counterpart.
        noise_fixed = bounds_and_defaults["_NOISE_FIXED"]
        bounds_and_defaults["NOISE_SIGMA"] = noise_fixed
        bound_active = bounds_and_defaults["BOUND"]
        print(f"--scale-bound: BOUND fitted in "
              f"[{bound_active.Min}, {bound_active.Max}], "
              f"NOISE_SIGMA frozen at {noise_fixed.Default}")
    else:
        # Scale-How=Noise (legacy default): NOISE_SIGMA uses its
        # dataclass default (fittable). BOUND is overridden to the
        # frozen counterpart. This restores bit-exact pre-refactor
        # behavior — when the user hasn't asked for --scale-bound,
        # BOUND lands at the same (1, 1, 1) range it had as the old
        # ``InitVals.BOUND``.
        bound_fixed = bounds_and_defaults["_BOUND_FIXED"]
        bounds_and_defaults["BOUND"] = bound_fixed


    fixed_params = dict(biasFn=biasFn,
                        biasFn_df_cols=biasFn_df_cols,
                        biasFn_kwargs=biasFn_kwargs_li,
                        driftFn=driftFn,
                        driftFn_df_cols=driftFn_df_cols,
                        driftFn_kwargs=driftFn_kwargs_li,
                        noiseFn=noiseFn,
                        noiseFn_df_cols=noiseFn_df_cols,
                        noiseFn_kwargs=noiseFn_kwargs_li,
    )


    fixed_params_names = [k for k in fixed_params.keys()]
    fixed_params_vals = [v for v in fixed_params.values()]
    fixed_params_names.insert(0, "df")
    fixed_params_vals.insert(0, None) # Temp

    # Extra args:
    fixed_params_names += ["include_Q", "include_RewardRate", "dt", "t_dur", "return_df", "is_loss_no_dir"]
    fixed_params_vals += [include_Q,    include_RewardRate,    dt,   t_dur,  False, is_loss_no_dir]

    fixed_params_names = np.asarray(fixed_params_names)
    fixed_params_vals = np.asarray(fixed_params_vals, dtype=object)

    # print("Fixed Params Names:", fixed_params_names)
    # print("Fixed Params Vals:", fixed_params_vals)
    assert len(fixed_params_names) == len(fixed_params_vals)
    # for fix_param_name, fix_param_val in zip(fixed_params_names, fixed_params_vals):
    #     print(fix_param_name, "=", fix_param_val)

    # Asymmetric learning-rate gating: orthogonal to model identity.
    # The two explicit ``uses_asym_*`` flags (set by the CLI / GUI /
    # caller) are the only source. They're combined with the
    # auto-detected ``include_Q`` / ``include_RewardRate`` so a flag
    # against an incompatible model surfaces as a no-op here — the
    # CLI pre-flight in model_runner.py is the friendly error layer.
    include_Q_asym = include_Q and uses_asym_q
    include_RewardRate_asym = include_RewardRate and uses_asym_rr

    # Declarative per-param gating table — the single place that says
    # "this param enters the fit vector iff <flag>". Params for which
    # the gate fires False get appended to ``manually_passed_params``,
    # so _makeOneRunWrapper passes the documented sentinel
    # (NaN for ALPHA/BETA, None for the *_UNREWARDED pair) to makeOneRun.
    # Adding a new flag-gated param is one entry here, no new if-block.
    _PARAM_FIT_GATES = {
        "ALPHA":            include_Q,
        "BETA":             include_RewardRate,
        "ALPHA_UNREWARDED": include_Q_asym,
        "BETA_UNREWARDED":  include_RewardRate_asym,
    }
    manually_passed_params = ["driftFn_kwargs", "noiseFn_kwargs",
                              "biasFn_kwargs", "is_loss_no_dir"]
    for name, gate in _PARAM_FIT_GATES.items():
        if not gate:
            manually_passed_params.append(name)
    makeOneRun_params_names = np.asarray([param for param in _makeOneRun_params_names
                                         if param not in manually_passed_params])
    def assertInBoundsAndDefaults(x):
        x = x.upper()
        assert x in bounds_and_defaults, f"{x} not in bounds_and_defaults"
        return x, (bounds_and_defaults[x][0], bounds_and_defaults[x][1]), bounds_and_defaults[x][2]
    # Create first as set to group duplicate params where the same param is used
    # in multiple functions
    # Sorry for the hack, but include ALPHA and BETA only needed as they are
    # always present in the function signature even if not used.
    _makeOneRun_fit_params = [_param for _param in makeOneRun_params_names
                              if _param not in fixed_params_names]

    fit_params_li = {assertInBoundsAndDefaults(x)
                    for li in (_makeOneRun_fit_params,
                               driftFn_kwargs_li, noiseFn_kwargs_li,
                               biasFn_kwargs_li)
                    for x in li}
    # MLE-only fittable parameters: don't go through driftFn/biasFn/noiseFn or
    # makeOneRun, so they wouldn't otherwise enter the fit vector. The chisq
    # path silently ignores LAPSE_RATE; only the MLE objective consumes it.
    mle_only_param_names = (
        {"LAPSE_RATE"} if fit_mode == "mle" else set())
    for mle_only in mle_only_param_names:
        fit_params_li.add(assertInBoundsAndDefaults(mle_only))
    fit_params_li = list(fit_params_li)
    fit_params_names, fit_params_bounds, fit_params_init = zip(*fit_params_li)
    # Converto to lists
    fit_params_names, fit_params_bounds, fit_params_init = map(np.asarray, (
        fit_params_names, fit_params_bounds, fit_params_init))


    print("Fit Params Names:", fit_params_names)
    # print("Fit Params init:", fit_params_init)
    # print("FIt Params bounds:", fit_params_bounds)


    def indxsOfParams(names, params):
        params = {p.upper() for p in params}
        # print("Params:", params)
        # print("Names:", {name.upper() for name in names})
        return np.array([i for i, name in enumerate(names)
                         if name.upper() in params], dtype=int)

    logicFn_fit_idxs = indxsOfParams(fit_params_names, makeOneRun_params_names)
    logicFn_fix_idxs = indxsOfParams(fixed_params_names, makeOneRun_params_names)
    driftFn_fit_idxs = indxsOfParams(fit_params_names, driftFn_kwargs_li)
    driftFn_fix_idxs = indxsOfParams(fixed_params_names, driftFn_kwargs_li)
    noiseFn_fit_idxs = indxsOfParams(fit_params_names, noiseFn_kwargs_li)
    noiseFn_fix_idxs = indxsOfParams(fixed_params_names, noiseFn_kwargs_li)
    biasFn_fit_idxs = indxsOfParams(fit_params_names, biasFn_kwargs_li)
    biasFn_fix_idxs = indxsOfParams(fixed_params_names, biasFn_kwargs_li)

    DEBUG = True
    if DEBUG:
        def _print_idxs(name, x_idxs, fix_idxs):
            if len(x_idxs):
                print(f"{name} x_idxs:", x_idxs, "==", fit_params_names[x_idxs])
            else:
                print(f"{name} x_idxs: None")
            if len(fix_idxs):
                print(f"{name} fix_idxs:", fix_idxs, "==", fixed_params_names[fix_idxs])
            else:
                print(f"{name} fix_idxs: None")
        _print_idxs("LogicFn", logicFn_fit_idxs, logicFn_fix_idxs)
        _print_idxs("DriftFn", driftFn_fit_idxs, driftFn_fix_idxs)
        _print_idxs("NoiseFn", noiseFn_fit_idxs, noiseFn_fix_idxs)
        _print_idxs("BiasFn",  biasFn_fit_idxs,  biasFn_fix_idxs)

    # assert that we used all the params
    assert len(logicFn_fit_idxs) + len(logicFn_fix_idxs) == len(makeOneRun_params_names)
    assert len(driftFn_fit_idxs) + len(driftFn_fix_idxs) == len(driftFn_kwargs_li)
    assert len(noiseFn_fit_idxs) + len(noiseFn_fix_idxs) == len(noiseFn_kwargs_li)
    assert len(biasFn_fit_idxs) + len(biasFn_fix_idxs) == len(biasFn_kwargs_li)
    # Now assert that we used all the params from both the fixed and fit params
    used_fix_idxs = (set(logicFn_fix_idxs) | set(driftFn_fix_idxs) |
                     set(noiseFn_fix_idxs) | set(biasFn_fix_idxs))
    unused_fix_idxs = list(set(range(len(fixed_params_names))) - used_fix_idxs)
    # Flag-gated params (ALPHA, BETA, ALPHA_UNREWARDED, BETA_UNREWARDED)
    # have defaults on makeOneRun, so when the gate is False they never
    # appear in either x or fixed_params — they're just absent. Pad the
    # unused-count with sentinels so the assert below still balances out
    # against ``len(manually_passed_params)``.
    flag_gated_count = sum(1 for g in _PARAM_FIT_GATES.values() if not g)
    unused_fix_idxs += [np.nan] * flag_gated_count
    len_manual_params = len(manually_passed_params)
    # TODO: Remove the manual params from the unused_fix_idxs so we get a
    # filtered list of unused params
    print("Unused fix idxs:", len(unused_fix_idxs))
    print("len_manual_params:", len_manual_params)
    assert len(unused_fix_idxs) - len_manual_params == 0, (
      f"Unused (look TODO) fixed params: {fixed_params_names[unused_fix_idxs]}")
    used_fit_idxs = (set(logicFn_fit_idxs) | set(driftFn_fit_idxs) |
                     set(noiseFn_fit_idxs) | set(biasFn_fit_idxs))
    # MLE-only params (LAPSE_RATE) deliberately don't dispatch to any sub-fn
    # — they're consumed by the MLE objective directly. Exclude them from
    # the "every fit param is used by a dispatch table" assertion.
    mle_only_fit_idxs = {i for i, n in enumerate(fit_params_names)
                         if str(n).upper() in mle_only_param_names}
    unused_fit_idxs = list(
        set(range(len(fit_params_names))) - used_fit_idxs - mle_only_fit_idxs)
    assert not len(unused_fit_idxs), (
                      f"Unused fit params: {fit_params_names[unused_fit_idxs]}")




    df = initDF(df, include_Q, include_RewardRate)
    keep_cols = ["DVabs", "DV", "DVstr", "valid",
                 "calcStimulusTime", "ChoiceCorrect", "ChoiceLeft",
                 "Name", "Date", "SessionNum", "TrialNumber", "SessId",
                 "SimRT", "SimStartingPoint", "SimChoiceCorrect", "SimChoiceLeft",
                #  "SimMatchReal"
                 ]
    if include_Q:
        keep_cols += ["Q_L", "Q_R", "Q_val"]
    if include_RewardRate:
        keep_cols += ["RewardRate"]
    df = df[keep_cols]

    if include_Q:
        assert "ALPHA" in fit_params_names
    else:
        assert "ALPHA" not in fit_params_names
    if include_RewardRate:
        assert "BETA" in fit_params_names
    else:
        assert "BETA" not in fit_params_names

    all_subjects = df.Name.unique()
    remaining_subjects = [subject for subject in all_subjects
                          if subject not in evolvs_res]
    print("Skipping:", [subject for subject in all_subjects
                       if subject not in remaining_subjects])

    # When multiple dict keys map to the same function object (e.g. the
    # "-asym" aliases in BIAS_FN_DICT / DRIFT_FN_DICT point to the same
    # ``_biasQVal`` / ``_noiseGainRewardRate``), a plain comprehension
    # keeps the LAST key seen — which silently picks the asymmetric
    # alias and enables the asymmetric LR for callers that thought they
    # were fitting the legacy variant. Iterating in reverse means the
    # FIRST (canonical) key wins, so the default reverse-lookup behavior
    # is the safe symmetric one. Callers who actually want the
    # asymmetric variant must pass it through the new ``bias_fn_str`` /
    # ``drift_fn_str`` kwargs (overridden below).
    reverse_DriftLookup = {v: k for k, v in reversed(DRIFT_FN_DICT.items())}
    reverse_BiasLookup = {v: k for k, v in reversed(BIAS_FN_DICT.items())}
    reverse_NoiseLookup = {v: k for k, v in reversed(NOISE_FN_DICT.items())}
    assert driftFn in reverse_DriftLookup
    assert biasFn in reverse_BiasLookup
    assert noiseFn in reverse_NoiseLookup
    driftFn_str = reverse_DriftLookup[driftFn]
    biasFn_str = reverse_BiasLookup[biasFn]
    noiseFn_str = reverse_NoiseLookup[noiseFn]
    # Prefer the caller-supplied canonical names when available. The
    # reverse_*Lookup picks the LAST dict key for a given function
    # object, which the "-asym" aliases in BIAS_FN_DICT/DRIFT_FN_DICT
    # (same fn, different key) hijack — without this override the user's
    # "--bias 'Q-Val (Offset)'" silently turns into "Q-Val-asym (Offset)",
    # which then fails validate_mle_config. Passing the str through
    # runModel preserves the exact name the user / CLI provided.
    if bias_fn_str is not None:
        biasFn_str = bias_fn_str
    if drift_fn_str is not None:
        driftFn_str = drift_fn_str
    model_config = None
    if fit_mode == "mle":
        model_config = MLEModelConfig(
            drift_fn_str=driftFn_str,
            bias_fn_str=biasFn_str,
            noise_fn_str=noiseFn_str,
            include_Q=include_Q,
            include_RewardRate=include_RewardRate,
            dt=dt,
            t_dur=t_dur,
            mle_array_backend=mle_array_backend,
            mle_device_id=mle_device_id,
            mle_cupy_fallback=mle_cupy_fallback,
            mle_gpu_memory_gb=mle_gpu_memory_gb,
            mle_show_progress=mle_show_progress,
            mle_terminal_c=float(mle_terminal_c),
            # DE population floor. ``--mle-min-population`` translates
            # into this; programmatic callers can leave it None to
            # take the dataclass default (``MIN_POPULATION_CANDIDATES``).
            mle_min_population_candidates=int(
                mle_min_population_candidates
                if mle_min_population_candidates is not None
                else MIN_POPULATION_CANDIDATES),
            # Per-condition sample-balancing columns. Empty tuple →
            # unweighted (legacy). Filename is NOT affected (evolveFP
            # doesn't see this), so A/B tests overwrite the same
            # pickle. ``tuple(...)`` for the frozen dataclass and to
            # neutralize any list the caller passes.
            mle_condition_columns=tuple(mle_condition_columns or ()),
            # Weighted choice-vs-RT loss. Like the conditions above, NOT
            # reflected in the filename (evolveFP doesn't see these), so
            # A/B tests overwrite the same pickle. mle_choice_norm
            # defaults to "conditional" at this user-facing boundary
            # (the dataclass default is "marginal" for back-compat with
            # old pickles / direct-construct tests). See mle.py.
            mle_choice_weight=float(mle_choice_weight),
            mle_rt_weight=float(mle_rt_weight),
            mle_choice_norm=mle_choice_norm,
            # Outer joint-loss weights (--mle-mle-weight / --mle-chi2-weight).
            # Default (1.0, 0.0) ⇒ pure MLE; mle_chi2_weight > 0 switches
            # _processSubject to the joint MLE+Chi² driver. Also NOT reflected
            # in the filename (evolveFP doesn't see these), so A/B tests
            # overwrite the same pickle.
            mle_mle_weight=float(mle_mle_weight),
            mle_chi2_weight=float(mle_chi2_weight),
            # The flag-gated asymmetric-LR contract on MLEModelConfig
            # (see mle.py:_compute_latent_arrays). True ⇒ ALPHA_UNREWARDED
            # / BETA_UNREWARDED MUST be in the params dict at eval time —
            # strict access, KeyError on miss.
            uses_asymmetric_alpha=include_Q_asym,
            uses_asymmetric_beta=include_RewardRate_asym,
            uses_per_trial_bound=uses_per_trial_bound,
            uses_scaled_bound=scale_bound,
        )
        # Pre-flight: the gate table + fit-param list must agree, else
        # the MLE objective hits KeyError mid-DE rather than failing
        # loudly here.
        if model_config.uses_asymmetric_alpha:
            assert "ALPHA_UNREWARDED" in fit_params_names, (
                "uses_asymmetric_alpha=True but ALPHA_UNREWARDED is not in "
                "fit_params_names; check _PARAM_FIT_GATES and bias_fn_str")
        if model_config.uses_asymmetric_beta:
            assert "BETA_UNREWARDED" in fit_params_names, (
                "uses_asymmetric_beta=True but BETA_UNREWARDED is not in "
                "fit_params_names; check _PARAM_FIT_GATES and drift_fn_str")
    evolve_dump_FP = evolveFP(driftFn_str, biasFn_str, noiseFn_str, t_dur, dt,
                              is_loss_no_dir, fit_mode,
                              uses_asym_q=uses_asym_q,
                              uses_asym_rr=uses_asym_rr,
                              uses_scaled_bound=scale_bound,
                              mle_mle_weight=mle_mle_weight,
                              mle_chi2_weight=mle_chi2_weight)

    # Joint-loss references (loss/ref normalization): the pure-MLE
    # (chi2_weight=0) and chisq merged result files for THIS model. Built always
    # (cheap path construction); only loaded per-subject in _processSubject when
    # mle_chi2_weight>0.
    ref_mle_FP = evolveFP(driftFn_str, biasFn_str, noiseFn_str, t_dur, dt,
                          is_loss_no_dir, "mle",
                          uses_asym_q=uses_asym_q, uses_asym_rr=uses_asym_rr,
                          uses_scaled_bound=scale_bound)  # chi2_weight=0 → canonical
    ref_chi2_FP = evolveFP(driftFn_str, biasFn_str, noiseFn_str, t_dur, dt,
                           is_loss_no_dir, "chisq",
                           uses_asym_q=uses_asym_q, uses_asym_rr=uses_asym_rr,
                           uses_scaled_bound=scale_bound)

    is_gpu_mle = fit_mode == "mle" and model_config.requires_gpu
    if num_cpus is None:
        if is_gpu_mle:
            num_cpus = 1
        else:
            num_cpus = multiprocessing.cpu_count()
    elif is_gpu_mle and num_cpus != 1:
        # Item E: the MLE DE call sets vectorized=True (which makes scipy
        # ignore `workers`) and runs the popsize loop inside this process.
        # A multiprocessing.Pool is therefore (a) unused by scipy and (b)
        # actively harmful for CuPy backends since each child would init its
        # own CUDA context fighting for the same GPU. Force num_cpus=1 for
        # MLE regardless of the user's --num-cpus flag.
        raise ValueError(
                    f"fit_mode='mle' with GPU backend {model_config.mle_array_backend} "
                    f"requires num_cpus=1; got {num_cpus}.")

    if num_cpus != 1:
        # Reusable pool to avoid ulimit file exhaustion. This parallelizes the
        # chisq DE's per-candidate evaluations across cores (scipy `workers`);
        # MLE forces num_cpus=1 above, so it never builds a pool.
        if _pool is None or _pool._processes != num_cpus:
            _pool = multiprocessing.Pool(num_cpus)
        workers = _pool.map
    else:
        assert num_cpus == 1
        workers = num_cpus
        global _running_locally
        _running_locally = True


    partialProcess = partial(_processSubject,
                             fixed_params_names=fixed_params_names,
                             fixed_params_vals=fixed_params_vals,
                             fit_params_names=fit_params_names,
                             fit_params_bounds=fit_params_bounds,
                             fit_params_init=fit_params_init,
                             logicFn_x_idxs=logicFn_fit_idxs, logicFn_fix_idxs=logicFn_fix_idxs,
                             biasFn_x_idxs=biasFn_fit_idxs, biasFn_fix_idxs=biasFn_fix_idxs,
                             driftFn_x_idxs=driftFn_fit_idxs, driftFn_fix_idxs=driftFn_fix_idxs,
                             noiseFn_x_idxs=noiseFn_fit_idxs, noiseFn_fix_idxs=noiseFn_fix_idxs,
                             include_Q=include_Q, include_RewardRate=include_RewardRate,
                             is_loss_no_dir=is_loss_no_dir,
                             dt=dt, t_dur=t_dur, workers=workers,
                             evolve_dump_FP=evolve_dump_FP,
                             dry_run=dry_run,
                             fit_mode=fit_mode,
                             model_config=model_config,
                             ref_mle_FP=ref_mle_FP,
                             ref_chi2_FP=ref_chi2_FP)
    for subject in remaining_subjects:
        subject_df = df[df.Name == subject]
        if not _running_locally:
            dump_FP = pathlib.Path(f"data/RLModel/df_dump/{fit_mode}_"
                                   f"{subject}_{driftFn_str}"
                                   f"_{noiseFn_str}_{biasFn_str}.pkl")
            if not dump_FP.parent.exists():
                dump_FP.parent.mkdir(exist_ok=True)
            print("Dumping:", dump_FP)
            subject_df.to_pickle(dump_FP)
            subject_df = dump_FP
        dict_res = partialProcess(subject_df)
        evolvs_res[subject] = dict_res
        if not dry_run:
            # Reload-merge save (not a full overwrite) so concurrent
            # processes fitting other subjects aren't clobbered.
            _merge_save_evolve(evolve_dump_FP, subject, dict_res)
        print("Subject:", subject, "done")
        if dry_run:
            break

    return evolvs_res
