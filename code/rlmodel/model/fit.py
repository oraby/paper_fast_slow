from .bias import BIAS_FN_DICT
from .drift import DRIFT_FN_DICT
from .noise import NOISE_FN_DICT
from .logic import makeOneRun
from .array_backend import assert_gpu_backend, resolve_array_backend
from .mle import (
    MLEModelConfig,
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
import multiprocessing
import multiprocessing.pool
import pathlib
import pickle



_NON_FITTABLE_MAKEONERUN_PARAMS = {"seed", "skip_loss"}
_makeOneRun_params_names = inspect.signature(makeOneRun).parameters.keys()
_makeOneRun_params_names = np.asanyarray([p for p in _makeOneRun_params_names
                                          if p not in _NON_FITTABLE_MAKEONERUN_PARAMS])

def _makeOneRunWrapper(x, x_params_names, fixed_params_names, fixed_params_vals,
                       logicFn_x_idxs, logicFn_fix_idxs,
                       biasFn_x_idxs,  biasFn_fix_idxs,
                       driftFn_x_idxs, driftFn_fix_idxs,
                       noiseFn_x_idxs, noiseFn_fix_idxs,
                       ):
    assert len(x_params_names) == len(x)
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
    if not include_Q:
        logicFn_kwargs["ALPHA"] = np.nan
    if not include_RewardRate:
        logicFn_kwargs["BETA"] = np.nan
    DEBUG = False
    if DEBUG:
        print("LogicFn kwargs:", logicFn_kwargs)
        print("DriftFn kwargs:", driftFn_kwargs)
        print("NoiseFn kwargs:", noiseFn_kwargs)
        print("BiasFn kwargs:", biasFn_kwargs)
    # print("x:", x)
    # print("x_params_names:", x_params_names)
    # print("driftFn_kwargs:", driftFn_kwargs)

    return makeOneRun(**logicFn_kwargs, driftFn_kwargs=driftFn_kwargs,
                      noiseFn_kwargs=noiseFn_kwargs, biasFn_kwargs=biasFn_kwargs)


def _mleVectorizedObjectiveWrapper(x_matrix, x_params_names, subject_df,
                                   model_config):
    """Vectorized DE objective.

    SciPy calls this once per generation with ``x_matrix.shape == (n_params, S)``
    and expects ``(S,)`` losses back. By doing the popsize loop inside our
    process we keep a single CuPy context across the whole generation, avoid
    the multiprocessing.Pool boundary, and pay one parameter-vector transfer
    per generation instead of one per candidate.
    """
    return objective_from_population(
        x_matrix, x_params_names, subject_df, model_config)

class _NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class _NoDaemonContext(type(multiprocessing.get_context())):
    Process = _NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


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
                    fit_mode, model_config=None):

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
        prepared_subject = prepare_mle_data(subject_df)

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

        if dry_run:
            payload = result_payload(
                optim_res=None,
                params_names=fit_params_names,
                params_init=fit_params_init,
                params_bounds=fit_params_bounds,
                subject_df=prepared_subject,
                model_config=model_config,
            )
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
        res = differential_evolution(
            _mleVectorizedObjectiveWrapper,
            bounds=fit_params_bounds,
            args=(fit_params_names, prepared_subject, model_config),
            x0=fit_params_init,
            disp=True,
            workers=1,
            polish=True,
            popsize=1024*100,
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
        )
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

    res = differential_evolution(_makeOneRunWrapper,
                                 bounds=fit_params_bounds,
                                 args=(fit_params_names,
                                       fixed_params_names,
                                       fixed_params_vals,
                                       logicFn_x_idxs, logicFn_fix_idxs,
                                       biasFn_x_idxs, biasFn_fix_idxs,
                                       driftFn_x_idxs, driftFn_fix_idxs,
                                       noiseFn_x_idxs, noiseFn_fix_idxs,
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

def evolveFP(drift_fn_str, bias_fn_str, noise_fn_str, t_dur, dt,
            is_loss_no_dir, fit_mode):
    loss_no_dir_str = "" if not is_loss_no_dir else "_loss_no_dir"
    main_str = (f"data/RLModel/{fit_mode}_{drift_fn_str}_"
                f"bias{bias_fn_str}_{noise_fn_str}"
                f"{loss_no_dir_str}_{t_dur}s_dt{dt}.pkl")
    return pathlib.Path(main_str)


_pool = None # Ruse pool between runs
def simulateDDM(df, bounds_and_defaults, dt, t_dur, biasFn, driftFn, noiseFn,
                is_loss_no_dir, num_cpus, evolvs_res : dict, fit_mode,
                dry_run=False, mle_array_backend="numpy",
                mle_device_id=None, mle_cupy_fallback="error",
                mle_batch_size=None, mle_gpu_memory_gb=None):
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

    # Remove Params we are going to pass manually
    manually_passed_params = ["driftFn_kwargs", "noiseFn_kwargs",
                              "biasFn_kwargs", "is_loss_no_dir"]
    extra_ignored_count = 0
    if not include_Q:
        manually_passed_params.append("ALPHA")
        extra_ignored_count += 1
    if not include_RewardRate:
        manually_passed_params.append("BETA")
        extra_ignored_count += 1
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
    # ALPHA and BETA has default values so they dont show up if not used
    unused_fix_idxs += [np.nan] * extra_ignored_count # Just to keep the same length
    len_manual_params = len(manually_passed_params)
    # TODO: Remove the manual params from the unused_fix_idxs so we get a
    # filtered list of unused params
    print("Unused fix idxs:", len(unused_fix_idxs))
    print("len_manual_params:", len_manual_params)
    assert len(unused_fix_idxs) - len_manual_params == 0, (
      f"Unused (look TODO) fixed params: {fixed_params_names[unused_fix_idxs]}")
    used_fit_idxs = (set(logicFn_fit_idxs) | set(driftFn_fit_idxs) |
                     set(noiseFn_fit_idxs) | set(biasFn_fit_idxs))
    unused_fit_idxs = list(set(range(len(fit_params_names))) - used_fit_idxs)
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

    reverse_DriftLookup = {v:k for k,v in DRIFT_FN_DICT.items()}
    reverse_BiasLookup = {v:k for k,v in BIAS_FN_DICT.items()}
    reverse_NoiseLookup = {v:k for k,v in NOISE_FN_DICT.items()}
    assert driftFn in reverse_DriftLookup
    assert biasFn in reverse_BiasLookup
    assert noiseFn in reverse_NoiseLookup
    driftFn_str = reverse_DriftLookup[driftFn]
    biasFn_str = reverse_BiasLookup[biasFn]
    noiseFn_str = reverse_NoiseLookup[noiseFn]
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
            mle_batch_size=mle_batch_size,
            mle_gpu_memory_gb=mle_gpu_memory_gb,
        )
    evolve_dump_FP = evolveFP(driftFn_str, biasFn_str, noiseFn_str, t_dur, dt,
                              is_loss_no_dir, fit_mode)

    IS_PARALLEL_EXECUTION_ENABLED = False
    if num_cpus is None:
        num_cpus = 1 if fit_mode == "mle" else multiprocessing.cpu_count()
    elif fit_mode == "mle" and num_cpus != 1:
        # Item E: the MLE DE call sets vectorized=True (which makes scipy
        # ignore `workers`) and runs the popsize loop inside this process.
        # A multiprocessing.Pool is therefore (a) unused by scipy and (b)
        # actively harmful for CuPy backends since each child would init its
        # own CUDA context fighting for the same GPU. Force num_cpus=1 for
        # MLE regardless of the user's --num-cpus flag.
        backend_note = (
            " (CuPy contexts would contend for the GPU)"
            if mle_array_backend == "cupy" else "")
        print(
            f"WARNING: MLE mode runs single-process; ignoring "
            f"--num-cpus {num_cpus}{backend_note}.")
        num_cpus = 1

    if IS_PARALLEL_EXECUTION_ENABLED:
        workers = num_cpus / 2
        workers = max(workers, 1)
    else:
        if num_cpus != 1:
            # Reusable pool to avoid ulimit file exhaustion
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
                             model_config=model_config)
    if not IS_PARALLEL_EXECUTION_ENABLED:
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
                with open(evolve_dump_FP, 'wb') as f:
                    pickle.dump(evolvs_res, f)
            print("Subject:", subject, "done")
            if dry_run:
                break
    else:
        with NestablePool(3) as p:
            for dict_res in p.imap_unordered(partialProcess,
                                             [df[df.Name == subject] for subject in remaining_subjects]):
                subject_df = dict_res["subject_df"]
                subject = subject_df.Name.iloc[0]
                evolvs_res[subject] = dict_res
                if not dry_run:
                    with open(evolve_dump_FP, 'wb') as f:
                        pickle.dump(evolvs_res, f)
                print("Subject:", subject, "done")
                if dry_run:
                    break

    return evolvs_res
