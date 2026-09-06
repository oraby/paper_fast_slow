"""Golden test: does the extracted Fig. 1l code reproduce the notebook's?

Run it from the repo root::

    uv run python code/rlmodel/golden_fig1l.py

(It reads the fits through ``model.fitio.loadFit``, so the conda env is no
longer needed — see ``docs/data-portability.md``. Still a script rather than a
pytest test because it needs the real fits, which are not in git.)

It runs both implementations over the same ``chisq_*`` fits and compares them:

- **baseline** — ``model_analysis.ipynb``'s ORIGINAL in-cell ``runSubjectData``
  + ``collectMetric``, reproduced verbatim below, except that (a) the stale
  ``..._3s_dt0.005.pkl`` filename key is replaced with ``fit.evolveFP`` (those
  3s pickles were deleted; the fits on disk are 4.8s ``chisq_*``), and (b) a
  missing ``BOUND`` / ``NOISE_SIGMA`` falls back to 1.0. See ``_PARAM_FALLBACK``
  below — if the fits do carry both params the fallback never fires and the
  baseline is the original code exactly.
- **extracted** — ``model.aggregate.collect_metrics(FIG1L_SPECS)``.

Both are reduced to the scalar metrics Fig. 1l actually plots and compared with
``pd.testing.assert_frame_equal``. Expect equality: same fits, same seed (0),
same math.
"""
from __future__ import annotations

import pathlib
import importlib
import sys

import numpy as np
import pandas as pd


def _bootstrap():
    """Make the deep package imports work when run as a plain script (the
    notebooks do the same thing in their first cell).

    The package prefix is read from the checkout's own directory name rather
    than hardcoded, so this still runs if the repo is cloned under a different
    name or nested somewhere else.
    """
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]           # the checkout, whatever it is called
    sys.path.insert(0, str(repo_root.parent))
    return repo_root, f"{repo_root.name}.code.rlmodel.model"


REPO_ROOT, _MODEL_PKG = _bootstrap()

_model = importlib.import_module(_MODEL_PKG)
compare = importlib.import_module(f"{_MODEL_PKG}.compare")
fit = importlib.import_module(f"{_MODEL_PKG}.fit")
_aggregate = importlib.import_module(f"{_MODEL_PKG}.aggregate")
FIG1L_SPECS = _aggregate.FIG1L_SPECS
MIN_NUM_TRIALS = _aggregate.MIN_NUM_TRIALS
collect_metrics = _aggregate.collect_metrics
subject_metrics = _aggregate.subject_metrics
BIAS_FN_DICT = importlib.import_module(f"{_MODEL_PKG}.bias").BIAS_FN_DICT
DRIFT_FN_DICT = importlib.import_module(f"{_MODEL_PKG}.drift").DRIFT_FN_DICT
NOISE_FN_DICT = importlib.import_module(f"{_MODEL_PKG}.noise").NOISE_FN_DICT
runAndPlot = importlib.import_module(f"{_MODEL_PKG}.plotter").runAndPlot
loadFit = importlib.import_module(f"{_MODEL_PKG}.fitio").loadFit
_util = importlib.import_module(f"{_MODEL_PKG}.util")
biasFnColsAndKwargs = _util.biasFnColsAndKwargs
driftFnColsAndKwargs = _util.driftFnColsAndKwargs
noiseFnColsAndKwargs = _util.noiseFnColsAndKwargs

# The frozen scale axis is absent from a fit (BOUND for a noise-scaled fit,
# NOISE_SIGMA for a scale-bound one). The original cell indexed params_dict
# unconditionally; mirror model/compare.py's documented 1.0 default so the
# baseline can consume the current fits at all.
_PARAM_FALLBACK = {"BOUND": 1.0, "NOISE_SIGMA": 1.0}


# --------------------------------------------------------------------------
# Baseline: model_analysis.ipynb's original cell-14 / cell-20 code
# --------------------------------------------------------------------------
def runSubjectData_original(subject, subject_fit_res, noiseFn, biasFn, driftFn,
                            t_dur, dt, is_loss_no_dir, df_behavior):
    x = subject_fit_res["OptimRes"].x
    include_Q = subject_fit_res["include_Q"]
    include_RewardRate = subject_fit_res["include_RewardRate"]

    subject_df = df_behavior[df_behavior.Name == subject].copy()

    params_names = subject_fit_res["params_names"]
    assert len(x) == len(params_names)
    params_dict = {name: val for name, val in zip(params_names, x)}

    def _p(name):
        if name in params_dict:
            return params_dict[name]
        if name in _PARAM_FALLBACK:
            return _PARAM_FALLBACK[name]
        raise KeyError(name)

    ALPHA = params_dict["ALPHA"] if include_Q else np.nan
    BETA = params_dict["BETA"] if include_RewardRate else np.nan

    biasFn_df_cols, biasFn_kwargs_li = biasFnColsAndKwargs(biasFn)
    driftFn_df_cols, driftFn_kwargs_li = driftFnColsAndKwargs(driftFn)
    noiseFn_df_cols, noiseFn_kwargs_li = noiseFnColsAndKwargs(noiseFn)
    biasFn_kwargs = {n: params_dict[n] for n in biasFn_kwargs_li}
    driftFn_kwargs = {n: params_dict[n] for n in driftFn_kwargs_li}
    noiseFn_kwargs = {n: params_dict[n] for n in noiseFn_kwargs_li}

    loss, fitted_df = runAndPlot(
        df=subject_df, ALPHA=ALPHA, BETA=BETA,
        DRIFT_COEF=_p("DRIFT_COEF"), NOISE_SIGMA=_p("NOISE_SIGMA"),
        BOUND=_p("BOUND"), NON_DECISION_TIME=_p("NON_DECISION_TIME"),
        biasFn=biasFn, driftFn=driftFn, noiseFn=noiseFn,
        t_dur=t_dur, dt=dt, fig=None, axs=None, is_small_fig_mode=False,
        include_Q=include_Q, include_RewardRate=include_RewardRate,
        plot_bias_dir=False, psych_plot=False,
        biasFn_kwargs=biasFn_kwargs, biasFn_df_cols=biasFn_df_cols,
        driftFn_kwargs=driftFn_kwargs, driftFn_df_cols=driftFn_df_cols,
        noiseFn_kwargs=noiseFn_kwargs, noiseFn_df_cols=noiseFn_df_cols,
        is_loss_no_dir=is_loss_no_dir, verbose=False)
    return loss, fitted_df, _p("BOUND"), include_Q, include_RewardRate


def collect_baseline(df_behavior):
    """The original per-model loop, over the fits the presets address."""
    t_dur, dt, noise_fn_str = 4.8, 0.005, "Normal(0, 1)"
    # (drift, bias) per FIG1L_SPECS, in the same order.
    combos = [("Classic", "None_"), ("Classic", "Q-Val (Offset)"),
              ("NoiseGain-RewardRate", "None_"),
              ("NoiseGain-RewardRate", "Q-Val (Offset)")]
    rows = []
    for spec, (drift_fn_str, bias_fn_str) in zip(FIG1L_SPECS, combos):
        fp = fit.evolveFP(drift_fn_str, bias_fn_str, noise_fn_str, t_dur, dt,
                          is_loss_no_dir=False, fit_mode="chisq")
        path = REPO_ROOT / fp
        print(f"  baseline: {path.name}")
        res_dict = loadFit(path)
        noiseFn = NOISE_FN_DICT[noise_fn_str]
        driftFn = DRIFT_FN_DICT[drift_fn_str]
        biasFn = BIAS_FN_DICT[bias_fn_str]
        for subject in sorted(res_dict):
            loss, fitted_df, _b, _q, _rr = runSubjectData_original(
                subject, res_dict[subject], noiseFn, biasFn, driftFn, t_dur, dt,
                is_loss_no_dir=False, df_behavior=df_behavior)
            if len(fitted_df) < MIN_NUM_TRIALS:
                print(f"    {subject}: {len(fitted_df)} trials -> skipped")
                continue
            row = subject_metrics(subject, fitted_df)
            row["SpecLabel"] = spec.label
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def _comparable(df):
    """The scalar metrics Fig. 1l plots, keyed + sorted for comparison."""
    keys = ["SpecLabel", "Name"]
    metrics = ["R2_Psych", "RewardRateCorr", "R2_WinLose", "NumTrials"]
    return (df[keys + metrics].sort_values(keys).reset_index(drop=True))


def main():
    # Both default to paths relative to the notebooks' cwd (code/rlmodel), so
    # pass absolutes to stay runnable from anywhere.
    print("Loading behavior + discovering fits...")
    df_behavior = compare.prepare_behavior_df(
        df_fp=str(REPO_ROOT / "data" / "behavior" / "df_behavior.pkl"))
    fits = compare.discover_fits(result_dir=str(REPO_ROOT / "data" / "RLModel"))

    print("\nFit params (does the original's unconditional BOUND lookup work?)")
    for entry in fits.values():
        for subject, cols in list(entry.subjects.items())[:1]:
            for col in cols:
                print(f"  {col.filename}\n    params_names="
                      f"{list(col.payload['params_names'])}")
            break
        break

    if not fits:
        sys.exit(
            "discover_fits() found no readable fits in data/RLModel.\n"
            "discover_fits silently skips pickles it cannot read, so check\n"
            "the skip reasons above. The fits are stored portably now, so an\n"
            "unreadable one is a real problem rather than the wrong\n"
            "environment: see docs/data-portability.md, and check whether\n"
            "the copy in data_bak/ loads.")

    print("\n=== baseline (notebook's original code) ===")
    baseline = collect_baseline(df_behavior)

    print("\n=== extracted (aggregate.collect_metrics) ===")
    extracted = collect_metrics(fits, FIG1L_SPECS, df_behavior,
                                num_evaluations=1)

    a, b = _comparable(baseline), _comparable(extracted)
    print(f"\nbaseline rows={len(a)}  extracted rows={len(b)}")
    pd.testing.assert_frame_equal(a, b)
    print("\nMATCH — the extracted code reproduces the notebook's Fig. 1l "
          "metrics exactly.")


if __name__ == "__main__":
    main()
