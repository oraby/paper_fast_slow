"""Aggregate model evaluation — the single controller for "which models do we
compare, and how well does each reproduce behavior?".

Extracted from ``model_analysis.ipynb``'s in-cell ``getFitDict`` /
``runSubjectData`` / ``collectMetric`` (the ``Plot Aggregates`` → ``Fig. 1l``
path) so the analysis is importable, testable, and reusable across figures —
the user's "export notebook code to files / don't duplicate" directive.

Model selection
---------------
A model to evaluate is an :class:`EvalSpec`: a ``(model_key, column_label)``
coordinate in the space ``compare.discover_fits`` already builds off the
saved-fit filenames (see ``mle_reeval.FitFileId`` and
``compare.classify_column``). That coordinate is what lets one preset say
"reward-rate, noise-scaled, Chi²-fit" and another say "same model, but the
scale-bound fit" or "same model, but the joint MLE+Chi²=0.1 fit" — without any
figure knowing a filename. Presets for the paper's figures live at the bottom
of this module.

Repeat evaluation
-----------------
The DDM forward pass is stochastic. ``num_evaluations=1`` runs the historical
single trajectory (``seed=0``); ``N > 1`` runs seeds ``0..N-1``, so iteration 0
always reproduces the single-evaluation result. Each iteration is reduced to
its metric row immediately — only the ``seed=0`` simulation dataframe is
retained (into ``sim_cache``), because holding N full trial-level frames per
subject × model does not fit in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pathlib
import pickle
import re

import numpy as np
import pandas as pd

from . import compare
from .plotter import calcWinLoseUpdates, _dvQuantileFn
from ...behavior.bias import calcBias
from ...behavior.rewardrate import plotSubjectRewardRateRt
from ...figcode.prevoutcomecurquantile import quantilePrevOutcomeCur
from ...figcode.psychometric import _psychFitBasic


# Analysis constants — the values ``model_analysis.ipynb`` sets in its
# "Define Analysis Parameters" / "Collect relevant data points" cells.
MIN_NUM_TRIALS = 2_500
MIN_TRIALS_PER_SESS_RR = 5
REWARD_RATE_NUM_PAST_TRIALS = 5
REWARD_RATE_BY_SESS = True

DV_STRS = ["Easy", "Med", "Hard"]

# Random multi-starts per psychometric fit (``_psychFitBasic``'s ``nfits``).
# This is the single most expensive part of an evaluation — ~64% of
# subject_metrics, which itself dwarfs the simulation — so it is worth lowering
# for exploratory passes. It changes the fitted values, hence it is threaded
# through explicitly and recorded in the metrics cache rather than read from a
# mutable global.
N_PSYCH_FITS = 20

# Where load_or_collect_metrics keeps its collected frames. Relative to the
# notebooks' cwd (code/rlmodel), matching compare.DEFAULT_RESULT_DIR.
DEFAULT_METRICS_CACHE_DIR = "../../data/RLModel/metrics"


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalSpec:
    """One model to evaluate == one bar group in an aggregate figure.

    ``model_key`` / ``column_label`` address a fit in the coordinate space of
    ``compare.discover_fits``: the key is the abstract model
    (``FitFileId.model_key`` — drift alias + bias + noise + timing + asym) and
    the label is the fitting criterion (``compare.classify_column`` —
    ``"Chi²-Noise"``, ``"Chi²-Bound"``, ``"MLE"``, ``"MLE=1, Chi²=0.1"``, …).

    ``gap_after`` adds blank x-space after this group, in bar-slot units, so a
    figure can visually separate sub-families (e.g. the half-column between the
    no-Q-val and +Q-val pairs).
    """
    label: str
    model_key: str
    column_label: str
    color: str
    gap_after: float = 0.0


def available_coords(fits):
    """``[(model_key, column_label), …]`` present in ``fits`` — the menu an
    ``EvalSpec`` may address. Used for error messages."""
    return sorted((entry.model_key, col.column_label)
                  for entry in fits.values()
                  for cols in entry.subjects.values()
                  for col in cols)


def resolve_spec(fits, spec):
    """``{subject: ColumnFit}`` for ``spec``.

    Raises ``KeyError`` listing what *is* on disk when the spec addresses a fit
    that hasn't been run — the failure that matters in practice, since a
    missing pickle is otherwise a silently empty bar.
    """
    entry = fits.get(spec.model_key)
    if entry is None:
        raise KeyError(
            f"No fits for model_key {spec.model_key!r} (spec {spec.label!r}). "
            f"Available: {sorted({k for k, _ in available_coords(fits)})}")
    resolved = {}
    for subject, cols in entry.subjects.items():
        for col in cols:
            if col.column_label == spec.column_label:
                resolved[subject] = col
                break
    if not resolved:
        raise KeyError(
            f"Model {spec.model_key!r} has no column {spec.column_label!r} "
            f"(spec {spec.label!r}). Available columns for this model: "
            f"{sorted({c.column_label for cols in entry.subjects.values() for c in cols})}")
    return resolved


# --------------------------------------------------------------------------
# Metric primitives (lifted from the notebook's collectMetric cell)
# --------------------------------------------------------------------------
def _calcR2(data_real, data_model, print_res=False, subject=None):
    data_real = np.asanyarray(data_real)
    data_model = np.asanyarray(data_model)
    real_mean = data_real.mean()
    SS_res = ((data_real - data_model)**2).sum()
    SS_reg = ((data_model - real_mean)**2).sum()
    SS_total = SS_res + SS_reg
    r2 = SS_reg / SS_total
    if print_res:
        print("Subject:", subject, "Real Mean:", real_mean,
              "Var Mean:", SS_total, "Var Diff:", SS_res, "R2:", r2)
    return r2


def _prevOutcomeCountRT(df, rt_col, col_prefix):
    LIMIT_AT = 2
    df = df[(-LIMIT_AT <= df[f"{col_prefix}PrevOutcomeCount"]) &
            (df[f"{col_prefix}PrevOutcomeCount"] <= LIMIT_AT)]

    prev_outcome = sorted(df[f"{col_prefix}PrevOutcomeCount"].unique())
    prev_outcomes_rt = {"OutcomeCount": [], "RTMean": []}
    for prev_outcome_val in prev_outcome:
        prev_outcome_df = df[df[f"{col_prefix}PrevOutcomeCount"] ==
                             prev_outcome_val]
        rt_mean = prev_outcome_df[rt_col].mean()
        prev_outcomes_rt["OutcomeCount"].append(prev_outcome_val)
        prev_outcomes_rt["RTMean"].append(rt_mean)
    return pd.DataFrame(prev_outcomes_rt)


def _fitData(df, is_abs, left_col, choice_col, n_psych_fits=N_PSYCH_FITS):
    if is_abs:
        groupby_col = "DVstr"
    else:
        groupby_col = ["DVstr", df.DV > 0]
    stims, stim_count, stims_perf = [], [], []
    for dv_str_side, dv_df in df.groupby(groupby_col):
        dvs = dv_df["DV"]
        if is_abs:
            dvs = dvs.abs()
        stims.append(dvs.mean())
        stim_count.append(len(dv_df))
        if is_abs:
            stims_perf.append(dv_df[choice_col].mean())
        else:
            stims_perf.append(dv_df[left_col].mean())
    pars, fitFn = _psychFitBasic(stims=stims, stim_count=stim_count,
                                 nfits=n_psych_fits,
                                 stim_ratio_correct=stims_perf,
                                 combine_sides=is_abs,)
    res = {"DV": [], "Perf": []}
    for stim in stims:
        res["DV"].append(stim)
        res["Perf"].append(fitFn(stim))
    return pd.DataFrame(res)


def _reward_rate_series(subject, fitted_df, col_postfix, rt_col):
    """Per-bin mean reward-rate-vs-RT for one subject, real or simulated."""
    rr = plotSubjectRewardRateRt(
        subject=subject, subject_df=fitted_df, col_postfix=col_postfix,
        rt_col=rt_col, num_past_trials=REWARD_RATE_NUM_PAST_TRIALS,
        BY_SESS=REWARD_RATE_BY_SESS, save_figs=False, RT_ZSCORE=False,
        min_trials_per_sess_rr=MIN_TRIALS_PER_SESS_RR, plot=False)
    rr = rr[rr.GUI_TimeOutIncorrectChoice == "All"]
    return rr.groupby("bin").RewardRateRTAvg.mean()


def subject_metrics(subject, fitted_df, *, n_psych_fits=N_PSYCH_FITS):
    """Every real-vs-model metric for ONE subject's simulation dataframe.

    A faithful port of ``model_analysis.ipynb``'s ``collectMetric`` inner loop,
    returning what that cell appended into its ``anlys_dict`` — one flat dict
    == one row.

    Beyond the notebook's keys this adds the scalar ``RewardRateCorr`` (the
    per-subject real-vs-model Pearson r that the Fig. 1l plotting cell used to
    compute inline from the ``RewardRate5*`` series), so plotting stays free of
    analysis. The ``RewardRate5Real`` / ``RewardRate5Model`` series are still
    returned for the figures that consume the full curve.
    """
    row = {"Name": subject, "NumTrials": len(fitted_df)}

    win_update_real, lose_update_real = calcWinLoseUpdates(fitted_df,
                                                           col_prefix="")
    win_update_model, lose_update_model = calcWinLoseUpdates(fitted_df,
                                                             col_prefix="Sim")
    row["R2_WinLose"] = _calcR2(
        np.asanyarray([win_update_real, lose_update_real]),
        np.asanyarray([win_update_model, lose_update_model]))

    subject_df_real = fitted_df
    fast_real, typical_real, slow_real = _dvQuantileFn(
        subject_df_real, "calcStimulusTime", as_df=True)
    fast_real_mean = quantilePrevOutcomeCur(
        fast_real, col_prev_choice_correct="PrevChoiceCorrect")[0]
    typical_real_mean = quantilePrevOutcomeCur(
        typical_real, col_prev_choice_correct="PrevChoiceCorrect")[0]
    slow_real_mean = quantilePrevOutcomeCur(
        slow_real, col_prev_choice_correct="PrevChoiceCorrect")[0]

    # Prev outcome Count RT:
    prev_outcome_count_rt_real = _prevOutcomeCountRT(
        subject_df_real, "calcStimulusTime", col_prefix="")
    prev_outcome_count_rt_sim = _prevOutcomeCountRT(
        fitted_df, "SimRT", col_prefix="Sim")
    assert len(prev_outcome_count_rt_real) == len(prev_outcome_count_rt_sim)
    row["R2_PrevOutcomeCount"] = _calcR2(prev_outcome_count_rt_real["RTMean"],
                                         prev_outcome_count_rt_sim["RTMean"])

    subject_df_sim = fitted_df
    fast_model, typical_sim, slow_model = _dvQuantileFn(
        subject_df_sim, "SimRT", as_df=True)
    fast_model_mean = quantilePrevOutcomeCur(
        fast_model, col_prev_choice_correct="SimPrevChoiceCorrect")[0]
    typical_model_mean = quantilePrevOutcomeCur(
        typical_sim, col_prev_choice_correct="SimPrevChoiceCorrect")[0]
    slow_model_mean = quantilePrevOutcomeCur(
        slow_model, col_prev_choice_correct="SimPrevChoiceCorrect")[0]
    row["PrevOutcomeCurQuantileReal"] = fast_real_mean - slow_real_mean
    row["PrevOutcomeCurQuantileModel"] = fast_model_mean - slow_model_mean
    row["R2_PrevOutcomeQuantile"] = _calcR2(
        np.asanyarray([fast_real_mean, typical_real_mean, slow_real_mean]),
        np.asanyarray([fast_model_mean, typical_model_mean, slow_model_mean]))

    reward_rate_real = _reward_rate_series(subject, fitted_df, "",
                                           "calcStimulusTime")
    reward_rate_sim = _reward_rate_series(subject, fitted_df, "Sim", "SimRT")
    common_bins = reward_rate_real.index.intersection(reward_rate_sim.index)
    row["R2_RewardRate"] = _calcR2(reward_rate_real.loc[common_bins],
                                   reward_rate_sim.loc[common_bins])
    row["RewardRate5Real"] = reward_rate_real
    row["RewardRate5Model"] = reward_rate_sim
    row["RewardRateCorr"] = reward_rate_real.loc[common_bins].corr(
        reward_rate_sim.loc[common_bins])

    for subkey, bias_df_real, bias_df_sim in (("All", subject_df_real, subject_df_sim),
                                              ("Fast", fast_real, fast_model),
                                              ("Slow", slow_real, slow_model)):
        groupby_col = ["Name"]
        bias_real = calcBias(bias_df_real, "ChoiceLeft",
                             groupby_cols=groupby_col).unstack()
        bias_sim = calcBias(bias_df_sim, "SimChoiceLeft",
                            groupby_cols=groupby_col).unstack()
        assert len(bias_real) == len(bias_sim) == 1
        row[f"MotorBias{subkey}Real"] = bias_real.iloc[0]
        row[f"MotorBias{subkey}Model"] = bias_sim.iloc[0]

    data_slow_real, data_fast_real = [], []
    data_slow_model, data_fast_model = [], []
    for is_abs in [True, False]:
        _fit = lambda df, left, choice: _fitData(  # noqa: E731 — local shorthand
            df, is_abs=is_abs, left_col=left, choice_col=choice,
            n_psych_fits=n_psych_fits)
        psych_real_fast = _fit(fast_real, "ChoiceLeft", "ChoiceCorrect")
        psych_real_slow = _fit(slow_real, "ChoiceLeft", "ChoiceCorrect")
        psych_model_fast = _fit(fast_model, "SimChoiceLeft", "SimChoiceCorrect")
        psych_model_slow = _fit(slow_model, "SimChoiceLeft", "SimChoiceCorrect")

        diff_real = psych_real_slow["Perf"] - psych_real_fast["Perf"]
        diff_model = psych_model_slow["Perf"] - psych_model_fast["Perf"]
        if is_abs:
            row["PsychAbsReal"] = diff_real.sum()
            row["PsychAbsModel"] = diff_model.sum()
        else:
            row["PsychReal"] = diff_real.sum()
            row["PsychModel"] = diff_model.sum()
            row["R2_Psych"] = _calcR2(diff_real, diff_model)
            data_fast_real = psych_real_fast["Perf"]
            data_slow_real = psych_real_slow["Perf"]
            data_fast_model = psych_model_fast["Perf"]
            data_slow_model = psych_model_slow["Perf"]

    row["R2_Slow"] = _calcR2(data_slow_real, data_slow_model)
    row["R2_Fast"] = _calcR2(data_fast_real, data_fast_model)
    row["R2_Total"] = row["R2_Fast"] + row["R2_Slow"]

    row["WinStayReal"] = win_update_real
    row["WinStayModel"] = win_update_model
    row["LoseSwitchReal"] = -lose_update_real
    row["LoseSwitchModel"] = -lose_update_model
    return row


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------
def collect_metrics(fits, specs, df_behavior, *, num_evaluations=1,
                    n_psych_fits=N_PSYCH_FITS,
                    min_num_trials=MIN_NUM_TRIALS, sim_cache=None,
                    verbose=True):
    """Evaluate every ``spec`` × subject × iteration into one tidy dataframe.

    Returns one row per ``(spec, subject, iteration)`` with a ``SpecLabel`` /
    ``Name`` / ``Iteration`` / ``Seed`` key and every :func:`subject_metrics`
    metric.

    Subjects with fewer than ``min_num_trials`` simulated trials are dropped
    before their metrics are computed. The notebook applied this filter at plot
    time instead (``_filterDF`` on ``NumTrials``); both key off ``len(sim_df)``,
    so the surviving rows are the same set — doing it here just avoids
    simulating a subject ``num_evaluations`` times only to discard it. Pass
    ``min_num_trials=0`` to keep every subject.

    ``num_evaluations`` repeats each subject's simulation under seeds
    ``0..N-1``; ``N == 1`` uses seed 0 alone, matching the historical result.

    ``sim_cache`` (a dict, when given) collects ``{spec.label: {subject:
    (loss, fitted_df, BOUND, include_Q, include_RewardRate)}}`` for the
    ``seed=0`` run only — the shape the notebook's downstream per-subject
    figure cells consume. Later iterations are reduced to their metric row and
    their dataframes dropped, so memory stays flat in ``num_evaluations``.
    """
    if num_evaluations < 1:
        raise ValueError(f"num_evaluations must be >= 1, got {num_evaluations}")
    rows = []
    for spec in specs:
        resolved = resolve_spec(fits, spec)
        if verbose:
            print(f"Evaluating {spec.label!r} "
                  f"({spec.model_key} / {spec.column_label}) — "
                  f"{len(resolved)} subject(s) × {num_evaluations} eval(s)")
        for subject in sorted(resolved):
            col_fit = resolved[subject]
            include_Q, include_RewardRate = compare._include_flags(col_fit.payload)
            if include_Q is None or include_RewardRate is None:
                raise KeyError(
                    f"Fit {col_fit.filename!r} / subject {subject!r} is missing "
                    f"the include_Q / include_RewardRate flags; without them the "
                    f"simulation would silently run the wrong model.")
            for iteration in range(num_evaluations):
                seed = iteration
                sim_df, bound, _bias_kwargs, error = compare._compute_sim(
                    subject, col_fit.fid, col_fit.payload, df_behavior,
                    include_Q=include_Q, include_RewardRate=include_RewardRate,
                    seed=seed)
                if error is not None:
                    print(f"  {spec.label} / {subject} / seed {seed}: {error}")
                    continue
                if len(sim_df) < min_num_trials:
                    print(f"  Subject {subject} has less than {min_num_trials} "
                          f"trials: {len(sim_df)}")
                    break
                if iteration == 0 and sim_cache is not None:
                    loss = getattr(col_fit.payload.get("OptimRes"), "fun", np.nan)
                    sim_cache.setdefault(spec.label, {})[subject] = (
                        loss, sim_df, bound, include_Q, include_RewardRate)
                row = subject_metrics(subject, sim_df,
                                      n_psych_fits=n_psych_fits)
                row.update(SpecLabel=spec.label, ModelKey=spec.model_key,
                           ColumnLabel=spec.column_label,
                           Iteration=iteration, Seed=seed)
                rows.append(row)
    if not rows:
        raise ValueError("No subject×model combination produced metrics — "
                         "every simulation errored or fell below "
                         f"min_num_trials={min_num_trials}.")
    lead = ["SpecLabel", "ModelKey", "ColumnLabel", "Name", "Iteration", "Seed",
            "NumTrials"]
    df = pd.DataFrame(rows)
    return df[lead + [c for c in df.columns if c not in lead]]


# --------------------------------------------------------------------------
# Disk cache
# --------------------------------------------------------------------------
def _spec_key(specs):
    """Stable digest of what the specs address — a guard against reusing a
    cache after a preset was edited under the same ``cache_name``."""
    ident = repr([(s.label, s.model_key, s.column_label) for s in specs])
    return hashlib.sha1(ident.encode()).hexdigest()[:12]


def _source_fit_paths(fits, specs, result_dir):
    """Absolute paths of the fit pickles ``specs`` resolve to.

    ``ColumnFit`` carries only the basename, so the directory it was discovered
    from has to be supplied to stat the files for the freshness check.
    """
    result_dir = pathlib.Path(result_dir)
    paths = set()
    for spec in specs:
        for col_fit in resolve_spec(fits, spec).values():
            paths.add(result_dir / col_fit.filename)
    return sorted(paths)


def _cache_invalid_reason(payload, cache_fp, *, spec_key, num_evaluations,
                          n_psych_fits, source_paths):
    """Why the cache can't serve this request, or ``None`` if it can."""
    if payload.get("spec_key") != spec_key:
        return "the specs changed since it was written"
    if payload.get("n_psych_fits") != n_psych_fits:
        return (f"it was built with n_psych_fits="
                f"{payload.get('n_psych_fits')}, not {n_psych_fits}")
    cached_n = payload.get("num_evaluations", 0)
    if cached_n < num_evaluations:
        # Rows are seeds 0..cached_n-1, so a bigger request needs new seeds.
        # A smaller one is served fine — the plot subsamples.
        return f"it holds {cached_n} evaluation(s), fewer than {num_evaluations}"
    cache_mtime = cache_fp.stat().st_mtime
    newer = [p.name for p in source_paths
             if p.exists() and p.stat().st_mtime > cache_mtime]
    if newer:
        return f"these fits are newer than it: {', '.join(sorted(newer)[:3])}"
    return None


def load_or_collect_metrics(fits, specs, df_behavior, *, cache_name,
                            num_evaluations=1, n_psych_fits=N_PSYCH_FITS,
                            cache_dir=DEFAULT_METRICS_CACHE_DIR,
                            result_dir=compare.DEFAULT_RESULT_DIR,
                            force_recompute=False, sim_cache=None,
                            verbose=True, **collect_kwargs):
    """:func:`collect_metrics` with an on-disk cache.

    Collecting a large ``num_evaluations`` costs minutes-to-tens-of-minutes, so
    it should survive a kernel restart. ``cache_name`` is a short caller-chosen
    slug (``"fig1l"``, ``"scale_bound"``, …) naming
    ``{cache_dir}/metrics_{cache_name}.pkl``.

    The cache is rejected — and everything recollected — when
    ``force_recompute`` is set, the file is missing or unreadable, the specs or
    ``n_psych_fits`` changed, it holds fewer evaluations than asked for, or any
    source fit pickle is newer than it. The reason is always printed: a silent
    re-run here costs half an hour.

    ``sim_cache`` is only filled by an actual collection — a cache hit does no
    simulation and therefore has no per-subject frames to give. Callers that
    need them (the per-subject figure cells) must check whether it is empty.
    """
    cache_fp = pathlib.Path(cache_dir) / f"metrics_{cache_name}.pkl"
    spec_key = _spec_key(specs)
    source_paths = _source_fit_paths(fits, specs, result_dir)

    reason = None
    if force_recompute:
        reason = "force_recompute=True"
    elif not cache_fp.exists():
        reason = "no cache yet"
    else:
        try:
            with cache_fp.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 — a bad cache must not be fatal
            reason = f"it could not be read ({exc!r})"
        else:
            reason = _cache_invalid_reason(
                payload, cache_fp, spec_key=spec_key,
                num_evaluations=num_evaluations, n_psych_fits=n_psych_fits,
                source_paths=source_paths)
            if reason is None:
                if verbose:
                    print(f"Loaded {cache_fp} "
                          f"({payload['num_evaluations']} evaluation(s); "
                          f"using {num_evaluations}).")
                return payload["metrics_df"]

    if verbose:
        print(f"Collecting {num_evaluations} evaluation(s) — {reason}. "
              f"This is the slow path.")
    metrics_df = collect_metrics(fits, specs, df_behavior,
                                 num_evaluations=num_evaluations,
                                 n_psych_fits=n_psych_fits,
                                 sim_cache=sim_cache, verbose=verbose,
                                 **collect_kwargs)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    with cache_fp.open("wb") as f:
        pickle.dump({"metrics_df": metrics_df, "spec_key": spec_key,
                     "num_evaluations": num_evaluations,
                     "n_psych_fits": n_psych_fits,
                     "source_files": [p.name for p in source_paths]}, f)
    if verbose:
        print(f"Wrote {cache_fp}")
    return metrics_df


def per_spec_frames(metrics_df, iteration=0):
    """``{spec.label: one-row-per-subject frame}`` for a single iteration.

    The per-subject-scatter figures (correlation grids etc.) want one frame per
    model with a row per subject — the shape the notebook's ``collectMetric``
    returned. This is that view of the tidy frame, defaulting to the ``seed=0``
    iteration so those figures are unaffected by repeat evaluation.
    """
    sub = metrics_df[metrics_df.Iteration == iteration]
    return {label: frame.reset_index(drop=True)
            for label, frame in sub.groupby("SpecLabel", sort=False)}


def rows_for_spec_from(metrics_df, spec):
    """Rows of an existing metrics frame matching ``spec``'s fit, relabeled.

    Lets one figure reuse a column another figure already collected — e.g. the
    MLE-weight comparison prepending the Chi²-only fit that Fig. 1l already
    evaluated — instead of re-simulating it. The match is on
    ``(ModelKey, ColumnLabel)`` (a fit's identity, independent of the
    ``SpecLabel`` it was collected under); the returned copy's ``SpecLabel`` is
    set to ``spec.label`` so it slots into ``spec``'s bar group.
    """
    mask = ((metrics_df.ModelKey == spec.model_key) &
            (metrics_df.ColumnLabel == spec.column_label))
    got = metrics_df[mask].copy()
    if got.empty:
        available = sorted(set(zip(metrics_df.ModelKey, metrics_df.ColumnLabel)))
        raise KeyError(
            f"The given metrics frame has no rows for model "
            f"{spec.model_key!r} / column {spec.column_label!r} "
            f"(spec {spec.label!r}). Available (model, column) pairs: "
            f"{available}")
    got["SpecLabel"] = spec.label
    return got


# Characters no filesystem tolerates in a name, plus the newlines/tabs that
# EvalSpec labels carry for their two-line plot legends (e.g.
# "RewardRate\n+ Q-Val"). Collapsed to a single space so a label can be used
# directly as a file name.
_FILENAME_ILLEGAL = re.compile(r'[<>:"/\\|?*\r\n\t]+')


def safe_filename(name):
    """An ``EvalSpec.label`` sanitized into a single-line, filesystem-safe token.

    The labels embed newlines for their plot legends; those reach a save path
    as-is unless stripped, and a newline in a filename raises
    ``OSError: [Errno 22] Invalid argument`` on Windows. Use this wherever a
    label becomes (part of) a file name.
    """
    return _FILENAME_ILLEGAL.sub(" ", str(name).replace("·", "-")).strip()


# --------------------------------------------------------------------------
# Figure presets
# --------------------------------------------------------------------------
# The model_key format is ``{drift_alias}|{bias}|{noise}|{t_dur}|{dt}|{asym}``
# (``mle_reeval.FitFileId.model_key``). Everything the paper compares is a
# symmetric, 4.8s, dt=0.005, Normal(0, 1) fit.
def model_key(drift_alias, bias, *, noise="Normal(0, 1)", t_dur=4.8, dt=0.005,
              asym="sym"):
    return f"{drift_alias}|{bias}|{noise}|{t_dur:g}|{dt:g}|{asym}"


_Q = "Q-Val (Offset)"
_NO_BIAS = "None_"

# Chi²-Noise == no _scaledB == fitted noise, fixed threshold.
COL_CHI2_NOISE = "Chi²-Noise"
COL_CHI2_BOUND = "Chi²-Bound"
COL_MLE = "MLE"


# Fig. 1l — the four models, all from their Chi² fits. Reward-rate is the
# noise-based variant; Classic likewise is the noise-based, fixed-threshold fit.
FIG1L_SPECS = (
    EvalSpec("Classic", model_key("Classic", _NO_BIAS), COL_CHI2_NOISE, "gray"),
    EvalSpec("Q-Val", model_key("Classic", _Q), COL_CHI2_NOISE, "red"),
    EvalSpec("RewardRate", model_key("RewardRate", _NO_BIAS), COL_CHI2_NOISE, "blue"),
    EvalSpec("RewardRate\n+ Q-Val", model_key("RewardRate", _Q), COL_CHI2_NOISE, "green"),
)

# Reward-rate implemented as noise-gain (fixed threshold) vs. as a scaled bound
# (fixed noise), without and with Q-Val. The half-column gap separates the
# no-Q-val pair from the +Q-val pair.
SCALE_BOUND_SPECS = (
    EvalSpec("RewardRate (Noise)\nFixed Threshold",
             model_key("RewardRate", _NO_BIAS), COL_CHI2_NOISE, "blue"),
    EvalSpec("RewardRate (Scale-Bound)\nFixed Noise",
             model_key("RewardRate", _NO_BIAS), COL_CHI2_BOUND, "purple",
             gap_after=0.5),
    EvalSpec("RewardRate (Noise)\nFixed Threshold + Q-Val",
             model_key("RewardRate", _Q), COL_CHI2_NOISE, "green"),
    EvalSpec("RewardRate (Scale-Bound)\nFixed Noise + Q-Val",
             model_key("RewardRate", _Q), COL_CHI2_BOUND, "olive"),
)

# One model (reward-rate noise-gain + Q-Val), three fitting criteria.
MLE_WEIGHT_SPECS = (
    EvalSpec("Pure MLE", model_key("RewardRate", _Q), COL_MLE, "darkorange"),
    EvalSpec("MLE + Chi²=0.1", model_key("RewardRate", _Q),
             "MLE=1, Chi²=0.1", "chocolate"),
    EvalSpec("MLE + Chi²=0.5", model_key("RewardRate", _Q),
             "MLE=1, Chi²=0.5", "sienna"),
)

# The Chi²-only fit of the MLE_WEIGHT_SPECS model — its Chi²-Noise column.
# This is the same (model_key, column_label) as FIG1L_SPECS[3], so the MLE
# figure can prepend it as a reference column by borrowing the rows already
# collected for Fig. 1l (see rows_for_spec_from) instead of re-simulating.
CHI2_ONLY_RRQ_SPEC = EvalSpec("Chi²-only", model_key("RewardRate", _Q),
                              COL_CHI2_NOISE, "gray")
