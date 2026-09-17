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
from .initvals import MLE_TERMINAL_C
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
# Subject selection
# --------------------------------------------------------------------------
# Fitted from ``data/RLModel/df_2p_missing.pkl`` purely so
# ``model_neural_correlate.ipynb`` has model latents for every imaged animal.
# They are NOT part of the behavioral cohort: they are absent from
# ``data/behavior/df_behavior.pkl``, so anything in ``model_analysis.ipynb``
# that touches them is scoring a fit against no trials. They still appear
# inside the fit pickles, hence ``resolve_spec`` finds them and they have to be
# dropped explicitly rather than by accident.
IMAGING_ONLY_SUBJECTS = ("GP4-23", "GP4-28")


def select_subjects(df, *, exclude=(), keep=None, column="Name",
                    what="rows", verbose=True):
    """Subject filter for an already-built frame — a view, never a recompute.

    ``exclude`` drops the named subjects; ``keep`` (when not ``None``) restricts
    to the named ones. Both are applied, ``exclude`` first. A name in
    ``exclude`` that is not in the frame is reported rather than ignored: the
    usual cause is a typo in a subject id (``GP-23`` for ``GP4-23``), which
    would otherwise silently filter nothing.

    This is deliberately a *post*-filter and never part of a cache key. The
    collected frames cost hours; making the excluded set a cache input would
    mean every edit to that list threw them away, and would leave a cache whose
    contents depend on a parameter it does not record. Collect the superset
    once, filter on the way out.
    """
    exclude = tuple(exclude or ())
    present = set(df[column].unique())
    if exclude:
        missing = [s for s in exclude if s not in present]
        if missing and verbose:
            print(f"select_subjects: nothing to exclude for {missing} — not in "
                  f"the {what} (present: {sorted(present)}).")
        hit = [s for s in exclude if s in present]
        if hit:
            df = df[~df[column].isin(hit)]
            if verbose:
                print(f"Excluded {hit} from the {what}.")
    if keep is not None:
        keep = set(keep)
        dropped = sorted(set(df[column].unique()) - keep)
        df = df[df[column].isin(keep)]
        if dropped and verbose:
            print(f"Restricted the {what} to {len(keep)} subject(s); "
                  f"dropped {dropped}.")
    # The original index is preserved deliberately. This filter is also applied
    # to df_behavior, where a silent reindex would be a behavior change; the
    # frames that want 0..n-1 (per_spec_frames, the concat in the MLE cell)
    # reset it themselves.
    return df


def select_sim_cache_subjects(sim_cache, *, exclude=(), verbose=True):
    """:func:`select_subjects` for a ``sim_cache`` — ``{spec: {subject: …}}``.

    Mutates in place (the notebook holds the dict the collection filled) and
    returns it, so the per-subject figure cells that iterate it never see an
    excluded animal.
    """
    exclude = set(exclude or ())
    if not exclude:
        return sim_cache
    for label, subjects_dict in sim_cache.items():
        for subject in exclude & set(subjects_dict):
            del subjects_dict[subject]
            if verbose:
                print(f"Excluded {subject!r} from the sim_cache of {label!r}.")
    return sim_cache


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalSpec:
    """One model to evaluate == one bar group in an aggregate figure.

    ``model_key`` / ``column_label`` address a fit in the coordinate space of
    ``compare.discover_fits``: the key is the abstract model
    (``FitFileId.model_key`` — drift alias + bias + noise + timing) and
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
    return frame_from_rows(rows)


# The key columns, in the order they lead every metrics frame. Shared with the
# sharded cluster collection (``metrics_shards.merge_shards``) so a merged frame
# is column-identical to a locally collected one.
LEAD_COLUMNS = ["SpecLabel", "ModelKey", "ColumnLabel", "Name", "Iteration",
                "Seed", "NumTrials"]


def frame_from_rows(rows):
    """``[row dict, …]`` → the tidy metrics frame, key columns first."""
    df = pd.DataFrame(rows)
    return df[LEAD_COLUMNS + [c for c in df.columns if c not in LEAD_COLUMNS]]


# --------------------------------------------------------------------------
# Pure-MLE loss per subject
# --------------------------------------------------------------------------
# What a fit pickle actually stores is ``OptimRes.fun`` -- the value of the
# objective THAT fit minimized. That is a pure negative log-likelihood only for
# the "MLE" column (Chi2 weight 0); for a joint column it also carries the
# weighted Chi2 term, and for a Chi2 column it is a Chi2 statistic, not a
# likelihood at all. So it is NOT comparable across the fitting-criterion
# figure's bars.
#
# The comparable quantity is every column's fitted params re-scored under the
# SAME pure-MLE objective (``compare.mle_eval_result``, the grid header's
# "MLE-Score"). That is what is collected here, together with the valid-trial
# count that entered the sum -- an unnormalized negative log-likelihood is a
# sum over trials, so a subject with twice the trials contributes roughly twice
# the loss and the across-subject mean would mostly measure session counts.
MLE_LOSS_LEAD_COLUMNS = ["SpecLabel", "ModelKey", "ColumnLabel", "Name",
                         "NegLogLik", "NTrialsLoss", "NegLogLikPerTrial"]

# Nullable integers: a subject whose re-evaluation errored has no trial count,
# and a plain int64 column cannot hold that gap without silently going float.
_MLE_LOSS_INT_COLUMNS = ("NTrialsLoss", "NTrialsTotal")


def _fit_objective(payload):
    """``OptimRes.fun`` -- the value of the objective the fit itself minimized.

    Reported alongside the re-scored likelihood as a provenance check: for a
    pure-MLE column the two should agree (same objective, same params), while
    for a joint or Chi2 column they are different quantities by construction.
    """
    return float(getattr(payload.get("OptimRes"), "fun", np.nan))


def collect_mle_losses(fits, specs, df_behavior, *,
                       mle_terminal_c=MLE_TERMINAL_C.Default,
                       lapse_override=None, verbose=True):
    """Score every ``spec`` x subject under the pure-MLE objective.

    Returns one row per ``(spec, subject)`` -- no ``Iteration``/``Seed``, unlike
    :func:`collect_metrics`: the likelihood is a deterministic function of the
    fitted params and the subject's trials, so there is no trajectory to repeat.

    Columns:

    ``NegLogLik``
        The pure negative log-likelihood of the column's fitted params, summed
        over the subject's valid trials -- the "fit value before
        normalization".
    ``NTrialsLoss`` / ``NTrialsTotal``
        Trials that actually entered that sum (valid and finite-loglik) and the
        subject's total trials.
    ``NegLogLikPerTrial``
        ``NegLogLik / NTrialsLoss`` -- the "fit value under normalization", and
        the only one comparable between subjects of different lengths.
    ``FitObjective``
        The fit's own stored ``OptimRes.fun`` (see :func:`_fit_objective`).
    ``Error``
        ``None``, or the failure that left this subject's loss NaN. One bad
        subject degrades its row rather than the whole collection, matching
        ``compare``'s per-column error handling.

    ``mle_terminal_c`` must match what the fits were run under (the fitting
    default, hence the default here) or the re-scored likelihood is of a
    different objective than the one that was minimized. ``lapse_override``
    substitutes a lapse rate for every column; the natural use is giving a Chi2
    fit -- whose params dict has no ``LAPSE_RATE``, so lambda defaults to 0 --
    the same lapse as the MLE fits it is being compared against.
    """
    rows = []
    for spec in specs:
        resolved = resolve_spec(fits, spec)
        if verbose:
            print(f"MLE loss for {spec.label!r} "
                  f"({spec.model_key} / {spec.column_label}) — "
                  f"{len(resolved)} subject(s)")
        for subject in sorted(resolved):
            col_fit = resolved[subject]
            include_Q, include_RewardRate = compare._include_flags(col_fit.payload)
            if include_Q is None or include_RewardRate is None:
                raise KeyError(
                    f"Fit {col_fit.filename!r} / subject {subject!r} is missing "
                    f"the include_Q / include_RewardRate flags; without them the "
                    f"likelihood would silently score the wrong model.")
            res, _lapse, error = compare.mle_eval_result(
                subject, col_fit.fid, col_fit.payload, df_behavior,
                include_Q, include_RewardRate, mle_terminal_c, lapse_override)
            row = {"SpecLabel": spec.label, "ModelKey": spec.model_key,
                   "ColumnLabel": spec.column_label, "Name": subject,
                   "FitObjective": _fit_objective(col_fit.payload),
                   "Filename": col_fit.filename, "Error": error}
            if res is None:
                print(f"  {spec.label} / {subject}: {error}")
                row.update(NegLogLik=np.nan, NTrialsLoss=pd.NA,
                           NTrialsTotal=pd.NA, NegLogLikPerTrial=np.nan)
            else:
                neg_loglik = float(res.neg_loglik)
                n_loss = int(res.n_trials_loss)
                if not n_loss:
                    # The likelihood of an empty frame is 0, not an error, so
                    # this would otherwise pass as a real (and best-possible)
                    # score. It means the subject is in the fit pickle but has
                    # no trials in df_behavior — the imaging-only animals (see
                    # IMAGING_ONLY_SUBJECTS) are exactly this case.
                    print(f"  WARNING: {spec.label} / {subject} scored on 0 "
                          f"valid trials — it is in the fit but has no trials "
                          f"in df_behavior. Recording NaN, not 0.")
                row.update(NegLogLik=neg_loglik, NTrialsLoss=n_loss,
                           NTrialsTotal=int(res.n_trials_total),
                           NegLogLikPerTrial=(neg_loglik / n_loss if n_loss
                                              else np.nan))
            rows.append(row)
    if not rows:
        raise ValueError("No subject×model combination produced an MLE loss.")
    return mle_loss_frame_from_rows(rows)


def mle_loss_frame_from_rows(rows):
    """``[row dict, …]`` → the tidy MLE-loss frame, key columns first."""
    df = pd.DataFrame(rows)
    df = df[MLE_LOSS_LEAD_COLUMNS +
            [c for c in df.columns if c not in MLE_LOSS_LEAD_COLUMNS]]
    for col in _MLE_LOSS_INT_COLUMNS:
        df[col] = df[col].astype("Int64")
    return df


def mle_loss_summary(mle_losses, metric="NegLogLikPerTrial"):
    """Per-model mean ± SEM of ``metric`` over subjects.

    One row per ``SpecLabel``, in the frame's own group order, with ``Mean`` /
    ``SEM`` / ``NumSubjects``. Errored subjects (NaN ``metric``) drop out, so
    ``NumSubjects`` is the n the SEM was actually computed over. This is both
    what the bar figure annotates under each model title and the table to print
    beside it, so the two cannot disagree.
    """
    scored = mle_losses.dropna(subset=[metric])
    grouped = scored.groupby("SpecLabel", sort=False)[metric]
    # Deliberately plain int64, not the nullable Int64 the frame's trial counts
    # use: a masked column here would drag a whole ``.loc[label]`` row into
    # pandas' Float64 dtype, turning the n==1 SEM into ``pd.NA`` — which the
    # float formatting downstream cannot take. No group can have a NA size.
    return pd.DataFrame({"Mean": grouped.mean(), "SEM": grouped.sem(),
                         "NumSubjects": grouped.size()})


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


# How a rejected cache can be rejected. ``STRUCTURAL`` means the frame does not
# contain the asked-for rows at all, so there is nothing to fall back on;
# ``STALE`` means the rows are all there and self-consistent, just older than a
# fit pickle. Only the second is safe to serve under ``require_cache`` — see
# :func:`load_or_collect_metrics`.
CACHE_STRUCTURAL = "structural"
CACHE_STALE = "stale"


def _cache_invalid_reason(payload, cache_fp, *, spec_key, num_evaluations,
                          n_psych_fits, source_paths):
    """``(kind, reason)`` the cache can't serve this request, or ``None``."""
    if payload.get("spec_key") != spec_key:
        return CACHE_STRUCTURAL, "the specs changed since it was written"
    if payload.get("n_psych_fits") != n_psych_fits:
        return CACHE_STRUCTURAL, (f"it was built with n_psych_fits="
                                  f"{payload.get('n_psych_fits')}, not "
                                  f"{n_psych_fits}")
    cached_n = payload.get("num_evaluations", 0)
    if cached_n < num_evaluations:
        # Rows are seeds 0..cached_n-1, so a bigger request needs new seeds.
        # A smaller one is served fine — the plot subsamples.
        return CACHE_STRUCTURAL, (f"it holds {cached_n} evaluation(s), fewer "
                                  f"than {num_evaluations}")
    newer = _newer_source_fits(cache_fp, source_paths)
    if newer:
        return CACHE_STALE, f"these fits are newer than it: {', '.join(newer[:3])}"
    return None


def _newer_source_fits(cache_fp, source_paths):
    """Names of the fit pickles modified after ``cache_fp`` was written.

    Shared by the metrics and the MLE-loss caches: both are derived products of
    the same pickles, so a refit must invalidate either of them.
    """
    cache_mtime = cache_fp.stat().st_mtime
    return sorted(p.name for p in source_paths
                  if p.exists() and p.stat().st_mtime > cache_mtime)


def sim_cache_path(cache_name, cache_dir=DEFAULT_METRICS_CACHE_DIR):
    """Sidecar path holding the ``seed=0`` simulation frames for a metrics cache.

    Kept beside ``metrics_{cache_name}.pkl`` rather than inside it: the frames
    are trial-level and one to two orders of magnitude bigger than the metric
    rows, and only the per-subject figure cells want them.
    """
    return pathlib.Path(cache_dir) / f"simcache_{cache_name}.pkl"


def save_sim_cache(sim_cache, cache_name, cache_dir=DEFAULT_METRICS_CACHE_DIR):
    """Persist a ``collect_metrics`` sim cache next to its metrics cache."""
    fp = sim_cache_path(cache_name, cache_dir)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("wb") as f:
        pickle.dump(sim_cache, f)
    return fp


def load_sim_cache(cache_name, cache_dir=DEFAULT_METRICS_CACHE_DIR):
    """The persisted sim cache, or ``{}`` when there is none / it is unreadable.

    Unreadable is not fatal: the frames are a convenience for the per-subject
    figures, and the metrics — the thing the paper's bars are made of — live in
    a separate file.
    """
    fp = sim_cache_path(cache_name, cache_dir)
    if not fp.exists():
        return {}
    try:
        with fp.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:  # noqa: BLE001 — a bad sidecar must not be fatal
        print(f"Could not read {fp}: {exc!r}")
        return {}


class CacheUnavailable(RuntimeError):
    """``require_cache`` was set and the cache genuinely cannot serve the request.

    Raised instead of silently starting a collection that costs hours. The
    message names the reason and what to change.
    """


def _require_cache_check(force_recompute, require_cache):
    if force_recompute and require_cache:
        raise ValueError(
            "force_recompute=True and require_cache=True contradict each "
            "other: the first demands a fresh collection, the second forbids "
            "one. Pick whichever you meant.")


def _resolve_cache_reason(kind, reason, *, cache_fp, require_cache, payload,
                          verbose):
    """Under ``require_cache``, decide whether a rejected cache is still usable.

    Returns ``True`` to serve ``payload`` anyway. A ``CACHE_STALE`` rejection
    means every asked-for row is present and internally consistent — only a fit
    pickle's mtime moved — so serving it is a defensible, loudly-announced
    choice. A ``CACHE_STRUCTURAL`` one means the rows are not there, and no
    flag can conjure them, so it raises.
    """
    if not require_cache:
        return False
    if kind == CACHE_STALE and payload is not None:
        if verbose:
            print(f"WARNING: {cache_fp} is stale ({reason}), but "
                  f"require_cache=True — using it as-is rather than "
                  f"recollecting. Clear require_cache to rebuild it.")
        return True
    raise CacheUnavailable(
        f"require_cache=True but {cache_fp} cannot serve this request: "
        f"{reason}. Nothing was recollected. Either fix the mismatch (see the "
        f"reason above) or pass require_cache=False to pay for a collection.")


def load_or_collect_metrics(fits, specs, df_behavior, *, cache_name,
                            num_evaluations=1, n_psych_fits=N_PSYCH_FITS,
                            cache_dir=DEFAULT_METRICS_CACHE_DIR,
                            result_dir=compare.DEFAULT_RESULT_DIR,
                            force_recompute=False, require_cache=False,
                            exclude_subjects=(), sim_cache=None,
                            sim_cache_file=True,
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

    ``require_cache=True`` forbids that re-run. A cache rejected only for
    staleness (a fit pickle is newer, but every asked-for row is present) is
    used anyway with a loud warning; one rejected structurally — missing,
    unreadable, wrong specs, too few evaluations — raises
    :class:`CacheUnavailable` rather than quietly starting an hours-long
    collection. Use it when re-collecting is not an option this session.

    ``exclude_subjects`` drops those subjects from the returned frame and from
    ``sim_cache``. It is applied on the way *out* and is deliberately not part
    of the cache key, so editing the list never invalidates a collection — see
    :func:`select_subjects`.

    ``sim_cache`` collects the ``seed=0`` per-subject frames. A cache hit does
    no simulation and so has none to give — with ``sim_cache_file`` (the
    default) they are instead persisted to / restored from the
    :func:`sim_cache_path` sidecar, so the per-subject figure cells keep working
    across a kernel restart and after a cluster run (see
    ``model/metrics_shards.py``). Pass ``sim_cache_file=False`` to skip writing
    it. Callers should still tolerate an empty ``sim_cache``: the sidecar is
    absent for caches collected before it existed.
    """
    _require_cache_check(force_recompute, require_cache)
    cache_fp = pathlib.Path(cache_dir) / f"metrics_{cache_name}.pkl"
    spec_key = _spec_key(specs)
    source_paths = _source_fit_paths(fits, specs, result_dir)

    def _serve(metrics_df):
        """Restore the sidecar frames, apply the subject filter, return."""
        if sim_cache is not None and sim_cache_file:
            restored = load_sim_cache(cache_name, cache_dir)
            sim_cache.update(restored)
            if verbose and restored:
                print(f"Restored {sum(len(v) for v in restored.values())}"
                      f" seed-0 simulation frame(s) from "
                      f"{sim_cache_path(cache_name, cache_dir)}.")
        if sim_cache is not None:
            select_sim_cache_subjects(sim_cache, exclude=exclude_subjects,
                                      verbose=verbose)
        return select_subjects(metrics_df, exclude=exclude_subjects,
                               what=f"{cache_name} metrics", verbose=verbose)

    payload = None
    reason = kind = None
    if force_recompute:
        kind, reason = CACHE_STRUCTURAL, "force_recompute=True"
    elif not cache_fp.exists():
        kind, reason = CACHE_STRUCTURAL, "no cache yet"
    else:
        try:
            with cache_fp.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 — a bad cache must not be fatal
            kind, reason = CACHE_STRUCTURAL, f"it could not be read ({exc!r})"
        else:
            verdict = _cache_invalid_reason(
                payload, cache_fp, spec_key=spec_key,
                num_evaluations=num_evaluations, n_psych_fits=n_psych_fits,
                source_paths=source_paths)
            if verdict is None:
                if verbose:
                    print(f"Loaded {cache_fp} "
                          f"({payload['num_evaluations']} evaluation(s); "
                          f"using {num_evaluations}).")
                return _serve(payload["metrics_df"])
            kind, reason = verdict

    if _resolve_cache_reason(kind, reason, cache_fp=cache_fp,
                             require_cache=require_cache, payload=payload,
                             verbose=verbose):
        return _serve(payload["metrics_df"])

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
    if sim_cache and sim_cache_file:
        fp = save_sim_cache(sim_cache, cache_name, cache_dir)
        if verbose:
            print(f"Wrote {fp}")
    # The cache above is written UNFILTERED, so the excluded set stays a
    # display choice a later run can revisit without re-simulating.
    if sim_cache is not None:
        select_sim_cache_subjects(sim_cache, exclude=exclude_subjects,
                                  verbose=verbose)
    return select_subjects(metrics_df, exclude=exclude_subjects,
                           what=f"{cache_name} metrics", verbose=verbose)


def mle_loss_cache_path(cache_name, cache_dir=DEFAULT_METRICS_CACHE_DIR):
    """Where :func:`load_or_collect_mle_losses` keeps ``cache_name``'s frame."""
    return pathlib.Path(cache_dir) / f"mleloss_{cache_name}.pkl"


def _mle_loss_cache_invalid_reason(payload, cache_fp, *, spec_key,
                                   mle_terminal_c, lapse_override,
                                   source_paths):
    """``(kind, reason)`` the MLE-loss cache can't serve this, or ``None``."""
    if payload.get("spec_key") != spec_key:
        return CACHE_STRUCTURAL, "the specs changed since it was written"
    if payload.get("mle_terminal_c") != mle_terminal_c:
        return CACHE_STRUCTURAL, (f"it was scored with mle_terminal_c="
                                  f"{payload.get('mle_terminal_c')}, not "
                                  f"{mle_terminal_c}")
    if payload.get("lapse_override") != lapse_override:
        return CACHE_STRUCTURAL, (f"it was scored with lapse_override="
                                  f"{payload.get('lapse_override')}, not "
                                  f"{lapse_override}")
    newer = _newer_source_fits(cache_fp, source_paths)
    if newer:
        return CACHE_STALE, f"these fits are newer than it: {', '.join(newer[:3])}"
    return None


def load_or_collect_mle_losses(fits, specs, df_behavior, *, cache_name,
                               mle_terminal_c=MLE_TERMINAL_C.Default,
                               lapse_override=None,
                               cache_dir=DEFAULT_METRICS_CACHE_DIR,
                               result_dir=compare.DEFAULT_RESULT_DIR,
                               force_recompute=False, require_cache=False,
                               exclude_subjects=(), subjects=None,
                               verbose=True):
    """:func:`collect_mle_losses` with an on-disk cache.

    Same contract as :func:`load_or_collect_metrics` — a caller-chosen slug
    naming ``{cache_dir}/mleloss_{cache_name}.pkl``, rejected (with the reason
    printed) when ``force_recompute`` is set, the file is missing or unreadable,
    the specs or the scoring settings changed, or a source fit pickle is newer
    than it. ``require_cache`` likewise serves a merely-stale cache with a
    warning and raises :class:`CacheUnavailable` on a structural miss.

    ``exclude_subjects`` and ``subjects`` filter the returned frame on the way
    out and are not part of the cache key (see :func:`select_subjects`).
    ``subjects`` is how the annotation is kept honest: scoring runs over every
    subject in the fit files, but the bars it annotates are drawn over only
    those that cleared ``collect_metrics``' ``min_num_trials``. Pass that
    figure's ``metrics_df.Name.unique()`` and the two describe the same
    animals.
    """
    _require_cache_check(force_recompute, require_cache)
    cache_fp = mle_loss_cache_path(cache_name, cache_dir)
    spec_key = _spec_key(specs)
    source_paths = _source_fit_paths(fits, specs, result_dir)

    def _serve(mle_losses):
        return select_subjects(mle_losses, exclude=exclude_subjects,
                               keep=subjects,
                               what=f"{cache_name} MLE losses", verbose=verbose)

    payload = None
    reason = kind = None
    if force_recompute:
        kind, reason = CACHE_STRUCTURAL, "force_recompute=True"
    elif not cache_fp.exists():
        kind, reason = CACHE_STRUCTURAL, "no cache yet"
    else:
        try:
            with cache_fp.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 — a bad cache must not be fatal
            kind, reason = CACHE_STRUCTURAL, f"it could not be read ({exc!r})"
        else:
            verdict = _mle_loss_cache_invalid_reason(
                payload, cache_fp, spec_key=spec_key,
                mle_terminal_c=mle_terminal_c, lapse_override=lapse_override,
                source_paths=source_paths)
            if verdict is None:
                if verbose:
                    print(f"Loaded {cache_fp}.")
                return _serve(payload["mle_losses"])
            kind, reason = verdict

    if _resolve_cache_reason(kind, reason, cache_fp=cache_fp,
                             require_cache=require_cache, payload=payload,
                             verbose=verbose):
        return _serve(payload["mle_losses"])

    if verbose:
        print(f"Scoring every fit under pure MLE — {reason}.")
    mle_losses = collect_mle_losses(
        fits, specs, df_behavior, mle_terminal_c=mle_terminal_c,
        lapse_override=lapse_override, verbose=verbose)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    with cache_fp.open("wb") as f:
        pickle.dump({"mle_losses": mle_losses, "spec_key": spec_key,
                     "mle_terminal_c": mle_terminal_c,
                     "lapse_override": lapse_override,
                     "source_files": [p.name for p in source_paths]}, f)
    if verbose:
        print(f"Wrote {cache_fp}")
    # Written unfiltered, like the metrics cache: the filters are a view.
    return _serve(mle_losses)

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
# The model_key format is ``{drift_alias}|{bias}|{noise}|{t_dur}|{dt}|sym``
# (``mle_reeval.FitFileId.model_key``). Everything the paper compares is a
# 4.8s, dt=0.005, Normal(0, 1) fit.
def model_key(drift_alias, bias, *, noise="Normal(0, 1)", t_dur=4.8, dt=0.005):
    return f"{drift_alias}|{bias}|{noise}|{t_dur:g}|{dt:g}|sym"


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

# All four reward-rate CHANNELS side by side, without and with Q-Val: the
# reward rate modulating the noise (sigma *= r), the threshold
# (b = BOUND*(2-r)), or the drift (mu *= g(r), --use-drift-rr) under either
# g(r) mapping — 2-r (a high reward rate SLOWS decisions) and 1+r (it speeds
# them up, the same direction as the noise/threshold channels).
#
# Noise and Bound share one model_key and differ only by criterion column,
# because they are the same abstract model on two scale axes. Each Drift
# mapping is a DIFFERENT model, so each has its own model_key
# (``RewardRate (Drift)`` / ``RewardRate (Drift 1+r)``, see
# drift.display_alias_for_drift) and is read from its Chi²-Noise column —
# i.e. its fitted-noise / fixed-threshold fit, matching the scale-axis
# convention of the RR (Noise) bar next to it.
#
# Labels abbreviate RewardRate to "RR": eight bars, and the full name made
# the tick labels collide. Note ``EvalSpec.label`` is the join key between
# collection, plotting and the sim_cache (and part of the metrics cache
# digest), so renaming a bar re-collects it — it does not silently reuse
# another bar's rows.
#
# Deliberately its OWN tuple rather than extra entries on SCALE_BOUND_SPECS:
# that tuple is a 2x2 (noise-vs-bound x no-Q-vs-Q) whose gap_after encodes
# the pairing, and FIG1L_SPECS is positionally zipped by golden_fig1l.py.
# Those two keep the unabbreviated labels — they back paper figures.
_RR_DRIFT = "RewardRate (Drift)"
_RR_DRIFT_1P = "RewardRate (Drift 1+r)"
DRIFT_RR_SPECS = (
    EvalSpec("RR (Noise)",
             model_key("RewardRate", _NO_BIAS), COL_CHI2_NOISE, "blue"),
    EvalSpec("RR (Bound)",
             model_key("RewardRate", _NO_BIAS), COL_CHI2_BOUND, "purple"),
    EvalSpec("RR (Drift 2-r)",
             model_key(_RR_DRIFT, _NO_BIAS), COL_CHI2_NOISE, "crimson"),
    EvalSpec("RR (Drift 1+r)",
             model_key(_RR_DRIFT_1P, _NO_BIAS), COL_CHI2_NOISE, "orange",
             gap_after=0.5),
    EvalSpec("RR (Noise)\n+ Q-Val",
             model_key("RewardRate", _Q), COL_CHI2_NOISE, "green"),
    EvalSpec("RR (Bound)\n+ Q-Val",
             model_key("RewardRate", _Q), COL_CHI2_BOUND, "olive"),
    EvalSpec("RR (Drift 2-r)\n+ Q-Val",
             model_key(_RR_DRIFT, _Q), COL_CHI2_NOISE, "darkred"),
    EvalSpec("RR (Drift 1+r)\n+ Q-Val",
             model_key(_RR_DRIFT_1P, _Q), COL_CHI2_NOISE, "chocolate"),
)
