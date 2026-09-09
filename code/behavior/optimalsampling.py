'''Reward-optimal sampling time (Figure 2B).

Backend for the ``## Mixed`` section of ``behavior.ipynb``, which replaced two
earlier copies of the same analysis (one that never wrote its figures, one that
wrote a non-manuscript ``results/optimal_sampling_time.svg``).

The question is whether animals sample for about as long as maximises their
reward *rate*. Sampling longer buys accuracy but costs time, so there is an
interior optimum. Per animal:

1. **Speed-accuracy curve.** Trials are grouped by difficulty, binned by
   sampling time (0.05 s bins), and performance is averaged per bin; bins with
   fewer than ``MIN_BIN_COUNT`` trials are dropped. The three difficulties are
   averaged together and optionally Gaussian-smoothed.
2. **Fitted curve, Methods eq. (1).** The binned curve is fitted by non-linear
   least squares with a saturating function pinned to chance at t = 0::

       p(t) = 50 + (50 - lapse) * t**beta / (t**beta + alpha**beta)

   evaluated over all sampling times, including those the animal never used.
3. **Expected trial duration, Methods eq. (2).** A trial costs its sampling
   time plus the non-sampling overhead -- returning to the centre port,
   collecting reward, or serving the error timeout -- which differs after
   correct and incorrect choices::

       E[T|t] = (tau_C + t) * p(t)/100 + (tau_I + t) * (1 - p(t)/100)

4. **Reward rate, Methods eq. (3).** ``TPH(t) = 3600 / E[T|t]`` trials per
   hour, of which a fraction ``p(t)/100`` are rewarded, and the optimum is
   ``t* = argmax_t R(t)``.

Two panels come out of this. :func:`plotSubjectRewardCurve` is Figure 2B-left:
one animal's normalised reward-rate curve, coloured by predicted performance,
with ``t*`` marked and the observed sampling time (mean +- SD) overlaid.
:func:`plotAlignedPopulation` is Figure 2B-right: every animal as a row, its
performance curve shifted so its own ``t*`` sits at 0, so that the observed
times can be compared across animals on a common axis.

**The optimum is not always interior.** Eq. (3) maximises reward per *hour*,
not per trial, so sampling longer only pays if the accuracy it buys outruns the
time it costs. When it does not, ``t*`` sits at the shortest sampling time the
grid allows -- the model's advice is to guess immediately and run more trials.
This is a property of the equation rather than something seen in this dataset
(all 17 animals come out between 0.53 s and 1.88 s), but it matters when
reading a fit, and near the switch ``t*`` is unstable. See
``docs/manuscript-methods-map.md`` for the worked example.

**The two panels are not computed with the same smoothing.** The notebook runs
the population panel at ``smooth_window=1`` and re-runs the whole computation
per animal at ``smooth_window=4`` for the single-animal panels. That is
preserved here as ``POPULATION_SMOOTH_WINDOW`` / ``SUBJECT_SMOOTH_WINDOW``.

**Non-sampling overhead is measured on the current trial.** Trial duration is
``next(TrialStartSysTime) - TrialStartSysTime`` minus that trial's sampling
time, split by the current trial's outcome. The inline version carried a
``extra_time_mode="prev"`` switch for a legacy alignment that measured the gap
against the *previous* trial's stimulus time and outcome; nothing used it and
it is not reproduced.
'''
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, NamedTuple, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pandas.api.types import is_timedelta64_dtype
from scipy.optimize import curve_fit

from ..pipeline.utils import filterNanGaussianConserving

#: An animal needs more than this many trials to be included (Methods).
MIN_TRIALS = 1_000
#: Sampling-time bins for the speed-accuracy curve, in seconds.
BIN_STEP = 0.05
BIN_RANGE = (0.3, 5.0)
#: Bins with fewer than this many trials are dropped (Methods).
MIN_BIN_COUNT = 5
#: Longest sampling time the fitted curve is evaluated over, in seconds.
NUM_SECS = 5.0
#: Evaluation step for the single-animal panel and the population panel. The
#: population panel is coarser because it evaluates one row per animal.
SUBJECT_STEP = 0.001
POPULATION_STEP = 0.01
#: Gaussian smoothing applied to the binned performance curve. The notebook
#: uses a different width for each panel; see the module docstring.
POPULATION_SMOOTH_WINDOW = 1
SUBJECT_SMOOTH_WINDOW = 4
#: Performance range the red-green colour map spans, in percent.
PERF_NORM = (45.0, 90.0)
#: x-limits of the population panel, in seconds relative to each animal's t*.
POPULATION_XLIM = (-2, 4.5)

_GROUP = ["Name", "Date", "SessionNum"]


class OverallMetrics(NamedTuple):
    '''Population-level inputs to the reward-rate curve.'''
    correct_extra: float
    incorrect_extra: float
    bins: np.ndarray
    perf: np.ndarray


def perfModel(sampling_time, alpha: float, beta: float,
              lapse: float) -> np.ndarray:
    '''Methods eq. (1): saturating performance curve, pinned to 50% at t=0.'''
    t = np.clip(np.asarray(sampling_time, dtype=float), 0.0, None)
    return 50.0 + (50.0 - lapse) * (t**beta / (t**beta + alpha**beta))


def fitPerfCurve(bins, perf, new_bins) -> np.ndarray:
    '''Fit :func:`perfModel` to a binned curve and evaluate it on ``new_bins``.

    Falls back to linear interpolation when fewer than three finite bins
    survive, which is not enough to constrain three parameters.
    '''
    bins = np.asarray(bins, dtype=float)
    perf = np.asarray(perf, dtype=float)
    ok = np.isfinite(bins) & np.isfinite(perf)
    if ok.sum() < 3:
        return np.interp(new_bins, bins[ok], perf[ok])
    popt, _ = curve_fit(perfModel, bins[ok], perf[ok], p0=(1.0, 1.5, 0.0),
                        bounds=([1e-6, 1.0, 0.0], [100.0, 10.0, 20.0]),
                        maxfev=20_000)
    return perfModel(new_bins, *popt)


def rewardCurve(new_bins, perf, correct_extra: float,
                incorrect_extra: float) -> tuple:
    '''Methods eqs. (2)-(3): trials per hour and rewarded trials per hour.'''
    new_bins = np.asarray(new_bins, dtype=float)
    perf = np.asarray(perf, dtype=float)
    exp_trial_time = ((correct_extra + new_bins) * (perf / 100)
                      + (incorrect_extra + new_bins) * (1 - perf / 100))
    trials_per_hour = 3600 / exp_trial_time
    return trials_per_hour, trials_per_hour * perf / 100


def optimalSamplingTime(new_bins, reward) -> float:
    '''``t* = argmax_t R(t)``, in seconds.'''
    return float(np.asarray(new_bins)[int(np.nanargmax(reward))])


def _ensureSeconds(series: pd.Series) -> pd.Series:
    '''Timedelta columns become float seconds; anything else passes through.'''
    return series.dt.total_seconds() if is_timedelta64_dtype(series) else series


def avgExtraTime(df: pd.DataFrame, duration_col: str="TrialDuration",
                 stimulus_col: str="calcStimulusTime",
                 agg: str="mean") -> float:
    '''Non-sampling time per trial: per session, then per animal, then mean.'''
    if duration_col not in df.columns or stimulus_col not in df.columns:
        return np.nan
    _agg = {"mean": pd.Series.mean, "median": pd.Series.median}[agg]
    per_session = df.groupby(_GROUP).apply(
        lambda sub: _agg(sub[duration_col] - sub[stimulus_col]),
        include_groups=False)
    return float(per_session.groupby("Name").mean().mean())


def avgPerf(df: pd.DataFrame) -> float:
    '''Mean ChoiceCorrect per session, averaged across sessions, in percent.'''
    return df.groupby(_GROUP).ChoiceCorrect.mean().mean() * 100


def defaultBins() -> np.ndarray:
    lo, hi = BIN_RANGE
    return np.arange(lo, hi + BIN_STEP, BIN_STEP)


def buildPerfCurve(df: pd.DataFrame, bins=None,
                   min_bin_count: int=MIN_BIN_COUNT,
                   smooth_window: Optional[float]=None) -> tuple:
    '''Speed-accuracy curve: per-difficulty, binned by sampling time.

    Returns ``(per_difficulty, mean_perf, bin_centres)``. The three difficulty
    curves are averaged with ``nanmean``, so a bin survives if any difficulty
    had enough trials in it.
    '''
    bins = defaultBins() if bins is None else np.asarray(bins, dtype=float)
    per_difficulty = {}
    for dv_str, dv_df in df.groupby(df.DVstr):
        curve = []
        for _bin, bin_df in dv_df.groupby(pd.cut(dv_df.calcStimulusTime,
                                                 bins=bins), observed=False):
            curve.append(np.nan if len(bin_df) < min_bin_count
                         else avgPerf(bin_df))
        per_difficulty[dv_str] = curve

    # A bin can be empty at every difficulty, which nanmean reports as an
    # empty slice. That is the expected way a sparse bin drops out.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_perf = np.nanmean(list(per_difficulty.values()), axis=0)
    if smooth_window is not None:
        mean_perf = filterNanGaussianConserving(mean_perf, smooth_window,
                                                axis=-1)
    centres = np.array([(l + r) / 2 for l, r in zip(bins[:-1], bins[1:])])
    return per_difficulty, mean_perf, centres


def _withTrialDuration(df: pd.DataFrame) -> pd.DataFrame:
    '''Add the current trial's wall-clock duration from the next trial's start.'''
    df = df.sort_values(_GROUP + ["TrialStartSysTime"]).copy()
    nxt = df.groupby(_GROUP).TrialStartSysTime.shift(-1)
    df["TrialDuration"] = _ensureSeconds(nxt - df.TrialStartSysTime)
    return df


def collectMetrics(df: pd.DataFrame, bins=None,
                   smooth_window: Optional[float]=None,
                   min_bin_count: int=MIN_BIN_COUNT,
                   min_trials: int=MIN_TRIALS,
                   extra_time_agg: str="mean") -> tuple:
    '''Population and per-animal inputs to the reward-rate curve.

    Returns ``(OverallMetrics, {name: dict})``. Animals with no correct or no
    incorrect trials, or whose performance curve is empty after binning, are
    skipped rather than raising.
    '''
    bins = defaultBins() if bins is None else np.asarray(bins, dtype=float)
    df = df.groupby("Name").filter(lambda sub: len(sub) > min_trials)
    df = _withTrialDuration(df)

    finite = df[np.isfinite(df.TrialDuration)
                & np.isfinite(df.calcStimulusTime)]
    overall_correct = avgExtraTime(finite[finite.ChoiceCorrect == True],
                                   agg=extra_time_agg)
    overall_incorrect = avgExtraTime(finite[finite.ChoiceCorrect == False],
                                     agg=extra_time_agg)

    _, mean_perf, centres = buildPerfCurve(df, bins, min_bin_count,
                                           smooth_window)
    ok = ~np.isnan(mean_perf)
    overall = OverallMetrics(overall_correct, overall_incorrect,
                             centres[ok], mean_perf[ok])

    subj_metrics = {}
    for name, sub_df in df.groupby("Name"):
        sub_ok = sub_df[np.isfinite(sub_df.TrialDuration)
                        & np.isfinite(sub_df.calcStimulusTime)]
        correct = sub_ok[sub_ok.ChoiceCorrect == True]
        incorrect = sub_ok[sub_ok.ChoiceCorrect == False]
        if not len(correct) or not len(incorrect):
            continue

        _, sub_perf, _ = buildPerfCurve(sub_df, bins, min_bin_count,
                                        smooth_window)
        sub_ok_bins = ~np.isnan(sub_perf)
        if not sub_ok_bins.any():
            continue

        stim = sub_df.calcStimulusTime
        subj_metrics[name] = dict(
            correct_extra=avgExtraTime(correct, agg=extra_time_agg),
            incorrect_extra=avgExtraTime(incorrect, agg=extra_time_agg),
            bins=centres[sub_ok_bins],
            perf=sub_perf[sub_ok_bins],
            observed_sampling_time=float(stim.mean()),
            # The paper's error bars are +- SD, not SEM.
            observed_sampling_time_sd=float(stim.std()),
            observed_perf=float(sub_df.ChoiceCorrect.mean() * 100),
            observed_reward_per_hour=_observedRewardPerHour(sub_df, correct,
                                                            incorrect,
                                                            extra_time_agg),
        )
    return overall, subj_metrics


def _observedRewardPerHour(sub_df, correct, incorrect, agg) -> float:
    '''Rewards per hour actually collected, from real trial durations.'''
    correct_extra = avgExtraTime(correct, agg=agg)
    incorrect_extra = avgExtraTime(incorrect, agg=agg)
    stim = np.asarray(sub_df.calcStimulusTime.values, dtype=float)
    corr = np.asarray(sub_df.ChoiceCorrect.values, dtype=bool)
    valid = np.isfinite(stim) & (stim >= 0)
    stim, corr = stim[valid], corr[valid]
    durations = np.where(corr, stim + correct_extra, stim + incorrect_extra)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if not durations.size:
        return np.nan
    return float(corr.sum() / (durations.sum() / 3600.0))


def _colormap():
    return LinearSegmentedColormap.from_list("RedGreen", [(1, 0, 0), (0, 1, 0)],
                                             N=100)


def _perfNorm():
    return mpl.colors.Normalize(vmin=PERF_NORM[0], vmax=PERF_NORM[1])


def _evalGrid(step: float) -> np.ndarray:
    return np.arange(0.0, NUM_SECS + step, step)


def plotSubjectRewardCurve(overall: OverallMetrics, subject_name: str,
                           subj_metrics: Mapping[str, dict],
                           ax: Optional[plt.Axes]=None,
                           save_prefix: Optional[Union[str, Path]]=None,
                           save_figs: bool=False):
    '''Figure 2B-left: one animal's reward-rate curve against its optimum.

    The bars are the min-max normalised reward rate over sampling time,
    coloured by the fitted performance at that time; the dashed line is ``t*``
    and the X is the observed sampling time (mean +- SD).
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")
    m = subj_metrics[subject_name]

    new_bins = _evalGrid(SUBJECT_STEP)
    perf = fitPerfCurve(m["bins"], m["perf"], new_bins)
    _, reward = rewardCurve(new_bins, perf, overall.correct_extra,
                            overall.incorrect_extra)
    lo, hi = np.nanmin(reward), np.nanmax(reward)
    denom = (hi - lo) if (hi - lo) > 0 else np.nan
    normed = (reward - lo) / denom
    t_opt = optimalSamplingTime(new_bins, reward)

    cm, norm = _colormap(), _perfNorm()
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar(new_bins, normed, color=cm(norm(perf)), width=SUBJECT_STEP,
           edgecolor="none", alpha=0.8)
    ax.set_xlim(new_bins.min(), new_bins.max())
    ax.set_xlabel("Sampling time (s)")
    ax.set_ylabel("Normalized ∑ Reward Rate")
    ax.set_title(f"Optimal sampling time maximizes reward rate "
                 f"({subject_name})")
    ax.axvline(t_opt, color="k", ls="--", lw=1.5, alpha=0.9,
               label="Reward-optimal")

    if np.isfinite(denom):
        ax.errorbar(m["observed_sampling_time"],
                    (m["observed_reward_per_hour"] - lo) / denom,
                    xerr=m["observed_sampling_time_sd"], fmt="X",
                    color=cm(norm(m["observed_perf"])), markersize=8,
                    markeredgecolor="k", ecolor="k", label="Observed",
                    zorder=5, capsize=4)

    mappable = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    mappable.set_array(perf)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.05, shrink=0.5,
                        orientation="horizontal", location="top",
                        anchor=(1.2, -2.5), format=lambda x, _: f"{x:.0f}%")
    cbar.set_label("Performance")
    cbar.ax.xaxis.set_ticks_position("bottom")
    ax.legend()

    if save_figs:
        _save(fig, save_prefix,
              f"subj__{subject_name.replace('.', '_')}.svg")
    return fig


def plotAlignedPopulation(subj_metrics: Mapping[str, dict],
                          ax: Optional[plt.Axes]=None,
                          save_prefix: Optional[Union[str, Path]]=None,
                          save_figs: bool=False):
    '''Figure 2B-right: every animal aligned on its own reward optimum.

    One row per animal: the fitted performance curve shifted so that ``t*``
    sits at 0, with the observed sampling time (mean +- SD) marked. Rows are
    sorted by how far the observed time falls from the optimum.
    '''
    if save_figs and save_prefix is None:
        raise ValueError("save_figs=True needs a save_prefix")

    new_bins = _evalGrid(POPULATION_STEP)
    deltas = np.arange(-NUM_SECS, NUM_SECS + POPULATION_STEP, POPULATION_STEP)
    cm, norm = _colormap(), _perfNorm()

    rows, meta = [], []
    for name, m in subj_metrics.items():
        if len(np.asarray(m["bins"])) < 2:
            continue
        perf = fitPerfCurve(m["bins"], m["perf"], new_bins)
        _, reward = rewardCurve(new_bins, perf, m["correct_extra"],
                                m["incorrect_extra"])
        if not np.isfinite(reward).any():
            continue
        t_opt = optimalSamplingTime(new_bins, reward)

        at = deltas + t_opt
        row = np.full_like(deltas, np.nan, dtype=float)
        inside = (at >= 0.0) & (at <= NUM_SECS)
        row[inside] = np.interp(at[inside], new_bins, perf)
        rgba = cm(norm(row))
        rgba[~np.isfinite(row), 3] = 0.0                # outside the domain
        rows.append(rgba)
        meta.append((name, m["observed_sampling_time"] - t_opt,
                     m["observed_perf"], m["observed_sampling_time_sd"]))

    if not rows:
        raise ValueError("no animal had enough bins to plot")

    order = np.argsort([d for _, d, _, _ in meta])
    rows = [rows[i] for i in order]
    meta = [meta[i] for i in order]

    fig, ax = ((ax.get_figure(), ax) if ax is not None
               else plt.subplots(figsize=(10, max(3, 0.35 * len(meta)))))
    ax.imshow(np.stack(rows, axis=0), aspect="auto", origin="lower",
              extent=[deltas.min(), deltas.max(), -0.5, len(meta) - 0.5],
              interpolation="nearest")
    ax.axvline(0.0, color="k", lw=1.0, alpha=0.8)

    for row_idx, (_name, delta, perf_pct, sd) in enumerate(meta):
        ax.errorbar(delta, row_idx, xerr=sd, fmt="X", color=cm(norm(perf_pct)),
                    markersize=12, markeredgecolor="k", markeredgewidth=2,
                    ecolor="k", zorder=5, capsize=4)

    ax.set_yticks(np.arange(len(meta)))
    ax.set_yticklabels(str(i + 1) for i in range(len(meta)))
    ax.set_xlabel("Δ sampling time relative to reward-optimal (s)")
    ax.set_xlim(*POPULATION_XLIM)
    ax.set_title("Across-subject performance around each subject’s "
                 "reward-optimal sampling time")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_figs:
        _save(fig, save_prefix, "all_subj_aligned.svg")
    return fig


def _save(fig, save_prefix, name: str) -> None:
    save_fp = Path(save_prefix) / "optimal_sampling" / name
    save_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fp, bbox_inches="tight", dpi=300)
