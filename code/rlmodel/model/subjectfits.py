'''Per-subject model-fit figures (Figures 2E, 2F and S4A).

Backend for the per-subject cells of ``rlmodel_analysis.ipynb``. Both figures
read the ``seed=0`` simulation frames that ``aggregate.load_or_collect_metrics``
puts in ``sim_cache`` -- ``{spec label: {subject: (loss, fitted_df, BOUND,
include_Q, include_RewardRate)}}``. On a metrics-cache hit they come from the
tracked sidecar ``data/RLModel/metrics/simcache_fig2g.pkl``, so a normal run
can draw them without re-simulating anything.

- ``plotSubjectFits`` (Figures 2F and S4A) -- the full fit panel of every
  model for each subject, one file per model:
  ``{save_prefix}/figs/{subject}/{model}.svg``. 2F is the
  "RewardRate + Q-Val" panel's psychometric, S4A the Classic / Q-Val /
  RewardRate ones.
- ``plotRewardRates`` (Figure 2E) -- z-scored sampling time against the
  past-5-trial reward rate, the mouse against every model, written to
  ``{save_prefix}/RewardRate/subject_{subject}.svg`` (and ``all_subjects.svg``
  for the pooled view).

``subjects`` picks which subjects to draw (``None`` = all), so the paper's
example and the full loop go through the same code.
'''
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ...behavior.rewardrate import loopRewardRateAnalysis
from .aggregate import safe_filename
from .plotter import PsychometricPlot, createFig, plotPlots

#: The example mouse of Figures 2E, 2F and S4A.
EXAMPLE_SUBJECT = "Avgat3"


def _wanted(subject, subjects):
    return subjects is None or subject in subjects


def _finish(fig, plt_show):
    if plt_show:
        plt.show()
    else:
        plt.close(fig)


def plotSubjectFits(sim_cache, *, save_prefix=None, subjects=None,
                    save_figs=False, plt_show=None, plt_agg=False,
                    t_dur=4.8, dt=0.005):
    """Per-subject (and optionally across-subject) model-fit panels.

    Returns the saved paths (empty unless ``save_figs``).
    """
    if plt_show is None:
        plt_show = not save_figs
    assert plt_show or save_figs, "Either plt_show or save_figs must be True"
    if save_figs:
        assert save_prefix is not None, "Pass save_prefix to save"

    saved = []
    for model_name, subjects_dict in sim_cache.items():
        df_li = []
        for subject, (loss, fitted_df, BOUND, include_Q,
                      include_RewardRate) in subjects_dict.items():
            df_li.append(fitted_df)
            if not _wanted(subject, subjects):
                continue
            fig, axs = createFig(is_small_fig_mode=False)
            plotPlots(fitted_df, axs=axs, is_small_fig_mode=False,
                      BOUND=BOUND, include_Q=include_Q,
                      include_RewardRate=include_RewardRate,
                      psych_plot=PsychometricPlot.SlowFast,
                      plot_bias_dir=False, t_dur=t_dur, dt=dt)
            fig.suptitle(f"Subject: {subject}\nModel: {model_name}\n"
                         f"Loss: {loss:.4f}", y=1.1)
            if save_figs:
                fp = (Path(save_prefix, "figs", subject) /
                      f"{safe_filename(model_name)}.svg")
                fp.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fp, bbox_inches="tight")
                saved.append(fp)
            _finish(fig, plt_show)

        if not plt_agg:
            continue
        fig, axs = createFig(is_small_fig_mode=False)
        fig.suptitle(f"All Subjects\nModel: {model_name}", y=1.05)
        plotPlots(pd.concat(df_li), axs=axs, is_small_fig_mode=False,
                  BOUND=BOUND, include_Q=include_Q,
                  include_RewardRate=include_RewardRate,
                  psych_plot=PsychometricPlot.SlowFast,
                  plot_bias_dir=False, t_dur=t_dur, dt=dt)
        if save_figs:
            fp = (Path(save_prefix, "figs", "all_subjects") /
                  f"{safe_filename(model_name)}.svg")
            fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fp, bbox_inches="tight")
            saved.append(fp)
        _finish(fig, plt_show)
    return saved


def plotSubjectRewardRate(sim_cache, df_behavior, *, subject_name=None,
                          min_trials_per_sess_rr, save_prefix=None,
                          save_figs=False):
    """One reward-rate panel: the mouse against every model in ``sim_cache``.

    ``subject_name=None`` pools every subject. Returns the figure and the
    saved path (``None`` unless ``save_figs``).
    """
    if subject_name is not None:
        df_behavior = df_behavior[df_behavior.Name == subject_name]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    loop_kwargs = dict(
        plot_singles_on_all=False, plot_subjects=False, ax_all=ax,
        exclude_reward_rate_bins=[0], run_stats=False,
        min_trials_per_sess_rr=min_trials_per_sess_rr,
        ax_all_used_timeout_time="All"
    )
    loopRewardRateAnalysis(df_behavior, descrp="Behavior", **loop_kwargs)
    for comb_str, comb_dict in sim_cache.items():
        # sim_cache is keyed by the EvalSpec label.
        subjects_dict = {subject: fitted_df for subject,
                         (loss, fitted_df, BOUND, include_Q, include_RewardRate)
                         in comb_dict.items()
                         if subject_name is None or subject == subject_name}
        df_subjects = pd.concat(subjects_dict.values())
        loopRewardRateAnalysis(df_subjects, descrp=comb_str,
                               print_only_sim=True, **loop_kwargs)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, zorder=-1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize="small")

    ax.set_title("All Subjects" if subject_name is None
                 else f"Subject: {subject_name}")
    fp = None
    if save_figs:
        assert save_prefix is not None, "Pass save_prefix to save"
        label = (f"subject_{subject_name}" if subject_name is not None
                 else "all_subjects")
        fp = Path(save_prefix, "RewardRate", f"{label}.svg")
        fp.parent.mkdir(parents=True, exist_ok=True)
        print("Saving:", label)
        fig.savefig(fp, bbox_inches="tight")
    plt.show()
    return fig, fp


def plotRewardRates(sim_cache, df_behavior, *, min_num_trials,
                    min_trials_per_sess_rr, subjects=None, plot_pooled=True,
                    save_prefix=None, save_figs=False):
    """Figure 2E: the pooled panel (``plot_pooled``) and one per subject.

    Subjects with fewer than ``min_num_trials`` trials are dropped first.
    Returns the saved paths.
    """
    df_behavior = df_behavior.groupby("Name").filter(
                                         lambda x: len(x) >= min_num_trials)
    df_behavior["SessId"] = (df_behavior.Name + "_" +
                             df_behavior.Date.astype(str) + "_" +
                             df_behavior.SessionNum.astype(str))
    kwargs = dict(min_trials_per_sess_rr=min_trials_per_sess_rr,
                  save_prefix=save_prefix, save_figs=save_figs)
    saved = []
    if plot_pooled:
        saved.append(plotSubjectRewardRate(sim_cache, df_behavior,
                                           subject_name=None, **kwargs)[1])
    for subject in df_behavior.Name.unique():
        if _wanted(subject, subjects):
            saved.append(plotSubjectRewardRate(sim_cache, df_behavior,
                                               subject_name=subject,
                                               **kwargs)[1])
    return [fp for fp in saved if fp is not None]
