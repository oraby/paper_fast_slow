'''Raw (seconds) sampling-time KDE figures, for mice and for human subjects.

One drawing primitive (``plotSTGroup``: a faint normalized histogram plus a
KDE, labelled with the group's mean +- SEM and trial count) shared by three
figures:

* ``plotFMHFSubjects`` -- Ext. Fig. 1d-right: one figure per freely-moving /
  head-fixed mouse, with a curve per experiment-type x mouse-state group.
  ``zscore=True`` pools that mouse's groups and z-scores them together.
* ``plotHumanSubjects`` -- one figure per human subject, with a curve per
  experiment type (the raw seconds, not the z-scored time).
* ``plotMiceVsHumansST`` -- every RDK mouse pooled into one curve against every
  human speed-context trial pooled into another, raw seconds.
* ``plotMouseHumanSTKde`` -- one mouse and one human subject brought together
  on a single axes, with a selectable subset of the mouse groups. With
  ``zscore=True`` each species is z-scored against its own pool: the mouse's
  plotted groups are pooled and z-scored together, and so, separately, are the
  human's experiment types -- which lines the two species up on a common axis
  without letting either one's scale set the other's.

The session-collection pass for the mice (``collectFMHFSessions``) is the logic
that used to live in the ``## Ext. Fig. 1d-right`` notebook cell, unchanged:
sessions need a non-null DV, RDK sessions below ``min_rdk_easy_perf`` easy-trial
performance are dropped, sampling times above ``max_st`` are dropped, a subject
is only accepted if every one of its groups has at least ``min_sess_per_state``
surviving sessions, and only the last ``max_sess`` of those are kept.
'''
from __future__ import annotations

import pathlib
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from ..common.definitions import ExperimentType, MouseState

# Sampling time is capped at 5s in both species, so one binning fits all.
DEF_BINS = np.arange(0, 5.1, 0.1)
DEF_XLIM = (0, 5)
MAX_ST = 4.9
# Z-scored axes are data-driven; only the bin width is fixed.
Z_BIN_WIDTH = 0.1

# Mouse groups, keyed as "{ExperimentType}{MouseState}" (see the enums in the
# notebook: LightIntensity/RDK x FreelyMoving/HeadFixed).
MOUSE_GROUPS = ("LightIntensityFreelyMoving", "LightIntensityHeadFixed",
                "RDKFreelyMoving", "RDKHeadFixed")
MOUSE_LABELS = {"LightIntensityFreelyMoving": "Light-Chasing Freely Moving",
                "LightIntensityHeadFixed": "Light-Chasing Head Fixed",
                "RDKFreelyMoving": "RDK Freely Moving",
                "RDKHeadFixed": "RDK Head Fixed"}
MOUSE_COLORS = {"LightIntensityFreelyMoving": "blue",
                "LightIntensityHeadFixed": "red",
                "RDKFreelyMoving": "green", "RDKHeadFixed": "orange"}
# The two groups the combined mouse+human figure shows.
COMBINED_MOUSE_GROUPS = ("LightIntensityFreelyMoving", "RDKHeadFixed")

DEF_ST_COL = "calcStimulusTime"

# The pooled mice-vs-humans figure: every mouse against the human speed
# context (maximizing correct outcomes in a fixed time).
MICE_POOL_LABEL = "Mice - RDK"
HUMAN_SPEED_LABEL = "Humans - Speed Context"
HUMAN_SPEED_SESSION_TYPE = "Competition"

# Human experiment types, as (label, session_type).
HUMAN_SESSION_TYPES = (("Accuracy Competition", "ReactionTime"),
                       ("Maximizing Correct-Outcome Competition",
                        "Competition"))


def plotSTGroup(st: pd.Series, ax, color, label: str, bins=DEF_BINS,
                clip=DEF_XLIM, show_hist: bool=True, ls: str="-",
                unit: str="s"):
    '''Faint normalized histogram + KDE of one group's sampling time.

    ``unit`` is appended to the mean/SEM in the legend ("s" for raw seconds,
    "" for z-scored values).
    '''
    st = st[st.notnull()]
    if show_hist:
        hist, _ = np.histogram(st, bins=bins)
        hist = hist.astype(float) / hist.sum()
        ax.stairs(hist, bins, color=color, fill=False, alpha=0.3)
    label = (f"{label} - RT: {st.mean():.2f}{unit} "
             f"±{stats.sem(st):.2f}{unit} SEM\n"
             f"(n={len(st):,} trials)")
    sns.kdeplot(st, ax=ax, color=color, label=label, fill=False, clip=clip,
                ls=ls)
    return st


def _finishAxes(ax, title: str, print_str: str="", xlim=DEF_XLIM,
                xlabel: str="Stimulus Time (s)"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    if print_str:
        ax.text(0, -.1, print_str.replace("\t", "    "), transform=ax.transAxes,
                ha="left", va="top")
    ax.set_xlim(*xlim)


def _save(fig, save_prefix, subdir: str, fname: str):
    save_fp = pathlib.Path(save_prefix, subdir, f"{fname}.svg")
    save_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_fp, bbox_inches="tight")
    return save_fp


def zScorePooled(series_by_label: dict) -> dict:
    '''Z-score every series against the pool of all of them together.

    The whole point of the combined figure's ``zscore`` flag: one species is one
    pool, so the relative offset between that species' own groups survives while
    its overall scale is normalized away.
    '''
    pooled = pd.concat(list(series_by_label.values()))
    mean, std = pooled.mean(), pooled.std(ddof=0)   # ddof=0, as scipy's zscore
    return {label: (st - mean)/std
            for label, st in series_by_label.items()}


def zBinsAndLim(series_by_label: dict, bin_width: float=Z_BIN_WIDTH):
    '''Bins and x-limits covering every z-scored value, at a fixed bin width.'''
    pooled = pd.concat(list(series_by_label.values()))
    lo = np.floor(pooled.min()/bin_width)*bin_width
    hi = np.ceil(pooled.max()/bin_width)*bin_width
    return np.arange(lo, hi + bin_width/2, bin_width), (lo, hi)


def collectFMHFSessions(df: pd.DataFrame, min_sess_per_state: int,
                        max_sess: int=10, max_st: float=MAX_ST,
                        min_rdk_easy_perf: float=75,
                        verbose: bool=True) -> dict:
    '''Per-subject accepted sessions, one entry per experiment x state group.

    Returns ``{subject: {group_key: df or None, "PrintStr": str}}`` for the
    subjects that have enough sessions in every group they contribute.
    '''
    accepted_subject_dict = {}
    for subject, subject_df in df.groupby("Name"):
        used_sess_dfs = {group: None for group in MOUSE_GROUPS}
        used_sess_dfs["PrintStr"] = ""
        print_str = f"{subject}\n"
        use_subject = True
        for exp_type, exp_df in subject_df.groupby("GUI_ExperimentType"):
            state_prefix = str(ExperimentType(exp_type))
            for state, state_df in exp_df.groupby("GUI_MouseState"):
                state_str = str(MouseState(state))
                print_str += f"\t{state_prefix} {state_str}\n"
                stat_dfs_li = []
                print_str_li = []
                for sess, sess_df in state_df.groupby(["Name", "Date",
                                                       "SessionNum"]):
                    sess_df = sess_df[sess_df.DV.notnull()]
                    easy_perf = sess_df[sess_df.DV.abs() >= .8
                                        ].ChoiceCorrect.mean()*100
                    if (exp_type == ExperimentType.RDK and
                            easy_perf < min_rdk_easy_perf):
                        continue
                    st = sess_df.calcStimulusTime.copy()
                    st = st[st.notnull()]
                    st = st[st <= max_st]
                    num_trials = len(sess_df)
                    min_sample_min = sess_df.GUI_MinSampleMin.unique()
                    min_sample_max = sess_df.GUI_MinSampleMax.unique()
                    min_simple_mean = sess_df.MinSample.mean()

                    sess_df = sess_df[sess_df.index.isin(st.index)]
                    print_str_li += [
                        f"\t\t{sess[1]}/{sess[2]} - EasyPerf: {easy_perf:.2f}% - "
                        f"MinSample: Min-Max:{min_sample_min}->{min_sample_max} - "
                        f"Mean:{min_simple_mean:.2f} -"
                        f" ST: {st.mean():.2f} - {num_trials} trials\n"]
                    stat_dfs_li.append(sess_df)

                if len(stat_dfs_li) >= min_sess_per_state or (
                        not exp_type and len(stat_dfs_li) >= 1):
                    if len(stat_dfs_li) > max_sess:
                        stat_dfs_li = stat_dfs_li[-max_sess:]
                        print_str_li = print_str_li[-max_sess:]
                    if verbose:
                        print(f"{state_prefix}{state_str}")
                    used_sess_dfs[f"{state_prefix}{state_str}"] = pd.concat(
                                        stat_dfs_li).reset_index(drop=True)
                    print_str += "".join(print_str_li)
                else:
                    use_subject = False
        if use_subject:
            used_sess_dfs["PrintStr"] = print_str
            accepted_subject_dict[subject] = used_sess_dfs
    return accepted_subject_dict


def mouseSubjectGroups(subject_dict: dict, groups=MOUSE_GROUPS) -> dict:
    '''``{group_key: sampling-time series}`` for the groups that have data.'''
    return {group: subject_dict[group].calcStimulusTime for group in groups
            if subject_dict.get(group) is not None}


def plotSeriesGroups(series_by_label: dict, ax, colors: dict,
                     labels: Optional[dict]=None, label_prefix: str="",
                     **kwargs) -> int:
    '''Draw one curve per entry of ``{key: series}``; returns how many drew.

    ``colors`` is keyed like ``series_by_label``; ``labels`` optionally renames
    those keys for the legend (the mouse groups use it, the humans do not).
    '''
    for key, st in series_by_label.items():
        legend_key = key if labels is None else labels[key]
        plotSTGroup(st, ax=ax, color=colors[key],
                    label=f"{label_prefix}{legend_key}", **kwargs)
    return len(series_by_label)


def zScoreStyle(series_by_label: dict, zscore: bool, per: str="Subject"):
    '''Bins / limits / labels / file-name suffix for a raw or z-scored axis.

    Returns ``(plot_kwargs, axes_kwargs, title_extra, fname_extra)``; the caller
    has already z-scored the series, this only dresses the axis for them.
    '''
    if not zscore:
        return (dict(bins=DEF_BINS, clip=DEF_XLIM, unit="s"),
                dict(xlim=DEF_XLIM, xlabel="Stimulus Time (s)"), "", "")
    bins, xlim = zBinsAndLim(series_by_label)
    return (dict(bins=bins, clip=xlim, unit=""),
            dict(xlim=xlim, xlabel=f"Within-{per} Z-Scored Stimulus Time"),
            f" (Z-Scored per {per})", "_zscored")


def plotFMHFSubjects(df: pd.DataFrame, min_sess_per_state: int,
                     groups=MOUSE_GROUPS, colors: Optional[dict]=None,
                     labels: Optional[dict]=None, annotate: bool=True,
                     zscore: bool=False, figsize=(6, 6), save_prefix=None,
                     save_figs: bool=False, subdir: str="fm_hf",
                     verbose: bool=True) -> dict:
    '''Ext. Fig. 1d-right: one sampling-time figure per accepted mouse.

    ``zscore=False`` (the default) plots the raw seconds; ``zscore=True`` pools
    that mouse's plotted groups and z-scores them together, so its groups keep
    their relative offsets while its overall scale is normalized away. The
    z-scored figures save under a separate ``*_zscored.svg`` file name.
    '''
    colors = MOUSE_COLORS if colors is None else colors
    labels = MOUSE_LABELS if labels is None else labels
    accepted = collectFMHFSessions(df, min_sess_per_state=min_sess_per_state,
                                   verbose=verbose)
    figs = {}
    for subject, subject_dict in accepted.items():
        group_st = mouseSubjectGroups(subject_dict, groups=groups)
        if not group_st:
            print(f"No data for {subject}")
            continue
        if zscore:
            group_st = zScorePooled(group_st)
        plot_kwargs, axes_kwargs, title_extra, fname_extra = zScoreStyle(
                                            group_st, zscore, per="Subject")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plotSeriesGroups(group_st, ax=ax, colors=colors, labels=labels,
                         **plot_kwargs)
        _finishAxes(ax, f"{subject} - Stimulus Time Distribution{title_extra}",
                    print_str=subject_dict["PrintStr"] if annotate else "",
                    **axes_kwargs)
        if save_figs:
            _save(fig, save_prefix, subdir,
                  f"stimulus_time_{subject}{fname_extra}")
        plt.show()
        figs[subject] = fig
    return figs


def humanSubjectGroups(df_users: pd.DataFrame, subject: str,
                       session_types=HUMAN_SESSION_TYPES,
                       max_st: Optional[float]=MAX_ST) -> dict:
    '''``{label: raw sampling-time series}`` for one human subject.'''
    subject_df = df_users[df_users.Name == subject]
    groups = {}
    for label, session_type in session_types:
        st = subject_df[subject_df.session_type == session_type
                        ].calcStimulusTime
        st = st[st.notnull()]
        if max_st is not None:
            st = st[st <= max_st]
        if len(st):
            groups[label] = st
    return groups


def humanSubjectPrintStr(df_users: pd.DataFrame, subject: str,
                         session_types=HUMAN_SESSION_TYPES) -> str:
    '''Per-session summary text, the human counterpart of ``PrintStr``.'''
    subject_df = df_users[df_users.Name == subject]
    print_str = f"{subject}\n"
    for label, session_type in session_types:
        sess_df = subject_df[subject_df.session_type == session_type]
        print_str += f"\t{label}\n"
        for sess, one_sess in sess_df.groupby(["Date", "SessionNum"]):
            st = one_sess.calcStimulusTime
            st = st[st.notnull()]
            if not len(st):
                continue
            print_str += (f"\t\t{sess[0]}/{sess[1]} - "
                          f"ST: {st.mean():.2f} - {len(one_sess)} trials\n")
    return print_str


def plotHumanSubjects(df_users: pd.DataFrame, colors: dict,
                      subjects: Optional[Union[list, tuple]]=None,
                      session_types=HUMAN_SESSION_TYPES, annotate: bool=True,
                      figsize=(6, 6), save_prefix=None, save_figs: bool=False,
                      subdir: str="humans_st", max_st: Optional[float]=MAX_ST
                      ) -> dict:
    '''One raw (seconds) sampling-time figure per human subject.

    ``colors`` maps the ``session_types`` labels to a color. Both experiment
    types of a subject go on the same axes.
    '''
    if subjects is None:
        subjects = sorted(df_users.Name.unique())
    figs = {}
    for subject in subjects:
        groups = humanSubjectGroups(df_users, subject,
                                    session_types=session_types, max_st=max_st)
        if not groups:
            print(f"No data for {subject}")
            continue
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plotSeriesGroups(groups, ax=ax, colors=colors)
        print_str = (humanSubjectPrintStr(df_users, subject, session_types)
                     if annotate else "")
        _finishAxes(ax, f"{subject} - Stimulus Time Distribution",
                    print_str=print_str)
        if save_figs:
            _save(fig, save_prefix, subdir, f"stimulus_time_{subject}")
        plt.show()
        figs[subject] = fig
    return figs


def plotMouseHumanSTKde(lc_hf_df: pd.DataFrame, df_users: pd.DataFrame,
                        mouse_subject: str, human_subject: str, *,
                        human_colors: dict,
                        mouse_groups=COMBINED_MOUSE_GROUPS,
                        mouse_colors: Optional[dict]=None,
                        mouse_labels: Optional[dict]=None,
                        session_types=HUMAN_SESSION_TYPES,
                        min_sess_per_state: int=1, max_st: float=MAX_ST,
                        zscore: bool=False, figsize=(7, 6), save_prefix=None,
                        save_figs: bool=False, subdir: str="fm_hf",
                        verbose: bool=False):
    '''One mouse and one human subject on the same axes.

    Only the ``mouse_groups`` subset of the mouse's curves is drawn. With
    ``zscore=False`` (the default) the x-axis is the raw sampling time in
    seconds; with ``zscore=True`` each species is z-scored against its own pool
    -- the mouse's plotted groups pooled together, the human's experiment types
    pooled together -- and the figure saves under a separate file name.
    '''
    accepted = collectFMHFSessions(lc_hf_df[lc_hf_df.Name == mouse_subject],
                                   min_sess_per_state=min_sess_per_state,
                                   verbose=verbose)
    assert mouse_subject in accepted, (
        f"{mouse_subject} has no group with >= {min_sess_per_state} sessions")
    mouse_colors = MOUSE_COLORS if mouse_colors is None else mouse_colors
    mouse_labels = MOUSE_LABELS if mouse_labels is None else mouse_labels

    mouse_st = mouseSubjectGroups(accepted[mouse_subject], groups=mouse_groups)
    assert mouse_st, f"None of {mouse_groups} has data for {mouse_subject}"
    human_st = humanSubjectGroups(df_users, human_subject,
                                  session_types=session_types, max_st=max_st)
    assert human_st, f"No sampling time for {human_subject}"

    if zscore:
        # One pool per species, so each species keeps its own between-group
        # offsets but neither one's scale dominates the shared axis.
        mouse_st = zScorePooled(mouse_st)
        human_st = zScorePooled(human_st)
    plot_kwargs, axes_kwargs, title_extra, fname_extra = zScoreStyle(
                            {**mouse_st, **human_st}, zscore, per="Species")

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plotSeriesGroups(mouse_st, ax=ax, colors=mouse_colors, labels=mouse_labels,
                     label_prefix=f"Mouse {mouse_subject}: ", **plot_kwargs)
    plotSeriesGroups(human_st, ax=ax, colors=human_colors,
                     label_prefix=f"Human {human_subject}: ", **plot_kwargs)

    _finishAxes(ax, f"Stimulus Time Distribution - Mouse {mouse_subject} vs "
                    f"Human {human_subject}{title_extra}", **axes_kwargs)
    ax.legend(fontsize="x-small")
    if save_figs:
        _save(fig, save_prefix, subdir,
              f"stimulus_time_mouse_{mouse_subject}_human_{human_subject}"
              f"{fname_extra}")
    plt.show()
    return fig


def poolST(df: pd.DataFrame, st_col: str=DEF_ST_COL,
           max_st: Optional[float]=MAX_ST) -> pd.Series:
    '''Every non-null sampling time of ``df``, pooled and capped at ``max_st``.'''
    st = df[st_col]
    st = st[st.notnull()]
    return st if max_st is None else st[st <= max_st]


def plotMiceVsHumansST(df_mice: pd.DataFrame, df_users: pd.DataFrame, *,
                       colors: dict, mice_label: str=MICE_POOL_LABEL,
                       human_label: str=HUMAN_SPEED_LABEL,
                       human_session_type: str=HUMAN_SPEED_SESSION_TYPE,
                       max_st: Optional[float]=MAX_ST, figsize=(7, 6),
                       save_prefix=None, save_figs: bool=False,
                       subdir: str="", fname: str="stimulus_time_mice_vs_humans"
                       ) -> dict:
    '''Two KDEs of raw seconds: all RDK mice vs all human speed-context trials.

    Both curves pool every subject's trials together -- one curve for the whole
    mouse dataset, one for the ``human_session_type`` sessions of every human
    subject. ``colors`` is keyed by ``mice_label`` / ``human_label``.
    '''
    human_df = df_users[df_users.session_type == human_session_type]
    groups = {mice_label: poolST(df_mice, max_st=max_st),
              human_label: poolST(human_df, max_st=max_st)}
    for label, st in groups.items():
        assert len(st), f"No sampling time for {label}"
    n_subjects = {mice_label: df_mice.Name.nunique(),
                  human_label: human_df.Name.nunique()}
    legend_labels = {label: f"{label} (n={n_subjects[label]} subjects)"
                     for label in groups}

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plotSeriesGroups(groups, ax=ax, colors=colors, labels=legend_labels)
    _finishAxes(ax, "Stimulus Time Distribution - Mice vs Humans")
    ax.legend(fontsize="x-small")
    if save_figs:
        _save(fig, save_prefix, subdir, fname)
    plt.show()
    return fig
