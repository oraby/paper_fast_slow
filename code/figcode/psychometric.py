try: # Try import as a library first, then as a local subfolder
    from .psychofit import psychofit
except ImportError:
    from psychofit import psychofit
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def slowFastPsych(df, is_human_subject, combine_sides=False, plot_typical=False,
                  y_rng=None, plot_single_subjects=False,
                  plot_combined_subjects=True,  title_suffix="",
                  subject_plot_points=True, save_fp="", save_figs=False):

    title = "All Mice" if not is_human_subject else "All Human Subjects"
    if len(title_suffix):
        title += f" - {title_suffix}"
    if plot_combined_subjects:
        _slowFastPsychSubject(df, title, combine_sides=combine_sides,
                            is_human_subject=is_human_subject,
                            plot_typical=plot_typical, y_rng=y_rng,
                            many_subjects=True,
                            subject_plot_points=subject_plot_points,
                            save_fp=save_fp, save_figs=save_figs,)
    if not plot_single_subjects:
        return
    for subject, subject_df in df.groupby("Name"):
        if len(title_suffix):
            subject = f"{subject} - {title_suffix}"
        _slowFastPsychSubject(subject_df, subject, combine_sides=combine_sides,
                              is_human_subject=is_human_subject,
                              plot_typical=plot_typical, y_rng=y_rng,
                              many_subjects=False, save_fp=save_fp,
                              save_figs=save_figs)

def loopPsych(df, is_human_subject, title, combine_sides=False, nfits=None,
              y_rng=None, plot_single_subjects=True,
              single_subjects_in_legend=False,
              save_fp="", save_figs=False):

    def processSubject(df, grp_by_col, sub_title, sec_clr_is_gray,
                       grp_by_legend, many_subjects):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax = _psychAxes(sub_title, combine_sides=combine_sides, ax=ax)
        if many_subjects:
            num_subjects = df.Name.nunique()
            label = (f"{sub_title} (n={num_subjects} Subjects, "
                     f"{len(df):,} Trials)")
        else:
            num_sessions = df[["Name", "Date", "SessionNum"]].drop_duplicates()
            num_sessions = len(num_sessions)
            label = (f"{sub_title} (n={num_sessions} Sessions, "
                     f"{len(df):,} Trials)")
        _fitPsych(_getGroups(df, combine_sides, is_human_subject), ax=ax,
                  many_subjects=many_subjects, nfits=nfits, label=label,
                  combine_sides=combine_sides, color='k', linewidth=2, alpha=1,
                  plot_points=True)
        c, alpha = ("gray", 0.3) if sec_clr_is_gray else (None, 1)
        for grp_name, grp_df in df.groupby(grp_by_col):
            label = (f"{grp_name} ({len(grp_df):,} Trials)" if grp_by_legend
                     else None)
            _fitPsych(_getGroups(grp_df, combine_sides, is_human_subject),
                      ax=ax, many_subjects=False, nfits=nfits,
                      label=label, combine_sides=combine_sides, color=c,
                      linewidth=0.5, alpha=alpha, plot_points=False)

        if y_rng is not None:
            ax.set_ylim(*y_rng)
        # else:
        #     if not is_human_subject and df.Name.nunique() == 1:
        #         ax.set_ylim(45 if combine_sides else 10, 90)
        # Now handle the x-axis
        if not is_human_subject:
            ax.set_xlim(-.05 if combine_sides else -1.05, 1.05)
        else:
            ax.set_xlim(-.05 if combine_sides else -0.6, 0.6)
        ax.legend(loc="upper left", fontsize="x-small")
        if save_figs:
            full_save_fp = Path(
                f"{save_fp}{title}_{sub_title}"
                f"{'_by_subject' if sec_clr_is_gray else '_by_session'}"
                "_psych_.svg")
            full_save_fp.parent.mkdir(exist_ok=True)
            fig.savefig(full_save_fp)
        plt.show()

    if df.Name.nunique() > 1:
        sub_title = "All Human Subjects" if is_human_subject else "All Mice"
        processSubject(df, grp_by_col="Name", sub_title=sub_title,
                       sec_clr_is_gray=False,
                       grp_by_legend=single_subjects_in_legend,
                       many_subjects=True)
    if not plot_single_subjects:
        return
    for subject, subject_df in df.groupby("Name"):
        processSubject(subject_df, grp_by_col=["Name", "Date", "SessionNum"],
                       sub_title=subject, sec_clr_is_gray=True,
                       grp_by_legend=False, many_subjects=False)


def plotPsych(df, title, by_subject, by_session, combine_sides,
              is_human_subject, nfits=None, save_fp="", save_figs=False,
              ax=None, default_color="k", subject_name=None, plot_points=True):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        should_show = True
    else:
        should_show = False
    if subject_name is None:
        subject_name = ("All Subjects" if not df.Name.nunique() == 1 else
                        df.Name.iloc[0])

    full_title = f"{title} - {subject_name}"
    ax = _psychAxes(full_title, combine_sides=combine_sides, ax=ax)
    if df.Name.nunique() == 1:
        by_subject = False
    c = default_color if not by_subject else None
    color_cycle = 0
    labelled_single_sess_once = False # Have an entry in the legend
    def processSubject(subject, subject_df, linewidth_for_overall=None):
        nonlocal color_cycle, labelled_single_sess_once
        color = c if c is not None else f"C{color_cycle}"
        color_cycle += 1
        if by_session:
            for sess, ses_df in subject_df.groupby([
                                                 "Name", "Date", "SessionNum"]):
                # sess_str = (f"{sess[0]} - {sess[1].strftime('%Y-%m-%d')} - "
                #             f"Sess #{sess[2]} ({len(ses_df)} trials)")
                if not labelled_single_sess_once:
                    labelled_single_sess_once = True
                    sess_str = "Single Session"
                else:
                    sess_str=None
                _fitPsych(_getGroups(ses_df, combine_sides, is_human_subject),
                          many_subjects=False, ax=ax, label=sess_str, nfits=nfits,
                          combine_sides=combine_sides,
                          color=color, linewidth=0.5, alpha=0.3,
                          plot_points=False)
        if True: # Add a flag on whether to plot overall per subject when needed
            label = f"{subject} ({len(subject_df):,} Trials)"
            _fitPsych(_getGroups(subject_df, combine_sides, is_human_subject),
                      ax=ax, label=label, many_subjects=df.Name.nunique() > 1,
                      combine_sides=combine_sides,  color=color,
                      linewidth=(2 if linewidth_for_overall is None else
                                 linewidth_for_overall),
                      plot_points=plot_points)
    if by_subject:
        for subject, subject_df in df.groupby("Name"):
            processSubject(subject, subject_df)
        if True: # Add a flag on whether to plot overall for all subjects
                 # when needed
            by_session = False # Force plotting of just overall
            c = default_color
            processSubject("All Subjects", df, linewidth_for_overall=3)
    else:
        print("Plotting overall psychometric")
        processSubject(subject_name, df)
    ax.legend(loc="upper left", fontsize="x-small")
    if should_show and save_figs:
        fig.savefig(f"{save_fp}psych_{full_title}"
                    f"{'_by_subject' if by_subject else ''}"
                    f"{'_by_session' if by_session else ''}.svg")
    if should_show:
        plt.show()

def _getGroups(df, combine_sides, is_human_subject):
    # dv_bins = [.01,
    #            .05, .15, 1.1]
    # dv_bins = [.01,
    #            .02, .03, .04, .05, .06, .07, #.08, .16,
    #            .32, 1.1]
    if is_human_subject:
        dv_bins = [.0101, .05,
                    .1, .2, .5, 1.1]
    else:
        dv_bins = [.01,
                    .02, .04, .08, .16,
                    .32, .64, 1.1]
    if not combine_sides:
        dv_bins = [-val for val in dv_bins[::-1]] + [0] + dv_bins
        cut_col = df.DV
    else:
        dv_bins = [0] + dv_bins
        cut_col = df.DV.abs()
    # print("dv_bins:", dv_bins)
    return df.groupby(pd.cut(cut_col, bins=dv_bins))

def getGroupsDVstr(df, combine_sides):
    assert "DVstr" in df.columns, "DVstr column not found"
    groupby_col = df.DVstr
    if not combine_sides:
        groupby_col = [groupby_col, df.DV < 0]
    return df.groupby(groupby_col)


def _slowFastPsychSubject(df, title, is_human_subject, combine_sides=False,
                          plot_typical=False, ax=None, many_subjects=False,
                          plot_points=True, subject_plot_points=True,
                          ls=None, y_rng=None, save_fp="",
                          save_figs=False, nfits=None, ncpus=None):
    if save_figs:
        assert len(save_fp), "Please provide a save folder path"

    def makeTitleStr(label, _df):
        _df = _df[_df.calcStimulusTime.notnull()]
        num_trials = len(
           _df[["Name", "Date", "SessionNum", "TrialNumber"]].drop_duplicates())
        num_sessions = len(
                          _df[["Name", "Date", "SessionNum"]].drop_duplicates())
        n_subj = len(_df.Name.unique())
        num_subjects_str = f"{n_subj} Subjects / " if n_subj > 1 else ""
        return (f"{label}\n"
                 f"(n={num_subjects_str}{num_sessions:,} Session"
                 # s for plural sessions
                 f"{'s' if num_sessions > 1 else ''} / "
                 f"{num_trials:,} Trials)")

    if ax is None:
        should_show = True
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        should_show = False

    q1_df = df[df.quantile_idx == 1]
    # print("q1 len:", len(q1_df), "Total len:", len(df))
    fast_subj = _fitPsych(_getGroups(q1_df, combine_sides, is_human_subject),
                                     many_subjects=many_subjects,
                                     ax=ax, color="red", ls=ls, 
                                     label=makeTitleStr("Fast", q1_df),
                                     combine_sides=combine_sides, nfits=nfits, 
                                     plot_points=plot_points,
                                     subject_plot_points=subject_plot_points,
                                     ncpus=ncpus)[3]
    if plot_typical:
        q2_df = df[df.quantile_idx == 2]
        typ_subj = _fitPsych(_getGroups(q2_df, combine_sides, is_human_subject),
                                        many_subjects=many_subjects,
                                        ax=ax, color="orange", 
                                        label=makeTitleStr("Typical", q2_df),
                                        ls=ls, combine_sides=combine_sides, 
                                        plot_points=plot_points,
                                        subject_plot_points=subject_plot_points,
                                        nfits=nfits, ncpus=ncpus)[3]
    q3_df = df[df.quantile_idx == 3]
    slow_subj = _fitPsych(_getGroups(q3_df, combine_sides, is_human_subject),
                                     many_subjects=many_subjects,
                                     ax=ax, color="yellow", ls=ls,
                                     label=makeTitleStr("Slow", q3_df),
                                     combine_sides=combine_sides,
                                     plot_points=plot_points,
                                     subject_plot_points=subject_plot_points,
                                     nfits=nfits,ncpus=ncpus)[3]

    ax = _psychAxes("", combine_sides=combine_sides, ax=ax)
    if many_subjects:
        if not is_human_subject:
            ax.set_xlim(0 if combine_sides else -1.05, 1.05)
            if y_rng is not None:
                ax.set_ylim(*y_rng)
            # else:
            #     ax.set_ylim(45 if combine_sides else 10, 90)
        else:
            LIM = 1 #0.6
            ax.set_xlim(0 if combine_sides else -LIM, LIM)
        if plot_points:
            res_df = {"subject":[], "dv":[],
                    "fast_perf":[], "slow_perf":[],
                    "fast_count":[], "slow_count":[]}
            for stim_interval in fast_subj.keys():
                fast_pts_and_counts = fast_subj[stim_interval]
                # Find matching slow subject points
                slow_pts_and_counts = [val for key, val in slow_subj.items()
                                        if key[1] == stim_interval[1]]
                dv = stim_interval[0]
                if len(slow_pts_and_counts):
                    slow_pts_and_counts = slow_pts_and_counts[0].copy()
                else:
                    slow_pts_and_counts = {}
                all_subjects = set(fast_pts_and_counts.keys()).union(
                                                        slow_pts_and_counts.keys())
                for subject in all_subjects:
                    subject_fast = fast_pts_and_counts.get(subject, (np.nan, 0))
                    subj_slow = slow_pts_and_counts.get(subject, (np.nan, 0))
                    res_df["subject"].append(subject)
                    res_df["dv"].append(dv)
                    res_df["fast_perf"].append(subject_fast[0])
                    res_df["slow_perf"].append(subj_slow[0])
                    res_df["fast_count"].append(subject_fast[1])
                    res_df["slow_count"].append(subj_slow[1])
            res_df = pd.DataFrame(res_df)
            # display(res_df)
            # Run paired t-test per DV with Holm-Bonferroni correction
            from statsmodels.stats.multitest import multipletests

            # Long format: one row per (subject, dv, condition)
            perf_long = res_df.melt(id_vars=["subject", "dv"],
                                    value_vars=["fast_perf", "slow_perf"],
                                    var_name="condition",
                                    value_name="perf",
                                    ).dropna(subset=["perf"]).copy()
            perf_long["condition"] = perf_long["condition"].map(
                                     {"fast_perf": "Fast", "slow_perf": "Slow"})

            # Treat dv as categorical for RM-ANOVA levels
            perf_long["dv"] = perf_long["dv"].astype(str)

            # Build subject x dv table with both conditions
            sd = perf_long.pivot_table(index=["subject", "dv"],
                                       columns="condition",
                                       values="perf",
                                       aggfunc="mean")
            # drop dv-rows missing either condition
            sd = sd.dropna(subset=["Fast", "Slow"], how="any")
            # --- Strict per-DV paired Fast vs Slow tests + Holm (FWER) ---
            assert not sd.empty, (
                "No paired Fast/Slow entries remain after dropping NaNs. "
                "Check that both fast_perf and slow_perf exist for subjects within DV bins."
            )

            dv_list = sorted(sd.reset_index()["dv"].unique())
            assert len(dv_list) > 0, "No DV bins available for testing."

            MIN_PAIRS = 2  # set to 3+ if you want stricter requirements

            # Hard-fail if any DV bin has insufficient paired subjects
            for dv in dv_list:
                tmp = sd.xs(dv, level="dv")
                assert tmp.shape[0] >= MIN_PAIRS, (
                    f"Not enough paired subjects for dv={dv}. "
                    f"Have {tmp.shape[0]}, need >= {MIN_PAIRS}."
                )
            # Run paired tests (two-sided by default) for every DV bin
            pvals, tstats = [], []
            for dv in dv_list:
                tmp = sd.xs(dv, level="dv")
                t, p = stats.ttest_rel(tmp["Fast"], tmp["Slow"], nan_policy="omit")
                tstats.append(float(t))
                pvals.append(float(p))

            # Holm family-wise error rate correction across DV bins
            reject, p_adj, _, _ = multipletests(pvals, method="holm")
            print("\nFast vs Slow per-dv paired tests (Holm-FWER corrected):")
            for dv, t, p, pa, r in zip(dv_list, tstats, pvals, p_adj, reject):
                print(f"  dv={dv}: t={t:.4f}, p={p:.4g}, p_holm={pa:.4g}, reject={bool(r)}")
                # Annotate only significant DV bins (as you currently do)
                if not r:
                    continue
                dv_f = float(dv)
                dv_df = res_df[(res_df.dv - dv_f).abs() < 1e-6]
                if subject_plot_points:
                    dv_max_y = 100 * dv_df[["fast_perf", "slow_perf"]].max().max()
                else:
                    dv_max_y = 100 * max(dv_df.fast_perf.mean() + dv_df.fast_perf.sem(),
                                         dv_df.slow_perf.mean() + dv_df.slow_perf.sem())
                stars = ("***" if pa <= 0.001 else
                         "**" if pa <= 0.01 else
                         "*")
                ax.text(dv_f, dv_max_y, stars,
                        ha="center", va="bottom", fontsize="large", color="k")


    ax.legend(loc="upper left" if not combine_sides else "lower right",
              fontsize="x-small")
    ax.set_title(title)
    if save_figs:
        full_save_fp = Path(
                    f"{save_fp}slow_fast_reaction_time"
                    f"{'_with_typical' if plot_typical else ''}_{title}.svg")
        full_save_fp.parent.mkdir(exist_ok=True)
        plt.savefig(full_save_fp)
    if should_show:
        plt.show()


def _psychAxes(subject_name="", ax=None, combine_sides=False):
    if len(subject_name):
        subject_name = f" {subject_name}"
    title=f"Psychometric Stim{subject_name}"
    x_label= f"RDK Coherence"
    if ax is None:
        ax = plt.axes()
    #axes.set_ylim(-.05, 1.05)
    ax.set_title(title)
    ax.set_ylim(45 if combine_sides else 0, 100)
    ax.set_xlim(0 if combine_sides else -1.05, 1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Choice {'Correct' if combine_sides else 'Left'} (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    # I wonder if we can use PercentFormatter for the x-axis as well
    x_ticks = np.arange(0,1.1,0.2) if combine_sides else np.arange(-1,1.1,0.4)
    def cohrStr(tick):
        cohr = int(round(100*tick))
        L_str = "L" if not combine_sides else ""
        return f"{abs(cohr)}%{'R' if cohr < 0 else '' if cohr==0 else L_str}"
    x_labels = list(map(cohrStr, x_ticks))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    if not combine_sides:
        ax.axvline(x=0, color='gray', linestyle='dashed', zorder=-10)
    ax.axhline(y=50, color='gray', linestyle='dashed', zorder=-10)
    [ax.spines[_dir].set_visible(False) for _dir in ["right", "top"]]
    return ax

def _fitPsych(dv_df_bins, ax, *, many_subjects, combine_sides=False, nfits=None,
              plot_points=True, correct_col="ChoiceCorrect",
              subject_plot_points=True,
              left_col="ChoiceLeft", ncpus=None, **kargs):
    parstart = np.array([.05,  1,   0.5,  0.5])
    parmin =   np.array([-1,   0.,   0.,  0])
    parmax =   np.array([1,    200.,  1,  1])
    if nfits is None:
        nfits = 100

    stims, stim_count, stim_ratio_correct, stim_ratio_sem = [], [], [], []
    subjects_pts = {}
    for dv_interval, dv_df in dv_df_bins:
        if combine_sides:
            dv_df = dv_df.copy()
            dv_df["DV"] = dv_df.DV.abs()
        if many_subjects:
            dv_df = dv_df.groupby("Name")
        DV = dv_df.DV
        dv_group = DV.mean()
        perf_col = dv_df[correct_col] if combine_sides else dv_df[left_col]
        if many_subjects:
            dv_group = dv_group.mean()
            perf_col = perf_col.mean()
        # print("DV mean:", dv_group, "len:", len(dv_df), "perf mean:", perf_col.mean())
        stims.append(dv_group)
        stim_count.append(len(dv_df))
        stim_ratio_correct.append(perf_col.mean())
        if plot_points and len(perf_col):
            # TODO: Remove the root cause of those few trials from source
            # TODO: Check if we have more than one subject, if so, then mean
            # and SEM should be calculated per subject first, then overall
            nan_perf = np.isnan(perf_col)
            nan_prcnt = 100*nan_perf.sum()/len(perf_col)
            assert (nan_perf.sum() < 5 or nan_prcnt < 0.5), (
                    f"Found {nan_perf.sum()} Trials ({nan_prcnt:.2f}%)")
            perf_col = perf_col[~nan_perf]
            stim_ratio_sem.append(stats.sem(perf_col))
            if many_subjects:
                cur_subjects_pts = {}
                for subject, subject_df in dv_df: # It's already a groupby object
                    subject_perf = (subject_df[correct_col] if combine_sides
                                    else subject_df[left_col])
                    subject_perf = subject_perf[~np.isnan(subject_perf)]
                    cur_subjects_pts[subject] = (subject_perf.mean(),
                                                len(subject_df))
                # subjects_pts.append(cur_subjects_pts)
                subjects_pts[(dv_group, dv_interval)] = {}
                for subject, (pt, cnt) in cur_subjects_pts.items():
                    subjects_pts[(dv_group, dv_interval)][subject] = (pt, cnt)
        else:
            stim_ratio_sem.append(0)
        if np.isnan(stim_ratio_sem[-1]):
            print("Mean:", 100*perf_col.mean(), "- SEM:", 100*stim_ratio_sem[-1], "- len(perf_col):", len(perf_col))
        # if np.isnan(stim_ratio_sem[-1]):
        #     display(perf_col[np.isnan(perf_col)])
    # Check if default values are used, in such case remove the last parameter
    pars, fitFn = _psychFitBasic(stims=stims, stim_count=stim_count,
                                 nfits=nfits,
                                 stim_ratio_correct=stim_ratio_correct,
                                 combine_sides=combine_sides,
                                 parstart=parstart, parmin=parmin,
                                 parmax=parmax, ncpus=ncpus)
    _range = np.arange(0 if combine_sides else -1, 1, 0.02)
    y_fit = fitFn(_range) * 100
    # print(f"y_fit: {y_fit}")
    ax.plot(_range, y_fit, **kargs)
    if plot_points:
        kargs = kargs.copy()
        kargs.pop("label")
        if "color" in kargs:
            c = kargs["color"]
        elif "c" in kargs:
            c = kargs["c"]
        else:
            c = None
        if c is not None:
            kargs["mfc"] = c
            kargs["mec"] = 'w' if c == 'k' else c
        stim_ratio_correct = np.array(stim_ratio_correct) * 100
        stim_ratio_sem = np.array(stim_ratio_sem) * 100
        # print(f"Stimulus: {stims}\nCorrect: {stim_ratio_correct}\nSEM: {stim_ratio_sem}")
        ax.errorbar(stims, stim_ratio_correct, yerr=stim_ratio_sem, fmt='o',
                    ms=8, elinewidth=2, **kargs)
        if subject_plot_points:
            if many_subjects:
                for stim in stims:
                    if np.isnan(stim):
                        continue
                    # Find stim, interval pair that matches
                    stim_interval = [key for key in subjects_pts.keys()
                                     if key[0] == stim]
                    assert len(stim_interval) == 1, (
                                       "Stimulus mismatch:", stim, subjects_pts)
                    subj_pts_and_counts = subjects_pts[stim_interval[0]]
                    assert len(subj_pts_and_counts) > 0, (
                                            "No subject points for stim:", stim)
                    subj_pts, subj_counts = zip(*subj_pts_and_counts.values())
                    subj_pts = np.array(subj_pts) * 100
                    # Filter out outliers out of y-axis range
                    if combine_sides:
                        mask = subj_pts >= 40
                        if not np.any(mask):
                            continue
                        subj_pts = subj_pts[mask]
                        subj_counts = np.array(subj_counts)[mask]
                    ax.scatter([stim]*len(subj_pts), subj_pts, c=c,
                            lw=.5, marker='X', alpha=1, s=15,
                            edgecolors=None if c != 'k' else 'w',
                            zorder=10, gid=f"subject_points_{c}_{stim}")
                    # Annotate with number of trials per subject
                    for subj_pt, cnt in zip(subj_pts, subj_counts):
                        ax.text(stim + 0.02, subj_pt,
                                f"{cnt} trials", ha='left', va='center',
                                fontsize=3)
            else:
                # Just annotate with number of trials
                for stim in stims:
                    cnt = stim_count[stims.index(stim)]
                    if cnt == 0:
                        continue
                    y = stim_ratio_correct[stims.index(stim)]
                    if combine_sides and y < 40:
                        continue # This gets rendered out of the figure area
                    ax.text(stim + 0.02, y,
                            f"{cnt} trials", ha='left', va='center',
                            fontsize=6)
    return pars[0], pars[1], (_range, y_fit), subjects_pts


_parstart = np.array([.05,  1,   0.5,  0.5])
_parmin =   np.array([-1,   0.,   0.,  0])
_parmax =   np.array([1,    200.,  1,  1])
def _psychFitBasic(stims, stim_count, stim_ratio_correct, combine_sides,
                   *, nfits, parstart=None, parmin=None, parmax=None,
                   ncpus=None):
    if parstart is None:
        parstart = _parstart.copy()
    if parmin is None:
        parmin = _parmin.copy()
    if parmax is None:
        parmax = _parmax.copy()
    data = np.array([stims, stim_count, stim_ratio_correct])
    #print(data)
    if not combine_sides:
        P_model = 'erf_psycho_2gammas'
    else:
        P_model = 'erf_psycho'
    num_params = 4 if "2gammas" in P_model else 3
    parstart = parstart[:num_params]
    parmin = parmin[:num_params]
    parmax = parmax[:num_params]
    pars, L = psychofit.mle_fit_psycho(data=data, P_model=P_model,
                                       parstart=parstart, parmin=parmin,
                                       parmax=parmax, nfits=nfits,
                                       ncpus=ncpus)
    fn = getattr(psychofit, P_model) # Get the right function from the module
    from functools import partial
    wrapFitFn = partial(fn, pars)
    return pars, wrapFitFn