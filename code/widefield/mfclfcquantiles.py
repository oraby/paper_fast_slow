'''MFC-vs-LFC activity difference across sampling-duration quantiles.

Backend for the widefield notebook's "Plot Difference between MFC and LFC
across Quantiles" figures. The measurement -- LFC (ALM) minus MFC (M2) mean
dF/F, compared across the within-session sampling-duration tertiles
(Fast/Typical/Slow) -- is made over a short window inside the sampling epoch.
`SamplingAnchor` selects where that window sits:

* `SamplingAnchor.END` -- the window ending at movement onset, i.e. the point of
  making a decision.
* `SamplingAnchor.MID` -- a window centred on the temporal middle of each
  trial's *own* sampling epoch.

Everything downstream of the alignment (`getTrialsMeanActivity`,
`plotMFCLFCQuantiles`) is shared between the two anchors.
'''
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from ..common.imaging import alignSampling
from ..pipeline import pipeline, tracesrestructure
from ..pipeline.pipeline import DFProcessor, assertTraceLimits

#: Width of the window the mean activity is averaged over. At the ~30Hz
#: acquisition rate this is 3 samples.
WINDOW_WIDTH_SEC = 0.1
#: Position of the MID anchor within the sampling epoch (0.5 == its middle).
MID_FRACTION = 0.5
#: The within-session sampling-duration tertiles, as split by `alignSampling`.
QUANTILE_LABEL = {1: "Fast", 2: "Typical", 3: "Slow"}
MFC_REGION = "M2"
LFC_REGION = "ALM"


class SamplingAnchor(Enum):
    '''Where inside the sampling epoch the MFC/LFC difference is measured.'''
    END = "end"
    MID = "mid"

    @property
    def descr(self):
        '''Human-readable epoch position, used in the figure title.'''
        return {SamplingAnchor.END: "Movement to Lateral Port",
                SamplingAnchor.MID: "Mid Sampling"}[self]

    @property
    def fig_suffix(self):
        '''END keeps the original (already published) figure file name.'''
        return {SamplingAnchor.END: "", SamplingAnchor.MID: "_mid_sampling"}[
                                                                          self]


class AlignTraceWithinEpoch(DFProcessor):
    '''Cut a fixed-width window anchored at a fraction of an epoch's duration.

    `tracesrestructure.AlignTraceAroundEpoch` can only anchor on an epoch's
    *start*. This one anchors at ``start + fraction*(end - start)`` of each
    matched epoch -- ``fraction=0.5`` being its temporal middle -- and keeps
    ``width_sec`` worth of samples centred on that anchor. For an even sample
    count the extra sample is taken before the anchor.

    The width is held constant in samples across trials, so quantiles that
    differ in sampling duration are still compared over equally long windows.
    It is deliberately not clipped to the epoch: a trial whose epoch is shorter
    than ``width_sec`` spills into the neighbouring epochs, matching how the
    end-of-sampling window is cut. `alignSampling` only keeps trials with at
    least ``time_before_sampling``/``time_after_sampling`` of neighbouring
    epoch acquired, so the spill stays inside acquired data.
    '''
    def __init__(self, epoch_name_li, fraction, width_sec,
                 use_epoch_name=None):
        assert 0 <= fraction <= 1, f"fraction must be in [0, 1], got {fraction}"
        assert width_sec > 0, f"width_sec must be > 0, got {width_sec}"
        self._epoch_name_li = epoch_name_li
        self._fraction = fraction
        self._width_sec = width_sec
        self._use_epoch_name = use_epoch_name

    def process(self, data):
        new_rows = []
        epochs_df = data[data.epoch.isin(self._epoch_name_li)]
        for _row_idx, row in epochs_df.iterrows():
            width_idx = max(int(round(self._width_sec*row.acq_sampling_rate)),
                            1)
            anchor_idx = row.trace_start_idx + int(round(
                  self._fraction*(row.trace_end_idx - row.trace_start_idx)))
            start_idx = max(0, anchor_idx - width_idx//2)
            end_idx = start_idx + width_idx - 1
            row = row.copy()
            if self._use_epoch_name is not None:
                row.epoch = self._use_epoch_name
            row["org_start_idx"] = start_idx
            row["org_mid_idx"] = anchor_idx
            row["org_end_idx"] = end_idx
            row.trace_start_idx = start_idx
            row.trace_end_idx = end_idx
            assertTraceLimits(row)
            new_rows.append(row)
        return pd.DataFrame(new_rows)

    def descr(self):
        return (f"Cut a {self._width_sec:g}s window centred at "
                f"{self._fraction:.0%} of {self._epoch_name_li}")


def alignSamplingWindow(df, anchor, normalization, time_before_sampling,
                        time_after_movement, num_quantiles=3,
                        width_sec=WINDOW_WIDTH_SEC):
    '''Cut, per trial, the `anchor` window of the sampling epoch.

    Returns one row per trial, holding only that window's traces, along with
    the `quantile_idx` (1: Fast, 2: Typical, 3: Slow) the trial's sampling
    duration falls in within its session.

    Note the two anchors' windows differ by one sample: END reuses
    `AlignTraceAroundEpoch`, whose window is the ``width_sec`` preceding
    movement onset *plus* the movement-onset sample itself (4 samples at
    ~30Hz), whereas MID is a true ``width_sec`` wide (3 samples).
    '''
    anchor = SamplingAnchor(anchor)
    df_around_sampling_q = alignSampling(
                              df, normalization=normalization,
                              time_before_sampling=time_before_sampling,
                              time_after_sampling=time_after_movement,
                              normalize_epoch_time=False,
                              num_quantiles=num_quantiles,
                              normalize_sessions_before_splitting=True,
                              concatenate_final_epochs=False)
    if anchor is SamplingAnchor.END:
        aligner = tracesrestructure.AlignTraceAroundEpoch(
                              epoch_name_li=["Movement to Lateral Port"],
                              time_before_sec=width_sec, time_after_sec=0,
                              limit_to_epoch_start=False,
                              limit_to_epoch_end=False)
    else:
        aligner = AlignTraceWithinEpoch(epoch_name_li=["Sampling"],
                                        fraction=MID_FRACTION,
                                        width_sec=width_sec,
                                        use_epoch_name=anchor.descr)
    chain = pipeline.Chain(
        pipeline.BySession(),
            aligner,
            pipeline.By("TrialNumber"),
                # END emits a pre-anchor and an anchor row that must be joined;
                # MID emits a single row, which this just cuts down to size.
                tracesrestructure.ConcatEpochs(ignore_existing_concat=True,
                                               assume_continuos=True),
        pipeline.RecombineResults(),
    )
    return chain.run(df_around_sampling_q)


def getTrialsMeanActivity(df, include_raw):
    '''Flatten per-trial windowed traces into one row per (trial, region).'''
    res_dict = {"BrainRegion":[],
                "Hemisphere":[],
                "Stimulus":[],
                "SamplingType":[],
                "ShortName":[],
                "Name":[],
                "TrialNumber":[],
                "StimulusTime":[],
                "mean_activity":[],
                "ChoiceCorrect":[],
                "Direction":[],
                "Difficulty":[],}
    if include_raw:
        res_dict["raw"] = []
    if "quantile_idx" in df.columns:
        res_dict["quantile_idx"] = []
        track_quantiles = True
    else:
        track_quantiles = False
    df = df[df.ChoiceCorrect.notnull()]
    for sampling_type, sampling_type_df in df.groupby("SamplingType"):
        for sess, sess_df in sampling_type_df.groupby("ShortName"):
            for trial_num, trial_df in sess_df.groupby("TrialNumber"):
                assert len(trial_df) == 1
                trial_df = trial_df.iloc[0]
                traces_dict = pipeline.getRowTracesSets(trial_df)["neuronal"]
                s, e = trial_df["trace_start_idx"], trial_df["trace_end_idx"] + 1
                for trace_id, traces_arr in traces_dict.items():
                    assert traces_arr.ndim == 1
                    traces_arr = traces_arr[s:e]
                    mean_activity = traces_arr.mean()
                    if "_" in trace_id:
                        brain_region, _dir = trace_id.rsplit("_", 1)
                    else:
                        brain_region, _dir = trace_id, "Bi"
                    res_dict["BrainRegion"].append(brain_region)
                    res_dict["Hemisphere"].append(_dir)
                    res_dict["SamplingType"].append(trial_df.SamplingType)
                    res_dict["Stimulus"].append(trial_df.Stimulus)
                    res_dict["ShortName"].append(sess)
                    res_dict["Name"].append(trial_df.Name)
                    res_dict["TrialNumber"].append(trial_num)
                    res_dict["StimulusTime"].append(trial_df.calcStimulusTime)
                    res_dict["mean_activity"].append(mean_activity)
                    res_dict["ChoiceCorrect"].append(trial_df.ChoiceCorrect)
                    res_dict["Direction"].append(trial_df.ChoiceLeft)
                    res_dict["Difficulty"].append(trial_df.DVstr)
                    if include_raw:
                        res_dict["raw"].append(traces_arr)
                    if track_quantiles:
                        res_dict["quantile_idx"].append(trial_df.quantile_idx)
    return pd.DataFrame(res_dict)


def _splitData(q_df):
    mfc = q_df[q_df.BrainRegion == MFC_REGION].groupby("ShortName")
    lfc = q_df[q_df.BrainRegion == LFC_REGION].groupby("ShortName")
    return mfc, lfc


def calcMFC_LFCDistance(q_df):
    mfc, lfc = _splitData(q_df)
    mfc_mean = mfc.mean_activity.mean().mean()
    lfc_mean = lfc.mean_activity.mean().mean()
    return lfc_mean - mfc_mean


def avgRawActivity(q_df, brain_region):
    mfc, lfc = _splitData(q_df)
    area_grpby = mfc if brain_region == MFC_REGION else lfc
    traces_avg = area_grpby.apply(lambda grp: grp.raw.values.mean(axis=0))
    traces_avg = traces_avg.mean(axis=0)  # mean across sessions
    return traces_avg


def _stack_series_of_arrays(s: pd.Series) -> np.ndarray:
    '''Turn a Series of 1D arrays into a 2D array (n_units x time).'''
    arrs = [np.asarray(a) for a in s.dropna().values]
    if len(arrs) == 0:
        return np.empty((0, 0))
    return np.vstack(arrs)


def plotMFCLFCQuantiles(df, groupby_by: str,
                        anchor=SamplingAnchor.END,
                        exclude_animals_containing=None,
                        save_figs: bool = False,
                        fig_save_prefix: str = "."):
    '''Plot the LFC - MFC difference per quantile, at `anchor`, + RM-ANOVA.'''
    anchor = SamplingAnchor(anchor)
    if exclude_animals_containing is None:
        exclude_animals_containing = []

    df = df[df.StimulusTime < 4.9].copy()
    for _str in exclude_animals_containing:
        df = df[~df.ShortName.str.contains(_str, na=False)]

    # -----------------------
    # Build long table for RM-ANOVA (robustly)
    # -----------------------
    assert groupby_by in ["Subject", "Session"]
    subject_col = "Name" if groupby_by == "Subject" else "ShortName"
    group_cols = [subject_col, "quantile_idx"]

    # Distance DV
    dist_s = df.groupby(group_cols).apply(calcMFC_LFCDistance)
    quantile_dist = dist_s.reset_index(name="delta_dist")

    # Add traces (object arrays) keyed on the same (subject, quantile_idx)
    mfc_s = df.groupby(group_cols).apply(avgRawActivity,
                                         brain_region=MFC_REGION)
    lfc_s = df.groupby(group_cols).apply(avgRawActivity,
                                         brain_region=LFC_REGION)

    quantile_dist = quantile_dist.merge(
        mfc_s.reset_index(name="avg_mfc"),
        on=group_cols,
        how="left"
    ).merge(
        lfc_s.reset_index(name="avg_lfc"),
        on=group_cols,
        how="left"
    )

    # Basic integrity: exactly one row per subject x quantile
    if not (quantile_dist.groupby(group_cols).size() == 1).all():
        raise ValueError("Found duplicate rows per (subject, quantile_idx). "
                         "Aggregate to one value per cell before ANOVA.")

    # Enforce integer quantile labels for stable ordering
    quantile_dist["quantile_idx"] = quantile_dist["quantile_idx"].astype(int)

    # -----------------------
    # Enforce balanced RM design (AnovaRM requirement)
    # -----------------------
    wide = quantile_dist.pivot(index=subject_col,
                               columns="quantile_idx",
                               values="delta_dist")

    # Keep only subjects that have all 3 quantiles
    required_levels = list(QUANTILE_LABEL)
    missing_cols = [q for q in required_levels if q not in wide.columns]
    if missing_cols:
        raise ValueError(f"Missing quantile levels in data: {missing_cols}")

    wide_bal = wide[required_levels]

    # Long (tidy) table for AnovaRM from the balanced wide table
    df_long = wide_bal.reset_index().melt(id_vars=subject_col,
                                          var_name="quantile_idx",
                                          value_name="delta_dist")
    df_long["quantile_idx"] = df_long["quantile_idx"].astype(int)

    # -----------------------
    # Repeated-measures ANOVA
    # -----------------------
    anova_res = AnovaRM(
        data=df_long,
        depvar="delta_dist",
        subject=subject_col,
        within=["quantile_idx"]
    ).fit()

    print("RM-ANOVA (AnovaRM) table:")
    print(anova_res.anova_table)

    f_stat = float(anova_res.anova_table["F Value"].iloc[0])
    p_value = float(anova_res.anova_table["Pr > F"].iloc[0])
    print(f"F-statistic: {f_stat:.6g}")
    print(f"p-value:     {p_value:.6g}")

    # -----------------------
    # Paired post-hoc (Holm), only if ANOVA significant
    # -----------------------
    res_dict = {}
    if p_value < 0.05:
        pairs = [(1, 2), (1, 3), (2, 3)]
        pvals = []
        tstats = []
        for a, b in pairs:
            t, p = stats.ttest_rel(wide_bal[a], wide_bal[b], nan_policy="omit")
            tstats.append(float(t))
            pvals.append(float(p))

        reject, p_adj, _, _ = multipletests(pvals, method="holm")
        print("\nPost-hoc paired t-tests (Holm-corrected):")
        for (a, b), t, p_raw, p_corr, r in zip(pairs, tstats, pvals, p_adj,
                                               reject):
            print(f"  {QUANTILE_LABEL[a]} vs {QUANTILE_LABEL[b]}: t={t:.4f}, "
                  f"p={p_raw:.4f}, p_holm={p_corr:.4f}, reject={bool(r)}")
            res_dict[(a, b)] = float(p_corr)

    # -----------------------
    # Plotting
    #   ax1: mean traces (LFC and MFC), grouped by quantile (from avg_* arrays)
    #   ax2: mean ± SEM of delta_dist (same DV as ANOVA)
    # -----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title("LFC and MFC Avg. Traces (per quantile)")

    # Plot traces (mean ± SEM across subjects/sessions) for each quantile
    for q in required_levels:
        sub = quantile_dist[quantile_dist["quantile_idx"] == q]
        lfc_mat = _stack_series_of_arrays(sub["avg_lfc"])
        mfc_mat = _stack_series_of_arrays(sub["avg_mfc"])

        if lfc_mat.size > 0:
            ax1.errorbar(
                x=np.arange(lfc_mat.shape[1]),
                y=lfc_mat.mean(axis=0),
                yerr=stats.sem(lfc_mat, axis=0, nan_policy="omit"),
                label=f"LFC ({QUANTILE_LABEL[q]}",
                c=("lightcoral" if q == 1 else "r" if q == 2 else "darkred")
            )
        if mfc_mat.size > 0:
            ax1.errorbar(
                x=np.arange(mfc_mat.shape[1]),
                y=mfc_mat.mean(axis=0),
                yerr=stats.sem(mfc_mat, axis=0, nan_policy="omit"),
                label=f"MFC ({QUANTILE_LABEL[q]})",
                c=("lightblue" if q == 1 else "b" if q == 2 else "darkblue")
            )
    ax1.legend(fontsize="x-small")
    ax1.spines[["top", "right"]].set_visible(False)

    # Bars for the DV tested in ANOVA: delta_dist
    means = df_long.groupby("quantile_idx")["delta_dist"].mean().reindex(
                                                              required_levels)
    sems  = df_long.groupby("quantile_idx")["delta_dist"].sem().reindex(
                                                              required_levels)

    x = np.array(required_levels)
    y = means.values
    y_sem = sems.values

    ax2.errorbar(x, y, yerr=y_sem, color="k")
    ax2.set_xticks(x)
    ax2.set_xticklabels([QUANTILE_LABEL[i] for i in required_levels])
    ax2.set_ylabel(f"δ$_{{{anchor.value}}}$(dF/F)")
    ax2.spines[["top", "right"]].set_visible(False)

    for xi, yi, si in zip(x, y, y_sem):
        ax2.annotate(f"{yi:.2f} ± {si:.2f}",
                     (xi, yi + (si if np.isfinite(si) else 0)),
                     va="bottom", ha="left")

    # Significance bars (if applicable)
    if p_value < 0.05 and res_dict:
        def draw_significance_bar(ax, x1, x2, y0, h, p_val):
            sgf_text = ("***" if p_val < 0.001 else
                        "**" if p_val < 0.01 else
                        "*" if p_val < 0.05 else "n.s.")
            ax.plot([x1, x1, x2, x2], [y0, y0 + h, y0 + h, y0], lw=1, c="k")
            ax.text((x1 + x2) * 0.5, y0 + h, sgf_text, ha="center",
                    va="bottom", color="k")

        y_max = np.nanmax(y + y_sem) if np.all(np.isfinite(y_sem)) else \
                np.nanmax(y)
        y_max = float(y_max) - 0.02
        h = 0.03

        for (a, b) in [(1, 2), (1, 3), (2, 3)]:
            p_adj = res_dict.get((a, b), None)
            if p_adj is not None and p_adj < 0.05:
                draw_significance_bar(ax2, a, b, y_max, h, p_adj)
                y_max += h + 0.02

    exclud_str = ("\nExcluding: " + ", ".join(exclude_animals_containing)) \
                 if exclude_animals_containing else ""
    num_sessions = df.ShortName.nunique()
    num_subjects = df.Name.nunique()
    n_str = f"N={num_sessions} sessions from {num_subjects} subjects" \
            if groupby_by == "Session" else f"N={num_subjects} subjects"
    ax2.set_title(
        f"LFC - MFC during {anchor.descr}\n"
        f"{n_str}; RM-ANOVA bars are mean ± SEM ({groupby_by}){exclud_str}",
        fontsize="small"
    )

    if save_figs:
        how = "by_" + groupby_by.lower()
        plt.savefig(f"{fig_save_prefix}/MFC_LFC_dist{anchor.fig_suffix}_"
                    f"{how}.svg", bbox_inches="tight")

    plt.show()
