from .optoprocessor import _classifyOptoConfig
from ..common.definitions import BrainRegion
from ..common.clr import BrainRegion as BrainRegionClr

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import List, Literal, Dict, Tuple

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


def optoFeedback(start_state, start_delay, max_dur, stimulus_time,
                 end_state, stim_type, df, min_choice_trials, num_iterations,
                 subject_name, z_score, plot_sem, save_prefix, save_figs=False,
                 only_brain_regions: List[BrainRegion] = [],
                 combine_control: bool = False,
                 z_score_how: Literal["Subject", "Session"] = "Session",
                 plot_as_difference: bool = False):
    """
    Top-level entry: prepares df, splits control / opto, and calls _plotReactionBars.
    """
    print("Df len:", len(df))
    if len(only_brain_regions):
        df = df[df.GUI_OptoBrainRegion.isin(only_brain_regions)]

    name = _classifyOptoConfig(start_state, start_delay, max_dur, stimulus_time,
                               end_state, stim_type)
    name = f"{subject_name} - {name}"

    # Remove over-stayed trials
    df = df[df.calcStimulusTime < df.GUI_StimulusTime - 0.1]

    CHECK_PREV = True
    col_prefix = "Prev" if CHECK_PREV else ""
    df_cntrl = df[df[f"{col_prefix}OptoEnabled"] == 0]
    df_opto = df[df[f"{col_prefix}OptoEnabled"] == 1]

    print("TODO: **********TODO: Use min_choice_trials")

    _plotReactionBars(
        df_cntrl, df_opto,
        df_col="calcStimulusTime",
        start_delay=start_delay,
        max_dur=max_dur,
        name=name,
        save_fp=f"{save_prefix}_rt_bars.svg",
        save_figs=save_figs,
        num_iterations=num_iterations,    # kept for API compatibility, not used
        z_score_time=z_score,
        combine_control=combine_control,
        z_score_how=z_score_how,
        plot_as_difference=plot_as_difference,
    )

    return df


# -------------------- helpers -------------------- #

def _apply_zscore(df: pd.DataFrame, df_col: str,
                  z_score_time: bool,
                  z_score_how: Literal["Subject", "Session"]) -> pd.DataFrame:
    """
    Z-score df_col within subject or session.
    """
    if not z_score_time:
        return df

    df = df.copy()
    if z_score_how == "Subject":
        grp = df.groupby("Name")[df_col]
    else:
        assert z_score_how == "Session"
        grp = df.groupby(["Name", "Date", "SessionNum"])[df_col]

    means = grp.transform("mean")
    stds = grp.transform("std")
    df[df_col] = (df[df_col] - means) / stds
    return df


def _sigstar(p: float) -> str:
    """
    Standard significance star mapping on (Holm-corrected) p-values.
        p < 0.001 -> ***
        p < 0.01  -> **
        p < 0.05  -> *
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def _add_sig_bracket(ax, x1, x2, y, text, h=None):
    """
    Draw a significance bracket between x1 and x2 at height y
    and return the top y (including the bracket height).
    """
    ymin, ymax = ax.get_ylim()
    # Dynamic bracket height based on plot range
    if h is None:
        h = (ymax - ymin) * 0.02 if ymax > ymin else 0.05
    
    ax.plot([x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="black", linewidth=1)
    ax.text((x1 + x2) / 2.0, y + h, text,
            ha="center", va="bottom", fontsize=10)
    return y + h * 3  # Return new clearance height


# -------------------- main plotting / stats -------------------- #

def _plotReactionBars(df_cntrl, df_opto,
                      df_col: Literal["calcStimulusTime", "ChoiceCorrect"],
                      start_delay, max_dur, name, save_fp, save_figs,
                      num_iterations: int,   # unused, kept for API compatibility
                      z_score_time: bool = True,
                      combine_control: bool = False,
                      z_score_how: Literal["Subject", "Session"] = "Session",
                      plot_as_difference: bool = False):

    assert df_col in ["calcStimulusTime", "ChoiceCorrect"]
    if z_score_time:
        assert df_col != "ChoiceCorrect"
        assert z_score_how in ["Subject", "Session"]

    # Recombine both dfs so z-scoring & grouping is easier
    df_all = pd.concat([df_cntrl, df_opto], ignore_index=True).reset_index(drop=True)
    df_all = df_all[["Name", "Date", "SessionNum",
                     "GUI_OptoBrainRegion", "PrevOptoEnabled",
                     "PrevChoiceCorrect", df_col]]

    # z-score if requested
    df_all = _apply_zscore(df_all, df_col=df_col,
                           z_score_time=z_score_time,
                           z_score_how=z_score_how)

    # -------------------- define regions -------------------- #
    # Region labels used in stats and plotting
    if combine_control:
        region_defs = {
            "Cntrl":     (lambda d: d["PrevOptoEnabled"] == 0),
            "Opto_MFC":  (lambda d: (d["PrevOptoEnabled"] == 1) &
                                   (d["GUI_OptoBrainRegion"] == BrainRegion.M2_Bi)),
            "Opto_LFC":  (lambda d: (d["PrevOptoEnabled"] == 1) &
                                   (d["GUI_OptoBrainRegion"] == BrainRegion.ALM_Bi)),
        }
    else:
        region_defs = {
            "Cntrl_MFC": (lambda d: (d["PrevOptoEnabled"] == 0) &
                                   (d["GUI_OptoBrainRegion"] == BrainRegion.M2_Bi)),
            "Opto_MFC":  (lambda d: (d["PrevOptoEnabled"] == 1) &
                                   (d["GUI_OptoBrainRegion"] == BrainRegion.M2_Bi)),
            "Cntrl_LFC": (lambda d: (d["PrevOptoEnabled"] == 0) &
                                   (d["GUI_OptoBrainRegion"] == BrainRegion.ALM_Bi)),
            "Opto_LFC":  (lambda d: (d["PrevOptoEnabled"] == 1) &
                                   (d["GUI_OptoBrainRegion"] == BrainRegion.ALM_Bi)),
        }

    region_names = list(region_defs.keys())

    # -------------------- per-subject, per-region, per-outcome means -------------------- #
    # Build a tidy df: Subject, Region, PrevOutcome (Correct/Incorrect), RT_mean
    rows = []
    for subj, subj_df in df_all.groupby("Name"):
        for region, mask_fn in region_defs.items():
            r_df = subj_df[mask_fn(subj_df)]
            if r_df.empty:
                continue
            for prev_flag, prev_label in [(1, "Correct"), (0, "Incorrect")]:
                tmp = r_df[r_df.PrevChoiceCorrect == prev_flag]
                if tmp.empty:
                    continue
                # Average per session, then average over sessions for this subject/region/outcome
                sess_means = tmp.groupby(["Name", "Date", "SessionNum"])[df_col].mean()
                rt_mean = sess_means.mean()
                rows.append({
                    "Subject": subj,
                    "Region": region,
                    "PrevOutcome": prev_label,
                    "RT": rt_mean,
                })

    subj_region_df = pd.DataFrame(rows)
    if subj_region_df.empty:
        print("No data after filtering. Nothing to plot.")
        return

    # -------------------- wide format for plotting -------------------- #
    # res_df: one row per subject, columns: Region_prev_correct / Region_prev_incorrect
    subjects = sorted(subj_region_df["Subject"].unique())
    res_dict = {"Subject": subjects}
    for region in region_names:
        res_dict[f"{region}_prev_correct"] = [np.nan] * len(subjects)
        res_dict[f"{region}_prev_incorrect"] = [np.nan] * len(subjects)

    subj_index = {s: i for i, s in enumerate(subjects)}
    for _, row in subj_region_df.iterrows():
        s = row["Subject"]
        r = row["Region"]
        o = row["PrevOutcome"]  # "Correct" or "Incorrect"
        idx = subj_index[s]
        col = f"{r}_prev_{o.lower()}"
        res_dict[col][idx] = row["RT"]

    res_df = pd.DataFrame(res_dict)

    # -------------------- STATS: Within-Region (Correct vs Incorrect) -------------------- #
    within_p_corr = {}
    within_t = {}
    
    # Raw p-values for within comparisons
    p_raw_within = {}
    keys_within = []

    for region in region_names:
        col_corr = f"{region}_prev_correct"
        col_incorr = f"{region}_prev_incorrect"
        if col_corr not in res_df or col_incorr not in res_df:
            continue
        vals_corr = res_df[col_corr].values
        vals_incorr = res_df[col_incorr].values
        mask = ~np.isnan(vals_corr) & ~np.isnan(vals_incorr)
        if mask.sum() < 2:
            continue
        
        t, p = stats.ttest_rel(vals_corr[mask], vals_incorr[mask])
        within_t[region] = t
        p_raw_within[region] = p
        keys_within.append(region)

    # Holm correction for within-region tests
    if keys_within:
        pvals = [p_raw_within[k] for k in keys_within]
        _, pvals_corr, _, _ = multipletests(pvals, method="holm")
        within_p_corr = {k: p for k, p in zip(keys_within, pvals_corr)}

    print("\n--- Within-Condition (Correct vs Incorrect) ---")
    for region in keys_within:
        print(f"{region}: t={within_t[region]:.2f}, p_raw={p_raw_within[region]:.4f}, p_holm={within_p_corr[region]:.4f}")


    # -------------------- STATS: Across-Region (RM ANOVA + Pairwise) -------------------- #
    
    # Helper for Cross-Region analysis
    def _analyze_cross_region(outcome_label: str):
        """
        Runs RM ANOVA and Pairwise t-tests for a specific outcome (Correct or Incorrect).
        Returns a dict of significant pairwise comparisons: {(RegionA, RegionB): p_holm}
        """
        subset_df = subj_region_df[subj_region_df["PrevOutcome"] == outcome_label].copy()
        
        # Check sufficiency
        if subset_df["Region"].nunique() < 2 or subset_df["Subject"].nunique() < 2:
            print(f"\nNot enough data for {outcome_label} cross-region stats.")
            return {}

        # RM ANOVA
        print(f"\n--- Cross-Region {outcome_label} (RM ANOVA) ---")
        try:
            aov = AnovaRM(subset_df, depvar="RT", subject="Subject", within=["Region"])
            res = aov.fit()
            print(res)
        except Exception as e:
            print(f"ANOVA failed: {e}")

        # Pairwise Post-hoc
        p_raw_cross = {}
        t_cross = {}
        pairs = []
        
        reg_list = region_names
        for i in range(len(reg_list)):
            for j in range(i + 1, len(reg_list)):
                r1, r2 = reg_list[i], reg_list[j]
                
                # Get data for both regions aligned by subject
                df1 = subset_df[subset_df.Region == r1].set_index("Subject")["RT"]
                df2 = subset_df[subset_df.Region == r2].set_index("Subject")["RT"]
                
                # Intersection
                common = df1.index.intersection(df2.index)
                if len(common) < 2:
                    continue
                
                t, p = stats.ttest_rel(df1.loc[common], df2.loc[common])
                p_raw_cross[(r1, r2)] = p
                t_cross[(r1, r2)] = t
                pairs.append((r1, r2))

        # Holm Correction
        sig_pairs_map = {}
        if pairs:
            pvals = [p_raw_cross[pair] for pair in pairs]
            _, pvals_corr, _, _ = multipletests(pvals, method="holm")
            
            print(f"--- Pairwise {outcome_label} (Holm) ---")
            for pair, p_c in zip(pairs, pvals_corr):
                print(f"{pair[0]} vs {pair[1]}: t={t_cross[pair]:.2f}, p_holm={p_c:.4f}")
                sig_pairs_map[pair] = p_c
        
        return sig_pairs_map

    # Run Cross-Region Stats
    cross_sig_correct = _analyze_cross_region("Correct")
    cross_sig_incorrect = _analyze_cross_region("Incorrect")


    # -------------------- PLOTTING -------------------- #
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Define X positions
    if combine_control:
        if plot_as_difference:
            xs = [1, 2, 3]
            labels_xt = ["Control", "Opto MFC", "Opto LFC"]
        else:
            xs = [1, 2, 4, 5, 7, 8]
            labels_xt = [
                "Cntrl\nPrev\nCorrect", "Cntrl\nPrev\nIncorrect",
                "Opto\nMFC\nPrev\nCorrect", "Opto\nMFC\nPrev\nIncorrect",
                "Opto\nLFC\nPrev\nCorrect", "Opto\nLFC\nPrev\nIncorrect",
            ]
        clrs = [
            "gray",
            BrainRegionClr[BrainRegion.M2_Bi],
            BrainRegionClr[BrainRegion.ALM_Bi],
        ]
    else:
        if plot_as_difference:
            xs = [1, 2, 3, 4]
            labels_xt = ["Cntrl MFC", "Opto MFC", "Cntrl LFC", "Opto LFC"]
        else:
            xs = [1, 2, 4, 5, 7, 8, 10, 11]
            labels_xt = [
                "Cntrl\nMFC\nPrev\nCorrect", "Cntrl\nMFC\nPrev\nIncorrect",
                "Opto\nMFC\nPrev\nCorrect", "Opto\nMFC\nPrev\nIncorrect",
                "Cntrl\nLFC\nPrev\nCorrect", "Cntrl\nLFC\nPrev\nIncorrect",
                "Opto\nLFC\nPrev\nCorrect", "Opto\nLFC\nPrev\nIncorrect",
            ]
        clrs = [
            "gray", BrainRegionClr[BrainRegion.M2_Bi],
            "gray", BrainRegionClr[BrainRegion.ALM_Bi],
        ]

    # Map regions to x-coordinates
    xs_loop = xs[::2] if not plot_as_difference else xs
    region_x_map = {} # Store x coords for brackets later
    
    y_stack_tracker = 0 # To track height for brackets
    
    # Draw Bars and Within-Region Stars
    for x0, region, clr in zip(xs_loop, region_names, clrs):
        col_corr = f"{region}_prev_correct"
        col_incorr = f"{region}_prev_incorrect"

        vals_corr = res_df[col_corr].values
        vals_incorr = res_df[col_incorr].values

        if plot_as_difference:
            # Mode: Plot difference (Inc - Corr)
            mask = ~np.isnan(vals_corr) & ~np.isnan(vals_incorr)
            if mask.sum() == 0:
                continue
            diffs = vals_incorr[mask] - vals_corr[mask]
            mean_diff = diffs.mean()
            err_diff = stats.sem(diffs) if mask.sum() > 1 else 0.0

            ax.bar(x0, mean_diff, yerr=err_diff,
                   width=0.4, color=clr, edgecolor="black")
            
            region_x_map[region] = {"diff": x0}
            
            # Update max height tracker
            current_top = mean_diff + err_diff
            if current_top > y_stack_tracker:
                y_stack_tracker = current_top

            # Within-region star (is diff different from 0 / paired t-test)
            if region in within_p_corr:
                star = _sigstar(within_p_corr[region])
                if star:
                    y_stack_tracker = _add_sig_bracket(ax, x0, x0, y_stack_tracker, star, h=0.01)

        else:
            # Mode: Plot raw Correct and Incorrect bars side-by-side
            mask_corr = ~np.isnan(vals_corr)
            mask_incorr = ~np.isnan(vals_incorr)
            
            mean_corr = vals_corr[mask_corr].mean() if mask_corr.sum() > 0 else np.nan
            mean_incorr = vals_incorr[mask_incorr].mean() if mask_incorr.sum() > 0 else np.nan
            err_corr = stats.sem(vals_corr[mask_corr]) if mask_corr.sum() > 1 else 0.0
            err_incorr = stats.sem(vals_incorr[mask_incorr]) if mask_incorr.sum() > 1 else 0.0

            x1 = x0 + 1

            ax.bar(x0, mean_corr, yerr=err_corr,
                   width=0.4, color=clr, edgecolor="g")
            ax.bar(x1, mean_incorr, yerr=err_incorr,
                   width=0.4, color=clr, edgecolor="r")

            region_x_map[region] = {"correct": x0, "incorrect": x1}

            # Update max height tracker
            current_top = max(mean_corr + err_corr, mean_incorr + err_incorr)
            if current_top > y_stack_tracker:
                y_stack_tracker = current_top

            # Within-region bracket (Correct vs Incorrect)
            if region in within_p_corr:
                star = _sigstar(within_p_corr[region])
                if star:
                    # Draw local bracket
                    y_stack_tracker = _add_sig_bracket(ax, x0, x1, y_stack_tracker, star)

    # -------------------- Draw Cross-Region Brackets -------------------- #
    # We draw brackets if p_holm < 0.05
    
    def _draw_cross_brackets(sig_map, key_suffix):
        nonlocal y_stack_tracker
        # Sort pairs by distance to draw smaller brackets first (aesthetic)
        sorted_pairs = sorted(sig_map.keys(), key=lambda p: abs(region_names.index(p[0]) - region_names.index(p[1])))
        
        for r1, r2 in sorted_pairs:
            p_val = sig_map[(r1, r2)]
            star = _sigstar(p_val)
            if not star: continue
            
            # Determine x coordinates
            if plot_as_difference:
                x_a = region_x_map[r1]["diff"]
                x_b = region_x_map[r2]["diff"]
            else:
                x_a = region_x_map[r1][key_suffix]
                x_b = region_x_map[r2][key_suffix]
            
            y_stack_tracker = _add_sig_bracket(ax, x_a, x_b, y_stack_tracker, star)

    if plot_as_difference:
        # For difference plots, we usually compare the "Difference" metric across regions.
        # However, the user asked for "Correct" and "Incorrect" ANOVA separately.
        # If we are plotting Difference, we mathematically should run ANOVA on the difference values.
        # But if the request is strict, we might skip drawing "Correct vs Correct" brackets on a "Difference" plot.
        # Assuming we treat the Difference metric as the variable of interest for cross-region in this mode:
        # Let's perform a quick check for Difference cross-region stats if in diff mode.
        pass # Not explicitly requested for Difference Mode logic, sticking to user's "Correct/Incorrect" request.
             # If plotting Diff, visuals for "Correct vs Correct" are ambiguous. 
             # I will skip cross-region brackets in "Difference" mode to avoid confusion, 
             # OR implies we should run stats on the Difference. 
             # Given the prompt, I will assume standard plot mode is the primary target for these brackets.
    else:
        # Draw Correct vs Correct brackets
        _draw_cross_brackets(cross_sig_correct, "correct")
        # Draw Incorrect vs Incorrect brackets
        _draw_cross_brackets(cross_sig_incorrect, "incorrect")


    # -------------------- Connect subject lines -------------------- #
    for _, row in res_df.iterrows():
        if plot_as_difference:
            ys = []
            xs_line = []
            for x0, region in zip(xs_loop, region_names):
                ys.append(row[f"{region}_prev_incorrect"] - row[f"{region}_prev_correct"])
                xs_line.append(x0)
            if len(ys) > 0:
                ax.plot(xs_line, ys,
                        color="gray", alpha=0.3, linestyle="-",
                        marker="o", markerfacecolor="white",
                        markeredgecolor="black", zorder=5)
        else:
            for region in region_names:
                col_corr = f"{region}_prev_correct"
                col_incorr = f"{region}_prev_incorrect"
                if region not in region_x_map:
                    continue
                # Connect Correct -> Incorrect within region
                x_corr = region_x_map[region]["correct"]
                x_incorr = region_x_map[region]["incorrect"]
                y_corr = row[col_corr]
                y_incorr = row[col_incorr]
                if np.isnan(y_corr) or np.isnan(y_incorr):
                    continue
                ax.plot([x_corr, x_incorr], [y_corr, y_incorr],
                        color="gray", alpha=0.3, linestyle="-",
                        marker="o", markerfacecolor="white",
                        markeredgecolor="black", zorder=5)

    # ---------- axis cosmetics ----------
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_xt)

    if df_col == "calcStimulusTime":
        ax.set_ylabel("Sampling Time " + ("(Z-Scored)" if z_score_time else "(s)"))
        ax.set_title(f"{name} - Sampling Time" +
                     (" (Z-Scored)" if z_score_time else ""))
    else:
        ax.set_ylabel("Performance %")
        ax.set_title(f"{name} - Performance ")
        ax.set_ylim(bottom=45)

    if z_score_time:
        ax.axhline(0, color="black", linestyle="--", linewidth=1)

    # Adjust Y lim for brackets
    ymin, ymax = ax.get_ylim()
    if y_stack_tracker > ymax:
        ax.set_ylim(top=y_stack_tracker + (ymax - ymin) * 0.05)

    if not z_score_time and df_col == "calcStimulusTime":
        bottom, top = ax.get_ylim()
        if bottom < 0.8:
            ax.set_ylim(bottom=0.8, top=top)

    ax.spines[["left", "top", "right"]].set_visible(False)

    if save_figs:
        save_fp = Path(save_fp)
        suffix = "_z_scored" if z_score_time else "_raw"
        suffix += "_diff" if plot_as_difference else "_raw"
        suffix += "_comm_cntrl" if combine_control else "_sep_cntrl"
        save_fp = save_fp.with_name(save_fp.stem + suffix + save_fp.suffix)
        plt.savefig(save_fp, bbox_inches="tight", dpi=300)

    plt.show()

    return res_df