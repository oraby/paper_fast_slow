from .util import splitdata
from ..common.clr import Difficulty as DifficultyClr
from ..common.definitions import ExpType
from ..figcode.psychometric import plotPsych
from enum import IntFlag, auto
import pandas as pd
import matplotlib.pyplot as plt
from enum import Flag, auto

class Plots(IntFlag):
    Chronometry = auto()
    ChronoPsych = auto()

AllPlots = sum([plot for plot in Plots])
NoPlots = 0

class GroupBy(Flag):
    Difficulty = auto()
    EqualSize = auto()
    Performance = auto()

def chronometry(df, plot_all_animals, plot_single_animals, **kargs):
    is_many_animals = df.Name.nunique() > 1
    if plot_all_animals and is_many_animals:
        _processAnimal("All Animals", df, is_many_animals=True, **kargs)
    if plot_single_animals:
        for animal_name, animal_df in df.groupby(df.Name):
            _processAnimal(animal_name, animal_df, is_many_animals=False,
                           **kargs)

def _processAnimal(animal_name, animal_df, *, is_many_animals, plots,
                   min_easiest_perf, exp_type, min_num_sampling_pts,
                   min_sampling_pts, grpby, save_figs, save_prefix):
    animal_df = splitdata.grpBySess(animal_df).filter(
                                 lambda ssn:(ssn.GUI_MinSampleType == 4).any())
    animal_df = splitdata.grpBySess(animal_df).filter(_fltrSsns,
                                            min_easiest_perf=min_easiest_perf,
                                            exp_type=exp_type)
    animal_df = animal_df[animal_df.Difficulty3.notnull()]
    if len(animal_df) < 200:
        if len(animal_df) > 10:
            print(f"Skipping {animal_name} with just {len(animal_df)} trials")
        return
    # print(animal_df.GUI_MinSampleType.unique())
    print(f"Subject: {animal_name}")
    chron_df = animal_df.groupby(animal_df.MinSample).filter(
                                                       lambda grp:len(grp) > 30)
    if plots & Plots.Chronometry:
        cur_min_sampling_pts = sorted(chron_df.MinSample.unique())
        if len(cur_min_sampling_pts) < min_num_sampling_pts:
            print(f"Skipping {animal_name} with {len(cur_min_sampling_pts)} "
                   "sampling points")
            return
        fig, axes = plt.subplots(1,1)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        print("Min sampling points:", cur_min_sampling_pts)
        _chronPlot(chron_df, axes, min_sampling_pts=min_sampling_pts,
                   grpby=grpby, is_many_animals=is_many_animals)
        if save_figs:
            fig.savefig(f"{save_prefix}/{animal_name}_chrono.svg")
        plt.show()

    if plots & Plots.ChronoPsych:
        plotPsych(chron_df, title="Chronometry", by_subject=True,
                  by_session=True, combine_sides=False, save_fp=save_prefix,
                  save_figs=save_figs, nfits=10 if not save_figs else None,
                  plot_points=False)
        if save_figs:
            fig.savefig(f"{save_prefix}/{animal_name}_psych_chrono.svg")
        plt.show()

def _chronPlot(df, axes, min_sampling_pts, grpby, is_many_animals):
    df = df[df.ChoiceCorrect.notnull()]
    df = df.copy()
    df['DVabs'] = df.DV.abs()
    colors = [DifficultyClr.Hard, DifficultyClr.Med, DifficultyClr.Easy]
    all_x_points = set()
    if not len(df.DVabs.unique()):
        return
    fn = splitdata.byDV if grpby == GroupBy.Difficulty else \
         splitdata.byPerf if grpby == GroupBy.Performance else \
         splitdata.byQuantile
    kargs = dict(periods=3, separate_zero=False, combine_sides=True)
    for idx, (dv_rng, _, dv_df) in reversed(list(enumerate(fn(df, **kargs)))):
        x_data = []
        y_data = []
        y_data_sem = []
        num_points = 0
        #for min_sampling, ms_data in DV_data.groupby(DV_data.MinSample):
        for min_sampling in min_sampling_pts:
            ms_data = dv_df[dv_df.MinSample.between(min_sampling - 0.001,
                                                    min_sampling + 0.001)]
            if not len(ms_data):
                continue
            x_data.append(min_sampling)
            cur_num_points = len(ms_data.ChoiceCorrect)
            if is_many_animals:
                ms_data = ms_data.groupby(ms_data.Name)
                cur_perf = 100*ms_data.ChoiceCorrect.mean().mean()
                cur_perf_sem = 100*ms_data.ChoiceCorrect.mean().sem()
                num_subjects = len(ms_data)
                # Plus minus sign:
                axes.annotate(f"{cur_perf:.2f}% ±{cur_perf_sem:.2f}%\n"
                              f"n={num_subjects}",  (min_sampling, cur_perf),
                              textcoords="offset points", xytext=(0,5),
                              ha='center', fontsize='small')
            else:
                cur_perf = 100*ms_data.ChoiceCorrect.mean()
                cur_perf_sem = 100*ms_data.ChoiceCorrect.sem()
            num_points += cur_num_points
            y_data.append(cur_perf)
            y_data_sem.append(cur_perf_sem)
        all_x_points.update(x_data)
        color=colors[idx]
        # First DV is 1.01, which would give 101% cohr.
        left, right = int(dv_rng.left*100), min(int(dv_rng.right*100), 100)
        axes.errorbar(x_data, y_data, yerr=y_data_sem, color=color,
                    label=f"{left}%-{right}% Coherence ({num_points:,} trials)")
    if is_many_animals:
        subjects_str = ", ".join(df.Name.unique())
        subjects_str = (f"All Animals ({subjects_str})\n"
                        f"Err bars: SEM across subjects")
    else:
        subjects_str = df.Name.iloc[0]
    axes.set_title(f"Chronometry - {subjects_str}")
    axes.set_xlabel("Sampling Duration (s)")
    axes.set_ylabel("Performance %")
    axes.set_xticks(sorted(list(all_x_points)))
    axes.legend(loc="upper left", prop={'size': 'x-small'})

def _fltrSsns(sess_df, exp_type, min_easiest_perf):
    sess_df = sess_df[sess_df.GUI_ExperimentType == exp_type]
    df_choice = sess_df[sess_df.ChoiceCorrect.notnull()]
    df_choice = df_choice[df_choice.Difficulty3.notnull()]
    if len(df_choice) and len(df_choice) < 50:
        print(f"Insufficient trials ({len(df_choice)}) for "
                    f"{sess_df.Date.iloc[0]}-Sess{sess_df.SessionNum.iloc[0]}")
        return False
    # Avoid floating-point comparision errors with int
    trial_difficulty_col = (df_choice.DV.abs() * 100).round()
    if exp_type != ExpType.RDK:
        trial_difficulty_col = (trial_difficulty_col/2)+50
    easiest_diff = df_choice[trial_difficulty_col == df_choice.Difficulty1]
    if len(easiest_diff):
        easiest_perf = \
          len(easiest_diff[easiest_diff.ChoiceCorrect == 1]) / len(easiest_diff)
    else:
        easiest_perf = -1
    easiest_perf *= 100
    if len(easiest_diff) and easiest_perf < min_easiest_perf:
        print(f"Bad performance ({easiest_perf:.2f}%) for "
              f"{sess_df.Date.iloc[0]}-Sess{sess_df.SessionNum.iloc[0]} - "
              f"Len: {len(df_choice)}")
    # print("easiest_perf >= min_easiest_perf:",
    #       easiest_perf >= min_easiest_perf)
    return easiest_perf >= min_easiest_perf
