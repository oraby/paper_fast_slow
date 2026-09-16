'''Decoder accuracy panels (Figure S12G).

Backend for the two "way to plot" sections of ``plottraces3.ipynb``, drawn from
the table :mod:`.decoders` produces.

The notebook kept **two** versions of this panel, both called ``plotAccuracy``
and both run, one cell after the other:

- ``plotAccuracyAcrossEpochs`` (the "newer way") draws one figure per strategy,
  with sampling start and end side by side -- ``Choice Decoder Accuracy - Fast
  Trials.svg`` and its slow twin;
- ``plotAccuracyAcrossStrategies`` (the "older way") draws one figure per
  epoch, with fast and slow side by side -- ``Choice Decoder Accuracy -
  Sampling Start by quantile.svg`` and so on.

Both write into ``results/2P/decoders/``, which nothing else creates, so the
first save makes it.

Each point is one session; the bars are the mean across sessions with the SEM,
and the dots behind them are the individual sessions.

Extracted from the notebook unchanged, except for those names and that
``fig_save_prefix`` is a parameter.
'''
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion

def plotAccuracyAcrossEpochs(df, metric, title, split_by_difficulty,
                 min=None, max=100, save_fig=False, fig_save_prefix=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    DVStrli = ["Easy", #"Med",
               "Hard"]
    epochs = df.epoch.unique()
    if split_by_difficulty:
        main_split_li = [None]
        main_split_col = None
        sub_split_col = "DVstr"
        sub_split_li = DVStrli
        x_labels = DVStrli
    # elif "ChoiceCorrect" in df.columns:
    #     main_split_li = [1, 0]
    #     main_split_col = "ChoiceCorrect"
    else:
        main_split_li = [None]
        sub_split_col = None
    xs_ticks = []
    xs_labels = []
    x_offset = 0
    df = df.copy()
    df[metric] = df[metric]*100
    display(df[metric])

    first_iter = True
    x_sub_offset = -0.05
    for br, br_df in df.groupby("BrainRegion"):
        br = BrainRegion(br)
        br_str = str(br).split("_")[0]
        clr = BRClr[br]
        num_sess = br_df.ShortName.nunique()
        # print(f"BR: {br_str} - Num. Sess: {num_sess}")
        label = f"{br_str} ({num_sess} Sess)"
        # print(label)
        xs = []
        ys_mean = []
        ys_err = []
        sessions_ys_mean = {}
        STEP = 2
        for x_idx, epoch in enumerate(epochs, start=1):
            x_idx += (x_idx - 1) + STEP
            epoch_df = br_df[br_df.epoch == epoch]
            xs.append(x_idx)
            by_sess = epoch_df.groupby("ShortName")
            for sess, sess_df in by_sess:
                if sess not in sessions_ys_mean:
                    sessions_ys_mean[sess] = []
                sess_metric_mean = sess_df[metric].mean()
                sessions_ys_mean[sess].append(sess_metric_mean)
            ys = by_sess[metric].mean()
            ys_sem = by_sess[metric].sem()
            ys_mean.append(ys.mean())
            ys_err.append(ys_sem.mean())
        xs = np.array(xs)
        plot_xs = xs + x_offset + x_sub_offset
        for sess, sess_ys_mean in sessions_ys_mean.items():
            ax.plot(xs + x_offset + x_sub_offset, sess_ys_mean,
                    color=clr, alpha=0.15)
        ax.errorbar(plot_xs, ys_mean, ys_err, color=clr, label=label)
        # Annotate eacch point
        for x, y, y_std in zip(plot_xs, ys_mean, ys_err):
            ax.annotate(f"{y:.2f} ±{y_std:.2f}", (x, y + y_std + 2),
                        color=clr, rotation=90)
        if first_iter:
            xs_ticks = xs
            xs_labels = [f"{epoch}\nStart" for epoch in epochs]
            first_iter = False
        x_sub_offset += 0.1

    # ax.axvline([1], label="Move To Port", ls="--", color="gray")
    xs = df.quantile_idx.unique()
    ax.set_xticks(xs_ticks)
    ax.set_xticklabels(xs_labels)
    chance = 100/2
    ax.axhline(chance, label=f"Chance Level ({chance:.3g}%)",
              ls="--", color="gray")
    ax.set_xlabel("Quantile Index")
    ax.set_ylabel("Decoding Accuracy %")
    ax.set_title(title)
    # ax.legend(fontsize="small", loc=(1.04, 0))
    # if min is not None:
    ax.set_ylim(min, max)
    # Despline
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    if save_fig:
        save_title = title
        if split_by_difficulty:
            save_title = f"{save_title} by difficulty"
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        os.makedirs(f"{fig_save_prefix}/decoders", exist_ok=True)
        fig.savefig(f"{fig_save_prefix}/decoders/{save_title}.svg",
                    bbox_inches="tight", dpi=300)
    plt.show()


def runAccuracyAcrossEpochs(df, save_figs=False, fig_save_prefix=None):
    display(df.head())
    for with_difficulty in [#True,
                            False]:
        for quantile_idx in [1, 3]:
            plot_df = df[(df.quantile_idx == quantile_idx)]
            plot_df = plot_df.query("DVstr.notnull()" if with_difficulty else "DVstr.isnull()")
            plot_df = plot_df.copy()
            quantile_str = "Fast" if quantile_idx == 1 else "Slow"
            plotAccuracyAcrossEpochs(df=plot_df, metric="accuracy", title=f"Choice Decoder Accuracy - {quantile_str} Trials",
                         split_by_difficulty=with_difficulty, min=35, max=89, save_fig=save_figs,
                         fig_save_prefix=fig_save_prefix)


def plotAccuracyAcrossStrategies(df, metric, title, split_by_quantile, split_by_difficulty,
                 min=None, max=100, save_fig=False, fig_save_prefix=None):
    labels = []
    data = []
    clrs = []
    fig, ax = plt.subplots(figsize=(8, 5))
    DVStrli = ["Easy", #"Med",
               "Hard"]
    QuantileIdxLi = [1, #2,
                     3]
    if split_by_quantile:
        main_split_li = DVStrli if split_by_difficulty else [None]
        main_split_col = "DVstr" if split_by_difficulty else None
        sub_split_col = "quantile_idx" if split_by_quantile else None
        sub_split_li = QuantileIdxLi
        x_labels = ["Fast", "Slow"]
    elif split_by_difficulty:
        main_split_li = [None]
        main_split_col = None
        sub_split_col = "DVstr"
        sub_split_li = DVStrli
        x_labels = DVStrli
    # elif "ChoiceCorrect" in df.columns:
    #     main_split_li = [1, 0]
    #     main_split_col = "ChoiceCorrect"
    else:
        main_split_li = [None]
        sub_split_col = None
    xs_ticks = []
    xs_labels = []
    x_offset = 0
    df = df.copy()
    df[metric] = df[metric]*100
    display(df[metric])
    for split_val in main_split_li:
        if split_val is not None:
            dv_df = df[df[main_split_col] == split_val]
        else:
            dv_df = df
            split_val = ""
        first_iter = True
        x_sub_offset = 0
        for br, br_df in dv_df.groupby("BrainRegion"):
        # for (br, sess), br_df in dv_df.groupby(["BrainRegion","ShortName"]):
            br = BrainRegion(br)
            br_str = str(br).split("_")[0]
            clr = BRClr[br]
            num_sess = br_df.ShortName.nunique()
            # print(f"BR: {br_str} - Num. Sess: {num_sess}")
            label = f"{split_val} {br_str} ({num_sess} Sess)"
            # print(label)
            if sub_split_col == "quantile_idx":
                br_df = br_df[br_df.quantile_idx != 2]
            xs = []
            ys_mean = []
            ys_err = []
            sessions_ys_mean = {}
            STEP = 2
            for x_idx, val in enumerate(sub_split_li, start=1):
                x_idx += (x_idx - 1) + STEP
                val_df = br_df[br_df[sub_split_col] == val]
                xs.append(x_idx)
                by_sess = val_df.groupby("ShortName")
                for sess, sess_df in by_sess:
                    if sess not in sessions_ys_mean:
                        sessions_ys_mean[sess] = []
                    sess_metric_mean = sess_df[metric].mean()
                    sessions_ys_mean[sess].append(sess_metric_mean)
                ys = by_sess[metric].mean()
                ys_sem = by_sess[metric].sem()
                ys_mean.append(ys.mean())
                ys_err.append(ys_sem.mean())
            xs = np.array(xs)
            plot_xs = xs + x_offset + x_sub_offset
            for sess, sess_ys_mean in sessions_ys_mean.items():
                ax.plot(xs + x_offset + x_sub_offset, sess_ys_mean,
                        color=clr, alpha=0.15)
            ax.errorbar(plot_xs, ys_mean, ys_err, color=clr, label=label)

            # quantile_grpby = br_df.groupby("quantile_idx", as_index=False)
            # xs = np.array(list(quantile_grpby.groups.keys()))
            # ys_mean = quantile_grpby[metric].mean()
            # ys_err = quantile_grpby[metric].std()
            # plot_xs = xs + x_offset + x_sub_offset
            # ax.errorbar(plot_xs, ys_mean, ys_err,
            #             color=clr, label=label)
            # Annotate eacch point
            for x, y, y_std in zip(plot_xs, ys_mean, ys_err):
                ax.annotate(f"{y:.2f} ±{y_std:.2f}", (x, y + y_std + 2),
                            color=clr, rotation=90)
            if first_iter:
                xs_ticks.extend(xs + x_offset/2)
                xs_labels.extend([f"{label}\n{split_val}" for label in x_labels])
                first_iter = False
            x_sub_offset += 0.1
            # print("Xs:", xs)
            # print("Y:", ys_mean)
            # print("Ystd:", ys_std)
        x_offset += 2.5
    # ax.axvline([1], label="Move To Port", ls="--", color="gray")
    xs = df.quantile_idx.unique()
    ax.set_xticks(xs_ticks)
    ax.set_xticklabels(xs_labels)
    chance = 100/2
    ax.axhline(chance, label=f"Chance Level ({chance:.3g}%)",
              ls="--", color="gray")
    ax.set_xlabel("Quantile Index")
    ax.set_ylabel("Decoding Accuracy %")
    ax.set_title(title)
    # ax.legend(fontsize="small", loc=(1.04, 0))
    # if min is not None:
    ax.set_ylim(min, max)
    # Despline
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    if save_fig:
        save_title = title
        if split_by_quantile:
            save_title = f"{save_title} by quantile"
        if split_by_difficulty:
            save_title = f"{save_title} by difficulty"
        assert fig_save_prefix is not None, "Pass fig_save_prefix to save"
        os.makedirs(f"{fig_save_prefix}/decoders", exist_ok=True)
        fig.savefig(f"{fig_save_prefix}/decoders/{save_title}.svg",
                    bbox_inches="tight", dpi=300)
    plt.show()


def runAccuracyAcrossStrategies(df, save_figs=False, fig_save_prefix=None):
    display(df.head())
    for with_quantiles in [True, False]:
        for with_difficulty in [#True,
                                False]:
            if not with_quantiles and not with_difficulty:
                continue
            for epoch in ["Sampling", "Movement"]:
                plot_df = df[(df.epoch == epoch)]
                plot_df = plot_df.query("quantile_idx.notnull()" if with_quantiles else "quantile_idx.isnull()")
                plot_df = plot_df.query("DVstr.notnull()" if with_difficulty else "DVstr.isnull()")
                plot_df = plot_df.copy()
                plotAccuracyAcrossStrategies(df=plot_df, metric="accuracy", title=f"Choice Decoder Accuracy - {epoch} Start",
                             split_by_quantile=with_quantiles, split_by_difficulty=with_difficulty,
                             min=30, max=90, save_fig=save_figs,
                         fig_save_prefix=fig_save_prefix)
