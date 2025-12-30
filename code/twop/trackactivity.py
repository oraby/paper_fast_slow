from .genrundata import makeLongTraceID
from ..common.definitions import BrainRegion
from ..common.clr import BrainRegion as BRClr
from ..pipeline import pipeline
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrackTracesActivity(pipeline.DFProcessor):
    def __init__(self, id_cols=["ShortName"], set_name="neuronal"):
        self._id_cols = id_cols
        self._set_name = set_name
        self._traces_mapping = {}

    def process(self, df):
        traces_mapping = {}
        time_ax = pipeline.TIME_AX_LOOKUP[self._set_name]
        for _, row in df.iterrows():
            traces_dict = pipeline.getRowTracesSets(row)[self._set_name]
            trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
            mapping_id = row[self._id_cols]
            mapping_id = tuple(mapping_id)
            cur_mapping = traces_mapping.get(mapping_id, {})
            if len(cur_mapping) == 0:
                cur_mapping = {trace_id: [] for trace_id in traces_dict.keys()}
            for trace_id, trace_data in traces_dict.items():
                trace_data = np.take(trace_data, trc_rng, axis=time_ax)
                trace_mean = np.nanmean(trace_data, axis=time_ax)
                cur_mapping[trace_id].append(trace_mean)
            traces_mapping[mapping_id] = cur_mapping

        for group_id, group_data in traces_mapping.items():
            cur_group_mapping = self._traces_mapping.get(group_id, {})
            for trace_id, trace_li in group_data.items():
                if trace_id not in cur_group_mapping:
                    cur_group_mapping[trace_id] = trace_li
                else:
                    cur_group_mapping[trace_id].extend(trace_li)
            self._traces_mapping[group_id] = cur_group_mapping
        return df

    def descr(self):
        return "Traclomg neurons that crossed a threshold"

def trackNeurons(df_src):
    activity_tracker = TrackTracesActivity()
    chain = pipeline.Chain(
            pipeline.BySession(),
                activity_tracker
    )
    chain.run(df_src)
    return activity_tracker._traces_mapping

def trackNeuronsActivityPerQuantile(df_src):
    # df_src = _preprocessDF(df_src)
    activity_tracker_dict = {}
    for quantile_idx, quantile_df in df_src.groupby("quantile_idx"):
        activty_tracker = trackNeurons(quantile_df)
        activity_tracker_dict[quantile_idx] = activty_tracker
    return activity_tracker_dict

def getActiveNeurons(tracking_dict, df_src, part=1, total_parts=1):
    active_neurons_dict = {#"epoch": [],
                           #"part": [],
                           "ShortName": [], # "session"
                           "brain_region": [],
                           "trace_id":[],
                           "trace_mean": [],
                           #"total_parts": []
    }
    assert part > 0
    assert part <= total_parts
    epoch = df_src.epoch.unique()
    assert len(epoch) == 1
    epoch = epoch[0]
    for group_id, group_data in tracking_dict.items():
        assert len(group_id) == 1
        sess_name = group_id[0]
        sess_df = df_src[df_src.ShortName == sess_name]
        sess_br = sess_df.BrainRegion.unique()[0]
        # sess_br = sess_br.value
        for trace_id, trace_li in group_data.items():
            trace_li = np.array(trace_li)
            trace_mean = np.nanmean(trace_li)
            if True: #trace_mean > threshold:
                active_neurons_dict["ShortName"].append(sess_name)
                active_neurons_dict["brain_region"].append(sess_br)
                active_neurons_dict["trace_id"].append(trace_id)
                active_neurons_dict["trace_mean"].append(trace_mean)
    active_neurons_df = pd.DataFrame(active_neurons_dict)
    active_neurons_df["epoch"] = epoch
    active_neurons_df["part"] = part
    active_neurons_df["total_parts"] = total_parts
    active_neurons_df = makeLongTraceID(active_neurons_df)
    return active_neurons_df

def getActiveNeuronsPerQuantile(tracking_quantile_dict, df_src, part=1,
                                total_parts=1):
    # df = _preprocessDF(df_src)
    df_li = []
    for quantile_idx, tracking_dict in tracking_quantile_dict.items():
        quantile_active_neurons_df = getActiveNeurons(tracking_dict, df_src,
                                                      part=part,
                                                      total_parts=total_parts)
        quantile_active_neurons_df["quantile_idx"] = quantile_idx
        df_li.append(quantile_active_neurons_df)
    active_neurons_df = pd.concat(df_li)
    return active_neurons_df

def visualizeQuantileActivity(df_neurons_activity, threshold, save_fig,
                              fig_save_prefix=None):
    if save_fig:
        assert fig_save_prefix is not None
    num_quantiles = df_neurons_activity.quantile_idx.nunique()
    num_br = df_neurons_activity.brain_region.nunique()
    def _setAxes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
    fig, axs = plt.subplots(nrows=num_br, ncols=num_quantiles,
                            figsize=(num_quantiles*10, num_br*10))
    if num_quantiles == 1:
        axs = np.expand_dims(axs, axis=1)
    fig2, ax_bar = plt.subplots(figsize=(8, 6))
    fig3, ax_violin = plt.subplots(figsize=(12, 6))
    fig4, axs_cdf = plt.subplots(1, 2, figsize=(20, 10))
    epoch = df_neurons_activity.epoch.unique()
    assert len(epoch) == 1
    epoch = epoch[0]
    offset_x = 0.5
    X_OFFSET_SCALE = 2
    bins = np.linspace(-1, 2, 50)
    print("Num bins:", len(bins))
    VERT_VIOLIN = False
    for row_axs, (br, br_df) in zip(axs,
                                   df_neurons_activity.groupby("brain_region")):
        br_clr = BRClr[BrainRegion(br)]
        br = str(BrainRegion(br)).split("_")[0]
        # Use enumerate to make the code work with special quantile labels
        for quantile_idx, (quantile_label, quantile_df) in enumerate(
                                        br_df.groupby("quantile_idx"), start=1):
            prcnt_above_thresh = ((quantile_df.trace_mean > threshold).sum() /
                                                               len(quantile_df))
            prcnt_above_thresh *= 100
            ax_bar.bar(quantile_idx*X_OFFSET_SCALE + offset_x,
                       prcnt_above_thresh, width=0.5, color=br_clr)
            ax = row_axs[quantile_idx-1]
            ax.set_title(f"{br} - Q={quantile_label} - {epoch} "
                         f"{prcnt_above_thresh:.2g}% above {threshold}T")
            ax.hist(quantile_df.trace_mean, bins=bins)
            ax.axvline(threshold, color="black", ls="--")
            ax.set_xlim(-1, 2)
            ax.set_xlabel("Mean Z-Score activity")
            _setAxes(ax)
            if br == "M2":
                clr = "cyan" if quantile_idx == 1 else (
                      "blue" if quantile_idx == 2 else
                      "indigo")
                violin_x_offset = 0
                ax_cdf = axs_cdf[0]
            else:
                clr = "lightcoral" if quantile_idx == 1 else (
                      "red" if quantile_idx == 2 else
                      "maroon")
                violin_x_offset = 4
                ax_cdf = axs_cdf[1]
            mean_traces = quantile_df.trace_mean#[quantile_df.trace_mean > 1]
            vs = ax_violin.violinplot(mean_traces, quantiles=[0.33, 0.66],
                                      # showmeans=True,
                                      points=len(mean_traces),
                                      positions=[violin_x_offset +
                                                 quantile_idx],
                                      vert=VERT_VIOLIN)
            for param in ("bodies", "cbars", "cmins", "cmaxes", "cmeans"):
                if param not in vs:
                    continue
                part = vs[param]
                if param != "bodies":
                    part = [part]
                for pc in part:
                    pc.set_facecolor(clr)
                    pc.set_edgecolor(clr)
                    pc.set_alpha(0.5)
            # ax_kde.hist(quantile_df.trace_mean, bins=bins, histtype="step",
            #             color=clr, label=f"{br} - Q={quantile_label}")
            import seaborn as sns
            sns.kdeplot(quantile_df.trace_mean, ax=ax_cdf, color=clr,
                                    label=f"{br} - Q={quantile_label}")
            # sorted_traces_means = np.sort(quantile_df.trace_mean)
            # xs = (100*np.arange(1, len(sorted_traces_means)+1) /
            #                                          len(sorted_traces_means))
            # ax_cdf.plot(xs, sorted_traces_means,
            #             label=f"{br} - Q={quantile_label}", color=clr)
        offset_x -= 0.5
    ax_bar.set_title(f"Neurons mean responses more than Z-Score "
                     f"Threshold={threshold} - epoch {epoch}")
    ax_bar.set_xticks(np.arange(1, num_quantiles+1)*X_OFFSET_SCALE)
    ax_bar.set_xticklabels([f"Q={q}" for q in np.arange(1, num_quantiles+1)])
    ax_bar.set_xlim(X_OFFSET_SCALE/2,
                    num_quantiles*X_OFFSET_SCALE + 1 + X_OFFSET_SCALE/2)
    _setAxes(ax_bar)
    ax_bar.set_ylim(0, 30)
    title = f"Neurons mean responses - epoch {epoch}"
    ax_violin.set_title(title)
    violin_xs = np.arange(1, 2*num_quantiles+2)
    violin_xs = violin_xs[violin_xs% (num_quantiles+1) != 0]
    if VERT_VIOLIN:
        ax_violin.set_xlabel("Quantile index")
        ax_violin.set_ylabel("Mean Z-Score activity")
        ax_violin.set_xticks(violin_xs)
        ax_violin.set_xticklabels([f"Q={q%(num_quantiles+1)}"
                                   for q in violin_xs])
    else:
        ax_violin.set_ylabel("Quantile index")
        ax_violin.set_xlabel("Mean Z-Score activity")
        ax_violin.set_yticks(violin_xs)
        ax_violin.set_yticklabels([f"Q={q%(num_quantiles+1)}"
                                   for q in violin_xs])
    _setAxes(ax_violin)
    [(_setAxes(ax) or True) and ax.set_title(title + " CDF") for ax in axs_cdf]
    if save_fig:
        base = f"{fig_save_prefix}/RT_Stats/"
        mid = f"quantiles_activity_{epoch}"
        _dir = 'vert' if VERT_VIOLIN else 'horz'
        fig.savefig(f"{base}hist_{mid}_threshold_{threshold}.pdf")
        fig2.savefig(f"{base}bar_{mid}_threshold_{threshold}.pdf")
        fig3.savefig(f"{base}violin_{mid}_{_dir}.pdf")
        fig4.savefig(f"{base}cdf_{mid}_{_dir}.pdf")
    plt.show()

def generateActiveNeuronsDF(neurons_activity_df, threshold):
    active_neurons_df = neurons_activity_df[
                                     neurons_activity_df.trace_mean > threshold]
    active_neurons_df = active_neurons_df.copy()
    # active_neurons_df["trace_id"] = active_neurons_df.session + \
    #                                 active_neurons_df.trace_id.astype(str)
    display(active_neurons_df)
    return active_neurons_df
