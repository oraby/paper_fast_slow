
def applyFunOnEpoch(fn, df, pval, fig_save_prefix, active_traces_df=None,
                     neurouns_total_count_df=None):
    for epoch, epoch_df in    df.groupby(df.epoch):
        epoch_total_counts_df = None if neurouns_total_count_df is None else \
                                getEpochTotalCountDf(epoch_df, epoch,
                                                     neurouns_total_count_df)
        if active_traces_df is not None:
            epoch_active_traces_df = active_traces_df[
                                               active_traces_df.epoch == epoch]
            applyFunOnEpochPart(fn, epoch_df=epoch_df, epoch=epoch,
                                epoch_active_traces_df=epoch_active_traces_df,
                                pval=pval, fig_save_prefix=fig_save_prefix,
                                epoch_total_counts_df=epoch_total_counts_df)
        else:
            epoch_name = epoch
            fn(epoch_df, epoch=epoch_name, pval=pval,
               fig_save_prefix=fig_save_prefix, active_traces_df=None,
               total_neurons_count_df=epoch_total_counts_df)

def getEpochTotalCountDf(epoch_df, epoch, neurouns_total_count_df):
    epoch_sess = epoch_df.ShortName.unique()
    epoch_counts_df = neurouns_total_count_df[
                           (neurouns_total_count_df.epoch == epoch) &
                           (neurouns_total_count_df.ShortName.isin(epoch_sess))]
    assert epoch_df.ShortName.nunique() == len(epoch_counts_df), (
                              print(sorted(epoch_df.ShortName.unique())) or
                              print(sorted(epoch_counts_df.ShortName.unique())))
    return epoch_counts_df

def applyFunOnEpochPart(fn, epoch_df, epoch, epoch_active_traces_df,
                        pval, fig_save_prefix, epoch_total_counts_df=None):
    assert len(epoch_active_traces_df)
    for part, part_traces_df in epoch_active_traces_df.groupby("part"):
        total_parts = part_traces_df.total_parts.iloc[0]
        if total_parts > 1:
            # epoch_name += f" {part}/{total_parts}"
            if part == 1:
                part = "Early"
            elif part == total_parts:
                part = "Late"
            elif part == 2 and total_parts == 3:
                part = "Mid"
            else:
                part = f"{part}/{total_parts}"
            epoch_name = f"{part} {epoch}"
        else:
            epoch_name = epoch
        fn(epoch_df, epoch=epoch_name, pval=pval,
            fig_save_prefix=fig_save_prefix,
            active_traces_df=part_traces_df,
            total_neurons_count_df=epoch_total_counts_df)
