# Copied from wf_evd_again.ipynb
from ..pipeline import pipeline
from ..behavior.util.assigndvstr import assignDVStr
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def _processSession(df, only_traces_ids=[], set_name="neuronal",
                    add_raw_data=False):
    dict_res = dict(
        trace_id=[],
        ShortName=[],
        Name=[],
        Date=[],
        SessionNum=[],
        BrainRegion=[],
        # Direction=[],
        Layer=[],
        MeanActivity=[],
        ChoiceCorrect=[],
        ChoiceLeft=[],
        DV=[],
        DVstr=[],
        PrevDV=[],
        PrevDVstr=[],
        Difficulty1=[],
        Difficulty3=[],
        PrevChoiceCorrect=[],
        PrevChoiceLeft=[],
        PrevOutcomeCount=[],
        calcStimulusTime=[],
        # Stimulus=[],
        SamplingType=[],
        SessPerformance=[],
        epochs_names=[],
        epochs_ranges=[],
        TrialNumber=[],
    )
    # if add_raw_data:
    #     dict_res["raw"] = []
    short_name = df.ShortName.unique()
    assert len(short_name) == 1
    short_name = short_name[0]
    df = df[df.ChoiceCorrect.notnull()]
    df = df[df.calcStimulusTime.notnull()]
    sess_performance = df.ChoiceCorrect.mean()*100
    # df_sampling = df[df.epoch == "Sampling"]
    # assert list(df.epoch.unique()) == ["Sampling"] # It's concatenated
    for _, row in df.iterrows():
        traces_dict = pipeline.getRowTracesSets(row)[set_name]
        time_ax = 0 # TODO : read this properly
        take_rng = np.arange(row.trace_start_idx, row.trace_end_idx+1)
        traces_dict = {k : v.take(take_rng, axis=time_ax)
                       for k, v in traces_dict.items()
                       if not len(only_traces_ids) or k in only_traces_ids}
        for trace_id, trace in traces_dict.items():
            trace_mean = np.nanmean(trace)
            # br, direction = trace_id.rsplit('_', 1)
            dict_res["trace_id"].append(trace_id)
            dict_res["ShortName"].append(row.ShortName)
            dict_res["Name"].append(row.Name)
            dict_res["Date"].append(row.Date)
            dict_res["SessionNum"].append(row.SessionNum)
            dict_res["BrainRegion"].append(row.BrainRegion)
            # dict_res["Direction"].append(direction)
            dict_res["Layer"].append(row.Layer)
            dict_res["MeanActivity"].append(trace_mean)
            # Extra stuff
            dict_res["calcStimulusTime"].append(row["calcStimulusTime"])
            dict_res["ChoiceCorrect"].append(row["ChoiceCorrect"])
            dict_res["ChoiceLeft"].append(row["ChoiceLeft"])
            dict_res["DV"].append(row["DV"])
            dict_res["DVstr"].append(row["DVstr"])
            dict_res["Difficulty1"].append(row["Difficulty1"])
            dict_res["Difficulty3"].append(row["Difficulty3"])
            # Same for previous rial
            dict_res["PrevChoiceCorrect"].append(row["PrevChoiceCorrect"])
            dict_res["PrevChoiceLeft"].append(row["PrevChoiceLeft"])
            dict_res["PrevOutcomeCount"].append(row["PrevOutcomeCount"])
            dict_res["PrevDV"].append(row["PrevDV"])
            dict_res["PrevDVstr"].append(row["PrevDVstr"])
            # dict_res["Stimulus"].append(row["Stimulus"])
            dict_res["SamplingType"].append(row["SamplingType"])
            dict_res["SessPerformance"].append(sess_performance)
            dict_res["epochs_names"].append(row["epochs_names"])
            dict_res["epochs_ranges"].append(row["epochs_ranges"])
            dict_res["TrialNumber"].append(row["TrialNumber"])
            if add_raw_data:
                for idx, data in enumerate(trace):
                    key = "Raw", idx
                    if key not in dict_res:
                        dict_res[key] = []
                    dict_res[key].append(data)
    if add_raw_data:
        raw_traces = {k:v for k,v in dict_res.items() if k[0] == "Raw"}
        dict_res = {(k, ''):v for k,v in dict_res.items() if k[0] != "Raw"}
        dict_res.update(raw_traces)
    # [print(k, len(v_arr)) for k,v_arr in dict_res.items()]
    res_df = pd.DataFrame(dict_res)
    # display(res_df)
    return res_df

def expandDfUngrouped(df, assign_PrevDVStr=True, add_raw_data=True):
    res_df_li = []
    if assign_PrevDVStr:
        df = assignDVStr(df.copy(), col="PrevDV", res_col="PrevDVstr")
    for sess, sess_df in tqdm(df.groupby("ShortName")):
        res_df = _processSession(sess_df, add_raw_data=add_raw_data)
        res_df_li.append(res_df)
    return pd.concat(res_df_li)

def expandDf(df, groupby_col="quantile_idx", add_raw_data=True):
    res_df_li = []
    df = assignDVStr(df.copy(), col="PrevDV", res_col="PrevDVstr")
    for groupby_val, groupby_df in df.groupby(groupby_col):
        print(f"Running for {groupby_col}={groupby_val}")
        res_df = expandDfUngrouped(groupby_df, assign_PrevDVStr=False,
                                   add_raw_data=add_raw_data)
        res_df[groupby_col] = groupby_val
        res_df_li.append(res_df)
    return pd.concat(res_df_li)
