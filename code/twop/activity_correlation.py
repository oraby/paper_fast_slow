from ..pipeline import pipeline
import numpy as np
import pandas as pd

def getTrialsMeanActivity(df):
    # TODO: Combine with wide-field here
    res_dict = {"BrainRegion":[],
                "Layer":[],
                "ShortName":[],
                "Name":[],
                "TrialNumber":[],
                "long_trace_id":[],
                "ChoiceLeft":[],
                "StimulusTime":[],
                "mean_activity":[],
                "num_neurons":[],
                "ChoiceCorrect":[],
                "Direction":[],
                "Difficulty":[],
                }
    df = df[df.ChoiceCorrect.notnull()]
    for br, br_df in df.groupby("BrainRegion"):
        for sess, sess_df in br_df.groupby("ShortName"):
            for trial_num, trial_df in sess_df.groupby("TrialNumber"):
                assert len(trial_df) == 1
                trial_df = trial_df.iloc[0]
                traces_dict = pipeline.getRowTracesSets(trial_df)["neuronal"]
                s, e = \
                      trial_df["trace_start_idx"], trial_df["trace_end_idx"] + 1
                # traces_li = [trace[s:e] for trace in traces_dict.values()]
                # traces_arr = np.array(traces_li)
                # mean_activity = traces_arr.mean(axis=1)
                # mean_activity = mean_activity.mean()
                num_neurons = len(traces_dict)
                for trace_id, trace in traces_dict.items():
                    trace = trace[s:e]
                    mean_activity = trace.mean()
                    res_dict["long_trace_id"].append(f"{sess}_{trace_id}")
                    res_dict["BrainRegion"].append(br)
                    res_dict["Layer"].append(trial_df.Layer)
                    res_dict["ShortName"].append(sess)
                    res_dict["Name"].append(trial_df.Name)
                    res_dict["TrialNumber"].append(trial_num)
                    res_dict["ChoiceLeft"].append(trial_df.ChoiceLeft)
                    res_dict["StimulusTime"].append(trial_df.calcStimulusTime)
                    res_dict["mean_activity"].append(mean_activity)
                    res_dict["num_neurons"].append(num_neurons)
                    res_dict["ChoiceCorrect"].append(trial_df.ChoiceCorrect)
                    res_dict["Direction"].append(trial_df.ChoiceLeft)
                    res_dict["Difficulty"].append(trial_df.DVstr)
    return pd.DataFrame(res_dict)


