from .pipeline import getRowTracesSets, updateRowTracesSet, DFProcessor
# Make the next import a local import, in case this module is imported as part
# of stand-alone library
# from ..behavior.util import splitdata
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d#, gaussian_filter

def keepOnlyTraces(*, dst_df, src_df, set_name=None, src_set_name=None,
                   sess_must_exist=True, remove_existing=False):
    filtered_rows = []
    # See comment at module start
    from ..behavior.util import splitdata
    for sess_info_src, sess_df_src in splitdata.grpBySess(src_df):
        assert len(sess_df_src) == 1, ("I don't know how to handle multi row "
                                       " src sessions")
        name = sess_df_src.Name.iloc[0]
        date = sess_info_src[-2]
        sess_num = sess_info_src[-1]
        sess_df_dst = dst_df.query("Date == @date and Name == @name and "
                                   "SessionNum == @sess_num")
        if sess_must_exist:
            assert len(sess_df_dst), "Dst dfs has doesn't exist"
        src_traces_set = getRowTracesSets(sess_df_src.iloc[0])
        for row_idx, row in sess_df_dst.iterrows():
            dst_new_traces_set = {}
            for cur_set_name, dst_old_traces_dict in \
                                                  getRowTracesSets(row).items():
                if set_name is not None and cur_set_name != set_name:
                    dst_new_traces_set[cur_set_name] = dst_old_traces_dict
                else:
                    if src_set_name is None:
                        cur_src_set_same = set_name
                    else:
                        cur_src_set_same = src_set_name
                    # Loop from dst to retain the order of the traces
                    src_keys_set = set(src_traces_set[cur_src_set_same].keys())
                    dst_new_traces_set[cur_set_name] = {
                        key:trace_data
                        for key, trace_data in dst_old_traces_dict.items()
                        if ((key in src_keys_set) if not remove_existing else
                            (key not in src_keys_set))}
            row = updateRowTracesSet(row, dst_new_traces_set)
            filtered_rows.append(row)
    return pd.DataFrame(filtered_rows)

def numTraces(df_row):
    return {set_name:len(traces_dict)
            for set_name, traces_dict in getRowTracesSets(df_row).items()}


class RenameTraces(DFProcessor):
    def __init__(self, preappend_col_name):
        self._preappend_col_name = preappend_col_name

    def process(self, df):
        res_rows = []
        for _, row in df.iterrows():
            new_traces_set = {}
            for set_name, traces_dict in getRowTracesSets(row).items():
                new_traces_set[set_name] = {
                            f"{key}_{row[self._preappend_col_name]}": val.copy()
                            for key, val in traces_dict.items()}
            row = row.copy()
            row = updateRowTracesSet(row, new_traces_set)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return ("Rename traces ids by preappending "
                f"{self._preappend_col_name} to their name")


class CopyTracesSortOrderProcessor(DFProcessor):
    def __init__(self, set_name, src_df, matchTrialsFn=None,
                 only_traces_ids=[],
                 match_sess_cols=["Name", "Date", "SessionNum"]):
        self._set_name = set_name
        self._src_df = src_df
        self._matchTrialsFn = matchTrialsFn
        self._only_traces_ids = only_traces_ids
        self._match_sess_cols = match_sess_cols

    def process(self, df):
        return copyTracesSortOrder(df, self._src_df, self._set_name,
                                   matchTrialsFn=self._matchTrialsFn,
                                   only_traces_ids=self._only_traces_ids,
                                   match_sess_cols=self._match_sess_cols)

    def descr(self):
        return f"Copying traces sort order from {self._set_name}"

def copyTracesSortOrder(dst_df, src_df, set_name, matchTrialsFn=None,
                        match_sess_cols=["Name", "Date", "SessionNum"],
                        only_traces_ids=[]):
    assert len(match_sess_cols)
    if len(only_traces_ids): # convert to set for a quicker lookup
        only_traces_ids = set(only_traces_ids)
    def _restrictToOnlyTracesIds(traces_sets):
        # Don't just use set as it will change the order of the traces, use dict
        org_order = traces_sets[set_name].keys()
        traces_set_keys = set(org_order)
        if len(only_traces_ids):
            traces_set_keys = traces_set_keys.intersection(only_traces_ids)
        # Reorder the traces to match the original order
        traces_set_keys = {key:None
                           for key in org_order if key in traces_set_keys}
        return traces_set_keys.keys()
    rows_with_resorted_traces = []
    # See comment at module start
    from ..behavior.util import splitdata
    for sess_info_dst, sess_df_dst in splitdata.grpBySess(dst_df):
        ex_dst_row = sess_df_dst.iloc[0]
        match_sess = \
                    ex_dst_row[match_sess_cols[0]] == src_df[match_sess_cols[0]]
        for col in match_sess_cols[1:]:
            match_sess = match_sess & (ex_dst_row[col] == src_df[col])
        assert match_sess.sum() > 0, "No matching sessions found"
        sess_df_src = src_df[match_sess]

        if matchTrialsFn is None:
            traces_keys_to_copy_sets = getRowTracesSets(sess_df_src.iloc[0])
            keys_to_copy_dst = _restrictToOnlyTracesIds(
                                                       traces_keys_to_copy_sets)
            if not len(keys_to_copy_dst):
                print("No traces to copy...")
                continue # TODO: I'm skipping the row, or should we copy the
                         # rest?
        for row_idx, row in sess_df_dst.iterrows():
            # Do initial check
            row_traces_keys_to_copy_set = getRowTracesSets(row)
            if matchTrialsFn is not None:
                keys_to_copy_dst = _restrictToOnlyTracesIds(
                                                    row_traces_keys_to_copy_set)
                # print("keys_to_copy_dst:", keys_to_copy_dst)
                if not len(keys_to_copy_dst):
                    print("2- No traces to copy...")
                    continue # TODO: Same as above, what should we do?
                row_src = matchTrialsFn(row, sess_df_src)
                # row_src = sess_df_src[
                #                    sess_df_src.TrialNumber == row.TrialNumber]
                # assert len(row_src) == 1, (
                #                       f"Expected 1 row, found {len(row_src)}")
                # row_src = row_src.iloc[0]
                row_src_traces_keys_to_copy_set = getRowTracesSets(row_src)
                keys_to_copy_src = _restrictToOnlyTracesIds(
                                                row_src_traces_keys_to_copy_set)
                # print("keys_to_copy_src:", keys_to_copy_src)
                if not len(keys_to_copy_src):
                    continue
                # Now do the actual ordered copy of keys
                keys_to_copy_dst = {k:None for k in keys_to_copy_src
                                    if k in keys_to_copy_dst}.keys()
            dst_new_traces_set = {}
            for cur_set_name, dst_old_traces_dict in \
                                            row_traces_keys_to_copy_set.items():
                if cur_set_name != set_name:
                    dst_new_traces_set[cur_set_name] = dst_old_traces_dict
                else:
                    # dst_keys_set = set(dst_old_traces_dict.keys())
                    dst_new_traces_set[cur_set_name] = {
                                                    key:dst_old_traces_dict[key]
                                                    for key in keys_to_copy_dst
                                                    #if key in dst_keys_set
                                                    }
            row = updateRowTracesSet(row, dst_new_traces_set)
            rows_with_resorted_traces.append(row)
    return pd.DataFrame(rows_with_resorted_traces)

def countTransientsRatePerMin(trace, baseline, z_score, frame_rate,
                              valid_transients_min_dur_sec):
    # Calculate the number of positive transiates that remains for at least
    # x number of seconds
    MIN_FRAMES = int(np.round(valid_transients_min_dur_sec * frame_rate))
    NUM_ZSCORE = 3
    pos_thres = baseline + NUM_ZSCORE*z_score
    # Find positive transients indices, but don't work on the trace directly,
    # use instead a moving averag trace as it will protect against sharp sudden
    # turns
    mv_avg_win = 5
    mv_avg_trace = np.convolve(trace, np.ones(mv_avg_win), "valid") / mv_avg_win
    pos_idxs = np.where(mv_avg_trace >= pos_thres)[0]
    # Use this trick: https://stackoverflow.com/a/7353335/11996983
    pos_transients = np.split(pos_idxs,
                                np.where(np.diff(pos_idxs) != 1)[0]+1)
    # Include one extra frame in the end so it would include the last trace
    if len(pos_transients[0]):  # Split can return len=1 array that is empty
        pos_transients = [list(rng) + [rng[-1] + 1] for rng in pos_transients]
    else:
        pos_transients = []
    # print("len(pos_idxs):", len(pos_idxs))
    # print("pos_idxs:", pos_idxs)
    # print("pos_transients:", pos_transients)
    accepted_pos_transients = [t for t in pos_transients
                                 if len(t) >= MIN_FRAMES]
    transients_rate = len(accepted_pos_transients)/(len(trace)/(frame_rate*60))
    return (transients_rate, pos_thres, accepted_pos_transients, pos_transients,
            mv_avg_trace, mv_avg_win, MIN_FRAMES)


def filterNanGaussianConserving(arr, sigma, axis):
    """Copied from: https://stackoverflow.com/a/61481246/11996983
    Apply a gaussian filter to an array with nans.
    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)
    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gaussian_filter1d(loss, sigma=sigma, mode='constant', cval=1,
                             axis=axis)
    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = gaussian_filter1d(gauss, sigma=sigma, mode='constant', cval=0,
                              axis=axis)
    gauss[nan_msk] = np.nan
    gauss += loss * arr
    return gauss