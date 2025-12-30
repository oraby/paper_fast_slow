from .pipeline import (DFProcessor, assertTraceLimits, createRowTracesSet,
                                             getRowTracesSets, TIME_AX_LOOKUP)
try:
    from IPython.display import display
except ModuleNotFoundError:
    pass
import numpy as np
import pandas as pd
from typing import List

class AlignTraceAroundEpoch(DFProcessor):
    def __init__(self, epoch_name_li : List[str], time_before_sec,
                 time_after_sec, limit_trial_start=False, limit_trial_end=False,
                 limit_to_epoch_start=[], limit_to_epoch_end=[],
                 limit_epoch_cols=["Name", "Date", "SessionNum", "TrialNumber"],
                 use_epoch_name=None):
        self._epoch_name_li = epoch_name_li
        self._time_before_sec = time_before_sec
        self._time_after_sec = time_after_sec
        self._limit_trial_start = limit_trial_start
        if isinstance(limit_to_epoch_start, str):
            limit_to_epoch_start = [limit_to_epoch_start]
        self._limit_to_epoch_start = limit_to_epoch_start
        if isinstance(limit_to_epoch_end, str):
            limit_to_epoch_end = [limit_to_epoch_end]
        self._limit_to_epoch_end = limit_to_epoch_end
        self._limit_epoch_cols = limit_epoch_cols
        self._use_epoch_name = use_epoch_name
        # TODO: Limit to an epoch of the next trial, e.g start of sampling
        self._limit_trial_end = limit_trial_end

    def process(self, data):
        new_rows = []
        epochs_df = data[data.epoch.isin(self._epoch_name_li)]
        for row_idx, row in epochs_df.iterrows():
            acq_rate = row.acq_sampling_rate
            idx_subtract = int(self._time_before_sec*acq_rate)
            # print(f"{self._time_after_sec *acq_rate = }")
            # Somehow, we need to take -1 out, or we end up with one more entry
            idx_add = max(int(self._time_after_sec * acq_rate) - 1, 0)
            if self._limit_trial_start or self._limit_trial_end:
                # Should we use row.trace_trial_offset_start_idx or the available
                # dataframe even if it's missing some entries?
                # trial_df = data[data.TrialNumber == row.TrialNumber]
                raise NotImplemented(
                                     "Limit just to the next or previous epoch")
                idx_subtract = min(row.trace_trial_offset_start_idx,
                                   idx_subtract)
            start_idx = max(0, row.trace_start_idx - idx_subtract)
            # print(f"start idx: {start_idx} - "
            #       f"{row.trace_start_idx - idx_subtract = }")
            if self._limit_to_epoch_start:
                start_epoch_row = self._getRelevantRow(row,
                                                     self._limit_to_epoch_start,
                                                     data)
                # print("start_epoch_row.trace_start_idx:",
                #       start_epoch_row.trace_start_idx,
                #       f" - start idx: {start_idx} - Use start idx?:",
                #       max(start_epoch_row.trace_start_idx, start_idx) ==
                #           start_idx)
                start_idx = max(start_epoch_row.trace_start_idx, start_idx)
            # TODO: Limit here to the end of the maximum trace
            # end_idx = min(max_trace_idx, row.trace_start_idx + idx_add)
            org_end_idx = row.trace_start_idx + idx_add
            end_idx = org_end_idx
            if self._limit_trial_end: # Limit to start of next trial
                if row.TrialNumber < data.TrialNumber.max():
                    next_trial = data[data.TrialNumber == row.TrialNumber + 1]
                    # Can we know whether the user passed a filtered df or that
                    # was the end of the acquisition?
                    end_idx = min(next_trial.trace_start_idx.iloc[0], end_idx)
                    # Should we just skip incomplete traces?
                # else: It's already at max_trace_idx
            if self._limit_to_epoch_end:
                end_epoch_row = self._getRelevantRow(row,
                                                     self._limit_to_epoch_end,
                                                     data)
                end_idx = min(end_epoch_row.trace_end_idx, end_idx)

            epoch_name = (row.epoch if self._use_epoch_name is None else\
                          self._use_epoch_name)
            # print("row.trace_start_idx:", row.trace_start_idx,
            #       "start idx:", start_idx)
            # print("row.trace_end_idx:", row.trace_end_idx,
            #       "end idx:", end_idx)
            if self._time_before_sec > 0:
                pre_start_row = row.copy()
                pre_start_row.epoch = \
                                   f"-{self._time_before_sec:.3g}s {epoch_name}"
                pre_start_row.trace_start_idx = start_idx
                pre_start_row.trace_end_idx = row.trace_start_idx - 1
                pre_start_row["org_start_idx"] = start_idx
                pre_start_row["org_mid_idx"] = row.trace_start_idx
                pre_start_row["org_end_idx"] = end_idx
                # print("final start idx:", start_idx,
                #       f"{pre_start_row.trace_end_idx = }")
                # row.trace_start_idx = start_idx # Leave this as it was
                assertTraceLimits(pre_start_row)
                new_rows.append(pre_start_row)
            row.trace_end_idx = end_idx
            row.epoch = epoch_name
            row["org_start_idx"] = start_idx
            row["org_mid_idx"] = row.trace_start_idx
            row["org_end_idx"] = end_idx
            assertTraceLimits(row)
            # TODO: Fix all the relative and absolute seconds
            new_rows.append(row)
        res = pd.DataFrame(new_rows)
        return res

    def _getRelevantRow(self, row, other_epoch_name_li, data):
        if row.epoch in other_epoch_name_li:
            # No need to find the other row
            end_epoch_row = row
        else:
            sub_df = data[data.epoch.isin(other_epoch_name_li)]
            all_trues = np.ones(len(sub_df)).astype(bool)
            for col in self._limit_epoch_cols:
                all_trues_temp = all_trues & (sub_df[col] == row[col])
                assert any(all_trues_temp), (
                        f"No data after filtering for {col} - "
                        f"Data before: {sub_df[all_trues].ShortName.unique()}")
                all_trues = all_trues_temp
            sub_df = sub_df[all_trues]
            assert len(sub_df) == 1, (f"Found {len(sub_df)} rows for {row} with "
                                     f"epochs: {sub_df.epoch.unique()} "
                                     f"while looking for {other_epoch_name_li} "
                                     f"in data of len={len(data)} with epochs: "
                                     f"{data.epoch.unique()}. Limited end?: "
                                     f"{self._limit_to_epoch_end}")
            end_epoch_row = sub_df.iloc[0]
        return end_epoch_row

    def descr(self):
        return (f"Aligning traces to {self._time_before_sec} sec(s) before "
                f"epoch(s): {self._epoch_name_li} " +
                (f" (or trial start if it occurred first)"
                 if self._limit_trial_start else "") +
                f" and {self._time_after_sec} sec(s) afterwards" +
                (f" (or trial end if it occurred first)"
                 if self._limit_trial_end else ""))

class ConcatEpochs(DFProcessor):
    def __init__(self, assume_continuos: bool, ignore_repeated=False,
                 ignore_existing_concat=False):
        super().__init__()
        if not assume_continuos:
            raise NotImplementedError("Didn't implement this other scenario "
                                      "yet")
        self._assume_continuos = assume_continuos
        self._ignore_repeated = ignore_repeated
        self._ignore_existing_concat = ignore_existing_concat

    def process(self, df):
        if not self._ignore_repeated:
            df = df.sort_values(by="state_id")
            epochs_name_li = df.epoch.unique()
            if len(epochs_name_li) != len(df):
                print(df.TrialNumber.unique())
                display(df[["Name", "Date", "epoch", "TrialNumber",
                            "trace_start_idx"]])
            assert len(epochs_name_li) == len(df), (
                          "Some epochs are repeated: " + str(df.epoch.tolist()))
        else:
            epochs_name_li = []
        epochs_ranges_li = []
        last_end_idx = -1
        set_concat_traces = {}
        if "epochs_names" in df.columns and not self._ignore_existing_concat:
            epoch_col = "epochs_names"
            col_is_list = True
        else:
            epoch_col = "epoch"
            col_is_list = False
        for row_idx, row in df.iterrows():
            trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx+1)
            epoch_name = row[epoch_col]
            if not self._ignore_repeated:
                assert not col_is_list, "This is unhandled yet"
                epoch_df = df[df[epoch_col] == epoch_name]
                assert len(epoch_df) == 1, ("Found more than one epoch. "
                              "Did you not split multiple sessions/conditions?")
            # print("indices:", indices)
            for set_name, traces_dict in getRowTracesSets(row).items():
                concat_traces = set_concat_traces.get(set_name, {})
                for trace_id, trace_data in traces_dict.items():
                    time_ax = TIME_AX_LOOKUP[set_name]
                    trace_data = trace_data.take(trc_rng, axis=time_ax)
                    if trace_id not in concat_traces:
                        concat_traces[trace_id] = trace_data
                    else: # We can use get but we have to reshape the empty
                          # array which is a pain in the ass.
                        concat_trace = np.concatenate(
                                        [concat_traces[trace_id], trace_data],
                                        axis=time_ax)
                        concat_traces[trace_id] = concat_trace
                set_concat_traces[set_name] = concat_traces
            if self._assume_continuos:
                if not col_is_list:
                    start_idx = last_end_idx + 1
                    last_end_idx = start_idx + (row.trace_end_idx -
                                                row.trace_start_idx)
                    epochs_ranges_li.append((start_idx, last_end_idx))
                else: # This was tested with only self._ignore_repeated = True
                    for idx_tup in row.epochs_ranges:
                        start_idx = last_end_idx + 1
                        last_end_idx = start_idx + (idx_tup[1] - idx_tup[0])
                        epochs_ranges_li.append((start_idx, last_end_idx))

                if self._ignore_repeated or col_is_list:
                    if col_is_list:
                        epochs_name_li += epoch_name
                    else:
                        epochs_name_li.append(epoch_name)
            else:
                raise NotImplementedError()
                # Should we just use trace_trial_offset_start_idx,
                #                    trace_trial_offset_last_idx
        ex_row = df.iloc[0].copy()
        ex_row["epochs_names"] = list(epochs_name_li)
        ex_row["epochs_ranges"] = epochs_ranges_li
        ex_row["trace_start_idx"] = epochs_ranges_li[0][0]
        ex_row["trace_end_idx"] = epochs_ranges_li[-1][1]
        # print("ex_trace_data.shape[time_ax]:", time_ax, concat_trace.shape)
        # print("epochs_ranges_li:", epochs_ranges_li)
        # print("idxs:", ex_row.trace_start_idx, ex_row.trace_end_idx)
        if "concat_trace" in locals():
            assert concat_trace.shape[time_ax] - 1 == epochs_ranges_li[-1][1]
        # for set_name, concat_traces_dict in set_concat_traces.items():
        #     updateRowTracesSet(ex_row, set_name, concat_traces_dict,
        #                                            modified_width=True)
        ex_row = createRowTracesSet(ex_row, set_concat_traces)
        return pd.DataFrame([ex_row])

    def descr(self):
        return "Concatenated epochs of the same trial-number together"


class CutLongTraces(DFProcessor):
    def __init__(self, maxTraceLenFn, cut_from_beginning : bool = False):
        self._maxTraceLenFn = maxTraceLenFn
        self._print_count = 100
        self._cut_from_beginning = cut_from_beginning
        self._self_checks = 0

    def process(self, df):
        cur_trace_lens = df.trace_end_idx - df.trace_start_idx + 1
        max_allowed_len = self._maxTraceLenFn(cur_trace_lens)
        if self._print_count < 10:
            print("Max allowed len:", max_allowed_len)
        unchanced_df = df[cur_trace_lens <= max_allowed_len]
        changed_df = df[cur_trace_lens > max_allowed_len]
        res_rows = []
        epochs_ranges_exist = "epochs_ranges" in df.columns
        for row_idx, row in changed_df.iterrows():
            new_traces_set = {}
            if not self._cut_from_beginning:
                new_range = np.arange(row.trace_start_idx,
                                      row.trace_start_idx + max_allowed_len + 1)
            else:
                new_range = np.arange(row.trace_end_idx - max_allowed_len,
                                      row.trace_end_idx + 1)
            for set_name, traces_dict in getRowTracesSets(row).items():
                time_ax = TIME_AX_LOOKUP[set_name]
                traces_dict = {trace_id: trace_data.take(new_range,
                                                         axis=time_ax)
                               for trace_id, trace_data in traces_dict.items()}
                new_traces_set[set_name] = traces_dict
                # if self._print_count < 10 and set_name == "neuronal":
                #     print("New trace shape:", next(iter(
                #                                  traces_dict.values())).shape)
                #     self._print_count += 1
            row = row.copy()
            row.trace_start_idx = 0
            row.trace_end_idx = max_allowed_len
            if epochs_ranges_exist:
                row = self._limitEpochsRanges(row, max_allowed_len)
            row = createRowTracesSet(row, new_traces_set)
            res_rows.append(row)
            if self._print_count < 10:
                # print("New trace len:", row.trace_end_idx -
                #                         row.trace_start_idx + 1)
                self._print_count += 1
        df = pd.concat([unchanced_df, pd.DataFrame(res_rows)])
        df = df.sort_values(by=["Name", "Date", "SessionNum", "TrialNumber"])
        # Do one last check
        cur_trace_lens = df.trace_end_idx - df.trace_start_idx
        assert all(cur_trace_lens <= max_allowed_len), (
                                      f"Bad cutting at: {df.ShortName.iloc[0]}")
        return df

    def _limitEpochsRanges(self, row, max_allowed_len):
        new_ranges = []
        # remaining_count = max_allowed_len
        rngs_zeroed = np.array(row.epochs_ranges)
        start_offset = rngs_zeroed[0][0]
        rngs_zeroed -= start_offset
        cur_idx = len(rngs_zeroed) - 1
        stop_at = max_allowed_len if not self._cut_from_beginning else \
                  rngs_zeroed[-1][1] - max_allowed_len
        cur_rng = rngs_zeroed[-1]
        while stop_at < cur_rng[0]:
            cur_idx -= 1
            cur_rng = rngs_zeroed[cur_idx]
        if not self._cut_from_beginning:
            rngs_zeroed = rngs_zeroed[:cur_idx+1]
            rngs_zeroed[cur_idx, 1] = stop_at
            row.epochs_names = row.epochs_names[:cur_idx+1]
        else:
            rngs_zeroed = rngs_zeroed[cur_idx:]
            rngs_zeroed[0, 0] = stop_at
            rngs_zeroed -= stop_at # I think we should zero it again as the
                                   # calling function has started the trace
                                   # from 0
            row.epochs_names = row.epochs_names[cur_idx:]
        # Again parent function has reset the trace len to 0, so don't add the
        # offset
        row.epochs_ranges = [(rng[0], rng[1]) for rng in rngs_zeroed]
        if self._self_checks < 100:
            self._self_checks += 1
            assert len(row.epochs_names) == len(row.epochs_ranges)
            last_idx = -1
            for rng in row.epochs_ranges:
                assert rng[0] == last_idx + 1, (
                                         f"Non-consecutive {row.epochs_ranges}")
                assert rng[1] >= rng[0]
                last_idx = rng[1]
            assert last_idx == row.trace_end_idx, (
                                           f"{last_idx} != {row.trace_end_idx}")
        return row

    def descr(self):
        return "Limit the length of traces to the length defined by user"


class ExtendShortTraces(DFProcessor):
    def __init__(self, minTraceLenFn, extend_at_beginning=False,
                 fill_value=np.nan, offset_epochs_ranges_starting_epoch=None):
        self._minTraceLenFn = minTraceLenFn
        self._extend_at_beginning = extend_at_beginning
        self._fill_value = fill_value
        if offset_epochs_ranges_starting_epoch is not None:
            assert extend_at_beginning, ("Can only offset epochs_ranges if "
                                         "extending from beginning")
        self._offset_epochs_ranges_starting_epoch = \
                                             offset_epochs_ranges_starting_epoch
        self._print_count = 100

    def process(self, df):
        cur_trace_lens = df.trace_end_idx - df.trace_start_idx + 1
        min_len_to_keep = self._minTraceLenFn(cur_trace_lens)
        if self._print_count < 10:
            print("Max allowed len:", min_len_to_keep)
        unchanced_df = df[cur_trace_lens >= min_len_to_keep]
        changed_df = df[cur_trace_lens < min_len_to_keep]
        epochs_ranges_exist = (self._offset_epochs_ranges_starting_epoch and
                               "epochs_ranges" in df.columns)
        res_rows = []
        for row_idx, row in changed_df.iterrows():
            new_traces_set = {}
            trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
            refill_amount = min_len_to_keep - len(trc_rng)
            assert refill_amount > 0, "Bad calculations"
            for set_name, traces_dict in getRowTracesSets(row).items():
                time_ax = TIME_AX_LOOKUP[set_name]
                ex_trace = next(iter(traces_dict.values()))
                ex_shape = list(ex_trace.shape)
                ex_shape[time_ax] = refill_amount
                refill_arr = np.full(ex_shape, self._fill_value)
                new_traces_dict = {}
                for trace_id, trace_data in traces_dict.items():
                    trace_data = trace_data.take(trc_rng, axis=time_ax)
                    if self._extend_at_beginning:
                        trace_data = np.concatenate([refill_arr, trace_data],
                                                    axis=time_ax)
                    else:
                        trace_data = np.concatenate([trace_data, refill_arr],
                                                    axis=time_ax)
                    new_traces_dict[trace_id] = trace_data
                new_traces_set[set_name] = new_traces_dict
                # if self._print_count < 10 and set_name == "neuronal":
                #     print(f"ex_trace shape: {ex_trace.shape} - "
                #           f"ex_shape: {ex_shape} - "
                #           f"new shape: {trace_data.shape}")
                #     self._print_count += 1
            row = row.copy()
            row.trace_start_idx = 0
            row.trace_end_idx = min_len_to_keep - 1
            if epochs_ranges_exist:
                epoch_idx = row.epochs_names.index(
                                      self._offset_epochs_ranges_starting_epoch)
                epochs_ranges = [list(tup) for tup in row.epochs_ranges]
                len_epochs = len(epochs_ranges)
                if epoch_idx != 0:
                    epochs_ranges[epoch_idx-1][1] += refill_amount
                for idx in np.arange(epoch_idx, len_epochs):
                    li = list(epochs_ranges[idx])
                    li[0] += refill_amount
                    li[1] += refill_amount
                    epochs_ranges[idx] = li
                row.epochs_ranges = [tuple(li) for li in epochs_ranges]
            row = createRowTracesSet(row, new_traces_set)
            res_rows.append(row)
            if self._print_count < 10:
                # print("New trace len:", row.trace_end_idx -
                #       row.trace_start_idx + 1)
                self._print_count += 1
        df = pd.concat([unchanced_df, pd.DataFrame(res_rows)])
        df = df.sort_values(by=["Name", "Date", "SessionNum", "TrialNumber"])
        return df

    def descr(self):
        return "Extend the length of traces to the length defined by user"
