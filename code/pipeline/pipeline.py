import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import inspect
from typing import List
import itertools
import multiprocessing

mp_pool = None # Multiprocessing pool, will be initialized on first use


class TIME_AX_LOOKUP:
    '''This class basically translates to:
    TIME_AX_LOOKUP = {"neuronal":0, "neuronal_raw":0, "SVT_MOV":1}
    '''
    mappings = {"neuronal":0, "SVT_MOV":1}
    def __class_getitem__(cls, key):
        while any((k in key for k in ("_sem", "_raw", "_blue", "_violet"))):
            key = key.rsplit("_", 1)[0]
        time_ax = cls.mappings.get(key)
        if time_ax is not None:
            return time_ax
        else:
            raise KeyError(f"Unknown key: {key}")


def getRowTracesSetsFullTrace(row):
    return row.traces_sets

def getRowTracesSets(row):
    return row.traces_sets
    # if row.sole_owner:
    #     return row.traces_sets
    # set_name_to_id_to_traces = {}
    # start_idx = row.trace_start_idx
    # end_idx = row.trace_end_idx
    # for set_name, traces_dict in row.traces_sets.items():
    #     time_ax = TIME_AX_LOOKUP[set_name]
    #     traces_dict = {trace_id:trace_data.take(np.arange(start_idx,
    #                                                       end_idx+1),
    #                    axis=time_ax)
    #                    for trace_id, trace_data in traces_dict.items()}
    #     set_name_to_id_to_traces[set_name] = traces_dict
    # return set_name_to_id_to_traces

def updateRowTracesSet(row, set_name_to_id_to_traces):
    row.traces_sets = set_name_to_id_to_traces
    assertTraceLimits(row)
    return row

def createRowTracesSet(row, set_name_to_id_to_traces, width_resized=False):
    row.traces_sets = set_name_to_id_to_traces
    expected_width = row.trace_end_idx - row.trace_start_idx + 1
    row.trace_start_idx = 0
    ex_set_name, ex_traces_dict = next(iter(row.traces_sets.items()))
    time_ax = TIME_AX_LOOKUP[ex_set_name]
    ex_trace_width = next(iter(ex_traces_dict.values())).shape[time_ax]
    if not width_resized:
        assert expected_width == ex_trace_width, ("Trace was unexpectedly "
            f"resized from size {expected_width} to {ex_trace_width}")
    row.trace_end_idx = ex_trace_width - 1
    assertTraceLimits(row)
    row["sole_owner"] = True
    return row

def assertTraceLimits(row, debug=False):
    assert 0 <= row.trace_start_idx <= row.trace_end_idx, ("Trace start idx: "
        f"{row.trace_start_idx:,} either < 0 or > end: {row.trace_end_idx}")
    for set_name, traces_dict in row.traces_sets.items():
        ex_trace = next(iter(traces_dict.values()))
        time_ax = TIME_AX_LOOKUP[set_name]
        total_len = ex_trace.shape[time_ax]
        if debug:
            print(f"Row row.trace_end_idx: {row.trace_end_idx} < "
                  f"total_len: {total_len}")
        assert row.trace_end_idx < total_len, ("Trace end idx: "
            f"{row.trace_end_idx:,} >= Total trace length: {total_len:,}")

def createDFTraces(df, set_name_to_traces_ids_to_traces_data):
    min_idx = df.trace_start_idx.min()
    # assert min_idx > 0
    assert min_idx >= 0
    max_idx = df.trace_start_idx.max() # TODO: maybe use trace_end_idx?
    # Check that all traces are within boundaries
    for set_name, traces_dict in set_name_to_traces_ids_to_traces_data.items():
        time_ax = TIME_AX_LOOKUP[set_name]
        val_iter = iter(traces_dict.values())
        first_width = next(val_iter).shape[time_ax]
        assert max_idx <= first_width, (f"Trace max length ({max_idx}) is "
            f"bigger than actual trace ({first_width})")
        for trace in val_iter:
            assert first_width == trace.shape[time_ax], ("Not all the traces "
                                                         "have the same length")
    # TODO: Wrap the data in a session.TraceAvg() wrapper
    df["traces_sets"] = [set_name_to_traces_ids_to_traces_data]*len(df)
    df["sole_owner"] = len(df) == 1
    return df

class DFProcessor:
    '''Abstract class for all pipeline processors'''
    def process(self, df):
        raise NotImplementedError("Child class should implement this")

    def descr(self) -> str:
        raise NotImplementedError()

    def supportsParallel(self):
        return True

    def runParallel(self, run_parallel:bool=None):
        '''If run_parallel is None, use the default run-in-parallel
        configuration.'''
        if run_parallel == True:
            assert self.supportsParallel(), ("This processor does not support "
                                             "parallel processing.")
        self._run_parallel = run_parallel
        raise NotImplementedError("Still being developed...")

class Chain():
    '''The main class to be used to create a pipeline chain of DFProcessors.
    TODO: Add a usage example here'''
    def __init__(self, *processors : DFProcessor, run_parallel=False,
                 print_descr=True):
        global mp_pool
        # super().__init__(_old_chain=_old_chain)
        [self._checkIsInst(processor) for processor in processors]
        self._processors = processors
        self._run_parallel = run_parallel
        if run_parallel and mp_pool is None:
            print("Initializing multiprocessing pool")
            mp_pool = multiprocessing.Pool(max(1,
                                               multiprocessing.cpu_count() - 1))
        if False: #'ipykernel' in sys.modules: Might be blocking termination of
                  # cells with keyboard interrupts
            from IPython.display import display
            from ipywidgets import widgets
            self._html_template = "<p>CONTENT</pr>"
            # self._display_id = display("Current Progress place holder",
            #                                                   display_id=True)
            self._html_wiget = widgets.HTML()
            display(self._html_wiget)
            self._print = self._mockPrint
        else:
            self._print = print
        self._print_descr = print_descr

    def _mockPrint(self, *args):
        txt = self._html_template.replace("CONTENT", " ".join(args))
        self._html_wiget.value =    self._html_wiget.value + txt

    def process(self, data):
        if not isinstance(data, list):
            data = [data]
        # print("Data is:", data)
        for processor in self._processors:
            if self._print_descr:
                self._print("Processing:", processor.descr())
            if isinstance(processor, RecombineResults): # Needs special handling
                data = [processor.process(data)]
            else:
                data = data if len(data) ==    1 else tqdm(data)
                if self._run_parallel:
                    data = mp_pool.map(processor.process, data, chunksize=1)
                else:
                    data = [self._handleDataChunk(processor, data_pt)
                            for data_pt in data]
                data = list(itertools.chain.from_iterable(data))
        # Be more friendly to the user
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        return data

    def _handleDataChunk(self, processor, data_pt):
        data_pt = processor.process(data_pt)
        if not isinstance(data_pt, list):
            data_pt = [data_pt]
        return data_pt

    def run(self, data):
        return self.process(data)

    def descr(self):
        return f"Running chain of processors: {self._processors}"

    def _checkIsInst(self, obj):
        assert not inspect.isclass(obj), (f"Pass an instance of class {obj}, "
                                           "not the class itself")


class RecombineResults(DFProcessor):
    # TODO: Add the ability to collapse only N-levels of grouping back together
    def process(self, data_li):
        #return self._joiner.join(data_li)
        return pd.concat(data_li)

    def descr(self):
        return "Collapsing the dataframes back into one dataframe"


class By(DFProcessor):
    '''Splits pipeline into multiple branches according to condition. Each branch
    separately by the next processers until a RecombineResults() is reached'''
    def __init__(self, col_or_li_of_cols, abs=False, bins=None):
        if not isinstance(col_or_li_of_cols, list) or \
           len(col_or_li_of_cols) == 1:
            # Suppress pandas groupby warning
            self._is_single_col = True
            if not isinstance(col_or_li_of_cols, list):
                col_or_li_of_cols = [col_or_li_of_cols]
        else:
            self._is_single_col = False
        self._col_or_li_of_cols = col_or_li_of_cols
        self._abs = abs
        self._bins = bins

    def process(self, df):
        # Drop groups keys, keep only values
        cols = [df[col] for col in self._col_or_li_of_cols]
        if self._abs:
            cols = [col.abs() for col in cols]
        if self._bins is not None and len(df):
            cols = [pd.cut(col, self._bins) for col in cols]
        if self._is_single_col:
            cols = cols[0] # Suppress pandas groupby warning
        res = [sub_df for k, sub_df in df.groupby(cols) if len(sub_df)]
        return res

    def descr(self):
        return f"Split data by trial {self._col_or_li_of_cols}"

    def supportsParallel(self):
        return False


class ByAnimal(By):
    def __init__(self):
        super().__init__(["Name"])


class BySession(By):
    def __init__(self):
        super().__init__(["Name", "Date", "SessionNum"])

### Helper classes


class DoNothingPipe(DFProcessor):
    '''Designed to be incorporated in one-line if-else signal chain to enable or
    disable processors depending on flags.'''
    def __init__(self):
        pass

    def process(self, data):
        return data

    def descr(self):
        return "Passing data through..."

    def supportsParallel(self):
        return False


class ApplyFullTrace(DFProcessor):
    def __init__(self, *df_processors : List[DFProcessor]):
        self._chain = Chain(*df_processors)

    def process(self, df):
        full_trace_rows = []
        sess_rows_remaapped_rng = {}
        # TODO: Create a sess_id rather than df.info_df_idx
        for sess_id, sess_df in df.groupby(df.ShortName):
            ex_row = sess_df.iloc[0].copy()
            if sess_df.sole_owner.sum() == 0: # Trace is not whole, it's been
                                              # already "cut"
                #full_trace_width = getRowFullTrace(ex_row)
                set_name_to_id_to_traces = getRowTracesSetsFullTrace(ex_row)
                # We need to get the full trace width to set the traces_end_idx
                ex_set_name, ex_trace_dict = next(iter(
                                              set_name_to_id_to_traces.items()))
                time_ax = TIME_AX_LOOKUP[ex_set_name]
                ex_trace = next(iter(ex_trace_dict.values()))
                ex_trace_width = ex_trace.shape[time_ax]
                ex_row.trace_start_idx = 0
                ex_row.trace_end_idx = ex_trace_width - 1
                ex_row = createRowTracesSet(ex_row, set_name_to_id_to_traces)
            else:
                # Spread the cut traces to a fake full trace
                print("***Not sure about hstack for 2D arrays, check it out...")
                new_traces_sets = {}
                cur_rng_idx = 0
                for row_idx, row in sess_df.iterrows():
                    rng = np.arange(row.trace_start_idx, row.trace_end_idx + 1)
                    assert row_idx not in sess_rows_remaapped_rng
                    rng_idx_end = cur_rng_idx + len(rng)
                    sess_rows_remaapped_rng[row_idx] = cur_rng_idx, rng_idx_end
                    cur_rng_idx = rng_idx_end
                    # Now do the actual concatenation
                    set_name_to_id_to_traces = getRowTracesSetsFullTrace(row)
                    for set_name, traces_dict in \
                                               set_name_to_id_to_traces.items():
                        # print("set_name:", set_name)
                        new_traces_dict = new_traces_sets.get(set_name, {})
                        time_ax = TIME_AX_LOOKUP[set_name]
                        # empty_trace = np.array([] if time_ax == 0 else [[]])
                        for trace_id, trace_data in traces_dict.items():
                            trace_data = trace_data.take(rng, axis=time_ax)
                            if trace_id not in new_traces_dict:
                                cur_trace = trace_data
                            else:
                                cur_trace = new_traces_dict[trace_id]
                                cur_trace = np.concatenate([cur_trace,
                                                            trace_data],
                                                            axis=time_ax)
                            new_traces_dict[trace_id] = cur_trace
                        new_traces_sets[set_name] = new_traces_dict
                ex_row = createRowTracesSet(ex_row, new_traces_sets,
                                            width_resized=True)
            full_trace_rows.append(ex_row)

        full_trace_df_org = pd.DataFrame(full_trace_rows)
        full_trace_df = self._chain.run(full_trace_df_org)
        # Check added or removed columns
        full_trace_columns = set(full_trace_df.columns)
        orig_columns = set(df.columns)
        added_cols = full_trace_columns - orig_columns
        removed_cols = orig_columns - full_trace_columns
        dfs_list = []
        for row_idx, full_trace_row in full_trace_df.iterrows():
            li_set_name_to_traces_dict = getRowTracesSets(full_trace_row)
            org_row = full_trace_df_org[full_trace_df_org.ShortName ==
                                                       full_trace_row.ShortName]
            # Assert that no trace length change happened
            assert len(org_row) == 1, f"Expected 1 row, not {len(org_row)}"
            org_row = org_row.iloc[0]
            assert org_row.trace_start_idx == full_trace_row.trace_start_idx
            assert org_row.trace_end_idx == full_trace_row.trace_end_idx
            # First, handle the added and removed columns
            related_df = df[df.ShortName == full_trace_row.ShortName].copy()
            assert len(related_df)
            for col in added_cols:
                related_df[col] = [full_trace_row[col]]*len(related_df)
            if len(removed_cols):
                related_df[col].drop(list(removed_cols), axis=1)
            # Is it right to use the session ID here or we should use another
            # unique id at the point of setting a trace?
            if related_df.sole_owner.sum() == 0:
                related_df = createDFTraces(related_df,
                                            li_set_name_to_traces_dict)
            else:
                # Now re-assign the fake full trace back to the cut traces
                related_row_li = []
                for related_row_idx, related_row in sess_df.iterrows():
                    set_name_to_id_to_traces = getRowTracesSetsFullTrace(
                                                             related_row).copy()
                    for set_name, traces_dict in \
                                               set_name_to_id_to_traces.items():
                        full_trace_dict = li_set_name_to_traces_dict[set_name]
                        time_ax = TIME_AX_LOOKUP[set_name]
                        related_row_rng = sess_rows_remaapped_rng[
                                                                related_row_idx]
                        rng = np.arange(related_row_rng[0],
                                                        related_row_rng[1])
                        new_traces_dict = {}
                        for trace_id in traces_dict.keys():
                            full_trace_data = full_trace_dict[trace_id]
                            trace_data = full_trace_data.take(rng, axis=time_ax)
                            new_traces_dict[trace_id] = trace_data
                        set_name_to_id_to_traces[set_name] = new_traces_dict
                    related_row = createRowTracesSet(related_row,
                                                     set_name_to_id_to_traces)
                    related_row_li.append(related_row)
                assert full_trace_data.shape[-1] == related_row_rng[-1], (
                           f"{full_trace_data.shape = } != {related_row_rng= }")
                related_df = pd.DataFrame(related_row_li)
            dfs_list.append(related_df)
        df = pd.concat(dfs_list)
        return df

    def descr(self):
        return (f"Applying on full trace")


class ApplyOnSubdata(DFProcessor):
    def __init__(self, col, col_val_or_lambda,
                 *df_processors : List[DFProcessor], only_full_df : bool=True,
                 use_not_eq=False):
        self._col = col
        self._col_val = col_val_or_lambda
        self._only_full_df = only_full_df
        self._use_not_eq = use_not_eq
        self._chain = Chain(*df_processors)

    def process(self, df):
        if callable(self._col_val):
                cmp_res = df[self._col].apply(self._col_val)
        else:
            cmp_res = df[self._col] == self._col_val
        if self._use_not_eq:
            cmp_res = ~cmp_res
        sub_df = df[cmp_res]
        if not len(sub_df) or (self._only_full_df and len(sub_df) != len(df)):
            return df
        rest_df = df[~cmp_res]
        sub_df = self._chain.run(sub_df)
        if len(rest_df):
            if isinstance(sub_df, list):
                sub_df = sub_df + [rest_df]
            else:
                sub_df = pd.concat([sub_df, rest_df])
        return sub_df

    def descr(self):
        # TODO: Do a depth level indentation for better readability
        return (
            f"Applying processors on "
            f"{'only whole groups' if self._only_full_df else 'any subgroups'} "
            f"where column '{self._col}' "
            f"{'!=' if self._use_not_eq else '=='} {self._col_val}.")


class CombineTraces(DFProcessor):
    def __init__(self, trace_id_rename_src_col, rename_prefix=""):
        self._trace_id_rename_src_col = trace_id_rename_src_col
        self._rename_prefix = rename_prefix

    def process(self, df):
        traces_lens = df.trace_end_idx - df.trace_start_idx
        assert len(traces_lens.unique()) == 1, ("Traces are not of the same "
                                        f"length: {traces_lens.value_counts()}")
        set_name_to_id_to_traces_all = {}
        for _, row in df.iterrows():
            for set_name, traces_dict in getRowTracesSets(row).items():
                cur_traces_data = set_name_to_id_to_traces_all.get(set_name, {})
                axis = TIME_AX_LOOKUP[set_name]
                for trace_id, trace_data in traces_dict.items():
                    data = trace_data.take(np.arange(row.trace_start_idx,
                                                     row.trace_end_idx+1),
                                           axis=axis)
                    trace_id = (f"{trace_id}_{self._rename_prefix}"
                                f"{row[self._trace_id_rename_src_col]}")
                    try:
                        trace_id = int(trace_id)
                    except ValueError:
                        pass
                    assert trace_id not in cur_traces_data, (
                                          f"Trace id {trace_id} already exists")
                    cur_traces_data[trace_id] = data
                set_name_to_id_to_traces_all[set_name] = cur_traces_data
        ex_row = df.iloc[0].copy()
        ex_row = createRowTracesSet(ex_row, set_name_to_id_to_traces_all)
        return pd.DataFrame([ex_row])

    def descr(self):
        return "Combining traces of input data together"

class LoopTraces(DFProcessor):
    def __init__(self, *processors_li, set_name="neuronal", only_traces_ids=[],
                 groupTracesFn=None, print_descr=False, show_progress=True):
        self._chain = Chain(*processors_li, print_descr=print_descr)
        self._only_traces_ids = only_traces_ids
        self._set_name = set_name
        self._groupTracesFn = groupTracesFn
        self._show_progress = show_progress

    def process(self, df):
        traces_ids_sets_dict = {}
        for _row_idx, row in df.iterrows():
            set_name_to_id_to_traces = getRowTracesSets(row)
            # print("set_name_to_id_to_traces:",
            #                          list(set_name_to_id_to_traces[0].keys()))
            # print("self._set_name:", self._set_name)
            for trace_id_org in set_name_to_id_to_traces[self._set_name].keys():
                if self._groupTracesFn is not None:
                    trace_id = self._groupTracesFn(trace_id_org)
                else:
                    trace_id = trace_id_org
                # print("trace_id:", trace_id, "trace_id_org:", trace_id_org)
                if not len(self._only_traces_ids) or \
                   trace_id in self._only_traces_ids:
                    trace_set = traces_ids_sets_dict.get(trace_id, set())
                    trace_set.add(trace_id_org)
                    traces_ids_sets_dict[trace_id] = trace_set
        # print("traces_ids_sets_dict:", list(traces_ids_sets_dict.items())[0])
        looper = traces_ids_sets_dict.items()
        if self._show_progress:
            looper = tqdm(looper, desc="Iterating traces")
        for _trace_id, trace_ids_sets in looper:
            # if len(self._only_traces_ids) and \
            #        cur_trace_id not in self._only_traces_ids:
            #     continue
            res_rows = []
            for row_idx, row in df.iterrows():
                trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx+1)
                set_name_to_id_to_traces = {}
                skip_row = False
                for set_name, traces_dict in getRowTracesSets(row).items():
                    if set_name == self._set_name:
                        # TODO: Add axis here in take() method
                        new_traces_dict = {k:v.take(trc_rng)
                                           for k,v in traces_dict.items()
                                           if k in trace_ids_sets}
                        skip_row = len(new_traces_dict) == 0
                        set_name_to_id_to_traces[self._set_name] =\
                                                                 new_traces_dict
                    else:
                        set_name_to_id_to_traces[set_name] = traces_dict
                if not skip_row:
                    row = createRowTracesSet(row.copy(),
                                             set_name_to_id_to_traces)
                    res_rows.append(row)
            # Now create a new dataframe and run our data on
            df_mod = pd.DataFrame(res_rows)
            self._chain.run(df_mod)
        return df

    def descr(self):
        return "Loop through the traces one by one..."
