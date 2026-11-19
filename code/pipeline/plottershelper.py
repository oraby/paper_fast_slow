from .pipeline import DFProcessor, getRowTracesSets
import numpy as np
import pandas as pd

class StoreTracesMinMax(DFProcessor):
    def process(self, df):
        ACROSS_ROWS = True
        if ACROSS_ROWS:
            set_name_trace_id_min_max = {}
            for row_idx, row in df.iterrows():
                for set_name, traces_dict in getRowTracesSets(row).items():
                    trace_id_min_max = set_name_trace_id_min_max.get(set_name,
                                                                     {})
                    for trace_id, trace in traces_dict.items():
                        old_min, old_max = trace_id_min_max.get(trace_id,
                                                              (np.inf, -np.inf))
                        cur_min, cur_max = np.min(trace), np.max(trace)
                        trace_id_min_max[trace_id] = (min(old_min, cur_min),
                                                      max(old_max, cur_max))
                    set_name_trace_id_min_max[set_name] = trace_id_min_max
        else:
            assert False, "Not implemented yet"
        res_rows = []
        for row_idx, row in df.iterrows():
            row = row.copy()
            row["TracesMinMax"] = set_name_trace_id_min_max
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return "For each row in our data, store each trace min and max value"
