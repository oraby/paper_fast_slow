from .pipeline import (DFProcessor, createRowTracesSet, getRowTracesSets,
                       TIME_AX_LOOKUP)
import numpy as np
import pandas as pd
from skimage import transform
from tqdm.auto import tqdm

class NormalizeTime(DFProcessor):
    # TODO: Add 'how' parameter, to normalize based on median or average
    def __init__(self, show_progress=True):
        self._show_progress = show_progress

    def process(self, df):
        trace_width = df.trace_end_idx - df.trace_start_idx + 1
        median_width = int(np.round(np.median(trace_width)))
        res_rows = []
        _iter = df.iterrows()
        if self._show_progress:
            _iter = tqdm(_iter, total=len(df), desc=f"Resizing traces",
                         leave=False)

        for row_id, row in _iter:
            trc_rng = np.arange(row.trace_start_idx, row.trace_end_idx+1)
            if len(trc_rng) == 0:
                continue # Should we fail instead?
            set_name_to_id_to_traces = {}
            for set_name, traces_dict in getRowTracesSets(row).items():
                normalized_traces = {}
                org_shape = next(iter(traces_dict.values())).shape
                time_ax = TIME_AX_LOOKUP[set_name]
                new_shape = list(org_shape) # It comes as a tuple
                new_shape[time_ax] = median_width
                # print("row.epoch", row_id, set_name, row.epoch,
                #       row.TrialNumber, "new_shape:", new_shape,
                #       "Old shape:", trc_rng[-1] - trc_rng[0], trc_rng[0],
                #       trc_rng[-1])
                for trace_id, trace_data in traces_dict.items():
                    trace_data = trace_data.take(trc_rng, axis=time_ax)
                    normalized_traces[trace_id] = transform.resize(trace_data,
                                                                   new_shape)
                set_name_to_id_to_traces[set_name] = normalized_traces
            row = createRowTracesSet(row, set_name_to_id_to_traces,
                                     width_resized=True)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self):
        return (f"Normalize traces time to have a common median length")
