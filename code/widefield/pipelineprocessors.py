"""Widefield trace processors: bilateral-region unification and whole-brain activity.

The pixel-wise dF/F maps of Figures 3B and S5C are not reproduced by this
repository. They are rebuilt from each session's raw widefield SVD components,
which are too large to share here; contact the authors if you are interested
in them.
"""
from ..pipeline.pipeline import (DFProcessor, getRowTracesSets,
                                 createRowTracesSet)
import numpy as np
import pandas as pd
from typing import List

Cols = List[DFProcessor]


class UnifyBilateralRegionsTraces(DFProcessor):
    def __init__(self, info_df_dict):
        '''Combine the left and right hemispheres brain-regions into a single
        brain-region.
        When computing the unified value, the weight of each brain-region within
        a hemisphere is a a function of its number of valid pixels,
        i.e the part of brain-region within our FOV.
        '''
        self._info_df_dict = info_df_dict

    def process(self, df):
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_traces_dict = getRowTracesSets(row)
            traces_dict = set_name_to_traces_dict["neuronal"]
            traces_keys = set([key.rsplit("_left", 1)[0].rsplit("_right", 1)[0]
                               for key in traces_dict.keys()])
            new_traces_dict = {}
            info_df = self._info_df_dict[row.info_df_id]
            for key_stripped in traces_keys:
                left_name = f"{key_stripped}_left"
                right_name = f"{key_stripped}_right"
                left_trace = traces_dict.get(left_name)
                right_trace = traces_dict.get(right_name)
                if left_trace is None:
                    assert right_trace is not None
                    new_traces_dict[key_stripped] = right_trace.copy()
                elif right_trace is None:
                    new_traces_dict[key_stripped] = left_trace.copy()
                else:
                    left_row = info_df[info_df.area_name == left_name]
                    right_row = info_df[info_df.area_name == right_name]
                    assert len(left_row) == 1, (
                        "Expected one left row for area "
                        f"{left_name} but found: {len(left_row)} - "
                        "are you passing updated info dict?")
                    assert len(right_row) == 1, (
                        "Expected one right row for area "
                        f"{right_name} but found: {len(right_row)} - "
                        "are you passing updated info dict?")
                    left_weight = len(left_row.iloc[0].area_valid_pix)
                    right_weight = len(right_row.iloc[0].area_valid_pix)
                    areas_avg = (
                           (left_trace*left_weight + right_trace*right_weight) /
                           (left_weight+right_weight))
                    new_traces_dict[key_stripped] = areas_avg
            # Copy into a new dict rather than changing the row's own
            set_name_to_traces_dict = set_name_to_traces_dict.copy()
            set_name_to_traces_dict["neuronal"] = new_traces_dict
            row = createRowTracesSet(row, set_name_to_traces_dict)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self) -> str:
        return (f"Unify left and right brain areas traces into one area")

class WholeBrainActivity(DFProcessor):
    def __init__(self, info_df_dict, relative_to_avg=False):
        self._info_df_dict = info_df_dict
        if relative_to_avg:
            raise NotImplementedError()

    def process(self, df):
        res_rows = []
        for row_id, row in df.iterrows():
            set_name_to_traces_dict = getRowTracesSets(row)
            traces_dict = set_name_to_traces_dict["neuronal"]

            new_traces_dict = traces_dict.copy()
            for _dir in ["_left", "_right"]:
                weighted_traces = []
                total_weights = 0
                info_df = self._info_df_dict[row.info_df_id]
                total_valid_pix = []
                total_invalid_pix = []
                for trace_id, trace_data in traces_dict.items():
                    area_info = info_df[info_df.area_name == trace_id]
                    assert len(area_info) == 1, (f"Can't find ID {trace_id}, "
                        "make sure you run WholeBrainActivity activity "
                        "processor before UnifyBilateralRegionsTraces")
                    if not trace_id.endswith(_dir):
                        continue
                    valid_pix = area_info.iloc[0].area_valid_pix
                    trace_weight = len(valid_pix)
                    weighted_traces.append(trace_data*trace_weight)
                    total_weights += trace_weight
                    # Now store the pixels so we can draw them later if needed
                    total_valid_pix.append(valid_pix)
                    total_invalid_pix.append(area_info.iloc[0].area_invalid_pix)
                new_traces_dict[f"Crtx{_dir}"] = \
                         np.sum(np.array(weighted_traces), axis=0)/total_weights
                assert new_traces_dict[f"Crtx{_dir}"].shape == trace_data.shape
                # Update the df entry if it exists, otherwise add it
                fields = {"name": "Cortex",
                          "acronym": "Crtx",
                          "area_name": f"Crtx{_dir}",
                          "label": 999,
                          "area_number": -999 if _dir == "_right" else 999,
                          "area_valid_pix": [np.concatenate(total_valid_pix)],
                          "area_invalid_pix": [
                                             np.concatenate(total_invalid_pix)],
                          "allen_rgb": [[0, 0, 0]]
                }
                crtx_row = info_df[info_df.area_name == fields["area_name"]]
                if len(crtx_row):
                    indx = crtx_row.index
                else: # We can use pandas's 'Setting With Enlargement'
                    indx = max(info_df.index) + 1
                for key, val in fields.items():
                    if isinstance(val, list):
                        val = [val] # Wrapped in a list for the enlargement assignment
                    info_df.loc[indx, key] = val
            set_name_to_traces_dict = set_name_to_traces_dict.copy()
            set_name_to_traces_dict["neuronal"] = new_traces_dict
            row = createRowTracesSet(row, set_name_to_traces_dict)
            res_rows.append(row)
        return pd.DataFrame(res_rows)

    def descr(self) -> str:
        return "Computing average cortex activity"
