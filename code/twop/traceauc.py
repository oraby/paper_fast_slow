'''Where a trace rises, and how much area those rises cover.

Shared machinery behind the rt/activity panels of ``plottraces3.ipynb``: the
active-neuron counts (:mod:`.activeneuroncount`, Figures S11A/S11B) and the
per-neuron correlations (Figures 4K, S10A-C) both reduce a trace the same way.

``getPosDeflections`` finds each stretch where the trace does not fall and
returns it as ``(the sample before it, its top)``. A flat run counts as rising,
so a trace that opens flat starts its first stretch at 0.

``getTraceAUC`` then walks **segments**, each running from one stretch's start
to the next one's start (or the end of the trace), and for each:

- takes ``threshold = min + TRACE_THRESH * (max - min)`` of that segment, so a
  small bump on a quiet neuron is measured like a large one on a loud neuron;
- masks the samples at or above it **plus everything leading up to the
  segment's peak**, which is the rise itself;
- and, after the segment, jumps to the first stretch starting beyond the
  masked region -- so stretches inside one already-measured segment are not
  counted twice.

The masked samples of every segment are then offset to start at zero,
concatenated, and integrated **once**, so the returned area is over the rises
together rather than per rise.

**When nothing is masked anywhere the area is 0 and all four bookkeeping lists
come back empty**, including entries already filled in for earlier segments.

Moved verbatim out of the notebook; only the module docstring, the leading
underscores and the ``TRACE_THRESH`` constant's home have changed.
'''
from __future__ import annotations

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: A rise is integrated from where it passes this share of its own height.
TRACE_THRESH = 0.8


def getPosDeflections(trace):
    trace_diff = np.diff(trace)
    trace_pos_idxs = np.where(trace_diff >= 0)[0] + 1
    if not len(trace_pos_idxs):
        return np.empty((0, 2))
    trace_pos_idxs_diff = np.diff(trace_pos_idxs)
    # Thanks https://stackoverflow.com/a/7353335/11996983
    trace_pos_non_concat_idxs = np.where(trace_pos_idxs_diff != 1)[0] + 1
    consect_idxs_grps_splits = np.split(trace_pos_idxs, trace_pos_non_concat_idxs)
    # Take the min before the first positive deflect and the last (max)
    # positive deflect
    pos_deflects_idxs = np.array([(grp[0]-1, grp[-1])
                                   for grp in consect_idxs_grps_splits])
    return pos_deflects_idxs


def getTraceAUC(trace, pos_deflects_idxs):
    if TRACE_THRESH is None:
        assert False, "Probably a bug?"
        integrated_auc = trapezoid(trace_to_integrate)
        traces_thresh_li = []
        traces_rng_li = []
        traces_min_max_idxs_li = []
        masks_li = []

    assert pos_deflects_idxs is not None
    traces_to_integrate_li = []
    traces_min_max_idxs_li = []
    traces_rng_li = []
    masks_li = []
    traces_thresh_li = []

    starts_idxs = pos_deflects_idxs[:,0]
    starts_li_idx = 0
    while starts_li_idx < len(starts_idxs):
        start_idx = starts_idxs[starts_li_idx]
        end_idx = starts_idxs[starts_li_idx + 1] if starts_li_idx + 1 < len(starts_idxs) else \
                  len(trace)
        sub_trace = trace[start_idx:end_idx]
        cur_min_idx, cur_max_idx = np.argmin(sub_trace), np.argmax(sub_trace)
        cur_min, cur_max = sub_trace[cur_min_idx], sub_trace[cur_max_idx]
        trace_thresh = (cur_max-cur_min)*TRACE_THRESH + cur_min
        sub_integrate_mask = sub_trace >= trace_thresh
        # But everything leading to the max should be included
        sub_integrate_mask[:cur_max_idx] = True
        # Bookkeeping
        traces_rng_li.append((start_idx, end_idx))
        traces_min_max_idxs_li.append((start_idx + cur_min_idx,
                                       start_idx + cur_max_idx))
        traces_thresh_li.append(trace_thresh)
        masks_li.append(sub_integrate_mask)
        if not sub_integrate_mask.any():
            starts_li_idx += 1
            continue
        sub_trace = sub_trace[sub_integrate_mask]
        # OFfset to zero
        sub_trace -= sub_trace.min()
        traces_to_integrate_li.append(sub_trace)
        # Find last occurance of the mask
        mask_end_idx = np.where(sub_integrate_mask)[0][-1]
        # Look for the first start after the end
        starts_li_idx = np.argmax(starts_idxs > start_idx + mask_end_idx)
        # print("Mask end:", mask_end_idx, "Starts idx:", starts_li_idx)
        if starts_li_idx == 0: # No more starts
            break
    if len(traces_to_integrate_li):
        trace_to_integrate = np.concatenate(traces_to_integrate_li)
        integrated_auc = trapezoid(trace_to_integrate)
    else:
        integrated_auc = 0
        traces_thresh_li = []
        traces_rng_li = []
        traces_min_max_idxs_li = []
        masks_li = []
    return (integrated_auc, traces_thresh_li, traces_rng_li,
            traces_min_max_idxs_li, masks_li)
