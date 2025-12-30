from ...common.definitions import ExpType
from .assigndvstr import assignDVStr
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from inspect import signature as sig
import multiprocessing as mp

def grpBySess(df, **grp_by_kargs):
  if len(df.Name.unique()) == 1:
    return df.groupby(["Date", "SessionNum"], **grp_by_kargs)
  else:
    return df.groupby(["Name", "Date", "SessionNum"], **grp_by_kargs)

# TODO: Add tests
def byDV(df, combine_sides=False, periods=3, separate_zero=True):
  bins = rngDV(periods=periods, combine_sides=combine_sides,
               separate_zero=separate_zero)
  # groups = []
  # DV = df.DV if not combine_sides else df.DV.abs()
  # # TODO: Use _splitByBins()
  # for (left, right)  in zip(rng, rng[1:]):
  #   if left >= 0:
  #     group_df = df[(left <= DV) & (DV < right)]
  #   else:
  #     group_df = df[(left < DV) & (df.DV <= right)]
  #   entry = pd.Interval(left=left, right=right), (left+right)/2, group_df
  #   groups.append(entry)
  # return groups
  return _splitByBins(df, bins, combine_sides=combine_sides)

def byPerf(df, combine_sides=False, periods=3, separate_zero=True,
           fit_fn_periods=10):
  bins = rngByPerf(df, periods=periods, separate_zero=separate_zero,
                   combine_sides=combine_sides, fit_fn_periods=fit_fn_periods)
  return _splitByBins(df, bins, combine_sides=combine_sides)

def byQuantile(df, combine_sides=False, periods=3, separate_zero=True):
  bins = rngByQuantile(df, periods=periods, separate_zero=separate_zero,
                       combine_sides=combine_sides)
  return _splitByBins(df, bins, combine_sides=combine_sides)


def _splitByBins(df, bins, combine_sides=False):
  groups = []
  DV = df.DV.abs() if combine_sides else df.DV
  for dv_rng, dv_df in df.groupby(pd.cut(DV, bins, include_lowest=True)):
    # We add zero to avoid having -0.0 printed
    dv_rng = pd.Interval(np.around(dv_rng.left, 2) + 0, dv_rng.right)
    entry = dv_rng, (dv_rng.left+dv_rng.right)/2, dv_df
    groups.append(entry)
  return groups


def rngDV(*, periods, combine_sides, separate_zero):
  periods += 1 # To include zero point
  rng = np.linspace(0, 1, periods) + 0.01
  if not combine_sides:
    _min = -rng[::-1]
    if not separate_zero:
      _min = _min[:-1]
      rng[0] = 0
    rng = np.concatenate([_min, rng])
  else:
    if not separate_zero:
      rng = rng[1:]
    rng = np.concatenate([[0], rng])
  return rng

def rngByPerf(df, *, periods, combine_sides, separate_zero, fit_fn_periods):
  df = df[df.ChoiceLeft.notnull()]
  stims, stim_count, stim_ratio_correct = [], [], []
  for _, _, dv_df in byDV(df, periods=fit_fn_periods,
                          combine_sides=combine_sides):
    if not len(dv_df):
      continue
    DV = dv_df.DV
    if combine_sides:
      DV = DV.abs()
    stims.append(DV.mean())
    stim_count.append(len(dv_df))
    perf_col = dv_df.ChoiceCorrect if combine_sides else dv_df.ChoiceLeft
    stim_ratio_correct.append(perf_col.mean())
  from ..psych.psych import psychFitBasic # Do the import here to avoid circular
  pars, fitFn = psychFitBasic(stims=stims, stim_count=stim_count, nfits=200,
                              stim_ratio_correct=stim_ratio_correct,
                              combine_sides=combine_sides)
  if combine_sides:
    possible_dvs = np.linspace(0,1,101)
  else:
    possible_dvs = np.linspace(-1,1,201)
  fits = fitFn(possible_dvs)
  min_perf = fits[0] if combine_sides else fits[possible_dvs == 0]
  max_perf_l, max_perf_r = fits[0], fits[-1]
  bins = [0]
  if separate_zero:
    if not combine_sides:
      bins = [-0.01] + bins
    bins += [0.01]
  cut_offs_perf = []
  def curBin(i, periods, is_neg):
    # If periods == 2, and we want to get 66.667% and 83.334%, then in ideal
    # case min_perf is 50% and max perf is 100%.
    max_perf = max_perf_r if not is_neg else max_perf_l
    cutoff_perf = min_perf + (max_perf-min_perf)*i/periods
    cutoff_idx = np.argmin(np.abs(fits-cutoff_perf))
    new_bin = possible_dvs[cutoff_idx]
    mult = 1 if not is_neg else -1
    # Do we need to handle close to zero cases here?
    if new_bin >= 1:
      new_bin = 1
      mult = -1*(periods-i)
    elif new_bin <= -1:
      new_bin = -1
      mult = periods-i
    while new_bin in bins:
      new_bin += 0.01*mult
    if not is_neg:
      bins.append(new_bin)
      cut_offs_perf.append(fits[cutoff_idx])
    else:
      bins.insert(0, new_bin)
      cut_offs_perf.insert(0, fits[cutoff_idx])

  for i in range(1, periods):
    curBin(i, periods, is_neg=False)
    if not combine_sides:
      curBin(i, periods, is_neg=True)
  bins += [1.01]
  if not combine_sides:
    bins = [-1.01] + bins
  # print("Closest dvs are: ", bins, "at perfms:", cut_offs_perf,
  #       "with min. perf:", min_perf, "and max perf L/R:", max_perf_l,
  #       max_perf_r)
  return bins

def rngByQuantile(df, *, periods, combine_sides, separate_zero):
  #_, bins = pd.qcut(df.DV.abs().rank(method='first'), periods, retbins=True)
  _, bins = pd.qcut(df.DV.abs(), periods, retbins=True, duplicates='drop')
  # print("Len:", bins)
  bins[-1] = 1.01
  if not combine_sides:
    if separate_zero:
      bins[0] = 0.01
      extra_bin = [0]
    else:
      bins[0] = 0
      extra_bin = []
    _min = -bins[::-1][:-1] # What would be a cleaner syntax?
    bins = np.concatenate([_min, extra_bin, bins])
  else:
    if not separate_zero:
      bins[0] = 0
    else:
      bin_offset_idx = 0 if bins[0] != 0 else 1
      bins = np.concatenate([[0, 0.01], bins[bin_offset_idx:]])
  # print("Returning bins:", bins)
  return bins

def splitByAssignedDifficulty(df, use_assignDVStr=True, combine_sides=False):
  if use_assignDVStr:
    if "DVstr" not in df.columns:
      df = assignDVStr(df) # TODO: Add combine_sides
    # _assertUniqEpochs(df)
    for difficulty_str, difficulty_df in df.groupby("DVstr"):
      # print("Running:", difficulty_str)
      yield difficulty_str, difficulty_df
    return
  difficulties = [df.Difficulty1, df.Difficulty2, df.Difficulty3,
                  df.Difficulty4]
  sign = df.ChoiceLeft if combine_sides else 1 # Create copy by multipling by 1
  difficulties = [col*sign for col in difficulties]
  difficulty_prcnt = (df.DV*100)
  rdk_exp = df.GUI_ExperimentType == ExpType.RDK
  difficulty_prcnt[~rdk_exp] = difficulty_prcnt[~rdk_exp]/2
  difficulty_prcnt.round().astype(int)
  if combine_sides:
    difficulty_prcnt = difficulty_prcnt.abs()
  diff1 = difficulty_prcnt == difficulties[0]
  diff2 = difficulty_prcnt == difficulties[1]
  diff3 = (difficulty_prcnt == difficulties[2]) | \
          (difficulty_prcnt == difficulties[3])
  for indexer, difficulty_str in zip([diff1, diff2, diff3],
                                     ["Easy", "Medium", "Hard"]):
    yield difficulty_str, df[indexer]

_splitDiff_prms = sig(splitByAssignedDifficulty).parameters

def splitStimulusTimeByQuantile(df, quantiles=3, col="calcStimulusTime",
    separate_by_session=True, separate_by_subject=True,
    bin_difficulties_separately=True, combine_sides=True,
    use_assignDVStr=_splitDiff_prms["use_assignDVStr"].default,
    cut_above_sec=None, uniq_sess_cols=["Name", "Date", "SessionNum", "TrialNumber"],
    ncpus=1):
    if separate_by_session:
        assert separate_by_subject

    #quantiles_df_li = []
    remainder_part_idx = 0
    org_order = df.index
    assert len(org_order.unique()) == len(df), (
            "Not unqiue idxs, call reset_index() before calling this function.")
    groupby_cols = []
    if separate_by_subject:
        groupby_cols.append("Name")
    if separate_by_session:
        groupby_cols += ["Date", "SessionNum"]

    _processDf_kwargs = dict(col=col, quantiles=quantiles,
                            cut_above_sec=cut_above_sec,
                            uniq_sess_cols=uniq_sess_cols,
                            remainder_part_idx=remainder_part_idx,
                            bin_difficulties_separately=bin_difficulties_separately,
                            combine_sides=combine_sides,
                            use_assignDVStr=use_assignDVStr,)
    if len(groupby_cols):
      quantiles_df_li = []
      if ncpus <= 1:
        for _, sub_df in tqdm(df.groupby(groupby_cols), leave=False):
          group_quantiles_df_li, grp_remained_part_idx = \
                     _processDf(sub_df, **_processDf_kwargs)
          _processDf_kwargs["remainder_part_idx"] = grp_remained_part_idx
          quantiles_df_li += group_quantiles_df_li
      else:
          with mp.Pool(ncpus) as pool:
              _processDf_kwargs["remainder_part_idx"] = remainder_part_idx
              li_args = [sub_df for _, sub_df in df.groupby(groupby_cols)]
              from functools import partial
              partial_processDf = partial(_processDf, **_processDf_kwargs)
              li_quantiles_df_li = pool.map(partial_processDf, li_args)
              for group_quantiles_df_li, grp_remained_part_idx in li_quantiles_df_li:
                  quantiles_df_li += group_quantiles_df_li

    else:
      quantiles_df_li = _processDf(df, remainder_part_idx=remainder_part_idx,
                                   **_processDf_kwargs)[0]

    quantiled_df = pd.concat(quantiles_df_li)
    assert len(quantiled_df) == len(df), f"{len(quantiled_df) = } != {len(df) = }"
    quantiled_df = quantiled_df.loc[org_order, :]
    return quantiled_df.groupby("quantile_idx")

def _processDf(df, col, quantiles, cut_above_sec, uniq_sess_cols, remainder_part_idx,
               bin_difficulties_separately, combine_sides, use_assignDVStr,):
    # _assertUniqEpochs(df)
    quantiles_df_li = []
    if cut_above_sec is not None:
        rest_df = df[df[col] > cut_above_sec]
        df = df[~df.index.isin(rest_df.index)]

    if bin_difficulties_separately:
        df = df.copy()
        grouper = splitByAssignedDifficulty(df, combine_sides=combine_sides,
                                            use_assignDVStr=use_assignDVStr)
    else:
        grouper = [("All", df)]
    grouper = list(grouper) # To avoid generator being exhausted
    # _assertUniqEpochs(pd.concat([_df for _, _df in grouper]))
    for difficulty_str, difficulty_df in grouper:
        difficulty_df = difficulty_df.copy()
        if len(difficulty_df) < quantiles:
            difficulty_df["quantile"] = np.nan
            difficulty_df["quantile_idx"] = np.nan
            quantiles_df_li.append(difficulty_df)
            continue
        # print("Difficulty str:", difficulty_str)
        # pd.cut() gave a lot of problems with duplicates and in producing equal
        # number of rows for each part, so we will do it manually.
        uniq_trials = difficulty_df[uniq_sess_cols + [col]
                                    ].value_counts()
        assert all(uniq_trials == 1), (
            display(difficulty_df[uniq_sess_cols + [col]]) or
            "There are duplicate trials in the same session. Please fix the data.")
        # display(uniq_trials)
        uniq_trials = uniq_trials.index.to_frame(index=False)
        uniq_trials = uniq_trials.sort_values(by=col)
        uniq_trials = uniq_trials.drop(columns=[col])
        each_part_size = int(len(uniq_trials) / quantiles)
        quantiles_part_size = [each_part_size]*quantiles
        remainder = len(uniq_trials) % quantiles
        # print("ShortName:", df.ShortName.unique(), difficulty_str,
        #       "- Remainder:", remainder,  "- Remainder idx: ", remainder_part_idx,)
        for _ in range(remainder):
            quantiles_part_size[remainder_part_idx] += 1
            remainder_part_idx = (remainder_part_idx + 1) % quantiles
        assert sum(quantiles_part_size) == len(uniq_trials)
        part_start = 0
        last_quantile_bin = difficulty_df[col].min() - 0.000000000000001
        # print("quantiles_part_size:", quantiles_part_size)
        final_len = 0
        for quantile_idx, quantile_part_size in enumerate(quantiles_part_size,
                                                          start=1):
            last_idx = part_start + quantile_part_size
            quantile_uniq_trials = uniq_trials[part_start:last_idx]
            # Use reset_index() to be able to re-order to the original index
            quantile_df = difficulty_df.reset_index().merge(quantile_uniq_trials,
                                                            how="inner")
            quantile_df = quantile_df.set_index("index")
            max_bin = quantile_df[col].max()
            quantile = pd.Interval(last_quantile_bin, max_bin, closed='right')
            quantile_df = quantile_df.copy()
            quantile_df["quantile"] = quantile
            quantile_df["quantile_idx"] = quantile_idx
            quantiles_df_li.append(quantile_df)
            quantile_len = len(quantile_df)
            assert quantile_part_size == quantile_len, (display(quantile_df) or
              display(quantile_uniq_trials.value_counts()) or
              f"Sizes should match {quantile_part_size = } != {quantile_len = }")
            final_len += quantile_len
            last_quantile_bin = max_bin
            part_start = last_idx
        assert final_len == len(difficulty_df), (
                                  f"{final_len = } != {len(difficulty_df) = } - "
                                  f"Did you include null {col} values?")
    if cut_above_sec is not None and len(rest_df):
        rest_df = rest_df.copy()
        if use_assignDVStr:
            rest_df = assignDVStr(rest_df)
        quantile = pd.Interval(last_quantile_bin, rest_df[col].max(),
                              closed='right')
        above_thresh_str = f"above_{cut_above_sec}s"
        rest_df["quantile"] = quantile
        rest_df["quantile_idx"] = above_thresh_str
        quantiles_df_li.append(rest_df)

    return quantiles_df_li, remainder_part_idx

def _assertUniqEpochs(df):
  uniq_df = df[["Name", "Date", "SessionNum", "TrialNumber", "epoch"]]
  assert len(uniq_df) == len(uniq_df.drop_duplicates())
