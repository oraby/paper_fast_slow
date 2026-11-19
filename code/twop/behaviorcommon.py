from ..behavior.util.splitdata import grpBySess
from ..common.definitions import BrainRegion
from ..pipeline import pipeline, tracesrestructure
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path


def assignDVStr(df, col="DV", res_col="DVstr"):
  raise ImportError("Import caiman.behavior.util.assigndvstr.assignDVstr() "
                    "instead.")

def filterIncompleteTrials(df):
  all_sess = []
  for _, grp_both in grpBySess(df):
    local_grps = []
    for choice_correct in [True, False]:
      grp = grp_both[grp_both.ChoiceCorrect == choice_correct]
      if not len(grp):
        continue
      grp = grp.copy()
      if "epoch" in grp.columns:
        # A groupby().apply() per column: slow, but fast enough here
        trial_nums = grp.groupby("TrialNumber").apply(
                                   lambda trial_df:trial_df.TrialNumber.iloc[0])
        trace_len = grp.groupby("TrialNumber").apply(
     lambda trial_df: (trial_df.trace_end_idx - trial_df.trace_start_idx).sum())
        valid_trial = trace_len == trace_len.median()
        for trial_num, valid_trial in zip(trial_nums, valid_trial):
          grp.loc[grp.TrialNumber == trial_num, "valid_trial"] = valid_trial
      else:
        trace_len = grp.trace_end_idx - grp.trace_start_idx
        grp["valid_trial"] = trace_len == trace_len.median()
      local_grps.append(grp)
    sess_df = pd.concat(local_grps)
    sess_df = sess_df.sort_values(["TrialNumber", "trace_start_idx"])
    all_sess.append(sess_df)
  df = pd.concat(all_sess)
  old_len = len(df)
  df = df[df.valid_trial == True]
  print(f"Filtered out: {old_len-len(df):,}/{old_len:,} Mismatched Trials")
  return df


def concatenateEpochs(df, by_session=True, by_correctness=False, by_prev_correct=False,
                      by_direction=False, by_prev_direction=False,
                      by_difficulty=False, by_trial=False,
                      ignore_existing_concat=False):
  chain = pipeline.Chain(
      pipeline.BySession() if by_session else pipeline.DoNothingPipe(),
        pipeline.By("ChoiceCorrect") if by_correctness else pipeline.DoNothingPipe(),
        pipeline.By("ChoiceCorrect") if by_prev_correct else pipeline.DoNothingPipe(),
        pipeline.By("ChoiceLeft") if by_direction else pipeline.DoNothingPipe(),
        pipeline.By("ChoiceLeft") if by_prev_direction else pipeline.DoNothingPipe(),
        pipeline.By("DVstr") if by_difficulty else pipeline.DoNothingPipe(),
        pipeline.By("TrialNumber") if by_trial else pipeline.DoNothingPipe(),
          tracesrestructure.ConcatEpochs(assume_continuos=True,
                                         ignore_existing_concat=ignore_existing_concat),
      pipeline.RecombineResults() if by_session else pipeline.DoNothingPipe(),
  )
  df = chain.run(df)
  return df


def savePrefixTmplt(fig_short_name, plot_id, parent_dir, df, is_unified,
                    fig_save_prefix, is_widefield, fig_postfix="",
                    normalization_str=None, ext=None, parent_subdir=None,
                    save_context=None, alternate_animal_folder=None,
                    alternative_plot_id_col=None, append_plot_id_col=None,
                    # This values here will be replaced with the corresponding
                    # variables.
                    save_fp=("{fig_save_prefix}/{session_type_path}/"
                             "{session_dir}/{parent_subdir}/"
                             "{(fig_short_name + \"_\") if len(fig_short_name) else \"\"}"
                             "{plot_id}{fig_postfix}{norm}.{ext}")
                            ):
  if ext is None:
    ext = "jpeg"
  assert is_widefield != None
  short_name = df.ShortName
  if len(short_name) and short_name.iloc[0] != str(parent_dir)[0]:
    parent_dir = str(Path(parent_dir))
    short_name = short_name.apply(Path).astype(str)
  animal_df = df[short_name == parent_dir]

  if is_widefield:
    stimulis_kw = (
       "stimulus_type" if "stimulus_type" in animal_df.columns else "Stimulus")
    sampling_kw = (
    "sampling_type" if "sampling_type" in animal_df.columns else "SamplingType")
    stim = animal_df[stimulis_kw].iloc[0]
    sample = animal_df[sampling_kw].iloc[0]
    session_type_path = f"{stim}/{sample}"
    if is_unified:
      plot_id += "_bi"
    session_dir = f"{Path(parent_dir).name}"
  else:
    if "SamplingType" in animal_df.columns:
      sample = animal_df["SamplingType"].iloc[0]
    else:
      sample = "Sessions"
    layer = ""
    brain_region = f"{BrainRegion(animal_df['BrainRegion'].iloc[0])}".split("_")[0]
    session_type_path = f"{sample}/{brain_region}/{layer}"
    animal_name = animal_df["ShortName"].iloc[0]
    if alternate_animal_folder is None:
      session_dir = f"{animal_name}/"
    else:
      session_dir = f"{alternate_animal_folder}/{animal_name}_"
  if alternative_plot_id_col is not None:
    plot_id = animal_df[alternative_plot_id_col].iloc[0]
  if append_plot_id_col is not None:
    plot_id = plot_id + str(animal_df[append_plot_id_col].iloc[0])
  if save_context is not None:
    if isinstance(save_context, pd.Series):
      save_context = "_".join(f"{idx}={val}"
                              for idx, val in save_context.items())
    plot_id = f"{plot_id}_{save_context}"
  if "dur_bin" in animal_df.columns:
    dur_bin = animal_df.dur_bin.iloc[0]
    dur_bin = f"{dur_bin.left}s_{dur_bin.right}s"
    plot_id_split = plot_id.split('_')
    plot_id = f"{plot_id_split[0]}_{dur_bin}_{'_'.join(plot_id_split[1:])}"
  if len(fig_postfix):
    fig_postfix = f"_{fig_postfix}"
  norm = f"_{normalization_str}" if normalization_str is not None else ""
  # Thanks https://stackoverflow.com/a/57597617/11996983
  save_fp = Path(eval(f"f'{save_fp}'"))
  save_fp.parent.mkdir(exist_ok=True, parents=True)
  return save_fp
