# Plan: Neuron ↔ MLE-parameter correlation notebook (2-photon × RL model)

## Context

We have (a) 2-photon per-trial, per-neuron calcium traces aligned to the
sampling window (+0.1 s before / +0.1 s after) and (b) trial-by-trial RL-model
latents from MLE fits (Q-left, Q-right, relative Q "Q-val", RewardRate). The
goal is a new **modelling** notebook that asks *which neurons' activity
correlates with which model parameter*, producing (1) a per-neuron correlation
table, (2) per-neuron activity-vs-parameter scatter plots, and (3) per–brain-region
"fraction tuned" pie charts. This is a delicate analysis, so the notebook stays
a **thin config layer** over reusable backend code, mirroring the existing
`model_compare.ipynb` → `model/compare.py` split.

All decisions confirmed with the user:
- **Activity source:** `data/2p/normalized_0.1s_before_sampling_0.1s_after_movement.pkl`
  (per-neuron z-scored **and** width/time-normalized). Both the stored full trace
  and the max-activity value come from this file.
- **Correlation:** Pearson (`scipy.stats.pearsonr`), within session.
- **Max-loss exclusion:** per **subject** (`Name`) — drop trials whose per-trial
  −loglik equals that subject's maximum (within tolerance).
- **Parameter mapping (before-trial latents):** Q-Left=`mle_Q_left_before`,
  Q-Right=`mle_Q_right_before`, Q-val=`mle_Q_rel_before` (relative Q),
  RewardRate=`mle_reward_rate_before`.

## Files

- **New backend module:** `code/rlmodel/model/neural_correlate.py` — all
  side-effecting/analysis code lives here (functions only; the notebook only
  sets config and calls them). Sits alongside `compare.py` / `mle_reeval.py`.
- **New tests:** `code/rlmodel/model/tests/test_neural_correlate.py`
  (run with `uv run pytest -k neural_correlate` from repo root).
- **New notebook:** `code/rlmodel/model_neural_correlate.ipynb` (cwd = `code/rlmodel`).

> Principle-4 note: the 2p reduction/loss code we reuse already lives in
> **modules** (`twop/activity_correlation.py`, `twop/dimreduc.py`,
> `model/mle_reeval.py`, `model/compare.py`) — nothing is being extracted out of
> a notebook, so no pre-extraction tests are required. The new correlation / pie
> code is written fresh in the module *with* tests.

## Reused backend (no duplication)

- **2p per-(neuron,trial) scalar table:** adapt the shape of
  `twop/activity_correlation.py::getTrialsMeanActivity` — same loop
  (`BrainRegion → ShortName → TrialNumber`), same window
  `traces_sets["neuronal"][trace_id][trace_start_idx : trace_end_idx+1]`, same
  `long_trace_id = f"{ShortName}_{trace_id}"` — but reduce with **`np.nanmax`**
  (equivalently `twop/dimreduc.py::DimReduc.max`) and additionally keep the full
  (time-normalized) trace object.
- **MLE re-evaluation → per-trial `mle_df`:** `model/mle_reeval.py`
  (`prepare_behavior_df`, `parse_fit_filename`, `fitted_params_from_result`,
  `build_mle_config`, `evaluate_params_under_mle`). `_build_mle_df`
  (`model/mle.py:1455`) already emits `mle_Q_left_before`, `mle_Q_right_before`,
  `mle_Q_rel_before`, `mle_reward_rate_before`, `mle_loglik`,
  `mle_valid_for_loss` joined onto `Name/Date/SessionNum/TrialNumber`.
- **Friendly model name:** `parse_fit_filename(fname).model_label`
  (e.g. `RewardRate · Q-Val (Offset) · Normal(0, 1) · 4.8s [asymQRR]`);
  filesystem-safe via a local sanitizer mirroring `compare.py::_safe_filename`.
- **Brain region → `MFC`/`LFC`:** `f"{BrainRegion(br)}"` (the `__format__`
  override maps `M2_Bi`→`MFC`, `ALM_Bi`→`LFC`). Never store the enum value or
  `*_Bi`.
- **Behavior/rcParams/preamble idioms:** copied verbatim from
  `model_compare.ipynb` (package-import preamble `root_parent_level=2`;
  `%matplotlib inline` + `svg.fonttype=none`/Arial rcParams).

## Backend functions (`neural_correlate.py`)

1. `prepare_behavior_df(...)` — thin re-export of `mle_reeval.prepare_behavior_df`.
2. `load_mle_per_trial(fit_pkl_path, df_behavior, *, mle_terminal_c=0.0, lapse_override=None)`
   - Parse filename → `fid`; load the `{subject: payload}` pickle.
   - Read `include_Q` / `include_RewardRate` from the payload; **print a clear
     report** of which parameter families are available (principle 6 — model may
     be reward-rate-only or Q-only). Absent families → their columns omitted.
   - Per subject: `evaluate_params_under_mle(...)` → concat `mle_df`.
   - Return per-trial df with `Name/Date/SessionNum/TrialNumber`, `mle_loglik`,
     `mle_valid_for_loss`, `model_label`, and renamed param columns
     `Q_L, Q_R, Q_val, RewardRate` (only the available ones).
3. `load_2p_activity(path)` — load the normalized pickle; keep one row per
   `(Name,Date,SessionNum,TrialNumber)` (guard: filter `epoch=="Sampling"` /
   drop_duplicates on identity keys); restrict to `BrainRegion ∈ {M2_Bi, ALM_Bi}`.
4. `build_neuron_trial_table(df_2p, mle_per_trial, *, exclude_max_loss=False, tol=1e-9)`
   → **intermediate dataframe** (principle 7), one row per (neuron, trial):
   `trace_id` (neuron name), `long_trace_id` (long name),
   `BrainRegion` (=`MFC`/`LFC` string), `ShortName` (session), `TrialNumber`,
   `max_activity` (=`np.nanmax` of the windowed time-normalized neuronal trace),
   `trace` (full time-normalized trace, object), and available `Q_L/Q_R/Q_val/RewardRate`.
   - Join model params on `(Name,Date,SessionNum,TrialNumber)` — reconcile dtypes
     (Date→normalized datetime; SessionNum/TrialNumber→int). Inner join naturally
     drops MLE padding trials (their TrialNumbers never occur in 2p).
   - `exclude_max_loss=True`: per subject, `max_loss = (-mle_loglik).max()` over
     valid trials; drop trials with `-mle_loglik >= max_loss - tol`.
5. `compute_neuron_correlations(table, *, method="pearson")` → **per-neuron
   correlation dataframe** (principle 8): `trace_id`, `long_trace_id`,
   `BrainRegion`, `ShortName`, and one `<param>_r` column per available parameter
   (Pearson r of that neuron's per-trial `max_activity` vs the parameter, within
   its session; NaN when constant/too-few-points). Optionally `<param>_p`.
6. `plot_neuron_results(table, corr_df, param, *, mode="display", top_x=20, zscore=False, save_root, model_name, ext="svg")`
   (principle 9): sort by `abs(<param>_r)` desc; scatter parameter (x) vs
   `max_activity` (y, y-label **"Neuron Activity"**; z-scored across the neuron's
   trials when `zscore=True`); x-label = friendly param name
   (Q-Left / Q-Right / Q-val / R-Learning RewardRate).
   - `mode="display"`: show the top-`top_x` neurons for `param`.
   - `mode="save"`: save **every** neuron to
     `{save_root}/{model_name}/traces/{param_name}/{signed_corr}_{long_name}[ _region]`.
     Append brain region only when `long_name` doesn't already contain it. Close
     the figure, don't show.
7. `plot_region_pie(corr_df, param, min_abs_corr, *, by_region=True, save=False, save_root, model_name)`
   (principle 10): a neuron is "tuned" if `abs(<param>_r) >= min_abs_corr`.
   Per session → % tuned; **average across sessions ± SEM**, rendered as a pie
   (gray + colored wedge, label `"{mean:.2f}%\n±{sem:.2f}%"`, exploded), mirroring
   `2pAnalysis.ipynb::loopPlotPieChart`. `by_region=True` → one pie per MFC/LFC;
   `by_region=False` → a single combined pie (so "everything" is possible without
   breaking down by region). **Always displayed**, even when `save=True`; when
   saving, data goes under `{save_root}/{model_name}/{param_name}_brain_region/`.

## Notebook layout (`model_neural_correlate.ipynb`, thin config)

1. Package-import preamble (verbatim from `model_compare.ipynb`).
2. `%matplotlib inline` + rcParams (svg.fonttype=none, Arial).
3. **Config globals** — `MODEL_FIT_PATH` (the model pickle, declared here),
   `TWOP_ACTIVITY_PATH` (the normalized pkl), `BASE_SAVE_PATH`
   (default `../../results/RLModel/neural_correlate`), `EXCLUDE_MAX_LOSS`,
   `MIN_ABS_CORR`, `TOP_X`, `ZSCORE_ACTIVITY`.
4. Load: `df_behavior = neural_correlate.prepare_behavior_df()`;
   `mle_pt = neural_correlate.load_mle_per_trial(MODEL_FIT_PATH, df_behavior)`;
   `df_2p = neural_correlate.load_2p_activity(TWOP_ACTIVITY_PATH)`.
5. `table = neural_correlate.build_neuron_trial_table(df_2p, mle_pt, exclude_max_loss=EXCLUDE_MAX_LOSS)`.
6. `corr_df = neural_correlate.compute_neuron_correlations(table)`.
7. Display scatters: `plot_neuron_results(..., param="Q_val", mode="display", top_x=TOP_X)`.
8. Save-all scatters: `plot_neuron_results(..., mode="save", save_root=BASE_SAVE_PATH, model_name=...)`.
9. Pie charts: `plot_region_pie(corr_df, param="Q_val", min_abs_corr=MIN_ABS_CORR, by_region=True)`.

Only `table`, `corr_df`, `df_2p`, `mle_pt`, `df_behavior` are cell-crossing data
variables; all logic with side effects is inside `neural_correlate.py` functions.

## Verification

- `uv run pytest -k neural_correlate -q` from repo root — new tests cover:
  `np.nanmax` reduction + windowing, `long_trace_id` and `MFC`/`LFC` mapping, the
  identity-key join, per-subject max-loss exclusion, Pearson r on a constructed
  linear relationship, parameter-availability reporting when
  `include_Q`/`include_RewardRate` is False, and the signed-corr/region save-path
  builder.
- Run the notebook end-to-end with `MODEL_FIT_PATH` set to
  `data/RLModel/mle_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005_asymQRR.pkl`
  and confirm: the availability report prints, `table`/`corr_df` are populated,
  display scatters render for the top neurons, save-mode writes the
  `{model}/traces/{param}/...` tree, and both by-region and combined pies render.
