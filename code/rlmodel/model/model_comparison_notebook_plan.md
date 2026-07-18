# Plan — `model_compare.ipynb`: per-subject × model fitting-criteria comparison grid

## Context

We have, on disk in `data/RLModel/`, several fits of the *same* DDM model produced
under different **fitting criteria**: pure MLE, joint MLE+Chi² (various Chi² weights),
pure Chi²-Noise, and pure Chi²-Bound. Today the only way to view a fit is the
one-fit-at-a-time interactive GUI (`model_interactive.ipynb` → `visualize.createWidget`),
and the MLE re-evaluation diagnostics live inline in `mle_debug.ipynb`. There is no
single view that puts all criteria for one subject × model **side by side** so we can
judge which criterion best reproduces behavior.

`model_compare.ipynb` fills that gap: for a chosen subject × model it renders a grid
whose **columns are the fitting criteria** and whose **rows are diagnostic plots**
(loss distributions + the existing behavioral panels). An interactive cell picks one
subject × model via dropdowns; a batch cell renders every combination found on disk and
saves each to `results/RLModel/fig_model_cmp/`. Per the user's directive, the notebook
itself stays thin (data load + config only); all logic lives in reusable modules, and
notebook-resident helpers (`mle_debug.ipynb`) are exported to files and de-duplicated.

### Decisions captured during planning
- **Containment-mixture ratio** (3rd header line, MLE/joint columns only) = the fitted
  **λ (`LAPSE_RATE`)** read from the fit's params. Chi² columns omit this line (they
  don't fit λ).
- **Row→plot mapping**: *Reward-Rate* row = Reward-Rate-vs-RT; *Alpha distribution* =
  Q-Value-Dist histogram; *Beta distribution* = Reward-Rate-Dist histogram; *Bias dist*
  = starting-point histogram.
- **Asymmetric-LR fits** (`asymQ/asymRR/asymQRR`) are **separate model-dropdown entries**
  (e.g. `RewardRate [asymQRR]`), keeping the criteria columns clean.

## Model identity vs. criteria columns (the discovery model)

A **"model"** (dropdown entry) = `(user-facing drift alias, bias, noise, t_dur, dt,
asym_variant)`. The **columns** within a subject × model figure are the fitting criteria.
This split is forced by the filenames: Chi²-Noise uses drift `NoiseGain-RewardRate`
while Chi²-Bound uses `Bound-RewardRate` + `_scaledB` — different drift strings for the
*same* abstract model — so grouping must key on the **drift alias**, not the raw string.

- Group drift via `drift.user_facing_drift_keys()` / the `_REWARDRATE_ALIAS_FOR_INTERNAL`
  reverse map (both in [model/drift.py](rlmodel/model/drift.py)); non-aliased drifts
  (`Classic`, `Decay Q (Offset)`) map to themselves.
- `_scaledB` and the joint weight suffix are **column axes**, never part of model identity.

**Criteria column classification** (from a parsed filename `(fit_mode, scaled_bound,
chi2_weight)`), rendered only when the file exists, in this order:
1. `MLE` — `mle`, `chi2_weight == 0`, not scaledB
2. `MLE={m}, Chi²={c}` (joint) — `mle`, `chi2_weight > 0`, not scaledB; ordered by
   ascending `c` (e.g. `Chi²=0.1`, then `Chi²=0.5`)
3. `Chi²-Noise` — `chisq`, not scaledB
4. `Chi²-Bound` — `chisq`, scaledB

Combos outside these four families (e.g. `mle`+scaledB, joint+scaledB) are **skipped in
v1** with a printed note.

## New/changed files

| File | Change |
|---|---|
| new [rlmodel/model_compare.ipynb](rlmodel/) | Thin notebook: import cells (mirror `model_interactive.ipynb` cells 0–4), load behavior + fit results, set global config, then two focus cells (interactive + batch) each calling `compare.plot_subject_model` / `compare.run_batch` |
| new [rlmodel/model/compare.py](rlmodel/model/) | The engine: column discovery/classification/order, the grid orchestrator `plot_subject_model`, `run_batch`, and thin per-row behavioral plotters over `plotter.py` |
| new [rlmodel/model/mle_reeval.py](rlmodel/model/) | Extracted from `mle_debug.ipynb`: `prepare_behavior_df`, robust `parse_fit_filename` (full drift/bias/noise + suffixes), `fitted_params_to_mle_df`, and the two loss plots refactored to take an `ax` (`plot_loss_distribution`, `plot_rt_hist_colored_by_loss`) |
| [rlmodel/model/plotter.py](rlmodel/model/plotter.py) | Refactor only (no behavior change): split `_plotDists` into `_plotQDist` / `_plotRewardRateDist` / `_plotBiasDist` (kept called by `_plotDists`); extract the Reward-Rate-vs-RT block from `plotPlots` into `_plotRewardRateVsRt(df, ax, subject)` |
| [rlmodel/mle_debug.ipynb](rlmodel/mle_debug.ipynb) | Replace the inline helper-def cells with `from .model.mle_reeval import …` (de-dup; keep the driver/inspection cells) |
| new [rlmodel/model/tests/test_model_compare.py](rlmodel/model/tests/) | Filename→(model identity, column label) classification + ordering; row-flag/include gating; headless (Agg) render of `plot_subject_model` over a tiny synthetic fit dict |

## Notebook structure (minimal)

- **Import/setup cells** — copied verbatim from `model_interactive.ipynb` cells 0–4
  (`%load_ext autoreload`, the `PKG` bootstrap, `%matplotlib widget`, SVG rcParams).
- **Load cell** — `df_behavior = compare.prepare_behavior_df()`;
  `fits = compare.discover_fits()` (globs `data/RLModel/`, parses each file into model
  identity + criterion, builds the subject × model → {criterion: fit-entry} index).
- **Config cell (globals)**:
  ```python
  FIG_COL_WIDTH, FIG_ROW_HEIGHT, DPI = 4, 3, 100          # size = (n_cols*W) x (n_rows*H)
  OUTPUT_DIR = Path("../../results/RLModel/fig_model_cmp")
  IMG_EXT = "png"
  MLE_SCORE_TERMINAL_C = 0.0        # terminal-C used for the MLE re-eval (header + loss rows)
  MLE_SCORE_LAPSE_OVERRIDE = None   # None = each fit's baked-in λ (chisq→0); float overrides
  ROW_FLAGS = {                     # toggle any row on/off
      "losses_dist": True, "hist_by_loss": True,
      "rt_corr_incorr": True, "rt_direction": True, "psychometric": True,
      "reward_rate": True, "beta_dist": True, "alpha_dist": True, "bias_dist": True,
  }
  ```
- **Interactive cell** — two `ipywidgets.Dropdown`s (Subject, Model) wired with
  `interactive_output` to `compare.plot_subject_model(subject, model, fits, df_behavior,
  row_flags=ROW_FLAGS, …config…)`.
- **Batch cell** — `compare.run_batch(fits, df_behavior, row_flags=ROW_FLAGS, out_dir=
  OUTPUT_DIR, …)` loops every subject × model, saving `{model}_{subject}.{IMG_EXT}`
  (`fig.savefig(..., dpi=DPI)`); filenames sanitized for the filesystem.

## Grid rows (each gated by `ROW_FLAGS`; header always on)

Per column the engine computes **two forward passes** from the same fitted params:
its **MLE re-eval** (`mle_reeval.fitted_params_to_mle_df`, using `MLE_SCORE_TERMINAL_C`
/ `MLE_SCORE_LAPSE_OVERRIDE`) → header + loss rows; and the **chisq simulation**
(`runAndPlot(df, …, axs=None)` → the reward-rate/prev-trial-annotated sim df) → behavioral
rows.

1. **Header** (text axis, `axis('off')`): line 1 = criterion label; line 2 =
   `MLE-Score: {neg_loglik:,.1f}` from the re-eval; line 3 (MLE/joint only) =
   `λ: {LAPSE_RATE:.3f}`.
2. **Losses distribution** — `mle_reeval.plot_loss_distribution(ax, mle_df)` (hist of
   clamped `-mle_loglik`).
3. **Hist colored by loss** — `mle_reeval.plot_rt_hist_colored_by_loss(ax, mle_df)`
   (stacked `calcStimulusTime`, red = outliers `loglik ≤ CUT_OFF`, blue = rest).
   *(v2 note in code: swap red/blue for a graded colormap by loss value.)*
4. **RT Hist (Correct/Incorrect)** — cell split into 2 stacked axes via
   `gridspec[r,c].subgridspec(2,1)`; `plotter._plotHist(df, ax_up, ax_down,
   "ChoiceCorrect", "SimChoiceCorrect", t_dur, dt)`.
5. **RT Hist (direction)** — same 2-axis split; `_plotHist(df, ax_up, ax_down,
   "ChoiceLeft", "SimChoiceLeft", t_dur, dt)`.
6. **Fast/slow psychometric** — `_psychAxes(ax)` then `plotter._plotPsychs(df, ax,
   PsychometricPlot.SlowFast)`.
7. **Reward-Rate** — `plotter._plotRewardRateVsRt(df, ax, subject)` (new extracted fn).
8. **Beta distribution** *(only if model includes RewardRate)* —
   `plotter._plotRewardRateDist(df, ax, …)`.
9. **Alpha distribution** *(only if model includes Q)* — `plotter._plotQDist(df, ax, …)`.
10. **Bias dist** — `plotter._plotBiasDist(df, ax, bound, biasFn_kwargs)`.

`include_Q` / `include_RewardRate` (gates rows 8–9 and passed to `runAndPlot`) come from
the fit payload's stored `include_Q`/`include_RewardRate`, falling back to column
detection via `model.util.{biasFn,driftFn,noiseFn}ColsAndKwargs` (the same logic
`visualize` uses).

## Reuse (existing code, do not re-implement)
- **Data/fit load**: `model_runner.{loadDF,_extendTrials,_reduceDFSize,DF_FP}`,
  `behavior.rewardrate.calcAvgRewardRate` (as in both notebooks' load cells).
- **MLE re-eval**: `mle.evaluate_neg_loglik` + `MLEModelConfig` (supports
  `uses_scaled_bound` for Chi²-Bound columns) — wrapped by
  `mle_reeval.fitted_params_to_mle_df`.
- **Forward sim + behavioral panels**: `plotter.runAndPlot(axs=None)` returns the fully
  annotated sim df; `plotter._plotHist`, `_plotPsychs`, and the newly-split dist helpers
  render onto our axes.
- **Filename/weights parsing**: adapt `extract_model_losses._parse_filename` (weights) +
  `model_interactive` cell-10 parser (drift/bias/noise + `_scaledB`/`_asym` suffixes) +
  `drift.resolve_drift_alias` into `mle_reeval.parse_fit_filename`.
- **Params extraction**: `mle_notebooks.data.fitted_params_from_result`.

## Verification
- **Unit** (`pytest rlmodel/model/tests/test_model_compare.py`): assert the parser+
  classifier maps the real on-disk names to the right (model identity, column label),
  e.g. `chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_Normal(0, 1)_4.8s_dt0.005.pkl` →
  model `RewardRate|Q-Val (Offset)|Normal(0, 1)|4.8|0.005|sym`, column `Chi²-Noise`; the
  `_scaledB` sibling → same model, column `Chi²-Bound`; `_mleW1_chi2W0.5` → column
  `MLE=1, Chi²=0.5`; and that column order is MLE → joints(asc) → Chi²-Noise → Chi²-Bound.
- **Headless render**: with `matplotlib.use("Agg")`, run `plot_subject_model` on a tiny
  synthetic `df_behavior` + 2-column fit dict and assert a Figure with the expected
  `(n_rows, n_cols)` axis count comes back and `savefig` writes a file; toggle a
  `ROW_FLAGS` entry and assert the row count drops.
- **Regression**: `pytest rlmodel/model/tests/` stays green (the `plotter._plotDists`
  split and Reward-Rate-vs-RT extraction preserve behavior); re-run `mle_debug.ipynb`
  after the import swap to confirm the extracted helpers still drive its cells.
- **End-to-end (manual)**: run the interactive cell, pick a subject × model with ≥2
  criteria on disk, confirm columns/rows match the flags and the header shows MLE-Score
  (+λ on MLE columns); run the batch cell and confirm one `{model}_{subject}.png` per
  combination lands in `results/RLModel/fig_model_cmp/`.

## Risks / notes
- **Two forward models per column** (MLE re-eval + chisq sim) ⇒ batch cost scales with
  subjects × models × columns; `run_batch` prints progress and closes each figure
  (`plt.close`) to bound memory.
- **Chi²-Bound MLE re-eval** must pass the internal drift (`Bound-RewardRate`) and
  `uses_scaled_bound=True` into `MLEModelConfig`; mirror `visualize`'s scaledB config path.
- **MLE-Score comparability**: a single notebook-global `MLE_SCORE_TERMINAL_C` /
  `MLE_SCORE_LAPSE_OVERRIDE` is applied to every column's re-eval so scores are
  comparable; documented in the config cell.
- `mle_debug.ipynb` edits are mechanical (def-cells → imports); its driver/inspection
  cells are untouched.
