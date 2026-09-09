# Repository audit — findings from the manuscript mapping pass

Everything here was verified against the working tree on 2026-08-30 (branch
`master2`). Each finding names the fork it belongs to, so the cleanup work can
be picked up independently.

Fork topics this feeds:

1. Model documentation
2. Removing model parts not used by the manuscript
3. Unit tests for the behaviour code
4. One unified `uv` environment
5. Reorganising the two-photon notebooks and backend
6. Removing code and figures not on the path to a manuscript figure
7. Notebook parameterisation for automated execution
8. General adherence to `CLAUDE.md` and best practice

---

## Baseline: the test suite

`uv run pytest` from the repo root: **805 passed, 7 failed, 1 skipped** in ~47 s.

All seven failures are the same bug — the default figure extension changed from
`svg` to `pdf` in the plotting code, and the assertions were not updated:

| Test | Asserts | Code writes |
|---|---|---|
| `test_neural_correlate.py::test_region_bars_runs_headless` | `bars_combined.svg` | `.pdf` (default `ext="pdf"`) |
| `…::test_plot_factor_bars_runs_headless` | `bars_Q_RewardRate_DV_combined.svg` | `.pdf` |
| `…::test_plot_factor_bars_fastslow_dv` | `bars_fastslowDV_combined.svg` | `.pdf` |
| `…::test_plot_results_save_min_abs_corr_filters` | `*.svg` glob | `.pdf` |
| `…::test_param_traces_save_runs_and_filters` | `*.svg` glob | `.pdf` |
| `…::test_fastslow_scatter_and_traces_save` | `*.svg` glob | `.pdf` |
| `twop/tests/test_corrthreshregions.py::test_plot_runs_headless_and_saves` | `rt_corr_regions_bars_above_0.3.svg` | `rt_corr_regions_bars_above_0.3.pdf` |

Six of them simply need `ext="svg"` passed (the sibling tests already do);
`corrthreshregions.plotRegionBars` has no `ext` parameter at all and hardcodes
`.pdf` at [`corrthreshregions.py:200`](../code/twop/plot/corrthreshregions.py#L200)
— worth giving it the same knob. **The manuscript figures are SVG**, so the
default extension is a real question, not only a test question.

*Fork 3 / 8.*

### Test coverage by package

| Package | Test files | Notes |
|---|---|---|
| `rlmodel/model` | 31 files | Well covered. |
| `twop` | 5 files | `twop/tests/` has **no `__init__.py`** unlike the other test packages. |
| `behavior` | 2 files (`stdispersion`, `stkde`) | The Methods-documented analyses are untested — see below. |
| `common` | 1 file | |
| `widefield` | 1 file | |
| `pipeline` | 1 file — **not collected**: `pipeline/tests` is missing from `testpaths` in `pyproject.toml`. |
| `opto` | **0** | Hierarchical bootstrap (Figures 3D, 4C, S6G) is untested. |
| `figcode` | **0** | Psychometrics, stay/switch, sampling-time heatmaps are untested. |

*Fork 3.*

---

## The environment clash — root cause found

**Symptom.** Model fit pickles in `data/RLModel/` load in the conda `py312` env
but not in the repo's `.venv`, even though both report pandas 2.3.3.

**Minimal reproduction** (verified):

```python
# in conda py312 (pandas 2.3.3, conda build)
pd.DataFrame({"s": pd.array(["a", "b", "c"], dtype="string")}).to_pickle("probe.pkl")

# in .venv (pandas 2.3.3, PyPI wheel)
pd.read_pickle("probe.pkl")
# NotImplementedError: (string[python], array(['a', 'b', 'c'], dtype=object))
#   at pandas/_libs/arrays.pyx:103 NDArrayBacked.__setstate__
```

**Cause.** The conda-built pandas 2.3.3 pickles `StringArray` in the legacy
2-tuple `NDArrayBacked` state; the PyPI-built pandas 2.3.3 rejects it. It is
**one-directional** — pickles written by the `.venv` pandas read fine in conda.
Ruled out: numpy version (fails identically at 2.4.6 and 2.5.0), `pyarrow`
(absent from both), `future.infer_string` (False in both). A fresh
`uv run --isolated --with pandas==2.3.3` fails too, so this is not a stale
`.venv`.

**What is affected.** Every `data/RLModel/*.pkl` fit — the `subject_df` inside
carries `DVstr` and `Name` as pandas `string` dtype. `data/behavior/*.pkl` and
`data/2p/*.pkl` load fine in both environments.

**Three environments currently in play:**

| Environment | Python | pandas | numpy | Role |
|---|---|---|---|---|
| `.venv` (uv, `pyproject.toml` + `uv.lock`) | 3.14.3 | 2.3.3 (PyPI) | 2.5.0 | `uv run pytest`, scripts |
| conda `py312` | 3.14.5 | 2.3.3 (conda) | 2.4.6 | Notebook kernel; the only env that reads the fit pickles |
| conda `wfield` | 3.11.5 | 1.5.3 | 1.26.2 | widefield/`wfield` work; also cannot read the fit pickles (`No module named 'numpy._core.numeric'`) |

**Recommended fix for fork 4.** Stop persisting pandas extension dtypes in the
artifacts. Cast `DVstr` / `Name` to `object` (or `category` with object
categories) before `to_pickle` in `rlmodel/model/fit.py`, and add a one-off
re-write pass over the existing `data/RLModel/*.pkl` from the conda env. Then a
single uv environment can own everything. The `wfield` package (pinned to old
numpy) is only imported optionally by
[`widefield/pipelineprocessors.py`](../code/widefield/pipelineprocessors.py),
which already degrades gracefully — it can stay an extra, not a base
requirement.

*Fork 4.*

### Dependency declaration gaps

Imported by the code but absent from `pyproject.toml`:

| Package | Used by |
|---|---|
| `matplotlib_venn` | Figure 4G and 6E (`2pAnalysis.ipynb`), `twop/plot/rocplot.py`, `twop/plot/statsregiondist.py` |
| `cv2` (opencv) | imaging / tracking paths |
| `tifffile` | `common/plottracesprocessor.py` TIFF export |
| `requests` | `data_downloader.ipynb` |
| `dill` | `common/cache.py` (guarded by bare `except:`) |
| `cupy` | 3 modules under `rlmodel/model/` (optional GPU MLE backend) |
| `wfield` | `widefield/pipelineprocessors.py` (guarded, prints to stderr) |

`code/README.md` still advertises the **old conda list** — `pandas<2.0.0`,
`pyddm 0.8.0`, `python=3.11`, `h5py` — none of which matches reality.
`pyddm` is no longer used anywhere except a comment in
[`model/visualize.py:871`](../code/rlmodel/model/visualize.py#L871).

*Fork 4 / 8.*

---

## Inline notebook code that produces published panels

`CLAUDE.md` requires that "main code should live in python files; the notebook
itself calls the backend code." These panels violate that, and none of them has
tests:

| Panel | Notebook | Inline functions | Manuscript status |
|---|---|---|---|
| ~~**Figure 2A**~~ | — | **extracted** to `behavior/varexplained.py` (21 tests) | Main figure; four numbers quoted in Results |
| ~~**Figure 2B**~~ | — | **extracted** to `behavior/optimalsampling.py` (27 tests); the three duplicated copies collapsed to one | Main figure; eqs. 1–3 in Methods. Extraction found a published-figure bug — see below |
| **Figure 1I-right** | `behavior.ipynb` | `plotSubjectsQuantileUpdate` | Main figure; *p* = 0.0046 |
| **Figure S2B, S2M** | `behavior.ipynb` | `assignZScoredST`, `errorsDistribution`, `processSubject`, `_plotGroup`, `localSlowFasPsych` | Supplementary |
| **Figure S3G** | `behavior.ipynb` | `_plotOverTime`, `loopDifficulties`, `loopWinStay`, `staySwitchCDF` | Supplementary; carries a documented 2-mouse exclusion rule |
| **Figures S3J–M** | `Tracking.ipynb` | all of it — the notebook imports **no** repo module | Supplementary |
| **Figures 4G, 6B, 6C, 6E, S9A–B, S14A** | `2pAnalysis.ipynb` | `_fastSlowOverlap`, `extractTracesPreferences`, `_plotBrainRegionTuning`, … | Main + supplementary |
| **Figures 4K, S10A–C, S11A-mid/right, S11B, S12G** | `plottraces3.ipynb` | the AUC/peak correlation machinery, the early/late active-neuron binning, the decoder loop | Main + supplementary |
| **Figure 7D** | `model_to_behavior.ipynb` | `plotQ_R_Heatmap` | Main figure |

### Defect found by extracting Figure 2B

The published `results/behavior/optimal_sampling/all_subj_aligned.svg`
(Figure 2B-right) draws **all 17 error bars with one animal's SD**. The
population loop read `m["emp_sampling_time_sem"]` from a variable left over
from an earlier loop, so every bar got the last animal's value rather than its
own:

```python
for name, m in subj_metrics.items():      # m ends up bound to the last animal
    ...
for yi, (name, delta_emp, t_opt, p_emp) in enumerate(meta):
    ax.errorbar(delta_emp, yi, xerr=m["emp_sampling_time_sem"], ...)
```

Measured off the published SVG: 17 bars, every one 130.0 px wide = ±0.621 s,
which is exactly `vgatchr2-sk`'s SD. The real per-animal SDs span 0.347–0.924 s
(2.7×), so the error bars are visibly wrong. Everything else in the panel is
correct — all 17 Δ-positions reproduce to 0.00 s.

`behavior/optimalsampling.py` uses each animal's own SD, pinned by
`test_population_panel_uses_each_animals_own_sd`. **The published SVG needs
regenerating**, and the figure in the manuscript with it. No quoted number
changes: the legend cites no values from this panel.

Two naming notes carried over: the field was called `..._sem` but always held
`.std()`, and the manuscript legend correctly says SD — so only the variable
name was wrong. It is `observed_sampling_time_sd` now.

Inline-versus-module balance (top-level `def`s in code cells against `from .`
imports, measured on the extracted code cells):

| Notebook | Lines of code | Inline `def`s | Module imports |
|---|---|---|---|
| `plottraces3.ipynb` | 4,020 | 65 | 37 |
| `2pAnalysis.ipynb` | 3,732 | 78 | 23 |
| `behavior.ipynb` | 3,353 | 41 | 29 |
| `TwoPLoad.ipynb` | 2,100 | 29 | 20 |
| `Tracking.ipynb` | 1,166 | 20 | **0** |
| `widefield.ipynb` | 744 | 9 | 8 |
| `TwoPTraces.ipynb` | 686 | 5 | 8 |
| `opto.ipynb` | 506 | 5 | 18 |
| `2pSeqWithinDeviation.ipynb` | 346 | 1 | 2 |

`opto.ipynb` and `2pSeqWithinDeviation.ipynb` are the models to imitate: thin
notebook, fat modules.

*Forks 3, 5, 8.*

### `plottraces3.ipynb` has cells that cannot run

Four cells carry an explicit upstream note that they raise `NameError` as
written (`plotSgfActivitySum` defined only inside a commented-out block;
`_iqrPRCNT`, `max_firing_all_df`, `svm_normed_df` never defined). A fifth
depends on run order (`res_corr_raw` assigned inside a conditional). This
notebook cannot be executed end-to-end, which blocks fork 7.

*Forks 5, 7.*

---

## Orphaned backends

These modules are imported by **nothing** — not by another module, not by any
notebook in the repository:

| Module | Lines | What it draws | Live implementation |
|---|---|---|---|
| `twop/plot/statspriorcuroverlap.py` | 98 | Figure 6E prior/current overlap | **inline** `_plotBrainRegionTuning` in `2pAnalysis.ipynb`; output filenames differ (`prior_cur_overlap_prcnt_of_*.jpeg` vs the published `*_prior_current_tuning.svg`) |
| `twop/plot/statssamplingfeedback.py` | 138 | Figure 6B sampling/feedback tuning | inline in `2pAnalysis.ipynb` |
| `twop/plot/statsregiondist.py` | 217 | region distribution charts | — |
| `twop/plot/statsearlylatesampling.py` | 114 | early/late sampling stats | inline in `plottraces3.ipynb` |
| `twop/plot/statsbetweenregions.py` | 163 | cross-region tuning | — |
| `twop/plot/plottracetrialsheatmap.py` | 127 | Figure 4H trial heatmap | inline / `twop/plottracesavg.py` |
| `twop/plot/plotutil.py` | 52 | helper for the five above | — |
| `twop/trackactivity.py` | 242 | — | — |
| `twop/tuningliklihood.py` | 140 | — | — |
| `opto/permute2regions.py` | 272 | alternative permutation test to the hierarchical bootstrap | `opto/bootstrap2regions.py` |
| `pipeline/plottershelper.py` | 31 | — | — |
| `figcode/prevoutcomecurquantile_bak.py` | 190 | earlier Figure 1I-left | `prevoutcomecurquantile.py` |
| `figcode/prevoutcomecurquantile_new.py` | 217 | ditto | `prevoutcomecurquantile.py` |
| `twop/relogit/vignette.py` | 87 | upstream demo of the vendored package | — |

That was **1,888 lines** of unreferenced code.

> **Resolved 2026-09-06 (D0).** The `twop/plot` cluster — the seven `stats*` /
> heatmap modules plus `plotutil.py` — was **deleted**, 1,052 lines. The
> hypothesis above (that they might be a better factoring worth adopting) did
> not survive contact with the evidence: they all write `.jpeg`, `results/`
> contains zero JPEG, none of their output filenames appears anywhere in
> `results/`, and they hold no hypothesis test the inline code lacks. The
> notebook's inline functions demonstrably write the published SVGs. See
> [`execution-plan.md`](execution-plan.md) for the full account.
>
> The rest of the table — `trackactivity.py`, `tuningliklihood.py`,
> `permute2regions.py`, `plottershelper.py`, the two
> `prevoutcomecurquantile` copies and `relogit/vignette.py` — is untouched and
> still open.

**Decision needed before deleting the `twop/plot/stats*` family:** they look
like a *better* factoring of what is currently inline. It may be cheaper to
adopt them (and delete the inline copies) than to delete them. Comparing their
output against the published SVGs is the first task of fork 5.

`figcode/prevoutcomecurquantile_{bak,new}.py` and `twop/relogit/vignette.py` are
unambiguous deletions.

*Forks 5, 6.*

---

## Model code the manuscript does not use

The manuscript fits four models (DDM, +QL, +RL, +QL+RL) plus four reward-rate
channels for Figure S4C. Registered but never used in any published figure:

**`model/bias.py::BIAS_FN_DICT`** — used: `None_`, `Q-Val (Offset)`.
Unused: `Fixed (Dir)`, `Fixed (Corr/Incorr)`, `μ, σ (Dir)`, `μ, σ (Corr/Incorr)`,
and the offset-free `Q-Val`.

**`model/noise.py::NOISE_FN_DICT`** — used: `Normal(0, 1)`.
Unused: `Decaying Q-Val`.

**`model/drift.py::DRIFT_FN_DICT`** — used: `Classic`, `NoiseGain-RewardRate`,
`Bound-RewardRate`, `DriftGain-RewardRate`, `DriftGain(1+r)-RewardRate`.
Unused: the entire **`Decay Q` family** — `Decay Q`, `Decay Q (Offset)`, and the
`* Decay Q` / `* Decay Q (Offset)` variants of NoiseGain, Bound, DriftGain and
DriftGain(1+r) — ten registry entries plus their implementing functions
`_decayQ`, `_noiseGainDecayingQ`, `_boundGainDecayingQ`, `_driftGainDecayingQ`.

Evidence that the `Decay Q` family is genuinely unused: the model list in
`model_viewer.ipynb` has all five `Decay Q` entries **commented out**; no
`data/RLModel/*.pkl` filename contains "Decay"; no `results/RLModel/`
subdirectory does either.

The corresponding `InitVals` fields `Q_VAL_DECAY_RATE`, `Q_VAL_COEF`,
`BIAS_MU`, `BIAS_SIGMA`, `BIAS_FIXED`, `ALPHA_UNREWARDED`, `BETA_UNREWARDED`
also become removable — note that `ALPHA_UNREWARDED` / `BETA_UNREWARDED` have a
dedicated test file (`tests/test_asymmetric_lr.py`) and a design document
(`asymmetric_lr_orthogonalization_plan.md`), so that one is a deliberate
unpublished experiment rather than dead weight.

**Removal is not purely subtractive.** The `Decay Q` / `Decaying Q-Val` names
are also reached by:

- `model/mle.py:141` — a branch keyed on `noise_fn_str == "Decaying Q-Val"`;
- `model_runner.py:482` — CLI help text for the `--asym` flag;
- `tests/test_asymmetric_lr.py:474` and `tests/test_mle_smoke.py:122` — both use
  `Decaying Q-Val` as a fixture;
- `model/visualize.py:781` — a GUI branch matching `"Fixed (Corr/Incorr)"` and
  `"μ, σ (Corr/Incorr)"`.

So fork 2 has to decide the fate of the asymmetric-learning-rate experiment at
the same time; the two are entangled through the noise function.

**Fit artifacts on disk vs. figures.** `data/RLModel/` holds 12 χ² fits and
3 MLE fits (pure MLE, joint w_χ²=0.1, joint w_χ²=0.5). Figure 2G uses four χ²
fits; Figure S4B/C uses the reward-rate channels; Figures 7A–B use the joint
w_χ²=0.5 fit. `results/RLModel/neural_correlate/` carries output for **both**
joint weightings, and the pure-MLE fit is an S14B comparison point — so all
three MLE pickles are load-bearing. None of the 15 fit files is a candidate for
deletion.

*Fork 2.*

### Model design documents in `rlmodel/`

Fifteen `.md` planning documents sit alongside the code
(`mle_assessment_and_workplan.md`, `mle_state_consolidation_plan.md`,
`scale_bound_equivalence_plan.md`, `slurm_script_plan.md`,
`Model_neural_correlation_plan.md`, `model/cupy_mle_plan.md`,
`model/gpu_problems.md`, `model/mle_optimization_plan.md`, …). These are working
notes, not documentation. Fork 1 should decide which of them graduate into
`docs/` and which are deleted before the repo goes public.

`code/in-code-2panalysis-ipynb-we-use-majestic-stream.md` and
`model/is-the-current-rlmodel-nashaat-oraby-mle-synthetic-fiddle.md` are
transcript-style artifacts that should not ship.

*Forks 1, 6.*

---

## Stale documentation

### `rlmodel/README.md`

| Statement | Reality |
|---|---|
| "maximum allowed duration (3s)" in the χ² section | `T_dur = 4.8` (`initvals.py`), matching the paper |
| Figure references `Fig. 1l`, `Fig. 1k`, `Fig. 5f middle`, `Ext. Fig. 5a` | Current numbering is `Figure 2G`, `Figure 2E`, `Figure 7D`, `Figure S4A/C` |
| No mention of MLE or joint MLE+χ² fitting | Both are Methods-documented and produce Figures 7A–B, S14B–H |
| "Many prior works use MLE-based fitting… We are open to adopting this approach if reviewers find it more appropriate" | MLE *was* adopted; this paragraph now reads as a stale reviewer response |
| Cluster instructions say "in the conda env (the fit pickles need it)" | True today — and is exactly the constraint fork 4 removes |

### `code/README.md`

Dependency section lists the retired conda environment (see above). The
"Missing" section already names the two gaps that forks 4 and 7 close: a
dependency metadata file and a papermill-style batch runner. It refers to
`TwoPAnalysis.ipynb` while the file is `2pAnalysis.ipynb`, and does not mention
`2pSeqWithinDeviation.ipynb`, `TwoPLoad.ipynb`, `behavior_v2.ipynb`,
`opto.ipynb` (listed) or the six `rlmodel/*.ipynb` notebooks.

### Root `README.md`

Titles the paper *"The neural mechanisms of fast versus slow decision-making"*;
the current title is *"Cortical mechanisms of fast versus slow decision making"*.

*Forks 1, 8.*

---

## Notebook parameterisation (fork 7)

Current state: every notebook defines a `global_save_figs` / `SAVE_FIGS` flag
and a `save_prefix` in an early cell, which is the right shape for
`papermill`-style injection. What is missing:

- No `parameters`-tagged cell, so papermill cannot override them.
- The flag is not honoured consistently. `behavior.ipynb` has **11** call sites
  passing `save_fig=True` / `save_figs=True` literally rather than
  `global_save_figs` — including Figure 1I-left (`prevOutcomeCurQuantile`), the
  Figure S3E heatmap cells, the stay/switch cells and the per-subject loops. One
  of them (`save_figs=True, #global_save_figs`) shows the flag was deliberately
  overridden and never restored. Setting `global_save_figs = False` therefore
  does **not** stop the notebook writing into `results/`.
- `plottraces3.ipynb` cannot run end-to-end (see above).
- Notebook run order matters in several places (`res_corr_raw`,
  `max_firing_all_df`), so a clean-kernel run is not currently reproducible.
- No runner script exists. `code/README.md` already scopes the intended one:
  a `uv` script plus papermill, with a flag to choose paper figures only versus
  all per-subject figures.

Data-path handling is also relative-cwd dependent (`"../data/behavior/…"`),
which will break under a runner invoked from the repo root.

*Fork 7.*

---

## Smaller items

- **`data/to_delete/`** is tracked-adjacent (untracked, but present) and holds
  `df_behavior_full.pkl`, `df_behavior_org.pkl`, `df_opto_all_org.pkl`,
  `mle_debug_cur.ipynb`, `trial_by_trial_mle_conversation.html`, `uvdeps.txt`
  and more. Clear before publishing.
- **`behavior_v2.ipynb`** (12 MB) sits beside `behavior.ipynb` (14 MB) with no
  note on which is current. `code/README.md` names only `behavior.ipynb`.
- **`rlmodel/model_GUI_cache.pkl`** and `rlmodel/run_cmd.txt` are working state.
- **Notebook output size**: `2pAnalysis.ipynb` and `TwoPLoad.ipynb` are ~10 MB
  each with outputs embedded; `behavior.ipynb` 14 MB, `plottraces3.ipynb` 17 MB.
  Stripping outputs before the public push is worth a decision — against it,
  outputs are the only record of results for cells that cannot currently rerun.
- **`results/` holds 3,754 files, 3,686 of them SVG or PDF** — the overwhelming
  majority per-subject and per-session diagnostics: `2P/` 1,909, `RLModel/`
  1,445, `behavior/` 249, `optogenetics/` 109, `tracking/` 28, `WF/` 13. Fork 6
  should decide what ships: only the files that back manuscript panels (a few
  dozen), or the full per-subject set that the README advertises as a feature.
- **`figcode/psychofit/`** is a vendored copy of cortex-lab/psychofit with its
  own `setup.py` and tests; the Methods cite the upstream URL. Either declare it
  a dependency or document the vendoring.
- **`twop/relogit/`** is a second vendored package, with its own
  `requirements.txt` that nothing reads.
- **`conftest.py`** carries a Windows conda DLL workaround derived from
  `sys.prefix`. Once fork 4 lands a single uv environment, check whether it is
  still needed.
- **`Z`-formula divergence**: the MLE and χ² paths compute the DDM starting
  point differently, and the paper states only the MLE form. Tracked in
  `rlmodel/methods_model_revision.md`; the fix belongs in `model/bias.py`.
  *This one changes a published number if the χ² path is corrected — decide
  before touching it.*

---

## Suggested order of work

The forks are not independent. A workable sequence:

1. **Fork 4 (unified uv)** first — everything else is easier once one
   interpreter reads every artifact. The blocking change is small: stop
   pickling pandas `string` dtype.
2. **Fork 3 (behaviour tests)** next, extracting Figures 2A and 2B from
   `behavior.ipynb` into modules as the first two targets. They are
   Methods-documented, they carry quoted numbers, and 2B is triplicated.
3. **Fork 2 (trim the model)** and **fork 1 (model docs)** together — deleting
   the `Decay Q` family and the unused bias/noise functions makes the
   documentation shorter, and the equations in
   [`manuscript-methods-map.md`](manuscript-methods-map.md#the-model) are the
   source of truth for what must stay.
4. **Fork 5 (2P reorganisation)** — largest surface. Start by deciding the fate
   of the orphaned `twop/plot/stats*` modules.
5. **Fork 6 (cleanup)** and **fork 7 (papermill)** last, once the notebooks are
   thin enough to run end-to-end.
