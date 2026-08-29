# Centralize model evaluation + multi-eval aggregates

## Context

`rlmodel/model_analysis.ipynb` builds its headline panel — **`# Plot Aggregates` → `## Fig. 1l`** (cell 24, `plotAggregates`) — from ~350 lines of in-cell code: `getFitDict` (cell 14), `runSubjectData` (cell 14), `collectMetric` (cell 20), `plotAggregates` (cell 24). None of it is importable, testable, or reusable, and the model set is a hardcoded `MODELS` list.

Two problems make this urgent:

1. **The notebook cannot run today.** `getFitDict` builds the key `f"{drift}_bias{bias}_{noise}_{t_dur}s_dt{dt}.pkl"` with `t_dur=3`. Those six 3s pickles are deleted from the worktree (still in git HEAD). The fits on disk are now `chisq_*` / `mle_*` at **4.8s**, written by `fit.evolveFP`.
2. **A parallel, better controller already exists.** `model/compare.py` + `model/mle_reeval.py` already parse those filenames into a structured identity and run the forward pass generically. `model_analysis.ipynb` predates it and duplicates it worse.

Goal: extract Fig. 1l's analysis into importable modules built on the *existing* `compare.py` controller, add an optional per-subject repeat-evaluation count, then express the two new model comparisons as pure configuration.

### Key finding: one coordinate system already covers every requested figure

`mle_reeval.parse_fit_filename` → `FitFileId` (drift/bias/noise/t_dur/dt/scaled_bound/weights) and `compare.classify_column` together map each file to **`(model_key, column_label)`**. I ran this over `data/RLModel/` — every requested figure is already a *selection* in that space, no new parsing needed:

| Requested | `model_key` | `column_label` | File on disk |
|---|---|---|---|
| Fig 1l · Classic | `Classic\|None_\|…` | `Chi²-Noise` | `chisq_Classic_biasNone_…4.8s_dt0.005.pkl` |
| Fig 1l · Q-value | `Classic\|Q-Val (Offset)\|…` | `Chi²-Noise` | `chisq_Classic_biasQ-Val (Offset)_…` |
| Fig 1l · RewardRate | `RewardRate\|None_\|…` | `Chi²-Noise` | `chisq_NoiseGain-RewardRate_biasNone_…` |
| Fig 1l · RR + Q-value | `RewardRate\|Q-Val (Offset)\|…` | `Chi²-Noise` | `chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_…` |
| New A·1 RR (Noise)/Fixed Thr. | `RewardRate\|None_\|…` | `Chi²-Noise` | `chisq_NoiseGain-RewardRate_biasNone_…` |
| New A·2 RR (Scale-Bound)/Fixed Noise | `RewardRate\|None_\|…` | `Chi²-Bound` | `chisq_Bound-RewardRate_biasNone_…_scaledB.pkl` |
| New A·3 = A·1 + Q-Val | `RewardRate\|Q-Val (Offset)\|…` | `Chi²-Noise` | `chisq_NoiseGain-RewardRate_biasQ-Val (Offset)_…` |
| New A·4 = A·2 + Q-Val | `RewardRate\|Q-Val (Offset)\|…` | `Chi²-Bound` | `chisq_Bound-RewardRate_biasQ-Val (Offset)_…_scaledB.pkl` |
| New B·1 Pure MLE | `RewardRate\|Q-Val (Offset)\|…` | `MLE` | `mle_NoiseGain-RewardRate_biasQ-Val (Offset)_…pkl` |
| New B·2 MLE + Chi²-0.1 | `RewardRate\|Q-Val (Offset)\|…` | `MLE=1, Chi²=0.1` | `…_mleW1_chi2W0.1.pkl` |
| New B·3 MLE + Chi²-0.5 | `RewardRate\|Q-Val (Offset)\|…` | `MLE=1, Chi²=0.5` | `…_mleW1_chi2W0.5.pkl` |

**Follow-up (drift reward-rate channel).** A third reward-rate channel —
`r` modulating the *drift* (`--use-drift-rr`, `μ = V·DV·(2−r)`) — extends the
same coordinate system with no parsing changes. Because it is a different
model rather than a different scale axis, `drift.display_alias_for_drift` gives
it its own `drift_alias`, hence its own `model_key` row:

| Figure bar | `model_key` | `column_label` | File on disk |
|---|---|---|---|
| RR (Drift 2-r) | `RewardRate (Drift)\|None_\|…` | `Chi²-Noise` | `chisq_DriftGain-RewardRate_biasNone_…` |
| RR (Drift 2-r) + Q-Val | `RewardRate (Drift)\|Q-Val (Offset)\|…` | `Chi²-Noise` | `chisq_DriftGain-RewardRate_biasQ-Val (Offset)_…` |
| RR (Drift 1+r) | `RewardRate (Drift 1+r)\|None_\|…` | `Chi²-Noise` | `chisq_DriftGain(1+r)-RewardRate_biasNone_…` |
| RR (Drift 1+r) + Q-Val | `RewardRate (Drift 1+r)\|Q-Val (Offset)\|…` | `Chi²-Noise` | `chisq_DriftGain(1+r)-RewardRate_biasQ-Val (Offset)_…` |

Each mapping's two scale axes still share one `model_key` and split by column,
exactly like the noise/bound pair (`…_scaledB.pkl` ⇒ `Chi²-Bound`); the two
mappings themselves are separate models. Expressed as
`aggregate.DRIFT_RR_SPECS` (8 bars = 4 channels × ±Q-Val, labels abbreviated
to `RR` so they don't collide, and `figsize=(20, 8)` for the same reason) plus
its own notebook cell, per the "add two cells" pattern below — `FIG1L_SPECS`
and `SCALE_BOUND_SPECS` are untouched.

All 11 files exist and classify cleanly. `Chi²-Noise` == non-`_scaledB` == fitted noise / fixed threshold, which matches "all reward-rate used are Noise based, and in Classic we also use the noise-based fixed-threshold version".

### Confirmed decisions

- Golden-test the refactor against the notebook's **original in-cell code on the new `chisq_*` 4.8s fits** (Fig. 1l numbers will differ from the published 3s panel — the 3s inputs are gone).
- Multi-eval stats: **dot = subject's mean ± STD over its N evals; bar = mean ± SEM over the per-subject means (n = #subjects)**.
- New comparisons: **one Fig-1l-style axes, grouped bars, half-column gap**.
- Metrics: **`R2_Psych` + Reward-Rate correlation**, same pair as Fig. 1l.

---

## Implementation

### 1. Thread `seed` through the simulation (enables multi-eval)

`model/logic.py:466` `makeOneRun(..., seed=0)` already seeds every RNG (`logic.py:547-550`, fanned out to `bias`/`drift`/`noise.rnd_default_rng`). But `plotter.runAndPlot` (`model/plotter.py:38`) **never accepts or forwards it**, so every run is seed 0.

- `model/plotter.py`: add `seed=0` to `runAndPlot`'s signature; forward `seed=seed` to both `makeOneRun` call sites (the `PROFILE` branch at :87 and the normal one at :100).
- `model/compare.py`: add `seed=0` to `_compute_sim` (:264) and pass it to `runAndPlot`.

Both default to `0` → every existing caller is unchanged. Seeding convention: `num_evaluations == 1` → `seed=0` (the default); `> 1` → `seed = 0 + i` for `i` in `0..N-1` (so iteration 0 reproduces the single-eval result exactly).

### 2. New module `model/aggregate.py` — the one controller

Reuses, does not reimplement: `compare.discover_fits`, `compare.classify_column`, `compare._compute_sim`, `compare.prepare_behavior_df`, `mle_reeval.FitFileId`.

**Spec = one bar group** (replaces the hardcoded `MODELS` list):

```python
@dataclass(frozen=True)
class EvalSpec:
    label: str          # "RewardRate\n+ Q-Val"
    model_key: str      # FitFileId.model_key
    column_label: str   # "Chi²-Noise" | "Chi²-Bound" | "MLE" | "MLE=1, Chi²=0.1"
    color: str
    gap_after: float = 0.0   # extra x-space after this group (the ½ column)
```

Presets built from the table above: `FIG1L_SPECS`, `SCALE_BOUND_SPECS` (`gap_after=0.5` on entry 2), `MLE_WEIGHT_SPECS`. A `resolve_spec(fits, spec)` helper raises an actionable error listing available `(model_key, column_label)` pairs when a spec matches nothing — the failure mode that matters when a fit hasn't been run yet.

**Metrics** — lift cell 20's `collectMetric` body verbatim into module-level functions, preserving the math:
- `_calcR2`, `_prevOutcomeCountRT`, `_fitData` → private module fns (currently nested in the cell).
- `subject_metrics(sim_df) -> dict` — every metric for ONE sim df. Keeps reusing `plotter.calcWinLoseUpdates`, `plotter._dvQuantileFn`, `behavior.bias.calcBias`, `figcode.prevoutcomecurquantile.quantilePrevOutcomeCur`, `figcode.psychometric._psychFitBasic`, `behavior.rewardrate.plotSubjectRewardRateRt`.
- Adds a scalar `RewardRateCorr` (the per-row `row_real.corr(row_model)` that cell 24 computes inline), while keeping `RewardRate5Real`/`Model` object columns so the other cells keep working.

**Collection** returns a **tidy DataFrame**, one row per `(spec, subject, iteration)`:

```python
def collect_metrics(fits, specs, df_behavior, *, num_evaluations=1,
                    min_num_trials=2_500, sim_cache=None) -> pd.DataFrame
# columns: SpecLabel, ModelKey, ColumnLabel, Name, Iteration, Seed,
#          NumTrials, R2_Psych, RewardRateCorr, R2_WinLose, … (all of collectMetric's keys)
```

**Memory (important):** the notebook's `cached_processed_df` retains every `fitted_df`; at N evals that is N× full trial-level frames per subject×model and will not fit. So: reduce each iteration to its metric row immediately and **retain only the `seed=0` sim df** into `sim_cache` — that is the one the downstream cells (`loopModelSubjectsPlotPrevOutcomeCount` cell 29, `loopModels` cell 32) consume, in the existing `{comb: {subject: (loss, fitted_df, BOUND, include_Q, include_RewardRate)}}` shape.

**Cost:** N × #specs × #subjects full DDM forward passes. Worth stating in the notebook markdown.

### 3. New module `model/aggregate_plot.py` — Fig-1l-style bars

`plot_aggregates(metrics_df, specs, *, metric_keys=("R2_Psych", "RewardRateCorr"), ax=None)` — ports cell 24's layout (grouped bars, per-group label annotation, IQR/OOB reporting, `axhline(0)`, despined, `ylim(…, 1.3)`), with two changes:

- **x layout** honours `spec.gap_after`, generalizing cell 24's hardcoded `global_offset_x += 3`.
- **multi-eval rendering**, keyed off `metrics_df.Iteration.nunique()`:
  - `N == 1` → dots all at the group's exact x, no per-dot error bar (identical to today).
  - `N > 1` → per-subject means spread evenly across the bar width (`width=0.6`) from start to end via `np.linspace`, each with an `ax.errorbar` STD whisker.
  - Bar height/whisker always = mean / `scipy.stats.sem` over the **per-subject means**.
  - Title/legend states the convention explicitly, e.g. `"bar: mean ± SEM across subjects (n=9) · dot: subject mean ± STD across 20 evals"`.

### 4. Notebook `model_analysis.ipynb`

- Cell 9 `MODELS` → a spec preset import; cells 14/20/24 collapse to thin calls (matching `model_compare.ipynb`'s established thin-notebook pattern: load, set config globals, call entry points).
- Add `NUM_EVALUATIONS = 1` and the existing `save_figs` flag as config globals.
- Add two cells for the new figures — `SCALE_BOUND_SPECS` and `MLE_WEIGHT_SPECS` — each a `collect_metrics` + `plot_aggregates` pair. They live here because this is Fig. 1l's home and they share its collection path.
- Cells 29/32 keep working off `sim_cache`.

---

## Verification

**Environment split (discovered, and it constrains the plan):** the real fit pickles embed `subject_df` (`rlmodel/README.md:218`) — which is why they're 30–300 MB and why they **fail to unpickle under `uv`'s pandas 2.3.3** (`NotImplementedError: (string[python], …)`, a newer-pandas dtype). `compare.discover_fits` swallows this and silently skips them. Therefore:

- **Unit tests (`uv run pytest`)** must use synthetic payloads — exactly what `model/tests/test_model_compare.py` already does (`tmp_path` + pickled dicts + monkeypatched forward passes + `Agg`).
- **Real-data checks** run in the notebook (conda py312) environment.

**New `model/tests/test_aggregate.py`** (mirrors `test_model_compare.py`'s conventions):
- `runAndPlot` forwards `seed` to `makeOneRun`; `N=1` → `[0]`, `N=3` → `[0,1,2]` (monkeypatch `makeOneRun`, assert the seeds it receives).
- Iteration-0 metrics with `N=3` equal the `N=1` metrics (the seeding convention's contract).
- Stats: dot = mean±STD per subject, bar = mean±SEM over subject means, on a hand-built metrics frame with known values.
- `gap_after` shifts subsequent groups' bar x-positions by the expected amount.
- `resolve_spec` raises a listing error for an absent `(model_key, column_label)`.
- `sim_cache` retains only iteration 0.

**Golden test (notebook env) — the "same results" check:**
1. Add a scratch cell running the notebook's **original** `getFitDict`/`runSubjectData`/`collectMetric`/`plotAggregates` against the new `chisq_*` 4.8s fits, patching only the filename key (use `fit.evolveFP(..., fit_mode="chisq")` instead of the stale 3s f-string). Dump `processed_df_dict` → `baseline_fig1l.pkl`.
2. Assert `aggregate.collect_metrics(FIG1L_SPECS, num_evaluations=1)` reproduces it via `pd.testing.assert_frame_equal`, and that the regenerated Fig. 1l matches visually.

**Two known fidelity risks to resolve at step 1** (both are why the extraction is worth doing, and both may force the patch to exceed "just the filename key" — I could not settle them under `uv` because the pickles won't load):
- `runSubjectData` reads `params_dict["BOUND"]` unconditionally, but `compare._routed_kwargs` (`compare.py:230`) documents that the frozen scale axis is **absent** from a fit — `BOUND` for a noise-scaled fit, `NOISE_SIGMA` for a `_scaledB` one — and defaults it to `1.0`. If the new `chisq_*` params_names omit `BOUND`, the original `runSubjectData` `KeyError`s and the baseline cell must adopt the same `1.0` fallback. **First step: print `params_names` for one new chisq fit in the notebook.**
- `runSubjectData` re-applies `_extendTrials`/`_reduceDFSize` per subject and sets `SessId = Date_SessionNum`, whereas `prepare_behavior_df` (`mle_reeval.py:238`) does it once globally with `SessId = Name_Date_SessionNum`. Equivalent within a per-subject filter, but confirm the sim dfs match before trusting the golden numbers.

**Also:** `_psychFitBasic` uses `psychofit.mle_fit_psycho(nfits=20)` multi-start, which is not covered by our `seed` — a small nondeterminism source independent of the sim seed. Worth knowing if repeat evals show a variance floor.
