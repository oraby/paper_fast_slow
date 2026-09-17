# Execution plan — dependency analysis and suggested order

Dependency analysis of the eight cleanup workstreams, ordered so that forks can
run concurrently without colliding. Findings are from
[`repo-audit.md`](repo-audit.md) plus the artifact-portability probe recorded
below.

The eight workstreams as stated:

| # | Workstream | Referred to below as |
|---|---|---|
| 1 | Improve the model documentation | **B3** |
| 2 | Remove model parts not used in the manuscript | **B2** |
| 3 | Unit tests for the behaviour code | **C** |
| 4 | One unified `uv` environment | **A** (+ **G0.5**) |
| 5 | Reorganise the 2-photon notebooks and backend | **D** |
| 6 | Remove unused code and non-manuscript figures | folded into C/D/**F** |
| 7 | Notebook parameters for automated execution | **E** |
| 8 | General best-practice / `CLAUDE.md` conformance | folded into every task |

## Status board — 2026-09-10

| | Workstream | Status |
|---|---|---|
| **G0** | Trustworthy baseline | **done** |
| **A** | Unified `uv` | **done** — everything runs from the lockfile; guarded by `test_environment.py` |
| **B** | Model trim + docs | **DONE (B1–B3)** (2026-09-17) — `rlmodel/README.md` corrected against the code and then rewritten to carry the model's equations; every registry entry the manuscript does not use is gone, verified fit-for-fit against the shipped pickles |
| **C** | Behaviour tests | **Done, C1–C8.** Every extracted panel reproduces its committed figure and is tested; `figcode/`, `opto/` and `tracking/` went 0 → 112, 51 and 124 tests. C8 extracted `Tracking.ipynb`'s preprocessing (identical output on all 76,311 frames) and fixed two silent breakages: pandas 3 copy-on-write and a machine-time-zone dependence. S3J–M legend/Methods text is drafted (`methods_model_revision.md` Blocks 10–12); S3K is regenerated choice-normalised next revision; interpolation wording (#13) is open |
| **D** | 2-photon reorg | **DONE (D0-D4)** — all five 2P notebooks run top to bottom, 0 errors, writing nothing (`SAVE_FIGS`/`SAVE_DATA`); every listed panel is an extracted, tested module, verified figure-for-figure against pre-extraction runs; every load goes through `twop/dataload.py`. Two items are deliberately left for the next revision (unseeded permutation panels; the 15-row difference in the shipped filtered frame) — see [`repo-audit.md`](repo-audit.md) |
| **E** | Runner / papermill | **done**, one open item — all 11 figure notebooks parameterised, `code/run_notebooks.py`, cells tagged `paper-figure`/`per-subject`; `model_to_behavior.ipynb` still needs ~320 GiB for Figure 7D (see the E section) |
| **F** | Final cleanup | **not started** — `data/to_delete/` is still 563 MB |

Suite: **1705 passed, 1 skipped, 0 failed**, and green with
`FutureWarning`/`DeprecationWarning` promoted to errors.

---

## The finding that reshapes the order

The environment problem is **not** confined to workstream 4, and it is not the
pandas-version clash it looked like. Loading every `data/**/*.pkl` with a plain
`pickle.load` — no compat shim, no stubs, nothing on `sys.path` — showed **139
of 270 files (6.6 GB)** would not open on a bare machine.

Two causes, neither of which was on the original list:

| Cause | files | needs |
|---|---|---|
| `pandas.core.indexes.numeric` — `Int64Index`, removed in pandas 2.0 | 123 | pandas' own private compat shim |
| pandas `string` extension array, conda-built vs PyPI-built | 16 | conda **py312** |

plus four embedded foreign classes (`RunData`, `caiman…States`, `mat_struct`,
and the model's own bias/drift/noise functions).

`pd.read_pickle` already rescues the first group — it applies
`pandas.compat.pickle_compat`, which a plain `pickle.Unpickler` subclass does
not. So only 16 files genuinely need an old interpreter, and the `wfield` conda
env is not needed at all.

That shim is private and temporary, though ("Support pre-0.12 series pickle
compatibility"), so **G0 migrated all 270 files to a dependency-free payload**
rather than leaning on it. See [`data-portability.md`](data-portability.md).

**Consequence for ordering:** this was the shared prerequisite for D, E and A,
and it is now done — so D can start immediately and A no longer carries the
data problem.

### Second correction: widefield — **resolved**

`widefield/pipelineprocessors.py` called `wfield.utils.reconstruct` for the
pixel-wise maps (Figures 3B, S5C), and `wfield` is not on PyPI — so this looked
like it forced a git dependency or an optional extra.

It did not: the whole of what this project used is one eight-line pure-numpy
function. It is now vendored into `widefield/svdreconstruct.py`, byte-identical
to upstream, with provenance and the GPLv3 note recorded. No extra, no git
dependency, and no part of widefield is now outside the `uv` environment.

---

## Dependency graph

```
                    ┌──────────────────────────────────────┐
                    │  G0  Trustworthy baseline      (S)   │
                    │  green suite + shared loader         │
                    └───┬──────────┬──────────┬────────────┘
                        │          │          │
          ┌─────────────┘          │          └──────────────┐
          ▼                        ▼                         ▼
  ┌───────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ A  unified uv │      │ B  model trim    │      │ C  behaviour     │
  │           (M) │      │    + docs    (M) │      │    tests     (M) │
  └───────┬───────┘      └──────────────────┘      └────────┬─────────┘
          │                                                 │
          │                            (pattern, not a gate)│
          │                                                 ▼
          │                                    ┌──────────────────────┐
          │                                    │ D  2-photon reorg    │
          │                                    │                (XL)  │
          │                                    └───────────┬──────────┘
          │                                                │
          └────────────────────┬───────────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ E  runner / papermill│
                    │                  (M) │
                    └───────────┬──────────┘
                                ▼
                    ┌──────────────────────┐
                    │ F  final cleanup     │
                    │    + publish   (S–M) │
                    └──────────────────────┘
```

**Critical path: G0 → D → E → F**, with A joining before E. Its length is
dominated by **D**, which is the only XL item.

`B` is off the critical path entirely — it can start at G0 and finish whenever.

---

## G0 — Trustworthy baseline — **DONE** (2026-08-31)

Nothing downstream was measurable until the suite was green and complete,
because you could not otherwise tell a refactor's breakage from the failures
already present.

| # | Item | Outcome |
|---|---|---|
| 1 | Fix the failing tests | 6 fixed upstream; the 7th needed `plotRegionBars` to take an `ext` parameter (it hardcoded `.pdf`), now defaulting to `svg` |
| 2 | `code/twop/tests/__init__.py` | added — it was the only test package without one |
| 3 | Collect `code/pipeline/tests` | added to `testpaths`, plus its `__init__.py`; 2 tests that had never run now do |
| 4 | Literal save-flags in `behavior.ipynb` | all 11 routed through `global_save_figs`, and 4 stale `#global_save_figs` comments removed. Diff is exactly 11 lines |
| 5 | Shared unpickler | superseded — became the one-off migration below |
| 6 | `behavior_v2.ipynb` | deleted |

**Suite: 865 passed, 1 skipped, 0 failed** (was 811 passed / 1 failed at the
start of G0; 835 when G0 closed).

### Item 5 became a migration, not a shim

A runtime shim would have left every artifact still tied to one interpreter.
The survey (below) showed the defect is worse than a single missing module, so
G0 instead produced a **one-off conversion to a dependency-free payload**:

- [`code/util/portablepickle.py`](../code/util/portablepickle.py) —
  `toPortable` / `fromPortable` decompose frames into plain numpy arrays plus
  builtins; `LegacyUnpickler` reads the old files despite their dangling
  imports; `_importSuffix` resolves repo-local names whatever prefix they were
  pickled under. 21 round-trip tests.
- [`code/util/migratepickles.py`](../code/util/migratepickles.py) — the driver.
  Reads originals in whichever interpreter can read them, writes a mirror tree,
  and **never modifies the originals**.

**Run and verified:** 270/270 migrated to `data_portable/`, all loading under
`uv` and referencing only stdlib + numpy. The 139 not-already-portable
originals are backed up in `data_bak/`.

Full detail, including the survey and the two-stage procedure, is in
[`data-portability.md`](data-portability.md).

---

## A — Unified `uv` environment — **DONE** (2026-09-06)

| | Item | Outcome |
|---|---|---|
| A1 | Fit payloads stop naming this package | `rlmodel/model/fitio.py` stores the bias/drift/noise functions by registry key and `MLEModelConfig` as a dict; `twop/genrundata.saveRunDataDict` / `loadRunDataDict` do the same for `RunData` |
| A2 | Declare the undeclared dependencies | eight added to `pyproject.toml`, each annotated with the module that needs it |
| A3 | The `wfield` dependency | the one function used (8 lines of numpy) vendored into `widefield/svdreconstruct.py`, with provenance and the GPLv3 note |
| A4 | Retire conda | `conftest.py`'s DLL workaround removed — verified inert under `uv`, since the four conda directories it looked for do not exist in `.venv`. Stale "run this in the conda env" instructions in `golden_fig1l.py`, `metrics_runner.py` and `rlmodel/README.md` corrected; they stopped being true once the fits became portable |
| A5 | `code/README.md` | dependency section rewritten for `uv`; the notebook list corrected (it named `TwoPAnalysis.ipynb`, which does not exist, and omitted eight notebooks) |
| — | `cupy` | left undeclared, deliberately: it is imported lazily and callers fall back to NumPy, and the correct wheel name is CUDA-specific. Documented in `code/README.md` instead |

**Verified, not assumed.** All 170 modules under `code/` import in one `uv`
process, and every absolute import in every notebook cell resolves from the
locked environment — the only exception is `psychofit` in the vendored
library's own upstream demo notebook.

`code/util/tests/test_environment.py` now guards both properties, so a
dependency that is used but not declared fails the suite instead of working by
accident on one machine. It would have caught all eight of the undeclared ones.

Suite: **1045 passed, 1 skipped**.

**One deliberate exception.** The Slurm launcher (`rlmodel/slurm/`,
`model/metrics_shards.py`) still activates conda on the compute nodes. That is
a property of the cluster, not of this repo, and is out of scope here.

## B — Model trim and documentation *(medium; fully parallel, off critical path)*

Order **within** the workstream matters:

- **B1 first** — correct the *wrong* statements in `rlmodel/README.md`
  independently of any deletion: the χ² section says "maximum allowed duration
  (3s)" where the code and paper both say 4.8 s; every figure reference uses the
  oldest numbering (`Fig. 1l` → Figure 2G, `Fig. 5f middle` → Figure 7D); and
  MLE and joint MLE+χ² fitting are not documented at all despite producing
  Figures 7A–B and S14B–H.
- **B2** — delete the unused registry entries: 5 bias functions, 1 noise
  function, the 10-entry `Decay Q` drift family. **Done** — see below.
- **B3** — rewrite the model documentation against
  [`manuscript-methods-map.md`](manuscript-methods-map.md#the-model), which
  carries every equation as the paper states it. **Done** — see below.

**B3 must follow B2**, otherwise you write documentation for code you are about
to delete.

### B1 — done (2026-09-17)

Every claim in `rlmodel/README.md` was checked against the code and the saved
fits, not against the older docs:

| Was | Now |
|---|---|
| χ² "maximum allowed duration (3s)"; a bin "below the 0.1 quantile" | 4.8 s (`initvals.T_dur`); the first bin ends just above the fastest trial, seven bins per condition (`logic.chi2Loss`) |
| `Fig. 1l`, `Fig. 5f middle`, no figure numbers elsewhere | Figures 2E, 2F, 2G, S4, S14B and 7D; `fig1l` explained as a preset name |
| MLE mentioned only as something "we are open to adopting" | New "Fitting criteria" section: χ², MLE and joint, with flags, filenames and the figures each feeds |
| `python model_runner.py --help` (fails: relative import) | `uv run python -m code.rlmodel.model_runner --help`, run from the root |
| Q-value "i.e. log(Q_left/Q_right)" | the clipped log ratio ÷ log(100), in [−1, 1] (`state_updates.compute_q_value`) |
| Q/RR updates "in `logic.py`" | in `state_updates.py`; `logic.py` holds thin wrappers |
| Model table without registry keys | adds the keys (`Q-Val (Offset)`, `NoiseGain-RewardRate`, …) |
| Fit pickles: functions and `OptimRes` as objects, `/data/rlmodel/` | stored by registry key / as dicts, `fitio.loadFit` / `saveFit`, the trace stripped, `/data/RLModel/`; MLE and joint payload keys added |
| "Subjects with fewer than 2,500 trials are excluded" | all 22 are fitted; the threshold applies at analysis, leaving 9 (checked on the saved fits) |
| Padding trials "do not affect the Q-value updates" | they do update, but only after a session's last real trial |
| — | the Z-formula divergence stated, pointing to `methods_model_revision.md` |

The same stale description of the joint loss — each term divided by its
valid-trial count, and the weights "not in the filename" — was in
`model_runner.py`'s `--help`, `MLEModelConfig`'s comment and
`test_joint_loss.py`'s docstring. The code divides by the subject's reference
losses and suffixes joint filenames with the weights, and its own tests check
exactly that; the three texts now say so. No behaviour changed.

**The two gates were decided on 2026-09-17**, and both were decisions rather
than work:

- **The asymmetric-learning-rate experiment** goes, together with `Decay Q` —
  but only after being made to work, so the version left in history is a
  recoverable one.
- **The Z-formula divergence** is deferred to the χ²/joint refit. B2 therefore
  edited `bias.py` only to delete entries; `_biasQVal`'s formula is untouched,
  so the code still reproduces the shipped fits and Figures 2E–G / S4.

### B2 — done (2026-09-17)

Two commits, deliberately kept apart:

1. **`67ef1ac` — make the unpublished variants work first.** Three defects meant
   several of them could not be fitted under χ² at all: `util.decayingQ` took
   `np.arange` of the noise array's `(trials, steps)` shape and indexed its
   scalar rate, so every `Decaying Q-Val` evaluation raised; that noise drew
   from the global `np.random` instead of the seeded generator, so the loss
   changed between two evaluations of the same parameters; and the
   `μ, σ (Corr/Incorr)` bias called `_biasMeanDir` without `BIAS_COEF`. A sweep
   of every bias × drift × noise combination (420 under χ², 180 under MLE) is
   what found them.
2. **`417629e` — delete them.** The registry entries, their `InitVals` fields,
   the whole asymmetric-rate path (flags, config fields, GUI controls, filename
   suffix, design document), and the Decay-Q-only compute paths: the factored
   time-varying mu in `mle.py` / `mle_batch.py`, which leaves the batched
   solver constant-drift only. About 1,900 lines out of `code/rlmodel/`.

**Verification was against the shipped fits, not only the suite.** All 15
`data/RLModel/*.pkl` × 22 subjects were re-simulated and re-scored before and
after: identical χ² loss, identical simulated trials, identical MLE likelihood
(298/298 rows on every column).

That check earned its keep. With the asymmetric rates gone,
`update_q_values` multiplied the float32 Q-state by a plain Python `alpha`, so
the update ran in float32 where the removed `xp.where(…, alpha_unrewarded,
alpha)` had promoted it to float64. The χ² loss was unchanged but the simulated
Q trajectories differed in their last bits — invisible to the tests. `alpha`
and `beta` are now cast explicitly.

### B3 — done (2026-09-17)

`rlmodel/README.md` documented the code's structure but never the model.
`d8a6aec` gives it a "The model" section: the baseline DDM (accumulation,
bounds, how the non-decision time is applied, what makes a no-choice trial),
the Q-learning bias (update, normalised log ratio, starting point), the
R-learning noise gain, and the two combined — each equation with the parameters
it introduces and the code implementing it, checked against the source rather
than the older docs.

Three things were previously discoverable only by reading the code:

- under χ², the Q and reward-rate updates learn from the model's **own**
  simulated outcome, and a simulated response faster than 0.3 s is forced to
  unrewarded (`logic.FORCE_EWD`);
- `BOUND` and `NOISE_SIGMA` are a scale pair of which exactly one is ever
  fitted, `--scale-bound` choosing which;
- a table of every fittable parameter with its range and initial value.

**One bug found on the way, fixed separately (`2585ced`).** G0's portability
rewrite stored every fit's `OptimRes` as a plain dict and `loadFit` never
rebuilt it, so on the shipped fits re-simulating or re-scoring raised
`AttributeError`, `getattr(OptimRes, "fun", nan)` readers recorded NaN in
silence, and a joint MLE+χ² fit could not start at all. The tests all used
`SimpleNamespace` stand-ins, which is why none of them saw it.

---

## C — Behaviour extraction and tests *(medium; parallel with A and B)*

Targets in priority order, highest-value first:

- ~~**C1** Figure 2A — variance explained.~~ **Done**: `behavior/varexplained.py`,
  21 tests. Reproduces the four published bars, n=9, the 101.9% total and the
  14.3–22.2 condition numbers exactly.
- ~~**C2** Figure 2B — optimal sampling.~~ **Done**: `behavior/optimalsampling.py`,
  27 tests. Three duplicated copies collapsed to one (notebook 13.25 → 9.26 MB);
  all 17 published Δ-positions reproduce to 0.00 s. Dropped along the way: the
  two dead copies, the non-manuscript `plotMetrics3D` (`3d_plot.svg`) and
  `results/optimal_sampling_time.svg`, and a mid-notebook `%matplotlib widget`
  switch that would have broken any headless runner. **Fixed a published-figure
  bug** — see the audit.
- ~~**C3** Figure 1I-right, S3G, S2B, S2M.~~ **Done.** 1I-right and S3G to
  `behavior/stayswitchupdate.py` (25 tests, exclusion rule now derived);
  S2M to `behavior/fastslowperf.py` (15 tests, all three published Holm
  p-values reproduced); S2B to `behavior/stdistribution.py` (15 tests) —
  that cell did not run under the locked pandas, and its normalisation is
  now per-subject across contexts, reproducing the published counts.
- ~~**C4** Delete `figcode/prevoutcomecurquantile_{bak,new}.py`.~~ **Done**:
  407 lines. Git history confirmed the lineage (all three added together, only
  the live one updated since) and neither copy had an importer.
- ~~**C5** First tests for `figcode/` and `opto/`.~~ **Done.** 114 tests across
  six new files; both packages added to `testpaths`.
  - `opto/tests/test_bootstrapping.py` (16) — Figure 3D. Pins the three
    resampling levels, and pins *why* the nesting is there: the same trials
    nested under six animals give a bootstrap spread more than twice the flat
    one, and quadrupling trials per animal halves the flat spread while barely
    moving the clustered one.
  - `opto/tests/test_bootstrap2regions.py` (34) — Figures 4C, S6G. Sign-rule
    p-values, the two Holm families, stratified trial resampling, and the
    subject-weighted-vs-trial-pooled difference from `bootstrapPerf`.
  - `figcode/tests/` — `test_util.py` (13), `test_psychometric.py` (18),
    `test_stayswitch.py` (17), `test_prevoutcomecurquantile.py` (16).

  Side effects, all verified inert against saved before/after output: three
  pandas-3 deprecation sites fixed (the suite now raises no `FutureWarning`),
  and `bootstrapping.py`'s `__main__` demo removed — it was broken three ways
  and could never have run. Findings, including a *wrong* unused
  Benjamini–Hochberg helper and the two different hierarchical-bootstrap
  estimators, are in [`repo-audit.md`](repo-audit.md).

  **Not covered:** see C6 below, plus several `groupby(...).apply` sites in
  `opto/optoprocessor.py` and `optoreactiontime.py` that still warn.

- ~~**C6** Tests for the two remaining untested `figcode/` figure producers.~~
  **Done.** 50 tests — `test_stbydifficulty.py` (28; **Figures 1D, 1G**,
  S2E–F, S3A) and `test_stheatmap.py` (22; Figure S3E and the S3F
  statistics). `figcode/` is now 114 tests across 6 files.

  **Two published panels could not be regenerated at all**, and both are now
  fixed and verified against the committed figures — Figure 1D on
  `matplotlib.cm.get_cmap` (removed in 3.9) and Figure S3E on pandas ≥ 2
  returning a 1-tuple from `groupby(["one_col"])`. Same shape as the Figure
  S2B breakage in C3: a legacy API on a path nothing exercised. Full account,
  including three further `stheatmap` defects pinned but deliberately not
  fixed (a `mode` option that cannot run, a sort key that ignores its
  argument, a stray debug print), in [`repo-audit.md`](repo-audit.md).

  Pinned beyond the obvious: the fast/typical/slow band edges are **per-animal
  tertiles averaged across animals**, not tertiles of the pooled trials;
  Figure 1G's slope/θ aggregates per animal for the cohort panel but per
  session for a single animal, and the reported slope is **negated** because x
  runs Hard→Easy. All three are easy to "simplify" wrongly.

- ~~**C7** Extract `Tracking.ipynb` (**Figures S3J–M**).~~ **Done.** The four
  panels are in `code/tracking/` — `centroids.py` (S3J–L) and `strategy.py`
  (S3M) — with 49 tests. The notebook's figure cells went **302 → 25 lines**.

  Both figure functions read `save_prefix` as a **free variable** from the
  notebook's globals, so either raised `NameError` the moment it was called
  with `save_fig=True` from anywhere else. That is the same class of latent
  breakage C6 found, and it is why "it looks maintained" is not evidence.

  Verified three ways before anything changed: bar geometry identical to the
  notebook's own function on all 76,311 tracked frames; all four SVGs
  text-identical to the committed ones; and Figure S3M's Holm-corrected
  p-values reproducing exactly (0.073 / 0.722 / 0.105 / 0.722 → the published
  **0/4 significant**).

  **Not extracted, deliberately:** the ~370-line preprocessing chain (time
  sync → rotation → interpolation, ending at `df_track_centroid`) and ~295
  lines of video/AVI tooling that no figure uses. The preprocessing is the
  natural follow-on; the video tooling is a keep/delete call for fork F. See
  [`repo-audit.md`](repo-audit.md).

- **C follow-up: defect decisions (2026-09-14).** Resolved as decided:
  `_bh_fdr`, `_holm_step_down` and `_hl_diff_unpaired` deleted; `stheatmap`'s
  unusable `"mode"` option (and `_stHist`, which existed only for it) deleted;
  both fixed-permutation sort callbacks replaced by label-ordered constants,
  with Figure S3E still text-identical; `bootstrapPerf` gained an optional
  `rng` whose default keeps today's behaviour (there is no implicit seed to
  adopt); `plt.show()` moved out of `prevoutcomecurquantile` to its call site;
  the nine oversized `data/2p` files added to `.gitignore` by name; the empty
  `psychofit-FR03` tree removed. Left and documented: empty coherence bins
  (they turn out not to affect any fit) and the 0.025 / 0.05 star thresholds
  (no reported star depends on the choice). Details in
  [`repo-audit.md`](repo-audit.md).

- ~~**C8** Extract `Tracking.ipynb`'s preprocessing.~~ **Done.** `tracking/sync.py`,
  `rotate.py`, `interpolate.py` and `preprocess.py::buildCentroidFrame`, with 76
  tests; the notebook's preprocessing cells went 437 → 4 lines. Identical to the
  notebook at every stage on all 76,311 frames, with copy-on-write on and off. Fixed
  on the way: the gap filling that pandas 3 would have silently disabled (6,189 of
  11,212 frames changed in a two-session test of the old code) and the dependence
  on the machine's time zone (64 frames matched on a UTC machine, none on US
  Eastern). Only the video/AVI writers remain in the notebook, for fork F.

**Why C should precede D even though it does not block it:** C is the same
refactor (inline → module → test) at roughly one-quarter of D's scale, on a
subtree nobody else touches. Landing it first establishes the module layout,
the fixture style and the figure-regression approach that D then applies to
three much larger notebooks. Starting with D means inventing those conventions
under maximum pressure.

---

## D — Two-photon reorganisation *(extra-large; the critical path)*

Scale: 11,750 lines of notebook code and 143 inline `def`s across
`2pAnalysis.ipynb`, `plottraces3.ipynb`, `TwoPTraces.ipynb` and
`TwoPLoad.ipynb`, backing 20+ published panels.

- **D0 — decision. Resolved 2026-09-06: delete, not adopt.**
  The hypothesis in the audit was that these modules might be a *better*
  factoring of what is inline, making D2 a wiring job rather than an
  extraction. The evidence says otherwise, on five independent counts:

  1. **Nothing imports them.** Seven modules, plus `plotutil.py`, which is
     imported only by three of the seven — so the whole cluster is orphaned,
     not just its leaves.
  2. **They all write `.jpeg`.** `results/` holds 2,729 SVG, 957 PDF, 41 PNG,
     27 CSV — and **zero JPEG**.
  3. **None of their output names exists.** `pie_*_dist`, `pie_*_tuning`,
     `pie_*_comb`, `early_late_sampling`, `sampling_feedback_tuning`,
     `sigf_no_prior`, `prior_cur_overlap` — zero matches anywhere in
     `results/`.
  4. **The inline code demonstrably writes the published artifacts.** The
     notebook's `plotPriorCurrentTuning` writes
     `PriorCurrentTuning/{br}_{epoch}_prior_current_tuning.svg` and its Venn
     cell writes `FastSlowVenn/valid_10%_{br}.svg`; both directories are
     present and populated.
  5. **They contain no analysis the inline code lacks.** No hypothesis test in
     any of them — the only statistical calls are `sem()` for error bars,
     which the inline `_plotBrainRegionTuning` also computes.

  Last touched in the initial "Add Code" commit (2025-12-30) and never since,
  while `corrthreshregions.py` beside them was edited this month. They are a
  superseded JPEG-era generation, not a parallel implementation.

  **Deleted: 1,052 lines.** The suite went 1045 → 1037, exactly the eight
  module-import tests for the removed files and nothing else.

  Consequence for D2: it is an **extraction** job. There is no existing module
  layer to wire up, so the inline panels have to be moved out and tested as
  they go — the pattern workstream C establishes.

- **D1 — done 2026-09-10, and it found more than expected.**

  A static pass over every notebook (bind every assignment / import / def /
  for-target / comprehension / parameter, then look for Names loaded but never
  bound) is the check; it now reports **zero** undefined names in
  `plottraces3.ipynb`, `2pAnalysis.ipynb`, `behavior.ipynb`, `opto.ipynb`,
  `widefield.ipynb`, `Tracking.ipynb` and `2pSeqWithinDeviation.ipynb`.

  Fixed: five dead cells deleted from `plottraces3.ipynb` (each checked against
  the figure map and `results/` first), one partly-dead cell trimmed on the
  S12G path, a commented-out `import seaborn as sns` restored in
  `2pAnalysis.ipynb`, and 14 imports in `TwoPLoad.ipynb` repointed from a
  sibling project's layout to this repo's.

  One flagged cell was *wrongly* flagged: the cell building `res_corr_df` is
  sound in order and load-bearing for Figure 4K right and S10D.

  **Heading names in these notebooks do not indicate what is live.** "Old
  working way" contains the live `plotActivitySum` call; "Continue with old
  code" contains the Figure 4K path. Only dependency and output analysis works.

### The two blockers D1 surfaced — one resolved, one narrowed

Both were references to code absent from this repo. The author's earlier
project (`OneDrive/caiman/`) supplied the answers — and they were different
answers.

**1. `TwoPTraces.ipynb` / Figure 4H — resolved, and not by restoring anything.**

`loopCombinations` is defined nowhere here. It exists in the ancestor notebook
`caiman/TwoP/again/plottraces3.ipynb`, and comparing the two shows it was
*generalised* into `loopHeatmapCombinations` — which sits in the cell directly
above the broken calls. `loopCombinationsTraces` was updated to the new name;
these two call sites were missed.

Calling the successor with its defaults reproduces the ancestor's hardcoded
loops exactly:

| ancestor (hardcoded) | successor (default) |
|---|---|
| `(None, None)` | `sgf_data_cols_comb_li=[(None, None)]` |
| `[False]` | `combin_all_sess_li=[False]` |
| `[(True, True)]` | `heatmap_sorting_options_li=[RESORT_TO_ALL]` |
| `[Combinations.All]` | `combinations_li=[Combinations.All]` |
| `[False, True]` | `heatmap_firing_range_options_li=[SELF_RNG, SHARED_RNG]` |

The last row is the same pair iterated the other way, which changes the order
figures are produced in, not their content. The one parameter lost in the
generalisation, `only_single_sess`, is left at its default by both call sites.

So the fix was a two-word rename, not resurrected code. **Caveat:** this
restores the code path; it does not prove the figure reproduces, which needs
the 2P data and a full run.

**2. `TwoPLoad.ipynb` — repaired; one input is not shipped.**

It was never broken, only moved. Every relative path assumed its original home,
`OneDrive/caiman/TwoP/again/`, and there `pkl/` holds exactly the frames
`data/2p/` later received — `df_all_by_epoch_df_f_filtered.pkl`,
`traces_cut_feedback_filtered_full_df.pkl`, `normed_sampling_by_quantiles2.pkl`,
`svm_df.pkl` and more — several of them written by TwoPLoad itself. So
`code/README.md` is right that it builds frames the other 2P notebooks consume.
An earlier version of this section read the broken paths as evidence of the
opposite; that was wrong.

Re-anchored to `code/`, where every other notebook lives:

- bootstrap `root_parent_level` 2 → 1;
- `../results/opto2P` → `../results/2P`, and `fig_save_prefix` — used by cells
  20 and 41, defined nowhere — set to the same;
- `pkl/X` → `../data/2p/X`, one read and four writes;
- **every write through `savePortable`.** Three are switched off in the
  notebook, but cell 55 is not, and it overwrites
  `data/2p/df_all_by_epoch_df_f_filtered.pkl` — a migrated file in the download
  archive — which its plain `pickle.dump` would have made unportable again,
  silently;
- the raw input shipped as a trimmed, portable `data/2p/df_all_by_epoch.pkl`
  and read with a plain `pd.read_pickle` (see below);
- `shorth` moved to `twop/shorth.py`, with a 500-case equivalence test against
  the ancestor's verbatim code.

**`ref_accepted_traces` was not missing data or a neuron subset.** The
acceptance test changed from peak amplitude —
`np.max(trace) >= ref_accepted_traces`, threshold
`np.median(ref_trace.max_vals)` — to the std criterion the Methods describe,
`trace.std() >= ref_threshold_std`. The old threshold and the old test were
both commented out; only a reference line in a debug plot kept the old name,
inside `if False and (...)` in cells 29 and 32, so it could never execute. That
line is gone; if the plot is re-enabled it still colours accepted traces green.
Cell 29 is the Figure 6B y-axis computation: feedback responses normalised by
the neuron's sampling-epoch statistics and judged against the same threshold.

**The raw input, trimmed (2026-09-14).** The original
`caiman/TwoP/again/pkl/df_all_by_epoch.pkl` (1.17 GB, 33 sessions, 249 columns)
names `caiman`, `Int64Index` and scipy's private `mat_struct`, so a plain
`pickle.load` fails. `data/2p/df_all_by_epoch.pkl` (657 MB, git-ignored, ships
in `2p_data.zip`) keeps the 23 M2/ALM L2/3 sessions — exactly those in
`df_all_by_epoch_df_f_filtered.pkl` — and that file's 30 columns, all 22,483
rows. It names only `builtins`, `datetime`, `numpy` and `pandas`, and loads with
no repo module imported.

The trim was checked by running TwoPLoad's own cells 12, 20, 41, 43, 53 and 54
on both files: identical accepted neurons (1,446), identical rows and columns,
identical traces in every one of the 2,892 arrays. The other sessions and
columns change nothing — cell 20 already restricts to M2/ALM L2/3, and the one
dropped column the pipeline names, `anlys_path`, is read by `CalcBaseline` only
with `track_is_active=True`, which cell 53 does not pass.

Two findings that are *not* about the trim:

- today's environment reproduces the 2024 dF/F to float32 rounding only: 116
  of 2,892 arrays differ, by at most 3e-6 on values around 10 (correlation 1.0);
- the shipped `df_all_by_epoch_df_f_filtered.pkl` has 15 rows fewer than
  TwoPLoad writes (22,468 vs 22,483; traces otherwise identical): 14 trailing
  `Wait Trial Start` rows with no outcome, plus all four epochs of
  `GP4_80_S2_L51_D250_M2m` trial 159. Some step after TwoPLoad dropped them,
  so re-running cell 55 would not reproduce the shipped file exactly.

**Lineage, from a full search of `OneDrive/caiman/`.** `TwoP/again/load.ipynb`
→ `caiman/paper_fast_slow/code/TwoPLoad.ipynb` → this repo's `TwoPLoad.ipynb`.
`shorth` and `_iqrPRCNT` originate in `TwoP/again/ROC_tests_new15_09_23_local.ipynb`
(the "Coped from ROC_tests_new15_09" in cell 20's first line), with copies in
both `TwoP.ipynb`s. `ref_accepted_traces` is commented out in **every**
ancestor, `load.ipynb` included — it has never had a value anywhere. Other
`loopCombinations` definitions (`behavior/evdaccum*.ipynb`,
`evd_conditions_2p.ipynb`) are unrelated functions sharing the name.

The same search re-checked a D1 deletion: `plotSgfActivitySum` is defined in
`TwoP/again/plottraces3.ipynb` — but **commented out there too**, run with
`only_sgf=False` and `save_figs=False`. It never produced the significant-only
sum the Methods use for S12D-bottom / S12J-right; those come from
`2pAnalysis.ipynb` (`TrajectoryTuningPlot`, `movementneurons.py`).

### D1b — every 2P notebook run top to bottom (2026-09-15)

D1's static pass cannot see cell order, calls into changed signatures, or names
bound only inside a function. So each notebook was executed headless
(`nbclient`, the lockfile environment) with a write guard that mirrors every
write under the repo into a sandbox; a before/after snapshot of every repo file
confirmed nothing under `results/` or `data/` changed.

| Notebook | First run | After the fixes |
|---|---|---|
| `2pSeqWithinDeviation.ipynb` | clean, 1.3 min | — |
| `2pAnalysis.ipynb` | cells 0–70 clean; cell 72 passed a 4 h cell timeout | **clean, 6.6 min** |
| `TwoPTraces.ipynb` | 11 failing cells | **clean, 13.4 min** |
| `plottraces3.ipynb` | 11 failing cells | **clean, 8.0 min** |
| `TwoPLoad.ipynb` | 10 failing cells | **clean, 11.4 min** (after the two decisions below) |

Fixed:

- **Half-finished renames** from `825fd7f`. `avg()` in TwoPTraces had moved from
  `only_single_sess` to `plot_only_ids_li`, but `loopHeatmapCombinations` and
  `loopCombinationsTraces` still passed the old names — Figures 4H, 5B and the
  4E/S8B average traces never ran. (D1's "4H restored" was a rename that only a
  run could show was incomplete.) `plotSum()` in plottraces3 had moved from
  `append_start` to `append_where`; the three callers now pass `"end"` for the
  old default and `"start"` for `append_start=True`.
- **Leftovers deleted**: plottraces3 cells reading variables local to
  `runCombinations()`, early copies of two later cells, two displays of columns
  that no longer exist, and a shuffle cell that is commented out in the
  ancestor and read nowhere. TwoPLoad's cell 13 (it inspected a frame before
  it was built).
- **Out of order**: TwoPLoad's feedback-heatmap cell plotted from
  `all_res_unnormed_df` before it existed and read `accepted_sampling_traces`
  from a later cell. Moved after `DecideNeurons` (it is now cell 22), computing
  its accepted traces as its sibling `loopThresholds` already did. **TwoPLoad
  cell numbers above are as renumbered by this change.**
- **Saves that could never succeed**: plottraces3's `plotActivitySum` calls
  lacked the prefix and label it asserts when saving, and `plotSum`,
  `plotActivitySum` and the decoder plots write into `activity_sum/`,
  `activity_sum2/` and `decoders/`, which nothing creates. With
  `SAVE_FIGS = True` (the notebook's default) each would have raised.
- **2pAnalysis shuffles** (cells 71–76): 1,000-iteration nulls, over 6 hours,
  none saved under `data/`, read only by a `printStats` argument that is
  commented out. Behind `RUN_UNUSED_SHUFFLES = False`.

Latent, not fixed: plottraces3's "Take 2" cell redefines `loopNeuronsPlot`,
`plotNeuronCorr` and `_plotFilteredComb`, so `createShuffle` (which needs the
first versions) would break — but it only runs when `rt_corr_shuffled.pkl` is
missing, and that file ships.

Both TwoPLoad decisions were then resolved (2026-09-16), and it too runs clean.

**(a) Figure 6B's missing statistics — recomputed, not ported.** The panel
normalises each feedback response by that neuron's sampling mean and std: the
numbers `NormalizeZScore` computes and this repo's pipeline discards. Diffing
the two `tracesnormalize.py` files, the ancestor's differs in exactly three
ways — it returns those statistics and stores them in a `<set>_stats` column;
it moves two helpers from closures to methods (no behaviour change); and, to
carry the statistics, it unpacks `trace_data, *stats = self._normFunc(...)`.
That last one is a regression: `NormalizePercentile` still returns a bare dict,
which unpacks to its *keys*, so in the ancestor that normaliser is broken on
both paths — and this repo uses it in four places, including the heatmaps
behind 4H, 5B and 6A. Rather than change the shared normaliser, `twop/
zscorestats.py` recomputes the statistics from the same frame — the sampling
epochs cut by `alignSampling` with `NoNormalization`, which is exactly what the
z-score consumed, since `alignSampling` normalises per session *after* cutting
and skips that step entirely for `NoNormalization`. `test_zscorestats.py`
pins it: `(raw - mean) / std` reproduces the pipeline's own z-scored traces,
and the per-trial keying (a repeated trial number replaces the earlier segment)
matches the normaliser's dict. The cell now reports 52% of MFC and 25.9% of LFC
neurons as feedback-responsive.

**(b) The unnormed-feedback path — deleted.** Its first cell called
`_alignAroundEpoch(limit_trial_end=True)`. That guard raises for
`limit_trial_start or limit_trial_end`, but only the *start* case is
unimplemented (it needs `trace_trial_offset_start_idx`); the trial-end branch
is implemented 25 lines below. So the path did run once — the cells still
carried outputs — until a guard aimed at the other flag disabled it. Narrowing
that guard to `limit_trial_start` would revive it, if the results are ever
wanted. The three consumers were already superseded: the last of them defines a
`plotFeedbackExamples` that the live Figure 6B cell redefines, and that cell
reads `all_res_cut_feedback_df` with the old source commented out beside it.
Two later cells did still need the frame the deleted cell built by renaming
Reward/Punishment to "Feedback"; they drop to one row per trial and never read
`epoch`, so they now read `df_all_by_epoch` directly.

**E, done for the 2P notebooks (2026-09-16): a run writes nothing.** Each of
the five now declares `SAVE_FIGS = False` and `SAVE_DATA = False` in its
settings cell, and every save derives from them: the 13 literal `True` flags
(TwoPTraces 27 and 31; 2pAnalysis 54, 56, 57, 66, 68, 75, 94 and twice in 104;
TwoPLoad 32 and 49), the `SAVE_FIGS = True` defaults in plottraces3 and
2pSeqWithinDeviation, and the four unconditional data writes
(`df_all_by_epoch_df_f_filtered.pkl`, `sgf_all.pkl`, `seq_within_deviation_df`
/`seq_cross_deviation_df`/`seq_shuffle_calibration_df`). Cache-style writes are
left as they are — `data_runs_*.pkl` and `rt_corr_shuffled.pkl` only write when
the shipped file is missing. Verified by rerunning all five: **zero writes
attempted**, where TwoPTraces alone had been rewriting 92 committed heatmaps.

- **D2** Extract the inline panels. **Done 2026-09-17.** Every panel on the
  list is a tested module under `twop/`, and both notebooks still run clean.

  | Panels | Module | Tests |
  |---|---|---|
  | 4G | `fastslowvenn.py` | 13 |
  | 6B | `tunedneurons.py` | 10 |
  | 6C | `balancechange.py` | 15 |
  | 6E | `priorcurtuning.py` | 12 |
  | S9A | `tracereliability.py` (shared with TwoPLoad) | 16 |
  | S9B | `activetrialscdf.py` | 13 |
  | S14A | `prevoutcomemod.py` | 16 |
  | S11A, S11B | `activeneuroncount.py`, `activeneuronplots.py`, `traceauc.py` | 32 |
  | S10B, S10C, 4K right | `rtcorrhists.py`, `rtcorrpies.py` | 21 |
  | 4K left, S10A | `rtcorrneurons.py`, `rtcorrpanels.py` | 20 |
  | S12G | `decoders.py`, `decoderplots.py` | 10 |

  **How each was checked.** Run the notebook before and after with saving on,
  then compare the files (normalising matplotlib's per-run date and id salt).
  Result: **2pAnalysis 196/196 identical, plottraces3 30/30 identical, none
  missing**, plus figures that previously could not be saved at all. That loop
  earned its keep twice: it caught `only_active_trials=False` where the
  notebook passes `True` (9 figures moved), and a dropped per-quantile cache
  that broke twelve downstream cells.

  **Three pairs of same-named, different functions** were found, each defined
  in one cell and redefined in the next, with the second silently winning:
  `loopNeuronsPlot`/`plotNeuronCorr`/`_plotFilteredComb` (the list-vs-frame
  pair that `createShuffle` needed), `plotCorrThreshPieChart`, and
  `plotAccuracy`. All are kept and named apart.

  **Two de-duplications:** `shorth` (2pAnalysis had its own copy of
  TwoPLoad's), and the 383-line activity-criterion cell both notebooks carried.

  **Work that ran and was thrown away:** the decoder computed `svm_df` for 55 s
  and the next cell overwrote it from the cache; it is behind
  `LOAD_SVM_FROM_PICKLE` now. The same shape as 2pAnalysis's shuffle cells.

  **Per-panel findings** are in each module's docstring; the ones most likely
  to be "simplified" wrongly: 6C's error bar is the SEM across the window's
  *time points*, not across trials, and its example neuron is picked by a
  monotonic-rise rule; 4G's denominator is the union of the session's fast and
  slow trace ids; 6E's Venn circles are ratios to the overlap while only the
  labels carry percentages; S10C's summary percentage is out of the *region's*
  neurons; S9B's axis labels read the other way round from the data.

- **D3** Consolidate trace loading across the notebooks. **Done 2026-09-17.**
  The four 2P notebooks opened `data/2p` twenty-four times in four idioms, each
  path relative to the notebook, so a frame could only be found from `code/`.
  `twop/dataload.py` names every artifact, says what wrote it, resolves paths
  from the package, and points at `data_downloader.ipynb` when one is absent
  (19 tests). No literal `../data/2p` path is left in any notebook; writes stay
  where the frame is built, behind `SAVE_DATA`. TwoPTraces' custom unpickler
  went with it — it stubbed out `caiman` classes the data rewrite removed, and
  the file reads plainly now. Verified by rerunning all five notebooks: 0
  errors, and figures identical to the pre-extraction baselines.
- **D4** Delete whatever D0 resolves as dead. **Done** — D0's modules, D1's
  dead cells, and D1b's unnormed-feedback path.

**Depends on:** G0 item 5. **Blocks:** E, and the 2P half of F.

---

## E — Notebook parameterisation and runner — **done** (one open item)

**Scope.** The 11 notebooks that produce manuscript figures: `behavior`,
`Tracking`, `opto`, `widefield`, `TwoPLoad`, `2pAnalysis`, `TwoPTraces`,
`plottraces3`, `2pSeqWithinDeviation`, `rlmodel/model_analysis`,
`rlmodel/model_to_behavior`. Exploratory notebooks (`model_viewer` and the
like) are out of scope.

**E1 — one `parameters` cell per notebook.** `SAVE_FIGS`, `SAVE_DATA`,
`PAPER_FIGURES_ONLY`, all `False`; `widefield` adds `MFC_LFC_MAP` and
`DEFAULT_ALLEN_MAP`. `global_save_figs` was renamed to `SAVE_FIGS` everywhere,
Tracking's `True` default is gone, and no call passes `save_figs=True`
literally. With every flag off, all notebooks but `model_to_behavior` ran top
to bottom through the runner and a before/after snapshot of the repository
showed no file written.

**E2 — `code/run_notebooks.py`.** papermill, each notebook executed from its
own directory so the relative data paths hold; `--save-figs`, `--save-data`,
`--paper-figures-only`, `--only`, `--param NAME=VALUE` (injected only where a
notebook declares it; a name no notebook declares is refused as a typo),
`--list`. Executed copies go to `runs/<timestamp>/`. Usage is in
[`code/README.md`](../code/README.md#running-the-notebooks).

**E3 — `--paper-figures-only`.** Every cell that reads `SAVE_FIGS` is tagged
`paper-figure` or `per-subject`; per-subject saves are
`SAVE_FIGS and not PAPER_FIGURES_ONLY`. Fixed along the way — published panels
whose call passed a literal `False`, so no flag could ever regenerate them:
2pAnalysis 5C, 4F, 5E, 5F, S12D/F/H/I, S12J (via a separate
`MOVEMENT_SAVE_FIGS`, now folded into `SAVE_FIGS`), and TwoPTraces 4E/S8B,
4H, 5B, 6A bottom, 5c. `code/util/tests/test_run_notebooks.py` checks the
contract on every suite run.

Two things only a saving run could show, both found in the write sandbox:

- **behavior 1H asserted** the moment figures were saved. The cell overlays
  mice and humans on one shared axis and then saves that figure itself, but
  passed `save_figs` (with `save_prefix=None`) into the overlay calls too —
  which asserts on the missing prefix, and would otherwise have written the
  half-built figure and closed it before the real save. The overlay calls now
  only draw.
- **TwoPLoad's two `per-subject` cells never write anything**, whatever the
  flags: they pass `StdDistCollector.track` as the `processFn`, which ignores
  the `pdf` it is handed, and `PdfPages` opens its file only on the first
  `savefig`. The tag is correct in intent and inert in fact; left alone.

The classification is deliberately over-inclusive: a cell is `per-subject`
only when it is unambiguously bulk. Where a published panel is one example
drawn from a loop over every session or neuron, and the example's ID is not
recorded, the whole loop stays `paper-figure`: 4F's fast/slow traces and 6C's
feedback-tuned neuron (2pAnalysis), 4E/S8B's average traces (TwoPTraces),
2E/2F/S4A's example mouse and Ext. 5a-right (`model_analysis`).

**Verified per notebook in a write sandbox** (a headless run whose every write
is redirected to a mirror, plus a before/after snapshot of the repository).
Each ran twice, saving on, once paper-only; all 0 errors, every paper-only
write also written by the full run, and nothing written outside the mirror:

| notebook | paper-only | full |
|---|---|---|
| behavior | 76 | 142 |
| Tracking | 5 | 29 |
| opto | 39 | 110 |
| TwoPLoad | 3 | 3 |
| 2pAnalysis | 1,695 | 1,718 |
| TwoPTraces | 389 | 477 |

The narrow gaps are the over-inclusive loops above, and they are most of what
a paper-only run writes: 1,446 of 2pAnalysis's files are 4F's per-neuron
traces and 184 are 6C's, and 381 of TwoPTraces' are 4E's per-neuron traces.
Pinning those example IDs is what would make `--paper-figures-only` mean the
manuscript's figures — which is also what F's `results/` pruning needs.

**Open.** `rlmodel/model_to_behavior.ipynb` cannot finish on a workstation.
Its Figure 7D cell resamples the 272 fitted sessions 10,000 times into one
frame (~1.1 billion rows, ~320 GiB) and fails with `MemoryError` on a 49 GB
machine. Before E, the cell was never reached: the notebook loaded a
`…_3s_dt0.005.pkl` fit that no longer exists (now the chi-squared 4.8 s fit).
Needs a decision — a cluster run, a smaller `resample_count`, or streaming
the resamples.

## F — Final cleanup and publish *(small–medium; last)*

- Remove `data/to_delete/`, `rlmodel/model_GUI_cache.pkl`, `rlmodel/run_cmd.txt`.
- Triage the 15 planning `.md` files in `rlmodel/` — which graduate into `docs/`,
  which are deleted. `in-code-2panalysis-ipynb-we-use-majestic-stream.md` and
  `is-the-current-rlmodel-nashaat-oraby-mle-synthetic-fiddle.md` are
  transcript artifacts that should not ship.
- Prune `results/` (3,754 files, 3,686 SVG/PDF). **This depends on E**, because
  the `--paper-figures-only` flag is what defines the keep-set.
- Decide on stripping notebook outputs (~44 MB across four notebooks).
- Fix the root `README.md` title — it still says *"The neural mechanisms of fast
  versus slow decision-making"*; the current title is *"Cortical mechanisms of
  fast versus slow decision making"*. Fix the notebook list in
  `code/README.md`, which names `TwoPAnalysis.ipynb` (actual file:
  `2pAnalysis.ipynb`) and omits six others.
- Renumber the figure labels in every notebook heading to the current scheme —
  see the drift table in [`README.md`](README.md#numbering-drift).

---

## Fork topology — what can run concurrently

After G0 lands, three forks can run at once with no file overlap:

| Fork | Territory | Overlaps with |
|---|---|---|
| **A** environment | `pyproject.toml`, `uv.lock`, `conftest.py`, `data/RLModel/` | `rlmodel/model/fit.py` shared with B — sequence those commits |
| **B** model | `code/rlmodel/**` | as above |
| **C** behaviour | `code/behavior/**`, `code/figcode/**`, `behavior.ipynb` | none |

Then:

| Fork | Territory | Overlaps with |
|---|---|---|
| **D** two-photon | `code/twop/**`, the four 2P notebooks | none, once C has landed its conventions |
| **E** runner | every notebook + a new script | **conflicts with C and D** — must follow both |
| **F** cleanup | everywhere | **conflicts with everything** — must be last |

Deletions were deliberately folded into the workstream that owns the territory
(figcode deletions into C, `twop/plot/stats*` into D, rlmodel registry entries
into B) rather than being collected into a single cleanup fork. A cross-cutting
delete-everything fork would conflict with all three of the others.

---

## Decisions that gate work

Six, in the order they are needed:

| # | Decision | Gates | Note |
|---|---|---|---|
| 1 | ~~Is `behavior_v2.ipynb` current or dead?~~ | — | **resolved** — deleted |
| 2 | ~~Z-formula divergence: fix the χ² path, or footnote the paper?~~ | B2 | **resolved 2026-09-17** — deferred to the refit; B2 left `bias.py`'s formula alone |
| 3 | ~~Keep or delete the asymmetric-learning-rate experiment?~~ | B2 | **resolved 2026-09-17** — fixed, committed, then deleted |
| 4 | Adopt or delete the orphaned `twop/plot/stats*` modules? | D0, D4 | 909 lines; possibly a better factoring than the inline code |
| 5 | Ship only manuscript figures, or all per-subject results? | E, F | Shapes E's flag and F's pruning; the README currently advertises the full set as a feature. **Now also decides the `global_save_figs` default**, which currently differs between notebooks |
| 6 | Strip notebook outputs before publishing? | F | Against stripping: outputs are the only record for cells that cannot currently rerun |

One external dependency: the manuscript is still under revision (copies dated to
2026-08-28). **Freeze the figure set before F prunes `results/`**, or the
keep-set will be wrong.
