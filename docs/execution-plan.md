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
| **B** | Model trim + docs | **not started** — `Decay Q` still in the registry, `rlmodel/README.md` still says 3 s and uses the oldest figure numbers |
| **C** | Behaviour tests | **done, C1–C7** — six panels extracted to `behavior/` (45 → 147 tests); `figcode/`, `opto/` and the new `tracking/` given their first tests (0 → 114, 0 → 50, 0 → 49). C6 found and fixed **two panels that could not be regenerated** (Figures 1D, S3E); C7 extracted `Tracking.ipynb` (Figures S3J–M), whose panels raised `NameError` outside the notebook. All verified against the committed figures |
| **D** | 2-photon reorg | **D0, D1 done** — orphaned modules deleted; no notebook now loads an undefined name. **Two blockers surfaced** (see D1): `TwoPTraces.ipynb`'s Figure 4H sorting reference and `TwoPLoad.ipynb` both call code that has never existed in this repo. D2 (extraction) remains |
| **E** | Runner / papermill | **started ahead of plan** — 4 notebooks carry a `parameters` cell; see the E section for what that does and does not yet cover |
| **F** | Final cleanup | **not started** — `data/to_delete/` is still 563 MB |

Suite: **1371 passed, 1 skipped, 0 failed**, and green with
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
  function, the 10-entry `Decay Q` drift family.
- **B3** — rewrite the model documentation against
  [`manuscript-methods-map.md`](manuscript-methods-map.md#the-model), which
  carries every equation as the paper states it.

**B3 must follow B2**, otherwise you write documentation for code you are about
to delete.

**Two gates before B2 can start**, both of which are decisions rather than work:

- **The asymmetric-learning-rate experiment.** `Decaying Q-Val` is reached from
  `mle.py:141`, `model_runner.py:482`, `tests/test_asymmetric_lr.py` and
  `tests/test_mle_smoke.py`. Deleting the `Decay Q` family forces a decision on
  whether that unpublished experiment stays.
- **The Z-formula divergence.** The MLE and χ² paths compute the DDM starting
  point differently and the paper states only the MLE form. The fix lands in
  `bias.py`, which B2 also edits — and **correcting the χ² path would change a
  published number.** Resolve this before touching `bias.py`, not after.

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

### Two blockers D1 surfaced — decisions needed

Both are references to code that **has never existed in this repo** (checked
across all `.py` in git history and every notebook, live or commented):

1. **`TwoPTraces.ipynb` calls `loopCombinations`** (cells 19 and 22), which is
   defined nowhere; only `loopCombinationsTraces` exists. The following cells
   `assert success` / `assert success_quantiles`, so the section fails hard.
   That section builds the **all-trials sorting reference for Figure 4H** —
   the Methods say Fast/Slow heatmaps "were sorted in reference to all trials'
   heatmap". So Figure 4H is not reproducible from this checkout as it stands.

2. **`TwoPLoad.ipynb` cannot run at all.** Beyond the imports (fixed), it reads
   `pkl/df_all_by_epoch.pkl` and writes `../results/opto2P` — neither exists;
   the data lives in `data/2p/`. Two names are still unbound: `shorth` (a
   shortest-half estimator, cell 23) and `ref_accepted_traces` (assignment
   commented out, cell 30). It also has no `SAVE_FIGS` / `fig_save_prefix`
   cell, unlike every other notebook, contrary to `CLAUDE.md`.

   `README.md` describes it as building the frames the other 2P notebooks
   consume — but since it cannot run here, those frames in `data/2p/` came from
   somewhere else. Cell 23 does implement a documented Method (the active-trial
   threshold), so this is provenance worth keeping in *some* form.

   Options: repair it against the real paths, reduce it to the parts that are
   documented Methods, or drop it and say plainly in the README that the 2P
   frames are supplied pre-built by the download archive.

- **D2** Extract the inline panels: 4G, 4K, 6B, 6C, 6E, S9A–B, S10A–C,
  S11A-mid/right, S11B, S12G, S14A.
- **D3** Consolidate trace loading across the three notebooks, on top of G0's
  shared unpickler.
- **D4** Delete whatever D0 resolves as dead.

**Depends on:** G0 item 5. **Blocks:** E, and the 2P half of F.

---

## E — Notebook parameterisation and runner — **partly started**

Four notebooks now carry a papermill `parameters` cell: `behavior.ipynb`,
`opto.ipynb`, `widefield.ipynb` and `Tracking.ipynb`. `widefield.ipynb`'s goes
beyond the save flag (`MFC_LFC_MAP`, `DEFAULT_ALLEN_MAP`), which is the right
shape.

Two things to settle before this spreads further:

- **The default is inconsistent.** `Tracking.ipynb` defaults
  `global_save_figs = True`; the other three default `False`. A batch run will
  write `results/` from one notebook and not the others.
- **A `parameters` cell does not make a notebook parameterised.** 19 literal
  `save_fig=True` / `save_figs=True` call sites remain, so those cells ignore
  the flag entirely:

  | notebook | literals | `parameters` cell |
  |---|---|---|
  | `2pAnalysis.ipynb` | 9 | no |
  | `TwoPLoad.ipynb` | 3 | no |
  | `plottraces3.ipynb` | 3 | no |
  | `TwoPTraces.ipynb` | 2 | no |
  | `Tracking.ipynb` | 1 | yes |
  | `model_to_behavior.ipynb` | 1 | no |
  | `model_viewer.ipynb` | 1 | no |

  `behavior.ipynb` and `opto.ipynb` are clean on both counts.

Still to do: the remaining `parameters` cells, the 19 literals, repo-root-relative
data paths, the `--paper-figures-only` flag, and the runner itself. The 2P
notebooks should wait for **D** rather than be parameterised twice — and
`plottraces3.ipynb` cannot be run by any runner until D1 fixes its six
`NameError` cells.

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
| 2 | Z-formula divergence: fix the χ² path, or footnote the paper? | B2 | **Fixing it changes a published number** |
| 3 | Keep or delete the asymmetric-learning-rate experiment? | B2 | Entangled with `Decay Q` through `Decaying Q-Val` |
| 4 | Adopt or delete the orphaned `twop/plot/stats*` modules? | D0, D4 | 909 lines; possibly a better factoring than the inline code |
| 5 | Ship only manuscript figures, or all per-subject results? | E, F | Shapes E's flag and F's pruning; the README currently advertises the full set as a feature. **Now also decides the `global_save_figs` default**, which currently differs between notebooks |
| 6 | Strip notebook outputs before publishing? | F | Against stripping: outputs are the only record for cells that cannot currently rerun |

One external dependency: the manuscript is still under revision (copies dated to
2026-08-28). **Freeze the figure set before F prunes `results/`**, or the
keep-set will be wrong.
