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

## Status board — 2026-09-04

| | Workstream | Status |
|---|---|---|
| **G0** | Trustworthy baseline | **done** |
| **A** | Unified `uv` | **done** — everything runs from the lockfile; guarded by `test_environment.py` |
| **B** | Model trim + docs | **not started** — `Decay Q` still in the registry, `rlmodel/README.md` still says 3 s and uses the oldest figure numbers |
| **C** | Behaviour tests | **C1 done** — Figure 2A extracted to `behavior/varexplained.py` + 21 tests (behaviour package 45 → 66). Reproduces the published bars, the 101.9% total and the 14.3–22.2 condition numbers exactly. C2–C5 remain; Figure 2B is still inline and triplicated |
| **D** | 2-photon reorg | **D0 done** — the orphaned `twop/plot` modules are resolved and deleted. D1 (`plottraces3.ipynb`'s 6 `NameError` cells) and D2 (extraction) remain |
| **E** | Runner / papermill | **started ahead of plan** — 4 notebooks carry a `parameters` cell; see the E section for what that does and does not yet cover |
| **F** | Final cleanup | **not started** — `data/to_delete/` is still 563 MB |

Suite: **865 passed, 1 skipped, 0 failed**.

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

- **C1** Figure 2A — variance explained. Currently `_glmRTFn`, `_simplifyDF`,
  `loopSubjects`, `plotVarExplaind` inline in `behavior.ipynb`. Four numbers from
  it are quoted in the Results text.
- **C2** Figure 2B — optimal sampling. Inline, **triplicated** (~1,800 notebook
  lines across three copies; only the "Mixed" section is live). Methods
  eqs. 1–3. Extracting this deletes two copies at the same time.
- **C3** Figure 1I-right, S3G, S2B, S2M — the remaining inline behaviour panels.
- **C4** Delete `figcode/prevoutcomecurquantile_{bak,new}.py` (407 lines,
  superseded).
- **C5** First tests for `figcode/` (0 tests today) and `opto/` (0 tests today).
  The hierarchical bootstrap behind Figures 3D, 4C and S6G is the highest-value
  target — it is a resampling procedure with no regression test.

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

- **D1** Fix the five non-runnable cells in `plottraces3.ipynb`
  (`plotSgfActivitySum`, `_iqrPRCNT`, `max_firing_all_df`, `svm_normed_df` never
  defined; `res_corr_raw` order-dependent). **This is a hard prerequisite for E**
  — no runner can execute a notebook that raises `NameError`.
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
