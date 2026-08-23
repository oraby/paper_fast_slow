# Within-Strategy Trial-to-Trial Sequence Deviation

## Context

`2pAnalysis.ipynb` has two sequence sections — `### Monte Carlo Simulation /
Permutation testing of the sequence` (cell 52) and `### Seq firing deviation`
(cell 55). Both compare the **aggregate** Fast sequence to the **aggregate**
Slow sequence (per-neuron *median* firing order in Fast trials vs in Slow
trials) — a between-condition comparison of two mean sequences.

The user wants a **new** notebook that measures **within-strategy
trial-to-trial variability**: how much does each individual trial's firing
sequence deviate from that strategy's *own* reference sequence, and does that
deviation differ between Fast and Slow, per brain region (MFC, LFC)?

**Design decision (confirmed with user): per-strategy reference ("each vs
itself").** Fast trials are scored against a Fast reference; Slow trials against
a Slow reference. We compare each strategy's own reproducibility. Neuron
non-overlap between Fast/Slow (challenge 1) is handled naturally — each
reference uses only its own strategy's active neurons.

## Background from exploration (files / data, verified)

- **Notebook bootstrap to reuse verbatim** from `2pAnalysis.ipynb`: cell 1
  (package bootstrap so `from .twop...` works), cell 3 (mpl save params:
  `svg.fonttype='none'`, Arial), cell 7 (`fig_save_prefix = "../results/2P/"`).
- **Data load** to reuse: cell 12 loads `normed_sampling_df`; cell 32
  `assignOriginalQuantile` adds `quantile_idx`; cell 34 loads
  `../data/2p/{q1,q3}_res_dict.pkl`; cell 35 builds `max_firing_q1_df` (Fast=
  quantile 1) and `max_firing_q3_df` (Slow=quantile 3). For the new notebook the
  Fast/Slow `res_dict` pickles are enough (no need to rebuild from raw traces).
- **These per-neuron dataframes already contain per-trial firing** (verified by
  loading `q1_res_dict.pkl`): each row = one neuron (`trace_id` = long name, e.g.
  `GP4_23_s6_..._ALM_0`) in one session, with **parallel arrays** `max_idxs`
  (peak sample index per active trial), `active_trial_numbers`, `max_vals`, plus
  `prcnt_valid` (% of that strategy's trials the neuron is active), `ShortName`,
  `BrainRegion`. q1 & q3 share the same **1446-neuron** set across **23
  sessions** (13 `M2_Bi`/MFC, 10 `ALM_Bi`/LFC). Median ~13 active neurons per
  Fast trial; ~all trials have ≥2 active → per-trial sequences are meaningful.
- **Reference-rank recipe** (reuse from cell 36 `extractIQR`): filter
  `prcnt_valid >= 5`, `extractIQR` computes per-neuron `med` = median of
  `max_idxs`, sort by `med`, then `.rank(method="dense")`. This is the *same*
  recipe the existing MC/deviation code uses, minus the 3-way inner-join — we
  keep *all* strategy-active neurons, per session.
- **Colors / naming**: `code/common/clr.py` `BrainRegion` → `BRClr[br]`;
  `code/common/definitions.py` `BrainRegion` enum (`M2_Bi`→"MFC", `ALM_Bi`→"LFC").
- **Tests**: follow `code/pipeline/tests/test_behavior.py` (`sys.path.insert(0,
  parents[3])`, drop stdlib `code` shadow, `from code.twop... import`); repo-root
  `conftest.py` fixes the Windows conda DLL fault; run `uv run pytest` from repo
  root (per `CLAUDE.md`).

## Penalty definition (from the user's worked example — settled, not re-asked)

Within one trial, take the active neurons' **global reference ranks** (their rank
in the whole session-strategy reference, e.g. 1, 5, 10). Form two orderings:
- **expected** = those ranks sorted ascending → `[1, 5, 10]`
- **observed** = the same ranks ordered by the neurons' within-trial peak time
  (`max_idxs` for that trial) → e.g. `[5, 1, 10]`

Per-position penalty = `abs(expected[i] - observed[i]) / n_active_in_trial`,
attributed to the neuron **expected** at position `i`. So rank-1 → `|1-5|/3`,
rank-5 → `|5-1|/3`, rank-10 → `0`. `observed_rank` stored per neuron = the value
`observed[i]` (the reference rank of whichever neuron fired in that neuron's
expected slot). Ties in peak time are broken by reference rank (a tie adds no
artificial deviation). `n_active_in_trial` counts only reference-set neurons
(those passing the >5% filter) active in that trial.

## Implementation

### 1. Backend module `code/twop/seqdeviation.py`
- `extractIQR(df)` — **moved out of** `2pAnalysis.ipynb` cell 36 into this module
  (unit-tested); the notebook keeps using it too.
- `reference_ranks(strategy_df, min_prcnt_valid=5)` — per (`ShortName`) group:
  filter `prcnt_valid >= min_prcnt_valid`, `extractIQR`, dense-rank `med` →
  DataFrame `[ShortName, trace_id, ref_rank, n_ref_neurons]`.
- `trial_penalties(strategy_df, strategy_name, min_prcnt_valid=5)` — invert the
  neuron→trials arrays into trial→(neuron, peak_time); for each trial apply the
  penalty definition above using that strategy's `reference_ranks`; return the
  **long** DataFrame (one row per trial×neuron).
- `build_penalty_df(fast_df, slow_df, min_prcnt_valid=5)` — concat Fast+Slow.
- `plot_session_histograms(pen_df, save_figs)` — per session, overlaid Fast vs
  Slow penalty histograms (colored, e.g. Fast=red/Slow=gold as elsewhere).
- `plot_region_bars(pen_df, brain_region, save_figs)` — two bars (Fast, Slow) =
  mean over sessions of each session's mean per-trial penalty; dotted-circle
  lines connect each session's Fast→Slow average; called once per region.

### 2. Output DataFrame (one row per trial×neuron), saved to `../data/2p/seq_within_deviation_df.pkl`
Columns: `BrainRegion` (MFC/LFC), `ShortName`, `trace_id` (long name),
`trace_num` (parsed suffix), `trial_strategy` (Fast/Slow), `TrialNumber`,
`ref_rank`, `observed_rank`, `penalty`, `n_active_in_trial`, `n_ref_neurons`.

### 3. New notebook `code/2pSeqWithinDeviation.ipynb`
Reuses the bootstrap + mpl + data-load cells; defines a `SAVE_FIGS` flag and a
`REFERENCE_STRATEGY` flag (Fast/Slow) for focused single-strategy views — the
summary (part D) always computes both. Cells call the backend only (no heavy
logic inline): build `max_firing_q1_df`/`q3_df` → `build_penalty_df` → save df →
(C) per-session Fast/Slow histograms → (D) two-bar Fast-vs-Slow figure with
per-session connecting lines, once for MFC then LFC.

### 4. Unit tests `code/twop/tests/test_seqdeviation.py`
- `extractIQR` on a tiny hand-checked frame.
- Penalty on the user's worked example → `{1:|1-5|/3, 5:|5-1|/3, 10:0}`.
- `reference_ranks` (>5% filter + dense rank by `med`, per session).

## Verification

- `uv run pytest code/twop/tests/test_seqdeviation.py -q` — the worked example
  must reproduce `|1-5|/3, |5-1|/3, 0`.
- Run `2pSeqWithinDeviation.ipynb` top-to-bottom (conda `py312` kernel) to
  produce `seq_within_deviation_df.pkl` and all figures; spot-check `ref_rank`
  ranges and `n_ref_neurons` against `max_firing_q*_df` (e.g. MFC session
  `GP4_23_S1...` ≈157 Fast ref neurons).

## Deferred (per user)

Significance testing (Fast vs Slow) is **not** in this pass — after reviewing the
histograms/bars the user will decide on amendments and whether to add stats.

---
---

# Extension F — Controlled-shuffle calibration curve

*(Everything above is DONE and shipped — including "### 3. New notebook", which created
`2pSeqWithinDeviation.ipynb` and is history, not a proposal. This section is the only
remaining work, and it EXTENDS that same notebook. No new notebook is created.)*

## Context

Parts C–E score trial-to-trial sequence disorder and compare it against a **single**
fully-random shuffle (chance ≈ 0.667, the third bar in `plot_cross_bars`). That gives
only two anchors — "real" and "random" — so an observed value like 0.271 has no
intuitive meaning beyond "well below chance".

This extension makes the shuffle **continuous**: it walks a controlled perturbation
from the reference order (0%) to the fully reversed order (100%), measures the
gap-aware disorder at each level, and then reads each session's *observed* disorder
back off that curve as an equivalent shuffle level. The payoff is an interpretable
sentence — "the Fast sequence is about as disordered as inverting ~19% of the neuron
pairs, the Slow sequence ~23%" — and a figure where Fast and Slow sit on a common,
meaningful axis instead of two bare bars.

## Confirmed design decisions (user-chosen — do not re-litigate)

- **Mechanism = Mallows / pairwise inversion.** Level `s` = expected fraction of
  neuron **pairs** inverted w.r.t. the reference order (normalized Kendall distance).
  Chosen over "reverse a random subset" and "swap symmetric pairs" because it is the
  only option where **x = 50% *is* fully random**, making the 0.667 line land exactly
  mid-axis and giving x > 50% the meaning "systematically reversed / anti-ordered".
- **Calibration curve is per session**: built from that session's own trials (its real
  active-set sizes and rank gaps), **pooling both `condition`s** — consistent with the
  existing chance bar (`_session_shuffle_means(..., group_col=None)`). Each session's
  points invert **its own** curve; the big mean±sem dots invert the **region-mean** curve.
- **y-axis score = `gap_norm_penalty`** ("Mean per-trial gap-aware disorder").
- 4 figures total: {MFC, LFC} × {Fast reference, Slow reference}.
- Point colors follow **strategy**, not condition (`FAST_CLR` red / `SLOW_CLR` gold),
  matching `plot_cross_bars`. In the Fast-ref figure red = Fast (matched), gold = Slow
  (cross); in the Slow-ref figure red = Fast (cross), gold = Slow (matched).

## Math — verified numerically during planning (n=15, dense-tied and distinct ranks)

Mallows φ-model sampled by **repeated insertion (RIM)**: insert item `j` at position
`p ∈ {1..j}` with probability ∝ `φ^(j-p)`; the inversions item `j` adds are
`Z_j ∈ {0..j-1}` with `P(Z_j=z) ∝ φ^z`, so

```
E[d](φ,n) = Σ_{j=1..n}  ( Σ_{z=0..j-1} z·φ^z ) / ( Σ_{z=0..j-1} φ^z )
max_d     = n(n-1)/2          E[d](1,n) = n(n-1)/4   →  s = 0.5 exactly (verified)
```

`E[d]` is monotone in φ → **bisect φ ∈ [0,1]** to hit `s·max_d` for `s ≤ 0.5`.
For `s > 0.5`: sample at `φ(1-s)` then **reverse** each permutation (reversal inverts
every pair, so `d → max_d - d`). All three anchors are then exact by construction:

| s | 0 | .05 | .10 | .20 | .30 | .40 | **.50** | .60 | .75 | .90 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| φ | 0 | .283 | .456 | .663 | .794 | .900 | **1.0** | .900 | .734 | .456 | 0 |
| disorder | **.000** | .082 | .149 | .286 | .423 | .556 | **.670** | .778 | .910 | .991 | **1.000** |

The curve is strongly **concave**, so the shuffle axis is *not* a relabeling of y:
MFC Fast 0.271 → x ≈ **19%**, MFC Slow 0.333 → x ≈ **23%**.

## Backend — `code/twop/seqdeviation.py`

1. **Refactor for reuse (no behavior change):** extract the score formulas out of
   `_shuffled_trial_scores` into `_perm_scores(e, sigma, score_cols)` (takes a
   `(n_perm, n)` permutation array, returns the score dict). `_shuffled_trial_scores`
   keeps its signature and just calls it with uniform `sigma` — the existing 16 tests
   and `cross_shuffle_test` / the chance bar must stay green.
2. `_mallows_expected_d(phi, n)` — the closed form above.
3. `_mallows_phi(n, s)` — bisection on `[0,1]` (80 iters); `s=0 → φ=0`, `s=0.5 → φ=1`.
4. `_rim_perms(n, phi, size, rng)` — **RIM vectorized over the `size` axis**: loop the
   `n` insertion steps, sample `z` for all reps at once via inverse-CDF
   (`searchsorted`), and build the permutation with `np.where(idx == pos, j-1, ...)`
   fancy indexing. (Prototyped and verified in planning.)
5. `_perm_pool(n, s, rng, pool_size=500)` — **cached by `(n, s)`**: permutations depend
   only on `n` and `s`, never on the trial's rank values, so one pool serves every
   trial with that `n`. Handles `s > 0.5` via the reverse trick. This is the
   optimization that makes the whole thing tractable (~40 distinct `n` × 32 levels of
   pooled sampling instead of ~4000 trials × 32 levels).
6. `SHUFFLE_LEVELS` — `0–10%` step 1, `10–20%` step 2, `20–100%` step 5, de-duped
   → **32 levels** (as fractions 0.0–1.0).
7. `shuffle_calibration(cross_df, score_col="gap_norm_penalty", levels=SHUFFLE_LEVELS,
   n_rep=100, seed=0, pool_size=500, show_progress=True)` →
   long df `[BrainRegion, ShortName, level, rep, score]`. Per (region, session, level):
   for each trial draw `n_rep` permutations at random from `_perm_pool(n, s)`, score via
   `_perm_scores`, average over trials → one value per rep. `tqdm` (`from tqdm.auto
   import tqdm`, the house style — already a `pyproject.toml` dep) over session×level.
8. `_invert_curve(levels, curve_y, y)` → `np.interp(y, curve_y, levels*100)` (curve is
   monotone; `np.interp` clamps out-of-range y).
9. `plot_shuffle_calibration(cross_df, brain_region, calib, score_col="gap_norm_penalty",
   save_figs, fig_save_prefix)` — one figure per (region, reference); mirrors
   `plot_cross_bars`' signature/save convention.

## The figure

- **Background**: grey scatter of every (level×100, score) point from `calib` for that
  region (all sessions × reps), small + low alpha; region-mean curve drawn as a line.
- **Dashed horizontal line** at the **empirical** random level = the region-mean curve's
  value at `level = 0.50` (φ=1 *is* uniform random, so this is self-consistent rather
  than a hard-coded 0.667). Labeled "max-shuffle level (fully random order) = 0.67".
- **Per session**: red (Fast) + gold (Slow) points from
  `_session_means(br_df, score_col, group_col="trial_strategy")`, joined by a thin grey
  dotted line; each point's **x = invert that session's own curve at its y**.
- **Global mean**: one big red + one big gold dot, `y = mean ± sem` over sessions
  (yerr bars), `x = invert the region-mean curve at the mean y`.
- **x-axis**: 0–100%, plus **rotated 90° annotations** below the axis at x=0
  ("No rank deviation") and x=100 ("Reversed rank deviation").
- **y-axis**: "Mean per-trial gap-aware disorder"; save to
  `SeqWithinDeviation/shuffle_calib_{reference}ref_{score_col}_{region}.svg`.

## Notebook — EXTEND the existing `code/2pSeqWithinDeviation.ipynb`

**No new notebook.** Append a "Part F" section to the existing notebook, exactly like
Parts C/D/E: it already has the bootstrap, the mpl params, the `max_firing_q1_df` /
`max_firing_q3_df` load, and the `cross_fast_ref` / `cross_slow_ref` build — which is
precisely what the calibration consumes, so a separate notebook would only duplicate
that setup.

One markdown intro + two code cells (backend calls only, per `CLAUDE.md`):
- flags `SHUFFLE_CAL_REPS = 100`, `SHUFFLE_CAL_SEED = 0`; build `calib_df` for both
  references and **save to `../data/2p/seq_shuffle_calibration_df.pkl`** (it is the
  expensive step — cache it like `penalty_df`).
- plot cell: `for _cross in [cross_fast_ref, cross_slow_ref]: for _br in ["MFC","LFC"]:
  sd.plot_shuffle_calibration(...)` → 4 figures.

## Tests — `code/twop/tests/test_seqdeviation.py` (extend; currently 16 passing)

- `_mallows_expected_d(1.0, n) == n(n-1)/4`, and monotone in φ.
- `_mallows_phi(n, 0) == 0`, `_mallows_phi(n, 0.5) ≈ 1`.
- `_rim_perms` at φ=0 → identity; empirical mean Kendall distance at a fitted φ matches
  the target `s·max_d` (tolerance ~5%).
- **Endpoints/midpoint of the calibration** on a synthetic session: level 0 → disorder
  0, level 1 → 1.0, level 0.5 → ≈0.667 (±0.03); curve monotone non-decreasing.
- `_invert_curve` round-trips a known y.
- `_perm_scores` refactor: uniform `sigma` reproduces the current
  `_shuffled_trial_scores` output for a fixed seed.

## Verification

- `uv run pytest code/twop/tests/test_seqdeviation.py -q` from the repo root — all
  previous 16 tests must stay green (the `_perm_scores` refactor is the risk).
- Headless notebook run (scratchpad `run_nb.py`) → must reach "NOTEBOOK RUN OK" with
  the 4 new figures; render one to PNG and confirm visually: curve hits 0 at x=0 and
  1.0 at x=100, crosses the dashed line at x≈50, and the red/gold dots sit on the curve
  at their inverted x (expect MFC ≈19% Fast / ≈23% Slow).
- Sanity: `calib_df` level 0.5 mean ≈ the `chance_*` values already reported by
  `cross_shuffle_test` (≈0.66) — two independent paths to the same number.

## Runtime note

Pooling permutations by `(n, s)` keeps this to roughly a minute at `n_rep=100`; the
result is cached to pkl so re-plotting is instant. If it drags, lower
`SHUFFLE_CAL_REPS` — it only thins the background cloud, not the curve's accuracy.
