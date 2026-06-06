# MLE Terminal-C Redistribution — Implementation Plan

## Goal

Today, residual probability mass remaining inside the open interval `(-B, +B)`
at `t = T_max` is treated as "no-decision" and used as the likelihood of any
trial flagged `no_choice` (and any choice trial whose `RT > T_max` falls through
to `LOGLIK_FLOOR`). The fitter therefore prefers parameters that leave a lot of
residual interior mass when the subject made no choice — even though in real
data a true *terminal* no-choice is rare.

We add a configurable threshold `C ∈ [0, 1]` so that at `t = T_max` the leftover
interior mass is partitioned by absolute position into three buckets:

- `x >  C · B`  →  treated as a **left** choice (upper bound)
- `x < -C · B`  →  treated as a **right** choice (lower bound)
- `|x| ≤ C · B` →  remains genuine **no-decision** mass

The valid range, default, and CLI/argparse parser all draw from a single
source of truth: `MLE_TERMINAL_C = InitVal(0.0, 1.0, 0.0)` defined as a
module-level constant in `rlmodel/model/initvals.py` (lives outside the
`InitVals` dataclass because it's a model-config knob, not a DE-fittable
parameter). Every site — `MLEModelConfig`, the batched solver, the CLI
parser, the GUI slider — imports `MLE_TERMINAL_C` and uses its `.Default`,
`.Min`, `.Max` fields, so changing the range or default is a one-line edit.

Special cases (sanity checks):

- `C = 0`     →  everything except the exact `x = 0` line goes to a choice.
                 The discrete grid uses bin centers at `±dx/2, ±3dx/2, …`,
                 so the no-decision bucket is empty and no-choice trials
                 collapse to `LOGLIK_FLOOR`.
- `C = 1`     →  threshold equals the bound; every interior bin center
                 falls in `|x| ≤ C · B`, so the no-decision band contains
                 the entire interior mass and no-choice likelihood equals
                 the full survival mass — exactly reproducing the legacy
                 survival-only behavior (the upper bound was relaxed from
                 `< 1` to `≤ 1` so this boundary is the canonical
                 "legacy survival" setting).

## Scope

**In scope (this PR):**

- Apply `C` only inside the **batched** likelihood path (`BatchedDiffusionSolver`
  + `batched_choice_rt_loglik` + `objective_from_population`). That is the path
  used by `--fit-mode mle`.
- Apply terminal redistribution to **no-choice trials only** (i.e. trials where
  `valid=True` and `ChoiceLeft.isna()`). For these, the per-trial likelihood
  becomes `terminal_no_decision_mass` instead of `survival_at_tmax`.

**Out of scope / deferred (follow-up "Option B"):**

- Choice trials that timed out (`RT > T_max` or `RT` missing) currently fall
  through to `LOGLIK_FLOOR`. A natural extension is to give them
  `terminal_upper_mass` or `terminal_lower_mass` based on observed choice. We
  do *not* change that here — the user's framing is "no-choice mass gets
  redistributed," and that maps cleanly onto Option A.
- The non-batched **rowwise** path (`_evaluate_trial_likelihoods_rowwise` /
  `trial_choice_rt_loglik`) is *not* updated in this PR. The rowwise path
  exists only as a fallback / reference for debugging and is gated by
  `mle_use_batched_likelihood`. We add a comment noting the deferral and
  validate the assumption with a guard.

## File-by-file change list

### 1. `rlmodel/model/mle.py`

Add a field to the `MLEModelConfig` dataclass and to its validator:

```python
@dataclass(frozen=True)
class MLEModelConfig:
    ...
    mle_terminal_c: float = 0.0  # 0 ≤ C < 1; fraction of bound used as
                                  # no-decision band at t = T_max. C=0 ⇒
                                  # all residual mass goes to a choice;
                                  # C→1 ⇒ pure survival (legacy behavior).
```

```python
def validate_mle_config(model_config):
    ...
    c = float(model_config.mle_terminal_c)
    if not (0.0 <= c < 1.0):
        raise ValueError(
            f"mle_terminal_c must satisfy 0 <= C < 1; got {c}.")
```

Thread `mle_terminal_c` into the batched call site:

- `_evaluate_trial_likelihoods_batched` passes `mle_terminal_c=model_config.mle_terminal_c`
  into `batched_choice_rt_loglik`.
- `objective_from_population` passes the same into its single
  `batched_choice_rt_loglik` call.
- `_evaluate_trial_likelihoods_rowwise` raises (or warns) if `mle_terminal_c != 0.0`
  and the rowwise path is selected — keeps the deferred behavior visible.
  Actually, to avoid breaking anyone using the rowwise path with default
  `mle_terminal_c = 0.0`, we *only* warn when the rowwise path is selected
  *and* `mle_terminal_c != 0.0`. With `C = 0` the rowwise path is broken in a
  different way (survival is used for no-choice but C=0 would say "no
  no-choice mass exists") — we accept this divergence and document it.

### 2. `rlmodel/model/mle_batch.py`

#### 2a. `_BatchedSolverResult`

Add three new fields, all `(b,)` arrays on `xp`:

```python
@dataclass
class _BatchedSolverResult:
    ...
    survival_xp: object                       # (kept) full interior mass
    terminal_upper_mass_xp: object            # (b,) Σ p[x] for x >  C·B
    terminal_lower_mass_xp: object            # (b,) Σ p[x] for x < -C·B
    terminal_no_decision_mass_xp: object      # (b,) Σ p[x] for |x| ≤ C·B
    ...
```

Invariant: `terminal_upper + terminal_lower + terminal_no_decision == survival`.

#### 2b. `BatchedDiffusionSolver.solve`

- Accept a new `terminal_c: float = 0.0` keyword argument.
- Validate `0.0 <= terminal_c < 1.0`.
- After the timestep loop, compute the three terminal masses using a
  precomputed `self.x_grid` (already on `xp`):

  ```python
  thresh = terminal_c * float(bound)
  upper_mask = self.x_grid >  thresh    # (n_x,)
  lower_mask = self.x_grid < -thresh
  no_dec_mask = ~upper_mask & ~lower_mask
  terminal_upper_mass = (p * upper_mask).sum(axis=1)
  terminal_lower_mass = (p * lower_mask).sum(axis=1)
  terminal_no_decision_mass = (p * no_dec_mask).sum(axis=1)
  ```

- Set `survival_xp` to the existing `sum(p, axis=1)` (unchanged).
- For the early-exit `b == 0` branch, return zeros for the three new fields.

#### 2c. `batched_choice_rt_loglik`

- Add `terminal_c: float = 0.0` keyword argument; forward it into
  `_evaluate_batch` and into `solver.solve` via `_evaluate_batch`.

#### 2d. `_evaluate_batch`

- Add `terminal_c` parameter, forward it into `solver.solve`.
- In the per-trial likelihood reduction:

  ```python
  no_choice_like_xp = solver_result.terminal_no_decision_mass_xp
  like_xp = xp.where(no_choice_xp, no_choice_like_xp, density_at_t_xp)
  ```

  Behavior preserved when `C` is set close to 1: the no-decision mass
  becomes ≈ the survival mass, matching the old code.

### 3. `rlmodel/model_runner.py`

Add CLI flag:

```python
parser.add_argument(
    "--mle-terminal-c", type=float, default=0.0,
    help="Terminal-time no-decision band fraction C in [0, 1). At t=T_max, "
         "residual mass with |x| > C·B is reassigned to the closest choice; "
         "|x| <= C·B remains no-decision mass. C=0 (default) forces all "
         "residual mass to a choice; C close to 1 reproduces the legacy "
         "survival-only behavior.")
```

Validate in `main()` (after `parse_args`):

```python
if not (0.0 <= args.mle_terminal_c < 1.0):
    parser.error("--mle-terminal-c must satisfy 0 <= C < 1")
```

Thread the value through `runModel(...)` →`fit.simulateDDM(...)` →
`MLEModelConfig(...)`. The `runModel` signature gains a
`mle_terminal_c=0.0` keyword, as does `simulateDDM`.

### 4. `rlmodel/model/fit.py`

- `simulateDDM(...)` gains `mle_terminal_c=0.0` keyword.
- It passes that into `MLEModelConfig(...)` constructor (only when
  `fit_mode == "mle"`).

### 5. `rlmodel/model/mle_likelihood.py`

- Add a one-line comment in `trial_choice_rt_loglik` noting that this path
  doesn't honor `mle_terminal_c` (deferred to a follow-up).

## Test strategy

New tests live in `rlmodel/model/tests/test_mle_terminal_c.py`.

1. **C=0 collapses no-choice likelihood to ~ LOGLIK_FLOOR.**
   Build a tiny df with one no-choice trial; evaluate with `C=0`. Assert
   the per-trial `mle_loglik` equals `log(LOGLIK_FLOOR)` (or close to it,
   if the grid happens to have a bin centered at 0 the no-decision mass is
   that bin's content — verify against `(p * no_dec_mask).sum()` directly).

2. **C close to 1 reproduces survival behavior.**
   Same df, run with `C = 0.999` and again with the *old* code path
   (compare against `survival_at_tmax` from the batched result). Assert
   the no-choice `mle_loglik` matches `log(survival)` within tight tol.

3. **Terminal mass conservation.**
   For one no-choice trial, assert
   `terminal_upper + terminal_lower + terminal_no_decision ≈ survival`
   (within `1e-12`).

4. **Choice-trial likelihood is unaffected by C.**
   Use a choice trial with `RT < T_max`. Run with C=0 and C=0.5 and verify
   the per-trial `mle_loglik` is identical (densities are read at decision
   time, not at T_max).

5. **CLI validation.**
   `MLEModelConfig(mle_terminal_c=1.0)` raises ValueError on validate.

## Run order

1. Patch `MLEModelConfig` + validator.
2. Patch `BatchedDiffusionSolver.solve` + `_BatchedSolverResult` + call sites.
3. Patch `batched_choice_rt_loglik` + `_evaluate_batch` to pipe `terminal_c`.
4. Patch `mle._evaluate_trial_likelihoods_batched` + `objective_from_population`.
5. Patch `fit.simulateDDM` + `model_runner.runModel` + `main()` for CLI.
6. Add `test_mle_terminal_c.py`.
7. Run full `pytest rlmodel/model/tests/`.
