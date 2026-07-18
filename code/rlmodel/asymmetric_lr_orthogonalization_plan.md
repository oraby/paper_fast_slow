# Orthogonal asymmetric-LR opt-ins — plan + live TODO

Persistent in-repo record for the asym-orthogonalization work. The
checklist below is kept current as each step lands.

## Context (short form)

The asymmetric-LR capability was fused with two dedicated registry
entries (`"Q-Val-asym (Offset)"` in `BIAS_FN_DICT`,
`"NoiseGain-RewardRate-asym"` in `DRIFT_FN_DICT`), both pure aliases
detected via substring match (`"asym" in bias_fn_str`). That conflated
the asym capability with the model identity, locked it to those two
specific variants, and prevented Decay-Q drift variants (which DO
learn Q-values) from ever using asymmetric ALPHA.

This refactor makes asym a **first-class orthogonal capability**:

- Two boolean CLI flags `--asym-q` / `--asym-rr` enable it.
- The capability combines with any Q-learning model and any
  RewardRate model — independent of the bias / drift / noise strings.
- `MLEModelConfig.uses_asymmetric_alpha` / `_beta` (added in the
  previous consolidation work) become the single source of truth.
- File names get an optional `_asymQ` / `_asymRR` / `_asymQRR`
  suffix; symmetric fits keep the existing format (no migration).
- The GUI exposes two checkboxes ("Asymmetric Q-update",
  "Asymmetric RR-update") instead of dropdown entries.

**Audit confirmed**: no `-asym` saved-fit files exist on disk yet, so
no rename/regeneration is needed — just an update to
`models_loop.sh` to switch from the registry names to the new flags.

## TODO

### 0. Persistent plan file
- [x] Save this file under `rlmodel/`

### 1. Drop the `-asym` registry aliases
- [x] `rlmodel/model/bias.py:60` — remove `"Q-Val-asym (Offset)"` entry
- [x] `rlmodel/model/drift.py:177` — remove `"NoiseGain-RewardRate-asym"` entry

### 2. CLI flags in `model_runner.py`
- [x] Add `--asym-q` / `--asym-rr` argparse flags with help text
- [x] Add CLI pre-flight: `parser.error(...)` when a flag is set but the
      selected (bias, drift, noise) trio doesn't expose `Q_val` / `RewardRate`
      cols (same auto-detection `fit.py:350` uses)
- [x] Plumb `uses_asym_q` / `uses_asym_rr` through `runModel(...)`

### 3. Replace substring gating in `fit.py`
- [x] `fit.simulateDDM`: accept `uses_asym_q` / `uses_asym_rr` kwargs
- [x] Replace `include_Q_asym = include_Q and "asym" in bias_fn_str` (and
      the RR twin) with `include_Q_asym = include_Q and uses_asym_q`
- [x] Confirm `_PARAM_FIT_GATES` and `MLEModelConfig` plumbing downstream
      is unchanged

### 4. Filename suffix in `fit.evolveFP`
- [x] Extend signature with `uses_asym_q=False, uses_asym_rr=False`
- [x] Append `_asymQ` / `_asymRR` / `_asymQRR` suffix when the flags
      are set; no suffix when both False (backwards-compat)
- [x] Thread the flags through every `evolveFP` call site

### 5. GUI checkboxes in `visualize.py`
- [x] Drop hardcoded model-name disable gates at L293, L295
- [x] Drop substring match at L644-645 in `_evaluate_mle_loss_for_gui`
- [x] Add `"Asymmetric Q-update"` checkbox next to ALPHA slider; disable
      when `include_Q` is False
- [x] Add `"Asymmetric RR-update"` checkbox next to BETA slider; disable
      when `include_RewardRate` is False
- [x] Drive `ALPHA_UNREWARDED` / `BETA_UNREWARDED` slider enable from
      the new checkboxes
- [x] Drive `_evaluate_mle_loss_for_gui`'s flag construction from the
      checkboxes

### 6. Population explorer surface
- [x] `rlmodel/model/mle_notebooks/data.py:flatten_mle_results`: attach
      `mle_uses_asymmetric_alpha` / `_beta` columns from saved
      `model_config` (with `getattr(..., False)` so old pickles load
      cleanly)
- [x] `rlmodel/model/mle_notebooks/histograms.py`: add `"asym α"` /
      `"asym β"` to `DEFAULT_LATENT_COLUMNS`
- [x] `rlmodel/model/mle_notebooks/ddm_viewer.py`: surface asym flags
      in `plot_ddm_frame` title (optional polish)

### 7. `models_loop.sh` rewrite
- [x] Replace the two `-asym` registry-name combos with canonical names
      + new flags
- [x] Extend the loop parser to accept a 4th pipe-separated field of
      extra CLI args
- [x] Add Decay-Q + asym-q combos that the old design couldn't express

### 8. Tests
- [x] `test_asymmetric_lr.py`: update fixtures from `bias_name="Q-Val-asym (Offset)"`
      → canonical names + `uses_asym_q=True` plumbed through `_run_dry`
- [x] New test: Decay-Q + asym-q orthogonality (regression for the
      newly-possible combination)
- [x] New test: `fit.evolveFP` emits the correct asym suffix for each
      flag combination
- [x] New test: `model_runner` CLI `--asym-q` with a non-Q-learning
      model exits with `parser.error`
- [x] `test_mle_notebooks.py`: smoke test for the new
      `mle_uses_asymmetric_*` columns from `flatten_mle_results`

### 9. Verification
- [x] Full `pytest rlmodel/model/tests/` — expect 130 + new tests, 1
      skipped (CuPy)
- [x] CLI smoke: `--drift "Decay Q (Offset)" --asym-q --dry-run`
      produces `_asymQ.pkl` filename and `ALPHA_UNREWARDED` in Fit
      Params Names (orthogonality proof)
- [x] CLI smoke: `--drift Classic --bias None_ --asym-q` exits with
      `parser.error` naming the missing prerequisite
- [x] Notebooks (manual): `model_interactive.ipynb` + DDM viewer +
      population explorer all work for both new asym fits and existing
      symmetric pickles

## Files modified

| File | Change |
|---|---|
| `rlmodel/model/bias.py` | Drop `"Q-Val-asym (Offset)"` entry. |
| `rlmodel/model/drift.py` | Drop `"NoiseGain-RewardRate-asym"` entry. |
| `rlmodel/model_runner.py` | Add `--asym-q` / `--asym-rr` flags + pre-flight; plumb into `runModel`. |
| `rlmodel/model/fit.py` | Substring → flag args. Extend `evolveFP` with `_asymQ`/`_asymRR`/`_asymQRR` suffix. |
| `rlmodel/model/visualize.py` | Two new Asym checkboxes; drop substring + hardcoded model-name disables. |
| `rlmodel/model/mle_notebooks/data.py` | Attach `mle_uses_asymmetric_*` columns. |
| `rlmodel/model/mle_notebooks/histograms.py` | Add asym columns to `DEFAULT_LATENT_COLUMNS`. |
| `rlmodel/model/mle_notebooks/ddm_viewer.py` | Asym flags in figure title (optional). |
| `rlmodel/models_loop.sh` | Switch to canonical names + new flags; new Decay-Q + asym combos. |
| `rlmodel/model/tests/test_asymmetric_lr.py` | Fixtures + new orthogonality + evolveFP + CLI tests. |
| `rlmodel/model/tests/test_mle_notebooks.py` | Smoke test for new asym columns. |

## Backwards compatibility

- **Existing symmetric pickles on disk**: zero impact. Filename has no
  suffix (matches new `evolveFP` for `(False, False)`); saved
  `MLEModelConfig` either has the flags as `False` already (post-
  consolidation) or `getattr(..., False)` returns `False` (pre-).
- **No `-asym` saved-fit files exist** — audit confirmed. No rename or
  regenerate needed.
- **`models_loop.sh`** is the only consumer of the dropped registry
  keys and is rewritten in this PR.

## Checkpoint 1 — parked

> **Whether to check for invalid trials at mle.py.** Keep the explicit
> `xp.where(valid_t, ...)` mask in `_compute_latent_population_equal_sessions`.
> `_extendTrials` padded trials and EarlyWithdrawal trials don't carry
> `ChoiceLeft=NaN`, so `state_updates`' no-choice branch wouldn't catch
> them; pre-NaN-ing at load time would conflate the "valid-for-loss"
> model concept with the "no choice" data concept and ripple into
> Chisqr, plotting, and posterior tools.
