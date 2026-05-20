# Plan: CuPy-backed batched MLE with NumPy fallback

## Summary
- First write this plan to `rlmodel/model/cupy_mle_plan.md` as the implementation reference.
- A simple `xp=cupy` swap will not speed up the current MLE path because it is per-trial and still uses NumPy/SciPy in hot solver code.
- Implement real GPU support through backend/device selection plus a batched MLE likelihood path.
- CuPy does not provide a normal CPU fallback; local testing will validate fallback to NumPy when CuPy/CUDA is unavailable.

## Implementation Changes
- Restore CPU MLE correctness first:
  - Fix the current `observed_rt` NameError.
  - Mask missing/non-finite `DV` trials out of likelihood.
  - Ensure invalid objective evaluations return a finite penalty, never NaN.
  - Run focused MLE tests before GPU work.

- Add MLE backend config and CLI flags:
  - `mle_array_backend: "auto" | "numpy" | "cupy"`
  - `mle_device_id: int | None`
  - `mle_cupy_fallback: "numpy" | "error"`
  - `mle_batch_size`
  - CLI flags: `--mle-array-backend`, `--mle-device-id`, `--mle-cupy-fallback`.

- Add backend resolver:
  - Try CuPy import/device selection/allocation for `auto` or `cupy`.
  - Fall back to NumPy with a warning when configured.
  - Raise a clear error when fallback is disabled.
  - Record requested backend, actual backend, device id, and fallback warning in fit metadata.

- Add batched MLE likelihood:
  - Keep SciPy `differential_evolution` on CPU.
  - Compute teacher-forced Q/R state on CPU into dense trial arrays.
  - Run batched backend-native diffusion likelihood over those arrays.
  - Support all current MLE variants, including Classic, Q-Val, R, Decay-Q, and decaying noise variants.
  - Transfer only final likelihood/diagnostic arrays back to CPU.
  - Use the existing rowwise NumPy evaluator for CPU fallback by default; the batched matrix solver is selected for real CuPy execution and can be forced in tests/benchmarks.

- Multi-GPU behavior:
  - One process binds to one selected GPU with `--mle-device-id`.
  - Multiple GPUs are used by launching separate fits with different device ids.
  - No internal multi-GPU scheduler in this pass.

## Test Plan
- Existing focused MLE smoke and teacher-forcing tests must pass.
- Add NumPy batched-vs-rowwise equivalence tests across representative variants.
- Add backend resolver tests for CuPy import failure, fallback behavior, explicit error behavior, and fake device-id selection.
- Add optional CuPy integration tests skipped when CuPy/CUDA is unavailable.
- Add a manually runnable benchmark comparing rowwise NumPy, batched NumPy, and CuPy when available.

## Assumptions
- All MLE variants are in scope for v1.
- Local machine validates fallback/mocking, not actual CUDA performance.
- Target machine installs the appropriate CuPy package separately.
- Optimizer-level multi-GPU scheduling is out of scope for this implementation.
