# MLE Diffusion Fitting — Optimization Plan

Reference for the GPU-friendly MLE pipeline. Tracks what's already landed and
what's left, with concrete pointers into the code. The numbers below assume a
representative fit: **S = 60 candidates × N ≈ 5000 trials × n_t = 3000
timesteps**, with `dx = 0.02 → n_x = 100`.

## What's already done

| # | Optimization | Status | Where |
|---|---|---|---|
| D0 | `sigma*sqrt(dt)` noise scaling (Milestone 0) | ✅ done | [noise.py](noise.py) |
| D1 | State updates extracted, no `iterrows()` (Milestone 1) | ✅ done | [state_updates.py](state_updates.py), [mle.py](mle.py) `_compute_latent_population_equal_sessions` |
| D2 | First-passage reference + vectorized const_mu / time_mu | ✅ done | [diffusion/](diffusion/) |
| D3 | MLE objective + dispatcher | ✅ done | [mle.py](mle.py), [fit.py](fit.py) `_processSubject` |
| D4 | Bucketed-FFT batched solver | ✅ done | [mle_batch.py](mle_batch.py) `BatchedDiffusionSolver.solve` |
| D5 | Deferred `asnumpy` — gather likelihood on xp, single `(b,)` sync per batch | ✅ done | [mle_batch.py](mle_batch.py) `_evaluate_batch` |
| D6 | Vectorized DE (`vectorized=True`), workers=1, updating="deferred" | ✅ done | [fit.py](fit.py) MLE DE call |
| D7 | Single-process MLE (no `multiprocessing.Pool` contention for the GPU) | ✅ done | [fit.py](fit.py) `simulateDDM` |
| D8 | `--mle-backend {CPU,GPU}` CLI + GPU pre-flight probe | ✅ done | [array_backend.py](array_backend.py) `assert_gpu_backend`, [fit.py](fit.py) |
| D9 | Population-level batching: one solver call per generation for `S×N` items | ✅ done | [mle.py](mle.py) `objective_from_population`, `_compute_latent_population_equal_sessions` |
| D10 | Q/R updates vectorized across (candidates × sessions) at each trial position | ✅ done | [mle.py](mle.py) `_compute_latent_population_equal_sessions` |
| D11 | Backend-cached per-session arrays (`dv`, `valid`, `choice`, `reward` on xp) | ✅ done | [mle.py](mle.py) `_prepared_session_arrays_for_backend` + `_PREPARED_SESSION_BACKEND_CACHE` |
| D12 | **O1** — Drop `(b, n_t)` density tensors; in-loop decision-time gather | ✅ done | [mle_batch.py](mle_batch.py) `BatchedDiffusionSolver.solve` (`upper_at_decision`, `lower_at_decision`); `_BatchedSolverResult.upper_at_decision_xp` |
| D13 | **O4** — Stop expanding constant-mu `(b,)` → `(b, n_t)`; use `_prepare_mu` to keep mu in natural shape | ✅ done | [mle_batch.py](mle_batch.py) `_prepare_mu`, `_mu_isfinite_per_trial`, `_coerce_to_numpy` |
| D14 | **O2** — Cache bucket structure across timesteps (constant-mu fast path) | ✅ done | [mle_batch.py](mle_batch.py) `BatchedDiffusionSolver.solve` `cached_buckets` |
| D15 | **O3** — Cache per-bucket transition kernel + kernel-FFT (constant-mu fast path) | ✅ done | [mle_batch.py](mle_batch.py) `_kernel_fft` + per-bucket `kernel_fft` precompute |
| D16 | **O5** — Cache constant per-trial observations on xp; broadcast instead of `np.tile` | ✅ done | [mle.py](mle.py) `_prepared_session_arrays_for_backend` (already shared with D11) |
| D17 | **O7** — Vectorize across buckets within a timestep (per-trial gather of mass / kernel-FFT). Eliminates the Python inner bucket loop entirely on the constant-mu fast path; one set of batched ops per step regardless of bucket count. | ✅ done | [mle_batch.py](mle_batch.py) `BatchedDiffusionSolver.solve` `per_trial_mass_above` / `per_trial_kernel_fft` |
| D18 | **O7-tv** — Same vectorization for the time-varying-mu path (Decay-Q drift, Decaying-Q-Val noise). Pushes `mu_valid` to xp once, then per timestep calls `_transition_terms_batched` to produce `(b, n_x)` mass and `(b, 2n_x-1)` kernel tensors in a single CuPy dispatch — no bucketing. Removes the per-bucket Python loop that previously dominated Decay-Q runs (and NoiseGain-RewardRate via inflated bucket count). | ✅ done | [mle_batch.py](mle_batch.py) `_transition_terms_batched`, `BatchedDiffusionSolver.solve` time-varying branch |

## What's left — the hot loop today

Tracing one objective call (with the work above already done):

```text
objective_from_population:
    [Phase 1] _compute_latent_population_equal_sessions
        ✅ all on xp, vectorized across (S × n_sessions × trials_per_session)
        ✅ Q/R recurrence is a short loop over trial_pos (≈ 100–300 iterations)

    [Phase 5] batched_choice_rt_loglik → solver.solve
        ❌ allocates (b, n_t) upper_density_valid + lower_density_valid  ← O1
        ❌ recomputes bucket structure every timestep                     ← O2
        ❌ recomputes kernel + kernel-FFT every timestep                   ← O3
        ❌ pulls mu off the device on entry (np.asarray on cupy array)    ← O4
        ❌ rebuilds flat_observed_choice / flat_observed_rt per generation ← O5
```

`b = S × N` ≈ 300 000 valid items. `(b, n_t)` floats = **7 GB per density tensor
× 2 = 14 GB GPU memory wasted per evaluation.** The bucket loop runs roughly
`n_t × n_buckets` ≈ `3000 × 840` ≈ **2.5 million iterations per generation**,
each doing a CPU→GPU sync via `xp.asarray(local)`.

---

## Pending optimizations

### O1 — Drop the `(b, n_t)` density tensor; keep only decision-time gather

**Where:** [mle_batch.py:121-122](mle_batch.py#L121-L122) and the timestep loop body
at [mle_batch.py:152-153](mle_batch.py#L152-L153).

**Why:** `solver.solve` writes every `[trial, t]` density entry, but
`_evaluate_batch` ([mle_batch.py:343-346](mle_batch.py#L343-L346)) only ever
reads `upper_density_xp[trial_idx, decision_idx]` — exactly `b` of the `b × n_t`
values. The rest is allocated, written, and discarded.

**Change sketch:**

```python
# Before the timestep loop, given decision_idx[b]:
upper_at_decision = self.xp.zeros(b, dtype=float)
lower_at_decision = self.xp.zeros(b, dtype=float)
decision_idx_xp = self.xp.asarray(decision_idx)

# Inside the timestep loop, after computing upper_abs/lower_abs for the bucket:
hits = (decision_idx_xp[local_xp] == t_idx)
upper_at_decision[local_xp] = self.xp.where(
    hits, upper_abs / dt, upper_at_decision[local_xp])
lower_at_decision[local_xp] = self.xp.where(
    hits, lower_abs / dt, lower_at_decision[local_xp])
```

`solver.solve` returns these `(b,)` arrays instead of `(b, n_t)`. The gather
in `_evaluate_batch` reduces to `density_at_t = where(is_left, upper_at_decision,
lower_at_decision)` — no row-indexing needed.

**Impact:**
- Frees **~14 GB GPU memory** per evaluation.
- Removes ~6 GB of `xp.zeros` time at the start of each solve.
- Unblocks larger populations on a single GPU (today populations crash with
  OOM on > ~10 GB GPUs).

**Risk:** low. The decision-time gather already exists in
[mle_batch.py:333-347](mle_batch.py#L333-L347); we just push it inside the
timestep loop.

---

### O2 — Cache bucket structure across timesteps

**Where:** [mle_batch.py:136-156](mle_batch.py#L136-L156).

**Why:** the per-trial `(mu, sigma)` is invariant in `t` for constant-mu
variants (Classic, R-only, Q-Val-bias). For Decay-Q variants, while mu(t)
changes, the **bucket membership** (who shares a (μ₀, μ_decay, σ) factor) is
invariant in `t`. The current code calls `np.unique(keys)` and
`xp.asarray(local)` every timestep — burning ~840 CPU→GPU syncs × 3000
timesteps = **2.5 M syncs per generation**.

**Change sketch:** precompute buckets once before the timestep loop:

```python
buckets = _compute_buckets(mu_valid, sigma_valid, is_time_varying)
for bucket in buckets:
    bucket["local_xp"] = self.xp.asarray(bucket["local"])   # one transfer per bucket per fit
    bucket["sigma_t"]  = float(bucket["sigma"])
    bucket["mu_t_fn"]  = lambda t: ...   # constant or per-step expression

for t_idx in range(n_t):
    new_p = self.xp.empty_like(p)
    for bucket in buckets:
        mu_t = bucket["mu_t_fn"](t_idx)
        ...
```

**Impact:** removes 3000× redundant CPU `np.unique` calls and CPU→GPU transfers
of `local`. For cupy this can be tens of milliseconds saved per generation.

**Risk:** moderate. Need to tease bucket setup out into a helper and pass the
prepared buckets through the timestep loop. Numerical results unchanged.

---

### O3 — Cache kernel + kernel-FFT per bucket (constant-mu only)

**Where:** [mle_batch.py:175-200](mle_batch.py#L175-L200) — `_transition_terms`
and `_convolve_rows`.

**Why:** for constant-mu, the kernel `K(mu, sigma)` and the kernel FFT are
invariant in `t`. We currently rebuild them ~840 × 3000 = 2.5 M times per
generation when only ~840 distinct values exist.

**Change sketch:**

```python
# Once per bucket, after O2:
for bucket in buckets_constant_mu:
    bucket["kernel"], bucket["mass_above"], bucket["mass_below"] = self._transition_terms(
        bucket["mu"], bucket["sigma"], bound, dt, dx)
    k_pad = self.xp.zeros(self.fft_n)
    k_pad[:2 * n_x - 1] = bucket["kernel"]
    bucket["kernel_fft"] = self.xp.fft.rfft(k_pad)   # the expensive part

def _convolve_rows_cached(self, p_sub, kernel_fft, n_x):
    p_pad = self.xp.zeros((p_sub.shape[0], self.fft_n), dtype=float)
    p_pad[:, :n_x] = p_sub
    full = self.xp.fft.irfft(
        self.xp.fft.rfft(p_pad, axis=1) * kernel_fft[None, :],
        n=self.fft_n, axis=1)
    return full[:, n_x - 1 : 2 * n_x - 1]
```

**Impact:** for constant-mu, **eliminates the dominant per-step work**: 2.5 M
kernel + kernel-FFT computations collapse to ~840.

**Risk:** trivial. Pure cache; no semantic change.

---

### O4 — Stop forcing mu off the device; skip the (b, n_t) expansion

**Where:** [mle_batch.py:118](mle_batch.py#L118),
[mle_batch.py:385-392](mle_batch.py#L385-L392) (`_as_mu_matrix`).

**Why:** `np.asarray(mu_values, dtype=float)` forces a cupy → numpy transfer of
the whole (S × N × n_t) tensor (~7 GB) on entry. Then for constant-mu it
*expands* a (b,) vector to (b, n_t) on CPU — another ~7 GB allocated to hold
duplicated data.

**Change sketch:**

```python
def _as_mu_matrix(mu_values, n_trials, n_t, xp):
    # Accept xp arrays directly; don't materialize (b, n_t) for constant-mu.
    if hasattr(mu_values, "shape") and mu_values.ndim == 1:
        return mu_values, "constant"      # (b,) on xp, do NOT broadcast
    return mu_values, "time_varying"
```

Bucket code accesses one element per bucket either way (`mu_valid[local[0],
t_idx]` for time-varying or `mu_valid[local[0]]` for constant) — so the bucket
loop only needs to know which branch.

**Impact:** removes a 7-GB transfer for Decay-Q fits and 7 GB of wasted CPU
allocation for constant-mu fits.

**Risk:** low. Affects only the mu-shape handling at the solver entry point.

---

### O5 — Cache constant observations on xp; broadcast instead of tile

**Where:** [mle.py:245-247](mle.py#L245-L247).

**Why:** `np.tile(prepared.choice_left, n_valid)` and friends allocate fresh
`(S × N)` arrays per generation for data that doesn't change. With S=60, N=5000
that's ~24 MB allocated, populated, and pushed to GPU per generation across
three arrays.

**Change sketch:** push the `(N,)` observations to xp once (in
`prepare_mle_data` or a new `_prepared_observations_for_backend`), then use
zero-copy `xp.broadcast_to` rather than physical tile:

```python
choice_xp = backend_arrays["choice_flat_view"]   # zero-copy (N,) on xp
observed_rt_xp = backend_arrays["observed_rt_flat_view"]

# Solver accepts (n_candidates, N) shape directly, or callers use broadcast.
```

**Impact:** modest — eliminates ~24 MB/generation of small transfers + the
Python list-building overhead.

**Risk:** low. The trickiness is in agreeing on the broadcast convention with
the solver (which is currently flat `(S*N,)`).

---

### O6 — Decay-Q factorization (only after O1–O5)

**Where:** would touch both [mle.py](mle.py) `_compute_mu_array` /
`_compute_latent_population_equal_sessions` and the solver's bucket structure
in [mle_batch.py](mle_batch.py).

**Why:** for Decay-Q,
`μ[i, t] = DRIFT_COEF·DV[i] + ΔQ[i]·decay(t)·Q_VAL_COEF`. Trials sharing
`(DRIFT_COEF·DV, ΔQ·Q_VAL_COEF, σ)` share the *entire* mu trajectory.
Typically that's `S × unique_DVs × discretized_ΔQ` buckets — much smaller
than `S × N` if Q-states cluster.

**Impact:** for Decay-Q fits, the per-bucket kernel work (which can't be
cached in O3 because it depends on t) is amortized across many more trials.
Potentially 10–100× for Decay-Q.

**Risk:** higher — couples the solver to the latent factorization.
**Defer** until O1–O5 land and Decay-Q is still the bottleneck.

---

## Suggested order

1. **O1** — biggest single win, smallest blast radius. Unblocks larger
   population batches on a single GPU.
2. **O2 + O4** together — bucket caching naturally couples to "where does
   mu live". Both are needed before O3 has a clean caching surface.
3. **O3** — pure cache layered on top of O2. Drops per-step kernel cost for
   constant-mu variants close to zero.
4. **O5** — polish; small wins.
5. **O6** — only if Decay-Q is still hot after the above.

## Compounding estimate

Today (after D1–D11): population batching gives one solver call per
generation, but each call burns 14 GB GPU memory and 2.5 M Python iterations.

After **O1 alone**: 14 GB freed → can fit 7–10× larger populations in one
solver call. Net throughput ≈ 5–10× on memory-bound fits.

After **O1 + O2 + O3 + O4** on **constant-mu fits**: bucket-loop overhead
falls to ~840 iterations per generation (down from 2.5 M); kernel work to
~840 evaluations (down from 2.5 M). Expected total speedup vs current state:
**~50× per generation** for typical constant-mu fits, dominated by O1 (memory)
and O3 (kernel reuse). Net per-fit speedup will be lower because Phase 1
(latent computation) is still CPU-bounded — see *Future work* below.

## Future work (outside the current scope)

- **Phase 1 latents on GPU end-to-end.** Currently `theta` is pushed to xp at
  the top of `_compute_latent_population_equal_sessions` and stays there.
  Verify nothing in `xp.where`/`xp.clip` paths is silently falling back to
  numpy. The `state_updates.LOG_CIEL`, `LOG_CEIL_MAX` Python floats interact
  fine but mixed scalar/array ops can be slow on cupy.
- **JIT or graph capture.** With O1–O3, the timestep loop becomes a stable
  sequence of small kernels. CUDA Graph capture (cupy 13+) could collapse
  3000 kernel launches into one graph replay per generation.
- **Alternative outer optimizer.** Scipy DE still controls the population on
  CPU and forces param-vector transfers per generation. CMA-ES with on-device
  population (e.g., custom cupy DE) eliminates this floor.
