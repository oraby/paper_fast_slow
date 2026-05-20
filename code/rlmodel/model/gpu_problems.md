I've read enough. Here's a focused review of red flags ranked by likely impact. Five of these will defeat GPU speedup even on a CUDA box; the rest matter mostly once you fix the first ones.

Critical (will cause "no speedup" even with a working GPU)
1. Multi-process workers fight for the GPU
fit.py:455-457 creates a multiprocessing.Pool(num_cpus) and passes _pool.map as workers to differential_evolution. Each pool process:

imports CuPy and initializes its own CUDA context on the same device
has ~300-500 MB of context overhead
serializes on the GPU kernel queue (you don't get parallelism across processes, you get queuing)
can OOM the GPU at popsize=100
For GPU runs you want one process, one context. Pass workers=1 (or --num-cpus 1) when mle_array_backend=cupy. Better: when model_config.mle_array_backend == "cupy", force workers=1 in simulateDDM.

** Let's make it such that if cupy then the default behavior is to error on non
*" existent CuPy and it set numbers of CPUs to 1 unless otherwise specified by
** the user. i.e the right settings should work out of the box.

2. The auto backend silently runs the slowest path
mle.py:325-332:


if backend.actual_backend == "cupy" or model_config.mle_use_batched_likelihood:
    return _evaluate_trial_likelihoods_batched(...)
likes, info = _evaluate_trial_likelihoods_rowwise(...)  # Python for-loop per trial
On a non-CUDA host with --mle-array-backend=auto, the resolver falls back to numpy (correct), but mle_use_batched_likelihood defaults to False, so you run the rowwise loop. The rowwise loop calls first_passage_density per trial — ~N_trials separate scipy/numpy passes with all the FirstPassageResult allocation overhead.

If you're benchmarking "GPU vs CPU" by toggling --mle-array-backend, you're actually comparing two different algorithms, not two backends. The numpy side is at a structural disadvantage. Make _evaluate_trial_likelihoods_batched the default for numpy too — that's what the equivalence test in your plan is for.

** let's do what's suggested here and
** disable silent fallback

3. Per-objective iterrows() dominates wall time
mle.py:94: every objective call re-walks the dataframe with iter_df.iterrows(), building a Python list of dicts per trial. Differential evolution calls this popsize × generations times — easily 10,000+ evaluations per fit. Per-call cost: ~5000 trials × (iterrows + dict build + dataclass) ≈ tens of milliseconds purely in Python. That alone can be 80%+ of wall time, capping any GPU benefit at a small constant factor.

Fix: precompute per-trial features (DV, valid, ChoiceLeft, ChoiceCorrect, calcStimulusTime, SessId, sorted index) once per fit, store in numpy arrays. Inside the objective, only the parameter-dependent quantities (z[i], mu[i], sigma[i]) need recomputing — and that's vectorizable over trials.

** let's do that

4. The batched kernel is O(B · n_x² · n_t), not O(B · n_x · n_t)
mle_batch.py:139-156:


standardized_edges = (
    edges[None, None, :]
    - x_grid[None, :, None]
    - mean_shift[:, None, None]
) / sigma_sqrt_dt[:, None, None]
cdf_at_edges = _normal_cdf(xp, standardized_edges)   # (B, n_x, n_x+1)
...
p = xp.einsum("bs,bst->bt", p, bin_mass)             # B × n_x × n_x
At typical sizes (B=1024, n_x=200, n_t=3000) that's a (1024, 200, 201) tensor per step → 41 M doubles → 328 MB per step → reallocated 3000 times per objective. Even on a fast GPU this is mostly memory-bandwidth-bound.

vectorized_const_mu.py already does O(n_x · n_t) per trial via convolution / kernel-shift. The batched path doesn't reuse that trick — but it could, since trials sharing (sigma, mu_scalar) share the kernel exactly. For Decay-Q trials with the same coherence × DRIFT_COEF, the per-step mu(t) is shared too. Bucket trials by (sigma, mu) (or hash a quantized version of mu), precompute one kernel per bucket per step, and convolve over the bucket's p[B, n_x] matrix.

** alright, let's do that. if it helps to reuse buffers, let's create
** the vectorized function as a class that holds it's buffers.
** it can reallocate the buffer if the size changes
** another optimization that's worth investigating, is whether we can use
** function results caching, but we need to see
** how does that conflict with GPU memory

5. Lazy CuPy initialization makes you measure compile time
First call to cupy_ndtr, xp.einsum, xp.linspace etc. each JIT-compiles a kernel — easily 1-3 s each on first run. The resolve_array_backend warm-up only touches cp.asarray([1.0]) and cp.cuda.Stream.null.synchronize; it does NOT exercise the kernels used in the hot loop. If your benchmark is "one fit on one subject" without a warm-up evaluation, the first generation of DE pays the entire JIT bill.

Add a warm-up call right after backend resolution that runs _evaluate_batch once on a tiny synthetic (e.g., B=4, n_t=10, n_x=10) just to prime the kernel cache.

** I don't care about this for now. we can leave for later or not at all

High-impact (improve GPU efficiency once the above are fixed)
6. asnumpy after every batch forces device→host sync
mle_batch.py:158-162: five asnumpy calls per batch, each an implicit cudaStreamSynchronize. With ~5 batches per evaluation × 10,000 evaluations = 50,000 syncs per fit. Either:

defer all asnumpy until after the per-objective loop (collect into a single GPU buffer, transfer once), or
keep loglik on GPU and only sync the scalar total_loglik at the end.

** Let's defer numpy conversions

7. mu_values stored as dtype=object
mle.py:376:


mu_values=np.asarray([inp["mu"] for inp in likelihood_inputs], dtype=object),
Object arrays hold Python references, defeat numpy vectorization, and force _mu_matrix to do a per-trial Python loop + np.vstack per batch (CPU). For const-mu trials, mu is a float; for Decay-Q, it's a (n_t,) array. Branch once on uses_decay_q_drift/uses_decay_q_noise and build a (N, n_t) matrix (or a (N,) scalar array) once per fit, in numpy, no object dtype.

** Let's do as suggested but add inline documentation to aid the reader

8. _evaluate_batch rebuilds x_grid, edges, z_grid on every call
mle_batch.py:121-127: x_grid, edges, and the argmin-based z_idx_cpu are computed from scratch every batch. They're fixed by (bound, dx, n_x) and z only. Precompute and cache.

** let's leave that for later

9. cupyx.scipy.special.ndtr not exercised at startup
mle_batch.py:194-197: the except Exception swallows everything, so if cupyx.scipy.special.ndtr ever raises mid-run it silently switches to 0.5 * (1 + xp.erf(...)). Also: cupy.erf doesn't exist at the top level — xp.erf would fail on CuPy. (You'd need cupy.special.erf or cupyx.scipy.special.erf.) Probe both at backend-resolve time and pin which path you're using.

** Let's remove the silent handling and make sure to test the GPU path at startup. If cupyx.scipy.special.ndtr is unavailable, error immediately with a clear message.

10. State updates done in Python over state_by_sess
mle.py:94-188: the teacher-forcing loop touches a Python dict state_by_sess per trial. Sessions are independent — you can vectorize Q/R updates per-session using pandas.groupby('SessId').agg(...) or a numba/cython loop. For ~5000 trials × thousands of evaluations the savings are large.

** Let's do that

Medium
11. Batch size 1024 may be too small
With 5000 trials and 200 bins, full GPU memory for the (B, n_x, n_x+1) tensor at B=5000 is ~1.6 GB. On most GPUs that fits. Bigger batch = fewer kernel launches and asnumpy syncs. Default 1024 is conservative; expose it more prominently or auto-tune based on available GPU memory.

** Let the user specify the number of GBs (not MBs or bytes) to use, and then
** we can compute the batch size from that and report it at startup.

12. differential_evolution(disp=True) and polish=True
disp=True printing per-generation can ironically dominate when the objective is fast. And polish=True runs a sequential L-BFGS-B refinement at the end, which is single-threaded on CPU and doesn't benefit from your GPU path at all. For benchmarking, set disp=False, polish=False.

** No, let's keept it as it is. Real-world performance, not benchmarking
** is what we care about it for now.

13. lru_cache on resolve_array_backend is process-local
With multiprocessing.Pool, each child re-resolves the backend and re-warms CuPy. See item #1.

** Also not for now, we intend to have a single process for GPU runs.


