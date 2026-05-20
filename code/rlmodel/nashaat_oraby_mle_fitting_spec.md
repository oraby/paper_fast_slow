# Specification: Trial-by-trial MLE fitting for Nashaat et al. RL-DDM models

## Purpose

Implement a new maximum-likelihood-estimation fitting path for the Nashaat et al. RL-DDM codebase while preserving the existing simulation/Chi-square fitting path. The new path should adapt the trial-by-trial likelihood logic used by Gupta et al. to the four Nashaat et al. model variants:

1. Classic DDM
2. Q-learning + DDM
3. R-learning + DDM
4. Q-learning + R-learning + DDM

The implementation must keep the existing modeling workflow usable and must support debugging, visualization, and posterior simulation from MLE-fitted parameters.

## Scientific grounding

### Existing Nashaat et al. model family

The Nashaat et al. model currently evaluates four DDM-based variants. The baseline DDM evolves accumulated evidence according to a coherence-dependent drift and Gaussian noise. Q-learning modifies the starting point bias using left/right action values. R-learning modifies the noise gain using a trial-wise reward-rate variable. Q+R combines both effects.

Current Nashaat equations, before the requested noise correction, are effectively:

\[
x_{t+\Delta t} = x_t + k \cdot Cohr_n \cdot \Delta t + s \cdot \epsilon_t \cdot \Delta t
\]

where \(\epsilon_t \sim \mathcal{N}(0,1)\). Q-learning computes:

\[
Q_{chosen,n} = Q_{chosen,n-1} + \alpha(R_{n-1} - Q_{chosen,n-1})
\]

with the unchosen side unchanged, and maps values to a normalized starting point:

\[
Z_n = clip\left(\delta \log\frac{Q_{L,n}}{Q_{R,n}} + offset, -1, 1\right)
\]

R-learning updates reward rate as:

\[
RR_n = RR_{n-1} + \beta(Reward_{n-1} - RR_{n-1})
\]

and uses \(RR_n\) as a noise gain.

### Required noise correction

Both the old simulation path and the new MLE path must use standard diffusion scaling:

\[
x_{t+\Delta t} = x_t + \mu_n \Delta t + \sigma_n \sqrt{\Delta t}\epsilon_t
\]

where:

\[
\mu_n = k \cdot Cohr_n
\]

and:

\[
\sigma_n = s
\]

for Classic and Q-only models, or:

\[
\sigma_n = RR_n \cdot s
\]

for R and Q+R models, unless an explicit compatibility flag is added for reproducing the older \(\sigma\Delta t\) behavior.

Do not silently keep both conventions in the same default path. The requested default is \(\sqrt{\Delta t}\).

### Gupta-style MLE adaptation

The new MLE path should follow the conceptual logic of Gupta et al.: fit model parameters by maximizing the summed log likelihood of observed trial-level behavior. For choice-only fitting:

\[
\log L(\theta)=\sum_n \log P(c^{obs}_n \mid Cohr_n, Q_n, RR_n, \theta)
\]

For choice + RT fitting:

\[
\log L(\theta)=\sum_n \log p(c^{obs}_n, RT^{obs}_n \mid Cohr_n, Q_n, RR_n, \theta)
\]

Unlike the simulation path, MLE state propagation must use the subject's observed choices and outcomes, not sampled model choices and outcomes. Even if the observed choice has low model likelihood, the next trial's Q and R state must be updated from the subject's actual behavioral history.

## Scope

### In scope

- Add MLE fitting side-by-side with the existing fitting method.
- Add a configuration flag to select fitting mode.
- Refactor shared Q/R state updates into shared code.
- Correct old simulation noise scaling to use \(\sqrt{dt}\).
- Using a factory/code-generation function, generate multiple diffusion-function variants as source files
- Implement numerical first-passage density computation for choice + RT likelihood.
- Add generated diffusion-function files for debuggable and optimized variants.
- Add tests proving generated optimized and non-optimized functions are numerically consistent.
- Add visualization utilities for one-trial probability flow and trial likelihood diagnostics.
- Save trial-wise MLE-derived latent variables, including Q and R values, in the output dataframe.
- Allow posterior simulation from MLE-fitted parameters for non-MLE behavioral evaluation.
- The code should be structured to allow other variants of R-learning, e.g. drift-modulated or
bound modulated R-learning.
- Support the full existing `drift.py` × `bias.py` × `noise.py` function family in MLE, including time-varying-drift variants (Decay-Q). The MLE first-passage solver must accept time-varying drift; see resolved-decision G in the Spec Addendum.

### Out of scope unless explicitly decided later

- Adding collapsing bounds.
- Implement choice-only MLE in the initial milestone.

## Repository integration

### Current relevant repository structure

The repository currently places model logic in `code/rlmodel/model/`, with `model_runner.py` providing the CLI entry point. The README states that `model/logic.py` contains the single-run logic and the existing loss function, while `bias.py`, `drift.py`, and `noise.py` contain DDM components. It also states that Q-values and reward-rate updates are currently implemented in `logic.py` as `_calcQVal()`, `_updateNextQL_Q()`, and `_updateNextRewardRate()`.

### Required structure

Add the following files or equivalent modules:

```text
code/rlmodel/model/
    fit.py                         # existing, updated to dispatch fitting mode
    logic.py                       # existing, simulation path preserved
    state_updates.py               # new shared Q/R update logic
    mle.py                         # new MLE objective and optimizer integration
    mle_likelihood.py              # likelihood computation utilities
    first_passage.py               # public wrapper API; dispatches to the appropriate solver
    diffusion/                     # three hand-written first-passage solvers
        __init__.py                # re-exports the three solver functions
        single.py                  # reference, always logs, scalar or array mu, slow, easy to read
        vectorized_const_mu.py     # fast path for constant drift
        vectorized_time_mu.py      # fast path for time-varying drift (Decay-Q)
    mle_visualize.py               # one-trial diagnostics and probability-flow plots
    tests/
        test_state_updates.py
        test_noise_scaling.py
        test_diffusion_equivalence.py
        test_mle_teacher_forcing.py
        test_first_passage_mass_conservation.py
        test_mle_smoke.py
```

The exact test folder can follow the repository's existing test conventions if they exist. If no test framework exists, use `pytest`.

### Dataframe column mapping

The spec uses generic field names; the actual dataframe (from `loadDF` in `model_runner.py`, processed by `_extendTrials` and `_reduceDFSize`) uses fixed column names. Hard-code the following mapping in MLE code — no adapter layer.

| Spec name | Actual column | Notes |
|---|---|---|
| `coherence_signed` | `DV` | signed coherence; existing sign convention |
| `sampling_time` (RT) | `calcStimulusTime` | seconds |
| `observed_reward` | `ChoiceCorrect` | 1=rewarded, 0=not, NaN=no-choice |
| `observed_choice_left` | `ChoiceLeft` | 1=left, 0=right, NaN=no-choice |
| `valid_for_loss` | `valid` AND `ChoiceLeft.notna()` | excludes padding AND no-choice trials |
| `is_padding` | row inserted by `_extendTrials` with `valid=False` | from `model_runner.py:142` |
| no-choice marker | `valid=True` AND `ChoiceLeft.isna()` | real trial, no bound hit |
| session id | `SessId` | from `util.py:47` |

See Spec Addendum (end of document), decision E.

## Configuration and CLI/API behavior

### Required fitting-mode flag

Add a fitting-mode flag to the command-line runner and internal fit API.

Recommended flag:

```bash
python model_runner.py --fit-mode chisq
python model_runner.py --fit-mode mle
```

Allowed values:

- `chisq`: existing simulation + Chi-square loss path.
- `mle`: new trial-by-trial MLE path.

No `--mle-observation-model` flag. Choice-only MLE is out of scope for the initial milestone (Spec Addendum, decision D). The `mle` mode is choice+RT only.

### Backward compatibility

`--fit-mode` is required. Existing calls without `--fit-mode` must fail with a clear error message indicating that no mode was selected. There is no default. See Spec Addendum, decision C.


### Noise-scaling compatibility flag

No compatibility flag. Implement `sqrt_dt` only — per Resolved design decision #7, old numerical results may be overwritten. Record `noise_dt_scaling = "sqrt_dt"` in result metadata so output is self-describing.

## Core model-state behavior

### Shared state-update module

Move Q/R updates into `state_updates.py` so both simulation and MLE use identical state equations.

Required functions:

```python
def initialize_latent_state(include_Q: bool, include_RewardRate: bool) -> LatentState:
    ...


def update_q_values(
    q_left: float,
    q_right: float,
    observed_choice_left: bool | None,
    observed_reward: float | int | None,
    alpha: float,
) -> tuple[float, float]:
    ...


def update_reward_rate(
    reward_rate: float,
    observed_reward: float | int | None,
    beta: float,
) -> float:
    ...


def compute_starting_point_z(
    q_left: float,
    q_right: float,
    delta: float,
    offset: float,
    include_Q: bool,
) -> float:
    """Return starting point z in [-bound, bound] units (current bound=1, so [-1, 1]).

    If include_Q is False, returns 0 unconditionally — Classic and R-only variants
    have no starting-point bias in MLE. `offset` is only meaningful when include_Q=True.
    See Spec Addendum, decision F.
    """
    ...


def compute_trial_mu(
    coherence: float,
    drift_coef: float,
    time_grid: np.ndarray | None = None,
) -> float | np.ndarray:
    """Return drift for a trial.

    For constant-drift variants (Classic, R-only, Q-Val-bias) returns a scalar.
    For Decay-Q drift variants returns a per-timestep array of length len(time_grid).
    See Spec Addendum, decision G.
    """
    ...


def compute_trial_sigma(
    base_sigma: float,
    reward_rate: float,
    include_RewardRate: bool,
) -> float:
    ...
```

Use the same initial values as the existing model:

- `Q_left = 0.5`
- `Q_right = 0.5`
- `RewardRate = 0.5`

### Handling invalid and padded trials

Preserve existing behavior unless explicitly decided otherwise:

- Invalid trials should not contribute to the MLE loss.
- Invalid trials may still update Q/R if the existing code and dataframe semantics say they represent real behavioral trials.
- Padding trials should not contribute to the loss. It can update Q/R as this operation
doesn't affect final computation.

This is a design-decision risk: the coding agent must inspect the current dataframe columns and existing behavior before finalizing invalid/padded-trial handling.

## Old simulation path changes

### Noise scaling

Update every old DDM simulation path to use:

```python
x_next = x + mu * dt + sigma * np.sqrt(dt) * rng.normal()
```

rather than:

```python
x_next = x + mu * dt + sigma * dt * rng.normal()
```

For R-learning models:

```python
sigma = reward_rate * NOISE_SIGMA
x_next = x + mu * dt + sigma * np.sqrt(dt) * rng.normal()
```

### Keep old loss unless `fit_mode=mle`

The old path should continue to:

- simulate model choices and RTs,
- propagate simulated choices/outcomes for synthetic behavior,
- compute the existing Chi-square loss,
- save outputs in the existing format.

The only required behavioral change to the old path is noise scaling.

## New MLE path behavior

### Teacher-forced trial loop

MLE evaluation must be deterministic for a fixed parameter vector, subject dataframe, and numerical grid. It must not sample model choices for state propagation.

Pseudo-code:

```python
def neg_loglik(params, df, model_config):
    state = initialize_latent_state(
        include_Q=model_config.include_Q,
        include_RewardRate=model_config.include_RewardRate,
    )
    rows = [] if model_config.return_df else None
    total_loglik = 0.0

    for trial in iter_trials_by_session(df):
        z = compute_starting_point_z(
            state.q_left,
            state.q_right,
            delta=params.delta,
            offset=params.offset,
            include_Q=model_config.include_Q,
        )
        mu = compute_trial_mu(trial.coherence_signed, params.drift_coef)
        sigma = compute_trial_sigma(
            params.noise_sigma,
            state.reward_rate,
            model_config.include_RewardRate,
        )

        if trial.valid_for_loss:
            ll = trial_choice_rt_loglik(
                observed_choice=trial.choice,
                observed_rt=trial.sampling_time,
                z=z,
                mu=mu,
                sigma=sigma,
                bound=params.bound,
                non_decision_time=params.non_decision_time,
                dt=model_config.dt,
                tmax=model_config.t_dur,
                dx=model_config.dx,
                diffusion_backend=model_config.diffusion_backend,
            )
            total_loglik += ll
        else:
            ll = np.nan

        if rows is not None:
            rows.append(record_trial_state_and_likelihood(...))

        state = update_state_from_observed_trial(
            state=state,
            observed_choice=trial.choice,
            observed_reward=trial.reward,
            alpha=params.alpha if include_Q else None,
            beta=params.beta if include_RewardRate else None,
            is_padding=trial.is_padding,
            is_behavioral=trial.is_behavioral,
        )

    return -total_loglik, maybe_dataframe
```

### Choice + RT likelihood

For each trial, compute first-passage densities:

\[
f_{upper,n}(t)
\]

and:

\[
f_{lower,n}(t)
\]

where `upper` and `lower` correspond to the two choice bounds. The likelihood contribution is:

\[
L_n = f_{observed\_bound,n}(RT^{obs}_n - T_0)
\]

if using fixed non-decision time \(T_0\). Then:

\[
\ell_n = \log(\max(L_n, \epsilon))
\]

Use a small floor such as `1e-300` to avoid `log(0)`.

### Non-decision time

Initial implementation should use the Nashaat scalar `NON_DECISION_TIME` as a fixed shift:

\[
DT_n = RT^{obs}_n - T_0
\]

If \(DT_n \leq 0\), return `log(1e-300)` (the value of `log(loglik_floor)`) for that trial.

Do not implement Gupta's full inverse-Gaussian non-decision-time model unless explicitly requested later.

### No-choice trials

Resolved (see Resolved design decisions #3): treat no-choice trials as no-bound-hit. For MLE, the likelihood contribution is the survival mass at `Tmax`:

\[
L_n = S(Tmax) = 1 - \int_0^{Tmax} f_{upper}(t)dt - \int_0^{Tmax} f_{lower}(t)dt
\]

A no-choice trial is identified by `valid=True` AND `ChoiceLeft.isna()` per the dataframe column mapping above. Padding trials (`valid=False`) are excluded from the loss entirely. Q/R state still updates from observed history for no-choice trials when reward is known.

## First-passage density implementation

### Required public API

In `first_passage.py` expose:

```python
def first_passage_density(
    z: float,
    mu: float | np.ndarray,
    sigma: float,
    bound: float,
    dt: float,
    dx: float,
    tmax: float,
    backend: str = "single",
    return_log: bool = False,
) -> FirstPassageResult:
    """`mu` may be a scalar (constant drift) or an array of length tmax/dt
    (time-varying drift, for Decay-Q variants). See Spec Addendum, decision G."""
    ...
```

Return object:

```python
@dataclass
class FirstPassageResult:
    times: np.ndarray
    f_upper: np.ndarray
    f_lower: np.ndarray
    survival: np.ndarray
    x_grid: np.ndarray | None = None
    p_by_t: np.ndarray | None = None
    upper_mass_by_t: np.ndarray | None = None
    lower_mass_by_t: np.ndarray | None = None
    metadata: dict | None = None
```

### Numerical method

Use discrete probability-mass propagation equivalent to a Fokker-Planck finite-difference / transition-matrix method:

1. Discretize the accumulator axis between lower and upper absorbing bounds.
2. Initialize all mass at the starting point bin.
3. For each time step, redistribute mass from each current bin according to the Gaussian transition:

\[
x_{t+\Delta t} \mid x_t \sim \mathcal{N}(x_t + \mu \Delta t, \sigma^2 \Delta t)
\]

4. Count mass crossing the upper and lower bounds during that time bin.
5. Remove absorbed mass from the active distribution.
6. Continue until `tmax` or until survival mass is negligible.

The returned densities are:

\[
f_{upper}(t_i) \approx \frac{P(upper\ crossing\ in\ bin\ i)}{\Delta t}
\]

\[
f_{lower}(t_i) \approx \frac{P(lower\ crossing\ in\ bin\ i)}{\Delta t}
\]

### Mass conservation test

For each first-passage call:

\[
\sum_i f_{upper}(t_i)\Delta t + \sum_i f_{lower}(t_i)\Delta t + survival(Tmax) \approx 1
\]

Tests must verify this within a numerical tolerance.

## Diffusion solvers

### Purpose

The codebase needs three first-passage solver implementations: a slow readable reference (for visualization, debugging, and as the equivalence-test oracle) and two fast paths (one per drift shape) used during MLE fitting. Earlier drafts proposed code generation; this is now superseded by the hand-written paired-function design below.

### Why hand-written, not generated

The original codegen rationale was to avoid `if logged:` branches inside hot loops. Hand-written paired functions achieve the same outcome — the reference always logs and is intentionally slow; the fast paths never log. The choice happens at the call site (which function is invoked), not inside any loop. Benefits:

- Straight Python — no template indirection. Standard IDE tooling (jump-to-def, debugger, linter, type checker) works directly.
- No regeneration step in the dev loop, no auto-gen header bookkeeping, no risk of stale generated files in git.
- Future GPU swap (`xp=numpy → xp=cupy`) is one keyword argument per function. Adding `xp` as a templating axis would clutter a codegen system unnecessarily.
- Fewer tests — only numerical equivalence between reference and fast paths matters.

Cost: ~10–30 lines of duplicated diffusion math between the reference and each fast path. The math is short enough that this is cheaper than the codegen indirection.

### Three hand-written solvers

All three live in `model/diffusion/` as normal Python (no auto-gen header):

1. **`single.py`** — reference implementation
   - Single-trial, non-vectorized, **always logs** (returns the full `FirstPassageResult` with `x_grid`, `p_by_t`, `upper_mass_by_t`, `lower_mass_by_t` populated)
   - Handles both scalar and array `mu`
   - Intended for visualization, debugging, and as the equivalence-test oracle
   - Intentionally slow; not used for fitting
   - **No separate `diffusion_single_logged.py` twin.** Logging is built in. Memory cost is bounded by `tmax/dt × N_x_bins` (~5 MB at T_max=3s, dt=1ms, 200 bins) — negligible because the reference is never used for fitting.

2. **`vectorized_const_mu.py`** — fast path for constant drift
   - `mu_scalar` must be a scalar
   - Precomputes the Gaussian transition kernel once and reuses it across all timesteps
   - Returns the core `FirstPassageResult` fields (`times`, `f_upper`, `f_lower`, `survival`); logging fields left as `None`
   - Intended as the fitting-mode default for Classic, R-only, and Q-Val-bias variants

3. **`vectorized_time_mu.py`** — fast path for time-varying drift (Decay-Q)
   - `mu_array` must be a 1-D array of length `tmax/dt`
   - Uses kernel-shift trick or FFT convolution to handle per-timestep kernel changes
   - Returns the core `FirstPassageResult` fields; logging fields left as `None`
   - Intended as the fitting-mode default for Decay-Q variants
   - Expected runtime ~3–10× slower than `vectorized_const_mu`

The dispatcher in `first_passage.py` selects `const_mu` vs `time_mu` automatically based on the shape of `mu` (scalar vs array). Equivalence tests in `test_diffusion_equivalence.py` cover both fast paths against the reference.

Optional future extension: `diffusion_batch_trials.py` for vectorizing across multiple trials sharing the same grid — only sensible if many trials can be grouped efficiently. Not in initial milestone.

### Function signatures

```python
import numpy as np

def diffusion_single(
    z, mu, sigma, bound, dt, dx, tmax, *, xp=np,
) -> FirstPassageResult:
    """Reference implementation. Slow, always logs, scalar or array mu.

    Solves the discrete approximation to:

        dx = mu dt + sigma dW_t

    with absorbing bounds at +/- bound and starting point z. Over a discrete
    time step: x_{t+dt} | x_t ~ Normal(x_t + mu*dt, sigma^2*dt). The stochastic
    increment is sigma*sqrt(dt)*epsilon, epsilon ~ Normal(0, 1). Probability
    mass crossing either bound is removed and recorded as first-passage mass
    for the corresponding bound and time bin.

    For constant-drift variants (Classic, R-only, Q-Val-bias), `mu` is a scalar.
    For Decay-Q drift variants, `mu` is a per-timestep array of length tmax/dt.

    `xp` is the array module — defaults to numpy. Pass `cupy` (or any
    numpy-API-compatible module) for GPU. Gaussian PDF is computed via
    `xp.exp(...)` directly (not scipy.stats) so the function works with both
    numpy and cupy arrays. Not exercised in initial PR — placeholder for the
    future GPU path.
    """

def diffusion_vectorized_const_mu(
    z, mu_scalar, sigma, bound, dt, dx, tmax, *, xp=np,
) -> FirstPassageResult:
    """Fast path for constant drift. `mu_scalar` must be a scalar."""

def diffusion_vectorized_time_mu(
    z, mu_array, sigma, bound, dt, dx, tmax, *, xp=np,
) -> FirstPassageResult:
    """Fast path for time-varying drift. `mu_array` must be a 1-D array of length tmax/dt."""
```

### Tests for the three solvers

Two test files cover all behavior:

- `test_diffusion_equivalence.py` — for several `(z, mu, sigma)` settings, run all three solvers and verify `f_upper`, `f_lower`, `survival` agree within `rtol=1e-6`, `atol=1e-9`. Constant-drift inputs are compared via `single` and `const_mu`; time-varying inputs via `single` and `time_mu`.
- `test_first_passage_mass_conservation.py` — for each backend, verify `Σf_upper·dt + Σf_lower·dt + survival(Tmax) ≈ 1` within `atol=1e-6`.

If the vectorized implementations use a mathematically equivalent but numerically different method (e.g., FFT convolution), document and loosen tolerance explicitly in the test.

## Visualization utilities

### Required functions

Create `mle_visualize.py` with:

```python
def visualize_first_passage_result(result: FirstPassageResult, observed_choice=None, observed_rt=None, save_path=None):
    ...
```

This should plot:

- upper first-passage density over time,
- lower first-passage density over time,
- survival mass over time,
- optional observed RT marker,
- optional observed choice annotation.

For logged results, add:

```python
def visualize_probability_flow(logged_result: FirstPassageResult, save_path=None):
    ...
```

This should visualize the probability distribution over accumulator state and time, with absorbing bounds marked.

Add one utility function:

```python
def debug_one_trial_mle_flow(df, trial_index, params, model_config, save_dir=None):
    ...
```

This function should:

1. reconstruct Q/R state up to the requested trial using observed history,
2. compute `z`, `mu`, and `sigma` for that trial,
3. call the logged first-passage backend,
4. compute the trial likelihood,
5. call the visualization functions,
6. return a structured object containing the trial state, likelihood, and paths to saved figures.

## Output dataframe and saved artifacts

### MLE trial-level dataframe

When using MLE, the returned/saved dataframe must include trial-wise latent and likelihood values.

Required columns, names can be adapted to existing naming style:

```text
mle_Q_left_before
mle_Q_right_before
mle_Q_rel_before
mle_reward_rate_before
mle_z
mle_mu
mle_sigma
mle_decision_time_observed
mle_choice_prob_or_density
mle_loglik
mle_valid_for_loss
mle_survival_at_tmax
mle_upper_hit_prob_tmax
mle_lower_hit_prob_tmax
```

Also include after-update columns if useful:

```text
mle_Q_left_after
mle_Q_right_after
mle_reward_rate_after
```

### Saved fit-result structure

Extend current pickle structure with MLE-specific fields:

```python
{
    "fit_mode": "mle",
    "mle_observation_model": "choice_rt",
    "subject_df": original_df,
    "mle_df": dataframe_with_latents_and_loglik,
    "dt": ...,
    "dx": ...,
    "t_dur": ...,
    "noise_dt_scaling": "sqrt_dt",
    "include_Q": ...,
    "include_RewardRate": ...,
    "params_names": ...,
    "params_init": ...,
    "params_bounds": ...,
    "OptimRes": ...,
    "loglik": ...,
    "neg_loglik": ...,
    "n_trials_loss": ...,
    "n_trials_total": ...,
    "aic": ...,
    "bic": ...,
}
```

AIC and BIC are recommended because MLE provides a natural likelihood basis for model comparison:

\[
AIC = 2k - 2\log L
\]

\[
BIC = k\log N - 2\log L
\]

where `k` is the number of fitted parameters and `N` is the number of valid likelihood-contributing trials.

## Posterior simulation from MLE parameters

### Requirement

Evaluate whether it is sensible to take MLE-fitted parameters and run the model in simulation mode to generate synthetic behavior that can be evaluated with the existing non-MLE behavioral metrics.

### Recommended design

Yes, this is sensible and should be supported, but it should be treated as a posterior predictive check, not as part of the MLE objective.

Add a function:

```python
def simulate_from_fitted_params(
    df,
    fitted_params,
    model_config,
    n_repeats=1,
    seed=None,
    use_observed_history_for_inputs=False,
):
    ...
```

Default recommendation:

- Use the existing simulation path.
- Use MLE-fitted parameters.
- Let simulated choices/outcomes propagate simulated Q/R states, because this is a generative posterior predictive simulation.
- Save simulated outputs in the same format used by the existing Chi-square workflow so existing non-MLE performance plots can be reused.

Alternative option, for debugging only:

- `use_observed_history_for_inputs=True` can keep Q/R fixed to observed-history values while sampling only the current trial. This should not be described as a full synthetic session.

### Important distinction

MLE fitting uses observed-history propagation.

Posterior simulation uses simulated-history propagation.

This distinction must be explicit in code comments and documentation.

## Refactoring guidance for `logic.py`

Before creating a separate MLE logic class, inspect whether the existing `logic.py` flow can be preserved by branching only at three points:

1. diffusion function call,
2. loss / likelihood calculation,
3. source of choices/outcomes for state propagation.

If those are the only differences, keep the function structure and add clear flow control:

```python
if fit_mode == "chisq":
    # existing simulation behavior
elif fit_mode == "mle":
    # observed-history likelihood behavior
else:
    raise ValueError(...)
```

If adding these branches makes `logic.py` hard to read, create `mle.py` and keep `logic.py` focused on simulation. Shared state updates must still live in `state_updates.py`.

Do not duplicate Q/R update code between paths.

## Optimization

### Existing optimizer

The existing code uses differential evolution. Keep this as the initial optimizer for MLE unless it proves too slow.

### MLE objective numerical safeguards

Implement:

- parameter bounds matching existing `initvals.py`,
- `loglik_floor = 1e-300`,
- invalid parameter handling returning large finite loss,
- checks for NaN/Inf in density outputs,
- optional caching of first-passage transition matrices for repeated `mu`, `sigma`, `z` grids only if performance requires it.

### Performance risks

Choice + RT MLE can be much slower than the existing simulation loss because each trial may require a first-passage density computation. The coding agent should first implement the clear single-trial reference version, then optimize.

Potential optimization strategies:

- precompute grid cell boundaries and normal-CDF matrices,
- vectorize transition probability computation,
- cache transition matrices for repeated `mu`/`sigma` values,
- group trials by same or rounded `mu`, `sigma`, `z`,
- use Numba only if already acceptable for the project environment.

Do not introduce new heavy dependencies without approval.

## Tests

### Required tests

1. `test_noise_scaling.py`
   - Verify stochastic simulation uses `sigma * sqrt(dt)`, not `sigma * dt`.
   - Use a large number of one-step samples and check empirical variance approximately equals `sigma**2 * dt`.

2. `test_state_updates.py`
   - Verify Q update changes only the chosen side.
   - Verify R update follows `RR_new = RR_old + beta * (reward - RR_old)`.
   - Verify initial values are 0.5.

3. `test_diffusion_generated_equivalence.py`
   - Compare generated single, logged, and vectorized functions.

4. `test_first_passage_mass_conservation.py`
   - Verify absorbed upper + absorbed lower + survival approximately equals 1.

5. `test_mle_teacher_forcing.py`
   - Build a tiny fake dataframe where model-predicted choice would differ from observed choice.
   - Verify Q/R update uses observed choice/outcome, not sampled/model choice.

6. `test_mle_smoke.py`
   - Run MLE objective on a small synthetic dataset for all four models.
   - Assert finite loss and expected output columns.

7. Optional `test_posterior_simulation.py`
   - Fit or mock fitted parameters, run posterior simulation, assert existing plot/evaluation-compatible columns are present.

## Resolved design decisions

The following design decisions are now fixed for the initial implementation.

1. **MLE observation model**
   - Implement **choice + RT MLE** as the primary MLE observation model.
   - The likelihood contribution for a valid behavioral trial is the first-passage density of the observed bound evaluated at the observed decision time.

2. **Non-decision time model**
   - Follow the existing Nashaat et al. implementation.
   - Use a fixed non-decision time scalar.
   - Compute decision time as:

```text
DT_n = RT_obs_n - T0
```

   - Do not implement a non-decision-time distribution in the initial version.

3. **No-choice / no-bound-hit trials**
   - Follow the existing Nashaat et al. design as closely as possible.
   - Treat no-choice trials as **no-bound-hit trials**.
   - For MLE, the likelihood should use the survival/no-absorption mass at `Tmax`:

```text
L_n = S(Tmax)
S(Tmax) = 1 - integral_0_Tmax f_upper(t) dt - integral_0_Tmax f_lower(t) dt
```

   - If the existing dataframe has a clear no-choice marker, use it. If it does not, add an explicit mapping layer rather than inferring silently from unrelated columns.

4. **Invalid and padded trials**
   - Preserve the existing design used to support parallelized sessions.
   - Invalid/padding trials should be represented/analyzed in the output stream as needed, but excluded from loss calculations.
   - They should be handled consistently with the current session-parallelization logic.
   - The coding agent must inspect existing invalid/padding semantics and preserve them rather than redesigning them.

5. **Bound parameter**
   - Keep the bound fixed in the initial MLE implementation to mirror the current non-MLE implementation.
   - The implementation should be flexible enough to support a future bound function, but this future function should not be implemented now.
   - Add a clear internal interface such as:

```python
def compute_bound_for_trial(trial, state, params, model_config):
    return model_config.fixed_bound
```

   This keeps the future extension point explicit without changing the current model.

6. **R-learning noise-gain formula**
   - Initially leave the R-learning noise formula as in Nashaat et al.:

```text
sigma_n = RR_n * s
```

   - Do not add a minimum noise offset in the initial implementation.
   - Keep the code structured so a later minimum offset can be added explicitly, e.g.:

```python
sigma = reward_rate * base_sigma
# future optional extension: sigma = sigma_min + reward_rate * base_sigma
```

7. **Old results after sqrt(dt) correction**
   - It is acceptable for new runs to overwrite old results.
   - No backward-compatible old-result preservation is required.
   - Still record `noise_dt_scaling = "sqrt_dt"` in new result metadata so future outputs are self-describing.

8. **Multi-trial optimized backend**
   - A multi-trial optimized diffusion backend is optional.
   - If it is not feasible or would overcomplicate the implementation, the fitting function in the logic/MLE module should accept multiple trials but process them one by one internally.
   - The API should not require callers to manually loop over trials.

## Remaining design notes

The following are no longer blocking but should remain explicit in code comments and documentation.

1. **Posterior simulation from MLE parameters**
   - MLE fitting uses observed-history propagation.
   - Posterior simulation from fitted MLE parameters should use simulated-history propagation because it is a generative check.

2. **Future bound function**
   - The current bound is fixed.
   - Future support for dynamic/session/trial-dependent bounds should use the `compute_bound_for_trial(...)` extension point.

3. **Future noise floor**
   - The current R-learning noise formula has no floor.
   - Future support for a minimum diffusion offset should be an explicit model variant or configuration option.

## Acceptance criteria

The implementation is complete when:

1. `model_runner.py` can run both `chisq` mode (explicit `--fit-mode chisq`) and new `mle` mode (explicit `--fit-mode mle`). Calls without `--fit-mode` fail with a clear error.
2. Existing simulation path still runs and now uses `sqrt(dt)` noise scaling.
3. MLE path fits every drift × bias × noise combination supported by the chisq path, including Decay-Q drift variants. Bias support is restricted to `None_` and `Q-Val` / `Q-Val (Offset)` per Spec Addendum decision F.
4. MLE path propagates Q/R states using observed choices and outcomes.
5. MLE outputs contain trial-wise Q, R, z, mu, sigma, likelihood, and valid-loss flags.
6. Three diffusion solvers (`single`, `vectorized_const_mu`, `vectorized_time_mu`) exist as hand-written source files in `model/diffusion/`. No auto-gen headers, no regeneration command.
7. All three diffusion solvers produce numerically equivalent first-passage densities within `rtol=1e-6`, `atol=1e-9`.
8. First-passage density passes mass-conservation tests, including for time-varying drift.
9. One-trial visualization/debug function works and saves interpretable figures.
10. MLE-fitted parameters can be passed to a posterior simulation function and evaluated using existing non-MLE behavioral metrics. The user can specify the saved results of the recovered MLE parameters and the code uses them to run the model in simulation mode to generate synthetic behavior evaluable with the existing non-MLE behavioral metrics. This is treated as a posterior predictive check, not as part of the MLE objective. May require adding a new script and utility functions.

---

## Spec Addendum

The following items resolve open ambiguities and contradictions in earlier sections. Where this addendum conflicts with earlier text, the addendum wins.

### A. Variant coverage

MLE covers the full existing drift × bias × noise family from `drift.py`, `bias.py`, `noise.py` — **including Decay-Q drift variants**. The earlier "Out of scope: Implement Decaying-Q model variants" line has been superseded and removed.

### B. Initial milestone

The initial milestone implements the full family in one go, including Decay-Q drift variants. No incremental staging.

### C. Default `--fit-mode`

`--fit-mode` is required. Existing scripts that omit it must fail with a clear error message. No silent default.

### D. Choice-only MLE

Out of scope for the initial milestone. Initial implementation supports choice+RT only. No `--mle-observation-model` flag. CLI is just `--fit-mode {chisq, mle}`.

### E. Dataframe column mapping

Hard-coded in MLE code. No adapter layer.

| Spec name | Actual column | Notes |
|---|---|---|
| `coherence_signed` | `DV` | signed coherence; sign convention from existing code |
| `sampling_time` (RT) | `calcStimulusTime` | seconds |
| `observed_reward` | `ChoiceCorrect` | 1=rewarded, 0=not, NaN=no-choice |
| `observed_choice_left` | `ChoiceLeft` | 1=left, 0=right, NaN=no-choice |
| `valid_for_loss` | `valid` AND `ChoiceLeft.notna()` | excludes padding AND no-choice trials |
| `is_padding` | row inserted by `_extendTrials` with `valid=False` | from `model_runner.py:142` |
| no-choice marker | `valid=True` AND `ChoiceLeft.isna()` | real trial, no bound hit |
| session id | `SessId` | from `util.py:47` |

### F. Starting point `z` when `include_Q=False`

`z = 0` (centered, no bias). `compute_starting_point_z(..., include_Q=False)` returns 0 unconditionally. `offset` is only fit when `include_Q=True`. Existing chisq-path "Fixed (Dir)" / "Fixed (Corr/Incorr)" bias functions are not supported in MLE — only `None_` and `Q-Val` / `Q-Val (Offset)` bias functions are supported in the initial milestone.

### G. Decay-Q in first-passage solver (time-varying drift)

The first-passage solver API accepts `mu` as either a scalar (constant drift) or an array of length `tmax/dt` (time-varying drift, for Decay-Q). `compute_trial_mu` returns `float | np.ndarray` depending on the variant.

Required generated fast paths:

1. `diffusion_single.py` — handles scalar OR time-varying `mu`, reference implementation
2. `diffusion_single_logged.py` — same, with state logging
3. `diffusion_vectorized_const_mu.py` — fast path for constant-drift variants (Classic, R-only, Q-Val-bias)
4. `diffusion_vectorized_time_mu.py` — fast path for Decay-Q variants (kernel-shift or FFT convolution)

The dispatcher in `first_passage.py` selects `const_mu` vs `time_mu` based on whether the variant's drift is time-varying. Equivalence tests cover both fast paths against the single reference.

Performance expectation: Decay-Q variants run ~3–10× slower than constant-drift variants in the optimized backend.

