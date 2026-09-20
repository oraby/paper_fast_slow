# Methods revision — Model & model↔neural correlate

Replacement text for the *Model* subsection and a new model↔neural subsection,
brought in line with the code that produced the reported figures.

**How to use this file.** Each block below is anchored to a line range in
`Nashaat et al._Neuron_Revision.pdf`. Paste the block **body** into Word in
track-changes mode. The `--- rationale ---` footer of each block is *not* part of
the manuscript — it records what changed and against which source file, so the edit
can be justified to co-authors and reviewers.

Blocks 1–8 sit inside *Model* (PDF lines 1256–1318). Block 9 is a new subsection at
the end of *Two-Photon Data Analysis*. Blocks 10–12 are outside the model: the *Tracking*
subsection (PDF lines 1406–1419) and panels J–M of the Figure S3 legend (PDF lines
2118–2126). The line numbers are identical in `Nashaat et al._Neuron_Revision.pdf` and
both `_accept_revision.pdf` exports.

---

## BLOCK 1 — REPLACE PDF line 1268 (equation 3)

The decision variable evolves according to:

> 𝑥ₜ₊Δₜ = 𝑥ₜ + 𝑘 · Cohr · Δ𝑡 + 𝑠 · 𝜖ₜ · √Δ𝑡    (3)

--- rationale (do not paste) ---
The noise increment scales with √Δ𝑡, not Δ𝑡, so that the accumulated variance is
proportional to elapsed time and the fitted 𝑠 is independent of the integration
step. Previously mis-stated as `s · ε_t · Δt`.
Code: `model/noise.py:16-22` (`standard_normal(size) * sqrt(dt)`);
guard test `model/tests/test_noise_scaling.py:23-40`.

---

## BLOCK 2 — REPLACE PDF lines 1287–1294 (equations 6–8 and their definitions)

> 𝑄_L^{n=1} = 𝑄_R^{n=1} = 0.5    (6)

> 𝑄_{L|R}^n = 𝑄_{L|R}^{n−1} + { 𝛼 (Reward^{n−1} − 𝑄_{L|R}^{n−1}) → ChoiceDir^{n−1} = (L|R)
>                                0                                → ChoiceDir^{n−1} ≠ (L|R) }    (7)

The two action values were reduced to a single normalised relative value, and this
in turn set the accumulator's starting point:

> q^{n} = log( clip(Q_{L}^{n}, 0.01, 1) ⁄ clip(Q_{R}^{n}, 0.01, 1) ) ⁄ log(100)    (8)

> 𝑍^{n} = clip( 𝛿 · 𝑞^{n} + offset , −1, 1 )    (9)

Where **𝑛** is the trial number within a session. **Reward^n** flags whether a reward
is received on trial 𝑛 (Reward^n ∈ {0,1}); it equals the trial's correct/incorrect
outcome. **ChoiceDir^n** is choice direction on trial 𝑛 (ChoiceDir^n ∈ {0,1,nil}).
**𝑸_L^n, 𝑸_R^n** are action values for left/right at trial 𝑛 (𝑄_{L|R}^n ∈ [0,1]); only
the chosen side is updated, and trials without a recorded choice leave both
unchanged. **𝒒ⁿ** is the relative action value. Because an action value that is
repeatedly unrewarded decays toward zero and would make the ratio 𝑄_L ⁄ 𝑄_R diverge,
both values are first floored at 0.01, confining the ratio to [0.01, 100]; dividing
the log ratio by log(100) then rescales 𝑞ⁿ to span exactly [−1, 1], the range over
which the starting point is defined. Written in this form the quantity is the
base-100 logarithm of the ratio, so its value does not depend on which logarithm
base is used. **𝒁ⁿ** is the starting point expressed as a fraction of the decision
bound. 𝑄_L and 𝑄_R are re-initialised to 0.5 at the start of every session.

Model free parameters are: **offset**: motor bias term (offset ∈ [−1,1]),
**𝜶**: Q-learning update rate (𝛼 ∈ [0,1]), **𝜹**: starting offset scaling factor
(𝛿 ∈ [0,1]).

--- rationale (do not paste) ---
Three corrections to the old equation (8), `Z^n = clip(δ·[log(Q_L/Q_R) + offset], −1, 1)`:
 (i) the log ratio is floored at 0.01 and normalised by log(100), which the old form
     omitted — without it 𝑞ⁿ is unbounded and the stated 𝑍ⁿ ∈ [−1,1] does not follow;
 (ii) `offset` is added **after** scaling by 𝛿, not inside the bracket;
 (iii) the normalisation is split out as its own numbered equation so 𝑞ⁿ can be
     referenced by the reward-rate and neural-correlate sections;
 (iv) the definition paragraph now states *why* the denominator is log(100) — a bare
     log(100) reads as a magic constant, when it is fixed by the 0.01 floor: flooring
     confines 𝑄_L ⁄ 𝑄_R to [0.01, 100], and log(100) is exactly what maps that onto
     [−1, 1].
On the choice of denominator: log(𝑥) ⁄ log(100) is the base-100 logarithm written out,
so the base cancels and any implementation (ln, log₁₀, log₂) gives the same value. It
is numerically identical to the more compact log₁₀(𝑥) ⁄ 2, but that form is only
correct if the reader honours the base-10 subscript — using a natural log there is off
by a factor of ~2.3. The ratio form was kept for that reason. (The code uses the
natural log with LOG_CEIL_MAX = log(1/0.01); `model/state_updates.py:27-28`.)
NOTE: this renumbers the subsequent equations by +1 (old 9→10, 10→11, 11→12, …).
Code: `model/state_updates.py:45-54` (`compute_q_value`),
`model/state_updates.py:183-205` (`compute_starting_point_z`),
`model/mle.py:983`; per-session reset at `model/mle.py:1206-1207`.

---

## BLOCK 3 — REPLACE PDF lines 1299–1302 (equations 9–10 and their definitions)

> 𝑅𝑅ⁿ = 𝑅𝑅^{n−1} + 𝛽 (Reward^{n−1} − 𝑅𝑅^{n−1})    (10)

> 𝑥ₜ₊Δₜⁿ = 𝑥ₜⁿ + 𝑘 · Cohrⁿ · Δ𝑡 + 𝑅𝑅ⁿ · 𝑠 · 𝜖ₜ · √Δ𝑡    (11)

Where **𝑹𝑹ⁿ**: reward rate entering trial 𝑛 (𝑅𝑅ⁿ ∈ [0,1]), an exponentially weighted
average of recent outcomes updated on every trial and re-initialised to 0.5 at the
start of every session. Because 𝑅𝑅ⁿ multiplies the noise term directly, a high recent
reward rate yields noisier accumulation and therefore faster, less accurate decisions.

Model free parameters: **𝜷**: R-learning update rate (𝛽 ∈ [0,1]).

--- rationale (do not paste) ---
 (i) The last term of the update read `R^{n-1}`; it is `RR^{n-1}` (typo).
 (ii) Noise scaling corrected to √Δ𝑡, as in Block 1.
 (iii) Added the initial value, the per-session reset, the one-trial lag, and one
     sentence stating the sign of the effect (previously the reader had to infer it).
Code: `model/state_updates.py:102-120` (`update_reward_rate`),
`model/drift.py:262-265` (`noise *= noise_sigma; noise *= RewardRate[:, np.newaxis]`),
MLE equivalent at `model/mle.py:1002-1006`.

---

## BLOCK 4 — INSERT after PDF line 1302 (before "Q-Learning + R-Learning + DDM")

### Alternative reward-rate routings

Reward and urgency signals are more commonly routed to the drift rate or to the
decision bound, so we compared four channels through which the reward rate could
modulate the accumulator, holding the rest of the model fixed (Figure S4C):

| Channel | Modulation |
|---|---|
| Noise gain (used throughout) | 𝜎ₜ = 𝑅𝑅ⁿ · 𝑠 |
| Bound | 𝑏ₜ = (𝑎⁄2) · (2 − 𝑅𝑅ⁿ) |
| Drift (2 − 𝑟) | 𝜇ₜ = 𝑘 · Cohrⁿ · (2 − 𝑅𝑅ⁿ) |
| Drift (1 + 𝑟) | 𝜇ₜ = 𝑘 · Cohrⁿ · (1 + 𝑅𝑅ⁿ) |

The two drift mappings are mirror polarities of one another — under (2 − 𝑟) a high
reward rate slows decisions, under (1 + 𝑟) it speeds them — and both span the same
range of drift magnitudes as 𝑅𝑅ⁿ varies over [0,1], so the comparison is not
confounded by the size of the modulation. All four variants were fitted with the
Chi-square objective described below.

--- rationale (do not paste) ---
New. The manuscript asserts a noise-modulation account and rules out drift
(Discussion lines 428-436; Figure S4 legend lines 1772-1777) but never defines the
alternatives, so the comparison is not reproducible from the text. This block supplies
the four equations; the *result* of the comparison stays in the Fig. S4C legend.
Code: `model/drift.py:445-470` (channel constants),
`model/state_updates.py:123-180` (`bound_scale_from_reward_rate`,
`drift_scale_from_reward_rate`), figure spec `DRIFT_RR_SPECS` at
`model/aggregate.py:714-732`.

---

## BLOCK 5 — REPLACE PDF lines 1306–1316 (the fitting paragraph)

In all models, to reduce interaction effects when fitting both the decision boundary
and noise parameters simultaneously, we fixed the decision boundary to 𝑎 = 2, setting
bounds at (−1, 1). The integration step was set to Δ𝑡 = 0.005 s, with a maximum
allowable sampling time of 𝑇_max = 4.8 s. Trials that did not reach a decision
threshold within this time were categorised as no-choice and no-reward trials
(Figure S4, `Sampling Time` column). Each model was fitted to each mouse under two
objectives — a Chi-square statistic on simulated sampling-time distributions and an
exact trial-wise likelihood — and, for the reported fits, under a weighted
combination of the two.

**Chi-square objective.** Sampling times generated by the model were compared with
the subject's own, following the quantile approach of Ratcliff and Tuerlinckx⁸⁰.
Trials were divided into four cells crossing outcome (correct, incorrect) with choice
direction (left, right); difficulties were pooled within each cell, so trial
difficulty constrains the fit through the drift rather than through the binning.
Within each cell, the subject's sampling times were binned at the 0.1, 0.3, 0.5, 0.7
and 0.9 quantiles, with an additional edge placed just above the fastest observed
trial and a final edge at 𝑇_max, giving seven bins per cell. The lowest bin isolates
the fastest response and thereby sharpens estimation of the non-decision time, while
the uppermost bin absorbs late and undecided model responses. The statistic

> 𝜒² = Σ_cells Σ_bins ( 𝑁^sim − 𝑁^real )² ⁄ 𝑁^real    (12)

where 𝑁^real and 𝑁^sim are the numbers of observed and simulated trials falling in a
bin, was summed across all four cells. The simulator was run under a fixed random
seed, so the Chi-square loss is deterministic given a parameter set.

--- rationale (do not paste) ---
 (i) 𝑇_max corrected from 3 s to 4.8 s — every fit reported in this revision uses 4.8 s
     (`model/initvals.py:116`; the 3 s fit pickles have been superseded).
 (ii) The bin definition was wrong: the stated quantiles [0, .1, .3, .5, .7, .9, 1] give
     six bins, the code uses seven (an extra edge at min(RT)+1e-6, top edge at 𝑇_max
     rather than the empirical maximum).
 (iii) The old text said loss was computed "separately for correct and incorrect
     choices" — it is also split by choice direction, i.e. four cells, not two.
 (iv) Added the explicit statement that difficulties are pooled. Reviewers are likely
     to assume otherwise, and it bears on how strongly the psychometric function is
     constrained by the Chi² fit.
 (v) Added the fixed-seed note (relevant to reproducibility of the DE search).
 (vii) Bin counts are written 𝑁, not 𝑛: the manuscript uses 𝑛 as the trial index
     (`Q^n`, `RR^n`, `L^n`), so 𝑛^sim / 𝑛^real would overload the same letter.
 (vi) The paragraph is split so that the Chi² objective, the likelihood, and the joint
     objective each have a heading, as in the rest of Methods.
Code: `model/logic.py:410-461` (`chi2Loss`, `calcLoss`); single call site
`model/logic.py:579` confirms difficulties are not stratified.

---

## BLOCK 6 — INSERT after Block 5

**Maximum-likelihood objective.** For each trial we evaluated the exact first-passage
density of the discretised diffusion, obtained by propagating the accumulator's
probability distribution forward in time on a state grid (step 0.02) with absorbing
barriers at ±1; no simulation or density smoothing is involved. Choice and sampling
time enter as a single joint density: 𝑝(𝑡ⁿ, 𝑐ⁿ) is the first-passage density of the
accumulator, that is, the probability density that it first reaches the bound
corresponding to the animal's choice 𝑐ⁿ, evaluated at the decision time 𝑇_decision =
𝑡ⁿ − 𝑇₀ implied by the observed sampling time 𝑡ⁿ. Choice and sampling time are
therefore not separate terms in the likelihood.

Some trials are not plausibly generated by the diffusion at all — responses faster
than the non-decision time, to which the model assigns zero density, and inattentive
responses at any latency. Under a pure diffusion likelihood a handful of such trials
dominates the fit, so the density was mixed with a uniform contaminant of free rate
𝜆, spread evenly across the two choices and across the observable range of sampling
times (0, 𝑇_max]:

> 𝐿ⁿ = (1 − 𝜆) · 𝑝(𝑡ⁿ, 𝑐ⁿ) + 𝜆 ⁄ (2 · 𝑇_max)    (13)

On every trial the diffusion also places some probability on reaching neither bound
within 𝑇_max. The data contain no such observations by construction: trials on which
the animal did not terminate sampling within 𝑇_max, and trials with no recorded
choice, were marked invalid and excluded (see Exclusion criteria). Because the
retained trials are thus conditioned on a response having occurred, the model was
conditioned on the same event — each trial's likelihood was divided by the mixture's
total probability of producing a response, (1 − 𝜆)(𝑃_upper + 𝑃_lower) + 𝜆, where
𝑃_upper and 𝑃_lower are the probabilities of reaching each bound by 𝑇_max. Without this
renormalisation the retained trials' likelihood would not be a proper density. The
objective minimised was the negative sum of the resulting log-likelihoods over all
valid trials of a subject,

> −log 𝐿(𝜃) = − Σ_𝑛 log 𝐿ⁿ(𝜃)    (14)

Under this objective the latent variables 𝑄_L, 𝑄_R and 𝑅𝑅 were propagated
using the animal's observed choices and outcomes, whereas under the Chi-square
objective they were propagated using the model's own simulated choices and outcomes.

Model free parameters gain: **𝝀**: contamination (lapse) rate (𝜆 ∈ [0, 0.1]).

--- rationale (do not paste) ---
New — the previous revision fitted by Chi² only, so no likelihood was described.
𝜆 is a fitted free parameter present in every MLE and joint fit and appeared nowhere
in the manuscript. The conditional normalisation is stated explicitly because it
changes what the reported log-likelihood (and hence any AIC/BIC) means.
Four points of notation and logic worth preserving if this paragraph is edited:
 (a) 𝑝(𝑡ⁿ, 𝑐ⁿ) is defined against 𝑇_decision and 𝑇₀, which the manuscript already
     introduces in equations (4)–(5), so no new symbol is needed.
 (b) The trial index is 𝑛, matching the Q- and R-learning equations. It must NOT be 𝑖:
     the manuscript already binds 𝑖 to "number of time steps until boundary crossing"
     in equation (4), so Σᵢ would collide with an existing symbol.
 (c) 𝜆 is a contaminant over the *observable* range (0, 𝑇_max]; the 2·𝑇_max is only
     its normaliser (two choices × duration). It models fast guesses and inattentive
     responses that fall *inside* that range, and has nothing to do with the 𝑇_max
     cutoff on long trials.
 (d) Keep the model side and the data side distinct. Reaching neither bound by 𝑇_max is
     a probability the diffusion carries on EVERY trial, not a subset of trials. An
     earlier draft wrote "such trials are excluded from the data", which wrongly implied
     the model outcome picks out a data category. The data-side rule is separate and
     applies to observed sampling times: `calcStimulusTime > T_dur`, or a null choice,
     sets `valid = False` (`model_runner.py:110-121`). The renormalisation exists
     because the retained data are conditioned on a response having occurred, so the
     model must be conditioned on the same event.
On the magnitude of the conditional renormalisation (checked 2026-08-14, since it
reads as though it might be negligible): it is applied, and it is tested —
`_apply_choice_rt_weights` is called from `objective_from_population` (`model/mle.py:496`),
the early return is gated on `choice_norm == "marginal"` so it never fires for our
`conditional` configs, and `test_mle_choice_rt_weight.py:138-144` asserts the exact
production case (weights 1,1 → `loglik − log(P_L+P_R)`).
Its size: P(response by T_max) ≈ 1 whenever there is appreciable drift, so the term is
worth only a few nats against an NLL of ~2000 at a good fit. It is NOT a constant,
though — it depends on θ and varies per trial through coherence, z and σ_eff = RR·s —
so it moves the optimum rather than offsetting the loss. It matters most during the DE
search, where small-σ candidates are reachable (σ bounds [0, 5]): at σ_eff = 0.15 and
zero coherence, P(response) ≈ 0.003, i.e. ~18,000 nats.
Trade-off worth knowing if this is ever revisited: conditioning scores the model on the
SHAPE of the RT/choice distribution among responders and not on the response rate, so
the model is never penalised for predicting an implausible non-response rate. Only ~131
trials (~0.2%) were actually excluded, so the observed ~99.8% response rate is real
information that the `conditional` setting discards. Switching to `marginal` would keep
it, at the cost of a refit.
The teacher-forcing sentence matters: the two objectives condition the RL recursion on
different histories, which is the reason they can disagree and hence the reason the
joint objective is not simply redundant.
Code: `model/mle_batch.py:539-577` (transition kernel / absorbing barriers),
`model/mle_likelihood.py:110-126` (joint density + lapse mixture),
`model/mle.py:710-726` (`conditional` normalisation, the configured default),
`model/mle.py:314-324` (the sum), `model/initvals.py:78` (𝜆 bounds),
`model/mle.py:1206-1231` vs `model/logic.py:143-156` (observed vs simulated history).

---

## BLOCK 7 — INSERT after Block 6

**Joint objective.** The two objectives constrain different aspects of the data: the
likelihood is sensitive to the precise timing of individual responses, while the
Chi-square statistic matches the shape of the sampling-time distribution within each
outcome × direction cell. Fitted on its own, the likelihood penalised fast errors
toward the non-preferred side so heavily that it favoured parameter sets in which the
learning rates hold 𝑄 and 𝑅𝑅 effectively constant, disabling both reinforcement-learning
components. We therefore minimised a weighted combination of the two:

> 𝐿_joint(𝜃) = 𝑤_MLE · ( −log 𝐿(𝜃) ⁄ −log 𝐿* ) + 𝑤_𝜒² · ( 𝜒²(𝜃) ⁄ 𝜒²* )    (15)

Because the two losses are on unrelated scales, each was normalised by its own
reference value: −log 𝐿* and 𝜒²* are the minimised losses obtained when the same model
is fitted to the same subject under each objective alone. Each term therefore equals 1
at its own optimum, and the weights express a like-for-like trade-off. Reported fits
use 𝑤_MLE = 1 and 𝑤_𝜒² = 0.5; the comparison across weightings is shown in Figure S14B.

--- rationale (do not paste) ---
New. Two points that must be stated correctly:
 (i) The normalisation is by the **standalone-best loss**, not by trial count. The CLI
     help text and the `MLEModelConfig` docstring still describe division by valid-trial
     count — that wording is stale and must not be copied into the manuscript.
 (ii) The justification of 0.5 over 0.1 belongs to the Fig. S14B legend, per the
     author's instruction; Methods states only what the method is and one sentence on
     why a joint objective was needed at all.
Approaches trialled and *not* reported (deliberately omitted here): terminal-C
redistribution of residual probability mass, and separate weights for the
choice-direction and reaction-time components of the likelihood. Both exist in the code
but are inert in every saved fit (`mle_terminal_c = 1.0`,
`mle_choice_weight = mle_rt_weight = 1.0`).
Code: `model/fit.py:200-231` (the combination),
`model/fit.py:582-617` (`_load_reference_loss`, `OptimRes.fun`),
weight specs `model/aggregate.py:673-686`.

---

## BLOCK 8 — INSERT after Block 7

**Optimisation.** All objectives were minimised by differential evolution
(`scipy.optimize.differential_evolution`). Chi-square fits used a population size of
100 per parameter with mutation sampled from (0.5, 1.5) and a final local polish,
parallelised across CPU cores. Likelihood and joint fits evaluated the entire
population in a single vectorised pass per generation (deferred updating, no local
polish), with the population sized so that at least 64 candidates were evaluated per
generation. Model-based analyses (Figure 2, Figure 7, Figure S4) were computed using
mice with a minimum of 2,500 trials (9 mice).

--- rationale (do not paste) ---
New; the previous text named differential evolution but gave no settings, so the fit
was not reproducible. The final sentence absorbs the existing PDF lines 1317-1318 —
delete those two lines when this block is inserted, and note the figure reference
there currently reads "Figure S3K", which should be S4 (see Notes, item 3).
Code: `model/fit.py:410-422` (MLE/joint DE call),
`model/fit.py:474-493` (Chi² DE call), `model/mle.py:1350-1383` (population sizing).

---

## BLOCK 9 — INSERT after PDF line 1684 (new subsection at the end of *Two-Photon Data Analysis*)

### Correlating single-neuron activity with model latent variables

To ask whether individual neurons tracked the latent variables of the behavioural
model (Figure 7A&B; Figure S14E), the fitted DDM+Q_L+R_L model was re-evaluated on each
animal's actual trial sequence, yielding for every trial the state of the model
*entering* that trial: the two action values 𝑄_L and 𝑄_R, the normalised relative value
𝑞ⁿ, and the reward rate 𝑅𝑅ⁿ. The trial's signed sensory evidence (DV), which is the
model's drift input, was taken from the trial record.

Each neuron was reduced to one value per trial: the maximum of its z-scored
fluorescence within the analysis window (0.1 s before sampling onset to 0.1 s after
movement onset, time-normalised as described above). Within each session we then
computed, for each neuron and each variable, the Pearson correlation across trials
between that value and the variable, and counted the neuron as modulated by the
variable when |𝑟| ≥ 0.3. Because the three Q variables describe one underlying
quantity, a neuron was counted as Q-modulated if it met the criterion for any of 𝑄_L,
𝑄_R or 𝑞ⁿ; neurons meeting it for more than one were counted once.

The reported value for each region is the percentage of assessable neurons modulated
in a session, averaged across sessions, with the error bar giving the SEM across
sessions (22 sessions: 12 MFC, 10 LFC). Neurons for which a correlation was undefined
(fewer than three trials, or a constant activity or variable) were excluded from both
numerator and denominator.

Chance level was established by a per-neuron shuffle: each neuron's trial-wise
activity was permuted while its latent variables were held in place, the correlations
recomputed and the criterion re-applied, and the resulting percentages aggregated
exactly as on the observed side. Over 1,000 shuffles this yields a null distribution
of the session-averaged percentage, from which we took the 2.5th and 97.5th
percentiles as a 95% interval and a one-sided p-value,
𝑝 = (#{shuffle ≥ observed} + 1)/(𝑁 + 1).

To compare evidence coding between behavioural strategies (Figure 7B), correlations
with DV were recomputed separately within the fast and slow trials of each session,
defined as elsewhere in this study (fastest and slowest thirds within each difficulty).
Fast and slow percentages were compared with a two-sided, session-paired sign-flip
permutation test: sessions contributing to both subsets were paired, the observed
statistic was the mean paired difference in percentage modulated, and the null was
generated by drawing an independent random sign for every session, negating that
session's difference where the sign was negative, and recomputing the mean; this was
repeated over 1,000 permutations. Sign flipping is the exchangeable operation for
paired data: under the null the fast/slow labels within a session are interchangeable,
and swapping them negates that session's difference. Within each panel, p-values across bars were corrected using the
Holm–Bonferroni method.

--- rationale (do not paste) ---
New subsection. Figures 7A&B and S14E currently have no Methods support at all.
Two wording choices worth keeping:
 (i) "signed sensory evidence (DV), which is the model's drift input" — DV is a
     per-trial task quantity, constant within a trial. Do NOT describe it as a decision
     variable read out at a timepoint; that would be a different (and unperformed)
     analysis.
 (ii) The vs-chance test is stated as one-sided and the fast/slow test as two-sided,
     because they genuinely differ.
Note the neuronal scalar is the **maximum** of the z-scored trace, which differs from
`twop/activity_correlation.py`, where the mean is used — deliberate, and stated here so
the two analyses are not conflated.
Code: `model/neural_correlate.py:80-90` (variables), `:240-256` (re-evaluation),
`:451-462` (per-trial scalar), `:532-556` (correlation), `:882-889` (criterion),
`:1289-1316` (Q factor OR), `:892-952` (shuffle null, CI, p),
`:1030-1054` (sign-flip permutation), `:983-992` (Holm).
Session counts verified against
`results/RLModel/neural_correlate/…Chi²=0.5…/factor_bars/summary_fastslowDV_by_region.csv`.

---

## BLOCK 10 — REPLACE PDF lines 1406–1413, from "For each trial, during the sampling epoch" to "…polar histograms for each strategy." (*Tracking*, Figures S3J–L)

For each trial, during the sampling epoch (nose-poke fixation; behavioural
definition described elsewhere), we discarded the lowest and highest 10% of that
trial's centroid rotation angles to remove occasional tracking glitches, and took the
rotation-angle range, the difference between the largest and smallest remaining angle
(degrees), as a measure of how much the posture changed within the trial. Per-trial
ranges were binned in 5° bins, expressed as a fraction of each session's trials,
averaged across sessions, and plotted for each behavioural strategy (fast, typical,
slow; Figure S3J). Because the posture changed little within a trial, we summarised
each trial by a single value, its mean centroid rotation angle, and expressed it either
relative to the choice direction (Figure S3K) or relative to the preferred side of each
session (Figure S3L): sessions in which the body was rotated predominantly to one side
were mirrored so that the preferred side was common to all sessions. Mean angles were
binned in 5° bins to generate polar histograms for each strategy.

--- rationale (do not paste) ---
- S3J: the code measures the within-trial range, `abs(max − min)`, not the cumulative
  frame-to-frame change the current text describes. Range is the intended metric:
  frame-to-frame differences add up tracking jitter as well as movement, so a
  cumulative sum grows with trial length even for a still animal. The 10% trim and
  the per-session normalisation are what the published panel does and were not
  stated. Code: `tracking/centroids.py::trimTrialOutliers`,
  `::distanceTravelledHistogram`; notebook call `outliers_ratio=0.1`.
- S3J → K/L: J is what justifies using one value per trial in K, L and M. The new
  sentence makes that dependency explicit.
- S3K: the sentence is kept as written. **The committed S3K is not normalised to the
  choice direction** (`choice_normed=False`); the next revision regenerates it with
  `choice_normed=True`, which makes this sentence and the legend correct as they stand.
  Code: `tracking/centroids.py::normaliseToChoice`.
- S3L: "each animal's preferred side" → "each session". The code flips each session
  independently (a session with more negative than positive frames is mirrored), and
  per session is the intended rule. Code: `tracking/centroids.py::normaliseToPreferredSide`.

---

## BLOCK 11 — REPLACE PDF lines 1414–1419 (*Tracking*, Figure S3M)

To test whether behavioural strategy was related to posture, we measured, for each
trial, how far the posture departed from the animal's usual posture (Figure S3M). The
usual posture was the most frequent (modal) mean trial angle of each animal across all
of its sessions, after binning mean angles to the nearest 5°, and each trial was scored
by the absolute difference between its mean centroid rotation angle and that modal
angle. For each animal and strategy, we first assessed normality of the per-trial
distributions using the Shapiro–Wilk test and found all distributions to deviate
significantly from normality (P < 0.05). We therefore compared the three strategies
within each animal using a non-parametric Kruskal–Wallis test, corrected for multiple
comparisons across animals with the Holm–Bonferroni method; no significant differences
were detected in any animal (0 of 4 mice).

--- rationale (do not paste) ---
The current text says the test compares "rotation-angle distance travelled per
trial". The published panel, its y-axis ("Distance from mode Rotation Angle (deg)")
and its 0/4 result come from the distance-from-usual-posture score instead, which is
the intended analysis. The two are not interchangeable: on the same frames a
distance-travelled test gives 4/4 animals significant, as distance travelled grows with
trial length (slow trials have a median of 44 video frames against 18 for fast).
The Holm correction was applied but not stated; it matters here, since raw per-animal
p-values are 0.018 / 0.361 / 0.035 / 0.497 and Holm-corrected 0.073 / 0.722 / 0.105 /
0.722. Animals needed at least 10 trials in each strategy; all four qualified.
Code: `tracking/strategy.py` (`trialMeanAngles`, `distanceFromMode`,
`animalQuantileGroups`, `compareStrategies`); notebook call `outliers_ratio=0`.

---

## BLOCK 12 — REPLACE PDF lines 2118–2126, from "(J) Distribution of centroid rotation" to "…across strategies." (Figure S3 legend)

(J) Distribution of the within-trial range of the centroid rotation angle across
behavioural strategies (fast, red; typical, orange; slow, yellow). Mice exhibited
minimal movement during sampling, and distributions overlapped extensively. (K) Polar
distribution of the mean centroid rotation angle per trial, normalized to the choice
direction. (L) Polar distribution of the mean centroid rotation angle per trial,
normalized to each session's preferred side. (M) Animal-wise comparison of each
trial's distance from the animal's usual posture (absolute difference between the
trial's mean centroid rotation angle and the animal's modal angle) across strategies
(Kruskal–Wallis, Holm–Bonferroni across animals). No significant differences were
observed, confirming that posture during sampling does not differ across strategies.

--- rationale (do not paste) ---
Same three corrections as Blocks 10–11, applied to the legend: J is a within-trial
range; L is per session; M is distance from the usual posture, and its conclusion is
about posture rather than movement (movement is J's claim). K is unchanged pending the
`choice_normed=True` regeneration. The Results sentence at PDF line 155 ("posture or
movement during sampling … (Figure S3J-M)") already matches this split and needs no
change.

---

# Notes for the authors (not manuscript text)

1. **Figure S14B needs no new code.** The four columns requested — pure MLE,
   MLE=1/Chi²=0.1, MLE=1/Chi²=0.5, and Chi²-only — are already specified as
   `MLE_WEIGHT_SPECS` plus `CHI2_ONLY_RRQ_SPEC` in `model/aggregate.py:673-686`, and
   all three MLE-family fit pickles exist on disk. The pure-MLE penalty to be printed
   under each panel title is the `mle_raw_loss` field stored in each joint payload
   (`model/fit.py:333-358`).

2. **Figure S14E appears to be stale.** Its panel title reads
   `weighted Chi² (MLE=1, Chi²=0.1)`. That run's results folder contains **12**
   sessions (7 LFC + 5 MFC). Figure 7A's legend states **n = 22 sessions**, which
   matches only the Chi²=0.5 run (10 LFC + 12 MFC, regenerated 2025-08-08). Block 9
   above is written for the Chi²=0.5 run; regenerate S14E from it before submission,
   or the figure and its legend will disagree.

3. **Stray figure cross-references inside *Model*.** PDF lines 1258, 1317, 1324 and
   1332 cite "Figure S3N" / "Figure S3K", but the model supplementary figure is S4.
   Line 1333 cites "Figure 1L" where Figure 2G is meant. Flagged rather than silently
   changed, since the intended targets should be confirmed by an author.

4. **Equation renumbering.** Block 2 adds one equation, so everything from the old
   equation (9) onward shifts by +1. The numbering used in this file already reflects
   that: old (9)→(10), (10)→(11), and the new Chi²/likelihood/joint equations take
   (12)–(15). Existing in-text references to equation numbers should be checked.

5. **Deferred to the next Methods round.** PDF lines 1203–1204 still carry an
   unresolved author note — "*(Mention here the parameters used, and they were used to
   avoid redundancy information between variables)*" — in the explained-variance
   paragraph. It belongs to the behavioural section, not the model, and is out of
   scope here.

---

# Known code issue (not manuscript text — for a separate fix)

**The starting point 𝑍ⁿ is computed by two different formulas.**

| Path | Formula (with BOUND = 1) | Source |
|---|---|---|
| Likelihood | `clip(δ·q + offset, ±1)` | `model/mle.py:983` → `model/state_updates.py:202` |
| Chi² / simulator | `clip( clip(q + offset, ±1) · δ, ±1 )` | `model/logic.py:330` → `model/bias.py:49` |

The joint objective evaluates the same parameter vector through both. In the Chi²
form the motor bias is scaled by 𝛿 and 𝑍's reachable range is capped at ±𝛿; in the
likelihood form the bias is an unscaled additive shift and 𝑍 can span ±1. At real
fitted values (𝛿 ≈ 0.118, offset ≈ +0.30) 𝑍 spans [+0.18, +0.42] under the likelihood
but [−0.08, +0.12] under the simulator — the sign of the bias differs, not just its
size.

This is **not** the `--scale-bound` axis. That axis (`bound=None` vs `bound=b_t` in
`compute_starting_point_z`) is genuinely inert here, because BOUND is frozen at 1.0 and
both paths clip to ±1 and then multiply by 1. The divergence is an order-of-operations
difference that survives BOUND = 1 and is live in every fit reported.

**Cause.** Commit `6e4e05b "Model: Unify code paths between Chisqr and MLE"` migrated
`fit.py`, `logic.py`, `mle.py`, `state_updates.py` and `visualize.py`, but not
`bias.py`. `_biasQVal` is pre-unification code that was never migrated, and no test
asserts parity between the two. The likelihood form is the one specified in
`nashaat_oraby_mle_fitting_spec.md:35`.

**Agreed fix direction.** Change the Chi² side: make `_biasQVal` return
`clip(BIAS_COEF · Q_val + Q_VAL_OFFSET, −1, 1)` to match `compute_starting_point_z`,
and add a parity test asserting the two agree. Chi² refits are cheap relative to MLE.

**Refit scope.** Chi²-only fits **and** joint fits (whose Chi² term and 𝜒² reference
both change) would need re-running. Pure-MLE fits are unaffected. Figures 2E–G and S4
would shift; Figures 7A–B and S14E would not, because they correlate
`mle_Q_rel_before`, to which the offset is never applied.

Per the author's decision, the Methods text above states the likelihood form with no
footnote, since this is a code defect rather than a modelling choice.

---

## BLOCK 13 — INSERT after Block 8 (or as a Figure 7D legend note)

**Figure 7D — the Q/reward-rate surface.** For each of the 22 modelled mice, every
recorded session was resampled 1,000 times with stimulus strengths redrawn uniformly
over the coherence range (272,000 pseudo-sessions, 47.8 million simulated trials that
reached a decision), and each pseudo-session was simulated under that mouse's own
fitted parameters. Simulated sampling times were z-scored within mouse and averaged
in a grid of 10 reward-rate x 20 relative-Q bins, separately for the three difficulty
terciles. Each bin is the mean of the per-mouse means it contains, so no bin is
dominated by a single animal, and bins holding a single trial are left empty. The
surface is smoothed along the relative-Q axis with a NaN-conserving Gaussian kernel
(sigma = 1 bin). During simulation, the Q values and the reward rate each received an
independent Gaussian perturbation (SD 0.01, clipped to their [0,1] range) on every
trial.

--- rationale (do not paste) ---
Three of these are new and need to survive review, so the numbers behind each are
recorded here.

1. **Resample count (1,000, was 10,000).** Unchanged in substance: the fit this figure
   was originally built from embedded 20 mice and 40 sessions, so 10,000 resamples per
   session meant 400,000 pseudo-sessions. The shipped fits embed 22 mice and 272
   sessions, so the same 400,000 is reached at ~1,470. Independently, the published
   panel's own stored output (busiest bin = 2,959,436 observations) implies ~1,540,
   since that count is linear in the resample count (measured: 4,591 / 6,651 / 12,094 /
   21,825 / 41,001 at 1 / 2 / 5 / 10 / 20). Left at 10,000 against the current fits the
   cell needs ~420 GiB and cannot run.

2. **The per-trial nudge (SD 0.01).** Needed because of what the fits contain: the
   chi-squared 4.8 s fits put the Q learning rate at a median ALPHA of 0.826, with 12
   of 22 mice above 0.8 and 4 at the upper bound (0.95-0.996). At those rates a losing
   trial collapses the chosen side to ~(1 - ALPHA) while the other stays near 1, so
   Q_val = log(0.05)/log(100) ~= -0.65 and similar fixed points recur constantly:
   63.5% of trials fall within +/-0.1 of zero and the remainder sit on a lattice of
   spacing 0.09-0.13, against a bin width of 0.1. Neighbouring bins therefore hold
   different populations of trial and the surface steps. That stepping is not sampling
   noise -- it measures 0.18 (mean |second difference| along Q, beyond the noise floor)
   and does not fall between 1.0M and 4.8M simulated trials. A 0.01 nudge halves it and
   0.02 removes it, while the trial-level relationships are untouched: slope of z-scored
   sampling time on reward rate -3.026 -> -3.027, on relative Q -0.478 -> -0.478,
   correlation -0.552 -> -0.552, mean simulated RT 1.2419 s -> 1.2421 s, accuracy
   0.7295 -> 0.7295. The published 3 s fit did not need it: its median ALPHA was 0.604,
   and its surface is correspondingly smoother (roughness 0.200 vs 0.341 at matched
   trial counts).
   It remains a visualisation device, applied after fitting. The principled version --
   learning noise as a model term, fitted jointly so the parameters and the noise are
   mutually consistent -- is the revision this should become; it is not claimed here.

3. **Per-mouse bin means (was pooled over trials).** Because each mouse's lattice is
   its own, a bin can be one animal: at low reward rate the relative-Q -0.7..-0.6
   column was 97% `Avgat1`, whose z-scored sampling time sits +0.36 above the group on
   hard trials and ~0.13 below it on medium and easy ones. Pooling over trials therefore
   printed a trough on the Medium and Easy sheets and nothing on Hard. Averaging
   per-mouse means first removes it (Easy: -0.09 -> +0.47, against neighbours at +0.44
   and +0.45) and matches how n = 22 mice is treated elsewhere in the paper.

Also corrected, and not worth manuscript text: the binning used
`np.digitize(..., right=True) - 1`, which returns -1 at or below the first edge and so
wrapped onto the last bin -- 1,846 trials per million with relative Q = -1 were drawn
at +1 -- and the grid was sized by bin edges rather than bins, leaving the reward-rate
1.0 row, the Q +1.0 column and a whole difficulty plane unreachable. Bin values were
also plotted at bin edges rather than centres.

Code: `model/qrsurface.py` (surface, binning, per-mouse means),
`model/state_updates.py::nudge_q_values`/`nudge_reward_rate`,
`model/logic.py` (`latent_nudge_sd` threaded through `makeOneRun`),
`model_to_behavior.ipynb` cells 7/9/13/14; tests in
`model/tests/test_qrsurface.py`.
