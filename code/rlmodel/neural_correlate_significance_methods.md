# Significance testing for the neuron ↔ model-latent correlation figures

*Draft methods text for the "Model-tuned neural representations" analyses
(`model_neural_correlate.ipynb`), written to slot into Methods §A.3 alongside
the existing statistical descriptions and the Fig-summary table.*

## A.3.x Neuron ↔ model-latent tuning: significance

For each imaged neuron we correlated its per-trial sampling-window activity
(the maximum z-scored dF/F within the sampling epoch) against each trial's
model latent — the Q-Learning action values (Q-left, Q-right and the relative
value Q-val), the R-Learning reward rate, and the drift/evidence variable (DV)
— using a within-session Pearson correlation. A neuron was counted as *tuned*
(or, for a grouped factor, *modulated*) when the absolute correlation reached a
fixed threshold (|r| ≥ 0.3); neurons whose correlation was undefined (constant
latent or activity, fewer than three trials) were excluded from both the
numerator and the denominator. For a grouped factor (e.g. "Q-modulated",
pooling Q-left/right/val) a neuron was modulated when any of the grouped
correlations passed the threshold, so a neuron tuned to more than one grouped
latent was counted once. All percentages were computed per session and then
averaged across sessions; bars report the across-session mean ± s.e.m. (i.e.
the distribution of per-session percentages within each brain region).

To ask whether the observed proportion of tuned neurons exceeded chance, we
built a per-neuron activity-shuffle null: for each neuron the per-trial
activity was randomly permuted (the latent vector held fixed), the correlation
re-computed, and the |r| ≥ 0.3 tuning decision re-made. Repeating this
`N_SHUFFLES` = 1,000 times, and aggregating each shuffle to the same
session-then-across-session percentage as the observed data, yields a null
distribution of the "% tuned" statistic per region and latent. The permutation
p-value was the fraction of shuffles whose chance percentage was greater than
or equal to the observed percentage, with a +1/(N+1) correction so that it is
never exactly zero. Because permuting activity preserves its variance, only the
correlation's covariance term moves under the shuffle, and all shuffles for a
neuron were drawn from a single (`N_SHUFFLES` × n-trials) permutation matrix.
Within each figure panel the p-values across the compared latents/factors were
corrected for multiple comparisons with the Holm–Bonferroni method and
annotated as *p < 0.05, **p < 0.01, ***p < 0.001 (n.s. otherwise).

## A.3.x Fast versus slow decisions

Drift (DV, DVabs) tuning was additionally computed separately within the fast
(first RT tercile) and slow (last RT tercile) trial subsets, so a neuron may be
drift-correlated in one strategy but not the other. Each fast and each slow bar
was first tested against its own within-subset activity-shuffle chance level,
exactly as above (each subset carries ~⅓ of the trials, so its chance floor is
correspondingly higher). To test whether the proportion of drift-tuned neurons
differed between fast and slow strategies we used a session-paired permutation:
per-session fast and slow percentages were paired on session identity, and the
observed mean paired difference (fast − slow) was compared against a null built
by independently flipping the sign of each session's paired difference
(`N_PERM` = 1,000 permutations, +1/(N+1) correction). We adopted this paired
permutation rather than a paired t-test so that the fast/slow comparison uses
the same distribution-free machinery as the rest of the analysis. Paired
comparisons were Holm-corrected across brain regions.

## A.3.x Medial versus lateral frontal cortex

To compare the proportion of tuned/modulated neurons between MFC and LFC we
used a hierarchical bootstrap that resamples with replacement at the region,
session and neuron levels, matching the resampling scheme used for the
optogenetics analyses. Within each region we resampled sessions with
replacement, then neurons within each drawn session with replacement, and
recomputed the region's session-averaged percentage; the statistic was the
MFC − LFC difference. Over `N_BOOT` = 10,000 draws this yields a bootstrap
distribution of the between-region difference, from which we obtained a
two-sided p-value as twice the fraction of draws whose sign was opposite to the
observed difference (capped at 1). This MFC-vs-LFC test is used on the fast/slow
drift figures, applied within each speed and Holm-corrected across speeds. The
factor-modulation summary (Q · Reward-Rate · DV-fast · DV-slow) is instead drawn
as one figure per region on a shared y-axis, with no cross-region comparison:
each bar is tested against its own shuffle chance level (per-neuron activity
permutation, above) and Holm-corrected across the four bars within that region,
and the DV-fast-vs-slow bracket uses the session-paired permutation of the
preceding section.

## Summary-table line

| Figure | Test |
| --- | --- |
| Neuron-tuning / factor-modulation bars (vs chance) | Per-neuron activity-shuffle permutation (1,000 shuffles), +1/(N+1) p-value — Holm–Bonferroni across bars within each region figure |
| Fast vs slow drift-tuning bars | Session-paired sign-flip permutation (1,000 permutations) — Holm–Bonferroni across regions |
| MFC vs LFC (fast/slow drift bars only) | Hierarchical bootstrap (region → session → neuron, 10,000 draws), two-sided sign-change p — Holm–Bonferroni across speeds |

Significance thresholds throughout: *p < 0.05, **p < 0.01, ***p < 0.001.
