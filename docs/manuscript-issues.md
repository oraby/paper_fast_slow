# Manuscript issues found while extracting the figure code

Discrepancies between the manuscript
(`Nashaat et al.,_Neuron_Revision.docx`) and the code or figures that produce
it, found while moving inline notebook code into tested modules.

Nothing here was changed in the manuscript: the `.docx` is never edited
directly, and Methods wording goes to
[`code/rlmodel/methods_model_revision.md`](../code/rlmodel/methods_model_revision.md)
as blocks anchored to PDF line numbers. This file is the list to work from.

Each entry states how it was verified, so it can be re-checked rather than
taken on trust.

| # | Where | Issue | Severity |
|---|---|---|---|
| 1 | Fig. 2B-right | All 17 error bars carry one animal's SD | **Figure is wrong** |
| 2 | Fig. S3G legend | "n = 9 mice" — it is 18 | Caption wrong |
| 3 | Fig. 2A legend | Two of four bar colours misnamed | Caption wrong |
| 4 | Fig. S3G legend | Colours given as green/red; figure is blue/magenta | Caption wrong |
| 5 | Fig. S2B legend | "dashed lines indicate means" — they are medians | Caption wrong |
| 6 | Fig. 1I-right | Plot legend trial counts differ from the caption's | Cosmetic |
| 7 | Methods, S3G | "group mean of 3.28%" is the retained-animal mean | Ambiguous wording |
| 8 | Methods, Fig. 2A | The 101.9% sum is weaker evidence than claimed | Over-claim |
| 9 | Fig. S3M legend + Methods | Describes a test on distance travelled; the figure tests distance from each animal's modal posture. The conclusion depends on which | **Text and figure disagree; conclusion reverses** |
| 10 | Fig. S3K legend + Methods | "Normalized to the choice direction"; the figure is un-normalised | Caption and Methods wrong |
| 11 | Methods, Fig. S3J | Distance travelled defined as cumulative frame-to-frame change; the figure uses the within-trial range | Methods and figure disagree |
| 12 | Fig. S3L legend | "Each animal's preferred side"; the code flips each *session*, which differs for 1 of 12 | Minor |
| 13 | Methods, tracking | Interpolation described as linear over surrounding frames only; a second, geometric step is not mentioned | Under-described |

---

## 1. Figure 2B-right: every error bar uses one animal's SD

**Severity: the published figure is wrong.**

The population loop read the error-bar width from `m`, a variable left bound to
the last animal by an earlier loop:

```python
for name, m in subj_metrics.items():      # m ends up = the last animal
    ...
for yi, (name, delta_emp, t_opt, p_emp) in enumerate(meta):
    ax.errorbar(delta_emp, yi, xerr=m["emp_sampling_time_sem"], ...)
```

**Verification.** Measured off the committed
`results/behavior/optimal_sampling/all_subj_aligned.svg`: 17 horizontal error
bars, every one 130.0 px wide. With the axis spanning −2 to 4.5 s over
680.24 px, that is ±0.621 s for all of them — exactly `vgatchr2-sk`'s SD. The
real per-animal SDs span 0.347–0.924 s, a 2.7× range.

Everything else in the panel is correct: all 17 Δ-positions reproduce to
0.00 s.

**Fixed in code** (`behavior/optimalsampling.py`, pinned by
`test_population_panel_uses_each_animals_own_sd`). **The SVG and the
manuscript figure still need regenerating.** No quoted number changes — the
caption cites no values from this panel.

Related: the field was named `emp_sampling_time_sem` but always held `.std()`.
The caption correctly says SD, so only the variable name was wrong; it is
`observed_sampling_time_sd` now.

---

## 2. Figure S3G legend: "n = 9 mice" should be n = 18

The analysis uses all 20 animals minus the 2 excluded by the outlier rule.

**Verification.** `normalizeSTAcrossSubjects` z-scores within animal and drops
nobody; after removing `RDK_WT1` and `RDK_WT6`, 18 animals remain, and 18 still
have a non-null `Stay`. The figure itself carries no "n=" text, so the caption
is the only place the number appears.

The 9 appears to be borrowed from **S3H**, which *is* restricted to the nine
mice contributing more than 2,500 trials.

---

## 3. Figure 2A legend: two bar colours misnamed

The caption reads "stimulus coherence (27.15%±6.87%, grey), reward history
(34.36%±7.94%, **blue**), motor bias (21.02%±5.82%, **red**), Win-stay
(19.32%±5.46%, violet)".

**Verification.** Fill colours in the committed
`results/behavior/model_OLS_var_explained_w_filter.svg` are `#808080` (grey,
DVabs), `#ffa500` (**orange**, PrevOutcomeCount), black (ChoiceLeft), `#800080`
(purple, Stay).

So reward history is orange, not blue, and motor bias is black, not red. Grey
and violet are right. The percentages all match exactly.

---

## 4. Figure S3G legend: "win (green) and lose (red)"

**Verification.** Stroke colours in the committed
`results/behavior/win_lose_stay_switch_Mice Strategy over decision times.svg`
are `#1e90ff` (dodgerblue) for Win and `#bf00bf` (magenta) for Lose. The red,
orange and yellow strokes present in the file are the Fast/Typical/Slow
background bands, not the curves.

---

## 5. Figure S2B legend: "dashed lines indicate means"

The code draws `ax.axvline(values.median(), ...)` — the **median** of the
per-subject values, not the mean. Either the caption or the line should change.

---

## 6. Figure 1I-right: plot legend disagrees with the caption

The plot's own legend reads "Fast n=20,864 Trials / Slow n=20,054 Trials",
counting every trial in each tertile. The caption says "fast = 20,851 trials,
slow = 20,044 trials".

**Verification.** Both are reproducible: 20,864 / 20,054 are the raw tertile
counts, and 20,851 / 20,044 are the counts after dropping trials with no
preceding choice — which the Methods require ("Trials with no preceding choice
within the same session were excluded throughout").

So **the caption is the more accurate pair** and the plot legend overstates by
13 and 10 trials. Cosmetic, but the figure should probably use the same rule it
analyses under.

---

## 7. Methods (Figure S3G): "a group mean of 3.28%"

The exclusion rule is stated as two mice having "18.62% and 13.26% of trials,
compared with a group mean of 3.28%".

**Verification.** Across all 20 animals the mean outlier fraction is **4.55%**.
Across the 18 *retained* animals it is **3.28%** — the quoted figure. The
number is right; "group mean" just does not say which group. Suggest "mean of
the retained animals".

The rest of the rule reproduces exactly: `RDK_WT6` 18.62%, `RDK_WT1` 13.26%,
and these are precisely the two above the 90th percentile (12.04%).

---

## 8. Methods (Figure 2A): the 101.9% sum proves less than claimed

The Methods say the four leave-one-out contributions "summed to 101.9% of the
total log-likelihood improvement, confirming that little variance was shared
between predictors."

Log-likelihood improvement is logarithmic in the residual sum of squares, so
leave-one-out shares are only near-additive while the predictors explain
little. Two **perfectly orthogonal** regressors sum to ~100% at R² = 0.03 but
to ~160% at R² = 0.96.

**Verification.** Per-animal R² here is **0.006–0.081** (median 0.021) —
exactly the regime where the sum lands near 100% whether or not the design is
orthogonal. A sweep at fixed orthogonality gives sums of 158.7% at noise 0.3,
125.9% at noise 1, 110.6% at 2, 101.7% at 5, 100.1% at 12.

The conclusion still holds — the **condition numbers (14.3–22.2)** are a direct
collinearity check and they are reported. It is the supporting argument that is
too strong. A *low* sum would indicate collinearity; a ~100% sum mostly
indicates weak effects.

Both regimes are pinned in `behavior/tests/test_varexplained.py`.

---

## 9. Figure S3M: the text and the figure test different quantities

**Severity: the stated conclusion depends on which one is meant.**

The Methods say: "we compared the distributions of rotation-angle distance
travelled per trial across strategies for each animal (Figure S3M)", and the
legend: "(M) Animal-wise comparison of rotation-angle distance travelled per
trial (Kruskal-Wallis)". The Results cite S3J-M for "little to no
correlation" between posture or movement and sampling time.

The figure does something else. For each trial it takes the **mean** rotation
angle, finds each animal's **modal** angle (5 degree bins), and tests
|trial mean - mode| across the three strategies. That is a posture measure, not
a distance-travelled one; the y-axis is labelled "Distance from mode Rotation
Angle (deg)".

**Verification**, per-animal Kruskal-Wallis with Holm across the four mice, on
the same 76,311 tracked frames:

| Quantity tested | MLA-73 | MLA-74 | MLA-75 | MLA-76 | Significant |
|---|---|---|---|---|---|
| distance from modal angle (**the figure**) | 0.073 | 0.722 | 0.105 | 0.722 | **0/4** |
| distance travelled, as within-trial range | 0.024 | 0.028 | 0.0009 | 0.036 | 4/4 |
| distance travelled, as cumulative change (**the Methods' definition**) | <0.0001 | <0.0001 | <0.0001 | <0.0001 | 4/4 |

(Holm-corrected p. Every group failed Shapiro-Wilk in all three versions, so
the Kruskal-Wallis gate is satisfied throughout.)

**Why the figure's quantity is the defensible one.** Distance travelled
scales with trial length, and the strategies *are* trial length: slow trials
have a median of 44 video frames against 18 for fast ones. Cumulative distance
correlates with frame count (Spearman rho = 0.48), and per frame the animals
actually move *less* on slow trials (median 0.62 against 0.94 degrees). So a
distance-travelled test mostly rediscovers that slow trials are longer. The
posture measure is a trial mean and does not grow with duration.

The likely reading is that the analysis was changed to remove that confound and
the text was not updated, but that is an inference. **Decide which is intended:**
if the figure, the Methods and legend sentences for S3M need rewording; if the
text, the figure and its "0/4" change.

---

## 10. Figure S3K: "normalized to the choice direction"

The legend reads "(K) Polar distribution of the mean centroid rotation angle per
trial, normalized to the choice direction", and the Methods "relative to the
choice direction (Figure S3K)".

**Verification.** The committed panel is un-normalised: the notebook call
passes `choice_normed=False`, the saved file has no `_choice_normed` suffix and
no "(Normed to Choice Direction)" in its title, and all 84 bars match a
`choice_normed=False` render (r = 1.000000, constant area ratio) but not a
`True` one (r = 0.94). Confirmed by the author as intended. The legend and the
Methods sentence are what need changing.

---

## 11. Figure S3J: distance travelled is defined differently

The Methods define "the rotation-angle distance travelled as the cumulative
absolute frame-to-frame change in centroid rotation angle". The code uses the
within-trial **range**, `abs(max - min)`.

**Verification.** On the published (10%-trimmed) frames the two differ by a
median factor of 2.2 per trial; median range is 5.4 / 5.9 / 5.4 degrees for
fast / typical / slow, against 9.2 / 11.7 / 15.5 for the cumulative version.
The cumulative one rises with strategy for the trial-length reason in #9;
the range does not. Either the Methods sentence or the panel should change,
and the choice should match whatever is decided for #9.

---

## 12. Figure S3L: "each animal's preferred side"

The legend says the angles are "normalized to each animal's preferred side".
The code flips each **session** that leans left, not each animal.

**Verification.** For 11 of the 12 sessions the two rules agree. The exception
is MLA-76 on 2025-10-24 (day 3), which is balanced on its own and is left
unflipped per session, while the animal as a whole leans left and would be
flipped. Small effect on the panel; either say "session" or flip per animal.

---

## 13. Tracking Methods: interpolation is under-described

The Methods say "Occasional missing SLEAP key points were linearly
interpolated from surrounding frames". The code does that, but only across gaps
of up to three frames within a trial; any limb still missing is then
**reconstructed geometrically** from the opposite limb using that session's
average inter-limb offset. The second step is not mentioned.

---

## Not an error, but worth a sentence

`docs/manuscript-methods-map.md` records that Methods eq. (3) has a boundary
solution: when evidence accumulates too slowly to pay for the time it costs,
the reward-rate optimum is the shortest allowed sampling time. None of the 17
fitted animals is in that regime (all land between 0.53 s and 1.88 s), so it is
a property of the equation rather than an observation — but it bears on the
paper's framing of impulsivity, since under the paper's own normative objective
there are parameter regimes where sampling briefly *is* reward-maximising
rather than a failure to maximise.
