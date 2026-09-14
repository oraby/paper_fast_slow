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
| 9 | Fig. S3M legend + Methods | Describes a test on distance travelled; the figure tests distance from the usual posture, which is intended | **Text wrong** — replacement drafted |
| 10 | Fig. S3K legend + Methods | "Normalized to the choice direction"; the committed figure is un-normalised | **Figure to regenerate** choice-normed next revision; text stays |
| 11 | Methods, Fig. S3J | Distance travelled defined as cumulative frame-to-frame change; the figure uses the within-trial range, which is intended | **Text wrong** — replacement drafted |
| 12 | Fig. S3L legend + Methods | "Each animal's preferred side"; per session is intended and is what the code does | **Text wrong** — replacement drafted |
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

## 9. Figure S3M: the text describes the wrong quantity

**Decided: the figure is right; the legend and Methods change.**
Replacement text: `code/rlmodel/methods_model_revision.md`, Blocks 11 and 12.

The Methods (PDF lines 1414–1419) say the test compares "rotation-angle distance
travelled per trial across strategies", and the legend (PDF lines 2123–2126)
"Animal-wise comparison of rotation-angle distance travelled per trial … confirming
that movement during sampling does not differ across strategies".

The figure tests **posture**, not movement: each trial's mean rotation angle is
scored by its absolute distance from the animal's usual posture — its modal mean
angle across all sessions, in 5° bins — and the three strategies are compared per
animal. The y-axis ("Distance from mode Rotation Angle (deg)") is correct. J and M
are deliberately different measurements: J shows posture changes little within a
trial, which is what makes a single per-trial value a valid proxy in K, L and M.

**Verification.** Per-animal Kruskal–Wallis with Holm across the four mice:

| Quantity tested | MLA-73 | MLA-74 | MLA-75 | MLA-76 | Significant |
|---|---|---|---|---|---|
| distance from usual posture (**the figure, intended**) | 0.073 | 0.722 | 0.105 | 0.722 | **0/4** |
| distance travelled, as within-trial range | 0.024 | 0.028 | 0.0009 | 0.036 | 4/4 |
| distance travelled, as cumulative change | <0.0001 | <0.0001 | <0.0001 | <0.0001 | 4/4 |

So the current wording would describe a test that gives the opposite result.
Distance travelled grows with trial length — slow trials have a median of 44 video
frames against 18 for fast — so it is not a posture measure. The replacement text
also states the Holm correction, which was applied but not mentioned (raw
0.018 / 0.361 / 0.035 / 0.497).

---

## 10. Figure S3K: "normalized to the choice direction"

**Decided: the text stays; the figure is regenerated with `choice_normed=True` in the
next revision.**

The legend (PDF lines 2121–2122) and Methods (PDF lines 1410–1412) describe S3K as
normalised to the choice direction. The committed panel is not: the notebook call
passes `choice_normed=False`, the saved file has no `_choice_normed` suffix, and all
84 bars match a `False` render (r = 1.000000) but not a `True` one (r = 0.94).

**To do in the next revision:** in `Tracking.ipynb`, the S3J/S3K cell
(heading "Extended Fig. 4b-c") calls `plotCentroids(..., choice_normed=False, ...)`.
Pass `choice_normed=True` for S3K. That call also draws S3J, and the travel panel is
skipped whenever a normalisation is on, so S3J needs its own call with
`choice_normed=False` (or keep the existing call for J and add one for K). The file
will be saved as `centroid_rotation_choice_normed_All Tracked Subjects.svg`.
Nothing in the text changes.

---

## 11. Figure S3J: distance travelled is defined differently

**Decided: the range is the intended metric; the Methods change.**
Replacement text: `code/rlmodel/methods_model_revision.md`, Blocks 10 and 12.

The Methods (PDF lines 1406–1410) define "the rotation-angle distance travelled as
the cumulative absolute frame-to-frame change in centroid rotation angle". The code
uses the within-trial **range**, `abs(max − min)`, after discarding each trial's
lowest and highest 10% of angles.

Range is the better metric because tracking is not perfect: frame-to-frame
differences accumulate tracking jitter along with real movement, so a cumulative sum
grows with the number of frames even for a still animal. (On these data the
cumulative version is a median 2.2× the range and rises with strategy — 9.2 / 11.7 /
15.5° for fast / typical / slow, against 5.4 / 5.9 / 5.4° for the range — which is
what you would expect from longer trials collecting more jitter, not from more
movement.) The replacement also states the 10% trim and the per-session
normalisation, which the published panel uses.

---

## 12. Figure S3L: "each animal's preferred side"

**Decided: per session is intended; the legend and Methods change.**
Replacement text: `code/rlmodel/methods_model_revision.md`, Blocks 10 and 12.

The legend (PDF lines 2122–2123) and Methods (PDF line 1412) say "each animal's
preferred side". The code mirrors each **session** whose frames lean predominantly to
one side, independently of the animal's other sessions, and that is the intended
rule. For reference, a per-animal rule would treat one of the 12 sessions differently
(MLA-76, day 3, which is balanced on its own).

---

## 13. Tracking Methods: interpolation is under-described

**Open — not yet decided.** Not covered by Blocks 10–12.

The Methods say "Occasional missing SLEAP key points were linearly
interpolated from surrounding frames". What the code does, measured on all
76,311 frames:

- **16.7% of limb keypoints** (50,843) were missing before interpolation, so
  "occasional" undersells it.
- **Stage 1, in time:** each coordinate is interpolated linearly across its
  trial. Gaps of *any* length are filled — an earlier version of this note said
  "up to three frames", which was wrong: the code repeats a one-frame fill
  until nothing changes. Gaps at the start or end of a trial take the nearest
  valid value rather than being interpolated. 35,827 keypoints were filled
  this way.
- **Stage 2, from the other limbs:** a keypoint missing in *every* frame of its
  trial is rebuilt from the limbs present, using that session's average
  inter-limb offset. **14,438 keypoints, in 18.2% of frames**, came from this
  step, and it is not mentioned.
- 578 keypoints stayed missing (a lone hind limb has nothing to rebuild from);
  every frame still had a centroid.

Also unstated: tail detections scoring below 0.25, or lying nearer the head than
both hind limbs, are discarded first; and the centroid coordinates are
truncated to whole pixels before averaging.

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
