"""Aggregate bar figures — the ``Fig. 1l`` form, driven by an ``EvalSpec`` list.

Extracted from ``model_analysis.ipynb``'s ``Plot Aggregates`` → ``Fig. 1l``
cell. The layout is unchanged: one axes, one bar group per model, one bar per
metric within a group, subject dots overlaid, group label above.

Two generalizations over the original cell:

- Group x-positions honour ``EvalSpec.gap_after`` instead of a hardcoded
  ``global_offset_x += 3``, so a figure can open a half-column of space
  between sub-families.
- Repeat evaluations (``Iteration``) render as a per-subject spread across the
  bar width with an STD whisker. See :func:`plot_aggregates`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt


BAR_WIDTH = 0.6
# Blank x-slots between consecutive model groups. A group occupies
# len(metric_keys) slots, so the original cell's "+= 3" for its two metrics is
# this 1 plus the group's own 2 — expressed this way it stays correct for any
# number of metrics.
_INTER_GROUP_GAP = 1
_LABEL_Y = 1.3          # group label height (cell 24)
_TITLE_PAD = 28         # points; clears the group labels drawn at _LABEL_Y

# Short x-tick label per metric, and whether it is an R² (vs. a correlation).
_METRIC_TICK = {"R2_Psych": "Psych", "RewardRateCorr": "RR",
                "R2_WinLose": "WinLose"}


def _tick_label(metric):
    return _METRIC_TICK.get(metric, metric)


def _report_oob(data, metric, spec):
    """Print subjects outside the 1.5·IQR fence — the notebook's OOB report.

    Reporting only; unlike the sibling subject-colour cell, Fig. 1l keeps
    outliers in the bar.
    """
    q25, q75 = data.quantile(.25), data.quantile(.75)
    iqr = q75 - q25
    oob = data[(data < q25 - 1.5 * iqr) | (data > q75 + 1.5 * iqr)]
    if len(oob):
        print(f"OOB {metric} - {spec.label!r}:")
        for name, value in oob.items():
            print(f"\t{name}: {value}")


def _subject_stats(metrics_df, spec, metric):
    """``(mean_per_subject, std_per_subject)`` for one spec × metric.

    Each subject contributes the mean of its N evaluations; ``std`` is that
    subject's spread across evaluations (all-NaN when N == 1).
    """
    sub = metrics_df[metrics_df.SpecLabel == spec.label]
    grouped = sub.groupby("Name")[metric]
    return grouped.mean(), grouped.std()


def subsample_evaluations(metrics_df, num_evaluations):
    """Keep the first ``num_evaluations`` iterations (``None`` keeps all).

    Iteration *i* is seed *i* and the seeds are independent, so the first *n*
    is a valid, deterministic subsample — no RNG at plot time, and the result
    is identical to having collected only *n*. That is what lets one expensive
    collection at a large N serve any smaller N at plot time.
    """
    if num_evaluations is None:
        return metrics_df
    available = int(metrics_df.Iteration.nunique())
    if num_evaluations > available:
        raise ValueError(
            f"Asked to plot {num_evaluations} evaluations but the metrics "
            f"frame only holds {available}. Re-collect with "
            f"num_evaluations >= {num_evaluations}.")
    if num_evaluations < 1:
        raise ValueError(
            f"num_evaluations must be >= 1, got {num_evaluations}")
    return metrics_df[metrics_df.Iteration < num_evaluations]


def plot_aggregates(metrics_df, specs, *,
                    metric_keys=("R2_Psych", "RewardRateCorr"),
                    num_evaluations=None,
                    ax=None, figsize=(12, 8), title=None):
    """Grouped bars: one group per ``EvalSpec``, one bar per metric.

    ``metrics_df`` is :func:`aggregate.collect_metrics`'s tidy frame. The bar is
    always the mean over the **per-subject means**, with a SEM whisker over
    subjects (n = number of subjects) — so repeat evaluations tighten each
    subject's estimate without inflating the across-subject n.

    With one evaluation the dots sit at the group's exact x with no whisker
    (identical to the original figure). With N > 1 each subject's dot is spread
    evenly across the bar width and carries an STD whisker over its N
    evaluations.

    ``num_evaluations`` plots only the first *n* of the collected iterations
    (see :func:`subsample_evaluations`); ``None`` uses every one present.
    """
    metrics_df = subsample_evaluations(metrics_df, num_evaluations)
    num_evals = int(metrics_df.Iteration.nunique())
    num_subjects = int(metrics_df.Name.nunique())
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize)

    x_ticks, x_tick_labels = [], []
    global_offset_x = 0.0
    for spec in specs:
        for offset_x, metric in enumerate(metric_keys):
            x = global_offset_x + offset_x
            means, stds = _subject_stats(metrics_df, spec, metric)
            if not len(means):
                raise ValueError(
                    f"No rows for spec {spec.label!r} in metrics_df — was it "
                    f"collected with a different spec list?")
            _report_oob(means, metric, spec)

            bar_mean, bar_sem = means.mean(), stats.sem(means)
            ax.bar(x, bar_mean, yerr=bar_sem, color=spec.color,
                   width=BAR_WIDTH, label=spec.label)

            if num_evals == 1:
                dot_x = np.full(len(means), x)
            else:
                # Span the bar from start to end so each subject's spread is
                # readable instead of overplotted at the centre.
                half = BAR_WIDTH / 2
                dot_x = (np.linspace(x - half, x + half, len(means))
                         if len(means) > 1 else np.array([x]))
            ax.errorbar(dot_x, means.values,
                        yerr=None if num_evals == 1 else stds.values,
                        fmt="o", markerfacecolor="white", markeredgecolor="k",
                        markersize=4, ecolor="k", elinewidth=1, capsize=2,
                        linestyle="none", zorder=3)

            r_str = "$R^2$" if metric.startswith("R2") else "r"
            # Clear whatever is actually drawn: the dots, their STD whiskers
            # (multi-eval only), and the bar's own SEM whisker.
            tops = [means.max(), bar_mean + bar_sem, 0]
            if num_evals > 1:
                tops.append((means + stds.fillna(0)).max())
            max_y = max(tops)
            ax.annotate(f"{r_str}\n{bar_mean:.2f} ±{bar_sem:.2f}",
                        xy=(x, max_y + 0.02), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10)
            x_ticks.append(x)
            x_tick_labels.append(_tick_label(metric))

        # Group label, centred over the group's bars.
        ax.annotate(spec.label,
                    xy=(global_offset_x + (len(metric_keys) - 1) / 2, _LABEL_Y),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", fontsize=12)
        global_offset_x += len(metric_keys) + _INTER_GROUP_GAP + spec.gap_after

    ax.axhline(0, c="black", ls="--")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Correlation Metric ($R^2$, r)")
    ax.set_ylim(-1 if "RewardRateCorr" in metric_keys else 0, _LABEL_Y)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.spines[["top", "right"]].set_visible(False)

    if title is None:
        title = (f"bar: mean ± SEM across subjects (n={num_subjects})")
        if num_evals > 1:
            title += (f" · dot: subject mean ± STD across "
                      f"{num_evals} evaluations")
        else:
            title += " · dot: one subject (single evaluation)"
    # The group labels sit at the axes top (_LABEL_Y == the ylim ceiling), so
    # the title needs clearance or it lands on top of them.
    ax.set_title(title, fontsize=10, pad=_TITLE_PAD)
    return ax
