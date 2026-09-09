'''Tests for the sampling-time distribution across contexts (Figure S2B).

The panel lives or dies on the order of two operations: subjects are z-scored
**before** they are split by context, not after. Doing it the other way round
centres every context on zero and erases the difference the figure exists to
show. The first tests pin that ordering directly.

The animal group is the documented exception. With a single context, the
per-subject rule is degenerate — every animal's mean z-score becomes exactly
zero — so the animals are pooled instead, and both behaviours are pinned so
the asymmetry cannot be "tidied up" by accident.
'''
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ..stdistribution import (CONTEXT_COL, SUBJECT_KEY, collectContexts,
                              plotSTDistribution, subjectContextMeans)

ACCURACY, SPEED = "ReactionTime", "Competition"
SESSION_TYPES = [("Accuracy", ACCURACY), ("Speed", SPEED)]
COLORS = {"Accuracy": "C0", "Speed": "C1", "Mice": "C2"}
N = 200


def _humans(specs, seed=0):
    '''specs: {name: {session_type: (mean_seconds, sd)}}.'''
    rng = np.random.default_rng(seed)
    rows = []
    for name, contexts in specs.items():
        for session_type, (mean, sd) in contexts.items():
            rows.append(pd.DataFrame({
                SUBJECT_KEY: name, CONTEXT_COL: session_type,
                "calcStimulusTime": rng.normal(mean, sd, N),
                "ChoiceCorrect": rng.integers(0, 2, N).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def _mice(specs, seed=1):
    rng = np.random.default_rng(seed)
    return pd.concat([pd.DataFrame({
        SUBJECT_KEY: name, "calcStimulusTime": rng.normal(mean, sd, N),
        "ChoiceCorrect": rng.integers(0, 2, N).astype(float)})
        for name, (mean, sd) in specs.items()], ignore_index=True)


def _slowAccuracyCohort(seed=0):
    '''Every subject samples ~1 s longer under the accuracy instruction.'''
    return _humans({f"S{i}": {ACCURACY: (2.0 + i * 0.3, 0.3),
                              SPEED: (1.0 + i * 0.3, 0.3)}
                    for i in range(6)}, seed=seed)


# --------------------------------------------------------------------------
# The ordering that makes the panel work
# --------------------------------------------------------------------------

def test_between_context_difference_survives_normalisation():
    '''Pooling a subject's contexts before splitting keeps the offset.'''
    groups = collectContexts(_slowAccuracyCohort(), colors=COLORS,
                             session_types=SESSION_TYPES)
    accuracy, speed = (g["values"].mean() for g in groups)
    assert accuracy > speed
    assert accuracy - speed > 1.0


def test_between_subject_offsets_are_removed():
    '''A uniformly slow subject must not sit apart from a fast one.

    Both subjects have the same context *gap*; only their overall pace
    differs. After per-subject normalisation their points should coincide.
    '''
    df = _humans({"Slow": {ACCURACY: (5.0, 0.3), SPEED: (4.0, 0.3)},
                  "Fast": {ACCURACY: (1.2, 0.3), SPEED: (0.2, 0.3)}})
    groups = collectContexts(df, colors=COLORS, session_types=SESSION_TYPES)
    for group in groups:
        assert group["values"].std() < 0.15


def test_each_subject_is_centred_across_its_two_contexts():
    '''The two context means of one subject average to ~0, not each to 0.'''
    groups = collectContexts(_slowAccuracyCohort(), colors=COLORS,
                             session_types=SESSION_TYPES)
    per_subject = sum(g["values"].droplevel(CONTEXT_COL) for g in groups) / 2
    np.testing.assert_allclose(per_subject.values, 0, atol=0.05)


def test_groups_report_their_own_trial_counts():
    df = _humans({"S1": {ACCURACY: (2.0, .3), SPEED: (1.0, .3)},
                  "S2": {ACCURACY: (2.0, .3), SPEED: (1.0, .3)}})
    groups = collectContexts(df, colors=COLORS, session_types=SESSION_TYPES)
    assert [g["n_trials"] for g in groups] == [2 * N, 2 * N]
    assert all(len(g["values"]) == 2 for g in groups)


def test_session_type_order_follows_the_request():
    groups = collectContexts(_slowAccuracyCohort(), colors=COLORS,
                             session_types=[("Speed", SPEED),
                                            ("Accuracy", ACCURACY)])
    assert [g["label"] for g in groups] == ["Speed", "Accuracy"]


# --------------------------------------------------------------------------
# The animal exception
# --------------------------------------------------------------------------

def test_per_animal_normalisation_collapses_the_group():
    '''Documents why the animals are not treated like the humans.'''
    groups = collectContexts(_slowAccuracyCohort(),
                             _mice({"M1": (1.0, .4), "M2": (2.5, .4)}),
                             colors=COLORS, session_types=SESSION_TYPES,
                             animals_per_subject=True)
    animals = groups[-1]["values"]
    assert animals.std() == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(animals.values, 0, atol=1e-9)


def test_pooled_animals_keep_between_animal_differences():
    groups = collectContexts(_slowAccuracyCohort(),
                             _mice({"Quick": (1.0, .4), "Slow": (2.5, .4)}),
                             colors=COLORS, session_types=SESSION_TYPES)
    animals = groups[-1]["values"]
    assert animals.std() > 0.5
    assert animals["Slow"] > animals["Quick"]


def test_pooled_animals_are_the_default():
    groups = collectContexts(_slowAccuracyCohort(),
                             _mice({"M1": (1.0, .4), "M2": (2.5, .4)}),
                             colors=COLORS, session_types=SESSION_TYPES)
    assert groups[-1]["values"].std() > 0


def test_animals_without_a_choice_are_excluded():
    mice = _mice({"M1": (1.0, .4)})
    mice.loc[mice.index[:50], "ChoiceCorrect"] = np.nan
    groups = collectContexts(_slowAccuracyCohort(), mice, colors=COLORS,
                             session_types=SESSION_TYPES)
    assert groups[-1]["n_trials"] == N - 50


def test_animals_are_optional():
    groups = collectContexts(_slowAccuracyCohort(), colors=COLORS,
                             session_types=SESSION_TYPES)
    assert [g["label"] for g in groups] == ["Accuracy", "Speed"]


# --------------------------------------------------------------------------
# subjectContextMeans
# --------------------------------------------------------------------------

def test_subject_means_are_one_value_per_subject_per_context():
    df = _humans({"S1": {ACCURACY: (2.0, .3), SPEED: (1.0, .3)},
                  "S2": {ACCURACY: (2.0, .3), SPEED: (1.0, .3)}})
    assert len(subjectContextMeans(df, CONTEXT_COL)) == 4
    assert len(subjectContextMeans(df)) == 2


# --------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------

def _figure(**kwargs):
    return plotSTDistribution(
        _slowAccuracyCohort(), _mice({"M1": (1.0, .4), "M2": (2.5, .4),
                                      "M3": (1.7, .4), "M4": (2.1, .4)}),
        colors=COLORS, session_types=SESSION_TYPES, verbose=False, **kwargs)


def test_panel_runs_and_returns_an_omnibus_result():
    result = _figure()
    assert result["test_name"] in ("f_oneway", "kruskal")
    assert 0 <= result["pvalue"] <= 1
    assert result["posthoc"].shape == (3, 3)
    plt.close("all")


def test_panel_marks_a_median_per_group():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _figure(axes=axes)
    dashed = [ln for ln in axes[0].lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 3
    plt.close(fig)


def test_panel_saves_under_the_prefix(tmp_path):
    _figure(save_prefix=tmp_path, save_figs=True)
    assert (tmp_path / "humans_mice_reaction_time_dist.svg").exists()
    plt.close("all")


def test_panel_refuses_to_save_without_a_prefix():
    with pytest.raises(ValueError, match="save_prefix"):
        _figure(save_figs=True)
