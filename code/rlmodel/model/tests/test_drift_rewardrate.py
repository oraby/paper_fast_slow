"""DriftGain-RewardRate: the reward rate modulating the DRIFT.

Third reward-rate channel alongside NoiseGain (``sigma *= r``) and Bound
(``b = BOUND*(2-r)``). The per-step update is::

    d(t) = d(t-1) + DV*V*g(r_t)*dt + S*sqrt(dt)*eps

with the noise sigma and the bound both FLAT. Unlike the Bound channel this
is not a rescaling of an equivalent model, so there is no analytic identity
to lean on — what these tests pin instead is that

1. the gain lands on the coherence drift and nowhere else (sigma untouched),
   and
2. the three MLE compute paths (rowwise reference, batched, population/DE)
   all apply it identically, which is the failure mode that would otherwise
   make DE optimize a different model than the final eval scores.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from .. import drift as drift_mod
from ..bias import BIAS_FN_DICT
from ..drift import (DRIFT_FN_DICT, _driftClassic, _driftGainRewardRate,
                     channel_for_drift, display_alias_for_drift,
                     is_rewardrate_alias, resolve_drift_alias,
                     user_facing_drift_keys)
from ..fit import evolveFP
from ..mle import (MLEModelConfig, _compute_latent_arrays,
                   evaluate_neg_loglik, objective_from_population,
                   prepare_mle_data)
from ..mle_reeval import parse_fit_filename
from ..noise import NOISE_FN_DICT
from ..state_updates import drift_scale_from_reward_rate


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

_DRIFT_KEYS = ("DriftGain-RewardRate", "DriftGain(1+r)-RewardRate")


# ---------------------------------------------------------------------------
# The mapping g(r)
# ---------------------------------------------------------------------------

def test_drift_scale_2_minus_r_mapping():
    """Default mapping: g(r) = 2 - r, i.e. d += DV*(2V - r*V). A HIGH
    reward rate WEAKENS the drift. Bounded in [1, 2] over r in [0, 1]."""
    assert drift_scale_from_reward_rate(0.0) == 2.0
    assert drift_scale_from_reward_rate(0.5) == 1.5
    assert drift_scale_from_reward_rate(1.0) == 1.0
    r = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    g = drift_scale_from_reward_rate(r)
    np.testing.assert_array_equal(g, 2.0 - r)
    assert np.all(np.diff(g) < 0)                # decreasing in reward rate
    assert g.min() >= 1.0 and g.max() <= 2.0


def test_drift_scale_1_plus_r_mapping():
    """Flipped mapping: g(r) = 1 + r, i.e. d += DV*(V + r*V). A HIGH
    reward rate STRENGTHENS the drift — the same speed direction as the
    noise and bound channels. Same [1, 2] range, mirrored."""
    r = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    g = drift_scale_from_reward_rate(r, "1+r")
    np.testing.assert_array_equal(g, 1.0 + r)
    assert np.all(np.diff(g) > 0)                # increasing in reward rate
    assert g.min() >= 1.0 and g.max() <= 2.0
    # The two mappings are mirror images about r = 0.5.
    np.testing.assert_allclose(
        drift_scale_from_reward_rate(r, "1+r"),
        drift_scale_from_reward_rate(1.0 - r, "2-r"))


def test_drift_scale_rejects_unknown_map():
    with pytest.raises(ValueError, match="Unknown reward-rate drift map"):
        drift_scale_from_reward_rate(0.5, "2+r")


def test_drift_scale_preserves_nan():
    """logic.py NaN-pads trial slots past a session's real length; the
    gain must propagate that rather than turning it into a number."""
    assert np.isnan(drift_scale_from_reward_rate(np.nan))
    assert np.isnan(drift_scale_from_reward_rate(np.nan, "1+r"))


# ---------------------------------------------------------------------------
# Chi2 / generative drift functions
# ---------------------------------------------------------------------------

def _chisq_inputs(n_trials=4, n_steps=40, seed=0):
    rng = np.random.default_rng(seed)
    dvs = np.array([-0.5, -0.25, 0.25, 0.5])[:n_trials]
    noise = rng.standard_normal((n_trials, n_steps)) * np.sqrt(0.005)
    return dvs, noise


def test_chisq_drift_scales_drift_and_leaves_noise_alone():
    """Against Classic on the SAME noise draw: the per-step increment
    differs by exactly the coherence-drift gain, and the noise the two
    integrate is bit-identical (proving sigma is untouched — the
    NoiseGain channel would have multiplied it by r)."""
    dvs, noise = _chisq_inputs()
    reward_rate = np.array([0.1, 0.4, 0.7, 1.0])
    dt, drift_coef, noise_sigma = 0.005, 1.3, 1.5

    dx_classic = _driftClassic(
        starting_point=np.zeros(len(dvs)), nondectime=0.0, noise=noise.copy(),
        drift_coef=drift_coef, dvs=dvs, dt=dt, noise_sigma=noise_sigma)
    dx_drift = _driftGainRewardRate(
        starting_point=np.zeros(len(dvs)), nondectime=0.0, noise=noise.copy(),
        drift_coef=drift_coef, dvs=dvs, dt=dt, noise_sigma=noise_sigma,
        RewardRate=reward_rate, RR_DRIFT_MAP="2-r")

    # d_drift - d_classic accumulates only the extra drift, (g-1)*V*DV*dt
    # per step, so the difference grows linearly in the step index.
    gain = (2.0 - reward_rate)[:, None]
    steps = np.arange(noise.shape[1])[None, :]
    expected_extra = (gain - 1.0) * drift_coef * dvs[:, None] * dt * steps
    np.testing.assert_allclose(dx_drift - dx_classic, expected_extra, atol=1e-12)


def test_chisq_drift_1_plus_r_uses_the_other_mapping():
    dvs, noise = _chisq_inputs()
    reward_rate = np.array([0.1, 0.4, 0.7, 1.0])
    kwargs = dict(starting_point=np.zeros(len(dvs)), nondectime=0.0,
                  drift_coef=1.3, dvs=dvs, dt=0.005, noise_sigma=1.5,
                  RewardRate=reward_rate)
    dx_2r = _driftGainRewardRate(noise=noise.copy(), RR_DRIFT_MAP="2-r", **kwargs)
    dx_1r = _driftGainRewardRate(noise=noise.copy(), RR_DRIFT_MAP="1+r", **kwargs)
    # Same noise, different gain => the paths must differ everywhere the
    # coherence drift is nonzero.
    assert not np.allclose(dx_2r, dx_1r)
    # ...and agree exactly at r = 0.5, where 2-r == 1+r == 1.5.
    flat = np.full(len(dvs), 0.5)
    a = _driftGainRewardRate(noise=noise.copy(), RR_DRIFT_MAP="2-r",
                             **{**kwargs, "RewardRate": flat})
    b = _driftGainRewardRate(noise=noise.copy(), RR_DRIFT_MAP="1+r",
                             **{**kwargs, "RewardRate": flat})
    np.testing.assert_allclose(a, b, atol=1e-15)


# ---------------------------------------------------------------------------
# Registry / alias layer
# ---------------------------------------------------------------------------

def test_all_alias_channels_resolve_to_registry_keys():
    for alias, channels in drift_mod._REWARDRATE_ALIASES.items():
        assert is_rewardrate_alias(alias)
        for channel, impl in channels.items():
            assert impl in DRIFT_FN_DICT, (alias, channel, impl)


def test_use_drift_rr_overrides_scale_bound():
    """``--use-drift-rr`` wins: --scale-bound then only picks the fitted
    scale axis, it no longer selects the Bound- implementation."""
    for scale_bound in (False, True):
        assert resolve_drift_alias(
            "RewardRate", scale_bound, use_drift_rr=True,
            drift_rr_map="2-r") == "DriftGain-RewardRate"
        assert resolve_drift_alias(
            "RewardRate", scale_bound, use_drift_rr=True,
            drift_rr_map="1+r") == "DriftGain(1+r)-RewardRate"
    # Without it, the legacy noise/bound routing is unchanged.
    assert resolve_drift_alias("RewardRate", False) == "NoiseGain-RewardRate"
    assert resolve_drift_alias("RewardRate", True) == "Bound-RewardRate"


def test_resolve_drift_alias_rejects_unknown_map():
    with pytest.raises(ValueError, match="Unknown reward-rate drift map"):
        resolve_drift_alias("RewardRate", False, use_drift_rr=True,
                            drift_rr_map="nope")


def test_is_rewardrate_alias_gates_the_drift_channel():
    """The shared CLI / GUI gate: only R-learning aliases may enable it."""
    assert is_rewardrate_alias("RewardRate")
    for other in ("Classic", "NoiseGain-RewardRate", "DriftGain-RewardRate"):
        assert not is_rewardrate_alias(other)


def test_display_alias_keeps_drift_channel_distinct():
    """NoiseGain/Bound collapse onto the shared alias (same model, two
    scale axes); DriftGain must NOT, or its fits would land on the
    NoiseGain row of the model_compare / aggregate grid."""
    assert display_alias_for_drift("NoiseGain-RewardRate") == "RewardRate"
    assert display_alias_for_drift("Bound-RewardRate") == "RewardRate"
    assert display_alias_for_drift("DriftGain-RewardRate") == "RewardRate (Drift)"
    assert (display_alias_for_drift("DriftGain(1+r)-RewardRate")
            == "RewardRate (Drift 1+r)")
    assert display_alias_for_drift("Classic") == "Classic"


def test_channel_for_drift_round_trips_through_resolve():
    """``channel_for_drift`` is the inverse the GUI's batch save-figures
    loop uses to drive the widgets from a saved-fit drift name."""
    for impl in _DRIFT_KEYS + ("NoiseGain-RewardRate", "Bound-RewardRate"):
        alias, channel = channel_for_drift(impl)
        if channel.startswith("drift:"):
            resolved = resolve_drift_alias(
                alias, False, use_drift_rr=True,
                drift_rr_map=channel.split(":", 1)[1])
        else:
            resolved = resolve_drift_alias(alias, channel == "bound")
        assert resolved == impl
    assert channel_for_drift("Classic") is None


def test_drift_gain_declares_rewardrate_column():
    """The ``RewardRate: npt.NDArray`` annotation is what switches on
    R-learning (fit.simulateDDM's include_RewardRate) and the BETA fit
    gate; ``RR_DRIFT_MAP`` must stay invisible to that introspection."""
    from ..util import driftFnColsAndKwargs
    for key in _DRIFT_KEYS:
        cols, kwargs = driftFnColsAndKwargs(DRIFT_FN_DICT[key])
        assert "RewardRate" in cols, key
        assert "RR_DRIFT_MAP" not in cols and "RR_DRIFT_MAP" not in kwargs, key


# ---------------------------------------------------------------------------
# Saved-fit filenames
# ---------------------------------------------------------------------------

def _fp(drift, scaled_bound=False, bias="Q-Val (Offset)"):
    return evolveFP(drift, bias, "Normal(0, 1)", 4.8, 0.005,
                    False, "chisq", uses_scaled_bound=scaled_bound).name


def test_every_channel_gets_a_distinct_filename():
    """No new suffix is needed: the resolved drift name is already part of
    the pickle path, so the four channels x two scale axes never collide."""
    names = [_fp(d, sb)
             for d in ("NoiseGain-RewardRate", "Bound-RewardRate",
                       "DriftGain-RewardRate", "DriftGain(1+r)-RewardRate")
             for sb in (False, True)]
    assert len(set(names)) == len(names), names


def test_parse_fit_filename_round_trips_the_drift_channel():
    for drift, expected_alias in (
            ("DriftGain-RewardRate", "RewardRate (Drift)"),
            ("DriftGain(1+r)-RewardRate", "RewardRate (Drift 1+r)")):
        for scaled_bound in (False, True):
            fid = parse_fit_filename(_fp(drift, scaled_bound))
            assert fid.drift == drift
            assert fid.drift_alias == expected_alias
            assert fid.scaled_bound is scaled_bound
            assert fid.bias == "Q-Val (Offset)"
            assert fid.noise == "Normal(0, 1)"


def test_drift_channel_gets_its_own_model_key():
    """model_key is what aggregate/compare rows are keyed on. The drift
    channel must NOT share the NoiseGain/Bound row, but its own two
    scale-axis fits MUST share one (scaled_bound is a column axis)."""
    noise_key = parse_fit_filename(_fp("NoiseGain-RewardRate")).model_key
    bound_key = parse_fit_filename(_fp("Bound-RewardRate", True)).model_key
    drift_key = parse_fit_filename(_fp("DriftGain-RewardRate")).model_key
    drift_sb_key = parse_fit_filename(_fp("DriftGain-RewardRate", True)).model_key
    drift_1r_key = parse_fit_filename(_fp("DriftGain(1+r)-RewardRate")).model_key
    assert noise_key == bound_key            # same model, two scale axes
    assert drift_key == drift_sb_key         # ditto for the drift channel
    assert drift_key not in (noise_key, drift_1r_key)


def test_aggregate_spec_addresses_the_drift_model_key():
    """The DRIFT_RR_SPECS preset must name a model_key that a real
    DriftGain filename actually parses to, else the figure raises a
    KeyError at collection time."""
    from ..aggregate import COL_CHI2_BOUND, COL_CHI2_NOISE, DRIFT_RR_SPECS
    on_disk = {parse_fit_filename(_fp(d, sb, bias)).model_key
               for d in ("NoiseGain-RewardRate", "Bound-RewardRate",
                         "DriftGain-RewardRate", "DriftGain(1+r)-RewardRate")
               for sb in (False, True)
               for bias in ("None_", "Q-Val (Offset)")}
    assert {spec.model_key for spec in DRIFT_RR_SPECS} <= on_disk
    # Four channels x (no Q-Val, +Q-Val), each label distinct so
    # per_spec_frames (which keys on label) can't collide.
    assert len(DRIFT_RR_SPECS) == 8
    assert len({spec.label for spec in DRIFT_RR_SPECS}) == 8
    assert len({spec.color for spec in DRIFT_RR_SPECS}) == 8
    # Labels abbreviate RewardRate -> RR (eight bars would otherwise
    # collide on the x axis).
    assert all(s.label.startswith("RR (") for s in DRIFT_RR_SPECS)
    assert not any("RewardRate" in s.label for s in DRIFT_RR_SPECS)
    # Both drift mappings appear, and each is its OWN model_key -- the two
    # are different models, not two columns of one.
    drift_specs = [s for s in DRIFT_RR_SPECS if "Drift" in s.label]
    assert len(drift_specs) == 4
    assert len({s.model_key for s in drift_specs}) == 4
    assert sum("2-r" in s.label for s in drift_specs) == 2
    assert sum("1+r" in s.label for s in drift_specs) == 2
    # The drift bars read their fitted-noise column, matching the
    # RR (Noise) bar beside them; only the bound bars use Chi2-Bound.
    assert all(s.column_label == COL_CHI2_NOISE for s in drift_specs)
    assert sum(s.column_label == COL_CHI2_BOUND for s in DRIFT_RR_SPECS) == 2
    # One half-column gap, separating the no-Q-Val group from the +Q-Val
    # group -- i.e. after the 4th bar, not inside a group.
    gaps = [i for i, s in enumerate(DRIFT_RR_SPECS) if s.gap_after]
    assert gaps == [3], gaps


def test_variant_suffix_round_trips_evolve_fp():
    """``FitFileId.variant_suffix`` must rebuild exactly what evolveFP
    appended. ``model_interactive``'s ``subjects_defaults`` keys its
    entries ``f"{fit_mode}{variant_suffix}"``, so a wrong composition
    silently hides fits that are on disk."""
    for sb in (False, True):
        for w in (0.0, 0.5):
            name = evolveFP(
                "DriftGain-RewardRate", "Q-Val (Offset)", "Normal(0, 1)",
                4.8, 0.005, False, "mle",
                uses_scaled_bound=sb, mle_chi2_weight=w).name
            fid = parse_fit_filename(name)
            # The suffix is everything evolveFP put after "dt0.005".
            expected = name[name.index("dt0.005") + len("dt0.005"):-len(".pkl")]
            assert fid.variant_suffix == expected, name


def test_variant_suffix_agrees_with_the_gui_composition():
    """The filename side (``FitFileId.variant_suffix``) and the widget side
    (``visualize._variant_suffix``) compose the same key from equivalent
    state — that agreement is what lets the GUI find a saved fit at all.

    The reward-rate CHANNEL must NOT appear in either: it lives in the
    drift name. Ticking "RR as Drift" changes which drift the tree is
    indexed by, not the suffix.
    """
    from ..visualize import _variant_suffix
    for sb in (False, True):
        for rr_drift in (False, True):
            widgets_state = _widgets(scale_how="Bound" if sb else "Noise",
                                     rr_as_drift=rr_drift)
            drift = resolve_drift_alias("RewardRate", sb,
                                        use_drift_rr=rr_drift)
            fid = parse_fit_filename(
                evolveFP(drift, "None_", "Normal(0, 1)", 4.8, 0.005, False,
                         "mle", uses_scaled_bound=sb).name)
            assert fid.variant_suffix == _variant_suffix(widgets_state), (
                drift, sb, rr_drift)


def test_launch_save_name_tracks_the_drift_flags():
    """slurm/launch.py derives the log dir from the pickle name, so it has
    to parse --use-drift-rr / --drift-rr-map too."""
    from ...slurm.launch import _save_name
    base = ["--drift", "RewardRate", "--bias", "Q-Val (Offset)"]
    assert "NoiseGain-RewardRate" in _save_name(base, "chisq")
    assert "DriftGain-RewardRate" in _save_name(base + ["--use-drift-rr"], "chisq")
    assert "DriftGain(1+r)-RewardRate" in _save_name(
        base + ["--use-drift-rr", "--drift-rr-map", "1+r"], "chisq")
    scaled = _save_name(base + ["--use-drift-rr", "--scale-bound"], "chisq")
    assert "DriftGain-RewardRate" in scaled and scaled.endswith("_scaledB.pkl")


# ---------------------------------------------------------------------------
# MLE compute paths
# ---------------------------------------------------------------------------

def _build_fixture(n_sessions=2, trials_per_session=15, seed=0):
    """Same shape as test_bound_rewardrate's fixture: small, deterministic,
    with a reward-rate trajectory wide enough that a mis-scaled drift shows
    up in the loss."""
    rng = np.random.default_rng(seed)
    rows = []
    for sess in range(n_sessions):
        for trial in range(trials_per_session):
            dv = float(rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]))
            choice_left = float(rng.choice([0.0, 1.0]))
            rows.append(dict(
                Name="S1", SessId=f"sess_{sess}", SessionNum=sess,
                Date=pd.Timestamp("2026-01-01"), TrialNumber=trial + 1,
                DV=dv, DVstr=str(dv), valid=True,
                calcStimulusTime=float(rng.uniform(0.08, 0.6)),
                ChoiceLeft=choice_left,
                ChoiceCorrect=float(choice_left == (dv > 0)),
            ))
    return pd.DataFrame(rows)


_PARAMS = {
    "DRIFT_COEF": 1.2, "NOISE_SIGMA": 1.5, "BOUND": 1.0,
    "NON_DECISION_TIME": 0.04, "ALPHA": 0.3,
    "BETA": 0.4,                     # reward rate actually moves
    "BIAS_COEF": 0.0, "Q_VAL_OFFSET": 0.0, "LAPSE_RATE": 0.0,
}


def _config(drift, *, include_Q=False, batched=True, scaled_bound=False):
    return MLEModelConfig(
        drift_fn_str=drift, bias_fn_str="None_", noise_fn_str="Normal(0, 1)",
        include_Q=include_Q, include_RewardRate=True,
        dt=0.005, t_dur=0.8, dx=0.02,
        mle_use_batched_likelihood=batched,
        uses_scaled_bound=scaled_bound)


def test_config_derives_the_channel_and_mapping_from_the_drift_name():
    for drift, expected_map in (("DriftGain-RewardRate", "2-r"),
                                ("DriftGain(1+r)-RewardRate", "1+r")):
        cfg = _config(drift)
        assert cfg.uses_per_trial_drift is True
        assert cfg.rr_drift_map == expected_map
        assert cfg.sigma_rr_channel == "drift"
        assert cfg.uses_per_trial_bound is False
    for other in ("NoiseGain-RewardRate", "Bound-RewardRate", "Classic"):
        cfg = _config(other)
        assert cfg.uses_per_trial_drift is False
        assert cfg.sigma_rr_channel == "noise"


def test_latents_put_the_gain_on_mu_and_leave_sigma_flat():
    """The whole point of the channel: mu carries g(r), sigma does not,
    and there is no per-trial bound."""
    df = _build_fixture()
    data = prepare_mle_data(df)
    for drift, gain in (("DriftGain-RewardRate", lambda r: 2.0 - r),
                        ("DriftGain(1+r)-RewardRate", lambda r: 1.0 + r)):
        latents = _compute_latent_arrays(data, _PARAMS, _config(drift))
        rr = latents["reward_rate_before"]
        assert rr.min() < rr.max(), "fixture must move the reward rate"
        np.testing.assert_allclose(
            latents["mu"], _PARAMS["DRIFT_COEF"] * data.dv * gain(rr))
        np.testing.assert_allclose(
            latents["sigma"],
            np.full_like(rr, _PARAMS["NOISE_SIGMA"]))
        assert "bound_per_trial" not in latents


@pytest.mark.parametrize("drift,include_Q", [
    ("DriftGain-RewardRate", False),
    ("DriftGain(1+r)-RewardRate", False),
    ("DriftGain-RewardRate", True),
])
@pytest.mark.parametrize("scaled_bound", [False, True])
def test_rowwise_batched_and_population_agree(drift, include_Q, scaled_bound):
    """The three compute paths must produce the same loss. The population
    path is the one DE actually optimizes, so a divergence here means DE
    fits a different model than the final eval scores.

    Also covers --scale-bound composition: the per-candidate 1/B rescale
    applies on top of the per-trial gain.
    """
    df = _build_fixture()
    kw = dict(include_Q=include_Q, scaled_bound=scaled_bound)
    rowwise = evaluate_neg_loglik(
        _PARAMS, df, _config(drift, batched=False, **kw)).neg_loglik
    batched_cfg = _config(drift, batched=True, **kw)
    batched = evaluate_neg_loglik(_PARAMS, df, batched_cfg).neg_loglik
    names = list(_PARAMS)
    cand = np.array([[_PARAMS[k] for k in names]], dtype=float).T
    population = objective_from_population(
        cand, np.array(names), df, batched_cfg)[0]

    assert np.isfinite(rowwise) and np.isfinite(batched)
    assert abs(rowwise - batched) < 1.5, (rowwise, batched)
    # Single candidate => the population path builds the same latents and
    # makes the same solver call as the batched reference, so this is a
    # machine-precision comparison (measured ~0). A tight tolerance is what
    # makes it sensitive to a missing gain in the population path.
    assert abs(population - batched) < 1e-6, (population, batched)


def test_scale_bound_is_a_pure_reparametrization_here():
    """--scale-bound at BOUND=1 is the identity rescale, so it must not
    change the loss. Confirms the gain and the 1/B rescale compose in the
    right order rather than one clobbering the other."""
    df = _build_fixture()
    plain = evaluate_neg_loglik(
        _PARAMS, df, _config("DriftGain-RewardRate")).neg_loglik
    scaled = evaluate_neg_loglik(
        _PARAMS, df, _config("DriftGain-RewardRate", scaled_bound=True)
    ).neg_loglik
    np.testing.assert_allclose(plain, scaled, rtol=1e-9)


def test_validate_rejects_a_drift_channel_without_reward_rate_learning():
    cfg = MLEModelConfig(
        drift_fn_str="DriftGain-RewardRate", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)", include_Q=False,
        include_RewardRate=False,        # inconsistent
        dt=0.005, t_dur=0.8)
    with pytest.raises(ValueError, match="include_RewardRate is False"):
        evaluate_neg_loglik(_PARAMS, _build_fixture(), cfg)


def test_validate_rejects_both_bound_and_drift_channels():
    cfg = MLEModelConfig(
        drift_fn_str="DriftGain-RewardRate", bias_fn_str="None_",
        noise_fn_str="Normal(0, 1)", include_Q=False, include_RewardRate=True,
        dt=0.005, t_dur=0.8,
        uses_per_trial_bound=True)       # mutually exclusive channels
    with pytest.raises(ValueError, match="mutually exclusive"):
        evaluate_neg_loglik(_PARAMS, _build_fixture(), cfg)


# ---------------------------------------------------------------------------
# fit.simulateDDM wiring
# ---------------------------------------------------------------------------

def _simulate_ddm_dry_run(fit_mode, drift, scale_bound, **kwargs):
    """``simulateDDM`` dry run. Returns ``{subject: loss}`` for chisq and
    ``{subject: payload_dict}`` for mle. Neither writes to data/RLModel
    (the saves are gated on ``not dry_run``)."""
    from ..fit import simulateDDM
    from ..initvals import InitVals
    # Enough trials that the Ratcliff quantiles of the observed RTs are
    # strictly increasing -- chi2Loss histograms on them, and a tiny
    # fixture collapses them into non-monotonic bin edges.
    df = _build_fixture(n_sessions=2, trials_per_session=40)
    df["RepeatIdx"] = 1
    return simulateDDM(
        df, bounds_and_defaults=InitVals().toDict(), dt=0.005, t_dur=0.8,
        biasFn=BIAS_FN_DICT["None_"], driftFn=DRIFT_FN_DICT[drift],
        noiseFn=NOISE_FN_DICT["Normal(0, 1)"],
        is_loss_no_dir=False, num_cpus=1, evolvs_res={}, fit_mode=fit_mode,
        dry_run=True, drift_fn_str=drift, bias_fn_str="None_",
        scale_bound=scale_bound, **kwargs)


@pytest.mark.parametrize("scale_bound", [False, True])
def test_simulate_ddm_chisq_runs_under_either_scale_axis(scale_bound):
    """``Bound-RewardRate`` raises the --scale-bound soft-limit when BOUND
    is frozen, because it collapses to NoiseGain there. The drift channel
    must NOT — it is a distinct model under either scale axis, so both
    directions have to reach a finite loss.
    """
    res = _simulate_ddm_dry_run("chisq", "DriftGain-RewardRate", scale_bound)
    assert np.isfinite(res["S1"])


def test_simulate_ddm_mle_gates_beta_and_the_drift_channel():
    """The ``RewardRate`` df-column dependency must switch on
    ``include_RewardRate`` (hence BETA in the fit vector) and the derived
    config must report the drift channel back."""
    res = _simulate_ddm_dry_run("mle", "DriftGain-RewardRate", False,
                                mle_array_backend="numpy")
    payload = res["S1"]
    assert payload["include_RewardRate"] is True
    assert "BETA" in [str(n).upper() for n in payload["params_names"]]
    config = payload["model_config"]
    assert config.uses_per_trial_drift is True
    assert config.uses_per_trial_bound is False
    assert config.rr_drift_map == "2-r"


# ---------------------------------------------------------------------------
# model_interactive GUI layer
# ---------------------------------------------------------------------------

class _FakeWidget:
    """Stands in for an ipywidgets control: ``_resolved_drift_fn_value``
    only ever reads ``.value``."""

    def __init__(self, value):
        self.value = value


def _widgets(drift="RewardRate", scale_how="Noise", rr_as_drift=False,
             rr_map="2-r"):
    """Minimal widget dict for ``_variant_suffix`` and the drift resolution."""
    return {"Drift Fn": _FakeWidget(drift),
            "Bias Fn": _FakeWidget("None_"),
            "Noise Fn": _FakeWidget("Normal(0, 1)"),
            "Scale-How": _FakeWidget(scale_how),
            "Joint Wt": _FakeWidget(""),
            "RR as Drift": _FakeWidget(rr_as_drift),
            "RR-Drift Map": _FakeWidget(rr_map)}


@pytest.mark.parametrize("scale_how", ["Noise", "Bound"])
def test_gui_checkbox_overrides_the_scale_how_dropdown(scale_how):
    """Ticking "RR as Drift" must win over Scale-How, matching
    ``--use-drift-rr`` vs ``--scale-bound`` on the CLI."""
    from ..visualize import _resolved_drift_fn_value
    assert _resolved_drift_fn_value(
        _widgets(scale_how=scale_how, rr_as_drift=True)) == "DriftGain-RewardRate"
    assert _resolved_drift_fn_value(
        _widgets(scale_how=scale_how, rr_as_drift=True, rr_map="1+r")
    ) == "DriftGain(1+r)-RewardRate"
    # Unticked => the legacy Scale-How routing is untouched.
    expected = ("Bound-RewardRate" if scale_how == "Bound"
                else "NoiseGain-RewardRate")
    assert _resolved_drift_fn_value(_widgets(scale_how=scale_how)) == expected


def test_gui_drift_resolution_tolerates_missing_widgets():
    """The two new controls are read with membership checks, so a caller
    that predates them (or a partially-built widget dict) still resolves."""
    from ..visualize import _resolved_drift_fn_value
    legacy = {"Drift Fn": _FakeWidget("RewardRate"),
              "Scale-How": _FakeWidget("Bound")}
    assert _resolved_drift_fn_value(legacy) == "Bound-RewardRate"


def test_gui_tooltips_are_defined_for_the_new_controls():
    """ipywidgets >= 8 spelling. Both controls must actually carry help
    text — the checkbox label alone doesn't convey that it OVERRIDES the
    noise/threshold channel rather than adding to it."""
    import ipywidgets as widgets
    from ..visualize import _WIDGET_TOOLTIPS
    for label in ("RR as Drift", "RR-Drift Map"):
        assert label in _WIDGET_TOOLTIPS and _WIDGET_TOOLTIPS[label].strip()
    cb = widgets.Checkbox(value=False, description="RR as Drift",
                          tooltip=_WIDGET_TOOLTIPS["RR as Drift"])
    assert cb.tooltip == _WIDGET_TOOLTIPS["RR as Drift"]
    dd = widgets.Dropdown(options=["2-r", "1+r"], value="2-r",
                          description="RR-Drift Map",
                          tooltip=_WIDGET_TOOLTIPS["RR-Drift Map"])
    assert dd.tooltip == _WIDGET_TOOLTIPS["RR-Drift Map"]


def test_create_widget_places_every_new_widget(monkeypatch):
    """``createWidget`` asserts that every constructed widget is popped
    into exactly one layout column — adding a control without placing it
    raises ``AssertionError: Left checkboxes: {...}``. This pins that the
    "RR as Drift" checkbox and "RR-Drift Map" dropdown are placed.

    ``runAndPlot`` is stubbed out: we only care about widget construction,
    and the real plot path wants the full behavior-df schema.
    """
    import matplotlib
    matplotlib.use("Agg")
    from .. import visualize
    from ..initvals import InitVals
    from ..util import initDF

    calls = []
    monkeypatch.setattr(visualize, "runAndPlot",
                        lambda *a, **k: calls.append(k) or (None, None))

    rng = np.random.default_rng(0)
    rows = []
    for sess in range(2):
        for trial in range(30):
            dv = float(rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]))
            choice_left = float(rng.choice([0.0, 1.0]))
            rows.append(dict(
                Name="S1", SessId=f"sess_{sess}", SessionNum=sess,
                Date=pd.Timestamp("2026-01-01"), TrialNumber=trial + 1,
                DV=dv, DVstr=str(dv), valid=True, RepeatIdx=1,
                calcStimulusTime=float(rng.uniform(0.08, 0.6)),
                ChoiceLeft=choice_left,
                ChoiceCorrect=float(choice_left == (dv > 0))))
    df = initDF(pd.DataFrame(rows), include_Q=True, include_RewardRate=True)

    visualize.createWidget(
        init_vals=InitVals(), gui_cache={}, df=df, t_dur=0.8, dt=0.005,
        is_small_fig_mode=False, subjects_defaults=None, save_figs=False)
    # updateGUI ran (so the gating block executed) and reached the plot call.
    assert calls, "createWidget never reached runAndPlot"


def test_gui_variant_suffix_unchanged_by_the_drift_channel():
    """The channel lives in the drift NAME, not in a filename suffix, so
    ``_variant_suffix`` (scaledB + joint weights) must not gain a
    component — otherwise saved-fit lookups would miss."""
    from ..visualize import _variant_suffix
    base = _widgets()
    ticked = _widgets(rr_as_drift=True, rr_map="1+r")
    assert _variant_suffix(base) == _variant_suffix(ticked) == ""


# ---------------------------------------------------------------------------
# CLI pre-flight
# ---------------------------------------------------------------------------

def _run_runner(*extra):
    return subprocess.run(
        [sys.executable, "-m", "code.rlmodel.model_runner",
         "--bias", "None_", "--noise", "Normal(0, 1)",
         "--fit-mode", "chisq", "--dry-run", *extra],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT))


def test_cli_rejects_use_drift_rr_without_reward_rate_learning():
    proc = _run_runner("--drift", "Classic", "--use-drift-rr")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "--use-drift-rr" in proc.stderr and "R-learning" in proc.stderr


def test_cli_rejects_drift_rr_map_without_use_drift_rr():
    proc = _run_runner("--drift", "RewardRate", "--drift-rr-map", "1+r")
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "--drift-rr-map" in proc.stderr


def test_cli_resolves_the_drift_channel():
    """``_resolve_drift_alias_args`` reads the two new flags off the
    namespace and mutates ``args.drift`` in place."""
    from ...model_runner import _resolve_drift_alias_args
    args = argparse.Namespace(drift="RewardRate", scale_bound=True,
                              use_drift_rr=True, drift_rr_map="1+r")
    _resolve_drift_alias_args(args)
    assert args.drift == "DriftGain(1+r)-RewardRate"
    # Namespaces without the new attrs still resolve (getattr defaults).
    legacy = argparse.Namespace(drift="RewardRate", scale_bound=False)
    _resolve_drift_alias_args(legacy)
    assert legacy.drift == "NoiseGain-RewardRate"
