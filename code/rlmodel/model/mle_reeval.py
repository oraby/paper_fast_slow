"""Re-evaluate any saved fit's parameters under the MLE objective.

Extracted from ``mle_debug.ipynb`` so both that notebook and
``model_compare.ipynb`` share one implementation (the user's "export
notebook code to files / don't duplicate" directive).

Two audiences:

- ``mle_debug.ipynb`` keeps its original entry point
  ``fitted_params_to_mle_df(subject, variant_filename, df_behavior,
  fit_results_by_mode, ...)`` (unchanged 4-tuple return) plus the two
  loss plots, now refactored to take an ``ax``.
- ``model_compare.py`` uses the lower-level primitives
  ``parse_fit_filename`` / ``build_mle_config`` /
  ``evaluate_params_under_mle`` directly, because it discovers fits from
  disk itself (it does not carry the notebook's ``fit_results_by_mode``
  nested dict).

The re-evaluation always runs the *pure MLE* objective on a fit's
parameters, whatever criterion produced them (pure MLE, joint MLE+Chi²,
Chi²-Noise, Chi²-Bound). That gives the comparable "MLE-Score" and the
per-trial ``mle_df`` the loss-distribution plots consume.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from .mle import MLEModelConfig, evaluate_neg_loglik
from .initvals import MLE_TERMINAL_C
from .drift import display_alias_for_drift


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------
# Suffixes that fit.evolveFP appends after ``dt{dt}``, in order:
#   {asym}{scaledB}{weights}  — asymQ/asymRR/asymQRR, then _scaledB, then
#   _mleW{m}_chi2W{c}. We peel them off in reverse.
_WEIGHT_SUFFIX_RE = re.compile(r"_mleW(?P<m>[-+0-9.eE]+)_chi2W(?P<c>[-+0-9.eE]+)$")
_ASYM_SUFFIX_RE = re.compile(r"_asym(Q|RR|QRR)$")


@dataclass(frozen=True)
class FitFileId:
    """Everything a saved-fit filename encodes, decomposed.

    ``drift`` is the internal ``DRIFT_FN_DICT`` key from the filename
    (e.g. ``NoiseGain-RewardRate`` / ``Bound-RewardRate``); ``drift_alias``
    is its user-facing collapse (``RewardRate``) — the two Bound/Noise
    implementations of one abstract model share an alias, which is what
    makes Chi²-Noise and Chi²-Bound land in the same model_compare figure.
    The DriftGain- family deliberately does NOT share that alias (it maps to
    ``RewardRate (Drift)`` / ``RewardRate (Drift 1+r)``): the reward rate
    acting on the drift is a different model, not a different scale axis, so
    it gets its own row. See ``drift.display_alias_for_drift``.
    """
    fit_mode: str          # "mle" | "chisq"
    drift: str             # internal DRIFT_FN_DICT key
    drift_alias: str       # user-facing alias; == drift when not aliased
    bias: str              # BIAS_FN_DICT key, e.g. "Q-Val (Offset)" or "None_"
    noise: str             # NOISE_FN_DICT key, e.g. "Normal(0, 1)"
    t_dur: float
    dt: float
    asym_variant: str      # "" | "Q" | "RR" | "QRR"
    scaled_bound: bool     # True when the _scaledB suffix is present
    mle_weight: float      # joint outer MLE weight (1.0 when absent)
    chi2_weight: float     # joint outer Chi² weight (0.0 when absent → pure)

    @property
    def model_key(self) -> str:
        """Identity of the abstract model (one model_compare dropdown
        entry): drift ALIAS + bias + noise + timing + asym variant.
        ``scaled_bound`` and the joint weights are *column* axes, not
        identity, so they're intentionally excluded."""
        asym = f"asym{self.asym_variant}" if self.asym_variant else "sym"
        return (f"{self.drift_alias}|{self.bias}|{self.noise}"
                f"|{self.t_dur:g}|{self.dt:g}|{asym}")

    @property
    def model_label(self) -> str:
        base = f"{self.drift_alias} · {self.bias} · {self.noise} · {self.t_dur:g}s"
        return f"{base} [asym{self.asym_variant}]" if self.asym_variant else base

    @property
    def variant_suffix(self) -> str:
        """The filename suffix these orthogonal opt-ins compose to, in
        ``fit.evolveFP`` order: asym, then scaledB, then joint weights.

        Round-trips what ``parse_fit_filename`` peeled off. Consumers that
        key saved fits by variant (``model_interactive``'s
        ``subjects_defaults``, whose entries are looked up as
        ``f"{fit_mode}{variant_suffix}"``) need the composed string back,
        and must agree with ``visualize._variant_suffix`` — which builds
        the same string from GUI widget state — or the GUI silently fails
        to find a fit that is on disk.

        NOTE the reward-rate CHANNEL is deliberately absent: it lives in
        the drift name (``NoiseGain-`` / ``Bound-`` / ``DriftGain-``), not
        in a suffix.
        """
        asym = f"_asym{self.asym_variant}" if self.asym_variant else ""
        scaled = "_scaledB" if self.scaled_bound else ""
        weights = ("" if self.chi2_weight <= 0.0 else
                   f"_mleW{self.mle_weight:g}_chi2W{self.chi2_weight:g}")
        return f"{asym}{scaled}{weights}"


def parse_fit_filename(filename: str) -> FitFileId:
    """Decompose a saved-fit pickle name into a :class:`FitFileId`.

    Mirrors ``fit.evolveFP``'s composition::

        {fit_mode}_{drift}_bias{bias}_{noise}{loss_no_dir}_{t_dur}s_dt{dt}
        {asym}{scaledB}{weights}.pkl
    """
    stem = filename[:-4] if filename.endswith(".pkl") else filename
    fit_mode, rest = stem.split("_", 1)
    # drift / bias / noise: peel the leading char then split on "_" — mirrors
    # model_interactive.ipynb's parser so names containing spaces / parens
    # (e.g. "Q-Val (Offset)", "Normal(0, 1)") survive intact.
    first_char, rest = rest[0], rest[1:]
    drift_fn, rest = rest.split("_", 1)
    drift_fn = first_char + drift_fn
    first_char, rest = rest[0], rest[1:]
    bias_fn, rest = rest.split("_", 1)
    bias_fn = first_char + bias_fn
    if bias_fn == "biasNone":            # the "None_" bias → doubled "_"
        bias_fn, rest = "biasNone_", rest[1:]
    bias_fn = bias_fn[len("bias"):] if bias_fn.startswith("bias") else bias_fn
    first_char, rest = rest[0], rest[1:]
    noise_fn, rest = rest.split("_", 1)
    noise_fn = first_char + noise_fn
    if "loss_no_dir_" in rest:
        raise NotImplementedError(
            f"loss_no_dir fits are not supported by model_compare: {filename}")
    parsed_t_dur, rest = rest.split("s_", 1)
    dt_segment = rest.split("dt", 1)[1]
    # Peel suffixes in reverse evolveFP order: weights, scaledB, asym.
    mle_weight, chi2_weight = 1.0, 0.0
    w = _WEIGHT_SUFFIX_RE.search(dt_segment)
    if w:
        mle_weight, chi2_weight = float(w.group("m")), float(w.group("c"))
        dt_segment = dt_segment[:w.start()]
    scaled_bound = dt_segment.endswith("_scaledB")
    if scaled_bound:
        dt_segment = dt_segment[:-len("_scaledB")]
    asym_variant = ""
    a = _ASYM_SUFFIX_RE.search(dt_segment)
    if a:
        asym_variant = a.group(1)
        dt_segment = dt_segment[:a.start()]
    return FitFileId(
        fit_mode=fit_mode,
        drift=drift_fn,
        drift_alias=display_alias_for_drift(drift_fn),
        bias=bias_fn,
        noise=noise_fn,
        t_dur=float(parsed_t_dur),
        dt=float(dt_segment),
        asym_variant=asym_variant,
        scaled_bound=scaled_bound,
        mle_weight=mle_weight,
        chi2_weight=chi2_weight,
    )


# --------------------------------------------------------------------------
# MLE re-evaluation
# --------------------------------------------------------------------------
def fitted_params_from_result(result: dict) -> dict[str, float]:
    """Fitted params as an UPPERCASE name → value dict.

    Falls back to ``params_init`` when a payload has no ``OptimRes`` (dry
    runs), mirroring ``mle_notebooks.data.fitted_params_from_result``.
    """
    names = list(result["params_names"])
    optim_res = result.get("OptimRes")
    values = (np.asarray(result["params_init"], dtype=float)
              if optim_res is None else np.asarray(optim_res.x, dtype=float))
    return {str(name).upper(): float(value)
            for name, value in zip(names, values)}


def build_mle_config(fid: FitFileId, *, include_Q, include_RewardRate,
                     mle_terminal_c=MLE_TERMINAL_C.Default) -> MLEModelConfig:
    """Build the ``MLEModelConfig`` for re-evaluating a fit under MLE.

    The scale-axis / asymmetric-LR / per-trial-bound flags are derived from
    the parsed filename so a Chi²-Bound (``Bound-RewardRate`` + ``_scaledB``)
    or asymmetric fit is re-evaluated with the same model structure it was
    fit under. ``uses_per_trial_bound`` follows ``fit.simulateDDM``'s rule
    (any ``Bound-RewardRate*`` drift).
    """
    return MLEModelConfig(
        drift_fn_str=fid.drift,
        bias_fn_str=fid.bias,
        noise_fn_str=fid.noise,
        include_Q=bool(include_Q),
        include_RewardRate=bool(include_RewardRate),
        dt=float(fid.dt),
        t_dur=float(fid.t_dur),
        mle_terminal_c=float(mle_terminal_c),
        uses_scaled_bound=bool(fid.scaled_bound),
        uses_per_trial_bound="Bound-RewardRate" in fid.drift,
        uses_asymmetric_alpha=bool(include_Q) and fid.asym_variant in ("Q", "QRR"),
        uses_asymmetric_beta=bool(include_RewardRate) and fid.asym_variant in ("RR", "QRR"),
    )


def evaluate_params_under_mle(params, subject_df, model_config, *,
                              lapse_override=None):
    """Run the pure-MLE objective on ``params`` for one subject's trials.

    Returns the ``MLEEvalResult`` (``.neg_loglik``, ``.mle_df``, …). When
    ``lapse_override`` is a float it replaces the params' ``LAPSE_RATE``
    (a Chi²-fit params dict has none → λ defaults to 0 otherwise).
    """
    params = {str(k).upper(): float(v) for k, v in params.items()}
    if lapse_override is not None:
        params["LAPSE_RATE"] = float(lapse_override)
    return evaluate_neg_loglik(params, subject_df, model_config, return_df=True)


def fitted_params_to_mle_df(subject_name, variant_filename, df_behavior,
                            fit_results_by_mode, *,
                            mle_terminal_c=MLE_TERMINAL_C.Default,
                            lapse_rate_override=None):
    """``mle_debug.ipynb`` entry point (unchanged 4-tuple contract).

    Re-evaluates the subject's fit from ``variant_filename`` under MLE and
    returns ``(mle_df, eval_result, model_config, params)``. Works for both
    ``chisq_*`` and ``mle_*`` variants — the filename's prefix selects the
    ``fit_results_by_mode`` bucket.
    """
    fid = parse_fit_filename(variant_filename)
    fit_payloads = fit_results_by_mode.get(fid.fit_mode)
    if fit_payloads is None or variant_filename not in fit_payloads:
        raise KeyError(
            f"Variant {variant_filename!r} not in "
            f"fit_results_by_mode[{fid.fit_mode!r}]")
    if subject_name not in fit_payloads[variant_filename]:
        raise KeyError(
            f"Subject {subject_name!r} not in {variant_filename}. Available: "
            f"{sorted(fit_payloads[variant_filename].keys())}")

    fit_payload = fit_payloads[variant_filename][subject_name]
    params = fitted_params_from_result(fit_payload)
    model_config = build_mle_config(
        fid,
        include_Q=bool(fit_payload["include_Q"]),
        include_RewardRate=bool(fit_payload["include_RewardRate"]),
        mle_terminal_c=mle_terminal_c)
    subject_df = df_behavior[df_behavior.Name == subject_name].copy()
    eval_result = evaluate_params_under_mle(
        params, subject_df, model_config, lapse_override=lapse_rate_override)
    return eval_result.mle_df, eval_result, model_config, params


# --------------------------------------------------------------------------
# Behavior dataframe (shared with mle_debug.ipynb's load cell)
# --------------------------------------------------------------------------
def prepare_behavior_df(df_fp=None):
    """Load + prepare the behavior dataframe exactly as both notebooks do.

    ``df_fp`` defaults to the notebook-relative ``../../data/behavior/…``
    path (both notebooks run with cwd = ``code/rlmodel``). Imports are lazy
    so importing this module for its parse/eval primitives doesn't pull in
    the full ``model_runner`` chain.
    """
    from ..model_runner import loadDF, _extendTrials, _reduceDFSize, DF_FP
    from ...behavior.rewardrate import calcAvgRewardRate
    if df_fp is None:
        df_fp = f"../../{DF_FP}"
    df = loadDF(df_fp=df_fp)
    print("Calculate average reward rate (and create SessId column)...")
    df = calcAvgRewardRate(df)
    print("Reduce dataframe size to speed up df operations...")
    df = _reduceDFSize(df)
    print("Extend trials so all sessions have the same number of trials...")
    df = _extendTrials(df)
    df["SessId"] = df.apply(
        lambda x: f"{x['Name']}_{x['Date']}_{x['SessionNum']}", axis=1)
    df["DVabs"] = df.DV.abs()
    df["Q_val"] = np.nan
    df["Q_L"] = np.nan
    df["Q_R"] = np.nan
    df["RewardRate"] = np.nan
    return df


# --------------------------------------------------------------------------
# Loss-distribution plots (refactored from mle_debug's inline ``debug()``)
# --------------------------------------------------------------------------
def plot_loss_distribution(ax, mle_df, *, clip_at=-20, ylim=None, bins=None):
    """Histogram of per-trial loss (``-mle_loglik``), clipped so the floor
    spike stacks into the last bin. Valid trials only."""
    mle_df = mle_df[mle_df.valid == True]  # noqa: E712 — pandas mask
    loglik_cp = mle_df.mle_loglik.copy()
    loglik_cp[loglik_cp < clip_at] = clip_at - 0.5
    if bins is None:
        bins = np.arange(-5, 21, 0.5)
    ax.hist(-loglik_cp, bins=bins)
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_xlabel("per-trial −loglik")
    ax.set_ylabel("count")
    return ax


def plot_rt_hist_colored_by_loss(ax, mle_df, *, cut_off=-40, t_dur=3.0, dt=0.05):
    """Stacked RT histogram coloured by loss: red = outlier trials
    (``mle_loglik ≤ cut_off``), blue = the rest. No-choice trials (null RT)
    are dropped into the ``t=-1`` bin edge so they stay visible. Valid
    trials only.

    v2 idea (left as a follow-up): replace the red/blue split with a graded
    colormap keyed on the continuous ``mle_loglik`` value.
    """
    mle_df = mle_df[mle_df.valid == True]  # noqa: E712 — pandas mask
    outlier_mask = mle_df.mle_loglik < cut_off
    okay_df = mle_df.loc[~outlier_mask]
    outliers_df = mle_df.loc[outlier_mask].sort_values("calcStimulusTime").copy()
    outliers_df.loc[outliers_df.calcStimulusTime.isnull(), "calcStimulusTime"] = -1
    ax.hist([outliers_df.calcStimulusTime, okay_df.calcStimulusTime],
            bins=np.arange(0, t_dur + dt, dt), histtype="barstacked",
            color=["red", "blue"],
            label=[f"outliers (loglik ≤ {cut_off})", "other trials"])
    ax.legend(fontsize="x-small", loc="upper right")
    ax.set_xlabel("Stimulus time (s)")
    ax.set_ylabel("count")
    return ax
