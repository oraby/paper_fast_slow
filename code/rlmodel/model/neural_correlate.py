"""neural_correlate engine — correlate 2-photon neuron activity with MLE latents.

Cross-analysis backend for ``model_neural_correlate.ipynb``. For a chosen
saved **MLE fit**, we re-evaluate its trial-by-trial latents (Q-left, Q-right,
relative Q "Q-val", RewardRate) and ask, **per neuron**, whether that neuron's
per-trial sampling-window activity correlates with each latent.

Design mirrors ``model/compare.py``: the notebook stays a thin config layer and
calls the functions here. Nothing here is duplicated from a notebook —

- the per-(neuron, trial) windowing/reduction mirrors
  ``twop/activity_correlation.py::getTrialsMeanActivity`` but reduces with
  ``np.nanmax`` (max activity within the sampling window) and additionally keeps
  the full time-normalized trace;
- the MLE re-evaluation reuses ``model/mle_reeval.py`` primitives verbatim;
- the pie chart mirrors ``2pAnalysis.ipynb::loopPlotPieChart`` (session
  mean ± SEM).

Confirmed conventions (see the notebook / plan):
- Activity source is the **z-scored + time(width)-normalized** sampling-window
  df (``normalized_0.1s_before_sampling_0.1s_after_movement.pkl``); the stored
  trace and the ``np.nanmax`` value both come from it.
- Correlation is **Pearson**, within session.
- Latents are the **before-trial** values (state entering the trial).
- Brain region is always ``MFC`` / ``LFC`` (never the enum value or ``*_Bi``).
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace
import pathlib
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from scipy.stats import linregress, spearmanr, ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from .mle_reeval import (parse_fit_filename, fitted_params_from_result,
                         build_mle_config, evaluate_params_under_mle,
                         prepare_behavior_df)  # re-exported for the notebook
from ...common.definitions import BrainRegion
from .fitio import loadFit


# --------------------------------------------------------------------------
# Parameter registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """One model latent the notebook can correlate against neural activity.

    ``key``    — short column/folder name (e.g. ``"Q_L"``; matches the plan).
    ``mle_col``— source column on the re-evaluated ``mle_df`` (for ``"drift"``
                 params this is the behavioral column on the 2-photon row).
    ``family`` — ``"Q"`` (needs ``include_Q``) / ``"RR"`` (needs
                 ``include_RewardRate``) — MLE latents gated by include flags;
                 or ``"drift"`` — behavioral (DV / DVabs), always available.
    ``label``  — friendly axis label (e.g. ``"Q-Left"``).
    ``color``  — plot color, kept consistent between scatter and pie.
    ``signed`` — the value can be negative (DV, Q-val). In the trace plots the
                 gradient then encodes only the **magnitude** (0 → max), so the
                 symmetric extremes (e.g. -1 and +1) share a shade, and the
                 negative-side ranges are drawn dashed.
    """
    key: str
    mle_col: str
    family: str
    label: str
    color: str
    signed: bool = False


PARAM_SPECS = (
    ParamSpec("Q_L", "mle_Q_left_before", "Q", "Q-Left", "g"),
    ParamSpec("Q_R", "mle_Q_right_before", "Q", "Q-Right", "orange"),
    ParamSpec("Q_val", "mle_Q_rel_before", "Q", "Q-val", "purple", signed=True),
    ParamSpec("RewardRate", "mle_reward_rate_before", "RR",
              "R-Learning RewardRate", "teal"),
    # Drift: the neuron's evidence encoding — activity vs signed DV and vs |DV|.
    # These live on the 2-photon trial row, not the MLE latents.
    ParamSpec("DV", "DV", "drift", "DV", "steelblue", signed=True),
    ParamSpec("DVabs", "DVabs", "drift", "|DV|", "sienna"),
)
PARAM_BY_KEY = {spec.key: spec for spec in PARAM_SPECS}

# Behavioral (non-MLE) columns carried onto the neuron-trial table from the 2P
# trial row, for the drift / fast-vs-slow analysis.
DRIFT_KEYS = [s.key for s in PARAM_SPECS if s.family == "drift"]  # ["DV","DVabs"]
_FAST_Q, _SLOW_Q = 1, 3  # quantile_idx: fastest / slowest RT tercile

# The trial-identity key shared by the behavior/MLE and the 2-photon dataframes.
IDENTITY_COLS = ["Name", "Date", "SessionNum", "TrialNumber"]

# The y-axis label of every per-neuron activity plot. The activity source is
# already z-scored per neuron over its session's concatenated trials (see
# ``pipeline/tracesnormalize.py::NormalizeZScore``, applied when the sampling
# df was built), so the unit is SDs of that neuron's own session fluorescence —
# not ΔF/F. ``_ZSCORED_AGAIN_LABEL`` is for the opt-in ``zscore=True`` scatter,
# which re-standardizes over only the plotted trials on top of that.
_ACTIVITY_LABEL = "Neuron activity (z-score)"
_ZSCORED_AGAIN_LABEL = "Neuron activity (z-score, re-standardized)"

# The two 2-photon brain regions carried in the traces df (M2_Bi -> MFC,
# ALM_Bi -> LFC); everything else is filtered out, matching twop/fastslowstats.
_ANALYSIS_REGIONS = (BrainRegion.M2_Bi, BrainRegion.ALM_Bi)


# --------------------------------------------------------------------------
# Filesystem-safe names (mirrors compare.py::_safe_filename)
# --------------------------------------------------------------------------
_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*]')


def _safe_filename(name):
    return _ILLEGAL_FS.sub("_", str(name).replace("·", "-")).strip()


def _friendly_model_name(fid):
    """Minimal friendly model name for titles / save paths.

    ``FitFileId.model_label`` is the abstract model (drift·bias·noise·timing,
    which deliberately omits the fitting-criterion weights). For this notebook
    we also surface the **fitting criterion** so a weighted-Chi² fit is not
    silently shown as if it were pure MLE: a joint fit gets a
    ``weighted Chi²`` tag, and a Chi²-fit its Noise/Bound variant."""
    base = fid.model_label
    if fid.fit_mode == "mle" and fid.chi2_weight != 0.0:
        base += (f" · weighted Chi² (MLE={fid.mle_weight:g}, "
                 f"Chi²={fid.chi2_weight:g})")
    elif fid.fit_mode == "chisq":
        base += " · Chi²-Bound" if fid.scaled_bound else " · Chi²-Noise"
    return base


# --------------------------------------------------------------------------
# 1. MLE re-evaluation → per-trial latents
# --------------------------------------------------------------------------
def available_params(include_Q, include_RewardRate):
    """The :class:`ParamSpec`s a fit supports given its include flags."""
    return [s for s in PARAM_SPECS
            if (s.family == "Q" and include_Q)
            or (s.family == "RR" and include_RewardRate)]


def load_mle_per_trial(fit_pkl_path, df_behavior=None, *, subjects=None,
                       reuse_fit_df=True, mle_terminal_c=0.0,
                       lapse_override=None, verbose=True):
    """Load a saved MLE fit's per-trial latent dataframe.

    ``fit_pkl_path`` is a single ``{subject: payload}`` pickle (declared at the
    top of the notebook). Only the latent families the fit actually learns
    (``include_Q`` / ``include_RewardRate``) are surfaced — the availability is
    **reported** so a reward-rate-only or Q-only model is handled gracefully.

    ``subjects`` (e.g. from :func:`subjects_in_2p`) restricts loading to those
    subject names; the fit file usually holds many subjects without imaging, and
    the caller should pass only the ones present in the 2-photon data.

    ``reuse_fit_df`` (default **True**) reads the per-trial ``mle_df`` already
    stored in each payload — fast, and it does not need ``df_behavior``. Set it
    ``False`` to **recompute** the latents via ``mle_reeval`` (expensive; needs
    ``df_behavior``, and honours ``mle_terminal_c`` / ``lapse_override``).

    Returns ``(mle_pt, param_keys)`` where ``mle_pt`` has the identity columns,
    ``mle_loglik``, ``mle_valid`` (bool), ``model_label``, ``model_name`` (safe),
    and one renamed column per available latent (``Q_L``/``Q_R``/``Q_val``/
    ``RewardRate``); ``param_keys`` is the list of those available keys.
    """
    fit_pkl_path = pathlib.Path(fit_pkl_path)
    fid = parse_fit_filename(fit_pkl_path.name)
    subject_payloads = loadFit(fit_pkl_path)
    if not isinstance(subject_payloads, dict) or not subject_payloads:
        raise ValueError(f"{fit_pkl_path.name} is not a non-empty "
                         "{subject: payload} dict")

    # Restrict to the requested subjects (those with 2-photon data).
    if subjects is not None:
        want = {str(s) for s in subjects}
        selected = {s: p for s, p in subject_payloads.items() if str(s) in want}
        missing = sorted(want - {str(s) for s in subject_payloads})
        if verbose and missing:
            print(f"  {len(missing)} requested subject(s) absent from the fit "
                  f"file: {missing}")
        if not selected:
            raise ValueError(
                f"None of the requested subjects {sorted(want)} are in "
                f"{fit_pkl_path.name} (has: {sorted(map(str, subject_payloads))}).")
        subject_payloads = selected

    # Include flags: take the first (selected) payload, then assert the rest
    # agree so we don't silently mix Q-only and RR-only subjects in one file.
    first_payload = next(iter(subject_payloads.values()))
    include_Q = bool(first_payload["include_Q"])
    include_RewardRate = bool(first_payload["include_RewardRate"])
    for subj, payload in subject_payloads.items():
        if (bool(payload["include_Q"]) != include_Q
                or bool(payload["include_RewardRate"]) != include_RewardRate):
            raise ValueError(
                f"Subject {subj!r} has different include flags than the rest of "
                f"{fit_pkl_path.name}; cannot combine.")

    friendly = _friendly_model_name(fid)
    specs = available_params(include_Q, include_RewardRate)
    if verbose:
        have = ", ".join(s.key for s in specs) or "(none!)"
        _report = [f"Model: {friendly}",
                   f"  subjects loaded: {sorted(map(str, subject_payloads))}",
                   f"  mode: {'reuse stored mle_df' if reuse_fit_df else 'RECOMPUTE'}",
                   f"  include_Q={include_Q}  include_RewardRate={include_RewardRate}"]
        if not include_Q:
            _report.append("  NOTE: Q-values (Q_L, Q_R, Q_val) NOT available — "
                           "this fit does not learn Q.")
        if not include_RewardRate:
            _report.append("  NOTE: RewardRate NOT available — this fit does "
                           "not learn a reward rate.")
        _report.append(f"  Correlatable parameters: {have}")
        print("\n".join(_report))
    if not specs:
        raise ValueError(f"{fit_pkl_path.name} learns neither Q nor RewardRate; "
                         "nothing to correlate.")

    if not reuse_fit_df and df_behavior is None:
        raise ValueError("reuse_fit_df=False recomputes the latents and needs "
                         "df_behavior (from prepare_behavior_df()).")

    config = (None if reuse_fit_df else
              build_mle_config(fid, include_Q=include_Q,
                               include_RewardRate=include_RewardRate,
                               mle_terminal_c=mle_terminal_c))
    frames = []
    for subject, payload in subject_payloads.items():
        if reuse_fit_df:
            stored = payload.get("mle_df")
            if not isinstance(stored, pd.DataFrame):
                raise KeyError(
                    f"Subject {subject!r} has no stored 'mle_df' in "
                    f"{fit_pkl_path.name}; re-run with reuse_fit_df=False.")
            frames.append(stored)
        else:
            subject_df = df_behavior[df_behavior.Name == subject].copy()
            if subject_df.empty:
                if verbose:
                    print(f"  skip {subject!r}: no behavior rows")
                continue
            params = fitted_params_from_result(payload)
            res = evaluate_params_under_mle(params, subject_df, config,
                                            lapse_override=lapse_override)
            frames.append(res.mle_df)
    if not frames:
        raise ValueError("No subject produced an mle_df.")
    mle_df = pd.concat(frames, ignore_index=True)

    rename = {s.mle_col: s.key for s in specs}
    keep = IDENTITY_COLS + [s.mle_col for s in specs] + ["mle_loglik"]
    if "mle_valid_for_loss" in mle_df.columns:
        keep.append("mle_valid_for_loss")
    mle_pt = mle_df[keep].rename(columns=rename).copy()
    mle_pt["mle_valid"] = (mle_df["mle_valid_for_loss"].to_numpy(dtype=bool)
                           if "mle_valid_for_loss" in mle_df.columns
                           else True)
    mle_pt.drop(columns=[c for c in ["mle_valid_for_loss"]
                         if c in mle_pt.columns], inplace=True)
    mle_pt["model_label"] = friendly
    mle_pt["model_name"] = _safe_filename(friendly)
    return mle_pt, [s.key for s in specs]


# --------------------------------------------------------------------------
# 2. 2-photon activity source
# --------------------------------------------------------------------------
def load_2p_activity(path):
    """Load the sampling-window per-(trial × neuron) traces df.

    Keeps one row per ``(Name, Date, SessionNum, TrialNumber)`` (drops any
    non-sampling epochs / duplicates) and restricts to the analysis regions
    (M2_Bi / ALM_Bi). The neuronal traces live in
    ``row["traces_sets"]["neuronal"]`` keyed by ``trace_id``.
    """
    path = pathlib.Path(path)
    with path.open("rb") as f:
        df = pickle.load(f)
    if "epoch" in df.columns:
        sampling = df["epoch"].astype(str).str.lower().eq("sampling")
        if sampling.any():
            df = df[sampling]
    df = df[df.BrainRegion.isin(_ANALYSIS_REGIONS)]
    df = df.drop_duplicates(subset=IDENTITY_COLS, keep="first").copy()
    if "DV" in df.columns and "DVabs" not in df.columns:
        df["DVabs"] = df["DV"].abs()   # evidence magnitude (mirrors prepare_behavior_df)
    return df.reset_index(drop=True)


def subjects_in_2p(df_2p):
    """Sorted unique subject names (``Name``) present in the 2-photon df — pass
    to :func:`load_mle_per_trial` so only imaged subjects are loaded."""
    return sorted(df_2p["Name"].astype(str).unique())


# --------------------------------------------------------------------------
# 3. Per-(neuron, trial) table
# --------------------------------------------------------------------------
def _identity_key_frame(df):
    """A view of ``df`` with the identity columns coerced to hashable, join-
    safe scalars: Date -> normalized ``Timestamp``, Session/Trial -> ``int``."""
    out = pd.DataFrame(index=df.index)
    out["Name"] = df["Name"].astype(str)
    out["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    out["SessionNum"] = df["SessionNum"].round().astype("int64")
    out["TrialNumber"] = df["TrialNumber"].round().astype("int64")
    return out


def _max_loss_excluded_keys(mle_pt, *, tol=1e-9):
    """Set of identity-key tuples to drop when ``exclude_max_loss`` is on.

    Per **subject** (``Name``): among valid trials, find the maximum per-trial
    loss (``-mle_loglik``) and drop trials at (within ``tol`` of) that maximum.
    The lapse mixture means this worst value is not a fixed floor, so it must be
    computed from the data rather than assumed.
    """
    keys = _identity_key_frame(mle_pt).reset_index(drop=True)
    loss = -mle_pt["mle_loglik"].to_numpy(dtype=float)
    valid = mle_pt["mle_valid"].to_numpy(dtype=bool) & np.isfinite(loss)
    names = keys["Name"].to_numpy()
    excluded = set()
    for name in np.unique(names):
        sel = (names == name) & valid
        if not sel.any():
            continue
        max_loss = loss[sel].max()
        drop = sel & (loss >= max_loss - tol)
        for i in np.flatnonzero(drop):
            excluded.add((keys["Name"].iloc[i], keys["Date"].iloc[i],
                          keys["SessionNum"].iloc[i], keys["TrialNumber"].iloc[i]))
    return excluded


def _report_join_coverage(df_2p, kept_keys):
    """Print what the 2-photon → MLE join kept, and name any session lost whole.

    The join is an inner one, so unmatched 2-photon trials just disappear. That
    is intended for MLE padding trials, but a session can also vanish *entirely*
    (no fit for that recording, an identity-key mismatch, every trial invalid) —
    and a silently shorter session list is exactly the kind of thing that is
    noticed far too late. Trial-level attrition is summarised; session-level loss
    is named and warned about.
    """
    keys = _identity_key_frame(df_2p).reset_index(drop=True)
    matched = np.array([(k["Name"], k["Date"], k["SessionNum"], k["TrialNumber"])
                        in kept_keys for _, k in keys.iterrows()], dtype=bool)
    sessions = df_2p["ShortName"].to_numpy()
    all_sessions = pd.unique(sessions)
    kept_sessions = pd.unique(sessions[matched])
    lost = [s for s in all_sessions if s not in set(kept_sessions)]
    print(f"  join: {int(matched.sum())}/{len(matched)} 2-photon trials matched "
          f"a model trial · {len(kept_sessions)}/{len(all_sessions)} sessions "
          "survive")
    if lost:
        print(f"  {len(lost)} session(s) LOST ENTIRELY in the join: "
              f"{', '.join(map(str, lost))}")
        warnings.warn(
            f"build_neuron_trial_table: {len(lost)} session(s) matched no model "
            f"trial and are absent from the table ({', '.join(map(str, lost))}). "
            "Check the subject has a fit and that Date/SessionNum/TrialNumber "
            "agree between the 2-photon and MLE frames.")
    # A session kept but heavily truncated is worth a look too.
    for s in kept_sessions:
        m = sessions == s
        if matched[m].sum() < 0.5 * m.sum():
            print(f"  note: {s} kept only {int(matched[m].sum())}/{int(m.sum())} "
                  "of its trials")


def build_neuron_trial_table(df_2p, mle_pt, param_keys, *,
                             exclude_max_loss=False, tol=1e-9,
                             require_valid=True, verbose=True):
    """Build the intermediate per-(neuron, trial) dataframe (principle 7).

    One row per (neuron, matched trial) with: ``trace_id`` (neuron name),
    ``long_trace_id`` (long name ``{ShortName}_{trace_id}``), ``BrainRegion``
    (``MFC``/``LFC`` string), ``ShortName`` (session), ``TrialNumber``,
    ``max_activity`` (``np.nanmax`` of the windowed time-normalized neuronal
    trace), ``trace`` (the full time-normalized trace, kept as an object), and
    one column per available latent in ``param_keys``.

    Model latents are matched to 2-photon trials on the identity key; the inner
    match naturally drops MLE padding trials (their trial numbers never occur in
    2-photon data). ``require_valid`` also drops MLE-invalid (no-choice) trials.
    ``exclude_max_loss`` additionally drops the worst-loss trials per subject.

    ``verbose`` (default True) reports what the join kept and **names any session
    that was lost whole**, with a warning — an inner join makes a missing
    recording look identical to a dataset that never had it.
    """
    param_cols = [PARAM_BY_KEY[k].key for k in param_keys]

    # Index the per-trial latents by identity key for O(1) lookup.
    mle_keys = _identity_key_frame(mle_pt).reset_index(drop=True)
    valid = mle_pt["mle_valid"].to_numpy(dtype=bool)
    param_vals = {c: mle_pt[c].to_numpy(dtype=float) for c in param_cols}
    excluded = (_max_loss_excluded_keys(mle_pt, tol=tol)
                if exclude_max_loss else set())
    lookup = {}
    for i in range(len(mle_keys)):
        if require_valid and not valid[i]:
            continue
        key = (mle_keys["Name"].iloc[i], mle_keys["Date"].iloc[i],
               mle_keys["SessionNum"].iloc[i], mle_keys["TrialNumber"].iloc[i])
        if key in excluded:
            continue
        lookup[key] = {c: param_vals[c][i] for c in param_cols}

    # Epoch layout (sampling-window boundaries) is constant across the
    # width-normalized df; capture it once so the trace plot can draw the
    # sampling-start / movement-start dashed lines (see plot_neuron_param_traces).
    has_epochs = ("epochs_ranges" in df_2p.columns
                  and "epochs_names" in df_2p.columns and len(df_2p))
    epochs_ranges = (tuple(tuple(int(a) for a in rng)
                           for rng in df_2p["epochs_ranges"].iloc[0])
                     if has_epochs else None)
    epochs_names = (list(df_2p["epochs_names"].iloc[0]) if has_epochs else None)

    twop_keys = _identity_key_frame(df_2p).reset_index(drop=True)
    rows = []
    for pos in range(len(df_2p)):
        trial = df_2p.iloc[pos]
        k = twop_keys.iloc[pos]
        key = (k["Name"], k["Date"], k["SessionNum"], k["TrialNumber"])
        params = lookup.get(key)
        if params is None:
            continue  # no matching (valid, kept) model trial
        region = f"{BrainRegion(int(trial.BrainRegion))}"  # -> MFC / LFC
        short = trial.ShortName
        s = int(trial.trace_start_idx)
        e = int(trial.trace_end_idx) + 1
        # Behavioral drift + speed columns, constant across this trial's neurons.
        dv = float(trial["DV"]) if "DV" in df_2p.columns else np.nan
        dvabs = (float(trial["DVabs"]) if "DVabs" in df_2p.columns
                 else abs(dv))
        quantile_idx = (int(trial["quantile_idx"])
                        if "quantile_idx" in df_2p.columns
                        and pd.notnull(trial["quantile_idx"]) else -1)
        neuronal = trial["traces_sets"]["neuronal"]
        for trace_id, full_trace in neuronal.items():
            window = np.asarray(full_trace)[s:e]
            if window.size == 0 or np.all(np.isnan(window)):
                continue
            row = {
                "trace_id": trace_id,
                "long_trace_id": f"{short}_{trace_id}",
                "BrainRegion": region,
                "ShortName": short,
                "TrialNumber": int(k["TrialNumber"]),
                "max_activity": float(np.nanmax(window)),
                "trace": np.asarray(full_trace),
                "epochs_ranges": epochs_ranges,
                "epochs_names": epochs_names,
                "DV": dv,
                "DVabs": dvabs,
                "quantile_idx": quantile_idx,
            }
            row.update(params)
            rows.append(row)
    if verbose:
        _report_join_coverage(df_2p, lookup.keys())
    cols = (["trace_id", "long_trace_id", "BrainRegion", "ShortName",
             "TrialNumber", "max_activity", "trace", "epochs_ranges",
             "epochs_names", "DV", "DVabs", "quantile_idx"] + param_cols)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------
# 4. Per-neuron correlations
# --------------------------------------------------------------------------
# Fewest paired points a correlation is computed from. Below this the r is not
# just noisy but undefined, so the neuron drops out of its bin/subset entirely —
# which is why a sparse value bin loses whole sessions (see
# :func:`session_bin_coverage`).
_MIN_CORR_POINTS = 3


def _clean_xy(x, y):
    """Finite, paired ``(x, y)``; ``None`` when degenerate (constant, or fewer
    than ``_MIN_CORR_POINTS``) so the caller can emit NaNs instead of a spurious
    fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < _MIN_CORR_POINTS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return x, y


def _linfit(x, y):
    """Least-squares line through ``(x, y)`` plus its Pearson r / p-value.

    ``scipy.stats.linregress`` gives the fit line **and** ``rvalue`` (== the
    Pearson correlation) and ``pvalue`` (two-sided, H0: slope=0) in one call —
    so the scatter's fit line and the correlation table share one computation.
    Returns ``(slope, intercept, r, p)``; all NaN on degenerate input.
    """
    cleaned = _clean_xy(x, y)
    if cleaned is None:
        return np.nan, np.nan, np.nan, np.nan
    res = linregress(*cleaned)
    return (float(res.slope), float(res.intercept),
            float(res.rvalue), float(res.pvalue))


def _corr(x, y, method):
    """``(r, p)`` for one neuron × parameter, guarded. ``pearson`` reuses the
    least-squares fit (``_linfit``); ``spearman`` is the rank correlation."""
    if method == "spearman":
        cleaned = _clean_xy(x, y)
        if cleaned is None:
            return np.nan, np.nan
        r, p = spearmanr(*cleaned)
        return float(r), float(p)
    _slope, _intercept, r, p = _linfit(x, y)
    return r, p


def compute_neuron_correlations(table, param_keys, *, method="pearson"):
    """Per-neuron correlation dataframe (principle 8).

    Groups ``table`` by ``long_trace_id`` (which encodes the session, so this is
    within-session per neuron) and correlates each neuron's per-trial
    ``max_activity`` against each available latent. Returns one row per neuron
    with ``trace_id``, ``long_trace_id``, ``BrainRegion``, ``ShortName``,
    ``n_trials``, and ``{key}_r`` / ``{key}_p`` per parameter.
    """
    param_cols = [PARAM_BY_KEY[k].key for k in param_keys]
    out = []
    for long_id, neuron_df in table.groupby("long_trace_id"):
        first = neuron_df.iloc[0]
        rec = {
            "trace_id": first.trace_id,
            "long_trace_id": long_id,
            "BrainRegion": first.BrainRegion,
            "ShortName": first.ShortName,
            "n_trials": len(neuron_df),
        }
        for col in param_cols:
            r, p = _corr(neuron_df[col].to_numpy(),
                         neuron_df["max_activity"].to_numpy(), method)
            rec[f"{col}_r"] = r
            rec[f"{col}_p"] = p
        out.append(rec)
    if not out:   # keep the schema when a subset has no trials (empty value bin)
        cols = (["trace_id", "long_trace_id", "BrainRegion", "ShortName",
                 "n_trials"]
                + [f"{c}_{s}" for c in param_cols for s in ("r", "p")])
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols})
    return pd.DataFrame(out)


def split_fast_slow(table):
    """``(fast, slow)`` subsets of the neuron-trial table — fast = fastest RT
    tercile (``quantile_idx == 1``), slow = slowest (``quantile_idx == 3``); the
    middle tercile is excluded, matching the notebooks' fast/slow convention."""
    return (table[table["quantile_idx"] == _FAST_Q],
            table[table["quantile_idx"] == _SLOW_Q])


def drift_correlations(table, *, method="pearson"):
    """``(corr_fast, corr_slow)`` — per-neuron correlations of max activity vs
    the drift columns (``DV``, ``DVabs``), computed **within** the fast and slow
    trial subsets separately (a neuron may be drift-correlated in one but not the
    other)."""
    fast, slow = split_fast_slow(table)
    return (compute_neuron_correlations(fast, DRIFT_KEYS, method=method),
            compute_neuron_correlations(slow, DRIFT_KEYS, method=method))


# --------------------------------------------------------------------------
# 5. Per-neuron scatter plots (principle 9)
# --------------------------------------------------------------------------
def _region_in_name(name, region):
    return region.lower() in str(name).lower()


def _fmt_p(p, *, decimals=3):
    """Format a p-value in fixed-point (never scientific): ``0.032``. Values too
    small to show at ``decimals`` places become ``< 0.001`` rather than an
    exponent or a misleading ``0.000``."""
    if p is None or not np.isfinite(p):
        return "n/a"
    floor = 10.0 ** (-decimals)
    if 0 < p < floor:
        return f"< {floor:.{decimals}f}"
    return f"{p:.{decimals}f}"


def _p_phrase(p, *, decimals=3):
    """``"p = 0.032"`` / ``"p < 0.001"`` — :func:`_fmt_p` with the right relation
    (the ``<`` form already carries its own operator)."""
    s = _fmt_p(p, decimals=decimals)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


def _plot_one_neuron(ax, neuron_df, spec, *, zscore):
    x = neuron_df[spec.key].to_numpy(dtype=float)
    y = neuron_df["max_activity"].to_numpy(dtype=float)
    if zscore:
        sd = np.nanstd(y)
        y = (y - np.nanmean(y)) / sd if sd > 0 else y - np.nanmean(y)
    ax.scatter(x, y, s=14, color=spec.color, alpha=0.6, edgecolors="none")

    # Least-squares fit line; r / p from the same fit go in the legend.
    slope, intercept, r, p = _linfit(x, y)
    p_str = _p_phrase(p)
    if np.isfinite(slope):
        xs = np.array([np.nanmin(x), np.nanmax(x)])
        ax.plot(xs, slope * xs + intercept, color="k", lw=1.5,
                label=f"r = {r:+.3f}\n{p_str}")
        ax.legend(loc="best", fontsize="small", frameon=False, handlelength=1.0)

    ax.set_xlabel(spec.label)
    ax.set_ylabel(_ZSCORED_AGAIN_LABEL if zscore else _ACTIVITY_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{neuron_df.iloc[0].long_trace_id}\n"
                 f"{neuron_df.iloc[0].BrainRegion} · {p_str}", fontsize="small")


def plot_neuron_results(table, corr_df, param, *, mode="display", top_x=20,
                        zscore=False, min_abs_corr=None, save_root=None,
                        model_name=None, ext="svg"):
    """Scatter neuron max-activity (y, in z-score units) vs a latent (x).

    Neurons are sorted by ``|r|`` for ``param`` (descending).
    ``mode="display"`` shows the top-``top_x`` neurons inline. ``mode="save"``
    writes each neuron to
    ``{save_root}/{model_name}/traces/{param}/{r:+.3f}_{long_name}[_region].{ext}``
    (region appended only when the long name lacks it) and closes each figure
    without showing it. When ``min_abs_corr`` is set, save mode writes **only**
    neurons whose ``|r| >= min_abs_corr`` (the tuned subset).
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    if rcol not in corr_df.columns:
        raise KeyError(f"{param!r} not available in this fit "
                       f"(columns: {list(corr_df.columns)})")

    ranked = corr_df.dropna(subset=[rcol]).copy()
    ranked["_abs_r"] = ranked[rcol].abs()
    ranked = ranked.sort_values("_abs_r", ascending=False)

    by_neuron = dict(tuple(table.groupby("long_trace_id")))

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            ranked = ranked[ranked["_abs_r"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "traces" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, crow in ranked.iterrows():
            long_id = crow.long_trace_id
            neuron_df = by_neuron[long_id]
            r = crow[rcol]  # for the signed-correlation filename prefix
            fig, ax = plt.subplots(figsize=(4, 4))
            _plot_one_neuron(ax, neuron_df, spec, zscore=zscore)
            fname = f"{r:+.3f}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, crow.BrainRegion):
                fname += f"_{crow.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight")
            plt.close(fig)
        print(f"Saved {len(ranked)} neuron plots -> {out_dir}")
        return out_dir

    # display mode
    head = ranked.head(top_x)
    for _, crow in head.iterrows():
        neuron_df = by_neuron[crow.long_trace_id]
        fig, ax = plt.subplots(figsize=(4, 4))
        _plot_one_neuron(ax, neuron_df, spec, zscore=zscore)
        plt.show()
    return head


# --------------------------------------------------------------------------
# 5b. Per-neuron trace averages, grouped by parameter value range
# --------------------------------------------------------------------------
def _param_shade(base_color, t):
    """A single shade of ``base_color``'s hue for ``t`` in ``[0, 1]`` — ``t=0``
    lightest (pale, desaturated, bright), ``t=1`` darkest (saturated, dim).
    Built in HSV so the light→dark span is visible for any hue (a straight
    light-tint→base blend was invisible for already-dark hues like purple)."""
    h, s, _v = colorsys.rgb_to_hsv(*to_rgb(base_color))
    s = max(s, 0.55)                            # ensure the hue can be shaded
    return colorsys.hsv_to_rgb(h, s * (0.30 + 0.70 * t), 0.98 - 0.55 * t)


def _param_gradient(base_color, n):
    """``n`` shades of ``base_color``'s hue, **light → dark** (low value range =
    pale, high value range = dark)."""
    if n == 1:
        return [_param_shade(base_color, 0.6)]
    return [_param_shade(base_color, t) for t in np.linspace(0.0, 1.0, n)]


def _range_colors_styles(spec, value_ranges):
    """Per-range ``(colors, linestyles)`` for the trace overlay.

    Unsigned params keep the plain light→dark gradient over the ordered ranges
    (all solid). For a **signed** param (DV, Q-val) the gradient encodes only the
    range's **magnitude** — each range's colour comes from ``|midpoint| / maxmag``
    — so the symmetric extremes (e.g. ``[-1,-0.35]`` and ``[0.35,1]``) get the
    same shade, and ranges on the negative side (midpoint < 0) are drawn dashed.
    """
    if not spec.signed:
        return _param_gradient(spec.color, len(value_ranges)), \
            ["-"] * len(value_ranges)
    mids = [0.5 * (lo + hi) for lo, hi in value_ranges]
    maxmag = max((abs(m) for m in mids), default=0.0) or 1.0
    colors = [_param_shade(spec.color, abs(m) / maxmag) for m in mids]
    styles = ["--" if m < 0 else "-" for m in mids]
    return colors, styles


def _draw_epoch_lines(ax, epochs_ranges, epochs_names):
    """Dashed vertical lines at each epoch start after the first (sampling
    start, movement start), labelled — matching the other 2p trace notebooks."""
    if not epochs_ranges:
        return
    starts = [rng[0] for rng in epochs_ranges[1:]]
    labels = (list(epochs_names[1:]) if epochs_names is not None
              else [""] * len(starts))
    for xs in starts:
        ax.axvline(xs, ls="--", color="gray", alpha=0.7)
    ax.set_xticks(starts)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize="small")


def _plot_one_neuron_traces(ax, neuron_df, spec, value_ranges, r, *,
                            panel_label=None):
    """Overlay, in one axes, the mean ± SEM time-normalized trace for each
    value range of ``spec`` (gradient-coloured), with the sampling/movement
    epoch lines. ``panel_label`` (e.g. "Fast") gives a short title instead of
    the full neuron id — for multi-panel figures that carry the id in a
    suptitle."""
    vals = neuron_df[spec.key].to_numpy(dtype=float)
    traces = np.vstack([np.asarray(t, dtype=float)
                        for t in neuron_df["trace"].to_numpy()])
    x = np.arange(traces.shape[1])
    colors, styles = _range_colors_styles(spec, value_ranges)
    for i, (lo, hi) in enumerate(value_ranges):
        # Last range is closed on the right so the max value is included.
        in_range = _range_mask(vals, lo, hi,
                               closed_right=(i == len(value_ranges) - 1))
        m = int(in_range.sum())
        if m == 0:
            continue
        grp = traces[in_range]
        mean = np.nanmean(grp, axis=0)
        ax.plot(x, mean, color=colors[i], lw=1.8, ls=styles[i],
                label=f"[{lo:g}, {hi:g}]  n={m}")
        if m >= 2:  # SEM band needs >= 2 trials
            cnt = np.sum(~np.isnan(grp), axis=0)
            with np.errstate(invalid="ignore"):
                se = np.nanstd(grp, axis=0, ddof=1) / np.sqrt(np.maximum(cnt, 1))
            ax.fill_between(x, mean - se, mean + se, color=colors[i], alpha=0.18,
                            linewidth=0)
    _draw_epoch_lines(ax, neuron_df.iloc[0].get("epochs_ranges"),
                      neuron_df.iloc[0].get("epochs_names"))
    ax.set_xlabel("Normalized time")
    ax.set_ylabel(_ACTIVITY_LABEL)
    # Remove the left axis (spine + ticks), matching the other 2p trace plots.
    ax.spines[["top", "right", "left"]].set_visible(False)
    # ax.tick_params(left=False, labelleft=False)
    r_str = "n/a" if r is None or not np.isfinite(r) else f"{r:+.3f}"
    if panel_label is not None:
        ax.set_title(f"{panel_label} · r={r_str}", fontsize="small")
    else:
        ax.set_title(f"{neuron_df.iloc[0].long_trace_id}\n"
                     f"{neuron_df.iloc[0].BrainRegion} · {spec.label} · r={r_str}",
                     fontsize="small")
    ax.legend(fontsize="x-small", frameon=False, title=spec.label)


def _edges_to_ranges(bin_edges):
    """``[e0, e1, e2, …]`` bin edges → ``[(e0,e1), (e1,e2), …]`` intervals."""
    edges = list(bin_edges)
    if len(edges) < 2:
        raise ValueError("bin_edges needs at least 2 edges (i.e. >= 1 bin).")
    return list(zip(edges[:-1], edges[1:]))


def _range_mask(values, lo, hi, closed_right):
    """Boolean membership of ``values`` in ``[lo, hi)`` — or ``[lo, hi]`` when
    ``closed_right`` (used for the last bin of a set, so the maximum observed
    value is not dropped). The single place the binning convention lives; shared
    by the trace overlays and the value-binned tuning bars (section 8)."""
    v = np.asarray(values, dtype=float)
    return (v >= lo) & ((v <= hi) if closed_right else (v < hi))


def plot_neuron_param_traces(table, corr_df, param, bin_edges, *,
                             mode="display", top_x=20, min_abs_corr=None,
                             display_figsize=(6, 4), display_dpi=110,
                             save_figsize=(10, 7), save_dpi=300,
                             save_root=None, model_name=None, ext="svg"):
    """Per-neuron mean ± SEM time-normalized traces, grouped by value range.

    For each neuron, the trials are split by the value of ``param`` into the
    bins defined by ``bin_edges`` — a monotonically increasing list of edges
    (``N`` edges → ``N-1`` bins), e.g. the notebook's
    ``PARAM_VALUE_RANGES[param]``; a trial joins bin ``i`` when
    ``edge[i] <= value < edge[i+1]`` (the last bin includes its right edge).
    The mean ± SEM trace of each bin is overlaid in **one axes**,
    gradient-coloured (pale → the parameter's bar colour), with dashed
    sampling-start / movement-start epoch lines.

    Display mode uses the smaller ``display_figsize`` / ``display_dpi``; save
    mode uses the larger ``save_figsize`` / ``save_dpi`` (sized like the other 2p
    single-neuron plots). Neurons are ordered by ``|r|`` for ``param``.
    ``mode="display"`` shows the top-``top_x``; ``mode="save"`` writes each
    neuron under ``{save_root}/{model_name}/param_traces/{param}/`` and, when
    ``min_abs_corr`` is set, only the tuned subset (``|r| >= min_abs_corr``).
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    if rcol not in corr_df.columns:
        raise KeyError(f"{param!r} not available in this fit "
                       f"(columns: {list(corr_df.columns)})")
    value_ranges = _edges_to_ranges(bin_edges)

    ranked = corr_df.dropna(subset=[rcol]).copy()
    ranked["_abs_r"] = ranked[rcol].abs()
    ranked = ranked.sort_values("_abs_r", ascending=False)
    by_neuron = dict(tuple(table.groupby("long_trace_id")))

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            ranked = ranked[ranked["_abs_r"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "param_traces" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for _, crow in ranked.iterrows():
            long_id = crow.long_trace_id
            r = crow[rcol]
            fig, ax = plt.subplots(figsize=save_figsize, dpi=save_dpi)
            _plot_one_neuron_traces(ax, by_neuron[long_id], spec, value_ranges, r)
            fname = f"{r:+.3f}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, crow.BrainRegion):
                fname += f"_{crow.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight",
                        dpi=save_dpi)
            plt.close(fig)
        print(f"Saved {len(ranked)} neuron trace plots -> {out_dir}")
        return out_dir

    head = ranked.head(top_x)
    for _, crow in head.iterrows():
        fig, ax = plt.subplots(figsize=display_figsize, dpi=display_dpi)
        _plot_one_neuron_traces(ax, by_neuron[crow.long_trace_id], spec,
                                value_ranges, crow[rcol])
        plt.show()
    return head


# --------------------------------------------------------------------------
# 6. Per-brain-region "fraction tuned" bar chart, with a shuffle baseline
# --------------------------------------------------------------------------
def _available_param_keys(corr_df):
    """The parameter keys present in a correlation dataframe (have a ``_r``)."""
    return [s.key for s in PARAM_SPECS if f"{s.key}_r" in corr_df.columns]


def _session_tuned_percent(corr_df, rcol, min_abs_corr):
    """Series: for each session, % of assessable neurons that are tuned
    (``|r| >= min_abs_corr``). Neurons with undefined r are not assessable and
    drop out of both numerator and denominator."""
    assessable = corr_df.dropna(subset=[rcol]).copy()
    assessable["_tuned"] = assessable[rcol].abs() >= min_abs_corr
    # mean of a boolean per session == fraction tuned; ×100 == percent.
    return (100.0 * assessable.groupby("ShortName")["_tuned"].mean()).dropna()


def _shuffle_null_tuned(table, param_cols, min_abs_corr, n_shuffles, rng):
    """Per-neuron shuffle null for the "tuned?" decision (chance baseline).

    The shuffle is at the **neuron level**: each neuron's per-trial
    ``max_activity`` is permuted (its parameter vector stays put), the
    correlation is recomputed, and ``|r| >= min_abs_corr`` re-decides tuning.
    Because permuting the activity leaves its variance unchanged, only the
    fit's numerator moves, so all ``n_shuffles`` r's for a neuron come from one
    ``(n_shuffles, n_trials)`` permutation matrix.

    Returns ``(meta_df, null_tuned, assessable)``:
      - ``meta_df``: one row per neuron with ``long_trace_id`` / ``ShortName`` /
        ``BrainRegion`` (row order defines the neuron index used below);
      - ``null_tuned[param]``: ``(n_neurons, n_shuffles)`` bool — tuned under
        each shuffle;
      - ``assessable[param]``: ``(n_neurons,)`` bool — False for neurons whose
        correlation is undefined (constant param / activity, <3 trials); those
        drop out of both numerator and denominator, matching the observed side.
    """
    metas, null = [], {p: [] for p in param_cols}
    assess = {p: [] for p in param_cols}
    for long_id, ndf in table.groupby("long_trace_id"):
        metas.append((long_id, ndf.iloc[0].ShortName, ndf.iloc[0].BrainRegion))
        y = ndf["max_activity"].to_numpy(dtype=float)
        n = y.size
        yc = y - y.mean()
        sy = np.sqrt((yc ** 2).sum())
        perm = np.argsort(rng.random((n_shuffles, n)), axis=1)  # (K, n)
        yc_perm = yc[perm]                                      # (K, n)
        for col in param_cols:
            x = ndf[col].to_numpy(dtype=float)
            xc = x - x.mean()
            sx = np.sqrt((xc ** 2).sum())
            if n < 3 or sx == 0 or sy == 0:
                assess[col].append(False)
                null[col].append(np.zeros(n_shuffles, dtype=bool))
                continue
            r = (yc_perm @ xc) / (sy * sx)                     # (K,)
            assess[col].append(True)
            null[col].append(np.abs(r) >= min_abs_corr)
    meta_df = pd.DataFrame(metas,
                           columns=["long_trace_id", "ShortName", "BrainRegion"])
    # dtype=bool keeps the masks combinable with `&` even when the subset held no
    # neurons at all (an empty value bin -> zero-length arrays).
    null = {p: np.asarray(v, dtype=bool) for p, v in null.items()}
    assess = {p: np.asarray(v, dtype=bool) for p, v in assess.items()}
    return meta_df, null, assess


def _null_region_distribution(meta_df, null_p, assess_p, region_mask):
    """Session-mean % tuned under the shuffle for one region × parameter, as an
    ``(n_shuffles,)`` array (per session % tuned, then mean across sessions —
    the same aggregation as the observed side)."""
    idx = np.flatnonzero(region_mask & assess_p)
    if idx.size == 0:
        return None
    sess = meta_df["ShortName"].to_numpy()
    sess_pcts = [100.0 * null_p[np.flatnonzero((sess == s) & region_mask
                                               & assess_p)].mean(axis=0)
                 for s in pd.unique(sess[idx])]
    return np.vstack(sess_pcts).mean(axis=0)


# --------------------------------------------------------------------------
# Significance annotation + permutation / bootstrap tests
#
# The paper reports significance as permutation / hierarchical-bootstrap
# p-values with ``*/**/***`` stars and Holm correction (see the ``opto`` package,
# e.g. ``opto/bootstrap2regions.py`` and ``opto/optofeedback.py``). These helpers
# bring the neural-correlate bar charts in line with that convention:
#   - per-bar "vs chance" uses the per-neuron activity-shuffle null already built
#     by :func:`_shuffle_null_tuned` (``+1/(n+1)`` permutation p-value);
#   - fast-vs-slow uses a session-paired sign-flip permutation;
#   - MFC-vs-LFC (cross-region) uses a region→session→neuron hierarchical
#     bootstrap with a two-sided sign-change p-value.
# All of this is skipped when ``run_stats=False`` for fast plot-only runs.
# --------------------------------------------------------------------------
def _sigstar(p):
    """``*/**/***`` for p < 0.05 / 0.01 / 0.001, ``"n.s."`` above 0.05, ``""``
    when undefined — the mapping used across the paper's figures."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _holm_adjust(pvals):
    """Holm–Bonferroni step-down over a family of p-values (NaN-safe: NaNs are
    passed through and excluded from the correction). Mirrors the
    ``multipletests(method="holm")`` calls used in ``opto``."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if finite.any():
        out[finite] = multipletests(p[finite], method="holm")[1]
    return out


def _sign_change_p(observed, draws):
    """Two-sided sign-change p-value (Svoboda/Brody convention, as in
    ``opto/bootstrap2regions.py::_sign_change_p_two_sided``): the fraction of
    bootstrap ``draws`` whose sign is opposite the ``observed`` effect, doubled
    and capped at 1."""
    if observed is None or not np.isfinite(observed):
        return np.nan
    valid = np.asarray(draws, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.nan
    if observed == 0:
        return 1.0
    opposite = -1.0 if observed > 0 else 1.0
    p_one = float(np.mean(np.sign(valid) == opposite))
    return float(min(1.0, 2.0 * p_one))


def _add_sig_bracket(ax, x1, x2, y, text, *, tick=None, lw=1.0, fontsize=13):
    """Draw a ``[x1, x2]`` significance bracket at height ``y`` with ``text``
    (a star string) centred above it. Adapted from
    ``opto/optofeedback.py::_add_sig_bracket``. ``tick`` is the drop-down length
    of the bracket ends; returns the top y so callers can stack brackets."""
    if not text:
        return y
    if tick is None:
        span = ax.get_ylim()
        tick = 0.02 * (span[1] - span[0])
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color="k", lw=lw, zorder=6, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + tick, text, ha="center", va="bottom",
            fontsize=fontsize, color="k", zorder=6)
    return y + tick


def _paired_perm_fastslow(corr_fast, corr_slow, rcol, min_abs_corr, n_perm, rng,
                          *, region_values=None):
    """Session-paired sign-flip permutation of the fast−slow drift-tuning %.

    Pairs the per-session % of drift-tuned neurons in the fast and slow subsets
    (:func:`_session_tuned_percent`) on ``ShortName``; the observed statistic is
    the mean paired difference (fast − slow). The null flips the sign of each
    session's paired difference independently. Returns ``(obs_diff, p)`` with the
    ``+1/(n+1)`` correction; both NaN when fewer than two paired sessions.
    """
    cf = (corr_fast if region_values is None
          else corr_fast[corr_fast.BrainRegion.isin(region_values)])
    cs = (corr_slow if region_values is None
          else corr_slow[corr_slow.BrainRegion.isin(region_values)])
    fast_pct = _session_tuned_percent(cf, rcol, min_abs_corr)
    slow_pct = _session_tuned_percent(cs, rcol, min_abs_corr)
    common = fast_pct.index.intersection(slow_pct.index)
    if len(common) < 2:
        return np.nan, np.nan
    d = (fast_pct.loc[common] - slow_pct.loc[common]).to_numpy(dtype=float)
    obs = float(d.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p


def _pct_from_absr(abs_r, min_abs_corr):
    """% of assessable neurons tuned, from an ``(n_neurons, n_params)`` matrix of
    ``|r|`` (NaN where undefined). Assessable = any param defined; tuned = any
    defined ``|r| >= min_abs_corr`` — the factor OR, matching
    :func:`_session_factor_percent`. NaN when no assessable neuron."""
    if abs_r.size == 0:
        return np.nan
    assessable = np.isfinite(abs_r).any(axis=1)
    if not assessable.any():
        return np.nan
    tuned = (abs_r >= min_abs_corr).any(axis=1)  # NaN >= x is False
    return 100.0 * float(tuned[assessable].mean())


def _session_absr_mats(corr_df, param_keys):
    """List (one per session) of ``(n_neurons, n_params)`` ``|r|`` matrices for a
    per-neuron correlation frame — the fast unit for the region bootstrap."""
    cols = [f"{k}_r" for k in param_keys]
    return [np.abs(g[cols].to_numpy(dtype=float))
            for _, g in corr_df.groupby("ShortName")]


def _region_stat_from_mats(mats, min_abs_corr):
    """Session-mean % tuned (across-session mean of per-session %) for a region."""
    vals = [v for v in (_pct_from_absr(m, min_abs_corr) for m in mats)
            if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _boot_region_stat(mats, min_abs_corr, rng):
    """One hierarchical-bootstrap replicate of a region's session-mean %:
    resample sessions with replacement, then neurons within each drawn session
    with replacement."""
    n = len(mats)
    if n == 0:
        return np.nan
    vals = []
    for j in rng.integers(0, n, size=n):
        m = mats[j]
        if m.shape[0] == 0:
            continue
        v = _pct_from_absr(m[rng.integers(0, m.shape[0], size=m.shape[0])],
                           min_abs_corr)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _hier_bootstrap_region_diff(mats_a, mats_b, min_abs_corr, n_boot, rng):
    """MFC-vs-LFC (region A vs B) hierarchical bootstrap on the "% tuned" stat.

    ``mats_a`` / ``mats_b`` are per-session ``|r|`` matrices (see
    :func:`_session_absr_mats`) for the two regions. Both regions are resampled
    independently at the session then neuron level (:func:`_boot_region_stat`);
    the statistic is ``stat_a - stat_b``. Returns a dict with the observed diff,
    each region's observed stat, and the two-sided sign-change p over ``n_boot``.
    """
    a_obs = _region_stat_from_mats(mats_a, min_abs_corr)
    b_obs = _region_stat_from_mats(mats_b, min_abs_corr)
    if not (np.isfinite(a_obs) and np.isfinite(b_obs)):
        return dict(diff=np.nan, p=np.nan, stat_a=a_obs, stat_b=b_obs)
    obs = float(a_obs - b_obs)
    draws = np.array([_boot_region_stat(mats_a, min_abs_corr, rng)
                      - _boot_region_stat(mats_b, min_abs_corr, rng)
                      for _ in range(n_boot)], dtype=float)
    return dict(diff=obs, p=_sign_change_p(obs, draws),
                stat_a=float(a_obs), stat_b=float(b_obs))


def plot_region_bars(corr_df, table, param_keys=None, *, min_abs_corr=0.3,
                     n_shuffles=1000, by_region=True, seed=0, ci=(2.5, 97.5),
                     run_stats=True, save=False, save_root=None,
                     model_name=None, ext="svg"):
    """Grouped "fraction tuned" bars per brain region.

    For each brain region (``by_region=True`` → one panel per MFC/LFC;
    ``by_region=False`` → a single pooled panel) the x-axis is the model
    parameters. Each **coloured bar** is the observed % of tuned neurons
    (``|r| >= min_abs_corr``), computed per session then averaged across
    sessions; the error bar is the **SEM across sessions** (the distribution of
    per-session percentages within the region).

    When ``run_stats`` (default), each bar is tested against a per-neuron
    activity-shuffle chance level (``n_shuffles`` permutations, ``+1/(n+1)``
    p-value; see :func:`_shuffle_null_tuned`), the p-values are Holm-corrected
    across the bars of a panel, and significance is annotated as ``*/**/***``
    stars (``n.s.`` otherwise). Set ``run_stats=False`` to skip the permutation
    entirely for a fast plot-only run.

    Returns a tidy summary dataframe (one row per region × parameter). Always
    shown; ``save`` also writes the figure + summary under
    ``{save_root}/{model_name}/tuning_bars/``.

    "Tuned" is a per-**neuron** property (its activity vs the parameter across
    the session's trials); the bar is the % of such neurons, so the y-axis reads
    "Tuned neurons (%)".
    """
    if param_keys is None:
        param_keys = _available_param_keys(corr_df)
    param_cols = [PARAM_BY_KEY[k].key for k in param_keys]
    if not param_cols:
        raise ValueError("No available parameters to plot.")
    lo, hi = ci

    rng = np.random.default_rng(seed)
    if run_stats:
        meta_df, null, assess = _shuffle_null_tuned(
            table, param_cols, min_abs_corr, n_shuffles, rng)
    else:
        meta_df = null = assess = None

    if by_region:
        regions = [(r, [r]) for r in sorted(pd.unique(corr_df.BrainRegion))]
    else:
        regions = [("MFC & LFC", None)]

    x = np.arange(len(param_cols))
    fig, axs = plt.subplots(1, len(regions), figsize=(4.2 * len(regions), 4.4),
                            squeeze=False, sharey=True)
    rows = []
    for ax, (region, rvals) in zip(axs[0], regions):
        obs_means, obs_sems, pvals = [], [], []
        panel_rows = []
        for col in param_cols:
            rcol = f"{col}_r"
            grp = (corr_df if rvals is None
                   else corr_df[corr_df.BrainRegion.isin(rvals)])
            pcts = _session_tuned_percent(grp, rcol, min_abs_corr)
            obs_mean = float(pcts.mean()) if len(pcts) else np.nan
            obs_sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
            null_mean = null_lo = null_hi = pval = np.nan
            if run_stats:
                mask = (np.ones(len(meta_df), dtype=bool) if rvals is None
                        else np.isin(meta_df["BrainRegion"].to_numpy(), rvals))
                null_dist = _null_region_distribution(meta_df, null[col],
                                                      assess[col], mask)
                if null_dist is not None and np.isfinite(obs_mean):
                    null_mean = float(null_dist.mean())
                    null_lo, null_hi = np.percentile(null_dist, [lo, hi])
                    # +1/+1 correction so p is never exactly 0.
                    pval = float((np.sum(null_dist >= obs_mean) + 1)
                                 / (n_shuffles + 1))
            obs_means.append(obs_mean); obs_sems.append(obs_sem); pvals.append(pval)
            panel_rows.append({"BrainRegion": region, "param": col,
                               "observed_pct": obs_mean, "observed_sem": obs_sem,
                               "shuffle_pct": null_mean, f"ci{lo:g}": null_lo,
                               f"ci{hi:g}": null_hi, "n_sessions": len(pcts),
                               "p_vs_shuffle": pval})

        # Holm across the bars of this panel, then annotate stars.
        p_holm = _holm_adjust(pvals) if run_stats else np.full(len(pvals), np.nan)
        for r, ph in zip(panel_rows, p_holm):
            r["p_vs_shuffle_holm"] = float(ph) if np.isfinite(ph) else np.nan
        rows.extend(panel_rows)

        colors = [PARAM_BY_KEY[k].color for k in param_keys]
        obs_means = np.array(obs_means, dtype=float)
        obs_sems = np.array(obs_sems, dtype=float)
        ax.bar(x, obs_means, width=0.62, color=colors, alpha=0.85,
               edgecolor="k", linewidth=0.5, yerr=obs_sems, capsize=3, zorder=2)
        if run_stats:
            for xi, om, os_, ph in zip(x, obs_means, obs_sems, p_holm):
                if np.isfinite(om):
                    ax.text(xi, om + os_ + 0.6, _sigstar(ph), ha="center",
                            va="bottom", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_BY_KEY[k].label for k in param_keys],
                           rotation=20, ha="right", fontsize="small")
        ax.set_title(f"{region}  (|r| ≥ {min_abs_corr:g})")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0][0].set_ylabel("Tuned neurons (%)")
    fig.suptitle(f"Parameter tuning — {model_name or ''}", y=1.02)
    fig.tight_layout()

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "tuning_bars")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "by_region" if by_region else "combined"
        fig.savefig(out_dir / f"bars_{tag}.{ext}", bbox_inches="tight")
        summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(f"Saved tuning bars -> {out_dir}")
    plt.show()
    return summary


# --------------------------------------------------------------------------
# 6b. Factor bars — group several parameters into one "modulation" axis
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Factor:
    """A named group of parameters treated as **one** modulation axis.

    A neuron is "modulated" by the factor when ``|r| >= min_abs_corr`` for **any**
    of the factor's ``param_keys``. Because the correlation dataframe is one row
    per neuron (unique ``long_trace_id``), the "any" is an OR across that
    neuron's columns — so a neuron tuned to more than one of the grouped params
    is still counted **once** (no double-counting).

    ``name``  — short key / save-folder tag.
    ``label`` — x-axis label (e.g. "Q-modulated").
    ``color`` — bar colour.
    """
    name: str
    param_keys: tuple
    label: str
    color: str


def default_factors(corr_df=None):
    """The paper's three decision "factors": **Q** (Q-left/right/val OR'd into one
    Q-modulated axis), **Reward-Rate**, and **DV**. When ``corr_df`` is given,
    each factor is trimmed to the params actually present (a factor with none is
    dropped) so a Q-only or RR-only fit still works."""
    facs = (
        Factor("Q", ("Q_L", "Q_R", "Q_val"), "Q-modulated", "purple"),
        Factor("RewardRate", ("RewardRate",), "Reward-Rate modulated", "teal"),
        Factor("DV", ("DV",), "DV-modulated", "steelblue"),
    )
    if corr_df is None:
        return list(facs)
    trimmed = []
    for f in facs:
        present = tuple(k for k in f.param_keys if f"{k}_r" in corr_df.columns)
        if present:
            trimmed.append(replace(f, param_keys=present))
    return trimmed


def _factor_masks(corr_df, param_keys, min_abs_corr):
    """Per-neuron ``(assessable, tuned)`` boolean Series for a factor: assessable
    if **any** grouped ``|r|`` is defined; tuned if **any** defined ``|r|`` meets
    the threshold (``NaN >= x`` is False, so undefined params never tune)."""
    R = corr_df[[f"{k}_r" for k in param_keys]].abs()
    return R.notna().any(axis=1), (R >= min_abs_corr).any(axis=1)


def _session_factor_percent(corr_df, param_keys, min_abs_corr):
    """Series: per session, % of assessable neurons modulated by the factor
    (OR over the grouped params). Mirrors ``_session_tuned_percent`` but across a
    group; unassessable neurons drop from numerator and denominator."""
    assessable, tuned = _factor_masks(corr_df, param_keys, min_abs_corr)
    sub = corr_df.loc[assessable, ["ShortName"]].copy()
    sub["_tuned"] = tuned[assessable]
    return (100.0 * sub.groupby("ShortName")["_tuned"].mean()).dropna()


def _factor_null(null, assess, param_keys):
    """OR a factor's per-param shuffle arrays into one ``(null_factor, assess_factor)``.
    ``_shuffle_null_tuned`` already zeroes ``null`` where a param is unassessable,
    so a plain OR gives "tuned to any grouped param under the shuffle"."""
    nfac = np.zeros_like(null[param_keys[0]], dtype=bool)
    afac = np.zeros_like(assess[param_keys[0]], dtype=bool)
    for k in param_keys:
        nfac = nfac | null[k]
        afac = afac | assess[k]
    return nfac, afac


def plot_factor_bars(corr_df, table, factors=None, *, min_abs_corr=0.3,
                     n_shuffles=1000, by_region=True, seed=0, ci=(2.5, 97.5),
                     run_stats=True, title=None, save=False, save_root=None,
                     model_name=None, ext="svg"):
    """Grouped "% modulated neurons" bars, one bar per **factor** (a group of
    parameters OR'd together — see :class:`Factor`), with the same per-neuron
    activity-shuffle permutation test + ``*/**/***`` stars as
    :func:`plot_region_bars`.

    Use it for the three cross-factor views the notebook asks for:
      - a single **DV** factor (DV over *all* trials, not split fast/slow);
      - a single **Q** factor (Q-left/right/val collapsed into one
        "Q-modulated" bar, each neuron counted once);
      - **Q vs Reward-Rate vs DV** side by side (:func:`default_factors`), so the
        reader can compare the main drivers of the decision.

    ``by_region=True`` gives one panel per MFC/LFC; ``by_region=False`` pools all
    regions. Bars are session mean ± SEM (across sessions); p-values (vs the
    shuffle chance level) are Holm-corrected across the factors of a panel and
    annotated as stars when ``run_stats``. The grouped params must all have
    ``_r`` columns in ``corr_df`` (correlate against ``PARAM_KEYS + ["DV"]``).
    Returns a tidy summary (row per region × factor). ``save`` also writes the
    figure + summary under ``{save_root}/{model_name}/factor_bars/``.
    """
    if factors is None:
        factors = default_factors(corr_df)
    if not factors:
        raise ValueError("No factors to plot (no matching params in corr_df).")
    for f in factors:
        missing = [k for k in f.param_keys if f"{k}_r" not in corr_df.columns]
        if missing:
            raise KeyError(f"factor {f.name!r} needs {missing} in corr_df "
                           f"(columns: {list(corr_df.columns)})")
    lo, hi = ci
    all_cols = sorted({k for f in factors for k in f.param_keys})

    rng = np.random.default_rng(seed)
    if run_stats:
        meta_df, null, assess = _shuffle_null_tuned(
            table, all_cols, min_abs_corr, n_shuffles, rng)
    else:
        meta_df = null = assess = None

    if by_region:
        regions = [(r, [r]) for r in sorted(pd.unique(corr_df.BrainRegion))]
    else:
        regions = [("MFC & LFC", None)]

    x = np.arange(len(factors))
    fig, axs = plt.subplots(1, len(regions), figsize=(4.2 * len(regions), 4.4),
                            squeeze=False, sharey=True)
    rows = []
    for ax, (region, rvals) in zip(axs[0], regions):
        obs_means, obs_sems, pvals = [], [], []
        panel_rows = []
        for f in factors:
            grp = (corr_df if rvals is None
                   else corr_df[corr_df.BrainRegion.isin(rvals)])
            pcts = _session_factor_percent(grp, f.param_keys, min_abs_corr)
            obs_mean = float(pcts.mean()) if len(pcts) else np.nan
            obs_sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
            null_mean = null_lo = null_hi = pval = np.nan
            if run_stats:
                nfac, afac = _factor_null(null, assess, f.param_keys)
                mask = (np.ones(len(meta_df), dtype=bool) if rvals is None
                        else np.isin(meta_df["BrainRegion"].to_numpy(), rvals))
                null_dist = _null_region_distribution(meta_df, nfac, afac, mask)
                if null_dist is not None and np.isfinite(obs_mean):
                    null_mean = float(null_dist.mean())
                    null_lo, null_hi = np.percentile(null_dist, [lo, hi])
                    pval = float((np.sum(null_dist >= obs_mean) + 1)
                                 / (n_shuffles + 1))
            obs_means.append(obs_mean); obs_sems.append(obs_sem); pvals.append(pval)
            panel_rows.append({"BrainRegion": region, "factor": f.name,
                               "params": "+".join(f.param_keys),
                               "observed_pct": obs_mean, "observed_sem": obs_sem,
                               "shuffle_pct": null_mean, f"ci{lo:g}": null_lo,
                               f"ci{hi:g}": null_hi, "n_sessions": len(pcts),
                               "p_vs_shuffle": pval})

        p_holm = _holm_adjust(pvals) if run_stats else np.full(len(pvals), np.nan)
        for r, ph in zip(panel_rows, p_holm):
            r["p_vs_shuffle_holm"] = float(ph) if np.isfinite(ph) else np.nan
        rows.extend(panel_rows)

        obs_means = np.array(obs_means, dtype=float)
        obs_sems = np.array(obs_sems, dtype=float)
        ax.bar(x, obs_means, width=0.62, color=[f.color for f in factors],
               alpha=0.85, edgecolor="k", linewidth=0.5, yerr=obs_sems,
               capsize=3, zorder=2)
        if run_stats:
            for xi, om, os_, ph in zip(x, obs_means, obs_sems, p_holm):
                if np.isfinite(om):
                    ax.text(xi, om + os_ + 0.6, _sigstar(ph), ha="center",
                            va="bottom", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f.label for f in factors], rotation=20, ha="right",
                           fontsize="small")
        ax.set_title(f"{region}  (|r| ≥ {min_abs_corr:g})")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0][0].set_ylabel("Modulated neurons (%)")
    fig.suptitle(title or f"Factor modulation — {model_name or ''}", y=1.02)
    fig.tight_layout()

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "factor_bars")
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = ("_".join(f.name for f in factors)
               + ("_by_region" if by_region else "_combined"))
        fig.savefig(out_dir / f"bars_{tag}.{ext}", bbox_inches="tight")
        summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(f"Saved factor bars -> {out_dir}")
    plt.show()
    return summary


def _factor_region_stat(corr_df, meta_df, null, assess, param_keys,
                        region_values, *, min_abs_corr, n_shuffles, ci):
    """Observed + shuffle stats for one factor in one region, off a
    **precomputed** shuffle bundle (``meta_df``/``null``/``assess`` from
    :func:`_shuffle_null_tuned`). ``region_values`` is ``None`` (pool all) or a
    list of ``BrainRegion`` labels. Returns a stats dict."""
    lo, hi = ci
    grp = (corr_df if region_values is None
           else corr_df[corr_df.BrainRegion.isin(region_values)])
    pcts = _session_factor_percent(grp, param_keys, min_abs_corr)
    obs_mean = float(pcts.mean()) if len(pcts) else np.nan
    obs_sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
    nfac, afac = _factor_null(null, assess, param_keys)
    mask = (np.ones(len(meta_df), dtype=bool) if region_values is None
            else np.isin(meta_df["BrainRegion"].to_numpy(), list(region_values)))
    null_dist = _null_region_distribution(meta_df, nfac, afac, mask)
    if null_dist is None or not np.isfinite(obs_mean):
        return dict(obs_mean=obs_mean, obs_sem=obs_sem, null_mean=np.nan,
                    null_lo=np.nan, null_hi=np.nan, pval=np.nan,
                    n_sessions=len(pcts))
    return dict(obs_mean=obs_mean, obs_sem=obs_sem,
                null_mean=float(null_dist.mean()),
                null_lo=float(np.percentile(null_dist, lo)),
                null_hi=float(np.percentile(null_dist, hi)),
                pval=float((np.sum(null_dist >= obs_mean) + 1) / (n_shuffles + 1)),
                n_sessions=len(pcts))


def plot_factor_bars_fastslow_dv(corr_all, table, *, factors=None, dv_key="DV",
                                 min_abs_corr=0.3, n_shuffles=1000, n_perm=1000,
                                 n_boot=10000, by_region=True, seed=0,
                                 ci=(2.5, 97.5), run_stats=True, bar_width=0.62,
                                 slot=1.0, title=None, save=False, save_root=None,
                                 model_name=None, ext="svg"):
    """Like :func:`plot_factor_bars`, but the **DV** factor is split into two
    touching sub-bars — DV within **fast** (tomato) and **slow** (goldenrod)
    trials — instead of one all-trials DV bar. This is the paper's factor-
    modulation summary (Q · Reward-Rate · DV-fast · DV-slow).

    Layout: the non-DV factors (default Q + Reward-Rate) occupy one slot each;
    the DV pair is the next slot, with its **fast** bar at that slot centre so the
    Reward-Rate → fast-DV gap equals the Q → Reward-Rate gap, and the **slow** bar
    butted against it (no gap). DV fast/slow percentages are the % DV-correlated
    neurons computed **within** each RT-tercile subset (``quantile_idx`` 1 / 3).

    Bars are session mean ± SEM (across sessions). With ``by_region`` each region
    (MFC, LFC) is drawn as its **own figure**, the two sharing a common y-range so
    bar heights are directly comparable across figures. When ``run_stats`` every
    bar carries a ``*/**/***`` star for that bar **vs its per-neuron activity-
    shuffle chance level** (``+1/(N+1)`` permutation p), Holm-corrected across the
    four bars *within that region* — there is no cross-region (MFC-vs-LFC)
    comparison here. A bracket over the DV fast|slow pair reports the session-
    paired fast-vs-slow permutation (:func:`_paired_perm_fastslow`), Holm-corrected
    across regions. Set ``run_stats=False`` to skip all permutations. ``n_boot`` is
    retained for interface compatibility and is unused here.

    ``corr_all`` supplies the non-DV factors (correlate against
    ``PARAM_KEYS + ["DV"]``); the fast/slow DV correlations are derived from
    ``table`` here. Returns a tidy summary (row per region × bar); ``save`` writes
    one figure per region (``bars_fastslowDV_{MFC,LFC}.{ext}`` or
    ``…_combined`` when ``by_region=False``) under
    ``{save_root}/{model_name}/factor_bars/``.
    """
    lo, hi = ci
    non_dv = [f for f in (factors or default_factors(corr_all))
              if dv_key not in f.param_keys]
    if not non_dv:
        raise ValueError("no non-DV factors to plot")
    if f"{dv_key}_r" not in corr_all.columns:
        raise KeyError(f"{dv_key!r} correlations missing from corr_all "
                       f"(correlate against PARAM_KEYS + ['{dv_key}'])")

    # DV split into fast / slow, correlated WITHIN each RT-tercile subset.
    fast_table, slow_table = split_fast_slow(table)
    corr_fast = compute_neuron_correlations(fast_table, [dv_key])
    corr_slow = compute_neuron_correlations(slow_table, [dv_key])

    rng = np.random.default_rng(seed)
    nd_cols = sorted({k for f in non_dv for k in f.param_keys})
    if run_stats:
        bundle_all = _shuffle_null_tuned(table, nd_cols, min_abs_corr,
                                         n_shuffles, rng)
        bundle_fast = _shuffle_null_tuned(fast_table, [dv_key], min_abs_corr,
                                          n_shuffles, rng)
        bundle_slow = _shuffle_null_tuned(slow_table, [dv_key], min_abs_corr,
                                          n_shuffles, rng)
    else:
        bundle_all = bundle_fast = bundle_slow = None

    # Ordered bars, each tagged with its layout slot + within-slot offset.
    bars = []
    for gi, f in enumerate(non_dv):
        bars.append(dict(name=f.name, label=f.label, color=f.color,
                         corr=corr_all, bundle=bundle_all, keys=f.param_keys,
                         group=gi, within=0))
    g = len(non_dv)
    bars.append(dict(name=f"{dv_key}_fast", label=f"{dv_key} (fast)",
                     color=_FAST_COLOR, corr=corr_fast, bundle=bundle_fast,
                     keys=(dv_key,), group=g, within=0))
    bars.append(dict(name=f"{dv_key}_slow", label=f"{dv_key} (slow)",
                     color=_SLOW_COLOR, corr=corr_slow, bundle=bundle_slow,
                     keys=(dv_key,), group=g, within=1))
    xs = np.array([b["group"] * slot + b["within"] * bar_width for b in bars])

    if by_region:
        regions = [(r, [r]) for r in sorted(pd.unique(corr_all.BrainRegion))]
    else:
        regions = [("MFC & LFC", None)]

    # DV fast vs slow, session-paired within each region (Holm across regions).
    dv_fs = {}
    if run_stats:
        for region, rvals in regions:
            diff, p = _paired_perm_fastslow(corr_fast, corr_slow, f"{dv_key}_r",
                                            min_abs_corr, n_perm, rng,
                                            region_values=rvals)
            dv_fs[region] = dict(diff=diff, p=p)
        for (region, _), ph in zip(regions,
                                   _holm_adjust([dv_fs[r]["p"]
                                                 for r, _ in regions])):
            dv_fs[region]["p_holm"] = float(ph) if np.isfinite(ph) else np.nan

    # ---- pass 1: per-region bar stats + vs-chance stars (Holm within region) --
    # Each region is drawn in its own figure now, so significance stars report
    # each bar vs its per-neuron activity-shuffle chance level, Holm-corrected
    # across the four bars *within that region* (no cross-region comparison).
    region_data = {}
    rows = []
    for region, rvals in regions:
        bstats = []
        for b in bars:
            bar_corr = (b["corr"] if rvals is None
                        else b["corr"][b["corr"].BrainRegion.isin(rvals)])
            pcts = _session_factor_percent(bar_corr, list(b["keys"]), min_abs_corr)
            obs_mean = float(pcts.mean()) if len(pcts) else np.nan
            obs_sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
            null_mean = null_lo = null_hi = pval = np.nan
            if run_stats:
                b_meta, b_null, b_assess = b["bundle"]
                st = _factor_region_stat(b["corr"], b_meta, b_null, b_assess,
                                         b["keys"], rvals, min_abs_corr=min_abs_corr,
                                         n_shuffles=n_shuffles, ci=ci)
                null_mean, null_lo = st["null_mean"], st["null_lo"]
                null_hi, pval = st["null_hi"], st["pval"]
            bstats.append(dict(name=b["name"], keys=b["keys"], obs_mean=obs_mean,
                               obs_sem=obs_sem, null_mean=null_mean,
                               null_lo=null_lo, null_hi=null_hi, pval=pval,
                               n=len(pcts)))
        vs_chance_p = [s["pval"] for s in bstats]
        holm_p = (_holm_adjust(vs_chance_p) if run_stats
                  else np.full(len(bstats), np.nan))
        oms = np.array([s["obs_mean"] for s in bstats], float)
        osm = np.array([s["obs_sem"] for s in bstats], float)
        for s, hp in zip(bstats, holm_p):
            fs = dv_fs.get(region, {}) if s["name"].startswith(f"{dv_key}_") else {}
            rows.append({"BrainRegion": region, "factor": s["name"],
                         "params": "+".join(s["keys"]),
                         "observed_pct": s["obs_mean"], "observed_sem": s["obs_sem"],
                         "shuffle_pct": s["null_mean"], f"ci{lo:g}": s["null_lo"],
                         f"ci{hi:g}": s["null_hi"], "n_sessions": s["n"],
                         "p_vs_shuffle": s["pval"],
                         "p_vs_shuffle_holm": (float(hp) if np.isfinite(hp)
                                               else np.nan),
                         "dv_fast_minus_slow": fs.get("diff", np.nan),
                         "p_dv_fast_vs_slow": fs.get("p", np.nan),
                         "p_dv_fast_vs_slow_holm": fs.get("p_holm", np.nan)})
        # Base height of the DV fast-vs-slow bracket (drawn whenever the paired
        # test returned a value — n.s. included).
        dv_base = np.nan
        fs_star = _sigstar(dv_fs.get(region, {}).get("p_holm", np.nan))
        if (run_stats and fs_star and np.isfinite(oms[-2])
                and np.isfinite(oms[-1])):
            span = float(np.nanmax(oms + osm)) if np.isfinite(oms).any() else 0.0
            dv_base = max(oms[-2] + osm[-2], oms[-1] + osm[-1]) + 0.10 * span + 0.8
        region_data[region] = dict(oms=oms, osm=osm, holm_p=holm_p, rvals=rvals,
                                   dv_base=dv_base, fs_star=fs_star)

    # ---- shared y-range across the (separate) region figures ------------------
    _STAR_PAD, _BRK_TICK, _TXT_PAD = 0.6, 0.6, 1.2
    y_top = 0.0
    for region, d in region_data.items():
        tops = [om + os_ + _STAR_PAD + _TXT_PAD
                for om, os_ in zip(d["oms"], d["osm"]) if np.isfinite(om)]
        if np.isfinite(d["dv_base"]):
            tops.append(d["dv_base"] + _BRK_TICK + _TXT_PAD)
        if tops:
            y_top = max(y_top, max(tops))
    if y_top <= 0:
        y_top = 1.0

    # ---- pass 2: one figure per region, shared y-range ------------------------
    figs = []
    for region, rvals in regions:
        d = region_data[region]
        oms, osm = d["oms"], d["osm"]
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.bar(xs, oms, width=bar_width, color=[b["color"] for b in bars],
               alpha=0.85, edgecolor="k", linewidth=0.5, yerr=osm, capsize=3,
               zorder=2)
        if run_stats:
            for xi, om, os_, hp in zip(xs, oms, osm, d["holm_p"]):
                if np.isfinite(om):
                    ax.text(xi, om + os_ + _STAR_PAD, _sigstar(hp), ha="center",
                            va="bottom", fontsize=12)
            # DV fast-vs-slow paired bracket over the DV pair (always shown).
            if d["fs_star"] and np.isfinite(d["dv_base"]):
                _add_sig_bracket(ax, xs[-2], xs[-1], d["dv_base"], d["fs_star"],
                                 tick=_BRK_TICK, fontsize=11)
            handles = [Line2D([0], [0], marker="*", color="k", linestyle="None",
                              label="★ above bar: vs shuffle chance"),
                       Line2D([0], [0], color="k", lw=1,
                              label="⊓ bracket: DV fast vs slow")]
            ax.legend(handles=handles, fontsize="x-small", frameon=False,
                      loc="upper left", title="Holm-corrected")
        ax.set_xticks(xs)
        ax.set_xticklabels([b["label"] for b in bars], rotation=20, ha="right",
                           fontsize="small")
        ax.set_title(f"{region}  (|r| ≥ {min_abs_corr:g})")
        ax.set_ylabel("Modulated neurons (%)")
        ax.set_ylim(0, y_top)
        ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(title or f"Factor modulation (DV fast/slow) — "
                     f"{model_name or ''}", y=1.02)
        fig.tight_layout()
        figs.append((region, fig))

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "factor_bars")
        out_dir.mkdir(parents=True, exist_ok=True)
        for region, fig in figs:
            rtag = "combined" if not by_region else _safe_filename(region)
            fig.savefig(out_dir / f"bars_fastslowDV_{rtag}.{ext}",
                        bbox_inches="tight")
        stag = "by_region" if by_region else "combined"
        summary.to_csv(out_dir / f"summary_fastslowDV_{stag}.csv", index=False)
        print(f"Saved factor bars (DV fast/slow) -> {out_dir}")
    plt.show()
    return summary


# --------------------------------------------------------------------------
# 7. Fast vs slow: drift-correlated neurons (DV / DVabs)
# --------------------------------------------------------------------------
_FAST_COLOR = "tomato"     # lab convention (behavior.ipynb): fast = red-ish,
_SLOW_COLOR = "goldenrod"  # slow = yellow/gold.


def _rfmt(v):
    return f"{v:+.3f}" if v is not None and np.isfinite(v) else "nan"


def _pct_stats(corr, rcol, min_abs_corr):
    """(session-mean %, SEM, n_sessions) of drift-correlated neurons."""
    pcts = _session_tuned_percent(corr, rcol, min_abs_corr)
    mean = float(pcts.mean()) if len(pcts) else np.nan
    sem = float(pcts.sem()) if len(pcts) > 1 else 0.0
    return mean, sem, len(pcts)


def _bar_vs_chance(bundle, key, region_values, obs_mean, n_perm):
    """``+1/(n+1)`` permutation p of one bar vs its per-neuron activity-shuffle
    chance level, off a precomputed :func:`_shuffle_null_tuned` bundle."""
    meta_df, null, assess = bundle
    mask = (np.ones(len(meta_df), dtype=bool) if region_values is None
            else np.isin(meta_df["BrainRegion"].to_numpy(), region_values))
    null_dist = _null_region_distribution(meta_df, null[key], assess[key], mask)
    if null_dist is None or not np.isfinite(obs_mean):
        return np.nan
    return float((np.sum(null_dist >= obs_mean) + 1) / (n_perm + 1))


def plot_fast_slow_bars(corr_fast, corr_slow, param, *, table=None,
                        min_abs_corr=0.3, n_perm=1000, n_boot=10000,
                        by_region=True, seed=0, run_stats=True, save=False,
                        save_root=None, model_name=None, ext="svg"):
    """Fast-vs-slow % of drift-correlated neurons, with permutation significance.

    For ``param`` (``"DV"`` / ``"DVabs"``), a neuron is drift-correlated when
    ``|r| >= min_abs_corr`` (Pearson, within its fast/slow subset). Each bar is
    the session-mean ± SEM of that %; fast = tomato, slow = goldenrod.
    ``by_region=True`` → a fast|slow pair per MFC/LFC (**4 bars**);
    ``by_region=False`` → all regions pooled (**2 bars**).

    When ``run_stats`` three permutation/bootstrap layers are annotated:
      1. **vs chance** — a ``*/**/***`` star above each bar (per-neuron activity
         shuffle, ``n_perm`` permutations; needs the trial-level ``table`` so the
         fast/slow subsets can be shuffled — skipped with a warning if omitted);
      2. **fast vs slow** — a session-paired sign-flip permutation
         (:func:`_paired_perm_fastslow`) drawn as a bracket over each region's
         pair;
      3. **MFC vs LFC** within each speed — a region→session→neuron hierarchical
         bootstrap (:func:`_hier_bootstrap_region_diff`) drawn as spanning
         brackets (only when ``by_region`` with two regions).
    Each family is Holm-corrected across its comparisons. Returns a tidy summary
    (region × speed rows, plus cross-region rows); ``save`` also writes it.
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    for c in (corr_fast, corr_slow):
        if rcol not in c.columns:
            raise KeyError(f"{param!r} not in the correlation dataframe.")

    if by_region:
        regions = sorted(set(corr_fast.BrainRegion) | set(corr_slow.BrainRegion))
    else:
        regions = ["MFC & LFC"]

    rng = np.random.default_rng(seed)
    fast_bundle = slow_bundle = None
    if run_stats:
        if table is not None:
            fast_table, slow_table = split_fast_slow(table)
            fast_bundle = _shuffle_null_tuned(fast_table, [spec.key],
                                              min_abs_corr, n_perm, rng)
            slow_bundle = _shuffle_null_tuned(slow_table, [spec.key],
                                              min_abs_corr, n_perm, rng)
        else:
            warnings.warn("plot_fast_slow_bars: `table` not supplied — skipping "
                          "the per-bar vs-chance test (paired + cross-region "
                          "tests still run).")

    x = np.arange(len(regions))
    w = 0.36
    fig, ax = plt.subplots(figsize=(1.9 * len(regions) + 2.5, 4.4))
    rows, per_region, bar_tops = [], [], []
    for i, region in enumerate(regions):
        rvals = None if not by_region else [region]
        cf = corr_fast if rvals is None else corr_fast[corr_fast.BrainRegion == region]
        cs = corr_slow if rvals is None else corr_slow[corr_slow.BrainRegion == region]
        fmean, fsem, fn = _pct_stats(cf, rcol, min_abs_corr)
        smean, ssem, sn = _pct_stats(cs, rcol, min_abs_corr)
        ax.bar(x[i] - w / 2, fmean, width=w, yerr=fsem, capsize=3,
               color=_FAST_COLOR, edgecolor="k", linewidth=0.5)
        ax.bar(x[i] + w / 2, smean, width=w, yerr=ssem, capsize=3,
               color=_SLOW_COLOR, edgecolor="k", linewidth=0.5)

        fp = sp = pair_diff = pair_p = np.nan
        if run_stats:
            if fast_bundle is not None:
                fp = _bar_vs_chance(fast_bundle, spec.key, rvals, fmean, n_perm)
                sp = _bar_vs_chance(slow_bundle, spec.key, rvals, smean, n_perm)
            pair_diff, pair_p = _paired_perm_fastslow(
                corr_fast, corr_slow, rcol, min_abs_corr, n_perm, rng,
                region_values=rvals)
        per_region.append(dict(region=region, i=i, fmean=fmean, fsem=fsem,
                               smean=smean, ssem=ssem, fp=fp, sp=sp,
                               pair_p=pair_p))
        rows += [{"BrainRegion": region, "speed": "fast", "pct": fmean,
                  "sem": fsem, "n_sessions": fn, "p_vs_chance": fp,
                  "fast_minus_slow": pair_diff, "p_fast_vs_slow": pair_p},
                 {"BrainRegion": region, "speed": "slow", "pct": smean,
                  "sem": ssem, "n_sessions": sn, "p_vs_chance": sp,
                  "fast_minus_slow": pair_diff, "p_fast_vs_slow": pair_p}]
        bar_tops += [fmean + fsem, smean + ssem]
        print(f"\t{region}: fast {fmean:.2f}±{fsem:.2f}% ({fn} sess) | "
              f"slow {smean:.2f}±{ssem:.2f}% ({sn} sess)")

    ymax = max([t for t in bar_tops if np.isfinite(t)] + [1.0])
    step = 0.10 * ymax + 1.0

    if run_stats:
        n = len(per_region)
        # 1. vs-chance stars above each bar (Holm across all bars).
        vc_holm = _holm_adjust([pr["fp"] for pr in per_region]
                               + [pr["sp"] for pr in per_region])
        for k, pr in enumerate(per_region):
            if np.isfinite(pr["fmean"]):
                ax.text(pr["i"] - w / 2, pr["fmean"] + pr["fsem"] + 0.2 * step,
                        _sigstar(vc_holm[k]), ha="center", va="bottom",
                        fontsize=11)
            if np.isfinite(pr["smean"]):
                ax.text(pr["i"] + w / 2, pr["smean"] + pr["ssem"] + 0.2 * step,
                        _sigstar(vc_holm[n + k]), ha="center", va="bottom",
                        fontsize=11)
        # 2. fast-vs-slow paired bracket per region (Holm across regions). Drawn
        # even when n.s. so the comparison is always visible.
        for pr, ph in zip(per_region,
                          _holm_adjust([pr["pair_p"] for pr in per_region])):
            s = _sigstar(ph)
            if s:
                y = max(pr["fmean"] + pr["fsem"], pr["smean"] + pr["ssem"]) \
                    + 0.5 * step
                _add_sig_bracket(ax, pr["i"] - w / 2, pr["i"] + w / 2, y, s)
        # 3. MFC-vs-LFC per speed (hierarchical bootstrap, Holm across speeds).
        if by_region and len(regions) == 2:
            ra, rb = regions[0], regions[1]
            speeds = [("fast", corr_fast, -w / 2), ("slow", corr_slow, w / 2)]
            res_list = []
            for _, cc, _off in speeds:
                ma = _session_absr_mats(cc[cc.BrainRegion == ra], [spec.key])
                mb = _session_absr_mats(cc[cc.BrainRegion == rb], [spec.key])
                res_list.append(_hier_bootstrap_region_diff(
                    ma, mb, min_abs_corr, n_boot, rng))
            cross_holm = _holm_adjust([r["p"] for r in res_list])
            for (speed, _cc, off), res, ph in zip(speeds, res_list, cross_holm):
                rows.append({"BrainRegion": f"{ra} vs {rb}", "speed": speed,
                             "diff_region": res["diff"], "p_cross_region": res["p"],
                             "p_cross_region_holm": (float(ph)
                                                     if np.isfinite(ph) else np.nan)})
                s = _sigstar(ph)
                if s:
                    y = ymax + (1.4 if speed == "fast" else 2.6) * step
                    _add_sig_bracket(ax, 0 + off, 1 + off, y, s)
        ax.set_ylim(top=ymax + 4.0 * step)

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel(f"{spec.label}-correlated neurons (%)")
    ax.set_title(f"{spec.label} drift tuning, fast vs slow (|r| ≥ {min_abs_corr:g})"
                 f"\n{model_name or ''}", fontsize="medium")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Patch(facecolor=_FAST_COLOR, edgecolor="k", label="fast"),
                       Patch(facecolor=_SLOW_COLOR, edgecolor="k", label="slow")],
              frameon=False, fontsize="small")
    fig.tight_layout()

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "fast_slow_bars" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "by_region" if by_region else "combined"
        fig.savefig(out_dir / f"bars_{tag}.{ext}", bbox_inches="tight")
        summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(f"Saved fast/slow bars -> {out_dir}")
    plt.show()
    return summary


def _fastslow_ranking(corr_fast, corr_slow, rcol):
    """Neuron table (long_trace_id, BrainRegion, r_fast, r_slow, max_abs) ranked
    by ``max(|r_fast|, |r_slow|)`` descending; neurons undefined in both drop."""
    meta = pd.concat([corr_fast[["long_trace_id", "BrainRegion"]],
                      corr_slow[["long_trace_id", "BrainRegion"]]]) \
             .drop_duplicates("long_trace_id").set_index("long_trace_id")
    rank = meta.copy()
    rank["r_fast"] = corr_fast.set_index("long_trace_id")[rcol]
    rank["r_slow"] = corr_slow.set_index("long_trace_id")[rcol]
    rank["max_abs"] = rank[["r_fast", "r_slow"]].abs().max(axis=1)
    return rank.dropna(subset=["max_abs"]).sort_values("max_abs", ascending=False)


def _plot_one_neuron_overlay(ax, neuron_fast, neuron_slow, spec):
    """Scatter a neuron's fast (tomato) & slow (goldenrod) trials with per-speed
    least-squares fit lines; legend carries each speed's r / p."""
    for ndf, color, name in [(neuron_fast, _FAST_COLOR, "fast"),
                             (neuron_slow, _SLOW_COLOR, "slow")]:
        if ndf is None or len(ndf) == 0:
            continue
        x = ndf[spec.key].to_numpy(dtype=float)
        y = ndf["max_activity"].to_numpy(dtype=float)
        ax.scatter(x, y, s=14, color=color, alpha=0.5, edgecolors="none")
        slope, intercept, r, p = _linfit(x, y)
        if np.isfinite(slope):
            xs = np.array([np.nanmin(x), np.nanmax(x)])
            ax.plot(xs, slope * xs + intercept, color=color, lw=2,
                    label=f"{name}: r={r:+.3f}, {_p_phrase(p)}")
        else:
            ax.plot([], [], color=color, label=f"{name}: r=n/a")
    ax.set_xlabel(spec.label)
    ax.set_ylabel(_ACTIVITY_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize="small", frameon=False)


def plot_neuron_fastslow_scatter(table, corr_fast, corr_slow, param, *,
                                 mode="display", top_x=20, min_abs_corr=None,
                                 display_figsize=(6, 5), display_dpi=110,
                                 save_figsize=(9, 7), save_dpi=300,
                                 save_root=None, model_name=None, ext="svg"):
    """Per-neuron activity-vs-drift scatter, fast & slow overlaid.

    Neurons ranked by ``max(|r_fast|, |r_slow|)``. ``mode="display"`` shows the
    top-``top_x``; ``mode="save"`` writes each neuron under
    ``{save_root}/{model_name}/drift_scatter/{param}/`` and, when ``min_abs_corr``
    is set, only the subset with ``max(|r_fast|,|r_slow|) >= min_abs_corr``.
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    rank = _fastslow_ranking(corr_fast, corr_slow, rcol)
    tf, ts = split_fast_slow(table)
    by_fast = dict(tuple(tf.groupby("long_trace_id")))
    by_slow = dict(tuple(ts.groupby("long_trace_id")))

    def _draw(fig, ax, long_id, row):
        _plot_one_neuron_overlay(ax, by_fast.get(long_id), by_slow.get(long_id),
                                 spec)
        ax.set_title(f"{long_id}\n{row.BrainRegion} · {spec.label}",
                     fontsize="small")

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            rank = rank[rank["max_abs"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "drift_scatter" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for long_id, row in rank.iterrows():
            fig, ax = plt.subplots(figsize=save_figsize, dpi=save_dpi)
            _draw(fig, ax, long_id, row)
            fname = f"f{_rfmt(row.r_fast)}_s{_rfmt(row.r_slow)}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, row.BrainRegion):
                fname += f"_{row.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight",
                        dpi=save_dpi)
            plt.close(fig)
        print(f"Saved {len(rank)} fast/slow scatters -> {out_dir}")
        return out_dir

    for long_id, row in rank.head(top_x).iterrows():
        fig, ax = plt.subplots(figsize=display_figsize, dpi=display_dpi)
        _draw(fig, ax, long_id, row)
        plt.show()
    return rank.head(top_x)


def plot_neuron_fastslow_traces(table, corr_fast, corr_slow, param, bin_edges, *,
                                mode="display", top_x=20, min_abs_corr=None,
                                display_figsize=(11, 4), display_dpi=110,
                                save_figsize=(16, 6), save_dpi=300,
                                save_root=None, model_name=None, ext="svg"):
    """Per-neuron normalized trace averages by value range, in two panels
    (Fast | Slow). Each panel is the gradient-by-range mean ± SEM style with
    sampling/movement epoch lines. Ranking / display / save behave like
    :func:`plot_neuron_fastslow_scatter`; saved under
    ``{save_root}/{model_name}/drift_traces/{param}/``."""
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    value_ranges = _edges_to_ranges(bin_edges)
    rank = _fastslow_ranking(corr_fast, corr_slow, rcol)
    tf, ts = split_fast_slow(table)
    by_fast = dict(tuple(tf.groupby("long_trace_id")))
    by_slow = dict(tuple(ts.groupby("long_trace_id")))

    def _panel(ax, ndf, r, label):
        if ndf is None or len(ndf) == 0:
            ax.text(0.5, 0.5, f"no {label.lower()} trials", ha="center",
                    va="center", transform=ax.transAxes, color="0.5")
            ax.axis("off")
            return
        _plot_one_neuron_traces(ax, ndf, spec, value_ranges, r,
                                panel_label=label)

    def _draw(fig, axs, long_id, row):
        _panel(axs[0], by_fast.get(long_id), row.r_fast, "Fast")
        _panel(axs[1], by_slow.get(long_id), row.r_slow, "Slow")
        fig.suptitle(f"{long_id}  ·  {row.BrainRegion}  ·  {spec.label}",
                     fontsize="medium")

    if mode == "save":
        if save_root is None or model_name is None:
            raise ValueError("save mode needs save_root and model_name")
        if min_abs_corr is not None:
            rank = rank[rank["max_abs"] >= min_abs_corr]
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "drift_traces" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        for long_id, row in rank.iterrows():
            fig, axs = plt.subplots(1, 2, figsize=save_figsize, dpi=save_dpi,
                                    sharey=True)
            _draw(fig, axs, long_id, row)
            fname = f"f{_rfmt(row.r_fast)}_s{_rfmt(row.r_slow)}_{_safe_filename(long_id)}"
            if not _region_in_name(long_id, row.BrainRegion):
                fname += f"_{row.BrainRegion}"
            fig.savefig(out_dir / f"{fname}.{ext}", bbox_inches="tight",
                        dpi=save_dpi)
            plt.close(fig)
        print(f"Saved {len(rank)} fast/slow trace figures -> {out_dir}")
        return out_dir

    for long_id, row in rank.head(top_x).iterrows():
        fig, axs = plt.subplots(1, 2, figsize=display_figsize, dpi=display_dpi,
                                sharey=True)
        _draw(fig, axs, long_id, row)
        plt.show()
    return rank.head(top_x)


# --------------------------------------------------------------------------
# 8. Reward-rate levels: correlated neurons across binned reward rate
#
# A variation of the fast-vs-slow section above. There the trials are split by
# RT tercile (fast / slow strategy); here they are split by the **model's
# reward-rate latent** into value bins, and we ask the same question — what % of
# neurons are correlated with the drift (DV / |DV|), or with any other latent —
# at each reward-rate level.
#
# The bins are caller-defined (``np.arange`` edges or explicit ``(lo, hi)``
# tuples) precisely because reward rates are not uniformly common: the notebook
# can try a coarse binary low/high split, equal-width terciles, or two extreme
# bands with the crowded middle dropped, without touching this code.
#
# The across-bin test is a **session-paired** one, matching how the bars are
# built (per-session %, then the mean across sessions): a paired t-test for two
# bins, a repeated-measures ANOVA (+ Holm-corrected paired post-hocs) for more.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ValueBin:
    """One interval of a trial-level value (e.g. the reward-rate latent).

    Membership is ``[lo, hi)``, except the **last** bin of a set which is
    ``[lo, hi]`` so the maximum observed value is included — the same convention
    as the per-neuron trace overlays (:func:`_range_mask`).
    """
    lo: float
    hi: float
    closed_right: bool = False

    @property
    def label(self):
        return f"[{self.lo:g}, {self.hi:g}{']' if self.closed_right else ')'}"


def make_value_bins(bin_spec):
    """Normalize a bin specification into a list of :class:`ValueBin`.

    Two accepted forms — pick whichever reads better for the split at hand:

    - **edges** (a flat, increasing sequence; e.g. ``np.arange(0, 1.01, 0.25)``
      or ``[0, 0.5, 1.0]``): ``N`` edges → ``N-1`` *contiguous* bins;
    - **explicit ``(lo, hi)`` tuples** (e.g. ``[(0, 0.3), (0.7, 1.0)]``): the
      bins are taken as given, so a crowded middle band can be left out entirely
      — the reward-rate analogue of dropping the middle RT tercile in the
      fast/slow split.

    Only the final bin is closed on the right; values outside every bin (and
    NaNs) simply take part in no bin.
    """
    items = list(bin_spec)
    if not items:
        raise ValueError("bin_spec is empty; give bin edges or (lo, hi) tuples.")
    if np.ndim(items[0]) == 1:                       # explicit (lo, hi) tuples
        ranges = []
        for item in items:
            pair = list(item)
            if len(pair) != 2:
                raise ValueError(f"bin {item!r} is not a (lo, hi) pair.")
            ranges.append((float(pair[0]), float(pair[1])))
    else:                                            # flat edge sequence
        edges = [float(v) for v in items]
        if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
            raise ValueError(f"bin edges must strictly increase, got {edges}.")
        ranges = _edges_to_ranges(edges)
    for lo, hi in ranges:
        if not (hi > lo):
            raise ValueError(f"bin ({lo}, {hi}) is empty; needs hi > lo.")
    ordered = sorted(ranges)
    for (_, hi_a), (lo_b, _) in zip(ordered, ordered[1:]):
        if lo_b < hi_a:
            raise ValueError(f"bins overlap ({ordered}); a trial would be "
                             "counted twice.")
    return [ValueBin(lo, hi, closed_right=(i == len(ranges) - 1))
            for i, (lo, hi) in enumerate(ranges)]


@dataclass(frozen=True)
class QuantileBin:
    """One **within-session** quantile level of a trial value.

    Unlike :class:`ValueBin` this has no fixed cut point: every session is split
    at its *own* quantiles, so each session contributes the same number of trials
    to every level. ``lo``/``hi`` are the pooled min/max of the values that
    landed here across sessions, for reporting only — they overlap between
    neighbouring levels and must not be used to re-derive membership.
    """
    index: int          # 1-based, matching the lab's ``quantile_idx``
    n_bins: int
    lo: float
    hi: float

    @property
    def label(self):
        tag = ""
        if self.index == 1:
            tag = " (low)"
        elif self.index == self.n_bins:
            tag = " (high)"
        return f"Q{self.index}/{self.n_bins}{tag}"


def _session_quantile_index(table, column, n_quantiles, group_col):
    """Per-row 1-based within-session quantile index (0 = unassigned).

    Mirrors ``behavior/util/splitdata.py::_processDf``, the split already used
    for the RT terciles: unique trials are **sorted** and cut into equal-count
    parts rather than binned by value, because value bins on a tied or skewed
    distribution do not produce equal parts. The remainder is handed out
    round-robin, and the starting offset carries across sessions so the leftovers
    do not always land in the same level.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")
    if column not in table.columns:
        raise KeyError(f"{column!r} is not a column of the neuron-trial table.")
    per_trial = table.drop_duplicates(subset=[group_col, "TrialNumber"])
    q_of = {}
    skipped, rem_idx = [], 0
    for sess in sorted(per_trial[group_col].astype(str).unique()):
        g = per_trial[per_trial[group_col].astype(str) == sess]
        vals = g[column].to_numpy(dtype=float)
        trials = g["TrialNumber"].to_numpy()
        ok = np.isfinite(vals)
        vals, trials = vals[ok], trials[ok]
        if vals.size < n_quantiles:
            skipped.append(f"{sess} ({vals.size} trials)")
            continue
        order = np.argsort(vals, kind="stable")
        sizes = [vals.size // n_quantiles] * n_quantiles
        for _ in range(vals.size % n_quantiles):
            sizes[rem_idx] += 1
            rem_idx = (rem_idx + 1) % n_quantiles
        start = 0
        for q, size in enumerate(sizes, start=1):
            for pos in order[start:start + size]:
                q_of[(sess, trials[pos])] = q
            start += size
    if skipped:
        warnings.warn(
            f"session quantile split: {len(skipped)} session(s) have fewer than "
            f"{n_quantiles} usable trials and are unassigned ({', '.join(skipped)}).")

    key = list(zip(table[group_col].astype(str).to_numpy(),
                   table["TrialNumber"].to_numpy()))
    idx = np.fromiter((q_of.get(k, 0) for k in key), dtype=int, count=len(key))

    vals = table[column].to_numpy(dtype=float)
    bins = []
    for q in range(1, n_quantiles + 1):
        sel = vals[idx == q]
        sel = sel[np.isfinite(sel)]
        bins.append(QuantileBin(index=q, n_bins=n_quantiles,
                                lo=float(sel.min()) if sel.size else np.nan,
                                hi=float(sel.max()) if sel.size else np.nan))
    return bins, idx


@dataclass(frozen=True)
class QuantileRangeBin:
    """One **within-session** band of a trial value, delimited by *quantiles*.

    The quantile-space extremes split: ``(0, 0.25)`` means "this session's
    lowest quarter of trials", so the band tracks each session's own
    distribution instead of a shared cut point that can drift into the crowded
    part of one session and off the end of another. ``lo``/``hi`` are the pooled
    observed min/max of the values that landed here, for reporting only — they
    differ per session and must not be used to re-derive membership.
    """
    index: int          # 1-based, ordered by q_lo
    n_bins: int
    q_lo: float
    q_hi: float
    lo: float
    hi: float

    @property
    def label(self):
        tag = ""
        if self.n_bins > 1 and self.index == 1:
            tag = " (low)"
        elif self.n_bins > 1 and self.index == self.n_bins:
            tag = " (high)"
        return f"{self.q_lo * 100:g}-{self.q_hi * 100:g}%{tag}"


def make_quantile_ranges(bin_spec):
    """Normalize ``(q_lo, q_hi)`` quantile bands, sorted and validated.

    Bands are **fractions** in ``[0, 1]`` (``0.25``, not ``25``) and must leave a
    gap between them — see :func:`_session_quantile_range_index` for why touching
    bands are refused rather than silently splitting a tied value.
    """
    items = list(bin_spec)
    if not items:
        raise ValueError("bin_spec is empty; give (q_lo, q_hi) quantile bands.")
    ranges = []
    for item in items:
        pair = list(item) if np.ndim(item) == 1 else [item]
        if len(pair) != 2:
            raise ValueError(f"quantile band {item!r} is not a (q_lo, q_hi) "
                             "pair.")
        q_lo, q_hi = float(pair[0]), float(pair[1])
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError(f"quantile band ({q_lo}, {q_hi}) must satisfy "
                             "0 <= q_lo < q_hi <= 1 — these are FRACTIONS, so "
                             "the top quarter is (0.75, 1.0), not (75, 100).")
        ranges.append((q_lo, q_hi))
    ranges.sort()
    for (_, hi_a), (lo_b, _) in zip(ranges, ranges[1:]):
        if lo_b <= hi_a:
            raise ValueError(
                f"quantile bands {ranges} touch or overlap. This split closes "
                "both ends so a tied value is never cut in half, so touching "
                "bands would put the shared value in both. Leave a gap between "
                "them (the point of an extremes split), or use "
                "by='session_quantile' for contiguous equal-count levels.")
    return ranges


def _session_quantile_range_index(table, column, bin_spec, group_col):
    """Per-row 1-based band index (0 = unassigned) for per-session quantile bands.

    Each session's own quantiles set that band's value thresholds, and membership
    is **closed at both ends**: every trial tied with a threshold joins the band.
    That is the deliberate difference from :func:`_session_quantile_index`, which
    cuts the sorted trials into exactly equal parts and therefore has to send
    some of a run of tied values one way and the rest the other. Here a tie is
    kept whole and the bands come out unequal in size instead — the right trade
    when the value is a fitted latent that plateaus (a reward rate sitting at the
    same level for long stretches), because splitting a plateau by rank puts
    trials with *identical* reward rates in "low" and "high".

    Ties can make two bands claim the same trial — a session whose values barely
    move has its bottom and top band land on one value. Such trials are left
    unassigned (with a warning naming the sessions) rather than double-counted;
    those sessions then drop out of the paired test through the usual coverage
    reporting.
    """
    ranges = make_quantile_ranges(bin_spec)
    per_trial = table.drop_duplicates(subset=[group_col, "TrialNumber"])
    band_of, ambiguous, skipped = {}, {}, []
    for sess in sorted(per_trial[group_col].astype(str).unique()):
        g = per_trial[per_trial[group_col].astype(str) == sess]
        vals = g[column].to_numpy(dtype=float)
        trials = g["TrialNumber"].to_numpy()
        ok = np.isfinite(vals)
        vals, trials = vals[ok], trials[ok]
        if vals.size < 2:
            skipped.append(f"{sess} ({vals.size} trials)")
            continue
        masks = np.vstack([
            # Closed on BOTH ends, so a value tied with the threshold is never
            # split across the band boundary.
            (vals >= v_lo) & (vals <= v_hi)
            for v_lo, v_hi in (np.quantile(vals, [q_lo, q_hi])
                               for q_lo, q_hi in ranges)])
        clash = masks.sum(axis=0) > 1
        if clash.any():
            ambiguous[sess] = int(clash.sum())
            masks[:, clash] = False
        for i, m in enumerate(masks, start=1):
            for t in trials[m]:
                band_of[(sess, t)] = i
    if skipped:
        warnings.warn(
            f"session quantile-range split: {len(skipped)} session(s) have too "
            f"few usable trials and are unassigned ({', '.join(skipped)}).")
    if ambiguous:
        warnings.warn(
            f"session quantile-range split: {len(ambiguous)} session(s) have "
            "values so tied that a trial fell in more than one band; those "
            "trials are unassigned ("
            + ", ".join(f"{s} ({n} trials)" for s, n in ambiguous.items())
            + "). Those sessions have too little spread in "
            f"{column!r} to contribute to this comparison.")

    key = list(zip(table[group_col].astype(str).to_numpy(),
                   table["TrialNumber"].to_numpy()))
    idx = np.fromiter((band_of.get(k, 0) for k in key), dtype=int, count=len(key))

    vals = table[column].to_numpy(dtype=float)
    bins = []
    for i, (q_lo, q_hi) in enumerate(ranges, start=1):
        sel = vals[idx == i]
        sel = sel[np.isfinite(sel)]
        bins.append(QuantileRangeBin(
            index=i, n_bins=len(ranges), q_lo=q_lo, q_hi=q_hi,
            lo=float(sel.min()) if sel.size else np.nan,
            hi=float(sel.max()) if sel.size else np.nan))
    return bins, idx


def _trial_bin_index(table, column, bin_spec, *, by="value",
                     group_col="ShortName"):
    """``(bins, per-row 1-based bin index)`` for any splitting criterion.
    Index 0 means the row belongs to no bin. The single place the criteria are
    dispatched, so every downstream report handles all of them."""
    if by == "session_quantile":
        return _session_quantile_index(table, column, int(bin_spec), group_col)
    if by == "session_quantile_range":
        return _session_quantile_range_index(table, column, bin_spec, group_col)
    if by != "value":
        raise ValueError(f"unknown split criterion {by!r}; use 'value', "
                         "'session_quantile' or 'session_quantile_range'.")
    bins = make_value_bins(bin_spec)
    vals = table[column].to_numpy(dtype=float)
    idx = np.zeros(len(table), dtype=int)
    for i, b in enumerate(bins, start=1):
        idx[_range_mask(vals, b.lo, b.hi, b.closed_right)] = i
    return bins, idx


def split_trials(table, column, bin_spec, *, by="value", group_col="ShortName"):
    """``(bins, bin_tables)`` under either splitting criterion.

    ``by="value"`` — ``bin_spec`` is edges or ``(lo, hi)`` tuples; the cut points
    are the same for every session (see :func:`make_value_bins`).

    ``by="session_quantile"`` — ``bin_spec`` is an **integer** number of
    quantiles, and each session is split at its own quantiles into equal-count
    levels (see :func:`_session_quantile_index`). Use this when the value's
    distribution differs between sessions — as a fitted latent's does — because
    it is the only split that guarantees every session contributes a comparable
    number of trials to every level.

    ``by="session_quantile_range"`` — ``bin_spec`` is a list of ``(q_lo, q_hi)``
    quantile bands with gaps between them, e.g. ``[(0, 0.25), (0.75, 1)]`` for
    the extremes with the crowded middle dropped (see
    :func:`_session_quantile_range_index`). Like ``session_quantile`` the cut
    points are per session, but a run of tied values is kept whole rather than
    cut to make the levels exactly equal — so the levels end up unequal in size.
    """
    bins, idx = _trial_bin_index(table, column, bin_spec, by=by,
                                 group_col=group_col)
    return bins, [table[idx == i] for i in range(1, len(bins) + 1)]


def split_by_value_bins(table, column, bins):
    """Split the neuron-trial ``table`` into one sub-table per :class:`ValueBin`
    of ``table[column]`` (e.g. ``"RewardRate"``). Trials outside every bin are
    dropped — the direct analogue of :func:`split_fast_slow`, which drops the
    middle RT tercile."""
    if column not in table.columns:
        raise KeyError(f"{column!r} is not a column of the neuron-trial table "
                       f"(has: {list(table.columns)}). A reward-rate split needs "
                       "a fit that learns RewardRate.")
    vals = table[column].to_numpy(dtype=float)
    return [table[_range_mask(vals, b.lo, b.hi, b.closed_right)] for b in bins]


def quantile_bin_edges(table, column, n_bins, *, decimals=3):
    """Bin edges at the empirical quantiles of ``table[column]`` — ``n_bins``
    bins holding (approximately) the same number of **trials** each.

    The counterpart to hand-picked edges: reward rates are far from uniformly
    common, so equal-width bins can leave one bin nearly empty. Equal-*occupancy*
    bins trade the round numbers for balance. Rounded to ``decimals`` so the bin
    labels stay readable; returns a plain list, ready for
    :func:`reward_rate_correlations`.
    """
    if column not in table.columns:
        raise KeyError(f"{column!r} is not a column of the neuron-trial table.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    # One value per trial, not per (neuron, trial) row, so a session with many
    # neurons does not dominate the quantiles.
    per_trial = table.drop_duplicates(subset=["ShortName", "TrialNumber"])
    vals = per_trial[column].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"no finite {column!r} values to take quantiles of.")
    edges = np.round(np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1)),
                     decimals)
    edges[0] = min(edges[0], np.floor(vals.min() * 10 ** decimals)
                   / 10 ** decimals)
    edges[-1] = max(edges[-1], vals.max())
    if len(np.unique(edges)) != len(edges):
        raise ValueError(
            f"{column!r} is too concentrated for {n_bins} equal-occupancy bins "
            f"(duplicate edges in {list(edges)}); use fewer bins or explicit "
            "edges.")
    return [float(e) for e in edges]


def plot_session_value_hist(table, column="RewardRate", bins=None, *,
                            n_quantiles=None, value_bins=None,
                            quantile_ranges=None,
                            group_col="ShortName", ncols=4,
                            panel_size=(2.6, 1.9), sharey=False, save=False,
                            save_root=None, model_name=None, ext="svg"):
    """One histogram of ``column`` per session — how differently the sessions are
    distributed, at a glance.

    This is the picture behind the choice of splitting criterion: if the
    per-session histograms sit on top of each other, fixed cut points are fine;
    if they are shifted relative to one another (as fitted latents usually are),
    a shared cut point gives each session a different number of trials per level
    and only a per-session quantile split is balanced.

    One trial contributes once (the table is per neuron × trial, so it is
    de-duplicated first). Panels are coloured by brain region, and either
    overlay can be drawn on top:

    ``n_quantiles`` — dashed lines at that session's **own** equal-count cut
    points, i.e. exactly where ``by="session_quantile"`` would split it. They
    move from panel to panel, which is the point.

    ``value_bins`` — a ``by="value"`` bin spec, drawn as shaded spans that are
    **identical in every panel**. A session whose mass falls outside a span
    contributes nothing to that level, so this shows at a glance which sessions
    a fixed split will lose.

    ``quantile_ranges`` — a ``by="session_quantile_range"`` bin spec, drawn as
    shaded spans **recomputed per panel** from that session's own quantiles. Put
    next to the same bands as fixed values, this is the picture of how far a
    fixed cut point drifts across sessions.

    Returns ``(fig, per_session_df)``.
    """
    if column not in table.columns:
        raise KeyError(f"{column!r} is not a column of the neuron-trial table.")
    bins = np.arange(0, 1.1, 0.1) if bins is None else np.asarray(bins)
    per_trial = table.drop_duplicates(subset=[group_col, "TrialNumber"])
    sessions = sorted(per_trial[group_col].astype(str).unique())
    if not sessions:
        raise ValueError("no sessions to plot.")

    region_of = (per_trial.groupby(group_col)["BrainRegion"].first()
                 if "BrainRegion" in per_trial.columns else {})
    palette = {"MFC": "purple", "LFC": "teal"}
    fixed = make_value_bins(value_bins) if value_bins is not None else []
    span_colors = _param_gradient(PARAM_BY_KEY["RewardRate"].color, len(fixed)) \
        if fixed else []
    qranges = (make_quantile_ranges(quantile_ranges)
               if quantile_ranges is not None else [])
    qr_colors = _param_gradient(PARAM_BY_KEY["RewardRate"].color, len(qranges)) \
        if qranges else []

    nrows = int(np.ceil(len(sessions) / ncols))
    fig, axs = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                            sharey=sharey,
                            figsize=(panel_size[0] * ncols,
                                     panel_size[1] * nrows))
    stats_rows = []
    for ax, sess in zip(axs.ravel(), sessions):
        vals = per_trial.loc[per_trial[group_col].astype(str) == sess, column]
        vals = vals.to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        region = str(region_of.get(sess, "")) if len(region_of) else ""
        # Fixed value bins first, so the histogram draws over them.
        for fb, fc in zip(fixed, span_colors):
            ax.axvspan(fb.lo, fb.hi, color=fc, alpha=0.22, linewidth=0, zorder=0)
        # Quantile bands, recomputed from THIS session's values — outlined
        # rather than filled so they stay legible over a fixed-bin span.
        qr_edges, n_in_qr = [], []
        for (q_lo, q_hi), qc in zip(qranges, qr_colors):
            if vals.size < 2:
                qr_edges.append((np.nan, np.nan))
                n_in_qr.append(0)
                continue
            v_lo, v_hi = (float(v) for v in np.quantile(vals, [q_lo, q_hi]))
            qr_edges.append((round(v_lo, 3), round(v_hi, 3)))
            # Closed both ends, matching _session_quantile_range_index.
            n_in_qr.append(int(((vals >= v_lo) & (vals <= v_hi)).sum()))
            ax.axvspan(v_lo, v_hi, facecolor="none", edgecolor=qc, lw=1.2,
                       ls="--", zorder=3)
        ax.hist(vals, bins=bins, color=palette.get(region, "0.4"), alpha=0.8,
                edgecolor="white", linewidth=0.4, zorder=2)
        n_in_fixed = [int(_range_mask(vals, fb.lo, fb.hi, fb.closed_right).sum())
                      for fb in fixed]
        cuts = []
        if n_quantiles and vals.size >= n_quantiles:
            # The session's OWN cut points — the same equal-count boundaries the
            # session-quantile split uses, so this shows where it would cut.
            order = np.sort(vals)
            step = vals.size / n_quantiles
            cuts = [float(order[int(round(k * step)) - 1])
                    for k in range(1, n_quantiles)]
            for c in cuts:
                ax.axvline(c, ls="--", color="k", lw=1.0, alpha=0.8)
        title = (f"{sess}\n{region} · n={vals.size}"
                 + (f" · median={np.median(vals):.2f}" if vals.size else ""))
        if fixed:
            # Trials this session would give each fixed level; "!" marks a level
            # it cannot contribute to at all.
            title += ("\nper level: "
                      + " / ".join(f"{c}{'!' if c < _MIN_CORR_POINTS else ''}"
                                   for c in n_in_fixed))
        if qranges:
            title += ("\nper band: "
                      + " / ".join(f"{c}{'!' if c < _MIN_CORR_POINTS else ''}"
                                   for c in n_in_qr))
        ax.set_title(title, fontsize="x-small")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize="x-small")
        stats_rows.append({
            group_col: sess, "BrainRegion": region, "n_trials": int(vals.size),
            # p50, not "median": a column named `median` shadows
            # DataFrame.median, so `df.median` silently returns the method.
            "p0": float(vals.min()) if vals.size else np.nan,
            "p25": float(np.percentile(vals, 25)) if vals.size else np.nan,
            "p50": float(np.median(vals)) if vals.size else np.nan,
            "p75": float(np.percentile(vals, 75)) if vals.size else np.nan,
            "p100": float(vals.max()) if vals.size else np.nan,
            "quantile_cuts": [round(c, 3) for c in cuts],
            "n_per_fixed_bin": n_in_fixed,
            "usable_in_all_fixed": (all(c >= _MIN_CORR_POINTS for c in n_in_fixed)
                                    if fixed else None),
            # Where this session's quantile bands actually fall in value space —
            # the spread of these across sessions is the drift a fixed cut has.
            "quantile_band_edges": qr_edges,
            "n_per_quantile_band": n_in_qr,
            "usable_in_all_bands": (all(c >= _MIN_CORR_POINTS for c in n_in_qr)
                                    if qranges else None)})
    for ax in axs.ravel()[len(sessions):]:
        ax.axis("off")
    for ax in axs[-1]:
        ax.set_xlabel(column, fontsize="x-small")
    for row in axs:
        row[0].set_ylabel("trials", fontsize="x-small")
    fig.suptitle(f"{column} distribution per session — {model_name or ''}",
                 y=1.005)
    fig.tight_layout()

    per_session = pd.DataFrame(stats_rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "reward_rate_bars")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"session_{_safe_filename(column)}_hist.{ext}",
                    bbox_inches="tight")
        per_session.to_csv(
            out_dir / f"session_{_safe_filename(column)}_stats.csv", index=False)
        print(f"Saved per-session {column} histograms -> {out_dir}")
    plt.show()
    return fig, per_session


# --------------------------------------------------------------------------
# 9. Paired per-neuron r between two conditions
#
# The bar charts above ask "what % of neurons clear a threshold" — a count that
# throws away the size of each correlation. These functions keep the r itself and
# compare it **within neuron** across two conditions (fast vs slow, or low vs
# high reward rate), which is the more sensitive question and needs no threshold
# at test time.
# --------------------------------------------------------------------------
def select_tuned_neurons(corr_frames, param, *, min_abs_corr=0.3, how="any"):
    """Neuron ids reaching ``|r| >= min_abs_corr`` in ``how`` of the frames.

    ``how="any"`` (default) is the union — a neuron counts if it is tuned in at
    least one condition; ``how="all"`` is the intersection. Returns a sorted list
    of ``long_trace_id``.
    """
    rcol = f"{PARAM_BY_KEY[param].key}_r"
    sets = []
    for c in corr_frames:
        if rcol not in c.columns:
            raise KeyError(f"{param!r} missing from a correlation frame "
                           f"(columns: {list(c.columns)}).")
        r = c[rcol].abs()
        sets.append(set(c.loc[r >= min_abs_corr, "long_trace_id"]))
    if not sets:
        return []
    picked = set.union(*sets) if how == "any" else set.intersection(*sets)
    return sorted(picked)


def paired_neuron_r(corr_a, corr_b, param, *, neurons=None):
    """Each neuron's ``r`` in two conditions, aligned on ``long_trace_id``.

    Only neurons whose ``r`` is **defined in both** conditions can be paired, so
    those are what comes back — with ``ShortName`` / ``BrainRegion`` carried
    along for the session-level and per-region views. ``neurons`` restricts to a
    pre-selected set (e.g. from :func:`select_tuned_neurons`); ids in that set
    that are not pairable are dropped and counted in ``frame.attrs``.
    """
    rcol = f"{PARAM_BY_KEY[param].key}_r"
    meta_cols = ["long_trace_id", "ShortName", "BrainRegion"]
    a = corr_a[meta_cols + [rcol]].rename(columns={rcol: "r_a"})
    b = corr_b[["long_trace_id", rcol]].rename(columns={rcol: "r_b"})
    out = a.merge(b, on="long_trace_id", how="inner")
    requested = None
    if neurons is not None:
        requested = set(neurons)
        out = out[out.long_trace_id.isin(requested)]
    n_before = len(out)
    out = out.dropna(subset=["r_a", "r_b"]).reset_index(drop=True)
    out.attrs["n_dropped_undefined"] = n_before - len(out)
    if requested is not None:
        out.attrs["n_requested"] = len(requested)
        out.attrs["n_missing"] = len(requested - set(out.long_trace_id))
    return out


#: The fast / slow bar colours used by the fast-vs-slow figures, so a paired
#: plot of those two conditions matches them. Pass as ``bar_colors``.
FAST_SLOW_COLORS = (_FAST_COLOR, _SLOW_COLOR)


def reward_rate_colors(n=2):
    """The reward-rate level shades (pale = low, dark = high) that
    :func:`plot_reward_rate_bars` uses, so a paired plot of two levels matches
    it. The endpoints are the same for any ``n >= 2``, so the default pair is
    the lowest and highest level whatever the split's resolution."""
    return _param_gradient(PARAM_BY_KEY["RewardRate"].color, n)


def _paired_neuron_panel(ax, a, b, sessions, labels, *, use_abs=True,
                         bar_colors=("0.85", "0.6"), line_color="0.6",
                         show_session_test=True):
    """One paired-condition panel: bars (mean ± SEM) with the per-neuron lines
    drawn over them, plus the paired-t bracket. Returns the panel's stats dict.

    Split out of :func:`plot_paired_neuron_r` so the pooled and per-region
    figures draw and test identically — only the neuron subset differs.
    """
    frame = pd.DataFrame({labels[0]: a, labels[1]: b})
    neuron_test = paired_bins_test(frame)
    session_test = None
    if show_session_test and sessions is not None:
        per_sess = (pd.DataFrame({"ShortName": np.asarray(sessions),
                                  labels[0]: a, labels[1]: b})
                    .groupby("ShortName").mean())
        session_test = paired_bins_test(per_sess)

    x = np.array([0.0, 1.0])
    means = [float(np.mean(a)), float(np.mean(b))]
    sems = [float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            float(np.std(b, ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0.0]
    # Bars underneath in the condition's own colour, per-neuron lines and points
    # over them in grey: colour carries the condition, grey the single neurons.
    ax.bar(x, means, width=0.62, color=list(bar_colors), alpha=0.85,
           edgecolor="k", lw=1.0, yerr=sems, capsize=5,
           error_kw={"ecolor": "k", "elinewidth": 1.2, "zorder": 4}, zorder=1)
    for aa, bb in zip(a, b):
        # "_neuron": leading underscore keeps it out of legends, and makes the
        # per-neuron lines separable from the error-bar caps.
        ax.plot(x, [aa, bb], color=line_color, lw=0.7, alpha=0.55, zorder=2,
                label="_neuron")
    ax.scatter(np.zeros_like(a), a, s=16, color="0.25", alpha=0.6,
               edgecolors="none", zorder=3)
    ax.scatter(np.ones_like(b), b, s=16, color="0.25", alpha=0.6,
               edgecolors="none", zorder=3)
    # The two means joined in black. No error bars on it — the bars carry the
    # SEM already, and doubling it would just clutter the tops.
    ax.plot(x, means, color="k", lw=2.0, marker="o", markersize=6, zorder=5,
            label="_mean")

    both = np.concatenate([a, b])
    lo = float(np.nanmin(both))
    # The bracket must clear the taller of the points and the error bars.
    hi = max(float(np.nanmax(both)), *(m + s for m, s in zip(means, sems)))
    span = (hi - min(lo, 0.0)) or 1.0
    y = hi + 0.06 * span
    top = _add_sig_bracket(ax, 0.0, 1.0, y, _sigstar(neuron_test["p"]),
                           tick=0.02 * span, fontsize=12)
    # The p-label goes ABOVE the star. Offsetting in points, not data units,
    # keeps the gap clear of the 12pt glyph whatever the y-range happens to be.
    ax.annotate(str(neuron_test["label"]), xy=(0.5, top), xytext=(0, 16),
                textcoords="offset points", ha="center", va="bottom",
                fontsize="x-small", color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    ax.set_xlim(-0.6, 1.6)
    # Bars are drawn from 0, so 0 has to be in view; signed r also needs
    # headroom under the lowest point.
    ax.set_ylim(0.0 if use_abs else min(0.0, lo - 0.08 * span),
                top + 0.20 * span)
    ax.spines[["top", "right"]].set_visible(False)
    return {"labels": tuple(labels), "use_abs": use_abs, "n_neurons": len(a),
            "n_sessions": (int(pd.Series(sessions).nunique())
                           if sessions is not None else np.nan),
            "mean_a": means[0], "mean_b": means[1], "sem_a": sems[0],
            "sem_b": sems[1], "neuron_test": neuron_test,
            "session_test": session_test}


def plot_paired_neuron_r(paired, labels, param, *, use_abs=True, by_region=False,
                         title=None, bar_colors=("0.85", "0.6"),
                         line_color="0.6", show_session_test=True,
                         figsize=(4.4, 4.8), save=False, save_root=None,
                         model_name=None, tag=None, ext="svg"):
    """Per-neuron ``r`` in two conditions: bars ± SEM with paired lines over them.

    Each condition is a bar (mean ± SEM across neurons) and each neuron is a grey
    line joining its own two values, so the group effect and the pairing behind
    it are both readable; a black line joins the two means. A **paired t-test**
    across neurons is annotated.

    ``bar_colors`` should be the condition's own colours so the figure matches
    the bar charts of the same split — :data:`FAST_SLOW_COLORS` for fast vs slow,
    :func:`reward_rate_colors` for two reward-rate levels. It defaults to grey,
    which says nothing about what is being compared.

    ``by_region=True`` draws one panel per brain region instead of pooling them,
    each with its own neurons and its own tests. Panels are **not** cross-region
    corrected — they are separate analyses of separate populations, not a family.

    ``use_abs`` (default) plots ``|r|`` — the correlation's **strength**. That is
    almost always what you want when the neuron set was chosen by ``|r|``: a
    population containing both positively and negatively tuned neurons averages
    to ~0 in signed r, so a signed test would compare two nulls. Set it False to
    keep the sign (meaningful only for a same-signed population).

    Two caveats this function surfaces rather than hides:

    - **Pseudo-replication.** The annotated t-test pairs *neurons*, but neurons
      within a session are not independent, so its ``p`` is anti-conservative.
      With ``show_session_test`` a second paired t-test over **session means** —
      the unit the rest of this module tests on — is computed and reported in the
      returned stats and printed; prefer it when the two disagree.
    - **Regression to the mean.** If the neurons were selected for having a large
      ``|r|`` in one of these very conditions, they are selected partly on noise
      and will drift toward the mean in the other, which manufactures a
      difference. Selecting on a *different* split than the one being compared
      (e.g. picking on fast/slow, then comparing reward-rate levels) avoids this.

    Returns ``(fig, stats)``. ``stats`` is the panel's stats dict (both tests
    plus the means) when pooling, or ``{region: stats}`` when ``by_region``.
    """
    spec = PARAM_BY_KEY[param]
    if len(paired) == 0:
        raise ValueError("no pairable neurons — nothing to plot.")
    r_a = paired["r_a"].to_numpy(dtype=float)
    r_b = paired["r_b"].to_numpy(dtype=float)
    if use_abs:
        r_a, r_b = np.abs(r_a), np.abs(r_b)
    ylab = f"|r|  ({spec.label})" if use_abs else f"r  ({spec.label})"

    if by_region:
        panels = [(region, (paired.BrainRegion == region).to_numpy())
                  for region in sorted(pd.unique(paired.BrainRegion))]
    else:
        panels = [(None, np.ones(len(paired), dtype=bool))]

    fig, axs = plt.subplots(1, len(panels),
                            figsize=(figsize[0] * len(panels), figsize[1]),
                            squeeze=False)
    per_panel = {}
    for ax, (region, mask) in zip(axs[0], panels):
        st = _paired_neuron_panel(
            ax, r_a[mask], r_b[mask], paired.ShortName.to_numpy()[mask], labels,
            use_abs=use_abs, bar_colors=bar_colors, line_color=line_color,
            show_session_test=show_session_test)
        st["region"] = region or "MFC & LFC"
        per_panel[st["region"]] = st
        ax.set_ylabel(ylab)
        ax.set_title(f"{st['region']}  (n={st['n_neurons']})"
                     if by_region else
                     (title or f"{spec.label} correlation per neuron "
                               f"(n={st['n_neurons']})"),
                     fontsize="medium")
    if by_region and title:
        fig.suptitle(title, fontsize="medium")
    fig.tight_layout()

    for st in per_panel.values():
        head = f"[{st['region']}] " if by_region else "\t"
        print(f"{head}{labels[0]} {st['mean_a']:.3f}±{st['sem_a']:.3f}  vs  "
              f"{labels[1]} {st['mean_b']:.3f}±{st['sem_b']:.3f}   "
              f"(n={st['n_neurons']} neurons)")
        print(f"\t  across neurons: {st['neuron_test']['label']}")
        if st["session_test"] is not None:
            print(f"\t  across sessions (guards against pseudo-replication): "
                  f"{st['session_test']['label']}")

    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "paired_neuron_r")
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_filename(tag or f"{spec.key}_{labels[0]}_vs_{labels[1]}")
        if by_region:
            stem = f"{stem}_by_region"
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
        rows = []
        for st in per_panel.values():
            row = {"param": spec.key, "BrainRegion": st["region"],
                   "cond_a": labels[0], "cond_b": labels[1],
                   "use_abs": use_abs, "n_neurons": st["n_neurons"],
                   "n_sessions": st["n_sessions"],
                   "mean_a": st["mean_a"], "mean_b": st["mean_b"],
                   "sem_a": st["sem_a"], "sem_b": st["sem_b"],
                   "t_neurons": st["neuron_test"]["stat"],
                   "p_neurons": st["neuron_test"]["p"],
                   "df_neurons": st["neuron_test"]["df1"]}
            if st["session_test"] is not None:
                row.update(t_sessions=st["session_test"]["stat"],
                           p_sessions=st["session_test"]["p"],
                           df_sessions=st["session_test"]["df1"],
                           # sessions the paired test could USE (complete cases)
                           n_sessions_tested=st["session_test"]["n_sessions"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / f"{stem}.csv", index=False)
        print(f"Saved paired neuron r -> {out_dir}")
    plt.show()
    return fig, (per_panel if by_region else next(iter(per_panel.values())))


def _session_time_fraction(table, group_col="ShortName"):
    """Each row's position within its session, 0 (first trial) → 1 (last).

    Rank-based on ``TrialNumber``, so gaps in the trial numbering do not distort
    it. Returned as a Series aligned to ``table``'s index; a session with a
    single trial gets 0.5.
    """
    tn = table.groupby(group_col)["TrialNumber"]
    rank = tn.rank(method="dense") - 1.0
    span = tn.transform(lambda s: s.nunique() - 1)
    return (rank / span.where(span > 0)).fillna(0.5)


def bin_occupancy(table, column, bin_spec, *, by="value", group_col="ShortName"):
    """How much data each bin would get — run this *before* committing to bins.

    Returns a small dataframe (one row per bin) with the trial / neuron / session
    counts and the share of trials, so an over-fine split is visible as a
    near-empty bin instead of showing up later as a blank bar. ``dropped_trials``
    on the frame's attrs counts trials outside every bin. ``by`` selects the
    splitting criterion (see :func:`split_trials`).
    """
    bins, sub_tables = split_trials(table, column, bin_spec, by=by,
                                    group_col=group_col)
    per_trial_total = table.drop_duplicates(
        subset=["ShortName", "TrialNumber"]).shape[0]
    # Where each level sits in SESSION TIME (0 = first trial, 1 = last). A
    # latent that drifts over a session — the reward rate does — makes its low
    # level mostly early trials and its high level mostly late ones, so the
    # comparison silently becomes early-vs-late as well. Spread values near 0.5
    # mean the levels are time-matched; values near 0 / 1 mean they are not.
    pos = _session_time_fraction(table, group_col)
    rows, covered = [], 0
    for b, sub in zip(bins, sub_tables):
        uniq = sub.drop_duplicates(subset=["ShortName", "TrialNumber"])
        n_trials = uniq.shape[0]
        covered += n_trials
        rows.append({"bin": b.label, "lo": b.lo, "hi": b.hi,
                     "n_trials": n_trials,
                     "pct_of_trials": (100.0 * n_trials / per_trial_total
                                       if per_trial_total else np.nan),
                     "n_neurons": sub.long_trace_id.nunique(),
                     "n_sessions": sub.ShortName.nunique(),
                     "mean_session_time": (float(pos.loc[uniq.index].mean())
                                           if n_trials else np.nan)})
    out = pd.DataFrame(rows)
    out.attrs["dropped_trials"] = per_trial_total - covered
    return out


def session_bin_coverage(table, column, bin_spec, *, min_trials=None,
                         by="value", group_col="ShortName"):
    """Per-session view of which bins a session can actually contribute to.

    The across-bin test is **paired on sessions**, so a session only counts when
    it is usable in *every* bin. Two things cost sessions here, and this is the
    cell that separates them:

    - the session never **visits** a bin (a slow-drifting latent like the reward
      rate can sit inside one bin from start to finish); or
    - it visits but with **too few trials** — under ``min_trials`` (default
      :data:`_MIN_CORR_POINTS`) no neuron's correlation is even defined there, so
      the session is just as lost as if it had no trials at all. This is the
      easy one to miss: a bin holding 1-2 trials looks populated in a histogram.

    Returns one row per session: ``BrainRegion``, the observed ``min``/``max`` of
    ``column``, the trial count in each bin (``n_{bin label}``), ``bins_covered``
    (bins with any trial), ``bins_usable`` (bins with ``>= min_trials``), and
    ``complete`` (usable in every bin). ``frame.attrs["n_complete"]`` counts the
    complete ones — still an upper bound, since the trials also have to give a
    non-constant activity/latent pair.
    """
    min_trials = _MIN_CORR_POINTS if min_trials is None else int(min_trials)
    per_trial = (table.drop_duplicates(subset=["ShortName", "TrialNumber"])
                 .reset_index(drop=True))
    bins, idx = _trial_bin_index(per_trial, column, bin_spec, by=by,
                                 group_col=group_col)
    masks = {b.label: (idx == i) for i, b in enumerate(bins, start=1)}
    vals = per_trial[column].to_numpy(dtype=float)
    rows = []
    for sess, idx in per_trial.groupby("ShortName").indices.items():
        v = vals[idx]
        rec = {"ShortName": sess,
               "BrainRegion": per_trial["BrainRegion"].to_numpy()[idx][0],
               "n_trials": len(idx),
               f"min_{column}": float(np.nanmin(v)) if len(v) else np.nan,
               f"max_{column}": float(np.nanmax(v)) if len(v) else np.nan}
        counts = [int(masks[b.label][idx].sum()) for b in bins]
        for b, c in zip(bins, counts):
            rec[f"n_{b.label}"] = c
        rec["bins_covered"] = int(sum(c > 0 for c in counts))
        rec["bins_usable"] = int(sum(c >= min_trials for c in counts))
        rec["complete"] = all(c >= min_trials for c in counts)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["BrainRegion", "ShortName"])
    out.attrs["n_complete"] = int(out["complete"].sum()) if len(out) else 0
    out.attrs["min_trials"] = min_trials
    return out.reset_index(drop=True)


def session_attrition(df_2p, table, column, bin_spec, param, *,
                      corr_by_bin=None, common_neurons=True, min_trials=None,
                      by="value", group_col="ShortName"):
    """Full ledger of every recorded session, from the raw 2-photon df to the
    bars — one row per session, with **where** it was lost.

    Sessions leave at three different places, each reported by a different
    helper, which is why a count can look unexplained when only one of them is
    consulted. This puts all three in one table so the arithmetic closes:

    ``no model trials``    — the session matched no MLE trial and is absent from
                             ``table`` altogether (no fit for that subject, or an
                             identity-key mismatch). Never visible in
                             :func:`session_bin_coverage`, because that only sees
                             ``table``.
    ``too few trials``     — present, but under ``min_trials`` in some bin, so no
                             correlation can be computed there.
    ``no assessable neuron`` — enough trials, but every neuron's ``r`` came out
                             undefined in some bin (constant activity or latent),
                             or ``common_neurons`` removed them. Needs
                             ``corr_by_bin``.
    ``kept``               — contributes to the paired across-bin test.

    ``param`` names the correlate (e.g. ``"DV"``). Pass ``corr_by_bin`` from
    :func:`reward_rate_correlations` to resolve the last stage; without it those
    sessions are reported as ``kept (correlations not checked)``.

    Note there is no ``min_abs_corr`` here on purpose: whether a session appears
    in a bar depends on its neurons' ``r`` being **defined**, not on it clearing
    the tuning threshold. A session of entirely untuned neurons still counts —
    as a 0%.
    """
    min_trials = _MIN_CORR_POINTS if min_trials is None else int(min_trials)
    rcol = f"{PARAM_BY_KEY[param].key}_r"

    raw = df_2p.copy()
    raw["_region"] = [f"{BrainRegion(int(r))}" for r in raw["BrainRegion"]]
    raw_counts = raw.groupby("ShortName").agg(
        BrainRegion=("_region", "first"), n_trials_2p=("TrialNumber", "nunique"))

    in_table = set(table["ShortName"].unique())
    cov = (session_bin_coverage(table, column, bin_spec, min_trials=min_trials,
                                by=by, group_col=group_col)
           if len(table) else pd.DataFrame())
    bins, _idx = _trial_bin_index(table, column, bin_spec, by=by,
                                  group_col=group_col)
    cov_by_sess = ({r.ShortName: r for _, r in cov.iterrows()}
                   if len(cov) else {})

    # Which sessions still have an assessable neuron in each bin?
    assessable = None
    if corr_by_bin is not None:
        frames = list(corr_by_bin)
        if common_neurons:
            keep = _common_assessable(frames, rcol)
            frames = [c[c.long_trace_id.isin(keep)] for c in frames]
        assessable = [set(c.loc[c[rcol].notna(), "ShortName"].unique())
                      for c in frames]

    rows = []
    for sess, rc in raw_counts.iterrows():
        rec = {"ShortName": sess, "BrainRegion": rc.BrainRegion,
               "n_trials_2p": int(rc.n_trials_2p)}
        if sess not in in_table:
            rec.update(n_trials_kept=0, outcome="no model trials",
                       detail="matched no MLE trial (join)")
            rows.append(rec)
            continue
        c = cov_by_sess.get(sess)
        rec["n_trials_kept"] = int(c["n_trials"]) if c is not None else 0
        for b in bins:
            rec[f"n_{b.label}"] = int(c[f"n_{b.label}"]) if c is not None else 0
        thin = [b.label for b in bins
                if (c is None or c[f"n_{b.label}"] < min_trials)]
        if thin:
            rec.update(outcome="too few trials",
                       detail=f"< {min_trials} trials in {', '.join(thin)}")
        elif assessable is not None:
            blind = [b.label for b, a in zip(bins, assessable) if sess not in a]
            if blind:
                rec.update(outcome="no assessable neuron",
                           detail=f"no neuron with a defined r in "
                                  f"{', '.join(blind)}")
            else:
                rec.update(outcome="kept", detail="")
        else:
            rec.update(outcome="kept (correlations not checked)",
                       detail="pass corr_by_bin to resolve")
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["BrainRegion", "outcome", "ShortName"])
    out.attrs["counts"] = out.groupby(["BrainRegion", "outcome"]).size()
    return out.reset_index(drop=True)


def binned_correlations(table, column, bin_spec, params=None, *,
                        method="pearson", by="value", group_col="ShortName"):
    """Per-neuron correlations computed **within** each value bin of ``column``.

    Returns ``(bins, bin_tables, corr_by_bin)``: the normalized
    :class:`ValueBin` list, the per-bin trial sub-tables, and the per-bin
    correlation dataframes (one row per neuron, ``{param}_r`` / ``{param}_p``).
    ``params`` defaults to the drift columns (``DV``, ``DVabs``) — the same
    correlates the fast/slow section uses; pass e.g. ``["Q_val"]`` for a latent.
    A bin with no trials yields an empty (but correctly-columned) frame.
    """
    params = list(DRIFT_KEYS) if params is None else list(params)
    bins, bin_tables = split_trials(table, column, bin_spec, by=by,
                                    group_col=group_col)
    corr_by_bin = [compute_neuron_correlations(t, params, method=method)
                   for t in bin_tables]
    return bins, bin_tables, corr_by_bin


def reward_rate_correlations(table, bin_spec, params=None, *,
                             column="RewardRate", method="pearson",
                             by="value", group_col="ShortName"):
    """:func:`binned_correlations` on the model's reward-rate latent — the entry
    point the notebook uses.

    ``by="value"`` takes fixed cut points (:func:`make_value_bins`);
    ``by="session_quantile"`` takes an integer and splits **each session at its
    own quantiles**. Prefer the latter here: the reward rate is a *fitted*
    latent, so its distribution differs from session to session, and fixed cut
    points therefore hand each session a different number of trials per level —
    which changes both the chance level and which sessions survive at all.
    """
    return binned_correlations(table, column, bin_spec, params, method=method,
                               by=by, group_col=group_col)


def _common_assessable(corr_by_bin, rcol):
    """Neurons whose correlation is **defined in every bin** — the set the paired
    across-bin comparison is run on.

    Reward-rate bins are deliberately unbalanced, so a rare bin gives some
    neurons too few trials for a correlation. Dropping those neurons everywhere
    keeps the bars a comparison of the *same* population across bins instead of a
    different sub-population per bin.
    """
    sets = [set(c.loc[c[rcol].notna(), "long_trace_id"]) for c in corr_by_bin]
    return set.intersection(*sets) if sets else set()


def _region_sessions(corr_by_bin, region_values):
    """Every session contributing neurons to a region, across all bins — the
    denominator the session-drop report is measured against."""
    out = set()
    for corr in corr_by_bin:
        grp = (corr if region_values is None
               else corr[corr.BrainRegion.isin(region_values)])
        out |= set(grp["ShortName"].dropna().unique())
    return out


def _session_drop_report(pct_frame, region_sessions):
    """``(kept, dropped)`` sessions for the paired across-bin test, with the
    reason each dropped one fell out.

    Two ways a session leaves: it has **no assessable neuron** in a bin (too few
    trials there for a correlation, or a constant activity/latent), or it never
    visits the bin at all. Both surface here as the bin(s) it is missing, so the
    paired-``n`` is never just a smaller number with no explanation.
    """
    kept = list(pct_frame.dropna(axis=0, how="any").index)
    dropped = {}
    for sess in sorted(region_sessions):
        if sess in kept:
            continue
        if sess in pct_frame.index:
            missing = [c for c in pct_frame.columns
                       if pd.isna(pct_frame.loc[sess, c])]
        else:
            missing = list(pct_frame.columns)   # nothing assessable anywhere
        dropped[sess] = missing
    return kept, dropped


def _bin_pct_frame(corr_by_bin, bins, rcol, min_abs_corr, *, region_values=None):
    """Sessions × bins dataframe of the per-session % of tuned neurons — the
    paired unit of the across-bin test. A session absent from a bin is NaN."""
    cols = {}
    for b, corr in zip(bins, corr_by_bin):
        grp = (corr if region_values is None
               else corr[corr.BrainRegion.isin(region_values)])
        cols[b.label] = _session_tuned_percent(grp, rcol, min_abs_corr)
    return pd.DataFrame(cols, columns=[b.label for b in bins])


def paired_bins_test(pct_frame):
    """Session-paired across-bin test on a sessions × bins % matrix.

    Sessions missing any bin are dropped (complete cases), then:

    - **2 bins** → a paired t-test over sessions (``scipy.stats.ttest_rel``);
    - **>2 bins** → a one-way repeated-measures ANOVA with session as the
      subject factor (``statsmodels.stats.anova.AnovaRM``).

    Returns a dict with ``test`` / ``stat`` / ``p`` / ``df1`` / ``df2`` /
    ``n_sessions`` / ``label`` (a ready-to-plot string). Degenerate inputs — under
    two complete sessions, or percentages identical across bins in every session
    — return ``p = 1`` (t/F = 0) rather than a NaN from a 0/0 variance ratio.
    """
    frame = pct_frame.dropna(axis=0, how="any")
    n, k = len(frame), frame.shape[1]
    none = dict(test="none", stat=np.nan, p=np.nan, df1=np.nan, df2=np.nan,
                n_sessions=n, label="")
    if k < 2:
        return dict(none, label="need >= 2 bins")
    if n < 2:
        return dict(none, label=f"only {n} complete session(s)")

    vals = frame.to_numpy(dtype=float)
    if np.allclose(vals - vals.mean(axis=1, keepdims=True), 0.0):
        # No between-bin variation at all -> the test statistic is 0/0.
        name = "paired t-test" if k == 2 else "RM-ANOVA"
        return dict(test=name, stat=0.0, p=1.0, df1=(n - 1 if k == 2 else 0.0),
                    df2=np.nan, n_sessions=n,
                    label=f"{name}: no between-bin difference (p = 1.000)")

    if k == 2:
        res = ttest_rel(vals[:, 0], vals[:, 1])
        t, p = float(res.statistic), float(res.pvalue)
        return dict(test="paired t-test", stat=t, p=p, df1=float(n - 1),
                    df2=np.nan, n_sessions=n,
                    label=f"paired t({n - 1}) = {t:+.2f}, {_p_phrase(p)}")

    long = (frame.rename_axis("session").reset_index()
            .melt(id_vars="session", var_name="bin", value_name="pct"))
    anova = AnovaRM(long, depvar="pct", subject="session",
                    within=["bin"]).fit().anova_table
    F = float(anova["F Value"].iloc[0])
    p = float(anova["Pr > F"].iloc[0])
    df1 = float(anova["Num DF"].iloc[0])
    df2 = float(anova["Den DF"].iloc[0])
    return dict(test="RM-ANOVA", stat=F, p=p, df1=df1, df2=df2, n_sessions=n,
                label=f"RM-ANOVA F({df1:g},{df2:g}) = {F:.2f}, {_p_phrase(p)}")


def pairwise_bin_tests(pct_frame):
    """Holm-corrected paired t-tests between every pair of bins (the post-hoc
    for a significant RM-ANOVA). Returns a tidy dataframe; empty when fewer than
    two complete sessions."""
    frame = pct_frame.dropna(axis=0, how="any")
    labels = list(frame.columns)
    if len(frame) < 2:
        return pd.DataFrame()
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = frame[labels[i]].to_numpy(dtype=float)
            b = frame[labels[j]].to_numpy(dtype=float)
            if np.allclose(a - b, 0.0):
                t, p = 0.0, 1.0
            else:
                res = ttest_rel(a, b)
                t, p = float(res.statistic), float(res.pvalue)
            rows.append({"bin_a": labels[i], "bin_b": labels[j],
                         "mean_diff": float(np.mean(a - b)), "t": t, "p": p,
                         "n_sessions": len(frame)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = _holm_adjust(out["p"].to_numpy())
    return out


def plot_reward_rate_bars(corr_by_bin, bins, param, *, bin_tables=None,
                          min_abs_corr=0.3, n_perm=1000, by_region=True,
                          seed=0, run_stats=True, common_neurons=True,
                          value_label="Reward rate", save=False, save_root=None,
                          model_name=None, ext="svg"):
    """% of ``param``-correlated neurons at each reward-rate level.

    The reward-rate variation of :func:`plot_fast_slow_bars`: instead of two RT
    terciles the x-axis is the reward-rate bins (:func:`reward_rate_correlations`),
    shaded pale → dark with the reward-rate colour so low → high reads off the
    bar. A neuron counts at a bin when ``|r| >= min_abs_corr`` for ``param``
    *within that bin's trials*; each bar is the session mean ± SEM of that
    percentage. ``by_region=True`` draws one panel per MFC/LFC (shared y-range);
    ``by_region=False`` pools the regions into one panel.

    With ``common_neurons`` (default) only neurons whose correlation is defined
    in **every** bin are used — see :func:`_common_assessable`; the shuffle null
    is restricted to the same neurons so bars and chance level agree.

    When ``run_stats`` two families are annotated, each Holm-corrected:
      1. **vs chance** — a ``*/**/***`` star above each bar from the per-neuron
         activity shuffle (``n_perm`` permutations, ``+1/(n+1)`` p-value; needs
         ``bin_tables`` — skipped with a warning if omitted), Holm-corrected
         across the bars of a panel;
      2. **across bins** — the session-paired :func:`paired_bins_test` (paired
         t-test for 2 bins, RM-ANOVA beyond), Holm-corrected across panels, drawn
         as one bracket spanning the bins with its statistic printed above. For
         more than two bins the Holm-corrected pairwise post-hocs
         (:func:`pairwise_bin_tests`) are appended to the summary.

    Sessions are **never dropped silently**: any session that cannot be paired
    across all bins is named, with the bins it lacks, and raises a warning. The
    counts are also on every summary row as ``n_sessions_region`` /
    ``n_sessions_dropped`` / ``n_paired_sessions``, so a shrunken paired-``n``
    can always be traced back. Zero pairable sessions is handled, not an error:
    the bars still draw and the test reports ``"none"``.

    Returns a tidy summary keyed by a ``scope`` column:
    ``"bin"`` — one row per region × bin (the bars), carrying the panel's test
    result; ``"session"`` — the per-session detail behind each bar (its own
    ``pct``, ``n_trials``, ``n_neurons``, ``trials_per_neuron`` and whether it
    was ``paired``), since the bar is just the mean of these; ``"comparison"`` —
    the pairwise post-hocs. Filter with ``summary[summary.scope == "session"]``.
    ``save`` also writes the figure + summary CSV under
    ``{save_root}/{model_name}/reward_rate_bars/{param}/``.
    """
    spec = PARAM_BY_KEY[param]
    rcol = f"{spec.key}_r"
    for corr in corr_by_bin:
        if rcol not in corr.columns:
            raise KeyError(f"{param!r} not in the binned correlation dataframes "
                           f"(columns: {list(corr.columns)}).")
    if len(bins) != len(corr_by_bin):
        raise ValueError(f"{len(bins)} bins but {len(corr_by_bin)} correlation "
                         "frames.")
    if len(bins) < 2:
        raise ValueError("Need at least 2 bins to compare reward-rate levels.")

    # Kept unfiltered so the session-drop report can measure against every
    # session that contributed neurons, not only the ones that survived.
    corr_before_filter = list(corr_by_bin)

    # Restrict to the neurons assessable in every bin (bars AND shuffle null).
    if common_neurons:
        keep = _common_assessable(corr_by_bin, rcol)
        if not keep:
            warnings.warn(
                "plot_reward_rate_bars: no neuron has a defined correlation in "
                "EVERY bin, so common_neurons=True leaves nothing to plot — "
                "usually a bin that is empty or too sparse to correlate within. "
                "Widen/merge the bins, or pass common_neurons=False.")
        corr_by_bin = [c[c.long_trace_id.isin(keep)] for c in corr_by_bin]
        if bin_tables is not None:
            bin_tables = [t[t.long_trace_id.isin(keep)] for t in bin_tables]

    rng = np.random.default_rng(seed)
    bundles = None
    if run_stats:
        if bin_tables is not None:
            bundles = [_shuffle_null_tuned(t, [spec.key], min_abs_corr, n_perm,
                                           rng)
                       for t in bin_tables]
        else:
            warnings.warn("plot_reward_rate_bars: `bin_tables` not supplied — "
                          "skipping the per-bar vs-chance test (the across-bin "
                          "paired test still runs).")

    if by_region:
        regions = [(r, [r]) for r in
                   sorted(set().union(*(set(c.BrainRegion.dropna().unique())
                                        for c in corr_by_bin)))]
    else:
        regions = [("MFC & LFC", None)]
    if not regions:
        raise ValueError("No neurons left to plot (empty correlation frames).")

    colors = _param_gradient(PARAM_BY_KEY["RewardRate"].color, len(bins))
    x = np.arange(len(bins))

    # ---- pass 1: per-panel stats ---------------------------------------------
    panels, rows = {}, []
    for region, rvals in regions:
        pct_frame = _bin_pct_frame(corr_by_bin, bins, rcol, min_abs_corr,
                                   region_values=rvals)
        # Per column so an all-empty bin is a plain NaN bar, not a warning.
        n_obs = [int(pct_frame[c].notna().sum()) for c in pct_frame.columns]
        means = np.array([pct_frame[c].mean() if n else np.nan
                          for c, n in zip(pct_frame.columns, n_obs)], dtype=float)
        sems = np.array([pct_frame[c].sem() if n > 1 else 0.0
                         for c, n in zip(pct_frame.columns, n_obs)], dtype=float)
        sems = np.nan_to_num(sems, nan=0.0)
        vs_chance = []
        for bi, b in enumerate(bins):
            p = np.nan
            if bundles is not None:
                p = _bar_vs_chance(bundles[bi], spec.key, rvals, means[bi],
                                   n_perm)
            vs_chance.append(p)
        holm_chance = (_holm_adjust(vs_chance) if run_stats
                       else np.full(len(bins), np.nan))
        test = paired_bins_test(pct_frame)
        pairwise = (pairwise_bin_tests(pct_frame)
                    if len(bins) > 2 and run_stats else pd.DataFrame())
        region_sessions = _region_sessions(corr_before_filter, rvals)
        _kept, dropped = _session_drop_report(pct_frame, region_sessions)
        panels[region] = dict(pct_frame=pct_frame, means=means, sems=sems,
                              vs_chance=np.asarray(vs_chance, dtype=float),
                              holm_chance=holm_chance, test=test,
                              pairwise=pairwise, rvals=rvals,
                              n_sessions_total=len(region_sessions),
                              n_sessions_dropped=len(dropped))
        print(f"\t{region}: " + " | ".join(
            f"{b.label} {m:.1f}±{s:.1f}% ({n} sess)"
            for b, m, s, n in zip(bins, means, sems, n_obs)))
        if test["label"]:
            print(f"\t  across bins: {test['label']}  "
                  f"[{test['n_sessions']} of {len(region_sessions)} sessions "
                  "paired]")
        # Never let sessions vanish quietly: name them and say what they lack.
        if dropped:
            print(f"\t  {len(dropped)} session(s) EXCLUDED from the paired test "
                  f"(no data in every bin):")
            for sess, missing in dropped.items():
                print(f"\t    {sess}: nothing assessable in {', '.join(missing)}")
            warnings.warn(
                f"plot_reward_rate_bars [{region}]: {len(dropped)} of "
                f"{len(region_sessions)} sessions dropped from the paired "
                f"across-bin test because they have no assessable neuron in "
                f"every bin ({', '.join(sorted(dropped))}). Widen/merge the "
                "bins (see quantile_bin_edges / session_bin_coverage) if that "
                "is more sessions than you expect.")

    # Holm the across-bin test across panels (one comparison per region).
    across_holm = (_holm_adjust([panels[r]["test"]["p"] for r, _ in regions])
                   if run_stats else np.full(len(regions), np.nan))
    for (region, _), ph in zip(regions, across_holm):
        panels[region]["p_across_holm"] = float(ph) if np.isfinite(ph) else np.nan

    # ---- summary rows ---------------------------------------------------------
    for region, rvals in regions:
        d = panels[region]
        t = d["test"]
        paired_sessions = set(d["pct_frame"].dropna(axis=0, how="any").index)
        for bi, b in enumerate(bins):
            n_trials = n_rows = n_neurons = np.nan
            sub = None
            if bin_tables is not None:
                sub = (bin_tables[bi] if rvals is None
                       else bin_tables[bi][bin_tables[bi].BrainRegion.isin(rvals)])
                n_rows = int(len(sub))
                n_trials = int(sub.drop_duplicates(
                    subset=["ShortName", "TrialNumber"]).shape[0])
                n_neurons = int(sub.long_trace_id.nunique())
            rows.append({
                "scope": "bin",
                "BrainRegion": region, "ShortName": "", "bin": b.label,
                "bin_lo": b.lo,
                "bin_hi": b.hi, "param": spec.key, "pct": d["means"][bi],
                "sem": d["sems"][bi],
                "n_sessions": int(d["pct_frame"][b.label].notna().sum()),
                "n_neurons": n_neurons,
                # n_trials counts distinct TRIALS; n_rows counts (neuron x trial)
                # rows. Trials-per-neuron = n_rows / n_neurons is what sets the
                # correlation's precision, so both are worth having.
                "n_trials": n_trials, "n_rows": n_rows,
                # Raw alongside corrected: without the raw p there is no way to
                # tell a bar that Holm pushed over the line from one that was
                # never near it (the largest p in a family is multiplied by 1).
                "p_vs_chance": (float(d["vs_chance"][bi])
                                if np.isfinite(d["vs_chance"][bi]) else np.nan),
                "p_vs_chance_holm": (float(d["holm_chance"][bi])
                                     if np.isfinite(d["holm_chance"][bi])
                                     else np.nan),
                "test_across_bins": t["test"], "stat_across_bins": t["stat"],
                "p_across_bins": t["p"], "p_across_bins_holm": d["p_across_holm"],
                "df1": t["df1"], "df2": t["df2"],
                "n_paired_sessions": t["n_sessions"],
                # Audit trail for the paired-n: how many sessions the region has
                # and how many could not be paired across all bins.
                "n_sessions_region": d["n_sessions_total"],
                "n_sessions_dropped": d["n_sessions_dropped"]})

            # Per-session detail behind this bar: the bar is the mean of these,
            # so an outlying session (or one with far fewer trials than the rest)
            # is visible instead of being averaged away.
            if sub is None:
                continue
            per_sess_rows = sub.drop_duplicates(
                subset=["ShortName", "TrialNumber"]).groupby("ShortName")
            neurons_by_sess = sub.groupby("ShortName")["long_trace_id"].nunique()
            rows_by_sess = sub.groupby("ShortName").size()
            for sess, sg in per_sess_rows:
                pct = (d["pct_frame"].loc[sess, b.label]
                       if sess in d["pct_frame"].index else np.nan)
                n_neu = int(neurons_by_sess.get(sess, 0))
                n_row = int(rows_by_sess.get(sess, 0))
                rows.append({
                    "scope": "session", "BrainRegion": region,
                    "ShortName": sess, "bin": b.label, "bin_lo": b.lo,
                    "bin_hi": b.hi, "param": spec.key,
                    "pct": float(pct) if pd.notna(pct) else np.nan,
                    "n_neurons": n_neu, "n_trials": int(len(sg)),
                    "n_rows": n_row,
                    # What actually sets this session's correlation precision.
                    "trials_per_neuron": (n_row / n_neu if n_neu else np.nan),
                    "paired": sess in paired_sessions})

        for _, pr in d["pairwise"].iterrows():
            rows.append({"scope": "comparison", "BrainRegion": region,
                         "ShortName": "",
                         "bin": f"{pr.bin_a} vs {pr.bin_b}", "param": spec.key,
                         "test_across_bins": "paired t-test (post-hoc)",
                         "stat_across_bins": pr.t, "p_across_bins": pr.p,
                         "p_across_bins_holm": pr.p_holm,
                         "df1": pr.n_sessions - 1,
                         "n_paired_sessions": pr.n_sessions})

    # ---- pass 2: draw, on a shared y-range ------------------------------------
    # Paddings scale with the data range: fixed point-offsets collide with the
    # vs-chance stars as soon as a bar's SEM is tall.
    bar_top = 0.0
    for region, _ in regions:
        d = panels[region]
        tops = [m + s for m, s in zip(d["means"], d["sems"]) if np.isfinite(m)]
        if tops:
            bar_top = max(bar_top, max(tops))
    bar_top = bar_top if bar_top > 0 else 1.0
    _STAR_PAD = 0.03 * bar_top      # gap under a vs-chance star
    _BRK_TICK = 0.02 * bar_top      # bracket end drop-down
    _TXT_PAD = 0.02 * bar_top       # gap under the bracket's statistic text
    # The bracket must clear the tallest bar AND the star sitting above it.
    _BRK_BASE = bar_top + 4.0 * _STAR_PAD
    y_top = _BRK_BASE + _BRK_TICK + _TXT_PAD + 0.09 * bar_top

    fig, axs = plt.subplots(1, len(regions), figsize=(4.6 * len(regions), 4.6),
                            squeeze=False, sharey=True)
    for ax, (region, rvals) in zip(axs[0], regions):
        d = panels[region]
        ax.bar(x, d["means"], width=0.62, color=colors, alpha=0.9,
               edgecolor="k", linewidth=0.5, yerr=d["sems"], capsize=3, zorder=2)
        if run_stats:
            for xi, m, s, ph in zip(x, d["means"], d["sems"], d["holm_chance"]):
                if np.isfinite(m):
                    ax.text(xi, m + s + _STAR_PAD, _sigstar(ph), ha="center",
                            va="bottom", fontsize=11)
            # One bracket over the whole bin range = the across-bin paired test.
            # The star reflects the HOLM-corrected p, so the text must report it
            # too — a raw "p = 0.049" beside an "n.s." star reads as a bug.
            t = d["test"]
            star = _sigstar(d["p_across_holm"])
            if star and np.isfinite(d["means"]).any():
                label = t["label"]
                ph = d["p_across_holm"]
                if np.isfinite(ph) and not np.isclose(ph, t["p"]):
                    label += f"  →  Holm {_p_phrase(ph)}"
                _add_sig_bracket(ax, x[0], x[-1], _BRK_BASE, star,
                                 tick=_BRK_TICK, fontsize=11)
                ax.text((x[0] + x[-1]) / 2.0, _BRK_BASE + _BRK_TICK + _TXT_PAD,
                        label, ha="center", va="bottom", fontsize="x-small",
                        color="0.25")
        ax.set_xticks(x)
        ax.set_xticklabels([b.label for b in bins], rotation=20, ha="right",
                           fontsize="small")
        ax.set_xlabel(value_label)
        ax.set_title(f"{region}  (|r| ≥ {min_abs_corr:g})")
        ax.set_ylim(0, y_top)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0][0].set_ylabel(f"{spec.label}-correlated neurons (%)")
    fig.suptitle(f"{spec.label} tuning across {value_label.lower()} — "
                 f"{model_name or ''}", y=1.02)
    fig.tight_layout()
    if run_stats:
        # Below the axes, not inside them: the across-bin bracket spans the full
        # panel width, so any in-axes corner overlaps it.
        fig.legend(
            handles=[Line2D([0], [0], marker="*", color="k", linestyle="None",
                            label="★ above bar: vs shuffle chance"),
                     Line2D([0], [0], color="k", lw=1,
                            label="⊓ bracket: across bins (paired)")],
            fontsize="x-small", frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, 0.0), ncol=2, title="Holm-corrected across "
            "bars within a panel / across panels respectively",
            title_fontsize="x-small")

    summary = pd.DataFrame(rows)
    if save:
        if save_root is None or model_name is None:
            raise ValueError("save needs save_root and model_name")
        out_dir = (pathlib.Path(save_root) / _safe_filename(model_name)
                   / "reward_rate_bars" / spec.key)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "by_region" if by_region else "combined"
        fig.savefig(out_dir / f"bars_{tag}.{ext}", bbox_inches="tight")
        summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
        print(f"Saved reward-rate bars -> {out_dir}")
    plt.show()
    return summary
