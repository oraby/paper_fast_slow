"""Within-strategy trial-to-trial sequence deviation.

Companion to the ``### Monte Carlo ...`` and ``### Seq firing deviation``
sections of ``2pAnalysis.ipynb``. Those sections compare the *aggregate* Fast
sequence to the *aggregate* Slow sequence (per-neuron median firing order in one
strategy vs the other). Here we instead measure, **within a strategy**, how much
each *individual* trial's firing sequence deviates from that strategy's own
reference sequence, and compare that deviation between Fast and Slow.

Design (per-strategy reference, "each vs itself"): Fast trials are scored against
a Fast reference, Slow trials against a Slow reference. Neuron non-overlap
between the two strategies is handled naturally because each reference is built
from only its own strategy's active neurons.

Reference ranks reuse the *same* recipe as the existing notebook code
(:func:`extractIQR` + ``med.rank(method="dense")``), minus the 3-way inner-join:
we keep every strategy-active neuron (``prcnt_valid`` above the threshold), per
session.

Penalty (per the worked example in the task): within one trial take the active
neurons' global reference ranks (e.g. ``1, 5, 10``). ``expected`` = those ranks
sorted ascending; ``observed`` = the same ranks ordered by within-trial peak
time. The penalty attributed to the neuron *expected* at position ``i`` is
``abs(expected[i] - observed[i]) / n_active_in_trial``. So for expected
``[1, 5, 10]`` and observed ``[5, 1, 10]`` the penalties are
``|1-5|/3, |5-1|/3, 0``. Ties in peak time are broken by reference rank so a tie
contributes no artificial deviation.

Because that global ``penalty`` divides by the per-trial active count ``n``
(per-trial mean = displacement / n^2), it is confounded by how many neurons a
trial recruits. Each row therefore also carries a ``norm_penalty``: the active
neurons are re-ranked 1..n locally, and the Spearman footrule
``|local_ref - local_observed|`` is scaled so the per-trial *mean* is
footrule / (full-reversal footrule = ``floor(n^2/2)``) -- 0 for reference order,
1 for a full reversal, ~0.67 for a random order, independent of ``n``. This
trades away the global-rank magnitude (only the relative order of the active
neurons matters) for a size-comparable score.

The main dataframe (``build_penalty_df``) has one row per trial x neuron and is
meant to be saved and reused for later analysis / significance testing.
"""

import numpy as np
import pandas as pd

from ..common.definitions import BrainRegion

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRATEGY_FAST = "Fast"
STRATEGY_SLOW = "Slow"

# Match the impulsive="r"/deliberate="y" convention used elsewhere in the
# notebook, but slightly more legible.
FAST_CLR = "red"
SLOW_CLR = "gold"

# ">5% active" from the task description (strict, as written).
DEFAULT_MIN_PRCNT_ACTIVE = 5.0
# A trial needs at least this many reference-set neurons active to define a
# sequence. A 1-neuron trial has an identically-zero penalty (no possible
# re-ordering), which would only dilute the distribution; drop it by default.
DEFAULT_MIN_ACTIVE_IN_TRIAL = 2

PENALTY_COLUMNS = [
    "BrainRegion", "ShortName", "trace_id", "trace_num", "trial_strategy",
    "TrialNumber", "ref_rank", "observed_rank", "penalty",
    "local_ref_rank", "local_observed_rank", "norm_penalty",
    "gap_norm_penalty", "abs_gap_penalty",
    "n_active_in_trial", "n_ref_neurons",
]

# Human labels for the deviation scores, used on plot axes.
SCORE_LABELS = {
    "penalty": "Rank deviation penalty (global ranks)",
    "norm_penalty": "Per-trial normalized disorder (local; 0 ref .. 1 reversed)",
    "gap_norm_penalty": "Gap-aware disorder, self-normalized (0 ref .. 1 reversed)",
    "abs_gap_penalty": "Gap-aware displacement, fraction of sequence (0 .. 1)",
}


# ---------------------------------------------------------------------------
# Reference sequence (moved out of 2pAnalysis.ipynb cell 36 so it can be reused
# and unit-tested)
# ---------------------------------------------------------------------------
def extractIQR(df):
    """Per-neuron IQR / median of the peak-firing sample (``max_idxs``).

    Adds ``whislo, q1, med, q3, whishi`` columns and returns the frame sorted by
    ``med`` (the median peak-firing sample, i.e. the neuron's position in the
    firing sequence). Copied verbatim from ``2pAnalysis.ipynb`` so the notebook
    and this module compute the reference sequence identically.
    """
    df = df.copy()

    def IQR(row):
        max_idxs = pd.Series(row.max_idxs)
        _min, q1, median, q3, _max = max_idxs.quantile([0, .25, .5, .75, 1])
        IQR = q3 - q1
        whis_lo = max_idxs[max_idxs > q1 - 1.5 * IQR].min()
        whis_hi = max_idxs[max_idxs < q3 + 1.5 * IQR].max()
        row["whislo"] = whis_lo
        row["q1"] = q1
        row["med"] = median
        row["q3"] = q3
        row["whishi"] = whis_hi
        return row

    df = df.apply(IQR, axis=1)
    df = df.sort_values("med")
    return df


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def region_label(br):
    """Map a ``BrainRegion`` enum value (int) to its short label (MFC/LFC)."""
    try:
        return "{}".format(BrainRegion(int(br)))
    except (ValueError, KeyError):
        return str(br)


def _parse_trace_num(trace_id):
    """Neuron index = the integer suffix of the long ``trace_id``."""
    try:
        return int(str(trace_id).rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return np.nan


# ---------------------------------------------------------------------------
# Reference ranks
# ---------------------------------------------------------------------------
def reference_ranks(strategy_df, min_prcnt_active=DEFAULT_MIN_PRCNT_ACTIVE,
                    rank_method="dense"):
    """Per-session reference rank of every strategy-active neuron.

    Parameters
    ----------
    strategy_df : DataFrame
        A ``max_firing_q*_df`` (one row per neuron in a session) for a single
        strategy, with columns ``ShortName, trace_id, BrainRegion, prcnt_valid,
        max_idxs`` (peak sample per active trial).
    min_prcnt_active : float
        Keep neurons active in strictly more than this percent of the strategy's
        trials (">5%").
    rank_method : str
        ``pandas.Series.rank`` method for turning ``med`` into ``ref_rank``.
        Default ``"dense"`` matches the existing MC / Seq-deviation notebook
        code. Because the peak sample (``max_idxs``) spans only ~30 samples,
        ``"dense"`` collapses many neurons onto shared ranks (~3 neurons/rank);
        use ``"first"`` (unique, arbitrary tie order) or ``"min"`` to spread
        them out instead.

    Returns
    -------
    DataFrame with ``ShortName, trace_id, BrainRegion, med, ref_rank,
    n_ref_neurons``. ``ref_rank`` is the rank of ``med`` within the session
    (1 = earliest-firing).
    """
    parts = []
    for _sess, sdf in strategy_df.groupby("ShortName"):
        ref = sdf[sdf.prcnt_valid > min_prcnt_active]
        if len(ref) == 0:
            continue
        ref = extractIQR(ref)
        ref = ref.copy()
        ranks = ref["med"].rank(method=rank_method)
        if (ranks == ranks.round()).all():  # keep "average" method fractional
            ranks = ranks.astype(int)
        ref["ref_rank"] = ranks
        ref["n_ref_neurons"] = len(ref)
        parts.append(ref[["ShortName", "trace_id", "BrainRegion", "med",
                          "ref_rank", "n_ref_neurons"]])
    if not parts:
        return pd.DataFrame(columns=["ShortName", "trace_id", "BrainRegion",
                                     "med", "ref_rank", "n_ref_neurons"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-trial penalty
# ---------------------------------------------------------------------------
def _trial_penalty(trial_df):
    """Deviation scores for the active neurons of one trial.

    ``trial_df`` holds the reference-set neurons active in a single trial, with
    columns ``ref_rank`` and ``peak_time``. Returns the same rows (ordered by
    expected/reference position) with two scores attached:

    - ``penalty`` (global): ``|ref_rank - observed_rank| / n`` using the global
      1..N_ref ranks. Sensitive to absolute sequence position but confounded by
      the active-neuron count ``n`` (per-trial mean = displacement / n^2).
    - ``norm_penalty`` (per-trial normalized): re-rank the active neurons 1..n
      locally, take the Spearman footrule ``|local_ref - local_observed|`` and
      scale it so the trial MEAN equals footrule / max-footrule (full reversal =
      ``floor(n^2/2)``). 0 = reference order, 1 = fully reversed, ~0.67 random,
      independent of ``n`` -- but it discards the global rank magnitude (only the
      relative order of the active neurons matters).
    - ``gap_norm_penalty`` (gap-aware, self-normalized): like ``norm_penalty`` but
      the footrule keeps the *global* rank gaps and is divided by this set's own
      max reversal, so overtaking across a larger reference gap costs more.
      Trial MEAN in 0..1 relative to the worst case for this active set.
    - ``abs_gap_penalty`` is added in :func:`_score_trials` (it needs the full
      reference size ``n_ref_neurons``).
    """
    n = len(trial_df)
    # Expected order = reference sequence (ties broken deterministically).
    exp = trial_df.sort_values(["ref_rank", "trace_id"], kind="stable")
    # Observed order = within-trial peak time; ties broken by reference rank so
    # simultaneous peaks add no artificial deviation.
    obs = exp.sort_values(["peak_time", "ref_rank"], kind="stable")

    exp_ranks = exp["ref_rank"].to_numpy()
    obs_ranks = obs["ref_rank"].to_numpy()

    # Local within-trial ranks 1..n: position in the expected (reference) order
    # vs position in the observed (peak-time) order.
    local_ref_rank = np.arange(1, n + 1)
    obs_local = pd.Series(np.arange(1, n + 1), index=obs.index)  # neuron -> obs pos
    local_obs_rank = obs_local.reindex(exp.index).to_numpy()
    local_disp = np.abs(local_ref_rank - local_obs_rank)
    max_footrule = (n * n) // 2  # sum |i - (n+1-i)| for a full reversal
    # Scale so the per-trial mean of norm_penalty == footrule / max_footrule.
    norm = (n * local_disp / max_footrule) if max_footrule else np.zeros(n)

    # Gap-aware footrule keeps the global rank gaps; normalize by this set's own
    # max reversal (exp_ranks is already sorted ascending, so it is the sorted
    # rank vector). Scale per neuron so the trial mean == footrule / max reversal.
    global_disp = np.abs(exp_ranks - obs_ranks)
    max_reversal = np.abs(exp_ranks - exp_ranks[::-1]).sum()
    gap_norm = (n * global_disp / max_reversal) if max_reversal else np.zeros(n)

    out = exp.copy()
    out["observed_rank"] = obs_ranks
    out["penalty"] = global_disp / n
    out["local_ref_rank"] = local_ref_rank
    out["local_observed_rank"] = local_obs_rank
    out["norm_penalty"] = norm
    out["gap_norm_penalty"] = gap_norm
    out["n_active_in_trial"] = n
    return out


def _score_trials(trials_df, ref, trial_strategy_name, min_active_in_trial):
    """Score every trial in ``trials_df`` against a reference-rank table.

    ``ref`` supplies the *expected* order (``ref_rank``, ``n_ref_neurons``); the
    inner merge restricts to its neurons, so the reference set is whatever ``ref``
    contains. ``trials_df`` supplies the *observed* within-trial peak times
    (``active_trial_numbers`` / ``max_idxs``) of the neurons being scored -- which
    need not be the same strategy that built ``ref`` (that is the cross case).
    """
    if len(ref) == 0:
        return pd.DataFrame(columns=PENALTY_COLUMNS)

    keep = trials_df[["ShortName", "trace_id", "BrainRegion",
                      "active_trial_numbers", "max_idxs"]].merge(
        ref[["ShortName", "trace_id", "ref_rank", "n_ref_neurons"]],
        on=["ShortName", "trace_id"], how="inner")

    # Explode the parallel per-trial arrays (equal length per neuron).
    keep = keep.copy()
    keep["active_trial_numbers"] = keep["active_trial_numbers"].apply(
        lambda a: list(np.asarray(a)))
    keep["max_idxs"] = keep["max_idxs"].apply(lambda a: list(np.asarray(a)))
    long = keep.explode(["active_trial_numbers", "max_idxs"], ignore_index=True)
    long = long.rename(columns={"active_trial_numbers": "TrialNumber",
                                "max_idxs": "peak_time"})
    long = long.dropna(subset=["TrialNumber", "peak_time"])
    long["TrialNumber"] = long["TrialNumber"].astype(int)
    long["peak_time"] = long["peak_time"].astype(float)

    # Keep only trials with enough active reference neurons to form a sequence.
    sizes = long.groupby(["ShortName", "TrialNumber"])["ref_rank"].transform("size")
    long = long[sizes >= min_active_in_trial]
    if len(long) == 0:
        return pd.DataFrame(columns=PENALTY_COLUMNS)

    # Per-trial penalty (explicit loop avoids the groupby-apply-on-grouping-
    # columns deprecation and is plenty fast for ~a couple thousand trials).
    parts = [_trial_penalty(g) for _key, g
             in long.groupby(["ShortName", "TrialNumber"], sort=False)]
    res = pd.concat(parts, ignore_index=True)

    res["trial_strategy"] = trial_strategy_name
    res["BrainRegion"] = res["BrainRegion"].apply(region_label)
    res["trace_num"] = res["trace_id"].apply(_parse_trace_num)
    # Gap-aware, absolute scale: |global displacement| as a fraction of the full
    # reference length (n_ref_neurons - 1). Unlike gap_norm_penalty this is NOT
    # self-normalized, so tightly-clustered active sets stay small.
    denom = (res["n_ref_neurons"] - 1).clip(lower=1)
    res["abs_gap_penalty"] = np.abs(res["ref_rank"] - res["observed_rank"]) / denom
    return res[PENALTY_COLUMNS].reset_index(drop=True)


def trial_penalties(strategy_df, strategy_name,
                    min_prcnt_active=DEFAULT_MIN_PRCNT_ACTIVE,
                    min_active_in_trial=DEFAULT_MIN_ACTIVE_IN_TRIAL,
                    rank_method="dense"):
    """Long (trial x neuron) deviation-penalty frame for one strategy scored
    against its *own* reference (each-vs-itself; the within-strategy analysis)."""
    ref = reference_ranks(strategy_df, min_prcnt_active, rank_method)
    return _score_trials(strategy_df, ref, strategy_name, min_active_in_trial)


def build_penalty_df(fast_df, slow_df,
                     min_prcnt_active=DEFAULT_MIN_PRCNT_ACTIVE,
                     min_active_in_trial=DEFAULT_MIN_ACTIVE_IN_TRIAL,
                     rank_method="dense"):
    """Concatenated Fast + Slow trial x neuron penalty frame.

    ``fast_df`` / ``slow_df`` are ``max_firing_q1_df`` / ``max_firing_q3_df``.
    """
    fast = trial_penalties(fast_df, STRATEGY_FAST, min_prcnt_active,
                           min_active_in_trial, rank_method)
    slow = trial_penalties(slow_df, STRATEGY_SLOW, min_prcnt_active,
                           min_active_in_trial, rank_method)
    return pd.concat([fast, slow], ignore_index=True)


# ---------------------------------------------------------------------------
# "Same sequence?" analysis: score BOTH strategies' trials against ONE
# strategy's reference order, on the neurons commonly active in both.
# ---------------------------------------------------------------------------
CONDITION_MATCHED = "matched"   # the reference strategy's own trials (baseline)
CONDITION_CROSS = "cross"       # the other strategy's trials vs the reference

CROSS_COLUMNS = PENALTY_COLUMNS + ["condition", "reference"]


def common_reference_ranks(ref_df, other_df,
                           min_prcnt_active=DEFAULT_MIN_PRCNT_ACTIVE,
                           rank_method="dense"):
    """Reference order from ``ref_df``, restricted to neurons active
    (>``min_prcnt_active``%) in **both** strategies and re-ranked within that
    common set, per session. ``ref_rank`` follows ``ref_df``'s firing order;
    ``n_ref_neurons`` is the common-set size."""
    ref_active = reference_ranks(ref_df, min_prcnt_active, rank_method)
    other_active = reference_ranks(other_df, min_prcnt_active, rank_method)
    empty = pd.DataFrame(columns=["ShortName", "trace_id", "BrainRegion",
                                  "med", "ref_rank", "n_ref_neurons"])
    if len(ref_active) == 0 or len(other_active) == 0:
        return empty
    common = ref_active.merge(other_active[["ShortName", "trace_id"]],
                              on=["ShortName", "trace_id"], how="inner")
    parts = []
    for _sess, g in common.groupby("ShortName"):
        g = g.copy()
        ranks = g["med"].rank(method=rank_method)
        if (ranks == ranks.round()).all():
            ranks = ranks.astype(int)
        g["ref_rank"] = ranks
        g["n_ref_neurons"] = len(g)
        parts.append(g[["ShortName", "trace_id", "BrainRegion", "med",
                        "ref_rank", "n_ref_neurons"]])
    return pd.concat(parts, ignore_index=True) if parts else empty


def build_cross_penalty_df(fast_df, slow_df, reference=STRATEGY_FAST,
                           min_prcnt_active=DEFAULT_MIN_PRCNT_ACTIVE,
                           min_active_in_trial=DEFAULT_MIN_ACTIVE_IN_TRIAL,
                           rank_method="dense"):
    """Do Fast and Slow employ the *same* firing sequence?

    Build the ``reference`` strategy's firing order on the **common** neuron set
    (active in both strategies), then score both strategies' trials against it:

    - ``matched`` -- the reference strategy's own trials (the within-strategy
      floor: how much its own trials wobble around the reference order);
    - ``cross`` -- the other strategy's trials against the same reference order.

    ``cross`` ~ ``matched`` -> the two strategies share a sequence; ``cross`` >>
    ``matched`` -> the other strategy fires the common neurons in a different
    order. Use ``norm_penalty`` for the comparison (matched and cross trials
    differ in active-neuron count, which biases the global ``penalty``).

    Returns ``PENALTY_COLUMNS`` plus ``condition`` and ``reference``.
    """
    assert reference in (STRATEGY_FAST, STRATEGY_SLOW)
    ref_df, other_df = ((fast_df, slow_df) if reference == STRATEGY_FAST
                        else (slow_df, fast_df))
    other_name = STRATEGY_SLOW if reference == STRATEGY_FAST else STRATEGY_FAST

    common_ref = common_reference_ranks(ref_df, other_df, min_prcnt_active,
                                        rank_method)
    matched = _score_trials(ref_df, common_ref, reference, min_active_in_trial)
    matched["condition"] = CONDITION_MATCHED
    cross = _score_trials(other_df, common_ref, other_name, min_active_in_trial)
    cross["condition"] = CONDITION_CROSS

    out = pd.concat([matched, cross], ignore_index=True)
    out["reference"] = reference
    return out[CROSS_COLUMNS]


# ---------------------------------------------------------------------------
# Plots (part C: per-session histograms; part D: per-region bars)
# ---------------------------------------------------------------------------
def _save_or_show(fig, save_figs, fig_save_prefix, rel_path):
    import pathlib
    import matplotlib.pyplot as plt
    if save_figs and fig_save_prefix is not None:
        fp = pathlib.Path(fig_save_prefix) / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_session_histograms(pen_df, save_figs=False, fig_save_prefix=None,
                            bins=30, clip_pct=99, score_col="penalty",
                            per_trial=False):
    """Part C: one figure per session, Fast vs Slow score histograms.

    ``score_col`` selects ``"penalty"`` (global) or ``"norm_penalty"``
    (per-trial normalized). With ``per_trial=True`` the score is first averaged
    within each trial (recommended for ``norm_penalty`` -> a clean 0..1 per-trial
    disorder distribution); otherwise every neuron-trial is a sample.

    The x-range is clipped to the ``clip_pct`` percentile of the session's
    combined values for legibility (the ``penalty`` score has a long thin tail);
    the reported ``mean`` in each label is over all values (unclipped).
    """
    import matplotlib.pyplot as plt
    unit = "trials" if per_trial else "neuron-trials"
    for br_str, br_df in pen_df.groupby("BrainRegion"):
        for sess, sess_df in br_df.groupby("ShortName"):
            data = sess_df
            if per_trial:
                data = (sess_df.groupby(["trial_strategy", "TrialNumber"])
                        [score_col].mean().reset_index())
            cap = np.nanpercentile(data[score_col].values, clip_pct) \
                if len(data) else 1.0
            cap = max(float(cap), 0.5)  # floor -> never a zero-width range
            bin_edges = np.linspace(0, cap, bins + 1)
            fig, ax = plt.subplots(figsize=(8, 5))
            for strat, clr in [(STRATEGY_FAST, FAST_CLR),
                               (STRATEGY_SLOW, SLOW_CLR)]:
                vals = data[data.trial_strategy == strat][score_col].values
                if len(vals) == 0:
                    continue
                ax.hist(vals, bins=bin_edges, histtype="stepfilled", alpha=0.45,
                        color=clr, density=True,
                        label=(f"{strat}  (n={len(vals):,} {unit}, "
                               f"mean={vals.mean():.3g})"))
            ax.set_xlim(0, cap)
            ax.set_xlabel(SCORE_LABELS.get(score_col, score_col))
            ax.set_ylabel("Density")
            ax.set_title(f"{br_str} - {sess}\n"
                         f"Within-strategy trial-to-trial sequence deviation")
            ax.legend(fontsize="x-small")
            ax.spines[["right", "top"]].set_visible(False)
            _save_or_show(
                fig, save_figs, fig_save_prefix,
                f"SeqWithinDeviation/sessions/hist_{score_col}_{br_str}_{sess}.svg")


def _session_means(pen_df, score_col="penalty", group_col="trial_strategy"):
    """Per (session, ``group_col``) mean of the per-trial mean of ``score_col``."""
    if score_col not in pen_df.columns:
        raise KeyError(
            f"{score_col!r} is not a column of this dataframe (it has: "
            f"{list(pen_df.columns)}). This usually means the dataframe is stale "
            f"- rebuild it by re-running the build_penalty_df / "
            f"build_cross_penalty_df cell (Restart & Run All if %autoreload looks "
            f"out of sync).")
    trial_mean = (pen_df.groupby(["ShortName", group_col, "TrialNumber"])
                  [score_col].mean().reset_index())
    sess_mean = (trial_mean.groupby(["ShortName", group_col])
                 [score_col].mean().reset_index().rename(
                     columns={score_col: "score"}))
    return sess_mean


ALL_SCORES = ["penalty", "norm_penalty", "gap_norm_penalty", "abs_gap_penalty"]


def score_summary(pen_df, score_cols=ALL_SCORES, group_col="trial_strategy",
                  decimals=3):
    """Compact side-by-side comparison of every deviation score.

    Aggregation matches the bars: per-trial mean -> per-session mean -> mean over
    sessions. Returns a DataFrame indexed by ``(BrainRegion, group_col)`` with one
    column per score (e.g. Fast/Slow rows, or matched/cross rows for the cross
    frame with ``group_col="condition"``)."""
    region = (pen_df[["ShortName", "BrainRegion"]].drop_duplicates()
              .set_index("ShortName")["BrainRegion"])
    cols = {}
    for sc in score_cols:
        sm = _session_means(pen_df, sc, group_col)
        sm["BrainRegion"] = sm["ShortName"].map(region)
        cols[sc] = sm.groupby(["BrainRegion", group_col])["score"].mean()
    return pd.DataFrame(cols).round(decimals)


def cross_significance(cross_df, score_cols=("norm_penalty", "gap_norm_penalty"),
                       alpha=0.05):
    """Paired session-level significance of matched vs cross, per region & score.

    Sessions are the independent unit and matched/cross are paired within a
    session (same session, different trial subset). Each session contributes one
    matched and one cross value (its mean per-trial score). Shapiro-Wilk on the
    per-session *paired differences* selects the test: a paired t-test if the
    differences look normal (``p > alpha``), otherwise the Wilcoxon signed-rank
    test. Two-sided ("do the two distributions differ?").

    Returns a tidy table with the normality result, the chosen test, its
    statistic and p-value, and the two condition means (per region x score).
    """
    from scipy import stats
    if isinstance(score_cols, str):
        score_cols = [score_cols]
    reference = cross_df["reference"].iloc[0]
    rows = []
    for score_col in score_cols:
        for br, br_df in cross_df.groupby("BrainRegion"):
            sm = _session_means(br_df, score_col, group_col="condition")
            piv = sm.pivot(index="ShortName", columns="condition",
                           values="score").dropna()
            matched = piv[CONDITION_MATCHED].to_numpy()
            cross = piv[CONDITION_CROSS].to_numpy()
            diff = cross - matched
            n = len(diff)

            sh_w, sh_p = stats.shapiro(diff)
            normal = sh_p > alpha
            if normal:
                test = "paired t-test"
                res = stats.ttest_rel(cross, matched)   # two-sided
                stat = res.statistic
                pval = res.pvalue
            else:
                test = "Wilcoxon signed-rank"
                res = stats.wilcoxon(cross, matched)     # two-sided
                stat = res.statistic
                pval = res.pvalue

            rows.append(dict(
                reference=reference, BrainRegion=br, score=score_col,
                n_sessions=n, matched_mean=matched.mean(), cross_mean=cross.mean(),
                normality="Shapiro-Wilk", shapiro_W=sh_w, shapiro_p=sh_p,
                normal=normal, test=test, statistic=stat, p_value=pval,
                sig=_p_stars(pval)))
    return pd.DataFrame(rows)


def _p_stars(p):
    return ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05
            else "ns")


def _perm_scores(e, sigma, score_cols):
    """Per-trial scores for a batch of within-trial orderings.

    ``e`` = the trial's reference ranks, sorted ascending (the expected order).
    ``sigma`` = ``(n_perm, n)`` array of permutations; row ``p`` says the neuron
    expected at position ``i`` is observed at ``sigma[p, i]``, so the observed
    rank vector is ``e[sigma[p]]``. Returns a dict score -> array ``(n_perm,)``.
    Both footrules come from the same permutation, so the scores stay consistent
    with each other and with :func:`_trial_penalty`.
    """
    n = len(e)
    n_perm = len(sigma)
    global_foot = np.abs(e[None, :] - e[sigma]).sum(axis=1)          # gap-aware
    local_foot = np.abs(np.arange(n)[None, :] - sigma).sum(axis=1)   # rank-only
    max_footrule = (n * n) // 2
    max_reversal = np.abs(e - e[::-1]).sum()
    out = {}
    for sc in score_cols:
        if sc == "norm_penalty":
            out[sc] = local_foot / max_footrule if max_footrule else np.zeros(n_perm)
        elif sc == "gap_norm_penalty":
            out[sc] = global_foot / max_reversal if max_reversal else np.zeros(n_perm)
        elif sc == "abs_gap_penalty":
            out[sc] = global_foot / n            # (per-neuron mean displacement)
        else:
            raise ValueError(f"shuffle null not defined for score {sc!r}")
    return out


def _shuffled_trial_scores(e, n_perm, rng, score_cols):
    """For one trial (sorted reference ranks ``e``), the per-trial score under
    ``n_perm`` **uniformly random** within-trial orderings, for each requested
    score. Returns a dict score -> array of shape (n_perm,)."""
    sigma = np.argsort(rng.random((n_perm, len(e))), axis=1)   # random orderings
    return _perm_scores(e, sigma, score_cols)


def cross_shuffle_test(cross_df,
                       score_cols=("norm_penalty", "gap_norm_penalty"),
                       n_perm=1000, seed=0):
    """Within-trial shuffle test of the matched-vs-cross effect (vs random order).

    For each permutation, *every* trial's observed firing order is randomly
    shuffled, the per-trial score is recomputed from the trial's reference ranks,
    aggregated the same way as the bars (per-session mean -> mean over sessions),
    and the effect ``cross_mean - matched_mean`` is recomputed -> a null
    distribution of effects. The observed effect is compared against it, giving a
    z-score and a permutation p-value (one-sided cross>matched, and two-sided).

    Note this null treats trials as the resampling unit (asks "beyond random
    ordering?"), which is more liberal than the session-paired test in
    :func:`cross_significance` (which asks "consistent across sessions?"). Use
    both: the paired test for the biological replicate, this for the chance floor.
    """
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    reference = cross_df["reference"].iloc[0]
    conds = (CONDITION_MATCHED, CONDITION_CROSS)
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    rows = []
    for br, br_df in cross_df.groupby("BrainRegion"):
        # Null accumulators: score -> cond -> session -> summed (n_perm,) + counts.
        nsum = {sc: {c: defaultdict(lambda: np.zeros(n_perm)) for c in conds}
                for sc in score_cols}
        osum = {sc: {c: defaultdict(float) for c in conds} for sc in score_cols}
        cnt = {c: defaultdict(int) for c in conds}

        grp = br_df.groupby(["condition", "ShortName", "TrialNumber"], sort=False)
        for (cond, sess, _tn), g in grp:
            e = np.sort(g["ref_rank"].to_numpy())
            shuf = _shuffled_trial_scores(e, n_perm, rng, score_cols)
            for sc in score_cols:
                val = shuf[sc]
                if sc == "abs_gap_penalty":  # divide by this session's (N_ref - 1)
                    val = val / max(int(g["n_ref_neurons"].iloc[0]) - 1, 1)
                nsum[sc][cond][sess] += val
                osum[sc][cond][sess] += g[sc].mean()   # observed per-trial mean
            cnt[cond][sess] += 1

        for sc in score_cols:
            region = {}       # cond -> null (n_perm,)
            region_obs = {}   # cond -> observed scalar
            for c in conds:
                sess_null = [nsum[sc][c][s] / cnt[c][s] for s in cnt[c]]
                sess_obs = [osum[sc][c][s] / cnt[c][s] for s in cnt[c]]
                region[c] = np.mean(sess_null, axis=0)
                region_obs[c] = float(np.mean(sess_obs))
            null = region[CONDITION_CROSS] - region[CONDITION_MATCHED]
            obs = region_obs[CONDITION_CROSS] - region_obs[CONDITION_MATCHED]
            null_mean, null_std = null.mean(), null.std(ddof=1)
            z = (obs - null_mean) / null_std if null_std > 0 else np.nan
            p_one = (1 + np.sum(null >= obs)) / (n_perm + 1)
            p_two = (1 + np.sum(np.abs(null - null_mean)
                                >= abs(obs - null_mean))) / (n_perm + 1)
            rows.append(dict(
                reference=reference, BrainRegion=br, score=sc, n_perm=n_perm,
                obs_matched=region_obs[CONDITION_MATCHED],
                obs_cross=region_obs[CONDITION_CROSS],
                chance_matched=region[CONDITION_MATCHED].mean(),
                chance_cross=region[CONDITION_CROSS].mean(),
                observed_effect=obs, null_effect_mean=null_mean,
                null_effect_std=null_std, z_score=z,
                p_shuffle_1sided=p_one, p_shuffle_2sided=p_two,
                sig=_p_stars(p_two)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part F: controlled-shuffle calibration (reference order -> reversed order)
#
# cross_shuffle_test above compares the data against ONE fully-random shuffle, so
# it only has two anchors: "real" and "random" (~0.667). Here the shuffle is made
# *continuous*, which lets an observed disorder be read back off the curve as an
# equivalent shuffle level ("as disordered as inverting ~19% of the neuron pairs").
#
# Mechanism: the Mallows phi-model, where the level ``s`` is the expected fraction
# of neuron PAIRS inverted with respect to the reference order (the normalized
# Kendall distance). It is the interpolation whose midpoint is exactly the uniform
# random case, so all three anchors are exact by construction:
#   s = 0   -> phi = 0 -> identity           -> gap-aware disorder 0
#   s = 0.5 -> phi = 1 -> uniform random     -> disorder ~= 0.667
#   s = 1   -> phi = 0, then reversed        -> exact reversal -> disorder 1
# and s > 0.5 reads as "more reversed than chance".
# ---------------------------------------------------------------------------

# 1% steps to 10%, 2% to 20%, 5% to 100% -> 32 levels, as fractions.
SHUFFLE_LEVELS = np.unique(np.concatenate([
    np.arange(0, 11, 1), np.arange(10, 21, 2), np.arange(20, 101, 5),
])) / 100.0


def _mallows_expected_d(phi, n):
    """Expected Kendall distance from the reference order under Mallows(``phi``).

    Under repeated insertion, item ``j`` adds ``Z_j`` in ``{0..j-1}`` inversions
    with ``P(Z_j = z) ∝ phi^z``, so ``E[d] = sum_j E[Z_j]``. Monotone increasing
    in ``phi``: 0 at ``phi=0`` (identity), ``n(n-1)/4`` at ``phi=1`` (uniform,
    i.e. exactly half of ``max_d = n(n-1)/2``).
    """
    total = 0.0
    for j in range(1, n + 1):
        z = np.arange(j)
        w = np.asarray(float(phi)) ** z      # phi=0 -> [1, 0, 0, ...] -> E=0
        total += float((z * w).sum() / w.sum())
    return total


def _mallows_phi(n, level, iters=80):
    """Dispersion ``phi`` in [0, 1] whose expected Kendall distance is
    ``level * max_d``. ``level`` must be <= 0.5 (that is the ``phi <= 1`` half);
    higher levels are produced by sampling at ``1 - level`` and reversing."""
    if n < 2 or level <= 0:
        return 0.0
    target = level * n * (n - 1) / 2.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _mallows_expected_d(mid, n) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _rim_perms(n, phi, size, rng):
    """``size`` Mallows(``phi``) permutations by repeated insertion, vectorized
    over the ``size`` axis (the ``n`` insertion steps stay a loop).

    Item ``j`` is inserted at position ``j - z`` of the list built so far with
    ``P(z) ∝ phi^z`` -- ``z`` is exactly the number of inversions it adds.
    ``phi=0`` -> identity; ``phi=1`` -> uniform. Returns ``(size, n)``.
    """
    perm = np.zeros((size, 1), dtype=int)
    rows = np.arange(size)[:, None]
    for j in range(2, n + 1):
        w = np.asarray(float(phi)) ** np.arange(j)
        cdf = np.cumsum(w / w.sum())
        z = np.minimum(np.searchsorted(cdf, rng.random(size)), j - 1)
        pos = (j - 1 - z)[:, None]           # 0-based insertion slot, 0..j-1
        idx = np.arange(j)[None, :]
        src = np.clip(np.where(idx < pos, idx, idx - 1), 0, j - 2)
        perm = np.where(idx == pos, j - 1, perm[rows, src])
    return perm


def _perm_pool(n, level, rng, pool_size, cache):
    """Cached pool of ``pool_size`` shuffle permutations for ``(n, level)``.

    The permutations depend only on ``n`` and ``level``, never on the trial's rank
    values, so one pool serves every trial with that active count -- which is what
    makes the calibration tractable (a few hundred pools instead of one sampling
    per trial). Levels above 0.5 are sampled at ``1 - level`` and reversed;
    reversing inverts every pair (``d -> max_d - d``), so ``level=1`` comes out an
    exact reversal.
    """
    key = (n, round(float(level), 6))
    if key not in cache:
        flip = level > 0.5
        phi = _mallows_phi(n, (1.0 - level) if flip else level)
        pool = _rim_perms(n, phi, pool_size, rng)
        cache[key] = pool[:, ::-1].copy() if flip else pool
    return cache[key]


def shuffle_calibration(df, score_col="gap_norm_penalty",
                        levels=SHUFFLE_LEVELS, n_rep=100, seed=0,
                        pool_size=500, show_progress=True,
                        trial_keys=("condition", "TrialNumber")):
    """Per-session calibration curve: disorder vs controlled shuffle level.

    For every session and every level in ``levels``, each of the session's trials
    has its firing order re-drawn ``n_rep`` times from the Mallows model at that
    level and re-scored; the scores are averaged over trials exactly like the bars
    (per-trial -> per-session), giving ``n_rep`` session values per level.

    The curve is built **per session** from that session's own trials (its real
    active-set sizes and rank gaps). ``trial_keys`` identifies a trial within a
    session: ``("condition", "TrialNumber")`` for a cross frame (pools both
    conditions), or ``("trial_strategy", "TrialNumber")`` for the within-strategy
    ``penalty_df`` (pools Fast and Slow). A ``reference`` column, if present, is
    carried through (the reference direction changes the rank values).

    Returns a long df ``[reference?, BrainRegion, ShortName, level, rep, score]``;
    it is the expensive step, so cache it (the notebook saves it to a pickle).
    """
    from tqdm.auto import tqdm
    rng = np.random.default_rng(seed)
    reference = df["reference"].iloc[0] if "reference" in df.columns else None
    levels = np.asarray(levels, dtype=float)
    trial_keys = list(trial_keys)
    cache = {}
    rows = []

    groups = list(df.groupby(["BrainRegion", "ShortName"], sort=False))
    desc = f"shuffle calibration ({reference or 'within'} ref, {score_col})"
    for (br, sess), sdf in tqdm(groups, disable=not show_progress, desc=desc):
        # One (ranks, n_ref) per trial; trials pooled across the non-TrialNumber key.
        trials = [(np.sort(g["ref_rank"].to_numpy()),
                   int(g["n_ref_neurons"].iloc[0]))
                  for _key, g in sdf.groupby(trial_keys, sort=False)]
        if not trials:
            continue
        for level in levels:
            acc = np.zeros(n_rep)
            for e, n_ref in trials:
                pool = _perm_pool(len(e), level, rng, pool_size, cache)
                sigma = pool[rng.integers(0, len(pool), n_rep)]
                val = _perm_scores(e, sigma, [score_col])[score_col]
                if score_col == "abs_gap_penalty":
                    val = val / max(n_ref - 1, 1)
                acc += val
            acc /= len(trials)
            rows.extend(dict(reference=reference, BrainRegion=br, ShortName=sess,
                             level=float(level), rep=r, score=float(v))
                        for r, v in enumerate(acc))
    return pd.DataFrame(rows)


def _invert_curve(levels, curve_y, y):
    """Read the shuffle level (in %) off a monotone calibration curve at disorder
    ``y``. ``np.interp`` clamps ``y`` outside the curve's range."""
    levels = np.asarray(levels, dtype=float)
    curve_y = np.asarray(curve_y, dtype=float)
    order = np.argsort(curve_y)              # guard against tiny non-monotonicity
    return np.interp(y, curve_y[order], levels[order] * 100.0)


def shuffle_level_summary(df, calib, score_col="gap_norm_penalty",
                          group_col="trial_strategy"):
    """Express each session's observed disorder as a Mallows shuffle level (%).

    The gap-aware ``score_col`` is monotone-remapped through the region's mean
    calibration curve (:func:`shuffle_calibration`) so that 0% = reference order,
    **50% = random/chance**, 100% = fully reversed. Because the Mallows level is
    literally the fraction of neuron pairs inverted vs the reference, the number is
    directly interpretable (e.g. "16%": 16% of pairs out of order, well under the
    50% chance mark -> still ordered like the reference, not reversed).

    ``df`` is a within-strategy (``group_col="trial_strategy"``) or cross
    (``group_col="condition"``) penalty frame; ``calib`` is a matching
    :func:`shuffle_calibration` output. Returns ``[ShortName, BrainRegion,
    group_col, score, shuffle_pct]`` (one row per session x group)."""
    obs = _session_means(df, score_col, group_col)     # ShortName, group_col, score
    region_of = (df[["ShortName", "BrainRegion"]].drop_duplicates()
                 .set_index("ShortName")["BrainRegion"])
    obs["BrainRegion"] = obs["ShortName"].map(region_of)
    reg_curve = (calib.groupby(["BrainRegion", "level"])["score"].mean()
                 .reset_index())
    parts = []
    for br, g in obs.groupby("BrainRegion"):
        cur = reg_curve[reg_curve.BrainRegion == br].sort_values("level")
        g = g.copy()
        g["shuffle_pct"] = _invert_curve(cur["level"].to_numpy(),
                                         cur["score"].to_numpy(),
                                         g["score"].to_numpy())
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def plot_shuffle_level_bars(df, brain_region, calib, group_col="trial_strategy",
                            score_col="gap_norm_penalty", save_figs=False,
                            fig_save_prefix=None):
    """Chance-centered bars: the Fast-vs-Slow (``group_col="trial_strategy"``) or
    matched-vs-cross (``group_col="condition"``) comparison re-expressed on the
    Mallows shuffle-level axis via :func:`shuffle_level_summary`.

    Bars = mean over sessions of each session's shuffle-equivalent %, with a
    per-session connecting line and a dashed **random = 50%** reference; the y-axis
    runs 0 (reference order) to 100 (fully reversed). ``calib`` must match ``df``
    (built with the same ``reference`` for the cross case, or from ``penalty_df``
    with ``trial_keys=("trial_strategy","TrialNumber")`` for the within case)."""
    import matplotlib.pyplot as plt
    br_df = df[df.BrainRegion == brain_region]
    if len(br_df) == 0:
        return
    summ = shuffle_level_summary(br_df, calib, score_col, group_col)
    pivot = summ.pivot(index="ShortName", columns=group_col, values="shuffle_pct")

    if group_col == "trial_strategy":
        order = [STRATEGY_FAST, STRATEGY_SLOW]
        clr = {STRATEGY_FAST: FAST_CLR, STRATEGY_SLOW: SLOW_CLR}
        lab = {STRATEGY_FAST: STRATEGY_FAST, STRATEGY_SLOW: STRATEGY_SLOW}
        subtitle = "Within-strategy trial-to-trial variability"
    else:  # condition
        reference = br_df["reference"].iloc[0]
        other = STRATEGY_SLOW if reference == STRATEGY_FAST else STRATEGY_FAST
        order = [CONDITION_MATCHED, CONDITION_CROSS]
        clr = {CONDITION_MATCHED: (FAST_CLR if reference == STRATEGY_FAST else SLOW_CLR),
               CONDITION_CROSS: (SLOW_CLR if reference == STRATEGY_FAST else FAST_CLR)}
        lab = {CONDITION_MATCHED: f"{reference} (matched)",
               CONDITION_CROSS: f"{other} (cross)"}
        subtitle = f"Do {other} trials follow the {reference} sequence?"
    xpos = {g: i + 1 for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=(5, 6))
    for g in order:
        if g not in pivot.columns:
            continue
        vals = pivot[g].dropna()
        ax.bar(xpos[g], vals.mean(), yerr=vals.sem(), width=0.5, color=clr[g],
               alpha=0.85, capsize=4,
               label=(f"{lab[g]} ({len(vals)} sessions, "
                      f"{vals.mean():.0f}%±{vals.sem():.0f})"))

    cols = [g for g in order if g in pivot.columns]
    for _sess, row in pivot[cols].iterrows():
        xs = [xpos[g] for g in cols if not np.isnan(row[g])]
        ys = [row[g] for g in cols if not np.isnan(row[g])]
        if len(xs) >= 2:
            ax.plot(xs, ys, color="gray", alpha=0.3, lw=1, linestyle=":",
                    marker="o", markerfacecolor="none")

    ax.axhline(50, ls="--", color="0.4", lw=1)
    ax.text(0.55, 50, "random (chance)", va="bottom", ha="left",
            fontsize="x-small", color="0.35")
    ax.set_ylim(0, 100)
    ax.set_xticks([xpos[g] for g in cols])
    ax.set_xticklabels([lab[g] for g in cols])
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_ylabel("Disorder as shuffle level  (% of neuron pairs inverted "
                  "vs reference)\n0 = reference order,  50 = random,  100 = reversed")
    ax.set_title(f"{brain_region} - {subtitle}\n(chance-centered gap-aware disorder)")
    ax.legend(fontsize="x-small")
    ax.spines[["right", "top"]].set_visible(False)
    _save_or_show(
        fig, save_figs, fig_save_prefix,
        f"SeqWithinDeviation/shufflelevel_{group_col}_{score_col}_{brain_region}.svg")


def _session_shuffle_means(cross_df, score_col="norm_penalty", n_perm=1000,
                           seed=0, group_col="condition"):
    """Per-session chance level of ``score_col`` under random within-trial order.

    For every trial, average the shuffled score over ``n_perm`` random orderings
    (the same ``n_perm``-and-mean the shuffle test uses), then average over the
    session's trials. Mirrors :func:`_session_means` so the resulting bar is
    aggregated identically to the matched/cross bars. Grouped by ``group_col``
    (``"condition"``); pass ``group_col=None`` to pool all of a session's trials
    into one chance value. Returns ``[ShortName, group_col?, score]``."""
    rng = np.random.default_rng(seed)
    keys = ["ShortName", "TrialNumber"] if group_col is None \
        else ["ShortName", group_col, "TrialNumber"]
    per_trial = []
    for key, g in cross_df.groupby(keys, sort=False):
        e = np.sort(g["ref_rank"].to_numpy())
        v = _shuffled_trial_scores(e, n_perm, rng, [score_col])[score_col].mean()
        if score_col == "abs_gap_penalty":  # divide by this session's (N_ref - 1)
            v = v / max(int(g["n_ref_neurons"].iloc[0]) - 1, 1)
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["score"] = float(v)
        per_trial.append(row)
    if not per_trial:
        out_cols = [c for c in keys if c != "TrialNumber"] + ["score"]
        return pd.DataFrame(columns=out_cols)
    trial_df = pd.DataFrame(per_trial)
    sess_keys = [c for c in keys if c != "TrialNumber"]
    return (trial_df.groupby(sess_keys)["score"].mean().reset_index())


def plot_region_bars(pen_df, brain_region, save_figs=False,
                     fig_save_prefix=None, score_col="penalty"):
    """Part D: two bars (Fast, Slow) = mean over sessions of the session mean
    per-trial ``score_col``, with a dotted circle-marked line per session
    connecting its Fast and Slow averages. Call once per brain region."""
    import matplotlib.pyplot as plt
    br_df = pen_df[pen_df.BrainRegion == brain_region]
    sess_mean = _session_means(br_df, score_col)
    pivot = sess_mean.pivot(index="ShortName", columns="trial_strategy",
                            values="score")
    xpos = {STRATEGY_FAST: 1, STRATEGY_SLOW: 2}

    fig, ax = plt.subplots(figsize=(5, 6))
    for strat, clr in [(STRATEGY_FAST, FAST_CLR), (STRATEGY_SLOW, SLOW_CLR)]:
        if strat not in pivot.columns:
            continue
        vals = pivot[strat].dropna()
        ax.bar(xpos[strat], vals.mean(), yerr=vals.sem(), width=0.5,
               color=clr, alpha=0.85, capsize=4,
               label=(f"{strat} ({len(vals)} sessions, "
                      f"{vals.mean():.3g}±{vals.sem():.2g})"))

    if STRATEGY_FAST in pivot.columns and STRATEGY_SLOW in pivot.columns:
        for _sess, row in pivot.iterrows():
            f, s = row.get(STRATEGY_FAST, np.nan), row.get(STRATEGY_SLOW, np.nan)
            if np.isnan(f) or np.isnan(s):
                continue
            ax.plot([xpos[STRATEGY_FAST], xpos[STRATEGY_SLOW]], [f, s],
                    color="gray", alpha=0.4, lw=1, linestyle=":",
                    marker="o", markerfacecolor="none")

    ax.set_xticks([xpos[STRATEGY_FAST], xpos[STRATEGY_SLOW]])
    ax.set_xticklabels([STRATEGY_FAST, STRATEGY_SLOW])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylabel("Mean per-trial " + SCORE_LABELS.get(score_col, score_col))
    ax.set_title(f"{brain_region} - Within-strategy sequence deviation\n"
                 f"(per-session means; line = one session)")
    ax.legend(fontsize="x-small")
    ax.spines[["right", "top"]].set_visible(False)
    _save_or_show(fig, save_figs, fig_save_prefix,
                  f"SeqWithinDeviation/bars_{score_col}_{brain_region}.svg")


CONDITION_SHUFFLE = "shuffled"   # random within-trial order (chance floor)
SHUFFLE_CLR = "0.6"


def plot_cross_bars(cross_df, brain_region, save_figs=False,
                    fig_save_prefix=None, score_col="norm_penalty",
                    show_shuffle=True, n_perm=1000, shuffle_seed=0):
    """"Same sequence?" bars: matched (reference strategy's own trials) vs cross
    (the other strategy's trials), both scored against the reference order on the
    common-neuron set. ``cross_df`` comes from :func:`build_cross_penalty_df` for
    one ``reference``. Call once per brain region.

    With ``show_shuffle`` (default) a third bar shows the **chance** level: each
    trial's score averaged over ``n_perm`` random within-trial orderings, then
    averaged per session (pooling both conditions' trials) -- the ceiling both
    real bars should sit well below if the firing order is reproducible. The
    per-session dotted lines extend to this chance point so each session's
    matched -> cross -> chance gap is visible."""
    import matplotlib.pyplot as plt
    br_df = cross_df[cross_df.BrainRegion == brain_region]
    reference = br_df["reference"].iloc[0]
    other = STRATEGY_SLOW if reference == STRATEGY_FAST else STRATEGY_FAST
    clr = {CONDITION_MATCHED: (FAST_CLR if reference == STRATEGY_FAST else SLOW_CLR),
           CONDITION_CROSS: (SLOW_CLR if reference == STRATEGY_FAST else FAST_CLR),
           CONDITION_SHUFFLE: SHUFFLE_CLR}
    xtick = {CONDITION_MATCHED: f"{reference} trials",
             CONDITION_CROSS: f"{other} trials",
             CONDITION_SHUFFLE: f"shuffled\n({n_perm} perms)"}
    legend = {CONDITION_MATCHED: f"matched: {xtick[CONDITION_MATCHED]}",
              CONDITION_CROSS: f"cross: {xtick[CONDITION_CROSS]}",
              CONDITION_SHUFFLE: "shuffled order (chance)"}

    sess_mean = _session_means(br_df, score_col, group_col="condition")
    pivot = sess_mean.pivot(index="ShortName", columns="condition", values="score")
    xpos = {CONDITION_MATCHED: 1, CONDITION_CROSS: 2}
    order = [CONDITION_MATCHED, CONDITION_CROSS]
    if show_shuffle:
        chance = _session_shuffle_means(br_df, score_col, n_perm, shuffle_seed,
                                        group_col=None).set_index("ShortName")["score"]
        pivot[CONDITION_SHUFFLE] = chance
        xpos[CONDITION_SHUFFLE] = 3
        order.append(CONDITION_SHUFFLE)

    fig, ax = plt.subplots(figsize=(6 if show_shuffle else 5, 6))
    for cond in order:
        if cond not in pivot.columns:
            continue
        vals = pivot[cond].dropna()
        ax.bar(xpos[cond], vals.mean(), yerr=vals.sem(), width=0.5,
               color=clr[cond], alpha=0.85, capsize=4,
               hatch=("//" if cond == CONDITION_SHUFFLE else None),
               label=(f"{legend[cond]} ({len(vals)} sessions, "
                      f"{vals.mean():.3g}±{vals.sem():.2g})"))

    line_cols = [c for c in order if c in pivot.columns]
    for _sess, row in pivot[line_cols].iterrows():
        xs = [xpos[c] for c in line_cols if not np.isnan(row[c])]
        ys = [row[c] for c in line_cols if not np.isnan(row[c])]
        if len(xs) >= 2:
            ax.plot(xs, ys, color="gray", alpha=0.4, lw=1, linestyle=":",
                    marker="o", markerfacecolor="none")

    ax.set_xticks([xpos[c] for c in line_cols])
    ax.set_xticklabels([xtick[c] for c in line_cols])
    ax.set_xlim(0.5, (3.5 if show_shuffle else 2.5))
    ax.set_ylabel("Mean per-trial " + SCORE_LABELS.get(score_col, score_col))
    ax.set_title(f"{brain_region} - Do {other} trials follow the {reference} "
                 f"sequence?\n(common-active neurons; both vs {reference} reference)")
    ax.legend(fontsize="x-small")
    ax.spines[["right", "top"]].set_visible(False)
    _save_or_show(
        fig, save_figs, fig_save_prefix,
        f"SeqWithinDeviation/cross_{reference}ref_{score_col}_{brain_region}.svg")


def plot_shuffle_calibration(cross_df, brain_region, calib,
                             score_col="gap_norm_penalty", save_figs=False,
                             fig_save_prefix=None, recenter=False):
    """Part F: disorder vs a controlled shuffle from reference order (0%) to
    reversed order (100%), with each session's *observed* Fast/Slow disorder read
    back off the calibration as an equivalent shuffle level.

    ``calib`` is the output of :func:`shuffle_calibration` for the same
    ``reference`` as ``cross_df``. One figure per (region, reference):

    - grey background = every (level, score) sample for the region (sessions x
      reps); the region-mean curve is drawn on top;
    - dashed line at the empirical fully-random level (the region curve at 50%);
    - red = Fast, gold = Slow: one point per session, x inferred by inverting that
      session's own curve at its observed disorder, joined per session;
    - big red/gold dots = mean +/- sem over sessions, x from the region-mean curve.

    The x-axis therefore reads as "how far toward reversal", with 50% = chance.

    ``recenter`` (Part F3): pass the y-axis through the same Mallows recentering as
    the F2 bars, i.e. remap raw disorder -> shuffle-equivalent % so **random sits at
    50% on the y-axis too**. Because the recentering uses this very calibration, the
    curve becomes the straight diagonal y = x (random dead-centre at (50, 50)) and
    the observed dots land on it at their F2 shuffle %. It is the linearized
    consistency view of F1, on a scale shared with F2.
    """
    import matplotlib.pyplot as plt
    reference = cross_df["reference"].iloc[0]
    br_df = cross_df[cross_df.BrainRegion == brain_region]
    cal = calib[calib.BrainRegion == brain_region]
    if len(br_df) == 0 or len(cal) == 0:
        return

    # Per-level spread across sessions: average reps within a session first, so the
    # unit is the session (matching the mean+/-sem dots). NB the across-session SEM
    # is ~0.002 (the sessions agree almost exactly on the calibration), i.e. thinner
    # than the curve line -> invisible; show the session min-max envelope and +/-SD,
    # which is the spread you can actually see.
    sess_level = cal.groupby(["ShortName", "level"])["score"].mean().reset_index()
    lvl_stats = (sess_level.groupby("level")["score"]
                 .agg(["mean", "std", "min", "max"]).sort_index())
    levels = lvl_stats.index.to_numpy()
    reg_y = lvl_stats["mean"].to_numpy()
    reg_sd = np.nan_to_num(lvl_stats["std"].to_numpy())     # NaN if a single session
    reg_lo, reg_hi = lvl_stats["min"].to_numpy(), lvl_stats["max"].to_numpy()
    rand_y = float(np.interp(0.5, levels, reg_y))    # 50% == uniform random

    # Optional Mallows recentering of the y-axis (F3): remap raw disorder to its
    # shuffle-equivalent % via this region's own curve, so random -> 50 on y too.
    if recenter:
        def ty(v):
            return _invert_curve(levels, reg_y, v)
    else:
        def ty(v):
            return np.asarray(v, dtype=float)

    obs = _session_means(br_df, score_col, group_col="trial_strategy")
    obs_piv = obs.pivot(index="ShortName", columns="trial_strategy", values="score")
    sess_curve = {s: g.groupby("level")["score"].mean().sort_index()
                  for s, g in cal.groupby("ShortName")}
    clr = {STRATEGY_FAST: FAST_CLR, STRATEGY_SLOW: SLOW_CLR}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    # Across-session spread bands (connected across shuffle levels via fill_between):
    # light = full session min-max envelope, darker = +/-SD; region-mean curve on top.
    ax.fill_between(levels * 100, ty(reg_lo), ty(reg_hi), color="0.6", alpha=0.18,
                    linewidth=0, zorder=1, label="session min-max")
    ax.fill_between(levels * 100, ty(reg_y - reg_sd), ty(reg_y + reg_sd), color="0.5",
                    alpha=0.35, linewidth=0, zorder=1, label="+/-SD across sessions")
    ax.plot(levels * 100, ty(reg_y), color="0.3", lw=1.3, zorder=2,
            label="calibration (region mean)")
    rand_lab = "fully random order = 50%" if recenter \
        else f"fully random order = {rand_y:.2f}"
    ax.axhline(float(ty(rand_y)), ls="--", color="0.4", lw=1, zorder=2)
    ax.text(1, float(ty(rand_y)), rand_lab, va="bottom", ha="left",
            fontsize="x-small", color="0.3")

    # Per-session points, x inferred from each session's own calibration curve;
    # y is the observed disorder (recentered to shuffle % when ``recenter``).
    sess_x = {STRATEGY_FAST: [], STRATEGY_SLOW: []}
    sess_y = {STRATEGY_FAST: [], STRATEGY_SLOW: []}
    for sess, row in obs_piv.iterrows():
        cur = sess_curve.get(sess)
        if cur is None:
            continue
        lv, cy = cur.index.to_numpy(), cur.to_numpy()
        pts = {}
        for strat in (STRATEGY_FAST, STRATEGY_SLOW):
            y = row.get(strat, np.nan)
            if np.isnan(y):
                continue
            x = float(_invert_curve(lv, cy, y))
            pts[strat] = (x, float(ty(y)))
            sess_x[strat].append(x)
            sess_y[strat].append(float(ty(y)))
        if len(pts) == 2:
            xs, ys = zip(pts[STRATEGY_FAST], pts[STRATEGY_SLOW])
            ax.plot(xs, ys, color="gray", alpha=0.3, lw=0.8, ls=":", zorder=2)

    # Global mean: transparent, BEHIND the session dots, with x- and y-SEM (small
    # marker so both error bars show). x-SEM is the spread of the per-session
    # inverted shuffle levels; y-SEM the spread of the observed disorders.
    for strat in (STRATEGY_FAST, STRATEGY_SLOW):
        xs, ys = np.asarray(sess_x[strat]), np.asarray(sess_y[strat])
        if len(xs) == 0:
            continue
        mx, my, n = xs.mean(), ys.mean(), len(xs)
        xsem = xs.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        lab = (f"{strat}: {mx:.0f}±{xsem:.0f}% shuffle ({n} sessions)" if recenter
               else f"{strat}: {my:.3g} disorder = "
                    f"{mx:.0f}±{xsem:.0f}% shuffle ({n} sessions)")
        ax.errorbar(mx, my, xerr=xsem, fmt="o", ms=13, color=clr[strat],
                    ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5,
                    markeredgecolor="black", markeredgewidth=0.8, alpha=0.5,
                    zorder=4, label=lab)

    # Session dots: opaque, thin black edge, IN FRONT of the mean marker.
    for strat in (STRATEGY_FAST, STRATEGY_SLOW):
        if sess_x[strat]:
            ax.scatter(sess_x[strat], sess_y[strat], s=26, color=clr[strat],
                       alpha=1.0, edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xlim(-2, 102)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlabel("Rank-shuffle level (% of neuron pairs inverted vs reference)")
    other = STRATEGY_SLOW if reference == STRATEGY_FAST else STRATEGY_FAST
    if recenter:
        ax.set_ylim(-2, 102)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_ylabel("Disorder recentered to shuffle level (%)\n"
                      "0 = reference order,  50 = random,  100 = reversed")
        ax.set_title(f"{brain_region} - {reference} reference: calibration "
                     f"recentered so random = 50%\n(y remapped like the F2 bars; "
                     f"curve collapses to y = x)")
    else:
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.set_ylabel("Mean per-trial " + SCORE_LABELS.get(score_col, score_col))
        ax.set_title(f"{brain_region} - {reference} reference: observed disorder as "
                     f"a shuffle level\n(red = Fast, gold = Slow; x read off each "
                     f"session's calibration)")
    # Vertical descriptor labels under the 0% and 100% ends of the axis.
    tr = ax.get_xaxis_transform()
    ax.text(0, -0.07, "No rank deviation", transform=tr, rotation=90,
            ha="center", va="top", fontsize="x-small", color="0.3")
    ax.text(100, -0.07, "Reversed rank deviation", transform=tr, rotation=90,
            ha="center", va="top", fontsize="x-small", color="0.3")
    ax.legend(fontsize="x-small", loc="upper left")
    ax.spines[["right", "top"]].set_visible(False)
    fig.subplots_adjust(bottom=0.24)
    suffix = "_recentered" if recenter else ""
    _save_or_show(
        fig, save_figs, fig_save_prefix,
        f"SeqWithinDeviation/shuffle_calib_{reference}ref_{score_col}"
        f"_{brain_region}{suffix}.svg")
