"""Movement-time neurons: how many neurons peak *after* movement onset.

Backend for the ``# Movement time neurons`` section of ``2pAnalysis.ipynb``.

A neuron's firing position is its **median peak sample** across the trials in
which it was active (``max_idxs`` in the ``max_firing_*_df`` frames built by the
``# Sequence Extracation`` section). The sampling-aligned traces carry three
epochs, ``epochs_ranges = [(0, 2), (3, 29), (30, 32)]`` =
``['-0.1s Sampling', 'Sampling', 'Movement to Lateral Port']``, so "fires after
movement starts" means the median peak falls inside that last epoch.

Each method (``METHOD_SPECS``) is a **criterion** x **window** pair, all built
from the same per-neuron ``max_idxs``.

The window is the trailing ``n_last_bins`` bins of :func:`maxFiringHist`. With
the standard ``BINS_COUNT = 7`` the bins are ``[pre-sampling] + 5 x [sampling] +
[movement]``, so:

- ``n_last_bins=1`` (``WINDOW_MOVEMENT``) is the 7th bin = the movement epoch,
  samples ``(30, 33]``;
- ``n_last_bins=2`` (``WINDOW_INITIATION_MOVEMENT``) prepends the 6th bin,
  ``(24.6, 30]`` -- the last ~0.18 s of sampling, where movement is initiated
  (the very window ``makeActive`` selects its "movement" set on) -- giving
  "initiation **or** movement".

The criterion is one of:

``CRITERION_MEDIAN_BIN`` (``"last_bin"``, ``"last_two_bins"``)
    The neuron's median peak falls in the window's bins -- the requested
    definition. Note ``pd.cut`` bins are left-open, so a median sitting exactly
    on the window's first edge (30, resp. 24.6) falls in the bin *before* it.

``CRITERION_MEDIAN_THRESH`` (``"movement_epoch"``, ``"initiation_onset"``)
    The neuron's median peak is ``>=`` the window's first sample. Same idea
    without the binning edge effect, and for the movement window it reads the
    epoch straight off the data the way ``_setupAxs`` / ``makeActive`` do.
    Differs from the bin criterion only for medians exactly on that edge.

``CRITERION_Q1_THRESH`` (``"iqr_q1"``, ``"iqr_q1_two_bins"``)
    Stricter, and built the way ``makeActive`` selects its neuron sets: the
    **q1** of the per-trial peaks (from :func:`extractIQR`) is at/after the
    window's first sample, i.e. the neuron peaks inside the window in >=75% of
    its active trials -- not merely in the median trial.

``MOVEMENT_METHODS`` is the default run list (both windows under the bin
criterion, plus the two robustness checks on the movement-only window); any
other combination in ``METHOD_SPECS`` can be requested via ``methods=``.

Percentages are always reported as **mean +/- SEM over sessions** (a session is
the independent unit, as in the ``_plotPrevModulatedPie`` / Venn sections), with
every session's denominator being *all* of that session's neurons.

The choice-selective restriction reuses the notebook's own significance table:
pass the ``long_trace_id`` set from :func:`getSgfNeurons`
(``run_data_unnormalized[0].shortlong_df``, ``data_col="ChoiceLeft"``, no prior)
and the per-session counts gain the intersection "movement **and**
choice-selective" -- again as a percentage of all neurons.

MFC (``M2_Bi``) vs LFC (``ALM_Bi``) is compared on those per-session
percentages, and the test is chosen by a normality check that runs *first*:
Shapiro-Wilk on each region's per-session values (the two regions are
independent samples of different sessions, so there are no paired differences to
test). Both normal -> two-sample t-test, Levene's test picking Student vs Welch;
otherwise Mann-Whitney U, the notebook's ``defaultStatsTestFn``. Both p-values
are always reported so the gate's choice can be second-guessed.
"""

import numpy as np
import pandas as pd

from ..common.clr import BrainRegion as BRClr
from ..common.definitions import BrainRegion
from .seqdeviation import extractIQR, region_label
# Re-exported so this module stays the one import the notebook section needs.
from .statstest import (NORMALITY_ALPHA, TEST_MWU, TEST_TTEST_STUDENT,  # noqa: F401
                        TEST_TTEST_WELCH, normalityGatedTest)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Same default as ``BINS_COUNT`` in 2pAnalysis.ipynb.
DEFAULT_BINS_COUNT = 7

# How a neuron's peak position is turned into a yes/no ...
CRITERION_MEDIAN_BIN = "median_bin"      # median peak inside the window's bins
CRITERION_MEDIAN_THRESH = "median_thresh"  # median peak >= the window's start
CRITERION_Q1_THRESH = "q1_thresh"        # q1 of the peaks >= the window's start

# ... over which window: the trailing ``n_last_bins`` bins. 1 = the movement bin
# only; 2 = the movement-initiation bin (the last sampling bin, ~0.18 s before
# movement onset -- the same window ``makeActive`` selects on) plus the movement
# bin.
WINDOW_MOVEMENT = 1
WINDOW_INITIATION_MOVEMENT = 2

METHOD_LAST_BIN = "last_bin"
METHOD_LAST_TWO_BINS = "last_two_bins"
METHOD_MOVEMENT_EPOCH = "movement_epoch"
METHOD_INITIATION_ONSET = "initiation_onset"
METHOD_IQR_Q1 = "iqr_q1"
METHOD_IQR_Q1_TWO_BINS = "iqr_q1_two_bins"

#: Every supported method -> ``(criterion, n_last_bins)``. Add a combination
#: here to make it available; ``MOVEMENT_METHODS`` is only the default run list.
METHOD_SPECS = {
    METHOD_LAST_BIN: (CRITERION_MEDIAN_BIN, WINDOW_MOVEMENT),
    METHOD_LAST_TWO_BINS: (CRITERION_MEDIAN_BIN, WINDOW_INITIATION_MOVEMENT),
    METHOD_MOVEMENT_EPOCH: (CRITERION_MEDIAN_THRESH, WINDOW_MOVEMENT),
    METHOD_INITIATION_ONSET: (CRITERION_MEDIAN_THRESH,
                              WINDOW_INITIATION_MOVEMENT),
    METHOD_IQR_Q1: (CRITERION_Q1_THRESH, WINDOW_MOVEMENT),
    METHOD_IQR_Q1_TWO_BINS: (CRITERION_Q1_THRESH, WINDOW_INITIATION_MOVEMENT),
}

#: Run by default: the two windows under the median-in-bin criterion, plus the
#: bin-edge and the strict (q1) robustness checks on the movement-only window.
#: Pass ``methods=`` to run any other subset of ``METHOD_SPECS``.
MOVEMENT_METHODS = (METHOD_LAST_BIN, METHOD_LAST_TWO_BINS,
                    METHOD_MOVEMENT_EPOCH, METHOD_IQR_Q1)

METHOD_LABELS = {
    METHOD_LAST_BIN: "median peak during movement bin",
    METHOD_LAST_TWO_BINS: "median peak during initiation or movement bin",
    METHOD_MOVEMENT_EPOCH: "median peak at/after movement onset",
    METHOD_INITIATION_ONSET: "median peak at/after initiation onset",
    METHOD_IQR_Q1: "peak after movement onset in >=75% of active trials (q1)",
    METHOD_IQR_Q1_TWO_BINS: ("peak after initiation onset in >=75% of active "
                             "trials (q1)"),
}

# Percentages carried per session. All of them share the same denominator (all
# of the session's neurons) except ``prcnt_choice_of_movement``.
METRIC_MOVEMENT = "prcnt_movement"
METRIC_MOVEMENT_CHOICE = "prcnt_movement_choice"
# No movement restriction, so this is the same per-session mean +/- SEM that the
# "Sgf. Neurons Pie Charts" section already plots (loopPlotPieChart, ChoiceLeft
# = "Cur. Direction"). Kept as the input to the intersection above / as a check,
# not as a result of its own.
METRIC_CHOICE = "prcnt_choice"
METRIC_CHOICE_OF_MOVEMENT = "prcnt_choice_of_movement"
DEFAULT_METRICS = (METRIC_MOVEMENT, METRIC_MOVEMENT_CHOICE)
ALL_METRICS = (METRIC_MOVEMENT, METRIC_MOVEMENT_CHOICE, METRIC_CHOICE,
               METRIC_CHOICE_OF_MOVEMENT)

#: What the window's neurons are called, per ``n_last_bins``.
WINDOW_LABELS = {
    WINDOW_MOVEMENT: "Movement-time",
    WINDOW_INITIATION_MOVEMENT: "Movement-initiation + movement",
}

_METRIC_LABEL_TEMPLATES = {
    METRIC_MOVEMENT: "{window} neurons (% of all neurons)",
    METRIC_MOVEMENT_CHOICE: ("{window} AND choice-selective neurons "
                             "(% of all neurons)"),
    METRIC_CHOICE: "Choice-selective neurons (% of all neurons)",
    METRIC_CHOICE_OF_MOVEMENT: ("Choice-selective neurons "
                                "(% of {window_lower} neurons)"),
}

#: Window-neutral labels (the movement-only wording), for contexts with no
#: method in scope. Use :func:`metricLabel` when the method *is* known.
METRIC_LABELS = {
    metric: template.format(window=WINDOW_LABELS[WINDOW_MOVEMENT],
                            window_lower=WINDOW_LABELS[WINDOW_MOVEMENT].lower())
    for metric, template in _METRIC_LABEL_TEMPLATES.items()
}


def metricLabel(metric, method=None):
    """Axis/report label for ``metric``, naming ``method``'s window.

    Without a method this is ``METRIC_LABELS[metric]`` (movement-only wording);
    with one, a two-bin method reads "Movement-initiation + movement ...".
    """
    if method is None:
        return METRIC_LABELS[metric]
    _criterion, n_last_bins = methodSpec(method)
    window = WINDOW_LABELS[n_last_bins]
    return _METRIC_LABEL_TEMPLATES[metric].format(window=window,
                                                  window_lower=window.lower())

#: Metric -> the (numerator, denominator) count columns behind its percentage.
_METRIC_COUNTS = {
    METRIC_MOVEMENT: ("n_movement", "n_neurons"),
    METRIC_MOVEMENT_CHOICE: ("n_movement_choice", "n_neurons"),
    METRIC_CHOICE: ("n_choice", "n_neurons"),
    METRIC_CHOICE_OF_MOVEMENT: ("n_movement_choice", "n_movement"),
}

#: Panels per row in :func:`plotMethodsComparison` before it wraps.
MAX_PANEL_COLS = 3

#: Figure geometry, matching the other 2p analyses (``twop/plot/stats*.py``,
#: the notebook's pie / Venn figures): a roomy single-axes figure saved at 300
#: dpi with a tight bounding box.
FIG_SIZE = (10, 8)
SAVE_DPI = 300

# Same labels region_label produces, straight from the enum's __format__.
MFC = "{}".format(BrainRegion.M2_Bi)
LFC = "{}".format(BrainRegion.ALM_Bi)
REGIONS = (MFC, LFC)
REGION_CLR = {MFC: BRClr[BrainRegion.M2_Bi], LFC: BRClr[BrainRegion.ALM_Bi]}

SESSION_COLUMNS = ["method", "BrainRegion", "ShortName", "animal", "n_neurons",
                   "n_movement", "n_choice", "n_movement_choice",
                   METRIC_MOVEMENT, METRIC_MOVEMENT_CHOICE, METRIC_CHOICE,
                   METRIC_CHOICE_OF_MOVEMENT]

#: Bootstrap iterations for :func:`hierarchicalBootstrap`.
DEFAULT_N_BOOT = 10_000


# ---------------------------------------------------------------------------
# Moved out of 2pAnalysis.ipynb so it can be reused and unit-tested
# ---------------------------------------------------------------------------
def binEdges(epoch_ranges, num_bins):
    """Bin edges over a sampling-aligned trace: 1 pre-sampling bin, ``num_bins-2``
    equal sampling bins and 1 movement bin.

    ``epoch_ranges`` is ``normed_sampling_df.iloc[0].epochs_ranges``, i.e.
    ``[(before_start, before_end), (sampling_start, sampling_end),
    (movement_start, movement_end)]``. Same recipe as ``makeActive`` in the
    notebook, factored out so :func:`maxFiringHist` and the movement criteria
    below cannot drift apart.
    """
    before, mid, after = epoch_ranges
    # Normally we need num_bins+1, but remove the first and last bin and we get
    # num_bins-1
    bins = [before[0]] + \
           list(np.linspace(mid[0], after[0], num_bins - 1, endpoint=True)) + \
           [after[1] + 1]
    return np.array(bins)


def maxFiringHist(df, num_bins, epoch_ranges=None):
    """Bin every neuron by its median peak-firing sample.

    Moved verbatim out of ``2pAnalysis.ipynb`` (``# Sequence Extracation``), the
    bin-edge maths factored into :func:`binEdges`. Returns ``(sub_df, bins)``
    where ``sub_df`` has one row per neuron with ``fire_pos`` (the median of
    ``max_idxs``), the ``bin_val`` interval and its ``bin_idx`` (0 = earliest).

    NB ``bin_idx`` enumerates the bins that are *populated in ``df``*, not the
    ``num_bins`` bins themselves: on a subset where some bin stays empty the
    indices shift down. Anchor on ``bin_val_left`` (an actual bin edge) when the
    identity of a specific bin matters -- :func:`movementMask` does.
    """
    sub_df = df[["BrainRegion", "trace_id"]].copy()
    sub_df["fire_pos"] = df.max_idxs.apply(np.median)
    if epoch_ranges is None:
        print("No epochs ranges passed...")
        max_len = df.max_idxs.apply(np.max).max()
        bins = np.linspace(0, max_len, num_bins + 1, endpoint=True)
    else:
        bins = binEdges(epoch_ranges, num_bins)
    sub_df["bin_val"] = pd.cut(sub_df.fire_pos, bins, include_lowest=True)
    bin_val = pd.IntervalIndex(sub_df.bin_val)
    sub_df["bin_val_left"] = np.around(bin_val.left, decimals=1)
    sub_df["bin_val_right"] = np.around(bin_val.right, decimals=1)
    bin_vals_left_sorted = sorted(sub_df.bin_val_left.unique())
    bin_idx_map = {val: bin_idx
                   for bin_idx, val in enumerate(bin_vals_left_sorted)}
    # Can we have have floating point mismatches?
    sub_df["bin_idx"] = sub_df.bin_val_left.map(bin_idx_map)
    assert sub_df.bin_idx.isnull().sum() == 0
    return sub_df, bins


def getSgfNeurons(stats_df, pval, data_col="ChoiceLeft", prior_data_col=np.nan):
    """``long_trace_id`` set of the neurons significant for ``data_col``.

    Moved out of ``2pAnalysis.ipynb`` (``### Active trials helper functions``);
    the only change is that the p-value threshold is now an argument instead of
    the notebook's ``PVAL`` global. ``stats_df`` is a ``shortlong_df``
    (``run_data_*[0].shortlong_df``); ``DVstr``-split rows are always dropped and
    ``prior_data_col=np.nan`` keeps the no-prior (unconditioned) rows.
    """
    stats_df = stats_df[stats_df.DVstr.isnull()]
    stats_df = stats_df[stats_df.pval <= pval]
    stats_df = stats_df[stats_df.data_col == data_col]
    if isinstance(prior_data_col, float) and np.isnan(prior_data_col):
        stats_df = stats_df[stats_df.prior_data_col.isnull()]
    else:
        stats_df = stats_df[stats_df.prior_data_col == prior_data_col]
    return set(stats_df.long_trace_id)


# ---------------------------------------------------------------------------
# Movement-time criteria
# ---------------------------------------------------------------------------
#: Guards the float comparison against the rounded ``bin_val_left`` edges.
_EDGE_TOL = 1e-9


def movementOnset(epoch_ranges):
    """First sample of the movement epoch (``epoch_ranges[-1][0]``)."""
    return epoch_ranges[-1][0]


def windowOnset(epoch_ranges, bins_count=DEFAULT_BINS_COUNT,
                n_last_bins=WINDOW_MOVEMENT):
    """First sample of the window made of the trailing ``n_last_bins`` bins.

    ``n_last_bins=1`` is the movement bin, so this is exactly
    :func:`movementOnset` (30 for the standard 7 bins); ``n_last_bins=2`` adds
    the movement-initiation bin in front of it (24.6), i.e. the last stretch of
    sampling before the animal leaves the centre port.
    """
    return binEdges(epoch_ranges, bins_count)[-(n_last_bins + 1)]


def methodSpec(method):
    """``(criterion, n_last_bins)`` of a registered method."""
    assert method in METHOD_SPECS, \
        f"Unknown method {method!r}, expected one of {tuple(METHOD_SPECS)}"
    return METHOD_SPECS[method]


def movementMask(max_firing_df, epoch_ranges, method=METHOD_LAST_BIN,
                 bins_count=DEFAULT_BINS_COUNT):
    """Boolean Series (indexed like ``max_firing_df``): does this neuron peak
    inside the method's window, per its criterion (see the module docstring)?

    ``max_firing_df`` is a ``max_firing_*_df``: one row per neuron with
    ``BrainRegion, trace_id, max_idxs`` (peak sample per active trial).
    """
    criterion, n_last_bins = methodSpec(method)
    if not len(max_firing_df):
        return pd.Series([], dtype=bool, index=max_firing_df.index)

    if criterion == CRITERION_MEDIAN_BIN:
        hist_df, bins = maxFiringHist(max_firing_df, num_bins=bins_count,
                                      epoch_ranges=epoch_ranges)
        # Not ``bin_idx >= bins_count - n_last_bins``: that index counts only the
        # bins populated in this frame (see maxFiringHist), so it would shift on
        # a subset. The window's first left edge is unambiguous.
        first_left = np.around(bins[-(n_last_bins + 1)], decimals=1)
        return hist_df.bin_val_left >= first_left - _EDGE_TOL
    onset = windowOnset(epoch_ranges, bins_count, n_last_bins)
    if criterion == CRITERION_MEDIAN_THRESH:
        return max_firing_df.max_idxs.apply(np.median) >= onset
    # CRITERION_Q1_THRESH: q1 of the per-trial peaks at/after the window's
    # start, i.e. the neuron peaks inside the window in >=75% of the trials it
    # is active in.
    iqr_df = extractIQR(max_firing_df)
    return (iqr_df["q1"] >= onset).reindex(max_firing_df.index)


# ---------------------------------------------------------------------------
# Per-session counts
# ---------------------------------------------------------------------------
def animalOf(short_name):
    """Animal of a session name: ``GP4_85_s10_L70_D250_ALM`` -> ``GP4_85``.

    The first two underscore-separated tokens, upper-cased so ``gp4_81_S5...``
    and a hypothetical ``GP4_81_...`` are the same animal. Sessions are nested
    in animals (LFC has only 3 animals for its 10 sessions), which is what
    :func:`hierarchicalBootstrap` resamples over.
    """
    return "_".join(str(short_name).split("_")[:2]).upper()


def perSessionCounts(max_firing_df, epoch_ranges, choice_sgf_ids=None,
                     method=METHOD_LAST_BIN, bins_count=DEFAULT_BINS_COUNT,
                     min_prcnt_valid=0.0):
    """One row per session: neuron counts and percentages for ``method``.

    Parameters
    ----------
    max_firing_df : DataFrame
        ``max_firing_all_df`` (one row per neuron per session) with
        ``BrainRegion, ShortName, trace_id, max_idxs, prcnt_valid``.
    epoch_ranges : sequence
        ``normed_sampling_df.iloc[0].epochs_ranges``.
    choice_sgf_ids : set or None
        ``long_trace_id`` set of the choice-selective neurons
        (:func:`getSgfNeurons`). ``None`` leaves the choice columns at 0/NaN.
    min_prcnt_valid : float
        Drop neurons active in fewer than this % of the session's trials, from
        the numerator *and* the denominator. 0 (default) keeps every neuron
        that the max-firing extraction produced, matching the notebook's other
        ``max_firing_all_df`` analyses.

    Returns
    -------
    DataFrame with ``SESSION_COLUMNS``: raw counts plus ``prcnt_movement``,
    ``prcnt_movement_choice`` and ``prcnt_choice`` (all out of the session's
    total neurons) and ``prcnt_choice_of_movement`` (out of its movement-time
    neurons; NaN when the session has none).
    """
    df = max_firing_df
    if min_prcnt_valid:
        df = df[df.prcnt_valid >= min_prcnt_valid]
    # A neuron with no active trial has no firing position at all.
    df = df[df.max_idxs.apply(len) > 0]
    if not len(df):
        return pd.DataFrame(columns=SESSION_COLUMNS)

    df = df.copy()
    df["is_movement"] = movementMask(df, epoch_ranges, method=method,
                                     bins_count=bins_count)
    df["is_choice"] = (df.trace_id.isin(choice_sgf_ids)
                       if choice_sgf_ids is not None else False)

    rows = []
    for (br, sess), sess_df in df.groupby(["BrainRegion", "ShortName"]):
        n_neurons = len(sess_df)
        n_movement = int(sess_df.is_movement.sum())
        n_choice = int(sess_df.is_choice.sum())
        n_both = int((sess_df.is_movement & sess_df.is_choice).sum())
        rows.append({
            "method": method,
            "BrainRegion": region_label(br),
            "ShortName": sess,
            "animal": animalOf(sess),
            "n_neurons": n_neurons,
            "n_movement": n_movement,
            "n_choice": n_choice,
            "n_movement_choice": n_both,
            METRIC_MOVEMENT: 100 * n_movement / n_neurons,
            METRIC_MOVEMENT_CHOICE: 100 * n_both / n_neurons,
            METRIC_CHOICE: 100 * n_choice / n_neurons,
            METRIC_CHOICE_OF_MOVEMENT: (100 * n_both / n_movement
                                        if n_movement else np.nan),
        })
    return pd.DataFrame(rows, columns=SESSION_COLUMNS)


def perSessionCountsAllMethods(max_firing_df, epoch_ranges,
                               choice_sgf_ids=None,
                               methods=MOVEMENT_METHODS,
                               bins_count=DEFAULT_BINS_COUNT,
                               min_prcnt_valid=0.0):
    """:func:`perSessionCounts` for every method, stacked (``method`` column)."""
    return pd.concat(
        [perSessionCounts(max_firing_df, epoch_ranges, choice_sgf_ids,
                          method=method, bins_count=bins_count,
                          min_prcnt_valid=min_prcnt_valid)
         for method in methods],
        ignore_index=True)


# ---------------------------------------------------------------------------
# Hierarchical (animal -> session -> neuron) bootstrap
# ---------------------------------------------------------------------------
def _sessionCells(sess_rows):
    """The per-session 2x2 neuron table as multinomial cells.

    Columns of the returned array: ``[both, movement only, choice only,
    neither]``. Resampling neurons within a session with replacement is exactly
    a multinomial draw over these four cells, so the bootstrap needs no
    neuron-level frame -- the counts in ``sess_df`` already are the table.
    """
    both = sess_rows.n_movement_choice.to_numpy(dtype=float)
    mov_only = sess_rows.n_movement.to_numpy(dtype=float) - both
    choice_only = sess_rows.n_choice.to_numpy(dtype=float) - both
    neither = (sess_rows.n_neurons.to_numpy(dtype=float) - both - mov_only
               - choice_only)
    cells = np.column_stack([both, mov_only, choice_only, neither])
    assert (cells >= 0).all(), "n_movement_choice must be <= n_movement/n_choice"
    return cells


def _metricFromCells(cells, metric):
    """``(numerator, denominator)`` of ``metric`` from ``_sessionCells`` rows."""
    both, mov_only, choice_only, _neither = cells.T
    n_neurons = cells.sum(axis=1)
    if metric == METRIC_MOVEMENT:
        return both + mov_only, n_neurons
    if metric == METRIC_MOVEMENT_CHOICE:
        return both, n_neurons
    if metric == METRIC_CHOICE:
        return both + choice_only, n_neurons
    if metric == METRIC_CHOICE_OF_MOVEMENT:
        return both, both + mov_only
    raise ValueError(f"Unknown metric {metric!r}")


def hierarchicalBootstrap(sess_df, metric=METRIC_MOVEMENT,
                          method=METHOD_LAST_BIN, n_boot=DEFAULT_N_BOOT,
                          seed=0):
    """Resample animals -> sessions -> neurons and rebuild the region means.

    The session-level mean +/- SEM of :func:`regionSummary` treats sessions as
    independent, but they are nested in a handful of animals (see
    :func:`animalOf`), so it understates the uncertainty. This resamples all
    three levels with replacement -- animals, then that animal's sessions, then
    the neurons within each drawn session (a multinomial over
    :func:`_sessionCells`) -- and recomputes the same estimator, the unweighted
    mean over sessions of the per-session percentage.

    Returns ``(draws, observed)``: ``draws[region]`` is an ``n_boot`` array of
    resampled region means, ``observed[region]`` the value on the real data.
    """
    rng = np.random.default_rng(seed)
    method_df = sess_df[sess_df.method == method]
    draws, observed = {}, {}
    for region in REGIONS:
        region_df = method_df[method_df.BrainRegion == region]
        if not len(region_df):
            draws[region] = np.full(n_boot, np.nan)
            observed[region] = np.nan
            continue
        cells = _sessionCells(region_df)
        num, den = _metricFromCells(cells, metric)
        with np.errstate(invalid="ignore"):
            observed[region] = float(np.nanmean(
                np.where(den > 0, 100 * num / np.where(den > 0, den, 1),
                         np.nan)))

        # Which rows of ``cells`` belong to each animal.
        animals = region_df.animal.to_numpy()
        by_animal = {a: np.flatnonzero(animals == a) for a in np.unique(animals)}
        animal_names = np.array(sorted(by_animal))
        probs = cells / cells.sum(axis=1, keepdims=True)
        n_neurons = cells.sum(axis=1).astype(int)

        out = np.empty(n_boot)
        for b in range(n_boot):
            picked = []
            for animal in rng.choice(animal_names, size=len(animal_names),
                                     replace=True):
                rows = by_animal[animal]
                picked.append(rng.choice(rows, size=len(rows), replace=True))
            picked = np.concatenate(picked)
            # Resample the neurons of every drawn session.
            boot_cells = np.array([rng.multinomial(n_neurons[i], probs[i])
                                   for i in picked], dtype=float)
            b_num, b_den = _metricFromCells(boot_cells, metric)
            keep = b_den > 0
            out[b] = (100 * b_num[keep] / b_den[keep]).mean() if keep.any() \
                else np.nan
        draws[region] = out
    return draws, observed


def bootstrapProbability(draws_left, draws_right):
    """Two-sided bootstrap p ("direct probability"): how often the two regions'
    resampled means land on either side of each other.

    Exact ties count as half, so two identical distributions give p = 1 rather
    than 0. Ties are common once a region saturates (e.g. every session at
    100%), where a strict ``>=`` would report a spurious p = 0.
    """
    valid = ~(np.isnan(draws_left) | np.isnan(draws_right))
    if not valid.any():
        return np.nan
    left, right = draws_left[valid], draws_right[valid]
    p_greater = np.mean(left > right) + 0.5 * np.mean(left == right)
    return float(2 * min(p_greater, 1 - p_greater))


def bootstrapSummary(sess_df, metrics=DEFAULT_METRICS, methods=None,
                     n_boot=DEFAULT_N_BOOT, seed=0):
    """One row per (method, metric) with the hierarchical-bootstrap SE, 95% CI
    and p-value, to be read next to :func:`regionSummary`'s session-level test.

    A p-value that is significant per-session but not here is carried by
    repeated sessions of the same animal rather than by the animals.
    """
    if methods is None:
        methods = [m for m in MOVEMENT_METHODS if m in set(sess_df.method)]
    rows = []
    for method in methods:
        for metric in metrics:
            draws, observed = hierarchicalBootstrap(sess_df, metric=metric,
                                                    method=method,
                                                    n_boot=n_boot, seed=seed)
            row = {"method": method, "metric": metric, "n_boot": n_boot}
            for region in REGIONS:
                region_df = sess_df[(sess_df.method == method) &
                                    (sess_df.BrainRegion == region)]
                lo, hi = np.nanpercentile(draws[region], [2.5, 97.5])
                row[f"{region}_n_animals"] = region_df.animal.nunique()
                row[f"{region}_mean"] = observed[region]
                row[f"{region}_boot_se"] = float(np.nanstd(draws[region]))
                row[f"{region}_ci_lo"], row[f"{region}_ci_hi"] = lo, hi
            diff = draws[MFC] - draws[LFC]
            row["diff_mean"] = float(np.nanmean(diff))
            row["diff_ci_lo"], row["diff_ci_hi"] = np.nanpercentile(diff,
                                                                    [2.5, 97.5])
            row["pval_boot"] = bootstrapProbability(draws[MFC], draws[LFC])
            rows.append(row)
    return pd.DataFrame(rows)


def formatBootstrap(boot_df, bins_count=DEFAULT_BINS_COUNT):
    """:func:`bootstrapSummary` as printable lines."""
    lines = []
    for _, row in boot_df.iterrows():
        lines.append(
            f"{metricLabel(row['metric'], row['method'])}  "
            f"[{METHOD_LABELS[row['method']].format(bins_count=bins_count)}]")
        for region in REGIONS:
            lines.append(
                f"    {region}: {row[f'{region}_mean']:.2f}% "
                f"+/- {row[f'{region}_boot_se']:.2f} boot-SE  "
                f"95% CI [{row[f'{region}_ci_lo']:.2f}, "
                f"{row[f'{region}_ci_hi']:.2f}]  "
                f"({row[f'{region}_n_animals']} animals)")
        lines.append(
            f"    MFC-LFC: {row['diff_mean']:+.2f} "
            f"95% CI [{row['diff_ci_lo']:+.2f}, {row['diff_ci_hi']:+.2f}]  "
            f"p_boot={row['pval_boot']:.4f} "
            f"({_pStars(row['pval_boot'])})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MFC vs LFC summary
# ---------------------------------------------------------------------------
def _regionTest(left_vals, right_vals, alpha=NORMALITY_ALPHA):
    """MFC vs LFC on the per-session percentages, gated on normality.

    The two regions are *independent* samples (different sessions), so
    :func:`~.statstest.normalityGatedTest` runs Shapiro-Wilk per region rather
    than on paired differences: both normal -> two-sample t-test (Levene picks
    Student vs Welch), otherwise Mann-Whitney U, the notebook's
    ``defaultStatsTestFn``.
    """
    return normalityGatedTest(left_vals, right_vals, alpha=alpha,
                              left_name=MFC, right_name=LFC)


def regionSummary(sess_df, metrics=DEFAULT_METRICS, alpha=NORMALITY_ALPHA):
    """One row per (method, metric): MFC and LFC mean +/- SEM over sessions, the
    pooled neuron counts, and the normality-gated MFC-vs-LFC test.

    ``sess_df`` comes from :func:`perSessionCounts` /
    :func:`perSessionCountsAllMethods`. The mean and SEM are over *sessions*,
    each session weighing the same regardless of its neuron count; the pooled
    ``*_n``/``*_total`` columns are the raw neuron tallies behind them.

    The test columns come from :func:`_regionTest`: ``MFC_shapiro_pval`` /
    ``LFC_shapiro_pval`` and ``is_normal`` (the normality check that runs
    first), ``levene_pval``, then ``test``/``statistic``/``pval`` for the test
    that gate selected, plus ``ttest_pval`` and ``mwu_pval`` for reference.
    """
    rows = []
    for method, method_df in sess_df.groupby("method", sort=False):
        for metric in metrics:
            num_col, den_col = _METRIC_COUNTS[metric]
            row = {"method": method, "metric": metric}
            vals = {}
            for region in REGIONS:
                br_df = method_df[method_df.BrainRegion == region]
                vals[region] = br_df[metric].dropna()
                row[f"{region}_n_sessions"] = len(vals[region])
                row[f"{region}_mean"] = vals[region].mean()
                row[f"{region}_sem"] = vals[region].sem()
                row[f"{region}_n"] = int(br_df[num_col].sum())
                row[f"{region}_total"] = int(br_df[den_col].sum())
            row.update(_regionTest(vals[MFC], vals[LFC], alpha=alpha))
            rows.append(row)
    return pd.DataFrame(rows)


def formatSummary(summary_df, bins_count=DEFAULT_BINS_COUNT):
    """The :func:`regionSummary` table as printable ``mean +/- SEM`` lines,
    each with the normality check and the test it selected."""
    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            f"{metricLabel(row['metric'], row['method'])}  "
            f"[{METHOD_LABELS[row['method']].format(bins_count=bins_count)}]")
        for region in REGIONS:
            lines.append(
                f"    {region}: {row[f'{region}_mean']:.2f}% "
                f"+/- {row[f'{region}_sem']:.2f}% SEM  "
                f"({row[f'{region}_n_sessions']} sessions, "
                f"{row[f'{region}_n']:,}/{row[f'{region}_total']:,} neurons)")
        other_pval = (row["mwu_pval"] if row["test"] != TEST_MWU
                      else row["ttest_pval"])
        other_test = TEST_MWU if row["test"] != TEST_MWU else "t-test"
        lines.append(
            f"    Normality (Shapiro-Wilk): "
            f"MFC p={row[f'{MFC}_shapiro_pval']:.3g}, "
            f"LFC p={row[f'{LFC}_shapiro_pval']:.3g} -> "
            + ("normal, t-test usable" if row["is_normal"] else
               "not normal, t-test not usable"))
        lines.append(f"    MFC vs LFC: {row['test']} "
                     f"stat={row['statistic']:.3g}, p={row['pval']:.3g} "
                     f"({other_test} p={other_pval:.3g} for reference)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _saveOrShow(fig, save_figs, fig_save_prefix, rel_path):
    import matplotlib.pyplot as plt
    import pathlib
    if save_figs:
        save_fp = pathlib.Path(fig_save_prefix, rel_path)
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _pStars(pval):
    if np.isnan(pval):
        return "n.a."
    return ("***" if pval <= 0.001 else "**" if pval <= 0.01 else
            "*" if pval <= 0.05 else "n.s.")


def plotRegionBars(sess_df, metric=METRIC_MOVEMENT, method=METHOD_LAST_BIN,
                   bins_count=DEFAULT_BINS_COUNT, save_figs=False,
                   fig_save_prefix=None, ax=None):
    """MFC vs LFC bars of ``metric``: mean +/- SEM over sessions, one dot per
    session. The Mann-Whitney U p-value across sessions is annotated on top."""
    import matplotlib.pyplot as plt
    method_df = sess_df[sess_df.method == method]
    summary = regionSummary(method_df, metrics=(metric,)).iloc[0]

    import matplotlib.ticker as mtick
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    xpos = {MFC: 1, LFC: 2}
    for region in REGIONS:
        vals = method_df[method_df.BrainRegion == region][metric].dropna()
        mean, sem = summary[f"{region}_mean"], summary[f"{region}_sem"]
        ax.bar(xpos[region], mean, yerr=sem, width=0.55,
               color=REGION_CLR[region], alpha=0.8, capsize=8,
               error_kw=dict(lw=2, capthick=2),
               label=(f"{region}: {mean:.2f}% +/-{sem:.2f}% "
                      f"({len(vals)} sessions, "
                      f"{summary[f'{region}_n']:,}/"
                      f"{summary[f'{region}_total']:,} neurons)"))
        ax.plot(xpos[region] * np.ones_like(vals), vals,
                ls="none", marker="o", markersize=8, markerfacecolor="none",
                color="gray", alpha=0.7)

    y_top = max(ax.get_ylim()[1], 1e-9)
    ax.plot([xpos[MFC], xpos[LFC]], [y_top * 1.02] * 2, color="k", lw=1.5)
    ax.text(1.5, y_top * 1.04,
            f"{_pStars(summary.pval)} (p={summary.pval:.3g}, {summary.test})",
            ha="center", va="bottom", fontsize="large")
    ax.set_xticks([xpos[MFC], xpos[LFC]])
    ax.set_xticklabels(list(REGIONS), fontsize="x-large")
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(bottom=0, top=y_top * 1.15)
    ax.set_ylabel(metricLabel(metric, method), fontsize="large")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.tick_params(axis="y", labelsize="large")
    _criterion, n_last_bins = methodSpec(method)
    ax.set_title(f"{WINDOW_LABELS[n_last_bins]} neurons\n"
                 f"{METHOD_LABELS[method].format(bins_count=bins_count)}",
                 fontsize="large")
    ax.legend(fontsize="medium", loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    ax.spines[["right", "top"]].set_visible(False)
    if fig is not None:
        _saveOrShow(fig, save_figs, fig_save_prefix,
                    f"MovementNeurons/{method}_{metric}.svg")


def plotMethodsComparison(sess_df, metric=METRIC_MOVEMENT,
                          bins_count=DEFAULT_BINS_COUNT, save_figs=False,
                          fig_save_prefix=None):
    """One :func:`plotRegionBars` panel per method present in ``sess_df``,
    wrapped onto at most ``MAX_PANEL_COLS`` columns."""
    import matplotlib.pyplot as plt
    present = set(sess_df.method)
    methods = ([m for m in MOVEMENT_METHODS if m in present] +
               [m for m in METHOD_SPECS if m in present
                and m not in MOVEMENT_METHODS])
    n_cols = min(MAX_PANEL_COLS, len(methods))
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(FIG_SIZE[0] * n_cols,
                                     FIG_SIZE[1] * n_rows))
    axs = np.atleast_1d(axs).ravel()
    for idx, (method, ax) in enumerate(zip(methods, axs)):
        plotRegionBars(sess_df, metric=metric, method=method,
                       bins_count=bins_count, ax=ax)
        if idx % n_cols:  # the shared y-axis is named once per row
            ax.set_ylabel("")
    for ax in axs[len(methods):]:  # a partly filled last row
        ax.set_visible(False)
    fig.suptitle(METRIC_LABELS[metric])
    fig.tight_layout()
    _saveOrShow(fig, save_figs, fig_save_prefix,
                f"MovementNeurons/methods_{metric}.svg")
