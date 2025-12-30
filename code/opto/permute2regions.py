import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
from tqdm.auto import tqdm
from typing import Callable

def _computeSubjectEntries(trials_df: pd.DataFrame,
                           calcPerfFn: Callable[[float, float], float],
                           rng: np.random.Generator | None = None,
                           permute: bool = False) -> pd.DataFrame:
    """
    Build subject×region×phase table of effects: effect = mean(ctrl) - mean(opto).

    If permute=False:
        - Compute observed effects directly from trials (no label shuffling).

    If permute=True:
        - For each (Name, OptoBrainRegion, IsEarly) block, permute OptoEnabled
          labels *within each session* and recompute the effect. This generates
          a single permutation-null draw of subject effects.

    Expects columns:
      Name, OptoBrainRegion in {'MFC','LFC'}, IsEarly (bool), OptoEnabled (bool),
      SessId, ChoiceCorrect (numeric/binary).
    """
    if permute:
        assert rng is not None, "rng must be provided when permute=True"

    # def compute_effect(control_values: np.ndarray, opto_values: np.ndarray) -> float | np.nan:
    #     if control_values.size == 0 or opto_values.size == 0:
    #         return np.nan
    #     return float(np.mean(control_values) - np.mean(opto_values))

    rows: list[dict] = []

    for (name, region, is_early), block_df in trials_df.groupby(['Name', 'OptoBrainRegion', 'IsEarly']):
        if not permute:
            control = block_df.loc[block_df['OptoEnabled'] == False, 'ChoiceCorrect'].to_numpy()
            opto    = block_df.loc[block_df['OptoEnabled'] == True,  'ChoiceCorrect'].to_numpy()
        else:
            # Permute OptoEnabled within each session, then aggregate
            control_values_list, opto_values_list = [], []
            for _, session_trials in block_df.groupby('SessId'):
                is_opto_enabled = session_trials['OptoEnabled'].to_numpy()
                outcomes = session_trials['ChoiceCorrect'].to_numpy()
                if is_opto_enabled.size == 0:
                    continue
                permuted_is_opto_enabled = is_opto_enabled.copy()
                rng.shuffle(permuted_is_opto_enabled)
                permuted_control_values = outcomes[permuted_is_opto_enabled == False]
                permuted_opto_values = outcomes[permuted_is_opto_enabled == True]
                if len(permuted_control_values) and len(permuted_opto_values):
                    control_values_list.append(permuted_control_values)
                    opto_values_list.append(permuted_opto_values)
            if not len(control_values_list) and not len(opto_values_list):
                control = np.array([])
                opto    = np.array([])
            else:
                control = np.concatenate(control_values_list)
                opto    = np.concatenate(opto_values_list)

        # effect = compute_effect(control, opto)
        if control.size == 0 or opto.size == 0:
            continue
        effect = calcPerfFn(np.nanmean(control), np.nanmean(opto))
        rows.append({
            'Name': name,
            'OptoBrainRegion': region,
            'Time': 'Early' if is_early else 'Late',
            'effect': effect
        })
    return pd.DataFrame(rows)


def _groupSubjectsByRegionCount(phase_entries: pd.DataFrame):
    counts = phase_entries.groupby('Name').size()
    one_region_subjects = counts[counts == 1].index.to_numpy()
    two_region_subjects = counts[counts == 2].index.to_numpy()
    return one_region_subjects, two_region_subjects


def _permuteRegionLabelsUnpaired(phase_entries: pd.DataFrame,
                                 rng: np.random.Generator) -> pd.DataFrame:
    """
    Within a single phase (Early or Late), permute 'Region' labels:
      - For subjects with both regions present: swap MFC/LFC with p=0.5 (preserves within-subject dependence).
      - For subjects with a single region: assign labels so that global MFC/LFC counts match the original totals.
    """
    assert phase_entries['Time'].nunique() == 1, "Input must contain exactly one phase"
    phase_entries = phase_entries.copy()

    counts = phase_entries.groupby('Name').size()
    two_region_subjects = counts[counts == 2].index.to_numpy()
    one_region_subjects = counts[counts == 1].index.to_numpy()

    target_mfc = int((phase_entries['OptoBrainRegion'] == 'MFC').sum())
    target_lfc = int((phase_entries['OptoBrainRegion'] == 'LFC').sum())

    # one_region_subjects, two_region_subjects = groupSubjectsByRegionCount(phase_entries)

    result_parts: list[pd.DataFrame] = []

    # Handle paired subjects (exactly two rows): swap with probability 0.5
    paired_regions_subjs_df = phase_entries[phase_entries['Name'].isin(two_region_subjects)].copy()
    for name, subject_rows in paired_regions_subjs_df.groupby('Name'):
        assert len(subject_rows) == 2, f"{name} has {len(subject_rows)} rows (expected 2) in this phase"
        counts = subject_rows['OptoBrainRegion'].value_counts()
        assert counts.get('MFC', 0) == 1 and counts.get('LFC', 0) == 1, (
            display(subject_rows) or f"{name} is not 1×MFC + 1×LFC")
        subject_rows = subject_rows.copy()
        if rng.random() < 0.5:
            subject_rows.loc[subject_rows.index, 'OptoBrainRegion'] = \
                               subject_rows['OptoBrainRegion'].iloc[::-1].values
        result_parts.append(subject_rows)

    # After paired handling, they contribute exactly one MFC and one LFC each
    num_paired = len(result_parts)
    mfc_used = num_paired
    lfc_used = num_paired

    # Assign singles to reach original totals
    single_regions_subjs_df = phase_entries[
                         phase_entries['Name'].isin(one_region_subjects)].copy()
    if not single_regions_subjs_df.empty:
        indices = single_regions_subjs_df.index.to_numpy()
        indices = indices[rng.permutation(len(indices))]
        single_regions_subjs_df = single_regions_subjs_df.loc[indices].copy()

        need_mfc = max(target_mfc - mfc_used, 0)
        need_lfc = max(target_lfc - lfc_used, 0)

        labels = np.array(['MFC'] * need_mfc + ['LFC'] * need_lfc)
        remainder = len(single_regions_subjs_df) - len(labels)
        if remainder > 0:
            total = max(target_mfc + target_lfc, 1)
            p_mfc = target_mfc / total
            extra = rng.choice(np.array(['MFC', 'LFC']), size=remainder, replace=True, p=[p_mfc, 1 - p_mfc])
            labels = np.concatenate([labels, extra])

        rng.shuffle(labels)
        single_regions_subjs_df.loc[:, 'OptoBrainRegion'] = labels[:len(single_regions_subjs_df)]
        result_parts.append(single_regions_subjs_df)

    permuted_phase = pd.concat(result_parts, ignore_index=True)

    # Safety: preserve totals
    assert (permuted_phase['OptoBrainRegion'] == 'MFC').sum() == target_mfc, "MFC count changed"
    assert (permuted_phase['OptoBrainRegion'] == 'LFC').sum() == target_lfc, "LFC count changed"

    return permuted_phase


def _computeTStatistics(entries_df: pd.DataFrame) -> dict:
    """
    Returns nested dict: stats[phase]['crossregion'|'MFC'|'LFC'] = t-statistic (np.nan if not enough data).
    """
    # display(entries_df)
    stats = {'Early': {'crossregion': np.nan, 'MFC': np.nan, 'LFC': np.nan},
             'Late':  {'crossregion': np.nan, 'MFC': np.nan, 'LFC': np.nan}}
    for time in ("Early", "Late"):
        phase_entries = entries_df[entries_df['Time'] == time]
        mfc = phase_entries.loc[phase_entries['OptoBrainRegion'] == 'MFC', 'effect'].to_numpy()
        lfc = phase_entries.loc[phase_entries['OptoBrainRegion'] == 'LFC', 'effect'].to_numpy()

        stats[time]['crossregion'] = (
            np.nanmean(mfc) - np.nanmean(lfc) #ttest_ind(mfc, lfc, equal_var=False).statistic
            if (len(mfc) >= 2 and len(lfc) >= 2) else np.nan
        )
        stats[time]['MFC'] = np.nanmean(mfc)#ttest_1samp(mfc, 0.0).statistic if len(mfc) >= 2 else np.nan
        stats[time]['LFC'] = np.nanmean(lfc)#ttest_1samp(lfc, 0.0).statistic if len(lfc) >= 2 else np.nan
    return stats


def _maxAbsOverFamily(stats: dict, family_keys: list[tuple[str, str]]) -> float:
    values = []
    for phase, key in family_keys:
        value = stats.get(phase, {}).get(key, np.nan)
        if not np.isnan(value):
            values.append(abs(value))
    return np.max(values) if values else np.nan


def _adjustedPFromMaxDistribution(max_values: list[float], observed_abs_t: float) -> float:
    valid = [v for v in max_values if not np.isnan(v)]
    if np.isnan(observed_abs_t) or not valid:
        return np.nan
    count = np.sum(np.asarray(valid) >= observed_abs_t)
    return (1 + count) / (1 + len(valid))


def runPermutationMaxT(trials_df: pd.DataFrame,
                       calcPerfFn: Callable[[float, float], float],
                       iterations: int = 10000,
                       seed: int = 42,
                       adjust_scope: str = 'per_phase'):
    """
    Runs a max-T permutation test for:
      - Per phase (Early/Late): Region difference (MFC-LFC), MFC vs 0, LFC vs 0.

    adjust_scope:
      - 'per_phase' → adjust within each phase (3 tests).
      - 'global'    → adjust across all 6 tests.
      - 'both'      → return both sets of adjusted p-values.

    Returns:
      results: dict with adjusted p-values and observed t-stats
      observed_entries: DataFrame with observed subject×region×phase effects
    """
    assert adjust_scope in ('per_phase', 'global', 'both'), "Invalid adjust_scope"

    trials_df = trials_df.copy()
    rng = np.random.default_rng(seed)

    # ---------- 1) Observed subject×region×phase effects ----------
    observed_entries = _computeSubjectEntries(trials_df,
                                              calcPerfFn=calcPerfFn,
                                              rng=None, permute=False)
    observed_stats = _computeTStatistics(observed_entries)

    phase_keys = {
        'Early': [('Early', 'crossregion'), ('Early', 'MFC'), ('Early', 'LFC')],
        'Late':  [('Late',  'crossregion'), ('Late',  'MFC'), ('Late',  'LFC')],
    }
    all_keys = phase_keys['Early'] + phase_keys['Late']

    max_values_early, max_values_late, max_values_all = [], [], []

    # ---------- 2) Permutation loop: label-shuffle within sessions + region relabel ----------
    for _ in tqdm(range(iterations)):
        # 2a) Permute OptoEnabled within sessions, recompute subject effects (null draw)
        permuted_entries = _computeSubjectEntries(trials_df,
                                                  calcPerfFn=calcPerfFn,
                                                  rng=rng, permute=True)

        # 2b) Region-label permutation within each phase (unpaired; paired-safe)
        early_permuted = _permuteRegionLabelsUnpaired(permuted_entries[permuted_entries['Time'] == 'Early'], rng)
        late_permuted  = _permuteRegionLabelsUnpaired(permuted_entries[permuted_entries['Time'] == 'Late'],  rng)
        permuted_all = pd.concat([early_permuted, late_permuted], ignore_index=True)

        # 2c) Compute t-statistics for this permuted dataset
        permuted_stats = _computeTStatistics(permuted_all)

        # 2d) Record family maxima (for max-T)
        max_values_early.append(_maxAbsOverFamily(permuted_stats, phase_keys['Early']))
        max_values_late.append(_maxAbsOverFamily(permuted_stats, phase_keys['Late']))
        max_values_all.append(_maxAbsOverFamily(permuted_stats, all_keys))

    # ---------- 3) Adjusted p-values from empirical max distributions ----------
    results = {}

    if adjust_scope in ('per_phase', 'both'):
        for phase in ['Early', 'Late']:
            family_max = max_values_early if phase == 'Early' else max_values_late
            for key in ['crossregion', 'MFC', 'LFC']:
                obs_t = observed_stats[phase][key]
                results[(phase, key, 'p_adj_per_phase')] = _adjustedPFromMaxDistribution(
                    family_max, abs(obs_t)
                )

    if adjust_scope in ('global', 'both'):
        for phase, key in all_keys:
            obs_t = observed_stats[phase][key]
            results[(phase, key, 'p_adj_global')] = _adjustedPFromMaxDistribution(
                max_values_all, abs(obs_t)
            )

    # ---------- 4) Attach observed t-statistics ----------
    for phase in ['Early', 'Late']:
        for key in ['crossregion', 'MFC', 'LFC']:
            results[(phase, key, 'T_obs')] = observed_stats[phase][key]

    return results, observed_entries
