import numpy as np
import pandas as pd
from typing import Callable, Tuple, Dict
from tqdm.auto import tqdm
from scipy import stats
from statsmodels.stats import multitest


# ---------------------------
# Private helpers (approach-2)
# ---------------------------

def _effect_from_trials_block(block_df: pd.DataFrame,
                              calcPerfFn: Callable[[float, float], float],
                              rng: np.random.Generator | None = None,
                              resample_trials: bool = False) -> float:
    """
    Compute effect for one (Name, OptoBrainRegion, IsEarly) block:
      effect = calcPerfFn(control_mean, opto_mean).
    If resample_trials=True, bootstrap trials *within each label* per session
    (stratified) with replacement to avoid dropping a label by chance.
    """
    def concat_safe(chunks: list[np.ndarray]) -> np.ndarray:
        valid = [c for c in chunks if isinstance(c, np.ndarray) and c.size > 0]
        return np.concatenate(valid) if valid else np.array([])

    control_chunks: list[np.ndarray] = []
    opto_chunks: list[np.ndarray] = []

    for _, session_df in block_df.groupby('SessId'):
        # Split by label once
        ctrl_arr = session_df.loc[session_df['OptoEnabled'] == False, 'ChoiceCorrect'].to_numpy()
        opto_arr = session_df.loc[session_df['OptoEnabled'] == True,  'ChoiceCorrect'].to_numpy()

        if resample_trials:
            assert rng is not None, "rng is required when resample_trials=True"
            if ctrl_arr.size:
                idx = rng.choice(ctrl_arr.size, size=ctrl_arr.size, replace=True)
                ctrl_arr = ctrl_arr[idx]
            if opto_arr.size:
                idx = rng.choice(opto_arr.size, size=opto_arr.size, replace=True)
                opto_arr = opto_arr[idx]

        control_chunks.append(ctrl_arr)
        opto_chunks.append(opto_arr)

    control_concat = concat_safe(control_chunks)
    opto_concat    = concat_safe(opto_chunks)

    if control_concat.size == 0 or opto_concat.size == 0:
        return np.nan

    cntrl_mean = float(np.mean(control_concat))
    opto_mean = float(np.mean(opto_concat))
    # cntrl_mean = float(stats.trim_mean(control_concat, proportiontocut=0.2))
    # opto_mean = float(stats.trim_mean(opto_concat, proportiontocut=0.2))
    return float(calcPerfFn(cntrl_mean, opto_mean))


def _subject_entries_once(trials_df: pd.DataFrame,
                          calcPerfFn: Callable[[float, float], float],
                          rng: np.random.Generator | None = None,
                          resample_trials: bool = False) -> pd.DataFrame:
    """
    Build subject×region×phase entries for all available blocks in trials_df.
    If resample_trials=True, bootstrap trials (not subjects/sessions).
    Returns columns: ['Name','OptoBrainRegion','Time','effect'].
    """
    rows: list[dict] = []
    for (name, region, is_early), block_df in trials_df.groupby(['Name', 'OptoBrainRegion', 'IsEarly']):
        eff = _effect_from_trials_block(block_df, calcPerfFn, rng=rng, resample_trials=resample_trials)
        if not np.isnan(eff):
            rows.append({
                'Name': name,
                'OptoBrainRegion': 'MFC' if region == 'MFC' else 'LFC',
                'Time': 'Early' if bool(is_early) else 'Late',
                'effect': eff
            })
    return pd.DataFrame(rows)


def _bootstrap_iteration_subject_entries(trials_df: pd.DataFrame,
                                         calcPerfFn: Callable[[float, float], float],
                                         rng: np.random.Generator) -> pd.DataFrame:
    """
    One hierarchical bootstrap iteration:
      1) sample subjects with replacement
      2) for each sampled subject, sample sessions with replacement
      3) within each sampled session, bootstrap trials
      4) compute subject×region×phase effects on the bootstrapped data
    """
    subjects = trials_df['Name'].unique()
    if subjects.size == 0:
        return pd.DataFrame(columns=['Name',
                                     'OptoBrainRegion', 'Time', 'effect'])

    sampled_subjects = rng.choice(subjects, size=subjects.size, replace=True)
    rows: list[dict] = []

    for subject in sampled_subjects:
        subject_df = trials_df[trials_df['Name'] == subject]
        sessions = subject_df['SessId'].unique()
        if sessions.size == 0:
            continue
        sampled_sessions = rng.choice(sessions, size=sessions.size, replace=True)
        subject_boot = pd.concat([subject_df[subject_df['SessId'] == s] for s in sampled_sessions],
                                 ignore_index=True)

        for (region, is_early), block_df in subject_boot.groupby(['OptoBrainRegion', 'IsEarly']):
            eff = _effect_from_trials_block(block_df, calcPerfFn, rng=rng, resample_trials=True)
            if not np.isnan(eff):
                rows.append({
                    'Name': subject,
                    'OptoBrainRegion': 'MFC' if region == 'MFC' else 'LFC',
                    'Time': 'Early' if bool(is_early) else 'Late',
                    'effect': eff
                })

    return pd.DataFrame(rows)


def _aggregate_group_effects(entries_df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate subject entries into group means by region×phase,
    and compute cross-region difference Δ = MFC - LFC per phase.
    Returns keys: MFC_Early, MFC_Late, LFC_Early, LFC_Late, Delta_Early, Delta_Late.
    """
    out = {
        'MFC_Early': np.nan, 'MFC_Late': np.nan,
        'LFC_Early': np.nan, 'LFC_Late': np.nan,
        'Delta_Early': np.nan, 'Delta_Late': np.nan
    }
    for region in ('MFC', 'LFC'):
        for phase in ('Early', 'Late'):
            vals = entries_df.loc[
                (entries_df['OptoBrainRegion'] == region) &
                (entries_df['Time'] == phase), 'effect'
            ].to_numpy()
            if vals.size:
                out[f'{region}_{phase}'] = float(np.nanmean(vals))

    for phase in ('Early', 'Late'):
        m, l = out[f'MFC_{phase}'], out[f'LFC_{phase}']
        if not np.isnan(m) and not np.isnan(l):
            out[f'Delta_{phase}'] = float(m - l)

    return out


def _sign_change_p_two_sided(observed_value: float, draws: np.ndarray) -> float:
    """
    Two-sided p from the sign-change bootstrap:
      p_one = P(sign(draw) != sign(observed))
      p_two = min(1, 2 * p_one)
    If observed == 0 -> p_two = 1. NaN-safe.
    """
    if np.isnan(observed_value):
        return np.nan
    if observed_value == 0:
        return 1.0
    valid = ~np.isnan(draws)
    if not np.any(valid):
        return np.nan
    obs_sign = 1 if observed_value > 0 else -1
    p_one = np.mean(np.sign(draws[valid]) == -obs_sign)
    return float(min(1.0, 2.0 * p_one))

def _bh_fdr(pvals: pd.Series) -> pd.Series:
    """
    Benjamini–Hochberg (BH) FDR on a Series. NaNs preserved; index preserved.
    """
    s = pvals.copy()
    mask = s.notna()
    m = int(mask.sum())
    if m == 0:
        return s
    ordered = s[mask].sort_values()
    ranks = pd.Series(np.arange(1, m + 1, dtype=float), index=ordered.index)
    adj = (ordered * m / ranks).cummin().clip(upper=1.0)
    s.loc[adj.index] = adj.values
    return s

def _holm_step_down(pvals: pd.Series) -> pd.Series:
    """
    Holm–Bonferroni step-down adjusted p-values.
    Input:  Series with p-values (may contain NaNs), indexed by test keys.
    Output: Series of same shape with Holm-adjusted p-values (NaNs preserved).
    """
    s = pvals.copy()
    mask = s.notna()
    if not mask.any():
        return s

    order = s[mask].sort_values().index  # ascending by raw p
    m = len(order)

    adjusted = pd.Series(index=order, dtype=float)
    running_max = 0.0
    for i, k in enumerate(order, start=1):
        raw = float(s.loc[k])
        factor = m - i + 1
        adj = min(1.0, factor * raw)
        running_max = max(running_max, adj)   # enforce monotonicity
        adjusted.loc[k] = running_max

    s.loc[order] = adjusted.values
    return s

def _hl_diff_unpaired(x, y):
    if x.size==0 or y.size==0: return np.nan
    diffs = x[:,None] - y[None,:]
    return float(np.median(diffs))

def _aggregate_cross_region_effects(observed_trials_df: pd.DataFrame,
                                    calcPerfFn: Callable[[float, float], float],
                                    resample: bool = False,
                                    rng: np.random.Generator | None = None) -> Dict[str, float]:
    """
    Cross-region Δ = (trimmed mean over sessions of MFC effects) - (trimmed mean over sessions of LFC effects),
    computed per phase. If resample=True, bootstrap *sessions* with replacement within each phase×region.
    """
    if resample:
        assert rng is not None, "rng is required when resample=True"
    out = {'Delta_Early': np.nan, 'Delta_Late': np.nan}

    df = observed_trials_df.copy()

    if resample:
        res_df_li = []
        for _, is_early_df in df.groupby('IsEarly'):
            for _, region_df in is_early_df.groupby('OptoBrainRegion'):
                sess_ids = region_df['SessId'].unique()
                if sess_ids.size == 0:
                    continue
                sessions_sampled = rng.choice(sess_ids, size=sess_ids.size, replace=True)
                for idx, sess_id in enumerate(sessions_sampled):
                    sess_df = region_df[region_df['SessId'] == sess_id].copy()
                    sess_df['SessId'] = f"{sess_id}_boot{idx}"  # avoid collisions
                    res_df_li.append(sess_df)
        if not res_df_li:
            return out
        df = pd.concat(res_df_li, ignore_index=True).reset_index(drop=True)

    for phase_str, is_early in [('Early', True), ('Late', False)]:
        sub = df[df['IsEarly'] == is_early]
        mfc = sub[sub['OptoBrainRegion'] == 'MFC'].groupby('SessId').apply(
            lambda x: calcPerfFn(x[x.OptoEnabled==0].ChoiceCorrect.mean(),
                                 x[x.OptoEnabled==1].ChoiceCorrect.mean())
        ).dropna()
        lfc = sub[sub['OptoBrainRegion'] == 'LFC'].groupby('SessId').apply(
            lambda x: calcPerfFn(x[x.OptoEnabled==0].ChoiceCorrect.mean(),
                                 x[x.OptoEnabled==1].ChoiceCorrect.mean())
        ).dropna()

        if (mfc.size == 0) or (lfc.size == 0):
            out[f'Delta_{phase_str}'] = np.nan
        else:
            mfc_mean = stats.trim_mean(mfc.to_numpy(), 0.2)
            lfc_mean = stats.trim_mean(lfc.to_numpy(), 0.2)
            out[f'Delta_{phase_str}'] = float(mfc_mean - lfc_mean)

    return out

# ---------------------------
# Public API (approach-2)
# ---------------------------

def bootstrapSignTestApproach2(trials_df: pd.DataFrame,
                               calcPerfFn: Callable[[float, float], float],
                               iterations: int = 10_000,
                               seed: int = 42
                               ) -> Tuple[Dict[tuple, float], pd.DataFrame]:
    """
    Approach-2: Hierarchical bootstrap (subjects -> sessions -> trials)
    with sign-change p-values and BH/FDR correction across all tests.

    Parameters
    ----------
    trials_df : DataFrame with columns
        Name, OptoBrainRegion {'MFC','LFC'}, IsEarly (bool),
        OptoEnabled (bool), SessId, ChoiceCorrect (numeric/binary).
    calcPerfFn : (control_mean, opto_mean) -> effect (float)
        e.g., lambda c, o: c - o
              lambda c, o: 100.0*(o - c)/c if c!=0 else np.nan
    iterations : int
        Number of bootstrap iterations.
    seed : int
        RNG seed.

    Returns
    -------
    results : Dict[(phase, test, metric) -> float]
        phase ∈ {'Early','Late'}
        test  ∈ {'MFC','LFC','crossregion'}  (crossregion is Δ = MFC - LFC)
        metric ∈ {'observed','sd','ci_low','ci_high','p','p_fdr'}
    observed_entries : DataFrame
        Subject×region×phase effects computed from raw trials (no resampling).
    """
    trials_df = trials_df[trials_df.ChoiceCorrect.notna()].copy()
    trials_df = trials_df[trials_df.ChoiceCorrect.notna()].copy()
    trials_df['OptoEnabled'] = trials_df['OptoEnabled'].astype(int)
    rng = np.random.default_rng(seed)

    # 1) Observed entries and aggregates
    observed_entries = _subject_entries_once(trials_df, calcPerfFn, rng=None,
                                             resample_trials=False)
    observed_group =  _aggregate_group_effects(observed_entries)
    observed_cross = _aggregate_cross_region_effects(trials_df,
                                                     calcPerfFn,
                                                     resample=False)
    # observed_group = _computeRobustStats_hl(observed_entries)

    within_keys = ['MFC_Early', 'MFC_Late', 'LFC_Early', 'LFC_Late']
    cross_keys  = ['Delta_Early', 'Delta_Late']
    within_draws = {k: [] for k in within_keys}
    cross_draws  = {k: [] for k in cross_keys}

    # 2) Bootstrap loop
    for _ in tqdm(range(iterations), desc="Bootstrap iterations"):
        boot_entries = _bootstrap_iteration_subject_entries(trials_df, calcPerfFn, rng)
        if boot_entries.empty:
            for k in within_keys: within_draws[k].append(np.nan)
            for k in cross_keys:  cross_draws[k].append(np.nan)
            continue

        boot_group =_aggregate_group_effects(boot_entries)
        boot_crossgroup = _aggregate_cross_region_effects(trials_df,
                                                          calcPerfFn,
                                                          resample=True,
                                                          rng=rng)
        # boot_group = _computeRobustStats_hl(boot_entries)
        for k in within_keys: within_draws[k].append(boot_group[k])
        # for k in cross_keys:  cross_draws[k].append(boot_group[k])
        for k in cross_keys:  cross_draws[k].append(boot_crossgroup[k])

    # Convert to arrays
    within_draws = {k: np.asarray(v, dtype=float) for k, v in within_draws.items()}
    cross_draws  = {k: np.asarray(v, dtype=float) for k, v in cross_draws.items()}

    # 3) Summaries and p-values
    results: Dict[tuple, float] = {}

    def _summarize_and_store(phase: str, test: str, observed_value: float, draws: np.ndarray):
        valid = draws[~np.isnan(draws)]
        sd = float(np.std(valid, ddof=1)) if valid.size else np.nan
        if valid.size:
            ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan
        p = _sign_change_p_two_sided(observed_value, draws)

        results[(phase, test, 'observed')] = observed_value
        results[(phase, test, 'sd')]       = sd
        results[(phase, test, 'ci_low')]   = ci_low
        results[(phase, test, 'ci_high')]  = ci_high
        results[(phase, test, 'p')]        = p

    # within-region effects
    _summarize_and_store('Early', 'MFC', observed_group['MFC_Early'], within_draws['MFC_Early'])
    _summarize_and_store('Late',  'MFC', observed_group['MFC_Late'],  within_draws['MFC_Late'])
    _summarize_and_store('Early', 'LFC', observed_group['LFC_Early'], within_draws['LFC_Early'])
    _summarize_and_store('Late',  'LFC', observed_group['LFC_Late'],  within_draws['LFC_Late'])
    # cross-region deltas (MFC-LFC)
    # _summarize_and_store('Early', 'crossregion', observed_group['Delta_Early'], cross_draws['Delta_Early'])
    # _summarize_and_store('Late',  'crossregion', observed_group['Delta_Late'],  cross_draws['Delta_Late'])
    _summarize_and_store('Early', 'crossregion', observed_cross['Delta_Early'], cross_draws['Delta_Early'])
    _summarize_and_store('Late',  'crossregion', observed_cross['Delta_Late'],  cross_draws['Delta_Late'])

    ALPHA = 0.05
    # 1) Within-phase Holm (MFC vs 0, LFC vs 0)
    for phase in ['Early', 'Late']:
        pair_keys = [(phase, 'MFC', 'p'), (phase, 'LFC', 'p')]
        # keep only keys that exist
        p_series = pd.Series({k: results[k] for k in pair_keys if k in results}, dtype=float)
        if not p_series.empty:
            # p_holm = _holm_step_down(p_series)
            # for k in pair_keys:
            #     if k in p_holm.index:
            #         results[(k[0], k[1], 'p_holm_within_phase')] = float(p_holm.loc[k])
            rej, p_holm_within, _, _ = multitest.multipletests(p_series.values,
                                                               method='holm',
                                                               alpha=ALPHA)
            for i, k in enumerate(pair_keys):
                if k in p_series.index:
                    results[(k[0], k[1], 'p_holm_within_phase')] = float(
                                                               p_holm_within[i])

    # 2) Cross-region Holm across phases (Δ Early, Δ Late)
    cross_keys = [('Early', 'crossregion', 'p'), ('Late', 'crossregion', 'p')]
    p_series_cross = pd.Series({k: results[k] for k in cross_keys if k in results}, dtype=float)
    if not p_series_cross.empty:
        # p_holm_cross = _holm_step_down(p_series_cross)
        # for k in cross_keys:
        #     if k in p_holm_cross.index:
        #         results[(k[0], 'crossregion', 'p_holm_across_phases')] = float(p_holm_cross.loc[k])
        rej, p_holm_cross, _, _ = multitest.multipletests(p_series_cross.values,
                                                          method='holm',
                                                          alpha=ALPHA)
        for i, k in enumerate(cross_keys):
            if k in p_series_cross.index:
                results[(k[0], 'crossregion', 'p_holm_across_phases')] = float(p_holm_cross[i])

    return results, observed_entries