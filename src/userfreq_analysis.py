# ----------------------------------------------------------------------------------------
# User Posting Frequency and Lexical Quality Analysis
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from itertools import combinations
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu


LEXICAL_METRICS = [
    'mattr_score',  # mtld_score omitted; mattr preferred for short texts
    'yules_k',
    'zipf_score',
    'aoa_score',
    'nawl_ratio',
]

METRIC_LABELS = {
    'mattr_score': 'MATTR Score',
    'yules_k':     "Yule's K",
    'zipf_score':  'Zipf Score (avg. word frequency)',
    'aoa_score':   'Age of Acquisition (years)',
    'nawl_ratio':  'NAWL Ratio',
}


# -----------------------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Raise ValueError if any columns in `cols` are absent from `df`."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns adjusted p-values capped at 1.

    Adjusted p_(k) = min_{j >= k}( p_(j) · m / j ), where tests are sorted
    ascending by raw p-value and m is the total number of tests.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adj = np.minimum(1.0, sorted_p * m / np.arange(1, m + 1))
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    result = np.empty(m)
    result[order] = adj
    return result


def _qcut_with_labels(series: pd.Series, n: int, prefix: str = 'Q') -> pd.Categorical:
    """pd.qcut wrapper that auto-generates labels matching the actual bin count.

    When many speakers share the same post count, pd.qcut with duplicates='drop'
    may produce fewer than n bins.  This helper computes the bins first and then
    generates labels of the right length, so the two never go out of sync.
    """
    _, bins = pd.qcut(series, q=n, retbins=True, duplicates='drop')
    n_actual = len(bins) - 1
    labels = [f'{prefix}{i + 1}' for i in range(n_actual)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


# -----------------------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------------------

def assign_freq_quartiles(
    df: pd.DataFrame,
    freq_col: str = 'num_utterances_by_speaker',
    n_quartiles: int = 4,
) -> pd.DataFrame:
    """Assign each speaker to a posting-frequency quartile.

    Quartile boundaries are computed on the **speaker-level** distribution of
    `freq_col` (one value per unique speaker_id).  Because lexical_master.csv
    contains one row per user per month, computing quantiles on the raw
    row-level column would shift the boundaries upward by giving prolific users
    more influence.  This function deduplicates first to avoid that.

    Parameters
    ----------
    df : pd.DataFrame
        Master lexical DataFrame.  Must contain 'speaker_id' and `freq_col`.
    freq_col : str
        Column used for frequency binning.  Default is
        'num_utterances_by_speaker' (lifetime total posts across the corpus).
    n_quartiles : int
        Number of equal-frequency bins.  Default is 4.  Pass 5 for quintiles,
        10 for deciles, etc.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with 'freq_quartile' added.  Labels are ordered
        Categorical values 'Q1' … 'Q{n}'; Q1 = least frequent posters,
        Q{n} = most frequent.  Speakers with a NaN or zero `freq_col` value
        receive NaN in 'freq_quartile'.
    """
    _require_columns(df, ['speaker_id', freq_col])

    # One row per speaker with a valid, positive frequency value.
    speaker_freq = (
        df[['speaker_id', freq_col]]
        .drop_duplicates('speaker_id')
        .copy()
    )
    speaker_freq[freq_col] = pd.to_numeric(speaker_freq[freq_col], errors='coerce')
    valid = speaker_freq[freq_col].notna() & (speaker_freq[freq_col] > 0)
    speaker_freq = speaker_freq[valid]

    speaker_freq['freq_quartile'] = _qcut_with_labels(
        speaker_freq[freq_col], n_quartiles
    )

    out = df.copy()
    out = out.merge(
        speaker_freq[['speaker_id', 'freq_quartile']],
        on='speaker_id',
        how='left',
    )
    return out


def add_relative_month(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``relative_month``: months elapsed since each speaker's first post.

    Month 0 is the calendar month of a speaker's first appearance in the
    dataset.  Month 1 is the next calendar month regardless of whether the
    speaker posted that month, and so on.  Uses integer month arithmetic
    (year × 12 + month) for efficiency on large DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'speaker_id' and 'timestamp'.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with 'relative_month' (nullable Int64) added.
    """
    _require_columns(df, ['speaker_id', 'timestamp'])

    out = df.copy()
    ts = pd.to_datetime(out['timestamp'], errors='coerce')
    month_int = ts.dt.year * 12 + ts.dt.month

    first_month = (
        out.assign(_mi=month_int)
        .groupby('speaker_id')['_mi']
        .transform('min')
    )
    out['relative_month'] = (month_int - first_month).astype('Int64')
    return out


def prepare_panel(
    df: pd.DataFrame,
    freq_col: str = 'num_utterances_by_speaker',
    n_quartiles: int = 4,
    min_obs: int = 6,
) -> pd.DataFrame:
    """Prepare a speaker-month panel for the temporal frequency analysis.

    Convenience wrapper that (1) assigns posting-frequency quartiles via
    :func:`assign_freq_quartiles`, (2) adds a relative-month counter via
    :func:`add_relative_month`, and (3) filters to speakers with at least
    ``min_obs`` distinct monthly rows so every retained user has a trajectory
    worth plotting.

    For the **quartile-distribution analysis** (Analysis 2), call
    :func:`assign_freq_quartiles` directly instead — no temporal filter should
    be applied there, since all speakers should be included.

    Parameters
    ----------
    df : pd.DataFrame
        Raw lexical master DataFrame (one row per user-month).
    freq_col : str
        Frequency column for quartile assignment.
        Default is 'num_utterances_by_speaker'.
    n_quartiles : int
        Number of frequency bins (default 4).
    min_obs : int
        Minimum number of monthly rows a speaker must have to be retained.
        Default is 6 (roughly half a year of consistent activity).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with 'freq_quartile' and 'relative_month' added.
        A retention summary is printed to stdout.
    """
    out = assign_freq_quartiles(df, freq_col=freq_col, n_quartiles=n_quartiles)
    out = add_relative_month(out)

    # lexical_master has exactly one row per user-month, so row count = month count.
    obs_per_speaker = out.groupby('speaker_id').size()
    valid = obs_per_speaker[obs_per_speaker >= min_obs].index
    out = out[out['speaker_id'].isin(valid)].copy()

    n_retained = out['speaker_id'].nunique()
    n_total = df['speaker_id'].nunique()
    print(
        f"Retained {n_retained:,} / {n_total:,} speakers "
        f"({100 * n_retained / n_total:.1f}%) with >= {min_obs} monthly observations."
    )
    return out


# -----------------------------------------------------------------------------------------
# Analysis 1 — Lexical quality over relative user tenure
# -----------------------------------------------------------------------------------------

def plot_quality_over_relative_time(
    df: pd.DataFrame,
    metrics: Sequence[str] = LEXICAL_METRICS,
    max_relative_month: int = 24,
    min_speakers_per_point: int = 10,
    ci: bool = True,
) -> None:
    """Plot mean lexical quality over relative user tenure, by frequency quartile.

    Aligns all speakers by months since their first post (relative month) rather
    than calendar time, so early- vs. later-career writing quality can be
    compared across users who joined the platform at different times.  One line
    is drawn per frequency quartile, letting you see whether trajectories
    diverge, converge, or run parallel across groups.

    Requires :func:`prepare_panel` (or :func:`assign_freq_quartiles` +
    :func:`add_relative_month`) to have been called on `df` first.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'freq_quartile', 'relative_month', and all metric columns.
    metrics : list of str
        Lexical metrics to plot.  Defaults to LEXICAL_METRICS.
    max_relative_month : int
        Truncate the x-axis at this relative month (default 24).  Later months
        are typically too sparsely populated for reliable group means.
    min_speakers_per_point : int
        Suppress a (quartile, relative_month) cell when fewer than this many
        speakers contribute to the mean (default 10).  Prevents noisy estimates
        in the far right tail of the plot.
    ci : bool
        If True (default), shade a 95% CI band (± 1.96 SE) around each line.
    """
    _require_columns(df, ['freq_quartile', 'relative_month', *metrics])

    plot_df = df[df['relative_month'] <= max_relative_month].copy()
    plot_df['relative_month'] = plot_df['relative_month'].astype(int)

    quartile_order = sorted(
        plot_df['freq_quartile'].dropna().astype(str).unique().tolist()
    )
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        ax = axes[i]
        for j, quartile in enumerate(quartile_order):
            mask = plot_df['freq_quartile'].astype(str) == quartile
            grouped = (
                plot_df.loc[mask]
                .groupby('relative_month')[col]
                .agg(mean='mean', std='std', count='count')
                .reset_index()
            )
            grouped = grouped[grouped['count'] >= min_speakers_per_point]
            if grouped.empty:
                continue

            color = palette[j % len(palette)]
            ax.plot(
                grouped['relative_month'], grouped['mean'],
                color=color, linewidth=1.8, label=quartile,
            )
            if ci:
                margin = 1.96 * grouped['std'] / np.sqrt(grouped['count'])
                ax.fill_between(
                    grouped['relative_month'],
                    grouped['mean'] - margin,
                    grouped['mean'] + margin,
                    color=color, alpha=0.15, linewidth=0,
                )

        ax.set_ylabel(METRIC_LABELS.get(col, col))
        ax.grid(True, alpha=0.3)
        ax.legend(title='Freq. Quartile', ncol=2, fontsize=9)

    axes[-1].set_xlabel('Months Since First Post')
    fig.suptitle(
        'Lexical Quality Over User Tenure by Posting Frequency Quartile',
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# -----------------------------------------------------------------------------------------
# Analysis 2 — Lexical quality distributions across frequency quartiles
# -----------------------------------------------------------------------------------------

def plot_quality_by_freq_quartile(
    df: pd.DataFrame,
    metrics: Sequence[str] = LEXICAL_METRICS,
    kind: str = 'violin',
    agg_to_speaker: bool = True,
) -> None:
    """Compare lexical quality distributions across posting-frequency quartiles.

    Requires :func:`assign_freq_quartiles` (or :func:`prepare_panel`) to have
    been called on `df` so that a 'freq_quartile' column is present.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'freq_quartile', 'speaker_id', and all metric columns.
    metrics : list of str
        Lexical metrics to compare.  Defaults to LEXICAL_METRICS.
    kind : {'violin', 'box'}
        Plot style.  'violin' (default) shows the full kernel density; 'box'
        shows quartile boxes with outlier dots.
    agg_to_speaker : bool
        If True (default), average each speaker's monthly rows before plotting
        so that prolific users with many rows do not dominate the distributions.
        Each data point then represents one speaker's mean lexical quality across
        all their months.  Set to False to compare at the individual-post level.
    """
    _require_columns(df, ['freq_quartile', 'speaker_id', *metrics])

    plot_df = df.dropna(subset=['freq_quartile']).copy()

    if agg_to_speaker:
        plot_df = (
            plot_df
            .groupby(['speaker_id', 'freq_quartile'], observed=True)[list(metrics)]
            .mean()
            .reset_index()
        )

    plot_df['freq_quartile'] = plot_df['freq_quartile'].astype(str)
    quartile_order = sorted(plot_df['freq_quartile'].unique().tolist())
    palette = sns.color_palette('muted', len(quartile_order))

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        ax = axes[i]
        if kind == 'violin':
            sns.violinplot(
                data=plot_df, x='freq_quartile', y=col,
                hue='freq_quartile', order=quartile_order,
                palette=palette, inner='box', legend=False, ax=ax,
            )
        else:
            sns.boxplot(
                data=plot_df, x='freq_quartile', y=col,
                hue='freq_quartile', order=quartile_order,
                palette=palette, legend=False,
                flierprops=dict(marker='.', markersize=2, alpha=0.4),
                ax=ax,
            )
        ax.set_ylabel(METRIC_LABELS.get(col, col))
        ax.set_xlabel('')
        ax.grid(True, axis='y', alpha=0.3)

    agg_note = 'speaker means' if agg_to_speaker else 'post level'
    axes[-1].set_xlabel(
        'Posting Frequency Quartile  (Q1 = least frequent · Q4 = most frequent)'
    )
    fig.suptitle(
        f'Lexical Quality by Posting Frequency Quartile  ({agg_note})',
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def run_freq_quartile_tests(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    agg_to_speaker: bool = True,
) -> pd.DataFrame:
    """Run Kruskal-Wallis and pairwise Mann-Whitney U tests across frequency quartiles.

    For each metric a Kruskal-Wallis H-test checks whether any quartile
    distributions differ overall.  Pairwise Mann-Whitney U tests then compare
    every pair of quartiles, with Benjamini-Hochberg FDR correction applied
    jointly across all pairs and all metrics.

    Requires :func:`assign_freq_quartiles` (or :func:`prepare_panel`) to have
    been called on `df` so that 'freq_quartile' is present.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'freq_quartile', 'speaker_id', and all metric columns.
    metrics : list of str, optional
        Metrics to test.  Defaults to LEXICAL_METRICS.
    alpha : float
        Significance threshold after BH correction (default 0.05).
    agg_to_speaker : bool
        If True (default), average each speaker's rows first so that each
        speaker contributes a single observation per metric.

    Returns
    -------
    pd.DataFrame
        Tidy results table with columns:

        metric       – lexical metric name
        test         – 'kruskal-wallis' or 'mann-whitney'
        group_1      – first group label ('all' for Kruskal-Wallis rows)
        group_2      – second group label (NaN for Kruskal-Wallis rows)
        statistic    – H statistic (KW) or U statistic (MW)
        p_value      – raw two-sided p-value
        p_adjusted   – BH-corrected p-value (pairwise MW); raw p (KW)
        significant  – True if p_adjusted <= alpha
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    _require_columns(df, ['freq_quartile', 'speaker_id', *metrics])

    test_df = df.dropna(subset=['freq_quartile']).copy()

    if agg_to_speaker:
        test_df = (
            test_df
            .groupby(['speaker_id', 'freq_quartile'], observed=True)[list(metrics)]
            .mean()
            .reset_index()
        )

    test_df['freq_quartile'] = test_df['freq_quartile'].astype(str)
    quartiles = sorted(test_df['freq_quartile'].unique().tolist())
    pairs = list(combinations(quartiles, 2))

    records = []

    for metric in metrics:
        # --- Kruskal-Wallis: do any quartile distributions differ? ---
        groups = [
            test_df.loc[test_df['freq_quartile'] == q, metric].dropna().values
            for q in quartiles
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            kw_stat, kw_p = kruskal(*groups)
        else:
            kw_stat, kw_p = np.nan, np.nan

        records.append(dict(
            metric=metric, test='kruskal-wallis',
            group_1='all', group_2=np.nan,
            statistic=round(float(kw_stat), 4) if not np.isnan(kw_stat) else np.nan,
            p_value=kw_p, p_adjusted=np.nan, significant=False,
        ))

        # --- Pairwise Mann-Whitney U: which pairs drive any difference? ---
        for q1, q2 in pairs:
            g1 = test_df.loc[test_df['freq_quartile'] == q1, metric].dropna().values
            g2 = test_df.loc[test_df['freq_quartile'] == q2, metric].dropna().values
            if len(g1) >= 2 and len(g2) >= 2:
                mw_stat, mw_p = mannwhitneyu(g1, g2, alternative='two-sided')
            else:
                mw_stat, mw_p = np.nan, np.nan

            records.append(dict(
                metric=metric, test='mann-whitney',
                group_1=q1, group_2=q2,
                statistic=round(float(mw_stat), 2) if not np.isnan(mw_stat) else np.nan,
                p_value=mw_p, p_adjusted=np.nan, significant=False,
            ))

    result_df = pd.DataFrame(records)

    # BH correction applied jointly across all pairwise tests and all metrics.
    mw_valid = (result_df['test'] == 'mann-whitney') & result_df['p_value'].notna()
    if mw_valid.any():
        raw_p = result_df.loc[mw_valid, 'p_value'].to_numpy(dtype=float)
        adj_p = _bh_adjust(raw_p)
        result_df.loc[mw_valid, 'p_adjusted'] = np.round(adj_p, 6)
        result_df.loc[mw_valid, 'significant'] = adj_p <= alpha

    # KW is a single omnibus test per metric; use raw p directly.
    kw_valid = (result_df['test'] == 'kruskal-wallis') & result_df['p_value'].notna()
    result_df.loc[kw_valid, 'p_adjusted'] = result_df.loc[kw_valid, 'p_value'].round(6)
    result_df.loc[kw_valid, 'significant'] = result_df.loc[kw_valid, 'p_value'] <= alpha

    result_df['p_value'] = result_df['p_value'].round(6)

    return result_df
