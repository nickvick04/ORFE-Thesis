# ----------------------------------------------------------------------------------------
# User Posting Frequency and Lexical Quality Analysis
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from itertools import combinations
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, skew, kurtosis


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

# metrics that are negated before plotting so that higher always means greater lexical quality
NEGATED_METRICS = {'yules_k', 'zipf_score'}


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
    method: str = 'threshold',
    threshold_cuts: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Assign each speaker to a posting-frequency tier.

    Two binning methods are available.  The default, ``'threshold'``, uses
    fixed cut points motivated by the empirical percentile structure of
    lexical_master.csv, where the distribution is so right-skewed (skewness
    ≈ 114, P25 = P50 = 1) that equal-frequency quantile bins collapse — the
    standard ``pd.qcut`` with n=4 yields only 3 bins on the full corpus because
    the lower boundary edges are duplicates.

    **method='threshold'** (recommended)
        Fixed cuts at the percentile landmarks P75 (4 posts) and P95 (25 posts),
        plus a hard boundary for the one-time-poster spike at n=1:

        ============  =============  =============  ==========================
        Tier          Range          Percentile     Interpretation
        ============  =============  =============  ==========================
        T1 (1)        n = 1          P0 – P46       One-time posters (visited
                                                    once and did not return)
        T2 (2–4)      2 ≤ n ≤ 4      P46 – P75      Casual / occasional
        T3 (5–25)     5 ≤ n ≤ 25     P75 – P95      Active contributors
        T4 (26+)      n ≥ 26         P95 – P100     Power users
        ============  =============  =============  ==========================

        The cut points are roughly one order of magnitude apart on a log scale,
        keeping the tiers interpretable and the upper two groups meaningfully
        separated from the one-time-poster mass.  Custom cuts can be supplied
        via ``threshold_cuts``.

    **method='quantile'**
        Equal-frequency bins via ``pd.qcut``.  Due to the heavy spike at n=1,
        this will produce fewer than ``n_quartiles`` bins on the full corpus
        (3 instead of 4).  Suitable for subsets where the distribution is less
        degenerate (e.g., the temporal panel after applying the min_obs filter).

    All binning is computed on the **speaker-level** distribution (one value per
    unique speaker_id) to avoid prolific users with many monthly rows inflating
    the upper quantile boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Master lexical DataFrame.  Must contain 'speaker_id' and `freq_col`.
    freq_col : str
        Column used for frequency binning.  Default is
        'num_utterances_by_speaker' (lifetime total posts across the corpus).
    n_quartiles : int
        Number of equal-frequency bins.  Only used when ``method='quantile'``.
        Default is 4.
    method : {'threshold', 'quantile'}
        Binning strategy.  Default is ``'threshold'``.
    threshold_cuts : sequence of float, optional
        Custom bin edges for ``method='threshold'``.  Must be a strictly
        increasing sequence starting at 0 and ending at ``np.inf``.  Default
        is ``[0, 1, 4, 25, np.inf]``, which corresponds to the T1–T4 tiers
        described above.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with 'freq_quartile' added as an ordered Categorical.
        Speakers with a NaN or zero `freq_col` value receive NaN.
    """
    _require_columns(df, ['speaker_id', freq_col])

    if method not in ('threshold', 'quantile'):
        raise ValueError(f"method must be 'threshold' or 'quantile', got {method!r}")

    # Deduplicate to one row per speaker with a valid, positive count.
    speaker_freq = (
        df[['speaker_id', freq_col]]
        .drop_duplicates('speaker_id')
        .copy()
    )
    speaker_freq[freq_col] = pd.to_numeric(speaker_freq[freq_col], errors='coerce')
    valid = speaker_freq[freq_col].notna() & (speaker_freq[freq_col] > 0)
    speaker_freq = speaker_freq[valid]

    if method == 'threshold':
        if threshold_cuts is None:
            # Default cuts motivated by P75=4 and P95=25 of the full corpus.
            cuts = [0, 1, 4, 25, np.inf]
        else:
            cuts = list(threshold_cuts)

        # Build labels: for finite upper bounds show "lo–hi", for the last
        # open-ended bin show "lo+".
        # Bin 0 uses include_lowest=True with left edge 0, so actual values
        # start at 1 (all speakers have > 0 posts by construction).
        labels = []
        for i in range(len(cuts) - 1):
            lo = 1 if i == 0 else int(cuts[i]) + 1
            hi = cuts[i + 1]
            tier = f'T{i + 1}'
            if np.isinf(hi):
                labels.append(f'{tier} ({lo}+)')
            elif lo == int(hi):
                labels.append(f'{tier} ({lo})')
            else:
                labels.append(f'{tier} ({lo}\u2013{int(hi)})')

        speaker_freq['freq_quartile'] = pd.cut(
            speaker_freq[freq_col],
            bins=cuts,
            labels=labels,
            include_lowest=True,
        )

    else:  # method == 'quantile'
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
    method: str = 'threshold',
    threshold_cuts: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Prepare a speaker-month panel for the temporal frequency analysis.

    Convenience wrapper that (1) assigns posting-frequency tiers via
    :func:`assign_freq_quartiles`, (2) adds a relative-month counter via
    :func:`add_relative_month`, and (3) filters to speakers with at least
    ``min_obs`` distinct monthly rows so every retained user has a trajectory
    worth plotting.

    For the **quartile-distribution analysis** (Analysis 2), call
    :func:`assign_freq_quartiles` directly instead — no temporal filter should
    be applied there, since all speakers should be included.

    Note: within the filtered panel (min_obs ≥ 6), the one-time-poster tier
    (T1) is empty by construction, so the temporal plot will naturally show
    only the active tiers (T2–T4).

    Parameters
    ----------
    df : pd.DataFrame
        Raw lexical master DataFrame (one row per user-month).
    freq_col : str
        Frequency column for quartile assignment.
        Default is 'num_utterances_by_speaker'.
    n_quartiles : int
        Number of frequency bins; only used when ``method='quantile'``.
        Default is 4.
    min_obs : int
        Minimum number of monthly rows a speaker must have to be retained.
        Default is 6 (roughly half a year of consistent activity).
    method : {'threshold', 'quantile'}
        Binning strategy passed to :func:`assign_freq_quartiles`.
        Default is ``'threshold'``.
    threshold_cuts : sequence of float, optional
        Custom bin edges when ``method='threshold'``.  Passed through to
        :func:`assign_freq_quartiles`.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with 'freq_quartile' and 'relative_month' added.
        A retention summary is printed to stdout.
    """
    out = assign_freq_quartiles(
        df,
        freq_col=freq_col,
        n_quartiles=n_quartiles,
        method=method,
        threshold_cuts=threshold_cuts,
    )
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
        negate = col in NEGATED_METRICS
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

            if negate:
                grouped['mean'] = -grouped['mean']

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

        ylabel = METRIC_LABELS.get(col, col)
        if negate:
            ylabel = f'−{ylabel}'
        ax.set_ylabel(ylabel)
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
        'Posting Frequency Tier  (T1 = least frequent · T4 = most frequent)'
    )
    fig.suptitle(
        f'Lexical Quality by Posting Frequency Tier  ({agg_note})',
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


# -----------------------------------------------------------------------------------------
# Effect size analysis
# -----------------------------------------------------------------------------------------

def compute_effect_sizes(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    agg_to_speaker: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute rank-biserial correlation effect sizes for all pairwise tier comparisons.

    The rank-biserial correlation r is derived from the Mann-Whitney U statistic:

        r = 2·U₁₂ / (n₁·n₂) − 1

    where U₁₂ is the U statistic for group_1 vs group_2 (i.e. how often a
    randomly drawn observation from group_1 exceeds one from group_2).  Values
    range from −1 to +1; r > 0 means group_1 tends to score higher on the
    metric.

    For metrics in NEGATED_METRICS (Yule's K, Zipf Score), where lower raw
    values correspond to higher lexical quality, the sign of r is flipped so
    that positive r consistently means group_1 has *greater* lexical quality
    than group_2.

    Effect size magnitudes follow the benchmarks in Cohen (1988):

        |r| < 0.10  → negligible
        |r| < 0.30  → small
        |r| < 0.50  → medium
        |r| ≥ 0.50  → large

    BH correction is applied jointly across all pairwise comparisons and all
    metrics (same procedure as :func:`run_freq_quartile_tests`).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'freq_quartile', 'speaker_id', and all metric columns.
        Call :func:`assign_freq_quartiles` first.
    metrics : list of str, optional
        Metrics to compute effect sizes for.  Defaults to LEXICAL_METRICS.
    agg_to_speaker : bool
        If True (default), average each speaker's monthly rows first so every
        speaker contributes one observation per metric.
    alpha : float
        BH-corrected significance threshold (default 0.05).

    Returns
    -------
    pd.DataFrame
        Tidy table, one row per (metric, group_1, group_2), with columns:

        metric      – lexical metric name
        group_1     – reference tier label
        group_2     – comparison tier label
        n_1         – number of speakers in group_1
        n_2         – number of speakers in group_2
        U_stat      – Mann-Whitney U statistic (group_1 vs group_2)
        r           – rank-biserial correlation (sign-adjusted for NEGATED_METRICS)
        magnitude   – 'negligible', 'small', 'medium', or 'large'
        p_value     – raw two-sided p-value
        p_adjusted  – BH-corrected p-value
        significant – True if p_adjusted <= alpha
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
        negate = metric in NEGATED_METRICS
        for q1, q2 in pairs:
            g1 = test_df.loc[test_df['freq_quartile'] == q1, metric].dropna().values
            g2 = test_df.loc[test_df['freq_quartile'] == q2, metric].dropna().values

            if len(g1) < 2 or len(g2) < 2:
                records.append(dict(
                    metric=metric, group_1=q1, group_2=q2,
                    n_1=len(g1), n_2=len(g2),
                    U_stat=np.nan, r=np.nan,
                    magnitude='insufficient data',
                    p_value=np.nan, p_adjusted=np.nan, significant=False,
                ))
                continue

            U_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
            r = 2 * U_stat / (len(g1) * len(g2)) - 1
            if negate:
                r = -r  # flip: positive r → group_1 has greater lexical quality

            abs_r = abs(r)
            if abs_r < 0.10:
                magnitude = 'negligible'
            elif abs_r < 0.30:
                magnitude = 'small'
            elif abs_r < 0.50:
                magnitude = 'medium'
            else:
                magnitude = 'large'

            records.append(dict(
                metric=metric, group_1=q1, group_2=q2,
                n_1=len(g1), n_2=len(g2),
                U_stat=round(float(U_stat), 2),
                r=round(float(r), 4),
                magnitude=magnitude,
                p_value=p_val, p_adjusted=np.nan, significant=False,
            ))

    result_df = pd.DataFrame(records)

    valid = result_df['p_value'].notna()
    if valid.any():
        raw_p = result_df.loc[valid, 'p_value'].to_numpy(dtype=float)
        adj_p = _bh_adjust(raw_p)
        result_df.loc[valid, 'p_adjusted'] = np.round(adj_p, 6)
        result_df.loc[valid, 'significant'] = adj_p <= alpha

    result_df['p_value'] = result_df['p_value'].round(6)
    return result_df


def plot_effect_size_heatmap(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    agg_to_speaker: bool = True,
    alpha: float = 0.05,
) -> None:
    """Plot a heatmap of rank-biserial correlation effect sizes across frequency tiers.

    One heatmap is drawn per metric, arranged in a row.  Each cell [row, col]
    shows the rank-biserial r for (row tier) vs (col tier): positive values
    (blue) mean the row tier has greater lexical quality; negative values (red)
    mean the row tier has lower lexical quality.  The diagonal is zero by
    definition and is left blank.

    Significance conventions (BH-corrected at ``alpha``):

        ***  p_adjusted < 0.001
        **   p_adjusted < 0.01
        *    p_adjusted < 0.05
        ns   not significant

    Non-significant cells are visually distinguished with a grey diagonal cross
    so that significant differences stand out immediately.

    For NEGATED_METRICS (Yule's K, Zipf Score), r is sign-adjusted before
    plotting so that blue consistently indicates greater lexical quality,
    regardless of the raw metric direction.

    Calls :func:`compute_effect_sizes` internally.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'freq_quartile', 'speaker_id', and all metric columns.
        Call :func:`assign_freq_quartiles` first.
    metrics : list of str, optional
        Defaults to LEXICAL_METRICS.
    agg_to_speaker : bool
        Average each speaker's rows first (default True).
    alpha : float
        BH-corrected significance threshold (default 0.05).
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    effect_df = compute_effect_sizes(
        df, metrics=metrics, agg_to_speaker=agg_to_speaker, alpha=alpha
    )

    tiers = sorted(
        set(effect_df['group_1'].dropna().tolist()) |
        set(effect_df['group_2'].dropna().tolist())
    )
    n_tiers = len(tiers)
    tier_idx = {t: i for i, t in enumerate(tiers)}
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(4.8 * n_metrics, 4.2),
        squeeze=False,
    )
    axes = axes[0]  # unpack the single row

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        mdf = effect_df[effect_df['metric'] == metric]

        # Build full n×n matrices for r, significance stars, and p_adjusted.
        r_mat = np.full((n_tiers, n_tiers), np.nan)
        np.fill_diagonal(r_mat, 0.0)
        sig_mat = np.zeros((n_tiers, n_tiers), dtype=bool)
        padj_mat = np.ones((n_tiers, n_tiers))

        for _, row in mdf.iterrows():
            i, j = tier_idx[row['group_1']], tier_idx[row['group_2']]
            r_val = row['r'] if pd.notna(row['r']) else 0.0
            p_adj = row['p_adjusted'] if pd.notna(row['p_adjusted']) else 1.0
            sig   = bool(row['significant'])
            # antisymmetric: r(A→B) = −r(B→A)
            r_mat[i, j] =  r_val;  r_mat[j, i] = -r_val
            sig_mat[i, j] = sig;   sig_mat[j, i] = sig
            padj_mat[i, j] = p_adj; padj_mat[j, i] = p_adj

        # Draw heatmap with seaborn (annotation built separately below).
        annot = np.empty((n_tiers, n_tiers), dtype=object)
        for i in range(n_tiers):
            for j in range(n_tiers):
                if i == j:
                    annot[i, j] = ''
                    continue
                r_val = r_mat[i, j]
                p_adj = padj_mat[i, j]
                sig   = sig_mat[i, j]
                if   p_adj < 0.001: stars = '***'
                elif p_adj < 0.01:  stars = '**'
                elif p_adj < 0.05:  stars = '*'
                else:               stars = 'ns'
                annot[i, j] = f'{r_val:+.2f}\n{stars}'

        # Mask diagonal for seaborn (NaN → grey square).
        plot_mat = r_mat.copy()
        np.fill_diagonal(plot_mat, np.nan)
        mask_diag = np.zeros((n_tiers, n_tiers), dtype=bool)
        np.fill_diagonal(mask_diag, True)

        sns.heatmap(
            plot_mat,
            mask=mask_diag,
            annot=annot,
            fmt='',
            cmap='RdBu_r',
            vmin=-1, vmax=1, center=0,
            linewidths=0.6, linecolor='white',
            xticklabels=tiers, yticklabels=tiers,
            cbar_kws={'label': 'r  (rank-biserial)', 'shrink': 0.82},
            annot_kws={'size': 8.5},
            ax=ax,
        )

        # Shade diagonal cells grey.
        for k in range(n_tiers):
            ax.add_patch(plt.Rectangle(
                (k, k), 1, 1,
                color='#d0d0d0', zorder=2, linewidth=0,
            ))

        # Add a subtle grey cross to non-significant off-diagonal cells.
        for i in range(n_tiers):
            for j in range(n_tiers):
                if i != j and not sig_mat[i, j]:
                    cx, cy = j + 0.5, i + 0.5
                    ax.plot(
                        [j + 0.12, j + 0.88], [i + 0.12, i + 0.88],
                        color='#999999', linewidth=1.0, zorder=3,
                    )
                    ax.plot(
                        [j + 0.12, j + 0.88], [i + 0.88, i + 0.12],
                        color='#999999', linewidth=1.0, zorder=3,
                    )

        title = METRIC_LABELS.get(metric, metric)
        if metric in NEGATED_METRICS:
            title += '\n(sign-adjusted)'
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=8)
        ax.set_xlabel('Comparison Tier', fontsize=8.5)
        ax.set_ylabel('Reference Tier', fontsize=8.5)
        ax.tick_params(axis='both', labelsize=8)

    fig.suptitle(
        'Pairwise Effect Sizes by Frequency Tier  (rank-biserial r)\n'
        'Blue = row tier has greater lexical quality · '
        'grey cross = non-significant (BH-corrected)',
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------------------
# Frequency distribution diagnostics
# -----------------------------------------------------------------------------------------

def _extract_speaker_freq(
    df: pd.DataFrame,
    freq_col: str = 'num_utterances_by_speaker',
) -> pd.Series:
    """Return a Series of positive per-speaker post counts (one value per speaker)."""
    _require_columns(df, ['speaker_id', freq_col])
    s = (
        df[['speaker_id', freq_col]]
        .drop_duplicates('speaker_id')
        [freq_col]
        .pipe(pd.to_numeric, errors='coerce')
        .dropna()
    )
    return s[s > 0].reset_index(drop=True)


def summarize_freq_distribution(
    df: pd.DataFrame,
    freq_col: str = 'num_utterances_by_speaker',
    low_count_thresholds: Sequence[int] = (1, 2, 3, 5, 10),
) -> None:
    """Print descriptive statistics for the speaker posting-frequency distribution.

    Computes summary statistics and percentiles on the speaker-level post-count
    distribution, and shows what fraction of speakers fall at or below each of
    several low-count thresholds.  Useful for motivating bin-selection choices.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'speaker_id' and `freq_col`.
    freq_col : str
        Column holding per-speaker post counts.
    low_count_thresholds : sequence of int
        Post-count thresholds for the low-count breakdown table.
    """
    s = _extract_speaker_freq(df, freq_col)

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    pct_values = np.percentile(s, percentiles)

    print("=== Speaker Post-Count Distribution ===")
    print(f"  N speakers      : {len(s):>12,}")
    print(f"  Mean            : {s.mean():>12.2f}")
    print(f"  Median          : {s.median():>12.2f}")
    print(f"  Std dev         : {s.std():>12.2f}")
    print(f"  Skewness        : {skew(s):>12.2f}  (>1 = right-skewed)")
    print(f"  Excess kurtosis : {kurtosis(s):>12.2f}")
    print(f"  Min / Max       : {s.min():>6.0f} / {s.max():>,.0f}")
    print()
    print("=== Percentiles ===")
    for p, v in zip(percentiles, pct_values):
        print(f"  P{str(p):>5}  :  {v:>8.0f} posts")
    print()
    print("=== Low-Count Speaker Breakdown ===")
    for t in low_count_thresholds:
        frac = (s <= t).mean()
        print(f"  <= {t:>2} post(s)  : {frac:.1%} of speakers")


def plot_freq_distribution(
    df: pd.DataFrame,
    freq_col: str = 'num_utterances_by_speaker',
) -> None:
    """Plot a three-panel diagnostic of the speaker posting-frequency distribution.

    Panels
    ------
    1. Linear-scale histogram capped at P99 (shows the spike at low counts).
    2. Log₁₀-scale histogram with dashed lines at the P25 / P50 / P75 cut points.
    3. Empirical CDF on a log₁₀ x-axis with the same quartile markers.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'speaker_id' and `freq_col`.
    freq_col : str
        Column holding per-speaker post counts.
    """
    s = _extract_speaker_freq(df, freq_col)
    q_vals = np.percentile(s, [25, 50, 75])
    colors = ['#e07b54', '#c0392b', '#8e44ad']
    q_labels = ['P25', 'P50', 'P75']

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # Panel 1: linear-scale histogram capped at P99
    ax = axes[0]
    cap = np.percentile(s, 99)
    ax.hist(s[s <= cap], bins=80, color='steelblue', edgecolor='none', alpha=0.85)
    ax.set_xlabel('Posts per Speaker')
    ax.set_ylabel('Number of Speakers')
    ax.set_title('Distribution (capped at P99)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Panel 2: log₁₀-scale histogram with quartile markers
    ax = axes[1]
    ax.hist(np.log10(s), bins=80, color='steelblue', edgecolor='none', alpha=0.85)
    ax.set_xlabel('log₁₀(Posts per Speaker)')
    ax.set_ylabel('Number of Speakers')
    ax.set_title('Distribution (log₁₀ scale)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    for qv, c, lbl in zip(q_vals, colors, q_labels):
        ax.axvline(
            np.log10(qv), color=c, linestyle='--', linewidth=1.4,
            label=f'{lbl} = {qv:.0f} posts',
        )
    ax.legend(fontsize=8)

    # Panel 3: empirical CDF
    ax = axes[2]
    sorted_s = np.sort(s)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    ax.plot(np.log10(sorted_s), cdf, color='steelblue', linewidth=1.5)
    for qv, c, level in zip(q_vals, colors, [0.25, 0.50, 0.75]):
        ax.axvline(np.log10(qv), color=c, linestyle='--', linewidth=1.2)
        ax.axhline(level, color=c, linestyle=':', linewidth=1.0)
    ax.set_xlabel('log₁₀(Posts per Speaker)')
    ax.set_ylabel('Cumulative Fraction of Speakers')
    ax.set_title('Empirical CDF')
    ax.set_ylim(0, 1)

    fig.suptitle('Speaker Posting Frequency: Skew Diagnostic', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()


def diagnose_quartile_bins(
    df: pd.DataFrame,
    freq_col: str = 'num_utterances_by_speaker',
    n_quartiles: int = 4,
) -> None:
    """Print a diagnostic showing how many bins pd.qcut actually produces.

    When the posting-frequency distribution has many tied low values,
    ``pd.qcut(..., duplicates='drop')`` silently collapses duplicate bin
    edges and produces fewer bins than requested.  This function makes that
    visible by printing the actual bin edges and the speaker count in each bin.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'speaker_id' and `freq_col`.
    freq_col : str
        Column holding per-speaker post counts.
    n_quartiles : int
        Number of bins requested (default 4).
    """
    s = _extract_speaker_freq(df, freq_col)

    _, raw_bins = pd.qcut(s, q=n_quartiles, retbins=True, duplicates='drop')
    n_bins = len(raw_bins) - 1

    print(f"Requested bins : {n_quartiles}")
    print(f"Bins produced  : {n_bins}"
          + ("  <-- duplicate edges dropped" if n_bins < n_quartiles else ""))
    print(f"Bin edges      : {[int(b) for b in raw_bins]}")
    print()

    labels = [f'Q{i + 1}' for i in range(n_bins)]
    assigned = pd.cut(s, bins=raw_bins, labels=labels, include_lowest=True)
    counts = assigned.value_counts().sort_index()
    for label, lo, hi in zip(labels, raw_bins[:-1], raw_bins[1:]):
        n = counts.get(label, 0)
        print(
            f"  {label}: [{int(lo):>4}, {int(hi):>6}] posts"
            f"  →  {n:>10,} speakers  ({100 * n / len(s):.1f}%)"
        )
