import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np

# define the metrics
LEXICAL_METRICS = ['mtld_score', 'mattr_score', 'yules_k', 'zipf_score', 'aoa_score', 'nawl_ratio']
SYNTACTIC_METRICS = ['fragment_ratio', 'avg_t_units', 'clause_to_t_unit_ratio', 'mltu']

# human-readable axis labels for each metric
METRIC_LABELS = {
    'mtld_score':             'MTLD Score',
    'mattr_score':            'MATTR Score',
    'yules_k':                "Yule's K",
    'zipf_score':             'Zipf Score (avg. word frequency)',
    'aoa_score':              'Age of Acquisition (years)',
    'nawl_ratio':             'NAWL Ratio',
    'fragment_ratio':         'Fragment Ratio',
    'avg_t_units':            'Avg. T-Units per Utterance',
    'clause_to_t_unit_ratio': 'Clause-to-T-Unit Ratio',
    'mltu':                   'Mean Length of T-Unit (words)',
}

def _with_datetime_index(df):
    '''Return a dataframe indexed by datetime timestamp.'''
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index or a 'timestamp' column.")
    return out.sort_index()

def _require_columns(df, cols):
    '''Raise a readable error when required metric columns are missing.'''
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def plot_lexical_trends_monthly(df, metrics=LEXICAL_METRICS):
    '''Plot monthly mean trends for lexical metrics.'''
    ts_df = _with_datetime_index(df)
    _require_columns(ts_df, metrics)
    monthly = ts_df[metrics].resample("M").mean()

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        axes[i].plot(monthly.index, monthly[col], color="tab:blue", linewidth=2)
        axes[i].set_ylabel(METRIC_LABELS.get(col, col))
        axes[i].grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha="right")
    axes[-1].set_xlabel("Month")
    fig.suptitle("Monthly Lexical Complexity Trends", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_lexical_trends_yearly(df, metrics=LEXICAL_METRICS):
    '''Plot yearly mean trends for lexical metrics.'''
    ts_df = _with_datetime_index(df)
    _require_columns(ts_df, metrics)
    yearly = ts_df[metrics].resample("Y").mean()

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        axes[i].plot(yearly.index, yearly[col], color="tab:green", linewidth=2)
        axes[i].set_ylabel(METRIC_LABELS.get(col, col))
        axes[i].grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha="right")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Yearly Lexical Complexity Trends", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def _frequency_bucket(series, q=5):
    '''Map num_utterances_by_speaker into quantile-based user-frequency buckets.
    Uses pd.qcut so each bucket contains approximately equal numbers of users.
    q controls the number of buckets (default 5); duplicates are dropped so the
    actual number of buckets may be smaller when many users share the same count.'''
    _, bins = pd.qcut(series, q=q, retbins=True, duplicates='drop')
    bins_int = np.round(bins).astype(int)
    n = len(bins_int) - 1
    labels = []
    for i in range(n):
        lo = int(bins_int[i]) + (1 if i > 0 else 0)
        hi = int(bins_int[i + 1])
        labels.append(f"{lo}+" if i == n - 1 else f"{lo}-{hi}")
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

def plot_complexity_by_user_frequency_over_time(df, metrics=LEXICAL_METRICS, freq="M"):
    '''Plot metric trends over time by fixed user-frequency buckets.'''
    ts_df = _with_datetime_index(df)
    _require_columns(ts_df, metrics)
    if "num_utterances_by_speaker" not in ts_df.columns:
        raise ValueError("Missing required column: num_utterances_by_speaker")

    ts_df = ts_df.copy()
    ts_df["num_utterances_by_speaker"] = pd.to_numeric(
        ts_df["num_utterances_by_speaker"], errors="coerce"
    )
    ts_df = ts_df.dropna(subset=["num_utterances_by_speaker"])
    ts_df = ts_df[ts_df["num_utterances_by_speaker"] > 0]
    ts_df["frequency_bucket"] = _frequency_bucket(ts_df["num_utterances_by_speaker"])

    grouped = (
        ts_df
        .groupby([pd.Grouper(freq=freq), "frequency_bucket"], observed=False)[metrics]
        .mean()
        .reset_index()
    )

    bucket_order = ts_df["frequency_bucket"].cat.categories.tolist()
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        for b in bucket_order:
            subset = grouped[grouped["frequency_bucket"].astype(str) == b]
            axes[i].plot(subset["timestamp"], subset[col], linewidth=1.8, label=b)
        axes[i].set_ylabel(METRIC_LABELS.get(col, col))
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(title="Posts/User", ncol=3, fontsize=9)

    if freq == "M":
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif freq == "Y":
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha="right")
    axes[-1].set_xlabel("Time")
    fig.suptitle("Lexical Complexity Over Time by User Frequency Group", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_complexity_by_user_frequency(df, metrics=LEXICAL_METRICS, bins=10):
    '''Visualize average complexity by user posting frequency (num_utterances_by_speaker).'''
    _require_columns(df, metrics)
    if "num_utterances_by_speaker" not in df.columns:
        raise ValueError("Missing required column: num_utterances_by_speaker")

    plot_df = df.copy()
    plot_df["num_utterances_by_speaker"] = pd.to_numeric(
        plot_df["num_utterances_by_speaker"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["num_utterances_by_speaker"])
    plot_df = plot_df[plot_df["num_utterances_by_speaker"] > 0]

    # Use quantile bins so each group has similar sample size.
    plot_df["frequency_bucket"] = pd.qcut(
        plot_df["num_utterances_by_speaker"],
        q=bins,
        duplicates="drop",
    )

    grouped = plot_df.groupby("frequency_bucket", observed=False)[metrics].mean()
    grouped.index = grouped.index.astype(str)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        sns.barplot(
            x=grouped.index,
            y=grouped[col].values,
            ax=axes[i],
            color="tab:orange",
        )
        axes[i].set_ylabel(METRIC_LABELS.get(col, col))
        axes[i].grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("User Frequency Bucket (quantiles of num_utterances_by_speaker)")
    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle("Average Lexical Complexity by User Posting Frequency", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_lexical_metrics(df, metrics=LEXICAL_METRICS, rolling_window=None, resample_freq=None):
    '''Plots lexical metrics from a dataframe on one figure with subplots, with a
    shaded 95% confidence interval band around each mean line.

    Parameters:
    - metrics: list of metric column names to plot (default: LEXICAL_METRICS).
    - rolling_window: int for size of rolling average window (post count).
      The band shows ±1.96 * (rolling SD / sqrt(rolling n)) at each point.
    - resample_freq: str, e.g., 'D', 'W', 'M' to aggregate metrics over time.
      The band shows ±1.96 * (SD / sqrt(n)) within each period — i.e., the 95%
      confidence interval on the period mean, not the spread of individual posts.

    Note: bands are only drawn when rolling_window or resample_freq is supplied,
    since raw (unaggregated) values have no within-period uncertainty to display.'''

    ts_df = _with_datetime_index(df)
    _require_columns(ts_df, metrics)
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        if resample_freq:
            resampled  = ts_df[col].resample(resample_freq).agg(['mean', 'std', 'count'])
            mean       = resampled['mean']
            margin     = 1.96 * resampled['std'] / np.sqrt(resampled['count'])
            label      = f'{resample_freq}-resampled mean ± 95% CI'
        elif rolling_window:
            mean       = ts_df[col].rolling(window=rolling_window, min_periods=1).mean()
            roll_std   = ts_df[col].rolling(window=rolling_window, min_periods=1).std()
            roll_n     = ts_df[col].rolling(window=rolling_window, min_periods=1).count()
            margin     = 1.96 * roll_std / np.sqrt(roll_n)
            label      = f'rolling mean ± 95% CI (window={rolling_window})'
        else:
            mean       = ts_df[col]
            margin     = None
            label      = 'raw values'

        axes[i].plot(mean.index, mean, color='tab:blue', linewidth=1.8)
        if margin is not None:
            axes[i].fill_between(
                mean.index,
                mean - margin,
                mean + margin,
                color='tab:blue',
                alpha=0.25,
                linewidth=0,
            )
        axes[i].set_ylabel(METRIC_LABELS.get(col, col))
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    method_desc = label if (rolling_window or resample_freq) else 'raw'
    fig.suptitle(f'Lexical Metrics Over Time ({method_desc})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_syntactic_metrics(df, rolling_window=None, resample_freq=None):
    '''Plots all syntactic metrics from a dataframe on one figure with subplots.
    Also takes two other paramters:
    - rolling_window: int for the size of rolling average window
    - resample_freq: str, e.g., 'D', 'W', 'M' to aggregate metrics over time.'''
    
    ts_df = _with_datetime_index(df)
    metrics = SYNTACTIC_METRICS
    _require_columns(ts_df, metrics)
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
    
    # Determine method description for title
    if resample_freq:
        method_desc = f'{resample_freq}-resampled mean'
    elif rolling_window:
        method_desc = f'rolling window={rolling_window}'
    else:
        method_desc = 'raw values'
    
    for i, col in enumerate(metrics):
        if resample_freq:
            series = ts_df[col].resample(resample_freq).mean()
        elif rolling_window:
            series = ts_df[col].rolling(window=rolling_window, min_periods=1).mean()
        else:
            series = ts_df[col]

        axes[i].plot(series.index, series, color='tab:green')
        axes[i].set_ylabel(METRIC_LABELS.get(col, col))
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle(f'Syntactic Metrics Over Time ({method_desc})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
