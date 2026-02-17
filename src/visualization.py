import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# define the metrics
LEXICAL_METRICS = ['mtld_score', 'yules_k', 'zipf_score', 'aoa_score', 'nawl_ratio']
SYNTACTIC_METRICS = ['fragment_ratio', 'avg_t_units', 'clause_to_t_unit_ratio', 'mltu']

def _with_datetime_index(df):
    '''Return a dataframe indexed by datetime timestamp.'''
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).set_index("timestamp")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a datetime index or a 'timestamp' column.")
    return out.sort_index()

def plot_lexical_trends_monthly(df, metrics=LEXICAL_METRICS):
    '''Plot monthly mean trends for lexical metrics.'''
    ts_df = _with_datetime_index(df)
    monthly = ts_df[metrics].resample("M").mean()

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(metrics):
        axes[i].plot(monthly.index, monthly[col], color="tab:blue", linewidth=2)
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Month")
    fig.suptitle("Monthly Lexical Complexity Trends", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_complexity_by_user_frequency(df, metrics=LEXICAL_METRICS, bins=10):
    '''Visualize average complexity by user posting frequency (num_utterances_by_speaker).'''
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
        axes[i].set_ylabel(col)
        axes[i].grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("User Frequency Bucket (quantiles of num_utterances_by_speaker)")
    axes[-1].tick_params(axis="x", rotation=35)
    fig.suptitle("Average Lexical Complexity by User Posting Frequency", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_lexical_metrics(df, rolling_window=None, resample_freq=None):
    '''Plots all lexical metrics from a dataframe on one figure with subplots.
    Takes two other parameters:
    - rolling_window: int for size of rolling average window (post count).
    - resample_freq: str, e.g., 'D', 'W', 'M' to aggregate metrics over time.'''
    
    metrics = LEXICAL_METRICS
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
    
    for i, col in enumerate(metrics):
        if resample_freq:
            series = df[col].resample(resample_freq).mean()
            label = f'{resample_freq}-resampled mean'
        elif rolling_window:
            series = df[col].rolling(window=rolling_window, min_periods=1).mean()
            label = f'rolling window={rolling_window})'
        else:
            series = df[col]
            label = 'raw values'
        
        axes[i].plot(series.index, series, color='tab:blue')
        axes[i].set_ylabel(col)
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
    
    metrics = SYNTACTIC_METRICS
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n), sharex=True)
    
    # Determine method description for title
    if resample_freq:
        method_desc = f'{resample_freq}-resampled mean'
    elif rolling_window:
        method_desc = f'rolling window={rolling_window})'
    else:
        method_desc = 'raw values'
    
    for i, col in enumerate(metrics):
        if resample_freq:
            series = df[col].resample(resample_freq).mean()
        elif rolling_window:
            series = df[col].rolling(window=rolling_window, min_periods=1).mean()
        else:
            series = df[col]
        
        axes[i].plot(series.index, series, color='tab:green')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle(f'Syntactic Metrics Over Time ({method_desc})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
