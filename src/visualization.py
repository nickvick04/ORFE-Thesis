import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# define the metrics
LEXICAL_METRICS = ['mtld_score', 'yules_k', 'zipf_score', 'aoa_score', 'nawl_ratio']
SYNTACTIC_METRICS = ['fragment_ratio', 'avg_t_units', 'clause_to_t_unit_ratio', 'mltu']

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
