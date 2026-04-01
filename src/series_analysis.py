# ----------------------------------------------------------------------------------------
# Stationarity testing (ADF / KPSS) for subreddit-month aggregated time series
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from statsmodels.tsa.stattools import adfuller, kpss

LEXICAL_METRICS = [
    "mtld_score",
    "mattr_score",
    "yules_k",
    "zipf_score",
    "aoa_score",
    "nawl_ratio",
]

METRIC_LABELS = {
    "mtld_score":   "MTLD",
    "mattr_score":  "MATTR",
    "yules_k":      "Yule's K",
    "zipf_score":   "Zipf Score",
    "aoa_score":    "AoA",
    "nawl_ratio":   "NAWL Ratio",
}

# Ordered list of possible conclusions — order determines heatmap color mapping.
CONCLUSIONS = [
    "stationary",
    "unit root",
    "inconclusive (both reject)",
    "inconclusive (neither rejects)",
    "insufficient data",
]

CONCLUSION_COLORS = {
    "stationary":                    "#27ae60",
    "unit root":                     "#e74c3c",
    "inconclusive (both reject)":    "#f39c12",
    "inconclusive (neither rejects)":"#95a5a6",
    "insufficient data":             "#ecf0f1",
}

CONCLUSION_SHORT = {
    "stationary":                    "S",
    "unit root":                     "U",
    "inconclusive (both reject)":    "I+",
    "inconclusive (neither rejects)":"I−",
    "insufficient data":             "—",
}


# ----------------------------------------------------------------------------------------
# Internal helper
# ----------------------------------------------------------------------------------------

def _extract_series(
    df: pd.DataFrame,
    subreddit: str,
    metric: str,
    subreddit_col: str,
    time_col: str,
) -> pd.Series:
    """Return a sorted, NaN-free monthly time series for one (subreddit, metric) pair.

    year_month strings (e.g. '2015-03') are parsed into a DatetimeIndex so the
    series is correctly ordered regardless of row order in the DataFrame.
    """
    sub = df[df[subreddit_col] == subreddit].copy()
    sub[time_col] = pd.to_datetime(
        sub[time_col].astype(str), format="%Y-%m", errors="coerce"
    )
    sub = sub.dropna(subset=[time_col, metric]).sort_values(time_col)
    return sub.set_index(time_col)[metric].dropna()


# ----------------------------------------------------------------------------------------
# Core test runner
# ----------------------------------------------------------------------------------------

def run_stationarity_tests(
    agg: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    adf_maxlag: int = 4,
    min_obs: int = 10,
) -> pd.DataFrame:
    """Run ADF and KPSS stationarity tests on each (subreddit, metric) time series.

    ADF null hypothesis  : a unit root is present (non-stationary).
                           Small p-value → evidence of stationarity.
    KPSS null hypothesis : the series is stationary.
                           Small p-value → evidence of a unit root.

    The two tests are run together because ADF has low power against near-unit-root
    stationary processes (DeJong et al., 1992); using both with complementary nulls
    provides a more reliable classification (Kwiatkowski et al., 1992).

    Combined conclusion logic:
      stationary                   – ADF rejects unit root AND KPSS does not reject
      unit root                    – ADF does not reject AND KPSS rejects stationarity
      inconclusive (both reject)   – both tests reject their respective nulls
      inconclusive (neither rejects) – neither test rejects (series ambiguous / short)
      insufficient data            – fewer than min_obs observations

    Parameters
    ----------
    agg : pd.DataFrame
        Subreddit-month aggregated DataFrame produced by the Aggregation step.
        Must contain subreddit_col, time_col (as 'YYYY-MM' strings), and metric columns.
    metrics : sequence of str, optional
        Metric columns to test. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    alpha : float
        Significance level for both tests (default 0.05).
    adf_maxlag : int
        Maximum lag order passed to adfuller; selected by AIC within this bound
        (default 4).
    min_obs : int
        Minimum monthly observations required to run tests (default 10).

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
          subreddit, metric, n_obs,
          adf_stat, adf_p, adf_lags, adf_stationary,
          kpss_stat, kpss_p, kpss_stationary,
          conclusion
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    missing_metrics = [m for m in metrics if m not in agg.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")
    for col in (subreddit_col, time_col):
        if col not in agg.columns:
            raise ValueError(f"Missing required column: '{col}'")

    subreddits = sorted(agg[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:
        for metric in metrics:
            series = _extract_series(agg, subreddit, metric, subreddit_col, time_col)
            n = len(series)

            if n < min_obs:
                records.append(dict(
                    subreddit=subreddit, metric=metric, n_obs=n,
                    adf_stat=np.nan, adf_p=np.nan, adf_lags=np.nan,
                    adf_stationary=np.nan,
                    kpss_stat=np.nan, kpss_p=np.nan,
                    kpss_stationary=np.nan,
                    conclusion="insufficient data",
                ))
                continue

            # --- ADF ---
            adf_out = adfuller(series.values, maxlag=adf_maxlag, autolag="AIC")
            adf_stat      = float(adf_out[0])
            adf_p         = float(adf_out[1])
            adf_lags      = int(adf_out[2])
            adf_stationary = adf_p < alpha

            # --- KPSS (suppress statsmodels p-value boundary warnings) ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_out = kpss(series.values, regression="c", nlags="auto")
            kpss_stat      = float(kpss_out[0])
            kpss_p         = float(kpss_out[1])
            kpss_stationary = kpss_p > alpha  # fail to reject → stationary

            # --- Combined conclusion ---
            if adf_stationary and kpss_stationary:
                conclusion = "stationary"
            elif not adf_stationary and not kpss_stationary:
                conclusion = "unit root"
            elif adf_stationary and not kpss_stationary:
                conclusion = "inconclusive (both reject)"
            else:
                conclusion = "inconclusive (neither rejects)"

            records.append(dict(
                subreddit=subreddit,
                metric=metric,
                n_obs=n,
                adf_stat=round(adf_stat, 4),
                adf_p=round(adf_p, 6),
                adf_lags=adf_lags,
                adf_stationary=adf_stationary,
                kpss_stat=round(kpss_stat, 4),
                kpss_p=round(kpss_p, 6),
                kpss_stationary=kpss_stationary,
                conclusion=conclusion,
            ))

    RESULT_COLS = [
        "subreddit", "metric", "n_obs",
        "adf_stat", "adf_p", "adf_lags", "adf_stationary",
        "kpss_stat", "kpss_p", "kpss_stationary",
        "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)
    return pd.DataFrame(records, columns=RESULT_COLS)


# ----------------------------------------------------------------------------------------
# Summary and visualization
# ----------------------------------------------------------------------------------------

def summarize_stationarity(results: pd.DataFrame) -> None:
    """Print a plain-text summary of stationarity test results.

    Shows overall conclusion counts and the most common conclusion per metric.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_stationarity_tests().
    """
    if results.empty or "conclusion" not in results.columns:
        print("No stationarity results to summarize — check that agg is non-empty "
              "and contains the expected subreddit and year_month columns.")
        return

    total = len(results)
    counts = results["conclusion"].value_counts()

    print("=== Stationarity Test Summary ===")
    print(f"Total (subreddit \u00d7 metric) series tested: {total}\n")

    print("Overall conclusions:")
    for conclusion in CONCLUSIONS:
        count = counts.get(conclusion, 0)
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  {conclusion:<38} {count:>4}  ({pct:.1f}%)")

    print("\nBreakdown by metric:")
    for metric in LEXICAL_METRICS:
        sub = results[results["metric"] == metric]
        if sub.empty:
            continue
        top = sub["conclusion"].value_counts().index[0]
        label = METRIC_LABELS.get(metric, metric)
        print(f"  {label:<18} \u2192 most common: {top}")


def plot_stationarity_heatmap(
    results: pd.DataFrame,
    save_path: Optional["str | Path"] = None,
) -> None:
    """Plot a heatmap of stationarity conclusions for each (metric \u00d7 subreddit).

    Color coding:
      Green  (S)  \u2013 stationary
      Red    (U)  \u2013 unit root
      Orange (I+) \u2013 inconclusive (both reject)
      Grey   (I\u2212) \u2013 inconclusive (neither rejects)
      Light  (\u2014)  \u2013 insufficient data

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_stationarity_tests().
    save_path : str or Path, optional
        If provided, the figure is saved here rather than displayed.
    """
    pivot = results.pivot(index="metric", columns="subreddit", values="conclusion")

    ordered = [m for m in LEXICAL_METRICS if m in pivot.index]
    pivot = pivot.reindex(ordered)

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.7), max(3, n_rows * 1.0)))

    numeric = pivot.map(
        lambda c: CONCLUSIONS.index(c) if c in CONCLUSIONS else len(CONCLUSIONS) - 1
    )
    cmap = ListedColormap([CONCLUSION_COLORS[c] for c in CONCLUSIONS])
    ax.imshow(numeric.values, cmap=cmap, vmin=0, vmax=len(CONCLUSIONS) - 1, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.iloc[i, j]
            short = CONCLUSION_SHORT.get(val, "?")
            text_color = "white" if val in ("stationary", "unit root") else "#333333"
            ax.text(j, i, short, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [str(c).replace("subreddit-", "r/") for c in pivot.columns],
        rotation=40, ha="right", fontsize=9,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in pivot.index], fontsize=10)
    ax.set_title(
        "Stationarity Test Conclusions by Metric and Subreddit",
        fontsize=13, fontweight="bold", pad=14,
    )

    patches = [
        mpatches.Patch(color=CONCLUSION_COLORS[c], label=c.capitalize())
        for c in CONCLUSIONS
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.show()
