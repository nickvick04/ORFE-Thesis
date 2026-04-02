# ----------------------------------------------------------------------------------------
# Stationarity testing (ADF / KPSS) and autocorrelation testing (Ljung-Box)
# for subreddit-month aggregated time series
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
from statsmodels.stats.diagnostic import acorr_ljungbox
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


# ----------------------------------------------------------------------------------------
# First-differencing
# ----------------------------------------------------------------------------------------

def first_difference(
    agg: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
) -> pd.DataFrame:
    """Return a copy of *agg* with each metric column first-differenced within subreddit.

    First-differencing transforms a series Y_t into ΔY_t = Y_t − Y_{t−1}, which
    removes a stochastic trend (unit root) and is the standard remedy when ADF/KPSS
    tests indicate nonstationarity. The differenced DataFrame has the same structure
    as *agg* and can be passed directly to ``run_stationarity_tests`` or
    ``run_ljung_box_tests`` to confirm that the differenced series are stationary.

    The first observation per subreddit becomes NaN after differencing and is dropped,
    so the returned DataFrame has one fewer row per subreddit than *agg*.

    Parameters
    ----------
    agg : pd.DataFrame
        Subreddit-month aggregated DataFrame produced by the Aggregation step.
        Must be sorted by *time_col* within each subreddit (the standard output
        of the aggregation step already satisfies this).
    metrics : sequence of str, optional
        Metric columns to difference. Defaults to LEXICAL_METRICS. Non-metric
        columns (subreddit, year_month, etc.) are carried through unchanged.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').

    Returns
    -------
    pd.DataFrame
        First-differenced copy of *agg* with the same columns, sorted by
        (subreddit_col, time_col). Rows where differencing produced NaN (the
        first observation per subreddit) are dropped.

    Examples
    --------
    >>> agg_diff = first_difference(agg)
    >>> # Confirm unit roots are resolved
    >>> run_stationarity_tests(agg_diff)
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    missing_metrics = [m for m in metrics if m not in agg.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")
    for col in (subreddit_col, time_col):
        if col not in agg.columns:
            raise ValueError(f"Missing required column: '{col}'")

    agg_sorted = agg.sort_values([subreddit_col, time_col]).copy()

    agg_sorted[list(metrics)] = (
        agg_sorted
        .groupby(subreddit_col, sort=False)[list(metrics)]
        .diff()
    )

    agg_diff = agg_sorted.dropna(subset=list(metrics)).reset_index(drop=True)
    return agg_diff


# ----------------------------------------------------------------------------------------
# Autocorrelation testing (Ljung-Box)
# ----------------------------------------------------------------------------------------

LB_CONCLUSIONS = [
    "no autocorrelation",
    "autocorrelated",
    "insufficient data",
]

LB_CONCLUSION_COLORS = {
    "no autocorrelation": "#27ae60",
    "autocorrelated":     "#e74c3c",
    "insufficient data":  "#ecf0f1",
}

LB_CONCLUSION_SHORT = {
    "no autocorrelation": "✓",
    "autocorrelated":     "AC",
    "insufficient data":  "—",
}


def run_ljung_box_tests(
    agg: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    lags: Sequence[int] = (6, 12),
    min_obs: int = 15,
) -> pd.DataFrame:
    """Run Ljung-Box portmanteau tests on each (subreddit, metric) time series.

    The Ljung-Box test (Ljung & Box, 1978) tests the joint null hypothesis that
    autocorrelations up to lag h are all zero. Rejection of the null at a given
    lag indicates the presence of serial dependence that may need to be accounted
    for in downstream regression models (e.g. via Newey-West HAC standard errors).

    Two lags are tested by default:
      h = 6   – half a seasonal cycle; catches short-range AR structure.
      h = 12  – one full seasonal cycle; the primary decision lag for monthly data.

    The conclusion is based on the h = 12 result. If h = 12 is not in `lags`,
    the conclusion is based on the largest lag provided.

    Parameters
    ----------
    agg : pd.DataFrame
        Subreddit-month aggregated DataFrame (output of the Aggregation step).
        Must contain subreddit_col, time_col (as 'YYYY-MM' strings), and metric columns.
    metrics : sequence of str, optional
        Metric columns to test. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    alpha : float
        Significance level (default 0.05).
    lags : sequence of int
        Lags at which to evaluate the test statistic (default (6, 12)).
    min_obs : int
        Minimum monthly observations required to run the test (default 15).
        Must be greater than max(lags).

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
          subreddit, metric, n_obs,
          lb_stat_h{k}, lb_p_h{k}   for each k in lags,
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

    lags = sorted(lags)
    decision_lag = lags[-1]  # primary conclusion lag (h = 12 by default)
    min_obs = max(min_obs, decision_lag + 1)

    subreddits = sorted(agg[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:
        for metric in metrics:
            series = _extract_series(agg, subreddit, metric, subreddit_col, time_col)
            n = len(series)

            if n < min_obs:
                row = dict(
                    subreddit=subreddit, metric=metric, n_obs=n,
                    conclusion="insufficient data",
                )
                for k in lags:
                    row[f"lb_stat_h{k}"] = np.nan
                    row[f"lb_p_h{k}"]    = np.nan
                records.append(row)
                continue

            lb = acorr_ljungbox(series.values, lags=lags, return_df=True)

            row = dict(subreddit=subreddit, metric=metric, n_obs=n)
            for k in lags:
                row[f"lb_stat_h{k}"] = round(float(lb.loc[k, "lb_stat"]),  4)
                row[f"lb_p_h{k}"]    = round(float(lb.loc[k, "lb_pvalue"]), 6)

            p_decision = row[f"lb_p_h{decision_lag}"]
            row["conclusion"] = "autocorrelated" if p_decision < alpha else "no autocorrelation"
            records.append(row)

    stat_cols  = [f"lb_stat_h{k}" for k in lags]
    p_cols     = [f"lb_p_h{k}"    for k in lags]
    result_cols = ["subreddit", "metric", "n_obs"] + stat_cols + p_cols + ["conclusion"]

    if not records:
        return pd.DataFrame(columns=result_cols)
    return pd.DataFrame(records, columns=result_cols)


def summarize_ljung_box(results: pd.DataFrame) -> None:
    """Print a plain-text summary of Ljung-Box autocorrelation test results.

    Shows overall conclusion counts and the most common conclusion per metric.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_ljung_box_tests().
    """
    if results.empty or "conclusion" not in results.columns:
        print("No Ljung-Box results to summarize — check that agg is non-empty "
              "and contains the expected subreddit and year_month columns.")
        return

    total = len(results)
    counts = results["conclusion"].value_counts()

    # Detect which lags were tested from column names
    p_cols = sorted([c for c in results.columns if c.startswith("lb_p_h")])
    lag_labels = ", ".join(c.replace("lb_p_h", "h=") for c in p_cols)

    print("=== Ljung-Box Autocorrelation Test Summary ===")
    print(f"Lags tested: {lag_labels}   |   Conclusion based on: {p_cols[-1].replace('lb_p_h', 'h=')}")
    print(f"Total (subreddit × metric) series tested: {total}\n")

    print("Overall conclusions:")
    for conclusion in LB_CONCLUSIONS:
        count = counts.get(conclusion, 0)
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  {conclusion:<25} {count:>4}  ({pct:.1f}%)")

    print("\nBreakdown by metric:")
    for metric in LEXICAL_METRICS:
        sub = results[results["metric"] == metric]
        if sub.empty:
            continue
        top = sub["conclusion"].value_counts().index[0]
        label = METRIC_LABELS.get(metric, metric)
        print(f"  {label:<18} → most common: {top}")


def plot_ljung_box_heatmap(
    results: pd.DataFrame,
    save_path: Optional["str | Path"] = None,
) -> None:
    """Plot a heatmap of Ljung-Box conclusions for each (metric × subreddit).

    Color coding:
      Green  (✓)  – no autocorrelation
      Red    (AC) – autocorrelated
      Light  (—)  – insufficient data

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_ljung_box_tests().
    save_path : str or Path, optional
        If provided, the figure is saved here rather than displayed.
    """
    pivot = results.pivot(index="metric", columns="subreddit", values="conclusion")

    ordered = [m for m in LEXICAL_METRICS if m in pivot.index]
    pivot = pivot.reindex(ordered)

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.7), max(3, n_rows * 1.0)))

    numeric = pivot.map(
        lambda c: LB_CONCLUSIONS.index(c) if c in LB_CONCLUSIONS else len(LB_CONCLUSIONS) - 1
    )
    cmap = ListedColormap([LB_CONCLUSION_COLORS[c] for c in LB_CONCLUSIONS])
    ax.imshow(numeric.values, cmap=cmap, vmin=0, vmax=len(LB_CONCLUSIONS) - 1, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.iloc[i, j]
            short = LB_CONCLUSION_SHORT.get(val, "?")
            text_color = "white" if val in ("no autocorrelation", "autocorrelated") else "#333333"
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
        "Ljung-Box Autocorrelation Test Conclusions by Metric and Subreddit",
        fontsize=13, fontweight="bold", pad=14,
    )

    patches = [
        mpatches.Patch(color=LB_CONCLUSION_COLORS[c], label=c.capitalize())
        for c in LB_CONCLUSIONS
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.show()
