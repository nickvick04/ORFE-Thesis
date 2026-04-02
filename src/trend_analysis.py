# ----------------------------------------------------------------------------------------
# Baseline OLS trend regression with Newey-West (HAC) standard errors
# Model: y_t = β₀ + β₁·t + ε_t  (t = 0, 1, 2, … months)
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from series_analysis import (
    LEXICAL_METRICS,
    METRIC_LABELS,
    _extract_series,
)

# Automatic Newey-West bandwidth: Andrews (1991) / Newey-West (1994) rule of thumb.
# statsmodels uses this when maxlags=None is passed to cov_kwds.
_NW_AUTO = None

TREND_CONCLUSIONS = [
    "upward trend",
    "downward trend",
    "no significant trend",
    "insufficient data",
]

TREND_CONCLUSION_COLORS = {
    "upward trend":          "#2980b9",
    "downward trend":        "#e74c3c",
    "no significant trend":  "#95a5a6",
    "insufficient data":     "#ecf0f1",
}


# ----------------------------------------------------------------------------------------
# Core regression runner
# ----------------------------------------------------------------------------------------

def run_ols_trend(
    agg: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    hac_maxlags: Optional[int] = _NW_AUTO,
    min_obs: int = 10,
) -> pd.DataFrame:
    """Fit a baseline OLS trend regression with Newey-West HAC standard errors.

    Model (estimated separately for each subreddit × metric pair):

        y_t = β₀ + β₁·t + ε_t

    where t is an integer time index (0, 1, 2, …) so that β₁ is the estimated
    change in the metric per calendar month. HAC standard errors (Newey-West)
    are used to produce valid inference under serial correlation and
    heteroskedasticity in ε_t.

    Note on validity: this regression is well-specified when the series has a
    *deterministic* trend (trend-stationary). If ADF/KPSS results indicate a
    *stochastic* trend (unit root), the OLS estimates are superseded by an
    ARIMA-with-drift model and should be treated as a baseline only.

    Parameters
    ----------
    agg : pd.DataFrame
        Subreddit-month aggregated DataFrame produced by the Aggregation step.
        Must contain subreddit_col, time_col (as 'YYYY-MM' strings), and metric
        columns. The series is internally sorted by time_col before fitting.
    metrics : sequence of str, optional
        Metric columns to test. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    alpha : float
        Significance level for the trend coefficient β₁ (default 0.05).
    hac_maxlags : int or None
        Maximum lag order for the Newey-West kernel. None (default) triggers
        automatic selection via the rule of thumb: floor(4·(T/100)^(2/9)),
        which is the statsmodels default and appropriate for monthly data.
    min_obs : int
        Minimum observations required to run the regression (default 10).

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
          subreddit, metric, n_obs,
          beta_0, beta_1, se_beta_1, t_stat, p_value,
          ci_lower, ci_upper,
          hac_lags, r_squared,
          significant, conclusion
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
                    beta_0=np.nan, beta_1=np.nan, se_beta_1=np.nan,
                    t_stat=np.nan, p_value=np.nan,
                    ci_lower=np.nan, ci_upper=np.nan,
                    hac_lags=np.nan, r_squared=np.nan,
                    significant=np.nan, conclusion="insufficient data",
                ))
                continue

            # Integer time index: t = 0, 1, 2, …, n-1
            t = np.arange(n, dtype=float)
            X = sm.add_constant(t, prepend=True)   # columns: [const, t]
            y = series.values.astype(float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": hac_maxlags},
                )

            beta_0     = float(res.params[0])
            beta_1     = float(res.params[1])
            se_beta_1  = float(res.bse[1])
            t_stat     = float(res.tvalues[1])
            p_value    = float(res.pvalues[1])
            ci         = res.conf_int(alpha=alpha)
            ci_lower   = float(ci[0][1])
            ci_upper   = float(ci[1][1])
            r_squared  = float(res.rsquared)

            # Retrieve the actual HAC lag order used
            try:
                hac_lags = int(res.model.data.cov_kwds.get("maxlags") or
                               np.floor(4 * (n / 100) ** (2 / 9)))
            except Exception:
                hac_lags = np.nan

            significant = p_value < alpha
            if not significant:
                conclusion = "no significant trend"
            elif beta_1 > 0:
                conclusion = "upward trend"
            else:
                conclusion = "downward trend"

            records.append(dict(
                subreddit=subreddit,
                metric=metric,
                n_obs=n,
                beta_0=round(beta_0, 6),
                beta_1=round(beta_1, 6),
                se_beta_1=round(se_beta_1, 6),
                t_stat=round(t_stat, 4),
                p_value=round(p_value, 6),
                ci_lower=round(ci_lower, 6),
                ci_upper=round(ci_upper, 6),
                hac_lags=hac_lags,
                r_squared=round(r_squared, 4),
                significant=significant,
                conclusion=conclusion,
            ))

    RESULT_COLS = [
        "subreddit", "metric", "n_obs",
        "beta_0", "beta_1", "se_beta_1", "t_stat", "p_value",
        "ci_lower", "ci_upper",
        "hac_lags", "r_squared",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)
    return pd.DataFrame(records, columns=RESULT_COLS)


# ----------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------

def summarize_ols_trend(results: pd.DataFrame) -> None:
    """Print a plain-text summary of OLS trend regression results.

    Shows overall conclusion counts, per-metric breakdown, and the β₁ estimate
    (monthly trend) with its HAC standard error for every significant series.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_ols_trend().
    """
    if results.empty or "conclusion" not in results.columns:
        print("No OLS trend results to summarize — check that agg is non-empty "
              "and contains the expected subreddit and year_month columns.")
        return

    total = len(results)
    counts = results["conclusion"].value_counts()

    print("=== Baseline OLS Trend Regression Summary (HAC / Newey-West SEs) ===")
    print(f"Model: y_t = β₀ + β₁·t + ε_t   |   t = integer month index\n")
    print(f"Total (subreddit × metric) series: {total}\n")

    print("Overall conclusions:")
    for conclusion in TREND_CONCLUSIONS:
        count = counts.get(conclusion, 0)
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  {conclusion:<28} {count:>4}  ({pct:.1f}%)")

    print("\nBreakdown by metric:")
    for metric in LEXICAL_METRICS:
        sub = results[results["metric"] == metric]
        if sub.empty:
            continue
        label = METRIC_LABELS.get(metric, metric)
        sig = sub[sub["significant"] == True]
        if sig.empty:
            print(f"  {label:<18} →  no significant trends")
        else:
            parts = []
            for _, row in sig.iterrows():
                direction = "↑" if row["beta_1"] > 0 else "↓"
                parts.append(
                    f"{str(row['subreddit']).replace('subreddit-', 'r/')}: "
                    f"{direction} β₁={row['beta_1']:+.5f} (SE={row['se_beta_1']:.5f}, "
                    f"p={row['p_value']:.4f})"
                )
            print(f"  {label:<18} →  " + ";  ".join(parts))


# ----------------------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------------------

def plot_ols_trend_grid(
    agg: pd.DataFrame,
    results: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    save_path: Optional["str | Path"] = None,
) -> None:
    """Plot each metric time series with its fitted OLS trend line overlaid.

    Produces a grid of subplots — one per metric — with one line per subreddit.
    Significant trend lines are drawn solid; non-significant trends are dashed.
    The β₁ estimate and p-value are annotated in each panel.

    Parameters
    ----------
    agg : pd.DataFrame
        Subreddit-month aggregated DataFrame (input to run_ols_trend).
    results : pd.DataFrame
        Output of run_ols_trend().
    metrics : sequence of str, optional
        Metrics to plot. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings (default 'year_month').
    save_path : str or Path, optional
        If provided, saves the figure instead of displaying it.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    subreddits = sorted(agg[subreddit_col].dropna().unique())
    n_metrics  = len(metrics)
    n_cols     = 2
    n_rows     = int(np.ceil(n_metrics / n_cols))

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(subreddits)}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.8 * n_rows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for ax_idx, metric in enumerate(metrics):
        ax = axes_flat[ax_idx]
        label = METRIC_LABELS.get(metric, metric)

        for subreddit in subreddits:
            series = _extract_series(agg, subreddit, metric, subreddit_col, time_col)
            if series.empty:
                continue

            color = color_map[subreddit]
            short = str(subreddit).replace("subreddit-", "r/")

            # Plot raw monthly means
            ax.plot(
                series.index, series.values,
                color=color, alpha=0.35, linewidth=1.0,
            )

            # Retrieve regression result for this pair
            mask = (results["subreddit"] == subreddit) & (results["metric"] == metric)
            row  = results[mask]
            if row.empty or pd.isna(row["beta_1"].values[0]):
                continue

            b0   = row["beta_1"].values[0]  # slope
            b0_i = row["beta_0"].values[0]  # intercept
            pval = row["p_value"].values[0]
            sig  = bool(row["significant"].values[0])

            t  = np.arange(len(series), dtype=float)
            y_hat = b0_i + b0 * t

            linestyle = "-" if sig else "--"
            ax.plot(
                series.index, y_hat,
                color=color, linewidth=2.0, linestyle=linestyle,
                label=f"{short}  β₁={b0:+.4f}  p={pval:.3f}{'*' if sig else ''}",
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7.5, framealpha=0.85)

    # Hide any unused subplot panels
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(
        "OLS Trend Regression: y_t = β₀ + β₁·t + ε_t  (Newey-West HAC SEs)\n"
        "Solid = significant at α=0.05 · Dashed = not significant",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
