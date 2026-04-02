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
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

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


# ----------------------------------------------------------------------------------------
# Panel OLS with speaker fixed effects (within-estimator)
# Model: y_it = β₀ + β₁·t + β₂·X_it + α_i + ε_it
# ----------------------------------------------------------------------------------------

# Default controls — mirrors the X_it vector in the model specification.
# 'num_utterances_by_speaker_month' serves as the speaker activity proxy.
PANEL_CONTROLS = [
    "post_depth",
    "score",
    "num_direct_replies",
    "num_utterances_by_speaker_month",
]

PANEL_CONTROL_LABELS = {
    "post_depth":                      "Post Depth",
    "score":                           "Score",
    "num_direct_replies":              "Direct Replies",
    "num_utterances_by_speaker_month": "Speaker Activity",
}

# Short names used for output column suffixes
_CONTROL_SHORT = {
    "post_depth":                      "post_depth",
    "score":                           "score",
    "num_direct_replies":              "direct_replies",
    "num_utterances_by_speaker_month": "speaker_activity",
}


def _prepare_panel(
    df: pd.DataFrame,
    subreddit: str,
    subreddit_col: str,
    speaker_col: str,
    time_col: str,
    min_speaker_obs: int,
) -> pd.DataFrame:
    """Filter and prepare the speaker-month panel for one subreddit.

    lexical_master.csv already contains exactly one row per speaker per month
    (the longest post), so no aggregation is required. This function:
      1. Filters to the target subreddit.
      2. Computes t = months since each speaker's first post (exact, using
         year × 12 + month arithmetic rather than timedelta approximation).
      3. Drops speakers with fewer than min_speaker_obs monthly observations.
    """
    panel = df[df[subreddit_col] == subreddit].copy()

    # Exact relative time: t = 0 at each speaker's first observed month
    dt = pd.to_datetime(panel[time_col].astype(str), format="%Y-%m")
    panel["_ym_int"] = dt.dt.year * 12 + dt.dt.month
    panel["t"] = panel.groupby(speaker_col)["_ym_int"].transform(
        lambda x: x - x.min()
    )
    panel = panel.drop(columns=["_ym_int"])

    # Keep only speakers with sufficient longitudinal observations
    obs_counts = panel.groupby(speaker_col)[time_col].transform("count")
    panel = panel[obs_counts >= min_speaker_obs].reset_index(drop=True)

    return panel


def run_panel_ols(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    speaker_col: str = "speaker_id",
    time_col: str = "year_month",
    controls: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    min_speaker_obs: int = 3,
) -> pd.DataFrame:
    """Fit a panel OLS regression with speaker fixed effects (within-estimator).

    Model (estimated separately for each subreddit × metric pair):

        y_it = β₀ + β₁·t + β₂·X_it + α_i + ε_it

    where
      i   = speaker (entity index)
      t   = months since speaker's first post in this subreddit (0, 1, 2, …)
      X_it = [post_depth, score, num_direct_replies, speaker_activity]_it
      α_i = speaker fixed effect (absorbed via within-transformation)

    Implementation — within-estimator (entity demeaning):
      All variables are demeaned by speaker (y_it − ȳ_i, t_it − t̄_i, etc.)
      before OLS is run without an intercept. This is algebraically equivalent
      to including N − 1 speaker dummies but avoids forming them explicitly,
      which would be infeasible with millions of speakers.

    Standard errors are clustered at the speaker level to account for serial
    dependence within speakers across months.

    Note on degrees of freedom: statsmodels computes residual DOF as
    N·T − K, where K = number of regressors. The correct within-estimator DOF
    is N·T − N − K (accounting for the N absorbed fixed effects). With the
    speaker counts in this dataset the difference is negligible for inference.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by clean_and_prepare_lexical_df(). lexical_master.csv
        already contains one row per speaker per month (longest post), so no
        internal aggregation is performed. Must contain speaker_col, subreddit_col,
        time_col (as 'YYYY-MM' strings), the metric columns, and the control columns.
    metrics : sequence of str, optional
        Dependent variables to model. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    speaker_col : str
        Column identifying the speaker / entity (default 'speaker_id').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    controls : sequence of str, optional
        Control columns to include as X_it. Defaults to PANEL_CONTROLS:
        [post_depth, score, num_direct_replies, num_utterances_by_speaker_month].
        Any control absent from df is silently dropped from the model.
    alpha : float
        Significance level for the trend coefficient β₁ (default 0.05).
    min_speaker_obs : int
        Minimum number of monthly observations required per speaker to be
        included in the regression (default 3). Speakers with fewer months
        contribute no within-speaker variation and are uninformative for FE
        estimation.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
          subreddit, metric, n_obs, n_speakers,
          beta_1, se_beta_1, t_stat, p_value, ci_lower, ci_upper,
          beta_{c}, se_{c}  for each active control c (short name),
          r_squared_within,
          significant, conclusion
    """
    if metrics is None:
        metrics = list(LEXICAL_METRICS)
    if controls is None:
        controls = PANEL_CONTROLS

    for col in (subreddit_col, speaker_col, time_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    # Silently drop controls not present in df
    active_controls = [c for c in controls if c in df.columns]
    reg_vars = ["t"] + active_controls   # regression variable order

    subreddits = sorted(df[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:

        # Prepare panel once per subreddit (filter, compute t, enforce min_speaker_obs)
        panel_base = _prepare_panel(
            df, subreddit, subreddit_col, speaker_col, time_col, min_speaker_obs,
        )

        for metric in metrics:
            needed = [metric] + reg_vars
            available = [c for c in needed if c in panel_base.columns]
            panel = panel_base[available + [speaker_col]].dropna(subset=available)

            n_obs      = len(panel)
            n_speakers = panel[speaker_col].nunique()

            insufficient_record = dict(
                subreddit=subreddit, metric=metric,
                n_obs=n_obs, n_speakers=n_speakers,
                beta_1=np.nan, se_beta_1=np.nan,
                t_stat=np.nan, p_value=np.nan,
                ci_lower=np.nan, ci_upper=np.nan,
                r_squared_within=np.nan,
                significant=np.nan, conclusion="insufficient data",
            )
            for c in active_controls:
                short = _CONTROL_SHORT.get(c, c)
                insufficient_record[f"beta_{short}"] = np.nan
                insufficient_record[f"se_{short}"]   = np.nan

            if n_obs < 10 or n_speakers < 2:
                records.append(insufficient_record)
                continue

            # --- Within-transformation (entity demeaning) ---
            entity_means = panel.groupby(speaker_col)[available].transform("mean")
            dm = panel[available] - entity_means   # demeaned

            y_dm = dm[metric].values
            X_dm = dm[reg_vars].values    # [t_dm, ctrl1_dm, ctrl2_dm, …]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(y_dm, X_dm).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": panel[speaker_col].values},
                )

            # β₁ is the first regressor (t_dm)
            beta_1    = float(res.params[0])
            se_beta_1 = float(res.bse[0])
            t_stat    = float(res.tvalues[0])
            p_value   = float(res.pvalues[0])
            ci        = res.conf_int(alpha=alpha)
            ci_lower  = float(ci[0][0])
            ci_upper  = float(ci[1][0])

            # Within R²: proportion of within-entity variance explained
            ss_res  = float(np.sum(res.resid ** 2))
            ss_tot  = float(np.sum(y_dm ** 2))   # total demeaned variance
            r2_within = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            significant = p_value < alpha
            if not significant:
                conclusion = "no significant trend"
            elif beta_1 > 0:
                conclusion = "upward trend"
            else:
                conclusion = "downward trend"

            record = dict(
                subreddit=subreddit,
                metric=metric,
                n_obs=n_obs,
                n_speakers=n_speakers,
                beta_1=round(beta_1, 6),
                se_beta_1=round(se_beta_1, 6),
                t_stat=round(t_stat, 4),
                p_value=round(p_value, 6),
                ci_lower=round(ci_lower, 6),
                ci_upper=round(ci_upper, 6),
                r_squared_within=round(r2_within, 4),
                significant=significant,
                conclusion=conclusion,
            )

            # Control coefficients (β₂ vector)
            for k, c in enumerate(active_controls):
                short = _CONTROL_SHORT.get(c, c)
                param_idx = k + 1   # index 0 is β₁ (t)
                record[f"beta_{short}"] = round(float(res.params[param_idx]), 6)
                record[f"se_{short}"]   = round(float(res.bse[param_idx]),   6)

            records.append(record)

    # Build ordered column list
    control_cols = []
    for c in active_controls:
        short = _CONTROL_SHORT.get(c, c)
        control_cols += [f"beta_{short}", f"se_{short}"]

    RESULT_COLS = (
        ["subreddit", "metric", "n_obs", "n_speakers",
         "beta_1", "se_beta_1", "t_stat", "p_value", "ci_lower", "ci_upper"]
        + control_cols
        + ["r_squared_within", "significant", "conclusion"]
    )
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)
    return pd.DataFrame(records, columns=RESULT_COLS)


# ----------------------------------------------------------------------------------------
# Panel summary
# ----------------------------------------------------------------------------------------

def summarize_panel_ols(results: pd.DataFrame) -> None:
    """Print a plain-text summary of panel OLS regression results.

    Shows conclusion counts, per-metric β₁ estimates with clustered SEs,
    and the control coefficient signs for each significant series.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_panel_ols().
    """
    if results.empty or "conclusion" not in results.columns:
        print("No panel OLS results to summarize — check that df is non-empty "
              "and contains the expected speaker_id, subreddit, and year_month columns.")
        return

    total  = len(results)
    counts = results["conclusion"].value_counts()

    print("=== Panel OLS Summary (Speaker FE · Clustered SEs) ===")
    print("Model: y_it = β₀ + β₁·t + β₂·X_it + α_i + ε_it\n")
    print(f"Total (subreddit × metric) series: {total}")
    if "n_obs" in results.columns and "n_speakers" in results.columns:
        total_obs = results["n_obs"].sum()
        total_spk = results.groupby("subreddit")["n_speakers"].first().sum()
        print(f"Total speaker-month observations : {total_obs:,}")
        print(f"Total unique speakers            : {total_spk:,}\n")

    print("Overall conclusions:")
    for conclusion in TREND_CONCLUSIONS:
        count = counts.get(conclusion, 0)
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  {conclusion:<28} {count:>4}  ({pct:.1f}%)")

    print("\nBreakdown by metric (β₁ = monthly trend, t in speaker-relative months):")
    for metric in LEXICAL_METRICS:
        sub = results[results["metric"] == metric]
        if sub.empty:
            continue
        label = METRIC_LABELS.get(metric, metric)
        sig   = sub[sub["significant"] == True]
        if sig.empty:
            print(f"  {label:<18} →  no significant trends")
        else:
            parts = []
            for _, row in sig.iterrows():
                direction = "↑" if row["beta_1"] > 0 else "↓"
                parts.append(
                    f"{str(row['subreddit']).replace('subreddit-', 'r/')}: "
                    f"{direction} β₁={row['beta_1']:+.6f} "
                    f"(SE={row['se_beta_1']:.6f}, p={row['p_value']:.4f})"
                )
            print(f"  {label:<18} →  " + ";  ".join(parts))


# ----------------------------------------------------------------------------------------
# Panel coefficient plot
# ----------------------------------------------------------------------------------------

def plot_panel_coef(
    results: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    save_path: Optional["str | Path"] = None,
) -> None:
    """Coefficient plot (dot-and-whisker) of β₁ across subreddits and metrics.

    Each panel shows one metric. Points are the β₁ estimates; horizontal bars
    are 95 % confidence intervals. Filled markers indicate significance at
    α = 0.05; hollow markers indicate non-significance. A vertical reference
    line at β₁ = 0 aids interpretation.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_panel_ols().
    metrics : sequence of str, optional
        Metrics to include. Defaults to all metrics present in results.
    save_path : str or Path, optional
        If provided, saves the figure instead of displaying it.
    """
    if metrics is None:
        metrics = [m for m in LEXICAL_METRICS if m in results["metric"].values]

    subreddits = sorted(results["subreddit"].dropna().unique())
    n_metrics  = len(metrics)
    n_cols     = 2
    n_rows     = int(np.ceil(n_metrics / n_cols))

    palette   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(subreddits)}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.0 * n_rows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    y_positions = np.arange(len(subreddits))

    for ax_idx, metric in enumerate(metrics):
        ax    = axes_flat[ax_idx]
        label = METRIC_LABELS.get(metric, metric)
        sub   = results[results["metric"] == metric]

        for y_pos, subreddit in enumerate(subreddits):
            row = sub[sub["subreddit"] == subreddit]
            if row.empty or pd.isna(row["beta_1"].values[0]):
                continue

            beta_1   = row["beta_1"].values[0]
            ci_lower = row["ci_lower"].values[0]
            ci_upper = row["ci_upper"].values[0]
            sig      = bool(row["significant"].values[0])
            color    = color_map[subreddit]
            short    = str(subreddit).replace("subreddit-", "r/")

            # Confidence interval bar
            ax.plot([ci_lower, ci_upper], [y_pos, y_pos],
                    color=color, linewidth=1.8, zorder=2)
            # Point estimate — filled if significant, hollow if not
            marker_kwargs = dict(
                color=color if sig else "white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                markersize=9,
                zorder=3,
            )
            ax.plot(beta_1, y_pos, "o", **marker_kwargs,
                    label=f"{short}{'*' if sig else ''}")

        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [str(s).replace("subreddit-", "r/") for s in subreddits], fontsize=9
        )
        ax.set_xlabel("β₁  (change per speaker-month)", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(
        "Panel OLS — β₁ Coefficient Plot  (Speaker FE · Clustered SEs)\n"
        "Filled = significant at α=0.05 · Hollow = not significant",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


# ----------------------------------------------------------------------------------------
# ARIMAX with automatic order selection
# Model: Δy_t = β₀ + Σβ_k X_{k,t} + Σφ_ℓ Δy_{t-ℓ} + Σθ_r ε_{t-r} + ε_t
# ----------------------------------------------------------------------------------------

def _build_arimax_series(
    df: pd.DataFrame,
    subreddit: str,
    metric: str,
    controls: Sequence[str],
    subreddit_col: str,
    time_col: str,
) -> tuple:
    """Aggregate df to subreddit-month level and return aligned (y, X) for ARIMAX.

    Returns
    -------
    y : pd.Series
        Monthly metric mean, DatetimeIndex, NaN rows dropped.
    X : pd.DataFrame or None
        Monthly control means aligned to y; None if no controls are available.
    active_controls : list[str]
        Controls that were present in df and included in X.
    """
    sub = df[df[subreddit_col] == subreddit]

    agg_cols = {c: "mean" for c in [metric] + list(controls) if c in sub.columns}
    agg = sub.groupby(time_col, sort=True).agg(agg_cols).reset_index()
    agg[time_col] = pd.to_datetime(agg[time_col].astype(str), format="%Y-%m")
    agg = agg.sort_values(time_col).reset_index(drop=True)

    active_controls = [c for c in controls if c in agg.columns]
    agg = agg.dropna(subset=[metric] + active_controls).reset_index(drop=True)

    y = agg.set_index(time_col)[metric]
    X = agg.set_index(time_col)[active_controls] if active_controls else None

    return y, X, active_controls


def _select_arimax_order(
    y: pd.Series,
    X,
    max_p: int,
    max_q: int,
    ic: str,
) -> tuple:
    """Grid-search ARIMA(p,1,q) orders and return the best result by AIC or BIC.

    All (p, q) combinations in [0, max_p] × [0, max_q] are tried. Failures
    (non-convergence, singular matrices) are silently skipped. If every
    combination fails, returns (0, 0, None).
    """
    best_ic_val = np.inf
    best_order  = (0, 0)
    best_result = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = _ARIMA(
                        y, exog=X, order=(p, 1, q), trend="c",
                    ).fit(method_kwargs={"warn_convergence": False})
                ic_val = res.aic if ic == "aic" else res.bic
                if ic_val < best_ic_val:
                    best_ic_val = ic_val
                    best_order  = (p, q)
                    best_result = res
            except Exception:
                continue

    return best_order, best_result


def run_arimax_trend(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    controls: Optional[Sequence[str]] = None,
    max_p: int = 4,
    max_q: int = 4,
    ic: str = "aic",
    alpha: float = 0.05,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Fit an ARIMAX model with automatic order selection per subreddit × metric.

    Model (estimated separately for each subreddit × metric pair):

        Δy_t = β₀ + Σ β_k X_{k,t} + Σ φ_ℓ Δy_{t-ℓ} + Σ θ_r ε_{t-r} + ε_t

    where
      Δy_t    = first difference of the subreddit-month mean metric
      β₀      = drift constant (significant β₀ > 0 → upward trend)
      X_{k,t} = subreddit-month means of external predictors (controls)
      φ_ℓ     = AR coefficients on lagged differences
      θ_r     = MA coefficients on lagged residuals

    Order selection: all (p, q) combinations in [0, max_p] × [0, max_q] are
    fitted; the combination minimising AIC (or BIC if ic='bic') is selected.
    The differencing order d is fixed at 1 throughout, consistent with the
    first-differencing used in the stationarity analysis.

    X_{k,t} are aggregated from the utterance-level df to subreddit-month means
    internally, so no pre-aggregated DataFrame is required.

    Residual adequacy is checked via a Ljung-Box test at lag min(12, T/5) on
    the model residuals. Significant residual autocorrelation suggests the
    selected (p, q) order is insufficient and max_p or max_q should be raised.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by clean_and_prepare_lexical_df(). Must contain
        subreddit_col, time_col (as 'YYYY-MM' strings), metric columns, and
        control columns.
    metrics : sequence of str, optional
        Dependent variables to model. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    controls : sequence of str, optional
        Exogenous predictor columns X_{k,t}. Defaults to PANEL_CONTROLS.
        Controls absent from df are silently dropped.
    max_p : int
        Maximum AR order to consider during grid search (default 4).
    max_q : int
        Maximum MA order to consider during grid search (default 4).
    ic : str
        Information criterion for order selection: 'aic' (default) or 'bic'.
    alpha : float
        Significance level for drift and residual tests (default 0.05).
    min_obs : int
        Minimum subreddit-month observations required to fit the model
        (default 20). Must exceed max_p + max_q + number of controls.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
          subreddit, metric, n_obs, p, q,
          beta_0, se_beta_0, pval_beta_0,
          beta_{c}, se_{c}, pval_{c}  for each active control c,
          aic, bic,
          lb_stat_resid, lb_p_resid, resid_autocorr,
          significant_drift, conclusion
    """
    if metrics is None:
        metrics = list(LEXICAL_METRICS)
    if controls is None:
        controls = PANEL_CONTROLS

    for col in (subreddit_col, time_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    active_controls = [c for c in controls if c in df.columns]

    subreddits = sorted(df[subreddit_col].dropna().unique())
    records    = []

    for subreddit in subreddits:
        for metric in metrics:

            y, X, present_controls = _build_arimax_series(
                df, subreddit, metric, active_controls, subreddit_col, time_col,
            )
            n = len(y)

            # Build the insufficient-data record template
            insuff = dict(
                subreddit=subreddit, metric=metric, n_obs=n,
                p=np.nan, q=np.nan,
                beta_0=np.nan, se_beta_0=np.nan, pval_beta_0=np.nan,
                aic=np.nan, bic=np.nan,
                lb_stat_resid=np.nan, lb_p_resid=np.nan,
                resid_autocorr=np.nan,
                significant_drift=np.nan, conclusion="insufficient data",
            )
            for c in active_controls:
                short = _CONTROL_SHORT.get(c, c)
                insuff[f"beta_{short}"] = np.nan
                insuff[f"se_{short}"]   = np.nan
                insuff[f"pval_{short}"] = np.nan

            if n < min_obs:
                records.append(insuff)
                continue

            (p, q), res = _select_arimax_order(y, X, max_p, max_q, ic)

            if res is None:
                records.append(insuff)
                continue

            # --- Drift / constant (β₀) ---
            # statsmodels names it 'const' for ARIMA with trend='c'
            const_name = next(
                (nm for nm in res.param_names if nm in ("const", "intercept")),
                None,
            )
            if const_name is not None:
                beta_0      = round(float(res.params[const_name]),  6)
                se_beta_0   = round(float(res.bse[const_name]),     6)
                pval_beta_0 = round(float(res.pvalues[const_name]), 6)
            else:
                beta_0 = se_beta_0 = pval_beta_0 = np.nan

            # --- Residual Ljung-Box check ---
            lb_lag  = max(1, min(12, n // 5))
            lb      = acorr_ljungbox(res.resid, lags=[lb_lag], return_df=True)
            lb_stat = round(float(lb["lb_stat"].iloc[0]),   4)
            lb_p    = round(float(lb["lb_pvalue"].iloc[0]), 6)
            resid_autocorr = bool(lb_p < alpha)

            # --- Drift significance and conclusion ---
            significant_drift = (
                not np.isnan(pval_beta_0) and pval_beta_0 < alpha
            )
            if not significant_drift:
                conclusion = "no significant drift"
            elif beta_0 > 0:
                conclusion = "upward drift"
            else:
                conclusion = "downward drift"

            record = dict(
                subreddit=subreddit,
                metric=metric,
                n_obs=n,
                p=int(p),
                q=int(q),
                beta_0=beta_0,
                se_beta_0=se_beta_0,
                pval_beta_0=pval_beta_0,
                aic=round(float(res.aic), 4),
                bic=round(float(res.bic), 4),
                lb_stat_resid=lb_stat,
                lb_p_resid=lb_p,
                resid_autocorr=resid_autocorr,
                significant_drift=significant_drift,
                conclusion=conclusion,
            )

            # --- Exogenous coefficients (β_k) ---
            for c in present_controls:
                short = _CONTROL_SHORT.get(c, c)
                if c in res.param_names:
                    record[f"beta_{short}"] = round(float(res.params[c]),  6)
                    record[f"se_{short}"]   = round(float(res.bse[c]),     6)
                    record[f"pval_{short}"] = round(float(res.pvalues[c]), 6)
                else:
                    record[f"beta_{short}"] = np.nan
                    record[f"se_{short}"]   = np.nan
                    record[f"pval_{short}"] = np.nan

            records.append(record)

    # Build ordered column list
    control_cols = []
    for c in active_controls:
        short = _CONTROL_SHORT.get(c, c)
        control_cols += [f"beta_{short}", f"se_{short}", f"pval_{short}"]

    RESULT_COLS = (
        ["subreddit", "metric", "n_obs", "p", "q",
         "beta_0", "se_beta_0", "pval_beta_0"]
        + control_cols
        + ["aic", "bic", "lb_stat_resid", "lb_p_resid",
           "resid_autocorr", "significant_drift", "conclusion"]
    )
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)
    return pd.DataFrame(records, columns=RESULT_COLS)


# ----------------------------------------------------------------------------------------
# ARIMAX summary
# ----------------------------------------------------------------------------------------

def summarize_arimax_trend(results: pd.DataFrame) -> None:
    """Print a plain-text summary of ARIMAX trend results.

    Reports selected orders, drift significance, residual adequacy, and
    the β_k coefficients for each active control across all subreddit × metric
    pairs.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_arimax_trend().
    """
    if results.empty or "conclusion" not in results.columns:
        print("No ARIMAX results to summarize.")
        return

    total  = len(results)
    counts = results["conclusion"].value_counts()

    print("=== ARIMAX Trend Summary  (d=1, AIC order selection) ===")
    print("Model: Δy_t = β₀ + ΣβₖX_{k,t} + Σφ_ℓ Δy_{t-ℓ} + Σθ_r ε_{t-r} + ε_t\n")
    print(f"Total (subreddit × metric) series: {total}\n")

    print("Drift conclusions:")
    for conclusion in TREND_CONCLUSIONS:
        count = counts.get(conclusion, 0)
        pct   = 100 * count / total if total > 0 else 0.0
        print(f"  {conclusion:<28} {count:>4}  ({pct:.1f}%)")

    # Residual adequacy
    if "resid_autocorr" in results.columns:
        n_ac  = results["resid_autocorr"].sum()
        pct   = 100 * n_ac / total if total > 0 else 0.0
        print(f"\nResidual autocorrelation detected: {int(n_ac)}/{total} ({pct:.1f}%)")
        if n_ac > 0:
            print("  → Consider raising max_p or max_q for flagged series.")

    # Selected orders
    if "p" in results.columns and "q" in results.columns:
        order_counts = results.groupby(["p", "q"]).size().sort_values(ascending=False)
        print("\nMost common selected orders (p, q):")
        for (p, q), cnt in order_counts.head(5).items():
            print(f"  ARIMA({int(p)},1,{int(q)})  →  {cnt} series")

    print("\nBreakdown by metric (β₀ = drift in first-differenced series):")
    for metric in LEXICAL_METRICS:
        sub = results[results["metric"] == metric]
        if sub.empty:
            continue
        label = METRIC_LABELS.get(metric, metric)
        sig   = sub[sub["significant_drift"] == True]
        if sig.empty:
            print(f"  {label:<18} →  no significant drift")
        else:
            parts = []
            for _, row in sig.iterrows():
                direction = "↑" if row["beta_0"] > 0 else "↓"
                ac_flag   = "  ⚠ resid AC" if row.get("resid_autocorr") else ""
                parts.append(
                    f"{str(row['subreddit']).replace('subreddit-', 'r/')}: "
                    f"{direction} β₀={row['beta_0']:+.6f} "
                    f"(SE={row['se_beta_0']:.6f}, p={row['pval_beta_0']:.4f})"
                    f"  ARIMA({int(row['p'])},1,{int(row['q'])}){ac_flag}"
                )
            print(f"  {label:<18} →  " + ";  ".join(parts))


# ----------------------------------------------------------------------------------------
# ARIMAX fitted-vs-actual visualization
# ----------------------------------------------------------------------------------------

def plot_arimax_fit(
    df: pd.DataFrame,
    results: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    controls: Optional[Sequence[str]] = None,
    save_path: Optional["str | Path"] = None,
) -> None:
    """Plot actual vs ARIMAX in-sample fitted values for each metric × subreddit.

    Fitted values are shown in the original (levels) scale — statsmodels
    integrates the first-differenced predictions back automatically via
    ``predict()``. One subplot per metric; one line per subreddit.

    Parameters
    ----------
    df : pd.DataFrame
        Same utterance-level DataFrame passed to run_arimax_trend().
    results : pd.DataFrame
        Output of run_arimax_trend().
    metrics : sequence of str, optional
        Metrics to plot. Defaults to all metrics present in results.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings (default 'year_month').
    controls : sequence of str, optional
        Must match the controls used in run_arimax_trend() (default PANEL_CONTROLS).
    save_path : str or Path, optional
        If provided, saves the figure instead of displaying it.
    """
    if metrics is None:
        metrics = [m for m in LEXICAL_METRICS if m in results["metric"].values]
    if controls is None:
        controls = PANEL_CONTROLS

    active_controls = [c for c in controls if c in df.columns]
    subreddits      = sorted(results["subreddit"].dropna().unique())
    n_metrics       = len(metrics)
    n_cols          = 2
    n_rows          = int(np.ceil(n_metrics / n_cols))

    palette   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(subreddits)}

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.8 * n_rows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for ax_idx, metric in enumerate(metrics):
        ax    = axes_flat[ax_idx]
        label = METRIC_LABELS.get(metric, metric)

        for subreddit in subreddits:
            row_mask = (results["subreddit"] == subreddit) & (results["metric"] == metric)
            row      = results[row_mask]
            color    = color_map[subreddit]
            short    = str(subreddit).replace("subreddit-", "r/")

            y, X, present_controls = _build_arimax_series(
                df, subreddit, metric, active_controls, subreddit_col, time_col,
            )
            if len(y) == 0:
                continue

            # Plot actual series
            ax.plot(y.index, y.values, color=color, alpha=0.35, linewidth=1.0)

            # Refit with stored (p, q) to get in-sample predictions
            if row.empty or pd.isna(row["p"].values[0]):
                continue
            p_sel = int(row["p"].values[0])
            q_sel = int(row["q"].values[0])

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = _ARIMA(y, exog=X, order=(p_sel, 1, q_sel), trend="c").fit(
                        method_kwargs={"warn_convergence": False}
                    )
                # predict() returns values in the original levels scale
                fitted = res.predict(start=0, end=len(y) - 1)
                sig    = bool(row["significant_drift"].values[0])
                pval   = row["pval_beta_0"].values[0]
                ax.plot(
                    y.index, fitted,
                    color=color, linewidth=2.0,
                    linestyle="-" if sig else "--",
                    label=(
                        f"{short}  ARIMA({p_sel},1,{q_sel})"
                        f"  p={pval:.3f}{'*' if sig else ''}"
                    ),
                )
            except Exception:
                continue

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7.5, framealpha=0.85)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(
        "ARIMAX Fitted vs Actual  (d=1 · AIC order selection)\n"
        "Solid = significant drift at α=0.05 · Dashed = not significant",
        fontsize=12, fontweight="bold", y=1.01,
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
