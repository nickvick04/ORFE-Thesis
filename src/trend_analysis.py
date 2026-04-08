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
    """Plot OLS trend lines for lexical diversity and sophistication metrics.

    Produces **two separate figures**, each with three subplots arranged in a
    single row:

    * **Figure 1 — Lexical Diversity**: MTLD, MATTR, Yule's K
    * **Figure 2 — Lexical Sophistication**: Zipf Score, AoA, NAWL Ratio

    Each subplot shows the monthly mean time series (faint) with the fitted
    OLS trend line overlaid. Solid lines indicate a significant trend at
    α = 0.05; dashed lines indicate non-significance. The β₁ estimate and
    p-value are shown in each legend entry.

    Parameters
    ----------
    agg : pd.DataFrame
        Subreddit-month aggregated DataFrame (input to run_ols_trend).
    results : pd.DataFrame
        Output of run_ols_trend().
    metrics : sequence of str, optional
        Metrics to plot. If provided, each metric is assigned to whichever
        group it belongs to; metrics outside both groups are silently ignored.
        Defaults to all six LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings (default 'year_month').
    save_path : str or Path, optional
        Base path for saving. If provided, two files are written by inserting
        ``_diversity`` and ``_sophistication`` before the extension, e.g.
        ``plot.png`` → ``plot_diversity.png`` and ``plot_sophistication.png``.
    """
    _DIVERSITY      = ["mtld_score", "mattr_score", "yules_k"]
    _SOPHISTICATION = ["zipf_score", "aoa_score",   "nawl_ratio"]

    if metrics is None:
        div_metrics  = _DIVERSITY
        soph_metrics = _SOPHISTICATION
    else:
        div_metrics  = [m for m in metrics if m in _DIVERSITY]
        soph_metrics = [m for m in metrics if m in _SOPHISTICATION]

    subreddits = sorted(agg[subreddit_col].dropna().unique())
    palette    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map  = {s: palette[i % len(palette)] for i, s in enumerate(subreddits)}

    # Derive save paths for each figure
    if save_path is not None:
        p         = Path(save_path)
        div_path  = p.parent / (p.stem + "_diversity"      + p.suffix)
        soph_path = p.parent / (p.stem + "_sophistication" + p.suffix)
    else:
        div_path = soph_path = None

    def _render_group(group_metrics: list, group_label: str, out_path) -> None:
        """Render one 1×3 figure for a given metric group."""
        if not group_metrics:
            return

        n_cols = len(group_metrics)
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(7.5 * n_cols, 5.5),
        )
        # Ensure axes is always iterable even for n_cols == 1
        axes_flat = np.array(axes).flatten()

        for ax_idx, metric in enumerate(group_metrics):
            ax    = axes_flat[ax_idx]
            label = METRIC_LABELS.get(metric, metric)

            for subreddit in subreddits:
                series = _extract_series(agg, subreddit, metric, subreddit_col, time_col)
                if series.empty:
                    continue

                color = color_map[subreddit]
                short = str(subreddit).replace("subreddit-", "r/")

                # Raw monthly mean (faint background)
                ax.plot(
                    series.index, series.values,
                    color=color, alpha=0.30, linewidth=1.2,
                )

                # Regression row for this (subreddit, metric)
                mask = (results["subreddit"] == subreddit) & (results["metric"] == metric)
                row  = results[mask]
                if row.empty or pd.isna(row["beta_1"].values[0]):
                    continue

                b1   = row["beta_1"].values[0]   # slope
                b0   = row["beta_0"].values[0]   # intercept
                pval = row["p_value"].values[0]
                sig  = bool(row["significant"].values[0])

                t     = np.arange(len(series), dtype=float)
                y_hat = b0 + b1 * t

                linestyle = "-" if sig else "--"
                ax.plot(
                    series.index, y_hat,
                    color=color, linewidth=2.2, linestyle=linestyle,
                    label=f"{short}  β₁={b1:+.4f}  p={pval:.3f}{'*' if sig else ''}",
                )

            ax.set_title(label, fontsize=14, fontweight="bold", pad=8)
            ax.set_xlabel("Month", fontsize=12)
            ax.tick_params(axis="x", labelsize=10, rotation=30)
            ax.tick_params(axis="y", labelsize=10)
            ax.legend(fontsize=9.5, framealpha=0.88)

        fig.suptitle(
            f"OLS Trend Regression — Lexical {group_label}\n"
            "Model: y_t = β₀ + β₁·t + ε_t  (Newey-West HAC SEs)  ·  "
            "Solid = significant at α=0.05  ·  Dashed = not significant",
            fontsize=13, fontweight="bold",
        )
        # Leave deliberate headroom between suptitle and subplot titles
        fig.tight_layout(rect=[0, 0, 1, 0.88])

        if out_path is not None:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {out_path}")
        else:
            plt.show()
        plt.close(fig)

    _render_group(div_metrics,  "Diversity",      div_path)
    _render_group(soph_metrics, "Sophistication", soph_path)


# ----------------------------------------------------------------------------------------
# Panel OLS with user, time, and subreddit fixed effects (3-way within-estimator)
# Model: y_ist = β₁·F_it + β₂·X_ist + α_i + γ_t + δ_s + ε_ist
# ----------------------------------------------------------------------------------------

# Key explanatory variable — user activity frequency F_it
PANEL_FREQ_COL = "num_utterances_by_speaker_month"

# Default controls — X_ist vector (time-varying, excluding the frequency regressor)
PANEL_CONTROLS = [
    "post_depth",
    "score",
    "num_direct_replies",
]

PANEL_CONTROL_LABELS = {
    "post_depth":         "Post Depth",
    "score":              "Score",
    "num_direct_replies": "Direct Replies",
}

# Short names used for output column suffixes
_CONTROL_SHORT = {
    "post_depth":         "post_depth",
    "score":              "score",
    "num_direct_replies": "direct_replies",
}

PANEL_CONCLUSIONS = [
    "positive effect",
    "negative effect",
    "no significant effect",
    "insufficient data",
]


def _absorb_3way_fe(
    df: pd.DataFrame,
    cols: list,
    fe1_col: str,
    fe2_col: str,
    fe3_col: str,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> pd.DataFrame:
    """Iterative demeaning (Gauss-Seidel) to absorb three sets of fixed effects.

    Alternately subtracts group means for each fixed effect in turn until
    convergence (max absolute change < tol). Typically converges in < 10 passes
    for three fixed effects.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing the FE identifier columns.
    cols : list of str
        Variable columns to demean (outcome + regressors).
    fe1_col, fe2_col, fe3_col : str
        Column names for the three fixed-effect dimensions.
    tol : float
        Convergence tolerance on max absolute change (default 1e-8).
    max_iter : int
        Maximum number of full Gauss-Seidel passes (default 50).

    Returns
    -------
    pd.DataFrame
        Residualized (demeaned) values for each column in cols.
    """
    result = df[cols].copy().astype(float)
    fe_index = {
        fe1_col: df[fe1_col].values,
        fe2_col: df[fe2_col].values,
        fe3_col: df[fe3_col].values,
    }
    for _ in range(max_iter):
        old = result.copy()
        for fe_col, fe_vals in fe_index.items():
            tmp = result.copy()
            tmp[fe_col + "_fe_key_"] = fe_vals
            group_means = tmp.groupby(fe_col + "_fe_key_")[cols].transform("mean")
            result = result - group_means
        delta = (result - old).abs().max().max()
        if delta < tol:
            break
    return result


def run_panel_ols(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    speaker_col: str = "speaker_id",
    time_col: str = "year_month",
    freq_col: str = PANEL_FREQ_COL,
    controls: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    min_speaker_obs: int = 3,
    hac_maxlags: Optional[int] = _NW_AUTO,
) -> pd.DataFrame:
    """Fit panel OLS with user, time, and subreddit fixed effects; HAC standard errors.

    Model (estimated jointly across all subreddits, separately per metric):

        y_ist = β₁·F_it + β₂·X_ist + α_i + γ_t + δ_s + ε_ist

    where
      i    = user (speaker) index
      s    = subreddit index
      t    = calendar time (year-month)
      F_it = user activity frequency — key explanatory variable
             (num_utterances_by_speaker_month, or log-transformed)
      X_ist = [post_depth, score, num_direct_replies]_ist — time-varying controls
      α_i  = user fixed effects, capturing time-invariant user-level heterogeneity
              in writing ability (absorbed via within-transformation)
      γ_t  = time fixed effects, capturing platform-wide temporal shocks and
              trends in language use (absorbed via within-transformation)
      δ_s  = subreddit fixed effects, capturing persistent differences in
              linguistic norms across communities (absorbed via within-transformation)
      ε_ist = idiosyncratic error, E[ε|F,X,α,γ,δ] = 0 assumed

    Implementation — 3-way within-estimator (iterative demeaning):
      All variables (outcome and regressors) are iteratively demeaned by user,
      calendar time period, and subreddit (Gauss-Seidel alternating projections)
      until convergence. OLS is then run on the fully residualised variables
      without an intercept. This is algebraically equivalent to including the
      full set of user, time, and subreddit dummies but avoids forming them
      explicitly, which is infeasible with millions of users.

    Standard errors are Newey-West HAC (heteroskedasticity- and autocorrelation-
    consistent). Observations are sorted by (speaker, time) before the HAC
    estimator is applied so that the lag structure aligns with the within-speaker
    time ordering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by clean_and_prepare_lexical_df(). Must contain
        speaker_col, subreddit_col, time_col (as 'YYYY-MM' strings), freq_col,
        the metric columns, and the control columns.
    metrics : sequence of str, optional
        Dependent variables to model. Defaults to LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    speaker_col : str
        Column identifying the speaker / entity (default 'speaker_id').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    freq_col : str
        Column containing user activity frequency F_it
        (default 'num_utterances_by_speaker_month'). The regressor used in
        estimation is log(1 + freq_col) to compress the right-skewed count
        distribution; β_F is therefore interpretable as the effect of a
        unit increase in log-activity on the lexical quality metric.
    controls : sequence of str, optional
        Control columns X_ist. Defaults to PANEL_CONTROLS:
        [post_depth, score, num_direct_replies]. Any control absent from df is
        silently dropped from the model.
    alpha : float
        Significance level for the key coefficient β₁ (default 0.05).
    min_speaker_obs : int
        Minimum monthly observations required per speaker (default 3). Speakers
        with fewer observations contribute no within-speaker variation and are
        dropped before estimation.
    hac_maxlags : int or None
        Newey-West lag truncation. None (default) triggers automatic selection
        via the rule of thumb floor(4·(T/100)^(2/9)).

    Returns
    -------
    pd.DataFrame
        One row per metric with columns:
          metric, n_obs, n_users, n_subreddits, n_time_periods,
          beta_F, se_F, t_stat, p_value, ci_lower, ci_upper,
          beta_{c}, se_{c}  for each active control c (short name),
          r_squared_within, hac_lags,
          significant, conclusion
    """
    if metrics is None:
        metrics = list(LEXICAL_METRICS)
    if controls is None:
        controls = list(PANEL_CONTROLS)

    for col in (subreddit_col, speaker_col, time_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")
    if freq_col not in df.columns:
        raise ValueError(
            f"Missing frequency column: '{freq_col}'. "
            "Set freq_col= to the appropriate column name."
        )

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    # Silently drop controls not present in df
    active_controls = [c for c in controls if c in df.columns]

    # Drop speakers with insufficient longitudinal observations globally
    obs_counts = df.groupby(speaker_col)[time_col].transform("count")
    panel_all = df[obs_counts >= min_speaker_obs].copy()

    records = []

    for metric in metrics:
        needed = [metric, freq_col] + active_controls
        available = [c for c in needed if c in panel_all.columns]
        id_cols   = [speaker_col, subreddit_col, time_col]

        panel = (
            panel_all[available + id_cols]
            .dropna(subset=available)
            .sort_values([speaker_col, time_col])
            .reset_index(drop=True)
        )

        n_obs        = len(panel)
        n_users      = panel[speaker_col].nunique()
        n_subreddits = panel[subreddit_col].nunique()
        n_time       = panel[time_col].nunique()

        # Build the insufficient-data sentinel record
        insufficient_record = dict(
            metric=metric,
            n_obs=n_obs, n_users=n_users,
            n_subreddits=n_subreddits, n_time_periods=n_time,
            beta_F=np.nan, se_F=np.nan,
            t_stat=np.nan, p_value=np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
            r_squared_within=np.nan, hac_lags=np.nan,
            significant=np.nan, conclusion="insufficient data",
        )
        for c in active_controls:
            short = _CONTROL_SHORT.get(c, c)
            insufficient_record[f"beta_{short}"] = np.nan
            insufficient_record[f"se_{short}"]   = np.nan

        if n_obs < 10 or n_users < 2 or n_subreddits < 2:
            records.append(insufficient_record)
            continue

        # Log-transform F_it: log(1 + count) compresses right-skewed activity
        # distribution and makes β_F interpretable as a semi-elasticity.
        # log1p is used so that zero-utterance rows (if any survive filtering)
        # are mapped to 0 rather than -inf.
        _log_freq = "_log_F_it"
        panel[_log_freq] = np.log1p(panel[freq_col].values)

        # Regressor order: [log(F_it), ctrl1, ctrl2, …]
        reg_vars = [_log_freq] + [c for c in active_controls if c in available]

        # --- 3-way within-transformation (user × time × subreddit) ---
        all_vars = [metric] + reg_vars
        dm = _absorb_3way_fe(
            panel, all_vars,
            fe1_col=speaker_col,
            fe2_col=time_col,
            fe3_col=subreddit_col,
        )

        y_dm = dm[metric].values
        X_dm = dm[reg_vars].values   # shape (n_obs, 1 + n_controls)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.OLS(y_dm, X_dm).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_maxlags, "use_correction": True},
            )

        # β_F is the first regressor (F_it — user activity frequency)
        beta_F   = float(res.params[0])
        se_F     = float(res.bse[0])
        t_stat   = float(res.tvalues[0])
        p_value  = float(res.pvalues[0])
        ci       = res.conf_int(alpha=alpha)
        ci_lower = float(ci[0][0])
        ci_upper = float(ci[1][0])

        # Retrieve actual HAC lag order used
        try:
            hac_lags = int(
                hac_maxlags
                if hac_maxlags is not None
                else np.floor(4 * (n_obs / 100) ** (2 / 9))
            )
        except Exception:
            hac_lags = np.nan

        # Within R²: share of within-entity variance explained
        ss_res    = float(np.sum(res.resid ** 2))
        ss_tot    = float(np.sum(y_dm ** 2))
        r2_within = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        significant = p_value < alpha
        if not significant:
            conclusion = "no significant effect"
        elif beta_F > 0:
            conclusion = "positive effect"
        else:
            conclusion = "negative effect"

        record = dict(
            metric=metric,
            n_obs=n_obs,
            n_users=n_users,
            n_subreddits=n_subreddits,
            n_time_periods=n_time,
            beta_F=round(beta_F, 6),
            se_F=round(se_F, 6),
            t_stat=round(t_stat, 4),
            p_value=round(p_value, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            r_squared_within=round(r2_within, 4),
            hac_lags=hac_lags,
            significant=significant,
            conclusion=conclusion,
        )

        # Control coefficients (β₂ vector)
        for c in active_controls:
            if c not in reg_vars:
                continue
            short     = _CONTROL_SHORT.get(c, c)
            param_idx = reg_vars.index(c)
            record[f"beta_{short}"] = round(float(res.params[param_idx]), 6)
            record[f"se_{short}"]   = round(float(res.bse[param_idx]),    6)

        records.append(record)

    # Build ordered column list
    control_cols = []
    for c in active_controls:
        short = _CONTROL_SHORT.get(c, c)
        control_cols += [f"beta_{short}", f"se_{short}"]

    RESULT_COLS = (
        ["metric", "n_obs", "n_users", "n_subreddits", "n_time_periods",
         "beta_F", "se_F", "t_stat", "p_value", "ci_lower", "ci_upper"]
        + control_cols
        + ["r_squared_within", "hac_lags", "significant", "conclusion"]
    )
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)
    return pd.DataFrame(records, columns=RESULT_COLS)


# ----------------------------------------------------------------------------------------
# Panel summary
# ----------------------------------------------------------------------------------------

def summarize_panel_ols(results: pd.DataFrame) -> None:
    """Print a plain-text summary of panel OLS results (3-way FE, HAC SEs).

    Shows overall conclusion counts, per-metric β_F (activity frequency effect)
    with Newey-West standard errors, and significant β₂ control coefficients.

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

    print("=== Panel OLS Summary (User FE · Time FE · Subreddit FE · HAC SEs) ===")
    print("Model: y_ist = β₁·F_it + β₂·X_ist + α_i + γ_t + δ_s + ε_ist\n")
    print(f"Total metrics estimated: {total}")

    if "n_obs" in results.columns:
        total_obs = results["n_obs"].sum()
        print(f"Total user-subreddit-month observations: {total_obs:,}")
    if "n_users" in results.columns:
        total_users = results["n_users"].max()
        print(f"Unique users (max across metrics)       : {total_users:,}")
    if "n_subreddits" in results.columns:
        total_sr = results["n_subreddits"].max()
        print(f"Subreddits                              : {total_sr}")
    if "n_time_periods" in results.columns:
        total_tp = results["n_time_periods"].max()
        print(f"Calendar time periods                   : {total_tp}")

    print()
    print("Overall conclusions:")
    for conclusion in PANEL_CONCLUSIONS:
        count = counts.get(conclusion, 0)
        pct = 100 * count / total if total > 0 else 0.0
        print(f"  {conclusion:<28} {count:>4}  ({pct:.1f}%)")

    # -----------------------------
    # β_F (activity frequency) per metric
    # -----------------------------
    print("\nβ₁ (F_it — user activity frequency effect) by metric:")
    for metric in LEXICAL_METRICS:
        row = results[results["metric"] == metric]
        if row.empty:
            continue
        row = row.iloc[0]
        label = METRIC_LABELS.get(metric, metric)

        if pd.isna(row.get("beta_F", np.nan)):
            print(f"  {label:<18} →  insufficient data")
            continue

        direction = "↑" if row["beta_F"] > 0 else "↓"
        sig_marker = "*" if row["significant"] else ""
        hac_info = (
            f", HAC lags={int(row['hac_lags'])}"
            if "hac_lags" in results.columns and not pd.isna(row["hac_lags"])
            else ""
        )
        print(
            f"  {label:<18} →  {direction} β_F={row['beta_F']:+.6f} "
            f"(SE={row['se_F']:.6f}, t={row['t_stat']:.4f}, "
            f"p={row['p_value']:.4f}{hac_info}){sig_marker}  "
            f"[{row['conclusion']}]"
        )

    # -----------------------------
    # β₂ (X_ist control) effects
    # -----------------------------
    print("\nβ₂ effects (X_ist time-varying controls):")

    # Detect control beta columns dynamically (starts with "beta_", excludes "beta_F")
    control_beta_cols = [
        c for c in results.columns
        if c.startswith("beta_") and c != "beta_F"
    ]

    if not control_beta_cols:
        print("  No control columns found in results.")
        return

    for metric in LEXICAL_METRICS:
        row = results[results["metric"] == metric]
        if row.empty:
            continue
        row = row.iloc[0]
        label = METRIC_LABELS.get(metric, metric)

        ctrl_parts = []
        for beta_col in control_beta_cols:
            val = row.get(beta_col, float("nan"))
            if val != val:   # NaN check
                continue
            short  = beta_col[len("beta_"):]
            se_col = f"se_{short}"
            se_val = row.get(se_col, float("nan"))
            direction = "↑" if val > 0 else "↓"
            ctrl_parts.append(
                f"{short}: {direction} β₂={val:+.6f} (SE={se_val:.6f})"
            )

        if ctrl_parts:
            print(f"  {label:<18} →  " + ";  ".join(ctrl_parts))
        else:
            print(f"  {label:<18} →  no control estimates available")


# ----------------------------------------------------------------------------------------
# Panel coefficient plot
# ----------------------------------------------------------------------------------------

def plot_panel_coef(
    results: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    save_path: Optional["str | Path"] = None,
) -> None:
    """Coefficient plot (dot-and-whisker) of β_F across metrics.

    Displays a single forest-plot panel with one row per metric. The point
    estimate is β_F (activity frequency effect); horizontal bars are 95 %
    confidence intervals. Filled markers indicate significance at α = 0.05;
    hollow markers indicate non-significance. A vertical reference line at
    β_F = 0 aids interpretation.

    Parameters
    ----------
    results : pd.DataFrame
        Output of run_panel_ols().
    metrics : sequence of str, optional
        Metrics to include. Defaults to all metrics present in results in the
        canonical LEXICAL_METRICS order.
    save_path : str or Path, optional
        If provided, saves the figure instead of displaying it.
    """
    if metrics is None:
        metrics = [m for m in LEXICAL_METRICS if m in results["metric"].values]

    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, max(3, 1.0 * n_metrics)),
                           constrained_layout=True)

    y_positions = np.arange(n_metrics)

    for y_pos, metric in enumerate(metrics):
        row = results[results["metric"] == metric]
        if row.empty or pd.isna(row["beta_F"].values[0]):
            # Draw a hollow grey marker to indicate missing data
            ax.plot(0, y_pos, "o", color="white",
                    markeredgecolor="#aaaaaa", markeredgewidth=1.2,
                    markersize=8, zorder=3)
            continue

        beta_F   = row["beta_F"].values[0]
        ci_lower = row["ci_lower"].values[0]
        ci_upper = row["ci_upper"].values[0]
        sig      = bool(row["significant"].values[0])
        color    = palette[y_pos % len(palette)]
        label    = METRIC_LABELS.get(metric, metric)

        # Confidence interval bar
        ax.plot([ci_lower, ci_upper], [y_pos, y_pos],
                color=color, linewidth=2.0, zorder=2)

        # Point estimate — filled if significant, hollow if not
        ax.plot(
            beta_F, y_pos, "o",
            color=color if sig else "white",
            markeredgecolor=color,
            markeredgewidth=1.8,
            markersize=10,
            zorder=3,
            label=f"{label}  β_F={beta_F:+.5f}{'*' if sig else ''}",
        )

        # Annotate p-value to the right of the CI bar
        p_val = row["p_value"].values[0]
        ax.annotate(
            f"p={p_val:.3f}{'*' if sig else ''}",
            xy=(ci_upper, y_pos),
            xytext=(4, 0), textcoords="offset points",
            fontsize=7.5, va="center",
        )

    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [METRIC_LABELS.get(m, m) for m in metrics], fontsize=10
    )
    ax.set_xlabel("β_F  (effect of user activity frequency F_it)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.invert_yaxis()   # top-to-bottom ordering matches table layout

    fig.suptitle(
        "Panel OLS — β_F Coefficient Plot\n"
        "Model: y_ist = β₁·F_it + β₂·X_ist + α_i + γ_t + δ_s  (HAC SEs)\n"
        "Filled = significant at α=0.05 · Hollow = not significant",
        fontsize=11, fontweight="bold",
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
        handles, _ = ax.get_legend_handles_labels()
        if handles:
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
