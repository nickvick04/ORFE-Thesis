"""
regressions.py
==============
Regression models for the ArcticShift lexical quality analysis.

Implements all six models described in Chapter 4 of the methodology:

  1. Baseline OLS Trend Regression          (§4.4.6)
  2. First-Differenced OLS Regression       (§4.4.7)
  3. Autoregressive (AR) OLS Regression     (§4.4.8)
  4. Cross-User Weighted Least Squares      (§4.5)
  5. Fixed Effects Panel Regression         (§4.6)
  6. Cross-Subreddit Mixed-Effects Model    (§4.7)

All time-series models (1–3) operate on monthly-aggregated data and use
Newey-West (HAC) standard errors to account for serial correlation.
Models 4–6 operate at the utterance or user level.

Typical notebook usage
----------------------
    import pandas as pd
    from regressions import (
        run_baseline_ols,
        run_first_diff_ols,
        run_ar_ols,
        run_cross_user_wls,
        run_fixed_effects_panel,
        run_mixed_effects,
    )

    df = pd.read_csv("lexical_df_combined.csv")

    results_ols   = run_baseline_ols(df)
    results_fd    = run_first_diff_ols(df)
    results_ar    = run_ar_ols(df)
    results_wls   = run_cross_user_wls(df)
    results_fe    = run_fixed_effects_panel(df)
    results_mixed = run_mixed_effects(df)

Author: Nicholas Vickery, Princeton ORFE '26
"""

import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEXICAL_METRICS: list[str] = [
    "mtld_score",
    "mattr_score",
    "yules_k",
    "zipf_score",
    "aoa_score",
    "nawl_ratio",
]

METRIC_LABELS: dict[str, str] = {
    "mtld_score":  "MTLD",
    "mattr_score": "MATTR",
    "yules_k":     "Yule's K",
    "zipf_score":  "Zipf Score",
    "aoa_score":   "AoA",
    "nawl_ratio":  "NAWL Ratio",
}

# Control variables used in WLS, FE, and mixed-effects models.
# Corresponds to X_ust in equations (13), (16), (17).
CONTROLS: list[str] = [
    "post_depth",
    "edited",
    "score",
    "num_direct_replies",
    "controversiality",
]

# Automatic Newey-West bandwidth: statsmodels rule-of-thumb when maxlags=None.
_NW_AUTO: None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: Sequence[str], context: str = "") -> None:
    """Raise ValueError if any columns in `cols` are absent from `df`."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{context}] " if context else ""
        raise ValueError(f"{prefix}Missing required columns: {missing}")


def _aggregate_monthly(
    df: pd.DataFrame,
    metrics: Sequence[str],
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
) -> pd.DataFrame:
    """Aggregate utterance-level data to subreddit-month means.

    NaN values in any metric column are excluded from the mean automatically.
    The result is sorted by (subreddit, year_month) to ensure the time index
    is monotone before regression.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV (output of combine_lexical_csvs.py).
    metrics : sequence of str
        Metric columns to average.
    subreddit_col, time_col : str
        Column names for community and year-month period.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, year_month) with mean metric values.
    """
    _require_columns(df, [subreddit_col, time_col] + list(metrics), "_aggregate_monthly")
    agg = (
        df.groupby([subreddit_col, time_col])[list(metrics)]
        .mean()
        .reset_index()
        .sort_values([subreddit_col, time_col])
        .reset_index(drop=True)
    )
    return agg


def _extract_series(
    agg: pd.DataFrame,
    subreddit: str,
    metric: str,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
) -> pd.Series:
    """Return a clean, time-sorted Series for one (subreddit, metric) pair.

    NaN values are dropped so that downstream regressions never see
    missing observations in the middle of a series.
    """
    sub = agg[agg[subreddit_col] == subreddit].sort_values(time_col)
    return sub[metric].dropna().reset_index(drop=True)


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values capped at 1.

    Adjusted p_(k) = min_{j ≥ k}(p_(j) · m / j) where tests are sorted
    ascending by raw p-value and m is the total number of tests.
    """
    m = len(p_values)
    if m == 0:
        return p_values
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adj = np.minimum(1.0, sorted_p * m / np.arange(1, m + 1))
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    result = np.empty(m)
    result[order] = adj
    return result


def _get_hac_lags(res, n: int) -> Union[int, float]:
    """Best-effort retrieval of the HAC lag order actually used."""
    try:
        lags = res.model.data.cov_kwds.get("maxlags")
        return int(lags) if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# 1. Baseline OLS Trend Regression  (§4.4.6)
# ---------------------------------------------------------------------------

def run_baseline_ols(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    hac_maxlags: Optional[int] = _NW_AUTO,
    min_obs: int = 10,
    apply_bh: bool = True,
) -> pd.DataFrame:
    """Baseline OLS trend regression with Newey-West HAC standard errors (§4.4.6).

    Fits the model

        y_t = β₀ + β₁·t + ε_t

    separately for each (subreddit, metric) pair, where t = 0, 1, 2, … is an
    integer month index. β₁ is the estimated average change in the metric per
    calendar month. Newey-West standard errors produce valid inference under
    serial correlation and heteroskedasticity in ε_t.

    This regression is well-specified when the series has a deterministic trend
    (trend-stationary). If ADF/KPSS results indicate a stochastic trend (unit
    root), the estimates should be treated as a baseline only, and the
    first-differencing or AR models in §4.4.7–4.4.8 are preferred.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV from combine_lexical_csvs.py.
    metrics : sequence of str, optional
        Metric columns to model. Defaults to all six LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings, e.g. '2015-03' (default 'year_month').
    alpha : float
        Significance level for β₁ (default 0.05).
    hac_maxlags : int or None
        Maximum lag for the Newey-West kernel. None triggers the statsmodels
        automatic rule-of-thumb: floor(4·(T/100)^(2/9)).
    min_obs : int
        Minimum number of time periods required to fit (default 10).
    apply_bh : bool
        If True (default), append BH-adjusted p-values across all
        (subreddit, metric) pairs to correct for multiple comparisons.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
        subreddit, metric, n_obs,
        beta_0, beta_1, se_beta_1, t_stat, p_value, [p_value_bh,]
        ci_lower, ci_upper, hac_lags, r_squared,
        significant, conclusion.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    agg = _aggregate_monthly(df, metrics, subreddit_col, time_col)
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

            t = np.arange(n, dtype=float)
            X = sm.add_constant(t, prepend=True)
            y = series.values.astype(float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": hac_maxlags},
                )

            ci       = np.array(res.conf_int(alpha=alpha))
            records.append(dict(
                subreddit=subreddit, metric=metric, n_obs=n,
                beta_0=round(float(res.params[0]), 6),
                beta_1=round(float(res.params[1]), 6),
                se_beta_1=round(float(res.bse[1]), 6),
                t_stat=round(float(res.tvalues[1]), 4),
                p_value=round(float(res.pvalues[1]), 6),
                ci_lower=round(float(ci[1, 0]), 6),
                ci_upper=round(float(ci[1, 1]), 6),
                hac_lags=_get_hac_lags(res, n),
                r_squared=round(float(res.rsquared), 4),
                significant=np.nan, conclusion="",
            ))

    RESULT_COLS = [
        "subreddit", "metric", "n_obs",
        "beta_0", "beta_1", "se_beta_1", "t_stat", "p_value",
        "ci_lower", "ci_upper", "hac_lags", "r_squared",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(records, columns=RESULT_COLS)
    _apply_significance(out, p_col="p_value", beta_col="beta_1", alpha=alpha,
                        apply_bh=apply_bh, up="upward trend",
                        down="downward trend", null="no significant trend")
    return out


# ---------------------------------------------------------------------------
# 2. First-Differenced OLS Regression  (§4.4.7)
# ---------------------------------------------------------------------------

def run_first_diff_ols(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    hac_maxlags: Optional[int] = _NW_AUTO,
    min_obs: int = 10,
    apply_bh: bool = True,
) -> pd.DataFrame:
    """First-differenced OLS regression (§4.4.7).

    Fits the drift model

        Δy_t = β₀ + ε_t

    where Δy_t = y_t − y_{t-1} is the month-over-month change in the metric.
    β₀ (the drift) captures the average directional change per month. When β₀
    shares the same sign as β₁ from the baseline levels regression, this
    corroborates the existence of a genuine trend, since both the level and the
    increments trend in the same direction.

    Newey-West HAC standard errors are applied to guard against residual serial
    correlation in the differenced series.

    Note on specification: equation (11) in the methodology writes
    Δy_t = β₀ + β₁ + ε_t, which reduces to a single drift constant. This
    function estimates that constant as β₀ and reports it as `drift`.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV from combine_lexical_csvs.py.
    metrics, subreddit_col, time_col, alpha, hac_maxlags, min_obs, apply_bh
        As in run_baseline_ols.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
        subreddit, metric, n_obs,
        drift, se_drift, t_stat, p_value, [p_value_bh,]
        ci_lower, ci_upper, hac_lags, r_squared,
        significant, conclusion.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    agg = _aggregate_monthly(df, metrics, subreddit_col, time_col)
    subreddits = sorted(agg[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:
        for metric in metrics:
            series = _extract_series(agg, subreddit, metric, subreddit_col, time_col)
            n = len(series)

            if n < min_obs + 1:          # need at least min_obs differences
                records.append(dict(
                    subreddit=subreddit, metric=metric, n_obs=max(0, n - 1),
                    drift=np.nan, se_drift=np.nan,
                    t_stat=np.nan, p_value=np.nan,
                    ci_lower=np.nan, ci_upper=np.nan,
                    hac_lags=np.nan, r_squared=np.nan,
                    significant=np.nan, conclusion="insufficient data",
                ))
                continue

            dy = series.diff().dropna().values.astype(float)
            n_diff = len(dy)
            X = sm.add_constant(np.ones(n_diff), has_constant="add")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(dy, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": hac_maxlags},
                )

            ci = np.array(res.conf_int(alpha=alpha))
            records.append(dict(
                subreddit=subreddit, metric=metric, n_obs=n_diff,
                drift=round(float(res.params[0]), 6),
                se_drift=round(float(res.bse[0]), 6),
                t_stat=round(float(res.tvalues[0]), 4),
                p_value=round(float(res.pvalues[0]), 6),
                ci_lower=round(float(ci[0, 0]), 6),
                ci_upper=round(float(ci[0, 1]), 6),
                hac_lags=_get_hac_lags(res, n_diff),
                r_squared=round(float(res.rsquared), 4),
                significant=np.nan, conclusion="",
            ))

    RESULT_COLS = [
        "subreddit", "metric", "n_obs",
        "drift", "se_drift", "t_stat", "p_value",
        "ci_lower", "ci_upper", "hac_lags", "r_squared",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(records, columns=RESULT_COLS)
    _apply_significance(out, p_col="p_value", beta_col="drift", alpha=alpha,
                        apply_bh=apply_bh, up="positive drift",
                        down="negative drift", null="no significant drift")
    return out


# ---------------------------------------------------------------------------
# 3. Autoregressive (AR) OLS Regression  (§4.4.8)
# ---------------------------------------------------------------------------

def run_ar_ols(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    hac_maxlags: Optional[int] = _NW_AUTO,
    min_obs: int = 10,
    apply_bh: bool = True,
) -> pd.DataFrame:
    """AR(1) OLS regression with deterministic time trend (§4.4.8).

    Fits the model

        y_t = β₀ + β₁·t + φ·y_{t-1} + ε_t

    where the AR term φ·y_{t-1} absorbs persistence from the previous period,
    allowing β₁ to identify a genuine time trend net of autocorrelation. A
    significant positive β₁ here — consistent with the sign from the baseline
    levels regression — is strong evidence of a real upward trend, since it
    survives controlling for last month's level.

    Newey-West HAC standard errors guard against residual autocorrelation not
    fully absorbed by the single lag.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV from combine_lexical_csvs.py.
    metrics, subreddit_col, time_col, alpha, hac_maxlags, min_obs, apply_bh
        As in run_baseline_ols.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
        subreddit, metric, n_obs,
        beta_0, beta_1, se_beta_1, phi, se_phi,
        t_stat_beta1, p_value_beta1, [p_value_bh,]
        ci_lower, ci_upper, hac_lags, r_squared,
        significant, conclusion.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    agg = _aggregate_monthly(df, metrics, subreddit_col, time_col)
    subreddits = sorted(agg[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:
        for metric in metrics:
            series = _extract_series(agg, subreddit, metric, subreddit_col, time_col)
            n = len(series)

            if n < min_obs + 1:          # need at least one lagged observation
                records.append(dict(
                    subreddit=subreddit, metric=metric, n_obs=max(0, n - 1),
                    beta_0=np.nan, beta_1=np.nan, se_beta_1=np.nan,
                    phi=np.nan, se_phi=np.nan,
                    t_stat_beta1=np.nan, p_value_beta1=np.nan,
                    ci_lower=np.nan, ci_upper=np.nan,
                    hac_lags=np.nan, r_squared=np.nan,
                    significant=np.nan, conclusion="insufficient data",
                ))
                continue

            y     = series.values.astype(float)
            y_lag = y[:-1]
            y_cur = y[1:]
            t     = np.arange(1, len(y), dtype=float)   # t = 1, 2, …, n-1

            # Design matrix: [const, t, y_{t-1}]  →  params: [β₀, β₁, φ]
            X = np.column_stack([np.ones(len(t)), t, y_lag])
            n_fit = len(y_cur)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.OLS(y_cur, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": hac_maxlags},
                )

            ci = np.array(res.conf_int(alpha=alpha))
            records.append(dict(
                subreddit=subreddit, metric=metric, n_obs=n_fit,
                beta_0=round(float(res.params[0]), 6),
                beta_1=round(float(res.params[1]), 6),
                se_beta_1=round(float(res.bse[1]), 6),
                phi=round(float(res.params[2]), 6),
                se_phi=round(float(res.bse[2]), 6),
                t_stat_beta1=round(float(res.tvalues[1]), 4),
                p_value_beta1=round(float(res.pvalues[1]), 6),
                ci_lower=round(float(ci[1, 0]), 6),
                ci_upper=round(float(ci[1, 1]), 6),
                hac_lags=_get_hac_lags(res, n_fit),
                r_squared=round(float(res.rsquared), 4),
                significant=np.nan, conclusion="",
            ))

    RESULT_COLS = [
        "subreddit", "metric", "n_obs",
        "beta_0", "beta_1", "se_beta_1", "phi", "se_phi",
        "t_stat_beta1", "p_value_beta1",
        "ci_lower", "ci_upper", "hac_lags", "r_squared",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(records, columns=RESULT_COLS)
    _apply_significance(out, p_col="p_value_beta1", beta_col="beta_1", alpha=alpha,
                        apply_bh=apply_bh, up="upward trend",
                        down="downward trend", null="no significant trend")
    return out


# ---------------------------------------------------------------------------
# 4. Cross-User Weighted Least Squares  (§4.5)
# ---------------------------------------------------------------------------

def run_cross_user_wls(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    speaker_col: str = "speaker_id",
    freq_col: str = "log_freq_month",
    post_count_col: str = "num_utterances_by_speaker",
    controls: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    min_users: int = 10,
    apply_bh: bool = True,
) -> pd.DataFrame:
    """Cross-user Weighted Least Squares regression (§4.5).

    Aggregates all utterances to user-level means, then fits

        ȳ_u = β₀ + β₁·F̄_u + β₂·X̄_u + ε_u

    where ȳ_u is the user's mean metric, F̄_u is the user's mean
    log-transformed monthly post frequency, and X̄_u is a vector of mean
    control variables. Users are weighted by their total post count n_u so
    that users with more utterances — whose means are estimated more precisely
    — exert proportionally greater influence on the estimated coefficients.

    The model is fit separately for each (subreddit, metric) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV from combine_lexical_csvs.py.
    metrics : sequence of str, optional
        Metric columns to model. Defaults to all LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    speaker_col : str
        Column identifying individual users (default 'speaker_id').
    freq_col : str
        Log-transformed monthly posting frequency F_ut (default 'log_freq_month').
    post_count_col : str
        All-time post count used as WLS weights, n_u (default 'num_utterances_by_speaker').
    controls : sequence of str, optional
        Covariates to include as user-level means in X̄_u. Defaults to CONTROLS:
        post_depth, edited, score, num_direct_replies, controversiality.
    alpha : float
        Significance level for β₁ (default 0.05).
    min_users : int
        Minimum number of users required to fit (default 10).
    apply_bh : bool
        If True, apply BH FDR correction across all (subreddit, metric) pairs.

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
        subreddit, metric, n_users,
        beta_0, beta_1, se_beta_1, t_stat, p_value, [p_value_bh,]
        ci_lower, ci_upper, r_squared,
        significant, conclusion.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS
    if controls is None:
        controls = CONTROLS

    avail_controls = [c for c in controls if c in df.columns]
    _require_columns(
        df,
        [subreddit_col, speaker_col, freq_col, post_count_col] + list(metrics),
        "run_cross_user_wls",
    )

    subreddits = sorted(df[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:
        sub_df = df[df[subreddit_col] == subreddit].copy()

        # Aggregate to user level: mean of metrics, freq, and controls per user
        agg_cols = list(metrics) + [freq_col] + avail_controls
        user_agg = (
            sub_df.groupby(speaker_col)[agg_cols]
            .mean()
            .reset_index()
        )
        # All-time post count as weights: take the max (constant within a user)
        weight_df = (
            sub_df.groupby(speaker_col)[post_count_col]
            .max()
            .reset_index()
            .rename(columns={post_count_col: "_weight"})
        )
        user_agg = user_agg.merge(weight_df, on=speaker_col, how="left")
        user_agg["_weight"] = user_agg["_weight"].fillna(1.0)

        for metric in metrics:
            cols_needed = [metric, freq_col] + avail_controls + ["_weight"]
            sub_user = user_agg[cols_needed].dropna()
            n_users = len(sub_user)

            if n_users < min_users:
                records.append(dict(
                    subreddit=subreddit, metric=metric, n_users=n_users,
                    beta_0=np.nan, beta_1=np.nan, se_beta_1=np.nan,
                    t_stat=np.nan, p_value=np.nan,
                    ci_lower=np.nan, ci_upper=np.nan,
                    r_squared=np.nan, significant=np.nan,
                    conclusion="insufficient data",
                ))
                continue

            y = sub_user[metric].values.astype(float)
            X = np.column_stack(
                [np.ones(n_users), sub_user[freq_col].values.astype(float)]
                + [sub_user[c].values.astype(float) for c in avail_controls]
            )
            w = sub_user["_weight"].values.astype(float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.WLS(y, X, weights=w).fit()

            ci = np.array(res.conf_int(alpha=alpha))
            records.append(dict(
                subreddit=subreddit, metric=metric, n_users=n_users,
                beta_0=round(float(res.params[0]), 6),
                beta_1=round(float(res.params[1]), 6),   # F̄_u coefficient
                se_beta_1=round(float(res.bse[1]), 6),
                t_stat=round(float(res.tvalues[1]), 4),
                p_value=round(float(res.pvalues[1]), 6),
                ci_lower=round(float(ci[1, 0]), 6),
                ci_upper=round(float(ci[1, 1]), 6),
                r_squared=round(float(res.rsquared), 4),
                significant=np.nan, conclusion="",
            ))

    RESULT_COLS = [
        "subreddit", "metric", "n_users",
        "beta_0", "beta_1", "se_beta_1", "t_stat", "p_value",
        "ci_lower", "ci_upper", "r_squared",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(records, columns=RESULT_COLS)
    _apply_significance(out, p_col="p_value", beta_col="beta_1", alpha=alpha,
                        apply_bh=apply_bh, up="positive effect",
                        down="negative effect", null="no significant effect")
    return out


# ---------------------------------------------------------------------------
# 5. Fixed Effects Panel Regression  (§4.6)
# ---------------------------------------------------------------------------

def run_fixed_effects_panel(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    speaker_col: str = "speaker_id",
    time_col: str = "year_month",
    freq_col: str = "log_freq_month",
    controls: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    min_obs: int = 50,
    apply_bh: bool = True,
) -> pd.DataFrame:
    """Fixed effects panel regression (§4.6).

    Fits the three-way fixed effects model

        y_ust = β₁·F_ut + β₂·X_ust + α_u + γ_t + δ_s + ε_ust

    using linearmodels PanelOLS. User fixed effects (α_u) control for
    time-invariant individual characteristics such as writing ability or
    education. Time fixed effects (γ_t) absorb platform-wide shocks common to
    all users in a given month. Subreddit fixed effects (δ_s) capture
    persistent community-level linguistic norms.

    β₁ is the key coefficient: the within-user, within-period effect of a
    one-unit increase in log monthly posting frequency on lexical quality,
    holding constant user identity, time shocks, and subreddit norms.

    The model is fit across all subreddits jointly. Robust standard errors
    are used, as HAC is not natively supported by PanelOLS.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV from combine_lexical_csvs.py.
    metrics : sequence of str, optional
        Metric columns to model. Defaults to all LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    speaker_col : str
        Panel entity column (default 'speaker_id').
    time_col : str
        Panel time column, year-month strings (default 'year_month').
    freq_col : str
        Key explanatory variable F_ut: log monthly post count (default 'log_freq_month').
    controls : sequence of str, optional
        Time-varying covariates X_ust. Defaults to CONTROLS.
    alpha : float
        Significance level for β₁ (default 0.05).
    min_obs : int
        Minimum observations required to fit (default 50).
    apply_bh : bool
        If True, apply BH FDR correction across all metrics.

    Returns
    -------
    pd.DataFrame
        One row per metric with columns:
        metric, n_obs, n_users, n_periods,
        beta_1, se_beta_1, t_stat, p_value, [p_value_bh,]
        ci_lower, ci_upper, r_squared_within,
        significant, conclusion.
    """
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:
        raise ImportError(
            "linearmodels is required for run_fixed_effects_panel. "
            "Install it with: pip install linearmodels"
        ) from exc

    if metrics is None:
        metrics = LEXICAL_METRICS
    if controls is None:
        controls = CONTROLS

    avail_controls = [c for c in controls if c in df.columns]
    _require_columns(
        df,
        [speaker_col, time_col, subreddit_col, freq_col] + list(metrics),
        "run_fixed_effects_panel",
    )

    records = []

    for metric in metrics:
        cols = [speaker_col, time_col, subreddit_col, metric, freq_col] + avail_controls
        panel_df = df[cols].dropna().copy()
        n_obs = len(panel_df)

        if n_obs < min_obs:
            records.append(dict(
                metric=metric, n_obs=n_obs,
                n_users=np.nan, n_periods=np.nan,
                beta_1=np.nan, se_beta_1=np.nan,
                t_stat=np.nan, p_value=np.nan,
                ci_lower=np.nan, ci_upper=np.nan,
                r_squared_within=np.nan,
                significant=np.nan, conclusion="insufficient data",
            ))
            continue

        # Subreddit dummies (δ_s): included as exogenous regressors so the
        # within-estimator can identify cross-community differences on top of
        # user FE. Drop first to avoid the dummy trap.
        sub_dummies = pd.get_dummies(
            panel_df[subreddit_col], prefix="sub", drop_first=True
        ).astype(float)
        sub_dummy_cols = sub_dummies.columns.tolist()
        panel_df = pd.concat(
            [panel_df.reset_index(drop=True), sub_dummies.reset_index(drop=True)],
            axis=1,
        )

        # PanelOLS requires a MultiIndex of (entity, time) where the time
        # dimension is numeric or date-like. Convert year_month strings
        # (e.g. '2015-01') to the first day of each month as a datetime.
        panel_df[time_col] = pd.to_datetime(panel_df[time_col] + "-01")
        panel_df = panel_df.set_index([speaker_col, time_col])

        exog_cols = [freq_col] + avail_controls + sub_dummy_cols
        exog = sm.add_constant(panel_df[exog_cols], has_constant="add")
        endog = panel_df[metric]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = PanelOLS(
                endog, exog,
                entity_effects=True,
                time_effects=True,
                drop_absorbed=True,
            )
            res = mod.fit(cov_type="robust")

        if freq_col not in res.params.index:
            records.append(dict(
                metric=metric, n_obs=n_obs,
                n_users=np.nan, n_periods=np.nan,
                beta_1=np.nan, se_beta_1=np.nan,
                t_stat=np.nan, p_value=np.nan,
                ci_lower=np.nan, ci_upper=np.nan,
                r_squared_within=np.nan,
                significant=np.nan, conclusion="dropped (collinear)",
            ))
            continue

        ci = res.conf_int(level=1 - alpha)
        r_sq_w = float(res.rsquared_within) if hasattr(res, "rsquared_within") else np.nan
        n_users   = int(res.entity_info.total)   if hasattr(res, "entity_info") else np.nan
        n_periods = int(res.time_info.total)      if hasattr(res, "time_info")   else np.nan

        records.append(dict(
            metric=metric, n_obs=n_obs,
            n_users=n_users, n_periods=n_periods,
            beta_1=round(float(res.params[freq_col]), 6),
            se_beta_1=round(float(res.std_errors[freq_col]), 6),
            t_stat=round(float(res.tstats[freq_col]), 4),
            p_value=round(float(res.pvalues[freq_col]), 6),
            ci_lower=round(float(ci.loc[freq_col, "lower"]), 6),
            ci_upper=round(float(ci.loc[freq_col, "upper"]), 6),
            r_squared_within=round(r_sq_w, 4) if not np.isnan(r_sq_w) else np.nan,
            significant=np.nan, conclusion="",
        ))

    RESULT_COLS = [
        "metric", "n_obs", "n_users", "n_periods",
        "beta_1", "se_beta_1", "t_stat", "p_value",
        "ci_lower", "ci_upper", "r_squared_within",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(records, columns=RESULT_COLS)
    _apply_significance(out, p_col="p_value", beta_col="beta_1", alpha=alpha,
                        apply_bh=apply_bh,
                        up="positive activity effect",
                        down="negative activity effect",
                        null="no significant effect")
    return out


# ---------------------------------------------------------------------------
# 6. Cross-Subreddit Mixed-Effects Model  (§4.7)
# ---------------------------------------------------------------------------

def run_mixed_effects(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    speaker_col: str = "speaker_id",
    time_col: str = "year_month",
    freq_col: str = "log_freq_month",
    controls: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    min_obs: int = 50,
    apply_bh: bool = True,
) -> pd.DataFrame:
    """Cross-subreddit mixed-effects comparison model (§6.3).

    Fits the model

        y_ust = Σ_s θ_s · 1[subreddit = s] + β_1·F_ut + β_2·X_ust + a_u + γ_t + ε_ust

    where θ_s are subreddit fixed effects capturing the conditional mean
    difference in lexical quality for subreddit s relative to a reference
    community (the first subreddit alphabetically), β_1 is the activity effect
    coefficient on log monthly post frequency F_ut, β_2·X_ust are the
    time-varying control variables, γ_t are time fixed effects (month dummies)
    absorbing platform-wide shocks common to all users in a given month, and
    a_u ~ N(0, σ²_u) is a user-level random intercept estimated by REML.

    Unlike the fixed effects panel model (§6.2) where user FE absorbs all
    between-user and between-subreddit variation, the random intercept here
    allows for cross-subreddit comparisons while still controlling for
    persistent user heterogeneity.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level combined CSV from combine_lexical_csvs.py.
    metrics : sequence of str, optional
        Metric columns to model. Defaults to all LEXICAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    speaker_col : str
        Grouping column for user random intercepts (default 'speaker_id').
    time_col : str
        Column for time period used to build γ_t month dummies (default 'year_month').
    freq_col : str
        Key explanatory variable F_ut: log monthly post count (default 'log_freq_month').
    controls : sequence of str, optional
        Fixed-effect covariates β_2 · X_ust. Defaults to CONTROLS.
    alpha : float
        Significance level for the θ_s coefficients (default 0.05).
    min_obs : int
        Minimum observations required to fit (default 50).
    apply_bh : bool
        If True, apply BH FDR correction across all (metric, subreddit) pairs.

    Returns
    -------
    pd.DataFrame
        One row per (metric, subreddit) — one per non-reference subreddit —
        with columns:
        metric, subreddit, reference_subreddit, n_obs, n_users,
        beta_1, se_beta_1,
        delta, se_delta, z_stat, p_value, [p_value_bh,]
        ci_lower, ci_upper, log_likelihood,
        significant, conclusion.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS
    if controls is None:
        controls = CONTROLS

    avail_controls = [c for c in controls if c in df.columns]
    _require_columns(
        df,
        [speaker_col, subreddit_col, time_col, freq_col] + list(metrics),
        "run_mixed_effects",
    )

    subreddits_all = sorted(df[subreddit_col].dropna().unique())
    reference_sub  = subreddits_all[0]

    records = []

    for metric in metrics:
        cols = [speaker_col, subreddit_col, time_col, freq_col, metric] + avail_controls
        mdf = df[cols].dropna().copy().reset_index(drop=True)
        n_obs   = len(mdf)
        n_users = mdf[speaker_col].nunique()

        if n_obs < min_obs:
            for sub in subreddits_all[1:]:
                records.append(_missing_mixed_row(
                    metric, sub, reference_sub, n_obs, n_users, "insufficient data"
                ))
            continue

        # Subreddit dummies (θ_s): drop the reference category so the intercept
        # represents the reference subreddit's mean.
        sub_dummies = pd.get_dummies(
            mdf[subreddit_col], prefix="sub", drop_first=False
        ).astype(float)
        ref_col = f"sub_{reference_sub}"
        if ref_col in sub_dummies.columns:
            sub_dummies = sub_dummies.drop(columns=[ref_col])
        sub_dummy_cols = sub_dummies.columns.tolist()

        # Time dummies (γ_t): drop one period to avoid the dummy trap.
        time_dummies = pd.get_dummies(
            mdf[time_col], prefix="time", drop_first=True
        ).astype(float)
        time_dummy_cols = time_dummies.columns.tolist()

        mdf = pd.concat([mdf, sub_dummies, time_dummies], axis=1)

        # exog order: subreddit dummies, freq (β_1·F_ut), controls (β_2·X_ust),
        # time dummies (γ_t)
        exog_cols = sub_dummy_cols + [freq_col] + avail_controls + time_dummy_cols
        exog = sm.add_constant(mdf[exog_cols], has_constant="add")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = MixedLM(
                endog=mdf[metric].values.astype(float),
                exog=exog,
                groups=mdf[speaker_col].values,
            )
            try:
                res = mod.fit(reml=True, method="lbfgs")
            except Exception:
                try:
                    res = mod.fit(reml=False, method="nm")
                except Exception:
                    for sub in subreddits_all[1:]:
                        records.append(_missing_mixed_row(
                            metric, sub, reference_sub, n_obs, n_users,
                            "convergence failure"
                        ))
                    continue

        log_lik = round(float(res.llf), 4) if hasattr(res, "llf") else np.nan
        ci = res.conf_int(alpha=alpha)

        # β_1 coefficient on F_ut (same for every subreddit row of this metric)
        if freq_col in res.params.index:
            beta_1    = round(float(res.params[freq_col]), 6)
            se_beta_1 = round(float(res.bse[freq_col]), 6)
        else:
            beta_1 = se_beta_1 = np.nan

        for sub in subreddits_all:
            if sub == reference_sub:
                continue
            param_name = f"sub_{sub}"
            if param_name not in res.params.index:
                records.append(_missing_mixed_row(
                    metric, sub, reference_sub, n_obs, n_users, "not estimated"
                ))
                continue

            records.append(dict(
                metric=metric, subreddit=sub,
                reference_subreddit=reference_sub,
                n_obs=n_obs, n_users=n_users,
                beta_1=beta_1, se_beta_1=se_beta_1,
                delta=round(float(res.params[param_name]), 6),
                se_delta=round(float(res.bse[param_name]), 6),
                z_stat=round(float(res.tvalues[param_name]), 4),
                p_value=round(float(res.pvalues[param_name]), 6),
                ci_lower=round(float(ci.loc[param_name, 0]), 6),
                ci_upper=round(float(ci.loc[param_name, 1]), 6),
                log_likelihood=log_lik,
                significant=np.nan, conclusion="",
            ))

    RESULT_COLS = [
        "metric", "subreddit", "reference_subreddit", "n_obs", "n_users",
        "beta_1", "se_beta_1",
        "delta", "se_delta", "z_stat", "p_value",
        "ci_lower", "ci_upper", "log_likelihood",
        "significant", "conclusion",
    ]
    if not records:
        return pd.DataFrame(columns=RESULT_COLS)

    out = pd.DataFrame(records, columns=RESULT_COLS)
    out["significant"] = out["significant"].astype(object)
    valid = out["p_value"].notna()

    if apply_bh and valid.any():
        adj = _bh_adjust(out.loc[valid, "p_value"].values)
        out.loc[valid, "p_value_bh"] = np.round(adj, 6)
        out.loc[valid, "significant"] = out.loc[valid, "p_value_bh"] < alpha
    else:
        out.loc[valid, "significant"] = out.loc[valid, "p_value"] < alpha

    def _conclude_mixed(row):
        if pd.isna(row["significant"]):
            return row["conclusion"] if row["conclusion"] else "insufficient data"
        if not row["significant"]:
            return "no significant difference"
        direction = "higher" if row["delta"] > 0 else "lower"
        return f"{direction} quality than {row['reference_subreddit']}"

    out["conclusion"] = out.apply(_conclude_mixed, axis=1)
    return out


# ---------------------------------------------------------------------------
# Private post-processing helpers
# ---------------------------------------------------------------------------

def _apply_significance(
    out: pd.DataFrame,
    p_col: str,
    beta_col: str,
    alpha: float,
    apply_bh: bool,
    up: str,
    down: str,
    null: str,
) -> None:
    """Attach `significant` and `conclusion` columns to a results DataFrame in-place.

    Optionally appends a `p_value_bh` column with BH-adjusted p-values.
    """
    valid = out[p_col].notna()

    # Cast to object so pandas 2.x allows mixed NaN/bool assignment
    out["significant"] = out["significant"].astype(object)

    if apply_bh and valid.any():
        adj = _bh_adjust(out.loc[valid, p_col].values)
        out.loc[valid, "p_value_bh"] = np.round(adj, 6)
        out.loc[valid, "significant"] = out.loc[valid, "p_value_bh"] < alpha
    else:
        out.loc[valid, "significant"] = out.loc[valid, p_col] < alpha

    def _conclude(row):
        if pd.isna(row["significant"]):
            return row["conclusion"] if (row["conclusion"] and row["conclusion"] != "") else "insufficient data"
        if not row["significant"]:
            return null
        return up if row[beta_col] > 0 else down

    out["conclusion"] = out.apply(_conclude, axis=1)


def _missing_mixed_row(
    metric: str,
    subreddit: str,
    reference_subreddit: str,
    n_obs: int,
    n_users: int,
    conclusion: str,
) -> dict:
    """Return a placeholder row dict for run_mixed_effects when fitting fails."""
    return dict(
        metric=metric, subreddit=subreddit,
        reference_subreddit=reference_subreddit,
        n_obs=n_obs, n_users=n_users,
        beta_1=np.nan, se_beta_1=np.nan,
        delta=np.nan, se_delta=np.nan,
        z_stat=np.nan, p_value=np.nan,
        ci_lower=np.nan, ci_upper=np.nan,
        log_likelihood=np.nan,
        significant=np.nan, conclusion=conclusion,
    )
