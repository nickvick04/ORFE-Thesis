"""
longitudinal_trajectory.py
==========================
Longitudinal user trajectory model for lexical quality analysis.

Implements the growth-curve mixed-effects model described in
Section X.X of the methodology:

    y_{u,s,t} = (mu_alpha + alpha_u)
              + (mu_beta  + b_u) * log(1 + C_{u,t})
              + gamma * T_{u,t}
              + delta * x_{u,s,t}
              + eta' * z_{u,s,t}
              + eps_{u,s,t}

where
    C_{u,t}  = cumulative posts by user u through month t
    T_{u,t}  = months since user u's first observed post (tenure)
    x_{u,s,t} = log_freq_month (pre-computed in ArcticShift CSV)
    z_{u,s,t} = comment-level controls (depth, edited, score, replies,
                controversiality)
    alpha_u  ~ N(0, sigma_alpha^2)   random intercept
    b_u      ~ N(0, sigma_b^2)       random slope on log-cumulative exposure
    [alpha_u, b_u]' ~ N(0, Sigma)    freely estimated covariance

Designed for the ArcticShift lexical_df_combined.csv schema, which
provides year_month, edited, and log_freq_month as pre-computed columns.

The model is fit separately for each of the six lexical quality
metrics via REML, pooled across subreddits with subreddit dummies.
Only users with >= min_months months of data are included.
p-values for mu_beta and gamma are BH-adjusted across the 6 tests.

Author: Nicholas Vickery, Princeton ORFE '26
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResultsWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirrored from regressions.py)
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

CONTROLS: list[str] = [
    "post_depth",
    "edited",
    "score",
    "num_direct_replies",
    "controversiality",
]

INVERSE_METRICS: frozenset = frozenset({"yules_k", "zipf_score"})

# Minimum months of activity required for a user to be included.
DEFAULT_MIN_MONTHS: int = 6


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: Sequence[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{context}] " if context else ""
        raise ValueError(f"{prefix}Missing required columns: {missing}")


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction across m simultaneous tests."""
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


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_longitudinal_features(
    df: pd.DataFrame,
    min_months: int = DEFAULT_MIN_MONTHS,
    speaker_col: str = "speaker_id",
    monthly_count_col: str = "num_utterances_by_speaker_month",
    log_freq_col: str = "log_freq_month",
    year_month_col: str = "year_month",
    subreddit_col: str = "subreddit",
) -> pd.DataFrame:
    """Aggregate utterance-level ArcticShift data to user-month observations
    and compute cumulative exposure (C_{u,t}) and tenure (T_{u,t}).

    The ArcticShift CSV provides year_month, edited, and log_freq_month as
    pre-computed columns, so no timestamp parsing or renaming is needed.

    Steps
    -----
    1. Convert year_month string ("YYYY-MM") to pd.Period.
    2. Aggregate to one row per (speaker, subreddit, year_month):
       mean lexical metrics + mean controls.
    3. Compute C_{u,t} = cumulative posts through each month (per
       speaker, summed across subreddits to reflect total platform use).
    4. Compute T_{u,t} = months since the user's first observation.
    5. Drop users with fewer than min_months distinct months of data.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level ArcticShift combined CSV (lexical_df_combined.csv).
    min_months : int
        Minimum number of distinct months a user must appear in to be
        included. Default 6.
    speaker_col : str
        User-identifier column.
    monthly_count_col : str
        Column giving the number of posts by that user in that month
        (num_utterances_by_speaker_month).
    log_freq_col : str
        Pre-computed log monthly frequency column (log_freq_month).
        Used directly as x_{u,s,t} in the regression.
    year_month_col : str
        Pre-computed year-month string column ("YYYY-MM").
    subreddit_col : str
        Community identifier column.

    Returns
    -------
    pd.DataFrame
        One row per (speaker_id, subreddit, year_month) with added
        columns: log_cum_exposure, tenure_months.
        log_freq_month is carried through as-is for use as x_{u,s,t}.
    """
    required = [speaker_col, monthly_count_col, log_freq_col,
                year_month_col, subreddit_col]
    _require_columns(df, required, "build_longitudinal_features")

    logger.info("Converting year_month to Period ...")
    df = df.copy()
    df[year_month_col] = pd.PeriodIndex(df[year_month_col], freq="M")

    # ------------------------------------------------------------------
    # Aggregate to (speaker, subreddit, year_month)
    # ------------------------------------------------------------------
    agg_cols = LEXICAL_METRICS + CONTROLS + [monthly_count_col, log_freq_col]
    agg_cols = [c for c in agg_cols if c in df.columns]
    keep = [speaker_col, subreddit_col, year_month_col] + agg_cols
    df = df[[c for c in keep if c in df.columns]]

    logger.info("Aggregating to user-month level ...")
    user_month = (
        df.groupby([speaker_col, subreddit_col, year_month_col], observed=True)
        .agg({col: "mean" for col in agg_cols})
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Compute cumulative exposure C_{u,t} across all subreddits for
    # each user (total platform exposure, not subreddit-specific).
    # Sum the monthly count across subreddits, then cumsum over time.
    # ------------------------------------------------------------------
    logger.info("Computing cumulative exposure C_{u,t} ...")
    total_monthly = (
        user_month.groupby([speaker_col, year_month_col], observed=True)[monthly_count_col]
        .sum()
        .reset_index()
        .rename(columns={monthly_count_col: "total_posts_month"})
        .sort_values([speaker_col, year_month_col])
    )
    total_monthly["cum_posts"] = (
        total_monthly.groupby(speaker_col, observed=True)["total_posts_month"]
        .cumsum()
    )

    user_month = user_month.merge(
        total_monthly[[speaker_col, year_month_col, "cum_posts"]],
        on=[speaker_col, year_month_col],
        how="left",
    )

    # ------------------------------------------------------------------
    # Compute tenure T_{u,t}: months since user's first observation
    # ------------------------------------------------------------------
    logger.info("Computing tenure T_{u,t} ...")
    first_month = (
        user_month.groupby(speaker_col, observed=True)[year_month_col]
        .min()
        .rename("first_month")
        .reset_index()
    )
    user_month = user_month.merge(first_month, on=speaker_col, how="left")
    user_month["tenure_months"] = (
        user_month[year_month_col] - user_month["first_month"]
    ).apply(lambda x: x.n if hasattr(x, "n") else int(x))

    # ------------------------------------------------------------------
    # Log-transform cumulative exposure
    # ------------------------------------------------------------------
    user_month["log_cum_exposure"] = np.log1p(user_month["cum_posts"])
    # log_freq_month is already pre-computed in the ArcticShift CSV;
    # rename to the generic name used throughout the regression code.
    if log_freq_col != "log_freq_month":
        user_month.rename(columns={log_freq_col: "log_freq_month"}, inplace=True)

    # ------------------------------------------------------------------
    # Filter users with fewer than min_months observations
    # ------------------------------------------------------------------
    logger.info(f"Filtering users with < {min_months} months of data ...")
    month_counts = (
        user_month.groupby(speaker_col, observed=True)[year_month_col].nunique()
    )
    eligible = month_counts[month_counts >= min_months].index
    user_month = user_month[user_month[speaker_col].isin(eligible)].copy()
    logger.info(
        f"Retained {user_month[speaker_col].nunique():,} users "
        f"({len(user_month):,} user-month observations)."
    )

    # Integer month index (months since dataset start) for reference
    min_period = user_month[year_month_col].min()
    user_month["month_idx"] = (
        user_month[year_month_col] - min_period
    ).apply(lambda x: x.n if hasattr(x, "n") else int(x))

    user_month.drop(columns=["first_month"], inplace=True)
    return user_month


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_one_metric(
    panel: pd.DataFrame,
    metric: str,
    speaker_col: str,
    subreddit_col: str,
    random_slope: bool = True,
) -> dict:
    """Fit the longitudinal ME model for a single lexical metric.

    Returns a dict of coefficient estimates, standard errors, z-scores,
    and p-values for the key parameters (mu_beta, gamma, delta) plus
    the random-effects variance components.
    """
    sub = panel[[metric, "log_cum_exposure", "tenure_months",
                 "log_freq_month", speaker_col, subreddit_col]
                + [c for c in CONTROLS if c in panel.columns]].dropna()

    if sub[speaker_col].nunique() < 10:
        logger.warning(f"[{metric}] Fewer than 10 users after dropna — skipping.")
        return {"metric": metric, "converged": False}

    # Subreddit dummies (drop one for identification)
    sub = pd.get_dummies(sub, columns=[subreddit_col], drop_first=True, dtype=float)
    subreddit_dummies = [c for c in sub.columns if c.startswith(f"{subreddit_col}_")]

    # Build fixed-effects formula
    # log_freq_month is the pre-computed x_{u,s,t} from the ArcticShift CSV
    fixed_terms = (
        ["log_cum_exposure", "tenure_months", "log_freq_month"]
        + [c for c in CONTROLS if c in sub.columns]
        + subreddit_dummies
    )
    formula = f"{metric} ~ " + " + ".join(fixed_terms)

    # Random effects: intercept + optional slope on log_cum_exposure
    if random_slope:
        re_formula = "~ log_cum_exposure"
    else:
        re_formula = None  # random intercept only

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(
                formula,
                data=sub,
                groups=sub[speaker_col],
                re_formula=re_formula,
            )
            result: MixedLMResultsWrapper = model.fit(
                reml=True,
                method="lbfgs",
                maxiter=200,
                full_output=False,
                disp=False,
            )
        converged = result.converged
    except Exception as e:
        logger.warning(f"[{metric}] Model failed: {e}")
        return {"metric": metric, "converged": False}

    # ------------------------------------------------------------------
    # Extract key coefficients
    # ------------------------------------------------------------------
    def _safe(name: str, attr: str):
        try:
            return getattr(result, attr)[name]
        except Exception:
            return np.nan

    row: dict = {
        "metric":           metric,
        "metric_label":     METRIC_LABELS.get(metric, metric),
        "converged":        converged,
        "n_users":          sub[speaker_col].nunique(),
        "n_obs":            len(sub),
        # mu_beta: avg effect of log-cumulative exposure
        "mu_beta":          _safe("log_cum_exposure", "params"),
        "mu_beta_se":       _safe("log_cum_exposure", "bse"),
        "mu_beta_z":        _safe("log_cum_exposure", "tvalues"),
        "mu_beta_p":        _safe("log_cum_exposure", "pvalues"),
        # gamma: tenure effect
        "gamma":            _safe("tenure_months", "params"),
        "gamma_se":         _safe("tenure_months", "bse"),
        "gamma_z":          _safe("tenure_months", "tvalues"),
        "gamma_p":          _safe("tenure_months", "pvalues"),
        # delta: current-month activity effect (x_{u,s,t} = log_freq_month)
        "delta":            _safe("log_freq_month", "params"),
        "delta_se":         _safe("log_freq_month", "bse"),
        "delta_z":          _safe("log_freq_month", "tvalues"),
        "delta_p":          _safe("log_freq_month", "pvalues"),
        # Variance components
        "sigma2_eps":       result.scale if hasattr(result, "scale") else np.nan,
    }

    # Random-effects variance components
    try:
        vc = result.cov_re
        row["sigma2_alpha"] = float(vc.iloc[0, 0])
        if random_slope and vc.shape[0] > 1:
            row["sigma2_b"]     = float(vc.iloc[1, 1])
            row["cov_alpha_b"]  = float(vc.iloc[0, 1])
        else:
            row["sigma2_b"]    = np.nan
            row["cov_alpha_b"] = np.nan
    except Exception:
        row["sigma2_alpha"] = row["sigma2_b"] = row["cov_alpha_b"] = np.nan

    # Flip sign interpretation for inverse metrics so that a positive
    # mu_beta always means "improving lexical quality"
    if metric in INVERSE_METRICS:
        for key in ("mu_beta", "gamma", "delta"):
            if key in row and not np.isnan(row[key]):
                row[key] = -row[key]
        row["sign_flipped"] = True
    else:
        row["sign_flipped"] = False

    return row


def run_longitudinal_trajectory(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    min_months: int = DEFAULT_MIN_MONTHS,
    random_slope: bool = True,
    speaker_col: str = "speaker_id",
    subreddit_col: str = "subreddit",
    year_month_col: str = "year_month",
    monthly_count_col: str = "num_utterances_by_speaker_month",
    log_freq_col: str = "log_freq_month",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run the longitudinal trajectory model (Eq. X.X) for each metric.

    Designed for the ArcticShift lexical_df_combined.csv schema.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level ArcticShift combined dataset (lexical_df_combined.csv).
    metrics : sequence of str, optional
        Subset of LEXICAL_METRICS to model. Defaults to all six.
    min_months : int
        Minimum months of activity required per user. Default 6.
    random_slope : bool
        If True, include a random slope on log_cum_exposure in addition
        to the random intercept. If convergence is problematic on a
        large dataset, set to False for a random-intercept-only model.
    speaker_col, subreddit_col, year_month_col : str
        Column names for user ID, community, and pre-computed year-month
        string respectively.
    monthly_count_col : str
        Column with per-user per-month post count
        (num_utterances_by_speaker_month).
    log_freq_col : str
        Pre-computed log monthly frequency (log_freq_month), used
        directly as x_{u,s,t} in the regression.
    alpha : float
        Significance threshold after BH correction.

    Returns
    -------
    pd.DataFrame
        One row per metric with coefficient estimates, standard errors,
        z-scores, raw and BH-adjusted p-values, variance components,
        and a qualitative conclusion column.
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    # ------------------------------------------------------------------
    # Step 1: Build longitudinal panel
    # ------------------------------------------------------------------
    logger.info("Building longitudinal panel ...")
    panel = build_longitudinal_features(
        df,
        min_months=min_months,
        speaker_col=speaker_col,
        monthly_count_col=monthly_count_col,
        log_freq_col=log_freq_col,
        year_month_col=year_month_col,
        subreddit_col=subreddit_col,
    )

    # ------------------------------------------------------------------
    # Step 2: Fit model for each metric
    # ------------------------------------------------------------------
    rows = []
    for metric in metrics:
        if metric not in panel.columns:
            logger.warning(f"Metric '{metric}' not in panel — skipping.")
            continue
        logger.info(f"Fitting model for {metric} ...")
        row = _fit_one_metric(
            panel, metric, speaker_col, subreddit_col, random_slope=random_slope
        )
        rows.append(row)

    results = pd.DataFrame(rows)

    if results.empty or "mu_beta_p" not in results.columns:
        logger.warning("No results produced.")
        return results

    # ------------------------------------------------------------------
    # Step 3: BH correction across the 6 mu_beta tests and 6 gamma tests
    # ------------------------------------------------------------------
    for param in ("mu_beta", "gamma"):
        p_col = f"{param}_p"
        adj_col = f"{param}_p_bh"
        valid = results[p_col].notna() & results["converged"]
        p_raw = results.loc[valid, p_col].to_numpy()
        p_adj = _bh_adjust(p_raw)
        results.loc[valid, adj_col] = p_adj
        results.loc[~valid, adj_col] = np.nan

    # ------------------------------------------------------------------
    # Step 4: Qualitative conclusion for mu_beta
    # ------------------------------------------------------------------
    def _conclude(row: pd.Series) -> str:
        if not row.get("converged", False):
            return "no convergence"
        p = row.get("mu_beta_p_bh", np.nan)
        if np.isnan(p) or p > alpha:
            return "n.s."
        return "↑ improving" if row.get("mu_beta", 0) > 0 else "↓ declining"

    results["conclusion"] = results.apply(_conclude, axis=1)

    # Reorder columns for readability
    col_order = [
        "metric", "metric_label", "converged", "n_users", "n_obs",
        "mu_beta", "mu_beta_se", "mu_beta_z", "mu_beta_p", "mu_beta_p_bh",
        "gamma",   "gamma_se",   "gamma_z",   "gamma_p",  "gamma_p_bh",
        "delta",   "delta_se",   "delta_z",   "delta_p",
        "sigma2_alpha", "sigma2_b", "cov_alpha_b", "sigma2_eps",
        "sign_flipped", "conclusion",
    ]
    col_order = [c for c in col_order if c in results.columns]
    results = results[col_order].reset_index(drop=True)

    return results
