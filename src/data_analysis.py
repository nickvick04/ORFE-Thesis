# ----------------------------------------------------------------------------------------
# This code is designed to analyze post-cluster computation CSV files
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, List, Optional, Sequence
import warnings

import pymannkendall as mk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, acf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LEXICAL_METRICS = [
    "mtld_score",
    "mattr_score",
    "yules_k",
    "zipf_score",
    "aoa_score",
    "nawl_ratio",
]


def combine_pipeline_csvs_to_lexical_master(
    convokit_root: str | Path,
    output_filename: str = "lexical_master.csv",
    variations_only: bool = True,
    skip_read_errors: bool = False,
) -> Path:
    """Combine pipeline-produced CSV files under Convokit variations into one master CSV.

    Expected structure:
    Convokit/
      Age-Variation/
        subreddit_a.csv
        subreddit_b.csv
      Topic-Variation/
        ...
    """
    root = Path(convokit_root).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        alias_hint = ""
        if root.exists() and root.is_file():
            alias_hint = " (path is a file; if this is a macOS Alias, use the real folder path)"
        raise FileNotFoundError(f"Convokit root not found: {root}{alias_hint}")

    def _is_pipeline_csv(path: Path) -> bool:
        name = path.name
        return (
            name.endswith("_df.csv")
            or "_df_shard-" in name
            or name.endswith("_lexical_df.csv")
        )

    def _subreddit_from_filename(path: Path) -> str:
        name = path.name
        suffixes = ("_lexical_df.csv", "_df.csv")
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        shard_marker = "_df_shard-"
        if shard_marker in name and name.endswith(".csv"):
            return name.split(shard_marker, 1)[0]
        return path.stem

    variation_dirs = [p for p in root.iterdir() if p.is_dir()]
    if variations_only:
        variation_dirs = [p for p in variation_dirs if "variation" in p.name.lower()]

    csv_files: List[Path] = []
    for variation_dir in sorted(variation_dirs):
        variation_csvs = [
            p for p in sorted(variation_dir.glob("*.csv"))
            if p.name != output_filename and _is_pipeline_csv(p)
        ]
        csv_files.extend(variation_csvs)

    if not csv_files:
        raise FileNotFoundError(
            f"No pipeline CSV files found under selected folders in: {root}"
        )

    EXPECTED_COLS = {
        "timestamp", "utterance_id", "speaker_id", "raw_text",
        "num_utterances_by_speaker", "num_utterances_by_speaker_month",
        "post_depth", "score", "num_direct_replies",
    }

    dfs = []
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Reading {csv_path}")
        try:
            df = pd.read_csv(csv_path)

            # Guard: detect headerless CSVs (first data row used as column names).
            # Expected columns are all strings; if any core column is missing and
            # one of the column names looks numeric, the file is likely headerless.
            missing_core = EXPECTED_COLS - set(df.columns)
            numeric_col_names = [c for c in df.columns if str(c).replace(".", "", 1).lstrip("-").isdigit()]
            if missing_core and numeric_col_names:
                raise RuntimeError(
                    f"CSV appears to have been written without a header row "
                    f"(column names look like data values: {numeric_col_names[:5]}).\n"
                    f"Missing expected columns: {sorted(missing_core)}.\n"
                    f"Re-run the pipeline for this corpus to regenerate: {csv_path}"
                )

            df["source_variation"] = csv_path.parent.name
            df["subreddit"] = _subreddit_from_filename(csv_path)
            dfs.append(df)
        except (TimeoutError, OSError, pd.errors.ParserError) as exc:
            if skip_read_errors:
                print(f"Skipping unreadable file: {csv_path}\nReason: {exc}")
                continue
            raise RuntimeError(
                f"Failed to read CSV: {csv_path}\n"
                "If this is in a cloud-synced folder, ensure the file is downloaded locally "
                "or rerun with skip_read_errors=True."
            ) from exc

    if not dfs:
        raise RuntimeError("No readable CSV files were loaded.")

    master_df = pd.concat(dfs, ignore_index=True)
    output_path = root / output_filename
    master_df.to_csv(output_path, index=False)

    print(f"Combined {len(csv_files)} CSV files into:\n{output_path}")
    print(f"Total rows: {len(master_df)}")
    return output_path


def clean_and_prepare_lexical_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the master lexical DataFrame for analysis.

    Steps performed, in order:

    1. Normalize source_variation and subreddit columns:
       - source_variation: keep text before the first hyphen (lowercased)
       - subreddit: keep text after the first hyphen

    2. Convert the timestamp column from Unix seconds to a tz-naive UTC
       datetime, then derive two time columns:
       - date          : full datetime (NaT where conversion fails)
       - months_elapsed: integer months since the earliest valid observation,
                         used as the sole time regressor in OLS / stationarity
                         analysis

    3. Convert all six lexical metric columns to numeric, coercing any
       unparseable strings to NaN.

    4. Drop rows where fewer than two of the six lexical metrics are non-null,
       as these rows provide insufficient signal for any metric-level analysis.
    """
    required_cols = {"source_variation", "subreddit", "timestamp"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()

    # --- 1. Normalize variation / subreddit labels ---
    cleaned["source_variation"] = (
        cleaned["source_variation"]
        .astype(str)
        .str.strip()
        .str.split("-", n=1)
        .str[0]
        .str.lower()
    )
    cleaned["subreddit"] = (
        cleaned["subreddit"]
        .astype(str)
        .str.strip()
        .str.split("-", n=1)
        .str[-1]
    )

    # --- 2. Timestamp conversion ---
    unix_numeric = pd.to_numeric(cleaned["timestamp"], errors="coerce")
    cleaned["date"] = pd.to_datetime(unix_numeric, unit="s", errors="coerce")

    earliest = cleaned["date"].min()
    cleaned["months_elapsed"] = (
        (cleaned["date"].dt.year - earliest.year) * 12
        + (cleaned["date"].dt.month - earliest.month)
    ).where(cleaned["date"].notna())

    # --- 3. Metric columns to numeric ---
    metric_cols = [m for m in LEXICAL_METRICS if m in cleaned.columns]
    cleaned[metric_cols] = cleaned[metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # --- 4. Drop rows with fewer than 2 valid metrics ---
    valid_metric_count = cleaned[metric_cols].notna().sum(axis=1)
    cleaned = cleaned[valid_metric_count >= 2]

    return cleaned


def run_log_linear_regression(
    df: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run sklearn linear regression with notebook-style log transforms.

    Behavior mirrors the notebook:
    - log1p transform for numeric predictors in:
      {"num_utterances_by_speaker", "score", "num_direct_replies"}
    - if "timestamp" is in predictors, convert it to numeric "year"
    - one-hot encode categorical predictors
    - return model, metrics, and coefficient table
    """
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    missing_predictors = [p for p in predictors if p not in df.columns]
    if missing_predictors:
        raise ValueError(f"Missing predictor columns: {missing_predictors}")

    model_df = df[[target, *predictors]].copy()

    work_predictors = list(predictors)
    if "timestamp" in work_predictors:
        model_df["timestamp"] = pd.to_datetime(model_df["timestamp"], errors="coerce")
        model_df["year"] = model_df["timestamp"].dt.year
        work_predictors = [p for p in work_predictors if p != "timestamp"] + ["year"]

    categorical_predictors = [
        p
        for p in work_predictors
        if (
            pd.api.types.is_object_dtype(model_df[p])
            or pd.api.types.is_categorical_dtype(model_df[p])
            or pd.api.types.is_string_dtype(model_df[p])
            or pd.api.types.is_bool_dtype(model_df[p])
        )
    ]
    numeric_predictors = [p for p in work_predictors if p not in categorical_predictors]

    for col in numeric_predictors:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    log_transform_cols = [
        c
        for c in ("num_utterances_by_speaker", "score", "num_direct_replies")
        if c in numeric_predictors
    ]

    transformed_numeric_predictors: list[str] = []
    for col in numeric_predictors:
        if col in log_transform_cols:
            log_col = f"log_{col}"
            model_df[log_col] = np.log1p(model_df[col].clip(lower=0))
            transformed_numeric_predictors.append(log_col)
        else:
            transformed_numeric_predictors.append(col)

    feature_cols = transformed_numeric_predictors + categorical_predictors
    X = model_df[feature_cols]
    y = model_df[target]

    y_mask = y.notna()
    X = X.loc[y_mask]
    y = y.loc[y_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), transformed_numeric_predictors),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_predictors,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["regressor"].coef_
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df = coef_df.reindex(coef_df.coefficient.abs().sort_values(ascending=False).index)

    return {
        "model": model,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "coef_df": coef_df,
        "feature_columns": feature_cols,
    }


# ----------------------------------------------------------------------------------------
# Mann-Kendall Trend Test
# ----------------------------------------------------------------------------------------

def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values capped at 1.

    Adjusted p_(k) = min_{j >= k} ( p_(j) * m / j ) where tests are sorted
    by ascending raw p-value and m is the total number of tests.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adj = np.minimum(1.0, sorted_p * m / np.arange(1, m + 1))
    # Enforce monotonicity right-to-left so that a larger p cannot have a
    # smaller adjusted value than a p ranked above it.
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    result = np.empty(m)
    result[order] = adj
    return result


def run_mann_kendall_tests(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    time_col: str = "timestamp",
    freq: str = "Y",
    alpha: float = 0.05,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Run Mann-Kendall trend tests on lexical metrics aggregated over time.

    For each metric (and optionally each level of group_col), the time series
    is resampled to `freq` frequency, period means are computed, and the
    Mann-Kendall test is applied to the resulting sequence.  Benjamini-Hochberg
    FDR correction is applied jointly across all (metric x group) tests.

    Parameters
    ----------
    df : pd.DataFrame
        Master lexical DataFrame with a timestamp column and metric columns.
    metrics : list of str, optional
        Metric columns to test.  Defaults to LEXICAL_METRICS.
    time_col : str
        Name of the datetime column (default 'timestamp').
    freq : str
        Pandas resampling frequency string.  'Y' is recommended for a
        2007-2018 dataset (12 annual data points); 'M' gives monthly
        resolution (~132 points) at the cost of more noise.
    alpha : float
        FDR significance threshold after BH correction (default 0.05).
    group_col : str or None
        If provided, run tests separately for each value of this column
        (e.g., 'subreddit' or 'source_variation').  When None, a single
        test is run over the full dataset (group label = 'all').

    Returns
    -------
    pd.DataFrame
        One row per (metric, group) with columns:
          metric       – name of the lexical metric
          group        – group label ('all' when group_col is None)
          n_periods    – number of resampled time periods with valid data
          tau          – Kendall tau (rank correlation with time index)
          sens_slope   – Theil-Sen median slope (metric units per period)
          p_value      – raw two-sided Mann-Kendall p-value
          p_adjusted   – BH-corrected p-value
          significant  – True if p_adjusted <= alpha
          trend        – 'increasing', 'decreasing', or 'no trend'
    """
    if metrics is None:
        metrics = LEXICAL_METRICS

    missing_metrics = [m for m in metrics if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: '{time_col}'")

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).set_index(time_col).sort_index()

    # Build list of (label, sub-dataframe) pairs to iterate over.
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"Missing group column: '{group_col}'")
        groups: List[tuple] = [
            (str(name), grp) for name, grp in work.groupby(group_col)
        ]
    else:
        groups = [("all", work)]

    records = []
    for group_label, grp_df in groups:
        resampled = grp_df[list(metrics)].resample(freq).mean().dropna(how="all")

        for metric in metrics:
            series = resampled[metric].dropna().values
            n = len(series)

            if n < 4:
                # Mann-Kendall is unreliable with fewer than 4 observations.
                records.append(
                    dict(
                        metric=metric,
                        group=group_label,
                        n_periods=n,
                        tau=np.nan,
                        sens_slope=np.nan,
                        p_value=np.nan,
                        p_adjusted=np.nan,
                        significant=False,
                        trend="insufficient data",
                    )
                )
                continue

            result = mk.original_test(series)

            records.append(
                dict(
                    metric=metric,
                    group=group_label,
                    n_periods=n,
                    tau=round(result.Tau, 4),
                    sens_slope=round(result.slope, 6),
                    p_value=result.p,
                    p_adjusted=np.nan,
                    significant=False,
                    trend="",
                )
            )

    result_df = pd.DataFrame(records)

    # Apply BH correction across all tests with a valid p-value.
    valid = result_df["p_value"].notna()
    if valid.any():
        raw_p = result_df.loc[valid, "p_value"].to_numpy(dtype=float)
        adj_p = _bh_adjust(raw_p)
        result_df.loc[valid, "p_adjusted"] = np.round(adj_p, 6)
        result_df.loc[valid, "significant"] = adj_p <= alpha

    # Assign human-readable trend labels.
    def _label(row: pd.Series) -> str:
        if pd.isna(row["tau"]):
            return "insufficient data"
        if not row["significant"]:
            return "no trend"
        return "increasing" if row["tau"] > 0 else "decreasing"

    result_df["trend"] = result_df.apply(_label, axis=1)
    result_df["p_value"] = result_df["p_value"].round(6)

    return result_df


# ----------------------------------------------------------------------------------------
# Temporal Stationarity Tests (ADF / KPSS)
# ----------------------------------------------------------------------------------------

# Corpus-level metrics produced by lexical_temporal.csv.
TEMPORAL_METRICS = [
    "mattr_score",
    "mtld_score",
    "yules_k",
    "zipf_score",
    "aoa_score",
    "nawl_ratio",
]

# Human-readable labels reused across stationarity and ACF outputs.
TEMPORAL_METRIC_LABELS = {
    "mattr_score": "MATTR",
    "mtld_score":  "MTLD",
    "yules_k":     "Yule's K",
    "zipf_score":  "Zipf Score",
    "aoa_score":   "Age of Acquisition",
    "nawl_ratio":  "NAWL Ratio",
}


def _load_temporal_df(source: "str | Path | pd.DataFrame") -> pd.DataFrame:
    """Return a DataFrame from a file path or pass through an existing DataFrame."""
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    return source.copy()


def _extract_series(
    df: pd.DataFrame,
    subreddit: str,
    metric: str,
    subreddit_col: str,
    time_col: str,
) -> pd.Series:
    """Extract a sorted, NaN-free time series for one (subreddit, metric) pair.

    year_month strings (e.g. '2015-03') are parsed into a DatetimeIndex so
    the series is properly ordered even if the CSV is not sorted.
    """
    sub = df[df[subreddit_col] == subreddit].copy()
    sub[time_col] = pd.to_datetime(
        sub[time_col].astype(str), format="%Y-%m", errors="coerce"
    )
    sub = sub.dropna(subset=[time_col, metric]).sort_values(time_col)
    return sub.set_index(time_col)[metric].dropna()


def run_stationarity_tests(
    source: "str | Path | pd.DataFrame",
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    alpha: float = 0.05,
    adf_maxlag: int = 4,
) -> pd.DataFrame:
    """Run ADF and KPSS unit-root tests on each (subreddit, metric) time series
    in lexical_temporal.csv.

    ADF null hypothesis: a unit root is present (non-stationary).
    A small p-value rejects the null — evidence of stationarity.

    KPSS null hypothesis: the series is stationary.
    A small p-value rejects the null — evidence of a unit root.

    Using both tests together guards against the known weaknesses of each:
    ADF has low power against near-unit-root processes; KPSS over-rejects in
    the presence of structural breaks.  The combined conclusion classifies each
    series into one of four states:

        stationary            – ADF rejects unit root AND KPSS does not reject stationarity
        unit root             – ADF does not reject unit root AND KPSS rejects stationarity
        inconclusive (both)   – both tests reject (possible fractional integration)
        inconclusive (neither)– neither test rejects (insufficient power / short series)

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Path to lexical_temporal.csv or an already-loaded DataFrame.
    metrics : sequence of str, optional
        Metric columns to test.  Defaults to TEMPORAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings (default 'year_month').
    alpha : float
        Significance level for both tests (default 0.05).
    adf_maxlag : int
        Maximum lag order passed to adfuller; lag selected by AIC within this
        bound (default 4).

    Returns
    -------
    pd.DataFrame
        One row per (subreddit, metric) with columns:
          subreddit        – community label
          metric           – metric name
          n_obs            – number of monthly observations
          adf_stat         – ADF test statistic
          adf_p            – ADF p-value
          adf_lags         – lag order chosen by AIC
          adf_stationary   – True if adf_p < alpha
          kpss_stat        – KPSS test statistic
          kpss_p           – KPSS p-value (may be boundary-clipped by statsmodels)
          kpss_stationary  – True if kpss_p > alpha
          conclusion       – one of the four classification strings above
    """
    if metrics is None:
        metrics = TEMPORAL_METRICS

    df = _load_temporal_df(source)

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    if subreddit_col not in df.columns:
        raise ValueError(f"Missing subreddit column: '{subreddit_col}'")

    subreddits = sorted(df[subreddit_col].dropna().unique())
    records = []

    for subreddit in subreddits:
        for metric in metrics:
            series = _extract_series(df, subreddit, metric, subreddit_col, time_col)
            n = len(series)

            if n < 10:
                records.append(dict(
                    subreddit=subreddit, metric=metric, n_obs=n,
                    adf_stat=np.nan, adf_p=np.nan, adf_lags=np.nan,
                    adf_stationary=np.nan,
                    kpss_stat=np.nan, kpss_p=np.nan,
                    kpss_stationary=np.nan,
                    conclusion="insufficient data",
                ))
                continue

            # --- ADF test ---
            adf_out = adfuller(series.values, maxlag=adf_maxlag, autolag="AIC")
            adf_stat  = float(adf_out[0])
            adf_p     = float(adf_out[1])
            adf_lags  = int(adf_out[2])
            adf_stat_flag = adf_p < alpha

            # --- KPSS test (suppress interpolation boundary warnings) ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_out = kpss(series.values, regression="c", nlags="auto")
            kpss_stat      = float(kpss_out[0])
            kpss_p         = float(kpss_out[1])
            kpss_stat_flag = kpss_p > alpha  # fail to reject → stationary

            # --- Combined conclusion ---
            if adf_stat_flag and kpss_stat_flag:
                conclusion = "stationary"
            elif not adf_stat_flag and not kpss_stat_flag:
                conclusion = "unit root"
            elif adf_stat_flag and not kpss_stat_flag:
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
                adf_stationary=adf_stat_flag,
                kpss_stat=round(kpss_stat, 4),
                kpss_p=round(kpss_p, 6),
                kpss_stationary=kpss_stat_flag,
                conclusion=conclusion,
            ))

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------------
# ACF Plots
# ----------------------------------------------------------------------------------------

def plot_acf_grid(
    source: "str | Path | pd.DataFrame",
    metrics: Optional[Sequence[str]] = None,
    subreddit_col: str = "subreddit",
    time_col: str = "year_month",
    n_lags: int = 12,
    alpha: float = 0.05,
    save_path: Optional["str | Path"] = None,
) -> None:
    """Plot a grid of ACF (autocorrelation function) charts for each
    (metric, subreddit) pair in lexical_temporal.csv.

    Layout: one row per metric, one column per subreddit.  Each panel shows
    autocorrelation at lags 1 through n_lags with a ±1.96/√T significance band.
    Bars that exceed the band are highlighted in red to draw attention to
    lags with statistically significant autocorrelation.

    A high ACF(1) — and slowly decaying bars across many lags — is the visual
    signature of a near-unit-root process and signals that OLS trend estimates
    will require HAC standard errors or first-differencing before inference.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Path to lexical_temporal.csv or an already-loaded DataFrame.
    metrics : sequence of str, optional
        Metrics to include.  Defaults to TEMPORAL_METRICS.
    subreddit_col : str
        Column identifying the community (default 'subreddit').
    time_col : str
        Column containing year-month strings (default 'year_month').
    n_lags : int
        Number of lags to display on each ACF plot (default 12).
    alpha : float
        Significance level for the confidence band (default 0.05).
        Band is drawn at ±z * 1/√T where z = scipy.stats.norm.ppf(1 - alpha/2).
    save_path : str or Path, optional
        If provided, save the figure to this path instead of displaying it.
    """
    if metrics is None:
        metrics = TEMPORAL_METRICS

    df = _load_temporal_df(source)

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    subreddits = sorted(df[subreddit_col].dropna().unique())
    n_rows = len(metrics)
    n_cols = len(subreddits)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.2 * n_rows),
        sharey=False,
        squeeze=False,
    )

    # colour palette: one colour per subreddit column
    col_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row, metric in enumerate(metrics):
        for col, subreddit in enumerate(subreddits):
            ax = axes[row][col]
            series = _extract_series(df, subreddit, metric, subreddit_col, time_col)
            T = len(series)

            if T < n_lags + 2:
                ax.set_title(f"{subreddit}\n(insufficient data)", fontsize=9)
                ax.axis("off")
                continue

            acf_vals, confint = acf(series.values, nlags=n_lags, alpha=alpha, fft=True)
            lags       = np.arange(1, n_lags + 1)
            acf_plot   = acf_vals[1:]
            ci_low     = confint[1:, 0] - acf_vals[1:]
            ci_high    = confint[1:, 1] - acf_vals[1:]
            sig_band   = 1.96 / np.sqrt(T)
            sig_mask   = np.abs(acf_plot) > sig_band

            bar_colors = [
                col_colors[col % len(col_colors)] if not s else "#DC2626"
                for s in sig_mask
            ]

            ax.bar(lags, acf_plot, color=bar_colors, alpha=0.75, width=0.5)
            ax.fill_between(lags, ci_low, ci_high, alpha=0.15, color="grey")
            ax.axhline(0,          color="black", linewidth=0.8)
            ax.axhline( sig_band,  color="grey",  linewidth=1.0, linestyle="--")
            ax.axhline(-sig_band,  color="grey",  linewidth=1.0, linestyle="--")
            ax.set_xlim(0.25, n_lags + 0.75)
            ax.set_xticks(lags)
            ax.grid(True, axis="y", alpha=0.3)

            # first row: subreddit name as column header
            if row == 0:
                label = subreddit.replace("subreddit-", "r/")
                ax.set_title(label, fontsize=11, fontweight="bold")

            # first column: metric name as row label
            if col == 0:
                ax.set_ylabel(
                    TEMPORAL_METRIC_LABELS.get(metric, metric), fontsize=10
                )

            # annotate ACF(1) value
            ax.text(
                0.97, 0.92, f"ACF(1)={acf_vals[1]:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    for row in range(n_rows):
        for col in range(n_cols):
            axes[row][col].set_xlabel("Lag (months)", fontsize=8)

    fig.suptitle(
        "ACF of Monthly Corpus-Level Lexical Metrics by Community\n"
        "(red bars exceed ±1.96/√T significance threshold)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ACF grid saved to: {save_path}")
    else:
        plt.show()
