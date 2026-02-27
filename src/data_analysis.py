# ----------------------------------------------------------------------------------------
# This code is designed to analyze post-cluster computation CSV files
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, List, Optional, Sequence

import pymannkendall as mk

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LEXICAL_METRICS = ["mtld_score", "yules_k", "zipf_score", "aoa_score", "nawl_ratio"]


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

    dfs = []
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Reading {csv_path}")
        try:
            df = pd.read_csv(csv_path)
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


def clean_variation_and_subreddit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source_variation and subreddit based on hyphen-separated values.

    - source_variation: keep text before first hyphen (lowercased)
    - subreddit: keep text after first hyphen
    """
    required_cols = {"source_variation", "subreddit"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df.copy()
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
