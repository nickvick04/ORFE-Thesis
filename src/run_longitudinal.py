"""
run_longitudinal.py
===================
CLI runner for the longitudinal user trajectory model.

Designed for the ArcticShift lexical_df_combined.csv schema, which
provides year_month, edited, and log_freq_month as pre-computed columns.

Usage (local)
-------------
    python src/run_longitudinal.py \
        --data     /path/to/lexical_df_combined.csv \
        --out_dir  /path/to/Results/longitudinal \
        --min_months 6 \
        --no_random_slope          # optional: intercept-only for speed

Usage (via SLURM on Adroit)
---------------------------
    sbatch src/run_longitudinal.slurm

Author: Nicholas Vickery, Princeton ORFE '26
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit longitudinal user trajectory models for lexical quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to lexical_df_combined.csv (ArcticShift combined dataset).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/scratch/network/nv9344/Thesis/Visualizations/longitudinal"),
        help="Directory to write output CSV and log.",
    )
    parser.add_argument(
        "--min_months",
        type=int,
        default=6,
        help="Minimum months of activity required per user.",
    )
    parser.add_argument(
        "--no_random_slope",
        action="store_true",
        default=False,
        help=(
            "Use a random-intercept-only model. Faster but less expressive. "
            "Use if random-slope model fails to converge on your hardware."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Subset of metrics to model. Defaults to all six: "
            "mtld_score mattr_score yules_k zipf_score aoa_score nawl_ratio."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for BH-adjusted p-values.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Columns needed from the ArcticShift lexical_df_combined.csv.
# year_month, edited, and log_freq_month are all pre-computed — no
# timestamp parsing or column renaming required.
_USECOLS = [
    "utterance_id",
    "speaker_id",
    "subreddit",
    "year_month",
    "num_utterances_by_speaker_month",
    "log_freq_month",
    "post_depth",
    "edited",
    "score",
    "num_direct_replies",
    "controversiality",
    "mtld_score",
    "mattr_score",
    "yules_k",
    "zipf_score",
    "aoa_score",
    "nawl_ratio",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load lexical_df_combined.csv, keeping only the necessary columns."""
    logger.info(f"Loading data from {path} ...")
    t0 = time.time()

    # Intersect with columns actually present in the file
    header = pd.read_csv(path, nrows=0)
    available = [c for c in _USECOLS if c in header.columns]
    missing = [c for c in _USECOLS if c not in header.columns]
    if missing:
        logger.warning(f"Columns not found in CSV (will be skipped): {missing}")

    df = pd.read_csv(path, usecols=available, low_memory=False)

    logger.info(
        f"Loaded {len(df):,} rows, {df['speaker_id'].nunique():,} unique users "
        f"across {df['subreddit'].nunique()} subreddit(s) "
        f"in {time.time()-t0:.1f}s."
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_longitudinal.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logs will be written to {log_path}")

    # ------------------------------------------------------------------
    # Import model (done here to keep import errors visible in logs)
    # ------------------------------------------------------------------
    from longitudinal_trajectory import run_longitudinal_trajectory  # noqa: E402

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = load_data(args.data)

    # ------------------------------------------------------------------
    # Fit models
    # ------------------------------------------------------------------
    t0 = time.time()
    logger.info(
        f"Fitting longitudinal trajectory models "
        f"(min_months={args.min_months}, "
        f"random_slope={not args.no_random_slope}) ..."
    )
    results = run_longitudinal_trajectory(
        df,
        metrics=args.metrics,
        min_months=args.min_months,
        random_slope=not args.no_random_slope,
        # ArcticShift column names (all pre-computed)
        speaker_col="speaker_id",
        subreddit_col="subreddit",
        year_month_col="year_month",
        monthly_count_col="num_utterances_by_speaker_month",
        log_freq_col="log_freq_month",
        alpha=args.alpha,
    )
    logger.info(f"Modelling complete in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = args.out_dir / "longitudinal_results.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Print summary table to stdout
    # ------------------------------------------------------------------
    display_cols = [
        "metric_label", "n_users", "n_obs",
        "mu_beta", "mu_beta_se", "mu_beta_p_bh",
        "gamma", "gamma_p_bh",
        "sigma2_b", "conclusion",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    with pd.option_context("display.max_columns", None, "display.width", 120,
                           "display.float_format", "{:.6f}".format):
        logger.info("\n" + results[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
