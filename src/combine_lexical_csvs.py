"""
combine_lexical_csvs.py
=======================
Combines the four per-subreddit lexical CSVs produced by the ArcticShift
pipeline into a single regression-ready file.

Transformations applied
-----------------------
1.  raw_text is dropped — not needed for any regression and can be large.

2.  Known bot / automated accounts are removed using the speaker_id
        blocklist in bot_usernames.csv (same directory as this script).

3.  edited → binary int
        "False" / False / 0  →  0
        any Unix timestamp   →  1

4.  timestamp → datetime (UTC), then:
        year_month  (Period string, e.g. "2015-01") — used as the time
                    fixed effect γ_t in the panel regression and the
                    groupby key for the baseline OLS aggregation

5.  log_freq_month = log1p(num_utterances_by_speaker_month)
        This is F_ut in the fixed-effects panel regression and the
        building block of F̄_u in the cross-user WLS regression.

Output
------
    <OUTPUT_DIR>/lexical_df_combined.csv

    Column order:
        utterance_id, speaker_id, subreddit,
        timestamp, year_month,
        num_utterances_by_speaker, num_utterances_by_speaker_month,
        log_freq_month,
        post_depth, score, num_direct_replies, controversiality, edited,
        mtld_score, mattr_score, yules_k, zipf_score, aoa_score, nawl_ratio

Usage
-----
    # Default paths (matches Adroit layout used by run_arcticshift.py)
    python combine_lexical_csvs.py

    # Override input / output directories
    python combine_lexical_csvs.py --input_dir /path/to/csvs --output_dir /path/to/out

Author: Nicholas Vickery, Princeton ORFE '26
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — edit INPUT_DIR / OUTPUT_DIR if your paths differ
# ---------------------------------------------------------------------------

# Directory that contains the four <Subreddit>_lexical_df.csv files.
# This matches the output location written by run_lexical_pipeline_arcticshift_batches().
DEFAULT_INPUT_DIR = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift"

# Where to write lexical_df_combined.csv (defaults to same directory).
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

# CSV listing known bot / automated accounts to exclude from the combined output.
# Each row must have a 'username' column matching speaker_id values in the data.
# Path is resolved relative to this script file so it works from any working directory.
BOT_USERNAMES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_usernames.csv")

# Expected subreddit file stems — used to warn if any are missing.
EXPECTED_STEMS = ["College", "Parenting", "Retirement", "Teenagers"]

# Final column order for the output CSV.
OUTPUT_COLUMNS = [
    "utterance_id", "speaker_id", "subreddit",
    "timestamp", "year_month",
    "num_utterances_by_speaker", "num_utterances_by_speaker_month",
    "log_freq_month",
    "post_depth", "score", "num_direct_replies", "controversiality", "edited",
    "mtld_score", "mattr_score", "yules_k", "zipf_score", "aoa_score", "nawl_ratio",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bot_usernames(csv_path: str) -> set:
    """Load the set of known-bot speaker IDs from *csv_path*.

    The file must have a header row with at least a 'username' column.
    Returns an empty set (with a warning) if the file is missing or malformed,
    so the pipeline can still run without the filter applied.
    """
    if not os.path.isfile(csv_path):
        print(f"WARNING: bot_usernames.csv not found at {csv_path}. Bot filtering skipped.")
        return set()
    try:
        bot_df = pd.read_csv(csv_path, usecols=["username"], dtype=str)
        bots = set(bot_df["username"].dropna().str.strip())
        return bots
    except Exception as exc:
        print(f"WARNING: could not load bot usernames from {csv_path}: {exc}. Bot filtering skipped.")
        return set()


def _edited_to_binary(series: pd.Series) -> pd.Series:
    """Convert the raw 'edited' column to a 0/1 integer.

    The ArcticShift pipeline stores:
        False           — comment was never edited
        <unix_ts int>   — comment was edited; value is the edit timestamp

    After CSV round-trip the column arrives as mixed dtype (object), so
    we test for the string "False" and the boolean False explicitly.
    """
    def _convert(val):
        if val is False or val == "False" or val == 0 or val == "0":
            return 0
        try:
            # Any numeric value other than 0 is a Unix timestamp → edited
            numeric = float(val)
            return 0 if numeric == 0 else 1
        except (TypeError, ValueError):
            return 0  # treat unrecognised values as not-edited

    return series.apply(_convert).astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-subreddit lexical CSVs into one regression-ready file."
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing <Subreddit>_lexical_df.csv files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write lexical_df_combined.csv (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Discover input files
    # ------------------------------------------------------------------
    pattern = os.path.join(args.input_dir, "*_lexical_df.csv")
    found_files = sorted(glob.glob(pattern))

    if not found_files:
        sys.exit(
            f"No files matching '*_lexical_df.csv' found in {args.input_dir}.\n"
            "Check --input_dir or that the pipeline has finished."
        )

    print(f"Found {len(found_files)} CSV file(s):")
    for f in found_files:
        print(f"  {os.path.basename(f)}")

    # Warn if any expected subreddits are absent
    found_stems = {os.path.basename(f).split("_lexical_df")[0] for f in found_files}
    missing = set(EXPECTED_STEMS) - found_stems
    if missing:
        print(f"WARNING: expected subreddit file(s) not found: {sorted(missing)}")

    # ------------------------------------------------------------------
    # 2. Load and concatenate
    # ------------------------------------------------------------------
    dfs = []
    for fpath in found_files:
        df_sub = pd.read_csv(fpath, low_memory=False)
        print(f"  Loaded {os.path.basename(fpath)}: {len(df_sub):,} rows, {df_sub['subreddit'].iloc[0] if 'subreddit' in df_sub.columns and len(df_sub) > 0 else '?'}")
        dfs.append(df_sub)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df):,} rows across {df['subreddit'].nunique()} subreddit(s).")

    # Drop raw text — not used in any regression and expensive to store.
    if "raw_text" in df.columns:
        df.drop(columns=["raw_text"], inplace=True)

    # ------------------------------------------------------------------
    # 3. Remove known bot / automated accounts
    # ------------------------------------------------------------------
    bot_usernames = _load_bot_usernames(BOT_USERNAMES_CSV)
    if bot_usernames:
        before = len(df)
        df = df[~df["speaker_id"].isin(bot_usernames)].copy()
        removed = before - len(df)
        print(f"Bot filter: removed {removed:,} rows from {len(bot_usernames):,} known-bot accounts "
              f"({removed / before * 100:.2f}% of combined data).")
    else:
        print("Bot filter: no bot usernames loaded — skipping.")

    # ------------------------------------------------------------------
    # 5. edited → binary
    # ------------------------------------------------------------------
    df["edited"] = _edited_to_binary(df["edited"])

    # ------------------------------------------------------------------
    # 6. timestamp → datetime, then year_month
    # ------------------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts > 0:
        print(f"WARNING: {n_bad_ts:,} rows have unparseable timestamps and will have NaT in timestamp / NaN in time columns.")

    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

    # ------------------------------------------------------------------
    # 7. log_freq_month  (F_ut = log(1 + num_utterances_by_speaker_month))
    # ------------------------------------------------------------------
    df["log_freq_month"] = np.log1p(df["num_utterances_by_speaker_month"])

    # ------------------------------------------------------------------
    # 8. Reorder columns and write output
    # ------------------------------------------------------------------
    # Keep only columns that exist in this dataframe (guards against schema drift)
    cols_out = [c for c in OUTPUT_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    if extra_cols:
        print(f"Note: additional columns appended at end: {extra_cols}")
        cols_out += extra_cols

    df = df[cols_out]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "lexical_df_combined.csv")
    df.to_csv(output_path, index=False)

    print(f"\nDone — {len(df):,} rows written to:\n  {output_path}")
    print("\nColumn summary:")
    print(df.dtypes.to_string())
    print(f"\nSubreddit breakdown:\n{df['subreddit'].value_counts().to_string()}")
    print(f"\nEdited value counts:\n{df['edited'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
