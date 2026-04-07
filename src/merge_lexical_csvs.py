"""
merge_lexical_csvs.py

Merges all Teenagers_lexical_df_*.csv files into a single
Teenagers_lexical_df.csv. Run this on Adroit after all pipeline
jobs have finished.

Usage:
    python merge_lexical_csvs.py [--dir PATH] [--out PATH]

Defaults:
    --dir   .   (current working directory)
    --out   ./Teenagers_lexical_df.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Merge Teenagers lexical CSVs.")
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing the input CSVs (default: current directory)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path (default: <dir>/Teenagers_lexical_df.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.dir)
    output_path = args.out or os.path.join(input_dir, "Teenagers_lexical_df.csv")

    # Collect all matching input files
    pattern = os.path.join(input_dir, "Teenagers_lexical_df_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No files matching '{pattern}' found.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to merge:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    # Read and concatenate
    dfs = []
    for f in files:
        print(f"Reading {os.path.basename(f)} ...", end=" ", flush=True)
        df = pd.read_csv(f, low_memory=False)
        print(f"{len(df):,} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after merge: {len(merged):,}")
    print(f"Columns: {list(merged.columns)}")

    # Save
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged CSV to: {output_path}")


if __name__ == "__main__":
    main()
