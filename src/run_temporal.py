# ----------------------------------------------------------------------------------------
# This code runs the temporal aggregation pipeline on the combined lexical_master.csv,
# producing a (subreddit, year_month) panel of corpus-level lexical metrics.
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import os
import sys
import argparse
from pathlib import Path

from run_pipeline import run_temporal_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "Data"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate lexical_master.csv to monthly subreddit corpora."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_DATA_DIR / "lexical_master.csv"),
        help="Path to the combined utterance-level CSV (default: ORFE-Thesis/Data/lexical_master.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DATA_DIR / "lexical_temporal.csv"),
        help="Destination path for the monthly panel CSV (default: ORFE-Thesis/Data/lexical_temporal.csv)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Override the Data directory (adjusts default --input and --output paths)",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of worker processes for parallel tokenization (default 1). "
             "Set to match --cpus-per-task in your SLURM script.",
    )

    args = parser.parse_args()

    # if a custom data_dir is given but input/output were left at defaults, re-derive them
    if args.data_dir != str(DEFAULT_DATA_DIR):
        data_dir = Path(args.data_dir)
        if args.input == str(DEFAULT_DATA_DIR / "lexical_master.csv"):
            args.input = str(data_dir / "lexical_master.csv")
        if args.output == str(DEFAULT_DATA_DIR / "lexical_temporal.csv"):
            args.output = str(data_dir / "lexical_temporal.csv")

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found:\n{args.input}")
        sys.exit(1)

    run_temporal_pipeline(
        input_path=args.input,
        output_path=args.output,
        n_workers=args.cpus,
    )


if __name__ == "__main__":
    main()
