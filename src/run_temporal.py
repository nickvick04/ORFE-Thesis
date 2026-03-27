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
DEFAULT_CONVOKIT_ROOT = SCRIPT_DIR.parent.parent / "Thesis-Data" / "Convokit"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate lexical_master.csv to monthly subreddit corpora."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_CONVOKIT_ROOT / "lexical_master.csv"),
        help="Path to the combined utterance-level CSV (default: Thesis-Data/Convokit/lexical_master.csv)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_CONVOKIT_ROOT / "lexical_temporal.csv"),
        help="Destination path for the monthly panel CSV (default: Thesis-Data/Convokit/lexical_temporal.csv)",
    )

    parser.add_argument(
        "--convokit_root",
        type=str,
        default=str(DEFAULT_CONVOKIT_ROOT),
        help="Override the Convokit root directory (adjusts default --input and --output paths)",
    )

    args = parser.parse_args()

    # if a custom convokit_root is given but input/output were left at defaults, re-derive them
    if args.convokit_root != str(DEFAULT_CONVOKIT_ROOT):
        root = Path(args.convokit_root)
        if args.input == str(DEFAULT_CONVOKIT_ROOT / "lexical_master.csv"):
            args.input = str(root / "lexical_master.csv")
        if args.output == str(DEFAULT_CONVOKIT_ROOT / "lexical_temporal.csv"):
            args.output = str(root / "lexical_temporal.csv")

    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found:\n{args.input}")
        sys.exit(1)

    run_temporal_pipeline(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
