# ----------------------------------------------------------------------------------------
# This code is designed to run the entire analysis pipeline on the Adriot cluster
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

# imports
import os
import sys
import argparse
from run_pipeline import run_full_pipeline_cnvkt_batches
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONVOKIT_ROOT = SCRIPT_DIR.parent.parent / "Thesis-Data" / "Convokit"

def main():
    parser = argparse.ArgumentParser(
        description="Run lexical and syntactic analysis on a single Convokit corpus."
    )

    parser.add_argument(
        "corpus_name",
        type=str,
        help="Name of subreddit corpus folder (e.g., politics, askreddit)"
    )

    parser.add_argument(
        "--variation",
        type=str,
        required=True,
        help="Variation folder (e.g., Age-Variation, Topic-Variation, Culture-Variation)"
    )

    parser.add_argument(
        "--convokit_root",
        type=str,
        default=str(DEFAULT_CONVOKIT_ROOT),
        help="Path to Convokit root directory"
    )

    parser.add_argument(
            "--batch_size",
            type=int,
            default=100,
            help="Rows per batch for pipeline processing (lower = less memory, slower)"
        )

    args = parser.parse_args()

    # construct full path
    corpus_dir = os.path.join(
        args.convokit_root,
        args.variation,
        args.corpus_name
    )

    if not os.path.isdir(corpus_dir):
        print(f"ERROR: Corpus directory not found:\n{corpus_dir}")
        sys.exit(1)

    run_full_pipeline_cnvkt_batches(corpus_dir, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
