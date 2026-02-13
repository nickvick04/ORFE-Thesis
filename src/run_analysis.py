# ----------------------------------------------------------------------------------------
# This code is designed to run the entire analysis pipeline on the Adriot cluster
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

# imports
import os
import sys
import argparse
from run_pipeline import run_full_pipeline_cnvkt

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
        default="../../Thesis-Data/Convokit",
        help="Path to Convokit root directory"
    )

    args = parser.parse_args()

    # Construct full path
    corpus_dir = os.path.join(
        args.convokit_root,
        args.variation,
        args.corpus_name
    )

    if not os.path.isdir(corpus_dir):
        print(f"ERROR: Corpus directory not found:\n{corpus_dir}")
        sys.exit(1)

    run_full_pipeline_cnvkt(corpus_dir)

if __name__ == "__main__":
    main()