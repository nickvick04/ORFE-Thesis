# ----------------------------------------------------------------------------------------
# This code is designed to process and analyze all the Convokit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# imports
import os
from data_preprocessing import corpus_to_df, filter_df
from lexical_analysis_functions import compute_lexical_vals
from syntactic_analysis_functions import compute_syntactic_vals
from visualization import *
from convokit import Corpus

def run_full_pipeline_cnvkt(corpus_dir: str):
    '''Runs full preprocessing and analysis pipeline on a single Convokit
    corpus and writes a CSV to the corpus' parent Variation folder'''
    corpus_name = os.path.basename(corpus_dir)
    print(f"Processing corpus: {corpus_name}")

    # load corpus
    print(f"Loading corpus: {corpus_name}")
    corpus = Corpus(corpus_dir)

    # pipeline
    print(f"Converting corpus: {corpus_name}")
    df = corpus_to_df(corpus)
    df = filter_df(df)
    print(f"Analyzing corpus: {corpus_name}")
    df = compute_lexical_vals(df)
    df = compute_syntactic_vals(df)
    df.set_index('timestamp', inplace=True)

    # save csv next to corpus
    output_dir = os.path.dirname(corpus_dir)
    output_path = os.path.join(output_dir, f"{corpus_name}_df.csv")

    df.to_csv(output_path)
    print(f"Saved -> {output_path}\n")


def run_all_corpora__cnvkt(convokit_root: str):
    '''Runs pipeline for all corpora under Convokit root directory.'''

    for variation in os.listdir(convokit_root):
        variation_path = os.path.join(convokit_root, variation)

        if not os.path.isdir(variation_path):
            continue

        print(f"\n=== {variation} ===")

        for corpus_name in os.listdir(variation_path):
            corpus_dir = os.path.join(variation_path, corpus_name)

            if not os.path.isdir(corpus_dir):
                continue

            run_full_pipeline_cnvkt(corpus_dir)