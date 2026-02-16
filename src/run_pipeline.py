# ----------------------------------------------------------------------------------------
# This code is designed to process and analyze all the Convokit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# imports
import os
from data_preprocessing import corpus_to_df, corpus_to_df_batches, filter_df
from lexical_analysis_functions import compute_lexical_vals
from syntactic_analysis_functions import compute_syntactic_vals
from visualization import *
from convokit import Corpus

BATCH_SIZE = 5000

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

def run_full_pipeline_cnvkt_batches(corpus_dir: str, batch_size=5000):
    '''Runs full preprocessing and analysis pipeline on a single Convokit
    corpus and writes a CSV to the corpus' parent Variation folder in batches
    so as to reduce the memory capacity demanded of the cluster.'''

    # extract corpus
    corpus_name = os.path.basename(corpus_dir)
    print(f"Processing corpus: {corpus_name}")

    # get path to csv next to corpus
    output_dir = os.path.dirname(corpus_dir)
    output_path = os.path.join(output_dir, f"{corpus_name}_df.csv")

    # load corpus
    print(f"Loading corpus: {corpus_name}")
    corpus = Corpus(corpus_dir)

    # boolean and index to track
    first_batch = True
    i = 0
    print(f"Processing {corpus_name} in batches...")
    print(f"Currently processing batch: {i}")

    # iterate through batches, writing out results incrementally
    for df_batch in corpus_to_df_batches(corpus, batch_size=BATCH_SIZE):
        
        df_batch = filter_df(df_batch)

        print(f"Analyzing corpus batch: {corpus_name}")
        df_batch = compute_lexical_vals(df_batch)
        df_batch = compute_syntactic_vals(df_batch)

        df_batch.set_index('timestamp', inplace=True)
        # write to new file if first batch, o/w append to existing file
        df_batch.to_csv(output_path, mode="w" if first_batch else "a", header=first_batch)

        # udpate boolean and index
        first_batch = False
        i += 1
        # delete explicitly to save storage
        del df_batch

def run_all_corpora_cnvkt(convokit_root: str):
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