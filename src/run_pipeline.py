# ----------------------------------------------------------------------------------------
# This code is designed to process and analyze all the Convokit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# imports
import os
import gc
from data_preprocessing import corpus_to_df, corpus_longest_posts_batches, filter_df
from lexical_analysis_functions import compute_lexical_vals
from visualization import *
from convokit import Corpus

BATCH_SIZE = 1000

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
    from syntactic_analysis_functions import compute_syntactic_vals
    df = compute_syntactic_vals(df)
    df.set_index('timestamp', inplace=True)

    # save csv next to corpus
    output_dir = os.path.dirname(corpus_dir)
    output_path = os.path.join(output_dir, f"{corpus_name}_df.csv")

    df.to_csv(output_path)
    print(f"Saved -> {output_path}\n")

def run_full_pipeline_cnvkt_batches(corpus_dir: str, batch_size=BATCH_SIZE, num_shards=1, shard_index=0):
    '''Runs full preprocessing and analysis pipeline on a single Convokit
    corpus and writes a CSV to the corpus' parent Variation folder in batches
    so as to reduce the memory capacity demanded of the cluster.'''

    # extract corpus
    corpus_name = os.path.basename(corpus_dir)
    print(f"Processing corpus: {corpus_name}")

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    # get path to csv next to corpus
    output_dir = os.path.dirname(corpus_dir)
    if num_shards == 1:
        output_name = f"{corpus_name}_df.csv"
    else:
        output_name = f"{corpus_name}_df_shard-{shard_index:03d}-of-{num_shards:03d}.csv"
    output_path = os.path.join(output_dir, output_name)

    # load corpus
    print(f"Loading corpus: {corpus_name}")
    corpus = Corpus(corpus_dir)

    # boolean and index to track
    first_batch = True
    i = 0
    print(f"Processing {corpus_name} in batches...")
    print(f"Shard {shard_index + 1}/{num_shards}")
    print(f"Currently processing batch: {i}")

    # iterate through globally filtered longest-post rows in batches
    from syntactic_analysis_functions import compute_syntactic_vals
    for df_batch in corpus_longest_posts_batches(
        corpus,
        batch_size=batch_size,
        num_shards=num_shards,
        shard_index=shard_index,
    ):

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
        gc.collect()

def run_lexical_pipeline_cnvkt_batches(corpus_dir: str, batch_size=BATCH_SIZE):
    '''Runs lexical-only preprocessing and analysis on a single Convokit corpus in batches.'''

    corpus_name = os.path.basename(corpus_dir)
    print(f"Processing corpus (lexical only): {corpus_name}")

    output_dir = os.path.dirname(corpus_dir)
    output_name = f"{corpus_name}_lexical_df.csv"
    output_path = os.path.join(output_dir, output_name)

    print(f"Loading corpus: {corpus_name}")
    corpus = Corpus(corpus_dir)

    first_batch = True
    i = 0
    print(f"Processing {corpus_name} lexical batches...")
    print(f"Currently processing batch: {i}")

    for df_batch in corpus_longest_posts_batches(corpus, batch_size=batch_size):
        print(f"Analyzing lexical batch: {corpus_name}")
        df_batch = compute_lexical_vals(df_batch)

        df_batch.set_index('timestamp', inplace=True)
        df_batch.to_csv(output_path, mode="w" if first_batch else "a", header=first_batch)

        first_batch = False
        i += 1
        del df_batch
        gc.collect()

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
