# ----------------------------------------------------------------------------------------
# This code is designed to process and analyze all the Convokit data
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------
# imports
import os
import gc
import pandas as pd
from data_preprocessing import corpus_longest_posts_batches, corpus_longest_posts_batches_from_jsonl
from lexical_analysis_functions import compute_lexical_vals
from visualization import *

BATCH_SIZE = 1000

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
    from convokit import Corpus
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

        # write to new file if first batch, o/w append to existing file
        df_batch.to_csv(output_path, mode="w" if first_batch else "a", header=first_batch, index=False)

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

    print(f"Loading utterances.jsonl from disk: {corpus_name}")

    first_batch = True
    i = 0
    print(f"Processing {corpus_name} lexical batches...")
    print(f"Currently processing lexical batch: {i}")

    for df_batch in corpus_longest_posts_batches_from_jsonl(corpus_dir, batch_size=batch_size):
        print(f"Currently processing lexical batch: {i}")
        df_batch = compute_lexical_vals(df_batch)

        df_batch.to_csv(output_path, mode="w" if first_batch else "a", header=first_batch, index=False)

        first_batch = False
        i += 1
        del df_batch
        gc.collect()

def run_temporal_pipeline(input_path: str, output_path: str, n_workers: int = 1):
    '''Reads the combined utterance-level lexical_master.csv, aggregates all utterances
    to (subreddit, year_month) monthly corpora, computes corpus-level lexical metrics,
    and writes the resulting panel to output_path as a single CSV.

    Corpus-level metrics (MATTR, MTLD, Yule's K, NAWL) are computed on the concatenated
    token list for each monthly cell rather than averaged from utterance-level scores.
    Zipf and AoA are token-count-weighted means of the utterance-level scores.

    Output schema
    -------------
    subreddit, year_month, source_variation,
    n_utterances, n_speakers, corpus_token_count,
    mattr_score, mtld_score, yules_k,
    zipf_score, aoa_score, nawl_ratio

    Parameters
    ----------
    input_path  : str  Path to lexical_master.csv.
    output_path : str  Destination path for the output CSV (e.g. lexical_temporal.csv).
    n_workers   : int  Worker processes for parallel tokenization (default 1).
                       Should match --cpus-per-task in the SLURM script.
    '''

    from temporal import aggregate_temporal_metrics, NEEDED_COLS

    print(f"Reading utterance-level data from: {input_path}")
    df = pd.read_csv(input_path, usecols=NEEDED_COLS, low_memory=False)
    print(f"Loaded {len(df):,} utterances across {df['subreddit'].nunique()} subreddits.")

    temporal_df = aggregate_temporal_metrics(df, n_workers=n_workers)

    print(f"\nWriting temporal panel to: {output_path}")
    temporal_df.to_csv(output_path, index=False)
    print("Done.")
