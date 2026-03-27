# ----------------------------------------------------------------------------------------
# This code aggregates utterance-level data to (subreddit, year-month) corpora and
# computes corpus-level lexical measures for temporal analysis.
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

# general imports
import gc
import numpy as np
import pandas as pd
from multiprocessing import Pool

# import preprocessing and lexical functions from existing pipeline
from data_preprocessing import clean_tokens_lexical
from lexical_analysis_functions import mattr_score, mtld_score, yules_K, nawl_ratio

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------

# only load the columns we need; utterance-level lexical scores are not re-used directly
# except zipf_score and aoa_score, which are averaged with token-count weighting
NEEDED_COLS = [
    "utterance_id",
    "speaker_id",
    "raw_text",
    "timestamp",
    "subreddit",
    "source_variation",
    "zipf_score",
    "aoa_score",
]

# ----------------------------------------------------------------------------------------
# Tokenization Helper
# ----------------------------------------------------------------------------------------
def _tokenize_utterance(text):
    '''Apply standard lexical preprocessing to a single raw utterance.
    Returns a list of cleaned, lemmatized tokens. Returns an empty list on failure.
    Wraps clean_tokens_lexical so group-level apply calls are safe.'''

    try:
        return clean_tokens_lexical(str(text))
    except Exception:
        return []

# ----------------------------------------------------------------------------------------
# Corpus-Level Metric Computation
# ----------------------------------------------------------------------------------------
def compute_corpus_metrics(group_df):
    '''Compute corpus-level lexical metrics for a single (subreddit, year_month) cell.

    MATTR, MTLD, and Yule's K are computed on the concatenated corpus token list so that
    the metric reflects the full community vocabulary for that month rather than an average
    of per-utterance scores (which would be sensitive to post-length distributions and would
    produce unreliable estimates for the large fraction of short posts in the corpus).

    Zipf and AoA are computed as token-count-weighted means of the utterance-level scores
    already in lexical_master.csv. Weighting by token count is necessary because each
    utterance score is itself a per-word mean; equal-weighting utterances would give a
    5-word post the same influence as a 500-word post.

    NAWL is recomputed at the corpus level so that the ratio reflects total academic word
    usage across the full monthly corpus, not an average of per-utterance ratios (which
    would be distorted by the ~97.5% zero-inflation at utterance level).

    Parameters
    ----------
    group_df : pd.DataFrame
        Subset of the utterance-level DataFrame for a single (subreddit, year_month) cell.
        Must contain columns: raw_text, speaker_id, zipf_score, aoa_score.
        If a '_tokens' column is present (pre-computed by aggregate_temporal_metrics),
        it is used directly and raw_text is not re-tokenized.

    Returns
    -------
    dict
        Corpus-level metric values for this cell.
    '''

    # --- Use pre-computed tokens if available; otherwise tokenize on-the-fly ---
    if "_tokens" in group_df.columns:
        token_lists = group_df["_tokens"].tolist()
    else:
        token_lists = group_df["raw_text"].apply(_tokenize_utterance).tolist()

    token_counts = [len(tl) for tl in token_lists]
    corpus_tokens = [tok for tl in token_lists for tok in tl]
    n_corpus_tokens = len(corpus_tokens)

    # --- Corpus-level diversity metrics (must be computed on concatenated tokens) ---
    corpus_mattr = mattr_score(corpus_tokens)
    corpus_mtld  = mtld_score(corpus_tokens)
    corpus_yules = yules_K(corpus_tokens)
    corpus_nawl  = nawl_ratio(corpus_tokens)

    # --- Token-count-weighted mean Zipf ---
    zipf_pairs   = [(w, s) for w, s in zip(token_counts, group_df["zipf_score"]) if pd.notna(s) and w > 0]
    total_w_zipf = sum(w for w, _ in zipf_pairs)
    zipf_wm      = sum(w * s for w, s in zipf_pairs) / total_w_zipf if total_w_zipf > 0 else np.nan

    # --- Token-count-weighted mean AoA ---
    aoa_pairs   = [(w, s) for w, s in zip(token_counts, group_df["aoa_score"]) if pd.notna(s) and w > 0]
    total_w_aoa = sum(w for w, _ in aoa_pairs)
    aoa_wm      = sum(w * s for w, s in aoa_pairs) / total_w_aoa if total_w_aoa > 0 else np.nan

    return {
        "n_utterances":       len(group_df),
        "n_speakers":         group_df["speaker_id"].nunique(),
        "corpus_token_count": n_corpus_tokens,
        "mattr_score":        corpus_mattr,
        "mtld_score":         corpus_mtld,
        "yules_k":            corpus_yules,
        "zipf_score":         zipf_wm,
        "aoa_score":          aoa_wm,
        "nawl_ratio":         corpus_nawl,
    }

# ----------------------------------------------------------------------------------------
# Main Aggregation Function
# ----------------------------------------------------------------------------------------
def aggregate_temporal_metrics(df, n_workers=1):
    '''Aggregate an utterance-level DataFrame into a (subreddit, year_month) panel by
    computing corpus-level lexical metrics for each cell.

    Each unique (subreddit, year_month) pair becomes one output row. The raw text of all
    utterances in that cell is concatenated into a monthly corpus before metrics are computed,
    ensuring that diversity measures are not artificially deflated by short posts and that
    the output panel is unaffected by month-to-month variation in posting volume or
    post-length distributions.

    All utterances are tokenized before the groupby loop. When n_workers > 1, tokenization
    is parallelised across workers using multiprocessing.Pool. On Linux (Adroit), forked
    workers inherit the parent's loaded NLTK models so there is no per-worker startup cost.
    With n_workers=8 this reduces wall time by approximately 6-7x versus single-threaded.

    Parameters
    ----------
    df : pd.DataFrame
        Utterance-level DataFrame. Must contain columns listed in NEEDED_COLS.
    n_workers : int
        Number of parallel worker processes for tokenization (default 1 = single-threaded).
        Set to the number of CPUs allocated by SLURM (--cpus-per-task).

    Returns
    -------
    pd.DataFrame
        Panel DataFrame with one row per (subreddit, year_month) and columns:
        subreddit, year_month, source_variation, n_utterances, n_speakers,
        corpus_token_count, mattr_score, mtld_score, yules_k, zipf_score,
        aoa_score, nawl_ratio.
    '''

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "raw_text", "subreddit"])
    df["year_month"] = df["timestamp"].dt.to_period("M")

    # --- Tokenize all utterances upfront; parallelise if n_workers > 1 ---
    # This moves the NLTK bottleneck out of the groupby loop so each group's
    # compute_corpus_metrics call only needs to flatten pre-computed token lists.
    texts = df["raw_text"].tolist()
    if n_workers > 1:
        print(f"Tokenizing {len(texts):,} utterances across {n_workers} workers...")
        with Pool(n_workers) as pool:
            token_lists = pool.map(_tokenize_utterance, texts)
    else:
        print(f"Tokenizing {len(texts):,} utterances (single-threaded)...")
        token_lists = [_tokenize_utterance(t) for t in texts]
    df["_tokens"] = token_lists

    groups   = df.groupby(["subreddit", "year_month"], sort=True)
    n_groups = len(groups)
    print(f"Aggregating {len(df):,} utterances into {n_groups} (subreddit, year_month) cells...")

    rows = []
    for i, ((subreddit, year_month), group_df) in enumerate(groups):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing group {i + 1}/{n_groups}: {subreddit} | {year_month}")

        metrics = compute_corpus_metrics(group_df)

        rows.append({
            "subreddit":        subreddit,
            "year_month":       str(year_month),
            "source_variation": group_df["source_variation"].iloc[0],
            **metrics,
        })

        del group_df
        gc.collect()

    print(f"Aggregation complete: {len(rows)} monthly corpus rows produced.")
    return pd.DataFrame(rows)
