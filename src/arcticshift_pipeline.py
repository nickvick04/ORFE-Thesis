# ----------------------------------------------------------------------------------------
# ArcticShift lexical pipeline — mirrors run_lexical_pipeline_cnvkt_batches() but reads
# ArcticShift native JSONL format (author / body / created_utc / parent_id) instead of
# the Convokit utterances.jsonl schema (speaker / text / timestamp / reply_to).
#
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import io
import os
import gc
import json
import pandas as pd
from datetime import datetime

import zstandard as zstd

# Shared helpers from the existing preprocessing module.
# BOT_TEXT_RE / HAS_LETTER_RE are the same compiled regexes used by the Convokit path.
# _speaker_shard_index / _compute_post_depth are pure-logic helpers with no
# format-specific assumptions, so we reuse them directly.
from data_preprocessing import (
    BOT_TEXT_RE,
    HAS_LETTER_RE,
    _speaker_shard_index,
    _compute_post_depth,
)
from lexical_analysis_functions import compute_lexical_vals

BATCH_SIZE = 1000

# Authors that ArcticShift never anonymises but that should always be excluded.
# "[deleted]" appears on removed accounts; "AutoModerator" is a universal Reddit bot.
_EXCLUDED_AUTHORS = {"[deleted]", "AutoModerator"}


# ----------------------------------------------------------------------------------------
# ArcticShift-specific field extractors
# ----------------------------------------------------------------------------------------

def _find_arcticshift_jsonl(corpus_dir: str) -> str:
    """Return the path of the single *_comments.jsonl(.zst) file inside corpus_dir.

    Accepts both the decompressed r_<subreddit>_comments.jsonl and the
    compressed r_<subreddit>_comments.jsonl.zst (or bare _comments.zst).
    The .jsonl form is preferred when both exist.  Raises FileNotFoundError
    if neither is found.
    """
    jsonl_path = None
    zst_path   = None
    for fname in os.listdir(corpus_dir):
        full = os.path.join(corpus_dir, fname)
        if fname.endswith("_comments.jsonl"):
            jsonl_path = full
        elif fname.endswith("_comments.jsonl.zst") or fname.endswith("_comments.zst"):
            zst_path = full
    if jsonl_path is not None:
        return jsonl_path
    if zst_path is not None:
        return zst_path
    raise FileNotFoundError(
        f"No *_comments.jsonl or *_comments.zst file found in {corpus_dir}. "
        "Expected a file matching r_<subreddit>_comments.jsonl(.zst)."
    )


def _open_jsonl(path: str):
    """Return a text-mode file-like object for a .jsonl or .jsonl.zst/.zst file.

    For compressed files the zstandard streaming decompressor is used so the
    file is never fully materialised on disk.  The caller is responsible for
    closing the returned object (use as a context manager).
    """
    if path.endswith(".zst"):
        fh  = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _extract_arcticshift_fields(obj: dict):
    """Extract the four core utterance fields from an ArcticShift JSONL row.

    Returns
    -------
    (utt_id, speaker_id, raw_text, timestamp)
        utt_id     : bare comment ID string  (obj["id"])
        speaker_id : username string         (obj["author"])
        raw_text   : comment body string     (obj["body"])
        timestamp  : integer Unix epoch      (obj["created_utc"])
    Any field absent from obj is returned as None.
    """
    utt_id     = obj.get("id")
    speaker_id = obj.get("author")
    raw_text   = obj.get("body")
    timestamp  = obj.get("created_utc")
    return utt_id, speaker_id, raw_text, timestamp


def _extract_arcticshift_parent_id(obj: dict):
    """Return the normalised parent comment ID from an ArcticShift JSONL row.

    ArcticShift stores parent_id with a Reddit type prefix:
      t1_<id>  →  parent is a comment   (keep, strip prefix)
      t3_<id>  →  parent is a submission (treat as top-level, return None)

    Any other prefix (t2_, t4_, t5_) is treated as top-level for safety.
    """
    raw = obj.get("parent_id")
    if not raw:
        return None
    if "_" in raw:
        prefix, bare_id = raw.split("_", 1)
        if prefix == "t1":
            return bare_id   # comment → comment reply chain
        return None          # t3 submission parent → comment is top-level
    return raw               # no prefix: return as-is


# ----------------------------------------------------------------------------------------
# Two-pass streaming batch generator
# ----------------------------------------------------------------------------------------

def arcticshift_longest_posts_batches(
    corpus_dir: str,
    batch_size: int = BATCH_SIZE,
    num_shards: int = 1,
    shard_index: int = 0,
):
    """Stream an ArcticShift *_comments.jsonl file, keep the longest valid post
    per speaker per (year, month), and yield the selected rows as pd.DataFrame
    batches.

    Mirrors corpus_longest_posts_batches_from_jsonl() in data_preprocessing.py
    with three adaptations for ArcticShift data:

      1. Field mapping  — reads author/body/created_utc/parent_id instead of
                          speaker/text/timestamp/reply_to.
      2. Author filter  — drops "[deleted]" and "AutoModerator" by username
                          before any text-level checks, because ArcticShift
                          does not pre-clean these accounts.
      3. Parent IDs     — strips the Reddit type prefix (t1_) and ignores
                          submission-level parents (t3_) so that top-level
                          comments receive depth 0 and reply depth is computed
                          correctly within the comment tree.

    The output DataFrame has one extra column vs. the Convokit path:
      subreddit (str) — the subreddit name, available natively in ArcticShift.

    Parameters
    ----------
    corpus_dir  : Path to the ArcticShift subreddit folder containing
                  r_<subreddit>_comments.jsonl.
    batch_size  : Rows per yielded DataFrame (default 1000).
    num_shards  : Total number of parallel shard jobs (for SLURM arrays).
    shard_index : Zero-based index of this job's shard.

    Yields
    ------
    pd.DataFrame with columns:
        utterance_id, speaker_id, raw_text, timestamp, subreddit,
        num_utterances_by_speaker, num_utterances_by_speaker_month,
        post_depth, score, num_direct_replies,
        controversiality, edited
    """
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_index < num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    utt_path = _find_arcticshift_jsonl(corpus_dir)
    print(f"Streaming ArcticShift file: {utt_path}")

    # ------------------------------------------------------------------
    # Dictionaries populated in Pass 1
    # ------------------------------------------------------------------
    best_by_speaker_month   = {}   # (speaker_id, (year, month)) → metadata dict
    counts_by_speaker       = {}   # speaker_id → all-time post count
    counts_by_speaker_month = {}   # (speaker_id, (year, month)) → monthly count
    parent_by_utt           = {}   # utt_id → normalised parent utt_id or None
    score_by_utt              = {}   # utt_id → Reddit score int or None
    direct_reply_counts       = {}   # utt_id → number of direct replies
    subreddit_by_utt          = {}   # utt_id → subreddit string
    controversiality_by_utt   = {}   # utt_id → controversiality flag (0 or 1)
    edited_by_utt             = {}   # utt_id → False, or Unix timestamp of edit

    # ------------------------------------------------------------------
    # Pass 1 — scan the file to build the selection index
    # ------------------------------------------------------------------
    with _open_jsonl(utt_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            utt_id, speaker_id, raw_text, timestamp = _extract_arcticshift_fields(obj)
            parent_id       = _extract_arcticshift_parent_id(obj)
            score           = obj.get("score")
            subreddit       = obj.get("subreddit")
            controversiality = obj.get("controversiality")
            edited          = obj.get("edited")

            # Populate per-utterance lookup tables even for posts we will
            # later filter out, so that depth and reply-count calculations
            # can still traverse the full parent chain.
            if utt_id is not None:
                score_by_utt[utt_id]            = score
                parent_by_utt[utt_id]           = parent_id
                subreddit_by_utt[utt_id]        = subreddit
                controversiality_by_utt[utt_id] = controversiality
                edited_by_utt[utt_id]           = edited
                if parent_id is not None:
                    direct_reply_counts[parent_id] = (
                        direct_reply_counts.get(parent_id, 0) + 1
                    )

            # Skip rows missing any required field
            if utt_id is None or speaker_id is None or not raw_text or timestamp is None:
                continue

            # --- Author-level filters (ArcticShift-specific) ---
            if speaker_id in _EXCLUDED_AUTHORS:
                continue

            # --- Text-level filters (identical to Convokit path) ---
            lower_text = raw_text.lower()
            if lower_text in {"[deleted]", "[removed]"}:
                continue
            if BOT_TEXT_RE.search(raw_text):
                continue
            if not HAS_LETTER_RE.search(raw_text):
                continue

            # --- Timestamp validation ---
            try:
                dt = datetime.fromtimestamp(int(timestamp))
            except (TypeError, ValueError, OSError):
                continue

            year_month        = (dt.year, dt.month)
            speaker_month_key = (speaker_id, year_month)

            # --- Shard assignment: each speaker belongs to exactly one shard ---
            if _speaker_shard_index(speaker_id, num_shards) != shard_index:
                continue

            # --- Update speaker and speaker-month counts ---
            counts_by_speaker[speaker_id] = (
                counts_by_speaker.get(speaker_id, 0) + 1
            )
            counts_by_speaker_month[speaker_month_key] = (
                counts_by_speaker_month.get(speaker_month_key, 0) + 1
            )

            # --- Keep the longest post for this speaker-month ---
            post_length = len(raw_text)
            prev = best_by_speaker_month.get(speaker_month_key)
            if prev is None or post_length > prev["post_length"]:
                best_by_speaker_month[speaker_month_key] = {
                    "utterance_id": utt_id,
                    "timestamp":    dt,
                    "post_length":  post_length,
                }

    # Invert the selection map for O(1) lookup in Pass 2
    selected_utterance_to_key = {
        row["utterance_id"]: speaker_month_key
        for speaker_month_key, row in best_by_speaker_month.items()
    }
    depth_cache = {}
    print(
        f"Selected {len(best_by_speaker_month):,} speaker-month pairs "
        f"({len(selected_utterance_to_key):,} utterances) from {utt_path}"
    )

    # ------------------------------------------------------------------
    # Pass 2 — re-stream the file and emit only the selected utterances
    # ------------------------------------------------------------------
    rows         = []
    emitted_rows = 0

    with _open_jsonl(utt_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            utt_id, _, raw_text, _ = _extract_arcticshift_fields(obj)
            speaker_month_key = selected_utterance_to_key.get(utt_id)
            if speaker_month_key is None:
                continue

            speaker_id, _ = speaker_month_key
            rows.append({
                "utterance_id": utt_id,
                "speaker_id":   speaker_id,
                "raw_text":     raw_text,
                "timestamp":    best_by_speaker_month[speaker_month_key]["timestamp"],
                "subreddit":    subreddit_by_utt.get(utt_id),
                # --- enrichment columns ---
                "num_utterances_by_speaker":       counts_by_speaker[speaker_id],
                "num_utterances_by_speaker_month": counts_by_speaker_month[speaker_month_key],
                "post_depth":       _compute_post_depth(
                                        utt_id, parent_by_utt, depth_cache, set()
                                    ),
                "score":             score_by_utt.get(utt_id),
                "num_direct_replies": direct_reply_counts.get(utt_id, 0),
                "controversiality":  controversiality_by_utt.get(utt_id),
                "edited":            edited_by_utt.get(utt_id),
            })
            emitted_rows += 1

            if len(rows) >= batch_size:
                yield pd.DataFrame(rows)
                rows = []

    if rows:
        yield pd.DataFrame(rows)

    if emitted_rows == 0:
        raise RuntimeError(
            f"No rows emitted from {utt_path}. "
            "Check that author/body/created_utc fields are present and that "
            "filters are not excluding every row."
        )


# ----------------------------------------------------------------------------------------
# Pipeline entry point
# ----------------------------------------------------------------------------------------

def run_lexical_pipeline_arcticshift_batches(
    corpus_dir: str,
    batch_size: int = BATCH_SIZE,
    num_shards: int = 1,
    shard_index: int = 0,
):
    """Run the lexical-only pipeline on a single ArcticShift subreddit corpus.

    Streams r_<subreddit>_comments.jsonl from corpus_dir, selects the longest
    valid post per speaker per (year, month), computes six lexical metrics
    (MTLD, MATTR, Yule's K, Zipf, AoA, NAWL) in batches, and writes a single
    CSV file alongside the corpus directory.

    The output schema matches the Convokit lexical pipeline CSV exactly, with
    one additional column:
        subreddit (str) — available natively in ArcticShift exports.

    Output path
    -----------
    Single shard : <parent_of_corpus_dir>/<corpus_name>_lexical_df.csv
    Multi-shard  : <parent_of_corpus_dir>/<corpus_name>_lexical_df_shard-NNN-of-MMM.csv

    Parameters
    ----------
    corpus_dir  : Path to the ArcticShift subreddit folder (must contain
                  r_<subreddit>_comments.jsonl).
    batch_size  : Rows per batch — lower saves RAM at the cost of more I/O.
    num_shards  : Total parallel shard jobs (for SLURM arrays).
    shard_index : Zero-based index of this job's shard.
    """
    corpus_name = os.path.basename(corpus_dir)
    print(f"Processing ArcticShift corpus (lexical only): {corpus_name}")

    output_dir = os.path.dirname(corpus_dir)
    if num_shards == 1:
        output_name = f"{corpus_name}_lexical_df.csv"
    else:
        output_name = (
            f"{corpus_name}_lexical_df"
            f"_shard-{shard_index:03d}-of-{num_shards:03d}.csv"
        )
    output_path = os.path.join(output_dir, output_name)

    first_batch = True
    i = 0
    print(f"Processing ArcticShift lexical batches for: {corpus_name}")
    if num_shards > 1:
        print(f"Shard {shard_index + 1}/{num_shards}")

    for df_batch in arcticshift_longest_posts_batches(
        corpus_dir,
        batch_size=batch_size,
        num_shards=num_shards,
        shard_index=shard_index,
    ):
        print(f"Currently processing lexical batch: {i}")
        df_batch = compute_lexical_vals(df_batch)

        # First batch creates the file; subsequent batches append without re-writing headers.
        df_batch.to_csv(
            output_path,
            mode="w" if first_batch else "a",
            header=first_batch,
            index=False,
        )

        first_batch = False
        i += 1
        del df_batch
        gc.collect()

    print(f"Done — {i} batch(es) written to: {output_path}")
