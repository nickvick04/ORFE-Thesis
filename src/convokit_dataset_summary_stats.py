"""
convokit_dataset_summary_stats.py
==================================
Counts comments and unique speakers for each Convokit subreddit at every
stage of the processing pipeline, from raw JSONL through to the final
lexical_master.csv.

Nine corpora across three variation groups are reported:

  Age-Variation    : college, parent, teenagers
  Topic-Variation  : relationships, science, worldnews
  Culture-Variation: books, movies, religion

Four stages are reported per corpus
-------------------------------------
  Stage 0 — Raw JSONL
      Every valid JSON line in utterances.jsonl, regardless of speaker,
      body content, or timestamp.

  Stage 1 — After text filters
      Removes the same rows that corpus_longest_posts_batches_from_jsonl
      discards:
        • missing id / speaker / text / timestamp
        • text is [deleted] or [removed]
        • text matches bot-text patterns
        • text contains no Latin letter
        • unparseable timestamp
      (The Convokit pipeline has no author-name blocklist, unlike ArcticShift.)

  Stage 2 — After 1-per-(speaker, month) selection
      Row count = unique (speaker_id, year_month) pairs surviving Stage 1.
      Speaker count is unchanged from Stage 1.

  Stage 3 — Final lexical_master.csv
      Rows in the combined master CSV, read directly and grouped by the
      normalised subreddit label (e.g. "subreddit-college" → "college").

Usage
-----
    # Default Adroit paths
    python convokit_dataset_summary_stats.py

    # Override paths
    python convokit_dataset_summary_stats.py \\
        --data_dir /scratch/network/nv9344/Thesis/Thesis-Data/Convokit \\
        --csv      /scratch/network/nv9344/Thesis/Thesis-Data/lexical_master.csv

Author: Nicholas Vickery, Princeton ORFE '26
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ordered list of (variation_folder, corpus_folder) pairs.
# Variation folders must exist directly under --data_dir.
# Corpus folders must contain utterances.jsonl.
CORPORA: list[tuple[str, str]] = [
    ("Age-Variation",     "subreddit-college"),
    ("Age-Variation",     "subreddit-parent"),
    ("Age-Variation",     "subreddit-teenagers"),
    ("Topic-Variation",   "subreddit-relationships"),
    ("Topic-Variation",   "subreddit-science"),
    ("Topic-Variation",   "subreddit-worldnews"),
    ("Culture-Variation", "subreddit-books"),
    ("Culture-Variation", "subreddit-movies"),
    ("Culture-Variation", "subreddit-religion"),
]

# Display labels matching the normalised subreddit value in lexical_master.csv.
# The combine step stores e.g. "subreddit-college" in the subreddit column.
# Strip the "subreddit-" prefix to get the lookup key used in Stage 3.
def _csv_label(corpus_folder: str) -> str:
    """'subreddit-college' → 'college'"""
    return corpus_folder.split("-", 1)[-1]

DEFAULT_DATA_DIR = "/scratch/network/nv9344/Thesis/Thesis-Data/Convokit"
DEFAULT_CSV      = "/scratch/network/nv9344/Thesis/Thesis-Data/lexical_master.csv"

# Mirrors data_preprocessing.BOT_TEXT_PATTERNS
_BOT_TEXT_RE = re.compile(
    r"\bi am a bot\b"
    r"|\bthis (?:comment|post) was (?:posted|left by) a bot\b"
    r"|\bthis reply was generated automatically\b"
    r"|[\^*]*beep(?:\s+beep)?[\^*]*\s+[\^*]*boop(?:\s+boop)?[\^*]*",
    flags=re.IGNORECASE,
)

_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _extract_fields(obj: dict):
    """Extract (utt_id, speaker_id, text, timestamp) from a Convokit JSONL row.

    Mirrors data_preprocessing._extract_utterance_fields_json:
      speaker field: 'speaker' > 'user' > 'speaker_id', may be dict or str
      text field:    'text' > 'body' > 'raw_text'
    """
    utt_id   = obj.get("id", obj.get("utterance_id"))
    raw_text = obj.get("text", obj.get("body", obj.get("raw_text")))
    ts       = obj.get("timestamp")

    speaker = obj.get("speaker", obj.get("user", obj.get("speaker_id")))
    if isinstance(speaker, dict):
        speaker_id = speaker.get("id", speaker.get("speaker_id"))
    else:
        speaker_id = speaker

    return utt_id, speaker_id, raw_text, ts


def _is_valid_timestamp(ts) -> bool:
    try:
        datetime.fromtimestamp(int(ts))
        return True
    except (TypeError, ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Core: stream one utterances.jsonl and collect per-stage counts
# ---------------------------------------------------------------------------

def count_jsonl(jsonl_path: str) -> dict:
    """
    Stream *jsonl_path* once and return counts for Stages 0–2.

    Returns
    -------
    dict with keys:
        raw_comments, raw_speakers
        filtered_comments, filtered_speakers
        selected_comments, selected_speakers
        drop_missing_fields, drop_deleted_body, drop_bot_text,
        drop_no_letter, drop_bad_timestamp
    """
    raw_comments = 0
    raw_authors: set[str] = set()

    drop_missing_fields = 0
    drop_deleted_body   = 0
    drop_bot_text       = 0
    drop_no_letter      = 0
    drop_bad_timestamp  = 0

    filt_comments = 0
    filt_authors: set[str] = set()

    speaker_months: set[tuple[str, str]] = set()

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            utt_id, speaker_id, raw_text, ts = _extract_fields(obj)

            # ── Stage 0 ─────────────────────────────────────────────
            raw_comments += 1
            if speaker_id is not None:
                raw_authors.add(speaker_id)

            # ── Stage 1 filters ─────────────────────────────────────
            if not utt_id or not speaker_id or not raw_text or ts is None:
                drop_missing_fields += 1
                continue

            lower_text = str(raw_text).lower()
            if lower_text in {"[deleted]", "[removed]"}:
                drop_deleted_body += 1
                continue

            if _BOT_TEXT_RE.search(raw_text):
                drop_bot_text += 1
                continue

            if not _HAS_LETTER_RE.search(raw_text):
                drop_no_letter += 1
                continue

            if not _is_valid_timestamp(ts):
                drop_bad_timestamp += 1
                continue

            filt_comments += 1
            filt_authors.add(speaker_id)

            # ── Stage 2: unique (speaker, year_month) key ────────────
            try:
                dt = datetime.fromtimestamp(int(ts))
                ym = f"{dt.year}-{dt.month:02d}"
            except Exception:
                continue
            speaker_months.add((speaker_id, ym))

    return {
        "raw_comments":       raw_comments,
        "raw_speakers":       len(raw_authors),
        "filtered_comments":  filt_comments,
        "filtered_speakers":  len(filt_authors),
        "selected_comments":  len(speaker_months),
        "selected_speakers":  len(filt_authors),
        "drop_missing_fields":   drop_missing_fields,
        "drop_deleted_body":     drop_deleted_body,
        "drop_bot_text":         drop_bot_text,
        "drop_no_letter":        drop_no_letter,
        "drop_bad_timestamp":    drop_bad_timestamp,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

_NUM_W   = 13
_DELTA_W = 18


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{part / whole * 100:.1f}%"


def _delta_str(new: int, old: int) -> str:
    if old == 0:
        return "—"
    diff = new - old
    sign = "+" if diff >= 0 else "−"
    abs_diff = abs(diff)
    if abs_diff >= 1_000_000:
        magnitude = f"{abs_diff / 1_000_000:.2f}M"
    elif abs_diff >= 1_000:
        magnitude = f"{abs_diff / 1_000:.1f}K"
    else:
        magnitude = str(abs_diff)
    pct = abs_diff / old * 100
    return f"{sign}{magnitude} ({pct:.1f}%)"


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("Corpus",          22),
        ("Stage",           42),
        ("Comments",        _NUM_W),
        ("Δ Comments",      _DELTA_W),
        ("Speakers",        _NUM_W),
        ("Δ Speakers",      _DELTA_W),
    ]
    header = "".join(lbl.ljust(w) for lbl, w in cols)
    sep    = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    prev_variation = None
    for row in rows:
        if row["stage"] == "":
            print()
            continue

        # Print variation group header on first corpus of each group
        if row.get("variation") and row["variation"] != prev_variation:
            print(f"\n  [{row['variation']}]")
            prev_variation = row["variation"]

        label      = row["corpus"]
        stage_lbl  = row["stage"]
        comments   = row["comments"]
        speakers   = row["speakers"]
        d_comments = row.get("d_comments", "")
        d_speakers = row.get("d_speakers", "")

        print(
            f"{label:<22}"
            f"{stage_lbl:<42}"
            f"{comments:>{_NUM_W},}"
            f"{d_comments:>{_DELTA_W}}"
            f"{speakers:>{_NUM_W},}"
            f"{d_speakers:>{_DELTA_W}}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Before/after processing stats for Convokit subreddits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="Root Convokit directory containing variation sub-folders.")
    p.add_argument("--csv",      default=DEFAULT_CSV,
                   help="Path to lexical_master.csv (Stage 3).")
    p.add_argument("--verbose",  action="store_true",
                   help="Print per-filter drop counts for each corpus.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Convokit Dataset Summary Statistics")
    print("=" * 70)
    print(f"\nData directory : {args.data_dir}")
    print(f"Master CSV     : {args.csv}")

    # ── Stage 3: load lexical_master.csv ──────────────────────────────
    print("\nLoading lexical_master.csv … (this may take a few minutes)", flush=True)
    if not os.path.isfile(args.csv):
        sys.exit(f"\nERROR: lexical_master.csv not found at {args.csv}")

    csv_df = pd.read_csv(args.csv, usecols=["speaker_id", "subreddit"],
                         low_memory=False)

    # Normalise: "subreddit-college" → "college"
    csv_df["subreddit_norm"] = (
        csv_df["subreddit"].astype(str).str.strip().str.split("-", n=1).str[-1]
    )
    csv_stats: dict[str, dict] = {}
    for _, corpus_folder in CORPORA:
        label = _csv_label(corpus_folder)
        sub_df = csv_df[csv_df["subreddit_norm"] == label]
        csv_stats[label] = {
            "comments": len(sub_df),
            "speakers": sub_df["speaker_id"].nunique(),
        }
    del csv_df

    # ── Process each JSONL ────────────────────────────────────────────
    all_rows: list[dict] = []
    totals:   dict[str, dict] = {}

    for variation, corpus_folder in CORPORA:
        jsonl_path = os.path.join(args.data_dir, variation, corpus_folder,
                                  "utterances.jsonl")
        label = _csv_label(corpus_folder)

        print(f"\n[{variation} / {corpus_folder}] ", end="", flush=True)

        if not os.path.isfile(jsonl_path):
            print(f"WARNING: utterances.jsonl not found at {jsonl_path} — skipping.")
            continue

        size_gb = os.path.getsize(jsonl_path) / 1e9
        print(f"Streaming utterances.jsonl ({size_gb:.2f} GB) …", flush=True)

        counts = count_jsonl(jsonl_path)
        csv_c  = csv_stats.get(label, {"comments": 0, "speakers": 0})
        totals[label] = {"jsonl": counts, "csv": csv_c, "variation": variation}

        if args.verbose:
            print(f"  Filter breakdown:")
            print(f"    Missing required fields  : {counts['drop_missing_fields']:>10,}")
            print(f"    Deleted/removed body     : {counts['drop_deleted_body']:>10,}")
            print(f"    Bot-text pattern         : {counts['drop_bot_text']:>10,}")
            print(f"    No Latin letter in body  : {counts['drop_no_letter']:>10,}")
            print(f"    Unparseable timestamp    : {counts['drop_bad_timestamp']:>10,}")

        stages = [
            {
                "variation":  variation,
                "corpus":     label,
                "stage":      "Stage 0 — Raw JSONL",
                "comments":   counts["raw_comments"],
                "speakers":   counts["raw_speakers"],
                "d_comments": "",
                "d_speakers": "",
            },
            {
                "variation":  variation,
                "corpus":     "",
                "stage":      "Stage 1 — After text filters",
                "comments":   counts["filtered_comments"],
                "speakers":   counts["filtered_speakers"],
                "d_comments": _delta_str(counts["filtered_comments"], counts["raw_comments"]),
                "d_speakers": _delta_str(counts["filtered_speakers"], counts["raw_speakers"]),
            },
            {
                "variation":  variation,
                "corpus":     "",
                "stage":      "Stage 2 — After 1-per-speaker-month",
                "comments":   counts["selected_comments"],
                "speakers":   counts["selected_speakers"],
                "d_comments": _delta_str(counts["selected_comments"], counts["filtered_comments"]),
                "d_speakers": "—",
            },
            {
                "variation":  variation,
                "corpus":     "",
                "stage":      "Stage 3 — Final lexical_master.csv",
                "comments":   csv_c["comments"],
                "speakers":   csv_c["speakers"],
                "d_comments": _delta_str(csv_c["comments"],  counts["selected_comments"]),
                "d_speakers": _delta_str(csv_c["speakers"],  counts["selected_speakers"]),
            },
        ]
        all_rows.extend(stages)
        all_rows.append({"variation": "", "corpus": "", "stage": "",
                         "comments": 0, "speakers": 0})

    # Remove trailing blank
    while all_rows and all_rows[-1]["stage"] == "":
        all_rows.pop()

    _print_table(all_rows)

    # ── Grand totals ─────────────────────────────────────────────────
    if totals:
        print("Grand totals (all corpora combined)")
        print("─" * 60)

        def _sum(key1, key2):
            return sum(v[key1][key2] for v in totals.values()
                       if key1 in v and key2 in v[key1])

        grand = {
            "raw_c":  _sum("jsonl", "raw_comments"),
            "raw_s":  _sum("jsonl", "raw_speakers"),
            "filt_c": _sum("jsonl", "filtered_comments"),
            "filt_s": _sum("jsonl", "filtered_speakers"),
            "sel_c":  _sum("jsonl", "selected_comments"),
            "sel_s":  _sum("jsonl", "selected_speakers"),
            "csv_c":  _sum("csv",   "comments"),
            "csv_s":  _sum("csv",   "speakers"),
        }

        fmt = lambda lbl, c, s: print(
            f"  {lbl:<45} {c:>12,} comments   {s:>12,} speakers"
        )
        fmt("Stage 0 — Raw JSONL",               grand["raw_c"],  grand["raw_s"])
        fmt("Stage 1 — After text filters",       grand["filt_c"], grand["filt_s"])
        fmt("Stage 2 — After 1-per-speaker-month",grand["sel_c"],  grand["sel_s"])
        fmt("Stage 3 — Final lexical_master.csv", grand["csv_c"],  grand["csv_s"])
        print()
        print(f"  Overall retention (Stage 0 → Stage 3): "
              f"{_pct(grand['csv_c'], grand['raw_c'])} of raw comments, "
              f"{_pct(grand['csv_s'], grand['raw_s'])} of raw speakers")
        print()


if __name__ == "__main__":
    main()
