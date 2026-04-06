"""
dataset_summary_stats.py
========================
Counts comments and unique speakers for each ArcticShift subreddit at every
stage of the processing pipeline, from raw JSONL through to the final
regression-ready CSV.

Four stages are reported
------------------------
  Stage 0 — Raw JSONL
      Every valid JSON line in the *_comments.jsonl file, regardless of
      author, body content, or timestamp.

  Stage 1 — After text & author filters
      Removes the same rows that arcticshift_pipeline.py discards:
        • author is [deleted] or AutoModerator
        • body is [deleted] or [removed]
        • body matches bot-text patterns
        • body contains no Latin letter
        • missing id / author / body / timestamp, or unparseable timestamp

  Stage 2 — After 1-per-(speaker, month) selection
      Mirrors the pipeline's "longest post per speaker per calendar month"
      step.  Row count = number of unique (speaker_id, year_month) pairs
      surviving Stage 1; speaker count is unchanged from Stage 1 (every
      eligible speaker contributes at least one selected post).

  Stage 3 — Final combined CSV  (lexical_df_combined.csv)
      Rows written by combine_lexical_csvs.py after the additional
      bot_usernames.csv filter.  Read directly from the CSV.

Usage
-----
# Default Adroit paths
python dataset_summary_stats.py

# Override any path
python dataset_summary_stats.py \\
    --data_dir /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift \\
    --csv      /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv \\
    --bots     /scratch/network/nv9344/Thesis/ORFE-Thesis/src/bot_usernames.csv

Author: Nicholas Vickery, Princeton ORFE '26
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maps ArcticShift folder name → subreddit label used in lexical_df_combined.csv
SUBREDDITS: dict[str, str] = {
    "College":     "college",
    "High School": "highschool",
    "Parenting":   "Parenting",
    "Retirement":  "retirement",
}

DEFAULT_DATA_DIR = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift"
DEFAULT_CSV      = os.path.join(DEFAULT_DATA_DIR, "lexical_df_combined.csv")
DEFAULT_BOTS_CSV = "/scratch/network/nv9344/Thesis/ORFE-Thesis/src/bot_usernames.csv"

# Mirrors arcticshift_pipeline._EXCLUDED_AUTHORS
_EXCLUDED_AUTHORS: set[str] = {"[deleted]", "AutoModerator"}

# Mirrors data_preprocessing.BOT_TEXT_PATTERNS
_BOT_TEXT_RE = re.compile(
    r"\bi am a bot\b"
    r"|\bthis (?:comment|post) was (?:posted|left by) a bot\b"
    r"|\bthis reply was generated automatically\b"
    r"|[\^*]*beep(?:\s+beep)?[\^*]*\s+[\^*]*boop(?:\s+boop)?[\^*]*",
    flags=re.IGNORECASE,
)

# Mirrors data_preprocessing.HAS_LETTER_RE
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_jsonl(folder: str) -> str | None:
    """Return the path of the single *_comments.jsonl file in `folder`."""
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        return None
    for fname in entries:
        if fname.endswith("_comments.jsonl"):
            return os.path.join(folder, fname)
    return None


def _load_bot_usernames(csv_path: str) -> set[str]:
    """Load bot usernames from bot_usernames.csv; return empty set on failure."""
    if not os.path.isfile(csv_path):
        print(f"  [warn] bot_usernames.csv not found at {csv_path}; Stage 3 "
              "speaker/comment counts will not reflect bot removal.",
              file=sys.stderr)
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["username"], dtype=str)
        return set(df["username"].dropna().str.strip())
    except Exception as exc:
        print(f"  [warn] Could not read bot_usernames.csv: {exc}", file=sys.stderr)
        return set()


def _is_valid_timestamp(ts) -> bool:
    """Return True if ts can be interpreted as a Unix timestamp."""
    try:
        datetime.fromtimestamp(int(ts))
        return True
    except (TypeError, ValueError, OSError):
        return False


# ---------------------------------------------------------------------------
# Core: stream a JSONL file and collect per-stage counts
# ---------------------------------------------------------------------------

def count_jsonl(jsonl_path: str) -> dict:
    """
    Stream *jsonl_path* once and return a dict with counts for Stages 0–2.

    Keys returned
    -------------
    raw_comments        : int   — Stage 0 comments
    raw_speakers        : int   — Stage 0 unique speakers
    filtered_comments   : int   — Stage 1 comments (after all text/author filters)
    filtered_speakers   : int   — Stage 1 unique speakers
    selected_comments   : int   — Stage 2 comments (1 per speaker-month)
    selected_speakers   : int   — Stage 2 unique speakers (== filtered_speakers)

    drop_*              : int   — rows dropped at each named filter step
    """
    # Stage 0 accumulators
    raw_comments  = 0
    raw_authors:  set[str] = set()

    # Filter-step counters (for diagnostics)
    drop_missing_fields = 0
    drop_excluded_author = 0
    drop_deleted_body    = 0
    drop_bot_text        = 0
    drop_no_letter       = 0
    drop_bad_timestamp   = 0

    # Stage 1 accumulators
    filt_comments = 0
    filt_authors: set[str] = set()

    # Stage 2: track unique (speaker, year_month) pairs
    speaker_months: set[tuple[str, str]] = set()

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # malformed JSON — skip silently

            author = obj.get("author")
            body   = obj.get("body")
            utt_id = obj.get("id")
            ts     = obj.get("created_utc")

            # ── Stage 0 ────────────────────────────────────────────────
            raw_comments += 1
            if author is not None:
                raw_authors.add(author)

            # ── Stage 1 filters ────────────────────────────────────────
            if not utt_id or not author or not body or ts is None:
                drop_missing_fields += 1
                continue

            if author in _EXCLUDED_AUTHORS:
                drop_excluded_author += 1
                continue

            lower_body = body.lower()
            if lower_body in {"[deleted]", "[removed]"}:
                drop_deleted_body += 1
                continue

            if _BOT_TEXT_RE.search(body):
                drop_bot_text += 1
                continue

            if not _HAS_LETTER_RE.search(body):
                drop_no_letter += 1
                continue

            if not _is_valid_timestamp(ts):
                drop_bad_timestamp += 1
                continue

            filt_comments += 1
            filt_authors.add(author)

            # ── Stage 2: unique (speaker, year_month) key ──────────────
            try:
                dt = datetime.fromtimestamp(int(ts))
                ym = f"{dt.year}-{dt.month:02d}"
            except Exception:
                continue
            speaker_months.add((author, ym))

    return {
        # Stage 0
        "raw_comments":       raw_comments,
        "raw_speakers":       len(raw_authors),
        # Stage 1
        "filtered_comments":  filt_comments,
        "filtered_speakers":  len(filt_authors),
        # Stage 2
        "selected_comments":  len(speaker_months),
        "selected_speakers":  len(filt_authors),  # same people, fewer rows
        # Drop breakdown
        "drop_missing_fields":    drop_missing_fields,
        "drop_excluded_author":   drop_excluded_author,
        "drop_deleted_body":      drop_deleted_body,
        "drop_bot_text":          drop_bot_text,
        "drop_no_letter":         drop_no_letter,
        "drop_bad_timestamp":     drop_bad_timestamp,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

_NUM_W   = 13   # width of comment/speaker count columns
_DELTA_W = 18   # width of delta columns


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{part / whole * 100:.1f}%"


def _delta_str(new: int, old: int) -> str:
    """Compact delta: e.g. '−3.7M (77.1%)' or '+12K (1.4%)'."""
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
    pct = abs(diff) / old * 100
    return f"{sign}{magnitude} ({pct:.1f}%)"


def _print_table(rows: list[dict]) -> None:
    """Print the four-stage summary table."""
    cols = [
        ("Subreddit",   16),
        ("Stage",       42),
        ("Comments",    _NUM_W),
        ("Δ Comments",  _DELTA_W),
        ("Speakers",    _NUM_W),
        ("Δ Speakers",  _DELTA_W),
    ]
    header = "".join(lbl.ljust(w) for lbl, w in cols)
    sep    = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        if row["stage"] == "":   # blank separator row
            print()
            continue

        sub        = row["subreddit"]
        stage_lbl  = row["stage"]
        comments   = row["comments"]
        speakers   = row["speakers"]
        d_comments = row.get("d_comments", "")
        d_speakers = row.get("d_speakers", "")

        line = (
            f"{sub:<16}"
            f"{stage_lbl:<42}"
            f"{comments:>{_NUM_W},}"
            f"{d_comments:>{_DELTA_W}}"
            f"{speakers:>{_NUM_W},}"
            f"{d_speakers:>{_DELTA_W}}"
        )
        print(line)

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Before/after processing stats for ArcticShift subreddits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing subreddit folders (each with a *_comments.jsonl).",
    )
    p.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to lexical_df_combined.csv (Stage 3 / after processing).",
    )
    p.add_argument(
        "--bots",
        default=DEFAULT_BOTS_CSV,
        help="Path to bot_usernames.csv used by combine_lexical_csvs.py.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-filter drop counts for each subreddit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("ArcticShift Dataset Summary Statistics")
    print("=" * 70)
    print(f"\nData directory : {args.data_dir}")
    print(f"Combined CSV   : {args.csv}")
    print(f"Bot list       : {args.bots}")

    # ── Stage 3: load combined CSV ─────────────────────────────────────
    print("\nLoading lexical_df_combined.csv …", flush=True)
    if not os.path.isfile(args.csv):
        sys.exit(f"\nERROR: Combined CSV not found at {args.csv}")

    csv_df = pd.read_csv(args.csv, usecols=["subreddit", "speaker_id"],
                         low_memory=False)
    csv_stats: dict[str, dict] = {}
    for folder_name, csv_label in SUBREDDITS.items():
        sub_df = csv_df[csv_df["subreddit"] == csv_label]
        csv_stats[folder_name] = {
            "comments": len(sub_df),
            "speakers": sub_df["speaker_id"].nunique(),
        }
    del csv_df  # free memory

    # ── Process each subreddit's JSONL ────────────────────────────────
    all_rows: list[dict] = []   # rows for the summary table
    totals:   dict[str, dict] = {}

    for folder_name in SUBREDDITS:
        folder_path = os.path.join(args.data_dir, folder_name)
        jsonl_path  = _find_jsonl(folder_path)

        print(f"\n[{folder_name}] ", end="", flush=True)

        if jsonl_path is None:
            print(f"WARNING: no *_comments.jsonl found in {folder_path} — skipping.")
            continue

        file_size_gb = os.path.getsize(jsonl_path) / 1e9
        print(f"Streaming {os.path.basename(jsonl_path)} ({file_size_gb:.2f} GB) …",
              flush=True)

        counts = count_jsonl(jsonl_path)
        csv_c  = csv_stats[folder_name]
        totals[folder_name] = {"jsonl": counts, "csv": csv_c}

        if args.verbose:
            print(f"  Filter breakdown:")
            print(f"    Missing required fields  : {counts['drop_missing_fields']:>10,}")
            print(f"    Excluded author          : {counts['drop_excluded_author']:>10,}")
            print(f"    Deleted/removed body     : {counts['drop_deleted_body']:>10,}")
            print(f"    Bot-text pattern         : {counts['drop_bot_text']:>10,}")
            print(f"    No Latin letter in body  : {counts['drop_no_letter']:>10,}")
            print(f"    Unparseable timestamp    : {counts['drop_bad_timestamp']:>10,}")

        stages = [
            {
                "subreddit":  folder_name,
                "stage":      "Stage 0 — Raw JSONL",
                "comments":   counts["raw_comments"],
                "speakers":   counts["raw_speakers"],
                "d_comments": "",
                "d_speakers": "",
            },
            {
                "subreddit":  "",
                "stage":      "Stage 1 — After text/author filters",
                "comments":   counts["filtered_comments"],
                "speakers":   counts["filtered_speakers"],
                "d_comments": _delta_str(counts["filtered_comments"], counts["raw_comments"]),
                "d_speakers": _delta_str(counts["filtered_speakers"], counts["raw_speakers"]),
            },
            {
                "subreddit":  "",
                "stage":      "Stage 2 — After 1-per-speaker-month",
                "comments":   counts["selected_comments"],
                "speakers":   counts["selected_speakers"],
                "d_comments": _delta_str(counts["selected_comments"], counts["filtered_comments"]),
                "d_speakers": "—",
            },
            {
                "subreddit":  "",
                "stage":      "Stage 3 — Final CSV (after bot filter)",
                "comments":   csv_c["comments"],
                "speakers":   csv_c["speakers"],
                "d_comments": _delta_str(csv_c["comments"],  counts["selected_comments"]),
                "d_speakers": _delta_str(csv_c["speakers"],  counts["selected_speakers"]),
            },
        ]
        all_rows.extend(stages)
        # blank separator row between subreddits
        all_rows.append({"subreddit": "", "stage": "", "comments": 0,
                         "speakers": 0, "d_comments": "", "d_speakers": ""})

    # ── Summary table ─────────────────────────────────────────────────
    # Remove trailing blank separator
    while all_rows and all_rows[-1]["stage"] == "":
        all_rows.pop()

    _print_table(all_rows)

    # ── Grand totals across all four subreddits ────────────────────────
    if totals:
        print("Grand totals (all subreddits combined)")
        print("─" * 60)
        def _sum(key1, key2):
            return sum(v[key1][key2] for v in totals.values()
                       if key1 in v and key2 in v[key1])

        grand_raw_c  = _sum("jsonl", "raw_comments")
        grand_raw_s  = _sum("jsonl", "raw_speakers")
        grand_filt_c = _sum("jsonl", "filtered_comments")
        grand_filt_s = _sum("jsonl", "filtered_speakers")
        grand_sel_c  = _sum("jsonl", "selected_comments")
        grand_sel_s  = _sum("jsonl", "selected_speakers")
        grand_csv_c  = _sum("csv",   "comments")
        grand_csv_s  = _sum("csv",   "speakers")

        fmt = lambda label, c, s: print(
            f"  {label:<42} {c:>12,} comments   {s:>12,} speakers"
        )
        fmt("Stage 0 — Raw JSONL",                    grand_raw_c,  grand_raw_s)
        fmt("Stage 1 — After text/author filters",    grand_filt_c, grand_filt_s)
        fmt("Stage 2 — After 1-per-speaker-month",    grand_sel_c,  grand_sel_s)
        fmt("Stage 3 — Final CSV (after bot filter)", grand_csv_c,  grand_csv_s)
        print()
        print(f"  Overall retention (Stage 0 → Stage 3): "
              f"{_pct(grand_csv_c, grand_raw_c)} of raw comments, "
              f"{_pct(grand_csv_s, grand_raw_s)} of raw speakers")
        print()


if __name__ == "__main__":
    main()
