"""
teenagers_summary_stats.py
==========================
Reports the same four-stage comment / unique-speaker counts as
dataset_summary_stats.py, but for r/teenagers whose raw data is split
across multiple .zst files rather than a single *_comments.jsonl.

  r_teenagers_comments_2015-2023.zst
  r_teenagers_comments_2024.zst
  teenagers_comments_2025.zst   (note: no r_ prefix)

All files are streamed with a single shared accumulator so unique-author
and unique-(speaker, year_month) counts are correct across the full
dataset — no double-counting.

Stages reported
---------------
  Stage 0 — Raw: every valid JSON line across all .zst files.
  Stage 1 — After text/author filters (mirrors arcticshift_pipeline.py).
  Stage 2 — After 1-per-(speaker, month) selection.
  Stage 3 — Rows in Teenagers_lexical_df.csv after bot filter.

Usage
-----
    # Default Adroit paths
    python teenagers_summary_stats.py

    # Override input directory and/or CSV path
    python teenagers_summary_stats.py \\
        --data_dir /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/Teenagers \\
        --csv      /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/Teenagers_lexical_df.csv \\
        --bots     /scratch/network/nv9344/Thesis/ORFE-Thesis/src/bot_usernames.csv

Author: Nicholas Vickery, Princeton ORFE '26
"""

import argparse
import glob
import io
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import zstandard as zstd

# ---------------------------------------------------------------------------
# Configuration — default Adroit paths
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/Teenagers"
DEFAULT_CSV      = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/Teenagers_lexical_df.csv"
DEFAULT_BOTS_CSV = "/scratch/network/nv9344/Thesis/ORFE-Thesis/src/bot_usernames.csv"

# ---------------------------------------------------------------------------
# Filters — exact mirrors of dataset_summary_stats.py / arcticshift_pipeline.py
# ---------------------------------------------------------------------------

_EXCLUDED_AUTHORS: set[str] = {"[deleted]", "AutoModerator"}

_BOT_TEXT_RE = re.compile(
    r"\bi am a bot\b"
    r"|\bthis (?:comment|post) was (?:posted|left by) a bot\b"
    r"|\bthis reply was generated automatically\b"
    r"|[\^*]*beep(?:\s+beep)?[\^*]*\s+[\^*]*boop(?:\s+boop)?[\^*]*",
    flags=re.IGNORECASE,
)

_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_zst(path: str):
    """Stream a .zst file as text without fully decompressing to disk.

    Mirrors _open_jsonl() in arcticshift_pipeline.py.
    The caller must close the returned object (use as a context manager).
    """
    fh     = open(path, "rb")
    dctx   = zstd.ZstdDecompressor()
    stream = dctx.stream_reader(fh)
    return io.TextIOWrapper(stream, encoding="utf-8")


def _is_valid_timestamp(ts) -> bool:
    """Return True if ts can be interpreted as a Unix timestamp."""
    try:
        datetime.fromtimestamp(int(ts))
        return True
    except (TypeError, ValueError, OSError):
        return False


def _find_zst_files(data_dir: str) -> list[str]:
    """Return sorted list of *_comments*.zst files in data_dir."""
    pattern = os.path.join(data_dir, "*comments*.zst")
    files   = sorted(glob.glob(pattern))
    return files


def _load_bot_usernames(csv_path: str) -> set[str]:
    """Load bot usernames from bot_usernames.csv; return empty set on failure."""
    if not os.path.isfile(csv_path):
        print(f"  [warn] bot_usernames.csv not found at {csv_path}; "
              "Stage 3 speaker count will not reflect bot removal.",
              file=sys.stderr)
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["username"], dtype=str)
        return set(df["username"].dropna().str.strip())
    except Exception as exc:
        print(f"  [warn] Could not read bot_usernames.csv: {exc}", file=sys.stderr)
        return set()


# ---------------------------------------------------------------------------
# Core: stream all .zst files into one shared accumulator
# ---------------------------------------------------------------------------

def count_all_zst_files(zst_files: list[str], verbose: bool = False) -> dict:
    """
    Stream every .zst file in *zst_files* once, accumulating into a single
    shared state so counts are correct across the full multi-file dataset.

    Returns the same key schema as dataset_summary_stats.count_jsonl():
        raw_comments, raw_speakers,
        filtered_comments, filtered_speakers,
        selected_comments, selected_speakers,
        drop_* breakdown counters.
    """
    # ── Shared accumulators (span all files) ──────────────────────────────
    raw_comments  = 0
    raw_authors:   set[str]             = set()

    filt_comments = 0
    filt_authors:  set[str]             = set()

    speaker_months: set[tuple[str,str]] = set()   # Stage 2

    # Drop-reason counters
    drop_missing_fields  = 0
    drop_excluded_author = 0
    drop_deleted_body    = 0
    drop_bot_text        = 0
    drop_no_letter       = 0
    drop_bad_timestamp   = 0

    for zst_path in zst_files:
        fname         = os.path.basename(zst_path)
        file_size_gb  = os.path.getsize(zst_path) / 1e9
        print(f"  Streaming {fname} ({file_size_gb:.2f} GB) …", flush=True)

        file_raw = 0
        file_filt = 0

        with _open_zst(zst_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                author = obj.get("author")
                body   = obj.get("body")
                utt_id = obj.get("id")
                ts     = obj.get("created_utc")

                # ── Stage 0 ───────────────────────────────────────────────
                raw_comments += 1
                file_raw     += 1
                if author is not None:
                    raw_authors.add(author)

                # ── Stage 1 filters ───────────────────────────────────────
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
                file_filt     += 1
                filt_authors.add(author)

                # ── Stage 2 unique (speaker, year_month) ──────────────────
                try:
                    dt = datetime.fromtimestamp(int(ts))
                    ym = f"{dt.year}-{dt.month:02d}"
                except Exception:
                    continue
                speaker_months.add((author, ym))

        print(f"    → {file_raw:>12,} raw   |  {file_filt:>12,} after filters")

    return {
        "raw_comments":      raw_comments,
        "raw_speakers":      len(raw_authors),
        "filtered_comments": filt_comments,
        "filtered_speakers": len(filt_authors),
        "selected_comments": len(speaker_months),
        "selected_speakers": len(filt_authors),
        # Drop breakdown
        "drop_missing_fields":    drop_missing_fields,
        "drop_excluded_author":   drop_excluded_author,
        "drop_deleted_body":      drop_deleted_body,
        "drop_bot_text":          drop_bot_text,
        "drop_no_letter":         drop_no_letter,
        "drop_bad_timestamp":     drop_bad_timestamp,
    }


# ---------------------------------------------------------------------------
# Formatting helpers (mirrors dataset_summary_stats.py)
# ---------------------------------------------------------------------------

def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{part / whole * 100:.1f}%"


def _delta_str(new: int, old: int) -> str:
    if old == 0:
        return "—"
    diff    = new - old
    sign    = "+" if diff >= 0 else "−"
    abs_diff = abs(diff)
    if abs_diff >= 1_000_000:
        magnitude = f"{abs_diff / 1_000_000:.2f}M"
    elif abs_diff >= 1_000:
        magnitude = f"{abs_diff / 1_000:.1f}K"
    else:
        magnitude = str(abs_diff)
    pct = abs_diff / old * 100
    return f"{sign}{magnitude} ({pct:.1f}%)"


def _print_table(counts: dict, csv_comments: int, csv_speakers: int) -> None:
    _NUM_W   = 13
    _DELTA_W = 18

    rows = [
        ("Stage 0 — Raw (.zst files)",           counts["raw_comments"],      counts["raw_speakers"],      "",                                                                          ""),
        ("Stage 1 — After text/author filters",  counts["filtered_comments"], counts["filtered_speakers"], _delta_str(counts["filtered_comments"], counts["raw_comments"]),             _delta_str(counts["filtered_speakers"], counts["raw_speakers"])),
        ("Stage 2 — After 1-per-speaker-month",  counts["selected_comments"], counts["selected_speakers"], _delta_str(counts["selected_comments"], counts["filtered_comments"]),        "—"),
        ("Stage 3 — Final CSV (after bot filter)", csv_comments,              csv_speakers,                _delta_str(csv_comments,  counts["selected_comments"]),                      _delta_str(csv_speakers, counts["selected_speakers"])),
    ]

    cols = [("Stage", 44), ("Comments", _NUM_W), ("Δ Comments", _DELTA_W), ("Speakers", _NUM_W), ("Δ Speakers", _DELTA_W)]
    header = "".join(lbl.ljust(w) for lbl, w in cols)
    sep    = "─" * len(header)

    print()
    print("r/teenagers — Pipeline Summary")
    print(sep)
    print(header)
    print(sep)
    for stage_lbl, comments, speakers, d_comments, d_speakers in rows:
        print(
            f"{stage_lbl:<44}"
            f"{comments:>{_NUM_W},}"
            f"{d_comments:>{_DELTA_W}}"
            f"{speakers:>{_NUM_W},}"
            f"{d_speakers:>{_DELTA_W}}"
        )
    print(sep)
    print(f"\nOverall retention (Stage 0 → Stage 3): "
          f"{_pct(csv_comments, counts['raw_comments'])} of raw comments, "
          f"{_pct(csv_speakers, counts['raw_speakers'])} of raw speakers")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-by-stage summary stats for r/teenagers (multi-.zst).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="Directory containing the r/teenagers *_comments*.zst files.")
    p.add_argument("--csv",      default=DEFAULT_CSV,
                   help="Path to Teenagers_lexical_df.csv (Stage 3 / after processing).")
    p.add_argument("--bots",     default=DEFAULT_BOTS_CSV,
                   help="Path to bot_usernames.csv.")
    p.add_argument("--verbose",  action="store_true",
                   help="Print per-filter drop counts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("r/teenagers Dataset Summary Statistics")
    print("=" * 70)
    print(f"\nData directory : {args.data_dir}")
    print(f"Lexical CSV    : {args.csv}")
    print(f"Bot list       : {args.bots}")

    # ── Discover .zst files ───────────────────────────────────────────────
    zst_files = _find_zst_files(args.data_dir)
    if not zst_files:
        sys.exit(
            f"\nERROR: No *_comments*.zst files found in {args.data_dir}\n"
            "Check --data_dir or that the ArcticShift exports are present."
        )

    print(f"\nFound {len(zst_files)} .zst file(s):")
    for f in zst_files:
        print(f"  {os.path.basename(f)}")

    # ── Stage 3: load Teenagers_lexical_df.csv ────────────────────────────
    print(f"\nLoading {os.path.basename(args.csv)} …", flush=True)
    if not os.path.isfile(args.csv):
        sys.exit(f"\nERROR: Lexical CSV not found at {args.csv}")

    bot_usernames = _load_bot_usernames(args.bots)

    csv_df        = pd.read_csv(args.csv, usecols=["speaker_id"], low_memory=False)
    if bot_usernames:
        csv_df = csv_df[~csv_df["speaker_id"].isin(bot_usernames)]
    csv_comments  = len(csv_df)
    csv_speakers  = csv_df["speaker_id"].nunique()
    del csv_df

    # ── Stream all .zst files ─────────────────────────────────────────────
    print("\nStreaming .zst files (shared accumulator across all files) …")
    counts = count_all_zst_files(zst_files, verbose=args.verbose)

    if args.verbose:
        print("\nFilter breakdown (totals across all files):")
        print(f"  Missing required fields  : {counts['drop_missing_fields']:>12,}")
        print(f"  Excluded author          : {counts['drop_excluded_author']:>12,}")
        print(f"  Deleted/removed body     : {counts['drop_deleted_body']:>12,}")
        print(f"  Bot-text pattern         : {counts['drop_bot_text']:>12,}")
        print(f"  No Latin letter in body  : {counts['drop_no_letter']:>12,}")
        print(f"  Unparseable timestamp    : {counts['drop_bad_timestamp']:>12,}")

    # ── Print summary table ───────────────────────────────────────────────
    _print_table(counts, csv_comments, csv_speakers)


if __name__ == "__main__":
    main()
