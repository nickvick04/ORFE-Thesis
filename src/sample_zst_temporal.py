# ----------------------------------------------------------------------------------------
# sample_zst_temporal.py
#
# Downsamples a zst-compressed Reddit JSONL file to a target comment count.
# Supports two modes:
#
#   proportional (default, recommended):
#       Keeps a uniform fraction of comments across ALL months.
#       Preserves the natural temporal growth trend — later years remain
#       proportionally larger than earlier years, matching the structure of
#       other subreddits.  Use this for cross-subreddit comparison.
#       Keep probability = target_total / total_comments  (same for every month)
#
#   stratified:
#       Caps each (year, month) bucket at --max-per-month comments.
#       Months below the cap are kept in full; months above are sampled down.
#       Equalises temporal representation but flattens later-year growth —
#       use only if uniform temporal precision is explicitly desired.
#
# Both modes use two streaming passes over the input file so it is never
# fully loaded into memory.
#
# Input format is auto-detected from the file extension:
#   .zst   → stream-decompressed on the fly (no temp file needed)
#   .jsonl → read directly as plain text
# Output is always written as a .zst file.
#
# Usage:
#   # Recommended — proportional, targeting ~15M comments (zst input)
#   python sample_zst_temporal.py \
#       --input  /path/to/r_teenagers_comments_2015.zst \
#       --output /path/to/r_teenagers_comments_sampled.zst \
#       --target 15_000_000 \
#       --mode   proportional \
#       --seed   42
#
#   # Same for a plain .jsonl file (e.g. from the ArcticShift download tool)
#   python sample_zst_temporal.py \
#       --input  /path/to/r_teenagers_comments_2024.jsonl \
#       --output /path/to/r_teenagers_comments_2024_sampled.zst \
#       --target 2_000_000 \
#       --mode   proportional \
#       --seed   42
#
#   # Stratified (flat cap per month)
#   python sample_zst_temporal.py \
#       --input  /path/to/r_teenagers_comments_2015.zst \
#       --output /path/to/r_teenagers_comments_sampled.zst \
#       --max-per-month 156000 \
#       --mode   stratified \
#       --seed   42
#
# Optional flags:
#   --mode           proportional|stratified   Sampling mode (default: proportional)
#   --target         int    Total target comments for proportional mode (default: 15_000_000)
#   --max-per-month  int    Per-month cap for stratified mode          (default: 156_000)
#   --level          1-22   zstd compression level for output          (default: 3)
#   --seed           int    Random seed for reproducibility            (default: 42)
#   --report         int    Progress interval in lines                 (default: 1_000_000)
#
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import argparse
import io
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import zstandard as zstd


# ----------------------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a zst Reddit JSONL file (proportional or stratified)."
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to input *_comments.zst file.")
    parser.add_argument("--output", "-o", required=True,
                        help="Path for sampled output *_comments.zst file.")
    parser.add_argument("--mode", choices=["proportional", "stratified"],
                        default="proportional",
                        help="Sampling mode (default: proportional).")
    parser.add_argument("--target", type=int, default=15_000_000, metavar="N",
                        help="[proportional] Total target comment count (default: 15,000,000).")
    parser.add_argument("--max-per-month", type=int, default=156_000, metavar="N",
                        help="[stratified] Max comments per (year, month) (default: 156,000).")
    parser.add_argument("--level", type=int, default=3, metavar="1-22",
                        help="zstd compression level for output (default: 3).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--report", type=int, default=1_000_000, metavar="N",
                        help="Print progress every N lines (default: 1,000,000).")
    return parser.parse_args()


# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

def _open_input(path: str):
    """Return a text-mode streaming reader for a .zst or .jsonl file.

    .zst  → stream-decompressed via zstandard (no temp file needed).
    .jsonl → opened directly as plain UTF-8 text.
    The caller is responsible for closing the returned object.
    """
    if path.endswith(".zst"):
        fh   = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
    elif path.endswith(".jsonl"):
        return open(path, "r", encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported input format: {path!r}. "
            "Expected a file ending in .zst or .jsonl."
        )


def _year_month(obj: dict):
    """Return (year, month) from a comment's created_utc, or None on failure."""
    ts = obj.get("created_utc")
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        return (dt.year, dt.month)
    except (TypeError, ValueError, OSError):
        return None


# ----------------------------------------------------------------------------------------
# Pass 1 — count comments per (year, month)
# ----------------------------------------------------------------------------------------

def count_by_month(input_path: str, report_every: int) -> dict:
    """Stream the file once and return a {(year, month): count} dict."""
    print("Pass 1 — counting comments per (year, month)...")
    counts = defaultdict(int)
    total  = 0
    errors = 0

    with _open_input(input_path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            ym = _year_month(obj)
            if ym is None:
                errors += 1
                continue

            counts[ym] += 1
            total += 1

            if total % report_every == 0:
                print(f"  {total:>12,} lines scanned", flush=True)

    print(f"  Done. {total:,} valid lines across {len(counts)} month-buckets "
          f"({errors:,} errors).\n")
    return dict(counts)


# ----------------------------------------------------------------------------------------
# Keep-probability computation
# ----------------------------------------------------------------------------------------

def compute_keep_probs(counts: dict, mode: str, target: int, max_per_month: int) -> dict:
    """Return a {(year, month): keep_probability} dict."""
    total = sum(counts.values())

    if mode == "proportional":
        if total == 0:
            return {ym: 0.0 for ym in counts}
        p = min(1.0, target / total)
        print(f"Proportional mode: keeping {p*100:.2f}% of every month "
              f"(target {target:,} / total {total:,})\n")
        return {ym: p for ym in counts}

    else:  # stratified
        probs = {ym: min(1.0, max_per_month / cnt) for ym, cnt in counts.items()}
        capped = sum(1 for p in probs.values() if p < 1.0)
        print(f"Stratified mode: cap {max_per_month:,}/month — "
              f"{capped} of {len(counts)} buckets will be downsampled.\n")
        return probs


# ----------------------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------------------

def print_summary(counts: dict, keep_probs: dict) -> None:
    """Print expected per-year input and output counts."""
    year_input  = defaultdict(int)
    year_output = defaultdict(int)

    for (yr, mo), cnt in sorted(counts.items()):
        year_input[yr]  += cnt
        year_output[yr] += round(cnt * keep_probs[(yr, mo)])

    print(f"{'Year':<6} {'Input':>12} {'Expected output':>16} {'Kept %':>8}")
    print("-" * 46)
    for yr in sorted(year_input):
        inp = year_input[yr]
        out = year_output[yr]
        pct = 100 * out / inp if inp else 0
        print(f"{yr:<6} {inp:>12,} {out:>16,} {pct:>7.1f}%")

    total_in  = sum(year_input.values())
    total_out = sum(year_output.values())
    pct = 100 * total_out / total_in if total_in else 0
    print("-" * 46)
    print(f"{'Total':<6} {total_in:>12,} {total_out:>16,} {pct:>7.1f}%\n")


# ----------------------------------------------------------------------------------------
# Pass 2 — sample and write
# ----------------------------------------------------------------------------------------

def sample_and_write(
    input_path:  str,
    output_path: str,
    keep_probs:  dict,
    level:       int,
    seed:        int,
    report_every: int,
) -> None:
    rng  = random.Random(seed)
    cctx = zstd.ZstdCompressor(level=level)

    kept   = 0
    total  = 0
    errors = 0

    print("Pass 2 — sampling and writing...")
    with _open_input(input_path) as in_f, \
         open(output_path, "wb") as out_fh, \
         cctx.stream_writer(out_fh, closefd=False) as writer:

        for raw_line in in_f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue

            ym = _year_month(obj)
            if ym is None:
                errors += 1
                continue

            total += 1
            if rng.random() < keep_probs.get(ym, 1.0):
                writer.write((line + "\n").encode("utf-8"))
                kept += 1

            if total % report_every == 0:
                print(f"  {total:>12,} processed — kept {kept:>10,} "
                      f"({100*kept/total:.1f}%)", flush=True)

    print(f"\nDone.")
    print(f"  Total processed : {total:,}")
    print(f"  Kept            : {kept:,}  ({100*kept/total:.1f}%)" if total else "  Kept: 0")
    print(f"  JSON errors     : {errors:,}")
    print(f"  Output          : {output_path}")


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not 1 <= args.level <= 22:
        sys.exit(f"ERROR: --level must be 1–22, got {args.level}")

    print(f"Input  : {args.input}")
    print(f"Output : {args.output}")
    print(f"Mode   : {args.mode}")
    print(f"Seed   : {args.seed}\n")

    counts     = count_by_month(args.input, args.report)
    keep_probs = compute_keep_probs(
        counts, args.mode, args.target, args.max_per_month
    )
    print_summary(counts, keep_probs)
    sample_and_write(
        input_path   = args.input,
        output_path  = args.output,
        keep_probs   = keep_probs,
        level        = args.level,
        seed         = args.seed,
        report_every = args.report,
    )


if __name__ == "__main__":
    main()
