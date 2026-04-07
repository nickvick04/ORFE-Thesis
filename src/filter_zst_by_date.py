# ----------------------------------------------------------------------------------------
# filter_zst_by_date.py
#
# Streams a zst-compressed Reddit JSONL file and writes a new zst file containing
# only comments on or after a given UTC cutoff date.  Neither file is fully
# decompressed to disk at any point — the pipeline is entirely streaming.
#
# Usage (interactive):
#   python filter_zst_by_date.py \
#       --input  /path/to/r_teenagers_comments.zst \
#       --output /path/to/r_teenagers_comments_2015.zst
#
# Optional flags:
#   --after   YYYY-MM-DD   Inclusive start date (default: 2015-01-01)
#   --level   1-22         zstd compression level for output (default: 3, fast)
#   --report  N            Print a progress line every N lines (default: 1_000_000)
#
# Code Author: Nicholas Vickery, Princeton ORFE '26
# ----------------------------------------------------------------------------------------

import argparse
import io
import json
import sys
from datetime import datetime, timezone

import zstandard as zstd


# ----------------------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a zst-compressed Reddit JSONL file by date, "
            "writing a new zst file with only records on or after --after."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input *_comments.zst file.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the filtered output *_comments.zst file.",
    )
    parser.add_argument(
        "--after",
        default="2015-01-01",
        metavar="YYYY-MM-DD",
        help="Keep only comments with created_utc >= this date (default: 2015-01-01).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        metavar="1-22",
        help="zstd compression level for the output file (default: 3).",
    )
    parser.add_argument(
        "--report",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Print a progress line every N lines processed (default: 1,000,000).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------------------
# Main filtering logic
# ----------------------------------------------------------------------------------------

def filter_zst(
    input_path: str,
    output_path: str,
    cutoff_utc: int,
    compression_level: int = 3,
    report_every: int = 1_000_000,
) -> None:
    """Stream-filter input_path → output_path, keeping lines where
    created_utc >= cutoff_utc.  Both files stay compressed throughout.
    """
    dctx = zstd.ZstdDecompressor()
    cctx = zstd.ZstdCompressor(level=compression_level)

    kept    = 0
    skipped = 0
    errors  = 0

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Cutoff: {datetime.fromtimestamp(cutoff_utc, tz=timezone.utc).strftime('%Y-%m-%d')} "
        f"(Unix {cutoff_utc:,})"
    )
    print(f"Compression level: {compression_level}  |  Progress every {report_every:,} lines\n")

    with open(input_path, "rb") as in_fh, open(output_path, "wb") as out_fh:
        with dctx.stream_reader(in_fh) as reader, \
             cctx.stream_writer(out_fh, closefd=False) as writer:

            text_reader = io.TextIOWrapper(reader, encoding="utf-8")

            for raw_line in text_reader:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                ts = obj.get("created_utc")
                try:
                    ts_int = int(ts)
                except (TypeError, ValueError):
                    errors += 1
                    continue

                if ts_int < cutoff_utc:
                    skipped += 1
                else:
                    # Write the original line (+ newline) to keep output valid JSONL
                    writer.write((line + "\n").encode("utf-8"))
                    kept += 1

                total = kept + skipped
                if total % report_every == 0:
                    pct_kept = 100 * kept / total if total else 0
                    print(
                        f"  {total:>12,} processed — "
                        f"kept {kept:>10,} ({pct_kept:.1f}%)  |  "
                        f"skipped {skipped:>10,}  |  "
                        f"errors {errors:>6,}",
                        flush=True,
                    )

    total = kept + skipped
    pct_kept = 100 * kept / total if total else 0
    print(f"\nDone.")
    print(f"  Total lines processed : {total:,}")
    print(f"  Kept (>= cutoff)      : {kept:,}  ({pct_kept:.1f}%)")
    print(f"  Skipped (< cutoff)    : {skipped:,}  ({100 - pct_kept:.1f}%)")
    print(f"  JSON errors           : {errors:,}")
    print(f"\nFiltered file written to: {output_path}")


# ----------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Parse the --after date
    try:
        cutoff_dt  = datetime.strptime(args.after, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        cutoff_utc = int(cutoff_dt.timestamp())
    except ValueError:
        sys.exit(f"ERROR: --after must be in YYYY-MM-DD format, got: {args.after!r}")

    if not 1 <= args.level <= 22:
        sys.exit(f"ERROR: --level must be between 1 and 22, got: {args.level}")

    filter_zst(
        input_path=args.input,
        output_path=args.output,
        cutoff_utc=cutoff_utc,
        compression_level=args.level,
        report_every=args.report,
    )


if __name__ == "__main__":
    main()
