"""
run_arcticshift.py
==================
Entry point for running the ArcticShift lexical pipeline on the Princeton
Adroit cluster via SLURM.

Parallelism model
-----------------
Two orthogonal levels of parallelism are supported and can be combined:

  1. Subreddit-level (SLURM array tasks)
     One array task per subreddit.  With the default ``--num_shards 1``:
         sbatch --array=0-3 run_arcticshift.slurm
     Task IDs 0-3 map to College → Parenting → Retirement → Teenagers.

  2. Shard-level (within a subreddit, also via SLURM array tasks)
     When ``--num_shards N`` (N > 1) the pipeline streams only 1/N of the
     speakers in each pass, writing a separate CSV shard.  The combined
     array size is (num_subreddits × num_shards):
         sbatch --array=0-11 run_arcticshift.slurm --num_shards 3

     Mapping:   subreddit_idx = SLURM_ARRAY_TASK_ID // num_shards
                shard_index   = SLURM_ARRAY_TASK_ID  % num_shards

  3. Interactive / local fallback
     When SLURM_ARRAY_TASK_ID is not set and no ``--subreddit`` flag is
     given, all four subreddits are processed in parallel using
     multiprocessing.Pool (one worker per subreddit, up to CPU count).

Usage examples
--------------
# Interactive: all four subreddits in parallel
python run_arcticshift.py

# Interactive: single subreddit
python run_arcticshift.py --subreddit College

# SLURM array (one task per subreddit, no sharding)
#   In your .slurm script: #SBATCH --array=0-3
python run_arcticshift.py

# SLURM array with 3 shards per subreddit (12 tasks total)
#   In your .slurm script: #SBATCH --array=0-11
python run_arcticshift.py --num_shards 3

Author: Nicholas Vickery, Princeton ORFE '26
"""

import argparse
import multiprocessing
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — ensure the src directory is on sys.path so that
# arcticshift_pipeline and its dependencies are importable regardless of
# where the script is invoked from.
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from arcticshift_pipeline import run_lexical_pipeline_arcticshift_batches  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ordered list of ArcticShift subreddit folder names.
SUBREDDITS = ["College", "High School", "Parenting", "Retirement", "Teenagers"]

# Base directory containing the subreddit folders on Adroit.
DEFAULT_DATA_DIR = "/scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift"


# ---------------------------------------------------------------------------
# Worker function (used both directly and by multiprocessing.Pool)
# ---------------------------------------------------------------------------

def process_subreddit(
    subreddit: str,
    data_dir: str,
    num_shards: int,
    shard_index: int,
    batch_size: int,
) -> None:
    """Call the ArcticShift lexical pipeline for one subreddit."""
    corpus_dir = os.path.join(data_dir, subreddit)
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}\n"
            "Check that DATA_DIR and the subreddit name are correct."
        )
    print(
        f"[{subreddit}] Starting — shard {shard_index + 1}/{num_shards}",
        flush=True,
    )
    run_lexical_pipeline_arcticshift_batches(
        corpus_dir=corpus_dir,
        batch_size=batch_size,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    print(f"[{subreddit}] Finished — shard {shard_index + 1}/{num_shards}", flush=True)


# Multiprocessing requires a top-level picklable callable; this thin wrapper
# unpacks the single-argument tuple that Pool.map passes.
def _worker(args: tuple) -> None:
    process_subreddit(*args)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ArcticShift lexical pipeline (SLURM-aware).",
    )
    parser.add_argument(
        "--data_dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "Base directory containing subreddit folders "
            f"(default: {DEFAULT_DATA_DIR})"
        ),
    )
    parser.add_argument(
        "--subreddit",
        default=None,
        choices=SUBREDDITS,
        help=(
            "Process a single named subreddit.  Overrides SLURM_ARRAY_TASK_ID "
            "and the multiprocessing fallback."
        ),
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help=(
            "Total number of shards per subreddit.  Must match the per-subreddit "
            "stride used when scheduling the SLURM array (default: 1)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Rows per processing batch (default: 1000).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.num_shards < 1:
        sys.exit("--num_shards must be >= 1")

    # ------------------------------------------------------------------
    # Determine which subreddit (and shard) this invocation should handle.
    # Priority: explicit --subreddit flag > SLURM_ARRAY_TASK_ID > run-all.
    # ------------------------------------------------------------------

    slurm_task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID")

    if args.subreddit is not None:
        # Explicit flag: always shard 0 unless the caller also sets env vars.
        subreddit   = args.subreddit
        shard_index = 0
        num_shards  = args.num_shards
        print(
            f"Running single subreddit '{subreddit}' "
            f"(shard {shard_index + 1}/{num_shards})."
        )
        process_subreddit(subreddit, args.data_dir, num_shards, shard_index, args.batch_size)

    elif slurm_task_id_str is not None:
        # SLURM array job: decode task ID into (subreddit_idx, shard_index).
        task_id        = int(slurm_task_id_str)
        num_shards     = args.num_shards
        total_tasks    = len(SUBREDDITS) * num_shards

        if task_id >= total_tasks:
            sys.exit(
                f"SLURM_ARRAY_TASK_ID={task_id} is out of range for "
                f"{len(SUBREDDITS)} subreddits × {num_shards} shard(s) "
                f"(expected 0–{total_tasks - 1})."
            )

        subreddit_idx = task_id // num_shards
        shard_index   = task_id  % num_shards
        subreddit     = SUBREDDITS[subreddit_idx]

        print(
            f"SLURM task {task_id}: subreddit='{subreddit}', "
            f"shard {shard_index + 1}/{num_shards}",
            flush=True,
        )
        process_subreddit(subreddit, args.data_dir, num_shards, shard_index, args.batch_size)

    else:
        # Interactive / local fallback: run all subreddits in parallel.
        num_shards  = args.num_shards
        num_workers = min(len(SUBREDDITS), multiprocessing.cpu_count())
        print(
            f"No SLURM_ARRAY_TASK_ID detected.  Running all {len(SUBREDDITS)} "
            f"subreddit(s) in parallel with {num_workers} worker(s).",
            flush=True,
        )
        worker_args = [
            (sub, args.data_dir, num_shards, 0, args.batch_size)
            for sub in SUBREDDITS
        ]
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.map(_worker, worker_args)

    print("All done.", flush=True)


if __name__ == "__main__":
    main()
