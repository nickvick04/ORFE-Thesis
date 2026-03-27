#!/bin/bash
#SBATCH --job-name=syntactic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

# ---------------------------------------------------------------------------
# run_syntactic.slurm
# Runs the full (lexical + syntactic) pipeline on a single Convokit corpus.
#
# Required environment variables (pass via --export or set below):
#   CORPUS_NAME  e.g. subreddit-teenagers
#   VARIATION    e.g. Age-Variation | Topic-Variation | Culture-Variation
#
# Example single-corpus submission:
#   sbatch --export=ALL,CORPUS_NAME=subreddit-teenagers,VARIATION=Age-Variation \
#          run_syntactic.slurm
#
# Example SLURM array (4 shards across a large corpus):
#   sbatch --array=0-3 \
#          --export=ALL,CORPUS_NAME=subreddit-worldnews,VARIATION=Topic-Variation,NUM_SHARDS=4 \
#          run_syntactic.slurm
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs

CONVOKIT_ROOT="/scratch/network/nv9344/Thesis/ORFE-Thesis/Data/Convokit"

# Default shard settings (overridden automatically when using --array)
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Batch size: 100 rows keeps per-batch memory low while amortising JSONL
# streaming overhead.  Increase to 200-500 if RAM headroom allows.
BATCH_SIZE=100

echo "Job:         ${SLURM_JOB_ID}"
echo "Corpus:      ${CORPUS_NAME}"
echo "Variation:   ${VARIATION}"
echo "Shard:       ${SHARD_INDEX} / ${NUM_SHARDS}"
echo "Workers:     ${SLURM_CPUS_PER_TASK}"
echo "Batch size:  ${BATCH_SIZE}"

cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

python -u run_analysis.py "${CORPUS_NAME}" \
    --variation   "${VARIATION}"            \
    --convokit_root "${CONVOKIT_ROOT}"      \
    --batch_size  "${BATCH_SIZE}"           \
    --num_shards  "${NUM_SHARDS}"           \
    --shard_index "${SHARD_INDEX}"          \
    --workers     "${SLURM_CPUS_PER_TASK}"
