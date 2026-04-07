#!/bin/bash
#SBATCH --job-name=arcticshift-teenagers
#SBATCH --array=12-14                   # Teenagers (idx 4) × 3 shards: 4*3+0=12, 4*3+1=13, 4*3+2=14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256G              # Elevated: Pass 1 dicts hold all ~127M comments in memory (~80-120GB dict overhead alone)
#SBATCH --time=12:00:00                 # ~1.5h for 2 file passes + up to ~10h lexical (1/3 of speakers)
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%A_%a.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%A_%a.err

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

# No --subreddit flag: let SLURM_ARRAY_TASK_ID drive subreddit + shard selection.
# Mapping (num_shards=3):
#   task 12 → Teenagers shard 1/3
#   task 13 → Teenagers shard 2/3
#   task 14 → Teenagers shard 3/3
echo "Running ArcticShift lexical pipeline — SLURM task ${SLURM_ARRAY_TASK_ID} (Teenagers, shard $(( SLURM_ARRAY_TASK_ID - 12 + 1 ))/3)"

python -u run_arcticshift.py \
  --num_shards 3 \
  --batch_size 5000
