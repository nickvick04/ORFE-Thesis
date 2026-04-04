#!/bin/bash
#SBATCH --job-name=arcticshift-lexical
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --time=10:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%A_%a.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%A_%a.err

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

# One subreddit per array task
TASKS=(
  "College"
  "Parenting"
  "Retirement"
  "Teenagers"
)

SUBREDDIT="${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "Running ArcticShift lexical pipeline for: ${SUBREDDIT}"

python -u run_arcticshift.py \
  --subreddit "$SUBREDDIT" \
  --batch_size 1000
