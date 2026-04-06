#!/bin/bash
#SBATCH --job-name=dataset-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

echo "Starting dataset summary stats: $(date)"

python -u dataset_summary_stats.py \
  --data_dir /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift \
  --csv      /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv \
  --bots     /scratch/network/nv9344/Thesis/ORFE-Thesis/src/bot_usernames.csv \
  --verbose

echo "Finished: $(date)"
