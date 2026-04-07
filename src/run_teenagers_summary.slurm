#!/bin/bash
#SBATCH --job-name=teenagers-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G                        # Elevated vs. other subreddits: unique-author and
                                         # (speaker, year_month) sets span ~127M comments
#SBATCH --time=4:00:00                   # 3 .zst files streamed sequentially; ~1h each
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

echo "Starting teenagers summary stats: $(date)"

python -u teenagers_summary_stats.py \
  --data_dir /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/Teenagers \
  --csv      /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/Teenagers_lexical_df.csv \
  --bots     /scratch/network/nv9344/Thesis/ORFE-Thesis/src/bot_usernames.csv \
  --verbose

echo "Finished: $(date)"
