#!/bin/bash
#SBATCH --job-name=convokit-summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

# Shared paths
CONVOKIT_DIR="/scratch/network/nv9344/Thesis/Thesis-Data/Convokit"
MASTER_CSV="/scratch/network/nv9344/Thesis/Thesis-Data/lexical_master.csv"
VIZ_DIR="/scratch/network/nv9344/Thesis/Visualizations"

# ---------------------------------------------------------------------------
# 1. Dataset summary statistics (streams all 9 JSONL files + reads master CSV)
# ---------------------------------------------------------------------------
echo "========================================"
echo "Step 1: Dataset summary stats"
echo "Started: $(date)"
echo "========================================"

python -u convokit_dataset_summary_stats.py \
    --data_dir "$CONVOKIT_DIR" \
    --csv      "$MASTER_CSV"   \
    --verbose

echo "Step 1 finished: $(date)"

# ---------------------------------------------------------------------------
# 2. Activity plots (reads master CSV, writes PNG + PDF)
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "Step 2: Activity plots"
echo "Started: $(date)"
echo "========================================"

python -u convokit_plot_activity.py \
    --csv     "$MASTER_CSV" \
    --out_dir "$VIZ_DIR"

echo "Step 2 finished: $(date)"
echo ""
echo "All done: $(date)"
