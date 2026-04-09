#!/bin/bash
# =============================================================================
#  run_longitudinal.slurm
#  Longitudinal user trajectory model — Adroit SLURM job
#
#  Submit from the ORFE-Thesis root with:
#      sbatch src/run_longitudinal.slurm
# =============================================================================

#SBATCH --job-name=longitudinal-lexical
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------
# lexical_df_combined.csv is ~9 GB on disk; pandas expands to ~20-28 GB in
# memory. After aggregating to user-month level, the working dataset shrinks
# dramatically. The REML optimisation for the random-slope model can spike to
# 2-3x the aggregated frame size per metric. 128 GB is comfortable.
#
# If the job is OOM-killed, bump to 256G (matching run_arcticshift_teenagers).
# Alternatively, add --no_random_slope to use the lighter intercept-only model.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wall-time estimate
# ---------------------------------------------------------------------------
# Data load + aggregation:       ~5-15 min
# Per-metric REML (random slope): ~30-90 min each x 6 metrics
# Total:                          ~4-10 h
# 8 h is a comfortable ceiling. If consistently exceeded, split into
# per-metric jobs or increase --time to 20:00:00.
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
mkdir -p /scratch/network/nv9344/Thesis/Visualizations/longitudinal

# Point BLAS threads to the allocated CPU count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo ""

cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

python -u run_longitudinal.py \
    --data     /scratch/network/nv9344/Thesis/Thesis-Data/ArcticShift/lexical_df_combined.csv \
    --out_dir  /scratch/network/nv9344/Thesis/Visualizations/longitudinal \
    --min_months 6 \
    --alpha 0.05
    # Add --no_random_slope below if the random-slope model hits memory or
    # convergence issues:
    # --no_random_slope

echo ""
echo "End    : $(date)"
