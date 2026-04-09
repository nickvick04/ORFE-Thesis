#!/bin/bash
#SBATCH --job-name=me-pairwise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # numpy/BLAS multi-threading during REML optimisation
#SBATCH --mem=128G                 # see memory notes below
#SBATCH --time=16:00:00            # see time notes below
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------
# The combined ArcticShift CSV loads to ~20–28 GB in RAM (2–3× on-disk size).
# The ME model is the bottleneck:
#   - subreddit + time dummies: ~200 year-month columns × N rows × 8 bytes
#   - REML L-BFGS stores multiple dense gradient/Hessian approximations;
#     with millions of rows this can spike to 3–5× data size
#   - res.cov_params() adds one dense (n_params × n_params) matrix per metric
#     (negligible relative to the fitting cost)
#
# 128 GB covers:
#   ~28 GB (DataFrame) + ~50 GB (mixed-effects matrices) + ~50 GB headroom
#
# If the job is OOM-killed, bump to 256G.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wall-time estimate
# ---------------------------------------------------------------------------
# This script runs only the ME model (no OLS/WLS/FE variants).
# Typical runtime per metric: 30 min – 2 h (REML L-BFGS at scale).
# 6 metrics × 2 h = 12 h worst case; 16 h provides comfortable headroom.
#
# If runtimes exceed this, consider:
#   - Splitting metrics across array jobs (--array=0-5, index into LEXICAL_METRICS)
#   - Switching to method="nm" (Nelder-Mead) as the primary solver
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
mkdir -p /scratch/network/nv9344/Thesis/Visualizations
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

echo "Job ID : ${SLURM_JOB_ID}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo ""

python -u run_me_pairwise.py

echo ""
echo "End: $(date)"
