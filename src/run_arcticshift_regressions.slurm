#!/bin/bash
#SBATCH --job-name=arcticshift-regressions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # allows numpy/BLAS to use multiple threads
#SBATCH --mem=128G                 # see memory notes below
#SBATCH --time=20:00:00            # see time notes below
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------
# The combined ArcticShift CSV is expected to be similar in size to or larger
# than lexical_master.csv (~9.3 GB on disk).  pandas typically expands a CSV
# to 2–3× its on-disk size in memory, so the raw DataFrame alone needs ~20–28 GB.
#
# The bottleneck is run_mixed_effects (§4.7):
#   - time dummies: ~200 year-month columns × N rows × 8 bytes
#   - REML optimisation via L-BFGS stores multiple dense gradient/Hessian
#     approximations; with millions of rows this can spike to 3–5× data size
#
# 128 GB covers:
#   ~28 GB (raw DataFrame) + ~50 GB (mixed-effects matrices) + ~50 GB headroom
#
# If the job is killed with OOM, bump to 256G (matching run_arcticshift_teenagers).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wall-time estimate
# ---------------------------------------------------------------------------
# Models 1–3 (OLS variants):  ~5–15 min total
# Model 4 (Cross-user WLS):   ~15–30 min
# Model 5 (Fixed effects):    ~1–3 h  (PanelOLS entity-demeaning at scale)
# Model 6 (Mixed effects):    ~4–12 h (REML via L-BFGS, 6 metrics × large N)
#
# 20 h provides comfortable headroom.  If mixed effects consistently exceeds
# this, consider running it in a separate job with a 48 h limit.
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
mkdir -p /scratch/network/nv9344/Thesis/Visualizations
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo ""

python -u run_arcticshift_regressions.py

echo ""
echo "End: $(date)"
