#!/bin/bash
# =============================================================================
#  run_longitudinal.slurm
#  Longitudinal user trajectory model — Adroit SLURM job
#
#  Submit with:
#      sbatch slurm/run_longitudinal.slurm
#
#  Tune the --mem and --time values if needed:
#    - Default (random slope model, full dataset):  ~64 GB, ~3-4 hours
#    - Intercept-only model (--no_random_slope):    ~32 GB, ~1-2 hours
# =============================================================================

#SBATCH --job-name=longitudinal_lexical
#SBATCH --output=logs/longitudinal_%j.out
#SBATCH --error=logs/longitudinal_%j.err

# --- Resources ---------------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # statsmodels uses OpenBLAS threading
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# --- Notification (optional) -------------------------------------------------
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_NETID@princeton.edu

# =============================================================================
# Environment setup
# =============================================================================

echo "=============================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "CPUs        : $SLURM_CPUS_PER_TASK"
echo "Start time  : $(date)"
echo "=============================="

# Load Princeton's Anaconda module (adjust version as needed)
module purge
module load anaconda3/2024.2

# Activate your conda environment.
# Create it first with:
#   conda create -n orfe-thesis python=3.11
#   conda activate orfe-thesis
#   pip install -r requirements.txt
conda activate orfe-thesis

# Point OpenBLAS / MKL threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# =============================================================================
# Paths — edit DATA_PATH to wherever lexical_master.csv lives on /scratch
# =============================================================================

PROJECT_DIR="$HOME/ORFE-Thesis"          # adjust if repo lives elsewhere
DATA_PATH="/scratch/gpfs/$USER/Thesis-Data/lexical_master.csv"
OUT_DIR="$PROJECT_DIR/Results/longitudinal"

# Ensure logs and output directories exist
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUT_DIR"

# =============================================================================
# Run the analysis
# =============================================================================

cd "$PROJECT_DIR"

python src/run_longitudinal.py \
    --data        "$DATA_PATH"  \
    --out_dir     "$OUT_DIR"    \
    --min_months  6             \
    --chunksize   5000000       \
    --alpha       0.05
    # Remove the line below to enable random slopes (default).
    # Add --no_random_slope if you hit memory/convergence issues:
    # --no_random_slope

EXIT_CODE=$?

echo "=============================="
echo "End time    : $(date)"
echo "Exit code   : $EXIT_CODE"
echo "=============================="

exit $EXIT_CODE
