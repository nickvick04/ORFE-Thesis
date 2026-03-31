#!/bin/bash
# ============================================================
# SLURM job script — Trend Analysis (steps 1–7)
# Princeton Adroit cluster
#
# Submit: sbatch trend_analysis.slurm
# Monitor: squeue -u $USER
# Logs:    logs/trend_analysis_<jobid>.out / .err
# ============================================================

#SBATCH --job-name=trend_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/trend_analysis_%j.out
#SBATCH --error=logs/trend_analysis_%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=nickvick04@gmail.com

# ============================================================
# CONFIGURE THESE PATHS BEFORE SUBMITTING
# ============================================================
LEXICAL_MASTER="/scratch/network/YOUR_NETID/Thesis-Data/Convokit/lexical_master.csv"
OUTPUT_DIR="/scratch/network/YOUR_NETID/Thesis-Data/TrendResults"
SRC_DIR="/home/YOUR_NETID/ORFE-Thesis/src"

# Analysis parameters — adjust as needed
MIN_OBS=3           # minimum monthly observations per speaker
MIN_SPEAKERS=50     # minimum speakers per subreddit for per-subreddit FE
# ============================================================

echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Input      : $LEXICAL_MASTER"
echo "Output dir : $OUTPUT_DIR"
echo "============================================"

# --- Environment setup ---
module purge
module load anaconda3/2023.9           # update version to match your Adroit module

conda activate thesis_env              # update to your conda environment name

# Confirm linearmodels is available; install if not
python -c "import linearmodels" 2>/dev/null || {
    echo "linearmodels not found — installing..."
    pip install linearmodels --user --quiet
}

# Create output and log directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# --- Run trend analysis ---
cd "$SRC_DIR"

python trend_analysis.py \
    --input        "$LEXICAL_MASTER" \
    --output-dir   "$OUTPUT_DIR"    \
    --min-obs      $MIN_OBS         \
    --min-speakers $MIN_SPEAKERS

EXIT_CODE=$?

echo "============================================"
echo "End time  : $(date)"
echo "Exit code : $EXIT_CODE"
echo "============================================"

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: trend_analysis.py exited with code $EXIT_CODE" >&2
    exit $EXIT_CODE
fi

echo "Output files:"
ls -lh "$OUTPUT_DIR"
