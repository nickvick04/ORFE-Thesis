#!/bin/bash
#SBATCH --job-name=lexical-temporal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=48G
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

LEXICAL_MASTER="/scratch/network/nv9344/Thesis/Thesis-Data/Convokit/lexical_master.csv"
OUTPUT_DIR="/scratch/network/nv9344/Thesis/Thesis-Data/TrendResults"
SRC_DIR="/home/nv9344/Thesis/ORFE-Thesis/src"

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
module load anaconda3/2025.12
conda activate thesis-env

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