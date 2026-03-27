#!/bin/bash
#SBATCH --job-name=lexical-temporal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=nv9344@princeton.edu
#SBATCH --output=/scratch/network/nv9344/Thesis/logs/%x_%j.out
#SBATCH --error=/scratch/network/nv9344/Thesis/logs/%x_%j.err

module purge
module load anaconda3/2025.12
conda activate thesis-env

mkdir -p /scratch/network/nv9344/Thesis/logs
cd /scratch/network/nv9344/Thesis/ORFE-Thesis/src

INPUT="/scratch/network/nv9344/Thesis/Thesis-Data/Convokit/lexical_master.csv"
OUTPUT="/scratch/network/nv9344/Thesis/Thesis-Data/Convokit/lexical_temporal.csv"

echo "Starting temporal aggregation pipeline"
echo "Input:  ${INPUT}"
echo "Output: ${OUTPUT}"

python -u run_temporal.py \
  --input  "${INPUT}"    \
  --output "${OUTPUT}"   \
  --cpus   "${SLURM_CPUS_PER_TASK}"
