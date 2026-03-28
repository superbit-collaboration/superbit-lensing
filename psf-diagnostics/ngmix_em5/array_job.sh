#!/bin/bash
#SBATCH -J em5_maps
#SBATCH -p pscomp
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 15:00:00
#SBATCH --array=0-41
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# ================================
# Print job info
# ================================
echo "===================================="
echo "SLURM JOB STARTED"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "===================================="

# ================================
# Avoid thread oversubscription
# ================================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source ~/.bashrc
conda activate bit

python runner.py