#!/bin/sh
#SBATCH -t 36:59:59
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=180g
#SBATCH --partition=short
#SBATCH -J psf
#SBATCH -v
#SBATCH -o logs/em5out.log
#SBATCH -e logs/em5err.log

# Load configuration file
export CONDA_ENV="bit_v3"

# Ensure the conda command is available
source ~/.bashrc
conda activate $CONDA_ENV

date
which python

python /projects/mccleary_group/saha/codes/superbit-lensing/psf-diagnostics/model_diagnostics.py