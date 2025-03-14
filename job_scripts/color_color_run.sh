#!/bin/sh
#SBATCH -t 13:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J meds
#SBATCH -v
#SBATCH -o out.log
#SBATCH -e err.log

# Load configuration file
source "$SLURM_SUBMIT_DIR/config.sh"
date
which python

python $CODEDIR/superbit_lensing/color/color_color_v2.py ${cluster_name} --datadir=$DATADIR --redshift=$cluster_redshift 