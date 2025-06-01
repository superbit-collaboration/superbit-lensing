#!/bin/sh
#SBATCH -t 13:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J ngmix1
#SBATCH -v
#SBATCH -o logs/ngmixout.log
#SBATCH -e logs/ngmixerr.log

# Load configuration file
source "$SLURM_SUBMIT_DIR/config.sh"

date

which python

export ARRAROUTDIR="/projects/mccleary_group/saha/data/Abell3411/b/arr/run1"

# Ensure ARRAROUTDIR exists
mkdir -p $ARRAROUTDIR

python $CODEDIR/superbit_lensing/metacalibration/ngmix_fit.py \
-outdir=$ARRAROUTDIR \
-n 48 \
-seed=701428541 \
-psf_model=$PSF_MODEL \
-gal_model=$GAL_MODEL \
--overwrite \
--use_coadd \
$OUTDIR/${cluster_name}_${band_name}_meds.fits \
$ARRAROUTDIR/${cluster_name}_${band_name}_mcal.fits 
