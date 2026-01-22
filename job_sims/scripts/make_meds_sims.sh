#!/bin/sh
#SBATCH -t 13:59:59
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=180G
#SBATCH --partition=short
#SBATCH -J meds
#SBATCH -v
#SBATCH -o logs/medsout.log
#SBATCH -e logs/medserr.log

# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"
# Print Job ID
echo "Submitted job with ID: $SLURM_JOB_ID"

# Load configuration file
source "$SLURM_SUBMIT_DIR/config.sh"

date

which python

# Path checking
export PATH=$PATH:'/projects/mccleary_group/Software/texlive-bin/x86_64-linux'
echo $PATH
echo $PYTHONPATH

echo "Preparing for sims run"

python $CODEDIR/superbit_lensing/galsim/starcat_nedmaker.py ${cluster_name} ${band_name} $DATADIR $CODEDIR

echo "Proceeding with code..."

python $CODEDIR/superbit_lensing/medsmaker/scripts/process_2023.py \
-outdir=$OUTDIR \
-psf_mode=psfex \
-psf_seed=$psf_seed \
-detection_bandpass=${detection_band} \
-star_config_dir $CODEDIR/superbit_lensing/medsmaker/configs \
--meds_coadd ${cluster_name} ${band_name} $DATADIR

# Record end time
end=$(date +%s)
echo "Job finished at: $(date)"

# Compute elapsed time
runtime=$((end - start_time))

# Optionally, print in minutes/hours
echo "Total runtime: $(awk -v t=$runtime 'BEGIN {printf "%.2f minutes (%.2f hours)\n", t/60, t/3600}')"