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

# Path checking
export PATH=$PATH:'/work/mccleary_group/Software/texlive-bin/x86_64-linux'
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



# Check if the Python script ran successfully
'''if [ $? -eq 0 ]; then
    echo "Python script executed successfully. Running multiple_ngmixrun.sh..."
    bash "$SLURM_SUBMIT_DIR/multiple_ngmixrun.sh"
else
    echo "Python script failed. Exiting without running multiple_ngmixrun.sh."
    exit 1
fi'''