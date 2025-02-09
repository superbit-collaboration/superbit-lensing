#!/bin/sh
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J TARGET_BRAND
#SBATCH -v
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@northeastern.edu
#SBATCH -o slurm_outfiles/TARGET_BAND.out
#SBATCH -e slurm_outfiles/TARGET_BAND.err

# Load conda environment
source /path/to/your/miniconda3/etc/profile.d/conda.sh 
conda activate sblens
echo "Using Python from: $(which python)"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"

# Check/create the SLURM outfiles directory
dirname="slurm_outfiles"
if [ ! -d "$dirname" ]; then
    echo "Directory $dirname does not exist. Creating now."
    mkdir -p -- "$dirname"
    echo "$dirname created."
else
    echo "Directory $dirname exists."
fi

echo "Proceeding with code..."

# Export all necessary variables (adjust these as needed)
export TARGET="Cluster Name"      # e.g., Abell3411
export BAND="Band"                # e.g., b, g, or u
export DATADIR="/path/to/union"
export OUTDIR="/path/to/union/${TARGET}/${BAND}/out"
export CODEDIR="/path/to/your/superbit-lensing/superbit_lensing"
export REDSHIFT="Cluster Redshift"   # e.g., 0.1687
export DETECTION_BAND="Band"         # usually the same as BAND

# New variables specific to ngmix v2:
export PSF_MODEL="guass"
export GAL_MODEL="gauss"

echo "Output MEDS file will be: $OUTDIR/${TARGET}_${BAND}_meds.fits"

# ----- Step 1: Medsmaker -----
echo "Starting Medsmaker..."
python $CODEDIR/medsmaker/scripts/process_2023.py \
  -outdir=$OUTDIR \
  -psf_mode=psfex \
  -psf_seed=33876300 \
  -detection_bandpass=$DETECTION_BAND \
  -star_config_dir $CODEDIR/medsmaker/configs \
  --meds_coadd $TARGET $BAND $DATADIR

echo "Medsmaker step completed."

# ----- Step 2: Metacalibration using ngmix_v2 -----
echo "Starting Metacalibration..."
python $CODEDIR/metacalibration/ngmix_v2_fit_superbit.py \
  -outdir=$OUTDIR \
  -n 48 \
  -seed=4225165605 \
  -psf_model=$PSF_MODEL \
  -gal_model=$GAL_MODEL \
  --overwrite \
  $OUTDIR/${TARGET}_${BAND}_meds.fits \
  $OUTDIR/${TARGET}_${BAND}_mcal.fits

echo "Metacalibration step completed."

# ----- Step 3: Shear Profiles (Annular Catalog) -----
echo "Starting Annular Catalog creation..."
python $CODEDIR/shear_profiles/make_annular_catalog.py \
  -outdir=$OUTDIR \
  -cluster_redshift=$REDSHIFT \
  -detection_band=$BAND \
  --overwrite \
  -redshift_cat=$DATADIR/catalogs/redshifts/${TARGET}_NED_redshifts.csv \
  $DATADIR $TARGET $OUTDIR/${TARGET}_${BAND}_mcal.fits \
  $OUTDIR/${TARGET}_${BAND}_annular.fits

echo "Annular Catalog step completed."
