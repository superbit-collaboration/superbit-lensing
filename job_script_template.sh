#!/bin/sh
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J TARGET_BRAND_union
#SBATCH -v
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@northeastern.edu
#SBATCH -o slurm_outfiles/TARGET_BAND.out

source /path/to/your/miniconda3/etc/profile.d/conda.sh 

conda activate sbmcal_py11
echo $PATH
echo $PYTHONPATH
dirname="slurm_outfiles"
if [ ! -d "$dirname" ]
then
     echo " Directory $dirname does not exist. Creating now"
     mkdir -p -- "$dirname"
     echo " $dirname created"
 else
     echo " Directory $dirname exists"
 fi

 echo "Proceeding with code..."

export TARGET="Cluster Name" #ex. Abell3411
export BAND="Band" #(b, g, u)
export DATADIR="/path/to/union"
export OUTDIR="/path/to/union/TARGET/BAND/out"
export CODEDIR="/path/to/your/superbit-lensing/superbit-lensing"
export REDSHIFT="See superBIT_target_list" #message amit.m@northeastern.edu for list 
export DETECTION_BAND="Band" #usually b band

## medsmaker
python $CODEDIR/medsmaker/scripts/process_2023.py -outdir $OUTDIR $TARGET $BAND $DATADIR -psf_mode=psfex -psf_seed=33876300 -star_config_dir $CODEDIR/medsmaker/configs -detection_bandpass=$DETECTION_BAND --meds_coadd

## metacalibration
python $CODEDIR/metacalibration/ngmix_fit_superbit3.py $OUTDIR/"${TARGET}_${BAND}_meds.fits" $OUTDIR/"${TARGET}_${BAND}_mcal.fits" -outdir=$OUTDIR -n 48 -seed=4225165605 --overwrite 

## shear_profiles
python $CODEDIR/shear_profiles/make_annular_catalog.py $DATADIR $TARGET $OUTDIR/"${TARGET}_${BAND}_mcal.fits" $OUTDIR/"${TARGET}_${BAND}_annular.fits" -outdir=$OUTDIR --overwrite -cluster_redshift=$REDSHIFT -redshift_cat=$DATADIR/"catalogs/redshifts/${TARGET}_NED_redshifts.csv" -detection_band=$BAND
