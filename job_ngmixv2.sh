#!/bin/sh
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J 3411_b
#SBATCH -v
#SBATCH -o out.log
#SBATCH -e err.log

# Define variables
cluster_name="Abell3411"
band_name="b"
cluster_redshift="0.1687"
detection_band="b"

date

source ~/.bashrc
conda activate bit_v2

which python

###
### Define variables
###

export OUTDIR="/work/mccleary_group/saha/data/${cluster_name}/${band_name}/backups"
export DATADIR="/work/mccleary_group/saha/data"
export CODEDIR="/work/mccleary_group/saha/codes/3d2c93d-jmac/superbit-metacal"
export PSF_MODEL="coellip5"
export GAL_MODEL="gauss"

###
### Path checking
###

echo $OUTDIR/${cluster_name}_${band_name}_meds.fits

export PATH=$PATH:'/work/mccleary_group/Software/texlive-bin/x86_64-linux'
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

###
### Commands!
###
#: '
### Medsmaker
python $CODEDIR/superbit_lensing/medsmaker/scripts/process_2023.py \
-outdir=$OUTDIR \
-psf_mode=psfex \
-psf_seed=33876300 \
-detection_bandpass=${detection_band} \
-star_config_dir $CODEDIR/superbit_lensing/medsmaker/configs \
--meds_coadd ${cluster_name} ${band_name} $DATADIR
#'
echo "process_2023.py code is done, now moving to Metacal"
#: '
### Metacal
python $CODEDIR/superbit_lensing/metacalibration/ngmix_v2_fit_superbit.py \
-outdir=$OUTDIR \
-n 48 \
-seed=4225165605 \
-psf_model=$PSF_MODEL \
-gal_model=$GAL_MODEL \
--overwrite \
$OUTDIR/${cluster_name}_${band_name}_meds.fits \
$OUTDIR/${cluster_name}_${band_name}_mcal.fits #-start 2000 -end 2200
#'
#: '
echo "ngmix_fit_superbit.py code is done, now moving to make_annular_catalog.py"

### Annular & shear
python $CODEDIR/superbit_lensing/shear_profiles/make_annular_catalog.py \
-outdir=$OUTDIR \
-cluster_redshift=$cluster_redshift \
-detection_band=${band_name} \
--overwrite \
-redshift_cat=$DATADIR/catalogs/redshifts/${cluster_name}_NED_redshifts.csv \
$DATADIR ${cluster_name} $OUTDIR/${cluster_name}_${band_name}_mcal.fits \
$OUTDIR/${cluster_name}_${band_name}_annular.fits
#'
