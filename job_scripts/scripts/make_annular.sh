#!/bin/bash

source "./config.sh"

# First combine mcal files
python $CODEDIR/superbit_lensing/metacalibration/combine_mcal.py \
-nrun=$ngmix_nruns \
-data_dir=$DATADIR \
-run_name=$cluster_name \
-band=$band_name \
-outdir=$OUTDIR

# Then run annular.py code
python $CODEDIR/superbit_lensing/shear_profiles/make_annular_catalog_v2.py \
-outdir=$OUTDIR \
-cluster_redshift=$cluster_redshift \
-detection_band=${band_name} \
--overwrite \
-redshift_cat=$DATADIR/catalogs/redshifts/${cluster_name}_NED_redshifts.csv \
$DATADIR ${cluster_name} $OUTDIR/${cluster_name}_${band_name}_mcal_combined.fits \
$OUTDIR/${cluster_name}_${band_name}_annular_combined.fits