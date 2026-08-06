#!/bin/bash

source "./config.sh"

# Build the file-ending-dependent names once, so all the python calls use
# consistent values whether or not file_ending was set in config.sh.
if [[ -z "${file_ending:-}" ]]; then
    FILE_ENDING_ARG="fits"
    MCAL_COMBINED_FILE="$OUTDIR/${cluster_name}_${band_name}_mcal_combined.fits"
    ANNULAR_COMBINED_FILE="$OUTDIR/${cluster_name}_${band_name}_annular_combined.fits"
else
    FILE_ENDING_ARG="${file_ending}.fits"
    MCAL_COMBINED_FILE="$OUTDIR/${cluster_name}_${band_name}_mcal_combined.${file_ending}.fits"
    ANNULAR_COMBINED_FILE="$OUTDIR/${cluster_name}_${band_name}_annular_combined.${file_ending}.fits"
fi
echo "file_ending flag: $FILE_ENDING_ARG"
echo "Combined mcal file: $MCAL_COMBINED_FILE"
echo "Combined annular file: $ANNULAR_COMBINED_FILE"

# First combine mcal files
python $CODEDIR/superbit_lensing/metacalibration/combine_mcal.py \
-nrun=$ngmix_nruns \
-data_dir=$DATADIR \
-run_name=$cluster_name \
-band=$band_name \
-reconv_psf=$reconv_psf \
-outdir=$OUTDIR \
--file_ending="$FILE_ENDING_ARG" \

# Then run annular.py code
python $CODEDIR/superbit_lensing/shear_profiles/make_annular_catalog_v2_sims.py \
-outdir=$OUTDIR \
-cluster_redshift=$cluster_redshift \
-detection_band=${band_name} \
-reconv_psf=$reconv_psf \
--overwrite \
$DATADIR ${cluster_name} $MCAL_COMBINED_FILE \
$ANNULAR_COMBINED_FILE