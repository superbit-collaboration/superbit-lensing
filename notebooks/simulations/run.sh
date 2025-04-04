#!/bin/bash
source "./config.sh"

python $CODEDIR/superbit_lensing/shearnet/scripts/single_exposure.py --num_samples=$num_sample --sim_exp=$exp --nse_sd=$nse_sd --npix=$npix --scale=$scale --ngmix_model_psf=$ngmix_model_psf --ngmix_model_gal=$ngmix_model_gal --outfilename=$outfile