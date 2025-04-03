#!/bin/bash
source "./config.sh"

python $CODEDIR/superbit_lensing/shearnet/scripts/single_exposure.py --num_samples=$num_sample --sim_exp=$exp --nse_sd=1e-5 --npix=$npix --scale=$scale --outfilename=$outfile