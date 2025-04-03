export num_sample=2000
export nse_sd=1e-4
export npix=63
export scale=0.141
export exp='superbit'
export outfile='measurement_single_exposure_superbit_v3.fits'

export CODEDIR='/work/mccleary_group/saha/codes/superbit-lensing'

# Set Conda environment
export CONDA_ENV="bit_v3"

# Ensure the conda command is available
source ~/.bashrc
conda activate $CONDA_ENV