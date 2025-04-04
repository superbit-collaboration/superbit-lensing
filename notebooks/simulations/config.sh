export num_sample=3000
export nse_sd=1e-4
export npix=61
export scale=0.141
export exp='superbit'
export ngmix_model_psf="coellip5"
export ngmix_model_gal="gauss"
export outfile='measurement_single_exposure_superbit_v3_coellip5.fits'

export CODEDIR='/work/mccleary_group/saha/codes/superbit-lensing'

# Set Conda environment
export CONDA_ENV="bit_v3" #change to your preffered env

# Ensure the conda command is available
source ~/.bashrc #change this line if you have miniconda installed
conda activate $CONDA_ENV