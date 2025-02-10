# config.sh - Configuration file for submission.sh

# Define cluster-related variables
export cluster_name="PLCKG287d0p32d9"
export band_name="b"
export cluster_redshift="0.39"
export detection_band="b"

# Define directories
export DATADIR="/work/mccleary_group/saha/codes/superbit-lensing/data"
export CODEDIR="/work/mccleary_group/saha/codes/superbit-lensing"
export OUTDIR="${DATADIR}/${cluster_name}/${band_name}/out"

# Define ngmix parameters
export ngmix_nruns=50 
export PSF_MODEL="coellip5"
export GAL_MODEL="gauss"

# Seeds
export psf_seed=33876300
export base_ngmix_seed=701428540

# Set Conda environment
export CONDA_ENV="bit_v3"

# Ensure the conda command is available
source ~/.bashrc
conda activate $CONDA_ENV