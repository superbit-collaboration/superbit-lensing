# config.sh - Configuration file for submission.sh

# Define cluster-related variables
export cluster_name="sim_test11"
export band_name="b"
export cluster_redshift="0.245"
export detection_band="b"

# Define directories
export DATADIR="/n23data1/saha/simulated_data"
export CODEDIR="/home/saysaha/codes/superbit-lensing"
export OUTDIR="${DATADIR}/${cluster_name}/${band_name}/out"
export LOGDIR="${DATADIR}/${cluster_name}/${band_name}/logs"

# Define ngmix parameters
export ngmix_nruns=1 
export PSF_MODEL="em5"
export GAL_MODEL="gauss"
export reconv_psf="dilate"
export EXP='forecast' # 'forecast' or 'backcast'
# Seeds
export master_seed=42
export psf_seed=33876300
export base_ngmix_seed=701428540

# Set Conda environment
export CONDA_ENV="bit"

# Ensure the conda command is available
source ~/.bashrc
conda activate $CONDA_ENV
