# config.sh - Configuration file for submission.sh

# Define cluster-related variables
export cluster_name="sim1"
export band_name="b"
export cluster_redshift="0.245"
export detection_band="b"

# Pipeline control: which step to START from. The pipeline runs this step
# through the end of the chain, in order:
#     gen_mocks -> make_meds -> ngmix -> make_annular
# Set to one of: gen_mocks | make_meds | ngmix | make_annular
#   gen_mocks     -> full pipeline
#   make_meds     -> skip gen_mocks, start at make_meds
#   ngmix         -> only ngmix + make_annular
#   make_annular  -> only make_annular
export START_STEP="gen_mocks"

# Define directories
export DATADIR="/projects/mccleary_group/scratch/simulated_data_full/backcast"
export CODEDIR="/projects/mccleary_group/saha/codes/superbit-lensing"
export OUTDIR="${DATADIR}/${cluster_name}/${band_name}/out"
export LOGDIR="${DATADIR}/${cluster_name}/${band_name}/logs"

# Define ngmix parameters
export ngmix_nruns=1 
export PSF_MODEL="gauss"
export GAL_MODEL="exp"
export reconv_psf="azgauss"
export EXP='backcast'
# Seeds
export master_seed=143
export psf_seed=33876300
export base_ngmix_seed=701428540

export file_ending="azgauss"

# Set Conda environment
export CONDA_ENV="bit_v3"

# Ensure the conda command is available
source ~/.bashrc
conda activate $CONDA_ENV
