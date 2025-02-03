#!/bin/sh

# Example cluster submit script for Princeton's Della-Feynman
# The special SBATCH comments are for the SLURM scheduler
# Edit those, the code directory, and the config file for your run
# Then submit with:  sbatch job_script_example_della.sh

#SBATCH --job-name=galsim
#SBATCH --mem=102400
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --partition=physics

# change this to point to your copy, unless you want to risk it with mine
GALSIM_DIR=/home/sbenton/bit/superbit_lensing/GalSim
CONFIG_FILE=$GALSIM_DIR/superbit_parameters.yaml

set -x
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1
source /projects/WCJONES/bit/module_setup.sh
mpiexec -n 10 python $GALSIM_DIR/mock_superBIT_data.py config_file=$CONFIG_FILE
