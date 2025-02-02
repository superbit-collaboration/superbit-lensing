#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=30:00

# Use 2 nodes with 8 tasks each, for 16 MPI tasks:
#SBATCH -N 3
#SBATCH --tasks-per-node=6
#SBATCH --cpus-per-task=1

# Specify a job name:
#SBATCH -J galsim-mpi-forecast

# Specify an output file
#SBATCH -o mpi-output/galsim-%j.out
#SBATCH -e mpi-output/galsimMPI-%j.out

# Run a command
GALSIM_DIR=/users/jmcclear/data/superbit/superbit-metacal/GalSim
CONFIG_FILE=$GALSIM_DIR/superbit_parameters_debugforecast.yaml

srun --mpi=pmix python $GALSIM_DIR/mock_superBIT_data.py config_file=$CONFIG_FILE
