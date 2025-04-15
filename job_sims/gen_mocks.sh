#!/bin/sh
#SBATCH -t 13:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J gen_mock
#SBATCH -v
#SBATCH -o logs/mockout.log
#SBATCH -e logs/mockerr.log

source "$SLURM_SUBMIT_DIR/config.sh"

which python

python $CODEDIR/superbit_lensing/galsim/mock_superBIT_data_backcast.py $SLURM_SUBMIT_DIR/galsim_config.yaml -run_name ${cluster_name} -data_dir $DATADIR -ncores 18 --clobber --vb