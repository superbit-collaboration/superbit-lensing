#!/bin/sh
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem=60G
#SBATCH --partition=short
#SBATCH -J gen_mock
#SBATCH -v
#SBATCH -o logs/mockout.log
#SBATCH -e logs/mockerr.log

# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"
# Print Job ID
echo "Submitted job with ID: $SLURM_JOB_ID"

echo "Submitted job from: $SLURM_SUBMIT_DIR"

source "$SLURM_SUBMIT_DIR/config.sh"

which python

if [ "$EXP" == "forecast" ]; then
  python $CODEDIR/superbit_lensing/galsim/unit_tests/nfw_shear/mock_superBIT_data_forecast.py $SLURM_SUBMIT_DIR/galsim_config.yaml -run_name ${cluster_name} -data_dir $DATADIR -ncores 18 --clobber --vb --master_seed $master_seed
elif [ "$EXP" == "backcast" ]; then
  python $CODEDIR/superbit_lensing/galsim/unit_tests/nfw_shear/mock_superBIT_data_backcast_fiducial.py $SLURM_SUBMIT_DIR/galsim_config.yaml -run_name ${cluster_name} -data_dir $DATADIR -ncores 18 --clobber --vb --master_seed $master_seed
fi

# Record end time
end=$(date +%s)
echo "Job finished at: $(date)"

# Compute elapsed time
runtime=$((end - start_time))

# Optionally, print in minutes/hours
echo "Total runtime: $(awk -v t=$runtime 'BEGIN {printf "%.2f minutes (%.2f hours)\n", t/60, t/3600}')"