#!/bin/bash

# Example Usage: bash SimBlaster.sh backcast_sep23 30 backcast 1 42

# Define simulation parameters
RUN_NAME=${1:-"default_run"}
NUM_SIMS=${2:-5}  # Default to 5 simulations if not specified
EXP_TYPE=${3:-"forecast"} # Default experiment type is 'forecast', can be 'backcast'
START_SIM=${4:-1} # Default starting simulation number
START_SEED=${5:-42} # Default starting seed

# Define directories
DATADIR="/scratch/sa.saha/simulated_data"
CODEDIR="/projects/mccleary_group/saha/codes/superbit-lensing"

echo "Starting simulation batch: $RUN_NAME"
echo "Will run $NUM_SIMS simulations starting from sim$START_SIM"
echo "Using experiment type: $EXP_TYPE"

# Create a base directory for initial copy
BASE_DIR="job_sims_${RUN_NAME}_base"

# Copy the job_sims directory from CODEDIR once
echo "Copying ${CODEDIR}/job_sims to ./${BASE_DIR}"

# Check if the directory already exists
if [ -d "$BASE_DIR" ]; then
    echo "Directory $BASE_DIR already exists. Removing it..."
    rm -rf "$BASE_DIR"
fi

# Copy the directory
cp -r "${CODEDIR}/job_sims" "$BASE_DIR"

# Enter the base directory to update config.sh
cd "$BASE_DIR"

# Update DATADIR, CODEDIR, and EXP in config.sh for the base copy
echo "Updating DATADIR, CODEDIR, and EXP in base config.sh"
sed -i "s|export DATADIR=.*|export DATADIR=\"${DATADIR}/${RUN_NAME}\"|" config.sh
sed -i "s|export CODEDIR=.*|export CODEDIR=\"${CODEDIR}\"|" config.sh
sed -i "s|export EXP=.*|export EXP='${EXP_TYPE}'|" config.sh

# Return to the original directory
cd - > /dev/null

# Process each simulation
for (( i=START_SIM; i<START_SIM+NUM_SIMS; i++ )); do
    # Define directory for this simulation
    SIM_DIR="sim${i}"
    
    echo "Processing simulation $i of $NUM_SIMS"
    
    # Copy from the base directory
    echo "Copying ${BASE_DIR} to ./${SIM_DIR}"
    
    # Check if the directory already exists
    if [ -d "$SIM_DIR" ]; then
        echo "Directory $SIM_DIR already exists. Removing it..."
        rm -rf "$SIM_DIR"
    fi
    
    # Copy the directory
    cp -r "$BASE_DIR" "$SIM_DIR"
    
    # Enter the working directory
    cd "$SIM_DIR"
    
    # Calculate the current seed
    CURRENT_SEED=$((START_SEED + i - START_SIM))
    
    echo "Updating cluster_name and master_seed for sim$i"
    
    # Update simulation-specific values in config.sh
    sed -i "s|export cluster_name=.*|export cluster_name=\"sim$i\"|" config.sh
    sed -i "s|export master_seed=.*|export master_seed=$CURRENT_SEED|" config.sh
    
    echo "Running master.sh for sim$i"
    bash master.sh
    
    echo "Completed setup for sim$i"
    
    # Go back to original directory for next iteration
    cd - > /dev/null
    
    echo "------------------------"
done

# Clean up the base directory if desired
echo "Cleaning up base directory"
rm -rf "$BASE_DIR"

echo "All simulations have been submitted!"
echo "You can check job status with: squeue -u $(whoami)"