#!/bin/bash

# Source the configuration file
source config.sh

# Define variables
base_arraroutdir="${DATADIR}/${cluster_name}/${band_name}/arr/run"
base_seed=$base_ngmix_seed  # Starting seed value, can be modified as needed
job_script_template="ngmix_array_runs/job1.sh"  # Path to the job script template

# Loop to create and submit jobs based on the ngmix_nruns variable
for i in $(seq 1 $ngmix_nruns)
do
  # Define the new ARRAROUTDIR and seed for this job
  new_arraroutdir="${base_arraroutdir}${i}"
  new_seed=$((base_seed + i))
  job_name="ngmix${i}"  # Job name format ngmix1, ngmix2, ...

  # Define the job script name
  job_script_name="ngmix_array_runs/job${i}.sh"  # Adjusted path

  # Copy the job template to the new job script
  cp $job_script_template $job_script_name

  # Modify ARRAROUTDIR, seed, and job name in the new job script (job${i}.sh)
  sed -i "s|export ARRAROUTDIR=.*|export ARRAROUTDIR=\"$new_arraroutdir\"|" $job_script_name
  sed -i "s|-seed=.*|-seed=$new_seed \\\\|" $job_script_name
  sed -i "s|#SBATCH -J .*|#SBATCH -J $job_name|" $job_script_name

  # Submit the new job script
  sbatch $job_script_name

  echo "Job script $job_script_name created and submitted with seed $new_seed."
done