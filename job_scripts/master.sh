#!/bin/bash

# Source the configuration file
source "./config.sh"

# Submit first job and capture its ID
# Submit color_color_run.sh first
JOBID_CC=$(sbatch ./scripts/color_color_run.sh | awk '{print $4}')

# Submit make_meds.sh only after color_color_run.sh completes successfully
JOBID1=$(sbatch --dependency=afterok:$JOBID_CC ./scripts/make_meds.sh | awk '{print $4}')

# Define variables
base_arraroutdir="${DATADIR}/${cluster_name}/${band_name}/arr/run"
base_seed=$base_ngmix_seed  # Starting seed value, can be modified as needed
#job_script_template="$SLURM_SUBMIT_DIR/ngmix_array_runs/job1.sh"  # Path to the job script template
job_script_template="./scripts/ngmix_job_template.sh"

# Create an array to store all job IDs
job_ids=()

# Loop to create and submit jobs based on the ngmix_nruns variable
for i in $(seq 1 $ngmix_nruns)
do
  # Define the new ARRAROUTDIR and seed for this job
  new_arraroutdir="${base_arraroutdir}${i}"
  new_seed=$((base_seed + i))
  job_name="ngmix${i}"  # Job name format ngmix1, ngmix2, ...

  # Define the job script name
  job_script_name="./scripts/job${i}.sh" 

  # Copy the job template to the new job script
  cp $job_script_template $job_script_name

  # Modify ARRAROUTDIR, seed, and job name in the new job script (job${i}.sh)
  sed -i "s|export ARRAROUTDIR=.*|export ARRAROUTDIR=\"$new_arraroutdir\"|" $job_script_name
  sed -i "s|-seed=.*|-seed=$new_seed \\\\|" $job_script_name
  sed -i "s|#SBATCH -J .*|#SBATCH -J $job_name|" $job_script_name

  # Submit the new job script and capture the job ID
  job_id=$(sbatch --dependency=afterok:$JOBID1 $job_script_name | awk '{print $4}')
  
  # Add the job ID to our array
  job_ids+=($job_id)

  echo "Job script $job_script_name created and submitted with seed $new_seed, job ID: $job_id"
done

# Create a dependency string with all job IDs including JOBID_CC
dependency_string="afterok:$JOBID_CC"
for job_id in "${job_ids[@]}"
do
  dependency_string+=":$job_id"
done

# Submit a final job that runs make_annular.sh after all ngmix jobs AND color_color_run complete
sbatch --dependency=$dependency_string \
       --time=00:05:00 \
       --job-name=make_annular \
       --output=logs/annular_out.log \
       --error=logs/annular_err.log \
       --wrap="bash ./scripts/make_annular.sh"