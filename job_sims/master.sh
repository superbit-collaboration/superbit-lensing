#!/bin/bash

# Source the configuration file
source "./config.sh"
mkdir -p $LOGDIR

# ---------------------------------------------------------------------------
# Pipeline control
# Chronology: gen_mocks -> make_meds -> ngmix -> make_annular
# START_STEP (set in config.sh) picks where the chain starts; everything from
# that step to the end is submitted, with SLURM dependencies wired up so that
# only the steps actually submitted are chained together.
# ---------------------------------------------------------------------------
case "${START_STEP:-gen_mocks}" in
  gen_mocks)     START_ORD=1 ;;
  make_meds)     START_ORD=2 ;;
  ngmix)         START_ORD=3 ;;
  make_annular)  START_ORD=4 ;;
  *) echo "ERROR: invalid START_STEP='${START_STEP}' (use gen_mocks|make_meds|ngmix|make_annular)"; exit 1 ;;
esac
echo "Pipeline START_STEP=${START_STEP} (order=$START_ORD)"

JOBID1=""   # gen_mocks
JOBID2=""   # make_meds
job_ids=()  # ngmix jobs

# ----- Step 1: gen_mocks -----
if [ "$START_ORD" -le 1 ]; then
  JOBID1=$(sbatch --output="$LOGDIR/genmock_out.log" --error="$LOGDIR/genmock_err.log" \
                  ./scripts/gen_mocks.sh | awk '{print $4}')
  echo "Submitted gen_mocks: JOBID=$JOBID1"
fi

# ----- Step 2: make_meds -----
if [ "$START_ORD" -le 2 ]; then
  dep=""
  [ -n "$JOBID1" ] && dep="--dependency=afterok:$JOBID1"
  JOBID2=$(sbatch $dep --output="$LOGDIR/make_meds_sims_out.log" --error="$LOGDIR/make_meds_sims_err.log" \
                  ./scripts/make_meds_sims.sh | awk '{print $4}')
  echo "Submitted make_meds: JOBID=$JOBID2 ${dep:+($dep)}"
fi

# ----- Step 3: ngmix -----
if [ "$START_ORD" -le 3 ]; then
  # Define variables
  base_arraroutdir="${DATADIR}/${cluster_name}/${band_name}/arr/run"
  base_seed=$base_ngmix_seed  # Starting seed value, can be modified as needed
  job_script_template="./scripts/ngmix_job_template.sh"

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

    # Depend on make_meds only if it was submitted in this run
    dep=""
    [ -n "$JOBID2" ] && dep="--dependency=afterok:$JOBID2"

    # Submit the new job script and capture the job ID
    job_id=$(sbatch $dep \
                   --output="$LOGDIR/${job_name}_out.log" \
                   --error="$LOGDIR/${job_name}_err.log" \
                   $job_script_name | awk '{print $4}')

    # Add the job ID to our array
    job_ids+=($job_id)

    echo "Job script $job_script_name created and submitted with seed $new_seed, job ID: $job_id ${dep:+($dep)}"
  done
fi

# ----- Step 4: make_annular -----
if [ "$START_ORD" -le 4 ]; then
  # Build dependency on all ngmix jobs, only if any were submitted this run
  dep=""
  if [ ${#job_ids[@]} -gt 0 ]; then
    dependency_string="afterok"
    for job_id in "${job_ids[@]}"; do
      dependency_string+=":$job_id"
    done
    dep="--dependency=$dependency_string"
  fi

  sbatch $dep \
         --time=00:05:00 \
         --job-name=make_annular \
         --output=$LOGDIR/annular_out.log \
         --error=$LOGDIR/annular_err.log \
         --wrap="bash ./scripts/make_annular.sh"
  echo "Submitted make_annular ${dep:+($dep)}"
fi
