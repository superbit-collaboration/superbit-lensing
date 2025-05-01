# SuperBIT Lensing Pipeline

The SimBlaster.sh generates multiple simulations data, based on the config `$CODEDIR/job_sims/galsim_config.yaml`. Then it runs the pipeline on the simulated data. 
### **Setup Instructions**

1. **Edit the SimBlaster File**
   - `DATADIR`: Path to your data directory
   - `CODEDIR`: Path to your code directory

2. **Give Blaster Run**

   Submit the main job script:
   ```sh
   bash SimBlaster.sh {run_name} {num_sims} {exp}
