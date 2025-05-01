# SuperBIT Lensing Pipeline

This pipeline generates simulation data, based on the config `galsim_config.yaml`. Then it runs the pipeline on the simulated data. Finally it provies `{cluster_name}_{band}_annular_combined_with_truth.fits`, which will contain the catalog of obejcts including its truth values, which has been used in the simulations.
### **Setup Instructions**

1. **Edit the Configuration File**

   Open `config.sh` and update the following variables to match your setup:
   - `cluster_name`: Your target galaxy cluster
   - `band_name`: The observation band
   - `DATADIR`: Path to your data directory
   - `CODEDIR`: Path to your code directory
   - `EXP`: What you want to run, forecast or backcast

2. **Give Master Run**

   Submit the main job script:
   ```sh
   bash master.sh
