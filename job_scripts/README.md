# SuperBIT Lensing Pipeline

This pipeline uses `ngmix_v2_fit_superbit.py` and `make_annular_catalog_v2.py`. From the same meds file, it generates multiple mcalfiles using different noise realisation in ngmix. Finally it combines the values from all of them and makes final annular catalogue.
### **Setup Instructions**

1. **Edit the Configuration File**

   Open `config.sh` and update the following variables to match your setup:
   - `cluster_name`: Your target galaxy cluster
   - `band_name`: The observation band
   - `DATADIR`: Path to your data directory
   - `CODEDIR`: Path to your code directory

2. **Run MEDS-making and ngmix processing**

   Submit the main job script:
   ```sh
   sbatch make_meds.sh
- This script runs `process_2023.py` to generate MEDS files.

3. **Submit multiple ngmix runs**

    After the meds file has been created, run the follwing command from your current directory
    ```sh
    bash multiple_ngmixrun.sh
- This will create multiple mcal files in "data/cluster/band/arr/runx"

3. **Make final annular catalogue**

    After all your ngmix runs have been finished, run the following command
    ```sh
    bash make_annular.sh
- This will combine all your mcal files in "data/cluster/band/arr/runx", do id matching and combine the mcal values and finally run make_annular_catalog_v2.py on the combined file. The final annular file will be "Outdir/cluster_band_annular_combined.fits"