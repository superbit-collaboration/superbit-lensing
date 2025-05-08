# SuperBIT Lensing Pipeline

This pipeline uses `ngmix_fit.py` and `make_annular_catalog_v2.py`. From the same meds file, it generates multiple mcalfiles using different noise realisation in ngmix. Finally it combines the values from all of them and makes final annular catalogue.
### **Setup Instructions**

1. **Edit the Configuration File**

   Open `config.sh` and update the following variables to match your setup:
   - `cluster_name`: Your target galaxy cluster
   - `band_name`: The observation band
   - `DATADIR`: Path to your data directory
   - `CODEDIR`: Path to your code directory


2. **Give Master Shape Measurement Run**

   Submit the main job script:
   ```sh
   bash master.sh
3. **Give Color-Color runs**

   Submit jobs from current directory (where your config.sh is)
   ```sh
   sbatch ./scripts/color_color_run.sh