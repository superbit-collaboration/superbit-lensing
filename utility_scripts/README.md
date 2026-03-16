# Utility Scripts

## SimBlaster.sh

The SimBlaster.sh generates multiple simulations data, based on the config `$CODEDIR/job_sims/galsim_config.yaml`. Then it runs the pipeline on the simulated data.

### Setup Instructions

1. **Edit the SimBlaster File**
    - `DATADIR`: Path to your data directory
    - `CODEDIR`: Path to your code directory
2. **Run the Blaster Script**
Submit the main job script:
    ```
    bash SimBlaster.sh {run_name} {num_sims} {exp}
    ```
**Parameters:**
- `run_name`: Name of the simulation run
- `num_sims`: Number of simulations to generate
- `exp`: Experiment type, i.e. forecast/backcast

## data_downloader.py

A utility script for downloading and organizing astronomical data for clusters.

### Features

- Downloads data from hen.astro.utoronto.ca using SCP
- Creates a directory structure for each cluster
- Filters FITS files based on the IMG_QUAL header
- Organizes files by band (u, b, g)
- Deletes files that don't meet quality criteria

### Run the script: 
```
python data_downloader.py {cluster_name} [--username USERNAME] [--data-dir DATA_DIR]
```

**Parameters:**
- `cluster_name`: Name of the cluster (e.g., Abell3411)
- `-username`: (Optional) Your username for hen.astro.utoronto.ca. If not provided, you will be prompted.
- `-data-dir`: (Optional) Base directory for data. Default: /projects/mccleary_group/superbit/union


## dust_utils.py 
A utility function for applying milky way dust extinction corrections using the CSFD dust map


### Run the script: 
you can create a small running script 
```
from utility_scripts.dust_utils import deredden_catalog

deredden_catalog(
    catname='/path/to/input.fits',
    outname='/path/to/output.fits',
    ra_colname='ra',
    dec_colname='dec',
    cluster='Abell3411', # set to None to process all clusters
)
```

### Notes: 
The CSFD dust map data must be downloaded before use. A shared copy lives at `/projects/mccleary_group/amit.m/dust/dust/`. 
By default, deredden_catalog points there automatically (dust_map=False). If you have your own copy downloaded, `pass dust_map=True`.

Default band names are `['m_u', 'm_b', 'm_g']` and default wavelengths are `[3950, 4760, 5970]` Å.


