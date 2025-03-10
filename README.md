# superbit-lensing
Contains a collection of routines used to perform ngmix fits, including metacalibration, on SuperBIT images.

This repo has recently been significantly refactored into the new `superbit_lensing` module. The module includes the following four submodules which can be used independently if desired:

  - `galsim`: Contains scripts that generate the simulated SuperBIT observations used for validation and forecasting analyses. (Broken, will be fixed soon)
  - `medsmaker`: Contains small modifications to the original superbit-ngmix scripts that make coadd images, runs SExtractor & PSFEx, and creates MEDS files.
  - `metacalibration`: Contains scripts used to run the ngmix/metacalibration algorithms on the MEDS files produced by Medsmaker.
  - `shear-profiles`: Contains scripts to compute the tangential/cross shear profiles and output to a file, as well as plots of the shear profiles.

More detailed descriptions for each stage are contained in their respective directories.

Currently, the pipeline is designed to run on a single cluster at a time. In the future, a meta job script will be developed to automate batch processing across multiple clusters. For now, each job script is configured to process a single cluster, but multiple job scripts can run simultaneously, provided that each is properly set up for its respective cluster. The `job_script_template.sh` file serves as a template, containing detailed instructions on how to configure and execute the pipeline for a specific cluster.


## To build a specific run environment
Before running the pipeline, a specific environemnt for superbit-lensing must be created.

First, clone the repo:
```bash
git clone https://github.com/superbit-collaboration/superbit-lensing.git
```

cd to this repo:
```bash
cd /path/to/repos/superbit-lensing
```

Create env from yml (e.g. `sblens.yml`):
```bash
conda env create --name sblens --file sblens.yml
```

Activate new env:
```bash
conda activate sblens
```

pip install repo:
```bash
pip install -e . 
```

## For the experts

Contact @GeorgeVassilakis at vassilakis.g@northeastern.edu, @MayaAmit at amit.m@northeastern.edu, or @mcclearyj at j.mccleary@northeastern.edu you have any questions about running the pipeline - or even better, create an issue!

![IMG_7144](https://github.com/user-attachments/assets/8a028b03-fdaa-4fbc-a602-c739941cd503)

