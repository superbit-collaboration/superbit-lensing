# superbit-lensing
Contains a collection of routines used to perform ngmix fits, including metacalibration, on SuperBIT images.

This repo has recently been significantly refactored into the new `superbit-lensing` module, which you can include in your desired environment by running `python setup.py install` without the need to add the various submodules to your `PYTHONPATH`. The module includes the following four submodules which can be used independently if desired:

  - `galsim`: Contains scripts that generate the simulated SuperBIT observations used for validation and forecasting analyses.
  - `medsmaker`: Contains small modifications to the original superbit-ngmix scripts that make coadd images, runs SExtractor & PSFEx, and creates MEDS files.
  - `metacalibration`: Contains scripts used to run the ngmix/metacalibration algorithms on the MEDS files produced by Medsmaker.
  - `shear-profiles`: Contains scripts to compute the tangential/cross shear profiles and output to a file, as well as plots of the shear profiles.

More detailed descriptions for each stage are contained in their respective directories.

Currently, the pipeline is designed to run on a single cluster at a time. In the future, a meta job script will be developed to automate batch processing across multiple clusters. For now, each job script is configured to process a single cluster, but multiple job scripts can run simultaneously, provided that each is properly set up for its respective cluster. The `job_script_template.sh` file serves as a template, containing detailed instructions on how to configure and execute the pipeline for a specific cluster.


## To build a specific run environment
Before running the pipeline, a specific environemnt for superbit-lensing must be created. 

Create env from yaml (e.g. `sblens.yaml`):

`conda env create --name sblens --file sblens.yaml`

Activate new env:

`conda activate sblens`

cd to meds repo:

`cd /path/to/repos/meds`

Build it:

`python setup.py install`

cd to this repo:

`cd /path/to/repos/superbit-lensing`

pip install repo:

`pip install -e /path/to/repos/superbit-lensing`

## For the experts

If you want to add a new submodule to the pipeline, simply define a new subclass `MyCustomModule(SuperBITModule)` that implements the abstract `run()` function of the parent class and add it to `pipe.MODULE_TYPES` to register it with the rest of the pipeline. You should also implement the desired required & optional parameters that can be present in the module config with the class variables `_req_fields` and `_opt_fields`, which should be lists.

Contact @sweverett at spencer.w.everett@jpl.nasa.gov or @mccleary at j.mccleary@northeastern.edu you have any questions about running the pipeline - or even better, create an issue!

![IMG_7144](https://github.com/user-attachments/assets/8a028b03-fdaa-4fbc-a602-c739941cd503)

