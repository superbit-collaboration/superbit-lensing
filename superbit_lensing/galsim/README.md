This directory contains the GalSim scripts used to generate the simulated superbit images used for medsmaking & ngmix/metacalibration.

These scripts require the `galsim` python module to be installed. Excellent documentation for GalSim, including download instructions, is available here: 
https://galsim-developers.github.io/GalSim/_build/html/index.html

The actual scripts you'll want to run are `mock_superBIT_data.py` to generate mock observations, and `mock_superBIT_empirical.cluster.py` 
to generate mock Abell 2218 observations using the 2019 flight PSF. In particular, `mock_superBIT_empirical.cluster.py` will create observations mimicking each exposure in the 2019 flight by stepping through each of the PSFs (see below) and guessing at the exposure time.

The user can set exposure time, number of expected galaxies in observation, output directory, galaxy catalog to use, etc. by supplying a YAML parameter file to `mock_superBIT_data.py`; some examples are in `config_files`.

All scripts rely on the COSMOS_23.5 and COSMOS_25.2 catalogs to be stored in a `data` directory, as well as their supporting files. 
COSMOS catalogs can be accessed here: https://galsim-developers.github.io/GalSim/_build/html/real_gal.html#downloading-the-cosmos-catalog.

*Note*  The `cosmosdir` and `datadir` keywords specify the locations of COSMOS galaxy catalogs and any others used in simulations. These *must be updated*  by the user. 

`mock_superBIT_empirical_cluster.py` will look for a folder containing  PSF models within the `data/` directory.
The `data/` folder in this repository contains the PSFEx fits to the 2019 flight A2218 data (in `data/empirical_2019Flight_PSF/`). `data/` also contains
flight-jitter-model PSFs (in `data/flight_jitter_only_oversampled_1x/` with three different exposure times)


To generate `Chromatic` galaxies convolved with the SuperBIT throughputs in different filters, the mock data scripts expect a throughput
file in the `data/` directory. User can modify locations and names of files being used by setting the `bp_dir` and `bp_file` variables
in the `make_a_galaxy()` and `make_cluster_galaxy` methods.

This repository contains a few debugging scripts as well: `mock_superBIT_data.debug.py` and ` mock_superBIT_data.gaussian.py`. These are mostly includedfor archival reasons; it's unlikely the user will have to invoke them unless they too are debugging some new module. Five files per exposure
time will be created, as well as the accompanying "truth" catalogs.



Run in parallel with MPI
========================

The insertion of galaxies and stars into an image has been parallelized with
MPI, so throwing extra CPUs and RAM at the problem can make things go faster.
To execute, say, 2 processes in parallel run as:
```
mpiexec -n 2 python ./mock_superBIT_data.py config_file=my_params.yaml
```

This depends on mpi4py, though if you don't have it the code will still run
serially. To install it:
 1. On a cluster: an optimized version probably exists. You may have to load a module
 2. With conda: `conda install mpi4py`
 3. With pip: you first need to install an MPI library (eg `apt install openmpi` on ubuntu). Then `pip install mpi4py` will probably work.
