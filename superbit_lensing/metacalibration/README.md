Requirements:

pip install mof
NGMIX - https://github.com/esheldon/ngmix.git

This directory contains the scripts to run `ngmix` and implement the Metacalibration algorithm.

`ngmix_fit_superbit.py`: uses the `MaxMetacalBootstrapper` class in the `ngmix` module to perform metacalibration

`ngmix_fit_superbit2.py`: uses a plain `ngmix.Bootstrapper` instance to perform metacalibration "by hand"

`make_annular_catalog.py`: this script turns the output files from the `ngmix_fit_superbit.py` scripts into a single
catalog ready for weak lensing analysis, with X, Y, RA, Dec, and metacalibrated shapes. Note that the *actual shear responsivity correction* of ngmix shapes
is performed here! 
