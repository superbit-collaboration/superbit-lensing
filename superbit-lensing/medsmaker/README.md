# superbit-ngmix
running ngmix on SuperBIT data

Py3 required to run

The following python packages are required to run:
  - `ngmix` (obviously)
  - `esutil`
  - `meds`
  - `astropy`
  - `fitsio`
  - `astroquery`
  - `psfex` (AstrOmatic & esheldon, see below)
  - `sextractor` (AstrOmatic) 
  - `swarp` (AstrOmatic) 

If using conda to manage python installation, packages can be installed with e.g.

`conda install -c conda-forge esutil`

NOTE: `psfex` needs to be installed with git to be a python module, i.e.,
`git clone https://github.com/esheldon/psfex.git`

