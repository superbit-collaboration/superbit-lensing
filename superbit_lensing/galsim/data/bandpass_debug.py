import sys
import os
import math
import numpy
import logging
import time
import galsim
import galsim.des
import pdb
import glob
import scipy
import astropy


im_size = 64
pix_scale = 0.06
#bp_file = os.path.join(galsim.meta_data.share_dir, 'wfc_F814W.dat.gz')
ici = '/Users/jemcclea/Research/GalSim/examples'
bp_file=os.path.join(ici,'lum_throughput.csv')
#throughput_filen='/Users/jemcclea/Research/GalSim/examples/data/bandpasses/lum.csv' 
bp = galsim.Bandpass(bp_file,wave_type='nm',blue_limit = 310,red_limit=1100)#).thin().withZeropoint(25.94)

catfilename = 'real_galaxy_catalog_23.5.fits'
directory='data/COSMOS_23.5_training_sample'
cosmos_cat = galsim.COSMOSCatalog(cat_file_name, dir=directory)

psf = galsim.OpticalPSF(diam=2.4, lam=1000.) # bigger than HST F814W PSF.
indices = np.arange(10)
real_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='real',
                                     noise_pad_size=im_size*pix_scale)
param_gal_list = cosmos_cat.makeGalaxy(indices, gal_type='parametric', chromatic=True)

for ind in indices:
   real_gal = galsim.Convolve(real_gal_list[ind], psf)
   param_gal = galsim.Convolve(param_gal_list[ind], psf)
   im_real = galsim.Image(im_size, im_size)
   im_param = galsim.Image(im_size, im_size)
   real_gal.drawImage(image=im_real, scale=pix_scale)
   param_gal.drawImage(bp, image=im_param, scale=pix_scale)
   im_real.write('im_real_'+str(ind)+'.fits')
   im_param.write('im_param_'+str(ind)+'.fits')

