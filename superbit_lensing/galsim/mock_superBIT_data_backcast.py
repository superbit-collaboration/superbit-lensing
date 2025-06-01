# GalSim is opyright (c) 2012-2019 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
##
## TO DO: find a way to write n_obj, exptime, filter, and other useful info to a FITS header

import sys
import os
import math
import logging
import time
import signal
from datetime import datetime, timezone
import galsim
import galsim.des
import galsim.convolve
import pdb
from glob import glob
import pickle
import scipy
import yaml
import numpy as np
import fitsio
from astropy.io import fits
from numpy.random import SeedSequence, default_rng
from functools import reduce
from astropy.table import Table
from argparse import ArgumentParser
from mpi_helper import MPIHelper
from multiprocessing import Pool

import superbit_lensing.utils as utils

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('config_file', action='store', type=str,
                        help='Configuration file for mock sims')
    parser.add_argument('-run_name', action='store', type=str, default='',
                        help='Name of mock simulation run')
    parser.add_argument('-outdir', action='store', type=str,
                        help='Output directory of simulated files')
    parser.add_argument('-data_dir', action='store', type=str,
                        help='Base directory where run_name directory structure will be created')
    parser.add_argument('-ncores', action='store', type=int, default=1,
                        help='Number of cores to use for multiproessing')
    parser.add_argument('--mpi', action='store_true', default=False,
                        help='Use to turn on mpi')
    parser.add_argument('--clobber', action='store_true', default=False,
                        help='Turn on to overwrite existing files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Turn on for verbose prints')
    parser.add_argument('--master_seed', type=int, default=None,
                        help='Optional master seed for RNG initialization')
    return parser.parse_args()

class truth():

    def __init__(self):
        '''
        class to store attributes of a mock galaxy or star
        :x/y: object position in full image
        :ra/dec: object position in WCS --> may need revision?
        :g1/g2: NFW shear moments
        :mu: NFW magnification
        :z: galaxy redshift
        '''

        self.cosmos_index = -1
        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        self.g1 = 0.0
        self.g2 = 0.0
        self.kappa = 0.0
        self.cosmos_g1 =0.0
        self.cosmos_g2 = 0.0
        self.admom_g1 = 0.0
        self.admom_g2 = 0.0
        self.admom_sigma = 0.0
        self.admom_flag = 0.0
        self.mu = 1.0
        self.z = 0.0
        self.fwhm = 0.0
        self.mom_size = 0.0
        self.n = 0.0
        self.hlr = 0.0
        self.scale_h_over_r = 0.0

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

# Set up the try block with a timeout
def calculate_admoms_with_timeout(gal, wcs, image_pos, galaxy_truth, sbparams, timeout_seconds=10):
    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        admoms = galsim.hsm.FindAdaptiveMom(gal.drawImage(wcs=wcs.local(image_pos)))
        galaxy_truth.admom_g1 = admoms.observed_shape.g1
        galaxy_truth.admom_g2 = admoms.observed_shape.g2
        galaxy_truth.admom_sigma = admoms.moments_sigma * sbparams.pixel_scale
        galaxy_truth.admom_flag = 1
        return True
    except (galsim.errors.GalSimError, TimeoutError) as e:
        if isinstance(e, TimeoutError):
            print('sigma calculation timed out')
        else:
            print('sigma calculation failed')
        galaxy_truth.admom_g1 = -9999.
        galaxy_truth.admom_g2 = -9999.
        galaxy_truth.admom_sigma = -9999.
        galaxy_truth.admom_flag = 0
        return False
    finally:
        # Cancel the alarm
        signal.alarm(0)

def matter_power_spectrum(k_abs, h=0.7, ns=1.0, A=1.7e6):
    """
    Calculate the matter power spectrum using the BBKS transfer function.
    
    Parameters:
    -----------
    k_abs : array_like
        Absolute wavenumber values in h Mpc^-1
    h : float, optional
        Dimensionless Hubble parameter (default: 0.7)
    ns : float, optional
        Scalar spectral index (default: 1.0)
    A : float, optional
        Normalization constant (default: 1.7e6)
        
    Returns:
    --------
    P_E : ndarray
        E-mode power spectrum with the same shape as k_abs
    """
    # Convert input to numpy array if it's not already
    k_abs = np.asarray(k_abs)
    
    # Calculate the BBKS transfer function
    Gamma = 0.21  # shape parameter
    q = k_abs / (Gamma * h)
    
    # Handle potential divide-by-zero for k=0
    T = np.ones_like(k_abs)
    nonzero = q > 0
    
    if np.any(nonzero):
        x = np.log(1 + 2.34 * q[nonzero]) / (2.34 * q[nonzero])
        T[nonzero] = x * (1 + 3.89*q[nonzero] + (16.1*q[nonzero])**2 + 
                         (5.46*q[nonzero])**3 + (6.71*q[nonzero])**4)**(-0.25)
    
    # Calculate the power spectrum
    P_E = A * T**2 * k_abs**ns
    
    return P_E

def lss_lensing(nfw_halo, pos, nfw_z_source):
    """
    - For some much-needed tidiness in main(), place the function that shears each galaxy here
    - Usage is borrowed from demo9.py
    - nfw_halo is galsim.NFW() object created in main()
    - pos is position of galaxy in image
    - nfw_z_source is background galaxy redshift
    """

    g1,g2 = nfw_halo.getShear( pos)
    nfw_shear = galsim.Shear(g1=g1,g2=g2)
    nfw_mu = nfw_halo.getMagnification( pos)
    nfw_kappa = nfw_halo.getConvergence(pos)

    if nfw_mu < 0:
        print("Warning: mu < 0 means strong lensing!  Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        print("Warning: mu > 25 means strong lensing!  Using mu=25.")
        nfw_mu = 25

    return nfw_shear, nfw_mu, nfw_kappa

def nfw_lensing(nfw_halo, pos, nfw_z_source):
    """
    - For some much-needed tidiness in main(), place the function that shears each galaxy here
    - Usage is borrowed from demo9.py
    - nfw_halo is galsim.NFW() object created in main()
    - pos is position of galaxy in image
    - nfw_z_source is background galaxy redshift
    """

    g1,g2 = nfw_halo.getShear( pos , nfw_z_source )
    nfw_shear = galsim.Shear(g1=g1,g2=g2)
    nfw_mu = nfw_halo.getMagnification( pos , nfw_z_source )
    nfw_kappa = nfw_halo.getConvergence(pos, nfw_z_source)

    if nfw_mu < 0:
        print("Warning: mu < 0 means strong lensing!  Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        print("Warning: mu > 25 means strong lensing!  Using mu=25.")
        nfw_mu = 25

    return nfw_shear, nfw_mu, nfw_kappa

def make_obj_runner(batch_indices, *args, **kwargs):
    '''
    Handles the batch running of make_obj() over multiple cores
    '''

    res = []
    for i in batch_indices:
        res.append(make_obj(i, *args, **kwargs))

    return res

def make_obj(i, obj_type, *args, **kwargs):
    '''
    Runs the approrpriate "make_a_{obj}" function given object type.
    Particularly useful for multiprocessing wrappers
    '''

    logprint = args[-1]

    func = None

    func_map = {
        'gal': make_a_galaxy,
        'cluster_gal': make_cluster_galaxy,
        'star': make_a_star
    }

    obj_types = func_map.keys()
    if obj_type not in obj_types:
        raise ValueError(f'Object type must be one of {obj_types}!')

    func = func_map[obj_type]


    try:
        obj_index = int(i)
        logprint(f'Starting {obj_type} {i}')
        stamp, truth = func(*args, **kwargs,obj_index=i)
        logprint(f'{obj_type} {i} completed succesfully')
    except galsim.errors.GalSimError:
        logprint(f'{obj_type} {i} has failed, skipping...')
        return i, None, None

    return i, stamp, truth

def combine_objs(make_obj_outputs, full_image, truth_catalog, exp_num):
    '''
    (i, stamps, truths) are the output of make_obj
    exp_num is the exposure number. Only add to truth table if == 1
    '''

    # flatten outputs into 1 list
    make_obj_outputs = [item for sublist in make_obj_outputs
                        for item in sublist]

    for i, stamp, truth in make_obj_outputs:

        if (stamp is None) or (truth is None):
            continue

        # Find the overlapping bounds:
        bounds = stamp.bounds & full_image.bounds

        # Finally, add the stamp to the full image.
        try:
            full_image[bounds] += stamp[bounds]
        except galsim.errors.GalSimBoundsError as e:
            print(e)

        this_flux = np.sum(stamp.array)

        if exp_num == 1:
            row = [i, truth.cosmos_index, truth.x, truth.y,
                   truth.ra, truth.dec,
                   truth.g1, truth.g2,
                   truth.mu, truth.kappa, truth.cosmos_g1, truth.cosmos_g2, 
                   truth.admom_g1, truth.admom_g2, truth.admom_sigma, truth.admom_flag, truth.z,
                   this_flux, truth.fwhm, truth.mom_size,
                   truth.n, truth.hlr, truth.scale_h_over_r,
                   truth.obj_class
                   ]
            truth_catalog.addRow(row)

    return full_image, truth_catalog

def make_a_galaxy(ud, wcs, affine, cosmos_cat, nfw, psf, sbparams, logprint, obj_index=None):
    """
    Method to make a single galaxy object and return stamp for
    injecting into larger GalSim image
    """

    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.
    dec = sbparams.center_dec + (ud()-0.5) * sbparams.image_ysize_arcsec * galsim.arcsec
    ra = sbparams.center_ra + (ud()-0.5) * sbparams.image_xsize_arcsec / np.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)

    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)

    # We also need this in the tangent plane, which we call "world coordinates" here.
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)
    logprint.debug('created galaxy position')

    ## Draw a Galaxy from scratch
    ## Note units of sbparams.gain is assumed to be be e-/ADU.
    index = int(np.floor(ud()*len(cosmos_cat)))
    gal_z = cosmos_cat[index]['ZPDF']
    gal_flux = cosmos_cat[index][sbparams.bandpass] * sbparams.exp_time / sbparams.gain
    phi = cosmos_cat[index]['c10_sersic_fit_phi'] * galsim.radians
    q = cosmos_cat[index]['c10_sersic_fit_q']
    g1_cosmos = cosmos_cat[index]['c10_gaussian_g1']
    g2_cosmos = cosmos_cat[index]['c10_gaussian_g2']
    # Cosmos HLR is in units of HST pix, convert to arcsec.
    half_light_radius=cosmos_cat[index]['c10_sersic_fit_hlr']*0.03*np.sqrt(q)
    sigma = cosmos_cat[index]["c10_gaussian_sigma"]
    n = cosmos_cat[index]['c10_sersic_fit_n']
    logprint.debug(f'galaxy z={gal_z} flux={gal_flux} hlr={half_light_radius} ' + \
                   f'sersic_index={n}')

    # Sersic class requires index n >= 0.3
    if (n < 0.3):
        n = 0.3

    gal = galsim.Sersic(n = n,
                        flux = gal_flux,
                        half_light_radius = half_light_radius)
    #gal  = galsim.Gaussian(sigma=sigma, flux=gal_flux)

    gal = gal.shear(g1 = g1_cosmos, g2 = g2_cosmos)
    logprint.debug('created galaxy')

    ## Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    ## Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    ## Get the reduced shears and magnification at this point
    try:
        nfw_shear, mu, kappa = nfw_lensing(nfw, uv_pos, gal_z)
        g1=nfw_shear.g1; g2=nfw_shear.g2
        gal = gal.lens(g1, g2, mu)
    except galsim.errors.GalSimError:
        logprint(f'could not lens galaxy at z = {gal_z}, setting default values...')
        g1 = 0.0; g2 = 0.0
        mu = 1.0
        kappa = 0.0

    this_psf = psf.getPSF(image_pos)
    logprint.debug("obtained PSF at image position")

    gsp=galsim.GSParams(maximum_fft_size=32768)
    final = galsim.Convolve([this_psf, gal],gsparams=gsp)
    logprint.debug("Convolved galaxy and PSF at image position")
    #final = galsim.Convolve([psf, gal])

    #logprint.debug('Convolved star and PSF at galaxy position')

    stamp = final.drawImage(wcs=wcs.local(image_pos))
    stamp.setCenter(image_pos.x,image_pos.y)
    logprint.debug('drew & centered galaxy!')
    galaxy_truth=truth()
    galaxy_truth.cosmos_index = cosmos_cat[index]['NUMBER']
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=image_pos.x; galaxy_truth.y=image_pos.y
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.kappa = kappa
    galaxy_truth.cosmos_g1 = g1_cosmos; galaxy_truth.cosmos_g2 = g2_cosmos
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    galaxy_truth.flux = stamp.added_flux
    galaxy_truth.n = n; galaxy_truth.hlr = half_light_radius
    #galaxy_truth.inclination = inclination.deg # storing in degrees for human readability
    galaxy_truth.scale_h_over_r = q
    galaxy_truth.obj_class = 'gal'

    logprint.debug('created truth values')

    calculate_admoms_with_timeout(gal, wcs, image_pos, galaxy_truth, sbparams, timeout_seconds=10)

    logprint.debug('stamp made, moving to next galaxy')
    return stamp, galaxy_truth

def make_cluster_galaxy(ud, wcs, affine, centerpix, cluster_cat, psf, sbparams, logprint, obj_index=None):
    """
    Method to make a single galaxy object and return stamp for
    injecting into larger GalSim image

    Galaxies defined here are not lensed, and are magnified to
    look more "cluster-like."
    """

    # Choose a random position within 200 pixels of the sky_center
    radius = 200
    max_rsq = radius**2
    while True:  # (This is essentially a do..while loop.)
        x = (2.*ud()-1) * radius
        y = (2.*ud()-1) * radius
        rsq = x**2 + y**2

        if rsq <= max_rsq: break

    # We will need the image position as well, so use the wcs to get that,
    # plus a small gaussian jitter so cluster doesn't look too box-like
    image_pos = galsim.PositionD(x+centerpix.x+(ud()-0.5)*100,y+centerpix.y+(ud()-0.5)*100)
    world_pos = wcs.toWorld(image_pos)
    ra=world_pos.ra; dec = world_pos.dec

    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)

    # Fixed redshift for cluster galaxies
    gal_z = sbparams.nfw_z_halo
    g1 = 0.0; g2 = 0.0
    mu = 1.0

    '''index = int(np.floor(ud()*len(cosmos_cat)))
    gal_flux = cosmos_cat[index][sbparams.bandpass] * sbparams.exp_time / sbparams.gain
    phi = cosmos_cat[index]['c10_sersic_fit_phi'] * galsim.radians
    q = cosmos_cat[index]['c10_sersic_fit_q']    
    half_light_radius=cosmos_cat[index]['c10_sersic_fit_hlr']*0.03*np.sqrt(q)
    n = cosmos_cat[index]['c10_sersic_fit_n']  
    # Sersic class requires index n >= 0.3
    if (n < 0.3):
        n = 0.3

    gal = galsim.Sersic(n = n,
                        flux = gal_flux,
                        half_light_radius = half_light_radius)

    gal = gal.shear(q = q, beta = phi)'''  
    # Create galaxy    
    gal = cluster_cat.makeGalaxy(gal_type='parametric', rng=ud)
    logprint.debug('created cluster galaxy')

    # Apply a random rotation
    theta = ud()*2.0*np.pi*galsim.radians
    gal = gal.rotate(theta)

    # The "magnify" is just for drama
    gal *= sbparams.flux_scaling
    gal.magnify(2)
    logprint.debug(f'rescaled galaxy with scaling factor {sbparams.flux_scaling}')

    this_psf = psf.getPSF(image_pos)
    logprint.debug("obtained PSF at image position")

    gsp=galsim.GSParams(maximum_fft_size=32768)
    final = galsim.Convolve([this_psf, gal],gsparams=gsp)

    logprint.debug('Convolved star and PSF at galaxy position')


    # Draw galaxy image
    this_stamp_image = galsim.Image(128, 128,wcs=wcs.local(image_pos))
    #cluster_stamp = final.drawImage(bandpass,image=this_stamp_image)
    cluster_stamp = final.drawImage(image=this_stamp_image)

    #cluster_stamp.setCenter(ix_nominal,iy_nominal)
    cluster_stamp.setCenter(image_pos.x,image_pos.y)

    logprint.debug('drew & centered galaxy!')

    cluster_galaxy_truth=truth()
    cluster_galaxy_truth.ra=ra.deg; cluster_galaxy_truth.dec=dec.deg
    #cluster_galaxy_truth.x=ix_nominal; cluster_galaxy_truth.y=iy_nominal
    cluster_galaxy_truth.x=image_pos.x; cluster_galaxy_truth.y=image_pos.y
    cluster_galaxy_truth.g1=g1; cluster_galaxy_truth.g2=g2
    cluster_galaxy_truth.mu = mu; cluster_galaxy_truth.z = gal_z
    cluster_galaxy_truth.flux = cluster_stamp.added_flux
    cluster_galaxy_truth.obj_class = 'cluster_gal'
    logprint.debug('created truth values')

    calculate_admoms_with_timeout(gal, wcs, image_pos, cluster_galaxy_truth, sbparams, timeout_seconds=10)

    return cluster_stamp, cluster_galaxy_truth


def make_a_star(ud, pud, k, wcs, affine, psf, sbparams, logprint, obj_index=None):
    """
    makes a star-like object for injection into larger image.
    """

    # Choose a random RA, Dec around the sky_center.
    dec = sbparams.center_dec + (ud()-0.5) * sbparams.image_ysize_arcsec * galsim.arcsec
    ra = sbparams.center_ra + (ud()-0.5) * sbparams.image_xsize_arcsec / np.cos(dec) * galsim.arcsec
    world_pos = galsim.CelestialCoord(ra,dec)

    # We will need the image position as well, so use the wcs to get that
    image_pos = wcs.toImage(world_pos)

    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)

    # Draw star flux at random; based on either a semi-analytic distribution or GAIA stars
    # Default to blue stars, which are plenty bright

    index = obj_index - 1

    if sbparams.star_cat is not None:
        if sbparams.bandpass in ['crates_lum', 'crates_shape']:
            star_flux = sbparams.star_cat['bitflux_electrons_lum'][index]

        elif sbparams.bandpass in ['crates_b', 'crates_u']:
            star_flux = sbparams.star_cat['bitflux_electrons_b'][index]

        else:
            raise NotImplementedError('Star catalog sampling only implemented ' +\
                                      'for crates_shape and crates_b!')

        star_flux *= sbparams.exp_time / sbparams.gain

    else:
        pud = np.random.default_rng()
        p = pud.power(0.4)
        flux_p = (10/p) - 10.
        star_flux = flux_p

        if sbparams.bandpass=='crates_b':
            star_flux *= 0.8271672
        else:
            raise NotImplementedError('Star power law only implemented for crates_b!')

    # Generate PSF at location of star, convolve with optical model to make a star
    deltastar = galsim.DeltaFunction(flux=star_flux)
    this_psf = psf.getPSF(image_pos)
    logprint.debug("obtained PSF at image position")

    gsp=galsim.GSParams(maximum_fft_size=32768)
    star = galsim.Convolve([this_psf, deltastar],gsparams=gsp)
    logprint.debug("Convolved galaxy and PSF at image position")
    #star = galsim.Convolve([psf, deltastar])

    star_stamp = star.drawImage(wcs=wcs.local(image_pos)) # before it was scale = 0.206, and that was bad!
    star_stamp.setCenter(image_pos.x, image_pos.y)

    star_truth = truth()
    star_truth.ra = ra.deg; star_truth.dec = dec.deg
    star_truth.x = image_pos.x; star_truth.y = image_pos.y
    star_truth.obj_class = 'star'

    try:
        star_truth.fwhm = star.calculateFWHM()
    except galsim.errors.GalSimError:
        logprint.debug('fwhm calculation failed')
        star_truth.fwhm =- 9999.0

    try:
        admoms = galsim.hsm.FindAdaptiveMom(star_stamp)
        star_truth.admom_g1 = admoms.observed_shape.g1
        star_truth.admom_g2 = admoms.observed_shape.g2
        star_truth.admom_sigma = admoms.moments_sigma * sbparams.pixel_scale
        #galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
        star_truth.admom_flag = 1
    except galsim.errors.GalSimError:
        logprint.debug('sigma calculation failed')
        star_truth.mom_size=-9999.

    return star_stamp, star_truth

class SuperBITParameters:
    __req_params = ['run_name', 'outdir']
    __req_defaults = ['', None]

    def __init__(self, config_file, logprint, args=None):
        """
        Initialize default params and overwirte with config_file params and / or commmand line
        parameters.
        """

        self.logprint = logprint
        self.use_dir_structure = True  # Default to using directory structure
        
        self.logprint(f'Loading parameters from {config_file}')
        self._load_config_file(config_file)

        # Check for command line args to overwrite config_file and / or defaults
        if args is not None:
            self._load_command_line(args)
            
            # If outdir was explicitly provided in command line args, bypass directory structure
            if 'outdir' in args and args.outdir is not None:
                self.use_dir_structure = False
                self.logprint('Outdir explicitly provided, bypassing directory structure creation')

        # Check that certain params are set either on command line or in config
        utils.check_req_params(self, self.__req_params, self.__req_defaults)

        self._set_seeds()

        # Setup stellar injection
        self._setup_stars()
        
        # Create directory structure if needed
        self._create_outdir_structure()

        return

    def _load_config_file(self, config_file):
        """
        Load parameters from configuration file. Only parameters that exist in the config_file
        will be overwritten.
        """
        with open(config_file) as fsettings:
            config = yaml.load(fsettings, Loader=yaml.FullLoader)
        self._load_dict(config)

        return

    def _args_to_dict(self, args):
        """
        Converts argparse command line arguments to a dictionary.
        """
        return vars(args)
        # d = {}
        # for arg in args[1:]:
        #     optval = arg.split("=", 1)
        #     option = optval[0]
        #     value = optval[1] if len(optval) > 1 else None
        #     d[option] = value

        # return d

    def _load_command_line(self, args):
        """
        Load parameters from the command line argumentts. Only parameters that are provided in
        the command line will be overwritten.
        """
        self.logprint('Processing command line args')
        # Parse arguments here
        self._load_dict(self._args_to_dict(args))

        return

    def _load_dict(self, d):
        """
        Load parameters from a dictionary.
        """
        ignore = ['config_file', 'vb']
        for (option, value) in d.items():
            if option in ignore:
                # config file already processed above
                continue
            # This would be much simpler:
            # else:
                # setattr(self, option, value)
            elif option == "pixel_scale":
                self.pixel_scale = float(value)
            elif option == "sky_bkg":
                self.sky_bkg = float(value)
            elif option == "sky_sigma":
                self.sky_sigma = float(value)
            elif option == "gain":
                self.gain = float(value)
            elif option == "read_noise":
                self.read_noise = float(value)
            elif option == "dark_current":
                self.dark_current = float(value)
            elif option == "dark_current_std":
                self.dark_current_std = float(value)
            elif option == "image_xsize":
                self.image_xsize = int(value)
            elif option == "image_ysize":
                self.image_ysize = int(value)
            elif option == "center_ra":
                self.center_ra = float(value) * galsim.hours
            elif option == "center_dec":
                self.center_dec = float(value) * galsim.degrees
            elif option == "nexp":
                self.nexp = int(value)
            elif option == "exp_time":
                self.exp_time = float(value)
            elif option == "nobj":
                self.nobj = int(value)
            elif option == "nclustergal":
                self.nclustergal = int(value)
            elif option == "nstars":
                self.nstars = value
            elif option == "tel_diam":
                self.tel_diam = float(value)
            elif option == "lam":
                self.lam = float(value)
            elif option == "mass":
                self.mass = float(value)
            elif option == "nfw_conc":
                self.nfw_conc = float(value)
            elif option == "nfw_z_halo":
                self.nfw_z_halo = float(value)
            elif option == "omega_m":
                self.omega_m = float(value)
            elif option == "omega_lam":
                self.omega_lam = float(value)
            elif option == "cosmosdir":
                self.cosmosdir = str(value)
            elif option == "datadir":
                self.datadir = str(value)
            elif option == "cat_file_name":
                self.cat_file_name = str(value)
            elif option == "fit_file_name":
                self.fit_file_name = str(value)
            elif option == "cluster_cat_name":
                self.cluster_cat_name = str(value)
            elif option == "star_cat_name":
                self.star_cat_name = str(value)
            elif option == "bp_file":
                self.bp_file = str(value)
            elif option == "outdir":
                self.outdir = str(value)
            elif option == "data_dir":
                self.data_dir = str(value)
            elif option == "master_seed":
                if value is None:
                    self.master_seed = None
                else:
                    self.master_seed = int(value)
            elif option == "dithering_seed":
                self.dithering_seed = int(value)
            elif option == "nstruts":
                self.nstruts = int(value)
            elif option == "nstruts":
                self.nstruts = int(value)
            elif option == "strut_thick":
                self.strut_thick = float(value)
            elif option == "strut_theta":
                self.strut_theta = float(value)
            elif option == "obscuration":
                self.obscuration = float(value)
            elif option == "bandpass":
                self.bandpass=str(value)
            elif option == "jitter_fwhm":
                self.jitter_fwhm=float(value)
            elif option == "run_name":
                self.run_name=str(value)
            elif option == "clobber":
                self.clobber=bool(value)
            elif option == "mpi":
                self.mpi = bool(value)
            elif option == "ncores":
                self.ncores = int(value)
            elif option == "use_optics":
                self.use_optics = bool(value)
            elif option == "sample_gaia_cats":
                self.sample_gaia_cats = bool(value)
            elif option == "gaia_dir":
                self.gaia_dir = str(value)
            elif option == "noise_seed":
                try:
                    self.noise_seed = int(value)
                except:
                    self.noise_seed = None
            elif option == "galobj_seed":
                try:
                    self.galobj_seed = int(value)
                except:
                    self.galobj_seed = None
            elif option == "cluster_seed":
                try:
                    self.cluster_seed = int(value)
                except:
                    self.cluster_seed = None
            elif option == "stars_seed":
                try:
                    self.stars_seed = int(value)
                except:
                    self.stars_seed = None
            else:
                raise ValueError("Invalid parameter \"%s\" with value \"%s\"" % (option, value))

        # Derive image parameters from the base parameters
        self.image_xsize_arcsec = self.image_xsize * self.pixel_scale
        self.image_ysize_arcsec = self.image_ysize * self.pixel_scale
        self.center_coords = galsim.CelestialCoord(self.center_ra,self.center_dec)
        self.strut_angle = self.strut_theta * galsim.degrees

        # OUR NEW CATALOG IS ALREADY SCALED TO SUPERBIT 0.5 m MIRROR.
        # Scaling used for cluster galaxies, which are drawn from default GalSim-COSMOS catalog
        hst_eff_area = 2.4**2 #* (1.-0.33**2)
        sbit_eff_area = self.tel_diam**2 #* (1.-0.380**2)
        self.flux_scaling = (sbit_eff_area/hst_eff_area) * self.exp_time
        if not hasattr(self,'jitter_fwhm'):
            self.jitter_fwhm = 0.1

        return

    def _setup_stars(self):
        valid_args = ['nstars', 'star_cat_name', 'sample_gaia_cats', 'gaia_dir']

        for arg in valid_args:
            if not hasattr(self, arg):
                setattr(self, arg, None)

        assert (self.nstars is not None) or \
               (self.star_cat_name is not None) or \
               (self.sample_gaia_cats is not None)

        if (self.star_cat_name is not None) and (self.sample_gaia_cats is not None):
            raise AttributeError('Cannot set both `star_cat_name` and ' +\
                                 '`sample_gaia_cats`!')

        # if sampling from a GAIA cat, do that first
        if self.sample_gaia_cats is True:
            if self.gaia_dir is None:
                raise AttributeError('Must set `gaia_dir` if sampling from gaia cats!')

            gaia_cats = glob(f'{self.gaia_dir}/GAIA*.csv')
            sample_gaia_rng = np.random.default_rng(self.stars_seed)
            self.star_cat_name = sample_gaia_rng.choice(gaia_cats)

        if self.star_cat_name is not None:
            star_fname = os.path.join(self.datadir, self.star_cat_name)
            self.star_cat = Table.read(star_fname)
        else:
            self.star_cat = None

        if self.nstars is None:
            self.nstars = len(self.star_cat)

        return

    def _set_seeds(self):
        '''
        Handle the setting of various seeds
        '''

        seed_types = ['galobj_seed', 'cluster_seed', 'stars_seed', 'noise_seed',
                      'dithering_seed']
        Nseeds = len(seed_types)
        needed_seeds = Nseeds

        master_seed = None
        seeds = dict(zip(seed_types, Nseeds*[None]))

        if hasattr(self, 'master_seed'):
            # can't pass separate seeds if a master seed is passed
            for seed in seed_types:
                if hasattr(self, seed):
                    raise AttributeError(f'Cannot set {seed} if a ' +\
                                         'master_seed is set!')
            master_seed = self.master_seed

        else:
            for seed_name in seeds.keys():
                if hasattr(self, seed_name):
                    seeds[seed_name] = getattr(self, seed_name)
                    needed_seeds -= 1

        assert needed_seeds >= 0, "needed_seeds should never be negative"
        if needed_seeds > 0:
            # Create safe, independent obj seeds given a master seed
            new_seeds = utils.generate_seeds(
                needed_seeds, master_seed=master_seed
                )

            k = 0
            for seed_name, val in seeds.items():
                if val is None:
                    val = new_seeds.pop()
                    seeds[seed_name] = val
                    setattr(self, seed_name, val)
                    k += 1

            assert k == needed_seeds
            assert len(new_seeds) == 0
            assert not (None in dict(seeds).values())

        for seed_name, val in seeds.items():
            print(seed_name, val)

        return

    # TODO: This should be updated to be sensible. see issue #10
    def make_mask_files(self, logprint, clobber, cal_dir=None):
        """
        Create mask files for the simulation.
        
        Parameters:
        -----------
        logprint : function
            Function to log messages
        clobber : bool
            Whether to overwrite existing files
        cal_dir : str, optional
            Directory to save mask files. If None, uses self.run_outdir
        """
        # Determine directory to use
        base_dir = cal_dir if cal_dir is not None else self.run_outdir
        
        mask_dir = os.path.join(base_dir, 'mask_files')
        mask_file = 'forecast_mask.fits'
        mask_outfile = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
            self.logprint(f'Created mask directory: {mask_dir}')

        if os.path.exists(mask_outfile):
            self.logprint('Removing old mask file...')
            os.remove(mask_outfile)

        Nx = self.image_xsize
        Ny = self.image_ysize

        # x and y are flipped in fits convention vs. np array
        mask = np.zeros((Ny, Nx), dtype='i4')

        mask_fits = fitsio.FITS(mask_outfile, 'rw')

        for ext in range(self.nexp):
            try:
                mask_fits.write(mask, ext=ext, clobber=clobber)
                logprint(f'Wrote mask to {mask_outfile}')
            except OSError as e:
                logprint(f'OSError: {e}')
                raise e

        return

    # TODO: This should be updated to be sensible. see issue #10
    def make_weight_files(self, logprint, clobber, cal_dir=None):
        """
        Create weight files for the simulation.
        
        Parameters:
        -----------
        logprint : function
            Function to log messages
        clobber : bool
            Whether to overwrite existing files
        cal_dir : str, optional
            Directory to save weight files. If None, uses self.run_outdir
        """
        # Determine directory to use
        base_dir = cal_dir if cal_dir is not None else self.run_outdir
        
        weight_dir = os.path.join(base_dir, 'weight_files')
        weight_file = 'forecast_weight.fits'
        weight_outfile = os.path.join(weight_dir, weight_file)

        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
            self.logprint(f'Created weight directory: {weight_dir}')

        if os.path.exists(weight_outfile):
            self.logprint('Removing old weight file...')
            os.remove(weight_outfile)

        Nx = self.image_xsize
        Ny = self.image_ysize

        # x and y are flipped in fits convention vs. np array
        weight = np.ones((Ny, Nx), dtype='f8')

        weight_fits = fitsio.FITS(weight_outfile, 'rw')

        for ext in range(self.nexp):
            try:
                weight_fits.write(weight, ext=ext, clobber=clobber)
                logprint(f'Wrote weight to {weight_outfile}')
            except OSError as e:
                logprint(f'OSError: {e}')
                raise e

        return

    def _create_outdir_structure(self):
        """
        Create the required directory structure for the output files.
        Directory structure follows the pattern:
        
        data_dir/run_name/
        data_dir/run_name/b/cal
        data_dir/run_name/b/cat
        data_dir/run_name/b/coadd
        data_dir/run_name/b/out
        data_dir/run_name/det/cat
        data_dir/run_name/det/coadd
        data_dir/run_name/g/cal
        data_dir/run_name/g/cat
        data_dir/run_name/g/coadd
        data_dir/run_name/g/out
        data_dir/run_name/u/cal
        data_dir/run_name/u/cat
        data_dir/run_name/u/coadd
        data_dir/run_name/u/out
        
        If use_dir_structure is False, skips creating directories and just 
        ensures outdir exists and sets run_outdir to outdir.
        
        If data_dir is specified and use_dir_structure is True, it will be used as the base directory,
        otherwise outdir will be used as base.
        """
        # When not using directory structure, just ensure outdir exists and return
        if not self.use_dir_structure:
            if not os.path.exists(self.outdir):
                self.logprint(f'Creating output directory: {self.outdir}')
                os.makedirs(self.outdir)
            
            # Set run_outdir to outdir so all files get written there
            self.run_outdir = self.outdir
            self.logprint(f'Using {self.outdir} as output directory')
            return
            
        # Otherwise create the full directory structure
        # Determine base directory
        base_dir = getattr(self, 'data_dir', None) or self.outdir
        
        if not os.path.exists(base_dir):
            self.logprint(f'Creating base directory: {base_dir}')
            os.makedirs(base_dir)
            
        # Define the directory structure
        bands = ['b', 'g', 'u']
        subdirs = ['cal', 'cat', 'coadd', 'out']
        det_subdirs = ['cat', 'coadd']
        
        # Create main run directory
        run_dir = os.path.join(base_dir, self.run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            self.logprint(f'Created run directory: {run_dir}')
            
        # Create band directories with subdirectories
        for band in bands:
            band_dir = os.path.join(run_dir, band)
            if not os.path.exists(band_dir):
                os.makedirs(band_dir)
                
            for subdir in subdirs:
                full_dir = os.path.join(band_dir, subdir)
                if not os.path.exists(full_dir):
                    os.makedirs(full_dir)
                    self.logprint(f'Created directory: {full_dir}')
        
        # Create detection directories
        det_dir = os.path.join(run_dir, 'det')
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
            
        for subdir in det_subdirs:
            full_dir = os.path.join(det_dir, subdir)
            if not os.path.exists(full_dir):
                os.makedirs(full_dir)
                self.logprint(f'Created directory: {full_dir}')
                
        # Store the final run directory path for later use
        self.run_outdir = run_dir
        self.logprint(f'Completed directory structure setup at {run_dir}')

# function to help with reducing MPI results from each process to single result
def combine_images(im1, im2):
    """Combine two galsim.Image objects into one."""
    # easy since they support +. Try using in-place operation to reduce memory
    im1 += im2
    return im1

def combine_catalogs(t1, t2):
    """Combine two galsim.OutputCatalog objects into one"""
    # as far as I can tell, they expose no way of doing this aside from messing
    # with the internal lists directly.
    t1.rows.extend(t2.rows)
    t1.sort_keys.extend(t2.sort_keys)
    return t1

def main(args):
    """
    Make images using model PSFs and galaxy cluster shear:
      - The galaxies come from a processed COSMOS 2015 Catalog, scaled to match
        anticipated SuperBIT 2023 observations
      - The galaxy shape parameters are assigned in a probabilistic way through matching
        galaxy fluxes and redshifts to similar GalSim-COSMOS galaxies (see A. Gill+ 2023)
    """

    config_file = args.config_file
    run_name = args.run_name
    mpi = args.mpi
    ncores = args.ncores
    clobber = args.clobber
    vb = args.vb

    start_time = time.time()

    # Determine where to create the log file
    logfile = f'generate_mocks.log'
    temp_log = False
    logdir = None
    
    if args.outdir is not None:
        # When outdir is provided, save log there directly
        logdir = args.outdir
    elif args.data_dir is not None:
        # When data_dir is provided, we'll want to save in the band/out dir later
        # but we don't know the band yet, so create a temporary log and move it later
        temp_log = True
        
    # Initialize the log
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    if mpi is True:
        M = MPIHelper()

    # Define some parameters we'll use below.
    sbparams = SuperBITParameters(config_file, logprint, args=args)

    # if galsim config run_name does not agree with passed arg,
    # it should be overridden (to match pipeline run_name)
    assert run_name == sbparams.run_name

    # Set up the NFWHalo:
    nfw = galsim.NFWHalo(mass=sbparams.mass, conc=sbparams.nfw_conc, redshift=sbparams.nfw_z_halo,
                     omega_m=sbparams.omega_m, omega_lam=sbparams.omega_lam)

    logprint('Set up NFW halo for lensing')

    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.
    # cosmos_cat = galsim.COSMOSCatalog(sbparams.cat_file_name, dir=sbparams.datadir)
    # fitcat = Table.read(os.path.join(sbparams.cosmosdir, sbparams.fit_file_name))

    cosmos_cat = Table.read(os.path.join(sbparams.datadir,
                                         sbparams.cat_file_name))
    logprint(f'Read in {len(cosmos_cat)} galaxies from catalog and associated fit info')

    size_wg = (cosmos_cat['FLUX_RADIUS'] > 0) & (cosmos_cat['c10_sersic_fit_hlr'] < 50)
    cosmos_cat = cosmos_cat[size_wg]

    try:
        cluster_cat = galsim.COSMOSCatalog()
        print("Default galsim cosmos catalog worked")
    except:
        try:
            cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name,
                                               dir=sbparams.cosmosdir)
        except:
            cluster_cat = galsim.COSMOSCatalog(sbparams.cluster_cat_name)

    ### Now create PSF. First, define Zernicke polynomial component
    ### note: aberrations were definined for lam = 550, and close to the
    ### center of the camera. The PSF degrades at the edge of the FOV
    lam_over_diam = sbparams.lam * 1.e-9 / sbparams.tel_diam    # radians
    lam_over_diam *= 206265.

    aberrations = np.zeros(38)             # Set the initial size.
    aberrations[0] = 0.                       # First entry must be zero
    aberrations[1] = -0.00305127
    aberrations[4] = -0.02474205              # Noll index 4 = Defocus
    aberrations[11] = -0.01544329             # Noll index 11 = Spherical
    aberrations[22] = 0.00199235
    aberrations[26] = 0.00000017
    aberrations[37] = 0.00000004
    logprint(f'Calculated lambda over diam = {lam_over_diam} arcsec')

    # gaussian jitter component from gondola instabilities
    jitter_psf = galsim.Gaussian(flux=1, fwhm=sbparams.jitter_fwhm)

    # due to how the config is structured...
    if hasattr(sbparams, 'use_optics'):
        use_optics = sbparams.use_optics
    else:
        use_optics = True

    if use_optics is False:
        optics = None
        psf = jitter_psf
        logprint('\nuse_optics is False; using jitter-only PSF\n')

    elif use_optics is True:
        # will store the Zernicke component of the PSF
        optics = galsim.OpticalPSF(lam=sbparams.lam,diam=sbparams.tel_diam,
                        obscuration=sbparams.obscuration, nstruts=sbparams.nstruts,
                        strut_angle=sbparams.strut_angle, strut_thick=sbparams.strut_thick,
                        aberrations=aberrations)

        psf = galsim.Convolve([jitter_psf, optics])

        logprint('\n Use_optics is True; convolving telescope optics PSF profile\n')

    ###
    ### Make generic WCS
    ###

    # If you wanted to make a non-trivial WCS system, could set theta to a non-zero number
    fiducial_full_image = galsim.ImageF(sbparams.image_xsize, sbparams.image_ysize)

    theta = 0.0 * galsim.degrees
    dudx = np.cos(theta) * sbparams.pixel_scale
    dudy = -np.sin(theta) * sbparams.pixel_scale
    dvdx = np.sin(theta) * sbparams.pixel_scale
    dvdy = np.cos(theta) * sbparams.pixel_scale

    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=fiducial_full_image.true_center)
    sky_center = galsim.CelestialCoord(ra=sbparams.center_ra, dec=sbparams.center_dec)
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

    ##
    ## Define RNG for dither offsets
    ##
    rng = np.random.default_rng(sbparams.dithering_seed)

    ###
    ### MAKE SIMULATED OBSERVATIONS
    ### ITERATE n TIMES TO MAKE n SEPARATE IMAGES
    ###

    # Get current band for directory structure
    band = sbparams.bandpass.replace("crates_", "")
    
    # Determine output directory based on whether we're using directory structure
    if sbparams.use_dir_structure:
        # Use the band/cal directory in the hierarchical structure
        output_dir = os.path.join(sbparams.run_outdir, band, "cal")
        
        # Also prepare the out directory path for logging when using data_dir
        out_dir = os.path.join(sbparams.run_outdir, band, "out")
        
        # Ensure the cal directory exists (should already be created, but just in case)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logprint(f'Created directory: {output_dir}')
            
        # Ensure the out directory exists
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            logprint(f'Created directory: {out_dir}')
    else:
        # Use the outdir directly with no subdirectories
        output_dir = sbparams.run_outdir
        out_dir = output_dir
        
    # Set up truth catalog path
    truth_file_name = os.path.join(output_dir, f'{run_name}_truth_{band}.fits')
    
    # Move the log file to the correct location if using data_dir
    if temp_log and args.data_dir is not None:
        handler = log.handlers[0]  # Get the file handler
        old_log_path = handler.baseFilename
        new_log_path = os.path.join(out_dir, logfile)
        
        # Close the current log file
        handler.close()
        log.removeHandler(handler)
        
        # Move the file
        os.makedirs(os.path.dirname(new_log_path), exist_ok=True)
        if os.path.exists(old_log_path):
            # Copy content rather than move in case there are permission issues
            with open(old_log_path, 'r') as old_log, open(new_log_path, 'w') as new_log:
                new_log.write(old_log.read())
            
            # Try to remove the old log file, but don't fail if we can't
            try:
                os.remove(old_log_path)
            except:
                pass
                
        # Setup new logger pointed at the new location
        log = utils.setup_logger(logfile, logdir=out_dir)
        logprint = utils.LogPrint(log, vb)
        logprint(f'Moved log file to {new_log_path}')
    
    # Initialize truth catalog during first run
    if mpi is False or M.is_mpi_root():
        names = ['gal_num', 'cosmos_index','x_image', 'y_image',
                 'ra', 'dec', 'nfw_g1', 'nfw_g2', 'nfw_mu', 'nfw_kappa', 'cosmos_g1', 'cosmos_g2',
                 'admom_g1', 'admom_g2', 'admom_sigma', 'admom_flag', 'redshift', 'flux',
                 'truth_fwhm','truth_mom', 'n',
                 'hlr', 'scale_h_over_r', 'obj_class']
        types = [int, int, float, float, float, float, float,
                 float, float, float, float, float, float, float, float, int, float, float, float, float,
                 float, float, float, str]
        truth_catalog = galsim.OutputCatalog(names, types)

    all_psf_files = []
    search_path = os.path.join('/projects/mccleary_group/superbit/emp_psfs/psfex-output', '*.psf')
    all_psf_files.extend(glob(search_path))

    for i in np.arange(1, sbparams.nexp+1):
        if mpi is True:
            # get MPI processes in sync at start of each image
            M.barrier()

        outnum = str(i).zfill(3)
        outname = f'{run_name}_{outnum}_{band}_sim.fits'
        file_name = os.path.join(output_dir, outname)

        # Set up the image:
        full_image = galsim.ImageF(sbparams.image_xsize, sbparams.image_ysize)
        sky_level = sbparams.exp_time * sbparams.sky_bkg / sbparams.gain
        full_image.fill(sky_level)

        ## Define X & Y dither offsets
        dither_offsets = rng.integers(-100, 100, size=2)
        logprint(f'dithers are {dither_offsets}')
        full_image.setOrigin(dither_offsets[0], dither_offsets[1])
        full_image.wcs = wcs

        psf_file = all_psf_files[i]
        image_file = os.path.join(os.path.dirname(os.path.dirname(psf_file)), os.path.basename(psf_file).replace('_starcat.psf', '.fits'))
        psf = galsim.des.DES_PSFEx(psf_file, image_file)
        #####
        ## Loop over galaxy objects:
        #####
        print('Starting galaxy injections')

        if mpi is False:
            start = time.time()
            with Pool(ncores) as pool:
                # Create batches
                batch_indices = utils.setup_batches(sbparams.nobj, ncores)

                full_image, truth_catalog = combine_objs(
                    pool.starmap(
                        make_obj_runner,
                        ([
                          batch_indices[k],
                          'gal',
                          galsim.UniformDeviate(sbparams.galobj_seed+k+1),
                          wcs,
                          affine,
                          cosmos_cat,
                          nfw,
                          psf,
                          sbparams,
                          logprint
                          ] for k in range(ncores))
                        ),
                    full_image,
                    truth_catalog,
                    i
                    )

            dt = time.time() - start
            logprint(f'Total time for galaxy injections: {dt:.1f}s')

        else:
            # get local range to iterate over in this process
            local_start, local_end = M.mpi_local_range(sbparams.nobj)
            for k in range(local_start, local_end):
                time1 = time.time()

                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(sbparams.galobj_seed+k+1)

                try:
                    # make single galaxy object
                    stamp, truth = make_a_galaxy(ud=ud,
                                                wcs=wcs,
                                                affine=affine,
                                                cosmos_cat=cosmos_cat,
                                                psf=psf,
                                                nfw=nfw,
                                                sbparams=sbparams,
                                                logprint=logprint
                                                )
                    # Find the overlapping bounds:
                    bounds = stamp.bounds & full_image.bounds

                    # Finally, add the stamp to the full image.

                    full_image[bounds] += stamp[bounds]
                    time2 = time.time()
                    tot_time = time2-time1
                    logprint(f'Galaxy {k} positioned relative to center t={tot_time} s')
                    this_flux=np.sum(stamp.array)

                    if i == 1:
                        row = [ k, truth.cosmos_index, truth.x, truth.y, truth.ra, truth.dec, truth.g1,
                                truth.g2, truth.mu, truth.kappa, truth.cosmos_g1, truth.cosmos_g2, truth.admom_g1, truth.admom_g2, truth.admom_sigma, truth.admom_flag,  truth.z,
                                this_flux, truth.fwhm, truth.mom_size,
                                truth.n, truth.hlr, truth.scale_h_over_r, truth.obj_class]
                        truth_catalog.addRow(row)
                except galsim.errors.GalSimError:
                    logprint(f'Galaxy {k} has failed, skipping...')

        #####
        ### Inject cluster galaxy objects:
        #####
        '''print('Starting cluster galaxy injections')

        center_coords = galsim.CelestialCoord(sbparams.center_ra,sbparams.center_dec)
        centerpix = wcs.toImage(center_coords)

        if mpi is False:
            start = time.time()
            with Pool(ncores) as pool:
                batch_indices = utils.setup_batches(sbparams.nclustergal, ncores)

                full_image, truth_catalog = combine_objs(
                    pool.starmap(
                        make_obj_runner,
                        ([
                          batch_indices[k],
                          'cluster_gal',
                          galsim.UniformDeviate(sbparams.cluster_seed+k+1),
                          wcs,
                          affine,
                          centerpix,
                          cluster_cat,
                          psf,
                          sbparams,
                          logprint
                          ] for k in range(ncores))
                        ),
                    full_image,
                    truth_catalog,
                    i
                    )

            dt = time.time() - start
            logprint(f'Total time for cluster galaxy injections: {dt:.1f}s')

        else:
            # get local range to iterate over in this process
            local_start, local_end = M.mpi_local_range(sbparams.nclustergal)
            for k in range(local_start, local_end):

                time1 = time.time()

                # The usual random number generator using a different seed for each galaxy.
                ud = galsim.UniformDeviate(sbparams.cluster_seed+k+1)

                try:
                    # make single galaxy object
                    cluster_stamp,truth = make_cluster_galaxy(ud=ud,wcs=wcs,affine=affine,
                                                            centerpix=centerpix,
                                                            cluster_cat=cluster_cat,
                                                            psf=psf,
                                                            sbparams=sbparams,
                                                            logprint=logprint)
                    # Find the overlapping bounds:
                    bounds = cluster_stamp.bounds & full_image.bounds

                    # Finally, add the stamp to the full image.

                    full_image[bounds] += cluster_stamp[bounds]
                    time2 = time.time()
                    tot_time = time2-time1
                    logprint(f'Cluster galaxy {k} positioned relative to center t={tot_time} s')
                    this_flux=np.sum(cluster_stamp.array)

                    if i == 1:
                        row = [ k, truth.cosmos_index, truth.x, truth.y, truth.ra, truth.dec, truth.g1,
                                truth.g2, truth.mu, truth.kappa, truth.cosmos_g1, truth.cosmos_g2, truth.admom_g1, truth.admom_g2, truth.admom_sigma, truth.admom_flag,  truth.z,
                                this_flux, truth.fwhm, truth.mom_size,
                                truth.n, truth.hlr, truth.scale_h_over_r, truth.obj_class]
                        truth_catalog.addRow(row)
                except galsim.errors.GalSimError:
                    logprint(f'Cluster galaxy {k} has failed, skipping...')'''

        #####
        ### Now repeat process for stars!
        #####
        print('Starting star injections')

        if mpi is False:
            start = time.time()
            pud = np.random.default_rng(sbparams.stars_seed)

            with Pool(ncores) as pool:
                batch_indices = utils.setup_batches(sbparams.nstars, ncores)

                full_image, truth_catalog = combine_objs(
                    pool.starmap(
                        make_obj_runner,
                        ([
                          batch_indices[k],
                          'star',
                          galsim.UniformDeviate(sbparams.stars_seed+k+1),
                          pud,
                          batch_indices[k],
                          wcs,
                          affine,
                          psf,
                          sbparams,
                          logprint
                          ] for k in range(sbparams.ncores))
                        ),
                    full_image,
                    truth_catalog,
                    i
                    )

            dt = time.time() - start
            logprint(f'Total time for star injections: {dt:.1f}s')


        else:
            # get local range to iterate over in this process
            local_start, local_end = M.mpi_local_range(sbparams.nstars)
            for k in range(local_start, local_end):
                time1 = time.time()
                ud = galsim.UniformDeviate(sbparams.stars_seed+k+1)
                pud = np.random.default_rng(sbparams.stars_seed)
                star_stamp,truth = make_a_star(ud=ud,pud=pud,
                                            k=k,
                                            wcs=wcs,
                                            affine=affine,
                                            psf=psf,
                                            sbparams=sbparams,
                                            logprint=logprint
                                            )
                bounds = star_stamp.bounds & full_image.bounds

                # Add the stamp to the full image.
                try:
                    full_image[bounds] += star_stamp[bounds]

                    time2 = time.time()
                    tot_time = time2-time1

                    logprint(f'Star {k}: positioned relative to center, t={tot_time} s')
                    this_flux=np.sum(star_stamp.array)

                    if i == 1:
                        row = [ k, truth.cosmos_index, truth.x, truth.y, truth.ra, truth.dec,
                                truth.g1, truth.g2, truth.mu, truth.admom_g1, truth.admom_g2, truth.admom_sigma, truth.admom_flag,
                                truth.z, this_flux, truth.fwhm,truth.mom_size,
                                truth.n, truth.hlr, truth.scale_h_over_r, truth.obj_class]
                        truth_catalog.addRow(row)

                except galsim.errors.GalSimError:
                    logprint(f'Star {k} has failed, skipping...')

        # If not using MPI, then this is already done
        if mpi is True:
            # Gather results from MPI processes, reduce to single result on root
            # Using same names on left and right sides is hiding lots of MPI magic
            full_image = M.gather(full_image)
            truth_catalog = M.gather(truth_catalog)
            if M.is_mpi_root():
                full_image = reduce(combine_images, full_image)

                if i == 1:
                    truth_catalog = reduce(combine_catalogs, truth_catalog)
                else:
                    # do the adding of noise and writing to disk entirely on root
                    # root and the rest meet again at barrier at start of loop
                    continue

        # The first thing to do is to make the Gaussian noise uniform across the whole image.

        if (mpi is False) or (M.is_mpi_root()):

            # Add dark current
            logprint('Adding Dark current')
            dark_noise = sbparams.dark_current * sbparams.exp_time
            full_image += dark_noise

            # Add ccd noise
            logprint('Adding CCD noise')
            noise = galsim.CCDNoise(
                sky_level=0,
                gain=sbparams.gain,
                read_noise=sbparams.read_noise,
                rng=galsim.BaseDeviate(sbparams.noise_seed)
                )

            full_image.addNoise(noise)

            logprint.debug('Added noise to final output image')
            
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            try:
                full_image.write(file_name, clobber=clobber)
                logprint(f'Wrote image to {file_name}')
            except OSError as e:
                logprint(f'OSError: {e}')
                raise e

            # Write truth catalog to file after all exposures
            if i == sbparams.nexp:
                try:
                    truth_catalog.write(truth_file_name)
                    logprint(f'Wrote truth to {truth_file_name}')

                    # It can be useful to load the true PSF into memory for
                    # later tests. So we pickle it now and save the filename
                    # into the truth catalog header
                    psf_outfile = os.path.join(
                        output_dir, f'true_psf_{band}.pkl'
                        )
                    with open(psf_outfile, 'wb') as psf_pfile:
                        pickle.dump(psf, psf_pfile)

                    with fits.open(truth_file_name, mode='update') as handle:
                        handle[0].header['psf_pkl'] = psf_outfile

                except OSError as e:
                    logprint(f'OSError: {e}')
                    raise e

    logprint('\nCompleted all images\n')

    if (mpi is False) or (M.is_mpi_root()):
        # Create mask and weight files in the output directory
        logprint('Creating masks')
        logprint.warning('For now, we just write a simple mask file with all 1s')
        #sbparams.make_mask_files(logprint, clobber, cal_dir=output_dir)

        logprint('Creating weights')
        logprint.warning('For now, we just write a simple weight file with all 1s')
        #sbparams.make_weight_files(logprint, clobber, cal_dir=output_dir)

    # Log file was created before outdir is setup in some cases
    # If so, move from temp location to there
    #if temp_log is True:
    #    oldfile = os.path.join(logdir, logfile)
    #    newfile = os.path.join(sbparams.outdir, logfile)
    #    os.replace(oldfile, newfile)

    if (mpi is False) or (M.is_mpi_root()):
        logprint('\nDone!\n')

    end_time = time.time()
    logprint(f'\n\ngalsim execution time = {end_time - start_time}\n\n')
    
    current_time_utc = datetime.now(timezone.utc)
    logprint(f"Current time in UTC: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
