# Copyright (c) 2012-2019 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
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
#

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
from astropy.table import Table

class truth():

    def __init__(self):
        '''
        class to store attributes of a mock galaxy or star
        :x/y: object position in full image
        :ra/dec: object position in WCS --> may need revision?
        :g1/g2: NFW shear moments
        :mu: NFW magnification
        :z: galaxy redshift
        :variance: of stamp pixel noise
        '''

        self.x = None
        self.y = None
        self.ra = None
        self.dec = None
        self.g1 = 0.0
        self.g2 = 0.0
        self.mu = 1.0
        self.z = 0.0
        self.variance=0.0


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

    if nfw_mu < 0:
        import warnings
        warnings.warn("Warning: mu < 0 means strong lensing!  Using mu=25.")
        nfw_mu = 25
    elif nfw_mu > 25:
        import warnings
        warnings.warn("Warning: mu > 25 means strong lensing!  Using mu=25.")
        nfw_mu = 25

    return nfw_shear, nfw_mu

def get_wcs_info(psfname):

    imdir='/Users/jemcclea/Research/SuperBIT/A2218/Clean'
    imroot=psfname.split('psfex_output/')[1].replace('_cat.psf','.fits')
    imagen=os.path.join(imdir,imroot)
    imw=galsim.wcs.readFromFitsHeader(astropy.io.fits.getheader(imagen))

    return imw[0]


def make_a_galaxy(ud,wcs,psf,affine,fitcat):
    """
    Method to make a single galaxy object and return stamp for
    injecting into larger GalSim image
    """

    # Choose a random RA, Dec around the sky_center. Start with XY coords,
    # or else image doesn't get filled!
    center_dec=wcs.center.dec
    center_ra=wcs.center.ra
    center_coords = galsim.CelestialCoord(center_ra,center_dec)
    centerpix = wcs.toImage(center_coords)

    x = (2.*ud()-1) * 0.5 * image_xsize
    y = (2.*ud()-1) * 0.5 * image_ysize

    # We will need the image position as well, so use the wcs to get that
    image_pos = galsim.PositionD(x+centerpix.x,y+centerpix.y)
    world_pos = wcs.toWorld(image_pos)
    ra=world_pos.ra; dec = world_pos.dec

    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)
    logger.debug('made it through WCS calculations...')

    # Create chromatic galaxy
    bp_dir = '/Users/jemcclea/Research/GalSim/examples/data'
    bp_file=os.path.join(bp_dir,'lum_throughput.csv')
    bandpass = galsim.Bandpass(bp_file,wave_type='nm',blue_limit = 310,red_limit=1100)
    gal = cosmos_cat.makeGalaxy(gal_type='parametric', rng=ud,chromatic=True)
    logger.debug('created chromatic galaxy')


    # Obtain galaxy redshift from the COSMOS profile fit catalog
    gal_z=fitcat['zphot'][gal.index]
    real_mag = fitcat['mag_auto'][gal.index]

    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)

    # This automatically scales up the noise variance by flux_scaling**2.
    gal *= flux_scaling

    logger.debug('rescaled galaxy with scaling factor %f' % flux_scaling)
    # Get the reduced shears and magnification at this point
    try:
        nfw_shear, mu = nfw_lensing(nfw, uv_pos, gal_z)
        g1=nfw_shear.g1; g2=nfw_shear.g2
        gal = gal.lens(g1, g2, mu)

    except:
        print("could not lens galaxy at z = %f, setting default values..." % gal_z)
        g1 = 0.0; g2 = 0.0
        mu = 1.0

    # Generate PSF at location of galaxy
    # Convolve galaxy image with the PSF.
    this_psf = psf.getPSF(image_pos)
    logger.debug("obtained PSF at image position")

    gsp=galsim.GSParams(maximum_fft_size=32768)
    final = galsim.Convolve([this_psf, gal],gsparams=gsp)
    logger.debug("Convolved galaxy and PSF at image position")


    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    position=[ix_nominal,iy_nominal,ra.deg,dec.deg]

    this_stamp_image = galsim.Image(128, 128,wcs=wcs.local(image_pos))
    stamp = final.drawImage(bandpass,image=this_stamp_image, offset=offset, method='no_pixel')

    # If desired, one can also draw the PSF and output its moments too, as:
    #psf_stamp = psf.drawImage(scale=0.206, offset=offset, method='no_pixel')
    # Recenter the stamp at the desired position:
    stamp.setCenter(ix_nominal,iy_nominal)
    logger.debug("Created stamp and set center!")

    galaxy_truth=truth()
    galaxy_truth.ra=ra.deg; galaxy_truth.dec=dec.deg
    galaxy_truth.x=ix_nominal; galaxy_truth.y=iy_nominal
    galaxy_truth.g1=g1; galaxy_truth.g2=g2
    galaxy_truth.mu = mu; galaxy_truth.z = gal_z
    #galaxy_truth.flux = stamp.added_flux
    galaxy_truth.flux = real_mag


    try:
        galaxy_truth.fwhm=final.calculateFWHM()
        galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma
    except:
        galaxy_truth.fwhm=-9999.0
        galaxy_truth.mom_size=stamp.FindAdaptiveMom().moments_sigma

    return stamp, galaxy_truth

def make_cluster_galaxy(ud,wcs,psf,affine,cluster_cat,fitcat):
    """
    Method to make a single galaxy object and return stamp for
    injecting into larger GalSim image
    """
    # Choose a random RA, Dec around the sky_center.
    # Note that for this to come out close to a square shape, we need to account for the
    # cos(dec) part of the metric: ds^2 = dr^2 + r^2 d(dec)^2 + r^2 cos^2(dec) d(ra)^2
    # So need to calculate dec first.



    center_dec=wcs.center.dec
    center_ra=wcs.center.ra
    center_coords = galsim.CelestialCoord(center_ra,center_dec)
    centerpix = wcs.toImage(center_coords)

    radius = 120
    max_rsq = (radius)**2
    while True:  # (This is essentially a do..while loop.)
        x = (2.*ud()-1) * radius
        y = (2.*ud()-1) * radius
        rsq = x**2 + y**2
        if rsq <= max_rsq: break

    # We will need the image position as well, so use the wcs to get that
    image_pos = galsim.PositionD(x+centerpix.x+(ud()-0.5)*75,y+centerpix.y+(ud()-0.5)*75)
    world_pos = wcs.toWorld(image_pos)
    ra=world_pos.ra; dec = world_pos.dec


    # We also need this in the tangent plane, which we call "world coordinates" here,
    # since the PowerSpectrum class is really defined on that plane, not in (ra,dec).
    # This is still an x/y corrdinate
    uv_pos = affine.toWorld(image_pos)


    bp_dir = '/Users/jemcclea/Research/GalSim/examples/data'
    bp_file=os.path.join(bp_dir,'lum_throughput.csv')
    bandpass = galsim.Bandpass(bp_file,wave_type='nm',blue_limit = 310,red_limit=1100)
    gal = cluster_cat.makeGalaxy(gal_type='parametric', rng=ud,chromatic=True)
    #gal_flux = gal_f._original.flux
    real_mag = fitcat['mag_auto'][gal.index]

    #gal = galsim.InclinedExponential(80*galsim.degrees,half_light_radius=1,flux=gal_flux).rotate(20*galsim.degrees)
    logger.debug('created debug galaxy')

    # Apply a random rotation
    theta = ud()*2.0*numpy.pi*galsim.radians
    gal = gal.rotate(theta)

    # not sheared
    g1 = 0.0; g2 = 0.0
    mu = 1.0
    gal_z = 0.17
    # This automatically scales up the noise variance by flux_scaling**2.
    gal *= (flux_scaling*2)
    gal.magnify(10)
    logger.debug('rescaled galaxy with scaling factor %f' % flux_scaling)


    # Generate PSF at location of galaxy
    # Convolve galaxy image with the PSF.
    this_psf = psf.getPSF(image_pos)
    #this_psf = galsim.Gaussian(flux=1,fwhm=0.3)
    gsp=galsim.GSParams(maximum_fft_size=16384)
    final = galsim.Convolve([this_psf,gal],gsparams=gsp)
    logger.debug("Convolved galaxy and PSF at image position")


    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.

    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    position=[ix_nominal,iy_nominal,ra.deg,dec.deg]

    # We use method='no_pixel' here because the SDSS PSF image that we are using includes the
    # pixel response already.

    this_stamp_image = galsim.Image(256, 256,wcs=wcs.local(image_pos))
    cluster_stamp = final.drawImage(bandpass,image=this_stamp_image, offset=offset,method='no_pixel')
    cluster_stamp.setCenter(ix_nominal,iy_nominal)
    logger.debug('drew & centered galaxy!')
    new_variance=0

    cluster_galaxy_truth=truth()
    cluster_galaxy_truth.ra=ra.deg; cluster_galaxy_truth.dec=dec.deg
    cluster_galaxy_truth.x=ix_nominal; cluster_galaxy_truth.y=iy_nominal
    cluster_galaxy_truth.g1=g1; cluster_galaxy_truth.g2=g2
    cluster_galaxy_truth.mu = mu; cluster_galaxy_truth.z = gal_z
    #cluster_galaxy_truth.flux = cluster_stamp.added_flux
    cluster_galaxy_truth.flux = real_mag


    try:
        cluster_galaxy_truth.fwhm=final.calculateFWHM()
        cluster_galaxy_truth.mom_size=cluster_stamp.FindAdaptiveMom().moments_sigma
    except:
        cluster_galaxy_truth.fwhm=-9999.0
        cluster_galaxy_truth.mom_size=cluster_stamp.FindAdaptiveMom().moments_sigma


    logger.debug('created truth values')

    return cluster_stamp, cluster_galaxy_truth


def make_a_star(ud,wcs=None,psf=None,affine=None):
    logger.debug('entered make a star method...')

    # Choose a random RA, Dec around the sky_center. Start with XY coords,
    # or else image doesn't get filled!

    center_dec=wcs.center.dec
    center_ra=wcs.center.ra
    center_coords = galsim.CelestialCoord(center_ra,center_dec)
    centerpix = wcs.toImage(center_coords)

    x = (2.*ud()-1) * 0.5 * image_xsize
    y = (2.*ud()-1) * 0.5 * image_ysize

    # We will need the image position as well, so use the wcs to get that

    image_pos = galsim.PositionD(x+centerpix.x,y+centerpix.y)
    world_pos = wcs.toWorld(image_pos)
    ra=world_pos.ra; dec = world_pos.dec

    # We also need this in the tangent plane, which we call "world coordinates" here,
    # This is still an x/y corrdinate

    uv_pos = affine.toWorld(image_pos)
    logger.debug('made it through WCS calculations...')


    # Draw star flux at random; based on distribution of star fluxes in real images
    # Generate PSF at location of star, convolve simple Airy with the PSF to make a star

    flux_dist = galsim.DistDeviate(ud, function = lambda x:x**-1., x_min = 1226.2965, x_max = 1068964.0)
    star_flux = flux_dist()
    shining_star = galsim.DeltaFunction(flux=star_flux)
    logger.debug('created star object with flux')

    # Final profile is the convolution of PSF and galaxy image
    # Can include any number of things in the list, all of which are convolved
    # together to make the final flux profile.

    this_psf = psf.getPSF(image_pos)
    star=galsim.Convolve([shining_star,this_psf])
    logger.debug('convolved star & psf')

    # Account for the fractional part of the position, and
    # recenter the stamp at the desired position.
    # (cf. demo9.py for an explanation of  nominal position stuff.)

    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)
    star_image = galsim.Image(512, 512, wcs=wcs.local(image_pos))
    star_stamp = star.drawImage(image=star_image, offset=offset, method='no_pixel')
    star_stamp.setCenter(ix_nominal,iy_nominal)
    logger.debug('made a star_stamp')



    star_truth=truth()
    star_truth.ra = ra.deg; star_truth.dec = dec.deg
    star_truth.x = ix_nominal; star_truth.y = iy_nominal
    star_truth.flux=star_stamp.added_flux
    try:
        star_truth.fwhm=final.CalculateFWHM()
        star_truth.mom_size=star_stamp.FindAdaptiveMom().moments_sigma
    except:
        star_truth.fwhm=-9999.0
        star_truth.mom_size=star_stamp.FindAdaptiveMom().moments_sigma
    logger.debug('made it through star recentering')
    results = star_stamp.FindAdaptiveMom()
    logger.debug('HSM reports that the image has observed shape and size:')
    logger.debug('    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)', results.observed_shape.e1,
                results.observed_shape.e2, results.moments_sigma)

    return star_stamp, star_truth

def main(argv):
    """
    Make images using model PSFs and galaxy cluster shear:
      - The galaxies come from COSMOSCatalog, which can produce either RealGalaxy profiles
        (like in demo10) and parametric fits to those profiles.
      - Using parametric galaxies so that filter responses/system throughput can be convolved
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    global logger
    logger = logging.getLogger("mock_superbit_data")

    # Define some parameters we'll use below.
    # Normally these would be read in from some parameter file.
    global pixel_scale
    pixel_scale = 0.206                   # arcsec/pixel
    global image_xsize
    image_xsize = 6665                    # size of image in pixels
    global image_ysize
    image_ysize = 4453                    # size of image in pixels
    global image_xsize_arcsec
    image_xsize_arcsec = image_xsize*pixel_scale # size of big image in each dimension (arcsec)
    global image_ysize_arcsec
    image_ysize_arcsec = image_ysize*pixel_scale # size of big image in each dimension (arcsec)

    gain = 3.33
    dark_current = 0.33
    read_noise = 5

    global nobj
    nobj = 10500                       # number of galaxies in entire field
    global nstars
    nstars = 320                         # number of stars in the entire field
    global flux_scaling                  # Let's figure out the flux for a 0.5 m class telescope
    global tel_diam
    tel_diam = 0.5
    global lam
    lam = 625                            # Central wavelength for an airy disk
    global exp_time
    global noise_variance
    global sky_level

    psf_path = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-real/psfex_output'
    global nfw                        # will store the NFWHalo information
    global cosmos_cat                 # will store the COSMOS catalog from which we draw objects

    # Set up the NFWHalo:
    mass=5E14              # Cluster mass (Msol/h)
    nfw_conc = 4           # Concentration parameter = virial radius / NFW scale radius
    nfw_z_halo = 0.17       # redshift of the halo
    nfw_z_source = 0.6     # redshift of the lensed sources
    omega_m = 0.3          # Omega matter for the background cosmology.
    omega_lam = 0.7        # Omega lambda for the background cosmology.

    nfw = galsim.NFWHalo(mass=mass, conc=nfw_conc, redshift=nfw_z_halo,
                             omega_m=omega_m, omega_lam=omega_lam)
    logger.info('Set up NFW halo for lensing')

    # Read in galaxy catalog, as well as catalog containing
    # information from COSMOS fits like redshifts, hlr, etc.

    cat_file_name = 'real_galaxy_catalog_23.5.fits'
    fdir = 'data/COSMOS_23.5_training_sample'
    fit_file_name = 'real_galaxy_catalog_23.5_fits.fits'

    cosmos_cat = galsim.COSMOSCatalog(cat_file_name, dir=fdir)
    fitcat = Table.read(os.path.join(fdir,fit_file_name))
    logger.info('Read in %d galaxies from catalog and associated fit info', cosmos_cat.nobjects)

    cluster_cat = galsim.COSMOSCatalog('data/real_galaxy_catalog_23.5_example.fits')
    logger.info('Read in %d cluster galaxies from catalog', cosmos_cat.nobjects)


    # The catalog returns objects that are appropriate for HST in 1 second exposures.  So for our
    # telescope we scale up by the relative area and exposure time.
    # Will also multiply by the gain and relative pixel scales...

    hst_eff_area = 2.4**2
    sbit_eff_area = tel_diam**2

    ###
    ### LOOP OVER PSFs TO MAKE GROUPS OF IMAGES
    ###

    all_psfs=glob.glob('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-real/psfex_output/dwb_image_ifc*WCS_cat.psf')
    logger.info('Beginning loop over jitter/optical psfs')
    random_seed = 7839234
    i=0
    for psf_filen in all_psfs:
        logger.info('Beginning PSF %s...'% psf_filen)
        rng = galsim.BaseDeviate(random_seed)

        # This is specific to empirical PSFs

        try:
            timescale=psf_filen.split('target_')[1].split('_WCS')[0]
        except:
            timescale=psf_filen.split('sci_')[1].split('_WCS')[0]

        outname=''.join(['mockSuperbit_nodilate_',timescale,'_',str(i),'.fits'])
        truth_file_name=''.join(['./output/truth_nodilate_',timescale,'_',str(i),'.dat'])
        file_name = os.path.join('output',outname)

        # Set up the image:
        if timescale=='150':
            print("Automatically detecting a 150s exposure image, setting flux scale and sky accordingly")
            sky_level = 51     # ADU
            exp_time=150.

        else:
            print("Automatically detecting a 300s exposure image, setting flux scale and sky accordingly")
            sky_level = 102    # ADU
            exp_time=300.

        flux_scaling = (sbit_eff_area/hst_eff_area) * exp_time * gain #* (pixel_scale/0.05)#**2

        # Setting up a truth catalog

        names = [ 'gal_num', 'x_image', 'y_image',
                      'ra', 'dec', 'g1_meas', 'g2_meas', 'fwhm','mom_size',
                      'nfw_g1', 'nfw_g2', 'nfw_mu', 'redshift','flux', 'stamp_sum']
        types = [ int, float, float, float, float, float,
                      float, float, float, float, float,
                      float, float, float, float]
        truth_catalog = galsim.OutputCatalog(names, types)

        # Set up the image:

        full_image = galsim.ImageF(image_xsize, image_ysize)
        full_image.setOrigin(0,0)
        full_image.fill(sky_level)


        wcs=get_wcs_info(psf_filen)
        affine=wcs.affine(full_image.true_center)
        full_image.wcs = wcs

        # Now let's read in the PSFEx PSF model.
        psf_wcs=wcs
        psf_file = os.path.join(psf_path,psf_filen)
        psf = galsim.des.DES_PSFEx(psf_file,wcs=psf_wcs)
        logger.info('Constructed PSF object from PSFEx file')

        # Loop over galaxy objects:

        for k in range(nobj):
            time1 = time.time()

            # The usual random number generator using a different seed for each galaxy.
            ud = galsim.UniformDeviate(random_seed+k+1)

            try:
                # make single galaxy object
                logger.debug("about to make stamp...")
                stamp,truth = make_a_galaxy(ud=ud,wcs=wcs,psf=psf,affine=affine,fitcat=fitcat)
                logger.debug("stamp is made")
                # Find the overlapping bounds:
                bounds = stamp.bounds & full_image.bounds

                # Finally, add the stamp to the full image.
                full_image[bounds] += stamp[bounds]
                logger.debug("stamp added to full image")
                time2 = time.time()
                tot_time = time2-time1
                logger.info('Galaxy %d positioned relative to center t=%f s',
                                k, tot_time)
                g1_real=stamp.FindAdaptiveMom().observed_shape.g1
                g2_real=stamp.FindAdaptiveMom().observed_shape.g2
                sum_flux=numpy.sum(stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec,
                            g1_real, g2_real, truth.fwhm, truth.mom_size, truth.g1,
                            truth.g2, truth.mu, truth.z, truth.flux, sum_flux]
                truth_catalog.addRow(row)
                logger.debug("row added to truth catalog")

            except:
                logger.info('Galaxy %d has failed, skipping...',k)
                pdb.set_trace()


        #####
        ### Inject cluster galaxy objects:
        ### - Note that this "cluster" is just for aesthetics
        ### - So, 'n_cluster_gals' is arbitrary
        ### - You could concievably create a method to base the number of galaxies injected
        ###   using some scaling relation between (NFW) mass and richness to set n_cluster_gals
        ###   to something based in reality (though these are poorly constrained at low mass!).
        #####

        n_cluster_gals = 40

        for k in range(n_cluster_gals):
            time1 = time.time()

            # The usual random number generator using a different seed for each galaxy.
            ud = galsim.UniformDeviate(random_seed+k+1)

            try:
                # make single galaxy object
                cluster_stamp,truth = make_cluster_galaxy(
                    ud=ud,wcs=wcs,affine=affine,
                    psf=psf,cluster_cat=cluster_cat,
                    fitcat=fitcat)

                # Find the overlapping bounds:
                bounds = cluster_stamp.bounds & full_image.bounds

                # Finally, add the stamp to the full image.

                full_image[bounds] += cluster_stamp[bounds]
                time2 = time.time()
                tot_time = time2-time1
                logger.info('Cluster galaxy %d positioned relative to center t=%f s',
                                k, tot_time)
                sum_flux=numpy.sum(cluster_stamp.array)
                g1_real=cluster_stamp.FindAdaptiveMom().observed_shape.g1
                g2_real=cluster_stamp.FindAdaptiveMom().observed_shape.g2

                row = [ k,truth.x, truth.y, truth.ra, truth.dec, g1_real, g2_real, truth.fwhm,
                            truth.mom_size, truth.g1,truth.g2, truth.mu, truth.z, truth.flux, sum_flux]
                truth_catalog.addRow(row)
            except:
                logger.info('Cluster galaxy %d has failed, skipping...',k)
                pdb.set_trace()


        ####
        ### Now repeat process for stars!
        ####

        random_seed_stars=3221987

        for k in range(nstars):
            time1 = time.time()
            ud = galsim.UniformDeviate(random_seed_stars+k+1)
            try:

                star_stamp,truth=make_a_star(ud=ud,wcs=wcs,psf=psf,affine=affine)
                bounds = star_stamp.bounds & full_image.bounds

                # Add the stamp to the full image.
                full_image[bounds] += star_stamp[bounds]

                time2 = time.time()
                tot_time = time2-time1

                logger.info('Star %d: positioned relative to center, t=%f s',
                                k,  tot_time)

                g1_real=star_stamp.FindAdaptiveMom().observed_shape.g1
                g2_real=star_stamp.FindAdaptiveMom().observed_shape.g2
                #g1_real = -9999.
                #g2_real = -9999.
                sum_flux=numpy.sum(star_stamp.array)
                row = [ k,truth.x, truth.y, truth.ra, truth.dec,
                            g1_real, g2_real, truth.fwhm, truth.mom_size, truth.g1,
                            truth.g2, truth.mu, truth.z, truth.flux, sum_flux]
                truth_catalog.addRow(row)

            except:
                logger.info('Star %d has failed, skipping...',k)
                pdb.set_trace()


        # Add ccd noise
        logger.info('Adding CCD noise')
        noise = galsim.CCDNoise(
            rng, sky_level=0, gain=1/gain,
            read_noise=read_noise)
        full_image.addNoise(noise)

        logger.info('Added noise to final output image')


        # Now write the image to disk.
        full_image.write(file_name)

        # Add a FLUXSCL keyword for later stacking
        this_hdu=astropy.io.fits.open(file_name)
        this_hdu[0].header['FLXSCALE'] = 300.0/exp_time
        this_hdu.writeto(file_name,overwrite='True')
        logger.info('Wrote image to %r',file_name)


        # Write truth catalog to file.
        truth_catalog.write(truth_file_name)

        i=i+1
        logger.info('completed run %d for psf %s',i,psf_filen)

    logger.info('completed all images')

if __name__ == "__main__":
    main(sys.argv)
