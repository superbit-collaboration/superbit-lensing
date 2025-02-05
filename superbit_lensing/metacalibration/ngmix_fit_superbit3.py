import meds
import ngmix
from ngmix.medsreaders import NGMixMEDS
import numpy as np
import pickle
from astropy.table import Table, Row, vstack, hstack
import os, sys, time, traceback
from copy import deepcopy
import galsim
import galsim.des
import psfex
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

from multiprocessing import Pool
import multiprocessing

import superbit_lensing.utils as utils

import ipdb

parser = ArgumentParser()

parser.add_argument('medsfile', type=str,
                    help='MEDS file to process')
parser.add_argument('outfile', type=str,
                    help='Output filename')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory')
parser.add_argument('-start', type=int, default=None,
                    help='Starting index for MEDS processing')
parser.add_argument('-end', type=int, default=None,
                    help='Ending index for MEDS processing')
parser.add_argument('-n', type=int, default=1,
                    help='Number of cores to use')
parser.add_argument('-seed', type=int, default=None,
                    help='Metacalibration seed')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Set to make diagnstic plots')
parser.add_argument('--use_coadd', action='store_true', default=False,
                    help='Will use the coadd, if present')                      
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Overwrite output mcal file')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Make verbose')

class SuperBITNgmixFitter():
    """
    class to process a set of observations from a MEDS file.

    config: A dictionary that holds all relevant info for class construction,
            including meds file location
    """

    def __init__(self, config):

        # The meds file groups data together by source.
        # To run, we need to make an ObservationList object for each source.
        # Each ObservationList has a single Observation frome each epoch.
        # Each Observation object needs an image, a psf, a weightmap, and a wcs, expressed as a Jacobian object.
        #   The MEDS object has methods to get all of these.

        # Once that's done, we create a Bootstrapper, populate it with initial guesses
        #   and priors, and run it.
        # The results get compiled into a catalog (numpy structured array with named fields),
        #   which is either written to a .fits table or returned in memory.
        #
        # self.metcal will hold result of metacalibration
        # self.gal_results is the result of a simple fit to the galaxy shape within metcal bootstrapper

        self.config = config
        self.seed = config['seed']

        try:
            fname = os.path.join(config['outdir'], config['medsfile'])
            self.medsObj = NGMixMEDS(fname)
        except OSError:
            fname =config['medsfile']
            self.medsObj = NGMixMEDS(fname)

        self.catalog = self.medsObj.get_cat()

        self.has_coadd = bool(self.medsObj._meta['has_coadd'])
        self.use_coadd = config['use_coadd']

        try:
            self.verbose = config['verbose']
        except KeyError:
            self.verbose = False

        return

    def _get_priors(self):

        # This bit is needed for ngmix v2.x.x
        # won't work for v1.x.x
        rng = np.random.RandomState(self.seed)

        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014

        g_sigma = 0.3
        g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which would be zero)
        # and the sigma of the gaussians

        # units same as jacobian, probably arcsec
        row, col = 0.0, 0.0
        row_sigma, col_sigma = 0.2, 0.2 # a bit smaller than pix size of SuperBIT
        cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf)

        Tminval = -1.0 # arcsec squared
        Tmaxval = 1000
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

        # similar for flux.  Make sure the bounds make sense for
        # your images

        Fminval = -1.e1
        Fmaxval = 1.e5
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

        # now make a joint prior.  This one takes priors
        # for each parameter separately
        priors = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior)

        return priors

    def _make_psf(self,psf_pixel_scale = 0.01,nx = 256,ny = 256):

        """
        for debugging: make a GalSim PSF from scratch,
        based on the one implemented in simulations
        """
        import galsim
        import galsim.des

        jitter_fwhm = 0.1
        jitter = galsim.Gaussian(flux=1., fwhm=jitter_fwhm)

        lam_over_diam = 0.257831 # units of arcsec
        aberrations = np.zeros(38)             # Set the initial size.
        aberrations[0] = 0.                       # First entry must be zero
        aberrations[1] = -0.00305127
        aberrations[4] = -0.02474205              # Noll index 4 = Defocus
        aberrations[11] = -0.01544329             # Noll index 11 = Spherical
        aberrations[22] = 0.00199235
        aberrations[26] = 0.00000017
        aberrations[37] = 0.00000004

        optics = galsim.OpticalPSF(lam=625,diam=0.5,
                                       obscuration=0.38, nstruts=4,
                                       strut_angle=(90*galsim.degrees), strut_thick=0.087,
                                       aberrations=aberrations)

        psf = galsim.Convolve([jitter,optics])
        gpsf_cutout = psf.drawImage(scale=psf_pixel_scale,nx=nx,ny=ny)

        return gpsf_cutout

    def _make_psfex_cutouts(self,source_id=None):


        psfex_cutouts = []
        image_info = self.medsObj.get_image_info()
        ycoord = self.medsObj[source_id]['orig_row']
        xcoord = self.medsObj[source_id]['orig_col']
        file_id = self.medsObj[source_id]['file_id']

        # TO DO: make naming more general; requires some rewrite of medsmaker and would ideally go into the file_id loop
        #        have array of psf_pixel_scales if we go with drawing galsim.des PSFs
        #psf_name = '/users/jmcclear/data/superbit/forecasting-analysis/psfex_cutout_tests/psfex_output/superbit_gaussJitter_001_cat.psf'
        psf_name = '/users/jmcclear/data/superbit/forecasting-analysis/psf_stamp_tests/psfex_output/superbit_gaussJitter_001_cat.psf'

        pex = psfex.PSFEx(psf_name)

        for i in range(len(file_id)):

            im_name = image_info[file_id[i]][0]

            """
            im_name = im_name.replace('/users/jmcclear/data/superbit/superbit_lensing/GalSim/forecasting/',\
            '/Volumes/PassportWD/SuperBIT/mock-data-forecasting/')
            """
            psfex_des = galsim.des.DES_PSFEx(psf_name, im_name)
            this_psf_des=psfex_des.getPSF(galsim.PositionD(xcoord[i],ycoord[i]))
            psf_cutout = this_psf_des.drawImage(method='no_pixel')
            #print("automatically made a PSF model of size %d x %d" % (psf_cutout.array.shape[0],psf_cutout.array.shape[0]))

            #psf_cutout = pex.get_rec(xcoord[i],ycoord[i])
            psfex_cutouts.append(psf_cutout)

        return psfex_cutouts

    def _get_source_observations(self, iobj, weight_type='uberseg'):

        obslist = self.medsObj.get_obslist(iobj, weight_type)
        se_obslist = ngmix.ObsList(meta=deepcopy(obslist._meta))

        if self.use_coadd:
            print('Using coadd along with multi-epoch obs to do ngmix fitting')
            if not self.has_coadd:
                print('No coadd found, skipping...')
                for obs in obslist[:]:
                    se_obslist.append(obs)
                obslist = se_obslist
            elif self.has_coadd and obslist and (obslist[0].psf._pixels['ierr'] == np.inf)[0]:
                print('Coadd is present, however Coadd psf is missing (Comment out the make_external_header command in medsmaker_real.py), So skipping the coadd....')
                for obs in obslist[1:]:
                    se_obslist.append(obs)
                obslist = se_obslist
            else:
                for obs in obslist[:]:
                    se_obslist.append(obs)
                obslist = se_obslist
            
        else:
            print('Using only multi-epoch obs to do ngmix fitting')
            if self.has_coadd:
                se_obslist = ngmix.ObsList(meta=deepcopy(obslist._meta))
                for obs in obslist[1:]:
                    se_obslist.append(obs)
                obslist = se_obslist

        return obslist

class SuperBITPlotter(object):

    def __init__(self, pixel_scale=0.141):
        self.pixel_scale = pixel_scale

        return

    # def setup_jdict_list(self, index_start, index_end, meds_list):
    #     '''
    #     Need to grab jacobian dict list for later plotting in parallel
    #     '''

    #     self.jdict_list = []

    #     for i in range(index_start, index_end):
    #         try:
    #             self.jdict_list.append(meds_list.get_jacobian_list(i)[0])
    #         except ValueError:
    #             pass
    #     return

    @staticmethod
    def _make_imc(source_id, meds_list):
        imc_arr = []
        for imc in range(meds_list[source_id]['ncutout']):
                imc_arr.append(meds_list.get_cutout(source_id, imc, type='image'))

        return np.mean(imc_arr, axis=0)

    @classmethod
    def make_imc_list(cls, index_start, index_end, meds_list):
        '''
        assumes you have already checked that make_plots is True, as
        it will take time to make images
        '''

        imc_list = []

        for i in range(index_start, index_end):
            imc_list.append(cls._make_imc(i, meds_list))

        return imc_list

    def _get_adaptmom(self, im, jac, vb=False):

        """
        Utility function to return the result of a galsim.FindAdaptiveMom()
        fit for an arbitrary 2x2 ndarray
        """

        galsim_im = galsim.Image(im,scale=jac.get_scale())
        guess_centroid = galsim.PositionD(galsim_im.true_center.x-1, galsim_im.true_center.y-1)
        try:

            hsm_fit = galsim_im.FindAdaptiveMom(guess_centroid=guess_centroid)
            hsm_sigma = hsm_fit.moments_sigma * self.pixel_scale
            hsm_g1 = hsm_fit.observed_shape.g1
            hsm_g2 = hsm_fit.observed_shape.g2

        except galsim.errors.GalSimHSMError:

            if vb is True:
                print('HSM fit failed; source is likely spurious')
            hsm_sigma = -9999; hsm_g1 = -9999; hsm_g2 = -9999

        outdict = {'hsm_sigma':hsm_sigma, 'hsm_g1':hsm_g1, 'hsm_g2':hsm_g2}

        return outdict

    def make_plots(self, imc, gmix, jac, outname, verbose=False):
        """
        Method to make plots of cutout, ngmix psf-convolved
        galaxy model, and residuals, all with galaxy size information
        (Gaussian sigma, ~FWHM/2.355) and colorbars.
        """

        cutout_avg = imc
        model_image = gmix.make_image(cutout_avg.shape, jacobian=jac)

        model_hsm_sigma = self._get_adaptmom(model_image,jac,verbose)['hsm_sigma']
        real_hsm_sigma = self._get_adaptmom(cutout_avg,jac,verbose)['hsm_sigma']

        y, x = gmix.get_cen()
        y /= jac.get_scale()
        x /= jac.get_scale()

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))

        #norm=colors.PowerNorm(gamma=1.2)
        real=ax1.imshow(cutout_avg, vmax=(cutout_avg.max()*.90))
        ax1.plot((x+(cutout_avg.shape[1]-1)/2), (y+(cutout_avg.shape[0]-1)/2), 'xr', \
                markersize=11, label='gmix_cen+row0/col0')

        ax1.axvline((cutout_avg.shape[0]-1)*0.5, color='black')
        ax1.axhline((cutout_avg.shape[1]-1)*0.5, color='black')
        ax1.plot(jac.col0, jac.row0, 'sw', fillstyle='none', \
                markersize=10, label='row0/col0')
        ax1.legend()
        ax1.set_title('cutout HSM sigma = %.3f' % real_hsm_sigma,fontsize=10)

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(real, cax=cax)

        model=ax2.imshow(model_image, vmax=(cutout_avg.max()*.90))
        ax2.plot((x+(cutout_avg.shape[0]-1)/2), (y+(cutout_avg.shape[1]-1)/2), 'xr', \
                markersize=11, label='gmix_cen+row0/col0')
        ax2.axvline((cutout_avg.shape[0]-1)*0.5, color='black')
        ax2.axhline((cutout_avg.shape[1]-1)*0.5, color='black')
        ax2.plot(jac.col0, jac.row0, 'sw', fillstyle='none', \
                     markersize=10, label='row0/col0')
        ax2.legend()
        ax2.set_title('model HSM sigma = %.3f' % model_hsm_sigma,fontsize=10)

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(model, cax=cax)

        resi = ax3.imshow(cutout_avg - model_image)
        ax3.set_title('cutout - model',fontsize=10)

        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(resi, cax=cax)
        fig.tight_layout()

        fig.savefig(outname)
        plt.close(fig)

        return 0

def set_seed(config):
    seed = int(time.time())
    config['seed'] = seed

    return

def make_output_table(outfilename, mcal, identifying, return_table=False):
    """
    Create an output table and add in metacal/ngmix galaxy fit parameters

    :mcal: array containing output parameters from the metacalibration fit
    :identifying: contains ra, dec and coadd cat ID for cross-matching if needed

    TO DO: find more compact/pythonic way of doing this exercise?
    """

    join_tab = mcal_dict2tab(mcal, identifying)

    write_output_table(outfilename, join_tab)

    if return_table is True:
        return join_tab
    else:
        return 0

def mcal_dict2tab(mcal, ident):
    '''
    mcal is the dict returned by ngmix.get_metacal_result()

    ident is an array with MEDS identification info like id, ra, dec
    not returned by the function
    '''

    # Annoying, but have to do this to make Table from scalars
    for key, val in ident.items():
        ident[key] = np.array([val])

    tab_names = ['noshear', '1p', '1m', '2p', '2m', '1p_psf', '1m_psf', '2p_psf', '2m_psf','MC']
    for name in tab_names:
        tab = mcal[name]

        for key, val in tab.items():
            tab[key] = np.array([val])

        mcal[name] = tab

    id_tab = Table(data=ident)

    tab_noshear = Table(mcal['noshear'])
    tab_1p = Table(mcal['1p'])
    tab_1m = Table(mcal['1m'])
    tab_2p = Table(mcal['2p'])
    tab_2m = Table(mcal['2m'])
    tab_1p_psf = Table(mcal['1p_psf'])
    tab_1m_psf = Table(mcal['1m_psf'])
    tab_2p_psf = Table(mcal['2p_psf'])
    tab_2m_psf = Table(mcal['2m_psf'])
    tab_MC = Table(mcal['MC'])

    join_tab = hstack([id_tab, hstack([tab_noshear, tab_1p,  tab_1m, tab_2p, tab_2m, tab_1p_psf, tab_1m_psf, tab_2p_psf, tab_2m_psf, tab_MC], \
                                      table_names=tab_names)])

    return join_tab

def write_output_table(outfilename, tab, overwrite=False):
    tab.write(outfilename, format='fits', overwrite=overwrite)

    return

# def mcal_dict2table(mcal, ident):
#     '''
#     mcal is the dict returned by ngmix.get_metacal_result()
#     ident is an array with MEDS identification info like id, ra, dec
#     not returned by the function
#     '''

#     id_tab = Table(data=ident)

#     tab_noshear = Table(mcal['noshear'])
#     tab_1p = Table(mcal['1p'])
#     tab_1m = Table(mcal['1m'])
#     tab_2p = Table(mcal['2p'])
#     tab_2m = Table(mcal['2m'])

#     return mcal_arr

def mp_fit_one(source_id, obslist, prior, logprint, rng, pars=None):
    """
    Multiprocessing version of original _fit_one()

    Method to perfom metacalibration on an object. Returns the unsheared ellipticities
    of each galaxy, as well as entries for each shear step

    inputs:
    - source_id: MEDS ID
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - pars: mcal running parameters

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """

    index = source_id

    if pars is None:
        mcal_shear = 0.01
        lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
        max_pars = {'method':'lm', 'lm_pars':lm_pars, 'find_center':True}
        types = ['noshear', '1p', '1m', '2p', '2m', '1p_psf', '1m_psf', '2p_psf', '2m_psf']
        #psf = 'fitgauss'
        #metacal_pars={'step':mcal_shear, 'rng': rng, 'psf': psf, 'types': types}
        metacal_pars={'step':mcal_shear, 'rng': rng}
    else:
        mcal_shear = metacal_pars['step']
        max_pars = pars['max_pars']
        metacal_pars = pars['metacal_pars']

    # get image pixel scale (assumes constant across list)
    jacobian = obslist[0]._jacobian
    Tguess = 4*jacobian.get_scale()**2
    ntry = 4
    psf_model = 'gauss' # should come up with diagnostics for PSF quality
    # gal_model = 'exp'
    gal_model = 'gauss'

    # Run the actual metacalibration fits on the observed galaxies
    mcb = ngmix.bootstrap.MaxMetacalBootstrapper(obslist)
    mcb.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior,
                    ntry=ntry, metacal_pars=metacal_pars)
    mcal_res = mcb.get_metacal_result() # this is a dict

    mcal_res = add_mcal_responsivities(mcal_res, mcal_shear)

    r11 = mcal_res['MC']['r11']
    r22 = mcal_res['MC']['r22']

    # To generate a model image, these calls do need to be here
    mcb.fit_psfs(psf_model, 1.)
    mcb.fit_max(gal_model, max_pars, prior=prior)

    mcal_fit = mcb.get_max_fitter()

    return mcal_res, mcal_fit

def add_mcal_responsivities(mcal_res, mcal_shear=0.01):
    '''
    Compute and add the mcal responsivity values to the output
    result dict from get_metacal_result()
    '''

    # Define full responsivity matrix, take inner product with shear moments
    r11 = (mcal_res['1p']['g'][0] - mcal_res['1m']['g'][0]) / (2*mcal_shear)
    r12 = (mcal_res['2p']['g'][0] - mcal_res['2m']['g'][0]) / (2*mcal_shear)
    r21 = (mcal_res['1p']['g'][1] - mcal_res['1m']['g'][1]) / (2*mcal_shear)
    r22 = (mcal_res['2p']['g'][1] - mcal_res['2m']['g'][1]) / (2*mcal_shear)

    R = [ [r11, r12], [r21, r22] ]
    Rinv = np.linalg.inv(R)
    gMC = np.dot(Rinv,
                 mcal_res['noshear']['g']
                 )

    MC = {
        'r11':r11, 'r12':r12,
        'r21':r21, 'r22':r22,
        'g1_MC':gMC[0], 'g2_MC':gMC[1]
        }

    mcal_res['MC'] = MC

    return mcal_res

def setup_obj(i, meds_obj):
    '''
    Setup object property dictionary used to compile fit params later on
    '''

    # Mcal object properties
    obj = {}

    obj['meds_indx'] = i
    obj['id'] = meds_obj['id']
    obj['ra'] = meds_obj['ra']
    obj['dec'] = meds_obj['dec']
    obj['XWIN_IMAGE'] = meds_obj['XWIN_IMAGE']
    obj['YWIN_IMAGE'] = meds_obj['YWIN_IMAGE']
    obj['ncutout'] = meds_obj['ncutout']

    return obj

def check_obj_flags(obj, min_cutouts=1):
    '''
    Check if MEDS obj has any flags.

    obj: meds.MEDS row
        An element of the meds.MEDS catalog
    min_cutouts: int
        Minimum number of image cutouts per object

    returns: is_flagged (bool), flag_name (str)
    '''

    # check that at least min_cutouts is stored in image data
    if obj['ncutout'] < min_cutouts:
        return True, 'min_cutouts'

    # other flags...

    return False, None

def mp_run_fit(i, start_ind, obj, obslist, prior, imc,
               plotter, config, logprint, rng):
    '''
    parallelized version of original ngmix_fit_superbit3 code

    i: MEDS indx

    returns ...
    '''

    start = time.time()

    logprint(f'Starting fit for obj {i}')

    if obslist is None:
        logprint('obslist is None')

    try:
        # first check if object is flagged
        flagged, flag_name = check_obj_flags(obj)

        if flagged is True:
            raise Exception(f'Object flagged with {flag_name}')

        # mcal_res: the bootstrapper's get_mcal_result() dict
        # mcal_fit: the mcal model image
        mcal_res, mcal_fit = mp_fit_one(i, obslist, prior, logprint, rng)

        # Ain some identifying info like (ra,dec), id, etc.
        # for key in obj.keys():
        #     mcal_res[key] = obj[key]

        # convert result dict to a formatted table
        # obj here is the "identifying" table
        mcal_tab = mcal_dict2tab(mcal_res, obj)

        end = time.time()
        logprint(f'Fitting and conversion took {end-start} seconds')

        if config['make_plots'] is True:
            image_cutout = imc
            #jdict = plotter.jdict_list[i]
            jdict = plotter.jdict_list[i-start_ind] # need to change that so allows for i-start_index
            #logprint("\n\nlength of plotter.jdict_list is %d\n\n" % len(plotter.jdict_list))

            jac = ngmix.Jacobian(row=jdict['row0'],
                                 col=jdict['col0'],
                                 dvdrow=jdict['dvdrow'],
                                 dvdcol=jdict['dvdcol'],
                                 dudrow=jdict['dudrow'],
                                 dudcol=jdict['dudcol']
                                 )
            try:
                gmix = mcal_fit.get_convolved_gmix()
                filen = 'diagnostics-' + str(int(i)) + '.png'
                outname = os.path.join(config['im_savedir'], filen)
                plotter.make_plots(imc, gmix, jac, outname, logprint.vb)

                logprint('Plots made')

            except:
                # EM probably failed
                logprint('Bad gmix model, no image made')

    except Exception as e:
        logprint(f'Exception: {e}')
        logprint(f'object {i} failed, skipping...')

        return Table()

    end = time.time()

    logprint(f'Total runtime for object was {end-start} seconds')

    return mcal_tab

def main():

    args = parser.parse_args()

    vb = args.vb # if True, prints out values of R11/R22 for every galaxy
    medsfile = args.medsfile
    outfilename = args.outfile
    outdir = args.outdir
    index_start  = args.start
    index_end = args.end
    make_plots = args.plot
    nproc = args.n
    seed = args.seed
    use_coadd = args.use_coadd
    overwrite = args.overwrite
    identifying = {'meds_index':[], 'id':[], 'ra':[], 'dec':[]}
    mcal = {'noshear':[], '1p':[], '1m':[], '2p':[], '2m':[]}
    rng  = np.random.RandomState(seed)

    # Test for existence of the "outdir" argument. If the "outdir" argument is
    # not given, set it to a default value (current working directory).
    # If the "outdir" argument is given in config but the "outdir" directory
    # itself doesn't exist, create it.

    if outdir is None:
        outdir = os.getcwd()

    if not os.path.isdir(outdir):
       	  cmd = 'mkdir -p %s' % outdir
          os.system(cmd)

    if make_plots is True:
        print('per-object diagnostic plotting enabled')
    else:
        print('--plots=False; no per-object diagnostic plots will be generated')

    # Set up for saving plots
    im_savedir = os.path.join(outdir, 'metacal-plots/')
    if (make_plots == True):
       if not os.path.isdir(im_savedir):
       	  cmd = 'mkdir -p %s' % im_savedir
          os.system(cmd)

    # Added to handle rng initialization
    # Could put everything through here instead
    config = {}
    config['medsfile'] = medsfile
    config['outfile'] = outfilename
    config['outdir'] = outdir
    config['verbose'] = vb
    config['make_plots'] = make_plots
    config['im_savedir'] = im_savedir
    config['nproc'] = nproc
    config['use_coadd'] = use_coadd
    

    if seed is not None:
        config['seed'] = seed
    else:
        set_seed(config)

    logdir = outdir
    logfile = 'mcal_fitting.log'
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    logprint(f'MEDS file: {medsfile}')
    logprint(f'index start, end: {index_start}, {index_end}')
    logprint(f'outfile: {os.path.join(outdir, outfilename)}')
    logprint(f'make_plots: {make_plots}')
    logprint(f'im_savedir: {im_savedir}')
    logprint(f'nproc: {nproc}')
    logprint(f'Use coadd: {use_coadd}')
    logprint(f'vb: {vb}')
    logprint(f'seed: {config["seed"]}')

    BITfitter = SuperBITNgmixFitter(config)

    priors = BITfitter._get_priors()

    Ncat = len(BITfitter.catalog)
    if index_start == None:
        index_start = 0
    if index_end == None:
        index_end = Ncat

    if index_end > Ncat:
        logprint(f'Warning: index_end={index_end} larger than ' +\
                 f'catalog size of {Ncat}; running over full catalog')
        index_end = Ncat

    # Needed for making plots on each worker
    plotter = SuperBITPlotter()

    if make_plots is True:
        imc_list = plotter.make_imc_list(index_start, index_end, BITfitter.medsObj)
    else:
        imc_list = [None for i in range(index_start, index_end)]

    logprint(f'Starting metacal fitting with {nproc} cores')

    start = time.time()

    # for no multiprocessing:
    if nproc == 1:
        mcal_res = []
        for i in range(index_start, index_end):
            mcal_res.append(mp_run_fit(
                            i,
                            index_start,
                            setup_obj(i, BITfitter.medsObj[i]),
                            BITfitter._get_source_observations(
                                i, logprint=logprint
                                ),
                            priors,
                            imc_list[i-index_start],
                            plotter,
                            config,
                            logprint, rng)
                            )

        mcal_res = vstack(mcal_res)

    # for multiprocessing:
    else:
        with Pool(nproc) as pool:
            mcal_res = vstack(pool.starmap(mp_run_fit,
                                        [(i,
                                          index_start,
                                          setup_obj(i, BITfitter.medsObj[i]),
                                          BITfitter._get_source_observations(
                                              i, logprint=logprint
                                              ),
                                          priors,
                                          imc_list[i-index_start],
                                          plotter,
                                          config,
                                          logprint, rng
                                          ) for i in range(index_start, index_end)
                                          ]
                                        )
                            )

    end = time.time()

    T = end - start
    logprint(f'Total fitting and stacking time: {T} seconds')

    N = index_end - index_start
    logprint(f'{T/N} seconds per object (wall time)')
    logprint(f'{T/N*nproc} seconds per object (CPU time)')


    if not os.path.isdir(outdir):
       cmd='mkdir -p %s' % outdir
       os.system(cmd)

    out = os.path.join(outdir, outfilename)
    logprint(f'Writing results to {out}')

    write_output_table(out, mcal_res, overwrite=overwrite)

    logprint('Done!')

    return

if __name__ == '__main__':
    main()
    """
    try:
        main()
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    """
