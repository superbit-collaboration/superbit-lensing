import ngmix
from ngmix.medsreaders import NGMixMEDS
import numpy as np
from astropy.table import Table, Row, vstack, hstack
import os, sys, time, traceback
from copy import deepcopy
from argparse import ArgumentParser
import time

from multiprocessing import Pool
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
parser.add_argument('-psf_model', type=str, default='gauss',
                    help='PSF model to use')
parser.add_argument('-gal_model', type=str, default='gauss',
                    help='Galaxy model to use')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Set to make diagnstic plots')
parser.add_argument('--use_coadd', action='store_true', default=False,
                    help='Will use the coadd, if present')                    
parser.add_argument('--use_coadd_only', action='store_true', default=False,
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
            print(fname)
            self.medsObj = NGMixMEDS(fname)

        self.catalog = self.medsObj.get_cat()
        self.radius = self.medsObj["KRON_RADIUS"]
        self.Tmax_vals = self.Tmax_from_radius()
        
        self.has_coadd = bool(self.medsObj._meta['has_coadd'])
        self.use_coadd = config['use_coadd']
        self.use_coadd_only = config['use_coadd_only']

        try:
            self.verbose = config['verbose']
        except KeyError:
            self.verbose = False

        return

    def Tmax_from_radius(self):
        return (4*self.radius*0.141)**2

    def _get_priors(self, Tmaxval = 1000):

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

    def _get_source_observations(self, iobj, weight_type='uberseg'):

        obslist = self.medsObj.get_obslist(iobj, weight_type)
        se_obslist = ngmix.ObsList(meta=deepcopy(obslist._meta))

        if self.use_coadd_only:
            print('Using only coadd to do ngmix fitting')
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
                for obs in obslist[:1]:
                    se_obslist.append(obs)
                obslist = se_obslist
            return obslist

        elif self.use_coadd:
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
            return obslist
            
        else:
            print('Using only multi-epoch obs to do ngmix fitting')
            if self.has_coadd:
                se_obslist = ngmix.ObsList(meta=deepcopy(obslist._meta))
                for obs in obslist[1:]:
                    se_obslist.append(obs)
                obslist = se_obslist
            return obslist

def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

def get_coellip_ngauss(name):
    ngauss=int( name[7:] )
    return ngauss

def set_seed(config):
    seed = int(time.time())
    config['seed'] = seed

    return

def write_output_table(outfilename, tab, overwrite=False):
    tab.write(outfilename, format='fits', overwrite=overwrite)

    return

def mcal_dict2tab(mcal, obsdict, ident):
    '''
    mcal is the dict returned by ngmix.get_metacal_result()

    ident is an array with MEDS identification info like id, ra, dec
    not returned by the function
    '''

    # Annoying, but have to do this to make Table from scalars
    for key, val in ident.items():
        ident[key] = np.array([val])

    tab_names = ['noshear', '1p', '1m', '2p', '2m', '1p_psf', '1m_psf', '2p_psf', '2m_psf']
    for name in tab_names:
        tab = mcal[name]

        # Remove "pars_cov0" and "pars_cov" keys if they exist
        for key_to_remove in ["pars_cov0", "pars_cov"]:
            if key_to_remove in tab:
                del tab[key_to_remove]

        for key, val in tab.items():
            tab[key] = np.array([val])

        # Get the psf T by averaging over epochs (and eventually bands)
        tpsf_list = []
        gpsf_list = []
        obs = obsdict[name]

        for i in range(len(obs)):
            try:
                resdict = obs[i].psf.meta['result']
                gm = resdict.get_gmix()
                g1psf, g2psf, Tpsf = gm.get_g1g2T()
                gpsf = np.array([g1,g2])
                tpsf_list.append(Tpsf)
                gpsf_list.append(gpsf)
            except:
                pass

        tab['Tpsf'] = np.array([np.mean(tpsf_list)]) if tpsf_list else np.array([np.nan])
        tab['gpsf'] = np.array([np.mean(gpsf_list, axis=0)]) if gpsf_list else np.array([np.nan, np.nan])

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

    join_tab = hstack([id_tab, hstack([tab_noshear, tab_1p, tab_1m, tab_2p, tab_2m,
                                       tab_1p_psf, tab_1m_psf, tab_2p_psf, tab_2m_psf],
                                      table_names=tab_names)])

    return join_tab

def mp_fit_one(obslist, prior, rng, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
    """
    Multiprocessing version of original _fit_one()

    Method to perfom metacalibration on an object. Returns the unsheared ellipticities
    of each galaxy, as well as entries for each shear step

    inputs:
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - mcal_pars: mcal running parameters

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """
    # get image pixel scale (assumes constant across list)
    jacobian = obslist[0]._jacobian
    Tguess = 4*jacobian.get_scale()**2
    ntry = 20
    lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    psf_lm_pars={'maxfev': 4000, 'xtol':5.0e-5,'ftol':5.0e-5}

    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    # psf fitting
    if 'em' in psf_model:
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}
        psf_ngauss = get_em_ngauss(psf_model)
        psf_fitter = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in psf_model:
        psf_ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif psf_model == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    else:
        raise ValueError('psf_model must be one of emn, coellipn, or gauss')

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)

    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

    #types = ['noshear', '1p', '1m', '2p', '2m']
    psf = mcal_pars['psf']
    mcal_shear = mcal_pars['mcal_shear']
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        step = mcal_shear,
        #types=types,
    )

    resdict, obsdict = boot.go(obslist)

    return resdict, obsdict


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

def mp_run_fit(i, obj, obslist, prior,
               logprint, rng, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
    '''
    parallelized version of original ngmix_fit code

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
        resdict, obsdict = mp_fit_one(obslist, prior, rng, psf_model=psf_model, gal_model=gal_model, mcal_pars=mcal_pars)

        # Ain some identifying info like (ra,dec), id, etc.
        # for key in obj.keys():
        #     mcal_res[key] = obj[key]

        # convert result dict to a formatted table
        # obj here is the "identifying" table
        mcal_tab = mcal_dict2tab(resdict, obsdict, obj)

        end = time.time()
        logprint(f'Fitting and conversion took {end-start} seconds')

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
    psf_model = args.psf_model
    gal_model = args.gal_model
    overwrite = args.overwrite
    use_coadd = args.use_coadd
    use_coadd_only = args.use_coadd_only
    rng  = np.random.RandomState(seed)
    mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}

    if outdir is None:
        outdir = os.getcwd()

    if not os.path.isdir(outdir):
        cmd = 'mkdir -p %s' % outdir
        os.system(cmd)

    # Added to handle rng initialization
    # Could put everything through here instead
    config = {}
    config['medsfile'] = medsfile
    config['outfile'] = outfilename
    config['outdir'] = outdir
    config['verbose'] = vb
    config['make_plots'] = make_plots
    config['nproc'] = nproc
    config['use_coadd'] = use_coadd
    config['use_coadd_only'] = use_coadd_only

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
    logprint(f'nproc: {nproc}')
    logprint(f'psf_model: {psf_model}')
    logprint(f'gal_model: {gal_model}')
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

    logprint(f'Starting metacal fitting with {nproc} cores')

    start = time.time()

    # for no multiprocessing:
    if nproc == 1:
        mcal_res = []
        for i in range(index_start, index_end):
            mcal_res.append(mp_run_fit(
                            i,
                            setup_obj(i, BITfitter.medsObj[i]),
                            BITfitter._get_source_observations(i),
                            BITfitter._get_priors(Tmaxval = BITfitter.Tmax_vals[i]),
                            logprint, rng, psf_model, gal_model, mcal_pars)
                            )

        mcal_res = vstack(mcal_res)

    # for multiprocessing:
    else:
        with Pool(nproc) as pool:
            mcal_res = vstack(pool.starmap(mp_run_fit,
                                        [(i,
                                          setup_obj(i, BITfitter.medsObj[i]),
                                          BITfitter._get_source_observations(i),
                                          BITfitter._get_priors(Tmaxval = BITfitter.Tmax_vals[i]),
                                          logprint, rng, psf_model, gal_model, mcal_pars) for i in range(index_start, index_end)
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

    return 0

if __name__ == '__main__':
    rc = main()

    if rc !=0:
        print(f'process_mocks failed w/ return code {rc}!')