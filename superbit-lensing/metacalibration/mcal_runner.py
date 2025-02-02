import ngmix
from ngmix.medsreaders import NGMixMEDS
from ngmix.fitting import Fitter
import numpy as np
import os
from copy import deepcopy
from collections.abc import Mapping
from multiprocessing import Pool
import time
from astropy.table import Table, vstack, hstack
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import superbit-lensing.utils as utils

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-start', type=int, default=0,
                        help='Starting MEDS index for mcal fitting')
    parser.add_argument('-end', type=int, default=1,
                        help='Ending MEDS index for mcal fitting')
    parser.add_argument('-ncores', type=int, default=1,
                        help='Number of cores to use for mcal fitting')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

class CaseInsensitiveDict(Mapping):
    '''
    We want a case-insensitive dictionary to ease
    the mapping between fitter class names & user
    inputs
    '''

    def __init__(self, d):
        self._d = d
        self._s = dict((k.lower(), k) for k in d)
    def __contains__(self, k):
        return k.lower() in self._s
    def __len__(self):
        return len(self._s)
    def __iter__(self):
        return iter(self._s)
    def __getitem__(self, k):
        return self._d[self._s[k.lower()]]
    def actual_key_case(self, k):
        return self._s.get(k.lower())

# NOTE: This is where you must register ngmix fitters
# NOTE: Capitalization here matches ngmix conventions,
# but isn't case-sensitive when actually building
GAL_FITTERS = CaseInsensitiveDict({
    'default': ngmix.fitting.Fitter,
    'Fitter': ngmix.fitting.Fitter,
    'GaussMom': ngmix.gaussmom.GaussMom,
    'Coellip': ngmix.fitting.CoellipFitter,
    'Galsim': ngmix.fitting.GalsimFitter,
    'GalsimSpergel': ngmix.fitting.GalsimSpergelFitter,
    })
PSF_FITTERS = CaseInsensitiveDict({
    'default': ngmix.fitting.CoellipFitter,
    'Fitter': ngmix.fitting.Fitter,
    'Coellip': ngmix.fitting.CoellipFitter,
    'PSFFlux': ngmix.fitting.PSFFluxFitter,
    'GaussMom': ngmix.gaussmom.GaussMom,
    'Galsim': ngmix.fitting.GalsimFitter,
    'GalsimMoffat': ngmix.fitting.GalsimMoffatFitter,
    'GalsimPSFFlux': ngmix.fitting.GalsimPSFFluxFitter,
})
# These are the base classes for all allowed fitters
BASE_FITTERS = (
    ngmix.fitting.Fitter,
    ngmix.fitting.PSFFluxFitter,
    ngmix.fitting.GalsimFitter,
    ngmix.fitting.GalsimPSFFluxFitter,
    ngmix.gaussmom.GaussMom
    )

def build_fitter(fit_type, fitter_name, kwargs):
    '''
    fit_type: str
        Set to ether 'gal' or 'psf'. 'source' is also accepted
        in place of gal
    fitter_name: str
        Name of ngmix fitter
    kwargs: dict
        Keyword args to pass to fitter constructor
    '''

    # we'll allow some sloppiness and assume source means gal
    _allowed_fit_types = ['source', 'gal', 'psf']

    fit_type = fit_type.lower()
    name = fitter_name.lower()

    if fit_type not in _allowed_fit_types:
        raise ValueError(f'{fit_type} is not an allowed fit_type! ' +\
                         f'must be one of {_allowed_fit_types}')

    if fit_type in ['gal', 'source']:
        FITTERS = GAL_FITTERS
    elif fit_type == 'psf':
        FITTERS = PSF_FITTERS

    name = name.lower()

    if name in FITTERS.keys():
        # User-defined input construction
        fitter = FITTERS[name](**kwargs)
    else:
        raise ValueError(f'{name} is not a registered {fitter_type} fitter!')

    return fitter

class MetacalRunner(object):
    '''
    A helper class to organize interaction w/ various ngmix
    metacalibration fitters, bootstrappers, guessers, etc.
    '''

    _base_fitters = BASE_FITTERS
    _gal_fitters = GAL_FITTERS
    _psf_fitters = PSF_FITTERS

    # for custom masking / deblending of neighbors
    _deblend_types = ['seg']

    # uberseg will mask out any pixels that are closer
    # to another detected source than the fitted obj
    _default_weight_type = 'uberseg'

    def __init__(self, medsfile, vb=False, logprint=None):
        '''
        medsfile: str
            The medsfile that we will use for metacalibration
        vb: bool
            Turn on for verbose printing
        logprint: LogPrint
            A LogPrint object, which simultaneously handles
            logging & printing. Takes precedence over vb
        '''

        self.medsfile = medsfile

        if logprint is None:
            # don't log, just print
            logprint = utils.LogPrint(None, vb)
        else:
            vb = logprint.vb
        self.logprint = logprint
        self.vb = vb

        self.meds = NGMixMEDS(medsfile)
        self.has_coadd = bool(self.meds._meta['has_coadd'])
        self.cat = self.meds.get_cat()
        self.Nobjs = len(self.cat)

        self.pixel_scale = self.get_pixel_scale(0)

        self.fitter = None
        self.psf_fitter = None
        self.shear_step = None
        self.seed = None
        self.rng = None

        self.prior = None
        self.guesser = None
        self.psf_guesser = None
        self.lm_pars = None

        self.mcal_table = None

        return

    def set_seed(self, seed=None):
        '''
        seed: int
            The random seed to use for mcal runs. Will set
            a default using time() if none is passed
        '''

        if seed is None:
            seed = np.random.randint(0, 2**32-1)

        self.seed = seed

        self.set_rng()

        return

    def set_rng(self):
        self.rng = np.random.RandomState(self.seed)

        return

    def get_pixel_scale(self, iobj, icutout=0):
        '''
        Get pixel scale from image jacobian

        NOTE: While this function allows for the pixel_scale to vary across
        stamps, the way we use it assumes same pix_scale for all cutouts

        iobj: int
            The MEDS index to grab the pixel_scale for
        icutout: int
            The cutout index for the chosen MEDS object
        '''

        jac_list = self.get_jacobians(iobj)

        return jac_list[icutout].get_scale()

    def setup_fitters(self, gal_fitter, psf_fitter, gal_kwargs={},
                      psf_kwargs={}):
        '''
        gal_fitter: str, ngmix.fitting object
            Either the name of or instance of a ngmix Fitter class
        psf_fitter: str, ngmix.fitting object
            Either the name of or instance of a ngmix Fitter class
        gal_kwargs: dict
            A kwarg dictionary of args needed to build the gal Fitter
        psf_kwargs: dict
            A kwarg dictionary of args needed to build the psf Fitter
        '''

        fitters = {
            'gal': gal_fitter,
            'psf': psf_fitter
        }
        fitter_kwargs = {
            'gal': gal_kwargs,
            'psf': psf_kwargs
        }

        # make sure each fitter is valid, and construct if necessary
        for name in ['gal', 'psf']:
            fitter = fitters[name]
            kwargs = fitter_kwargs[name]
            kwargs.update({'fit_pars':self.lm_pars})
            if isinstance(fitter, str):
                # build_fitter() will automatically check for validity
                fitters[name] = build_fitter(
                    name, fitter, kwargs
                    )
            elif isinstance(fitter, self._base_fitters):
                # is guaranteed to be an allowed ngmix fitter
                break
            else:
                f = getattr(fitter, 'go', None)
                if f is not None:
                    if callable(f):
                        self.logprint(f'WARNING! {fitter} is not a ' +\
                                 'registered ngmix fitter, but does have ' +\
                                 ' a go() method. Will proceed')
                    else:
                        raise AttributeError(f'{fitter} does not have a ' +\
                                             'callable go() method!')
                else:
                    raise AttributeError(f'{fitter} does not have a ' +\
                                         'go() method!')

        self.fitter = fitters['gal']
        self.psf_fitter = fitters['psf']

        return

    def setup_bootstrapper(self, gal_fitter, psf_fitter, shear_step,
                           gal_kwargs={}, psf_kwargs={}, lm_pars=None,
                           guesser=None, psf_guesser=None, prior=None,
                           ntry=1):
        '''
        Initialize ngmix bootstrapper for metacal measurement

        gal_fitter: ngmix.Fitter
            The desired ngmix fitter for the source image
        psf_fitter: ngmix.Fitter
            The desired ngmix psf fitter
        shear_step: float
            The step size for the shear finite-difference derivative
        guesser: ngmix.Guesser
            A ngmix Guesser object for the source fit. Defaults to
            a TPSFFluxAndPrior guesser
        psf_guesser: ngmix.Guesser
            A ngmix Guesser object for the psf fit. Defaults to a
            SimplePSFGuesser
        prior: ngmix.prior
            A ngmix Prior object. Defaults to a simple joint prior
        lm_pars: dict
            A dictionary of Levenberg–Marquardt algorithm parameters
        ntry: int
            The number of times to try the fit before failure
        '''

        self.shear_step = shear_step

        #----------------------------------------------------------------------
        # Setup prior, guessers, and Levenberg–Marquardt parameters, if not
        # already set
        if self.prior is None:
            self.setup_prior(prior)

        # guessers can only be set together
        if (self.guesser is None) or (self.psf_guesser is None):
            self.setup_guessers(guesser, psf_guesser)

        if self.lm_pars is None:
            self.setup_lm_pars(lm_pars)

        #----------------------------------------------------------------------
        # Setup fitters

        # fitters can only be set together
        if (self.fitter is None) or (self.psf_fitter is None):
            self.setup_fitters(
                gal_fitter,
                psf_fitter,
                gal_kwargs=gal_kwargs,
                psf_kwargs=psf_kwargs
                )

        # the runners run the measurement code on observations
        psf_runner = ngmix.runners.PSFRunner(
            fitter=self.psf_fitter, guesser=self.psf_guesser, ntry=ntry
            )
        runner = ngmix.runners.Runner(
            fitter=self.fitter, guesser=self.guesser, ntry=ntry
            )

        #----------------------------------------------------------------------
        # the bootstrapper automates the metacal image shearing as well as both psf
        # and object measurements
        self.boot = ngmix.metacal.MetacalBootstrapper(
            runner=runner,
            psf_runner=psf_runner,
            step=shear_step,
            rng=self.rng,
            # TODO: check into this later, we may want to specify PSF explicitly
            # psf=args.psf
        )

        return

    def _get_fit_args(self, iobj):
        '''
        Get the args & kwargs to run the metacal measurement for
        a single object

        iobj: int
            MEDS index of object to be fit
        '''

        obs = self.get_obslist(iobj)
        obj_info = self.get_obj_info(iobj)

        args = [iobj, self.boot, obs, obj_info, self.shear_step]
        kwargs = {'logprint': self.logprint}

        return args, kwargs

    @staticmethod
    def _fit_one(iobj, bootstrapper, obs, obj_info, mcal_shear, logprint):
        '''
        A static method to wrap the mcal fitting to allow
        for multiprocessing

        iobj: int
            The MEDS index of the object to fit
        bootstrapper: ngmix.bootstrapper
            The ngmix bootstrapper that will automate the mcal fit
        obs: ngmix.Observation
            All requisite images & metadata of the obj to fit
        obj_info: A dictionary of object properties to add to standard
            mcal return dict
        mcal_shear: float
            The applied shear to the metacal images
        logprint: LogPrint
            A LogPrint object, which simultaneously handles
            logging & printing

        returns: astropy.Table
            A table that holds all mcal info for the obj, including
            responsivities
        '''

        logprint(f'Starting fit for obj {iobj}')

        try:
            # first check if object is flagged
            flagged, flag_name = check_obj_flags(obj_info)

            if flagged is True:
                raise Exception(f'Object flagged with {flag_name}')

            res_dict, obs_dict = bootstrapper.go(obs)

            # compute value-added cols such as gamma-only responsivity,
            # PSF size, "roundified" s2n, etc.
            add_mcal_cols(res_dict, obs_dict, mcal_shear)

            return mcal_dict2tab(res_dict, obs_dict, obj_info)

        except Exception as e:
            logprint(f'object {iobj}: Exception: {e}')
            logprint(f'object {iobj} failed, skipping...')

            return Table()

    def go(self, start, end, ncores=1):
        '''
        Run the metacal measurement from start to end.
        NOTE: cannot be parallelized
        '''

        if end < start:
            raise ValueError('end must be greater than start!')
        N = end - start

        self.logprint(f'Starting metacal fitting for {N} objects...')

        if ncores == 1:
            mcal_tabs = []
            k = 1
            for iobj in range(start, end):
                args, kwargs = self._get_fit_args(iobj)
                mcal_tabs.append(
                    MetacalRunner._fit_one(*args, **kwargs)
                    )
                k += 1

            self.logprint('Stacking mcal results...')
            self.mcal_table = vstack(mcal_tabs)

        else:
            # multiprocessing
            self.logprint(f'Running on {ncores} cores')
            with Pool(ncores) as pool:
                # TODO: I want to use self._get_fit_args() here, but
                # a little complicated...
                self.mcal_table = vstack(pool.starmap(
                    self._fit_one,
                    [(i,
                      self.boot,
                      self.get_obslist(i),
                      self.get_obj_info(i),
                      self.shear_step,
                      self.logprint
                      # TODO: make this work!
                      # *args, **kwargs = self._get_fit_args(i)
                      ) for i in range(start, end)
                     ])
                )

        Nfailed = N - len(self.mcal_table)
        self.logprint(f'{Nfailed} objects failed metacalibration fitting ' +\
                      'and are excluded from output catalog')
        self.logprint('Done!')

        return

    def get_obslist(self, iobj, weight_type=None):

        if weight_type is None:
            weight_type = self._default_weight_type

        obslist = self.meds.get_obslist(iobj, weight_type)

        # TODO: Implement actual deblending in the future!
        # obslist = self.deblend_neighbors(obslist)

        # We don't want to fit to the coadd, as its PSF is not
        # well defined
        if self.has_coadd is True:
            # NOTE: doesn't produce the right type...
            # obslist = obslist[1:]
            se_obslist = ngmix.ObsList(meta=deepcopy(obslist._meta))
            for obs in obslist[1:]:
                se_obslist.append(obs)
            obslist = se_obslist

        return obslist

    def deblend_neighbors(self, obslist, deblend_type='seg'):
        '''
        Given an obslist, deblend and/or model neighbors using
        desired method

        obslist: ngmix.Observation, ngmix.ObsList
            The ngmix observation or obs list that we are
            to deblend neighbors of
        '''

        deblend_types = self._deblend_types
        if deblend_type not in deblend_types:
            raise ValueError('deblend_type must be one of {deblend_types}!')

        if isinstance(obslist, ngmix.Observation):
            obslist = self._deblend_neighbors(obslist, deblend_type)
        elif isinstance(obslist, ngmix.ObsList):
            for i, obs in enumerate(obslist):
                obslist[i] = self._deblend_neighbors(obs, deblend_type)
        else:
            raise TypeError('obslist must be either a ngmix Observation ' +\
                            'or ObsList!')

        return obslist

    def _deblend_neighbors(self, obs, deblend_type):
        '''
        see deblend_neighbors()
        '''
        pass

    def get_jacobians(self, iobj):
        Njac = len(self.meds.get_jacobian_list(iobj))

        jacobians = [self.meds.get_ngmix_jacobian(iobj, icutout)
                     for icutout in range(Njac)]

        return jacobians

    def get_obj_info(self, iobj):
        '''
        Setup object property dictionary used to compile fit params later on
        '''

        obj = self.meds[iobj]

        # Mcal object properties
        obj_info = {}

        obj_info['meds_indx'] = iobj
        obj_info['id'] = obj['id']
        obj_info['ncutout'] = obj['ncutout']
        obj_info['ra'] = obj['ra']
        obj_info['dec'] = obj['dec']
        obj_info['XWIN_IMAGE'] = obj['XWIN_IMAGE']
        obj_info['YWIN_IMAGE'] = obj['YWIN_IMAGE']

        return obj_info

    def setup_prior(self, prior=None):

        if prior is None:
            self._setup_default_prior()
        else:
            self.prior = prior

        return

    def _setup_default_prior(self):

        pixel_scale = self.pixel_scale

        rng = self.rng

        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014

        g_sigma = 0.3
        g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which would be zero)
        # and the sigma of the gaussians

        # units same as jacobian, probably arcsec
        row, col = 0.0, 0.0
        row_sigma, col_sigma = pixel_scale, pixel_scale # use pixel_scale as a guess
        cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf).
        # NOTE: T units are arcsec^2 but can be slightly negative, especially for
        # stars if the PSF is mis-estimated

        Tminval = -1.0 # arcsec squared
        Tmaxval = 1000
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

        # similar for flux.  Make sure the bounds make sense for
        # your images

        Fminval = -1.e1
        Fmaxval = 1.e9
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

        # now make a joint prior.  This one takes priors
        # for each parameter separately
        self.prior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            F_prior,
            )

        return

    def setup_guessers(self, guesser=None, psf_guesser=None):
        '''
        Sets up guesser given a pixel scale. Prior must be setup first

        guesser: ngmix.Guesser
            A ngmix Guesser object for the source fit
        psf_guesser: ngmix.Guesser
            A ngmix Guesser object for the psf fit
        '''

        if guesser is None:
            self._setup_default_guesser()
        else:
            self.guesser = guesser

        if psf_guesser is None:
            self._setup_default_psf_guesser()
        else:
            self.psf_guesser = psf_guesser

        return

    def _setup_default_guesser(self):
        '''
        If no guesser for the source is provided, create one
        based on the PSF size (T), flux, and provided prior
        '''

        if self.prior is None:
            raise ValueError('prior must be setup before using the ' +\
                             'default guesser!')

        Tguess = 4*self.pixel_scale**2

        # make parameter guesses based on a psf flux and a rough T
        self.guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
            rng=self.rng,
            T=Tguess,
            prior=self.prior,
        )

        self.logprint(f'WARING: No guesser passed, using default: {self.guesser}')

        return

    def _setup_default_psf_guesser(self):
        '''
        If no guesser for the psf is provided, create one

        NOTE: need to check what is best to put here!
        '''

        # use simplest PSF guesser
        self.psf_guesser = ngmix.guessers.SimplePSFGuesser(self.rng)

        self.logprint(f'WARING: No psf guesser passed, ' +\
                      f'using default: {self.psf_guesser}')

        return

    def setup_lm_pars(self, lm_pars=None):
        if lm_pars is None:
            self._setup_default_lm_pars()
        else:
            self.lm_pars = lm_pars

        return

    def _setup_default_lm_pars(self):
        # Fairly standard par values for lm fit
        self.lm_pars = {
            'maxfev': 2000,
            'ftol': 1.0e-05,
            'xtol': 1.0e-05,
            }

        return

    def write_output(self, outfile, overwrite=False):
        '''
        Write mcal_table to outfile

        outfile: str
            The filename of the output mcal table
        '''

        if self.mcal_table is None:
            raise ValueError('mcal_table is still None! Try using go()')

        self.mcal_table.write(outfile, overwrite=overwrite)

        return

def add_mcal_cols(res_dict, obs_dict, mcal_shear):
    '''
    There are additional value-added cols that modern ngmix no
    longer returns automatically

    res_dict: dict
        The main result dictionary returned by the ngmix metacal
        bootstrapper.go() func
    obs_dict: dict
        The ngmix observation dictionary returned by the ngmix metacal
        boostrapper.go() func, containing meta info like the PSF fit
    mcal_shear: float
        The applied shear in the finite difference calculation
    '''

    # grab cols related to the PSF fit, such as Tpsf
    add_psf_cols(res_dict, obs_dict)

    # some quantities like the S2N are most robust for
    # the round version of the profile
    add_round_cols(res_dict, obs_dict)

    # R_gamma only for now - selections later
    add_mcal_responsivities(res_dict, mcal_shear)

    return

def add_psf_cols(res_dict, obs_dict):
    '''
    Add PSF fit quantities to the metacal results dictionary

    res_dict: dict
        The main result dictionary returned by the ngmix metacal
        bootstrapper.go() func
    obs_dict: dict
        The ngmix observation dictionary returned by the ngmix metacal
        boostrapper.go() func, containing meta info like the PSF fit
    '''

    # For now, we'll just add a mean Tpsf
    col = 'Tpsf'
    for shear_type in ['noshear', '1p', '1m', '2p', '2m']:
        obs = obs_dict[shear_type]

        if isinstance(obs, ngmix.Observation):
            Tpsf = obs.psf.meta['result']['T']

        elif isinstance(obs, ngmix.ObsList):
            # simple average:
            # Tpsf = np.mean([
            #     ob.psf.meta['result']['T']
            #     for ob in obs
            #     ])

            # weighted average:
            wsum = 0.0
            Tpsf_sum = 0.0
            for ob in obs:
                T = ob.psf.meta['result']['T']

                twsum = ob.weight.sum()
                wsum += twsum
                Tpsf_sum += T*twsum

            Tpsf = Tpsf_sum/wsum

        elif isinstance(obs, ngmix.MultiBandObsList):
            raise NotImplementedError('Multiband observations ' +\
                                      'not yet supported!')

        res_dict[shear_type][col] = Tpsf

    return

def add_round_cols(res_dict, obs_dict):
    '''
    Add "roundified" fit quantities to the metacal results dictionary

    mcal_dict: dict
        The main result dictionary returned by the ngmix metacal
        bootstrapper.go() func
    obs_dict: dict
        The ngmix observation dictionary returned by the ngmix metacal
        boostrapper.go() func, containing meta info like the PSF fit
    '''

    # For now, we'll just add T_r and s2n_r
    cols = {
        's2n_r': None,
        'T_r': None
    }

    for shear_type in ['noshear', '1p', '1m', '2p', '2m']:
        result = res_dict[shear_type]
        obs = obs_dict[shear_type]

        # get corresponding "roundified" version of model fit
        round_gm = result.get_gmix().make_round()
        cols['T_r'] = round_gm.get_T()

        if isinstance(obs, ngmix.Observation):
            cols['s2n_r'] = round_gm.get_model_s2n(obs)
        elif isinstance(obs, ngmix.ObsList):
            # simple average:
            # cols['s2n_r'] = np.mean([
            #     round_gm.get_model_s2n(ob)
            #     for ob in obs
            #     ])

            # weighted average:
            wsum = 0.0
            s2n_r_sum = 0.0
            for ob in obs:
                s2n = round_gm.get_model_s2n(ob)

                twsum = ob.weight.sum()
                wsum += twsum
                s2n_r_sum += s2n*twsum

            cols['s2n_r'] = s2n_r_sum/wsum

        elif isinstance(obs, ngmix.MultiBandObsList):
            raise NotImplementedError('Multiband observations ' +\
                                      'not yet supported!')

        for col, val in cols.items():
            res_dict[shear_type][col] = val

    return

def _compute_obs_s2n_r(result, obs):
    '''
    result: ngmix.fitting.Result
    obs: ngmix.Observation
    '''

    round_gm = result.get_gmix().make_round()
    s2n_r = round_gm.get_model_s2n(obs)

    return s2n_r

def add_mcal_responsivities(res_dict, mcal_shear):
    '''
    Compute and add the mcal responsivity values to the output
    result dict from get_metacal_result()
    NOTE: These are only for the selection-independent component!
    '''

    # Define full responsivity matrix, take inner product with shear moments
    r11 = (res_dict['1p']['g'][0] - res_dict['1m']['g'][0]) / (2*mcal_shear)
    r12 = (res_dict['2p']['g'][0] - res_dict['2m']['g'][0]) / (2*mcal_shear)
    r21 = (res_dict['1p']['g'][1] - res_dict['1m']['g'][1]) / (2*mcal_shear)
    r22 = (res_dict['2p']['g'][1] - res_dict['2m']['g'][1]) / (2*mcal_shear)

    R = [ [r11, r12], [r21, r22] ]
    Rinv = np.linalg.inv(R)
    gMC = np.dot(Rinv,
                 res_dict['noshear']['g']
                 )

    MC = {
        'r11':r11, 'r12':r12,
        'r21':r21, 'r22':r22,
        'g1_MC':gMC[0], 'g2_MC':gMC[1]
    }

    res_dict['MC'] = MC

    return

def mcal_dict2tab(mcal_dict, obs_dict, obj_info):
    '''
    mcal_dict: dict
        The main result dictionary returned by the ngmix metacal
        bootstrapper.go() func
    obs_dict: dict
        The ngmix observation dictionary returned by the ngmix metacal
        boostrapper.go() func, containing meta info like the PSF fit
    obj_info: np.recarray, Table
        An array or astropy table with MEDS identification info like
        id, ra, dec not returned by the bootstrapper
    '''

    # Annoying, but have to do this to make Table from scalars
    for key, val in obj_info.items():
        obj_info[key] = np.array([val])

    tab_names = ['noshear', '1p', '1m', '2p', '2m','MC']
    for name in tab_names:
        tab = mcal_dict[name]

        for key, val in tab.items():
            tab[key] = np.array([val])

        mcal_dict[name] = tab

    id_tab = Table(data=obj_info)

    tab_noshear = Table(mcal_dict['noshear'])
    tab_1p = Table(mcal_dict['1p'])
    tab_1m = Table(mcal_dict['1m'])
    tab_2p = Table(mcal_dict['2p'])
    tab_2m = Table(mcal_dict['2m'])
    tab_MC = Table(mcal_dict['MC'])

    join_tab = hstack([id_tab, hstack([tab_noshear,
                                       tab_1p,
                                       tab_1m,
                                       tab_2p,
                                       tab_2m,
                                       tab_MC
                                       ],
                                      table_names=tab_names)
                       ]
                      )

    return join_tab

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

def main(args):
    start = args.start
    end = args.end
    ncores = args.ncores
    vb = args.vb

    test_dir = '/Users/sweveret/repos/superbit-metacal/runs/real-test/'
    meds_dir = 'redo/real-base/r0/'
    meds_file = 'real-base_meds.fits'
    shear = 0.01
    seed = 723961

    base_meds_file = os.path.join(test_dir, meds_dir, meds_file)

    if vb is True:
        print('Setting up MetacalRunner...')
    mcal_runner = MetacalRunner(base_meds_file, vb=vb)

    if vb is True:
        print(f'Using seed={seed}')
    mcal_runner.set_seed(seed)

    if vb is True:
        print('Running bootstrapper for a gauss/coellip model...')
    gal_fitter = 'fitter'
    gal_kwargs = {'model': 'gauss'}
    psf_fitter = 'coellip'
    psf_kwargs = {'ngauss': 4}
    mcal_runner.setup_bootstrapper(
        gal_fitter, psf_fitter, shear,
        gal_kwargs=gal_kwargs, psf_kwargs=psf_kwargs,
        ntry=3
        )
    mcal_runner.go(start, end, ncores=ncores)

    if vb is True:
        print('Running bootstrapper for a ngmix.GaussMom/GaussMom model...')
    weight_fwhm = 1.2
    psf_kwargs = {'fwhm': weight_fwhm}
    fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
    psf_fitter = build_fitter('psf', 'gaussmom', kwargs=psf_kwargs)
    mcal_runner.setup_bootstrapper(fitter, psf_fitter, shear,
                                   psf_kwargs=psf_kwargs, ntry=3)
    mcal_runner.go(start, end, ncores=ncores)

    if vb is True:
        print('Running bootstrapper for a Exponential/Coellip model...')
    ngauss = 5
    fitter = 'exp'
    psf_fitter = 'coellip'
    mcal_runner.setup_bootstrapper(fitter, psf_fitter, shear,
                                   ntry=3)
    mcal_runner.go(start, end, ncores=ncores)

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nTests have completed without errors')
    else:
        print(f'\nTests failed with rc={rc}')

