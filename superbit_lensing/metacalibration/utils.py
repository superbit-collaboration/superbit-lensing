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

def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

def get_coellip_ngauss(name):
    ngauss=int( name[7:] )
    return ngauss

import numpy as np
import ngmix

class MetacalFitter:
    """Simple class containing metacalibration fitting methods"""
    
    def __init__(self, seed, obslist):
        """
        Initialize the MetacalFitter
        
        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        obslist : list
            Observation list for MEDS object
        """
        self.seed = seed
        self.obslist = obslist
        self.rng = np.random.RandomState(seed)
        self.prior = self.get_priors()
        self.resdict = None
        self.obsdict = None
        self.galaxy_residuals = None
        self.psf_residuals = None
        self.avg_galaxy_residual = None
        self.avg_psf_residual = None
    
    def get_priors(self):

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

    def mp_fit_one(self, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
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
        jacobian = self.obslist[0]._jacobian
        Tguess = 4*jacobian.get_scale()**2
        ntry = 20
        lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
        psf_lm_pars={'maxfev': 4000, 'xtol':5.0e-5,'ftol':5.0e-5}

        fitter = ngmix.fitting.Fitter(model=gal_model, prior=self.prior, fit_pars=lm_pars)
        guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=self.rng, T=Tguess, prior=self.prior)

        # psf fitting
        if 'em' in psf_model:
            em_pars={'tol': 1.0e-6, 'maxiter': 50000}
            psf_ngauss = get_em_ngauss(psf_model)
            psf_fitter = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
            psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=self.rng, ngauss=psf_ngauss)
        elif 'coellip' in psf_model:
            psf_ngauss = get_coellip_ngauss(psf_model)
            psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
            psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=self.rng, ngauss=psf_ngauss)
        elif psf_model == 'gauss':
            psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=psf_lm_pars)
            psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=self.rng)
        else:
            raise ValueError('psf_model must be one of emn, coellipn, or gauss')

        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)

        runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

        #types = ['noshear', '1p', '1m', '2p', '2m']
        psf = mcal_pars['psf']
        mcal_shear = mcal_pars['mcal_shear']
        boot = ngmix.metacal.MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner,
            rng=self.rng,
            psf=psf,
            step = mcal_shear,
            #types=types,
        )

        resdict, obsdict = boot.go(self.obslist)
        
        # Store results as instance attributes
        self.resdict = resdict
        self.obsdict = obsdict

        return resdict, obsdict

    def get_response(self, resdict=None, verbose=False):
        """
        Get the shear response matrix.
        
        If resdict is not provided, uses self.resdict
        """
        if resdict is None:
            if self.resdict is None:
                raise ValueError("No resdict available. Run mp_fit_one first or provide resdict.")
            resdict = self.resdict
            
        # Extract all g1 and g2 values
        g1_1p = resdict['1p']['g'][0]
        g1_1m = resdict['1m']['g'][0]
        g1_noshear = resdict['noshear']['g'][0]

        g2_2p = resdict['2p']['g'][1]
        g2_2m = resdict['2m']['g'][1]
        g2_noshear = resdict['noshear']['g'][1]

        # Compute shear response matrix elements (finite difference approximation)
        R11 = (g1_1p - g1_1m) / 0.02  # response of g1 to g1 shear
        R22 = (g2_2p - g2_2m) / 0.02  # response of g2 to g2 shear

        # Often we also compute the full matrix
        R12 = (resdict['2p']['g'][0] - resdict['2m']['g'][0]) / 0.02  # response of g1 to g2 shear
        R21 = (resdict['1p']['g'][1] - resdict['1m']['g'][1]) / 0.02  # response of g2 to g1 shear

        # Pack into matrix if needed
        R = np.array([[R11, R12],
                    [R21, R22]])
        if verbose:
            print("Shear response matrix R:")
            print(R)
        return R
    
    def save_residuals(self):
        """
        Compute and save residuals for all exposures.
        
        Returns:
        --------
        dict containing:
            - galaxy_residuals: array of galaxy residuals for each exposure
            - psf_residuals: array of PSF residuals for each exposure
            - avg_galaxy_residual: average galaxy residual across all exposures
            - avg_psf_residual: average PSF residual across all exposures
        """
        if self.resdict is None or self.obsdict is None:
            raise ValueError("No fitting results available. Run mp_fit_one first.")
        
        # Initialize lists to store all residuals
        galaxy_residuals = []
        psf_residuals = []
        
        # Get number of exposures
        n_exposures = len(self.obslist)
        
        for i in range(n_exposures):
            # Galaxy residuals
            im_fit = self.resdict['noshear'].make_image(obsnum=i)
            obs_im = self.obslist[i].image
            galaxy_residual = obs_im - im_fit
            galaxy_residuals.append(galaxy_residual)
            
            # PSF residuals
            psf_fit = self.obsdict['noshear'][i].psf.meta['result'].make_image()
            psf_im = self.obslist[i].psf.image
            psf_residual = psf_im - psf_fit
            psf_residuals.append(psf_residual)
        
        # Convert to numpy arrays and compute averages
        galaxy_residuals = np.array(galaxy_residuals)
        psf_residuals = np.array(psf_residuals)
        
        avg_galaxy_residual = np.mean(galaxy_residuals, axis=0)
        avg_psf_residual = np.mean(psf_residuals, axis=0)
        
        # Store as instance attributes
        self.galaxy_residuals = galaxy_residuals
        self.psf_residuals = psf_residuals
        self.avg_galaxy_residual = avg_galaxy_residual
        self.avg_psf_residual = avg_psf_residual
        
        # Return dictionary with all residuals
        return {
            'galaxy_residuals': galaxy_residuals,
            'psf_residuals': psf_residuals,
            'avg_galaxy_residual': avg_galaxy_residual,
            'avg_psf_residual': avg_psf_residual
        }