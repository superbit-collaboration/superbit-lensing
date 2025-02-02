import numpy as np
import os
import time
import ngmix
from astropy.table import Table, vstack
from ngmix.medsreaders import NGMixMEDS
from multiprocessing import Pool
from argparse import ArgumentParser
import superbit_lensing.utils as utils
import pudb

parser = ArgumentParser()

parser.add_argument('medsfile', type=str,
                    help='MEDS filename to run ngmix on')
parser.add_argument('outfile', type=str,
                    help='Name of output file')
parser.add_argument('config', type=str,
                    help='ngmix config filename')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory')
parser.add_argument('-start', type=int, default=None,
                    help='Starting index for MEDS processing')
parser.add_argument('-end', type=int, default=None,
                    help='Ending index for MEDS processing')
parser.add_argument('-n', type=int, default=1,
                    help='Number of CPU cores to use')
parser.add_argument('--clobber', action='store_true', default=False,
                    help='Set to overwrite existing outfile')
parser.add_argument('--test', action='store_true', default=False,
                    help='Use to do a test run w/ provided config')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Verbose')

class SuperBITngmixRunner(object):
    _req_fields = ['gal', 'psf', 'priors', 'fit_pars',
                   'run_name', 'pixel_scale', 'nbands',
                   'seed']
    _opt_fields = []

    _gal_req_fields = ['model']
    _psf_req_fields = ['model']
    _priors_req_fields = ['T_range',
                          'F_range',
                          'g_sigma']

    def __init__(self, medsfile, config, logprint):

        assert isinstance(medsfile, str)
        self.medsfile = medsfile

        if isinstance(config, str):
            config = utils.read_yaml(config)
        else:
            assert isinstance(config, dict)

        utils.check_fields(config,
                           self._req_fields,
                           self._opt_fields,
                           'ngmix')

        field_zip = zip(['gal', 'psf', 'priors'],
                        [self._gal_req_fields,
                         self._psf_req_fields,
                         self._priors_req_fields]
                        )
        for field, req in field_zip:
            field_config = config[field]
            utils.check_req_fields(field_config,
                                   req,
                                   f'ngmix `{field}`'
                                   )

        self.config = config

        assert isinstance(logprint, utils.LogPrint)
        self.logprint = logprint

        seed = config['seed']
        rng = np.random.RandomState(seed)
        self.set_rng(rng)

        self._setup_priors()
        self._setup_meds()

        return

    def set_rng(self, rng):
        self.rng = rng

    def _setup_priors(self):
        self.logprint('Setting up ngmix priors')

        gal_model = self.config['gal']['model']
        imscale = self.config['pixel_scale']
        rng = self.rng

        # do common prior setup
        g_sigma = self.config['priors']['g_sigma']
        g_prior = ngmix.priors.GPriorBA(sigma=g_sigma, rng=rng)

        cen_prior = ngmix.priors.CenPrior(
            cen1=0, cen2=0, sigma1=imscale, sigma2=imscale, rng=rng
            )

        T_range = self.config['priors']['T_range']
        T_prior = ngmix.priors.FlatPrior(minval=T_range[0],
                                         maxval=T_range[1],
                                         rng=rng)

        F_range = self.config['priors']['F_range']
        F_prior = ngmix.priors.FlatPrior(minval=F_range[0],
                                         maxval=F_range[1],
                                         rng=rng)

        if self.config['nbands'] > 1:
            F_prior = [F_prior]*nbands

        cols = ['cen', 'g', 'T', 'F']
        prior_list = [cen_prior, g_prior, T_prior, F_prior]

        if gal_model in ['cm', 'bdf']:
            # setup fracdev
            fd_mean = self.config['priors']['fracdev_mean']
            fd_sig = self.config['priors']['fracdev_sigma']
            fracdev_prior = ngmix.priors.Normal(mean=fd_mean,
                                                sigma=fd_sig,
                                                rng=rng)

            cols.append('fracdev')
            prior_list.append(fracdev_prior)

        else:
            raise ValueError(f'{gal_model} prior not yet implemented!')

        prior_dict = {}
        for col, prior in zip(cols, prior_list):
            prior_dict[col] = prior

        priors = build_ngmix_priors(gal_model, prior_dict)

        self.priors = priors

        return

    def _setup_meds(self):

        self.logprint('Loading MEDS file')

        self.meds = NGMixMEDS(self.medsfile)
        self.nobjs = len(self.meds._cat)

        return

    def _setup_obs_list(self, i):
        '''i: MEDS index'''
        return self.meds.get_obslist(i)

    def fit(self, ncores, index_start, index_end):

        self.logprint('Starting ngmix fit')
        self.logprint(f'Using {ncores} cores')

        start = time.time()

        ngmix_range = range(index_start, index_end)

        # for single core processing:
        if ncores == 1:
            ngmix_fits = []
            for i in ngmix_range:
                ngmix_fits.append(
                    _fit_one(i,
                            self._setup_obs_list(i),
                            self.priors,
                            self.config,
                            self.logprint,
                            ))

            res = vstack(ngmix_fits)

        # for multiprocessing:
        else:
            with Pool(ncores) as pool:
                res = vstack(pool.starmap(_fit_one,
                                          [(i,
                                          self._setup_obs_list(i),
                                          self.priors,
                                          self.config,
                                          self.logprint)
                                          for i in ngmix_range]
                                          )
                    )

        self.logprint('Fitting has completed')

        end = time.time()

        T = end - start
        self.logprint(f'Total fitting and stacking time: {T} seconds')

        N = index_end - index_start
        self.logprint(f'{T/N} seconds per object (wall time)')
        self.logprint(f'{T/N*ncores} seconds per object (CPU time)')

        return res

def setup_bootstrapper(obs_list):
    boot = ngmix.Bootstrapper(obs_list)
    return boot

def _fit_one(i, obs_list, priors, config, logprint):
    '''
    i: MEDS index
    obs_list: ngmix observation list for index i

    return: an astropy Table containing fit data
    '''

    pars = config['fit_pars']

    boot = setup_bootstrapper(obs_list)

    try:

        logprint('Starting PSF fit')
        pixscale = config['pixel_scale']
        fwhm_guess = 2*pixscale # arcsec corresponding to 2 pixels
        psf_Tguess = fwhm_guess*pixscale**2
        boot.fit_psfs(
            config['psf']['model'],
            psf_Tguess
            )

        # Guesses taken from ngmix.test_guessers.py
        T_center = 0.001
        flux_center = [1.]*config['nbands']
        guesser = ngmix.guessers.BDFGuesser(
                    T_center, flux_center, priors,
                )
        model = config['gal']['model']
        logprint(f'Starting {model} fit')
        boot.fit_max(
            model,
            pars,
            prior=priors,
            guesser=guesser,
            ntry=3
            )

        res = boot.get_fitter().get_result()

        return utils.ngmix_dict2table(res)

    except Exception as e:
        logprint(e)
        logprint(f'object {i} failed, skipping...')

        return Table()

def build_ngmix_priors(model, prior_dict):
    '''
    model: ngmix galaxy model name
    prior_dict: dictionary of ngmix prior objects for each
                relevant parameter prior
    '''

    if model == 'bdf':
        priors = ngmix.joint_prior.PriorBDFSep(
            cen_prior=prior_dict['cen'],
            g_prior=prior_dict['g'],
            T_prior=prior_dict['T'],
            fracdev_prior=prior_dict['fracdev'],
            F_prior=prior_dict['F']
            )
    else:
        raise ValueError(f'{model} priors is not yet implemented!')

    return priors

def write_output_table(outfile, table, clobber=False):
    table.write(outfile, format='fits', overwrite=clobber)

    return

def _make_test_ngmix_config(run_name=None):
    if run_name is None:
        run_name = 'bfd_test'

    test_config = {
        'gal': {
            'model': 'bdf',
        },
        'psf': {
            'model': 'gauss'
        },
        'priors': {
            'T_range': [-1., 1.e3],
            'F_range': [-100., 1.e9],
            'g_sigma': 0.1,
            'fracdev_mean': 0.5,
            'fracdev_sigma': 0.1
        },
        'fit_pars': {
            'method': 'lm',
            'lm_pars': {
                'maxfev':2000,
                'xtol':5.0e-5,
                'ftol':5.0e-5
                }
        },
        'pixel_scale': 0.144, # arcsec / pixel
        'nbands': 1,
        'seed': 172396,
        'run_name': run_name
    }

    return test_config

def _return_test_medsfile():
    medsfile = os.path.join(utils.get_test_dir(),
                            'ngmix_fit',
                            'test_meds.fits')
    return medsfile

def _return_test_outfile():
    outfile = os.path.join(utils.get_test_dir(),
                           'ngmix_fit',
                           'test_ngmix.fits')

    return outfile

def main():
    args = parser.parse_args()
    medsfile = args.medsfile
    outfile = args.outfile
    config_file = args.config
    outdir = args.outdir
    index_start  = args.start
    index_end = args.end
    ncores = args.n
    clobber = args.clobber
    test = args.test
    vb = args.vb

    if test is True:
        config = _make_test_ngmix_config()
        medsfile = _return_test_medsfile()
        outfile = _return_test_outfile()

        logfile = 'ngmix-fit-test.log'
        outdir = os.path.join(utils.TEST_DIR,
                              'ngmix_fit')
        log = utils.setup_logger(logfile, logdir=outdir)
        logprint = utils.LogPrint(log, vb)

    else:
        config = utils.read_yaml(config_file)

        z = zip(['medsfile', 'outfile'], [medsfile, outfile])
        for name, val in z:
            if val is None:
                raise ValueError(f'Must pass a {name} if --test is not True!')

        try:
            run_name = config['run_name']
        except KeyError:
            raise ValueError('run_name is a required field in the ngmix config!')

        logfile = f'{run_name}-ngmix-fit.log'
        outdir = os.path.join(utils.TEST_DIR, 'ngmix_fit')
        log = utils.setup_logger(logfile, logdir=outdir)
        logprint = utils.LogPrint(log, vb)

    sb_runner = SuperBITngmixRunner(medsfile, config, logprint)

    nobjs = sb_runner.nobjs
    if index_start is None:
        index_start = 0
    if index_end is None:
        index_end = nobjs

    if index_end > nobjs:
        logprint(f'Warning: index_end={index_end} larger than ' +\
                 f'catalog size of {nobjs}; running over full catalog')

    res = sb_runner.fit(ncores=ncores, index_start=index_start, index_end=index_end)

    logprint(f'Writing results to {outfile}')

    write_output_table(outfile, res, clobber=clobber)

    print('Done!')

    return

if __name__ == '__main__':
    main()
