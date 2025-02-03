import numpy as np
import os
import time
from argparse import ArgumentParser

# TODO: annoying thing we have to do while testing:
import sys
BASE = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
            )
        )
    )
sys.path.insert(0, BASE)
from mcal_runner import MetacalRunner, build_fitter
import superbit_lensing.utils as utils

import ipdb

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('medsfile', type=str,
                        help='MEDS file to process')
    parser.add_argument('outfile', type=str,
                        help='Output filename')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('-seed', type=int, default=None,
                        help='Seed for metacal runner')
    parser.add_argument('-start', type=int, default=None,
                        help='Starting index for MEDS processing')
    parser.add_argument('-end', type=int, default=None,
                        help='Ending index for MEDS processing')
    parser.add_argument('-shear', type=float, default=0.1,
                        help='The value of the applied shear')
    parser.add_argument('-ntry', type=int, default=3,
                        help='Number of tries before accepting a fit failure')
    parser.add_argument('-ncores', type=int, default=1,
                        help='Number of cores to use')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output mcal file')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    # TODO: Right now this won't do anything other than make the
    # script backwards compatible w/ older versions. Can either
    # implement the diagnostics plotting here or get rid of for
    # future runs
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Set to make diagnstic plots')

    # TODO: I want to do this w/ a ngmix/mcal config in the future,
    # but will ignore for the current sims
    # Can include filenames, seeds, run options, etc.
    # parser.add_argument('meds_config', type=str,
    #                     help='Metacalibration config file')

    return parser.parse_args()

def main(args):
    medsfile = args.medsfile
    outfile = args.outfile
    outdir = args.outdir
    seed = args.seed
    index_start  = args.start
    index_end = args.end
    shear = args.shear
    ntry = args.ntry
    ncores = args.ncores
    make_plots = args.plot
    overwrite = args.overwrite
    vb = args.vb

    #-----------------------------------------------------------------
    # Initial setup

    if outdir is not None:
        outdir = os.getcwd()

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    logdir = outdir
    logfile = 'mcal_fitting.log'
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    #-----------------------------------------------------------------
    # Create & setup metacal runner

    mcal_runner = MetacalRunner(medsfile, logprint=logprint)

    Ncat = mcal_runner.Nobjs

    if index_start == None:
        index_start = 0
    if index_end == None:
        index_end = Ncat

    if index_end <= index_start:
        raise ValueError('index_end must be greater than index_start!')

    if make_plots is True:
        logprint('Warning: make_plots currently does nothing for this ' +\
                 'version of metacal running. Only here to not break ' +\
                 'old versions!')


    if index_end > Ncat:
        logprint(f'Warning: index_end={index_end} larger than ' +\
                 f'catalog size of {Ncat}; running over full catalog')
        index_end = Ncat

    if seed is None:
        seed = np.random.randint(0, 2**32-1)
    logprint(f'Using metacal seed {seed}')
    mcal_runner.set_seed(seed)

    # TODO: here we can utilize a future mcal config to do lots of
    # generic fitter, guesser, etc. setup using the builder functions.
    # For now, we just will specify our best guess

    # standard Gaussian fit to profile shape
    gal_kwargs = {
        'model': 'gauss'
    }
    gal_fitter = build_fitter('gal', 'fitter', gal_kwargs)

    # NOTE: just a single Gauss for testing
    psf_kwargs = {
        'model': 'gauss'
    }
    psf_fitter = build_fitter('psf', 'fitter', psf_kwargs)

    # NOTE:Could do something else. For example:
    # 4 coelliptical gaussians for PSF fit
    # psf_kwargs = {
    #     'ngauss': 4
    # }
    # psf_fitter = build_fitter('psf', 'coellip', psf_kwargs)

    # NOTE: can set specific guessers if desired. For now, we default
    # to those recommended in the ngmix examples page
    # gal_guesser = pass
    # psf_guesser = pass

    mcal_runner.setup_bootstrapper(
        gal_fitter, psf_fitter, shear,
        gal_kwargs=gal_kwargs, psf_kwargs=psf_kwargs,
        ntry=ntry
        )

    #-----------------------------------------------------------------
    # Run metacal

    s = '' if ncores==1 else 's'
    logprint(f'Starting metacal fitting with {ncores} core{s}')

    start = time.time()

    mcal_runner.go(index_start, index_end, ncores=ncores)

    end = time.time()

    T = end - start
    logprint(f'Total fitting and stacking time: {T} seconds')

    N = index_end - index_start
    logprint(f'{T/N:.4} seconds per object (wall time)')
    logprint(f'{T/N*ncores:.4} seconds per object (CPU time)')
    #-----------------------------------------------------------------
    # Save output metacal catalog

    outfile = os.path.join(outdir, outfile)
    logprint(f'Writing results to {outfile}')
    mcal_runner.write_output(outfile, overwrite=overwrite)

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nMetacal fitting completed without error')
    else:
        print(f'\nMetacal fitting failed with rc={rc}')
