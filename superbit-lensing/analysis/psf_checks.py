import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.table import Table
import ngmix.moments as moments
from argparse import ArgumentParser

from superbit-lensing import utils

def parse_args():

    parser = ArgumentParser

    parser.add_argument('run_name', type=str,
                        help='The name of the run')
    parser.add_argument('-ref_fwhm', type=float, default=0.24,
                        help='The FWHM in arcsec of the simulated PSF')

    return parser.parse_args()

def main(args):
    run_name = args.run_name
    ref_fwhm = args.ref_fwhm

    rundir = os.path.join(utils.get_module_dir(), 'runs')
    basedir = os.path.join(rundir, run_name)

    mass = '7.8e14'
    z = 0.25
    cl = f'cl_m{mass}_z{z}'

    real = 0

    data_dir = os.path.join(basedir, cl, f'r{str(real)}')

    mcal_file = os.path.join(data_dir, f'{run_name}_mcal.fits')
    mcal = Table.read(mcal_file)

    plt.rcParams.update({'figure.facecolor':'w'})

    # Plot the mcal-provided Tpsf's
    plt.subplot(121)
    plt.hist(mcal['Tpsf_noshear'], ec='k', bins=20)
    plt.xlabel('Tpsf_noshear')
    plt.title('Fitted PSF Tpsf')

    # Convert Tpsf to a fwhm & plot
    plt.subplot(122)
    fwhm = moments.T_to_fwhm(mcal['Tpsf_noshear'])
    plt.hist(fwhm, ec='k', bins=20)
    plt.xlabel('ngmix.moments.T_to_fwhm(Tpsf_noshear)')
    plt.axvline(ref_fwhm, ls='--', c='k', lw=2, label='True PSF FWHM')
    plt.legend()
    plt.title('Fitted PSF fwhm')

    plt.gcf().set_size_inches(12,5)

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
