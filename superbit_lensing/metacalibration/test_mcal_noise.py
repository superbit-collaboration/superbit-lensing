import ngmix
import os
from astropy.table import Table, vstack
from argparse import ArgumentParser
from multiprocessing import Pool

from mcal_runner import MetacalRunner

import ipdb

def parse_args():
    parser = ArgumentParser()

    # parser.add_argument('-start', type=int, default=0,
    #                     help='Starting MEDS index for mcal fitting')
    # parser.add_argument('-end', type=int, default=1,
                        # help='Ending MEDS index for mcal fitting')
    parser.add_argument('meds_indx', type=int,
                        help='MEDS index for mcal fitting')
    parser.add_argument('outfile', type=str,
                        help='Output filename for test table')
    parser.add_argument('-nreal', type=int, default=100,
                        help='The number of noise realizations per obj')
    parser.add_argument('-ncores', type=int, default=1,
                        help='The number of cores to use for the test')
    parser.add_argument('-outdir', type=str, default='',
                        help='Output directory location')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

def main(args):
    # start = args.start
    # end = args.end
    indx = args.meds_indx
    nreal = args.nreal
    ncores = args.ncores
    outfile = args.outfile
    outdir = args.outdir
    vb = args.vb

    # NOTE: just 1 for now
    # assert start < end
    # we follow python conventions and go from 0-(N-1)
    # Nobjs = (end-start) - 1

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
    # gal_fitter = ngmix.fitting.Fitter('gauss')
    # psf_fitter = ngmix.fitting.CoellipFitter(ngauss=4)
    gal_fitter = 'fitter'
    gal_kwargs = {'model': 'gauss'}
    psf_fitter = 'coellip'
    psf_kwargs = {'ngauss': 4}
    mcal_runner.setup_bootstrapper(
        gal_fitter, psf_fitter, shear,
        gal_kwargs=gal_kwargs, psf_kwargs=psf_kwargs, ntry=3
        )

    mcal_res = []

    if ncores == 1:
        mcal_res = []
        for n in range(nreal):
            print(f'Starting noise realization {n+1} of {nreal}')
            mcal_res.append(mcal_runner.go(indx, indx+1))

        final = vstack(mcal_res)

    else:
        # mp version:
        with Pool(ncores) as pool:
            # final = vstack(pool.starmap(
            #     mcal_runner.go, [(indx, indx+1) for n in range(nreal)]
            # ))

            final = vstack(pool.starmap(mcal_runner._fit_one,
                                        [(n,
                                          mcal_runner.boot,
                                          mcal_runner.get_obslist(indx),
                                          mcal_runner.get_obj_info(indx),
                                          mcal_runner.shear_step,
                                          vb
                                          ) for n in range(nreal)
                                         ]
                                        )
                           )

    outfile = os.path.join(outdir, outfile)
    final.write(outfile)

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc == 0:
        print('\nTests have completed without errors')
    else:
        print(f'\nTests failed with rc={rc}')
