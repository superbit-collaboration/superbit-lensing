import os,sys
from pathlib import Path
import glob
import esutil as eu
import meds
from argparse import ArgumentParser
import superbit-lensing.utils as utils
from superbit-lensing.medsmaker.superbit import medsmaker_mocks as medsmaker

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('mock_dir', type=str,
                        help='Directory containing mock data')
    parser.add_argument('outfile', type=str,
                        help='Name of output MEDS file')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory for MEDS file')
    parser.add_argument('-fname_base', action='store', type=str, default=None,
                        help='Basename of mock image files')
    parser.add_argument('-run_name', action='store', type=str, default=None,
                        help='Name of mock simulation run')
    parser.add_argument('-psf_mode', action='store', default='piff',
                        choices=['piff', 'psfex', 'true'],
                        help='model exposure PSF using either piff or psfex')
    parser.add_argument('-psf_seed', type=int, default=None,
                        help='Seed for chosen PSF estimation mode')
    parser.add_argument('--meds_coadd', action='store_true', default=False,
                        help='Set to keep coadd cutout in MEDS file')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite files')
    parser.add_argument('--source_select', action='store_true', default=False,
                        help='Set to select sources during MEDS creation')
    parser.add_argument('--select_truth_stars', action='store_true', default=False,
                        help='Set to match against truth catalog for PSF model fits')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Verbosity')

    return parser.parse_args()

def main(args):
    mock_dir = args.mock_dir
    outfile = args.outfile
    outdir = args.outdir
    run_name = args.run_name
    psf_mode = args.psf_mode
    psf_seed = args.psf_seed
    use_coadd = args.meds_coadd
    overwrite = args.overwrite
    source_selection = args.source_select
    select_truth_stars = args.select_truth_stars
    vb = args.vb

    if args.outdir is None:
        outdir = mock_dir

    logfile = 'medsmaker.log'
    logdir = outdir
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    if args.fname_base is None:
        fname_base = run_name
    else:
        fname_base = args.fname_base


    science = glob.glob(os.path.join(mock_dir, fname_base) +\
                        '*[!truth,mcal,.sub,mock_coadd].fits')
    logprint(f'Science frames: {science}')

    outfile = os.path.join(outdir, outfile)

    logprint('Setting up configuration...')
    bm = medsmaker.BITMeasurement(
        image_files=science, data_dir=mock_dir, run_name=run_name,
        log=log, vb=vb
        )

    bm.set_working_dir(path=outdir)
    mask_dir = os.path.join(mock_dir,'mask_files')
    weight_dir = os.path.join(mock_dir,'weight_files')
    bm.set_mask(
        mask_name='forecast_mask.fits', mask_dir=mask_dir
        )
    bm.set_weight(
        weight_name='forecast_weight.fits', weight_dir=weight_dir
        )

    # Combine images, make a catalog.
    logprint('Making coadd & its catalog...')
    bm.make_coadd_catalog(source_selection=source_selection)

    # Make single-exposure catalogs
    logprint('Making single-exposure catalogs...')
    im_cats = bm.make_exposure_catalogs()

    # Build a PSF model for each image.
    logprint('Making PSF models...')
    
    star_params = {'CLASS_STAR':0.95,
                        'MIN_MAG':23,
                        'MAX_MAG':15,
                        'MIN_SIZE':1.8,
                        'MAX_SIZE':4,
                        'MIN_SNR': 20
                        }

    bm.make_psf_models(
        select_truth_stars=select_truth_stars,
        im_cats=im_cats,
        use_coadd=use_coadd,
        psf_mode=psf_mode,
        psf_seed=psf_seed
        )

    logprint('Making MEDS...')

    # Make the image_info struct.
    image_info = bm.make_image_info_struct(use_coadd=use_coadd)

    # Make the object_info struct.
    obj_info = bm.make_object_info_struct()

    # Make the MEDS config file.
    meds_config = bm.make_meds_config(use_coadd, psf_mode)

    # Create metadata for MEDS
    magzp = 30.
    meta = bm.meds_metadata(magzp, use_coadd)
    # Finally, make and write the MEDS file.

    medsObj = meds.maker.MEDSMaker(
        obj_info, image_info, config=meds_config,
        psf_data=bm.psf_models, meta_data=meta
        )

    logprint(f'Writing to {outfile}')
    medsObj.write(outfile)

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc !=0:
        print(f'process_mocks failed w/ return code {rc}!')
