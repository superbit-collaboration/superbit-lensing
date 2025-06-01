import os,sys
from pathlib import Path
from glob import glob
import esutil as eu
import meds
from argparse import ArgumentParser
import superbit_lensing.utils as utils
from superbit_lensing.medsmaker.superbit import medsmaker_real as medsmaker
import yaml

# Added for Hot/Cold SExtractor
from superbit_lensing.medsmaker.superbit.hotcold_sextractor import HotColdSExtractor

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('target_name', action='store', type=str, default=None,
                        help='Name of target to make MEDS for')
    parser.add_argument('bands', type=str,
                        help='List of bands for MEDS (separated by commas)')
    parser.add_argument('data_dir', type=str,
                        help='Path to data all the way up to and including oba_temp ')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory for MEDS file')
    parser.add_argument('-psf_mode', action='store', default='piff',
                        choices=['piff', 'psfex', 'true'],
                        help='model exposure PSF using either piff or psfex')
    parser.add_argument('-psf_seed', type=int, default=None,
                        help='Seed for chosen PSF estimation mode')
    parser.add_argument('-star_config_dir', type=str, default=None,
                        help='Path to the directory containing the YAML configuration files for star processing')
    parser.add_argument('--select_truth_stars', action='store_true', default=False,
                        help='Set to match against truth catalog for PSF model fits')
    parser.add_argument('--meds_coadd', action='store_true', default=False,
                        help='Set to keep coadd cutout in MEDS file')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Verbosity')

    return parser.parse_args()

def main(args):
    target_name = args.target_name
    data_dir = args.data_dir
    outdir = args.outdir
    psf_mode = args.psf_mode
    psf_seed = args.psf_seed
    use_coadd = args.meds_coadd
    overwrite = args.overwrite
    bands = args.bands
    star_config_dir = args.star_config_dir
    select_truth_stars = args.select_truth_stars
    vb = args.vb

    if star_config_dir is None:
        star_config_dir = str(Path(utils.MODULE_DIR, 'medsmaker/configs'))

    # NOTE: Need to parse "band1,band2,etc." due to argparse struggling w/ lists
    bands = bands.split(',')

    logfile = 'medsmaker.log'
    logdir = outdir
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb=vb)

    for band in bands:
        logprint(f'Processing band {band}...')
        if outdir is None:
            band_outdir = Path(data_dir) / target_name / band / 'meds'
        else:
            band_outdir = outdir

        # only makes it if it doesn't already exist
        utils.make_dir(str(band_outdir))

        # Load the specific YAML file for the current band
        yaml_file = f'{target_name}_{band}_starparams.yaml'
        yaml_path = os.path.join(star_config_dir, yaml_file)

        # Check if the YAML file exists, if not use defaults
        if os.path.exists(yaml_path):
            star_params = read_yaml_file(yaml_path)
        else:
            logprint(
                f'Warning: Configuration file {yaml_file} not ' +\
                'found in {star_config_dir}. Setting star_params to None'
                )
            star_params = None

        # Load in the science frames
        search = str(Path(data_dir) / target_name / band / 'cal' / '*cal.fits')
        science = glob(search)
        logprint(f'Science frames: {science}')
        outfile = f'{target_name}_{band}_meds.fits'
        outfile = os.path.join(band_outdir, outfile)

        # Create an instance of BITMeasurement
        logprint('Setting up BITMeasurement configuration...')
        bm = medsmaker.BITMeasurement(
            science,
            data_dir,
            target_name,
            band,
            band_outdir,
            log=log,
            vb=vb
            )

        # Set up astromatic (sex & psfex & swarp) configs
        astro_config_dir = str(Path(utils.MODULE_DIR,
                               'medsmaker/superbit/astro_config/')
                               )
        """
        CHANGED SECTION STARTS HERE
        """

        # TODO: Make this less hard-coded
        # Create an instance of HotColdSExtractor
        logprint('Setting up HotColdSExtractor configuration...')

        hc_config = '/projects/mccleary_group/vassilakis.g/superbit-lensing/superbit_lensing/medsmaker/superbit/astro_config/hc_config.yaml'

        hcs = HotColdSExtractor(
            science,
            hc_config,
            band,
            target_name,
            data_dir,
            astro_config_dir
            )

        # Run HotColdSExtractor on Coadd
        logprint('Making coadd detection catalog...')
        coadd_catalog = hcs.make_coadd_catalog()

        # Run HotColdSExtractor on Single Exposures
        logprint('Making single-exposure catalogs...')
        single_exposure_catalogs = hcs.make_exposure_catalogs()

        # Build a PSF model for each image.
        logprint('Making PSF models...')

        bm.make_psf_models(
            select_truth_stars=select_truth_stars,
            use_coadd=use_coadd,
            psf_mode=psf_mode,
            psf_seed=psf_seed,
            star_params=star_params,
        )

        logprint('Making MEDS...')

        # TODO: Change this to my own version of detection!
        # Get detection source file & catalog
        logprint('Getting detection source files & catalogs...')
        bm.get_detection_files()

        # Make the image_info struct.
        image_info = bm.make_image_info_struct(use_coadd=use_coadd)

        # Make the object_info struct.
        obj_info = bm.make_object_info_struct()

        # Make the MEDS config file.
        meds_config = bm.make_meds_config(use_coadd, psf_mode)

        # Create metadata for MEDS
        # TODO: update this when we actually do photometric calibration!
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
