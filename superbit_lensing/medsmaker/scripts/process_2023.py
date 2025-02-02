import os,sys
from pathlib import Path
from glob import glob
import esutil as eu
import meds
from argparse import ArgumentParser
import superbit_lensing.utils as utils
from superbit_lensing.medsmaker.superbit import medsmaker_real as medsmaker
from superbit_lensing.medsmaker.superbit.hotcold_sextractor import HotColdSExtractor
import yaml
import pdb


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
                        help='Path to the directory containing the YAML ' + \
                             'configuration files for star processing')
    parser.add_argument('-detection_bandpass', type=str, default='b',
                        help='Shape measurement (detection) bandpass')
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
    detection_bandpass = args.detection_bandpass
    vb = args.vb

    if star_config_dir == None:
        star_config_dir = str(Path(utils.MODULE_DIR, 'medsmaker/configs'))

    # NOTE: Need to parse "band1,band2,etc." due to argparse struggling w/ lists
    bands = bands.split(',')

    for band in bands:
        if outdir == None:
            band_outdir = Path(data_dir) / target_name / band / 'out'
        else:
            band_outdir = outdir

        # only makes it if it doesn't already exist
        utils.make_dir(str(band_outdir))

        logfile = 'medsmaker.log'
        logdir = band_outdir
        log = utils.setup_logger(logfile, logdir=logdir)
        logprint = utils.LogPrint(log, vb=vb)

        logprint(f'Processing band {band}...\n')

        # Load the specific YAML file for the current band
        starparams_yaml_file = f'{target_name}_{band}_starparams.yaml'
        starparams_yaml_path = os.path.join(star_config_dir, starparams_yaml_file)

        # Check if the YAML file exists, if not use defaults
        if os.path.exists(starparams_yaml_path):
            star_config = read_yaml_file(starparams_yaml_path)
        else:
            logprint(
                f'Warning: Configuration file {starparams_yaml_file} not ' +
                f'found in {star_config_dir}. Setting "star_params" to None'
            )
            star_config = None

        # Load in the science frames
        endings = ["cal", "clean"]
        science = []

        for ending in endings:
            search_path = os.path.join(data_dir, target_name, band, 'cal', f'*{ending}.fits')
            science.extend(glob(search_path))
            
        #science = science[0:2]
        
        logprint(f'\nUsing science frames: {science}\n')

        # Define output MEDS name
        outfile = f'{target_name}_{band}_meds.fits'
        outfile = os.path.join(band_outdir, outfile)

        # Set up astromatic (sex & psfex & swarp) configs
        astro_config_dir = str(
            Path(utils.MODULE_DIR, 'medsmaker/superbit/astro_config/')
        )

        # Create an instance of BITMeasurement
        logprint('Setting up BITMeasurement configuration...\n')
        bm = medsmaker.BITMeasurement(
             science,
             data_dir,
             target_name,
             band,
             detection_bandpass,
             band_outdir,
             log=log,
             vb=vb
        )

        # TODO: Make this less hard-coded
        # Create an instance of HotColdSExtractor
        logprint('Setting up HotColdSExtractor configuration...\n')

        hc_config = os.path.join(astro_config_dir, 'hc_config.yaml')

        hcs = HotColdSExtractor(
            science,
            hc_config,
            band,
            target_name, 
            data_dir,
            astro_config_dir,
            log=log,
            vb=vb
        )


        # Get detection source file & catalog
        logprint('Making coadd...\n')

        # Make single band coadd and its catalog
        bm.make_coadd_image(astro_config_dir)
        
        logprint('Making coadd catalog...\n')        
        #hcs.make_coadd_catalog(use_band_coadd=True)
        bm.make_coadd_catalog(astro_config_dir)

        # Set detection file attributes
        bm.set_detection_files(use_band_coadd=True)

        # Then make dual-image mode SExtractor catalogs
        #hcs.make_dual_image_catalogs(detection_bandpass)

        logprint('Making single-exposure catalogs... \n')
        bm.make_exposure_catalogs(astro_config_dir)
        bm.make_exposure_weights()
        
        # Set image catalogs attribute
        bm.set_image_cats()

        # Build  a PSF model for each image.
        logprint('Making PSF models... \n')
        bm.make_psf_models(
            use_coadd=use_coadd,
            psf_mode=psf_mode,
            psf_seed=psf_seed,
            star_config=star_config,
        )

        logprint('Making MEDS... \n')

        logprint('Making image_info struct... \n')
        # Make the image_info struct.
        image_info = bm.make_image_info_struct(use_coadd=use_coadd)

        logprint('Make the object_info struct... \n')
        # Make the object_info struct.
        obj_info = bm.make_object_info_struct()

        logprint('Make the MEDS config file... \n')
        # Make the MEDS config file.
        meds_config = bm.make_meds_config(use_coadd, psf_mode)

        logprint('Create metadata for MEDS... \n')
        # Create metadata for MEDS
        # TODO: update this when we actually do photometric calibration!
        magzp = 30.
        meta = bm.meds_metadata(magzp, use_coadd)

        logprint('Finally, make and write the MEDS file... \n')
        # Finally, make and write the MEDS file.
        medsObj = meds.maker.MEDSMaker(
                  obj_info, image_info, config=meds_config,
                  psf_data=bm.psf_models, meta_data=meta
                  )

        logprint(f'Writing to {outfile} \n')
        medsObj.write(outfile)

    logprint('Done!')

    return 0

if __name__ == '__main__':
    args = parse_args()
    rc = main(args)

    if rc !=0:
        print(f'process_mocks failed w/ return code {rc}!')
