import numpy as np
import meds
import os
import psfex
from astropy.io import fits
import string
from pathlib import Path
import pickle
from astropy import wcs
import fitsio
import esutil as eu
from astropy.table import Table
import astropy.units as u
import superbit_lensing.utils as utils
from superbit_lensing.medsmaker.superbit.psf_extender import psf_extender
import glob
import pdb
import copy

'''
Goals:
  - Take as input calibrated images
  - Build a psf model (PSFEx or PIFF)
  - run the meds maker (use meds.Maker)
'''

class BITMeasurement():
    def __init__(self, image_files, data_dir, target_name, 
                band, detection_bandpass, outdir, work_dir=None, 
                log=None, vb=False):
        '''
        data_path: path to the image data not including target name
        coadd: Set to true if the first image file is a coadd image (must be first)
        '''

        self.image_files = image_files
        self.data_dir = data_dir
        self.target_name = target_name
        self.outdir = outdir
        self.vb = vb
        self.band = band
        self.detection_bandpass = detection_bandpass 

        self.image_cats = []
        self.detect_img_path = None
        self.detect_cat_path = None
        self.detection_cat = None
        self.psf_models = None

        # Set up logger
        if log is None:
            logfile = 'medsmaker.log'
            log = utils.setup_logger(logfile)

        self.logprint = utils.LogPrint(log, vb)

        # Set up base (code) directory
        filepath = Path(os.path.realpath(__file__))
        self.base_dir = filepath.parents[1]

        # If desired, set a tmp output directory
        self._set_work_dir(work_dir)

        # Cluster/bandpass directory containing all the cal/, cat/, etc. folders
        self.cluster_band_dir = os.path.join(self.data_dir,
                                    self.target_name, self.band)

    def _set_work_dir(self, work_dir):
        '''
        In case one wants to have psf outputs or (when they were still being
        made) SExtractor and SWarp products saved elsewhere than outdir.
        '''
        if work_dir == None:
            self.work_dir = self.outdir
        else:
            self.work_dir = work_dir
        utils.make_dir(self.work_dir)

    def set_image_cats(self):
        '''
        Get list of single-epoch exposure catalogs using filenames of
        single-epoch exposures. It bugs me to be defining science images in
        process_2023.py but catalogs here, but w/e.
        Note that this assumes OBA convention for data organization:
        [target_name]/[band]/[cal, cat, coadd, etc.]
        '''
        catdir = os.path.join(self.cluster_band_dir, 'cat')
        #imcats = glob.glob(os.path.join(top_dir, 'cat/*cal_cat.fits'))
        ims = self.image_files
        cnames = map(lambda x:os.path.basename(x).replace('.fits',\
                            '_cat.fits'), ims)
        imcats = list(map(lambda x:os.path.join(catdir, x), cnames))

        if os.path.exists(imcats[0]) == False:
            raise FileNotFoundError(f'No cat files found at location {catdir}')
        else:
            self.image_cats = imcats

    def make_exposure_catalogs(self, config_dir):
        '''
        Make single-exposure catalogs
        '''
        if os.path.isdir(config_dir) is False:
            raise f'{configdir} does not exist, exiting'

        # Make catalog directory
        cat_dir = os.path.join(self.cluster_band_dir, 'cat')
        utils.make_dir(cat_dir)
        self.logprint(f'made catalog directory {cat_dir}')
        for image_file in self.image_files:
            sexcat = self._run_sextractor(image_file=image_file,
                                          config_dir=config_dir,
                                          cat_dir=cat_dir)
            self.image_cats.append(sexcat)

    def make_exposure_weights(self):
        '''
        Make inverse-variance weight maps because ngmix needs them and we 
        don't have them for SuperBIT.
        Use the SExtractor BACKGROUND_RMS check-image as a basis
        '''
        
        img_names = self.image_files

        for img_name in img_names:
            # Read in the BACKGROUND_RMS image
            rms_name = img_name.replace('.fits', '.bkg_rms.fits')
            
            with fits.open(rms_name) as rms:
                # Assuming the image data is in the primary HDU
                background_rms_map = rms[0].data  
                # Keep the original header to use for the weight map
                header = rms[0].header  

                # Make a weight map
                weight_map = 1 / (background_rms_map**2)

                # Save the weight_map to a new file
                wgt_file_name = img_name.replace('.fits', '.weight.fits')
                hdu = fits.PrimaryHDU(weight_map, header=header)
                hdu.writeto(wgt_file_name, overwrite=True)

            print(f'Weight map saved to {wgt_file_name}')        
        
    def _run_sextractor(self, image_file, cat_dir, config_dir,
                        weight_file=None, back_type='AUTO'):
        '''
        Utility method to invoke Source Extractor on supplied detection file
        Returns: file path of catalog
        '''
        cat_name = os.path.basename(image_file).replace('.fits','_cat.fits')
        cat_file = os.path.join(cat_dir, cat_name)

        image_arg  = f'"{image_file}[0]"'
        name_arg   = '-CATALOG_NAME ' + cat_file
        config_arg = f'-c {os.path.join(config_dir, "sextractor.real.config")}'
        param_arg  = f'-PARAMETERS_NAME {os.path.join(config_dir, "sextractor.param")}'
        nnw_arg    = f'-STARNNW_NAME {os.path.join(config_dir, "default.nnw")}'
        filter_arg = f'-FILTER_NAME {os.path.join(config_dir, "gauss_2.0_3x3.conv")}'
        bg_sub_arg = f'-BACK_TYPE {back_type}'
        bkg_name   = image_file.replace('.fits','.sub.fits')
        seg_name   = image_file.replace('.fits','.sgm.fits')
        rms_name   = image_file.replace('.fits','.bkg_rms.fits')
        checkname_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name},{rms_name}'

        if weight_file is not None:
            weight_arg = f'-WEIGHT_IMAGE "{weight_file}[1]" ' + \
                         '-WEIGHT_TYPE MAP_WEIGHT'
        else:
            weight_arg = '-WEIGHT_TYPE NONE'

        cmd = ' '.join([
                    'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                    param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                    ])

        self.logprint("sex cmd is " + cmd)
        os.system(cmd)

        print(f'cat_name is {cat_file} \n')
        return cat_file

    def make_coadd_image(self, config_dir=None):
        '''
        Runs SWarp on provided (reduced!) image files to make a coadd image
        for SEX and PSFEx detection.
        '''
        # Make output directory for coadd image if it doesn't exist
        coadd_dir = os.path.join(self.cluster_band_dir, 'coadd')
        utils.make_dir(coadd_dir)

        # Get an Astromatic config path
        if config_dir is None:
            config_dir = os.path.join(self.base_dir,
                                      'superbit/astro_config/')

        # Define coadd image & weight file names and paths
        coadd_outname = f'{self.target_name}_coadd_{self.band}.fits'
        coadd_file = os.path.join(coadd_dir, coadd_outname)
        self.coadd_img_file = coadd_file

        # Same for weights
        weight_outname = coadd_outname.replace('.fits', '.weight.fits')
        weight_file = os.path.join(coadd_dir, weight_outname)

        image_args = ' '.join(self.image_files)
        config_arg = f'-c {config_dir}/swarp.config'
        resamp_arg = f'-RESAMPLE_DIR {coadd_dir}'
        cliplog_arg = f'CLIP_LOGNAME {coadd_dir}'
        outfile_arg = f'-IMAGEOUT_NAME {coadd_file} ' + \
                      f'-WEIGHTOUT_NAME {weight_file} '

        cmd_arr = {'swarp': 'swarp', 
                    'image_arg': image_args, 
                    'resamp_arg': resamp_arg,
                    'outfile_arg': outfile_arg, 
                    'config_arg': config_arg
                    }
       
        # Make external headers if band == detection
        #self._make_external_headers(cmd_arr)

        # Actually run the command
        cmd = ' '.join(cmd_arr.values())
        self.logprint('swarp cmd is ' + cmd)
        os.system(cmd)
    
        # Join weight file with image file in an MEF
        self.augment_coadd_image()

    def _make_external_headers(self, cmd_arr):
        """ Make external swarp header files to register coadds to one another
        in different bandpassess. Allows SExtractor to be run in dual-image 
        mode and thus color cuts to be made """
        
        # Need to create a copy of cmd_arr, or these values get
        # passed to make_coadd_image!
        head_arr = copy.copy(cmd_arr)
        
        # This line pulls out the filename in -IMAGEOUT_NAME [whatever.fits]
        # argument then creates header name by replacing ".fits" with ".head"
        header_name = \
            head_arr['outfile_arg'].split(' ')[1].replace(".fits",".head")

        if self.band == self.detection_bandpass:
            self.logprint(f'\nSwarp: band {self.band} matches ' +
                         'detection bandpass setting')
            self.logprint('Making external headers for u, g, b '+ 
                          f'based on {self.band}\n')
            
            # First, make the detection bandpass header
            header_only_arg = '-HEADER_ONLY Y'
            head_arr['header_only_arg'] = header_only_arg
            
            header_outfile = ' '.join(['-IMAGEOUT_NAME', header_name])
            
            # Update swarp command list (dict)
            head_arr['outfile_arg'] = header_outfile
            
            swarp_header_cmd = ' '.join(head_arr.values())
            self.logprint('swarp header-only cmd is ' + swarp_header_cmd)
            os.system(swarp_header_cmd)
            
            ## Cool, now that's done, create headers for the other bands too.
            all_bands = ['u', 'b', 'g']
            bands_to_do = np.setdiff1d(all_bands, self.detection_bandpass)
            for band in bands_to_do:
                # Get name
                band_header = header_name.replace(
                f'/{self.detection_bandpass}/',f'/{band}/').replace(
                f'{self.detection_bandpass}.head',f'{band}.head'
                )
                
                # Copy detection bandpass header to other band coadd dirs
                cp_cmd = f'cp {header_name} {band_header}'
                print(f'copying {header_name} to {band_header}')
                os.system(cp_cmd)
                
        else:
            print(f'\nSwarp: looking for external header...')

    def augment_coadd_image(self, add_sgm=False):
        '''
        Something of a utility function to add weight and sgm extensions to
        a single-band coadd, should the need arise
        '''

        coadd_im_file  = self.coadd_img_file
        coadd_wt_file  = coadd_im_file.replace('.fits', '.weight.fits')
        coadd_sgm_file = coadd_im_file.replace('.fits', '.sgm.fits')

        # Now, have to combine the weight and image file into one.
        im = fits.open(coadd_im_file, mode='append'); im[0].name = 'SCI'
        if im.__contains__('WGT') == True:
            self.logprint(f"\n Coadd image {coadd_im_file} already contains " +
                          "an extension named 'WGT', skipping...\n")
        else:
            wt = fits.open(coadd_wt_file); wt[0].name = 'WGT'
            im.append(wt[0])

        if add_sgm == True:
            sgm = fits.open(coadd_sgm_file); sgm[0].name = 'SEG'
            im.append(sgm[0])

        # Save; use writeto b/c flush and close don't update the SCI extension
        im.writeto(coadd_im_file, overwrite=True)
        im.close()


    def make_coadd_catalog(self, config_dir=None):
        '''
        Wrapper for astromatic tools to make coadd detection image
        from provided exposures and return a coadd catalog
        '''
        # Get an Astromatic config path
        if config_dir is None:
            config_dir = os.path.join(self.base_dir,
                                           'superbit/astro_config/')

        # Where would single-band coadd be hiding?
        coadd_dir = os.path.join(self.cluster_band_dir, 'coadd')

        # Set coadd filepath if it hasn't been set
        coadd_outname = f'{self.target_name}_coadd_{self.band}.fits'
        #weight_outname = coadd_outname.replace('.fits', '.weight.fits')
        #weight_filepath = os.path.join(coadd_dir, weight_outname)

        try:
            self.coadd_img_file
        except AttributeError:
            self.coadd_img_file = os.path.join(coadd_dir, coadd_outname)

        # Set pixel scale
        self.pix_scale = utils.get_pixel_scale(self.coadd_img_file)

        # Run SExtractor on coadd
        cat_name = self._run_sextractor(
            self.coadd_img_file,
            weight_file=self.coadd_img_file,
            cat_dir=coadd_dir,
            config_dir=config_dir, 
            back_type='MANUAL'
        )

        try:
            le_cat = fits.open(cat_name)
            try:
                self.catalog = le_cat[2].data
            except:
                self.catalog = le_cat[1].data

        except Exception as e:
            self.logprint("coadd catalog could not be loaded; check name?")
            raise(e)

    def set_detection_files(self, use_band_coadd=False):
        '''
        Get detection source file & catalog, assuming OBA convention for data
        organization: [target_name]/[band]/[cal, cat, coadd, etc.]
        '''
        # "pref" is catalog directory ("cat/" for oba, "coadd/" otherwise)
        if use_band_coadd == True:
            det = self.band
            pref = 'coadd/'
        else:
            det = 'det'
            pref = 'cat/'

        det_dir = os.path.join(self.data_dir, self.target_name, det)
        coadd_img_name = f'coadd/{self.target_name}_coadd_{det}.fits'
        coadd_cat_name = f'{pref}{self.target_name}_coadd_{det}_cat.fits'

        detection_img_file = os.path.join(det_dir, coadd_img_name)
        detection_cat_file = os.path.join(det_dir, coadd_cat_name)

        if os.path.exists(detection_img_file) == False:
            raise FileNotFoundError('No detection coadd image found '+
                                    f'at {detection_img_file}')
        else:
            self.detect_img_file = detection_img_file

        if use_band_coadd == True:
            self.coadd_img_file = detection_img_file 
            self.detect_img_file = detection_img_file
            
        if os.path.exists(detection_cat_file) == False:
            raise FileNotFoundError('No detection catalog found ',
                                    f'at {detection_cat_file}\nCheck name?')
            
        else:
            self.detect_cat_path = detection_cat_file
            dcat = fits.open(detection_cat_file)
            # hdu=2 if FITS_LDAC, hdu=1 if FITS_1.0
            try:
                self.detection_cat = dcat[2].data
            except:
                self.detection_cat  = dcat[1].data


    def make_psf_models(self, config_path=None, select_truth_stars=False,
                        use_coadd=True, psf_mode='piff', psf_seed=None,
                        star_config=None):
        '''
        Make PSF models. If select_truth_stars is enabled, cross-references an
        externally-supplied star catalog before PSF fitting.
        '''
        self.psf_models = []
        image_files = copy.deepcopy(self.image_files)
        image_cats  = copy.deepcopy(self.image_cats)

        if star_config is None:
            star_config = {'MIN_MAG': 23,
                           'MAX_MAG': 16,
                           'MIN_SIZE': 1.,
                           'MAX_SIZE': 3.5,
                           'MIN_SNR': 10,
                           'CLASS_STAR': 0.95,
                           'MAG_KEY': 'MAG_AUTO',
                           'SIZE_KEY': 'FWHM_IMAGE',
                           'SNR_KEY': 'SNR_WIN',
                           'use_truthstars': False
                           }
            self.logprint(f"Using default star params: {star_config}")
            #star_config = utils.AttrDict(star_config)

        if config_path is None:
            config_path = os.path.join(self.base_dir, 'superbit/astro_config/')
            self.logprint(f'Using PSF config path {config_path}')

        if psf_seed is None:
            psf_seed = utils.generate_seeds(1)

        if use_coadd is True:
            coadd_im = self.coadd_img_file.replace('.fits', '.sub.fits')
            image_files.insert(0, coadd_im)
            coadd_cat = self.coadd_img_file.replace('.fits', '_cat.fits')
            image_cats.insert(0, coadd_cat)

        Nim = len(image_files)
        self.logprint(f'Nim = {Nim}')
        self.logprint(f'len(image_cats)={len(image_cats)}')
        self.logprint(f'image_cats = {image_cats}')

        assert(len(image_cats)==Nim)

        k = 0
        for i in range(Nim):

            image_file = image_files[i]
            image_cat = image_cats[i]

            if psf_mode == 'piff':
                piff_model = self._make_piff_model(
                    image_file, image_cat, config_path=config_path,
                    star_config=star_config,
                    psf_seed=psf_seed
                    )
                self.psf_models.append(piff_model)

            elif psf_mode == 'psfex':
                psfex_model = self._make_psfex_model(
                    image_cat, config_path=config_path,
                    star_config=star_config
                    )
                self.psf_models.append(psfex_model)

            elif psf_mode == 'true':
                true_model = self._make_true_psf_model()
                self.psf_models.append(true_model)

        return

    def set_psfex_model_files(self, use_coadd=False):
        '''
        Utility function to grab PSFEx model names for when you 
        don't want to have to run PSFEx all over again
        ''' 
        imcats = self.image_cats 

        psfex_outdir = os.path.join(
            os.path.dirname(imcats[0]), 'psfex-output'
        )

        psfex_model_files = [
            os.path.join(
                psfex_outdir, 
                os.path.basename(x).replace(
                    'cat.fits', 'starcat.psf'
                )
            ) for x in imcats
        ]

        if use_coadd:
            # Grab the PSF coadd and prepend it to list of PSFs
            coadd_psf_filename = os.path.basename(
                self.coadd_img_file
            ).replace('.fits', '_starcat.psf')

            coadd_psf_file = os.path.join(
                os.path.dirname(self.coadd_img_file),
                'psfex-output',
                coadd_psf_filename
            )
            
            psfex_model_files.insert(0, coadd_psf_file)

            
        print(f"Using PSF models: {psfex_model_files}")

        self.psf_models = [
            psfex.PSFEx(m) for m in psfex_model_files
        ]

    def _make_psfex_model(self, im_cat, config_path,
                          star_config, psf_seed=None):
        '''
        Gets called by make_psf_models for every image in self.image_files
        Wrapper for PSFEx. Requires a FITS format catalog with vignettes

        TODO: Implement psf_seed for PSFEx!
        '''

        # Where to store PSFEx output
        psfex_outdir = os.path.join(os.path.dirname(im_cat), 'psfex-output')
        utils.make_dir(psfex_outdir)

        # Are we using a reference star catalog?
        if star_config['use_truthstars'] == True:
            truthfile = star_config['truth_filename']
            self.logprint('using truth catalog %s' % truthfile)
            autoselect_arg = '-SAMPLE_AUTOSELECT N'
        else:
            truthfile = None
            autoselect_arg = '-SAMPLE_AUTOSELECT Y'

        # Get a star catalog!
        psfcat_name = self._select_stars_for_psf(
                      sscat=im_cat,
                      star_config=star_config,
                      truthfile=truthfile
                      )

        # Define output names
        outcat_name = os.path.join(
            psfex_outdir,
            psfcat_name.replace('_starcat.fits','.psfex_starcat.fits')
        )
        psfex_model_file = os.path.join(
            psfex_outdir,
            os.path.basename(
                psfcat_name.replace('.fits','.psf')
            )
        )

        # Now run PSFEx on that image and accompanying catalog
        psfex_config_arg = '-c '+ config_path + 'psfex.config'
        psfdir_arg = f'-PSF_DIR {psfex_outdir}'

        cmd = ' '.join(
            ['psfex', psfcat_name, psfdir_arg, psfex_config_arg, \
                '-OUTCAT_NAME', outcat_name, autoselect_arg]
        )
        self.logprint("psfex cmd is " + cmd)
        os.system(cmd)

        cleanup_cmd = ' '.join(
            ['mv chi* resi* samp* snap* proto* *.xml', psfex_outdir]
            )
        cleanup_cmd2 = ' '.join(
            ['mv count*pdf ellipticity*pdf fwhm*pdf', psfex_outdir]
            )
        os.system(cleanup_cmd)
        os.system(cleanup_cmd2)

        try:
            model = psfex.PSFEx(psfex_model_file)
        except:
            model = None
            print(f'WARNING:\n Could not find PSFEx model file {psfex_model_file}\n')
        return model


    def _make_piff_model(self, im_file, im_cat, config_path, psf_seed,
                         star_config=None):
        '''
        Method to invoke PIFF for PSF modeling
        Returns a "PiffExtender" object with the get_rec() and get_cen()
        functions expected by meds.maker

        First, let's get it to run on one, then we can focus on running list
        '''

        output_dir = os.path.join(self.outdir, 'piff-output',
                        os.path.basename(im_file).split('.fits')[0])
        utils.make_dir(output_dir)

        output_name = os.path.basename(im_file).replace('.fits', '.piff')
        output_path = os.path.join(output_dir, output_name)

        # update piff config w/ psf_seed
        base_piff_config = os.path.join(config_path, 'piff.config')
        run_piff_config = os.path.join(output_dir, 'piff.config')

        config = utils.read_yaml(base_piff_config)
        config['select']['seed'] = psf_seed
        utils.write_yaml(config, run_piff_config)

        # PIFF wants RA in hours, not degrees
        ra  = fits.getval(im_file, 'CRVAL1') / 15.0
        dec = fits.getval(im_file, 'CRVAL2')

        if star_config['use_truthstars'] == True:
            truthfile = star_config['truth_filename']
            self.logprint('using truth catalog %s' % truthfile)
        else:
            truthfile = None

        psfcat_name = self._select_stars_for_psf(
                      sscat=im_cat,
                      star_config=star_config,
                      truthfile=truthfile
                      )

        # Now run PIFF on that image and accompanying catalog
        image_arg  = f'input.image_file_name={im_file}'
        psfcat_arg = f'input.cat_file_name={psfcat_name}'
        coord_arg  = f'input.ra={ra} input.dec={dec}'
        output_arg = f'output.file_name={output_name} output.dir={output_dir}'

        cmd = f'piffify {run_piff_config} {image_arg} {psfcat_arg} ' + \
              f'{output_arg} {coord_arg}'

        self.logprint('piff cmd is ' + cmd)
        os.system(cmd)

        # use stamp size defined in config
        psf_stamp_size = config['psf']['model']['size']

        # Extend PIFF PSF to have needed PSFEx methods for MEDS
        kwargs = {
            'piff_file': output_path
        }
        piff_extended = psf_extender('piff', psf_stamp_size, **kwargs)

        return piff_extended


    def _make_true_psf_model(self, stamp_size=25, psf_pix_scale=None):
        '''
        Construct a PSF image to populate a MEDS file using the actual
        PSF used in the creation of single-epoch images

        NOTE: For now, this function assumes a constant PSF for all images
        NOTE: Should only be used for validation simulations!
        '''

        if psf_pix_scale is None:
            # make it higher res
            # psf_pix_scale = self.pix_scale / 4
            psf_pix_scale = self.pix_scale

        # there should only be one of these
        true_psf_file = glob.glob(
            os.path.join(self.data_dir, '*true_psf.pkl')
            )[0]

        with open(true_psf_file, 'rb') as fname:
            true_psf = pickle.load(fname)

        # Extend True GalSim PSF to have needed PSFEx methods for MEDS
        kwargs = {
            'psf': true_psf,
            'psf_pix_scale': psf_pix_scale
        }
        true_extended = psf_extender('true', stamp_size, **kwargs)

        return true_extended


    def _select_stars_for_psf(self, sscat, truthfile, star_config):
        '''
        Method to obtain stars from SExtractor catalog using the truth catalog
        Inputs
            sscat: input catalog from which to select stars
            truthcat: a pre-vetted catalog of stars
        '''
        
        ss_fits = fits.open(sscat)
        if len(ss_fits) == 3:
            # It is an ldac
            ext = 2
        else:
            ext = 1
        ss = ss_fits[ext].data

        if truthfile is not None:
            # Create star catalog based on reference ("truth") star catalog
            try:
                stars = Table.read(truthfile, format='fits',
                                   hdu=star_config['cat_hdu'])
            except:
                stars = Table.read(truthfile, format='ascii')

            # match sscat against truth star catalog; 0.72" = 5 SuperBIT pixels
            self.logprint("Selecting stars using truth catalog " +
                          f"with {len(stars)} stars")

            star_matcher = eu.htm.Matcher(16,
                                ra=stars[star_config['truth_ra_key']],
                                dec=stars[star_config['truth_dec_key']]
                                )
            matches, starmatches, dist = \
                                star_matcher.match(ra=ss['ALPHAWIN_J2000'],
                                dec=ss['DELTAWIN_J2000'],
                                radius=2/3600., maxmatch=1
                                )

            og_len = len(ss); ss = ss[matches]
            wg_stars = (ss['SNR_WIN'] > star_config['MIN_SNR'])

            self.logprint(f'{len(dist)}/{og_len} objects ' +
                          'matched to reference (truth) star catalog \n' +
                          f'{len(ss[wg_stars])} stars passed MIN_SNR threshold'
                          )

        else:
            # Do more standard stellar locus matching
            self.logprint("Selecting stars on CLASS_STAR, SIZE and MAG...")
            wg_stars = \
                (ss['CLASS_STAR'] > star_config['CLASS_STAR']) & \
                (ss[star_config['SIZE_KEY']] > star_config['MIN_SIZE']) & \
                (ss[star_config['SIZE_KEY']] < star_config['MAX_SIZE']) & \
                (ss[star_config['MAG_KEY']] < star_config['MIN_MAG']) & \
                (ss[star_config['MAG_KEY']] > star_config['MAX_MAG'])

        # Save output star catalog to file
        ss_fits[ext].data = ss[wg_stars]
        
        outname = sscat.replace('_cat.fits','_starcat.fits')
        ss_fits.writeto(outname, overwrite=True)

        return outname

    def make_image_info_struct(self, max_len_of_filepath=200, use_coadd=False):
        # max_len_of_filepath may cause issues down the line if the file path
        # is particularly long

        image_files = []; weight_files = []
        
        coadd_image  = self.detect_img_file
        coadd_weight = self.detect_img_file.replace('.fits', '.weight.fits') 
        coadd_segmap = self.detect_img_file.replace('.fits', '.sgm.fits') 
        
        for img in self.image_files:
            bkgsub_name = img.replace('.fits','.sub.fits')
            weight_name = img.replace('.fits', '.weight.fits')
            image_files.append(bkgsub_name)
            weight_files.append(weight_name)

        if use_coadd == True:
            image_files.insert(0, coadd_image)
            weight_files.insert(0, coadd_weight)

        # If used, will be put first
        Nim = len(image_files)
        image_info = meds.util.get_image_info_struct(Nim, max_len_of_filepath)

        i=0
        for image_file in range(Nim):
            image_info[i]['image_path']  =  image_files[i]
            image_info[i]['image_ext']   =  0
            image_info[i]['weight_path'] =  weight_files[i]
            image_info[i]['weight_ext']  =  0
            #image_info[i]['bmask_path']  =  None
            #image_info[i]['bmask_ext']   =  0
            image_info[i]['seg_path']    =  coadd_segmap # Use coadd segmap for uberseg!
            image_info[i]['seg_ext']     =  0

            # The default is for 0 offset between the internal numpy arrays
            # and the images, but we use the FITS standard of a (1,1) origin.
            # In principle we could probably set this automatically by checking
            # the images
            image_info[i]['position_offset'] = 1

            i+=1

        return image_info

    def make_meds_config(self, use_coadd, psf_mode, extra_parameters=None,
                         use_joblib=False):
        '''
        extra_parameters: dictionary of keys to be used to update the base
                          MEDS configuration dict
        '''
        # sensible default config.
        config = {
            'first_image_is_coadd': use_coadd,
            'cutout_types':['weight','seg','bmask'],
            'psf_type': psf_mode,
            'use_joblib': use_joblib
            }

        if extra_parameters is not None:
            config.update(extra_parameters)

        return config

    def meds_metadata(self, magzp, use_coadd):
        '''
        magzp: float
            The reference magnitude zeropoint
        use_coadd: bool
            Set to True if the first MEDS cutout is from the coadd
        '''

        meta = np.empty(1, [
            ('magzp_ref', float),
            ('has_coadd', bool)
            ])

        meta['magzp_ref'] = magzp
        meta['has_coadd'] = use_coadd

        return meta

    def _calculate_box_size(self, angular_size, size_multiplier = 2.5,
                            min_size = 16, max_size= 64):
        '''
        Calculate the cutout size for this survey.

        :angular_size: angular size of a source, with some kind of angular units.
        :size_multiplier: Amount to multiply angular size by to choose boxsize.
        :deconvolved:
        :min_size:
        :max_size:
        '''

        pixel_scale = utils.get_pixel_scale(self.detect_img_file)
        box_size_float = np.ceil(angular_size/pixel_scale)

        # Available box sizes to choose from -> 16 to 256 in increments of 2
        available_sizes = min_size * 2**(np.arange(np.ceil(np.log2(max_size)-np.log2(min_size)+1)).astype(int))

        # If a single angular_size was proffered:
        if isinstance(box_size_float, np.ndarray):
            available_sizes_matrix = available_sizes.reshape(1,available_sizes.size).repeat(angular_size.size,axis=0)
            box_size_float_matrix=box_size_float.reshape(box_size_float.size,1)
            #available_sizes_matrix[box_size_float.reshape(box_size_float.size,1) > available_sizes.reshape(1,available_sizes.size)] = np.max(available_sizes)+1
            available_sizes_matrix[box_size_float.reshape(box_size_float.size,1) > available_sizes_matrix] = np.max(available_sizes)+1
            box_size = np.min(available_sizes_matrix,axis=1)
        else:
            box_size = np.min( available_sizes[ available_sizes > box_size_float ] )
        return box_size

    def make_object_info_struct(self, catalog=None):
        if catalog is None:
            catalog = self.detection_cat

        obj_str = meds.util.get_meds_input_struct(catalog.size, \
                  extra_fields = [('KRON_RADIUS', float), \
                  ('number', int), ('XWIN_IMAGE', float), \
                  ('YWIN_IMAGE', float)]
                  )
        obj_str['id'] = catalog['NUMBER']
        obj_str['number'] = np.arange(catalog.size)+1
        obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'])
        obj_str['ra'] = catalog['ALPHAWIN_J2000']
        obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['XWIN_IMAGE'] = catalog['XWIN_IMAGE']
        obj_str['YWIN_IMAGE'] = catalog['YWIN_IMAGE']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']

        return obj_str

    def run(self,outfile='superbit_ims.meds', overwrite=False,
            source_selection=False, select_truth_stars=False, psf_mode='piff',
            use_coadd=True):
        # Make a MEDS, overwriteing if needed

        #### ONLY FOR DEBUG
        #### Set up the paths to the science and calibration data
        #self.set_working_dir()
        #self.set_path_to_psf()
        #self.set_path_to_science_data()
        # Add a WCS to the science
        #self.add_wcs_to_science_frames()
        ####################

        # Reduce the data.
        # self.reduce(overwrite=overwrite,skip_sci_reduce=True)
        # Make a mask.
        # NB: can also read in a pre-existing mask by setting self.mask_file
        #self.make_mask(mask_name='mask.fits',overwrite=overwrite)

        # Combine images, make a catalog.
        config_path = os.path.join(self.base_dir, 'superbit/astro_config/')
        self.make_coadd_catalog(sextractor_config_dir=config_path,
                          source_selection=source_selection)
        # Make catalogs for individual exposures
        self.make_exposure_catalogs(sextractor_config_dir=config_path)
        # Build a PSF model for each image.
        self.make_psf_models(select_truth_stars=select_truth_stars, use_coadd=False, psf_mode=psf_mode)
        # Make the image_info struct.
        image_info = self.make_image_info_struct()
        # Make the object_info struct.
        obj_info = self.make_object_info_struct()
        # Make the MEDS config file.
        meds_config = self.make_meds_config()
        # Create metadata for MEDS
        magzp = 30.
        meta = self._meds_metadata(magzp, use_coadd)
        # Finally, make and write the MEDS file.
        medsObj = meds.maker.MEDSMaker(
            obj_info, image_info, config=meds_config, psf_data=self.psf_models,
            meta_data=meta
            )
        medsObj.write(outfile)
