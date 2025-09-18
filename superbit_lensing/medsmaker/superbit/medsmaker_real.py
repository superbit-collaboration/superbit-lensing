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
from superbit_lensing.match import SkyCoordMatcher
from superbit_lensing.medsmaker.superbit.psf_extender import psf_extender
import glob
import pdb
import copy
import time

'''
Goals:
  - Take as input calibrated images
  - Build a psf model (PSFEx or PIFF)
  - run the meds maker (use meds.Maker)
'''

class BITMeasurement():
    def __init__(self, image_files, data_dir, target_name, 
                band, detection_bandpass, outdir, work_dir=None, 
                log=None, vb=False, ext_header=False):
        '''
        data_path: path to the image data not including target name
        coadd: Set to true if the first image file is a coadd image (must be first)
        '''

        self.image_files = image_files
        self.data_dir = data_dir
        self.target_name = target_name
        self.outdir = outdir
        self.vb = vb
        self.ext_header = ext_header
        self.band = band
        self.detection_bandpass = detection_bandpass

        self.image_cats = []
        self.detect_img_path = None
        self.detect_cat_path = None
        self.detection_cat = None
        self.psf_models = None
        self.tolerance_deg = 1e-4
        # Set up logger
        if log is None:
            logfile = 'medsmaker.log'
            log = utils.setup_logger(logfile)

        self.logprint = utils.LogPrint(log, vb)

        # Set up base (code) directory
        filepath = Path(os.path.realpath(__file__))
        self.base_dir = filepath.parents[1]

        project_root = filepath.parents[3]  # This points to superbit-lensing/
        self.exposure_mask_fname = str(project_root / "data" / "masks" / "mask_dark_55percent_300.npy")
        self.star_file = os.path.join(self.data_dir, "catalogs", "stars", f"{self.target_name}_gaia_dr3.fits")

        try:
            self.gaia_stars = Table.read(self.star_file, hdu=1)
        except Exception as e:
            print(f"[WARNING] Could not open star file: {e}. Trying GAIA query...")
            try:
                # Try querying GAIA directly
                self.gaia_stars = utils.gaia_query(self.target_name)
                print("[INFO] GAIA query successful.")

                # Try saving the query result for reuse
                try:
                    self.gaia_stars.write(self.star_file, format="fits", overwrite=True)
                    print(f"[INFO] Saved queried GAIA stars to {self.star_file}")
                except Exception as e_save:
                    print(f"[WARNING] Could not save GAIA stars to file: {e_save}")

            except Exception as e_query:
                raise RuntimeError(
                    f"Failed to obtain GAIA stars for {self.target_name}: "
                    f"could not read {self.star_file} and GAIA query also failed ({e_query})."
                )

        # If desired, set a tmp output directory
        self._set_work_dir(work_dir)

        # Cluster/bandpass directory containing all the cal/, cat/, etc. folders
        self.cluster_band_dir = os.path.join(self.data_dir,
                                    self.target_name, self.band)

    def check_cat_image_order(self, verbose=True):
        """
        Check that for every catalog file in self.image_cats there is a
        corresponding image file in self.image_files, preserving order.
        
        Catalog example:
            /scratch/.../b/cat/Abell3411_1_300_1683033980_clean_cat.fits
        Expected image:
            /scratch/.../b/cal/Abell3411_1_300_1683033980_clean.fits
        
        Parameters
        ----------
        verbose : bool, optional
            If True, print quick stats about matches/mismatches.
        
        Returns
        -------
        bool
            True if all catalogs match their images in order, else False
        """
        n_cats, n_imgs = len(self.image_cats), len(self.image_files)
        if n_cats != n_imgs:
            if verbose:
                print(f"[Mismatch] Counts differ: {n_cats} cats, {n_imgs} images")
                self.logprint(f"[Mismatch] Counts differ: {n_cats} cats, {n_imgs} images")
            return False

        n_match = 0
        n_fail = 0
        mismatches = []

        for idx, (cat, img) in enumerate(zip(self.image_cats, self.image_files)):
            # Expected image path: swap directory and drop "_cat"
            expected_img = cat.replace("/cat/", "/cal/").replace("_cat.fits", ".fits")

            if expected_img == img:
                n_match += 1
            else:
                n_fail += 1
                mismatches.append((idx, cat, img, expected_img))

        if verbose:
            print(f"[check_cat_image_order] Matches: {n_match}/{n_cats}, Fails: {n_fail}")
            self.logprint(f"[check_cat_image_order] Matches: {n_match}/{n_cats}, Fails: {n_fail}")
            if n_fail > 0:
                for idx, cat, img, exp_img in mismatches:
                    print(f"  -> Exposure {idx}:")
                    self.logprint(f"  -> Exposure {idx}:")
                    print(f"     Catalog: {cat}")
                    self.logprint(f"     Catalog: {cat}")
                    print(f"     Got image: {img}")
                    self.logprint(f"     Got image: {img}")
                    print(f"     Expected : {exp_img}")
                    self.logprint(f"     Expected : {exp_img}")

        return n_fail == 0

    def check_psf_model_order(self, verbose=True):
        """
        Check that for every PSF model in self.psf_model_files there is a
        corresponding image and catalog, preserving order.

        Expected structure:
            image: /.../b/cal/Abell3411_1_300_1683033980_clean.fits
            catalog: /.../b/cat/Abell3411_1_300_1683033980_clean_cat.fits
            psf model: /.../b/cat/psfex-output/Abell3411_1_300_1683033980_clean_starcat.psf

        Parameters
        ----------
        verbose : bool, optional
            If True, print quick stats about matches/mismatches.

        Returns
        -------
        bool
            True if all PSF models match their images and catalogs in order, else False.
        """
        n_images, n_cats, n_psfs = len(self.image_files), len(self.image_cats), len(self.psf_model_files[1:])
        if not (n_images == n_cats == n_psfs):
            if verbose:
                print(f"[Mismatch] Counts differ: {n_images} images, {n_cats} cats, {n_psfs} psfs")
                self.logprint(f"[Mismatch] Counts differ: {n_images} images, {n_cats} cats, {n_psfs} psfs")
            return False

        n_match = 0
        n_fail = 0
        mismatches = []

        for idx, (psf, img, cat) in enumerate(zip(self.psf_model_files[1:], self.image_files, self.image_cats)):
            # Expected catalog path from image
            expected_cat = img.replace("/cal/", "/cat/").replace(".fits", "_cat.fits")

            # Expected PSF file from catalog
            expected_psf = expected_cat.replace("/cat/", "/cat/psfex-output/").replace("_cat.fits", "_starcat.psf")

            if cat == expected_cat and psf == expected_psf:
                n_match += 1
            else:
                n_fail += 1
                mismatches.append((idx, img, cat, psf, expected_cat, expected_psf))

        if verbose:
            print(f"[check_psf_model_order] Matches: {n_match}/{n_images}, Fails: {n_fail}")
            self.logprint(f"[check_psf_model_order] Matches: {n_match}/{n_images}, Fails: {n_fail}")
            if n_fail > 0:
                for idx, img, cat, psf, exp_cat, exp_psf in mismatches:
                    print(f"  -> Exposure {idx}:")
                    self.logprint(f"  -> Exposure {idx}:")
                    if cat != exp_cat:
                        print(f"     Catalog mismatch:\n       got {cat}\n       exp {exp_cat}")
                        self.logprint(f"     Catalog mismatch:\n       got {cat}\n       exp {exp_cat}")
                    if psf != exp_psf:
                        print(f"     PSF mismatch:\n       got {psf}\n       exp {exp_psf}")
                        self.logprint(f"     PSF mismatch:\n       got {psf}\n       exp {exp_psf}")

        return n_fail == 0

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
            sexcat = self._run_sextractor_on_exposure(image_file=image_file,
                                          config_dir=config_dir,
                                          cat_dir=cat_dir, admoms=True)
            self.image_cats.append(sexcat)
        self.cat_dir = cat_dir

    def filter_files(self, std_threshold=0.2):
        """
        Filter out exposures with poor PSF FWHM stability based on matched stars.

        For each exposure:
        1. Read the star catalog (HDU=2).
        2. Match stars to GAIA using `SkyCoordMatcher`.
        3. Apply star quality cuts: MAG_AUTO > 16.5 and SNR_WIN > 20.
        4. Compute star FWHM from T_ADMOM values.
        5. Keep only finite, positive values and apply 1–99 percentile filtering.
        6. Calculate the standard deviation of the filtered FWHM distribution.
        7. Retain the exposure if the scatter is below `std_threshold`.

        The function updates `self.image_files` and `self.image_cats` in place
        to keep only the filtered set of exposures, and ensures catalogs and images
        remain properly aligned.

        Parameters
        ----------
        std_threshold : float, optional
            Maximum allowed standard deviation of PSF FWHM (arcsec).
            Exposures with higher scatter are discarded. Default is 0.2.

        Returns
        -------
        int
            Number of exposures retained after filtering.

        Raises
        ------
        ValueError
            If catalog and image lists are misaligned before or after filtering.
        """
        # Ensure catalog-image pairs are ordered correctly
        if not self.check_cat_image_order():
            raise ValueError("Catalogs and images are not aligned in order.")

        # Work on deep copies (preserve originals until filtering is complete)
        image_files = copy.deepcopy(self.image_files)
        image_cats  = copy.deepcopy(self.image_cats)

        total_exposures = len(image_files)
        image_cats_filtered = []
        image_files_filtered = []

        for catfile, imfile in zip(image_cats, image_files):
            # Read star catalog from HDU 2
            exp_data = Table.read(catfile, hdu=2)

            # Match to GAIA stars
            matcher = SkyCoordMatcher(
                exp_data, self.gaia_stars,
                cat1_ratag='ALPHAWIN_J2000', cat1_dectag='DELTAWIN_J2000',
                cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000',
                return_idx=True,
                match_radius=1 * self.tolerance_deg,
                verbose=False
            )
            matched1, matched2, idx1, idx2 = matcher.get_matched_pairs()

            # Apply star quality cuts
            valid_stars = (matched1['MAG_AUTO'] > 16.5) & (matched1['SNR_WIN'] > 20)
            matched1 = matched1[valid_stars]

            # Compute FWHM from T_ADMOM
            star_T_admom = matched1["T_ADMOM"]
            star_fwhm = 2.355 * np.sqrt(star_T_admom / 2)

            # Keep only valid finite positive values
            valid_mask = np.isfinite(star_fwhm) & (star_fwhm > 0)
            star_fwhm_valid = star_fwhm[valid_mask]

            if len(star_fwhm_valid) == 0:
                continue  # skip if no valid stars

            # Apply 1–99 percentile filtering
            p1, p99 = np.percentile(star_fwhm_valid, [1, 99])
            percentile_mask = (star_fwhm_valid >= p1) & (star_fwhm_valid <= p99)
            star_fwhm_filtered = star_fwhm_valid[percentile_mask]

            # Compute scatter
            std_fwhm = np.std(star_fwhm_filtered)

            # Keep if scatter is below threshold
            if std_fwhm < std_threshold:
                image_cats_filtered.append(catfile)
                image_files_filtered.append(imfile)

        # Update object state
        self.image_files = image_files_filtered
        self.image_cats  = image_cats_filtered

        # Verify order after filtering
        if not self.check_cat_image_order():
            raise ValueError("Catalogs and images became misaligned after filtering.")

        kept_exposures = len(self.image_cats)
        percent_kept = (kept_exposures / total_exposures * 100) if total_exposures > 0 else 0

        print(f"Kept {kept_exposures} out of {total_exposures} exposures ({percent_kept:.1f}%)")
        self.logprint(f"Kept {kept_exposures} out of {total_exposures} exposures ({percent_kept:.1f}%)")
        return len(self.image_cats)

    def make_exposure_weights(self):
        """
        Make inverse-variance weight maps for single exposures, taking into account
        the exposure mask. Uses the SExtractor BACKGROUND_RMS check-image as a basis.
        """

        img_names = self.image_files
        mask = np.load(self.exposure_mask_fname)

        for img_name in img_names:
            # Read in the BACKGROUND_RMS image
            rms_name = img_name.replace('.fits', '.bkg_rms.fits')
            wgt_file_name = img_name.replace('.fits', '.weight.fits')

            with fits.open(rms_name) as rms:
                background_rms_map = rms[0].data
                header = rms[0].header.copy()

            # Prevent division by zero or negative values
            safe_rms = np.where(background_rms_map > 0, background_rms_map, np.nan)
            weight_map = np.where(np.isfinite(safe_rms), 1.0 / (safe_rms**2), 0.0)

            # Apply exposure mask: mask==True → weight = 0
            weight_map[mask] = 0.0

            # Save the weight map
            hdu = fits.PrimaryHDU(weight_map, header=header)
            hdu.writeto(wgt_file_name, overwrite=True)

            msg = f"[make_exposure_weights] Weight map saved to {wgt_file_name}"
            print(msg)
            self.logprint(msg)

    def make_coadd_weight(self):
        """
        Make inverse-variance weight maps because ngmix needs them and we 
        don't have them for SuperBIT.
        Use the SExtractor BACKGROUND_RMS check-image as a basis.
        """

        coadd_img_name = self.coadd_img_file

        # Read in the BACKGROUND_RMS image
        rms_name = coadd_img_name.replace(".fits", ".bkg_rms.fits")
        wgt_file_name = coadd_img_name.replace(".fits", ".weight.fits")

        with fits.open(rms_name) as rms:
            background_rms_map = rms[0].data
            header = rms[0].header.copy()

        # Prevent division by zero or negative values
        safe_rms = np.where(background_rms_map > 0, background_rms_map, np.nan)
        weight_map = np.where(np.isfinite(safe_rms), 1.0 / (safe_rms**2), 0.0)

        # Save the weight map
        hdu = fits.PrimaryHDU(weight_map, header=header)
        hdu.writeto(wgt_file_name, overwrite=True)

        msg = f"[make_coadd_weight] Weight map saved to {wgt_file_name}"
        print(msg)
        self.logprint(msg)      
        
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

    def make_sextractor_weight(self):
        '''
        Make inverse-variance weight maps because ngmix needs them and we 
        don't have them for SuperBIT.
        Use the SExtractor BACKGROUND_RMS check-image as a basis
        '''
        
        img_names = self.image_files
        mask = np.load(self.exposure_mask_fname)

        self.sex_wgt_files = [img_name.replace('.fits', '.sex_weight.fits') for img_name in img_names]

        for img_name in img_names:
            
            with fits.open(img_name) as img:
                # Assuming the image data is in the primary HDU
                img_data = img[0].data  
                # Keep the original header to use for the weight map
                header = img[0].header  

                # Initialize weight_map with ones
                weight_map = np.ones_like(img_data, dtype=np.float32)

                # Set weight to 0 where mask is True
                weight_map[mask] = 0

                # Save the weight_map to a new file
                sex_wgt_file_name = img_name.replace('.fits', '.sex_weight.fits')
                hdu = fits.PrimaryHDU(weight_map, header=header)
                hdu.writeto(sex_wgt_file_name, overwrite=True)

            print(f'Weight map saved to {sex_wgt_file_name}')

    def make_exposure_bmask(self):
        '''
        Make inverse-variance weight maps because ngmix needs them and we 
        don't have them for SuperBIT.
        Use the SExtractor BACKGROUND_RMS check-image as a basis
        '''
        
        img_names = self.image_files
        mask = np.load(self.exposure_mask_fname)

        for img_name in img_names:
            
            with fits.open(img_name) as img:
                # Assuming the image data is in the primary HDU
                img_data = img[0].data  
                # Keep the original header to use for the weight map
                header = img[0].header  

                # Initialize weight_map with ones
                bmask_map = np.zeros_like(img_data, dtype=np.float32)

                # Set weight to 1e30 where mask is True
                bmask_map[mask] = 1.0

                # Save the weight_map to a new file
                bmask_file_name = img_name.replace('.fits', '.bmask.fits')
                hdu = fits.PrimaryHDU(bmask_map, header=header)
                hdu.writeto(bmask_file_name, overwrite=True)

            print(f'bmask map saved to {bmask_file_name}')                               

    def _run_sextractor_on_exposure(self, image_file, cat_dir, config_dir,
                        weight_file=None, back_type='AUTO', admoms=False):
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
            weight_arg = f'-WEIGHT_IMAGE "{weight_file}[0]" ' + \
                         '-WEIGHT_TYPE MAP_RMS'
        else:
            weight_file = image_file.replace('.fits','.sex_weight.fits')
            weight_arg = f'-WEIGHT_IMAGE "{weight_file}[0]" ' + \
                         '-WEIGHT_TYPE MAP_WEIGHT'

        cmd = ' '.join([
                    'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                    param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                    ])

        self.logprint("sex cmd is " + cmd)
        os.system(cmd)

        print(f'saved catalog with {len(Table.read(cat_file, hdu=2))} objects to {cat_file} \n')
        if admoms:
            print(f"adding admoms columns to {cat_file}")
            cat = utils.add_admom_columns(cat_file, mode="galsim")
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
        sex_weight_files = ' '.join(self.sex_wgt_files)
        weight_arg = f'-WEIGHT_IMAGE "{sex_weight_files}" ' + \
                      '-WEIGHT_TYPE MAP_WEIGHT'        
        config_arg = f'-c {config_dir}/swarp.config'
        resamp_arg = f'-RESAMPLE_DIR {coadd_dir}'
        cliplog_arg = f'CLIP_LOGNAME {coadd_dir}'
        outfile_arg = f'-IMAGEOUT_NAME {coadd_file} ' + \
                      f'-WEIGHTOUT_NAME {weight_file} '

        cmd_arr = {'swarp': 'swarp', 
                    'image_arg': image_args,
                    'weight_arg': weight_arg, 
                    'resamp_arg': resamp_arg,
                    'outfile_arg': outfile_arg, 
                    'config_arg': config_arg
                    }
       
        # Make external headers if band == detection
        if self.ext_header:
            self._make_external_headers(cmd_arr)

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
        self.psf_model_files = []
        image_files = copy.deepcopy(self.image_files)
        image_cats  = copy.deepcopy(self.image_cats)

        if star_config is None:
            star_config = {'MIN_MAG': 27,
                           'MAX_MAG': 16.5,
                           'MIN_SIZE': 1.,
                           'MAX_SIZE': 3.5,
                           'MIN_SNR': 20,
                           'CLASS_STAR': 0.95,
                           'MAG_KEY': 'MAG_AUTO',
                           'SIZE_KEY': 'FWHM_IMAGE',
                           'SNR_KEY': 'SNR_WIN',
                           'truth_ra_key': 'ALPHAWIN_J2000',
                           'truth_dec_key': 'DELTAWIN_J2000',
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
                psfex_model, psfex_model_file = self._make_psfex_model(
                    image_cat, config_path=config_path,
                    star_config=star_config
                    )
                self.psf_models.append(psfex_model)
                self.psf_model_files.append(psfex_model_file)

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
        if star_config['use_truthstars']:
            truthfile = star_config['truth_filename']
            self.logprint('using truth catalog %s' % truthfile)
        else:
            truthfile = None

        # Get a star catalog!
        psfcat_name = self._select_stars_for_psf(
                      sscat=im_cat,
                      star_config=star_config,
                      truthfile=truthfile
                      )

        if star_config["use_truthstars"] or self.gaia_query_happened:
            if self.gaia_query_happened:
                self.logprint("Yay! Gaia Query was successful")
            autoselect_arg = '-SAMPLE_AUTOSELECT N'
        else:
            self.logprint('Things are bad psfex is autoselecting objects')
            autoselect_arg = '-SAMPLE_AUTOSELECT Y'

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
            psfex_model_file = None
            print(f'WARNING:\n Could not find PSFEx model file {psfex_model_file}\n')
        return model, psfex_model_file


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

    def _select_stars_for_psf(self, sscat, star_config, truthfile=None):
        '''
        Method to obtain stars from SExtractor catalog using the truth catalog
        Inputs
            sscat: input catalog from which to select stars
            truthcat: a pre-vetted catalog of stars
        '''
        self.gaia_query_happened = True
        ss_fits = fits.open(sscat)
        if len(ss_fits) == 3:
            # It is an ldac
            ext = 2
        else:
            ext = 1
        ss = ss_fits[ext].data

        matcher = SkyCoordMatcher(ss, self.gaia_stars,
                                  cat1_ratag='ALPHAWIN_J2000', cat1_dectag='DELTAWIN_J2000',
                                  cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000',
                                  return_idx=True, match_radius=1 * self.tolerance_deg)

        ss, matched2, idx1, idx2 = matcher.get_matched_pairs()    

        wg_stars = (ss['MAG_AUTO'] > 16.5) & (ss['SNR_WIN'] > 20)
        # Save output star catalog to file
        ss_fits[ext].data = ss[wg_stars]
        
        outname = sscat.replace('_cat.fits','_starcat.fits')
        ss_fits.writeto(outname, overwrite=True)

        ss_fits_union = fits.HDUList([hdu.copy() for hdu in ss_fits])
        ss_fits_union[ext].data = ss
        union_outname = sscat.replace('_cat.fits','_starcat_union.fits')
        ss_fits_union.writeto(union_outname, overwrite=True)

        return outname

    def _select_stars_for_psf_v2(self, sscat, truthfile, star_config):
        '''
        Method to obtain stars from SExtractor catalog using the truth catalog
        Inputs
            sscat: input catalog from which to select stars
            truthcat: a pre-vetted catalog of stars
        '''
        self.gaia_query_happened = False
        ss_fits = fits.open(sscat)
        if len(ss_fits) == 3:
            # It is an ldac
            ext = 2
        else:
            ext = 1
        ss = ss_fits[ext].data

        if truthfile is not None:
            # Create star catalog based on reference ("truth") star catalog
            self.logprint(f"Attempting to read truth catalog from {truthfile}")
            
            try:
                # First attempt: Try reading as FITS with specified HDU
                self.logprint(f"Trying FITS format with HDU={star_config['cat_hdu']}")
                stars = Table.read(truthfile, format='fits', hdu=star_config['cat_hdu'])
                self.logprint(f"Successfully read truth catalog in FITS format with {len(stars)} entries")
            
            except Exception as e1:
                # Second attempt: Try reading as ASCII
                self.logprint(f"FITS reading failed: {e1}. Trying ASCII format...")
                
                try:
                    stars = Table.read(truthfile, format='ascii')
                    self.logprint(f"Successfully read truth catalog in ASCII format with {len(stars)} entries")
                
                except Exception as e2:
                    # Third attempt: Fall back to Gaia query
                    self.logprint(f"ASCII reading failed: {e2}. Falling back to Gaia query...")
                    
                    try:
                        stars = utils.gaia_query(cluster_name=self.target_name)
                        self.logprint(f"Successfully queried Gaia with {len(stars)} entries")
                        self.gaia_query_happened = True
                    
                    except Exception as e3:
                        # All attempts failed - log and re-raise the error
                        self.logprint(f"All catalog reading methods failed! Gaia query error: {e3}")
                        raise RuntimeError(f"Could not read truth catalog in any format and Gaia query failed")
            # match sscat against truth star catalog; 0.72" = 5 SuperBIT pixels

            star_matcher = eu.htm.Matcher(16,
                                ra=stars[star_config['truth_ra_key']],
                                dec=stars[star_config['truth_dec_key']]
                                )
            matches, starmatches, dist = \
                                star_matcher.match(ra=ss['ALPHAWIN_J2000'],
                                dec=ss['DELTAWIN_J2000'],
                                radius=1/3600., maxmatch=1
                                )

            og_len = len(ss); ss = ss[matches]
            if self.gaia_query_happened:
                wg_stars = (ss['SNR_WIN'] > star_config['MIN_SNR']) & (ss['MAG_AUTO'] > 16)
            else:
                wg_stars = (ss['SNR_WIN'] > star_config['MIN_SNR'])

            self.logprint(f'{len(dist)}/{og_len} objects ' +
                          'matched to reference (truth) star catalog \n' +
                          f'{len(ss[wg_stars])} stars passed MIN_SNR threshold'
                          )

        else:
            retry_attempts = 3  # Number of attempts to make
            
            for attempt in range(retry_attempts):
                try:
                    self.logprint(f"Attempting Gaia query (attempt {attempt+1}/{retry_attempts})...")
                    stars = utils.gaia_query(cluster_name=self.target_name)
                    
                    # match sscat against truth star catalog; 0.72" = 5 SuperBIT pixels
                    self.logprint(f"Selecting stars using Gaia Query with {len(stars)} stars")

                    star_matcher = eu.htm.Matcher(16,
                                        ra=stars[star_config['truth_ra_key']],
                                        dec=stars[star_config['truth_dec_key']]
                                        )
                    matches, starmatches, dist = \
                                        star_matcher.match(ra=ss['ALPHAWIN_J2000'],
                                        dec=ss['DELTAWIN_J2000'],
                                        radius=1/3600., maxmatch=1
                                        )

                    og_len = len(ss); ss = ss[matches]
                    wg_stars = (ss['SNR_WIN'] > star_config['MIN_SNR']) & (ss['MAG_AUTO'] > star_config['MAX_MAG'])

                    self.logprint(f'{len(dist)}/{og_len} objects ' +
                                'matched to reference (truth) star catalog \n' +
                                f'{len(ss[wg_stars])} stars passed MIN_SNR threshold'
                                )
                    self.gaia_query_happened = True
                    break  # Exit the retry loop if successful
                    
                except Exception as e:
                    self.logprint(f"Gaia Query attempt {attempt+1} failed: {e}")
                    if attempt < retry_attempts - 1:
                        self.logprint("Retrying Gaia query...")
                        time.sleep(3)  # Add a short delay before retrying
                    else:
                        self.logprint("All Gaia query attempts failed.")
            
            # If all Gaia query attempts failed, fall back to standard stellar locus matching
            if not self.gaia_query_happened:
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

        ss_fits_union = fits.HDUList([hdu.copy() for hdu in ss_fits])
        ss_fits_union[ext].data = ss
        union_outname = sscat.replace('_cat.fits','_starcat_union.fits')
        ss_fits_union.writeto(union_outname, overwrite=True)

        return outname

    def make_image_info_struct(self, max_len_of_filepath=200, use_coadd=False):
        # max_len_of_filepath may cause issues down the line if the file path
        # is particularly long

        image_files = []; weight_files = []
        bmask_files = []
        
        coadd_image  = self.detect_img_file.replace('.fits','.sub.fits')
        coadd_weight = self.detect_img_file.replace('.fits', '.weight.fits') 
        coadd_segmap = self.detect_img_file.replace('.fits', '.sgm.fits') 
        coadd_bmask = self.detect_img_file.replace('.fits', '.bmask.fits') 

        with fits.open(coadd_image) as img:
            img_data = img[0].data  
            header = img[0].header  # Preserve the header

            # Create an array of zeros with the same shape as the image
            bmask_data = np.zeros_like(img_data, dtype=np.float32)

            # Save the bmask file
            fits.PrimaryHDU(bmask_data, header=header).writeto(coadd_bmask, overwrite=True)
            print(f'Binary mask file saved to {coadd_bmask}')

        
        for img in self.image_files:
            bkgsub_name = img.replace('.fits','.sub.fits')
            weight_name = img.replace('.fits', '.weight.fits')
            bmask_name = img.replace('.fits', '.bmask.fits')
            image_files.append(bkgsub_name)
            weight_files.append(weight_name)
            bmask_files.append(bmask_name)

        if use_coadd == True:
            image_files.insert(0, coadd_image)
            weight_files.insert(0, coadd_weight)
            bmask_files.insert(0, coadd_bmask)

        # If used, will be put first
        Nim = len(image_files)
        image_info = meds.util.get_image_info_struct(Nim, max_len_of_filepath)

        for i in range(Nim):
            image_info[i]['image_path']  =  image_files[i]
            image_info[i]['image_ext']   =  0
            image_info[i]['weight_path'] =  weight_files[i]
            image_info[i]['weight_ext']  =  0
            image_info[i]['bmask_path']  =  bmask_files[i]
            image_info[i]['bmask_ext']   =  0
            image_info[i]['seg_path']    =  coadd_segmap # Use coadd segmap for uberseg!
            image_info[i]['seg_ext']     =  0

            # The default is for 0 offset between the internal numpy arrays
            # and the images, but we use the FITS standard of a (1,1) origin.
            # In principle we could probably set this automatically by checking
            # the images
            image_info[i]['position_offset'] = 1

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
                            min_size = 16, max_size= 256):
        '''
        Calculate the cutout size for this survey.

        :angular_size: angular size of a source, with some kind of angular units.
        :size_multiplier: Amount to multiply angular size by to choose boxsize.
        :deconvolved:
        :min_size:
        :max_size:
        '''

        pixel_scale = utils.get_pixel_scale(self.detect_img_file)
        box_size_float = np.ceil(angular_size * size_multiplier /pixel_scale)

        # Available box sizes to choose from -> 16 to 256 in increments of 2
        available_sizes = min_size * 2**(np.arange(np.ceil(np.log2(max_size)-np.log2(min_size)+1)).astype(int))

        def get_box_size(val):
            larger = available_sizes[available_sizes > val]
            return np.min(larger) if larger.size > 0 else np.max(available_sizes)

        if isinstance(box_size_float, np.ndarray):
            return np.array([get_box_size(val) for val in box_size_float])
        else:
            return get_box_size(box_size_float)

    def make_object_info_struct(self, catalog=None):
        if catalog is None:
            catalog = self.detection_cat

        obj_str = meds.util.get_meds_input_struct(catalog.size, \
                  extra_fields = [('KRON_RADIUS', float), ('FLUX_RADIUS', float), ('apt_radius', float), \
                  ('number', int), ('XWIN_IMAGE', float), \
                  ('YWIN_IMAGE', float)]
                  )
        obj_str['id'] = catalog['NUMBER']
        obj_str['number'] = np.arange(catalog.size)+1
        obj_str['box_size'] = self._calculate_box_size(catalog['KRON_RADIUS'] * catalog['A_IMAGE'] * utils.get_pixel_scale(self.detect_img_file))
        obj_str['ra'] = catalog['ALPHAWIN_J2000']
        obj_str['dec'] = catalog['DELTAWIN_J2000']
        obj_str['XWIN_IMAGE'] = catalog['XWIN_IMAGE']
        obj_str['YWIN_IMAGE'] = catalog['YWIN_IMAGE']
        obj_str['KRON_RADIUS'] = catalog['KRON_RADIUS']
        obj_str['FLUX_RADIUS'] = catalog['FLUX_RADIUS']
        obj_str['apt_radius'] = catalog['KRON_RADIUS'] * catalog['A_IMAGE'] * utils.get_pixel_scale(self.detect_img_file)

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
