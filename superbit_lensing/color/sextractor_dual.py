import os
from glob import glob
import numpy as np
from astropy.io import fits
import copy

def _run_sextractor_dual(image_file1, image_file2, cat_dir, config_dir, diag_dir=None, back_type='AUTO'):
    '''
    Utility method to invoke Source Extractor on supplied detection file
    Returns: file path of catalog
    '''
    if diag_dir is None:
        diag_dir = cat_dir
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    cat_name = os.path.basename(image_file2).replace('.fits','_cat.fits')
    cat_file = os.path.join(cat_dir, cat_name)

    image_arg  = f'"{image_file1}[0], {image_file2}[0]"'
    name_arg   = '-CATALOG_NAME ' + cat_file
    config_arg = f'-c {os.path.join(config_dir, "sextractor.real.config")}'
    param_arg  = f'-PARAMETERS_NAME {os.path.join(config_dir, "sextractor.param")}'
    nnw_arg    = f'-STARNNW_NAME {os.path.join(config_dir, "default.nnw")}'
    filter_arg = f'-FILTER_NAME {os.path.join(config_dir, "gauss_2.0_3x3.conv")}'
    bg_sub_arg = f'-BACK_TYPE {back_type}'

    bkg_file   = os.path.basename(image_file2).replace('.fits','.sub.fits')
    bkg_name   = os.path.join(diag_dir, bkg_file) 
    seg_file   = os.path.basename(image_file2).replace('.fits','.sgm.fits')
    seg_name   = os.path.join(diag_dir, seg_file)
    rms_file   = os.path.basename(image_file2).replace('.fits','.bkg_rms.fits')
    rms_name   = os.path.join(diag_dir, rms_file)
    #apt_file   = os.path.basename(image_file2).replace('.fits','.apt.fits')
    #apt_name   = os.path.join(diag_dir, apt_file)
    checkname_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name},{rms_name}'

    weight_arg = f'-WEIGHT_IMAGE "{image_file1}[1], {image_file2}[1]" ' + \
                    '-WEIGHT_TYPE MAP_WEIGHT'

    cmd = ' '.join([
                'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                ])

    print("sex cmd is " + cmd)
    os.system(cmd)

    print(f'cat_name is {cat_file} \n')
    return cat_file

def _run_sextractor_single(image_file1, cat_dir, config_dir, diag_dir=None, back_type='AUTO'):
    '''
    Utility method to invoke Source Extractor on supplied detection file
    Returns: file path of catalog
    '''
    os.makedirs(cat_dir, exist_ok=True)
    if diag_dir is None:
        diag_dir = cat_dir

    cat_name = os.path.basename(image_file1).replace('.fits','_cat.fits')
    cat_file = os.path.join(cat_dir, cat_name)

    image_arg  = f'"{image_file1}[0]"'
    name_arg   = '-CATALOG_NAME ' + cat_file
    config_arg = f'-c {os.path.join(config_dir, "sextractor.real.config")}'
    param_arg  = f'-PARAMETERS_NAME {os.path.join(config_dir, "sextractor.param")}'
    nnw_arg    = f'-STARNNW_NAME {os.path.join(config_dir, "default.nnw")}'
    filter_arg = f'-FILTER_NAME {os.path.join(config_dir, "gauss_2.0_3x3.conv")}'
    bg_sub_arg = f'-BACK_TYPE {back_type}'

    bkg_file   = os.path.basename(image_file1).replace('.fits','.sub.fits')
    bkg_name   = os.path.join(diag_dir, bkg_file) 
    seg_file   = os.path.basename(image_file1).replace('.fits','.sgm.fits')
    seg_name   = os.path.join(diag_dir, seg_file)
    rms_file   = os.path.basename(image_file1).replace('.fits','.bkg_rms.fits')
    rms_name   = os.path.join(diag_dir, rms_file)
    #apt_file   = os.path.basename(image_file1).replace('.fits','.apt.fits')
    #apt_name   = os.path.join(diag_dir, apt_file)
    checkname_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name},{rms_name}'

    weight_arg = f'-WEIGHT_IMAGE "{image_file1}[1]" ' + \
                    '-WEIGHT_TYPE MAP_WEIGHT'

    cmd = ' '.join([
                'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                ])

    print("sex cmd is " + cmd)
    os.system(cmd)

    print(f'cat_name is {cat_file} \n')
    return cat_file

def is_valid_fits(file_path):
    """Check if a file is a valid FITS image."""
    try:
        with fits.open(file_path, memmap=True) as hdul:
            return True  # Successfully opened, so it's a valid FITS file
    except Exception:
        return False  # Either doesn't exist or is corrupted

class make_coadds_for_dualmode():
    def __init__(self, data_dir, cluster_name, config_dir=None, overwrite_coadds=False):
        '''
        data_path: path to the image data not including target name
        coadd: Set to true if the first image file is a coadd image (must be first)
        '''

        self.data_dir = data_dir
        self.cluster_name = cluster_name
        self.exposure_mask_fname = "/work/mccleary_group/superbit/union/masks/mask_dark_55percent_300.npy" 
        self.bands = ["b", "g", "u"]
        self.image_files = {band: self.set_image_files(band) for band in self.bands}
        self.sex_wgt_files = {band: self.set_weight_files(self.image_files[band]) for band in self.bands}
        self.base_path = os.path.join(self.data_dir, self.cluster_name)
        self.dual_mode_dir = f"{self.base_path}/sextractor_dualmode"
        self.base_coadd_dir = f"{self.dual_mode_dir}/coadd"
        if overwrite_coadds:
            os.system(f"rm -rf {self.base_coadd_dir}")
        if config_dir is None:
            config_dir = "../medsmaker/superbit/astro_config"
            config_dir = os.path.abspath(config_dir)
        self.config_dir = config_dir
        self.make_coadd_dirs()
        self.coadd_file_names = self.make_coadds()

    def make_coadds(self):
        files = {}
        for band in self.bands:
            command_arr, coadd_file = self.make_coadd_arguments(band, config_dir=self.config_dir)
            # Check if the file exists and is a valid FITS image
            if os.path.exists(coadd_file) and is_valid_fits(coadd_file):
                print(f"Valid coadd FITS file exists: {coadd_file}")
            else:
                print(f"Coadd file does not exist or is invalid: {coadd_file}")
                if band == "b":  # Only run this for the first band
                    self._make_external_headers(command_arr, band)
                command = ' '.join(command_arr.values())
                print(f"The SWarp Command is {command}")
                os.system(command)
                self.augment_coadd_image(coadd_file)
            files[band] = coadd_file
        return files

    def set_image_files(self, band):
        # Load in the science frames
        endings = ["cal", "clean"]
        science = []

        for ending in endings:
            search_path = os.path.join(self.data_dir, self.cluster_name, band, 'cal', f'*{ending}.fits')
            science.extend(glob(search_path))
        return science

    def set_weight_files(self, image_files):
        '''
        Make inverse-variance weight maps because ngmix needs them and we 
        don't have them for SuperBIT.
        Use the SExtractor BACKGROUND_RMS check-image as a basis
        '''
        
        img_names = image_files
        mask = np.load(self.exposure_mask_fname)

        sex_wgt_files = [img_name.replace('.fits', '.sex_weight.fits') for img_name in img_names]

        for img_name in img_names:
            sex_wgt_file_name = img_name.replace('.fits', '.sex_weight.fits')
            if not os.path.exists(sex_wgt_file_name):
                print(f'Weight map not found for {img_name}, creating one...')
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
                    
                    hdu = fits.PrimaryHDU(weight_map, header=header)
                    hdu.writeto(sex_wgt_file_name, overwrite=True)

                print(f'Weight map saved to {sex_wgt_file_name}')
            else:
                print(f'Weight map found for {img_name}, skipping...')

        return sex_wgt_files
        
    def make_coadd_arguments(self, band, config_dir=None):
        '''
        Runs SWarp on provided (reduced!) image files to make a coadd image
        for SEX and PSFEx detection.
        '''
        # Make output directory for coadd image if it doesn't exist
        coadd_dir = os.path.join(self.base_coadd_dir, band)
        os.makedirs(coadd_dir, exist_ok=True)
        # Get an Astromatic config path
        if config_dir is None:
            # Ensure output directory exists
            config_dir = "../medsmaker/superbit/astro_config"
            config_dir = os.path.abspath(config_dir)

        # Define coadd image & weight file names and paths
        coadd_outname = f'{self.cluster_name}_coadd_{band}.fits'
        coadd_file = os.path.join(coadd_dir, coadd_outname)

        # Same for weights
        weight_outname = coadd_outname.replace('.fits', '.weight.fits')
        weight_file = os.path.join(coadd_dir, weight_outname)

        image_files = self.image_files.get(band)
        sex_wgt_files = self.sex_wgt_files.get(band)
        
        image_args = ' '.join(image_files)
        sex_weight_files = ' '.join(sex_wgt_files)
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
        return cmd_arr, coadd_file

    def augment_coadd_image(self, coadd_im_file):
        '''
        Something of a utility function to add weight and sgm extensions to
        a single-band coadd, should the need arise
        '''

        coadd_wt_file  = coadd_im_file.replace('.fits', '.weight.fits')

        # Now, have to combine the weight and image file into one.
        im = fits.open(coadd_im_file, mode='append'); im[0].name = 'SCI'
        if im.__contains__('WGT') == True:
            print(f"\n Coadd image {coadd_im_file} already contains " +
                          "an extension named 'WGT', skipping...\n")
        else:
            wt = fits.open(coadd_wt_file); wt[0].name = 'WGT'
            im.append(wt[0])

        # Save; use writeto b/c flush and close don't update the SCI extension
        im.writeto(coadd_im_file, overwrite=True)
        im.close()

    def _make_external_headers(self, cmd_arr, detection_band):
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
        
        # First, make the detection bandpass header
        header_only_arg = '-HEADER_ONLY Y'
        head_arr['header_only_arg'] = header_only_arg
        
        header_outfile = ' '.join(['-IMAGEOUT_NAME', header_name])
        
        # Update swarp command list (dict)
        head_arr['outfile_arg'] = header_outfile
        
        swarp_header_cmd = ' '.join(head_arr.values())
        print('swarp header-only cmd is ' + swarp_header_cmd)
        os.system(swarp_header_cmd)
        
        ## Cool, now that's done, create headers for the other bands too.
        all_bands = ['u', 'b', 'g']
        bands_to_do = np.setdiff1d(all_bands, detection_band)
        for band in bands_to_do:
            # Get name
            band_header = header_name.replace(
            f'/{detection_band}/',f'/{band}/').replace(
            f'{detection_band}.head',f'{band}.head'
            )
            
            # Copy detection bandpass header to other band coadd dirs
            cp_cmd = f'cp {header_name} {band_header}'
            print(f'copying {header_name} to {band_header}')
            os.system(cp_cmd)

    def make_coadd_dirs(self):
        '''
        Make coadd directories for each band
        '''
        for band in ['u', 'b', 'g']:
            os.makedirs(os.path.join(self.base_coadd_dir, band), exist_ok=True)