import os
from glob import glob
import numpy as np
from astropy.io import fits
import copy
import subprocess
import multiprocessing
from tqdm import tqdm
from superbit_lensing.utils import radec_to_xy, extract_vignette

def get_slurm_resources():
    """Detect available SLURM resources, fallback to local CPU count if not in SLURM."""
    ntasks = int(os.environ.get("SLURM_NTASKS", 0))  # SLURM tasks
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))  # Cores per task

    if ntasks > 0 and ncores > 0:
        return ntasks * ncores  # Total available cores in SLURM
    else:
        return multiprocessing.cpu_count()  # Use all cores if not in SLURM

def run_command(cmd, cores):
    """Run a command with allocated cores"""
    full_cmd = f"OMP_NUM_THREADS={cores} {cmd}"
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)

def _run_sextractor_dual(image_file1, image_file2, cat_dir, config_dir, diag_dir=None, back_type='AUTO', mag_zp=28.66794, use_weight=True):
    '''
    Utility method to invoke Source Extractor on supplied detection file
    Returns: file path of catalog
    '''
    if cat_dir is None:
        cat_dir = os.path.dirname(image_file1)    

    if diag_dir is None:
        diag_dir = cat_dir
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    cat_name = os.path.basename(image_file2).replace('.fits','_cat.fits')
    cat_file = os.path.join(cat_dir, cat_name)

    mag_arg = f'-MAG_ZEROPINT {mag_zp}'
    print(f'Using mag zero-point: {mag_zp}')

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
    
    if use_weight:
        weight_arg = f'-WEIGHT_IMAGE "{image_file1}[1], {image_file2}[1]" ' + \
                        '-WEIGHT_TYPE MAP_WEIGHT'
    else:
        weight_arg = '-WEIGHT_TYPE NONE'

    cmd = ' '.join([
                'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg, mag_arg
                ])

    print("sex cmd is " + cmd)
    os.system(cmd)

    print(f'cat_name is {cat_file} \n')
    return cat_file

def _run_sextractor_single(image_file1, cat_dir, config_dir, diag_dir=None, back_type='AUTO', mag_zp=28.66794, use_weight=True):
    '''
    Utility method to invoke Source Extractor on supplied detection file
    Returns: file path of catalog
    '''
    if cat_dir is None:
        cat_dir = os.path.dirname(image_file1)

    os.makedirs(cat_dir, exist_ok=True)
    if diag_dir is None:
        diag_dir = cat_dir

    mag_arg = f'-MAG_ZEROPINT {mag_zp}'
    print(f'Using mag zero-point: {mag_zp}')

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

    if use_weight:
        weight_arg = f'-WEIGHT_IMAGE "{image_file1}[1]" ' + \
                        '-WEIGHT_TYPE MAP_WEIGHT'
    else:
        weight_arg = '-WEIGHT_TYPE NONE'
        
    cmd = ' '.join([
                'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg, mag_arg
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
    def __init__(self, data_dir, cluster_name, config_dir=None, projection="TPV", overwrite_coadds=False):
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
        self.projection = projection
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
        commands_to_run = []

        for band in self.bands:
            command_arr, coadd_file = self.make_coadd_arguments(band,  config_dir=self.config_dir, projection=self.projection)

            if os.path.exists(coadd_file) and is_valid_fits(coadd_file):
                print(f"Valid coadd FITS file exists: {coadd_file}")
            else:
                print(f"Coadd file does not exist or is invalid: {coadd_file}")

                if band == "b":
                    self._make_external_headers(command_arr, band)

                command = ' '.join(command_arr.values())
                print(f"The SWarp Command is {command}")
                commands_to_run.append(command)

            files[band] = coadd_file

        # **Parallel Execution**
        total_cores = get_slurm_resources()
        num_tasks = len(commands_to_run)
        cores_per_task = max(1, total_cores // num_tasks) if num_tasks > 0 else 1

        # Calculate Efficiency
        efficiency = (num_tasks / total_cores) if total_cores > 0 else 0
        print(f"Total Cores: {total_cores}, Number of Tasks: {num_tasks}, Cores per Task: {cores_per_task}")
        print(f"Efficiency: {efficiency * 100:.2f}%")

        if num_tasks > 0:
            with multiprocessing.Pool(processes=num_tasks) as pool:
                pool.starmap(run_command, [(cmd, cores_per_task) for cmd in commands_to_run])
        
            # **Augment coadd images**
            for band, coadd_file in files.items():
                if os.path.exists(coadd_file):
                    self.augment_coadd_image(coadd_file)
                    print(f"Augmented coadd image for band {band}: {coadd_file}")

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
        
    def make_coadd_arguments(self, band, config_dir=None, projection='TPV'):
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
        proj_arg = f'-PROJECTION_TYPE {projection}'
        resamp_arg = f'-RESAMPLE_DIR {coadd_dir}'
        cliplog_arg = f'CLIP_LOGNAME {coadd_dir}'
        outfile_arg = f'-IMAGEOUT_NAME {coadd_file} ' + \
                        f'-WEIGHTOUT_NAME {weight_file} '

        cmd_arr = {'swarp': 'swarp', 
                    'image_arg': image_args,
                    'weight_arg': weight_arg, 
                    'resamp_arg': resamp_arg,
                    'outfile_arg': outfile_arg, 
                    'config_arg': config_arg,
                    'proj_arg': proj_arg
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

class update_vignet():
    def __init__(self, coadd_file, bg_sub_file, bkg_rms_file, cat_data):
        '''
        coadd_file: path to the coadd image
        bg_sub_file: path to the background subtracted image
        detection_cat_file: path to the SExtractor detection catalog
        '''
        self.coadd_file = coadd_file
        self.bg_sub_file = bg_sub_file
        self.bkg_rms_file = bkg_rms_file
        self.cat_data = cat_data
        
        # Load the image data
        print("Loading background subtracted image...")
        with fits.open(self.bg_sub_file) as hdul:
            self.bg_image_data = hdul[0].data
            self.bg_header = hdul[0].header
        
        print("Loading weight data...")
        with fits.open(self.coadd_file) as hdul:
            self.weight_data = hdul[1].data
            self.weight_header = hdul[1].header
        
        print("Loading background RMS data...")
        with fits.open(self.bkg_rms_file) as hdul:
            self.bkg_rms_data = hdul[0].data
            self.bkg_rms_header = hdul[0].header
        
        # Calculate RMS weight map
        print("Calculating RMS weight map...")
        self.rms_weight_map = 1 / (self.bkg_rms_data**2)
        
        # Initialize arrays in cat_data
        print("Initializing arrays...")
        self.initialize_arrays()
        
        # Process vignets in batches
        print("Processing vignets...")
        self.update_vignet_data()
        
        # Clean up large arrays we no longer need
        print("Cleaning up...")
        self.cleanup()

    def initialize_arrays(self):
        """Initialize empty arrays for our vignette data"""
        n_objects = len(self.cat_data)
        
        # Get sample vignette to determine shape
        ra = self.cat_data["ALPHAWIN_J2000"][0]
        dec = self.cat_data["DELTAWIN_J2000"][0]
        x_image, y_image = radec_to_xy(self.bg_header, ra, dec)
        sample_vignet, _ = extract_vignette(self.bg_image_data, self.bg_header, x_image, y_image)
        vignet_shape = sample_vignet.shape
        
        # Initialize arrays with proper shape
        self.cat_data["im_vignett"] = np.zeros((n_objects, *vignet_shape), dtype=np.float32)
        self.cat_data["weight_vignett"] = np.zeros((n_objects, *vignet_shape), dtype=np.float32)
        self.cat_data["rms_weight_vignett"] = np.zeros((n_objects, *vignet_shape), dtype=np.float32)
        self.cat_data["mask"] = np.zeros((n_objects, *vignet_shape), dtype=np.int8)  # Use smaller dtype
        self.cat_data["is_at_edge"] = np.zeros(n_objects, dtype=bool)  # Use boolean type

    def update_vignet_data(self):
        '''Update the vignet data directly, without accumulating lists'''
        batch_size = 500  # Process 500 objects at a time
        total_objects = len(self.cat_data)
        
        for batch_start in range(0, total_objects, batch_size):
            batch_end = min(batch_start + batch_size, total_objects)
            print(f"Processing batch {batch_start//batch_size + 1}/{(total_objects + batch_size - 1)//batch_size}...")
            
            for i in tqdm(range(batch_start, batch_end)):
                ra = self.cat_data["ALPHAWIN_J2000"][i]
                dec = self.cat_data["DELTAWIN_J2000"][i]
                x_image, y_image = radec_to_xy(self.bg_header, ra, dec)
                
                # Extract vignettes
                im_vignett, meta_bg = extract_vignette(self.bg_image_data, self.bg_header, x_image, y_image)
                weight_vignett, meta_wg = extract_vignette(self.weight_data, self.weight_header, x_image, y_image)
                rms_wt_vignett, meta_rms_wt = extract_vignette(self.rms_weight_map, self.bkg_rms_header, x_image, y_image)
                
                # Create mask directly
                raw_vignet = self.cat_data["VIGNET"][i]
                mask = np.ones_like(raw_vignet, dtype=np.int8)  # Use smaller dtype
                mask[raw_vignet < -1e29] = 0
                
                # Store directly in the arrays
                self.cat_data["im_vignett"][i] = im_vignett
                self.cat_data["weight_vignett"][i] = weight_vignett
                self.cat_data["rms_weight_vignett"][i] = rms_wt_vignett
                self.cat_data["mask"][i] = mask
                self.cat_data["is_at_edge"][i] = meta_bg["is_at_edge"]
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
            print(f"Memory usage after batch: {self.get_memory_usage()} MB")
    
    def get_memory_usage(self):
        """Return current memory usage in MB"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def cleanup(self):
        """Clean up large arrays we don't need anymore"""
        self.bg_image_data = None
        self.weight_data = None
        self.bkg_rms_data = None
        self.rms_weight_map = None
        import gc
        gc.collect()    