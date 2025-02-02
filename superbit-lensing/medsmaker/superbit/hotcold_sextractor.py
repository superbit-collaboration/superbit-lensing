import os
import numpy as np
import yaml
import astropy.io.fits as fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon, box
from rtree import index

import superbit-lensing.utils as utils

class HotColdSExtractor:

    def __init__(self, image_files, hc_config, band, target_name, 
                 data_dir, config_dir, log=None, vb=False):
        # Load the YAML configuration file
        with open(hc_config, 'r') as file:
            config = yaml.safe_load(file)

        # Load in config arguments
        self.modes = config['modes']
        self.buffer_radius = config['buffer_radius']
        self.n_neighbors = config['n_neighbors']

        self.data_dir = data_dir
        self.config_dir = config_dir
        self.target_name = target_name
        self.image_files = image_files
        self.band = band

        # Set up logger
        if log is None:
            logfile = 'hotcold.log'
            log = utils.setup_logger(logfile)

        self.logprint = utils.LogPrint(log, vb)

        # Initialize merged catalog and exclusion zone
        self.merged_data = []
        self.exclusion_zones = []

    def run(self, imagefile, catdir, back_type='AUTO'):
        '''
        Function that calls the SExtractor command building function based on the selected mode.
        '''
        cold_cat, hot_cat, default_cat = None, None, None

        # Run 'cold' (bright-mode) Source Extractor
        if 'cold' in self.modes:
            cold_cat = self._run_sextractor(self.config_dir, imagefile, self.catdir, "cold", self.data_dir, back_type)
            self.logprint(f"Cold mode catalog complete: {cold_cat}\n")

        # Run 'hot' (faint-mode) Source Extractor
        if 'hot' in self.modes:
            hot_cat = self._run_sextractor(self.config_dir, imagefile, self.catdir, "hot", self.data_dir, back_type)
            self.logprint(f"Hot mode catalog complete: {hot_cat}\n")

        # Run 'default' (one-mode) Source Extractor
        if 'default' in self.modes:
            default_cat = self._run_sextractor(self.config_dir, imagefile, self.catdir, "default", self.data_dir, back_type)
            self.logprint(f"Default mode catalog complete: {default_cat}\n")

        # If only default mode is selected, return immediately
        if 'default' in self.modes and 'cold' not in self.modes and 'hot' not in self.modes:
            return

        # Check if the catalogs exist before merging
        if os.path.exists(cold_cat) and os.path.exists(hot_cat):

            # Define output catalog names and catdirs
            base_name = os.path.basename(imagefile)
            outname = base_name.replace('.fits', '_cat.fits')  

            # Merge the catalogs
            self.logprint(f"Merging catalogs {cold_cat} and {hot_cat}")
            self._merge_catalogs(cold_cat, hot_cat, self.buffer_radius, self.n_neighbors, outname)
            self.logprint(f"Merged catalog complete: {outname}")

            # Delete the hot and cold catalogs after merging
            os.remove(hot_cat)
            os.remove(cold_cat)
            self.logprint(f"Deleted catalogs: {hot_cat} and {cold_cat}\n")
        else:
            raise FileNotFoundError(f"One or both of the catalogs {cold_cat}, {hot_cat} do not exist, merged catalog NOT created.")


    def _run_sextractor(self, sextractor_config_path, image_file, catdir, mode, datadir, back_type='AUTO'):
        '''
        Runs source extractor using os.system(cmd) on the given image file in the given mode
        '''
        self.logprint("Processing " + f'{image_file}' + " in mode " + f'{mode}')

        # Define config file path
        cpath = sextractor_config_path

        # Define output catalog name/path
        cat_dir = os.path.join(catdir)
        base_name = os.path.basename(image_file)

        # Different output catalog nomenclature for coadd versus single exposure
        if mode in ['cold', 'hot']:
            # For 'cold' or 'hot' modes, append '_cat_mode' to the base name
            cat_name = base_name.replace('.fits', '_cat_' + f'{mode}' + '.fits')
        elif mode == 'default':
            # For the default mode, append only '_cat' to the base name
            cat_name = base_name.replace('.fits', '_cat.fits')


        cat_file = os.path.join(cat_dir, cat_name)

        # Define image file, check if it exists
        image = os.path.join(datadir, image_file)
        if not os.path.exists(image):
            raise FileNotFoundError(f"The file '{image}' does not exist.")

        # Construct the SExtractor command
        cmd = self._construct_sextractor_cmd(image, cat_file, cpath, mode, back_type)
        self.logprint("sex cmd is " + cmd)

        # Print command to terminal and run
        os.system(cmd)

        self.logprint(f'cat_name is {cat_file}')
        return cat_file

    def make_exposure_catalogs(self):
        '''
        Wrapper script that runs through the list of single exposures,
        runs SExtractor on each, and returns a list of the catalogs.
        '''

        exposure_catalogs = []

        self.catdir = os.path.join(self.data_dir, self.target_name, self.band, "cat")

        for imagefile in self.image_files:
            sexcat = self.run(imagefile, self.catdir)
            exposure_catalogs.append(sexcat)

    def make_coadd_catalog(self, use_band_coadd=False):
        '''
        Wrapper script that runs SExtractor on the coadd image,
        and returns the catalog.

        Added functionality on whether to use on single band or det.
        '''
        if use_band_coadd == True:
            self.coadd_file = os.path.join(self.data_dir, self.target_name, self.band, "coadd", f"{self.target_name}_coadd_{self.band}.fits")
            self.catdir = os.path.join(self.data_dir, self.target_name, self.band, "coadd")
            self.run(self.coadd_file, self.catdir, back_type='MANUAL')
        else:
            self.coadd_file = os.path.join(self.data_dir, self.target_name, "det", "coadd", f"{self.target_name}_coadd_det.fits")
            self.catdir = os.path.join(self.data_dir, self.target_name, "det", "cat")
            self.run(self.coadd_file, self.catdir, back_type='MANUAL')

    def make_dual_image_catalogs(self, detection_bandpass):
        '''
        Wrapper script that runs SExtractor on two coadd images in both 'hot' and 'cold' modes,
        and then merges the resulting catalogs.
        '''

        band1_coadd_file = os.path.join(self.data_dir, self.target_name, detection_bandpass, "coadd", f"{self.target_name}_coadd_{detection_bandpass}.fits")
        band2_coadd_file = os.path.join(self.data_dir, self.target_name, self.band, "coadd", f"{self.target_name}_coadd_{self.band}.fits")

        self.catdir = os.path.join(self.data_dir, self.target_name, "det", "cat")
        hot_cat = self._run_sextractor_dual_mode(band1_coadd_file, band2_coadd_file, detection_bandpass, self.catdir, mode="hot")
        cold_cat = self._run_sextractor_dual_mode(band1_coadd_file, band2_coadd_file, detection_bandpass, self.catdir, mode="cold")

        # Set the output file name for the merged catalog
        outname = f"{self.target_name}_{detection_bandpass}_{self.band}_dual_cat.fits"
        self._merge_catalogs(hot_cat, cold_cat, self.buffer_radius, self.n_neighbors, outname)

    def _run_sextractor_dual_mode(self, image_file1, image_file2, detection_bandpass, catdir, mode):
        '''
        Runs source extractor in dual image mode using os.system(cmd) on the given image files in the given mode
        '''
        self.logprint("Processing dual image mode for " + f'{image_file1}' + " and " + f'{image_file2}')

        # Define output catalog name/path
        cat_dir = os.path.join(catdir)

        cat_name = f"{self.target_name}_{detection_bandpass}_{self.band}_dual_cat_{mode}.fits"
        cat_file = os.path.join(cat_dir, cat_name)

        # Construct the SExtractor command in dual image mode
        cmd = self._construct_sextractor_cmd(image_file1, cat_file, self.config_dir, mode, back_type='MANUAL', dual_image_mode=True, second_image=image_file2)
        self.logprint("sex cmd is " + cmd)

        # Run the command
        os.system(cmd)

        self.logprint(f'Dual image catalog is {cat_file}')
        return cat_file

    def _construct_sextractor_cmd(self, image, cat_file, cpath, mode, back_type='AUTO', dual_image_mode=False, second_image=None):
        '''
        Construct the sextractor command based on the given mode.
        '''
        # Extract the base name of the image (without path and .fits extension)
        image_basename = os.path.basename(image).replace('.fits', '')

        # Define Arguments for SExtractor command
        if dual_image_mode and second_image is not None:
            # Update the image argument to include both images
            image_arg =         f'"{image}[0],{second_image}[0]"'
        else:
            image_arg =         f'"{image}[0]"'

        weight_arg =            f'-WEIGHT_IMAGE "{image}[1]" -WEIGHT_TYPE MAP_WEIGHT'
        name_arg =              f'-CATALOG_NAME {cat_file}'
        param_arg =             f'-PARAMETERS_NAME {os.path.join(cpath, "sextractor.param")}'
        nnw_arg =               f'-STARNNW_NAME {os.path.join(cpath, "default.nnw")}'
        checktype_arg =         f'-CHECKIMAGE_TYPE -BACKGROUND,SEGMENTATION'

        bkg_name =              image.replace('.fits','.sub.fits')
        seg_name =              image.replace('.fits','.sgm.fits')
        bg_sub_arg =            f'-BACK_TYPE {back_type}'
        aper_name_base =        f"{image_basename}_apertures"

        if mode == "hot":
            config_arg =        f'-c {os.path.join(cpath, "sextractor.hot.config")}'
            filter_arg =        f'-FILTER_NAME {os.path.join(cpath, "gauss_2.0_3x3.conv")}'
            aper_name =         os.path.join(self.catdir, f"{aper_name_base}.hot.fits")
        elif mode == "cold":
            config_arg =        f'-c {os.path.join(cpath, "sextractor.cold.config")}'
            filter_arg =        f'-FILTER_NAME {os.path.join(cpath, "gauss_5.0_9x9.conv")}'
            aper_name =         os.path.join(self.catdir, f"{aper_name_base}.cold.fits")
        elif mode == "default":
            config_arg =        f'-c {os.path.join(cpath, "sextractor.real.config")}'
            filter_arg =        f'-FILTER_NAME {os.path.join(cpath, "gauss_2.0_3x3.conv")}'
            aper_name =         os.path.join(self.catdir, f"{aper_name_base}.default.fits")

        checkname_arg =         f'-CHECKIMAGE_NAME {bkg_name},{seg_name}'

        # Make the SExtractor command
        cmd = ' '.join([
            'sex', image_arg, weight_arg, name_arg,  checktype_arg, checkname_arg,
            param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                        ])

        return cmd

    def create_ellipse(self, ra, dec, a, b, theta_deg):
        # Check for NaN or Infinity values in the inputs
        if np.isnan(ra) or np.isnan(dec) or np.isnan(a) or np.isnan(b) or np.isnan(theta_deg):
            self.logprint("Skipped due to NaN values")
            return None
        if np.isinf(ra) or np.isinf(dec) or np.isinf(a) or np.isinf(b) or np.isinf(theta_deg):
            self.logprint("Skipped due to Infinity values")
            return None

        theta_rad = np.radians(180 - theta_deg) # Convert theta from degrees to radians (flip angle due to Polygon using different convention)
        t = np.linspace(0, 2*np.pi, 50)
        ellipse_x = ra + a * np.cos(t) * np.cos(theta_rad) - b * np.sin(t) * np.sin(theta_rad)
        ellipse_y = dec + a * np.cos(t) * np.sin(theta_rad) + b * np.sin(t) * np.cos(theta_rad)
        return Polygon(np.column_stack((ellipse_x, ellipse_y)))


    def get_kron_min_radius(self, fits_file):
        with fits.open(fits_file) as hdul:
            header = hdul[0].header  # the header is in the Primary HDU
            return header['SEXAPEK3'] # Kron minimum radius attribute in header

    def _merge_catalogs(self, cold_cat, hot_cat, buffer_radius, n_neighbors, outname):

        # Load in hot and cold catalogs
        with fits.open(cold_cat) as hdul1:
            cold_data = hdul1[2].data
        with fits.open(hot_cat) as hdul2:
            hot_data = hdul2[2].data

        cold_idx = index.Index()
        buffer_radius = self.buffer_radius
        n_neighbors = self.n_neighbors

        # Get the Kron minimum radius from the cold catalog
        kron_min_radius_cold = 3.0
        kron_min_radius_hot = 3.0

        self.logprint(f'Creating ellipses for {len(cold_data)} cold sources')
        # Loop over each source in the cold catalog
        for i, cold_source in enumerate(cold_data):
            # Convert the cold source parameters to a SkyCoord object
            cold_coord = SkyCoord(cold_source['ALPHAWIN_J2000'], cold_source['DELTAWIN_J2000'], unit='deg')
            
            # Create an ellipse for the cold source
            try:
                ellipse = self.create_ellipse(cold_coord.ra.deg, cold_coord.dec.deg, cold_source['A_WORLD']*kron_min_radius_cold, 
                                cold_source['B_WORLD']*kron_min_radius_cold, cold_source['THETA_IMAGE'])
                # Add a check here to see if the ellipse was created successfully
                if ellipse is None:
                    continue
            except Exception as e:
                self.logprint(f"Exception occurred: {e}")
                self.logprint(f"Cold source parameters: {cold_source}")
                continue

            # Add the ellipse to the exclusion zones and the r-tree index
            self.exclusion_zones.append(ellipse)
            cold_idx.insert(i, ellipse.bounds)

            # Add the cold source to the merged catalog
            self.merged_data.append(cold_source)
        self.logprint(f'Created {len(self.exclusion_zones)} exclusion zones')

        self.logprint(f'Creating ellipses for {len(hot_data)} hot sources')
        # Loop over each source in the hot catalog
        # Initialize a boolean array with the same length as hot_data and set all values to False
        wg = np.full(len(hot_data), False)

        for idx, hot_source in enumerate(hot_data):
            # Convert the hot source parameters to a SkyCoord object
            hot_coord = SkyCoord(hot_source['ALPHAWIN_J2000'], hot_source['DELTAWIN_J2000'], unit='deg')

            # Create an ellipse for the hot source
            try:
                hot_object = self.create_ellipse(hot_coord.ra.deg, hot_coord.dec.deg, hot_source['A_WORLD']*kron_min_radius_hot,
                                                hot_source['B_WORLD']*kron_min_radius_hot, hot_source['THETA_IMAGE'])
                # Add a check here to see if the hot object was created successfully
                if hot_object is None:
                    continue
            except Exception as e:
                self.logprint(f"Exception occurred: {e}")
                self.logprint(f"Hot source parameters: {hot_source}")
                continue

            # Create a buffer around the hot source
            hot_buffer = hot_object.buffer(buffer_radius)
            hot_buffer_bounds = box(*hot_buffer.bounds)

            # Get the list of 5 nearest cold sources
            nearest_sources_ids = list(cold_idx.nearest(hot_buffer_bounds.bounds, n_neighbors))
            
            # Get the actual nearest cold sources
            nearest_sources = [self.exclusion_zones[i] for i in nearest_sources_ids if 0 <= i < len(self.exclusion_zones)]
            
            if not any(source.contains(hot_object) or source.intersects(hot_object) for source in nearest_sources):
                wg[idx] = True

        self.logprint('Hot sources referenced to exclusion zone, merging catalogs')

        # Merge catalogs
        cold_fits_base = fits.open(cold_cat)

        # Add header comment noting that it is a merged catalog
        cold_fits_base[0].header['COMMENT'] = 'Hot/Cold used for this catalog'

        cold_table = Table.read(cold_cat, hdu=2)
        hot_table = Table.read(hot_cat, hdu=2) 

        merged_table = vstack([cold_table, hot_table[wg]])

        cold_fits_base[2].data = merged_table.as_array()

        # Write the merged catalog to a FITS file
        cold_fits_base.writeto(os.path.join(self.catdir, outname), overwrite=True)
