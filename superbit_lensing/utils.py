import logging
import os
import sys
import yaml
import re
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u
import pandas as pd
from numpy.random import SeedSequence, default_rng
import time
import numpy as np
import subprocess
from astropy.io import fits
import astropy.wcs as wcs
from esutil import htm
import pdb
import ipdb
import pyregion
from shapely.geometry import Point, Polygon

# Get the path to the root of the project (2 levels up from utils.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CLUSTERS_CSV = os.path.join(PROJECT_ROOT, 'data', 'SuperBIT_target_galactic_coords.csv')
TARGET_LIST = os.path.join(PROJECT_ROOT, 'data', 'SuperBIT_target_list.csv')
DESI_MASTER_FILE = "/projects/mccleary_group/superbit/desi_data/zall-pix-iron.fits"
desi_table = [
    'TARGETID', 'SURVEY', 'PROGRAM', 'OBJTYPE', 'SPECTYPE', 
    'TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR', 'ZWARN', 'ZCAT_NSPEC', 'ZCAT_PRIMARY'
]

class AttrDict(dict):
    '''
    It can be more convenient to access dict keys with dict.key than
    dict['key'], so cast the input dict into a class!
    '''

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def match_coords(cat1, cat2, ratag1=None, dectag1=None,
                ratag2=None, dectag2=None, radius=0.5):
    '''
    Utility function to match cat1 to cat 2 using celestial coordinates.
    This assumes cat1/cat2 are astropy.Table objects.
    '''
    # Either 'ra/dec' or 'ALPHAWIN_J2000/DELTAWIN_J2000'!
    try:
        if (ratag1 is not None) and (dectag1 is not None):
            cat1_ra = cat1[ratag1]
            cat1_dec =  cat1[dectag1]
        elif 'ra' in cat1.colnames:
            cat1_ra = cat1['ra']
            cat1_dec =  cat1['dec']
        elif 'ALPHAWIN_J2000' in cat1.colnames:
            cat1_ra = cat1['ALPHAWIN_J2000']
            cat1_dec =  cat1['DELTAWIN_J2000']
        else:
            raise KeyError('cat1: no "ra,dec" or ',
                           '"{ALPHA,DELTA}WIN_J2000" columns')
    except:
        raise NameError("\nCouldn't load catalog 1 RA & Dec\n")

    try:
        if (ratag2 is not None) and (dectag2 is not None):
            cat2_ra = cat2[ratag2]
            cat2_dec =  cat2[dectag2]
        elif 'ra' in cat2.colnames:
            cat2_ra = cat2['ra']
            cat2_dec =  cat2['dec']
        elif 'ALPHAWIN_J2000' in cat2.colnames:
            cat2_ra = cat2['ALPHAWIN_J2000']
            cat2_dec =  cat2['DELTAWIN_J2000']
        else:
            raise KeyError('cat2: no "ra,dec" or ',
                           '"{ALPHA,DELTA}WIN_J2000" columns')
    except:
        raise NameError("\nCouldn't load catalog 2 RA & Dec\n")

    cat1_matcher = htm.Matcher(16, ra=cat1_ra, dec=cat1_dec)

    cat2_ind, cat1_ind, dist = cat1_matcher.match(ra=cat2_ra,
                                                  dec=cat2_dec,
                                                  maxmatch=1,
                                                  radius=radius/3600.
                                                  )

    print(f'\n {len(dist)}/{len(cat1)} gals matched to truth \n')

    return cat1[cat1_ind], cat2[cat2_ind]

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

        return

class LogPrint(object):

    def __init__(self, log, vb):
        '''
        Requires a logging obj and verbosity level
        '''

        # Must be either a Logger object or None
        if log is not None:
            if not isinstance(log, logging.Logger):
                raise TypeError('log must be either a Logger ' +\
                                'instance or None!')

        self.log = log
        self.vb = vb

        return

    def __call__(self, msg):
        '''
        treat it like print()
        e.g. lprint = LogPrint(...); lprint('message')
        '''

        if self.log is not None:
            self.log.info(msg)
        if self.vb is True:
            print(msg)

        return

    def debug(self, msg):
        '''
        don't print for a debug
        '''
        self.log.debug(msg)

        return

    def warning(self, msg):
        self.log.warning(msg)
        if self.vb is True:
            print(msg)

        return

class Logger(object):

    def __init__(self, logfile, logdir=None):
        if logdir is None:
            logdir = './'

        self.logfile = os.path.join(logdir, logfile)

        # only works for newer versions of python
        # log = logging.basicConfig(filename=logfile, level=logging.DEBUG)

        # instead:
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        # log.setLevel(logging.ERROR)
        handler = logging.FileHandler(self.logfile, 'w', 'utf-8')
        handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
        log.addHandler(handler)

        self.log = log

        return

    # other useful things?
    # ...

def setup_logger(logfile, logdir=None):
    '''
    Utility function if you just want the log and not the Logger object
    '''

    if logdir is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    logger = Logger(logfile, logdir=logdir)

    return logger.log

def read_yaml(yaml_file):
    '''
    current package has a problem reading scientific notation as
    floats; see
    https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    '''

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(yaml_file, 'r') as stream:
        # return yaml.safe_load(stream) # see above issue
        return yaml.load(stream, Loader=loader)

def write_yaml(yaml_dict, yaml_outfile):
    with open(yaml_outfile, 'w') as yaml_file:
        yaml.dump(yaml_dict, yaml_file, default_flow_style=False)

    return

def generate_seeds(Nseeds, master_seed=None, seed_bounds=(0, 2**32-1)):
    '''
    generate a set of safe, independent seeds given a master seed

    Nseeds: int
        The number of desired independent seeds
    master_seed: int
        A seed that initializes the SeedSequence, if desired
    seed_bounds: tuple of ints
        The min & max values for the seeds to be sampled from
    '''

    if (not isinstance(Nseeds, int)) or (Nseeds < 1):
        raise ValueError('Nseeds must be a positive int!')

    for b in seed_bounds:
        if not isinstance(b, int):
            raise TypeError('seed_bounds must be a tuple of ints!')
        if seed_bounds[0] < 0:
            raise ValueError('seed_bounds must be positive!')
        if seed_bounds[1] < seed_bounds[0]:
            raise ValueError('seed_bounds values must be monotonic!')

    if master_seed is None:
        # local time in microseconds
        master_seed = int(time.time()*1e6)

    ss = SeedSequence(master_seed)
    child_seeds = ss.spawn(Nseeds)
    streams = [default_rng(s) for s in child_seeds]

    seeds = []
    for k in range(Nseeds):
        val = int(streams[k].integers(seed_bounds[0], seed_bounds[1]))
        seeds.append(val)

    return seeds

def check_req_params(config, params, defaults):
    '''
    Ensure that certain required parameters have their values set to
    something either than the default after a configuration file is read.
    This is needed to allow certain params to be set either on the command
    line or config file.

    config: An object that (potentially) has the param values stored as
    attributes
    params: List of required parameter names
    defaults: List of default values of associated params
    '''

    for param, default in zip(params, defaults):
        # Should at least be set by command line arg defaults, but double check:
        if (not hasattr(config, param)) or (getattr(config, param) == default):
            e_msg = f'Must set {param} either on command line or in passed config!'
            raise Exception(e_msg)

    return

def check_req_fields(config, req, name=None):
    for field in req:
        if not field in config:
            raise ValueError(f'{name}config must have field {field}')

    return

def check_fields(config, req, opt, name=None):
    '''
    req: list of required field names
    opt: list of optional field names
    name: name of config type, for extra print info
    '''
    assert isinstance(config, dict)

    if name is None:
        name = ''
    else:
        name = name + ' '

    if req is None:
        req = []
    if opt is None:
        opt = []

    # ensure all req fields are present
    check_req_fields(config, req, name=name)

    # now check for fields not in either
    for field in config:
        if (not field in req) and (not field in opt):
            raise ValueError(f'{field} not a valid field for {name}config!')

    return

def sigma2fwhm(sigma):
    c = np.sqrt(8.*np.log(2))
    return c * sigma

def fwhm2sigma(fwhm):
    c = np.sqrt(8.*np.log(2))
    return fwhm / c

def decode(msg):
    if isinstance(msg, str):
        return msg
    elif isinstance(msg, bytes):
        return msg.decode('utf-8')
    elif msg is None:
        return ''
    else:
        print(f'Warning: message={msg} is not a string or bytes')
        return msg

def run_command(cmd, logprint=None):

    if logprint is None:
        # Just remap to print then
        logprint = print

    args = [cmd.split()]
    kwargs = {'stdout':subprocess.PIPE,
              'stderr':subprocess.STDOUT,
              # 'universal_newlines':True,
              'bufsize':1}

    with subprocess.Popen(*args, **kwargs) as process:
        try:
            # for line in iter(process.stdout.readline, b''):
            for line in iter(process.stdout.readline, b''):
                logprint(decode(line).replace('\n', ''))

            stdout, stderr = process.communicate()

        except:
            logprint('')
            logprint('.....................ERROR....................')
            logprint('')

            logprint('\n'+decode(stderr))
            # try:
            #     logprint('\n'+decode(stderr))
            # except AttributeError:
            #     logprint('\n'+stderr)

            rc = process.poll()
            raise subprocess.CalledProcessError(rc,
                                                process.args,
                                                output=stdout,
                                                stderr=stderr)
            # raise subprocess.CalledProcessError(rc, cmd)

        rc = process.poll()

        # if rc:
        #     stdout, stderr = process.communicate()
        #     logprint('\n'+decode(stderr))
            # return 1

        if rc:
            stdout, stderr = process.communicate()
            logprint('\n'+decode(stderr))
            # raise subprocess.CalledProcessError(rc, cmd)
            raise subprocess.CalledProcessError(rc,
                                                process.args,
                                                output=stdout,
                                                stderr=stderr)

    # rc = popen.wait()

    # rc = process.returncode

    return rc

def ngmix_dict2table(d):
    '''
    convert the result of a ngmix fit to an astropy table
    '''

    # Annoying, but have to do this to make Table from scalars
    for key, val in d.items():
        d[key] = np.array([val])

    return Table(data=d)

def setup_batches(nobjs, ncores):
    '''
    Create list of batch indices for each core
    '''

    batch_len = [nobjs//ncores]*(ncores-1)

    s = int(np.sum(batch_len))
    batch_len.append(nobjs-s)

    batch_indices = []

    start = 0
    for i in range(ncores):
        batch_indices.append(range(start, start + batch_len[i]))
        start += batch_len[i]

    return batch_indices

def get_pixel_scale(image_filename):
    '''
    use astropy.wcs to obtain the pixel scale (a/k/a plate scale)
    for the input image. Returns pixel scale in arcsec/pixels.

    Input:
        image_filename: FITS image for which pixel scale is desired

    Return:
        pix_scale: image pixel scale in arcsec/pixels

    '''

    # Get coadd image header
    hdr = fits.getheader(image_filename)

    # Instantiate astropy.wcs.WCS header
    w=wcs.WCS(hdr)

    # Obtain pixel scale in degrees/pix & convert to arcsec/pix
    cd1_1 = wcs.utils.proj_plane_pixel_scales(w)[0]
    pix_scale = cd1_1 * 3600

    return pix_scale

def make_dir(d):
    '''
    Makes dir if it does not already exist
    '''
    if not os.path.exists(d):
        os.makedirs(d)

def get_base_dir():
    '''
    base dir is parent repo dir
    '''
    module_dir = get_module_dir()
    return os.path.dirname(module_dir)

def get_module_dir():
    return os.path.dirname(__file__)

def get_test_dir():
    base_dir = get_base_dir()
    return os.path.join(base_dir, 'tests')

def extract_ra_dec(data):
    """Try to extract RA and Dec from known column name options."""
    for ra_col, dec_col in [
        ('ra', 'dec'),
        ('ra_mcal', 'dec_mcal'),
        ('ALPHAWIN_J2000', 'DELTAWIN_J2000')
    ]:
        if ra_col in data.colnames and dec_col in data.colnames:
            return data[ra_col], data[dec_col]
    raise KeyError("No suitable RA/Dec columns found in the data.")

def analyze_mcal_fits(file_path, hdu=None, verbose=True, update_header=False):
    """
    Analyze a FITS file containing astronomical data with RA/Dec coordinates.
    
    Parameters:
    -----------
    file_path : str
        Path to the FITS file
    hdu : int, optional
        HDU index to read. If None, will try default and then HDU=2
    verbose : bool, default=True
        Whether to print results
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    import numpy as np
    from astropy.table import Table
    import os.path
    
    # Input validation
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the FITS file
    try:
        if hdu is not None:
            data = Table.read(file_path, format='fits', hdu=hdu)
        else:
            try:
                data = Table.read(file_path, format='fits')
            except Exception:
                data = Table.read(file_path, format='fits', hdu=2)
                if verbose:
                    print("Using HDU=2 for FITS reading")
    except Exception as e:
        raise IOError(f"Failed to read FITS file: {e}")
    
    # Known column names for RA and Dec
    ra_column_names = ['ra', 'ra_mcal', 'ALPHAWIN_J2000', 'RA']
    dec_column_names = ['dec', 'dec_mcal', 'DELTAWIN_J2000', 'DEC']
    
    # Extract RA and Dec
    ra, dec = None, None
    
    # Try to find RA column
    for col_name in ra_column_names:
        if col_name in data.colnames:
            ra = data[col_name]
            ra_col_used = col_name
            break
    
    # Try to find Dec column
    for col_name in dec_column_names:
        if col_name in data.colnames:
            dec = data[col_name]
            dec_col_used = col_name
            break
    
    if ra is None or dec is None:
        available_cols = ", ".join(data.colnames)
        raise KeyError(f"RA/Dec columns not found. Available columns: {available_cols}")
    
    if verbose:
        print(f"Using columns: RA={ra_col_used}, Dec={dec_col_used}")
    
    # Check for NaN values and filter them
    valid_mask = ~(np.isnan(ra) | np.isnan(dec))
    if np.sum(~valid_mask) > 0 and verbose:
        print(f"Filtered out {np.sum(~valid_mask)} rows with NaN values")
    
    ra = ra[valid_mask]
    dec = dec[valid_mask]
    filtered_data = data[valid_mask]
    
    # Compute full min and max boundaries
    ra_min, ra_max = np.min(ra), np.max(ra)
    dec_min, dec_max = np.min(dec), np.max(dec)
    
    # Compute the center of the field of view
    ra_center = (ra_max + ra_min) / 2
    dec_center = (dec_max + dec_min) / 2
    
    # Compute the radius needed to cover the entire field
    # Using Haversine formula for spherical coordinates would be more accurate
    # But this approximation works for small fields
    ra_extent = (ra_max - ra_min) / 2
    dec_extent = (dec_max - dec_min) / 2
    covering_radius = np.sqrt(ra_extent**2 + dec_extent**2)
    
    # Compute middle 50% boundaries
    ra_lower = ra_min + 0.25 * (ra_max - ra_min)
    ra_upper = ra_max - 0.25 * (ra_max - ra_min)
    dec_lower = dec_min + 0.25 * (dec_max - dec_min)
    dec_upper = dec_max - 0.25 * (dec_max - dec_min)
    
    # Filter data within middle 50%
    mask = (ra >= ra_lower) & (ra <= ra_upper) & (dec >= dec_lower) & (dec <= dec_upper)
    central_data = filtered_data[mask]
    
    # Handle spherical coordinates properly for area calculation
    # For small fields, this approximation is reasonable
    # cos(dec_center) factor accounts for RA convergence at the poles
    area_arcmin2 = (ra_upper - ra_lower) * np.cos(np.radians(dec_center)) * 60 * (dec_upper - dec_lower) * 60
    
    # Compute object density
    num_objects = len(central_data)
    density_per_arcmin2 = num_objects / area_arcmin2 if area_arcmin2 > 0 else 0
    # Calculate recommended pixel size for convergence maps
    min_objects_per_pixel = 5
    pixel_size_arcmin = np.sqrt(min_objects_per_pixel / density_per_arcmin2) if density_per_arcmin2 > 0 else np.inf    
    # Prepare results dictionary
    
    # Print results if verbose
    if verbose:
        print(f"Full RA range: {ra_min:.6f}° to {ra_max:.6f}°")
        print(f"Full Dec range: {dec_min:.6f}° to {dec_max:.6f}°")
        print(f"Field center: RA = {ra_center:.6f}°, Dec = {dec_center:.6f}°")
        print(f"Covering circle radius: {covering_radius:.6f}°")
        print(f"Middle 50% RA range: {ra_lower:.6f}° to {ra_upper:.6f}°")
        print(f"Middle 50% Dec range: {dec_lower:.6f}° to {dec_upper:.6f}°")
        print(f"Total number of objects in middle 50%: {num_objects}")
        print(f"Survey area (middle 50%): {area_arcmin2:.2f} arcmin²")
        print(f"Density: {density_per_arcmin2:.2f} objects per arcmin²")
        print(f"Recommended pixel size for convergence map (≥5 objects/pixel): {pixel_size_arcmin:.2f} arcmin")
        print(f"Expected objects per pixel: {density_per_arcmin2 * pixel_size_arcmin * pixel_size_arcmin:.2f}")
    
    # Create results dictionary
    results = {
        "RA_MIN": ra_min, 
        "RA_MAX": ra_max,
        "DEC_MIN": dec_min,
        "DEC_MAX": dec_max,
        "RA_CENTER": ra_center,
        "DEC_CENTER": dec_center,
        "COVER_RAD": covering_radius,
        "RA_50_MIN": ra_lower,
        "RA_50_MAX": ra_upper,
        "DEC_50_MIN": dec_lower,
        "DEC_50_MAX": dec_upper,
        "N_OBJ_50P": num_objects,
        "AREA_AMIN": area_arcmin2,
        "DENS_AMIN": density_per_arcmin2,
        "TOT_OBJS": len(data),
        "recommended_pixel_size_arcmin": pixel_size_arcmin,
    }
    
    # Update the FITS header if requested
    if update_header:
        # Open the FITS file to update its header
        with fits.open(file_path, mode='update') as hdul:
            # Determine which HDU has the data
            data_hdu = 0  # Primary HDU by default
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, 'data') and hdu.data is not None and len(hdu.data) > 0:
                    data_hdu = i
                    break
            
            # Add metadata to header
            header = hdul[data_hdu].header
            header['HISTORY'] = 'Analysis metadata added by analyze_mcal_fits function'
            
            # Add all results to header with comments
            header['RA_MIN'] = (ra_min, 'Minimum RA value in dataset')
            header['RA_MAX'] = (ra_max, 'Maximum RA value in dataset')
            header['DEC_MIN'] = (dec_min, 'Minimum Dec value in dataset')
            header['DEC_MAX'] = (dec_max, 'Maximum Dec value in dataset')
            header['RA_CNTR'] = (ra_center, 'Center RA of field')
            header['DEC_CNTR'] = (dec_center, 'Center Dec of field')
            header['CVR_RAD'] = (covering_radius, 'Covering radius in degrees')
            #header['RA50_MIN'] = (ra_lower, 'Minimum RA of middle 50% region')
            #header['RA50_MAX'] = (ra_upper, 'Maximum RA of middle 50% region')
            #header['DEC50_MIN'] = (dec_lower, 'Minimum Dec of middle 50% region')
            #header['DEC50_MAX'] = (dec_upper, 'Maximum Dec of middle 50% region')
            header['NOBJ_50P'] = (num_objects, 'Number of objects in middle 50% region')
            header['AREA_AM2'] = (area_arcmin2, 'Area in middle 50% region (arcmin^2)')
            header['DENS_AM2'] = (density_per_arcmin2, 'Object density per arcmin^2')
            header['TOT_OBJS'] = (len(data), 'Total number of objects in dataset')
            header['PIX_AMIN'] = (pixel_size_arcmin, 'Recommended pixel size (arcmin)')
            header['OBJ_PIX'] = (density_per_arcmin2 * pixel_size_arcmin * pixel_size_arcmin, 
                                    'Expected objects per pixel')
            # Save changes
            hdul.flush()
            print(f"Updated FITS header in {file_path} with analysis metadata")
    
    return results

def get_sky_footprint_center_radius(data_table, buffer_fraction=0.05):
    """
    Given an astropy Table with RA/Dec in 'ALPHAWIN_J2000' and 'DELTAWIN_J2000',
    returns the center coordinate as the midpoint of the range and a radius 
    (in degrees) that covers all objects.

    Parameters
    ----------
    data_table : astropy.table.Table
        A FITS binary table with 'ALPHAWIN_J2000' and 'DELTAWIN_J2000' columns.
    buffer_fraction : float, optional
        Fractional increase in radius to account for margin (default is 5%).
            
    Returns
    -------
    ra_center : float
        Right Ascension of center (in degrees), calculated as (max + min)/2
    dec_center : float
        Declination of center (in degrees), calculated as (max + min)/2
    radius_deg : float
        Radius in degrees that encloses all objects from the center
    """
    for ra_col, dec_col in [('RA', 'DEC'), ('ra', 'dec'), ('ALPHAWIN_J2000', 'DELTAWIN_J2000')]:
        if ra_col in data_table.columns and dec_col in data_table.columns:
            ra, dec = data_table[ra_col], data_table[dec_col]
            break
    
    # Handle RA wrap-around for midpoint calculation
    ra_wrapped = np.array(ra)
    ra_range = np.max(ra) - np.min(ra)
    
    # If data spans the 0/360 boundary
    if ra_range > 180:
        # Adjust RAs that are > 180 degrees away from the reference point
        mask = ra > 180
        ra_wrapped[mask] -= 360
    
    # Calculate min and max for wrapped RA
    ra_min = np.min(ra_wrapped)
    ra_max = np.max(ra_wrapped)
    
    # Calculate the center as the midpoint of the range
    ra_center_wrapped = (ra_min + ra_max) / 2
    
    # Adjust back to 0-360 range if needed
    if ra_center_wrapped < 0:
        ra_center_wrapped += 360
    
    # Calculate midpoint for declination (no wrap-around needed)
    dec_min = np.min(dec)
    dec_max = np.max(dec)
    dec_center = (dec_min + dec_max) / 2
    
    # Create center SkyCoord
    center = SkyCoord(ra=ra_center_wrapped, dec=dec_center, unit='deg')
    
    # Create SkyCoord for all points
    coords = SkyCoord(ra=ra, dec=dec, unit='deg')
    
    # Calculate separations from center to all points
    separations = coords.separation(center)
    
    # Get the maximum separation (with buffer)
    radius_deg = separations.max().deg * (1 + buffer_fraction)
    
    return center.ra.deg, center.dec.deg, radius_deg

def get_cluster_info(cluster_name):
    cluster_data = pd.read_csv(TARGET_LIST)
    idx = cluster_data['SuperBIT_name'] == cluster_name
    if not idx.any():
        raise ValueError(f"Cluster name '{cluster_name}' not found in {CLUSTERS_CSV}")
    ra_center = cluster_data.loc[idx, 'RA'].values[0]
    dec_center = cluster_data.loc[idx, 'DEC'].values[0]
    redshift = cluster_data.loc[idx, 'redshift'].values[0]

    return float(ra_center), float(dec_center), float(redshift)

def gaia_query(cluster_name=None, rad_deg=0.5, ra_center=None, dec_center=None, catalog_id = "I/355/gaiadr3", pure=True):
    """
    Query Gaia DR3 data around a given cluster name or specified RA/Dec.

    Parameters
    ----------
    cluster_name : str, optional
        Name of the cluster to look up from CSV. Ignored if ra_center and dec_center are provided.
    rad_deg : float
        Search radius in degrees.
    ra_center : float, optional
        RA center in degrees. Overrides cluster_name if provided.
    dec_center : float, optional
        Dec center in degrees. Overrides cluster_name if provided.

    Returns
    -------
    astropy.table.Table
        Gaia catalog table with columns renamed to match internal naming.
    """
    Vizier.ROW_LIMIT = -1
    Vizier.columns = ['**']
    
    if ra_center is not None and dec_center is not None:
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    elif cluster_name is not None:
        cluster_data = pd.read_csv(CLUSTERS_CSV)
        idx = cluster_data['Name'] == cluster_name
        if not idx.any():
            raise ValueError(f"Cluster name '{cluster_name}' not found in {CLUSTERS_CSV}")
        ra_center = cluster_data.loc[idx, 'RA'].values[0]
        dec_center = cluster_data.loc[idx, 'Dec'].values[0]
        print(f"Using cluster coordinates: RA={ra_center}, Dec={dec_center}")
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    else:
        raise ValueError("Either cluster_name or both ra_center and dec_center must be provided.")

    radius = rad_deg * u.deg
    try:
        result = Vizier.query_region(coord, radius=radius, catalog=catalog_id)
        print("The default query was successful")
    except:
        Vizier.VIZIER_SERVER = 'vizier.iucaa.in'
        result = Vizier.query_region(coord, radius=radius, catalog=catalog_id)
        print("The custom query with vizier.iucaa.in was successful")

    if not result:
        raise RuntimeError("Gaia query returned no results.")

    gaia_table = result[0]
    final_table = gaia_table #['RAJ2000', 'DEJ2000']
    final_table.rename_column('RAJ2000', 'ALPHAWIN_J2000')
    final_table.rename_column('DEJ2000', 'DELTAWIN_J2000')

    # Filter out rows with NaNs in RA or Dec
    mask = ~np.isnan(final_table['ALPHAWIN_J2000']) & ~np.isnan(final_table['DELTAWIN_J2000'])
    if not np.all(mask):
        print(f"Warning: {np.count_nonzero(~mask)} rows with NaN coordinates removed.")
    final_table = final_table[mask]

    # Add object class column
    final_table = add_object_class(final_table)

    if pure:
        return final_table[final_table["class"]=='star']

    return final_table

def gaia_query_v2(cluster_name=None, rad_deg=0.5, ra_center=None, dec_center=None, catalog_id = "I/355/gaiadr3", pure=True):
    """
    Query Gaia DR3 directly using TAP service with specific columns needed for classification
    """
    if ra_center is not None and dec_center is not None:
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    elif cluster_name is not None:
        cluster_data = pd.read_csv(CLUSTERS_CSV)
        idx = cluster_data['Name'] == cluster_name
        if not idx.any():
            raise ValueError(f"Cluster name '{cluster_name}' not found in {CLUSTERS_CSV}")
        ra_center = cluster_data.loc[idx, 'RA'].values[0]
        dec_center = cluster_data.loc[idx, 'Dec'].values[0]
        print(f"Using cluster coordinates: RA={ra_center}, Dec={dec_center}")
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    else:
        raise ValueError("Either cluster_name or both ra_center and dec_center must be provided.")
    try:
        print(f"\nUsing Gaia TAP service directly...")
        print(f"Coordinates: RA={ra_center:.6f}, Dec={dec_center:.6f}, Radius={rad_deg}°")

        # Define the columns needed based on add_object_class function
        columns = [
            # Basic identification and position
            'source_id',
            'ra', 
            'ra_error',
            'dec',
            'dec_error',
            
            # Photometry
            'phot_g_mean_mag',
            'phot_g_mean_mag_error',
            'phot_bp_mean_mag',
            'phot_bp_mean_mag_error', 
            'phot_rp_mean_mag',
            'phot_rp_mean_mag_error',
            'phot_bp_rp_excess_factor',  # This is E(BP/RP) in your function
            
            # Astrometry
            'parallax',
            'parallax_error',
            'pmra',
            'pmra_error',
            'pmdec',
            'pmdec_error',
            
            # Classification columns
            'classprob_dsc_combmod_star',      # PSS
            'classprob_dsc_combmod_galaxy',    # PGal
            'classprob_dsc_combmod_quasar',    # PQSO
            'in_qso_candidates',               # QSO flag
            'in_galaxy_candidates',            # Gal flag
            'non_single_star',                 # NSS
            
            # Quality indicators
            'astrometric_excess_noise',
            'astrometric_excess_noise_sig',
            'astrometric_sigma5d_max',        # amax
            'ruwe',                           # RUWE
            
            # Additional useful columns
            'radial_velocity',
            'radial_velocity_error',
            'teff_gspphot',
            'logg_gspphot',
            'mh_gspphot'
        ]

        # Construct ADQL query with column aliases for easier use
        query = f"""
        SELECT TOP 10000
            {', '.join(columns)},
            phot_bp_mean_mag - phot_rp_mean_mag as bp_rp
        FROM gaiadr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_center}, {dec_center}, {rad_deg})
        ) = 1
        """
        
        start = time.time()
        job = Gaia.launch_job(query)
        result = job.get_results()
        
        # Create column aliases to match your add_object_class function
        result.rename_column('classprob_dsc_combmod_star', 'PSS')
        result.rename_column('classprob_dsc_combmod_galaxy', 'PGal')
        result.rename_column('classprob_dsc_combmod_quasar', 'PQSO')
        result.rename_column('in_qso_candidates', 'QSO')
        result.rename_column('in_galaxy_candidates', 'Gal')
        result.rename_column('non_single_star', 'NSS')
        result.rename_column('astrometric_sigma5d_max', 'amax')
        result.rename_column('ruwe', 'RUWE')
        result.rename_column('phot_bp_rp_excess_factor', 'E(BP/RP)')
        result.rename_column('bp_rp', 'BP-RP')
        result.rename_column('ra', 'ALPHAWIN_J2000')
        result.rename_column('dec', 'DELTAWIN_J2000')

        
        elapsed = time.time() - start
        print(f"✓ Success! Retrieved {len(result)} sources via TAP")
        print(f"  Time taken: {elapsed:.2f} seconds")
        print(f"  Columns: {len(result.columns)}")
        
        # Add object class column
        final_table = add_object_class(result)

        if pure:
            return final_table[final_table["class"]=='star']

        return final_table

    except ImportError:
        print("✗ astroquery.gaia not available")
        return None
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)}")
        return None

def ned_query(cluster_name=None, rad_deg=0.5, ra_center=None, dec_center=None):
    """
    Query NED for objects around a given cluster name or RA/Dec.

    Parameters
    ----------
    cluster_name : str, optional
        Name of the cluster to look up from CSV. Ignored if ra_center and dec_center are provided.
    rad_deg : float
        Search radius in degrees.
    ra_center : float, optional
        RA center in degrees. Overrides cluster_name if provided.
    dec_center : float, optional
        Dec center in degrees. Overrides cluster_name if provided.

    Returns
    -------
    astropy.table.Table
        NED query results, filtered to remove rows with NaN RA or Dec.
    """
    if ra_center is not None and dec_center is not None:
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    elif cluster_name is not None:
        cluster_data = pd.read_csv(CLUSTERS_CSV)
        idx = cluster_data['Name'] == cluster_name
        if not idx.any():
            raise ValueError(f"Cluster name '{cluster_name}' not found in {CLUSTERS_CSV}")
        ra_center = cluster_data.loc[idx, 'RA'].values[0]
        dec_center = cluster_data.loc[idx, 'Dec'].values[0]
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    else:
        raise ValueError("Either cluster_name or both ra_center and dec_center must be provided.")

    table = Ned.query_region(coord, radius=rad_deg * u.deg, equinox='J2000.0')

    # Check and drop rows with NaNs in RA or Dec
    if 'Redshift' in table.colnames:
        mask = ~np.isnan(table['Redshift'])
        if not np.all(mask):
            print(f"Warning: {np.count_nonzero(~mask)} rows with NaN coordinates removed.")
        table = table[mask]

    return table

def desi_query(cluster_name=None, rad_deg=0.5, ra_center=None, dec_center=None):
    """
    Query DESI for objects around a given cluster name or RA/Dec.

    Parameters
    ----------
    cluster_name : str, optional
        Name of the cluster to look up from CSV. Ignored if ra_center and dec_center are provided.
    rad_deg : float
        Search radius in degrees.
    ra_center : float, optional
        RA center in degrees. Overrides cluster_name if provided.
    dec_center : float, optional
        Dec center in degrees. Overrides cluster_name if provided.

    Returns
    -------
    astropy.table.Table
        DESI query results.
    """
    if ra_center is not None and dec_center is not None:
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    elif cluster_name is not None:
        cluster_data = pd.read_csv(CLUSTERS_CSV)
        idx = cluster_data['Name'] == cluster_name
        if not idx.any():
            raise ValueError(f"Cluster name '{cluster_name}' not found in {CLUSTERS_CSV}")
        ra_center = cluster_data.loc[idx, 'RA'].values[0]
        dec_center = cluster_data.loc[idx, 'Dec'].values[0]
        coord = SkyCoord(ra=ra_center, dec=dec_center, unit='deg')
    else:
        raise ValueError("Either cluster_name or both ra_center and dec_center must be provided.")    
    
    try:
        col_data = {}
        with fits.open(desi_file, memmap=True) as hdul:
            for col in desi_table:
                col_data[col] = hdul[1].data[col]

        desi = Table(col_data)
        desi = desi[
            (desi['ZCAT_PRIMARY'] == True) &
            (desi['OBJTYPE'] == 'TGT') &
            (desi['ZWARN'] == 0)
        ]

    except Exception as e:
        print(f"Failed to read desi file ({e}), creating an empty catalog.")
        desi = Table(names=desi_table, dtype=['f8'] * len(desi_table))    

    desi_ra = desi['TARGET_RA'].astype(float)
    desi_dec = desi['TARGET_DEC'].astype(float)

    desi_coord = SkyCoord(ra = desi_ra, dec = desi_dec, unit = u.deg)
    distances = coord.separation(desi_coord)
    rad_deg = rad_deg * u.deg 
    mask = distances <= rad_deg

    return desi[mask]

def radec_to_xy(header, ra, dec):
    """
    Convert RA, Dec to pixel coordinates using WCS from image header.
    If TPV projection is specified but no PV polynomials found, fall back to TAN-SIP.
    
    Parameters:
    fits_filename (str): Path to the FITS file
    ra (float): Right Ascension in degrees
    dec (float): Declination in degrees
    
    Returns:
    tuple: (x, y) pixel coordinates
    """
    
    # Extract WCS parameters from header
    crpix1 = header.get('CRPIX1')
    crpix2 = header.get('CRPIX2')
    crval1 = header.get('CRVAL1')
    crval2 = header.get('CRVAL2')
    cd1_1 = header.get('CD1_1')
    cd1_2 = header.get('CD1_2', 0.0)  # Default to 0 if not present
    cd2_1 = header.get('CD2_1', 0.0)  # Default to 0 if not present
    cd2_2 = header.get('CD2_2')
    
    # Check if header specifies TPV projection
    ctype1 = header.get('CTYPE1', '')
    ctype2 = header.get('CTYPE2', '')
    is_tpv = 'TPV' in ctype1 or 'TPV' in ctype2
    
    # Check if PV polynomials exist
    has_pv_terms = False
    for key in header.keys():
        if key.startswith('PV'):
            has_pv_terms = True
            break
    
    # Create WCS object manually
    wcs_manual = WCS(naxis=2)
    wcs_manual.wcs.crpix = [crpix1, crpix2]
    wcs_manual.wcs.cdelt = [cd1_1, cd2_2]  # Using CD matrix diagonals as CDELT
    wcs_manual.wcs.crval = [crval1, crval2]
    
    # Set projection type based on checks
    if is_tpv and has_pv_terms:
        # Use TPV if it's specified and has polynomials
        wcs_manual.wcs.ctype = [ctype1, ctype2]
        #print("Using TPV projection with PV terms")
    elif is_tpv and not has_pv_terms:
        # Fall back to TAN-SIP if TPV is specified but no PV terms
        wcs_manual.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        #print("TPV specified but no PV terms found, falling back to TAN-SIP")
        
        # Check for SIP coefficients (should be present for TAN-SIP)
        has_sip = False
        for key in header.keys():
            if key.startswith('A_') or key.startswith('B_'):
                has_sip = True
                # Copy SIP coefficients if present
                wcs_manual.sip = WCS(header).sip
                break
        
        if not has_sip:
            # If no SIP coefficients, fall back to plain TAN
            wcs_manual.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            #print("No SIP coefficients found, falling back to plain TAN")
    else:
        # Use what's in the header or default to TAN
        if 'CTYPE1' in header and 'CTYPE2' in header:
            wcs_manual.wcs.ctype = [ctype1, ctype2]
            #print(f"Using projection from header: {ctype1}, {ctype2}")
        else:
            wcs_manual.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            #print("No projection specified in header, using TAN")
    
    # Convert coordinates using the manual WCS
    coords = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
    x, y = wcs_manual.world_to_pixel(coords)
        
    return x, y

def extract_vignette(image_data, header, x_image, y_image, size=51):
    """
    Extract a square vignette centered on X_IMAGE, Y_IMAGE from a FITS image
    
    Parameters:
    fits_file_path (str): Path to the FITS file
    x_image (float): X pixel coordinate
    y_image (float): Y pixel coordinate
    size (int): Size of the vignette in pixels (default: 51)
    
    Returns:
    numpy.ndarray: Vignette image data
    dict: Metadata about the extraction
    """
    # Make sure size is odd to have a center pixel
    if size % 2 == 0:
        size += 1
    
    half_size = size // 2
    
    # Get image dimensions
    img_height, img_width = image_data.shape
    
    # Convert to integer pixel coordinates
    x_int = int(round(float(x_image)))
    y_int = int(round(float(y_image)))
    
    # Calculate vignette boundaries
    x_min = max(0, x_int - half_size)
    x_max = min(img_width, x_int + half_size + 1)
    y_min = max(0, y_int - half_size)
    y_max = min(img_height, y_int + half_size + 1)
    
    # Check if the vignette would be at the edge of the image
    is_at_edge = (x_int - half_size < 0 or 
                    x_int + half_size >= img_width or 
                    y_int - half_size < 0 or 
                    y_int + half_size >= img_height)
    
    # Extract the vignette
    vignette = image_data[y_min:y_max, x_min:x_max]
    
    # Create a full-sized vignette with padding if necessary
    full_vignette = np.zeros((size, size), dtype=vignette.dtype)
    
    # Calculate where to place the extracted vignette in the full-sized array
    v_y_min = max(0, half_size - (y_int - y_min))
    v_y_max = v_y_min + (y_max - y_min)
    v_x_min = max(0, half_size - (x_int - x_min))
    v_x_max = v_x_min + (x_max - x_min)
    
    # Place the vignette in the full-sized array
    full_vignette[v_y_min:v_y_max, v_x_min:v_x_max] = vignette
    
    # Get pixel scale from header if available
    try:
        wcs = WCS(header)
        pixel_scale_matrix = wcs.pixel_scale_matrix * 3600  # to arcsec/pixel
        x_scale = abs(pixel_scale_matrix[0, 0])
        y_scale = abs(pixel_scale_matrix[1, 1])
    except:
        x_scale = None
        y_scale = None
    
    # Return vignette and metadata
    metadata = {
        'center_x': float(x_image),
        'center_y': float(y_image),
        'vignette_size': size,
        'image_bounds': (img_width, img_height),
        'extraction_region': (x_min, x_max, y_min, y_max),
        'is_at_edge': is_at_edge,
        'pixel_scale_x': x_scale,
        'pixel_scale_y': y_scale,
        'angular_size_x': size * x_scale if x_scale else None,  # in arcsec
        'angular_size_y': size * y_scale if y_scale else None   # in arcsec
    }
    
    return full_vignette, metadata

def g1g2_to_gt_gc(g1, g2, ra, dec, ra_c, dec_c, resolution = 0.3, key="ra-dec"):
    """
    Convert reduced shear to tangential and cross shear (Eq. 10, 11 in McCleary et al. 2023).
    args:
    - g1, g2: Reduced shear components.
    - ra, dec: Right ascension and declination of the catalogue,i.e. shear_df['ra'], shear_df['dec'].
    - ra_c, dec_c: Right ascension and declination of the cluster-centre.
    
    returns:
    - gt, gc: Tangential and cross shear components.
    - phi: Polar angle in the plane of the sky.
    """ 
    ra_max, ra_min, dec_max, dec_min = np.max(ra), np.min(ra), np.max(dec), np.min(dec)
    aspect_ratio = (ra_max - ra_min) / (dec_max - dec_min)
    pix_ra = int((np.max(ra) - np.min(ra)) / resolution)
    pix_ra = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    pix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max, pix_ra), np.linspace(dec_min, dec_max, pix_dec))
    print(ra_grid.shape, dec_grid.shape, ra_c, dec_c)

    phi = np.arctan2(dec_grid - dec_c, ra_grid - ra_c)
    #print(phi.shape)
    if key == "ra-dec":
        phi = phi[:, ::-1]
    elif key == "x-y":
        phi = phi
    else:
        raise ValueError("Unknown key, must be either 'ra-dec' or 'x-y'")
#    phi = phi[:, ::-1] # flip the phi array to match the shape of g1, g2
    
    # Calculate the tangential and cross components
    gt = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)
    gc = -g1 * np.sin(2 * phi) + g2 * np.cos(2 * phi)

    return gt, gc, phi

def read_ds9_ctr(filename):
    contours = []
    current = []
    coord_system = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() in ('fk5', 'image', 'physical', 'wcs'):
                coord_system = line.lower()
            elif line.lower() == "line":
                if current:
                    contours.append(current)
                    current = []
            else:
                try:
                    parts = list(map(float, line.split()))
                    current.append(parts)
                except ValueError:
                    continue
        if current:
            contours.append(current)

    # Fallback to FK5 if unknown
    if coord_system is None:
        coord_system = 'fk5'
    return contours, coord_system

def build_clean_tan_wcs(header):
    # Extract key WCS parameters
    crpix1 = header.get('CRPIX1')
    crpix2 = header.get('CRPIX2')
    naxis1 = header.get('NAXIS1')
    naxis2 = header.get('NAXIS2')
    crval1 = header.get('CRVAL1')
    crval2 = header.get('CRVAL2')

    cd1_1 = header.get('CD1_1')
    cd1_2 = header.get('CD1_2', 0.0)
    cd2_1 = header.get('CD2_1', 0.0)
    cd2_2 = header.get('CD2_2')

    # Check if it's a TPV header
    ctype1 = header.get('CTYPE1', '')
    ctype2 = header.get('CTYPE2', '')
    is_tpv = 'TPV' in ctype1 or 'TPV' in ctype2

    # Check for any PV distortion terms
    has_pv_terms = any(key.startswith('PV') for key in header.keys())

    # Build WCS manually
    wcs_manual = WCS(naxis=2)
    wcs_manual.wcs.crpix = [crpix1, crpix2]
    wcs_manual.wcs.crval = [crval1, crval2]
    wcs_manual.naxis1 = naxis1
    wcs_manual.naxis2 = naxis2
    wcs_manual.wcs.cd = [[cd1_1, cd1_2], [cd2_1, cd2_2]]
    wcs_manual.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # Force TAN
    wcs_manual.wcs.cunit = ['deg', 'deg']

    return wcs_manual

def separate_catalog_by_regions(reg_file, catalog):
    # Read the region file with pyregion
    print(f"Reading regions from {reg_file}")
    regions = pyregion.open(reg_file)
    print(f"Found {len(regions)} regions")
    
    # Convert regions to shapely polygons
    print("Converting regions to polygons...")
    polygons = []
    for reg in regions:
        if reg.name == "polygon":
            # Extract the polygon vertices
            vertices = reg.coord_list
            points = [(vertices[i], vertices[i+1]) for i in range(0, len(vertices), 2)]
            polygons.append(Polygon(points))
    
    print(f"Converted {len(polygons)} polygons")
    
    
    # Using X_IMAGE and Y_IMAGE columns
    x_col = 'XWIN_IMAGE'
    y_col = 'YWIN_IMAGE'
    print(f"Using {x_col} and {y_col} as pixel coordinate columns")
    
    # Check each object against all polygons
    in_region_mask = np.zeros(len(catalog), dtype=bool)
    
    for i, (x, y) in enumerate(zip(catalog[x_col], catalog[y_col])):
        if i % 1000 == 0:  # Progress indicator
            print(f"Checking object {i}/{len(catalog)}")
        
        # Create a Point for the current object
        point = Point(x, y)
        
        # Check if the point is inside any polygon
        for polygon in polygons:
            if polygon.contains(point):
                in_region_mask[i] = True
                break
    
    # Create the two catalogs
    inside_catalog = catalog[in_region_mask]
    outside_catalog = catalog[~in_region_mask]
    
    return inside_catalog, outside_catalog, in_region_mask

def add_object_class(gaia_table, verbose=False):
    """
    Add a 'class' column to Gaia DR3 Astropy Table.
    Only classifies as 'star' if BOTH conditions are met:
    1. PSS > 0.9995 (high confidence from Gaia classifier)
    2. Passes all extended source tests (no galaxy indicators)
    
    Also counts and prints statistics on objects with PSS > 0.9995 
    that still show extended source characteristics.
    """
    # Create default classification as 'galaxy'
    class_column = ['galaxy'] * len(gaia_table)
    
    # Create counters for statistics
    total_objects = len(gaia_table)
    high_pss_count = 0
    extended_with_high_pss = 0
    extended_reasons = {}
    
    # Process each object
    for i, row in enumerate(gaia_table):
        # First identify quasars
        if row['QSO'] == 1 or (row['PQSO'] is not None and row['PQSO'] > 0.5):
            class_column[i] = 'quasar'
            continue
        
        # Check for high confidence PSS
        high_conf_star = row['PSS'] is not None and row['PSS'] > 0.9995
        if high_conf_star:
            high_pss_count += 1
        
        # Check if any extended source criteria are met
        extended_source = False
        reason = None
        
        # Gaia's own galaxy classification
        if row['Gal'] == 1 or (row['PGal'] is not None and row['PGal'] > 0.5):
            extended_source = True
            reason = "Gal flag or PGal > 0.5"
        
        # amax = Longest semi-major axis of the error ellipsoid 
        elif row['amax'] is not None and row['amax'] > 8.0:
            extended_source = True
            reason = "amax > 8.0"
            
        # E(BP/RP) = BP/RP excess factor
        elif row['E(BP/RP)'] is not None and row['E(BP/RP)'] > 2.0:
            extended_source = True
            reason = "E(BP/RP) > 2.0"
            
        # RUWE = Renormalised Unit Weight Error
        elif row['RUWE'] is not None and row['RUWE'] > 2.0:
            extended_source = True
            reason = "RUWE > 2.0"
            
        # Color + astrometric criteria
        elif (row['BP-RP'] is not None and row['BP-RP'] > 1.5 and 
              row['amax'] is not None and row['amax'] > 4.0):
            extended_source = True
            reason = "BP-RP > 1.5 & amax > 4.0"
        
        # Count high PSS objects that show extended source characteristics
        if high_conf_star and extended_source:
            extended_with_high_pss += 1
            if reason in extended_reasons:
                extended_reasons[reason] += 1
            else:
                extended_reasons[reason] = 1
        
        # Only classify as star if BOTH conditions are met
        if high_conf_star and not extended_source:
            # Check for star system
            if row['NSS'] is not None and row['NSS'] > 0:
                class_column[i] = 'star-system'
            else:
                class_column[i] = 'star'
    
    # Add the column to the table
    gaia_table['class'] = class_column
    
    if verbose:
        # Print statistics
        print(f"Total objects in catalog: {total_objects}")
        print(f"Objects with PSS > 0.9995: {high_pss_count} ({high_pss_count/total_objects*100:.2f}%)")
        print(f"High PSS objects showing extended source characteristics: {extended_with_high_pss} ({extended_with_high_pss/high_pss_count*100:.2f}% of high PSS objects)")
        
        if extended_with_high_pss > 0:
            print("\nReasons for extended source classification in high PSS objects:")
            for reason, count in sorted(extended_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} objects ({count/extended_with_high_pss*100:.2f}%)")
    
    return gaia_table

def add_object_class_v1(gaia_table):
    """
    Add a simple 'class' column to Gaia DR3 Astropy Table with four categories:
    'star', 'star-system', 'galaxy', or 'quasar'
    Default classification is 'star' if no other specific criteria are met.
    """
    # Create a new column with default 'star' classification
    class_column = ['star'] * len(gaia_table)
    
    # Update classification for each object
    for i, row in enumerate(gaia_table):
        # Check for quasar
        if row['QSO'] == 1 or (row['PQSO'] is not None and row['PQSO'] > 0.5):
            class_column[i] = 'quasar'
        # Check for galaxy
        elif row['Gal'] == 1 or (row['PGal'] is not None and row['PGal'] > 0.5):
            class_column[i] = 'galaxy'
        # Check for star system
        elif row['NSS'] is not None and row['NSS'] > 0:
            class_column[i] = 'star-system'
        # Everything else remains as 'star'
    
    # Add the new column to the table
    gaia_table['class'] = class_column
    
    return gaia_table

# If the file doesn't exist yet and you want to create it with your data in HDU=2
def write_to_hdu2(mega_catalog, output_file, overwrite=True):
    # Create a primary HDU (HDU=0) that's empty
    primary_hdu = fits.PrimaryHDU()
    
    # Create an empty HDU=1 (since we want our data in HDU=2)
    empty_hdu = fits.ImageHDU()
    
    # Convert your catalog to an HDU
    if hasattr(mega_catalog, 'to_table'):
        # If it's an Astropy Table object
        table = mega_catalog.to_table()
    else:
        # If it's already a Table
        table = mega_catalog
    
    data_hdu = fits.table_to_hdu(table)
    
    # Create a HDUList with our HDUs
    hdul = fits.HDUList([primary_hdu, empty_hdu, data_hdu])
    
    # Write to file
    hdul.writeto(output_file, overwrite=overwrite)
    print(f"Data successfully written to HDU=2 in {output_file}")

def update_hdu2(mega_catalog, output_file):
    mega_catalog = Table.read(mega_catalog)
    try:
        # Try to open the existing file
        with fits.open(output_file, mode='update') as hdul:
            # Convert mega_catalog to a Table if needed
            if hasattr(mega_catalog, 'to_table'):
                table = mega_catalog.to_table()
            else:
                table = mega_catalog
            
            # Create a new HDU from the table
            new_hdu = fits.table_to_hdu(table)
            
            # Check if HDU=2 already exists
            if len(hdul) > 2:
                # Replace existing HDU=2
                hdul[2] = new_hdu
                print(f"Replaced existing HDU=2 in {output_file}")
            else:
                # If HDU=2 doesn't exist, make sure HDU=1 exists first
                if len(hdul) == 1:
                    hdul.append(fits.ImageHDU())
                
                # Then append the new HDU to make it HDU=2
                hdul.append(new_hdu)
                print(f"Added data as HDU=2 in {output_file}")
            
            # Save changes
            hdul.flush()
    
    except FileNotFoundError:
        # If file doesn't exist, create it with HDU=2
        write_to_hdu2(mega_catalog, output_file)

BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
