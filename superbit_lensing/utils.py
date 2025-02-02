import logging
import os
import sys
import yaml
import re
from astropy.table import Table
from numpy.random import SeedSequence, default_rng
import time
import numpy as np
import subprocess
from astropy.io import fits
import astropy.wcs as wcs
from esutil import htm
import pdb
import ipdb

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

BASE_DIR = get_base_dir()
MODULE_DIR = get_module_dir()
TEST_DIR = get_test_dir()
