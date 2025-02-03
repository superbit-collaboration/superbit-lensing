import numpy as np
import os
import time
import shutil
from numpy.random import SeedSequence, default_rng

import utils
from config import make_run_config_from_dict

class JobsManager(object):
    '''
    A class that sets up a series of cluster jobs
    '''

    _req = ['run_name', 'base_dir', 'nfw_dir', 'gs_base_config',
           'mass_bins', 'z_bins']

    # values are defaults if not present
    _opt = {
        'realizations': 1,
        'ncores_per_job': 1,
        'memory_per_job': 32, # GB
        'psf_mode': 'piff',
        'vb': True,
        'run_diagnostics': True,
        'master_seed': None
    }

    def __init__(self, config_file, fresh=False):
        '''
        A class that manages the creation of multiple CluserJob's

        config: str
            Filepath of a configuration dictionary for managing multiple jobs
        fresh: bool
            Set to True to clean out any existing runs in specified
            paths
        '''

        self.config_file = config_file
        self.config = utils.read_yaml(config_file)
        self.fresh = fresh
        self.jobs = None

        self.parse_config()

        return

    def run(self):
        '''
        Create each individual ClusterJob along with the necessary
        directory structure, config files, etc.
        '''

        print('Creating jobs...')
        self.jobs = self.create_jobs()
        print('Setting jobs seeds...')
        self.set_job_seeds()
        print('Creating job configs...')
        self.make_job_configs()

        print('Done!')

        return

    def parse_config(self):
        '''
        Make sure the config satisfies requirements
        '''

        for key in self._req:
            if key not in self.config.keys():
                raise ValueError(f'{key} must be in the run prep config!')

        for key, default in self._opt.items():
            if key not in self.config.keys():
                print(f'{key} not in config; using default of {default}')
                self.config[key] = default

        masses = self.config['mass_bins']
        redshifts = self.config['z_bins']
        realizations = self.config['realizations']

        for vals, name in zip([masses, redshifts], ['mass_bins', 'z_bins']):
            if isinstance(vals, list):
                for v in vals:
                    if not isinstance(v, float):
                        raise TypeError(f'`{name}` entries must be a float!')
            elif isinstance(v, float):
                pass
            else:
                raise TypeError('`{name}` must be a float or list of floats!')

        if isinstance(realizations, list):
            for i in realizations:
                if not isinstance(i, int):
                    raise TypeError('Each element of `realizations` must be an int!')
        else:
            if isinstance(realizations, int):
                # assume this gives the # of realizations
                self.config['realizations'] = [i for i in range(realizations)]
            else:
                raise TypeError('`realizations` must be an int or list!')

        return

    def create_jobs(self):
        '''
        Create all ClusterJob's and required run directories
        given input configuration

        Returns a list of ClusterJob's
        '''

        config = self.config
        fresh = self.fresh

        base_dir = config['base_dir']
        run_name = config['run_name']
        nfw_dir = config['nfw_dir']

        psf_mode = config['psf_mode']

        if os.path.exists(base_dir):
            if fresh is False:
                raise Exception(f'{base_dir} already exists; set `fresh` ' +\
                                'to True to overwrite all files!')
            else:
                print(f'{base_dir} already exists; cleaning dir as `fresh==True`')
                shutil.rmtree(base_dir)

                end_dir = base_dir.split('\\')[-1]
                if end_dir != base_dir:
                    print(f'Warning: dir {end_dir} does not match base_name. Is that correct?')
                    print('Continuing for now...')

        utils.make_dir(base_dir)

        masses = config['mass_bins']
        redshifts = config['z_bins']
        realizations = config['realizations']
        ncores = config['ncores_per_job']
        memory = config['memory_per_job']
        run_diagnostics = config['run_diagnostics']
        vb = config['vb']

        jobs = [] # list of ClusterJob's
        jindx = 0
        for m in masses:
            for z in redshifts:

                # We need to have a clean mapping of masses into
                # subdir names
                mantissa, exp = self.fexp(m)
                cl_name = f'cl_m{mantissa}e{exp}_z{z}'

                cluster_dir = os.path.join(base_dir, cl_name)
                utils.make_dir(cluster_dir)

                # Set truth nfw filename
                nfw_fname = f'nfw_{cl_name}.fits'
                nfw_file = os.path.join(nfw_dir, nfw_fname)

                for r in realizations:
                    # make subdirectory
                    real_dir = os.path.join(cluster_dir, f'r{r}')
                    utils.make_dir(real_dir)

                    job_dict = {
                        'run_name': run_name,
                        'base_dir': real_dir,
                        'nfw_file': nfw_file,
                        'cl_name': cl_name,
                        'mass': m,
                        'z': z,
                        'realization': r,
                        'job_index': jindx,
                        'mass_mantissa': mantissa,
                        'mass_exp': exp,
                        'ncores': ncores,
                        'memory': memory,
                        'psf_mode': psf_mode,
                        'run_diagnostics': run_diagnostics,
                        'vb': vb
                    }
                    jobs.append(ClusterJob(job_dict))
                    jindx += 1

        return jobs

    @staticmethod
    def fexp(x, precision=1):
        '''
        Returns the mantissa and exponent of float x in base 10
        for a given precision (i.e. # of digits after decimal)

        Acts like np.fexp, but for decimal numbers
        '''

        exp = int(np.log10(x))
        mantissa = round(x / 10**exp, precision)

        return mantissa, exp

    def set_job_seeds(self):
        '''
        Safely set multiple, uncorrelated seeds across jobs

        See https://numpy.org/doc/stable/reference/random/parallel.html
        for details
        '''

        Njobs = len(self.jobs)

        # Can set a master seed for all jobs if you want predictable
        # seeds throughout, say for validation testing
        master_seed = self.config['master_seed']
        if master_seed is None:
            # Set master seed of *all* job seed generation
            # to be local time in microseconds
            master_seed = int(time.time()*1e6)
        else:
            print(f'WARNING: using master_seed={master_seed}\nfor all ' +\
                  'subsequent job seeds. You probably only want this ' +\
                  'for testing purposes')

        ss = SeedSequence(master_seed)

        child_seeds = ss.spawn(Njobs)
        streams = [default_rng(s) for s in child_seeds]

        for i in range(len(self.jobs)):
            job_seed = int(streams[i].random()*1e16)
            # "master seed" here is for a specific job, not all jobs
            self.jobs[i]['gs_master_seed'] = job_seed

        return

    def make_job_configs(self):
        '''
        Make individual cluster job pipeline configs from job list
        '''

        # common galsim config for all jobs
        gs_base_config = utils.read_yaml(self.config['gs_base_config'])

        for i in range(len(self.jobs)):

            self.jobs[i].generate_job_seeds()

            # Make gs job config from base config
            self.jobs[i].make_gs_config(gs_base_config)

            # Make pipeline run config
            self.jobs[i].make_run_config()

        return

class ClusterJob(object):
    '''
    This class stores all relevant info for a given cluster simulation job

    Right now it is not much more than a dict, but may want to increase
    complexity later
    '''

    _req_params = ['base_dir', 'run_name', 'mass', 'z', 'job_index',
                   'nfw_file', 'ncores', 'memory', 'realization']

    _opt_params = ['gs_master_seed', 'gs_config', 'psf_mode',
                   'run_diagnostics', 'vb']

    def __init__(self, job_config):
        self._parse_job_config(job_config)
        self._config = job_config

        return

    def _parse_job_config(self, job_config):
        for name in self._req_params:
            if name not in job_config:
                raise KeyError(f'The passed job_config does not contain {name}!')

        return

    def generate_job_seeds(self):
        '''
        generate needed seeds given job master seed
        '''

        seed_names = [
            'galobj_seed', 'cluster_seed', 'stars_seed', 'noise_seed',
            'dithering_seed', 'psf_seed', 'mcal_seed', 'nfw_seed'
            ]
        Nseeds = len(seed_names)
        seeds = utils.generate_seeds(
            Nseeds, master_seed=self._config['gs_master_seed']
            )

        self.seeds = dict(zip(seed_names, seeds))

        # some seeds are stored in the gs_config
        self.gs_seeds = {}
        gs_seed_names = [
            'galobj_seed', 'cluster_seed', 'stars_seed', 'noise_seed',
            'dithering_seed'
            ]
        for name in gs_seed_names:
            self.gs_seeds[name] = self.seeds[name]

        # some seeds are separate from the gs_config
        self.misc_seeds = {}
        seed_names = ['psf_seed', 'mcal_seed', 'nfw_seed']
        for name in seed_names:
            self.misc_seeds[name] = self.seeds[name]

        return

    def make_gs_config(self, gs_base_config):
        '''
        Make a job-specific GalSim config file given a base config
        '''

        gs_config = gs_base_config.copy()

        base_dir = self._config['base_dir']
        run_dir = self._config['run_name']
        run_name = self._config['run_name']

        gs_name = f'{run_name}_gs_config.yaml'
        gs_filepath = os.path.join(base_dir, gs_name)
        self._config['gs_config'] = gs_filepath

        # update base GalSim config w/ needed job-specific changes
        updates = {
            'mass': self._config['mass'], # Msol / h
            'nfw_z_halo': self._config['z'],
            'outdir': self._config['base_dir'],
        }

        updates.update(self.gs_seeds)

        # incorporate all updates for specific job
        gs_config.update(updates)

        utils.write_yaml(gs_config, gs_filepath)

        return

    def make_run_config(self):
        '''
        Make pipeline run config for given job
        '''

        config_dict = {}

        # parse output filename
        run_name = self._config['run_name']
        cl_name = self._config['cl_name']
        filename = f'{run_name}_{cl_name}.yaml'

        # Some key names are the same as those used by
        # config.make_run_config()
        same_keys = [
            'run_name', 'gs_config', 'nfw_file', 'ncores', 'vb',
            'run_diagnostics', 'psf_mode'
            ]
        for key in same_keys:
            config_dict[key] = self._config[key]

        # Now for the rest
        # NOTE: config_overwrite and run_overwrite not needed
        # as already handled by run manager w/ --fresh
        config_dict['outfile'] = filename
        config_dict['outdir'] = self._config['base_dir']

        # for seeds not in gs_config
        config_dict['seeds'] = self.misc_seeds

        make_run_config_from_dict(config_dict)

        return

    def __setitem__(self, key, item):
        self._config[key] = item

    def __getitem__(self, key):
        return self._config[key]

    def __repr__(self):
        return repr(self._config)

    def __len__(self):
        return len(self._config)

    def __delitem__(self, key):
        del self._config[key]

    def __contains__(self, item):
        return item in self._config

    def __iter__(self):
        return iter(self._config)
