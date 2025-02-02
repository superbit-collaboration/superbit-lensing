from abc import ABC, abstractmethod
import os
import yaml
import logging
import subprocess

import utils
from diagnostics import build_diagnostics

import ipdb

class SuperBITModule(dict):
    '''
    Class for an arbitrary module in the SuperBIT weak lensing pipeline

    These modules are not intended to be run on their own, but as components
    called in the `SuperBITPipeline` class

    The pipeline handles the usual things like logging, verbosity, etc.
    '''

    _req_fields = []
    _opt_fields = []
    _flag_fields = []

    def __init__(self, name, config, set_defaults=False):

        self.name = name

        if not isinstance(config, dict):
            raise TypeError(f'The passed config for module {name} was not a dict!')

        self._config = config
        self._check_config(set_defaults=set_defaults)

        self.diagnostics = build_diagnostics(name, config)

        return

    def _check_config(self, set_defaults=False):
        '''
        Make sure all required elements of the module config are present.
        '''

        # Check that required fields are there
        for field in self._req_fields:
            if field not in self._config.keys():
                raise KeyError(f'Field "{field}" must be present in the {self.name} config!')

        # Make sure there are no surprises
        for field in self._config.keys():
            if (field not in self._req_fields) and \
               (field not in self._opt_fields) and \
               (field not in self._flag_fields):
                raise KeyError(f'{field} is not a valid field for the {self.name} module!')

        # Make sure all fields are at least initialized to None
        if set_defaults is True:
            for field in self._opt_fields:
                if field not in self._config.keys():
                    self._config[field] = None

        # anything else?
        # ...

        return

    def _run_setup(self, logprint):
        logprint(f'Starting module {self.name}')

        # ...

        return

    def _setup_options(self, run_options):
        options = ''
        for opt in self._opt_fields:
            if opt in self._config:
                options += f' -{opt}={self._config[opt]}'

        for flag in self._flag_fields:
            if flag in self._config:
                # since passed in a config, it could possibly
                # be set to False
                if self._config[flag] is True:
                    options += f' --{flag}'

        vb = ' --vb' if run_options['vb'] is True else ''
        options += vb

        return options

    def _run_cleanup(self, logprint):

        logprint(f'Module {self.name} completed succesfully')

        # ...
        return

    @abstractmethod
    def run(self, run_options, logprint):
        pass

    def run_diagnostics(self, run_options, logprint):
        self.diagnostics.run(run_options, logprint)

        return

    def _run_command(self, cmd, logprint):
        '''
        Run bash command
        '''

        logprint(f'\n{cmd}\n')

        # for live prints but no error handling:
        # process = subprocess.Popen(cmd.split(),
        #                            stdout=subprocess.PIPE,
        #                            stderr=subprocess.PIPE,
        #                            bufsize=1)

        # for line in iter(process.stdout.readline, b''):
        #     logprint(line.decode('utf-8').replace('\n', ''))

        # output, error = process.communicate()

        args = [cmd.split()]
        kwargs = {'stdout':subprocess.PIPE,
                  # TODO: check this!
                  # 'stderr':subprocess.PIPE,
                  'stderr':subprocess.STDOUT,
                  'bufsize':1}
        with subprocess.Popen(*args, **kwargs) as process:
            try:
                for line in iter(process.stdout.readline, b''):
                    logprint(utils.decode(line).replace('\n', ''))

                stdout, stderr = process.communicate()

            except:
                process.kill()
                raise
                # stdout, stderr = process.communicate()
                # logprint('\n'+utils.decode(stderr))

                # return 1
                # process.kill()

            rc = process.poll()

            if rc:
                # stdout, stderr = process.communicate()
                err = subprocess.CalledProcessError(rc,
                                                    process.args,
                                                    output=stdout,
                                                    stderr=stderr)

                logprint('\n'+utils.decode(err.stderr))
                raise err

                # return rc

        # try:
        #     process = subprocess.run(cmd.split(),
        #                              stdout=subprocess.PIPE,
        #                              stderr=subprocess.PIPE,
        #                              # stderr=subprocess.PIPE,
        #                              # stderr=subprocess.STDOUT,
        #                              bufsize=1,
        #                              check=True)
        # except subprocess.CalledProcessError as e:
        #     logprint(e.stderr)
        #     return 1

        # output, error = process.communicate()
        # rc = process.returncode

        # if rc != 0:
        #     logprint(f'call returned in error with rc={rc}:')
        #     logprint(output)
        #     logprint(error)
        #     return rc

        # process.wait()

        # now get the return code from the executed process
        # streamdata = process.communicate()[0]
        # res = process.communicate()
        rc = process.returncode

        # process.stdout.close()

        return rc

    def __getitem__(self, key):
        val = self._config.__getitem__(self, key)

        return val

    def __setitem__(self, key, val):

        self._config.__setitem__(self, key, val)

        return

class SuperBITPipeline(SuperBITModule):

    # We abuse the required fields variable a bit here, as
    # the pipeline module is atypical.
    _req_fields = ['run_options']
    _req_run_options_fields = ['run_name', 'order', 'vb']

    # _opt_fields = {get_module_types().keys()}
    _opt_fields = [] # Gets setup in constructor
    _opt_run_options_fields = ['ncores', 'run_diagnostics']

    def __init__(self, config_file, log=None):

        config = utils.read_yaml(config_file)

        self._setup_opt_fields()

        super(SuperBITPipeline, self).__init__('pipeline',
                                               config)
                                               # set_defaults=False)
        self._check_config()

        self.log = log
        self.vb = self._config['run_options']['vb']

        self.logprint = utils.LogPrint(log, self.vb)

        if self._config['run_options']['ncores'] in [None, 'none', 'None']:
            ncores = os.cpu_count()
            self._config['run_options']['ncores'] = ncores
            self.logprint(f'`ncores` was not set; using all available ({ncores})')

        col = 'run_diagnostics'
        if col in self._config['run_options']:
            self.do_diagnostics = self._config['run_options'][col]
        else:
            self.do_diagnostics = False
            self._config['run_options'][col] = False

        module_names = list(self._config['run_options']['order'])
        # module_names.remove('run_options') # Not really a module

        self.modules = []
        for name in module_names:
            self.modules.append(build_module(name,
                                             self._config[name],
                                             self.logprint))

        return

    def _check_config(self, set_defaults=False):
        '''
        Make sure all required elements of the module config are present.
        '''

        # super(SuperBITPipeline, self)._check_config(set_defaults=set_defaults)

        # The pipeline module required fields variable is a bit special
        for field in self._req_run_options_fields:
            if field not in self._config['run_options'].keys():
                raise KeyError(f'Must have an entry for "{field}" in the "run_options" ' + \
                'field for the {self.name} config!')

        if set_defaults is True:
            for field in self._opt_run_options_fields:
                if field not in self._config['run_options'].keys():
                    self._config['run_options'][field] = None

        return

    def _setup_opt_fields(cls):
        cls._opt_fields = MODULE_TYPES.keys()

        return

    def run(self):
        '''
        Run each module in order
        '''

        self.logprint('\nStarting pipeline run\n')
        for module in self.modules:
            rc = module.run(self._config['run_options'], self.logprint)

            if rc !=0:
                self.logprint.warning(f'Exception occured during {module.name}.run()')
                return 1

            if self.do_diagnostics is True:
                module.run_diagnostics(self._config['run_options'], self.logprint)

        self.logprint('\nFinished pipeline run')

        return 0

class GalSimModule(SuperBITModule):
    _req_fields = ['config_file', 'outdir']
    _opt_fields = ['config_dir']
    _flag_fields = ['use_mpi', 'use_srun', 'clobber', 'vb']

    def __init__(self, name, config):
        super(GalSimModule, self).__init__(name, config)

        self.gs_config_path = None
        self.gs_config = None

        return

    def run(self, run_options, logprint):
        '''
        Relevant type checks and param init's have already
        taken place
        '''

        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        self.gs_config_path = os.path.join(self._config['config_dir'],
                                           self._config['config_file'])
        self.gs_config = utils.read_yaml(self.gs_config_path) # Do we need this?

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)
        # rc = utils.run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        galsim_dir = os.path.join(utils.MODULE_DIR, 'galsim')
        galsim_filepath = os.path.join(galsim_dir, 'mock_superBIT_data.py')

        outdir = self._config['outdir']
        base = f'python {galsim_filepath} {self.gs_config_path} -outdir={outdir}'

        # multiple mpi flags breaks this func...
        # options = self._setup_options(run_options)
        options = ''

        if 'run_name' not in self._config:
            run_name = run_options['run_name']
            options += f' -run_name={run_name}'

        if 'clobber' in self._config:
            options += ' --clobber'

        if run_options['vb'] is True:
            options += ' --vb'

        cmd = base + options

        ncores = run_options['ncores']
        if ncores > 1:
            if hasattr(self._config, 'use_mpi'):
                if self._config['use_mpi'] is True:
                    cmd = f'mpiexec -n {ncores} ' + cmd
            if hasattr(self._config, 'use_srun'):
                if self._config['use_srun'] is True:
                    cmd = f'srun -mpi=pmix ' + cmd
            else:
                cmd = cmd + f' -ncores={ncores}'

        return cmd

class GridTestModule(GalSimModule):
    def _setup_run_command(self, run_options):

        galsim_dir = os.path.join(utils.MODULE_DIR, 'galsim')
        filepath = os.path.join(galsim_dir, 'grid_test.py')

        outdir = self._config['outdir']
        base = f'python {filepath} {self.gs_config_path} -outdir={outdir}'

        # multiple mpi flags breaks this func...
        # options = self._setup_options(run_options)
        options = ''

        if 'run_name' not in self._config:
            run_name = run_options['run_name']
            options += f' -run_name={run_name}'

        if 'clobber' in self._config:
            options += ' --clobber'

        if run_options['vb'] is True:
            options += ' --vb'

        cmd = base + options

        ncores = run_options['ncores']
        if ncores > 1:
            if hasattr(self._config, 'use_mpi'):
                if self._config['use_mpi'] is True:
                    cmd = f'mpiexec -n {ncores} ' + cmd
            if hasattr(self._config, 'use_srun'):
                if self._config['use_srun'] is True:
                    cmd = f'srun -mpi=pmix ' + cmd
            else:
                cmd = cmd + f' -ncores={ncores}'

        return cmd

class MedsmakerMockModule(SuperBITModule):
    _req_fields = ['mock_dir', 'outfile']
    _opt_fields = ['fname_base', 'run_name', 'outdir', 'psf_mode', 'psf_seed']
    _flag_fields = ['meds_coadd', 'source_select', 'select_truth_stars',
                    'overwrite', 'vb']

    def __init__(self, name, config):
        super(MedsmakerMockModule, self).__init__(name, config)

        # ...

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        mock_dir = self._config['mock_dir']
        outfile = self._config['outfile']
        outdir = self._config['outdir']

        filepath = os.path.join(utils.get_module_dir(),
                                'medsmaker',
                                'scripts',
                                'process_mocks.py')

        base = f'python {filepath} {mock_dir} {outfile}'

        options = self._setup_options(run_options)

        if 'run_name' not in self._config:
            run_name = run_options['run_name']
            options += f' -run_name={run_name}'

        cmd = base + options

        return cmd

class MedsmakerModule(SuperBITModule):
    _req_fields = ['target_name', 'bands', 'data_dir']
    _opt_fields = ['outdir', 'psf_mode', 'psf_seed', 'star_config_dir']
    _flag_fields = ['meds_coadd', 'select_truth_stars', 'overwrite', 'vb']

    def __init__(self, name, config):
        super(MedsmakerModule, self).__init__(name, config)

        # ...

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        target_name = self._config['target_name']
        bands = self._config['bands']
        data_dir = self._config['data_dir']

        filepath = os.path.join(
            utils.get_module_dir(),
            'medsmaker',
            'scripts',
            'process_2023.py'
            )

        base = f'python {filepath} {target_name} {bands} {data_dir}'

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class MetacalModule(SuperBITModule):
    _req_fields = ['meds_file', 'outfile']
    _opt_fields = ['outdir','start', 'end', 'n', 'seed']
    _flag_fields = ['plot', 'overwrite', 'vb']

    def __init__(self, name, config):
        super(MetacalModule, self).__init__(name, config)

        col = 'outdir'
        if col not in self._config:
            self._config[col] = os.getcwd()

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        run_name = run_options['run_name']
        outdir = self._config['outdir']
        meds_file = self._config['meds_file']
        outfile = self._config['outfile']
        mcal_dir = os.path.join(utils.get_module_dir(),
                                'metacalibration')
        filepath = os.path.join(mcal_dir, 'ngmix_fit_superbit3.py')

        base = f'python {filepath} {meds_file} {outfile}'

        # Set up some default values that require the run config
        col = 'n'
        if col not in self._config:
            self._config['n'] = run_options['ncores']

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class MetacalModuleV2(MetacalModule):
    '''
    Same as MetacalModule, but using ngmix v2.X
    '''

    _req_fields = ['meds_file', 'outfile']
    _opt_fields = ['outdir','start', 'end', 'seed', 'ncores', 'shear', 'ntry']
    _flag_fields = ['overwrite', 'vb']

    def _setup_run_command(self, run_options):

        run_name = run_options['run_name']
        outdir = self._config['outdir']
        meds_file = self._config['meds_file']
        outfile = self._config['outfile']
        mcal_dir = os.path.join(utils.get_module_dir(),
                                'metacalibration')
        filepath = os.path.join(mcal_dir, 'run_mcal.py')

        base = f'python {filepath} {meds_file} {outfile}'

        # Set up some default values that require the run config
        col = 'ncores'
        if col not in self._config:
            self._config[col] = run_options[col]

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class NgmixFitModule(SuperBITModule):
    _req_fields = ['meds_file', 'outfile', 'config']
    _opt_fields = ['outdir', 'start', 'end', 'n', 'clobber', 'vb']

    def __init__(self, name, config):
        super(NgmixFitModule, self).__init__(name, config)

        col = 'outdir'
        if col not in self._config:
            self._config[col] = os.getcwd()

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        outdir = self._config['outdir']

        meds_file = self._config['meds_file']
        outfile = self._config['outfile']
        outfile = os.path.join(outdir, outfile)
        config = self._config['config']

        ngmix_dir = os.path.join(utils.get_module_dir(),
                                'ngmix_fit')
        filepath = os.path.join(ngmix_dir, 'ngmix_fit.py')

        base = f' python {filepath} {meds_file} {outfile} {config}'

        # Setup some options that rquire the run config
        col = 'n'
        if col not in self._config:
            self._config['n'] = run_options['ncores']

        options = self._setup_options(run_options)

        cmd = base + options

        return cmd

class ShearProfileModule(SuperBITModule):
    _req_fields = ['se_file', 'mcal_file', 'outfile']
    _opt_fields = ['outdir', 'run_name', 'truth_file', 'nfw_file', 'Nresample',
                   'rmin', 'rmax', 'nbins', 'nfw_seed']
    _flag_fields = ['overwrite', 'vb']

    def __init__(self, name, config):
        super(ShearProfileModule, self).__init__(name, config)

        col = 'outdir'
        if col not in self._config:
            self._config[col] = os.getcwd()

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        outdir = self._config['outdir']
        se_file = self._config['se_file']
        mcal_file = self._config['mcal_file']
        outfile = self._config['outfile']

        if 'outdir' in self._config:
            outdir = self._config['outdir']
        else:
            outdir = ''
        outfile = os.path.join(outdir, outfile)

        shear_dir = os.path.join(utils.get_module_dir(),
                                 'shear_profiles')
        filepath = os.path.join(shear_dir, 'make_annular_catalog.py')

        base = f'python {filepath} '

        base += f'{se_file} {mcal_file} {outfile} '

        options = self._setup_options(run_options)

        if 'run_name' not in self._config:
            run_name = run_options['run_name']
            options += f' -run_name={run_name}'

        cmd = base + options

        return cmd

class AnalysisModule(SuperBITModule):
    _req_fields = ['basedir']
    _opt_fields = ['shear_cut', 'outdir']
    _flag_fields = ['overwrite', 'vb']

    def __init__(self, name, config):
        super(AnalysisModule, self).__init__(name, config)

        #...

        return

    def run(self, run_options, logprint):
        logprint(f'\nRunning module {self.name}\n')
        logprint(f'config:\n{self._config}')

        cmd = self._setup_run_command(run_options)

        rc = self._run_command(cmd, logprint)

        return rc

    def _setup_run_command(self, run_options):

        basedir = run_options['outdir']

        analysis_dir = os.path.join(utils.MODULE_DIR, 'analysis')
        filepath = os.path.join(
            analysis_dir, 'run_analysis.py'
            )

        base = f'python {filepath} {basedir}'

        options = self._setup_options(run_options)

        if 'overwrite' not in options:
            if 'overwrite' in run_options:
                if run_options['overwrite'] is True:
                    options += ' --overwrite'

        cmd = base + options

        return cmd

def build_module(name, config, logprint):
    name = name.lower()

    if name in MODULE_TYPES.keys():
        # User-defined input construction
        module = MODULE_TYPES[name](name, config)
    else:
        # Attempt generic input construction
        logprint(f'Warning: {name} is not a pre-defined module type.')
        logprint('Attempting generic module construction.')
        logprint('Module is not guaranteed to run succesfully.')

        module = SuperBITModule(name, config)

    return module

def make_test_ngmix_config(config_file='ngmix_test.yaml', outdir=None,
                           run_name=None, clobber=False):
    if outdir is not None:
        filename = os.path.join(outdir, config_file)

    if run_name is None:
        run_name = 'pipe_test'

    if (clobber is True) or (not os.path.exists(filename)):
        with open(filename, 'w') as f:
            CONFIG = {
                'gal': {
                    'model': 'bdf',
                },
                'psf': {
                    'model': 'gauss'
                },
                'priors': {
                    'T_range': [-1., 1.e3],
                    'F_range': [-100., 1.e9],
                    'g_sigma': 0.1,
                    'fracdev_mean': 0.5,
                    'fracdev_sigma': 0.1
                },
                'fit_pars': {
                    'method': 'lm',
                    'lm_pars': {
                        'maxfev':2000,
                        'xtol':5.0e-5,
                        'ftol':5.0e-5
                        }
                },
                'pixel_scale': 0.144, # arcsec / pixel
                'nbands': 1,
                'seed': 172396,
                'run_name': run_name
            }
            yaml.dump(CONFIG, f, default_flow_style=False)

    return filename

def make_test_config(config_file='pipe_test.yaml', outdir=None, overwrite=False,
                     imsim='galsim'):
    '''
    can make imsim = 'grid_test' for other module
    '''
    if outdir is not None:
        filename = os.path.join(outdir, config_file)

    if (overwrite is True) or (not os.path.exists(filename)):
        run_name = 'pipe_test'
        outdir = os.path.join(utils.TEST_DIR, run_name)
        se_file = os.path.join(outdir, f'{run_name}_mock_coadd_cat.ldac')
        meds_file = os.path.join(outdir, f'{run_name}_meds.fits')
        mcal_file = os.path.join(outdir, f'{run_name}_mcal.fits')
        ngmix_test_config = make_test_ngmix_config('ngmix_test.yaml',
                                                   outdir=outdir,
                                                   run_name=run_name)

        # dummy truth
        nfw_file = os.path.join(
            utils.BASE_DIR, 'runs/truth/nfw_truth_files/nfw_cl_m2.2e15_z0.44.fits')

        with open(filename, 'w') as f:
            # Create dummy config file
            CONFIG = {
                'run_options': {
                    'run_name': run_name,
                    'outdir': outdir,
                    'vb': True,
                    'ncores': 8,
                    'run_diagnostics': True,
                    'order': [
                        f'{imsim}',
                        'medsmaker',
                        'metacal',
                        # 'metacal_v2',
                        'shear_profile',
                        # 'ngmix_fit'
                        ]
                    },
                f'{imsim}': {
                    'config_file': 'pipe_test.yaml',
                    # 'config_file': 'superbit_parameters_forecast.yaml',
                    'config_dir': os.path.join(utils.MODULE_DIR,
                                               'galsim',
                                               'config_files'),
                    'outdir': outdir,
                    'clobber': overwrite
                },
                'medsmaker': {
                    'mock_dir': outdir,
                    'outfile': meds_file,
                    'fname_base': run_name,
                    'run_name': run_name,
                    'outdir': outdir,
                    'overwrite': overwrite,
                    'meds_coadd': True,
                    'psf_mode': 'piff'
                },
                'metacal': {
                # 'metacal_v2': {
                    'meds_file': meds_file,
                    'outfile': mcal_file,
                    'outdir': outdir,
                    'end': 250,
                    'overwrite': overwrite
                },
                'ngmix_fit': {
                    'meds_file': meds_file,
                    'outfile': f'{run_name}_ngmix.fits',
                    'config': ngmix_test_config,
                    'outdir': outdir,
                    'end': 100
                },
                'shear_profile': {
                    'se_file': se_file,
                    'mcal_file': mcal_file,
                    'outfile': f'{run_name}_annular.fits',
                    'nfw_file': nfw_file,
                    'outdir': outdir,
                    'run_name': run_name,
                    'Nresample': 1, # to run much faster
                    'overwrite': overwrite,
                },
            }

            yaml.dump(CONFIG, f, default_flow_style=False)

    return filename

def get_module_types():
    return MODULE_TYPES

# NOTE: This is where you must register a new module
MODULE_TYPES = {
    'galsim': GalSimModule,
    'grid_test': GridTestModule,
    'medsmaker': MedsmakerModule,
    'medsmaker_mock': MedsmakerMockModule,
    'metacal': MetacalModule,
    'metacal_v2': MetacalModuleV2,
    'ngmix_fit': NgmixFitModule,
    'shear_profile': ShearProfileModule,
    'analysis': AnalysisModule,
    }
