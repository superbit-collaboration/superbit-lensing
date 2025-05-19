import os
import sys
from glob import glob
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import time

from .match import MatchedTruthCatalog
import superbit_lensing.utils as utils

import ipdb

class Diagnostics(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config

        # plotdir is the directory where all pipeline plots are saved
        # plot_outdir is the output plot directory for *this* diagnostic type
        self.plotdir = None
        self.plot_outdir = None

        # outdir is the location of module outputs
        if 'outdir' in config:
            self.outdir = config['outdir']
        else:
            # Will be set using run_options
            self.outdir = None

        return

    def run(self, run_options, logprint):
        logprint(f'Running diagnostics for {self.name}')

        # If outdir wasn't set in init, do it now
        if self.outdir is None:
            try:
                self.outdir = run_options['outdir']
            except KeyError as e:
                logprint('ERROR: Outdir must be set in either module ' +\
                         'config or run_options!')
                raise e

        self._setup_plot_dirs()

        return

    def run_cmd(cmd, logprint):
        logprint(f'cmd = {cmd}')
        os.system(cmd)

        return

    def _setup_plot_dirs(self):
        '''
        Must be run after self.outdir is set
        '''

        assert(hasattr(self, 'outdir'))

        self.plotdir = os.path.join(self.outdir, 'plots')
        self.plot_outdir = os.path.join(self.plotdir, self.name)

        for p in [self.plotdir, self.plot_outdir]:
            utils.make_dir(p)

        return

class TruthDiagnostics(Diagnostics):
    '''
    Some modules have a corresponding truth catalog
    to compare to
    '''

    def __init__(self, name, config):
        super(TruthDiagnostics, self).__init__(name, config)

        self.true = None
        self.matched_cat = None

        self.truth_file = None

        # Used for matching measured catalog to truth
        self.ratag, self.dectag = 'ra', 'dec'

        return

    def run(self, run_options, logprint):
        super(TruthDiagnostics, self).run(run_options, logprint)

        run_name = run_options['run_name']

        self._setup_truth_cat()

        return

    def _setup_truth_cat(self):
        '''
        Some diagnostics will require a truth catalog

        Assumes the truth cat is in self.outdir, which
        must be set beforehand
        '''

        assert(hasattr(self, 'outdir'))

        truth_file = self._get_truth_file()

        self.true = Table.read(truth_file)

        return

    def _setup_matched_cat(self, meas_file):
        true_file = self._get_truth_file()

        self.matched_cat = MatchedTruthCatalog(true_file, meas_file)

        return

    def _get_truth_file(self):
        truth_files = glob(os.path.join(self.outdir, '*truth*.fits'))

        # After update, there should only be one
        N = len(truth_files)
        if N != 1:
            raise Exception(f'There should only be 1 truth table, not {N}!')

        self.truth_file = truth_files[0]

        return self.truth_file

class GalSimDiagnostics(TruthDiagnostics):

    def __init__(self, name, config):
        super(GalSimDiagnostics, self).__init__(name, config)

        # ...

        return

    def run(self, run_options, logprint):

        super(GalSimDiagnostics, self).run(run_options, logprint)

        ## Check consistency of truth tables
        self.plot_compare_truths(run_options, logprint)

        return

    def plot_compare_truths(self, run_options, logprint):
        '''
        NOTE: No longer relevant, we have updated code
        to produce only one truth cat

        That is why this function does not use self.truth
        '''

        # Not obvious to me why there are multiple tables - this here
        # just to prove this.

        logprint('Diagnostic: Comparing truth catalogs...')

        truth_tables = glob(os.path.join(self.outdir, 'truth*.fits'))
        N = len(truth_tables)

        tables = []
        for fname in truth_tables:
            tables.append(Table.read(fname))

        cols = ['ra', 'flux', 'hlr']
        Nrows = len(cols)
        Nbins = 30
        ec = 'k'
        alpha = 0.75

        for i in range(1, Nrows+1):
            plt.subplot(Nrows, 1, i)

            col = cols[i-1]

            k = 1
            for t in tables:
                plt.hist(t[col], bins=Nbins, ec=ec, alpha=alpha, label=f'Truth_{k}')
                k += 1

            plt.xlabel(f'True {col}')
            plt.ylabel('Counts')
            plt.legend()

            if ('flux' in col) or ('hlr' in col):
                plt.yscale('log')

        plt.gcf().set_size_inches(9, 3*Nrows+2)

        outfile = os.path.join(self.plotdir, 'compare_truth_tables.pdf')
        plt.savefig(outfile, bbox_inches='tight')

        return

class MedsmakerDiagnostics(Diagnostics):
    pass

class MetacalDiagnostics(TruthDiagnostics):

    def __init__(self, name, config):
        super(MetacalDiagnostics, self).__init__(name, config)

        return

    def run(self, run_options, logprint):
        super(MetacalDiagnostics, self).run(run_options, logprint)

        outdir = self.config['outdir']
        outfile = os.path.join(outdir, self.config['outfile'])

        self._setup_matched_cat(outfile)

        self.plot_shear_calibration_g1g2(run_options, logprint)

        return

    def plot_shear_calibration_g1g2(self, run_options, logprint):
        outdir = self.config['outdir']
        outfile = os.path.join(outdir, self.config['outfile'])

        shear_file = self.config['outfile']
        true_file = self.truth_file
        run_name = run_options['run_name']
        out_dir = self.plot_outdir
        vb = run_options['vb']

        # TODO: finish! (on new PR)
        # os.

        return

class MetacalV2Diagnostics(MetacalDiagnostics):
    pass

class NgmixFitDiagnostics(TruthDiagnostics):

    def __init__(self, name, config):
        super(NgmixFitDiagnostics, self).__init__(name, config)

        return

    def run(self, run_options, logprint):
        super(NgmixFitDiagnostics, self).run(run_options, logprint)

        outdir = self.config['outdir']
        outfile = os.path.join(outdir, self.config['outfile'])

        # TODO: Fix in the future!
        print('WARNING: NgmixFitDiagnostics cannot create ' +\
              'a matched catalog as (ra,dec) are not currently ' +\
              'in the ngmix catalog. Fix this to continue.')
        return

        # self._setup_matched_cat(outfile)

        # self.compare_to_truth(run_options, logprint)

        # return

    def compare_to_truth(self, run_options, logprint):
        '''
        Plot meas vs. true for a variety of quantities
        '''

        self.plot_pars_compare(run_options, logprint)

        # use matched catalog
        # self.matched_cat ...

        return

    def plot_pars_compare(self, run_options, logprint):
        logprint('Comparing meas vs. true ngmix pars')

        # gal_model = ngmix_config[]
        # true_pars = self.truth['']

        return

class ShearProfileDiagnostics(TruthDiagnostics):
    def run(self, run_options, logprint):

        super(ShearProfileDiagnostics, self).run(run_options, logprint)

        annular_file = os.path.join(self.outdir, self.config['outfile'])
        self._setup_matched_cat(annular_file)

        self._run_shear_calibration(run_options, logprint)

        return

    def _run_shear_calibration(self, run_options, logprint):

        run_name = run_options['run_name']

        shear_script = os.path.join(self.outdir, 'diagnostic_shear_calibration.py')
        annular_file = os.path.join(self.outdir, self.config['outfile'])
        truth_file = self.truth_file
        outdir = self.plot_outdir

        cmd = f'python {shear_script} {annular_file} {truth_file} -outdir={outdir}'

        logprint('Running shear calibration diagnostics\n')
        self.run_command(cmd, logprint)

        return

def get_diagnostics_types():
    return DIAGNOSTICS_TYPES

# NOTE: This is where you must register a new diagnostics type
DIAGNOSTICS_TYPES = {
    'pipeline': Diagnostics,
    'galsim': GalSimDiagnostics,
    'medsmaker': MedsmakerDiagnostics,
    'metacal': MetacalDiagnostics,
    'metacal_v2': MetacalV2Diagnostics,
    'ngmix_fit': NgmixFitDiagnostics,
    'shear_profile': ShearProfileDiagnostics,
}

def build_diagnostics(name, config):
    name = name.lower()

    if name in DIAGNOSTICS_TYPES.keys():
        # User-defined input construction
        diagnostics = DIAGNOSTICS_TYPES[name](name, config)
    else:
        # Attempt generic input construction
        print(f'Warning: {name} is not a defined diagnostics type. ' +\
              'Using the default.')
        diagnostics = Diagnostics(name, config)

    return diagnostics

def combine_simulation_catalogs(base_path, output_file=None, start_sim=0, num_sims=40, verbose=True):
    """
    Combines simulation catalogs by vstacking all files.
    
    Parameters:
    -----------
    base_path : str
        Base path where simulation files are stored, not including sim number
    output_file : str, optional
        Path where the combined mega catalog will be saved
    num_sims : int, optional
        Number of simulation files to combine (default: 40)
    verbose : bool, optional
        Whether to print progress information (default: True)
    
    Returns:
    --------
    mega_catalog : astropy.table.Table
        The combined mega catalog
    """
    start_time = time.time()
    
    if verbose:
        print(f"Starting to combine {num_sims} simulation catalogs...")
    
    all_tables = []
    
    success_count = 0
    # Loop through all simulation files
    for sim_num in range(start_sim, start_sim+num_sims):
        sim_file = f"{base_path}/sim{sim_num}/b/out/sim{sim_num}_b_annular_combined_with_truth.fits"
        
        if verbose:
            print(f"Reading sim {sim_num}: {sim_file}")
        
        try:
            # Read the FITS file
            table = Table.read(sim_file)
            
            # Add a column to identify which simulation this data came from
            table['sim_id'] = sim_num
            
            # Append to our list of tables
            all_tables.append(table)
            
            if verbose:
                print(f"  Added {len(table)} rows from sim{sim_num}")
            success_count += 1

        except Exception as e:
            print(f"ERROR: Failed to read {sim_file}: {e}")
    
    if not all_tables:
        raise ValueError("No tables were successfully read. Cannot create mega catalog.")
    elif success_count < num_sims:
        print(f"WARNING: Only {success_count} out of {num_sims} simulations were successfully read.")

    # Vertical stack all tables
    if verbose:
        print("Stacking all tables...")
    
    mega_catalog = vstack(all_tables, metadata_conflicts='silent')
    
    # Save the combined catalog

    if output_file is not None:
        mega_catalog.write(output_file, format='fits', overwrite=True)
        if verbose:
            print(f"Saving mega catalog with {len(mega_catalog)} total rows to {output_file}")    
    end_time = time.time()
    if verbose:
        print(f"Finished in {end_time - start_time:.2f} seconds")
    
    return mega_catalog