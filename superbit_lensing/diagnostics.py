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

def compute_metacal_quantities(mcal, qual_cuts, mcal_shear, shape_noise=0.14, cluster_redshift=None):
    """
    Compute metacalibration quantities for weak lensing analysis.
    
    Parameters
    ----------
    mcal : astropy.table.Table or similar
        The metacal catalog containing galaxy measurements
    qual_cuts : dict
        Dictionary containing quality cuts with keys:
        - 'min_Tpsf': minimum T/Tpsf ratio
        - 'max_sn': maximum signal-to-noise
        - 'min_sn': minimum signal-to-noise  
        - 'min_T': minimum size T
        - 'max_T': maximum size T
        - 'admom_flag' (optional): required admom flag value
        - 'min_admom_sigma' (optional): minimum admom sigma value
    mcal_shear : float
        The metacal shear value (typically 0.01)
    shape_noise : float
        The shape noise value
    cluster_redshift : float, optional
        Cluster redshift for background selection
        
    Returns
    -------
    selected : astropy.table.Table
        The selected catalog with computed metacal quantities added as columns
    """
    
    # Extract individual selection cuts
    min_Tpsf = qual_cuts['min_Tpsf']
    max_sn = qual_cuts['max_sn']
    min_sn = qual_cuts['min_sn']
    min_T = qual_cuts['min_T']
    max_T = qual_cuts['max_T']
    
    # Extract optional admom cuts
    admom_flag = qual_cuts.get('admom_flag', None)
    min_admom_sigma = qual_cuts.get('min_admom_sigma', None)

    max_gpsf = qual_cuts.get('max_gpsf', None)

    if cluster_redshift is not None:
        # Add in a little bit of a safety margin
        min_redshift = float(cluster_redshift) + 0.025
    else:
        min_redshift = 0

    cut_msg = (
        f"#\n# cuts applied: Tpsf_ratio > {min_Tpsf:.2f}"
        + f"\n# SN > {min_sn:.1f} T > {min_T:.2f}"
        + f"\n# redshift = {min_redshift:.3f}"
    )
    if admom_flag is not None:
        cut_msg += f"\n# admom_flag = {admom_flag}"
    if min_admom_sigma is not None:
        cut_msg += f"\n# admom_sigma > {min_admom_sigma:.2f}"

    if max_gpsf is not None:
        cut_msg +=f"\n# g_psf < {max_gpsf}"
    cut_msg += " \n#\n"
    print(cut_msg)

    # Apply selection cuts for different shear configurations
    # Base selection for noshear
    noshear_mask = (
        (mcal['T_noshear'] >= min_Tpsf * mcal['Tpsf_noshear'])
        & (mcal['T_noshear'] < max_T)
        & (mcal['T_noshear'] >= min_T)
        & (mcal['s2n_noshear'] > min_sn)
        & (mcal['s2n_noshear'] < max_sn)
        # Note: redshift cut commented out in original for noshear
        & (mcal['redshift'] > min_redshift)
    )
    
    # Add optional admom cuts for noshear (only applied here since admom columns don't have suffixes)
    if admom_flag is not None:
        noshear_mask &= (mcal['admom_flag'] == admom_flag)
    if min_admom_sigma is not None:
        noshear_mask &= (mcal['admom_sigma'] > min_admom_sigma)
    if max_gpsf is not None:
        noshear_mask &= (mcal["gpsf_noshear"][:, 1]>-max_gpsf) &(mcal["gpsf_noshear"][:, 1]<max_gpsf) & (mcal["gpsf_noshear"][:, 0]>-max_gpsf) &(mcal["gpsf_noshear"][:, 0]<max_gpsf)
    
    noshear_selection = mcal[noshear_mask]

    # Selection for 1p shear
    selection_1p = mcal[
        (mcal['T_1p'] >= min_Tpsf * mcal['Tpsf_1p'])
        & (mcal['T_1p'] <= max_T)
        & (mcal['T_1p'] >= min_T)
        & (mcal['s2n_1p'] > min_sn)
        & (mcal['s2n_1p'] < max_sn)
        & (mcal['redshift'] > min_redshift)
    ]

    # Selection for 1m shear
    selection_1m = mcal[
        (mcal['T_1m'] >= min_Tpsf * mcal['Tpsf_1m'])
        & (mcal['T_1m'] <= max_T)
        & (mcal['T_1m'] >= min_T)
        & (mcal['s2n_1m'] > min_sn)
        & (mcal['s2n_1m'] < max_sn)
        & (mcal['redshift'] > min_redshift)
    ]

    # Selection for 2p shear
    selection_2p = mcal[
        (mcal['T_2p'] >= min_Tpsf * mcal['Tpsf_2p'])
        & (mcal['T_2p'] <= max_T)
        & (mcal['T_2p'] >= min_T)
        & (mcal['s2n_2p'] > min_sn)
        & (mcal['s2n_2p'] < max_sn)
        & (mcal['redshift'] > min_redshift)
    ]

    # Selection for 2m shear
    selection_2m = mcal[
        (mcal['T_2m'] >= min_Tpsf * mcal['Tpsf_2m'])
        & (mcal['T_2m'] <= max_T)
        & (mcal['T_2m'] >= min_T)
        & (mcal['s2n_2m'] > min_sn)
        & (mcal['s2n_2m'] < max_sn)
        & (mcal['redshift'] > min_redshift)
    ]

    # Compute response matrix components for gamma (shear response)
    r11_gamma = (np.mean(noshear_selection['g_1p'][:, 0]) -
                 np.mean(noshear_selection['g_1m'][:, 0])) / (2. * mcal_shear)
    r22_gamma = (np.mean(noshear_selection['g_2p'][:, 1]) -
                 np.mean(noshear_selection['g_2m'][:, 1])) / (2. * mcal_shear)
    r12_gamma = (np.mean(noshear_selection['g_2p'][:, 0]) -
                 np.mean(noshear_selection['g_2m'][:, 0])) / (2. * mcal_shear)
    r21_gamma = (np.mean(noshear_selection['g_1p'][:, 1]) -
                 np.mean(noshear_selection['g_1m'][:, 1])) / (2. * mcal_shear)

    # Compute response matrix components for selection bias
    r11_S = (np.mean(selection_1p['g_noshear'][:, 0]) -
             np.mean(selection_1m['g_noshear'][:, 0])) / (2. * mcal_shear)
    r22_S = (np.mean(selection_2p['g_noshear'][:, 1]) -
             np.mean(selection_2m['g_noshear'][:, 1])) / (2. * mcal_shear)
    r12_S = (np.mean(selection_2p['g_noshear'][:, 0]) -
             np.mean(selection_2m['g_noshear'][:, 0])) / (2. * mcal_shear)
    r21_S = (np.mean(selection_1p['g_noshear'][:, 1]) -
             np.mean(selection_1m['g_noshear'][:, 1])) / (2. * mcal_shear)

    def mean_err(x):
        return np.std(x, ddof=1) / np.sqrt(len(x))

    err_r11_S = np.sqrt(
        mean_err(selection_1p['g_noshear'][:,0])**2 +
        mean_err(selection_1m['g_noshear'][:,0])**2
    ) / (2*mcal_shear)

    err_r22_S = np.sqrt(
        mean_err(selection_2p['g_noshear'][:,1])**2 +
        mean_err(selection_2m['g_noshear'][:,1])**2
    ) / (2*mcal_shear)

    err_r12_S = np.sqrt(
        mean_err(selection_2p['g_noshear'][:,0])**2 +
        mean_err(selection_2m['g_noshear'][:,0])**2
    ) / (2*mcal_shear)

    err_r21_S = np.sqrt(
        mean_err(selection_1p['g_noshear'][:,1])**2 +
        mean_err(selection_1m['g_noshear'][:,1])**2
    ) / (2*mcal_shear)

    # Compute PSF and gamma corrections
    c1_psf = np.mean((noshear_selection['g_1p_psf'][:, 0] + noshear_selection['g_1m_psf'][:, 0])/2 - 
                     noshear_selection['g_noshear'][:, 0])
    c2_psf = np.mean((noshear_selection['g_2p_psf'][:, 1] + noshear_selection['g_2m_psf'][:, 1])/2 - 
                     noshear_selection['g_noshear'][:, 1])
    c1_gamma = np.mean((noshear_selection['g_1p'][:, 0] + noshear_selection['g_1m'][:, 0])/2 - 
                       noshear_selection['g_noshear'][:, 0])
    c2_gamma = np.mean((noshear_selection['g_2p'][:, 1] + noshear_selection['g_2m'][:, 1])/2 - 
                       noshear_selection['g_noshear'][:, 1])

    # Construct response matrices
    R_gamma = np.array([
        [r11_gamma, r12_gamma],
        [r21_gamma, r22_gamma]
    ])

    R_S = np.array([
        [r11_S, r12_S],
        [r21_S, r22_S]
    ])

    err_R_S = np.array([
        [err_r11_S, err_r12_S],
        [err_r21_S, err_r22_S]
    ])


    # Compute the final response matrix
    R = R_gamma + R_S
    R_inv = np.linalg.inv(R)

    # Correction vectors
    c_psf = np.array([c1_psf, c2_psf])
    c_gamma = np.array([c1_gamma, c2_gamma])
    c_total = c_psf + c_gamma

    # Print diagnostics
    print("Gamma Response Matrix (R_gamma):")
    print(R_gamma)
    print("\nSelection Bias Response Matrix (R_S):")
    print(R_S)
    print("\n Selection response error bar: ")
    print(err_R_S)
    print("\nPSF Correction Vector (c_psf):")
    print(c_psf)
    print("\nGamma Correction Vector (c_gamma):")
    print(c_gamma)
    print(f'\n{len(noshear_selection)} objects passed selection criteria')
    print(f'shape noise is {shape_noise}')

    # Create a copy of the selected catalog to avoid modifying the original
    selected = noshear_selection.copy()

    # Get covariance and compute weights
    g_cov_noshear = selected['g_cov_noshear']
    
    # Transform the covariance matrix
    corrected_cov = np.einsum('ij,njk,lk->nil', R_inv, g_cov_noshear, R_inv)
    
    tot_covar = shape_noise + corrected_cov[:, 0, 0] + corrected_cov[:, 1, 1]
    weight = 1. / tot_covar

    # Compute per-object response values
    try:
        r11 = (selected['g_1p'][:, 0] - selected['g_1m'][:, 0]) / (2. * mcal_shear)
        r12 = (selected['g_2p'][:, 0] - selected['g_2m'][:, 0]) / (2. * mcal_shear)
        r21 = (selected['g_1p'][:, 1] - selected['g_1m'][:, 1]) / (2. * mcal_shear)
        r22 = (selected['g_2p'][:, 1] - selected['g_2m'][:, 1]) / (2. * mcal_shear)
        c1_psf_obj = ((selected['g_1p_psf'][:, 0] + selected['g_1m_psf'][:, 0])/2 - 
                      selected['g_noshear'][:, 0])
        c2_psf_obj = ((selected['g_2p_psf'][:, 1] + selected['g_2m_psf'][:, 1])/2 - 
                      selected['g_noshear'][:, 1])
        c1_gamma_obj = ((selected['g_1p'][:, 0] + selected['g_1m'][:, 0])/2 - 
                        selected['g_noshear'][:, 0])
        c2_gamma_obj = ((selected['g_2p'][:, 1] + selected['g_2m'][:, 1])/2 - 
                        selected['g_noshear'][:, 1])

        # Add per-object quantities to table
        selected.add_columns(
            [r11, r12, r21, r22, c1_gamma_obj, c2_gamma_obj, c1_psf_obj, c2_psf_obj],
            names=['r11', 'r12', 'r21', 'r22', 'c1', 'c2', 'c1_psf', 'c2_psf']
        )

    except ValueError as e:
        print('WARNING: mcal r{ij} value-added cols not added; ' +
              'already present in catalog')

    # Compute per-object metacal corrected shears
    try:
        R_obj = np.array([[r11, r12], [r21, r22]])
        g1_MC = np.zeros_like(r11)
        g2_MC = np.zeros_like(r22)

        N = len(g1_MC)
        for k in range(N):
            Rinv = np.linalg.inv(R_obj[:, :, k])
            gMC = np.dot(Rinv, selected[k]['g_noshear'])
            g1_MC[k] = gMC[0]
            g2_MC[k] = gMC[1]

        selected.add_columns(
            [g1_MC, g2_MC],
            names=['g1_MC', 'g2_MC']
        )

    except ValueError as e:
        print('WARNING: mcal g{1/2}_MC value-added cols not added; ' +
              'already present in catalog')

    # Apply bias corrections and response matrix
    g_biased = selected['g_noshear'] - c_total  # Shape: (n, 2)
    g_corrected = np.einsum('ij,nj->ni', R_inv, g_biased)  # Shape: (n, 2)

    # Add final corrected quantities
    selected['g1_Rinv'] = g_corrected[:, 0]
    selected['g2_Rinv'] = g_corrected[:, 1]
    selected['g_cov_Rinv'] = corrected_cov
    selected['R11_S'] = r11_S
    selected['R22_S'] = r22_S
    selected['weight'] = weight

    return selected


def compute_R_S(
    mcal,
    qual_cuts,
    mcal_shear,
    cluster_redshift=None,
    overwrite_calibration=True,
    R_diagonal=True,
):
    """
    Compute metacalibration response matrices and apply shear calibration.

    Applies quality cuts across all five metacal shear variants (noshear,
    1p, 1m, 2p, 2m), computes the shear response matrix R_gamma, the
    selection bias response matrix R_S, and returns the selected catalog
    with response-corrected ellipticities.

    Parameters
    ----------
    mcal : astropy.table.Table or similar
        The metacal catalog containing galaxy measurements. Expected columns
        include ``T_<suffix>``, ``Tpsf_<suffix>``, ``s2n_<suffix>``,
        ``g_noshear``, ``g_<suffix>``, ``g_cov_noshear``, ``r11``, ``r22``,
        ``r12``, ``r21``, ``redshift``, and optionally ``admom_flag``,
        ``admom_sigma``, ``gpsf_noshear``.
    qual_cuts : dict
        Dictionary containing quality cuts with keys:

        - ``min_Tpsf``  : float — minimum T/Tpsf ratio
        - ``max_sn``    : float — maximum signal-to-noise
        - ``min_sn``    : float — minimum signal-to-noise
        - ``min_T``     : float — minimum size T
        - ``max_T``     : float — maximum size T
        - ``admom_flag``      (optional) : int   — required admom flag value
        - ``min_admom_sigma`` (optional) : float — minimum admom sigma
        - ``max_gpsf``        (optional) : float — maximum PSF ellipticity
    mcal_shear : float
        The metacalibration shear step (typically 0.01).
    cluster_redshift : float, optional
        Cluster redshift for background galaxy selection.  A safety margin
        of +0.025 is added internally.
    overwrite_calibration : bool, optional
        If True (default), write ``g1_Rinv`` and ``g2_Rinv`` columns to
        the output catalog.
    R_diagonal : bool, optional
        If True (default), apply only the diagonal elements of the response
        matrix when correcting ellipticities.  If False, apply the full
        2×2 inverse.

    Returns
    -------
    selected : astropy.table.Table
        Copy of the noshear-selected catalog with additional columns:
        ``g1_Rinv``, ``g2_Rinv`` (if *overwrite_calibration*),
        ``g_cov_Rinv``, ``R11_S``, ``R22_S``.
    """

    # ------------------------------------------------------------------ #
    #  Unpack quality cuts
    # ------------------------------------------------------------------ #
    min_Tpsf = qual_cuts['min_Tpsf']
    max_sn = qual_cuts['max_sn']
    min_sn = qual_cuts['min_sn']
    min_T = qual_cuts['min_T']
    max_T = qual_cuts['max_T']

    admom_flag = qual_cuts.get('admom_flag', None)
    min_admom_sigma = qual_cuts.get('min_admom_sigma', None)
    max_gpsf = qual_cuts.get('max_gpsf', None)

    if cluster_redshift is not None:
        min_redshift = float(cluster_redshift) + 0.025
    else:
        min_redshift = 0.0

    # ------------------------------------------------------------------ #
    #  Log the cuts being applied
    # ------------------------------------------------------------------ #
    cut_msg = (
        f"#\n# cuts applied: Tpsf_ratio > {min_Tpsf:.2f}"
        f"\n# SN > {min_sn:.1f}  T > {min_T:.2f}"
        f"\n# redshift > {min_redshift:.3f}"
    )
    if admom_flag is not None:
        cut_msg += f"\n# admom_flag = {admom_flag}"
    if min_admom_sigma is not None:
        cut_msg += f"\n# admom_sigma > {min_admom_sigma:.2f}"
    if max_gpsf is not None:
        cut_msg += f"\n# g_psf < {max_gpsf}"
    cut_msg += "\n#\n"
    print(cut_msg)

    # ------------------------------------------------------------------ #
    #  Helper: build a standard selection mask for a given suffix
    # ------------------------------------------------------------------ #
    def _base_mask(suffix):
        """Return the boolean mask common to all shear variants."""
        return (
            (mcal[f'T_{suffix}'] >= min_Tpsf * mcal[f'Tpsf_{suffix}'])
            & (mcal[f'T_{suffix}'] < max_T)
            & (mcal[f'T_{suffix}'] >= min_T)
            & (mcal[f's2n_{suffix}'] > min_sn)
            & (mcal[f's2n_{suffix}'] < max_sn)
            & (mcal['redshift'] > min_redshift)
        )

    # ------------------------------------------------------------------ #
    #  Noshear selection (with optional admom / gpsf cuts)
    # ------------------------------------------------------------------ #
    noshear_mask = _base_mask('noshear')

    if admom_flag is not None:
        noshear_mask &= (mcal['admom_flag'] == admom_flag)
    if min_admom_sigma is not None:
        noshear_mask &= (mcal['admom_sigma'] > min_admom_sigma)
    if max_gpsf is not None:
        gpsf = mcal['gpsf_noshear']
        noshear_mask &= (
            (gpsf[:, 0] > -max_gpsf) & (gpsf[:, 0] < max_gpsf)
            & (gpsf[:, 1] > -max_gpsf) & (gpsf[:, 1] < max_gpsf)
        )

    noshear_selection = mcal[noshear_mask]

    # ------------------------------------------------------------------ #
    #  Sheared-variant selections (1p, 1m, 2p, 2m)
    # ------------------------------------------------------------------ #
    selection_1p = mcal[_base_mask('1p')]
    selection_1m = mcal[_base_mask('1m')]
    selection_2p = mcal[_base_mask('2p')]
    selection_2m = mcal[_base_mask('2m')]

    # ------------------------------------------------------------------ #
    #  Shear response matrix  R_gamma
    # ------------------------------------------------------------------ #
    r11_gamma = np.mean(noshear_selection['r11'])
    r22_gamma = np.mean(noshear_selection['r22'])
    r12_gamma = np.mean(noshear_selection['r12'])
    r21_gamma = np.mean(noshear_selection['r21'])

    R_gamma = np.array([
        [r11_gamma, r12_gamma],
        [r21_gamma, r22_gamma],
    ])

    # ------------------------------------------------------------------ #
    #  Selection bias response matrix  R_S  (and its uncertainty)
    # ------------------------------------------------------------------ #
    def _mean_err(x):
        """Standard error on the mean."""
        return np.std(x, ddof=1) / np.sqrt(len(x))

    r11_S = (np.mean(selection_1p['g_noshear'][:, 0])
             - np.mean(selection_1m['g_noshear'][:, 0])) / (2.0 * mcal_shear)
    r22_S = (np.mean(selection_2p['g_noshear'][:, 1])
             - np.mean(selection_2m['g_noshear'][:, 1])) / (2.0 * mcal_shear)
    r12_S = (np.mean(selection_2p['g_noshear'][:, 0])
             - np.mean(selection_2m['g_noshear'][:, 0])) / (2.0 * mcal_shear)
    r21_S = (np.mean(selection_1p['g_noshear'][:, 1])
             - np.mean(selection_1m['g_noshear'][:, 1])) / (2.0 * mcal_shear)

    err_r11_S = np.sqrt(
        _mean_err(selection_1p['g_noshear'][:, 0])**2
        + _mean_err(selection_1m['g_noshear'][:, 0])**2
    ) / (2.0 * mcal_shear)
    err_r22_S = np.sqrt(
        _mean_err(selection_2p['g_noshear'][:, 1])**2
        + _mean_err(selection_2m['g_noshear'][:, 1])**2
    ) / (2.0 * mcal_shear)
    err_r12_S = np.sqrt(
        _mean_err(selection_2p['g_noshear'][:, 0])**2
        + _mean_err(selection_2m['g_noshear'][:, 0])**2
    ) / (2.0 * mcal_shear)
    err_r21_S = np.sqrt(
        _mean_err(selection_1p['g_noshear'][:, 1])**2
        + _mean_err(selection_1m['g_noshear'][:, 1])**2
    ) / (2.0 * mcal_shear)

    R_S = np.array([
        [r11_S, r12_S],
        [r21_S, r22_S],
    ])
    err_R_S = np.array([
        [err_r11_S, err_r12_S],
        [err_r21_S, err_r22_S],
    ])

    # ------------------------------------------------------------------ #
    #  PSF and gamma additive bias corrections
    # ------------------------------------------------------------------ #
    c1_psf = np.mean(
        (noshear_selection['g_1p_psf'][:, 0] + noshear_selection['g_1m_psf'][:, 0]) / 2.0
        - noshear_selection['g_noshear'][:, 0]
    )
    c2_psf = np.mean(
        (noshear_selection['g_2p_psf'][:, 1] + noshear_selection['g_2m_psf'][:, 1]) / 2.0
        - noshear_selection['g_noshear'][:, 1]
    )
    c1_gamma = np.mean(
        (noshear_selection['g_1p'][:, 0] + noshear_selection['g_1m'][:, 0]) / 2.0
        - noshear_selection['g_noshear'][:, 0]
    )
    c2_gamma = np.mean(
        (noshear_selection['g_2p'][:, 1] + noshear_selection['g_2m'][:, 1]) / 2.0
        - noshear_selection['g_noshear'][:, 1]
    )

    # ------------------------------------------------------------------ #
    #  Total response and correction vectors
    # ------------------------------------------------------------------ #
    R = R_gamma + R_S
    R_inv = np.linalg.inv(R)

    c_psf = np.array([c1_psf, c2_psf])
    c_gamma = np.array([c1_gamma, c2_gamma])
    c_total = c_psf + c_gamma

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    print("Gamma Response Matrix (R_gamma):")
    print(R_gamma)
    print("\nSelection Bias Response Matrix (R_S):")
    print(R_S)
    print("\nTotal Response Matrix (R):")
    print(R)
    print("\nSelection Response Uncertainty:")
    print(err_R_S)
    print("\nPSF Correction Vector (c_psf):")
    print(c_psf)
    print("\nGamma Correction Vector (c_gamma):")
    print(c_gamma)
    print("\nTotal Correction Vector (c_total = c_psf + c_gamma):")
    print(c_total)
    print(f"\n{len(noshear_selection)} objects passed selection criteria")

    # ------------------------------------------------------------------ #
    #  Mean ellipticity diagnostic
    # ------------------------------------------------------------------ #
    g1_noshear = noshear_selection['g_noshear'][:, 0]
    g2_noshear = noshear_selection['g_noshear'][:, 1]

    mean_g1 = np.mean(g1_noshear)
    mean_g2 = np.mean(g2_noshear)
    err_g1 = np.std(g1_noshear, ddof=1) / np.sqrt(len(g1_noshear))
    err_g2 = np.std(g2_noshear, ddof=1) / np.sqrt(len(g2_noshear))

    print(f"\nWeighted mean g_noshear (selected):")
    print(f"  <g1> = {mean_g1:.6f} +/- {err_g1:.6f}")
    print(f"  <g2> = {mean_g2:.6f} +/- {err_g2:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, gi, label, mean_gi, err_gi in zip(
        axes,
        [g1_noshear, g2_noshear],
        [r'$g_1$', r'$g_2$'],
        [mean_g1, mean_g2],
        [err_g1, err_g2],
    ):
        ax.hist(gi, bins=80, range=(-1, 1), histtype='stepfilled',
                color='slategrey', alpha=0.45, edgecolor='slategrey')
        ax.axvline(mean_gi, color='crimson', ls='-', lw=1.5,
                   label=rf'$\langle {label[1:-1]} \rangle = {mean_gi:.5f} \pm {err_gi:.5f}$')
        ax.axvline(0, color='k', ls=':', lw=0.8, alpha=0.5)
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel('Count', fontsize=12)
        ax.legend(fontsize=10, frameon=False)
        ax.tick_params(axis='both', direction='in', which='both')

    fig.suptitle('Weighted g_noshear distribution (selected galaxies)', fontsize=14)
    fig.tight_layout()
    # fig.savefig('g_noshear_hist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    # print("Saved g_noshear histogram -> g_noshear_hist.png")

    # ------------------------------------------------------------------ #
    #  Apply calibration to the selected catalog
    # ------------------------------------------------------------------ #
    selected = noshear_selection.copy()

    # Transform the per-galaxy covariance through R_inv
    g_cov_noshear = selected['g_cov_noshear']
    corrected_cov = np.einsum('ij,njk,lk->nil', R_inv, g_cov_noshear, R_inv)

    # Subtract additive bias, then divide by response
    g_biased = selected['g_noshear'] - c_total  # (n, 2)

    if R_diagonal:
        g_corrected = np.empty_like(g_biased)  # FIX: was np.array_like (doesn't exist)
        g_corrected[:, 0] = g_biased[:, 0] / R[0, 0]
        g_corrected[:, 1] = g_biased[:, 1] / R[1, 1]
    else:
        g_corrected = np.einsum('ij,nj->ni', R_inv, g_biased)  # (n, 2)

    if overwrite_calibration:
        selected['g1_Rinv'] = g_corrected[:, 0]
        selected['g2_Rinv'] = g_corrected[:, 1]
    selected['g_cov_Rinv'] = corrected_cov
    selected['R11_S'] = r11_S
    selected['R22_S'] = r22_S

    return selected, R_S, c_total, mean_g1, mean_g2