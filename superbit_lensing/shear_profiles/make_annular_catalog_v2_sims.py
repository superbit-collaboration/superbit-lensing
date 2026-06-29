import numpy as np
import ipdb
from astropy.table import Table, vstack, hstack, join
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import hstack
import glob
import sys, os
from astropy.io import fits
from esutil import htm
from argparse import ArgumentParser


from annular_jmac import Annular, ShearCalc
from make_redshift_cat_sims import make_redshift_catalog
from superbit_lensing import utils
from superbit_lensing.match import SkyCoordMatcher

MINIMAL_TYPES = ['noshear', '1p', '1m', '2p', '2m']
DILATE_TYPES = ['noshear', '1p', '1m', '2p', '2m', '1p_psf', '1m_psf', '2p_psf', '2m_psf']

DEFAULT_MCAL_PARS = {'psf': 'dilate', 'mcal_shear': 0.01, 'types' : DILATE_TYPES}
AZGAUSS_MCAL_PARS = {'psf': 'azgauss', 'mcal_shear': 0.01, 'types' : MINIMAL_TYPES}

# Shear steps that map onto the rows/columns of the 2x2 metacal matrices:
# step '1' -> column 0 (g1), step '2' -> column 1 (g2)
_SHEAR_STEPS = ('1', '2')


def mcal_response(tab, mcal_shear, suffix=''):
    """Per-object 2x2 mcal response, R[i, j, n] for component i, shear step j,
    object n. Pass suffix='_psf' for the PSF response."""
    R = np.array([
        [(tab[f'g_{step}p{suffix}'][:, i] - tab[f'g_{step}m{suffix}'][:, i]) / (2. * mcal_shear)
         for step in _SHEAR_STEPS]
        for i in range(2)
    ])
    return R


def mcal_additive_bias(tab, suffix=''):
    """Per-object additive bias, c[i, n] for component i. The diagonal step
    (i -> step i+1) is used, matching the standard mcal convention."""
    c = np.array([
        (tab[f'g_{step}p{suffix}'][:, i] + tab[f'g_{step}m{suffix}'][:, i]) / 2.
        - tab['g_noshear'][:, i]
        for i, step in enumerate(_SHEAR_STEPS)
    ])
    return c


def mcal_selection_response(selections, mcal_shear):
    """2x2 selection-bias response from the per-step selected catalogs,
    R_S[i, j] for component i and shear step j."""
    R = np.empty((2, 2))
    for j, step in enumerate(_SHEAR_STEPS):
        # The plus/minus selections contain different objects, so the mean of
        # each must be taken before differencing.
        gp = np.mean(selections[f'{step}p']['g_noshear'], axis=0)
        gm = np.mean(selections[f'{step}m']['g_noshear'], axis=0)
        R[:, j] = (gp - gm) / (2. * mcal_shear)
    return R

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to cluster data')
    parser.add_argument('run_name', type=str,
                        help='Run name (target name for real data)')
    parser.add_argument('mcal_file', type=str,
                        help='Metacal catalog filename')
    parser.add_argument('outfile', type=str,
                        help='Output selected source catalog filename')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('-config', '-c', type=str,
                        default=os.path.join(script_dir, 'configs/default_annular_config_sims.yaml'),
                        help='Configuration file with annular cuts etc.')
    parser.add_argument('-detection_band', type=str, default='b',
                        help='Detection bandpass [default: b]')
    parser.add_argument('-cluster_redshift', type=str, default=None,
                        help='Redshift of cluster')
    parser.add_argument('-reconv_psf', type=str, default='dilate',
                        help='Reconvolution psf kernel (default: dilate)')
    parser.add_argument('--psf_correction', action='store_true', default=False,
                        help='Apply PSF correction')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite output files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Turn on for verbose prints')

    return parser.parse_args()


class AnnularCatalog():

    """
    This class calculates shear-calibrated ellipticities based on the
    metacal procedure. This will then match against the parent
    SExtractor catalog (set in option in main)
    """

    def __init__(self, cat_info, config):
        """
        cat_info: dict
            A dictionary that must contain the paths for the SExtractor
            catalog, the mcal catalog, and the output catalog filename
        annular_bins: dict
            A dictionary holding the definitions of the annular bins
        config: dict
            Configuration parameters loaded from YAML file
        """

        self.cat_info = cat_info
        self.config = config
        self.detect_cat = cat_info['detect_cat']
        self.data_dir = cat_info['data_dir']
        self.mcal_file = cat_info['mcal_file']
        self.outfile = cat_info['mcal_selected']
        self.outdir = cat_info['outdir']
        self.run_name = cat_info['run_name']
        self.detection_band = cat_info['detection_band']
        self.redshift_cat = cat_info['redshift_cat']
        self.cluster_redshift = cat_info['cluster_redshift']
        self.mcal_pars = cat_info['mcal_pars']
        self.psf_correction = cat_info['psf_correction']


        if self.outdir is not None:
            self.outfile = os.path.join(self.outdir, self.outfile)
        else:
            self.outdir = ''

        self.outfile_w_truth = self.outfile.replace(".fits", "_with_truth.fits")

        self.det_cat = Table.read(self.detect_cat, hdu=2)
        self.mcal = Table.read(self.mcal_file)
        self.joined = None
        self.joined_gals = None
        self.selected = None
        self.outcat = None

        self.Ndet = len(self.det_cat)
        self.Nmcal = len(self.mcal)

        return

    def join(self, overwrite=False):

        # rename a few cols for joining purposes
        colmap = {
            'ALPHAWIN_J2000': 'ra',
            'DELTAWIN_J2000': 'dec',
            'NUMBER': 'id'
            }
        for old, new in colmap.items():
            self.det_cat.rename_column(old, new)

        # TODO: Should we remove a few duplicate cols here?

        self.joined = join(
            self.det_cat, self.mcal, join_type='inner',
            keys=['id', 'ra', 'dec'], table_names=['se', 'mcal']
            )

        Nmin = min([self.Ndet, self.Nmcal])
        Nobjs = len(self.joined)

        if (Nobjs != Nmin):
            raise ValueError('There was an error while joining the SE and mcal ' +
                             f'catalogs;' +
                             f'\nlen(SE)={self.Ndet}' +
                             f'\nlen(mcal)={self.Nmcal}' +
                             f'\nlen(joined)={Nobjs}'
                             )

        print(f'{Nobjs} mcal objects joined to reference SExtractor cat')

        try:
            # save full catalog to file
            if self.run_name is None:
                p = ''
            else:
                p = f'{self.run_name}_'

            outfile = os.path.join(self.outdir, f'{p}full_joined_catalog.fits')
            #self.joined.write(outfile, overwrite=overwrite)

        # we *want* it to fail loudly!
        except OSError as err:
            print('Cannot overwrite {outfile} unless `overwrite` is set to True!')
            raise err

        return


    def _redshift_select(self, redshift_cat, overwrite=False):
        '''
        Select background galaxies from larger transformed shear catalog:
            - Load in file containing redshifts
            - Select background galaxies behind galaxy cluster
            - Match in RA/Dec to transformed shear catalog
            - Filter self.r, self.gtan, self.gcross to be background-only
            - If it's a forecast run, store the number of galaxies injected
              into the simulation
        '''

        joined_cat = self.joined

        try:
            redshifts = Table.read(redshift_cat)
            #if vb is True:
            print(f'Read in redshift catalog {redshift_cat}')

        except FileNotFoundError as fnf_err:
            print(f"Can't load redshift catalog {redshift_cat}, check name/type?")
            raise fnf_err

        try:
            # In case this is a forecasting run and the redshift catalog
            # is actually a truth catalog
            truth = redshifts
            zcat_gals = truth[truth['obj_class'] == 'gal']
            self.n_truth_gals = len(truth_gals)  # type: ignore
            cluster_gals = truth[truth['obj_class']=='cluster_gal']
            cluster_redshift = np.mean(cluster_gals['redshift'])
            self.cluster_redshift = cluster_redshift
            ra_col = 'ra'; dec_col = 'dec'; z_col='redshift'

        except:
            # Assume real data -- need to have cluster redshift defined!
            # And some basic RA/Dec columns
            ra_col = 'RA'; dec_col = 'DEC'; z_col = 'Redshift'
            if self.cluster_redshift == None:
                print('No cluster_redshift argument supplied; ' +\
                        'no redshift cuts will be made')


        if (~np.isin(ra_col, redshifts.colnames)) & (~np.isin(dec_col, redshifts.colnames)):
            print(f'Redshift catalog missing columns {ra_col}, {dec_col}')

        zcat_gals = redshifts[redshifts[z_col] > 0]

        z_matcher = htm.Matcher(16,
                                ra = zcat_gals[ra_col],
                                dec = zcat_gals[dec_col]
                                )

        joined_file_ind, z_ind, dist = z_matcher.match(
                                            ra = joined_cat['ra'],
                                            dec = joined_cat['dec'],
                                            maxmatch = 1,
                                            radius = 1e-4
                                            )

        print(f"# {len(dist)} of {len(joined_cat['ra'])}"+\
                    " objects matched to redshift catalog objects")

        gals_joined_cat = joined_cat[joined_file_ind]
        gals_joined_cat.add_column(redshifts[z_col][z_ind], name='redshift')

        try:
            if self.run_name is None:
                p = ''
            else:
                p = f'{self.run_name}_'

                outfile = os.path.join(self.outdir,
                                    f'{p}gals_joined_catalog.fits'
                                    )
                #gals_joined_cat.write(outfile, overwrite=overwrite)

        except OSError as err:
            print('Cannot overwrite {outfile} unless `overwrite` is set to True!')
            raise err

        self.joined_gals = gals_joined_cat

        return


    def make_table(self, overwrite=False):
        """
        - Remove foreground galaxies from sample using redshift info in truth file
        - Select from catalog on g_cov, T/T_psf, etc.
        - Correct g1/g2_noshear for the Rinv quantity (see Huff & Mandelbaum 2017)
        - Save shear-response-corrected ellipticities to an output table
        """

        # Access truth file name
        cat_info = self.cat_info
        data_dir = self.cat_info['data_dir']
        redshift_cat = self.redshift_cat

        # Filter out foreground galaxies using redshifts in truth file
        self._redshift_select(redshift_cat, overwrite=overwrite)


        # Apply selection cuts and produce responsivity-corrected shear moments
        self._compute_metacal_quantities()

        # Save selected galaxies to file
        for key, val in self.config['mcal_cuts'].items():
            self.selected.meta[key] = val

        self.selected.write(self.outfile, format='fits', overwrite=overwrite)

        truthfile = f"{cat_info['data_dir']}/{self.run_name}/{cat_info['detection_band']}/cal/{self.run_name}_truth_{cat_info['detection_band']}.fits"

        truth = Table.read(truthfile)
        
        self.selected_with_truth = join(
            self.selected, truth, join_type='inner',
            keys=['ra', 'dec'], table_names=['mcal', 'truth']
            )

        # Convert RA/Dec to SkyCoord objects
        tolerance_deg = 1./3600.
        matcher_truth = SkyCoordMatcher(self.selected, truth, cat1_ratag='ra', cat1_dectag='dec',
                                return_idx=True, match_radius=1 * tolerance_deg)
        matched_selected, matched_truth, idx1, idx2 = matcher_truth.get_matched_pairs()

        # Apply mask and combine tables
        self.selected_with_truth = hstack([matched_selected, matched_truth], table_names=["mcal", "truth"])

        print(f"Number of objects joined from truth file : {len(self.selected_with_truth)}")

        self.selected_with_truth.write(self.outfile_w_truth, format='fits', overwrite=overwrite)
        print(f'The final file with object truth values: {self.outfile_w_truth}')

        print("==Some Stats==\n")
        _ = utils.analyze_mcal_fits(self.outfile_w_truth, update_header=True)

        return

    def _compute_metacal_quantities(self):
        """
        - Cut sources on S/N and minimum size (adapted from DES cuts).
        - compute mean r11 and r22 for galaxies: responsivity & selection
        - divide "no shear" g1/g2 by r11 and r22, and return
        """

        # Load configuration parameters
        qual_cuts = self.config['mcal_cuts']
        mcal_shear = self.config['mcal_shear']
        shape_noise = self.config['shape_noise']

        has_psf = self.mcal_pars['psf'] == 'dilate'

        # Define individual selection cuts
        min_Tpsf = qual_cuts['min_Tpsf']
        max_sn = qual_cuts['max_sn']
        min_sn = qual_cuts['min_sn']
        min_T = qual_cuts['min_T']
        max_T = qual_cuts['max_T']
        

        if self.cluster_redshift != None:
            # Add in a little bit of a safety margin -- maybe a bad call for simulated data?
            min_redshift = float(self.cluster_redshift) + 0.025
        else:
            min_redshift = 0

        print(
            f"#\n# cuts applied: Tpsf_ratio > {min_Tpsf:.2f}"
            + f"\n# SN > {min_sn:.1f} T > {min_T:.2f}"
            + f"\n# redshift = {min_redshift:.3f} \n#\n"
        )

        mcal = self.joined_gals

        shear_keys = [f"g_{suffix}" for suffix in self.mcal_pars['types']]

        finite = np.ones(len(mcal), dtype=bool)
        for key in shear_keys:
            finite &= np.isfinite(mcal[key][:, 0]) & np.isfinite(mcal[key][:, 1])

        mcal = mcal[finite]

        def apply_cuts(suffix):
            return mcal[
                (mcal[f'T_{suffix}'] >= min_Tpsf * mcal[f'Tpsf_{suffix}']) \
                & (mcal[f'T_{suffix}'] <= max_T) \
                & (mcal[f'T_{suffix}'] >= min_T) \
                & (mcal[f's2n_{suffix}'] > min_sn) \
                & (mcal[f's2n_{suffix}'] < max_sn) \
                # & (mcal['redshift'] > min_redshift)
            ]

        noshear_selection = apply_cuts('noshear')
        selection_1p = apply_cuts('1p')
        selection_1m = apply_cuts('1m')
        selection_2p = apply_cuts('2p')
        selection_2m = apply_cuts('2m')

        # assuming delta_shear in ngmix_fit is 0.01
        selections = {'1p': selection_1p, '1m': selection_1m,
                      '2p': selection_2p, '2m': selection_2m}

        # assuming delta_shear in ngmix_fit is 0.01
        # Mean responses (mean over objects of the per-object response)
        R_gamma = np.mean(mcal_response(noshear_selection, mcal_shear), axis=2)
        R_S = mcal_selection_response(selections, mcal_shear)

        # Mean additive biases
        c_gamma = np.mean(mcal_additive_bias(noshear_selection), axis=1)

        # Selection-response diagonal terms stored later as value-added columns
        r11_S, r22_S = R_S[0, 0], R_S[1, 1]
        
        # Compute the final response matrix
        R = R_gamma + R_S
        R_inv = np.linalg.inv(R)        
        c_total =  c_gamma
        
        if has_psf:
            R_psf = np.mean(mcal_response(noshear_selection, mcal_shear, suffix='_psf'), axis=2)
            c_psf = np.mean(mcal_additive_bias(noshear_selection, suffix='_psf'), axis=1)
            c_total = c_total + c_psf


        print("Gamma Response Matrix (R_gamma):")
        print(R_gamma)
        print("\nSelection Bias Response Matrix (R_S):")
        print(R_S)
        if has_psf:
            print("\nPSF Response Matrix (R_psf):")
            print(R_psf)
            print("\nPSF Correction Vector (c_psf):")
            print(c_psf)

        print("\nGamma Correction Vector (c_gamma):")
        print(c_gamma)

        print(f'{len(noshear_selection)} objects passed selection criteria')

        # Populate the selCat attribute with "noshear"-selected catalog
        self.selected = noshear_selection

        print(f'shape noise is {shape_noise}')
        g_cov_noshear = self.selected['g_cov_noshear']

        # Transform the covariance matrix
        corrected_cov = np.einsum('ij,njk,lk->nil', R_inv, g_cov_noshear, R_inv)

        tot_covar = shape_noise + corrected_cov[:, 0, 0] + corrected_cov[:, 1, 1]
        weight = 1. / tot_covar

        # Per-object response and bias terms (rows = component, cols = shear step)
        (r11, r12), (r21, r22) = mcal_response(noshear_selection, mcal_shear)
        c1_gamma, c2_gamma = mcal_additive_bias(noshear_selection)

        # Size response
        rT_1 = ( noshear_selection['T_1p'] - noshear_selection['T_1m'] ) / (2.*mcal_shear)
        rT_2 = ( noshear_selection['T_2p'] - noshear_selection['T_2m'] ) / (2.*mcal_shear)

        # Assemble the value-added columns; PSF terms only for the 'dilate' kernel
        value_added = {
            'r11': r11, 'r12': r12, 'r21': r21, 'r22': r22,
            'c1': c1_gamma, 'c2': c2_gamma,
            'rT_1': rT_1, 'rT_2': rT_2,
        }
        if has_psf:
            (r11_psf, r12_psf), (r21_psf, r22_psf) = \
                mcal_response(noshear_selection, mcal_shear, suffix='_psf')
            c1_psf, c2_psf = mcal_additive_bias(noshear_selection, suffix='_psf')
            value_added.update({
                'r11_psf': r11_psf, 'r12_psf': r12_psf,
                'r21_psf': r21_psf, 'r22_psf': r22_psf,
                'c1_psf': c1_psf, 'c2_psf': c2_psf,
            })

        try:
            #---------------------------------
            # Now add value-adds to table
            self.selected.add_columns(
                list(value_added.values()), names=list(value_added.keys())
            )

        except ValueError as e:
            # In some cases, these cols are already computed
            print('WARNING: mcal r{ij} value-added cols not added; ' +\
                'already present in catalog')

        try:
            R = np.array([[r11, r12], [r21, r22]])
            g1_MC = np.zeros_like(r11)
            g2_MC = np.zeros_like(r22)

            N = len(g1_MC)
            for k in range(N):
                Rinv = np.linalg.inv(R[:,:,k])
                gMC = np.dot(Rinv, noshear_selection[k]['g_noshear'])
                g1_MC[k] = gMC[0]
                g2_MC[k] = gMC[1]

            self.selected.add_columns(
                [g1_MC, g2_MC],
                names = ['g1_MC', 'g2_MC']
            )

        except ValueError as e:
            # In some cases, these cols are already computed
            print('WARNING: mcal g{1/2}_MC value-added cols not added; ' +\
                'already present in catalog')

        g_biased = self.selected['g_noshear'] - c_total  # Shape: (n, 2)

        g_corrected = np.einsum('ij,nj->ni', R_inv, g_biased)  # Shape: (n, 2)

        self.selected['g1_Rinv'] = g_corrected[:, 0]
        self.selected['g2_Rinv'] = g_corrected[:, 1]
        self.selected['g_cov_Rinv'] = corrected_cov
        self.selected['R11_S'] = r11_S
        self.selected['R22_S'] = r22_S
        self.selected['weight'] = weight

        return

    def run(self, overwrite, vb=False):

        # match master metacal catalog to source extractor cat
        self.join(overwrite=overwrite)

        # source selection; saves table to self.outfile
        self.make_table(overwrite=overwrite)


def main(args):

    data_dir = args.data_dir
    target_name = args.run_name
    mcal_file = args.mcal_file
    outfile = args.outfile
    outdir = args.outdir
    config_yaml = args.config
    cluster_redshift = args.cluster_redshift
    reconv_psf = args.reconv_psf
    psf_correction = args.psf_correction
    detection_band = args.detection_band
    overwrite = args.overwrite
    vb = args.vb
    
    if reconv_psf=='dilate':
        mcal_pars = DEFAULT_MCAL_PARS
    elif reconv_psf=='azgauss':
        mcal_pars = AZGAUSS_MCAL_PARS
    else:
        raise ValueError("Invalid reconv_psf value. Use 'dilate' or 'azgauss'.")

    # Load config file
    config = utils.read_yaml(config_yaml)

    ## Get center of galaxy cluster for fitting
    ## Throw error if image can't be read in
    detect_cat = os.path.join(data_dir, target_name, detection_band,
        f'coadd/{target_name}_coadd_{detection_band}_cat.fits'
    )

    ## Make dummy redshift catalog -- should be a flag!
    print("Making redshift catalog")
    redshift_cat = make_redshift_catalog(
        datadir=data_dir, target=target_name,
        band=detection_band, detect_cat_path=detect_cat
    )

    ## n.b outfile is the name of the metacalibrated &
    ## quality-selected galaxy catalog

    cat_info={
        'data_dir': data_dir,
        'detect_cat': detect_cat,
        'mcal_file': mcal_file,
        'run_name': target_name,
        'detection_band': detection_band,
        'mcal_selected': outfile,
        'outdir': outdir,
        'redshift_cat': redshift_cat,
        'cluster_redshift': cluster_redshift,
        'mcal_pars': mcal_pars,
        'psf_correction': psf_correction
        }

    annular_cat = AnnularCatalog(cat_info, config)

    # run everything
    annular_cat.run(overwrite=overwrite, vb=vb)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print('make_annular_catalog.py has completed succesfully')
    else:
        print(f'make_annular_catalog.py has failed w/ rc={rc}')    
        