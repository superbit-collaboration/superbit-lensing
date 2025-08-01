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
from make_redshift_cat import make_redshift_catalog
from superbit_lensing import utils
from superbit_lensing.match import SkyCoordMatcher


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
    parser.add_argument('-redshift_cat', type=str, default=None,
                        help='File containing redshifts')
    parser.add_argument('-nfw_file', type=str, default=None,
                        help='Theory NFW shear catalog')
    parser.add_argument('-Nresample', type=int, default=10,
                        help='The number of NFW redshift resamples to compute')
    parser.add_argument('-rmin', type=float, default=100,
                        help='Starting radius value (in pixels)')
    parser.add_argument('-rmax', type=float, default=5200,
                        help='Ending radius value (in pixels)')
    parser.add_argument('-nfw_seed', type=int, default=None,
                        help='Seed for nfw redshift resampling')
    parser.add_argument('-nbins', type=int, default=18,
                        help='Number of radial bins')
    parser.add_argument('--center', type=str, default='image',
                    help='Center type for shear profile calculation; image or xray')
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

    def __init__(self, cat_info, annular_info, config):
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
        self.annular_info = annular_info
        self.config = config
        self.detect_cat = cat_info['detect_cat']
        self.mcal_file = cat_info['mcal_file']
        self.outfile = cat_info['mcal_selected']
        self.outdir = cat_info['outdir']
        self.run_name = cat_info['run_name']
        self.redshift_cat = cat_info['redshift_cat']
        self.cluster_redshift = cat_info['cluster_redshift']
        self.nfw_file = cat_info['nfw_file']
        self.Nresample = cat_info['Nresample']

        self.rmin = annular_info['rmin']
        self.rmax = annular_info['rmax']
        self.nbins = annular_info['nbins']
        self.coadd_center = annular_info['coadd_center']

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
        truthfile = f"{cat_info['data_dir']}/{self.run_name}/{cat_info['band']}/cal/{self.run_name}_truth_{cat_info['band']}.fits"
        self.truth = Table.read(truthfile)
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
                                            radius = 1./3600.
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
        # Convert RA/Dec to SkyCoord objects
        tolerance_deg = 1./3600.
        matcher_truth = SkyCoordMatcher(gals_joined_cat, self.truth, cat1_ratag='ra', cat1_dectag='dec',
                                return_idx=True, match_radius=1 * tolerance_deg)
        matched_selected, matched_truth, idx1, idx2 = matcher_truth.get_matched_pairs()

        # Apply mask and combine tables
        self.joined_gals = hstack([matched_selected, matched_truth], table_names=["mcal", "truth"])
        self.joined_gals.rename_column('redshift_mcal', 'redshift')
        self.joined_gals.remove_column('redshift_truth')
        self.joined_gals.rename_column('ra_mcal', 'ra')
        self.joined_gals.rename_column('dec_mcal', 'dec')
        self.joined_gals.remove_column('ra_truth')
        self.joined_gals.remove_column('dec_truth')        

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

        truthfile = f"{cat_info['data_dir']}/{self.run_name}/{cat_info['band']}/cal/{self.run_name}_truth_{cat_info['band']}.fits"

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
        mcal = mcal[mcal['admom_sigma']>0.19]

        noshear_selection = mcal[
            (mcal['T_noshear'] >= min_Tpsf * mcal['Tpsf_noshear'])\
            & (mcal['T_noshear'] < max_T) \
            & (mcal['T_noshear'] >= min_T) \
            & (mcal['s2n_noshear'] > min_sn) \
            & (mcal['s2n_noshear'] < max_sn) \
            #& (mcal['redshift'] > min_redshift)
        ]

        selection_1p = mcal[
            (mcal['T_1p'] >= min_Tpsf * mcal['Tpsf_1p']) \
            & (mcal['T_1p'] <= max_T) \
            & (mcal['T_1p'] >= min_T) \
            & (mcal['s2n_1p'] > min_sn) \
            & (mcal['s2n_1p'] < max_sn) \
            & (mcal['redshift'] > min_redshift)
        ]

        selection_1m = mcal[
            (mcal['T_1m'] >= min_Tpsf * mcal['Tpsf_1m']) \
            & (mcal['T_1m'] <= max_T) \
            & (mcal['T_1m'] >= min_T) \
            & (mcal['s2n_1m'] > min_sn) \
            & (mcal['s2n_1m'] < max_sn) \
            & (mcal['redshift'] > min_redshift)
        ]

        selection_2p = mcal[
            (mcal['T_2p'] >= min_Tpsf * mcal['Tpsf_2p']) \
            & (mcal['T_2p'] <= max_T) \
            & (mcal['T_2p'] >= min_T) \
            & (mcal['s2n_2p'] > min_sn) \
            & (mcal['s2n_2p'] < max_sn) \
            & (mcal['redshift'] > min_redshift)
        ]

        selection_2m = mcal[
            (mcal['T_2m'] >= min_Tpsf*mcal['Tpsf_2m']) \
            & (mcal['T_2m'] <= max_T) \
            & (mcal['T_2m'] >= min_T) \
            & (mcal['s2n_2m'] > min_sn) \
            & (mcal['s2n_2m'] < max_sn) \
            & (mcal['redshift'] > min_redshift)
        ]

        # assuming delta_shear in ngmix_fit is 0.01
        r11_gamma = (np.mean(noshear_selection['g_1p'][:,0]) -
                    np.mean(noshear_selection['g_1m'][:,0])) / (2.*mcal_shear)
        r22_gamma = (np.mean(noshear_selection['g_2p'][:,1]) -
                    np.mean(noshear_selection['g_2m'][:,1])) / (2.*mcal_shear)
        r12_gamma = (np.mean(noshear_selection['g_2p'][:, 0]) -
                    np.mean(noshear_selection['g_2m'][:, 0])) / (2.*mcal_shear)
        r21_gamma = (np.mean(noshear_selection['g_1p'][:, 1]) -
                    np.mean(noshear_selection['g_1m'][:, 1])) / (2.*mcal_shear)

        # assuming delta_shear in ngmix_fit is 0.01
        r11_S = (np.mean(selection_1p['g_noshear'][:,0]) -
                np.mean(selection_1m['g_noshear'][:,0])) / (2.*mcal_shear)
        r22_S = (np.mean(selection_2p['g_noshear'][:,1]) -
                np.mean(selection_2m['g_noshear'][:,1])) / (2.*mcal_shear)
        r12_S = (np.mean(selection_2p['g_noshear'][:, 0]) -
                np.mean(selection_2m['g_noshear'][:, 0])) / (2.*mcal_shear)
        r21_S = (np.mean(selection_1p['g_noshear'][:, 1]) -
                np.mean(selection_1m['g_noshear'][:, 1])) / (2.*mcal_shear)

        c1_psf = np.mean((noshear_selection['g_1p_psf'][:,0] + noshear_selection['g_1m_psf'][:,0])/2 - noshear_selection['g_noshear'][:,0])
        c2_psf = np.mean((noshear_selection['g_2p_psf'][:, 1] + noshear_selection['g_2m_psf'][:, 1])/2 - noshear_selection['g_noshear'][:, 1])
        c1_gamma = np.mean((noshear_selection['g_1p'][:,0] + noshear_selection['g_1m'][:,0])/2 - noshear_selection['g_noshear'][:,0])
        c2_gamma = np.mean((noshear_selection['g_2p'][:, 1] + noshear_selection['g_2m'][:, 1])/2 - noshear_selection['g_noshear'][:, 1])

        # Gamma response matrix
        R_gamma = np.array([
            [r11_gamma, r12_gamma],
            [r21_gamma, r22_gamma]
        ])

        # Selection response matrix
        R_S = np.array([
            [r11_S, r12_S],
            [r21_S, r22_S]
        ])

        # Compute the final response matrix
        R = R_gamma + R_S
        R_inv = np.linalg.inv(R)

        # PSF additive bias
        c_psf = np.array([c1_psf, c2_psf])

        # Gamma correction vector
        c_gamma = np.array([c1_gamma, c2_gamma])

        c_total = c_psf + c_gamma

        print("Gamma Response Matrix (R_gamma):")
        print(R_gamma)
        print("\nSelection Bias Response Matrix (R_S):")
        print(R_S)

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

        try:
            r11 = ( noshear_selection['g_1p'][:,0] - noshear_selection['g_1m'][:,0] ) / (2.*mcal_shear)
            r12 = ( noshear_selection['g_2p'][:,0] - noshear_selection['g_2m'][:,0] ) / (2.*mcal_shear)
            r21 = ( noshear_selection['g_1p'][:,1] - noshear_selection['g_1m'][:,1] ) / (2.*mcal_shear)
            r22 = ( noshear_selection['g_2p'][:,1] - noshear_selection['g_2m'][:,1] ) / (2.*mcal_shear)
            c1_psf = ( (noshear_selection['g_1p_psf'][:,0] + noshear_selection['g_1m_psf'][:,0])/2 - noshear_selection['g_noshear'][:,0])
            c2_psf = ((noshear_selection['g_2p_psf'][:, 1] + noshear_selection['g_2m_psf'][:, 1])/2 - noshear_selection['g_noshear'][:, 1])
            c1_gamma = ((noshear_selection['g_1p'][:, 0] + noshear_selection['g_1m'][:, 0])/2 - noshear_selection['g_noshear'][:, 0])
            c2_gamma = ((noshear_selection['g_2p'][:, 1] + noshear_selection['g_2m'][:, 1])/2 - noshear_selection['g_noshear'][:, 1])

            #---------------------------------
            # Now add value-adds to table
            self.selected.add_columns(
                [r11, r12, r21, r22, c1_gamma, c2_gamma, c1_psf, c2_psf],
                names=['r11', 'r12', 'r21', 'r22', 'c1', 'c2', 'c1_psf', 'c2_psf']
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

    
def get_xray_center(run_name, csv_path='superbit-lensing/data/catalogs/superbit_xray_centers.csv'):
    """
    Get X-ray center coordinates from CSV file
    """
    try:
        centers = Table.read(csv_path)
        mask = centers['Name'] == run_name
        if not any(mask):
            raise ValueError(f"No X-ray center found for cluster {run_name}")
        return centers['RA'][mask][0], centers['Dec'][mask][0]
    except Exception as e:
        print(f"Error reading X-ray centers: {e}")
        raise

def main(args):

    data_dir = args.data_dir
    target_name = args.run_name
    mcal_file = args.mcal_file
    outfile = args.outfile
    outdir = args.outdir
    config_yaml = args.config
    cluster_redshift = args.cluster_redshift
    detection_band = args.detection_band
    redshift_cat = args.redshift_cat
    nfw_file = args.nfw_file
    Nresample = args.Nresample
    rmin = args.rmin
    rmax = args.rmax
    nfw_seed = args.nfw_seed
    nbins = args.nbins
    overwrite = args.overwrite
    vb = args.vb

    # Load config file
    config = utils.read_yaml(config_yaml)

    # Define position args
    xy_cols = ['XWIN_IMAGE_se', 'YWIN_IMAGE_se']
    shear_args = ['g1_Rinv', 'g2_Rinv']

    ## Get center of galaxy cluster for fitting
    ## Throw error if image can't be read in
    detect_cat = os.path.join(data_dir, target_name, detection_band,
        f'coadd/{target_name}_coadd_{detection_band}_cat.fits'
    )
    detect_im = os.path.join(data_dir, target_name, detection_band,
        f'coadd/{target_name}_coadd_{detection_band}.fits'
    )
    print(f'using detection catalog {detect_cat}')
    print(f'using detection image {detect_im}')

    if args.center == 'image':
        try:
            assert os.path.exists(detect_im) is True
            hdr = fits.getheader(detect_im)
            #This is the image center:
            xcen = hdr['CRPIX1']; ycen = hdr['CRPIX2']
            coadd_center = [xcen, ycen]
            print(f'Read image data and setting image center to ({xcen},{ycen})')
        except Exception as e:
            print('\n\n\nNo coadd image center found, cannot calculate tangential shear\n\n.')
            raise e
    else:  # xray center
        try:
            ra, dec = get_xray_center(target_name)
            # Convert RA/Dec to pixel coordinates using WCS from image header
            hdr = fits.getheader(detect_im)
            wcs = WCS(hdr)
            xcen, ycen = wcs.all_world2pix(ra, dec, 0)
            coadd_center = [xcen, ycen]
            print(f'Using X-ray center at RA,Dec = ({ra},{dec})')
            print(f'Converted to pixel coordinates: ({xcen},{ycen})')
        except Exception as e:
            print('\n\n\nError getting X-ray center coordinates\n\n.')
            raise e

    ## Make dummy redshift catalog -- should be a flag!
    print("Making redshift catalog")
    redshift_cat = make_redshift_catalog(
        datadir=data_dir, target=target_name,
        band=detection_band, detect_cat_path=detect_cat
    )


    if nfw_seed is None:
        nfw_seed = utils.generate_seeds(1)

    ## n.b outfile is the name of the metacalibrated &
    ## quality-selected galaxy catalog

    cat_info={
        'data_dir': data_dir,
        'detect_cat': detect_cat,
        'mcal_file': mcal_file,
        'run_name': target_name,
        'mcal_selected': outfile,
        'outdir': outdir,
        'redshift_cat': redshift_cat,
        'cluster_redshift': cluster_redshift,
        'nfw_file': nfw_file,
        'Nresample': Nresample,
        'nfw_seed': nfw_seed,
        'band': detection_band
        }

    annular_info = {
        'rmin': rmin,
        'rmax': rmax,
        'nbins': nbins,
        'coadd_center': coadd_center,
        'xy_args': xy_cols,
        'shear_args': shear_args
        }

    annular_cat = AnnularCatalog(cat_info, annular_info, config)

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
        