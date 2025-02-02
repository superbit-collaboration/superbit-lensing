from astropy.table import vstack, hstack, Table
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from esutil import htm
import sys
import os
import astropy
import collections
from argparse import ArgumentParser
from statsmodels.stats.weightstats import DescrStatsW

import superbit-lensing.utils as utils
from superbit-lensing.shear_profiles.bias import _compute_shear_bias
from superbit-lensing.shear_profiles.annular_jmac import _compute_profile

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('-shear_cats', type=str, default=None,
                        help = 'Tables to read in: xxx_transformed_shear_tab.fits')
    parser.add_argument('-nfw_cats', type=str, default=None,
                        help='Reference NFW shear tables to read in')
    parser.add_argument('-shear_cut', type=float, default=None,
                        help='Max tangential shear to define scale cuts')
    parser.add_argument('-outfile', type=str, default=None,
                        help = 'Name of ouput mean shear profile')
    parser.add_argument('-minrad', type=float, default=100,
                        help='Starting radius value (in pixels)')
    parser.add_argument('-maxrad', type=float, default=5200,
                        help='Ending radius value (in pixels)')
    parser.add_argument('-nbins', type=int, default=18,
                        help='Number of radial bins')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output mcal file')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

class CatalogStacker():

    def __init__(self, cat_list=None):

        self.cat_list = cat_list
        self.alpha_list = []
        self.stacked_cat = None
        self.mean_a = None
        self.std_a = None
        self.avg_nobj = None
        self.num_alphas = None

    def _get_avg_nobj(self):
        '''
        Returns the average length of catalogs in cat_list.
        If cat_list is galaxy catalogs,  it's an average source density
        if cat_list is shear profile catalogs, it's the average number of radial bins
        '''

        len_stacked = len(self.stacked_cat)
        len_catlist = len(self.cat_list)
        avg_cat_len = len_stacked/len_catlist

        self.avg_nobj = avg_cat_len

        return

    def _get_alpha_stats(self):

        num_alphas  = len(self.alpha_list)
        mean_alpha = np.mean(self.alpha_list)
        std_alpha = np.std(self.alpha_list)/np.sqrt(num_alphas)

        self.mean_a = mean_alpha
        self.std_a = std_alpha
        self.num_alphas = num_alphas

        return

    def get_catalogs(self, catnames=None):
        '''
        Either stack the supplied catnames, or
        stack the catalogs already in self
        '''

        if catnames == None:
            catnames = self.cat_list

        assert isinstance(catnames, list), f"catnames should be a list, got {type(catnames)} instead"

        stack = []
        for i in np.arange(len(catnames)):
            tab = Table.read(catnames[i],format='fits')
            stack.append(tab)

            # Skip the alpha list if it doesn't exist
            try:
                self.alpha_list.append(tab.meta['ALPHA'])
            except KeyError:
                pass

        stacked_catalog = vstack(stack, metadata_conflicts='silent')
        stacked_catalog.meta = collections.OrderedDict()

        self.stacked_cat = stacked_catalog

        return

    def _get_cat_stats(self):

        self._get_avg_nobj()

        if (len(self.alpha_list) > 1):
            self._get_alpha_stats()

        return

    def run(self):

        # Concatenate all the catalogs
        self.get_catalogs()

        # Get summary statistics
        self._get_cat_stats()

        return

def shear_curve(r, a, b, c):
    '''
    A simple model for the tangential shear profile used to determine
    best estimate for where the profile crosses the shear cut
    '''

    # ratio = r / rs

    # return rho0 / ( ratio * (1 + ratio)**2 )

    # return a + b*(c**r)
    # return a + b*r + c*(r)**2
    return a * np.exp(-b * r) + c

def plot_curve_fit(pars, r, gtan_mean, err_gtan, mean_nfw_gtan, shear_cut_flag,
                   shear_cut=None, show=False, outfile=None, s=(9,6)):
    plt.errorbar(r, gtan_mean, err_gtan, c='tab:blue', label='Meas <gtan>')
    plt.plot(r, mean_nfw_gtan, c='tab:red', label='True <nfw_gtan>')
    plt.plot(r, shear_curve(r, *pars), ls=':', lw=2, c='k', label='Curve fit')

    # z = np.polyfit(r, gtan_mean, 3)
    # f = np.poly1d(z)
    # plt.plot(r, f(r), ls='--', label='Curve fit')

    if shear_cut is not None:
        rmin = np.min(r[~shear_cut_flag])
        plt.axvspan(0, rmin, facecolor='k', alpha=0.1)

    plt.axhline(0, lw=2, ls='--', c='k')
    plt.legend()

    plt.xlabel(r'$R$ (pixels)')
    plt.ylabel(r'<g_tan>')
    plt.title('Curve fit to mean shear profile')

    plt.gcf().set_size_inches(s)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def add_mean_profile_alpha(mean_cat):
    '''
    mean_cat: astropy.Table
        The stacked mean tangential shear profile catalog on which
        we will compute the alpha, ignoring radial bins that have
        been flagged

    returns: mean_cat w/ alpha & sig_alpha in metadata
    '''

    # only consider bins outside of shear cut
    cat = mean_cat[mean_cat['shear_cut_flag'] == 0]

    # alpha & sig_alpha are added to catalog metadata
    alpha, sig_alpha = _compute_shear_bias(cat)

    mean_cat.meta.update({
        'mean_profile_alpha': alpha,
        'mean_profile_sig_alpha': sig_alpha
        })

    return mean_cat

def main(args):

    shear_cats = args.shear_cats
    nfw_cats = args.nfw_cats
    shear_cut = args.shear_cut
    outfile = args.outfile
    minrad = args.minrad
    maxrad = args.maxrad
    nbins = args.nbins
    show = args.show
    overwrite = args.overwrite
    vb = args.vb

    if shear_cats is None:
        shear_cats = 'r*/*_transformed_shear_tab.fits'
    if nfw_cats is None:
       nfw_cats = 'r*/subsampled_nfw_cat.fits'
    if outfile is None:
        outfile = './mean_shear_profile_cat.fits'

    if shear_cut is not None:
        if shear_cut <= 0:
            raise ValueError('shear_cut must be positive')

    outdir = os.path.dirname(outfile)
    stacked_cat_name = os.path.join(outdir,'all_source_gal_shears.fits')
    mean_shear_name = outfile

    logfile = 'mean_shear_profile.log'
    log = utils.setup_logger(logfile, logdir=outdir)
    logprint = utils.LogPrint(log, vb)

    # Concatenate single-realization NFW catalogs
    try:
        nfw_filelist = glob.glob(nfw_cats)
        if len(nfw_filelist) == 0:
            stacked_nfw = None
        else:
            all_nfws = CatalogStacker(nfw_filelist)
            all_nfws.run()
            stacked_nfw = all_nfws.stacked_cat
    except OSError:
        stacked_nfw = None

    # Get source density, shear cats and also average alpha
    shearcat_list = glob.glob(shear_cats)
    all_shears = CatalogStacker(shearcat_list)
    all_shears.run()

    avg_n_sources = np.ceil(all_shears.avg_nobj)
    logprint('')
    logprint(f'Avg number of galaxies in catalog is {avg_n_sources}')
    logprint('')

    stacked_shear = all_shears.stacked_cat
    stacked_shear.sort('r')

    # Calculate mean shear profile, including the
    # NFW shear profile if such a catalog is provided
    rbins = np.linspace(minrad, maxrad, nbins)
    shear_profile = _compute_profile(
        stacked_shear, rbins, nfw_tab=stacked_nfw
        )

    #---------------------------------------------------------------------------
    # Now fit a curve to the mean profile to determine where to apply the
    # shear cut
    # pars, pars_cov = curve_fit(shear_curve, midpoint_r, gtan_mean, maxfev=2000)
    pars, pars_cov = curve_fit(
        shear_curve, shear_profile['midpoint_r'], shear_profile['mean_gtan'],
        sigma=shear_profile['err_gtan'], maxfev=20000,
        p0=(.1, .0001, .1)
        )
    logprint(f'pars: {pars}')

    if shear_cut is not None:
        shear_cut_flag = np.abs(shear_curve(shear_profile['midpoint_r'], *pars)) > shear_cut
    else:
        shear_cut_flag = np.zeros(len(shear_profile['midpoint_r']), dtype=bool)

    plt_outfile = os.path.join(outdir, 'mean_shear_profile_curve_fit.png')
    plot_curve_fit(
        pars, shear_profile['midpoint_r'], shear_profile['mean_gtan'],
        shear_profile['err_gtan'], shear_profile['mean_nfw_gtan'],
        shear_cut_flag, shear_cut=shear_cut, show=show, outfile=plt_outfile
        )

    #---------------------------------------------------------------------------
    # save cols & metadata

    shear_profile.add_columns(
        [shear_cut_flag], names=['shear_cut_flag']
        )
    shear_profile.meta['mean_n_gals'] = avg_n_sources

    # compute mean profile alpha & sig alpha taking shear-cut into account
    shear_profile = add_mean_profile_alpha(shear_profile)

    logprint(f'Writing out concatenated source galaxy shears to {stacked_cat_name}')
    stacked_shear.write(stacked_cat_name, format='fits', overwrite=overwrite)

    logprint(f'Writing out mean shear profile catalog to {mean_shear_name}')
    shear_profile.write(mean_shear_name, format='fits', overwrite=overwrite)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
