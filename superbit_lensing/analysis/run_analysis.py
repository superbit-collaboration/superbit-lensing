import os
from glob import glob
from astropy.table import Table
from argparse import ArgumentParser

import superbit_lensing.utils as utils
from superbit_lensing.shear_profiles.shear_plots import ShearProfilePlotter

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('basedir', type=str,
                        help='Location of run files to analyze. Should be ' +\
                        'a list of cluster directories')
    parser.add_argument('-shear_cut', type=float, default=None,
                        help='Max tangential shear to define scale cuts')
    parser.add_argument('-minrad', type=float, default=100,
                        help='Starting radius value (in pixels)')
    parser.add_argument('-maxrad', type=float, default=5200,
                        help='Ending radius value (in pixels)')
    parser.add_argument('-nbins', type=int, default=18,
                        help='Number of radial bins')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output mcal file')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

class MeanShearProfilePlotter(ShearProfilePlotter):
    def get_alpha(self):
        '''
        Compute alpha over the mean profile bins, ignoring any past
        the shear cut
        '''

        # already computed in get_mean_shear_profile.py
        alpha = self.cat.meta['mean_profile_alpha']
        sig_alpha = self.cat.meta['mean_profile_sig_alpha']

        return alpha, sig_alpha

class AnalysisRunner(object):

    def __init__(self, basedir, minrad, maxrad, nbins, shear_cut=None,
                 outdir=None, logprint=None, vb=False):
        '''
        basedir: str
            Location of run files to analyze. Should be a list of cluster
            directories
        shear_cut: float
            The max absolute <gtan> value beyond which data bins
            are removed
        minrad, maxrad: float
            Minimum and maximum radius for shear profile calculation
        nbins:
            Number of bins for shear profile calculation
        outdir: str
            The output directory for analysis plots & cats
        logprint: LogPrint
            A LogPrint instance
        '''

        self.basedir = basedir
        self.shear_cut = shear_cut
        self.minrad = minrad
        self.maxrad = maxrad
        self.nbins = nbins

        if logprint is None:
            logprint = utils.LogPrint(None, vb)
        self.logprint = logprint

        if outdir is None:
            outdir = os.path.join(self.basedir, 'analysis')
        self.outdir = outdir
        utils.make_dir(self.outdir)

        self.analysis_dir = os.path.join(utils.MODULE_DIR, 'analysis')

        return

    def go(self, overwrite=False, show=False):

        clusters = glob(os.path.join(self.basedir, 'cl*/'))

        if len(clusters) == 0:
            self.logprint(f'No clusters found in {self.basedir}')

        for cluster in clusters:
            try:
                cl = os.path.basename(os.path.abspath(cluster))
                self.logprint(f'Starting cluster {cl}')
                self.logprint(f'Computing mean shear profile...')
                self.run_mean_shear_profile(
                    cluster, overwrite=overwrite, show=show
                    )
                self.logprint(f'Making plots...')
                self.make_plots(
                    cluster, overwrite=overwrite, show=show
                    )
            except Exception as e:
                print(f'ERROR: Cluster {cl} failed with the following ' +\
                      f'error:\n{e}')

        return

    def run_mean_shear_profile(self, cluster, overwrite=False, show=False):
        '''
        Compute the mean shear profile and associated alpha, after removing
        bins past the shear cut

        cluster: str
            The cluster dir that holds the output results for each realization
        overwrite: bool
            Set to overwrite output files
        show: bool
            Set to show plots
        '''

        minrad = self.minrad
        maxrad = self.maxrad
        nbins = self.nbins

        cl = os.path.basename(os.path.abspath(cluster))

        profile_script = os.path.join(
            self.analysis_dir, 'get_mean_shear_profile.py'
            )

        # regular expressions to grab all relevant catalogs for a given cluster
        shear_cats = os.path.join(
            cluster, 'r*/*_transformed_shear_tab.fits'
            )

        # regular expression to grab all reference NFW catalogs for a given cluster
        nfw_cats = os.path.join(
            cluster, 'r*/subsampled_nfw_cat.fits'
            )

        outfile = os.path.join(
            self.outdir, cl, 'mean_shear_profile_cat.fits'
            )

        base = f'python {profile_script} '
        opts = f'-shear_cats={shear_cats} -nfw_cats={nfw_cats} ' +\
               f'-minrad={minrad} -maxrad={maxrad} -nbins={nbins} ' +\
               f'-outfile={outfile}'
        cmd = base + opts

        if self.shear_cut is not None:
            cmd += f' -shear_cut={self.shear_cut}'

        if overwrite is True:
            cmd += ' --overwrite'

        if show is True:
            cmd += ' --show'

        if self.logprint.vb is True:
            cmd += ' --vb'

        self.logprint('Computing mean shear profile\n')
        self.logprint(f'cmd = {cmd}')
        os.system(cmd)

        return

    def make_plots(self, cluster, overwrite=False, show=False):

        cl = os.path.basename(os.path.abspath(cluster))
        outdir = os.path.join(self.outdir, cl)

        mean_catfile = os.path.join(
            self.outdir, cl, 'mean_shear_profile_cat.fits'
            )

        outfile = os.path.join(outdir, 'mean_shear_profile.png')
        self.plot_mean_profile(
            mean_catfile, cluster, outfile=outfile, overwrite=overwrite,
            show=show
            )

        outfile = os.path.join(outdir, 'stacked_shear_calibration_g1g2.png')
        self.plot_stacked_shear_calibration_g1g2(
            outfile=outfile, overwrite=overwrite
            )

        return

    def plot_mean_profile(self, catfile, cluster, outfile=None, overwrite=None,
                           show=None, size=(12,9)):

        cl = os.path.basename(cluster)

        if self.shear_cut is not None:
            shear_cut = True
        else:
            shear_cut = False

        shear_plotter = MeanShearProfilePlotter(catfile)

        shear_plotter.plot_tan_profile(
            title='Mean profile for cluster {cl}', outfile=outfile, show=show,
            size=size, shear_cut=shear_cut
            )

        return

    def plot_stacked_shear_calibration_g1g2(self, outfile=None, show=None,
                                            overwrite=False):
        stacked_cat_name = os.path.join(
            self.outdir, 'all_source_gal_shears.fits'
            )

        # TODO ...

        return

def main(args):

    basedir = args.basedir
    shear_cut = args.shear_cut
    minrad = args.minrad
    maxrad = args.maxrad
    nbins = args.nbins
    overwrite = args.overwrite
    show = args.show
    vb = args.vb

    logdir = os.path.join(basedir, 'analysis')
    logfile = 'analysis.log'
    log = utils.setup_logger(logfile, logdir=logdir)
    logprint = utils.LogPrint(log, vb)

    runner = AnalysisRunner(
        basedir, shear_cut=shear_cut, logprint=logprint,
        minrad=minrad, maxrad=maxrad, nbins=nbins
        )

    logprint('Starting AnalysisRunner')
    runner.go(overwrite=overwrite, show=show)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print('run_analysis.py has completed successfully')
    else:
        print(f'run_analysis.py has failed w/ rc={rc}')
