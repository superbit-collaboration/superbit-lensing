import numpy as np
import os
from astropy.table import Table
from superbit_lensing.diagnostics import process_catalog, compute_R_S
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from superbit_lensing.plotter import PSFLeakagePanelMaker

MINIMAL_TYPES = ['noshear', '1p', '1m', '2p', '2m']
DILATE_TYPES = ['noshear', '1p', '1m', '2p', '2m', '1p_psf', '1m_psf', '2p_psf', '2m_psf']

DEFAULT_MCAL_PARS = {'psf': 'dilate', 'mcal_shear': 0.01, 'types' : DILATE_TYPES}
AZGAUSS_MCAL_PARS = {'psf': 'azgauss', 'mcal_shear': 0.01, 'types' : MINIMAL_TYPES}

_SHEAR_STEPS = ('1', '2')

DEFAULT_CONFIG = {
                # Binning
                "n_bins": 5,              # number of bins in x and y
                "append_high_bin": 1e10,   # upper overflow bin
                "kernel": 3,
                "statistic": "median",
                'smoothing': True,

                # Cuts
                "percentile_cut": 95,      # percentile to set x/y max limits
                "x_min": 1.0,              # lower x limit
                "y_min": 10,               # lower y limit

                # Plotting
                "cmap": "magma",           # colormap
                "dpi": 600,                # figure resolution
                "linewidth": 0.003,         # grid edge line width
                "lognorm_vmin": 20,        # min count for lognorm in panel 0

                # Fonts and ticks
                "xlabel_fontsize": 12,
                "ylabel_fontsize": 12,
                "tick_fontsize": 9.5,
                "n_ticks_axis": 15,        # number of ticks on x/y axes
                "n_ticks_cbar": 5       
}

DEFUALT_SELECTION_CUT = {
    'min_Tpsf': 1.0,
    'max_sn': 1000,
    'min_sn': 15.0,
    'min_T': 0.06,
    'max_T': 100,
    'max_gpsf': 0.05
    #'admom_flag': 1,          # Optional: require admom_flag == 1
    #'min_admom_sigma': 0.1    # Optional: require admom_sigma > 0.3
}

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


class Calibrator:
    """
    Class to calibrate shear measurements using metacalibration.
    """

    def __init__(self, catalog, config = DEFAULT_CONFIG, plot_outfile=None, seed=None, reconv_psf='dilate'):
        
        if isinstance(catalog, str):
            self.catalog = Table.read(catalog)
        else:
            self.catalog = catalog
            
        self.plot_outfile = plot_outfile
        self.seed = seed
        self.config = config
        self.reconv_psf = reconv_psf
        if reconv_psf == 'dilate':
            self.mcal_pars = DEFAULT_MCAL_PARS
            shear_keys = [f"g_{suffix}" for suffix in DILATE_TYPES]
            self.has_psf = True
        elif reconv_psf == 'azgauss':
            self.mcal_pars = AZGAUSS_MCAL_PARS
            shear_keys = [f"g_{suffix}" for suffix in MINIMAL_TYPES]
            self.has_psf = False
        else:
            raise ValueError(f"Invalid reconv_psf value: {reconv_psf}. Must be 'dilate' or 'azgauss'.")

        finite = np.ones(len(self.catalog), dtype=bool)
        for key in shear_keys:
            finite &= np.isfinite(self.catalog[key][:, 0]) & np.isfinite(self.catalog[key][:, 1])

        finite &= np.isfinite(self.catalog["T_noshear"] / self.catalog['Tpsf_noshear']) & np.isfinite(self.catalog["s2n_noshear"]) & (self.catalog["T_noshear"] / self.catalog['Tpsf_noshear'] > 0) & (self.catalog["s2n_noshear"] > 0)
        
        self.catalog = self.catalog[finite]
        if self.has_psf:
            r_colnames = ['r11', 'r12', 'r21', 'r22', 'r11_psf', 'r12_psf', 'r21_psf', 'r22_psf']
            c_colnames = ['c1_psf', 'c2_psf', 'c1', 'c2']
        else:
            r_colnames = ['r11', 'r12', 'r21', 'r22']
            c_colnames = ['c1', 'c2']
        for colname in r_colnames + c_colnames:
            if colname not in self.catalog.colnames:
                self.compute_response()

        
    def compute_response(self, mcal_shear=0.01):
        """
        Compute the response of the shear measurements.
        """
        (r11, r12), (r21, r22) = mcal_response(self.catalog, mcal_shear)
        c1_gamma, c2_gamma = mcal_additive_bias(self.catalog)
        value_added = {
            'r11': r11, 'r12': r12, 'r21': r21, 'r22': r22,
            'c1': c1_gamma, 'c2': c2_gamma,
        }
        if self.has_psf:
            (r11_psf, r12_psf), (r21_psf, r22_psf) = \
                mcal_response(self.catalog, mcal_shear, suffix='_psf')
            c1_psf, c2_psf = mcal_additive_bias(self.catalog, suffix='_psf')
            value_added.update({
                'r11_psf': r11_psf, 'r12_psf': r12_psf,
                'r21_psf': r21_psf, 'r22_psf': r22_psf,
                'c1_psf': c1_psf, 'c2_psf': c2_psf,
            })
            
        self.catalog.add_columns(
            list(value_added.values()), names=list(value_added.keys())
        )
        
    
    def gridder(self, config):
        x_min, y_min = config["x_min"], config["y_min"]
        n_bins = config["n_bins"]
        xbin_var = self.catalog["T_noshear"] / self.catalog['Tpsf_noshear']
        ybin_var = self.catalog["s2n_noshear"]

        x_p = np.percentile(xbin_var, config["percentile_cut"])
        y_p = np.percentile(ybin_var, config["percentile_cut"])

        x_bins = np.append(np.logspace(np.log10(x_min), np.log10(x_p), n_bins),
                        config["append_high_bin"])
        y_bins = np.append(np.logspace(np.log10(y_min), np.log10(y_p), n_bins),
                        config["append_high_bin"])
        
        mean_r = (self.catalog["r11"] + self.catalog["r22"]) / 2
        R, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, mean_r,
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])

        R11, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r11"],
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])
        R22, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r22"],
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])
        R12, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r12"],
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])
        R21, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r21"],
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])
        # now for c
        C1, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["c1"],
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])
        C2, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["c2"],
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])    

        sigma_e = (self.catalog['g_noshear'][:, 0]**2 + self.catalog['g_noshear'][:, 1]**2) / 2
        sigma_e2, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, sigma_e,
                                        statistic=config["statistic"],
                                        bins=[x_bins, y_bins])
        weight = R**2 / sigma_e2
        res_dict = {
            "R": R,
            "R11": R11,
            "R22": R22,
            "R12": R12,
            "R21": R21,
            "C1": C1,
            "C2": C2,
            "sigma_e2": sigma_e2,
            "weight": weight,
            "x_bins": x_bins,
            "y_bins": y_bins,
            'x_p': x_p,
            'y_p': y_p,
            'xbin_var': xbin_var,
            'ybin_var': ybin_var
        }
                
        # now for the PSF responses
        if self.has_psf:
            R11_psf, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r11_psf"],
                                            statistic=config["statistic"],
                                            bins=[x_bins, y_bins])
            R22_psf, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r22_psf"],
                                            statistic=config["statistic"],
                                            bins=[x_bins, y_bins])
            R12_psf, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r12_psf"],
                                            statistic=config["statistic"],
                                            bins=[x_bins, y_bins])
            R21_psf, _, _, _ = binned_statistic_2d(xbin_var, ybin_var, self.catalog["r21_psf"],
                                            statistic=config["statistic"],
                                            bins=[x_bins, y_bins])
            res_dict.update({
                "R11_psf": R11_psf,
                "R22_psf": R22_psf,
                "R12_psf": R12_psf,
                "R21_psf": R21_psf
            })

        if config['smoothing']:
            for key in res_dict:
                if key not in ['x_bins', 'y_bins', 'x_p', 'y_p', 'xbin_var', 'ybin_var', 'sigma_e2', 'weight']:
                    res_dict[key] = gaussian_filter(res_dict[key], sigma=config['kernel'], mode='nearest')
                    
        self.res_dict = res_dict
        return res_dict
        

    def response_plotter(self):
        fig, axes = plt.subplots(6, 2, figsize=(17, 40))
        axes = axes.flatten()
        
        # first plot: response vs SNR
        data_snr = process_catalog(self.catalog, bin_by='snr', has_psf=self.has_psf)
        res_dict = self.gridder(self.config)
        axes[0].errorbar(data_snr['bin_mean'], data_snr['r11_mean'], yerr=data_snr['r11_err'], 
                        fmt='o-', label='r11', capsize=4, markersize=5, color='blue', alpha=0.7)
        axes[0].errorbar(data_snr['bin_mean'], data_snr['r22_mean'], yerr=data_snr['r22_err'], 
                    fmt='s-', label='r22', capsize=4, markersize=5, color='red', alpha=0.7)
        axes[0].errorbar(data_snr['bin_mean'], data_snr['r12_mean'], yerr=data_snr['r12_err'], 
                    fmt='o--', label='r12', capsize=4, markersize=5, color='green', alpha=0.7)
        axes[0].errorbar(data_snr['bin_mean'], data_snr['r21_mean'], yerr=data_snr['r21_err'], 
                    fmt='s--', label='r21', capsize=4, markersize=5, color='orange', alpha=0.7)
        axes[0].axhline(1.0, color='grey', linestyle='--', alpha=0.7)
        axes[0].axhline(0.0, color='grey', linestyle='--', alpha=0.7)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('SNR', fontsize=self.config['xlabel_fontsize'])
        axes[0].set_ylabel('Response', fontsize=self.config['ylabel_fontsize'])
        axes[0].set_title('Response vs SNR', fontsize=self.config['xlabel_fontsize'] + 2)
        axes[0].legend(fontsize=self.config['tick_fontsize'])
        
        # second plot: response_psf vs SNR
        if self.has_psf:
            axes[1].errorbar(data_snr['bin_mean'], data_snr['r11_psf_mean'], yerr=data_snr['r11_psf_err'], 
                            fmt='o-', label='r11_psf', capsize=4, markersize=5, color='blue', alpha=0.7)
            axes[1].errorbar(data_snr['bin_mean'], data_snr['r22_psf_mean'], yerr=data_snr['r22_psf_err'], 
                        fmt='s-', label='r22_psf', capsize=4, markersize=5, color='red', alpha=0.7)
            axes[1].errorbar(data_snr['bin_mean'], data_snr['r12_psf_mean'], yerr=data_snr['r12_psf_err'], 
                        fmt     ='o--', label='r12_psf', capsize=4, markersize=5, color='green', alpha=0.7)
            axes[1].errorbar(data_snr['bin_mean'], data_snr['r21_psf_mean'], yerr=data_snr['r21_psf_err'], 
                        fmt ='s--', label='r21_psf', capsize=4, markersize=5, color='orange', alpha=0.7)
            axes[1].axhline(0.0, color='grey', linestyle='--', alpha=0.7)
            axes[1].set_xscale('log')
            axes[1].set_xlabel('SNR', fontsize=self.config['xlabel_fontsize'])
            axes[1].set_ylabel('Response PSF', fontsize=self.config['ylabel_fontsize'])
            axes[1].set_title('Response PSF vs SNR', fontsize=self.config['xlabel_fontsize'] + 2)
            axes[1].legend(fontsize=self.config['tick_fontsize'])
        
        # third plot: 2D histogram of counts
        pcm, X, Y = plot_counts(axes[2], res_dict['xbin_var'], res_dict['ybin_var'], res_dict['x_bins'], res_dict['y_bins'],
                        self.config, self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"])
        add_colorbar(pcm, fig, axes[2], "counts", self.config)
        
        # fourth plot: 2D histogram of R
        pcm = axes[3].pcolormesh(X, Y, res_dict['R'].T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[3], r"$\langle R \rangle$", self.config, logscale=False)
        optimize_ax(axes[3], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)
        
        # fifth plot: 2D histogram of sigma_e^2
        pcm = axes[4].pcolormesh(X, Y, np.sqrt(res_dict['sigma_e2']).T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[4], r"$\sqrt{\sigma_e^2}$", self.config, logscale=False)
        optimize_ax(axes[4], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)
        
        # sixth plot: 2D histogram of weight
        pcm = axes[5].pcolormesh(X, Y, res_dict['weight'].T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[5], r"$\langle R \rangle^2 / \sigma_e^2$", self.config, logscale=False)
        optimize_ax(axes[5], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)

        # seventh plot: 2D histogram of R11
        pcm = axes[6].pcolormesh(X, Y, res_dict['R11'].T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[6], r"$\langle R_{11} \rangle$", self.config, logscale=False)
        optimize_ax(axes[6], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config) 
        
        # eighth plot: 2D histogram of R22
        pcm = axes[7].pcolormesh(X, Y, res_dict['R22'].T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[7], r"$\langle R_{22} \rangle$", self.config, logscale=False)
        optimize_ax(axes[7], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)
        
        if self.has_psf:
            # ninth plot: 2D histogram of R11_psf
            pcm = axes[8].pcolormesh(X, Y, res_dict['R11_psf'].T, cmap=self.config["cmap"],
                            edgecolors="k", linewidth=self.config["linewidth"])
            add_colorbar(pcm, fig, axes[8], r"$\langle R_{11}^{\rm PSF} \rangle$", self.config, logscale=False)
            optimize_ax(axes[8], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)
            
            # tenth plot: 2D histogram of R22_psf
            pcm = axes[9].pcolormesh(X, Y, res_dict['R22_psf'].T, cmap=self.config["cmap"],
                            edgecolors="k", linewidth=self.config["linewidth"])
            add_colorbar(pcm, fig, axes[9], r"$\langle R_{22}^{\rm PSF} \rangle$", self.config, logscale=False)
            optimize_ax(axes[9], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)
        
        # eleventh plot: 2D histogram of C1
        pcm = axes[10].pcolormesh(X, Y, res_dict['C1'].T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[10], r"$\langle c_1 \rangle$", self.config, logscale=False)
        optimize_ax(axes[10], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)        
        
        # twelfth plot: 2D histogram of C2
        pcm = axes[11].pcolormesh(X, Y, res_dict['C2'].T, cmap=self.config["cmap"],
                         edgecolors="k", linewidth=self.config["linewidth"])
        add_colorbar(pcm, fig, axes[11], r"$\langle c_2 \rangle$", self.config, logscale=False)
        optimize_ax(axes[11], self.config["x_min"], self.config["y_min"], res_dict["x_p"], res_dict["y_p"], self.config)  
        
        plt.tight_layout()
        if self.plot_outfile:
            plt.savefig(self.plot_outfile, dpi=self.config["dpi"])
        
        else:
            plt.show()  
        
    def selection_n_calibration(self, selection_cut, cluster_redshift=None, mcal_shear=0.01, PFS_response_correction=True, R_diagonal=True, constant_Rpsf=False, constant_C_gamma=False):
        selected_catalog, R_S, c_gamma, mean_g1, mean_g2, R_PSF = compute_R_S(
                mcal=self.catalog,
                qual_cuts=selection_cut,
                mcal_shear=mcal_shear,
                cluster_redshift=cluster_redshift,
                overwrite_calibration=True,
                R_diagonal=R_diagonal,
                PFS_response_correction=PFS_response_correction,
                has_psf=self.has_psf
            )
        if not constant_Rpsf:
            R_PSF = None
        if not constant_C_gamma:
            c_gamma = None
        resdict = self.gridder(self.config)
        calibrate_catalog(selected_catalog, resdict, R_S, mean_g1, mean_g2, suffix=f"{self.config['n_bins']}x{self.config['n_bins']}", psf_correction=(PFS_response_correction & self.has_psf), R_PSF=R_PSF, c_gamma=c_gamma)
        self.selected_catalog = selected_catalog
        
        
        return selected_catalog
        
# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def optimize_ax(ax, x_min, y_min, x_max, y_max, cfg):
    """Apply log scaling, limits, tick locators, and axis labels."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max * 1.15)
    ax.set_ylim(y_min, y_max * 1.1)

    locator = LogLocator(base=10.0, subs=[1, 2, 5], numticks=cfg["n_ticks_axis"])
    formatter = ScalarFormatter()

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel('T/T$_{\\rm PSF}$', fontsize=cfg["xlabel_fontsize"])
    ax.set_ylabel('SNR', fontsize=cfg["ylabel_fontsize"])

    # Set tick label size
    ax.tick_params(axis='both', which='major', labelsize=cfg["tick_fontsize"])


def plot_counts(ax, x, y, x_bins, y_bins, cfg, x_min, y_min, x_p, y_p):
    """2D histogram with log color scale."""
    hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    X, Y = np.meshgrid(xedges, yedges)

    pcm = ax.pcolormesh(
        X, Y, hist.T,
        cmap=cfg["cmap"],
        norm=LogNorm(vmin=cfg["lognorm_vmin"]),
        edgecolors="k",
        linewidth=cfg["linewidth"]
    )
    optimize_ax(ax, cfg["x_min"], cfg["y_min"], x_p, y_p, cfg)
    return pcm, X, Y

def add_colorbar(pcm, fig, ax, label, cfg, logscale=True):
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=cfg["xlabel_fontsize"])
    if logscale:
        cbar.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1, 2, 5],
                                                   numticks=cfg["n_ticks_cbar"]))
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    cbar.ax.tick_params(labelsize=cfg["tick_fontsize"])
    return cbar

def assign_weights(catalog, x_bins, y_bins, weights):
    """
    Assign 2D binned weights to each object.

    Parameters
    ----------
    catalog : astropy Table or dict-like
        Catalog with columns "T_noshear" / "Tpsf_noshear" and "s2n_noshear".
    x_bins, y_bins : array
        Bin edges for T/Tpsf and S/N axes.
    weights : 2D array
        Shape (len(x_bins)-1, len(y_bins)-1).

    Returns
    -------
    w_obj : array
        Weight assigned to each object in catalog.
    """
    x_obj = catalog["T_noshear"] / catalog["Tpsf_noshear"]
    y_obj = catalog["s2n_noshear"]

    mask = np.isfinite(x_obj) & np.isfinite(y_obj) & (x_obj > 0) & (y_obj > 0)
    w_obj = np.zeros(len(x_obj))

    x_idx = np.searchsorted(x_bins, x_obj, side='right') - 1
    y_idx = np.searchsorted(y_bins, y_obj, side='right') - 1

    # Clamp to valid bin range (lower → first bin, upper → mega-bin)
    x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
    y_idx = np.clip(y_idx, 0, len(y_bins) - 2)

    w_obj[mask] = weights[x_idx[mask], y_idx[mask]]
    return w_obj


def calibrate_catalog(catalog, grid, R_S, mean_g1, mean_g2, suffix, psf_correction=False,R_PSF=None, c_gamma=None):
    """
    Assign gridded weights and calibrated ellipticities to a catalog.

    New columns written (where {s} = suffix, e.g. '7x7'):
        w_{s}      — R²/σ_e² weight
        w_inv_{s}  — 1/σ_e² weight
        g1_cal_{s} — calibrated g1
        g2_cal_{s} — calibrated g2

    Parameters
    ----------
    catalog : Table
        Catalog to receive new columns (modified in place).
    grid : dict
        Output of compute_grid_calibration.
    R_S : array
        2×2 selection response matrix from compute_R_S.
    mean_g1, mean_g2 : float
        Additive bias to subtract before dividing by response.
    suffix : str
        Column name suffix, e.g. '7x7' or '5x5'.
    """
    x_bins, y_bins = grid['x_bins'], grid['y_bins']

    weight_col = assign_weights(catalog, x_bins, y_bins, grid['weight'])
    sn_col = assign_weights(catalog, x_bins, y_bins, grid['sigma_e2'])
    inv_weight_col = 1.0 / sn_col

    r11_col = assign_weights(catalog, x_bins, y_bins, grid['R11']) + R_S[0, 0]
    r22_col = assign_weights(catalog, x_bins, y_bins, grid['R22']) + R_S[1, 1]
    
    if c_gamma is not None:
        c1_col = c_gamma[0]
        c2_col = c_gamma[1]
    else:
        c1_col = assign_weights(catalog, x_bins, y_bins, grid['C1'])
        c2_col = assign_weights(catalog, x_bins, y_bins, grid['C2'])

    g1_noshear = catalog['g_noshear'][:, 0]  #- c1_col #- mean_g1
    g2_noshear = catalog['g_noshear'][:, 1]  #- c2_col #- mean_g2

    if psf_correction:
        r11_psf_col = assign_weights(catalog, x_bins, y_bins, grid['R11_psf'])
        r22_psf_col = assign_weights(catalog, x_bins, y_bins, grid['R22_psf'])
        paneler = PSFLeakagePanelMaker(e1_gal=g1_noshear, e2_gal=g2_noshear, e1_psf=catalog['gpsf_noshear'][:,0], e2_psf=catalog['gpsf_noshear'][:,1], r11_psf=catalog['r11_psf'], r22_psf=catalog['r22_psf'], weights = catalog['weight'], NBIN=10, MIN_COUNT=20)
        
        
        if R_PSF is not None:
            g1_noshear = g1_noshear - R_PSF[0,0] * catalog['gpsf_noshear'][:,0]
            g2_noshear = g2_noshear - R_PSF[1,1] * catalog['gpsf_noshear'][:,1]
        else:
            g1_noshear = g1_noshear - r11_psf_col * catalog['gpsf_noshear'][:,0]
            g2_noshear = g2_noshear - r22_psf_col * catalog['gpsf_noshear'][:,1]
            print("PSF correction applied using PSFLeakagePanelMaker.")
            g1_noshear_1 = paneler.e1_gal
            g2_noshear_1 = paneler.e2_gal

    g1_cal = np.divide(
        g1_noshear, r11_col,
        out=np.zeros_like(g1_noshear), where=r11_col != 0,
    )
    g2_cal = np.divide(
        g2_noshear, r22_col,
        out=np.zeros_like(g2_noshear), where=r22_col != 0,
    )
    
    if R_PSF is not None:
        g1_cal_1 = np.divide(
            g1_noshear_1, r11_col,
            out=np.zeros_like(g1_noshear_1), where=r11_col != 0,
        )
        g2_cal_1 = np.divide(
            g2_noshear_1, r22_col,
            out=np.zeros_like(g2_noshear_1), where=r22_col != 0,
        )
    

    catalog[f"w_{suffix}"] = weight_col
    catalog[f"w_inv_{suffix}"] = inv_weight_col
    if (f"g1_cal_{suffix}" in catalog.colnames) & (f"g2_cal_{suffix}" in catalog.colnames):
        # raise ValueError(f"Column g1_cal_{suffix} and g2_cal_{suffix} already exist in catalog. Choose a different suffix.")
        catalog[f"g1_cal_{suffix}_psf_corr"] = g1_cal
        catalog[f"g2_cal_{suffix}_psf_corr"] = g2_cal
        catalog[f"g1_cal_{suffix}_psf_corr_v2"] = g1_cal_1
        catalog[f"g2_cal_{suffix}_psf_corr_v2"] = g2_cal_1
    else:
        catalog[f"g1_cal_{suffix}"] = g1_cal
        catalog[f"g2_cal_{suffix}"] = g2_cal
    catalog[f'g1_uncl_{suffix}'] = g1_noshear
    catalog[f'g2_uncl_{suffix}'] = g2_noshear