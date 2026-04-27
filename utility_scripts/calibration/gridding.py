"""
Weight and shear response diagnostics for metacalibration catalogs.

Produces a 3×2 panel figure (counts, <R>, R11, R22, sigma_e, weight)
binned in T/Tpsf vs S/N space, then writes per-object calibrated
ellipticities and weights back to the catalog file.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FuncFormatter
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from astropy.table import Table

from superbit_lensing.diagnostics import compute_R_S

# ================================================================== #
#  Configuration
# ================================================================== #
QUAL_CUTS = {
    'min_Tpsf': 1.0,
    'max_sn': 1000,
    'min_sn': 10.0,
    'min_T': 0.0,
    'max_T': 100,
    # 'admom_flag': 1,
    # 'min_admom_sigma': 0.0,
}

CONFIG = {
    # Binning
    "n_bins": 7,
    "append_high_bin": 1e10,

    # Cuts
    "percentile_cut": 95,
    "x_min": 1.0,
    "y_min": 10,

    # Plotting
    "cmap": "magma",
    "dpi": 600,
    "linewidth": 0.003,
    "lognorm_vmin": 20,

    # Fonts and ticks
    "xlabel_fontsize": 12,
    "ylabel_fontsize": 12,
    "tick_fontsize": 9.5,
    "n_ticks_axis": 15,
    "n_ticks_cbar": 5,
}

CATALOG_PATH = "/scratch/sa.saha/data_final/mega_annular_catalog_20260424_111327.fits"
MCAL_SHEAR = 0.01


# ================================================================== #
#  Helper functions — plotting
# ================================================================== #
def optimize_ax(ax, x_min, y_min, x_max, y_max, cfg):
    """Apply log scaling, limits, tick locators, and axis labels."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max * 1.15)
    ax.set_ylim(y_min, y_max * 1.1)

    plain_fmt = FuncFormatter(lambda x, _: f'{x:g}')
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(
            LogLocator(base=10.0, subs=[1, 2, 5], numticks=cfg["n_ticks_axis"])
        )
        axis.set_major_formatter(plain_fmt)
        axis.set_minor_formatter(FuncFormatter(lambda x, _: ''))

    ax.set_xlabel(r'T/T$_{\rm PSF}$', fontsize=cfg["xlabel_fontsize"])
    ax.set_ylabel('SNR', fontsize=cfg["ylabel_fontsize"])
    ax.tick_params(axis='both', which='major', labelsize=cfg["tick_fontsize"])


def plot_counts(ax, x, y, x_bins, y_bins, cfg, x_min, y_min, x_p, y_p):
    """2D histogram with log colour scale."""
    hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    X, Y = np.meshgrid(xedges, yedges)

    pcm = ax.pcolormesh(
        X, Y, hist.T,
        cmap=cfg["cmap"],
        norm=LogNorm(vmin=cfg["lognorm_vmin"]),
        edgecolors="k",
        linewidth=cfg["linewidth"],
    )
    optimize_ax(ax, x_min, y_min, x_p, y_p, cfg)
    return pcm, X, Y


def plot_statistic(ax, X, Y, data, cfg, x_min, y_min, x_p, y_p,
                   use_lognorm=False):
    """Plot a 2D binned statistic on a pcolormesh grid."""
    kwargs = dict(cmap=cfg["cmap"], edgecolors="k", linewidth=cfg["linewidth"])
    if use_lognorm:
        kwargs["norm"] = LogNorm()
    pcm = ax.pcolormesh(X, Y, data.T, **kwargs)
    optimize_ax(ax, x_min, y_min, x_p, y_p, cfg)
    return pcm


def add_colorbar(pcm, fig, ax, label, cfg, logscale=True):
    """Attach a colorbar to a panel."""
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=cfg["xlabel_fontsize"])
    if logscale:
        cbar.ax.yaxis.set_major_locator(
            LogLocator(base=10.0, subs=[1, 2, 5], numticks=cfg["n_ticks_cbar"])
        )
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:g}')
        )
    cbar.ax.tick_params(labelsize=cfg["tick_fontsize"])
    return cbar


# ================================================================== #
#  Helper functions — analysis
# ================================================================== #
def binned_selection_response(cat, x_bins, y_bins, mcal_shear):
    """Compute R11_S and R22_S using sheared quantities in gridded bins."""

    def mean_in_bin(x1name, x2name, yname, gcol):
        x = cat[x1name] / cat[x2name]
        y = cat[yname]
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y, g = x[mask], y[mask], cat[gcol][mask]
        stat0, _, _, _ = binned_statistic_2d(
            x, y, g[:, 0], statistic='median', bins=[x_bins, y_bins])
        stat1, _, _, _ = binned_statistic_2d(
            x, y, g[:, 1], statistic='median', bins=[x_bins, y_bins])
        return stat0, stat1

    g1_1p, _ = mean_in_bin("T_1p", "Tpsf_1p", "s2n_1p", "g_noshear")
    g1_1m, _ = mean_in_bin("T_1m", "Tpsf_1m", "s2n_1m", "g_noshear")
    _, g2_2p = mean_in_bin("T_2p", "Tpsf_2p", "s2n_2p", "g_noshear")
    _, g2_2m = mean_in_bin("T_2m", "Tpsf_2m", "s2n_2m", "g_noshear")

    R11_S = (g1_1p - g1_1m) / (2.0 * mcal_shear)
    R22_S = (g2_2p - g2_2m) / (2.0 * mcal_shear)

    return R11_S, R22_S, (R11_S + R22_S) / 2


def assign_weights(cat, x_bins, y_bins, weights):
    """
    Assign 2D binned weights to each object.

    Parameters
    ----------
    cat : astropy Table or dict-like
        Catalog with columns "T_noshear" / "Tpsf_noshear" and "s2n_noshear".
    x_bins, y_bins : array
        Bin edges for T/Tpsf and S/N axes.
    weights : 2D array
        Shape (len(x_bins)-1, len(y_bins)-1).

    Returns
    -------
    w_obj : array
        Weight assigned to each object in cat.
    """
    x_obj = cat["T_noshear"] / cat["Tpsf_noshear"]
    y_obj = cat["s2n_noshear"]

    mask = np.isfinite(x_obj) & np.isfinite(y_obj) & (x_obj > 0) & (y_obj > 0)
    w_obj = np.zeros(len(x_obj))

    x_idx = np.searchsorted(x_bins, x_obj, side='right') - 1
    y_idx = np.searchsorted(y_bins, y_obj, side='right') - 1

    # Clamp to valid bin range (lower → first bin, upper → mega-bin)
    x_idx = np.clip(x_idx, 0, len(x_bins) - 2)
    y_idx = np.clip(y_idx, 0, len(y_bins) - 2)

    w_obj[mask] = weights[x_idx[mask], y_idx[mask]]
    return w_obj


def compute_grid_calibration(x_clean, y_clean, r11_clean, r22_clean,
                             values_e, cat, n_bins, cfg):
    """
    Build log-spaced bin edges and compute smoothed response / weight maps.

    Parameters
    ----------
    x_clean, y_clean : array
        Cleaned T/Tpsf and S/N arrays.
    r11_clean, r22_clean : array
        Per-galaxy r11/r22 values (already masked to match x/y_clean).
    values_e : array
        Per-galaxy (e1² + e2²)/2 (already masked).
    cat : Table
        Full catalog (used for binned selection response).
    n_bins : int
        Number of log-spaced bins per axis.
    cfg : dict
        CONFIG dictionary.

    Returns
    -------
    grid : dict
        Keys: 'x_bins', 'y_bins', 'R11', 'R22', 'R', 'sigma_e2', 'weight',
              'RS', 'X', 'Y'.
    """
    x_min, y_min = cfg["x_min"], cfg["y_min"]
    x_p = np.percentile(x_clean, cfg["percentile_cut"])
    y_p = np.percentile(y_clean, cfg["percentile_cut"])

    x_bins = np.append(
        np.logspace(np.log10(x_min), np.log10(x_p), n_bins),
        cfg["append_high_bin"],
    )
    y_bins = np.append(
        np.logspace(np.log10(y_min), np.log10(y_p), n_bins),
        cfg["append_high_bin"],
    )

    R11, _, _, _ = binned_statistic_2d(
        x_clean, y_clean, r11_clean,
        statistic='median', bins=[x_bins, y_bins],
    )
    R22, _, _, _ = binned_statistic_2d(
        x_clean, y_clean, r22_clean,
        statistic='median', bins=[x_bins, y_bins],
    )
    R = (R11 + R22) / 2

    _, _, RS = binned_selection_response(cat, x_bins, y_bins, MCAL_SHEAR)
    # R = R + RS

    R11 = gaussian_filter(R11, sigma=2)
    R22 = gaussian_filter(R22, sigma=2)
    R = gaussian_filter(R, sigma=2)

    sigma_e2, _, _, _ = binned_statistic_2d(
        x_clean, y_clean, values_e,
        statistic='mean', bins=[x_bins, y_bins],
    )
    weight = R**2 / sigma_e2

    # Mesh for pcolormesh (from histogram edges)
    hist, xedges, yedges = np.histogram2d(
        x_clean, y_clean, bins=[x_bins, y_bins],
    )
    X, Y = np.meshgrid(xedges, yedges)

    return {
        'x_bins': x_bins, 'y_bins': y_bins,
        'R11': R11, 'R22': R22, 'R': R,
        'sigma_e2': sigma_e2, 'weight': weight,
        'RS': RS, 'X': X, 'Y': Y,
        'x_p': x_p, 'y_p': y_p, 'hist': hist,
    }


def calibrate_catalog(annular_cat, grid, R_S, mean_g1, mean_g2, suffix):
    """
    Assign gridded weights and calibrated ellipticities to a catalog.

    New columns written (where {s} = suffix, e.g. '7x7'):
        w_{s}      — R²/σ_e² weight
        w_inv_{s}  — 1/σ_e² weight
        g1_cal_{s} — calibrated g1
        g2_cal_{s} — calibrated g2

    Parameters
    ----------
    annular_cat : Table
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

    weight_col = assign_weights(annular_cat, x_bins, y_bins, grid['weight'])
    sn_col = assign_weights(annular_cat, x_bins, y_bins, grid['sigma_e2'])
    inv_weight_col = 1.0 / sn_col

    r11_col = assign_weights(annular_cat, x_bins, y_bins, grid['R11']) + R_S[0, 0]
    r22_col = assign_weights(annular_cat, x_bins, y_bins, grid['R22']) + R_S[1, 1]

    g1_noshear = annular_cat['g_noshear'][:, 0] - mean_g1
    g2_noshear = annular_cat['g_noshear'][:, 1] - mean_g2

    g1_cal = np.divide(
        g1_noshear, r11_col,
        out=np.zeros_like(g1_noshear), where=r11_col != 0,
    )
    g2_cal = np.divide(
        g2_noshear, r22_col,
        out=np.zeros_like(g2_noshear), where=r22_col != 0,
    )

    annular_cat[f"w_{suffix}"] = weight_col
    annular_cat[f"w_inv_{suffix}"] = inv_weight_col
    annular_cat[f"g1_cal_{suffix}"] = g1_cal
    annular_cat[f"g2_cal_{suffix}"] = g2_cal


# ================================================================== #
#  Main
# ================================================================== #
def main():
    cfg = CONFIG

    # -------------------------------------------------------------- #
    #  Load catalog and apply initial star cut
    # -------------------------------------------------------------- #
    cat = Table.read(CATALOG_PATH)
    star = cat['CLASS_STAR'] > 0.1
    print(f"Objects discarding as stars: {np.sum(star)}")
    cat = cat[~star]

    # -------------------------------------------------------------- #
    #  Metacalibration response and selection bias
    # -------------------------------------------------------------- #
    selected_catalog, R_S, c_total, mean_g1, mean_g2 = compute_R_S(
        mcal=cat,
        qual_cuts=QUAL_CUTS,
        mcal_shear=MCAL_SHEAR,
    )

    # -------------------------------------------------------------- #
    #  Prepare binning variables (with stricter star cut for plots)
    # -------------------------------------------------------------- #
    e1 = cat["g_noshear"][:, 0]
    e2 = cat["g_noshear"][:, 1]

    xbin_var = cat["T_noshear"] / cat["Tpsf_noshear"]
    ybin_var = cat["s2n_noshear"]

    # star_strict = cat['CLASS_STAR'] > 0.5
    mask = (
        np.isfinite(xbin_var) & np.isfinite(ybin_var)
        & (xbin_var > 0) & (ybin_var > 0) # & ~star_strict
    )
    x_clean, y_clean = xbin_var[mask], ybin_var[mask]

    print(f"{cfg['percentile_cut']}th percentile for T/Tpsf: "
          f"{np.percentile(x_clean, cfg['percentile_cut']):.2f}")
    print(f"{cfg['percentile_cut']}th percentile for S/N: "
          f"{np.percentile(y_clean, cfg['percentile_cut']):.2f}")
    print(f"Total galaxies after cleaning: {len(x_clean)}")

    # Quantities that both grids need (already masked)
    r11_clean = cat["r11"][mask]
    r22_clean = cat["r22"][mask]
    values_e = (e1[mask]**2 + e2[mask]**2) / 2

    # -------------------------------------------------------------- #
    #  Compute gridded calibration for both resolutions
    # -------------------------------------------------------------- #
    grid_7 = compute_grid_calibration(
        x_clean, y_clean, r11_clean, r22_clean, values_e,
        cat, n_bins=7, cfg=cfg,
    )
    grid_5 = compute_grid_calibration(
        x_clean, y_clean, r11_clean, r22_clean, values_e,
        cat, n_bins=5, cfg=cfg,
    )

    print(f"\n7×7 grid: {len(grid_7['x_bins'])-1} x {len(grid_7['y_bins'])-1} bins")
    print(f"5×5 grid: {len(grid_5['x_bins'])-1} x {len(grid_5['y_bins'])-1} bins")

    # -------------------------------------------------------------- #
    #  Figure: 3×2 diagnostic panels (using the 7×7 grid)
    #  Row 1 — Counts, <R>
    #  Row 2 — R_11,   R_22
    #  Row 3 — σ_e,    Weight
    # -------------------------------------------------------------- #
    g = grid_7
    x_min, y_min = cfg["x_min"], cfg["y_min"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), dpi=cfg["dpi"])
    axes = axes.ravel()
    common = dict(cfg=cfg, x_min=x_min, y_min=y_min, x_p=g['x_p'], y_p=g['y_p'])

    # Panel 0 — Counts
    pcm, X, Y = plot_counts(
        axes[0], x_clean, y_clean, g['x_bins'], g['y_bins'],
        cfg, x_min, y_min, g['x_p'], g['y_p'],
    )
    add_colorbar(pcm, fig, axes[0], "count", cfg)

    # Panel 1 — <R>
    pcm = plot_statistic(axes[1], g['X'], g['Y'], g['R'], **common)
    add_colorbar(pcm, fig, axes[1], r"$\langle R \rangle$", cfg, logscale=False)

    # Panel 2 — R_11
    pcm = plot_statistic(axes[2], g['X'], g['Y'], g['R11'], **common)
    add_colorbar(pcm, fig, axes[2], r"$\langle R_{11} \rangle$", cfg, logscale=False)

    # Panel 3 — R_22
    pcm = plot_statistic(axes[3], g['X'], g['Y'], g['R22'], **common)
    add_colorbar(pcm, fig, axes[3], r"$\langle R_{22} \rangle$", cfg, logscale=False)

    # Panel 4 — sqrt(sigma_e^2)
    pcm = plot_statistic(axes[4], g['X'], g['Y'], np.sqrt(g['sigma_e2']), **common)
    add_colorbar(pcm, fig, axes[4], r"$\sqrt{\sigma_e^2}$", cfg)

    # Panel 5 — Weight
    pcm = plot_statistic(axes[5], g['X'], g['Y'], g['weight'], **common)
    add_colorbar(pcm, fig, axes[5], "weight", cfg, logscale=False)

    plt.subplots_adjust(wspace=0.45, hspace=0.35)
    plt.savefig("weight_response.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

    # -------------------------------------------------------------- #
    #  Outlier statistics (based on 7×7 percentiles)
    # -------------------------------------------------------------- #
    x_p, y_p = g['x_p'], g['y_p']
    n_outliers_x = np.sum(x_clean > x_p)
    n_outliers_y = np.sum(y_clean > y_p)
    n_outliers_both = np.sum((x_clean > x_p) & (y_clean > y_p))

    pct = cfg["percentile_cut"]
    print("\nOutlier statistics:")
    print(f"Galaxies with T/Tpsf > {pct}%: "
          f"{n_outliers_x} ({100 * n_outliers_x / len(x_clean):.1f}%)")
    print(f"Galaxies with S/N > {pct}%: "
          f"{n_outliers_y} ({100 * n_outliers_y / len(y_clean):.1f}%)")
    print(f"Galaxies with both > {pct}%: {n_outliers_both}")

    # -------------------------------------------------------------- #
    #  Write calibrated columns back to the catalog
    #  Columns:  w_7x7, w_inv_7x7, g1_cal_7x7, g2_cal_7x7
    #            w_5x5, w_inv_5x5, g1_cal_5x5, g2_cal_5x5
    # -------------------------------------------------------------- #
    annular_cat = Table.read(CATALOG_PATH)

    calibrate_catalog(annular_cat, grid_7, R_S, mean_g1, mean_g2, suffix="7x7")
    calibrate_catalog(annular_cat, grid_5, R_S, mean_g1, mean_g2, suffix="5x5")

    annular_cat.write(CATALOG_PATH, overwrite=True)
    print(f"\nUpdated file saved: {CATALOG_PATH}")


if __name__ == "__main__":
    main()