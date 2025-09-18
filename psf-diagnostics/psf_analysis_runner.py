"""
PSF Analysis Pipeline - Main Runner
Executes the complete PSF analysis with all configurations at the top
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from ngmix.shape import e1e2_to_g1g2
from superbit_lensing.utils import ClusterShearCorrelation, RhoStats
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# Import custom functions from the functions module
from psf_analysis_functions import (
    _get_scalar, get_median_radec, minmax_norm, z_norm, maybe_smooth,
    bin_statistic_2d, binned_err_stats, format_colorbar,
    build_exposures_list, compute_global_metrics, compute_per_cluster_metrics
)

# ============================================================================
# =========================== MAIN CONFIGURATION =============================
# ============================================================================

# Output folder for all plots
OUTPUT_FOLDER = "psf_analysis_plots"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ----------------------------- CATALOG PATH ---------------------------------
#catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem.fits" # order 3 (old)
#catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem_20250815_115222.fits" # order 3 (new)
#catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem_20250813_165650.fits" # order 2 (new)
#catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem_20250817_110445_temp.fits" # order 4 (new)
catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem_20250819_182300.fits" # order 5 (new)
#catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem_20250820_180134.fits" # order 6 (new)
#catalog_path = "/scratch/sa.saha/data/all_clusters_star_catalog_withem_20250911_190003.fits" # order 5 (with better star selection)

# ---------------------- EXPOSURE SELECTION CONFIG ---------------------------
SELECTION_SCOPE = 'global'   # 'per_cluster' or 'global'
N_BEST_EXPOSURES = 10        # When per_cluster: best exposures per cluster
N_GLOBAL_EXPOSURES = 800     # When global: best exposures total
SELECTION_METRIC = 'COMBINED' # 'FWHM_MEDIAN', 'FWHM_STD', 'N_GOOD_STARS', or 'COMBINED'
COMBINED_ALPHA = 0.5          # Weight of FWHM_MEDIAN (alpha) vs FWHM_STD (1-alpha)
COMBINED_METHOD = 'minmax'    # 'minmax' or 'zscore'
LOWER_IS_BETTER = True        # For FWHM-like metrics yes, for N_GOOD_STARS set False

# ---------------------- BINNING CONFIGURATION -------------------------------
TARGET_NBINS = 20            # Number of bins in smaller dimension
MIN_STARS_PER_BIN = 3        # Minimum stars per bin (bins with fewer will be masked)
APPLY_GAUSSIAN = True        # Apply Gaussian smoothing to binned data
GAUSS_SIGMA = 1.0            # Standard deviation of Gaussian kernel (in pixels)
SHOW_MODEL_ROW = False       # Set True to include model row in binned comparison

# ---------------------- COLUMN NAMES CONFIGURATION --------------------------
COLUMN_CONFIG = {
    # Filter columns
    'set_column': 'SET',
    'flux_column': 'FLUX_AUTO',
    
    # Position columns
    'x_column': 'XWIN_IMAGE',
    'y_column': 'YWIN_IMAGE',
    
    # Observed shape columns
    'e1_obs': 'e1_admom_obs',
    'e2_obs': 'e2_admom_obs',
    'T_obs': 'T_admom_obs',
    
    # Model shape columns
    'e1_model': 'e1_admom_model',
    'e2_model': 'e2_admom_model',
    'T_model': 'T_admom_model'
}

# ---------------------- COLOR SCALE LIMITS -----------------------------------
COLOR_LIMITS = {
    'e_lim': 0.06,        # for e1 and e2
    'e_res_lim': 0.01,    # for e1 and e2 residuals
    'T_res_lim': 0.01     # for T residuals
}

# ---------------------- RHO STATS CONFIGURATION -----------------------------
RHO_STATS_CONFIG = {
    'catalog_file': "/projects/mccleary_group/saha/codes/.codes/psf_diagnostics/psfex_v2/truth_file/stacked_annular_combined.fits",
    'M500': 6.31e14,      # Msun/h
    'z_cluster': 0.246,
    'z_min_offset': 0.05,
    'pixel_scale': 0.1408
}

# ---------------------- OUTPUT CONFIGURATION --------------------------------
OUTPUT_DPI = 600

# ---------------------- PLOT FONT SIZES --------------------------------------
FONT_SIZE_MAIN = 16
FONT_SIZE_SMALL = 10
FONT_SIZE_LARGE = 18

# ============================================================================
# ========================= MAIN PROCESSING ==================================
# ============================================================================

plt.rcParams.update({'font.size': FONT_SIZE_MAIN})

print(f"Loading catalog from: {catalog_path}")
catalog = Table.read(catalog_path)
print(f"Total objects in catalog: {len(catalog)}")

# Filter
flux_mask = catalog['FLUX_AUTO'] > 0
catalog_filtered = catalog[flux_mask]
print(f"Objects with FLUX_AUTO > 0: {len(catalog_filtered)} (removed {len(catalog)-len(catalog_filtered)})")

# Validate columns
sel_upper = SELECTION_METRIC.upper()
if sel_upper in ('FWHM_MEDIAN', 'FWHM_STD', 'N_GOOD_STARS'):
    required_cols = [sel_upper]
elif sel_upper == 'COMBINED':
    required_cols = ['FWHM_MEDIAN', 'FWHM_STD']
else:
    required_cols = [SELECTION_METRIC]

missing = [c for c in required_cols if c not in catalog_filtered.colnames]
if missing:
    raise ValueError(f"Required column(s) missing from catalog: {missing}")

print(f"Selection scope: {SELECTION_SCOPE}; metric: {SELECTION_METRIC}")

# Build exposures list
unique_clusters = np.unique(catalog_filtered['CLUSTER_NAME'])
exposures = build_exposures_list(catalog_filtered, unique_clusters)
print(f"Total unique exposures found: {len(exposures)}")

# Compute metric_value(s)
if SELECTION_SCOPE == 'global':
    # Compute metric globally
    exposures = compute_global_metrics(exposures, sel_upper, COMBINED_METHOD, COMBINED_ALPHA)
    
    # Sort globally and pick top N_GLOBAL_EXPOSURES
    def sort_key_global(x):
        m = x['metric_value']
        if np.isnan(m):
            return np.inf if LOWER_IS_BETTER else -np.inf
        return m if LOWER_IS_BETTER else -m

    exposures_sorted = sorted(exposures, key=sort_key_global)
    selected_exposures = exposures_sorted[:min(N_GLOBAL_EXPOSURES, len(exposures_sorted))]
    selected_set = {(e['cluster'], e['exp_num']) for e in selected_exposures}

    print(f"Selected {len(selected_exposures)} exposures globally (requested {N_GLOBAL_EXPOSURES}).")

    # Build keep mask
    keep_mask = np.zeros(len(catalog_filtered), dtype=bool)
    for (cluster, exp) in selected_set:
        keep_mask |= (catalog_filtered['CLUSTER_NAME'] == cluster) & (catalog_filtered['EXP_NUM'] == exp)

    # Build cluster_summary
    cluster_summary = []
    for cluster in unique_clusters:
        sel_for_cluster = [e for e in selected_exposures if e['cluster'] == cluster]
        cluster_summary.append({
            'cluster': cluster,
            'total_exps': len(np.unique(catalog_filtered[catalog_filtered['CLUSTER_NAME'] == cluster]['EXP_NUM'])),
            'selected_exps': [e['exp_num'] for e in sel_for_cluster],
            'n_selected': len(sel_for_cluster),
            'total_objects_kept': sum(((catalog_filtered['CLUSTER_NAME'] == cluster) & (catalog_filtered['EXP_NUM'] == e['exp_num'])).sum() for e in sel_for_cluster)
        })

else:
    # PER-CLUSTER selection
    keep_mask = np.zeros(len(catalog_filtered), dtype=bool)
    cluster_summary = []
    for cluster in unique_clusters:
        cluster_mask = catalog_filtered['CLUSTER_NAME'] == cluster
        cluster_data = catalog_filtered[cluster_mask]
        
        # Compute metrics for this cluster
        exp_info = compute_per_cluster_metrics(cluster_data, sel_upper, COMBINED_METHOD, COMBINED_ALPHA)
        
        def sort_key(x):
            m = x['metric_value']
            if np.isnan(m):
                return np.inf if LOWER_IS_BETTER else -np.inf
            return m if LOWER_IS_BETTER else -m

        exp_info_sorted = sorted(exp_info, key=sort_key)
        best_exps = exp_info_sorted[:min(N_BEST_EXPOSURES, len(exp_info_sorted))]
        best_exp_nums = [e['exp_num'] for e in best_exps]
        for exp_num in best_exp_nums:
            keep_mask |= (catalog_filtered['CLUSTER_NAME'] == cluster) & (catalog_filtered['EXP_NUM'] == exp_num)

        summary_dict = {
            'cluster': cluster,
            'total_exps': len(np.unique(cluster_data['EXP_NUM'])),
            'selected_exps': best_exp_nums,
            'n_selected': len(best_exp_nums),
            'total_objects_kept': sum(((catalog_filtered['CLUSTER_NAME'] == cluster) & (catalog_filtered['EXP_NUM'] == e)).sum() for e in best_exp_nums)
        }
        cluster_summary.append(summary_dict)

# Apply mask
final_catalog = catalog_filtered[keep_mask]
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Original exposures: {len(exposures)}")
print(f"Final catalog size: {len(final_catalog)} rows")
if SELECTION_SCOPE == 'global':
    print(f"Requested global exposures: {N_GLOBAL_EXPOSURES}; selected: {len(selected_exposures)}")
else:
    print(f"Requested per-cluster exposures: {N_BEST_EXPOSURES} each")

exps_per_cluster = [s['n_selected'] for s in cluster_summary]
unique, counts = np.unique(exps_per_cluster, return_counts=True)

# ============================================================================
# ================================ PLOTTING ==================================
# ============================================================================

# ---------------------- PLOT 1: EXPOSURE SELECTION -------------------------
metric_lookup = {(e['cluster'], e['exp_num']): e['metric_value'] for e in exposures}
fwhm_median_lookup = {(e['cluster'], e['exp_num']): e['fwhm_median'] for e in exposures}
fwhm_std_lookup = {(e['cluster'], e['exp_num']): e['fwhm_std'] for e in exposures}

selected_pairs = set()
if SELECTION_SCOPE == 'global':
    selected_pairs = {(e['cluster'], e['exp_num']) for e in selected_exposures}
else:
    selected_pairs = {(s['cluster'], e) for s in cluster_summary for e in s['selected_exps']}

all_med = []
all_std = []
sel_med = []
sel_std = []

for (cluster, exp), mv in metric_lookup.items():
    med = fwhm_median_lookup.get((cluster, exp), np.nan)
    std = fwhm_std_lookup.get((cluster, exp), np.nan)
    if np.isnan(med) or np.isnan(std):
        continue
    all_med.append(med); all_std.append(std)
    if (cluster, exp) in selected_pairs:
        sel_med.append(med); sel_std.append(std)

fig, (ax1) = plt.subplots(1, 1, figsize=(13, 10))
ax1.scatter(all_med, all_std, alpha=0.4, s=18, label='All exposures')
if sel_med:
    ax1.scatter(sel_med, sel_std, alpha=0.95, s=50, edgecolors='black', linewidths=0.6, label='Selected')
ax1.set_xlim(right=0.8)
ax1.set_xlabel('FWHM_MEDIAN (arcsec)'); ax1.set_ylabel('FWHM_STD (arcsec)')
ax1.set_title('FWHM_MEDIAN vs FWHM_STD (exposures)')
ax1.legend(); ax1.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'exposure_selection_final.pdf'), dpi=150, bbox_inches='tight')
plt.show()

print("\nDone. Final catalog ready for next processing step.")

# ---------------------- PLOT 2: PSF BINNED COMPARISON ----------------------
test_mask = (
    final_catalog[COLUMN_CONFIG['set_column']] == 'test'
) & (final_catalog[COLUMN_CONFIG['flux_column']] > 0) & (final_catalog['MAG_AUTO']>16.5) & (final_catalog['SNR_WIN']>20)
test_catalog = final_catalog[test_mask]
print(f"Test stars with {COLUMN_CONFIG['flux_column']} > 0: {len(test_catalog)}")

# Get positions
x = test_catalog[COLUMN_CONFIG['x_column']]
y = test_catalog[COLUMN_CONFIG['y_column']]

# Check the actual ranges
x_min, x_max = np.nanmin(x), np.nanmax(x)
y_min, y_max = np.nanmin(y), np.nanmax(y)
x_range = x_max - x_min
y_range = y_max - y_min

print(f"\nX range: {x_min:.1f} to {x_max:.1f} (range: {x_range:.1f} pixels)")
print(f"Y range: {y_min:.1f} to {y_max:.1f} (range: {y_range:.1f} pixels)")
print(f"Aspect ratio (X/Y): {x_range/y_range:.2f}")

# Calculate bins to ensure square bins
target_bin_size = min(x_range, y_range) / TARGET_NBINS
nbins_x = int(np.round(x_range / target_bin_size))
nbins_y = int(np.round(y_range / target_bin_size))

x_edges = np.linspace(x_min, x_max, nbins_x + 1)
y_edges = np.linspace(y_min, y_max, nbins_y + 1)

x_bin_size = x_range / nbins_x
y_bin_size = y_range / nbins_y

print(f"\nTarget bin size: {target_bin_size:.1f} pixels")
print(f"Actual bin sizes:")
print(f"X bin size: {x_bin_size:.1f} pixels")
print(f"Y bin size: {y_bin_size:.1f} pixels")
print(f"Number of bins: {nbins_x} x {nbins_y}")
print(f"Bin size ratio (should be ~1 for square bins): {x_bin_size/y_bin_size:.3f}")

e1_obs = test_catalog[COLUMN_CONFIG['e1_obs']]
e2_obs = test_catalog[COLUMN_CONFIG['e2_obs']]
T_obs = test_catalog[COLUMN_CONFIG['T_obs']]

e1_model = test_catalog[COLUMN_CONFIG['e1_model']]
e2_model = test_catalog[COLUMN_CONFIG['e2_model']]
T_model = test_catalog[COLUMN_CONFIG['T_model']]

# Changing e1, e2 to g1, g2
e1_obs, e2_obs = e1e2_to_g1g2(e1_obs, e2_obs)
e1_model, e2_model = e1e2_to_g1g2(e1_model, e2_model)

# Calculate residuals
e1_residual = e1_obs - e1_model
e2_residual = e2_obs - e2_model
T_residual = T_obs - T_model

# Calculate binned statistics with smoothing
e1_obs_binned   = maybe_smooth(bin_statistic_2d(x, y, e1_obs, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)
e2_obs_binned   = maybe_smooth(bin_statistic_2d(x, y, e2_obs, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)
T_obs_binned    = maybe_smooth(bin_statistic_2d(x, y, T_obs, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)

e1_model_binned = maybe_smooth(bin_statistic_2d(x, y, e1_model, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)
e2_model_binned = maybe_smooth(bin_statistic_2d(x, y, e2_model, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)
T_model_binned  = maybe_smooth(bin_statistic_2d(x, y, T_model, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)

e1_res_binned   = maybe_smooth(bin_statistic_2d(x, y, e1_residual, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)
e2_res_binned   = maybe_smooth(bin_statistic_2d(x, y, e2_residual, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)
T_res_binned    = maybe_smooth(bin_statistic_2d(x, y, T_residual, x_edges, y_edges, MIN_STARS_PER_BIN), APPLY_GAUSSIAN, GAUSS_SIGMA)

# Plotting
nrows = 3 if SHOW_MODEL_ROW else 2
fig_width = 15
cell_aspect = nbins_y / nbins_x
fig_height = fig_width * (nrows / 3) * cell_aspect

fig, axes = plt.subplots(nrows, 3, figsize=(fig_width, fig_height), constrained_layout=True)
axes = np.array(axes).reshape(nrows, 3)

e_lim = COLOR_LIMITS['e_lim']
e_res_lim = COLOR_LIMITS['e_res_lim']
T_res_lim = COLOR_LIMITS['T_res_lim']

# Observed row
im1 = axes[0,0].imshow(e1_obs_binned.T, origin='lower', cmap='RdBu_r', 
                       vmin=-e_lim, vmax=e_lim, aspect='equal',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[0,0].set_aspect('equal')
format_colorbar(im1, axes[0,0], -e_lim, e_lim)

im2 = axes[0,1].imshow(e2_obs_binned.T, origin='lower', cmap='RdBu_r',
                       vmin=-e_lim, vmax=e_lim, aspect='equal',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[0,1].set_aspect('equal')
format_colorbar(im2, axes[0,1], -e_lim, e_lim)

im3 = axes[0,2].imshow(T_obs_binned.T, origin='lower', cmap='viridis', aspect='equal',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[0,2].set_aspect('equal')
format_colorbar(im3, axes[0,2], np.nanmin(T_obs_binned), np.nanmax(T_obs_binned))

# Model row (if enabled)
if SHOW_MODEL_ROW:
    im4 = axes[1,0].imshow(e1_model_binned.T, origin='lower', cmap='RdBu_r',
                           vmin=-e_lim, vmax=e_lim, aspect='equal',
                           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    axes[1,0].set_aspect('equal')
    format_colorbar(im4, axes[1,0], -e_lim, e_lim)

    im5 = axes[1,1].imshow(e2_model_binned.T, origin='lower', cmap='RdBu_r',
                           vmin=-e_lim, vmax=e_lim, aspect='equal',
                           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    axes[1,1].set_aspect('equal')
    format_colorbar(im5, axes[1,1], -e_lim, e_lim)

    im6 = axes[1,2].imshow(T_model_binned.T, origin='lower', cmap='viridis', aspect='equal',
                           extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    axes[1,2].set_aspect('equal')
    format_colorbar(im6, axes[1,2], np.nanmin(T_model_binned), np.nanmax(T_model_binned))

# Residual row
row_idx = 2 if SHOW_MODEL_ROW else 1

im7 = axes[row_idx,0].imshow(e1_res_binned.T, origin='lower', cmap='RdBu_r', 
                             vmin=-e_res_lim, vmax=e_res_lim, aspect='equal',
                             extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[row_idx,0].set_aspect('equal')
format_colorbar(im7, axes[row_idx,0], -e_res_lim, e_res_lim)

im8 = axes[row_idx,1].imshow(e2_res_binned.T, origin='lower', cmap='RdBu_r', 
                             vmin=-e_res_lim, vmax=e_res_lim, aspect='equal',
                             extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[row_idx,1].set_aspect('equal')
format_colorbar(im8, axes[row_idx,1], -e_res_lim, e_res_lim)

im9 = axes[row_idx,2].imshow(T_res_binned.T, origin='lower', cmap='RdBu_r',
                             vmin=-T_res_lim, vmax=T_res_lim, aspect='equal',
                             extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
axes[row_idx,2].set_aspect('equal')
format_colorbar(im9, axes[row_idx,2], -T_res_lim, T_res_lim)

# Axis labels
for i in range(nrows):
    for j in range(3):
        if i == nrows-1:
            axes[i, j].set_xlabel('X [pixels]')
        else:
            axes[i, j].set_xticklabels([])
        if j == 0:
            axes[i, j].set_ylabel('Y [pixels]')
        else:
            axes[i, j].set_yticklabels([])

# Column titles
col_titles = [r"$e_1$", r"$e_2$", r"$T$"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, pad=10)

# Row labels
if SHOW_MODEL_ROW:
    row_labels = ["Observed", "Model", "Residual"]
    y_positions = [0.83, 0.50, 0.17]
else:
    row_labels = ["Observed", "Residual"]
    y_positions = [0.765, 0.28]

for label, ypos in zip(row_labels, y_positions):
    fig.text(1.00, ypos, label,
             rotation=270, va="center", ha="center",
             fontsize=16, transform=fig.transFigure,
             clip_on=False)

plt.savefig(os.path.join(OUTPUT_FOLDER, 'psf_binned_comparison_square_bins.pdf'), dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()

print(f"\nResidual statistics:")
print(f"e1 residual RMS: {np.nanstd(e1_residual):.4f}")
print(f"e2 residual RMS: {np.nanstd(e2_residual):.4f}")
print(f"T residual RMS: {np.nanstd(T_residual):.4f}")

# ---------------------- PLOT 3: RHO STATISTICS -----------------------------
rho = RhoStats(test_catalog, COLUMN_CONFIG, pixel_scale=RHO_STATS_CONFIG['pixel_scale'])

analyzer = ClusterShearCorrelation(
    catalog_file=RHO_STATS_CONFIG['catalog_file'],
    M500=RHO_STATS_CONFIG['M500'],
    z_cluster=RHO_STATS_CONFIG['z_cluster']
)

corr = analyzer.run_analysis(z_min_offset=RHO_STATS_CONFIG['z_min_offset'])

fig, axs = rho.plot(safezone_corr=corr, fraction=0.05)
fig.savefig(os.path.join(OUTPUT_FOLDER, "rho_stats_treecorr.pdf"), dpi=600, bbox_inches="tight")

# ---------------------- PLOT 4: PSF RESIDUALS VS MAGNITUDE -----------------
plt.rcParams.update({'font.size': FONT_SIZE_SMALL})

test_mask = (
    final_catalog[COLUMN_CONFIG['set_column']] == 'test'
) & (final_catalog[COLUMN_CONFIG['flux_column']] > 0)
test_catalog = final_catalog[test_mask]

e1_obs = test_catalog[COLUMN_CONFIG['e1_obs']]
e2_obs = test_catalog[COLUMN_CONFIG['e2_obs']]
T_obs = test_catalog[COLUMN_CONFIG['T_obs']]

e1_model = test_catalog[COLUMN_CONFIG['e1_model']]
e2_model = test_catalog[COLUMN_CONFIG['e2_model']]
T_model = test_catalog[COLUMN_CONFIG['T_model']]

e1_obs, e2_obs = e1e2_to_g1g2(e1_obs, e2_obs)
e1_model, e2_model = e1e2_to_g1g2(e1_model, e2_model)

e1_residual = e1_obs - e1_model
e2_residual = e2_obs - e2_model
T_residual = T_obs - T_model

mask = (
    (test_catalog['MAG_AUTO'] < 24) &
    np.isfinite(e1_residual) &
    np.isfinite(e2_residual) &
    np.isfinite(T_residual)
)

mag = test_catalog['MAG_AUTO'][mask]
e1_res = e1_residual[mask]
e2_res = e2_residual[mask]
T_res = T_residual[mask]
print(f"Number of objects for the statistics : {len(mag)}")

bins = np.linspace(13, 21, 20)
mag = mag - 1.33

bc, Tmed, Terr = binned_err_stats(mag, T_res, bins)
bc, Tfmed, Tferr = binned_err_stats(mag, T_res / test_catalog['T_admom_obs'][mask], bins)
bc, e1med, e1err = binned_err_stats(mag, e1_res, bins)
bc, e2med, e2err = binned_err_stats(mag, e2_res, bins)

fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)

axes[0].errorbar(bc, Tmed, yerr=Terr, fmt='o', color='darkcyan', alpha=0.8, ms=5)
axes[0].axhline(0, color='black', lw=1, linestyle='--')
axes[0].set_ylabel(r'$\delta T\ \mathrm{(arcsec^2)}$')
axes[0].set_ylim(-0.01, 0.04)

axes[1].errorbar(bc, Tfmed, yerr=Tferr, fmt='o', color='royalblue', alpha=0.8, ms=5)
axes[1].axhline(0, color='black', lw=1, linestyle='--')
axes[1].set_ylabel(r'$\delta T/T_*$')
axes[1].set_ylim(-0.015, 0.035)

axes[2].errorbar(bc, e1med, yerr=e1err, fmt='o', color='orange', alpha=0.85, ms=5, label=r'$e_1$')
axes[2].errorbar(bc, e2med, yerr=e2err, fmt='s', color='teal', alpha=0.85, ms=5, label=r'$e_2$')
axes[2].axhline(0, color='black', lw=1, linestyle='--')
axes[2].set_ylabel(r'$\delta e$')
axes[2].legend()
axes[2].set_ylim(-0.004, 0.004)

for ax in axes:
    ax.axvspan(13, 15.17, color='lightblue', alpha=0.25)
    ax.axvspan(20.2, 21, color='lightblue', alpha=0.25)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

axes[2].set_xlabel('Magnitude')
axes[0].set_xlim(13, 21)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "psf_residuals_mag.pdf"), dpi=OUTPUT_DPI, bbox_inches="tight")
plt.show()

# ---------------------- PLOT 5: FWHM DISTRIBUTION ---------------------------
plt.rcParams.update({'font.size': FONT_SIZE_LARGE})

fwhm_meds = np.array([e['fwhm_median'] for e in exposures], dtype=float)
good_stars = np.array([e['n_good_stars'] for e in exposures], dtype=float)

fwhm_meds = fwhm_meds[np.isfinite(fwhm_meds)]
good_stars = good_stars[np.isfinite(good_stars)]

fwhm_med = np.nanmedian(fwhm_meds) if fwhm_meds.size else np.nan
stars_med = np.nanmedian(good_stars) if good_stars.size else np.nan

if fwhm_meds.size:
    fwhm_min, fwhm_max = np.nanmin(fwhm_meds), np.nanmax(fwhm_meds)
    fwhm_pad = 0.03 * (fwhm_max - fwhm_min + 1e-6)
    fwhm_bins = np.linspace(fwhm_min - fwhm_pad, fwhm_max + fwhm_pad, 30)
else:
    fwhm_bins = 30

if good_stars.size:
    gs_min, gs_max = int(np.floor(np.nanmin(good_stars))), int(np.ceil(np.nanmax(good_stars)))
    if gs_max - gs_min < 10:
        gs_bins = np.arange(max(0, gs_min-1), gs_max+2)
    else:
        gs_bins = np.linspace(gs_min, gs_max, 30)
else:
    gs_bins = 30

fig, axes = plt.subplots(1, 1, figsize=(12, 7), sharey=False)

ax = axes
ax.hist(fwhm_meds, bins=fwhm_bins, color='#2E8BC0', alpha=0.85, edgecolor='white', linewidth=0.6)
ax.axvline(fwhm_med, color='#0C2D48', linestyle='--', lw=1.6, label=f'median = {fwhm_med:.3f}')
ax.set_xlabel('Seeing FWHM median (arcsec)', fontsize=18)
ax.set_ylabel('Number of exposures', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_xlim([0.3, 0.7])
ax.grid(alpha=0.05, linestyle='--')
ax.legend(frameon=True, fontsize=18)

fig.savefig(
    os.path.join(OUTPUT_FOLDER, "fwhm_distribution.pdf"),
    bbox_inches='tight',
    dpi=600,
    transparent=False,
)

# ---------------------- PLOT 6: STAR COUNTS FOOTPRINT ----------------------
plt.rcParams.update({'font.size': FONT_SIZE_MAIN})

ra_all, dec_all, ngood_all = [], [], []
for e in exposures:
    if 'coords' in e and np.all(np.isfinite(e['coords'])):
        ra_all.append(e['coords'][0])
        dec_all.append(e['coords'][1])
        ngood_all.append(e.get('n_good_stars', np.nan))

ra_all = np.array(ra_all)
dec_all = np.array(dec_all)
ngood_all = np.array(ngood_all, dtype=float)

skycoords = SkyCoord(ra=ra_all*u.deg, dec=dec_all*u.deg, frame='icrs')
l = skycoords.galactic.l.deg
b = skycoords.galactic.b.deg

plt.figure(figsize=(12, 6))
ax = plt.subplot(111, projection="mollweide")

l_rad = np.radians(l)
b_rad = np.radians(b)
l_rad_shift = np.radians(l - 180.0)

sc = ax.scatter(l_rad_shift, b_rad, c=ngood_all, cmap="RdBu_r", edgecolor='k', s=90, alpha=0.8)
cb = plt.colorbar(sc, orientation='horizontal', pad=0.05, fraction=0.05)
cb.set_label("Stars per exposure")

ax.grid(True)
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "stars_per_exp_footprint.pdf"),
    bbox_inches='tight',
    dpi=600,
    transparent=False,
)

print(f"\nAll plots saved to: {OUTPUT_FOLDER}/")
print("Analysis complete!")