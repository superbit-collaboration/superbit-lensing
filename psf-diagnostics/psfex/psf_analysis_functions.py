"""
PSF Analysis Functions Module
Contains all helper functions for PSF analysis pipeline
"""

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter

# ============================================================================
# =========================== HELPER FUNCTIONS ===============================
# ============================================================================

def _get_scalar(arr, default=np.nan):
    """Helper: robust scalar extraction"""
    try:
        val = arr[0]
        if hasattr(val, 'mask') and val.mask:
            return default
        return float(val)
    except Exception:
        return default

def get_median_radec(data):
    """Get median RA/Dec from data"""
    ra_array = data["ALPHAWIN_J2000"]
    dec_array = data["DELTAWIN_J2000"]
    return np.nanmedian(ra_array), np.nanmedian(dec_array)

def minmax_norm(arr):
    """Utility: minmax normalization with nan-safety"""
    arr = np.array(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isclose(mx, mn):
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def z_norm(arr):
    """Utility: z-score normalization with nan-safety"""
    arr = np.array(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)
    mn = np.nanmean(arr)
    sd = np.nanstd(arr)
    if np.isclose(sd, 0):
        return np.zeros_like(arr)
    return (arr - mn) / sd

def maybe_smooth(arr, apply_gaussian, gauss_sigma):
    """Apply Gaussian smoothing if enabled"""
    if apply_gaussian:
        return gaussian_filter(arr, sigma=gauss_sigma)
    else:
        return arr

def bin_statistic_2d(x, y, values, x_edges, y_edges, min_stars_per_bin):
    """Function to calculate binned median"""
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(values))
    result, _, _, _ = stats.binned_statistic_2d(
        x[mask], y[mask], values[mask], 
        statistic='median', 
        bins=[x_edges, y_edges]
    )
    # Also get counts for masking
    counts, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
    # Mask bins with too few stars
    result[counts < min_stars_per_bin] = np.nan
    return result

def binned_err_stats(x, y, bins):
    """Calculate binned error statistics"""
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    med = np.full(len(bin_centers), np.nan)
    err = np.full(len(bin_centers), np.nan)
    for i in range(len(bins) - 1):
        m = (x >= bins[i]) & (x < bins[i+1])
        if np.any(m):
            med[i] = np.nanmean(y[m])
            err[i] = np.nanstd(y[m]) / np.sqrt(np.sum(m))
    return bin_centers, med, err

def format_colorbar(im, ax, vmin, vmax):
    """Helper function for colorbars"""
    import matplotlib.pyplot as plt
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{vmin:.2f}", "", f"{((vmin+vmax)/2):.2f}", "", f"{vmax:.2f}"])
    return cbar

def build_exposures_list(catalog_filtered, unique_clusters):
    """Build list of exposures from catalog"""
    exposures = []
    for cluster in unique_clusters:
        cluster_mask = catalog_filtered['CLUSTER_NAME'] == cluster
        cluster_data = catalog_filtered[cluster_mask]
        unique_exps = np.unique(cluster_data['EXP_NUM'])
        for exp in unique_exps:
            exp_mask = cluster_data['EXP_NUM'] == exp
            exp_data = cluster_data[exp_mask]
            fwhm_median = _get_scalar(exp_data['FWHM_MEDIAN']) if 'FWHM_MEDIAN' in exp_data.colnames else np.nan
            fwhm_std = _get_scalar(exp_data['FWHM_STD']) if 'FWHM_STD' in exp_data.colnames else np.nan
            n_good_stars = _get_scalar(exp_data['N_GOOD_STARS']) if 'N_GOOD_STARS' in exp_data.colnames else np.nan
            exposures.append({
                'cluster': cluster,
                'exp_num': exp,
                'coords': get_median_radec(exp_data),
                'fwhm_median': fwhm_median,
                'fwhm_std': fwhm_std,
                'n_good_stars': n_good_stars,
                'n_objects': len(exp_data),
                'metric_value': np.nan  # to fill next
            })
    return exposures

def compute_global_metrics(exposures, sel_upper, combined_method, combined_alpha):
    """Compute metric values globally across all exposures"""
    if sel_upper == 'COMBINED':
        med = np.array([e['fwhm_median'] for e in exposures], dtype=float)
        std = np.array([e['fwhm_std'] for e in exposures], dtype=float)
        if combined_method == 'minmax':
            med_norm = minmax_norm(med)
            std_norm = minmax_norm(std)
        elif combined_method == 'zscore':
            med_norm = z_norm(med)
            std_norm = z_norm(std)
        else:
            raise ValueError("Unknown COMBINED_METHOD")
        for i, e in enumerate(exposures):
            a = med_norm[i]
            b = std_norm[i]
            if np.isnan(a) and np.isnan(b):
                combined = np.nan
            elif np.isnan(a):
                combined = b
            elif np.isnan(b):
                combined = a
            else:
                combined = combined_alpha * a + (1.0 - combined_alpha) * b
            e['metric_value'] = float(combined) if not np.isnan(combined) else np.nan
    else:
        # Single-column metrics: directly copy into metric_value
        for e in exposures:
            if sel_upper == 'FWHM_MEDIAN':
                e['metric_value'] = e['fwhm_median']
            elif sel_upper == 'FWHM_STD':
                e['metric_value'] = e['fwhm_std']
            elif sel_upper == 'N_GOOD_STARS':
                e['metric_value'] = e['n_good_stars']
    return exposures

def compute_per_cluster_metrics(cluster_data, sel_upper, combined_method, combined_alpha):
    """Compute metrics for a single cluster"""
    unique_exps = np.unique(cluster_data['EXP_NUM'])
    exp_info = []
    for exp in unique_exps:
        exp_mask = cluster_data['EXP_NUM'] == exp
        exp_data = cluster_data[exp_mask]
        fwhm_median = _get_scalar(exp_data['FWHM_MEDIAN']) if 'FWHM_MEDIAN' in exp_data.colnames else np.nan
        fwhm_std = _get_scalar(exp_data['FWHM_STD']) if 'FWHM_STD' in exp_data.colnames else np.nan
        n_good_stars = _get_scalar(exp_data['N_GOOD_STARS']) if 'N_GOOD_STARS' in exp_data.colnames else np.nan
        exp_info.append({
            'exp_num': exp,
            'fwhm_median': fwhm_median,
            'fwhm_std': fwhm_std,
            'n_good_stars': n_good_stars,
            'n_objects': len(exp_data),
            'metric_value': np.nan
        })
    
    # compute metric per-cluster
    if sel_upper == 'COMBINED':
        med = np.array([e['fwhm_median'] for e in exp_info], dtype=float)
        std = np.array([e['fwhm_std'] for e in exp_info], dtype=float)
        if combined_method == 'minmax':
            med_norm = minmax_norm(med)
            std_norm = minmax_norm(std)
        elif combined_method == 'zscore':
            med_norm = z_norm(med)
            std_norm = z_norm(std)
        else:
            raise ValueError("Unknown COMBINED_METHOD")
        for i, e in enumerate(exp_info):
            a = med_norm[i]
            b = std_norm[i]
            if np.isnan(a) and np.isnan(b):
                combined = np.nan
            elif np.isnan(a):
                combined = b
            elif np.isnan(b):
                combined = a
            else:
                combined = combined_alpha * a + (1.0 - combined_alpha) * b
            e['metric_value'] = float(combined) if not np.isnan(combined) else np.nan
    else:
        for e in exp_info:
            if sel_upper == 'FWHM_MEDIAN':
                e['metric_value'] = e['fwhm_median']
            elif sel_upper == 'FWHM_STD':
                e['metric_value'] = e['fwhm_std']
            elif sel_upper == 'N_GOOD_STARS':
                e['metric_value'] = e['n_good_stars']
            else:
                try:
                    val = cluster_data[cluster_data['EXP_NUM'] == e['exp_num']][SELECTION_METRIC][0]
                    e['metric_value'] = float(val)
                except Exception:
                    e['metric_value'] = np.nan
    
    return exp_info