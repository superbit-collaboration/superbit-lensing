"""
color_cuts.py

Core pixel mask logic for foreground/background separation in color-color space.
Extracted and modularized from Pixel_Mask_Notebook.ipynb.

Single-catalog design: one mega catalog serves as both the training set
(rows with NED/DESI redshifts) and the full photometric sample.

Produces:
  - A pixel voting map trained on the redshift catalog
  - A foreground catalog (objects in foreground-dominated pixels)
  - A background catalog (cluster objects not in the foreground)
  - A pixel mask diagnostic image saved to disk
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
from matplotlib.patches import Patch
from astropy.table import Table
from astropy.io import fits
from superbit_lensing.match import SkyCoordMatcher


# ---------------------------------------------------------------------------
# Column name constants — change here if catalog schema changes
#
# Single-catalog design: training and photometric selection use the same
# catalog and therefore the same column names throughout.
# ---------------------------------------------------------------------------

TRAIN_COLOR_BG_COL     = 'color_bg'
TRAIN_COLOR_UB_COL     = 'color_ub'
TRAIN_COLOR_BG_ERR_COL = 'color_bg_err'
TRAIN_COLOR_UB_ERR_COL = 'color_ub_err'
TRAIN_REDSHIFT_COL     = 'Z_best'
TRAIN_ZSOURCE_COL      = 'Z_source'

COLOR_BG_COL = 'color_bg'
COLOR_UB_COL = 'color_ub'
CLUSTER_COL  = 'CLUSTER'


# ---------------------------------------------------------------------------
# Training mask
# ---------------------------------------------------------------------------

def build_training_mask(redshift_cat, err_thresh=0.5,
                        reliable_sources=('NED', 'DESI')):
    """
    Build a boolean mask selecting high-quality spectroscopic training objects.

    Selects rows where:
      - Z_source is in reliable_sources (NED or DESI)
      - color errors are below err_thresh
      - no NaN values in colors or redshift

    Parameters
    ----------
    redshift_cat : astropy.table.Table
        Catalog with Z_best, Z_source, color_bg, color_ub,
        color_bg_err, color_ub_err columns.
    err_thresh : float
    reliable_sources : tuple of str

    Returns
    -------
    training_mask : np.ndarray of bool
    """
    z_source     = np.array([s.strip() if isinstance(s, str) else ''
                              for s in redshift_cat[TRAIN_ZSOURCE_COL]])
    color_bg_err = redshift_cat[TRAIN_COLOR_BG_ERR_COL].astype(float)
    color_ub_err = redshift_cat[TRAIN_COLOR_UB_ERR_COL].astype(float)
    color_bg     = redshift_cat[TRAIN_COLOR_BG_COL].astype(float)
    color_ub     = redshift_cat[TRAIN_COLOR_UB_COL].astype(float)
    redshift     = redshift_cat[TRAIN_REDSHIFT_COL].astype(float)

    reliable_mask   = np.isin(z_source, list(reliable_sources))
    good_color_mask = (color_bg_err < err_thresh) & (color_ub_err < err_thresh)
    valid_data_mask = ~(np.isnan(color_bg) | np.isnan(color_ub) | np.isnan(redshift))

    return reliable_mask & good_color_mask & valid_data_mask


# ---------------------------------------------------------------------------
# Red sequence removal
# ---------------------------------------------------------------------------

def load_and_remove_redseq(training_mask, redshift_cat,
                            redseq_catalog_path, cluster_name):
    """
    Remove all red sequence members for a cluster from the training mask.

    Parameters
    ----------
    training_mask : np.ndarray of bool
    redshift_cat : astropy.table.Table
    redseq_catalog_path : str
        Mega RS catalog with target_name and is_red_sequence columns.
    cluster_name : str

    Returns
    -------
    training_mask : np.ndarray of bool
    """
    if not os.path.exists(redseq_catalog_path):
        print(f'Warning: RS catalog not found: {redseq_catalog_path}')
        return training_mask

    rs_cat = Table.read(redseq_catalog_path)

    cluster_mask = rs_cat['target_name'] == cluster_name
    cluster_rs   = rs_cat[cluster_mask & rs_cat['is_red_sequence'].astype(bool)]

    if len(cluster_rs) == 0:
        print(f'  No RS members found for {cluster_name} — skipping RS removal.')
        return training_mask

    print(f'  Found {len(cluster_rs)} RS members for {cluster_name}')

    if 'id' not in cluster_rs.colnames or 'id' not in redshift_cat.colnames:
        print('  Warning: missing id column — cannot perform RS removal.')
        return training_mask

    rs_flag       = np.isin(redshift_cat['id'], cluster_rs['id'].tolist())
    n_removed     = np.sum(training_mask & rs_flag)
    training_mask = training_mask & ~rs_flag
    print(f'  RS removal: excluded {n_removed} members from training')

    return training_mask


# ---------------------------------------------------------------------------
# Core pixel mask
# ---------------------------------------------------------------------------

def create_pixel_voting_map_purity(color_bg, color_ub, redshift, z_thresh,
                                   xlim, ylim, pixel_size, purity_threshold,
                                   training_mask=None, weighting=True,
                                   color_bg_err=None, color_ub_err=None):
    """
    Create a pixel-based voting map in color-color space with purity threshold.

    Each pixel is classified as foreground-dominated (vote = -1) if the
    (optionally weighted) foreground fraction meets purity_threshold, and
    background-dominated (vote = +1) otherwise.

    Parameters
    ----------
    color_bg, color_ub : array-like
    redshift : array-like
    z_thresh : float
    xlim, ylim : tuple
    pixel_size : float
    purity_threshold : float
    training_mask : boolean array, optional
    weighting : bool
    color_bg_err, color_ub_err : array-like, optional

    Returns
    -------
    vote_map, x_edges, y_edges, bg_hist, fg_hist, total_objects, actual_counts
    """
    if weighting and (color_bg_err is None or color_ub_err is None):
        raise ValueError(
            'color_bg_err and color_ub_err must be provided when weighting=True'
        )

    if training_mask is not None:
        cb  = np.asarray(color_bg)[training_mask]
        cub = np.asarray(color_ub)[training_mask]
        z   = np.asarray(redshift)[training_mask]
        if weighting:
            cb_err  = np.asarray(color_bg_err)[training_mask]
            cub_err = np.asarray(color_ub_err)[training_mask]
    else:
        cb  = np.asarray(color_bg)
        cub = np.asarray(color_ub)
        z   = np.asarray(redshift)
        if weighting:
            cb_err  = np.asarray(color_bg_err)
            cub_err = np.asarray(color_ub_err)

    valid = ~(np.isnan(cb) | np.isnan(cub) | np.isnan(z))
    if weighting:
        valid &= ~(np.isnan(cb_err) | np.isnan(cub_err))
    cb  = cb[valid];  cub = cub[valid];  z = z[valid]
    if weighting:
        cb_err = cb_err[valid];  cub_err = cub_err[valid]

    background_mask = z >  z_thresh
    foreground_mask = z <= z_thresh

    x_edges = np.arange(xlim[0], xlim[1] + pixel_size, pixel_size)
    y_edges = np.arange(ylim[0], ylim[1] + pixel_size, pixel_size)

    bg_counts, _, _ = np.histogram2d(cb[background_mask], cub[background_mask],
                                     bins=[x_edges, y_edges])
    fg_counts, _, _ = np.histogram2d(cb[foreground_mask], cub[foreground_mask],
                                     bins=[x_edges, y_edges])
    actual_counts = bg_counts + fg_counts

    if weighting:
        combined_err = (cb_err + cub_err) / 2.0
        epsilon      = 1e-10
        weights      = 1.0 / (combined_err**2 + epsilon)

        print(f'Foreground median error : {np.median(combined_err[foreground_mask]):.3f}')
        print(f'Background median error : {np.median(combined_err[background_mask]):.3f}')

        bg_hist, _, _ = np.histogram2d(cb[background_mask], cub[background_mask],
                                       bins=[x_edges, y_edges],
                                       weights=weights[background_mask])
        fg_hist, _, _ = np.histogram2d(cb[foreground_mask], cub[foreground_mask],
                                       bins=[x_edges, y_edges],
                                       weights=weights[foreground_mask])
    else:
        bg_hist = bg_counts
        fg_hist = fg_counts

    total_objects = bg_hist + fg_hist
    has_objects   = total_objects > 0

    purity   = np.zeros_like(bg_hist)
    purity[has_objects] = fg_hist[has_objects] / total_objects[has_objects]

    vote_map = np.zeros_like(bg_hist)
    vote_map[has_objects] = np.where(
        purity[has_objects] >= purity_threshold, -1, 1
    )

    return vote_map, x_edges, y_edges, bg_hist, fg_hist, total_objects, actual_counts


# ---------------------------------------------------------------------------
# Pixel mask diagnostic plot
# ---------------------------------------------------------------------------

def save_pixel_mask_image(vote_map, x_edges, y_edges, actual_counts,
                          cluster_name, z_thresh, purity_threshold,
                          pixel_size, min_count, outpath,
                          color_bg_train=None, color_ub_train=None,
                          redshift_train=None):
    """
    Save the pixel mask diagnostic image to disk.

    Parameters
    ----------
    vote_map : 2D np.ndarray
    x_edges, y_edges : np.ndarray
    actual_counts : 2D np.ndarray
    cluster_name : str
    z_thresh : float
    purity_threshold : float
    pixel_size : float
    min_count : int
    outpath : str
    color_bg_train, color_ub_train, redshift_train : array-like, optional

    Returns
    -------
    vote_map_display : 2D np.ndarray
    """
    vote_map_display = vote_map.copy()
    mask_pixels = (vote_map == -1) & (actual_counts >= min_count)
    vote_map_display[mask_pixels] = 2

    cmap = ListedColormap([
        to_rgba('blue', alpha=0.4),   # -1: light blue
        to_rgba('white', alpha=0.0),  #  0: empty
        to_rgba('red',  alpha=0.7),   # +1: BG dominated
        to_rgba('blue', alpha=1.0),   # +2: hard mask
    ])

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(
        vote_map_display.T,
        origin='lower',
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect='auto',
        cmap=cmap,
        vmin=-1, vmax=2,
        interpolation='nearest'
    )

    if color_bg_train is not None:
        cb  = np.asarray(color_bg_train)
        cub = np.asarray(color_ub_train)
        z   = np.asarray(redshift_train)
        bg_m = z >  z_thresh
        fg_m = z <= z_thresh
        if np.any(bg_m):
            ax.scatter(cb[bg_m], cub[bg_m], c='red',  alpha=0.6, s=2,
                       edgecolors='none', zorder=2)
        if np.any(fg_m):
            ax.scatter(cb[fg_m], cub[fg_m], c='blue', alpha=0.2, s=2,
                       edgecolors='none', zorder=2)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    blue_pixels  = int(np.sum(vote_map == -1))
    green_pixels = int(np.sum(vote_map_display == 2))

    legend_elements = [
        Patch(facecolor=to_rgba('blue', alpha=0.4), edgecolor='black', linewidth=0.5,
              label=f'FG dominated ({blue_pixels} pixels)'),
        Patch(facecolor=to_rgba('blue', alpha=1.0), edgecolor='black', linewidth=0.5,
              label=f'Mask (≥{min_count} objects) ({green_pixels} pixels)'),
        Patch(facecolor=to_rgba('red',  alpha=0.7), edgecolor='black', linewidth=0.5,
              label='BG dominated'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='black')

    ax.set_xlabel('B - G Color', fontsize=13)
    ax.set_ylabel('U - B Color', fontsize=13)
    ax.set_title(
        f'Pixel Mask — {cluster_name}  (z_thresh={z_thresh:.3f}, '
        f'τ={purity_threshold:.2f}, pixel={pixel_size})',
        fontsize=13
    )
    ax.tick_params(axis='both', labelsize=11, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, alpha=0.3, linewidth=1.0)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved pixel mask image: {outpath}')

    return vote_map_display


# ---------------------------------------------------------------------------
# Apply pixel mask → foreground catalog
# ---------------------------------------------------------------------------

def apply_pixel_mask_to_catalog(vote_map_display, x_edges, y_edges,
                                full_catalog, cluster_name,
                                purity_threshold, pixel_size):
    """
    Apply the pixel mask to the full catalog for one cluster,
    returning the foreground catalog.

    Each object's color_bg and color_ub are looked up in the pixel grid.
    Objects landing in hard mask pixels (vote_map_display == 2) are foreground.

    Parameters
    ----------
    vote_map_display : 2D np.ndarray
    x_edges, y_edges : np.ndarray
    full_catalog : astropy.table.Table
        Must contain CLUSTER, color_bg, color_ub columns.
    cluster_name : str
    purity_threshold : float
    pixel_size : float

    Returns
    -------
    foreground_catalog : astropy.table.Table
    """
    cluster_mask    = full_catalog[CLUSTER_COL] == cluster_name
    cluster_objects = full_catalog[cluster_mask]

    if len(cluster_objects) == 0:
        print(f'No objects found for cluster {cluster_name}')
        return Table()

    print(f'Found {len(cluster_objects)} objects for {cluster_name}')

    obj_bg = cluster_objects[COLOR_BG_COL].astype(float)
    obj_ub = cluster_objects[COLOR_UB_COL].astype(float)

    valid_colors  = ~(np.isnan(obj_bg) | np.isnan(obj_ub))
    valid_indices = np.where(valid_colors)[0]
    obj_bg = obj_bg[valid_colors]
    obj_ub = obj_ub[valid_colors]

    print(f'Objects with valid colors: {len(obj_bg)}')

    keep = np.zeros(len(obj_bg), dtype=bool)

    for idx, (bg_val, ub_val) in enumerate(zip(obj_bg, obj_ub)):
        i = np.searchsorted(x_edges, bg_val) - 1
        j = np.searchsorted(y_edges, ub_val) - 1
        if 0 <= i < len(x_edges) - 1 and 0 <= j < len(y_edges) - 1:
            keep[idx] = (vote_map_display[i, j] == 2)

    final_indices      = valid_indices[keep]
    foreground_catalog = cluster_objects[final_indices]

    print(f'Pixel mask selected {len(foreground_catalog)} foreground objects '
          f'({len(foreground_catalog)/len(cluster_objects)*100:.1f}%)')

    return foreground_catalog


# ---------------------------------------------------------------------------
# Shear response recalculation (only used when shapes=True)
# ---------------------------------------------------------------------------

def calculate_shear_response(catalog, verbose=True):
    """
    Recalculate shear response correction for a filtered (background) catalog.
    Only called when shapes=True in the config.

    Parameters
    ----------
    catalog : astropy.table.Table
    verbose : bool

    Returns
    -------
    catalog : astropy.table.Table
    """
    response_cols = ['r11', 'r12', 'r21', 'r22']
    bias_cols     = ['c1_psf', 'c2_psf', 'c1', 'c2']

    missing_critical = [c for c in response_cols if c not in catalog.colnames]
    if missing_critical:
        raise ValueError(f'Missing critical columns: {missing_critical}')

    r11_avg = np.mean(catalog['r11'])
    r12_avg = np.mean(catalog['r12'])
    r21_avg = np.mean(catalog['r21'])
    r22_avg = np.mean(catalog['r22'])

    R_avg = np.array([[r11_avg, r12_avg],
                      [r21_avg, r22_avg]])
    try:
        R_inv = np.linalg.inv(R_avg)
    except np.linalg.LinAlgError:
        raise ValueError('Response matrix is singular and cannot be inverted')

    if verbose:
        print(f'\nResponse matrix (from {len(catalog)} background objects):')
        print(f'  R = [[{r11_avg:.6f}, {r12_avg:.6f}], [{r21_avg:.6f}, {r22_avg:.6f}]]')

    c_total = np.zeros(2)
    if all(c in catalog.colnames for c in bias_cols):
        c_total = np.array([
            np.mean(catalog['c1_psf']) + np.mean(catalog['c1']),
            np.mean(catalog['c2_psf']) + np.mean(catalog['c2']),
        ])
        if verbose:
            print(f'  c_total = [{c_total[0]:.6f}, {c_total[1]:.6f}]')
    else:
        if verbose:
            print('  No bias correction applied (columns missing)')

    g_noshear = catalog['g_noshear']
    if len(g_noshear.shape) == 1:
        g1 = np.array([g[0] for g in g_noshear])
        g2 = np.array([g[1] for g in g_noshear])
    else:
        g1 = g_noshear[:, 0]
        g2 = g_noshear[:, 1]

    g_biased    = np.column_stack((g1 - c_total[0], g2 - c_total[1]))
    g_corrected = np.einsum('ij,nj->ni', R_inv, g_biased)

    catalog['g1_Rinv_new'] = g_corrected[:, 0]
    catalog['g2_Rinv_new'] = g_corrected[:, 1]

    if verbose:
        print('Added g1_Rinv_new and g2_Rinv_new to catalog')

    return catalog


# ---------------------------------------------------------------------------
# Build background catalog
# ---------------------------------------------------------------------------

def make_background_catalog(source_catalog, foreground_catalog,
                             tolerance_deg=1e-4, calculate_response=False,
                             verbose=True):
    """
    Remove foreground objects from source_catalog by RA/Dec sky matching,
    returning the background catalog.

    When shapes=False, source_catalog is the color catalog cluster slice.
    When shapes=True, source_catalog is the per-cluster annular shear catalog.

    Parameters
    ----------
    source_catalog : astropy.table.Table
    foreground_catalog : astropy.table.Table
    tolerance_deg : float
        Sky matching radius in degrees.
    calculate_response : bool
        If True, recalculate shear response (shapes=True only).
    verbose : bool

    Returns
    -------
    background_catalog : astropy.table.Table
    """
    n_total = len(source_catalog)

    if verbose:
        print(f'  Source objects       : {n_total:,}')
        print(f'  Foreground objects   : {len(foreground_catalog):,}')
        print(f'  Matching radius      : {tolerance_deg} deg')

    if len(foreground_catalog) == 0:
        print('  Warning: empty foreground catalog — returning full source catalog')
        return source_catalog

    matcher = SkyCoordMatcher(
        source_catalog, foreground_catalog,
        cat1_ratag='ra', cat1_dectag='dec',
        cat2_ratag='ra', cat2_dectag='dec',
        return_idx=True,
        match_radius=tolerance_deg
    )
    _, _, idx1, _ = matcher.get_matched_pairs()
    idx1 = np.array(idx1)

    background_mask = np.ones(n_total, dtype=bool)
    if len(idx1) > 0:
        background_mask[idx1] = False

    background_catalog = source_catalog[background_mask]
    n_bg = len(background_catalog)

    if verbose:
        print(f'  Foreground matches removed : {len(idx1):,}')
        print(f'  Background remaining       : {n_bg:,} '
              f'({n_bg/n_total:.3f} of total)')

    if calculate_response:
        try:
            background_catalog = calculate_shear_response(background_catalog,
                                                          verbose=verbose)
        except ValueError as e:
            if verbose:
                print(f'  Warning: shear response not recalculated — {e}')

    return background_catalog