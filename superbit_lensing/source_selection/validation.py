"""
validation.py

Cross-validation contamination estimation and tau sweep for the
source_selection module.

Two main entry points:

  run_cv_at_tau()
      5-fold stratified CV on the global redshift catalog at a single tau.
      Returns mean bg_contam, fg_contam, total_contam across folds.

  run_tau_sweep()
      Loops over tau values for one cluster. At each tau:
        - builds pixel mask and catalogs (in memory)
        - computes source density of the background catalog
        - records number of foreground objects removed
        - runs CV to get contamination estimate
      Produces a contamination-vs-source-density plot saved to disk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from .color_cuts import (
    TRAIN_COLOR_BG_COL as COLOR_BG_COL,
    TRAIN_COLOR_UB_COL as COLOR_UB_COL,
    TRAIN_COLOR_BG_ERR_COL as COLOR_BG_ERR_COL,
    TRAIN_COLOR_UB_ERR_COL as COLOR_UB_ERR_COL,
    TRAIN_REDSHIFT_COL as REDSHIFT_COL,
    create_pixel_voting_map_purity,
    apply_pixel_mask_to_catalog,
    make_background_catalog,
)


# ---------------------------------------------------------------------------
# Source density
# ---------------------------------------------------------------------------

def compute_source_density(catalog, ra_col='ra', dec_col='dec'):
    """
    Estimate source density in objects / arcmin² using the central 50%
    of the field, matching the convention in utils.analyze_mcal_fits.

    Parameters
    ----------
    catalog : astropy.table.Table
    ra_col, dec_col : str

    Returns
    -------
    density : float
    """
    if len(catalog) == 0:
        return 0.0

    ra  = np.asarray(catalog[ra_col],  dtype=float)
    dec = np.asarray(catalog[dec_col], dtype=float)

    valid = ~(np.isnan(ra) | np.isnan(dec))
    ra    = ra[valid]
    dec   = dec[valid]

    if len(ra) == 0:
        return 0.0

    ra_min, ra_max   = ra.min(),  ra.max()
    dec_min, dec_max = dec.min(), dec.max()

    ra_lo  = ra_min  + 0.25 * (ra_max  - ra_min)
    ra_hi  = ra_max  - 0.25 * (ra_max  - ra_min)
    dec_lo = dec_min + 0.25 * (dec_max - dec_min)
    dec_hi = dec_max - 0.25 * (dec_max - dec_min)

    mask     = (ra >= ra_lo) & (ra <= ra_hi) & (dec >= dec_lo) & (dec <= dec_hi)
    n_center = np.sum(mask)

    dec_center   = (dec_max + dec_min) / 2.0
    area_arcmin2 = (
        (ra_hi - ra_lo) * np.cos(np.radians(dec_center)) * 60.0
        * (dec_hi - dec_lo) * 60.0
    )

    if area_arcmin2 <= 0:
        return 0.0

    return n_center / area_arcmin2


# ---------------------------------------------------------------------------
# Per-fold contamination
# ---------------------------------------------------------------------------

def _contamination_for_fold(train_idx, val_idx, redshift_cat,
                             z_thresh, xlim, ylim, pixel_size,
                             purity_threshold, min_count,
                             weighting, err_thresh):
    """
    Train the pixel mask on train_idx, evaluate on val_idx.

    Convention: 1 = foreground (removed), 0 = background (kept).

    Returns
    -------
    dict with keys bg_contam, fg_contam, total_contam.
    """
    color_bg = redshift_cat[COLOR_BG_COL].astype(float)
    color_ub = redshift_cat[COLOR_UB_COL].astype(float)
    redshift = redshift_cat[REDSHIFT_COL].astype(float)
    cb_err   = redshift_cat[COLOR_BG_ERR_COL].astype(float)
    cub_err  = redshift_cat[COLOR_UB_ERR_COL].astype(float)

    fold_train_mask = np.zeros(len(redshift_cat), dtype=bool)
    fold_train_mask[train_idx] = True

    vote_map, x_edges, y_edges, _, _, _, actual_counts = \
        create_pixel_voting_map_purity(
            color_bg, color_ub, redshift, z_thresh,
            xlim, ylim, pixel_size, purity_threshold,
            training_mask=fold_train_mask,
            weighting=weighting,
            color_bg_err=cb_err,
            color_ub_err=cub_err,
        )

    vote_map_display = vote_map.copy()
    vote_map_display[(vote_map == -1) & (actual_counts >= min_count)] = 2

    val_bg = color_bg[val_idx]
    val_ub = color_ub[val_idx]
    val_z  = redshift[val_idx]

    true_labels = (val_z <= z_thresh).astype(int)

    pred_fg = np.zeros(len(val_idx), dtype=bool)
    for k, (bg_val, ub_val) in enumerate(zip(val_bg, val_ub)):
        if np.isnan(bg_val) or np.isnan(ub_val):
            continue
        i = np.searchsorted(x_edges, bg_val) - 1
        j = np.searchsorted(y_edges, ub_val) - 1
        if 0 <= i < len(x_edges) - 1 and 0 <= j < len(y_edges) - 1:
            pred_fg[k] = (vote_map_display[i, j] == 2)

    predictions = pred_fg.astype(int)

    TP = np.sum((predictions == 1) & (true_labels == 1))
    FP = np.sum((predictions == 1) & (true_labels == 0))
    TN = np.sum((predictions == 0) & (true_labels == 0))
    FN = np.sum((predictions == 0) & (true_labels == 1))

    bg_contam    = FN / (TN + FN) if (TN + FN) > 0 else np.nan
    fg_contam    = FP / (TP + FP) if (TP + FP) > 0 else np.nan
    total_contam = (FP + FN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else np.nan

    return dict(bg_contam=bg_contam, fg_contam=fg_contam, total_contam=total_contam)


# ---------------------------------------------------------------------------
# CV at a single tau
# ---------------------------------------------------------------------------

def run_cv_at_tau(redshift_cat, training_mask, z_thresh,
                  xlim, ylim, pixel_size, purity_threshold, min_count,
                  n_folds=5, weighting=True, err_thresh=0.5):
    """
    Run n_folds stratified cross-validation at a single purity_threshold
    on the global redshift catalog.

    Parameters
    ----------
    redshift_cat : astropy.table.Table
    training_mask : np.ndarray of bool
    z_thresh : float
    xlim, ylim : tuple
    pixel_size : float
    purity_threshold : float
    min_count : int
    n_folds : int
    weighting : bool
    err_thresh : float

    Returns
    -------
    dict with keys:
        bg_contam_mean,  bg_contam_std,
        fg_contam_mean,  fg_contam_std,
        total_contam_mean, total_contam_std
    """
    qual_indices = np.where(training_mask)[0]
    qual_cat     = redshift_cat[qual_indices]
    redshift_q   = qual_cat[REDSHIFT_COL].astype(float)
    labels       = (redshift_q > z_thresh).astype(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    bg_contams    = []
    fg_contams    = []
    total_contams = []

    for train_local, val_local in skf.split(qual_indices, labels):
        train_global = qual_indices[train_local]
        val_global   = qual_indices[val_local]

        result = _contamination_for_fold(
            train_global, val_global, redshift_cat,
            z_thresh, xlim, ylim, pixel_size, purity_threshold,
            min_count, weighting, err_thresh,
        )

        bg_contams.append(result['bg_contam'])
        fg_contams.append(result['fg_contam'])
        total_contams.append(result['total_contam'])

    def _safe_stats(arr):
        a = np.array([x for x in arr if not np.isnan(x)])
        return (float(np.mean(a)), float(np.std(a))) if len(a) > 0 else (np.nan, np.nan)

    bg_mean,    bg_std    = _safe_stats(bg_contams)
    fg_mean,    fg_std    = _safe_stats(fg_contams)
    total_mean, total_std = _safe_stats(total_contams)

    return dict(
        bg_contam_mean=bg_mean,       bg_contam_std=bg_std,
        fg_contam_mean=fg_mean,       fg_contam_std=fg_std,
        total_contam_mean=total_mean, total_contam_std=total_std,
    )


# ---------------------------------------------------------------------------
# Tau sweep
# ---------------------------------------------------------------------------

def run_tau_sweep(cluster_name, z_thresh, redshift_cat, training_mask,
                  full_catalog, source_cluster_cat,
                  tau_values, xlim, ylim, pixel_size, min_count,
                  shapes=False, n_folds=5, weighting=True, err_thresh=0.5,
                  outdir=None):
    """
    For a single cluster, sweep over tau values. At each tau:
      1. Build pixel mask on the full training set.
      2. Apply mask to full_catalog → foreground catalog.
      3. Build background catalog from source_cluster_cat (in memory).
      4. Compute source density of the background catalog.
      5. Record number of foreground objects removed.
      6. Run n-fold CV → contamination estimates.

    Saves a contamination-vs-source-density plot to outdir.

    Parameters
    ----------
    cluster_name : str
    z_thresh : float
    redshift_cat : astropy.table.Table
    training_mask : np.ndarray of bool
    full_catalog : astropy.table.Table
    source_cluster_cat : astropy.table.Table
    tau_values : list of float
    xlim, ylim : tuple
    pixel_size : float
    min_count : int
    shapes : bool
    n_folds : int
    weighting : bool
    err_thresh : float
    outdir : str or None

    Returns
    -------
    results : list of dict
    """
    if outdir is None:
        outdir = '.'

    color_bg = redshift_cat[COLOR_BG_COL].astype(float)
    color_ub = redshift_cat[COLOR_UB_COL].astype(float)
    redshift = redshift_cat[REDSHIFT_COL].astype(float)
    cb_err   = redshift_cat[COLOR_BG_ERR_COL].astype(float)
    cub_err  = redshift_cat[COLOR_UB_ERR_COL].astype(float)

    results = []

    for tau in tau_values:
        tau_key = round(tau, 6)
        print(f'\n  tau = {tau_key:.3f}')

        # 1. Build pixel mask
        vote_map, x_edges, y_edges, _, _, _, actual_counts = \
            create_pixel_voting_map_purity(
                color_bg, color_ub, redshift, z_thresh,
                xlim, ylim, pixel_size, tau_key,
                training_mask=training_mask,
                weighting=weighting,
                color_bg_err=cb_err,
                color_ub_err=cub_err,
            )

        vote_map_display = vote_map.copy()
        vote_map_display[(vote_map == -1) & (actual_counts >= min_count)] = 2

        # 2. Foreground catalog
        fg_cat = apply_pixel_mask_to_catalog(
            vote_map_display, x_edges, y_edges,
            full_catalog, cluster_name,
            purity_threshold=tau_key,
            pixel_size=pixel_size,
        )
        n_foreground = len(fg_cat)

        # 3. Background catalog (in memory, no disk write)
        bg_cat = make_background_catalog(
            source_cluster_cat, fg_cat,
            calculate_response=False,
            verbose=False,
        )

        # 4. Source density
        density = compute_source_density(bg_cat)
        print(f'     Foreground removed : {n_foreground}')
        print(f'     Source density     : {density:.3f} obj/arcmin²')

        # 5. CV contamination
        cv_result = run_cv_at_tau(
            redshift_cat, training_mask, z_thresh,
            xlim, ylim, pixel_size, tau_key, min_count,
            n_folds=n_folds, weighting=weighting, err_thresh=err_thresh,
        )
        print(f'     bg_contam  : {cv_result["bg_contam_mean"]:.3f} '
              f'± {cv_result["bg_contam_std"]:.3f}')
        print(f'     fg_contam  : {cv_result["fg_contam_mean"]:.3f} '
              f'± {cv_result["fg_contam_std"]:.3f}')
        print(f'     tot_contam : {cv_result["total_contam_mean"]:.3f} '
              f'± {cv_result["total_contam_std"]:.3f}')

        results.append(dict(
            tau=tau_key,
            source_density=density,
            n_foreground=n_foreground,
            **cv_result,
        ))

    _plot_tau_sweep(cluster_name, results, pixel_size, outdir)

    return results


def _plot_tau_sweep(cluster_name, results, pixel_size, outdir):
    """
    Plot contamination (y) vs source density (x) for each tau.
    Three curves: bg_contam, fg_contam, total_contam.
    Each point annotated with tau value and number of foreground objects removed.
    """
    taus          = [r['tau']            for r in results]
    densities     = [r['source_density'] for r in results]
    n_foregrounds = [r['n_foreground']   for r in results]

    bg_means  = np.array([r['bg_contam_mean']    for r in results])
    fg_means  = np.array([r['fg_contam_mean']    for r in results])
    tot_means = np.array([r['total_contam_mean'] for r in results])
    bg_stds   = np.array([r['bg_contam_std']     for r in results])
    fg_stds   = np.array([r['fg_contam_std']     for r in results])
    tot_stds  = np.array([r['total_contam_std']  for r in results])

    fig, ax = plt.subplots(figsize=(11, 7))

    styles = [
        (bg_means,  bg_stds,  '#E53935', 'o', 'BG contamination (FG→BG)'),
        (fg_means,  fg_stds,  '#1E88E5', 's', 'FG contamination (BG→FG)'),
        (tot_means, tot_stds, '#6A1B9A', 'D', 'Total misclassification'),
    ]

    for means, stds, color, marker, label in styles:
        ax.errorbar(densities, means, yerr=stds,
                    color=color, marker=marker, markersize=6,
                    linestyle='-', linewidth=1.5, capsize=3,
                    label=label, alpha=0.85)

        # Tau annotation above each point
        for x, y, tau in zip(densities, means, taus):
            ax.annotate(f'τ={tau:.2f}', (x, y),
                        textcoords='offset points', xytext=(6, 5),
                        fontsize=7, color=color)

    # Foreground count annotation below each point (once, in gray)
    for x, y, n_fg in zip(densities, bg_means, n_foregrounds):
        ax.annotate(f'{n_fg}', (x, y),
                    textcoords='offset points', xytext=(6, -14),
                    fontsize=6.5, color='gray')

    ax.axhline(0.10, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(ax.get_xlim()[0], 0.103, '10%', fontsize=8, color='gray')

    ax.set_xlabel('Source density (objects / arcmin²)', fontsize=12)
    ax.set_ylabel('Contamination / misclassification rate', fontsize=12)
    ax.set_title(
        f'{cluster_name}: Contamination vs Source Density\n'
        f'(tau sweep, pixel={pixel_size})',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    outpath = os.path.join(outdir, f'{cluster_name}_contamination_vs_density.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved tau sweep plot: {outpath}')