"""
make_background_catalog.py

Main driver script for the source_selection module.

Single-catalog design:
  One mega color catalog serves both roles:
    Training set  : rows where Z_source is NED or DESI (~5,558 objects)
    Full sample   : all rows used for foreground/background object selection

shapes: false (default)
    Color cuts applied to the catalog. Background = catalog objects not
    classified as foreground. No shear columns required.

shapes: true
    Same color cuts, but foreground objects are removed from a separate
    per-cluster shear catalog. Shear response is recalculated.

For each cluster, produces:
  {outdir}/{cluster_name}/
    {cluster_name}_foreground.fits
    {cluster_name}_background.fits
    {cluster_name}_pixel_mask.png
    {cluster_name}_contamination_vs_density.png   (tau_sweep only)

Usage
-----
  python -m superbit_lensing.source_selection.make_background_catalog \
      -c configs/default_source_selection.yaml
"""

import os
import sys
import argparse
import numpy as np
from astropy.table import Table
from astropy.io import fits

from .color_cuts import (
    build_training_mask,
    load_and_remove_redseq,
    create_pixel_voting_map_purity,
    save_pixel_mask_image,
    apply_pixel_mask_to_catalog,
    union_redseq_into_foreground,
    make_background_catalog,
    union_specz_foreground_into_foreground,
    TRAIN_COLOR_BG_COL,
    TRAIN_COLOR_UB_COL,
    TRAIN_COLOR_BG_ERR_COL,
    TRAIN_COLOR_UB_ERR_COL,
    TRAIN_REDSHIFT_COL,
    TRAIN_ZSOURCE_COL, 
    CLUSTER_COL,
)
from .validation import run_cv_at_tau, run_tau_sweep, compute_source_density

from superbit_lensing import utils


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run color-cut source selection to produce per-cluster '
                    'foreground and background catalogs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-c', '--config', type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs', 'default_source_selection.yaml'
        ),
        help='Path to YAML config file.'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite existing output files.'
    )
    parser.add_argument(
        '--vb', action='store_true', default=False,
        help='Verbose output.'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-cluster processing
# ---------------------------------------------------------------------------

def process_cluster(cluster_cfg, global_cfg, catalog,
                    overwrite=False, vb=False):
    """
    Run the full source-selection pipeline for one cluster.

    Parameters
    ----------
    cluster_cfg : dict
        Must contain 'name' and 'redshift'.
        Optionally 'shear_catalog' (required when shapes=True).
    global_cfg : dict
    catalog : astropy.table.Table
        The single mega catalog — used for both training and object selection.
    overwrite : bool
    vb : bool
    """
    cluster_name = cluster_cfg['name']
    z_cluster    = float(cluster_cfg['redshift'])
    z_thresh     = round(z_cluster + 0.025, 6)
    

    print(f'\n{"="*60}')
    print(f'Processing {cluster_name}  (z={z_cluster:.3f}, z_thresh={z_thresh:.3f})')
    print(f'{"="*60}')

    # ---- Read global parameters ------------------------------------------
    pixel_size       = float(global_cfg['pixel_size'])
    min_count        = int(global_cfg['min_count'])
    err_thresh       = float(global_cfg['err_thresh'])
    xlim             = tuple(global_cfg['xlim'])
    ylim             = tuple(global_cfg['ylim'])
    weighting        = bool(global_cfg.get('weighting', True))
    with_redsequence = bool(global_cfg.get('with_redsequence', False))
    tau_sweep        = bool(global_cfg.get('tau_sweep', False))
    purity_threshold = round(float(global_cfg.get('purity_threshold', 0.5)), 6)
    tau_values       = [round(float(t), 6)
                        for t in global_cfg.get('tau_values', [0.5])]
    n_folds          = int(global_cfg.get('n_folds', 5))
    shapes           = bool(global_cfg.get('shapes', False))
    union_redseq    = bool(global_cfg.get('union_redseq_into_foreground', False))
    union_specz     = bool(global_cfg.get('union_specz_into_foreground', False))



    # ---- Output directory ------------------------------------------------
    cluster_outdir = os.path.join(global_cfg['output_dir'], cluster_name)
    os.makedirs(cluster_outdir, exist_ok=True)

    fg_path   = os.path.join(cluster_outdir, f'{cluster_name}_foreground.fits')
    bg_path   = os.path.join(cluster_outdir, f'{cluster_name}_background.fits')
    mask_path = os.path.join(cluster_outdir, f'{cluster_name}_pixel_mask.png')

    for path in (fg_path, bg_path):
        if os.path.exists(path) and not overwrite:
            print(f'Output already exists (use --overwrite to redo): {path}')
            return

    # ---- Source catalog --------------------------------------------------
    if shapes:
        shear_catalog_path = cluster_cfg.get('shear_catalog', None)
        if shear_catalog_path is None:
            print(f'  ERROR: shapes=True but no shear_catalog specified '
                  f'for {cluster_name} — skipping.')
            return
        if not os.path.exists(shear_catalog_path):
            print(f'  ERROR: shear catalog not found: {shear_catalog_path} — skipping.')
            return
        print(f'\nLoading shear catalog: {shear_catalog_path}')
        source_cluster_cat = Table.read(shear_catalog_path)
        print(f'  {len(source_cluster_cat):,} objects')
    else:
        print('\nshapes=False: using color catalog as source catalog.')
        cluster_mask       = catalog[CLUSTER_COL] == cluster_name
        source_cluster_cat = catalog[cluster_mask]
        print(f'  {len(source_cluster_cat):,} objects for {cluster_name}')

    if len(source_cluster_cat) == 0:
        print(f'  No objects found for {cluster_name} — skipping.')
        return

    # ---- Training mask ---------------------------------------------------
    print('\nBuilding training mask...')
    training_mask = build_training_mask(catalog, err_thresh=err_thresh)
    print(f'  Quality objects available for training: {np.sum(training_mask):,}')

    # ---- Red sequence removal --------------------------------------------
    if with_redsequence:
        redseq_catalog_path = global_cfg.get('redseq_catalog', None)
        if redseq_catalog_path is None:
            print('Warning: with_redsequence=True but redseq_catalog not set in config.')
        else:
            print('\nRemoving red sequence members from training...')
            training_mask = load_and_remove_redseq(
                training_mask, catalog,
                redseq_catalog_path, cluster_name
            )
            print(f'  Training objects after RS removal: {np.sum(training_mask):,}')

    # ---- Tau sweep -------------------------------------------------------
    if tau_sweep:
        print(f'\nRunning tau sweep over {len(tau_values)} tau values...')
        run_tau_sweep(
            cluster_name=cluster_name,
            z_thresh=z_thresh,
            redshift_cat=catalog,
            training_mask=training_mask,
            full_catalog=catalog,
            source_cluster_cat=source_cluster_cat,
            tau_values=tau_values,
            xlim=xlim,
            ylim=ylim,
            pixel_size=pixel_size,
            min_count=min_count,
            shapes=shapes,
            n_folds=n_folds,
            weighting=weighting,
            err_thresh=err_thresh,
            outdir=cluster_outdir,
        )
        print(f'\nProducing final catalogs at purity_threshold={purity_threshold:.2f}...')

    # ---- Build final pixel mask ------------------------------------------
    color_bg = catalog[TRAIN_COLOR_BG_COL].astype(float)
    color_ub = catalog[TRAIN_COLOR_UB_COL].astype(float)
    redshift = catalog[TRAIN_REDSHIFT_COL].astype(float)
    cb_err   = catalog[TRAIN_COLOR_BG_ERR_COL].astype(float)
    cub_err  = catalog[TRAIN_COLOR_UB_ERR_COL].astype(float)

    print('\nBuilding pixel voting map...')
    vote_map, x_edges, y_edges, _, _, _, actual_counts = \
        create_pixel_voting_map_purity(
            color_bg, color_ub, redshift, z_thresh,
            xlim, ylim, pixel_size, purity_threshold,
            training_mask=training_mask,
            weighting=weighting,
            color_bg_err=cb_err,
            color_ub_err=cub_err,
        )

    # ---- Save pixel mask image -------------------------------------------
    print('\nSaving pixel mask image...')
    vote_map_display = save_pixel_mask_image(
        vote_map, x_edges, y_edges, actual_counts,
        cluster_name, z_thresh, purity_threshold,
        pixel_size, min_count, outpath=mask_path,
        color_bg_train=color_bg[training_mask],
        color_ub_train=color_ub[training_mask],
        redshift_train=redshift[training_mask],
    )

    print(f'  Total FG pixels (vote=-1) : {int(np.sum(vote_map == -1))}')
    print(f'  Hard mask pixels (vote=2) : {int(np.sum(vote_map_display == 2))}')

    # ---- Foreground catalog ----------------------------------------------
    print('\nApplying pixel mask to catalog...')
    foreground_cat = apply_pixel_mask_to_catalog(
        vote_map_display, x_edges, y_edges,
        catalog, cluster_name,
        purity_threshold=purity_threshold,
        pixel_size=pixel_size,
    )
    n_pixel_mask = len(foreground_cat)

    if union_redseq:
        redseq_catalog_path = global_cfg.get('redseq_catalog', None)
        if redseq_catalog_path is None:
            print('  Warning: union_redseq_into_foreground=True but redseq_catalog not set.')
        else:
            print('\nUnioning RS members into foreground catalog...')
            cluster_color_mask = catalog[CLUSTER_COL] == cluster_name
            color_cluster_cat  = catalog[cluster_color_mask]
            foreground_cat, n_rs_added = union_redseq_into_foreground(
                foreground_cat, color_cluster_cat,
                redseq_catalog_path, cluster_name,
            )
            print(f'  Pixel mask hits      : {n_pixel_mask}')
            print(f'  Unique RS added      : {n_rs_added}')
            print(f'  Total foreground     : {len(foreground_cat)}')

    if union_specz:
        print('\nUnioning spec-z confirmed FG into foreground catalog...')
        cluster_color_mask = catalog[CLUSTER_COL] == cluster_name
        color_cluster_cat  = catalog[cluster_color_mask]
        n_before_specz = len(foreground_cat)
        foreground_cat, n_specz_added = union_specz_foreground_into_foreground(
            foreground_cat, color_cluster_cat, z_thresh,
        )
        print(f'  Foreground before spec-z union : {n_before_specz}')
        print(f'  Unique spec-z FG added          : {n_specz_added}')
        print(f'  Total foreground after union    : {len(foreground_cat)}')

    if len(foreground_cat) == 0:
        print(f'  Warning: empty foreground catalog for {cluster_name}. '
              f'Check pixel mask parameters.')
        return

    print(f'\nSaving foreground catalog: {fg_path}')
    foreground_cat.write(fg_path, format='fits', overwrite=overwrite)


    

    # ---- Background catalog ----------------------------------------------
    print('\nBuilding background catalog...')
    background_cat = make_background_catalog(
        source_cluster_cat, foreground_cat,
        calculate_response=shapes,
        verbose=vb,
    )

    print(f'\nSaving background catalog: {bg_path}')
    background_cat.write(bg_path, format='fits', overwrite=overwrite)

    # ---- Source density + header via utils.analyze_mcal_fits -------------
    print('\nAnalyzing background catalog...')
    density_results = utils.analyze_mcal_fits(
        bg_path, update_header=True, verbose=vb
    )
    density = density_results['DENS_AMIN']
    print(f'Background source density: {density:.3f} objects / arcmin²')

    
    # ---- CV contamination → FITS header (always) -------------------------
    print(f'\nRunning {n_folds}-fold CV at tau={purity_threshold:.2f}...')
    cv_result = run_cv_at_tau(
        catalog, training_mask, z_thresh,
        xlim, ylim, pixel_size, purity_threshold, min_count,
        n_folds=n_folds, weighting=weighting, err_thresh=err_thresh,
    )

    print(f'  bg_contam  : {cv_result["bg_contam_mean"]:.3f} '
        f'± {cv_result["bg_contam_std"]:.3f}')
    print(f'  fg_contam  : {cv_result["fg_contam_mean"]:.3f} '
        f'± {cv_result["fg_contam_std"]:.3f}')
    print(f'  tot_contam : {cv_result["total_contam_mean"]:.3f} '
        f'± {cv_result["total_contam_std"]:.3f}')

    _write_contamination_to_header(
        bg_path, cv_result, purity_threshold, n_folds, z_thresh, shapes
    )


def _write_contamination_to_header(bg_path, cv_result, purity_threshold,
                                   n_folds, z_thresh, shapes):
    """
    Write CV contamination estimates into the FITS header.
    Source density is already written by utils.analyze_mcal_fits.
    """
    with fits.open(bg_path, mode='update') as hdul:
        data_hdu = 0
        for i, hdu in enumerate(hdul):
            if hasattr(hdu, 'data') and hdu.data is not None:
                try:
                    if len(hdu.data) > 0:
                        data_hdu = i
                        break
                except TypeError:
                    pass

        hdr = hdul[data_hdu].header
        hdr['HISTORY']  = 'Contamination estimates added by source_selection'
        hdr['SHAPES']   = (shapes,           'Whether shear catalog was used')
        hdr['TAU']      = (purity_threshold, 'Purity threshold (tau) used for color cuts')
        hdr['Z_THRESH'] = (z_thresh,         'Redshift threshold (z_cluster + 0.025)')
        hdr['CV_FOLDS'] = (n_folds,          'Number of CV folds for contamination estimate')
        hdr['BG_CONT']  = (round(cv_result['bg_contam_mean'],    4),
                           'Mean BG contamination (FG->BG) from CV')
        hdr['BG_CSTD']  = (round(cv_result['bg_contam_std'],     4),
                           'Std BG contamination across CV folds')
        hdr['FG_CONT']  = (round(cv_result['fg_contam_mean'],    4),
                           'Mean FG contamination (BG->FG) from CV')
        hdr['FG_CSTD']  = (round(cv_result['fg_contam_std'],     4),
                           'Std FG contamination across CV folds')
        hdr['TOT_CONT'] = (round(cv_result['total_contam_mean'], 4),
                           'Mean total misclassification rate from CV')
        hdr['TOT_CSTD'] = (round(cv_result['total_contam_std'],  4),
                           'Std total misclassification across CV folds')
        hdr['N_SPECZ'] = (n_specz_added, 'Spec-z confirmed FG removed beyond pixel mask')

    print(f'  Contamination estimates written to header: {bg_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):

    print(f'Reading config: {args.config}')
    cfg = utils.read_yaml(args.config)

    # Single catalog: training + object selection
    print(f'\nLoading catalog: {cfg["color_catalog"]}')
    catalog = Table.read(cfg['color_catalog'])
    n_with_z = int(np.sum(~np.isnan(catalog[TRAIN_REDSHIFT_COL].astype(float))))
    z_src = np.array([s.strip() if isinstance(s, str) else ''
                      for s in catalog[TRAIN_ZSOURCE_COL]])
    n_reliable = int(np.sum(np.isin(z_src, ['NED', 'DESI'])))
    print(f'  {len(catalog):,} total objects')
    print(f'  {n_with_z:,} objects with Z_best')
    print(f'  {n_reliable:,} NED/DESI objects (training set)')

    shapes = bool(cfg.get('shapes', False))
    print(f'\nshapes = {shapes}')

    clusters = cfg.get('clusters', [])
    if not clusters:
        print('No clusters specified in config — exiting.')
        return 1

    print(f'\n{len(clusters)} cluster(s) to process: '
          f'{[c["name"] for c in clusters]}')

    for cluster_cfg in clusters:
        try:
            process_cluster(
                cluster_cfg=cluster_cfg,
                global_cfg=cfg,
                catalog=catalog,
                overwrite=args.overwrite,
                vb=args.vb,
            )
        except Exception as e:
            print(f'\nERROR processing {cluster_cfg["name"]}: {e}')
            if args.vb:
                import traceback
                traceback.print_exc()
            print('Continuing to next cluster...')
            continue

    print('\nAll clusters processed.')
    return 0


if __name__ == '__main__':
    args = parse_args()
    rc   = main(args)

    if rc == 0:
        print('make_background_catalog.py completed successfully.')
    else:
        print(f'make_background_catalog.py failed with rc={rc}')
        sys.exit(rc)