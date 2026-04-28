"""
plot_cluster_comparison.py

Overlay tau sweep results for multiple clusters on two plots:
  1. BG contamination vs tau
  2. Source density vs tau

Each curve corresponds to one cluster, using that cluster's actual
z_thresh (z_cluster + 0.025). Training uses global NED/DESI objects.

Usage
-----
  python plot_cluster_comparison.py -c configs/default_source_selection.yaml
  python plot_cluster_comparison.py -c configs/default_source_selection.yaml \
      --outdir /path/to/output
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.table import Table

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../..')
))

from superbit_lensing.source_selection.color_cuts import (
    build_training_mask,
    load_and_remove_redseq,
    CLUSTER_COL,
)
from superbit_lensing.source_selection.validation import run_tau_sweep
from superbit_lensing import utils


# ---------------------------------------------------------------------------
# Clusters to compare
# ---------------------------------------------------------------------------

CLUSTERS = [
    {'name': 'Abell3365',          'redshift': 0.093},
    {'name': 'Abell3411',          'redshift': 0.170},
    {'name': 'Abell2163',          'redshift': 0.200},
    {'name': 'AbellS0592',         'redshift': 0.220},
    {'name': 'AbellS780',          'redshift': 0.240},
    {'name': 'RXCJ1314d4m2515',    'redshift': 0.250},
    {'name': 'RXCJ2003d5m2323',    'redshift': 0.320},
    {'name': 'SMACSJ2031d8m4036',  'redshift': 0.330},
    {'name': 'MACSJ1931d8m2635',   'redshift': 0.350},
    {'name': 'MACSJ0416d1m2403',   'redshift': 0.420},
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Overlay tau sweep curves for multiple clusters.'
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
        '--outdir', type=str, default=None,
        help='Output directory for plots. Defaults to output_dir in config.'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):

    cfg = utils.read_yaml(args.config)

    outdir = args.outdir or cfg['output_dir']
    os.makedirs(outdir, exist_ok=True)

    print(f'Loading catalog: {cfg["color_catalog"]}')
    catalog = Table.read(cfg['color_catalog'])
    print(f'  {len(catalog):,} total objects')

    pixel_size       = float(cfg['pixel_size'])
    min_count        = int(cfg['min_count'])
    err_thresh       = float(cfg['err_thresh'])
    xlim             = tuple(cfg['xlim'])
    ylim             = tuple(cfg['ylim'])
    weighting        = bool(cfg.get('weighting', True))
    with_redsequence = bool(cfg.get('with_redsequence', False))
    tau_values       = [round(float(t), 6) for t in cfg.get('tau_values', [0.5])]
    n_folds          = int(cfg.get('n_folds', 5))
    redseq_path      = cfg.get('redseq_catalog', None)

    # ---------------------------------------------------------------------------
    # Set up two figures
    # ---------------------------------------------------------------------------
    colors = cm.plasma(np.linspace(0.1, 0.9, len(CLUSTERS)))

    fig_contam, ax_contam = plt.subplots(figsize=(12, 7))
    fig_density, ax_density = plt.subplots(figsize=(12, 7))

    # ---------------------------------------------------------------------------
    # Run tau sweep per cluster and collect results
    # ---------------------------------------------------------------------------
    for cluster_cfg, color in zip(CLUSTERS, colors):
        cluster_name = cluster_cfg['name']
        z_cluster    = cluster_cfg['redshift']
        z_thresh     = round(z_cluster + 0.025, 6)

        print(f'\n{"="*55}')
        print(f'{cluster_name}  (z={z_cluster:.3f}, z_thresh={z_thresh:.3f})')
        print(f'{"="*55}')

        cluster_mask = catalog[CLUSTER_COL] == cluster_name
        if not np.any(cluster_mask):
            print(f'  Warning: {cluster_name} not found in catalog — skipping.')
            continue

        source_cluster_cat = catalog[cluster_mask]
        print(f'  {len(source_cluster_cat):,} objects')

        training_mask = build_training_mask(catalog, err_thresh=err_thresh)
        print(f'  Training objects: {np.sum(training_mask):,}')

        if with_redsequence and redseq_path:
            training_mask = load_and_remove_redseq(
                training_mask, catalog, redseq_path, cluster_name
            )
            print(f'  Training after RS removal: {np.sum(training_mask):,}')

        results = run_tau_sweep(
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
            shapes=False,
            n_folds=n_folds,
            weighting=weighting,
            err_thresh=err_thresh,
            outdir=None,  # suppress per-cluster plot
        )

        taus      = np.array([r['tau']             for r in results])
        bg_means  = np.array([r['bg_contam_mean']  for r in results])
        bg_stds   = np.array([r['bg_contam_std']   for r in results])
        densities = np.array([r['source_density']  for r in results])
        label     = f'{cluster_name} (z={z_cluster:.3f})'

        # Plot 1: bg_contam vs tau
        ax_contam.errorbar(
            taus, bg_means, yerr=bg_stds,
            color=color, marker='o', markersize=5,
            linestyle='-', linewidth=1.5, capsize=3,
            label=label, alpha=0.85
        )

        # Plot 2: source density vs tau
        ax_density.plot(
            taus, densities,
            color=color, marker='o', markersize=5,
            linestyle='-', linewidth=1.5,
            label=label, alpha=0.85
        )

    # ---------------------------------------------------------------------------
    # Format plot 1: bg_contam vs tau
    # ---------------------------------------------------------------------------
    ax_contam.axhline(0.10, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax_contam.text(tau_values[0], 0.103, '10%', fontsize=8, color='gray')

    ax_contam.set_xlabel('Purity threshold τ', fontsize=13)
    ax_contam.set_ylabel('BG contamination (FG→BG)', fontsize=13)
    ax_contam.set_title(
        'BG Contamination vs Purity Threshold\n'
        'Tau sweep across clusters (weighted pixel mask)',
        fontsize=14, fontweight='bold'
    )
    ax_contam.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax_contam.grid(True, alpha=0.3)
    ax_contam.set_xticks(tau_values)
    ax_contam.set_xticklabels([f'{t:.2f}' for t in tau_values], rotation=45)

    fig_contam.tight_layout()
    contam_path = os.path.join(outdir, 'cluster_comparison_bg_contam_vs_tau.png')
    fig_contam.savefig(contam_path, dpi=150, bbox_inches='tight')
    plt.close(fig_contam)
    print(f'\nSaved: {contam_path}')

    # ---------------------------------------------------------------------------
    # Format plot 2: source density vs tau
    # ---------------------------------------------------------------------------
    ax_density.set_xlabel('Purity threshold τ', fontsize=13)
    ax_density.set_ylabel('Source density (objects / arcmin²)', fontsize=13)
    ax_density.set_title(
        'Background Source Density vs Purity Threshold\n'
        'Tau sweep across clusters (weighted pixel mask)',
        fontsize=14, fontweight='bold'
    )
    ax_density.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax_density.grid(True, alpha=0.3)
    ax_density.set_xticks(tau_values)
    ax_density.set_xticklabels([f'{t:.2f}' for t in tau_values], rotation=45)

    fig_density.tight_layout()
    density_path = os.path.join(outdir, 'cluster_comparison_density_vs_tau.png')
    fig_density.savefig(density_path, dpi=150, bbox_inches='tight')
    plt.close(fig_density)
    print(f'Saved: {density_path}')

    return 0


if __name__ == '__main__':
    args = parse_args()
    rc   = main(args)
    if rc == 0:
        print('plot_cluster_comparison.py completed successfully.')
    else:
        sys.exit(rc)