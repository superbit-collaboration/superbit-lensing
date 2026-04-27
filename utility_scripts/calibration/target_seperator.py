"""
target_separator.py

Select background galaxies for weak lensing by removing
known low-redshift and foreground contaminants from a
metacalibration catalog, then compute shear responsivity.

Usage:
    python target_separator.py <config.yaml>
"""

import os
import sys
import time
import yaml
from omegaconf import OmegaConf
import numpy as np
from astropy.table import Table

import superbit_lensing.utils as utils
from superbit_lensing.match import SkyCoordMatcher
from superbit_lensing.diagnostics import compute_R_S


# ============================================================
# Config
# ============================================================

def load_config(config_path):
    """Load and validate a YAML config file."""
    with open(config_path, 'r') as f:
        cfg = OmegaConf.load(f)

    # Derived quantities
    cfg['redshift_cut'] = cfg['target_redshift'] + cfg['redshift_offset']
    cfg['tolerance_deg'] = cfg['tolerance_arcsec'] / 3600.0

    # Resolve NED redshift cache path
    cfg['ned_redshifts_path'] = os.path.join(
        cfg['data_dir'], "catalogs", "redshifts",
        f"{cfg['target_name']}_NED_redshifts.csv"
    )

    # Attach foreground file to qual_cuts so compute_R_S sees it if needed
    cfg['qual_cuts']['foreground_file'] = cfg['foreground_file']

    return cfg


# ============================================================
# NED redshifts
# ============================================================

def load_ned_redshifts(cfg, center_ra, center_dec, radius):
    """
    Load NED redshifts from a cached CSV, falling back to a
    live NED cone search with retries on failure.
    """
    ned_path = cfg['ned_redshifts_path']
    ned_cfg = cfg['ned']

    try:
        print(f"Loading NED redshifts from {ned_path}")
        ned = Table.read(ned_path, format='csv')
        print(f"  Loaded {len(ned)} objects")
        return ned
    except Exception as e:
        print(f"  Could not load cached file: {e}")

    # Live query with retries
    print("Falling back to NED cone search ...")
    current_radius = radius

    for attempt in range(ned_cfg['max_attempts']):
        try:
            print(f"  Attempt {attempt + 1}/{ned_cfg['max_attempts']}  "
                  f"(radius = {current_radius:.4f} deg)")
            ned = utils.ned_query(
                rad_deg=current_radius,
                ra_center=center_ra,
                dec_center=center_dec,
            )
            print(f"  Success: {len(ned)} objects returned")

            # Cache for next time
            try:
                ned.write(ned_path, format='csv', overwrite=True)
                print(f"  Saved to {ned_path}")
            except Exception as save_err:
                print(f"  Warning: could not save cache — {save_err}")

            return ned

        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < ned_cfg['max_attempts'] - 1:
                current_radius /= ned_cfg['radius_shrink_factor']
                print(f"  Retrying in {ned_cfg['wait_time']}s with "
                      f"radius = {current_radius:.4f} deg ...")
                time.sleep(ned_cfg['wait_time'])

    if ned_cfg['fail_silently']:
        print("  All NED attempts failed — proceeding with empty catalog")
        return Table(names=['RA', 'DEC', 'Redshift'],
                     dtype=['f8', 'f8', 'f8'])
    else:
        raise RuntimeError("All NED query attempts failed.")


# ============================================================
# Catalog cleaning helpers
# ============================================================

def remove_by_match(tab, contaminants, tolerance_deg,
                    cat1_ra='RA', cat1_dec='DEC',
                    cat2_ra='ra', cat2_dec='dec',
                    label="contaminant cut"):
    """
    Cross-match `tab` against `contaminants` and remove
    all matched rows from `tab`.

    Returns the cleaned table.
    """
    matcher = SkyCoordMatcher(
        contaminants, tab,
        cat1_ratag=cat1_ra, cat1_dectag=cat1_dec,
        cat2_ratag=cat2_ra, cat2_dectag=cat2_dec,
        return_idx=True, match_radius=tolerance_deg,
    )
    _, _, _, idx_tab = matcher.get_matched_pairs()

    remove_idx = np.array(idx_tab)
    mask = np.ones(len(tab), dtype=bool)
    mask[remove_idx] = False

    n_total = len(mask)
    n_remove = np.sum(~mask)
    n_remain = np.sum(mask)
    print(f"  {label}: {n_remove}/{n_total} removed, {n_remain} remaining")

    return tab[mask]


def remove_low_redshift(tab, ned, redshift_cut, tolerance_deg):
    """
    Match against NED, then remove only those matches whose
    redshift falls below `redshift_cut`.
    """
    matcher = SkyCoordMatcher(
        ned, tab,
        cat1_ratag='RA', cat1_dectag='DEC',
        cat2_ratag='ra', cat2_dectag='dec',
        return_idx=True, match_radius=tolerance_deg,
    )
    matched_ned, _, _, idx_tab = matcher.get_matched_pairs()

    low_z = matched_ned["Redshift"] < redshift_cut
    remove_idx = np.array(idx_tab)[low_z]

    mask = np.ones(len(tab), dtype=bool)
    mask[remove_idx] = False

    n_total = len(mask)
    n_remove = np.sum(~mask)
    n_remain = np.sum(mask)
    print(f"  NED low-z cut (z < {redshift_cut:.4f}): "
          f"{n_remove}/{n_total} removed, {n_remain} remaining")

    return tab[mask]


# ============================================================
# Main pipeline
# ============================================================

def run(cfg):
    """Run the full target separation pipeline."""

    print(f"\n{'=' * 60}")
    print(f"  Target: {cfg['target_name']}  (z = {cfg['target_redshift']})")
    print(f"{'=' * 60}\n")

    # 1. Load mega catalog and select target
    mega = Table.read(cfg['mega_file'])
    tab = mega[mega["CLUSTER"] == cfg['target_name']]
    print(f"Objects in {cfg['target_name']}: {len(tab)}")

    # 2. Sky footprint for NED query
    center_ra, center_dec, radius = utils.get_sky_footprint_center_radius(
        tab, buffer_fraction=0.2
    )

    # 3. NED low-redshift removal
    ned = load_ned_redshifts(cfg, center_ra, center_dec, radius)
    tab = remove_low_redshift(
        tab, ned,
        cfg['redshift_cut'],
        cfg['tolerance_deg'],
    )

    # 4. Known foreground removal
    foreground = Table.read(cfg['foreground_file'])
    tab = remove_by_match(
        tab, foreground, cfg['tolerance_deg'],
        cat1_ra='ra', cat1_dec='dec',
        label="Known foreground cut (color-color)",
    )

    # 5. Metacalibration selection + shear responsivity
    selected, R_S, c_total, mean_g1, mean_g2 = compute_R_S(
        mcal=tab,
        qual_cuts=cfg['qual_cuts'],
        mcal_shear=cfg['mcal_shear'],
    )

    # 6. Write output
    selected.write(cfg['outfile'], format='fits',
                   overwrite=cfg['overwrite'])
    print(f"\nWrote {len(selected)} objects to {cfg['outfile']}")

    # 7. Summary stats
    print("\n== Summary Stats ==\n")
    _ = utils.analyze_mcal_fits(cfg['outfile'], update_header=True)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    run(cfg)