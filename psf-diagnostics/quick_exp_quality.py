import os
import glob
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
# import astropy.io.fits
# import astropy.units
import numpy as np
from datetime import datetime
import glob
import re
# import warnings
import ngmix
import filecmp
from superbit_lensing.utils import gaia_query, make_psfex_model, radec_to_xy, extract_vignette, add_admom_columns, get_psf_model_file
from superbit_lensing.match import SkyCoordMatcher
import multiprocessing

DATA_DIR = "/scratch/sa.saha/new_data"
CODE_DIR = "/projects/mccleary_group/saha/codes/superbit-lensing"

base_dir = DATA_DIR
tolerance_deg = 1e-4
def ensure_admom_columns(cat_name):
    with fits.open(cat_name, mode='readonly') as hdul:
        cols = hdul[2].columns.names  # assuming data in first extension

    required_cols = {"T_ADMOM", "E1_ADMOM", "E2_ADMOM"}

    # Check if all required columns are already present
    if required_cols.issubset(cols):
        print(f"[Info] {cat_name} already has ADMOM columns — skipping addition.")
    else:
        print(f"[Info] Missing ADMOM columns in {cat_name}. Adding them...")
        cat = add_admom_columns(cat_name, mode="galsim")
        return cat  # return the updated catalog if needed

clusters = [
    "Abell3526",
    "Abell2163",
    "SMACSJ2031d8m4036",
    "MS1008d1m1224",
    "Abell2384a",
    # "MACSJ1931d8m2635",
    "RXCJ1514d9m1523",
    "Abell2345",
    # "SPTCLJ0411",
    "AbellS780",
    "AbellS0592",
    "Abell3411",
    "1E0657_Bullet",
    "Abell141",
    "Abell1689",
    "Abell2384b",
    "Abell3365",
    "Abell3571",
    "Abell3716S",
    "Abell3827",
    "COSMOS113",
    "COSMOSa",
    "COSMOSb",
    "COSMOSg",
    "COSMOSo",
    "MACSJ0723d3_7327_JWST",
    "MACSJ1105d7m1014",
    "MS2137m2353",
    "PLCKG287d0p32d9",
    "RXCJ1314d4m2515",
    "RXCJ2003d5m2323",
    "Z20_SPT_CLJ0135m5904",
    "Abell3192",
    "Abell3667",
    "ACT_CL_J0012_0855_J0012_0857",
    "COSMOSk",
    "MACSJ0416d1m2403",
    "RXJ1347d5m1145"
]
summary_rows = []

for cluster_idx, cluster in enumerate(clusters):
    print(f"\n [{cluster_idx+1}/{len(clusters)}] Processing Cluster: {cluster}")
    cat_folder = os.path.join(base_dir, cluster, "b", "cat")
    cat_files = sorted(glob.glob(os.path.join(cat_folder, "*_clean_cat.fits")))
    star_file = os.path.join(base_dir, 'catalogs',"stars" ,f"{cluster}_gaia_dr3.fits")
    try:
        gaia_stars = Table.read(star_file)
    except Exception as e:
        print(f'[WARNING] Could not open star file : {e}, doing a fresh new query')
        gaia_stars = gaia_query(cluster)
        try:
            gaia_stars.write(star_file, format='fits', overwrite=True)
            print(f'[INFO] Saved the queried file to {star_file}')
        except Exception as e:
            print(f'[WARNING] could not save it to fits file for future use')

    print(f"Found {len(gaia_stars)} GAIA stars")
    for exp_idx, cat_name in enumerate(cat_files):
        print(f"\n  [{exp_idx+1}/{len(cat_files)}] Processing exposure: {os.path.basename(cat_name)}")
        psf_model_file = get_psf_model_file(cat_name)
        ensure_admom_columns(cat_name)
        ss_fits = fits.open(cat_name)
        if len(ss_fits) == 3:
            # It is an ldac
            ext = 2
        else:
            ext = 1
        cat = ss_fits[ext].data

        print(f"    Catalog contains {len(cat)} objects")
        xim = cat["XWIN_IMAGE"]
        yim = cat["YWIN_IMAGE"]
        
        # Define central 50% region (25% area)
        xmin, xmax = np.min(xim), np.max(xim)
        ymin, ymax = np.min(yim), np.max(yim)
        
        width = xmax - xmin
        height = ymax - ymin
        
        # Scale factor for central 50% area
        scale = np.sqrt(0.25)  # ≈ 0.7071
        
        # Trim margins
        x_margin = (1 - scale) / 2 * width
        y_margin = (1 - scale) / 2 * height
        
        x_low, x_high = xmin + x_margin, xmax - x_margin
        y_low, y_high = ymin + y_margin, ymax - y_margin

        print(f"    Matching with GAIA stars...")
        matcher = SkyCoordMatcher(cat, gaia_stars,
                                  cat1_ratag='ALPHAWIN_J2000', cat1_dectag='DELTAWIN_J2000',
                                  cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000',
                                  return_idx=True, match_radius=1 * tolerance_deg)

        all_stars, matched2, idx1, idx2 = matcher.get_matched_pairs()

        print(f"    Matched {len(all_stars)} stars")
        # Initial cuts
        valid = (all_stars['MAG_AUTO'] > 16.5) & (all_stars['SNR_WIN'] > 20)

        # Filter
        good_stars = all_stars[valid]
        bad_stars = all_stars[~valid]

        # First Store fwhm scatter
        star_T_admom = good_stars["T_ADMOM"]
        star_fwhm = 2.355 * np.sqrt(star_T_admom / 2)
        # Keep only valid finite positive values
        valid_mask = np.isfinite(star_fwhm) & (star_fwhm > 0)
        star_fwhm_valid = star_fwhm[valid_mask]

        if len(star_fwhm_valid) == 0:
            print("[WARNING] NO VALID STARS FOUND. RED ALERT!!!!!")
            continue  # skip if no valid stars

        # Apply 1–99 percentile filtering
        p1, p99 = np.percentile(star_fwhm_valid, [1, 99])
        percentile_mask = (star_fwhm_valid >= p1) & (star_fwhm_valid <= p99)
        star_fwhm_filtered = star_fwhm_valid[percentile_mask]

        # Compute scatter
        if len(star_fwhm_filtered) > 0:
            std_fwhm = np.std(star_fwhm_filtered)
            median_fwhm = np.median(star_fwhm_filtered)
        else:
            std_fwhm = np.nan
            median_fwhm = np.nan

        # Now ellipticity        
        xim, yim = good_stars["XWIN_IMAGE"], good_stars["YWIN_IMAGE"]
        inner_region = (xim >= x_low) & (xim <= x_high) & (yim >= y_low) & (yim <= y_high)
        stars_ourskirts = good_stars[~inner_region]

        e1_stars_ot, e2_stars_ot = stars_ourskirts['E1_ADMOM'], stars_ourskirts['E2_ADMOM']
        ellip_mag = np.sqrt(e1_stars_ot**2 + e2_stars_ot**2)

        valid_ellip = np.isfinite(ellip_mag) & (ellip_mag >= 0)
        ellip_mag_valid = ellip_mag[valid_ellip]
        if len(ellip_mag_valid) > 0:
            # Apply percentile filtering for outliers
            p1_e, p99_e = np.percentile(ellip_mag_valid, [1, 99])
            ellip_filtered = ellip_mag_valid[(ellip_mag_valid >= p1_e) & (ellip_mag_valid <= p99_e)]
            median_e = np.median(ellip_filtered)
        else:
            median_e = np.nan

        print(f"    Selected {len(good_stars)} good stars after T cuts")
        row = dict(
            cluster_name=cluster,
            exp=exp_idx + 1,  # exposure number
            median_e=median_e,
            std_fwhm=std_fwhm,
            median_fwhm=median_fwhm,
            n_good_stars=len(good_stars),
            psf_model_file=psf_model_file
        )
        summary_rows.append(row)     

# --- After all clusters processed ---
if summary_rows:
    summary_table = Table(rows=summary_rows)
    summary_table.meta['DATE'] = datetime.now().isoformat()

    output_file = os.path.join(DATA_DIR, "psf_summary_table.fits")
    summary_table.write(output_file, overwrite=True)

    print(f"\n✅ Summary table saved to: {output_file}")
    print(f"   Contains {len(summary_table)} exposures from {len(clusters)} clusters.")
else:
    print("[Warning] No exposures processed successfully — no table created.")                   

