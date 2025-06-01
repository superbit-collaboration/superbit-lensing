import os
import argparse
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ipdb
import time
from superbit_lensing.match import SkyCoordMatcher
from superbit_lensing.color import sextractor_dual as sex
from superbit_lensing.utils import gaia_query, ned_query, get_sky_footprint_center_radius

lovoccs_table = ['RA', 'DEC', 'Redshift', 'Redshift_error']
desi_table = [
    'TARGETID', 'SURVEY', 'PROGRAM', 'OBJTYPE', 'SPECTYPE', 
    'TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR', 'ZWARN', 'ZCAT_NSPEC', 'ZCAT_PRIMARY'
]
desi_table_primary = ['TARGET_RA', 'TARGET_DEC', 'ZCAT_PRIMARY', 'OBJTYPE', 'ZWARN']
ned_table = ['RA', 'DEC', 'Redshift']
DESI_MASTER_FILE = "/projects/mccleary_group/superbit/desi_data/zall-pix-iron.fits"

MAG_ZP_b = 28.66794
MAG_ZP_g = 27.490537
MAG_ZP_u = 26.48623

def main(args):
    print("=== Arguments Passed to the Script ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================\n")
    # Construct file paths based on the inputs
    cluster_name = args.cluster_name
    datadir = args.datadir
    projection = args.swarp_projection_type
    config_dir = args.config_dir
    tolerance_deg = args.tolerance
    redshift = args.redshift
    snr_threshold =  args.snr_threshold
    no_weight = args.no_weight
    delz = 0.02

    base_path = os.path.join(datadir, cluster_name)
    redshift_file = os.path.join(datadir, f'catalogs/redshifts/{cluster_name}_NED_redshifts.csv')
    lovoccs_file = os.path.join(datadir, f'catalogs/lovoccs/{cluster_name}_lovoccs_redshifts.fits')
    if os.path.exists(DESI_MASTER_FILE):
        desi_file = DESI_MASTER_FILE
        print(f"Using DESI_MASTER_FILE: {DESI_MASTER_FILE}")
    else:
        desi_file = os.path.join(datadir, f'catalogs/desi/{cluster_name}_desi_spectra.fits')
        print(f"Using fall_back file: {desi_file}")

    # Construct file paths
    file_b_stars_union = f"{base_path}/b/coadd/{cluster_name}_coadd_b_starcat_union.fits"
    file_b_stars_fallback = f"{base_path}/b/coadd/{cluster_name}_coadd_b_starcat.fits"
    file_b_stars_fallback_2nd = os.path.join(datadir, f'catalogs/stars/{cluster_name}_gaia_starcat.fits')

    # Check for starcat_union first, else fall back to starcat
    if os.path.exists(file_b_stars_union):
        file_b_stars = file_b_stars_union
    elif os.path.exists(file_b_stars_fallback):
        file_b_stars = file_b_stars_fallback
        print(f"Warning: Using fallback star catalog for band b: {file_b_stars}")
    else:
        file_b_stars = file_b_stars_fallback_2nd
        print(f"Warning: Using Gaia fallback star catalog for band b: {file_b_stars}")
    
    try:
        star_data_b = Table.read(file_b_stars, format='fits', hdu=2)
    except Exception as e:
        print(f"Failed to read star catalog for band b ({e}), trying to query Gaia...")
        try:
            star_data_b = gaia_query(cluster_name)
            print(f"Gaia query was sucessfull, found {len(star_data_b)} objects")
            star_data_b.write(file_b_stars_fallback_2nd, format='fits', overwrite=True)
        except Exception as e:
            print(f"Failed to query Gaia for star catalog for band b ({e}), creating an empty catalog.")   
            star_data_b = Table(names=['ALPHAWIN_J2000', 'DELTAWIN_J2000'], dtype=['f8', 'f8'])
    print(f"Number of stars in the star catalog: {len(star_data_b)}")

    # Ensure output directory exists

    dual_mode_path = f"{base_path}/sextractor_dualmode"
    plot_dir = f"{dual_mode_path}/plots"
    output_dir = f"{dual_mode_path}/out"
    output_file = f"{plot_dir}/{cluster_name}_ubg_color_color.png"
    out_file_cm_bg = f"{plot_dir}/{cluster_name}_b_g_color_mag.png"
    out_file_cm_ub = f"{plot_dir}/{cluster_name}_u_b_color_mag.png"
    output_fits = f"{output_dir}/{cluster_name}_colors_mags.fits"
    output_star_fits = f"{output_dir}/{cluster_name}_colors_mags_stars.fits"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    diag_path = os.path.join(dual_mode_path, "diags")
    dual_cat_path = os.path.join(dual_mode_path, "cat")
    os.makedirs(diag_path, exist_ok=True)
    os.makedirs(dual_cat_path, exist_ok=True)
    
    swarp = sex.make_coadds_for_dualmode(datadir, cluster_name, config_dir=config_dir, overwrite_coadds=args.overwrite_coadds, projection=projection)
    coadd_files = swarp.coadd_file_names
    file_b, file_g, file_u = coadd_files['b'], coadd_files['g'], coadd_files['u']

    # Ensure input files exist
    if not os.path.exists(file_b) or not os.path.exists(file_g) or not os.path.exists(file_u):
        raise FileNotFoundError(f"One or more input files not found:\n{file_b}\n{file_g}\n{file_u}")    
    file_b_cat = f"{dual_cat_path}/{cluster_name}_coadd_b_cat.fits"
    file_g_cat = f"{dual_cat_path}/{cluster_name}_coadd_g_cat.fits"
    file_u_cat = f"{dual_cat_path}/{cluster_name}_coadd_u_cat.fits"

    if args.overwrite_cats:
        print("WARNING: Overwriting catalogs, this may take a while...")
        os.system(f"rm -rf {dual_cat_path}/*")
        os.system(f"rm -rf {diag_path}/*")
        os.system(f"rm -rf {output_dir}/*")

    try:
        # Read the FITS files
        data_b = Table.read(file_b_cat, format='fits', hdu=2)
        ra_b = data_b["ALPHAWIN_J2000"]

    except:
        print("WARNING: source extractor has not been run for band b, so running it again")
        if no_weight:
            sex._run_sextractor_single(file_b, dual_cat_path, config_dir, diag_dir=diag_path, mag_zp=MAG_ZP_b, use_weight=False)
        else:
            sex._run_sextractor_single(file_b, dual_cat_path, config_dir, diag_dir=diag_path, mag_zp=MAG_ZP_b)
        data_b = Table.read(file_b_cat, format='fits', hdu=2)
        ra_b = data_b["ALPHAWIN_J2000"]

    # Get footprint of the cluster
    center_ra_b, center_dec_b, radius_b = get_sky_footprint_center_radius(data_b, buffer_fraction=0.2)
    print(f"Cluster center: {center_ra_b}, {center_dec_b}")
    print(f"Cluster footprint radius: {radius_b}")

    try:
        # Read the FITS files
        data_g = Table.read(file_g_cat, format='fits', hdu=2)
        ra_g = data_g["ALPHAWIN_J2000"]
    except:
        print("WARNING: source extractor has not been run for band g, so running it again")
        if no_weight:
            sex._run_sextractor_dual(file_b, file_g, dual_cat_path, config_dir, diag_dir=diag_path, mag_zp=MAG_ZP_g,  use_weight=False)        
        else:
            sex._run_sextractor_dual(file_b, file_g, dual_cat_path, config_dir, diag_dir=diag_path,  mag_zp=MAG_ZP_g)
        data_g = Table.read(file_g_cat, format='fits', hdu=2)
    try:
        data_u = Table.read(file_u_cat, format='fits', hdu=2)
        ra_u = data_u["ALPHAWIN_J2000"]
    except:
        print("WARNING: source extractor has not been run for band u, so running it again")
        if no_weight:
            sex._run_sextractor_dual(file_b, file_u, dual_cat_path, config_dir, diag_dir=diag_path,  mag_zp=MAG_ZP_u, use_weight=False)
        else:
            sex._run_sextractor_dual(file_b, file_u, dual_cat_path, config_dir, diag_dir=diag_path,  mag_zp=MAG_ZP_u)
        data_u = Table.read(file_u_cat, format='fits', hdu=2)
    
    bg_sub_b_file = f"{diag_path}/{cluster_name}_coadd_b.sub.fits"
    bg_sub_g_file = f"{diag_path}/{cluster_name}_coadd_g.sub.fits"
    bg_sub_u_file = f"{diag_path}/{cluster_name}_coadd_u.sub.fits"
    bkg_rms_b_file = f"{diag_path}/{cluster_name}_coadd_b.bkg_rms.fits"
    bkg_rms_g_file = f"{diag_path}/{cluster_name}_coadd_g.bkg_rms.fits"
    bkg_rms_u_file = f"{diag_path}/{cluster_name}_coadd_u.bkg_rms.fits"

    try:
        ned_cat = Table.read(redshift_file, format='csv')
        ned_cat = ned_cat[ned_table]
    except Exception as e:
        print(f"Failed to read redshift file ({e}), Trying ned query...")
        try:
            ned_cat = ned_query(rad_deg=radius_b, ra_center=center_ra_b, dec_center=center_dec_b)
            print(f"NED query was sucessfull, found {len(ned_cat)} objects")
            ned_cat.write(redshift_file, format='csv', overwrite=True)
        except Exception as e:
            try:
                print(f"Failed to query NED for redshift file ({e}), trying again....")
                time.sleep(3)
                ned_cat = ned_query(rad_deg=radius_b/1.1, ra_center=center_ra_b, dec_center=center_dec_b)
                print(f"2nd NED query was sucessfull, found {len(ned_cat)} objects")
                ned_cat.write(redshift_file, format='csv', overwrite=True)
            except Exception as e:
                print(f"Ned query failed for 2nd time ({e}), creating an empty catalog.")
                ned_cat = Table(names=ned_table, dtype=['f8'] * len(ned_table))

    try:
        # Read the Lovoccs catalog
        lovoccs = Table.read(lovoccs_file, format='fits', hdu=1)
        lovoccs = lovoccs[lovoccs_table]
    except Exception as e:
        print(f"Failed to read lovoccs file ({e}), creating an empty catalog.")
        lovoccs = Table(names=lovoccs_table, dtype=['f8'] * len(lovoccs_table))

    try:
        col_data = {}
        with fits.open(desi_file, memmap=True) as hdul:
            for col in desi_table:
                col_data[col] = hdul[1].data[col]

        desi = Table(col_data)
        desi = desi[
            (desi['ZCAT_PRIMARY'] == True) &
            (desi['OBJTYPE'] == 'TGT') &
            (desi['ZWARN'] == 0)
        ]

    except Exception as e:
        print(f"Failed to read desi file ({e}), creating an empty catalog.")
        desi = Table(names=desi_table, dtype=['f8'] * len(desi_table))

    print("Star matching and object discarding is starting...")
    try:
        matcher_b_stars = SkyCoordMatcher(data_b, star_data_b, match_radius=tolerance_deg)
        matched_data_b_stars, matched_star_data_b = matcher_b_stars.get_matched_pairs()

        # Filter out stars from the main catalogs using both RA and DEC
        data_b_coords = list(zip(data_b['ALPHAWIN_J2000'], data_b['DELTAWIN_J2000']))
        matched_b_stars_coords = list(zip(matched_data_b_stars['ALPHAWIN_J2000'], matched_data_b_stars['DELTAWIN_J2000']))
        mask = np.array([(ra, dec) not in matched_b_stars_coords for ra, dec in data_b_coords])
    except Exception as e:
        print(f"Error during star matching in band b: {e}")
        mask = np.ones(len(data_b), dtype=bool)
    data_b_filtered = data_b[mask]
    data_g_filtered = data_g[mask]
    data_u_filtered = data_u[mask]

    discarded_count = np.sum(~mask)

    print(f"Band b - Total objects: {len(data_b)}, Stars: {len(star_data_b)}, Discarded: {discarded_count}, Proceeding to next step: {len(data_b_filtered)}")
    print(f"Band g - Total objects: {len(data_g)}, Stars: {len(star_data_b)}, Discarded: {discarded_count}, Proceeding to next step: {len(data_g_filtered)}")
    print(f"Band u - Total objects: {len(data_u)}, Stars: {len(star_data_b)}, Discarded: {discarded_count}, Proceeding to next step: {len(data_u_filtered)}")


    # Step 2: Interband Matching
    matched_data_b = data_b_filtered
    matched_data_g = data_g_filtered
    matched_data_u = data_u_filtered

    # Step 3: Combine Matched Data
    flux_b = matched_data_b['FLUX_AUTO']
    flux_g = matched_data_g['FLUX_AUTO']
    flux_u = matched_data_u['FLUX_AUTO']

    #valid_flux = (flux_b > 0) & (flux_g > 0) & (flux_u > 0)
    valid_flux = np.ones(len(matched_data_b), dtype=bool)
    matched_data_b_valid = matched_data_b[valid_flux]
    matched_data_g_valid = matched_data_g[valid_flux]
    matched_data_u_valid = matched_data_u[valid_flux]
    print(f"Number of objects with positive flux in all bands: {np.sum(valid_flux)}")
    print(f"Number of objects with invalid flux in all bands: {np.sum(~valid_flux)}")
    m_b = matched_data_b_valid["MAG_AUTO"] #-2.5 * np.log10(flux_b[valid_flux])
    m_g = matched_data_g_valid["MAG_AUTO"] #-2.5 * np.log10(flux_g[valid_flux])
    m_u = matched_data_u_valid["MAG_AUTO"] #-2.5 * np.log10(flux_u[valid_flux])
    m_b_err = matched_data_b_valid["MAGERR_AUTO"]  # Error in b-band magnitude
    m_g_err = matched_data_g_valid["MAGERR_AUTO"]  # Error in g-band magnitude  
    m_u_err = matched_data_u_valid["MAGERR_AUTO"]  # Error in u-band magnitude    
    
    color_index_bg = m_b - m_g
    color_index_ub = m_u - m_b

    # Calculate errors on color indices
    color_index_bg_err = np.sqrt(m_b_err**2 + m_g_err**2)
    color_index_ub_err = np.sqrt(m_u_err**2 + m_b_err**2)    

    # Use SkyCoordMatcher for NED matching
    matcher_ned = SkyCoordMatcher(ned_cat, matched_data_b, cat1_ratag='RA', cat1_dectag='DEC',
                 cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', return_idx=True, match_radius=1 * tolerance_deg)
    matched_ned, matched_data_b_ned, idx1_ned, idx2_ned = matcher_ned.get_matched_pairs()

    print(f"Number of matched galaxies with known ned redshifts: {len(matched_ned)}")

    matcher_lovoccs = SkyCoordMatcher(lovoccs, matched_data_b, cat1_ratag='RA', cat1_dectag='DEC',
                 cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', return_idx=True, match_radius=1 * tolerance_deg)
    matched_lovoccs, matched_data_b_lovoccs, idx1_lovoccs, idx2_lovoccs = matcher_lovoccs.get_matched_pairs()
    print(f"Number of matched galaxies with known lovoccs redshifts: {len(matched_lovoccs)}")

    matcher_desi = SkyCoordMatcher(desi, matched_data_b, cat1_ratag='TARGET_RA', cat1_dectag='TARGET_DEC',
                 cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', return_idx=True, match_radius=1 * tolerance_deg)
    matched_desi, matched_data_b_desi, idx1_desi, idx2_desi = matcher_desi.get_matched_pairs()
    print(f"Number of matched galaxies with known desi redshifts: {len(matched_desi)}")
    
    # --- Initialize redshift arrays for all matched_data_b entries ---
    del desi
    n = len(matched_data_b)
    redshifts_ned = np.full(n, np.nan)
    redshifts_lovoccs = np.full(n, np.nan)
    redshift_err_lovoccs = np.full(n, np.nan)
    redshifts_desi = np.full(n, np.nan)
    redshift_err_desi = np.full(n, np.nan)

    # Fill values for matched indices
    redshifts_ned[idx2_ned] = matched_ned["Redshift"]
    redshifts_lovoccs[idx2_lovoccs] = matched_lovoccs["Redshift"]
    redshift_err_lovoccs[idx2_lovoccs] = matched_lovoccs["Redshift_error"]
    redshifts_desi[idx2_desi] = matched_desi["Z"]
    redshift_err_desi[idx2_desi] = matched_desi["ZERR"]

    # --- Initialize DESI metadata columns ---
    desi_data_arrays = {}
    for col in desi_table:
        dtype = matched_desi[col].dtype
        kind = dtype.kind

        if kind == 'f':  # float
            fill_value = np.nan
        elif kind == 'i':  # integer
            fill_value = -1
        elif kind == 'b':  # boolean
            fill_value = False
        elif kind in ('U', 'S'):  # string (unicode or bytes)
            fill_value = ''
        else:
            # Skip unsupported dtypes
            print(f"Skipping column '{col}' due to unsupported dtype kind '{kind}'")
            continue

        desi_data_arrays[col] = np.full(n, fill_value, dtype=dtype)
    # Fill matched DESI metadata
    for col in desi_table:
        desi_data_arrays[col][idx2_desi] = matched_desi[col]

    z_best = np.full(len(matched_data_b), np.nan)
    z_best_err = np.full(len(matched_data_b), np.nan)
    z_source = np.full(len(matched_data_b), '', dtype='U10')

    # Priority: DESI > NED > LoVoCCS
    for i in range(len(matched_data_b)):
        if not np.isnan(redshifts_desi[i]):
            z_best[i] = redshifts_desi[i]
            z_best_err[i] = redshift_err_desi[i]
            z_source[i] = 'DESI'
        elif not np.isnan(redshifts_ned[i]):
            z_best[i] = redshifts_ned[i]
            z_best_err[i] = -1
            z_source[i] = 'NED'
        elif not np.isnan(redshifts_lovoccs[i]):
            z_best[i] = redshifts_lovoccs[i]
            z_best_err[i] = redshift_err_lovoccs[i]
            z_source[i] = 'LoVoCCS'

    if args.vignet_updater:
        print("Vignet updater is starting...")
        vignet_updater_b = sex.update_vignet(file_b, bg_sub_b_file, bkg_rms_b_file, matched_data_b_valid)
        vignet_updater_g = sex.update_vignet(file_g, bg_sub_g_file, bkg_rms_g_file, matched_data_g_valid)
        vignet_updater_u = sex.update_vignet(file_u, bg_sub_u_file, bkg_rms_u_file, matched_data_u_valid)

    if args.save_fits:
        ra_dec_table = matched_data_b['ALPHAWIN_J2000', 'DELTAWIN_J2000']
        ra_dec_table.rename_columns(['ALPHAWIN_J2000', 'DELTAWIN_J2000'], ['ra', 'dec'])
        # Create a new table with the selected columns and computed values
        final_table = Table()
        final_table['id'] = matched_data_b[valid_flux]["NUMBER"]
        final_table['ra'] = ra_dec_table['ra'][valid_flux]
        final_table['dec'] = ra_dec_table['dec'][valid_flux]
        final_table['m_b'] = m_b
        final_table['m_b_err'] = matched_data_b[valid_flux]["MAGERR_AUTO"]
        final_table['m_g'] = m_g
        final_table['m_g_err'] = matched_data_g[valid_flux]["MAGERR_AUTO"]
        final_table['m_u'] = m_u
        final_table['m_u_err'] = matched_data_u[valid_flux]["MAGERR_AUTO"]
        final_table['R_b'] = matched_data_b["FLUX_RADIUS"][valid_flux]
        final_table['R_g'] = matched_data_g["FLUX_RADIUS"][valid_flux]
        final_table['R_u'] = matched_data_u["FLUX_RADIUS"][valid_flux]
        final_table['R_b_prepsf'] = 0.6 * matched_data_b["FLUX_RADIUS"][valid_flux]
        final_table['R_g_prepsf'] = 0.6 * matched_data_g["FLUX_RADIUS"][valid_flux]
        final_table['R_u_prepsf'] = 0.6 * matched_data_u["FLUX_RADIUS"][valid_flux]
        final_table['FLUX_AUTO_b'] = flux_b[valid_flux]
        final_table['FLUXERR_AUTO_b'] = matched_data_b["FLUXERR_AUTO"][valid_flux]
        final_table['FLUX_AUTO_g'] = flux_g[valid_flux]
        final_table['FLUXERR_AUTO_g'] = matched_data_g["FLUXERR_AUTO"][valid_flux]
        final_table['FLUX_AUTO_u'] = flux_u[valid_flux]
        final_table['FLUXERR_AUTO_u'] = matched_data_u["FLUXERR_AUTO"][valid_flux]
        final_table['VIGNET_b'] = matched_data_b["VIGNET"][valid_flux]
        final_table['VIGNET_g'] = matched_data_g["VIGNET"][valid_flux]
        final_table['VIGNET_u'] = matched_data_u["VIGNET"][valid_flux]
        if args.vignet_updater:
            final_table['VIGNET_b_im'] = vignet_updater_b.cat_data["im_vignett"]
            final_table['VIGNET_g_im'] = vignet_updater_g.cat_data["im_vignett"]
            final_table['VIGNET_u_im'] = vignet_updater_u.cat_data["im_vignett"]
            final_table['VIGNET_b_wt'] = vignet_updater_b.cat_data["weight_vignett"]
            final_table['VIGNET_g_wt'] = vignet_updater_g.cat_data["weight_vignett"]
            final_table['VIGNET_u_wt'] = vignet_updater_u.cat_data["weight_vignett"]
            final_table['VIGNET_b_bkg_wt'] = vignet_updater_b.cat_data["rms_weight_vignett"]
            final_table['VIGNET_g_bkg_wt'] = vignet_updater_g.cat_data["rms_weight_vignett"]
            final_table['VIGNET_u_bkg_wt'] = vignet_updater_u.cat_data["rms_weight_vignett"]
            final_table['MASK_b'] = vignet_updater_b.cat_data["mask"]
            final_table['MASK_g'] = vignet_updater_g.cat_data["mask"]
            final_table['MASK_u'] = vignet_updater_u.cat_data["mask"]
            final_table['at_edge'] = vignet_updater_b.cat_data["is_at_edge"]
        final_table['SNR_b'] = matched_data_b["SNR_WIN"][valid_flux]
        final_table['SNR_g'] = matched_data_g["SNR_WIN"][valid_flux]
        final_table['SNR_u'] = matched_data_u["SNR_WIN"][valid_flux]
        final_table['SNR_combined'] = np.sqrt(final_table['SNR_b']**2 + final_table['SNR_g']**2 + final_table['SNR_u']**2)
        final_table['color_bg'] = color_index_bg
        final_table['color_ub'] = color_index_ub
        final_table['color_bg_err'] = color_index_bg_err
        final_table['color_ub_err'] = color_index_ub_err
        final_table['Z_ned'] = redshifts_ned[valid_flux]
        final_table['Z_lovoccs'] = redshifts_lovoccs[valid_flux]
        final_table['ZERR_lovoccs'] = redshift_err_lovoccs[valid_flux]
        final_table['Z_desi'] = redshifts_desi[valid_flux]
        final_table['ZERR_desi'] = redshift_err_desi[valid_flux]
        # Add DESI metadata columns
        for col in desi_table:
            final_table[col] = desi_data_arrays[col][valid_flux]

        final_table['Z_best'] = z_best[valid_flux]
        final_table['ZERR_best'] = z_best_err[valid_flux]
        final_table['Z_source'] = z_source[valid_flux]
        # Save as a FITS file
        final_table.write(output_fits, format='fits', overwrite=True)        

    # Step 5: Classify NED Matches by Redshift
    cluster_redshift = redshift
    cluster_redshift_up = cluster_redshift + delz
    cluster_redshift_down = cluster_redshift - delz
    z_matched = z_best

    # Filter out entries where z_best is NaN (or empty, if applicable)
    valid_z_best_indices = ~np.isnan(z_best)  # Create mask to exclude NaNs
    print(f"Number of objects with valid redshifts: {np.sum(valid_z_best_indices)}")
    # Apply this mask to all relevant data arrays
    z_best_filtered = z_best[valid_z_best_indices]
    matched_data_b_filtered = matched_data_b[valid_z_best_indices]
    matched_data_g_filtered = matched_data_g[valid_z_best_indices]
    matched_data_u_filtered = matched_data_u[valid_z_best_indices]

    #filter with snr
    #snr_threshold = -1e30
    snr_mask = (matched_data_b_filtered['SNR_WIN']>snr_threshold) & (matched_data_g_filtered['SNR_WIN']>snr_threshold) & (matched_data_u_filtered['SNR_WIN']>snr_threshold)
    print(f"Number of objects with SNR > {snr_threshold} in all bands: {np.sum(snr_mask)}")
    z_best_filtered = z_best_filtered[snr_mask]
    matched_data_b_filtered = matched_data_b_filtered[snr_mask]
    matched_data_g_filtered = matched_data_g_filtered[snr_mask]
    matched_data_u_filtered = matched_data_u_filtered[snr_mask]

    # Now, use the filtered z_best for classification
    z_matched = z_best_filtered  # Use the valid (non-NaN) redshift values

    high_z_indices = np.where(z_matched > cluster_redshift_up)[0]
    low_z_indices = np.where(z_matched <= cluster_redshift_down)[0]
    mid_z_indices = np.where((z_matched > cluster_redshift_down) & (z_matched <= cluster_redshift_up))[0]

    # Filter matched data by redshift categories (high_z, mid_z, low_z)
    high_z_b = matched_data_b_filtered[high_z_indices]
    high_z_g = matched_data_g_filtered[high_z_indices]
    high_z_u = matched_data_u_filtered[high_z_indices]

    low_z_b = matched_data_b_filtered[low_z_indices]
    low_z_g = matched_data_g_filtered[low_z_indices]
    low_z_u = matched_data_u_filtered[low_z_indices]

    mid_z_b = matched_data_b_filtered[mid_z_indices]
    mid_z_g = matched_data_g_filtered[mid_z_indices]
    mid_z_u = matched_data_u_filtered[mid_z_indices]

    print(f"Galaxies with z > {cluster_redshift_up:.2f}: {len(high_z_indices)}")
    print(f'Galaxies with {cluster_redshift_down:.2f} < z ≤ {cluster_redshift_up:.2f}: {len(mid_z_indices)}')
    print(f"Galaxies with z ≤ {cluster_redshift_down:.2f}: {len(low_z_indices)}")

    # Step 6: Compute Magnitudes for NED Matches
    flux_b_high = high_z_b['FLUX_AUTO']
    flux_g_high = high_z_g['FLUX_AUTO']
    flux_u_high = high_z_u['FLUX_AUTO']

    flux_b_low = low_z_b['FLUX_AUTO']
    flux_g_low = low_z_g['FLUX_AUTO']
    flux_u_low = low_z_u['FLUX_AUTO']

    flux_b_mid = mid_z_b['FLUX_AUTO']
    flux_g_mid = mid_z_g['FLUX_AUTO']
    flux_u_mid = mid_z_u['FLUX_AUTO']

    valid_flux_high = (flux_b_high > 0) & (flux_g_high > 0) & (flux_u_high > 0)
    valid_flux_low = (flux_b_low > 0) & (flux_g_low > 0) & (flux_u_low > 0)
    valid_flux_mid = (flux_b_mid > 0) & (flux_g_mid > 0) & (flux_u_mid > 0)
    print(f"Number of objects with invalid fluxes in high_z: {np.sum(~valid_flux_high)}")
    print(f"Number of objects with invalid fluxes in mid_z: {np.sum(~valid_flux_mid)}")
    print(f"Number of objects with invalid fluxes in low_z: {np.sum(~valid_flux_low)}")

    m_b_high = high_z_b[valid_flux_high]["MAG_AUTO"] #-2.5 * np.log10(flux_b_high[valid_flux_high])
    m_g_high = high_z_g[valid_flux_high]["MAG_AUTO"] #-2.5 * np.log10(flux_g_high[valid_flux_high]) 
    m_u_high = high_z_u[valid_flux_high]["MAG_AUTO"] #-2.5 * np.log10(flux_u_high[valid_flux_high])
    color_index_bg_high = m_b_high - m_g_high
    color_index_ub_high = m_u_high - m_b_high

    m_b_low = low_z_b[valid_flux_low]["MAG_AUTO"] #-2.5 * np.log10(flux_b_low[valid_flux_low])
    m_g_low = low_z_g[valid_flux_low]["MAG_AUTO"] #-2.5 * np.log10(flux_g_low[valid_flux_low])
    m_u_low = low_z_u[valid_flux_low]["MAG_AUTO"] #-2.5 * np.log10(flux_u_low[valid_flux_low])
    color_index_bg_low = m_b_low - m_g_low
    color_index_ub_low = m_u_low - m_b_low

    m_b_mid = mid_z_b[valid_flux_mid]["MAG_AUTO"] #-2.5 * np.log10(flux_b_mid[valid_flux_mid])
    m_g_mid = mid_z_g[valid_flux_mid]["MAG_AUTO"]  #-2.5 * np.log10(flux_g_mid[valid_flux_mid])
    m_u_mid = mid_z_u[valid_flux_mid]["MAG_AUTO"] #-2.5 * np.log10(flux_u_mid[valid_flux_mid])
    color_index_bg_mid = m_b_mid - m_g_mid
    color_index_ub_mid = m_u_mid - m_b_mid


    # Interband Star Matching
    matched_star_data_b = data_b[~mask]
    matched_star_data_g = data_g[~mask]
    matched_star_data_u = data_u[~mask]

    # Step 7: Compute Magnitudes for Matched Stars
    flux_stars_b = matched_star_data_b['FLUX_AUTO']
    flux_stars_g = matched_star_data_g['FLUX_AUTO']
    flux_stars_u = matched_star_data_u['FLUX_AUTO']

    valid_flux_stars = (flux_stars_b > 0) & (flux_stars_g > 0) & (flux_stars_u > 0)
    discard_count = np.sum(~valid_flux_stars)
    print(f"Number of stars discarded due to invalid flux: {discard_count}")
    m_stars_b = matched_star_data_b[valid_flux_stars]['MAG_AUTO'] #-2.5 * np.log10(flux_stars_b[valid_flux_stars])
    m_stars_g = matched_star_data_g[valid_flux_stars]['MAG_AUTO'] #-2.5 * np.log10(flux_stars_g[valid_flux_stars])
    m_stars_u = matched_star_data_u[valid_flux_stars]['MAG_AUTO'] #-2.5 * np.log10(flux_stars_u[valid_flux_stars])
    color_bg_stars = m_stars_b - m_stars_g
    color_ub_stars = m_stars_u - m_stars_b

    if args.save_fits:
        ra_dec_table = matched_star_data_b['ALPHAWIN_J2000', 'DELTAWIN_J2000']
        ra_dec_table.rename_columns(['ALPHAWIN_J2000', 'DELTAWIN_J2000'], ['ra', 'dec'])
        # Create a new table with the selected columns and computed values
        final_table = Table()
        final_table['ra'] = ra_dec_table['ra'][valid_flux_stars]
        final_table['dec'] = ra_dec_table['dec'][valid_flux_stars]
        final_table['m_b'] = m_stars_b
        final_table['m_g'] = m_stars_g
        final_table['m_u'] = m_stars_u
        final_table['R_b'] = matched_star_data_b["FLUX_RADIUS"][valid_flux_stars]
        final_table['R_g'] = matched_star_data_g["FLUX_RADIUS"][valid_flux_stars]
        final_table['R_u'] = matched_star_data_u["FLUX_RADIUS"][valid_flux_stars]
        final_table['R_b_prepsf'] = 0.6 * matched_star_data_b["FLUX_RADIUS"][valid_flux_stars]
        final_table['R_g_prepsf'] = 0.6 * matched_star_data_g["FLUX_RADIUS"][valid_flux_stars]
        final_table['R_u_prepsf'] = 0.6 * matched_star_data_u["FLUX_RADIUS"][valid_flux_stars]
        final_table['FLUX_AUTO_b'] = flux_stars_b[valid_flux_stars]
        final_table['FLUX_AUTO_g'] = flux_stars_g[valid_flux_stars]
        final_table['FLUX_AUTO_u'] = flux_stars_u[valid_flux_stars]
        final_table['VIGNET_b'] = matched_star_data_b["VIGNET"][valid_flux_stars]
        final_table['VIGNET_g'] = matched_star_data_g["VIGNET"][valid_flux_stars]
        final_table['VIGNET_u'] = matched_star_data_u["VIGNET"][valid_flux_stars]
        final_table['color_bg'] = color_bg_stars
        final_table['color_ub'] = color_ub_stars

        # Save as a FITS file
        final_table.write(output_star_fits, format='fits', overwrite=True)        
        # Modify the FITS header to store descriptions inline
        with fits.open(output_star_fits, mode='update') as hdul:
            hdr = hdul[1].header  # Access the table header

            # Add descriptions to column definitions
            hdr['TTYPE1'] = ('ra', 'Right Ascension (J2000) [degree]')
            hdr['TTYPE2'] = ('dec', 'Declination (J2000) [degree]')
            hdr['TTYPE3'] = ('m_b', 'Magnitude in b-band')
            hdr['TTYPE4'] = ('m_g', 'Magnitude in g-band')
            hdr['TTYPE5'] = ('m_u', 'Magnitude in u-band')
            hdr['TTYPE6'] = ('R_b', 'Half-light radius in b-band [arcsec]')
            hdr['TTYPE7'] = ('R_g', 'Half-light radius in g-band [arcsec]')
            hdr['TTYPE8'] = ('R_u', 'Half-light radius in u-band [arcsec]')
            hdr['TTYPE9'] = ('R_b_prepsf', 'pre-PSF Half-light radius in b-band [arcsec]')
            hdr['TTYPE10'] = ('R_g_prepsf', 'pre-PSF Half-light radius in g-band [arcsec]')
            hdr['TTYPE11'] = ('R_u_prepsf', 'pre-PSF Half-light radius in u-band [arcsec]')
            hdr['TTYPE12'] = ('FLUX_AUTO_b', 'SE flux measurement in b-band')
            hdr['TTYPE13'] = ('FLUX_AUTO_g', 'SE flux measurement in g-band')
            hdr['TTYPE14'] = ('FLUX_AUTO_u', 'SE flux measurement in u-band')
            hdr['TTYPE15'] = ('VIGNET_b', 'VIGNET in b-band')
            hdr['TTYPE16'] = ('VIGNET_g', 'VIGNET in g-band')
            hdr['TTYPE17'] = ('VIGNET_u', 'VIGNET in u-band')
            hdr['TTYPE18'] = ('color_bg', 'Color index (m_b - m_g)')
            hdr['TTYPE19'] = ('color_ub', 'Color index (m_u - m_b)')

            # Save changes
            hdul.flush()

    # Step 8: Plot the Color-Magnitude Diagram
    plt.figure(figsize=(8, 6))
    plt.scatter(color_index_bg, color_index_ub, s=5, alpha=0.10, color='blue', label='Galaxies')
    if args.plot_stars:
        plt.scatter(color_bg_stars, color_ub_stars, s=5, alpha=0.10, color='red', label='Stars')

    if args.plot_redshifts:
        plt.scatter(color_index_bg_high, color_index_ub_high, s=10, alpha=0.3, color='orange', label=f'High-z (z > {cluster_redshift_up:.2f}): : {len(high_z_indices)}')
        plt.scatter(color_index_bg_mid, color_index_ub_mid, s=10, alpha=0.3, color='lime', label=f'Members ({cluster_redshift_down:.2f} < z ≤ {cluster_redshift_up:.2f}): {len(mid_z_indices)}')
        plt.scatter(color_index_bg_low, color_index_ub_low, s=10, alpha=0.3, color='red', label=f'Low-z (z ≤ {cluster_redshift_down:.2f}): {len(low_z_indices)}')

    #plt.ylim(-4.2, 3.8)
    #plt.xlim(-20, -2)
    plt.xlabel(f'$m_b - m_g$')
    plt.ylabel(f'$m_u - m_b$')
    plt.title(f'{cluster_name}, Redshift={redshift}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to '{output_file}'")

    if args.plot_color_mag:
        plt.figure(figsize=(8, 6))
        plt.scatter(m_b, color_index_bg, s=5, alpha=0.10, color='blue', label='Galaxies')
        if args.plot_stars:
            plt.scatter(m_stars_b, color_bg_stars, s=5, alpha=0.10, color='red', label='Stars')

        if args.plot_redshifts:
            plt.scatter(m_b_high, color_index_bg_high, s=10, alpha=0.3, facecolors='orange', label=f'High-z (z > {cluster_redshift_up:.2f}): : {len(high_z_indices)}')
            plt.scatter(m_b_low, color_index_bg_low, s=10, alpha=0.3, facecolors='red', label=f'Low-z (z ≤ {cluster_redshift_down:.2f}): {len(low_z_indices)}')
            plt.scatter(m_b_mid, color_index_bg_mid, s=10, alpha=0.3, facecolors='lime', label=f'Members ({cluster_redshift_down:.2f} < z ≤ {cluster_redshift_up:.2f}): {len(mid_z_indices)}')


        #plt.ylim(-4.2, 3.8)
        #plt.xlim(-20, -2)
        plt.xlabel(f'$m_b$')
        plt.ylabel(f'$m_b - m_g$')
        plt.title(f'{cluster_name}, Redshift={redshift}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.savefig(out_file_cm_bg, dpi=300)
        print(f"Plot saved to '{out_file_cm_bg}'")        

        plt.figure(figsize=(8, 6))
        plt.scatter(m_b, color_index_ub, s=5, alpha=0.10, color='blue', label='Galaxies')
        if args.plot_stars:
            plt.scatter(m_stars_b, color_ub_stars, s=5, alpha=0.10, color='red', label='Stars')

        if args.plot_redshifts:
            plt.scatter(m_b_high, color_index_ub_high, s=10, edgecolors='black', facecolors='orange', label=f'High-z (z > {cluster_redshift_up:.2f}): : {len(high_z_indices)}')
            plt.scatter(m_b_mid, color_index_ub_mid, s=10, edgecolors='black', facecolors='lime', label=f'Members ({cluster_redshift_down:.2f} < z ≤ {cluster_redshift_up:.2f}): {len(mid_z_indices)}')
            plt.scatter(m_b_low, color_index_ub_low, s=10, edgecolors='black', facecolors='red', label=f'Low-z (z ≤ {cluster_redshift_down:.2f}): {len(low_z_indices)}')

        #plt.ylim(-4.2, 3.8)
        #plt.xlim(-20, -2)
        plt.xlabel(f'$m_b$')
        plt.ylabel(f'$m_u - m_b$')
        plt.title(f'{cluster_name}, Redshift={redshift}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.savefig(out_file_cm_ub, dpi=300)
        print(f"Plot saved to '{out_file_cm_ub}'")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine bands for a given cluster.")
    parser.add_argument("cluster_name", type=str, 
                        default=os.getenv("cluster_name", ""), 
                        help="Name of the cluster (default: value from $cluster_name)")
    parser.add_argument("--datadir", type=str, 
                        default=os.getenv("DATADIR", ""), 
                        help="Directory containing the data files")
    parser.add_argument("--config_dir", type=str, 
                        default=os.path.join(os.getenv("CODEDIR", ""), "superbit_lensing/medsmaker/superbit/astro_config"), 
                        help="Directory containing the config files")
    parser.add_argument("--tolerance", type=float, 
                        default=1e-4, 
                        help="Angular tolerance in degrees (default: 1e-4)")
    parser.add_argument("--redshift", type=float, 
                        default=float(os.getenv("cluster_redshift", 0.5)), 
                        help="Redshift threshold for classification (default: value from $cluster_redshift or 0.5)")
    parser.add_argument("--snr_threshold", type=float,
                        default=-1e30,
                        help="Signal-to-noise ratio threshold for galaxy selection (default: -1e30)")
    parser.add_argument("--swarp_projection_type", type=str,
                        default="TPV",
                        help="Projection type for swarp (default: 'TPV')")
    parser.add_argument("--overwrite_coadds", action="store_true", help="Overwrite existing coadds")
    parser.add_argument("--overwrite_cats", action="store_true", help="Overwrite existing catalogs")
    parser.add_argument('--no_weight', action='store_true', help='Do not Use the coadd weight image for SE')
    parser.add_argument('--vignet_updater', action='store_true', help='Update the vignette image')
    parser.add_argument("--plot_color_mag", action="store_true", help="Plot the color-magnitude diagrams")
    parser.add_argument("--plot_stars", action="store_true", help="Plot stars in the color-magnitude diagram")
    parser.add_argument("--plot_redshifts", action="store_true", help="Plot NED galaxies in the color-magnitude diagram")
    parser.add_argument('--save_fits', action='store_true', help='Save the output FITS file')
    
    args = parser.parse_args()

    main(args)