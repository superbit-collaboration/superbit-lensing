import os
import argparse
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from superbit_lensing.match import SkyCoordMatcher
from superbit_lensing.color import sextractor_dual as sex

def main(args):
    print("=== Arguments Passed to the Script ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================\n")
    # Construct file paths based on the inputs
    cluster_name = args.cluster_name
    datadir = args.datadir
    config_dir = args.config_dir
    tolerance_deg = args.tolerance
    redshift = args.redshift
    delz = 0.02

    base_path = os.path.join(datadir, cluster_name)
    redshift_file = os.path.join(datadir, f'catalogs/redshifts/{cluster_name}_NED_redshifts.csv')
    lovoccs_file = os.path.join(datadir, f'catalogs/lovoccs/{cluster_name}_lovoccs_redshifts.fits')

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
        print(f"Failed to read star catalog for band b ({e}), creating an empty catalog.")
        star_data_b = Table(names=['ALPHAWIN_J2000', 'DELTAWIN_J2000'], dtype=['f8', 'f8'])
    print(f"Number of stars in the star catalog: {len(star_data_b)}")

    try:
        ned_cat = Table.read(redshift_file, format='csv')
        ned_cat = ned_cat['RA', 'DEC', 'Redshift']
    except Exception as e:
        print(f"Failed to read redshift file ({e}), creating an empty catalog.")
        ned_cat = Table(names=['RA', 'DEC', 'Redshift'], dtype=['f8', 'f8', 'f8'])

    if args.plot_lovoccs:
        try:
            # Read the Lovoccs catalog
            lovoccs = Table.read(lovoccs_file, format='fits', hdu=1)
            lovoccs = lovoccs['RA', 'DEC', 'Redshift']
            
            # Combine both catalogs
            ned_cat = vstack([ned_cat, lovoccs])  
            print("Successfully combined NED and Lovoccs catalogs.")
        except Exception as e:
            print(f"No Lovoccs file found, skipping... ({e})")


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
    
    swarp = sex.make_coadds_for_dualmode(datadir, cluster_name, config_dir=config_dir, overwrite_coadds=args.overwrite_coadds)
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

    try:
        # Read the FITS files
        data_b = Table.read(file_b_cat, format='fits', hdu=2)
        ra_b = data_b["ALPHAWIN_J2000"]

    except:
        print("WARNING: source extractor has not been run for band b, so running it again")
        sex._run_sextractor_single(file_b, dual_cat_path, config_dir, diag_dir=diag_path)
        data_b = Table.read(file_b_cat, format='fits', hdu=2)
        ra_b = data_b["ALPHAWIN_J2000"]

    try:
        # Read the FITS files
        data_g = Table.read(file_g_cat, format='fits', hdu=2)
        ra_g = data_g["ALPHAWIN_J2000"]
    except:
        print("WARNING: source extractor has not been run for band g, so running it again")        
        sex._run_sextractor_dual(file_b, file_g, dual_cat_path, config_dir, diag_dir=diag_path)
        data_g = Table.read(file_g_cat, format='fits', hdu=2)
    try:
        data_u = Table.read(file_u_cat, format='fits', hdu=2)
        ra_u = data_u["ALPHAWIN_J2000"]
    except:
        print("WARNING: source extractor has not been run for band u, so running it again")
        sex._run_sextractor_dual(file_b, file_u, dual_cat_path, config_dir, diag_dir=diag_path)
        data_u = Table.read(file_u_cat, format='fits', hdu=2)
    
    print("Star matching and object discarding is starting...")
    try:
        matcher_b_stars = SkyCoordMatcher(data_b, star_data_b, match_radius=tolerance_deg)
        # Use SkyCoordMatcher for star matching in band b
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

    valid_flux = (flux_b > 0) & (flux_g > 0) & (flux_u > 0)
    print(f"Number of objects with positive flux in all bands: {np.sum(valid_flux)}")
    m_b = -2.5 * np.log10(flux_b[valid_flux])
    m_g = -2.5 * np.log10(flux_g[valid_flux])
    m_u = -2.5 * np.log10(flux_u[valid_flux])
    color_index_bg = m_b - m_g
    color_index_ub = m_u - m_b

    # Use SkyCoordMatcher for NED matching
    matcher_ned = SkyCoordMatcher(ned_cat, matched_data_b, cat1_ratag='RA', cat1_dectag='DEC',
                 cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', return_idx=True, match_radius=1 * tolerance_deg)
    matched_ned, matched_data_b_ned, idx1, idx2 = matcher_ned.get_matched_pairs()

    print(f"Number of matched galaxies with known redshifts: {len(matched_ned)}")
    # Initialize redshift array with NaNs for all objects
    redshifts = np.full(len(matched_data_b), np.nan)

    # Assign redshifts to matched indices
    redshifts[idx2] = matched_ned["Redshift"]

    if args.save_fits:
        ra_dec_table = matched_data_b['ALPHAWIN_J2000', 'DELTAWIN_J2000']
        ra_dec_table.rename_columns(['ALPHAWIN_J2000', 'DELTAWIN_J2000'], ['ra', 'dec'])
        # Create a new table with the selected columns and computed values
        final_table = Table()
        final_table['ra'] = ra_dec_table['ra'][valid_flux]
        final_table['dec'] = ra_dec_table['dec'][valid_flux]
        final_table['m_b'] = m_b
        final_table['m_g'] = m_g
        final_table['m_u'] = m_u
        final_table['R_b'] = matched_data_b["FLUX_RADIUS"][valid_flux]
        final_table['R_g'] = matched_data_g["FLUX_RADIUS"][valid_flux]
        final_table['R_u'] = matched_data_u["FLUX_RADIUS"][valid_flux]
        final_table['R_b_prepsf'] = 0.6 * matched_data_b["FLUX_RADIUS"][valid_flux]
        final_table['R_g_prepsf'] = 0.6 * matched_data_g["FLUX_RADIUS"][valid_flux]
        final_table['R_u_prepsf'] = 0.6 * matched_data_u["FLUX_RADIUS"][valid_flux]
        final_table['FLUX_AUTO_b'] = flux_b[valid_flux]
        final_table['FLUX_AUTO_g'] = flux_g[valid_flux]
        final_table['FLUX_AUTO_u'] = flux_u[valid_flux]
        final_table['MAG_AUTO_b'] = matched_data_b["MAG_AUTO"][valid_flux]
        final_table['MAG_AUTO_g'] = matched_data_g["MAG_AUTO"][valid_flux]
        final_table['MAG_AUTO_u'] = matched_data_u["MAG_AUTO"][valid_flux]
        final_table['VIGNET_b'] = matched_data_b["VIGNET"][valid_flux]
        final_table['VIGNET_g'] = matched_data_g["VIGNET"][valid_flux]
        final_table['VIGNET_u'] = matched_data_u["VIGNET"][valid_flux]
        final_table['color_bg'] = color_index_bg
        final_table['color_ub'] = color_index_ub
        final_table['redshift'] = redshifts[valid_flux]

        # Save as a FITS file
        final_table.write(output_fits, format='fits', overwrite=True)        
        # Modify the FITS header to store descriptions inline
        with fits.open(output_fits, mode='update') as hdul:
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
            hdr['TTYPE15'] = ('MAG_AUTO_b', 'SE magnitude in b-band')
            hdr['TTYPE16'] = ('MAG_AUTO_g', 'SE magnitude in g-band')
            hdr['TTYPE17'] = ('MAG_AUTO_u', 'SE magnitude in u-band')
            hdr['TTYPE18'] = ('VIGNET_b', 'VIGNET in b-band')
            hdr['TTYPE19'] = ('VIGNET_g', 'VIGNET in g-band')
            hdr['TTYPE20'] = ('VIGNET_u', 'VIGNET in u-band')
            hdr['TTYPE21'] = ('color_bg', 'Color index (m_b - m_g)')
            hdr['TTYPE22'] = ('color_ub', 'Color index (m_u - m_b)')
            hdr['TTYPE23'] = ('redshift', 'Available redshift')

            # Save changes
            hdul.flush()

    # Step 5: Classify NED Matches by Redshift
    cluster_redshift = redshift
    cluster_redshift_up = cluster_redshift + delz
    cluster_redshift_down = cluster_redshift - delz
    z_matched = matched_ned['Redshift']

    high_z_indices = np.where(z_matched > cluster_redshift_up)[0]
    low_z_indices = np.where(z_matched <= cluster_redshift_down)[0]
    mid_z_indices = np.where((z_matched > cluster_redshift_down) & (z_matched <= cluster_redshift_up))[0]

    high_z_b = matched_data_b_ned[high_z_indices]
    low_z_b = matched_data_b_ned[low_z_indices]

    print(f"Galaxies with z > {cluster_redshift_up:.2f}: {len(high_z_indices)}")
    print(f'Galaxies with {cluster_redshift_down:.2f} < z ≤ {cluster_redshift_up:.2f}: {len(mid_z_indices)}')
    print(f"Galaxies with z ≤ {cluster_redshift_down:.2f}: {len(low_z_indices)}")

    # Step 6: Compute Magnitudes for NED Matches
    flux_b_high = high_z_b['FLUX_AUTO']
    flux_g_high = matched_data_g[idx2][high_z_indices]['FLUX_AUTO']
    flux_u_high = matched_data_u[idx2][high_z_indices]['FLUX_AUTO']

    flux_b_low = low_z_b['FLUX_AUTO']
    flux_g_low = matched_data_g[idx2][low_z_indices]['FLUX_AUTO']
    flux_u_low = matched_data_u[idx2][low_z_indices]['FLUX_AUTO']

    flux_b_mid = matched_data_b_ned[mid_z_indices]['FLUX_AUTO']
    flux_g_mid = matched_data_g[idx2][mid_z_indices]['FLUX_AUTO']
    flux_u_mid = matched_data_u[idx2][mid_z_indices]['FLUX_AUTO']

    valid_flux_high = (flux_b_high > 0) & (flux_g_high > 0) & (flux_u_high > 0)
    valid_flux_low = (flux_b_low > 0) & (flux_g_low > 0) & (flux_u_low > 0)
    valid_flux_mid = (flux_b_mid > 0) & (flux_g_mid > 0) & (flux_u_mid > 0)

    m_b_high = -2.5 * np.log10(flux_b_high[valid_flux_high])
    m_g_high = -2.5 * np.log10(flux_g_high[valid_flux_high])
    m_u_high = -2.5 * np.log10(flux_u_high[valid_flux_high])
    color_index_bg_high = m_b_high - m_g_high
    color_index_ub_high = m_u_high - m_b_high

    m_b_low = -2.5 * np.log10(flux_b_low[valid_flux_low])
    m_g_low = -2.5 * np.log10(flux_g_low[valid_flux_low])
    m_u_low = -2.5 * np.log10(flux_u_low[valid_flux_low])
    color_index_bg_low = m_b_low - m_g_low
    color_index_ub_low = m_u_low - m_b_low

    m_b_mid = -2.5 * np.log10(flux_b_mid[valid_flux_mid])
    m_g_mid = -2.5 * np.log10(flux_g_mid[valid_flux_mid])
    m_u_mid = -2.5 * np.log10(flux_u_mid[valid_flux_mid])
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
    m_stars_b = -2.5 * np.log10(flux_stars_b[valid_flux_stars])
    m_stars_g = -2.5 * np.log10(flux_stars_g[valid_flux_stars])
    m_stars_u = -2.5 * np.log10(flux_stars_u[valid_flux_stars])
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
        final_table['MAG_AUTO_b'] = matched_star_data_b["MAG_AUTO"][valid_flux_stars]
        final_table['MAG_AUTO_g'] = matched_star_data_g["MAG_AUTO"][valid_flux_stars]
        final_table['MAG_AUTO_u'] = matched_star_data_u["MAG_AUTO"][valid_flux_stars]
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
            hdr['TTYPE15'] = ('MAG_AUTO_b', 'SE magnitude in b-band')
            hdr['TTYPE16'] = ('MAG_AUTO_g', 'SE magnitude in g-band')
            hdr['TTYPE17'] = ('MAG_AUTO_u', 'SE magnitude in u-band')
            hdr['TTYPE18'] = ('VIGNET_b', 'VIGNET in b-band')
            hdr['TTYPE19'] = ('VIGNET_g', 'VIGNET in g-band')
            hdr['TTYPE20'] = ('VIGNET_u', 'VIGNET in u-band')
            hdr['TTYPE21'] = ('color_bg', 'Color index (m_b - m_g)')
            hdr['TTYPE22'] = ('color_ub', 'Color index (m_u - m_b)')

            # Save changes
            hdul.flush()

    # Step 8: Plot the Color-Magnitude Diagram
    plt.figure(figsize=(8, 6))
    plt.scatter(color_index_bg, color_index_ub, s=5, alpha=0.10, color='blue', label='Galaxies')
    if args.plot_stars:
        plt.scatter(color_bg_stars, color_ub_stars, s=5, alpha=0.10, color='red', label='Stars')

    if args.plot_ned:
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

        if args.plot_ned:
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

        if args.plot_ned:
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
    parser.add_argument("--overwrite_coadds", action="store_true", help="Overwrite existing coadds")
    parser.add_argument("--overwrite_cats", action="store_true", help="Overwrite existing catalogs")
    parser.add_argument("--plot_color_mag", action="store_true", help="Plot the color-magnitude diagrams")
    parser.add_argument("--plot_stars", action="store_true", help="Plot stars in the color-magnitude diagram")
    parser.add_argument("--plot_ned", action="store_true", help="Plot NED galaxies in the color-magnitude diagram")
    parser.add_argument("--plot_lovoccs", action="store_true", help="Plot LOVOCCS galaxies in the color-magnitude diagram")
    parser.add_argument('--save_fits', action='store_true', help='Save the output FITS file')
    
    args = parser.parse_args()

    main(args)