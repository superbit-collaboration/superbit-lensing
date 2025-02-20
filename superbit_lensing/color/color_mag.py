import os
import argparse
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from superbit_lensing.match import SkyCoordMatcher

def main(args):
    # Construct file paths based on the inputs
    cluster_name = args.cluster_name
    bands = args.bands
    datadir = args.datadir
    tolerance_deg = args.tolerance
    redshift = args.redshift
    delz = 0.02

    base_path = os.path.join(datadir, cluster_name)
    redshift_file = os.path.join(datadir, f'catalogs/redshifts/{cluster_name}_NED_redshifts.csv')
    file_b = f"{base_path}/{bands[0]}/coadd/{cluster_name}_coadd_{bands[0]}_cat.fits"
    file_g = f"{base_path}/{bands[1]}/coadd/{cluster_name}_coadd_{bands[1]}_cat.fits"
    file_b_stars = f"{base_path}/{bands[0]}/coadd/{cluster_name}_coadd_{bands[0]}_starcat_union.fits"
    file_g_stars = f"{base_path}/{bands[1]}/coadd/{cluster_name}_coadd_{bands[1]}_starcat_union.fits"
    output_file = f"{base_path}/color_magnitude_diagram.png"

    # Ensure input files exist
    if not os.path.exists(file_b) or not os.path.exists(file_g):
        raise FileNotFoundError(f"One or more input files not found:\n{file_b}\n{file_g}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Read the FITS files
    data_b = Table.read(file_b, format='fits', hdu=2)
    data_g = Table.read(file_g, format='fits', hdu=2)
    star_data_b = Table.read(file_b_stars, format='fits', hdu=2)
    star_data_g = Table.read(file_g_stars, format='fits', hdu=2)
    ned_cat = Table.read(redshift_file, format='csv')

    # Step 1: Star Matching and Object Discarding
    print("Star matching and object discarding is starting...")

    # Use SkyCoordMatcher for star matching in band b
    matcher_b_stars = SkyCoordMatcher(data_b, star_data_b, match_radius=tolerance_deg)
    matched_data_b_stars, matched_star_data_b = matcher_b_stars.get_matched_pairs()

    # Use SkyCoordMatcher for star matching in band g
    matcher_g_stars = SkyCoordMatcher(data_g, star_data_g, match_radius=tolerance_deg)
    matched_data_g_stars, matched_star_data_g = matcher_g_stars.get_matched_pairs()

    # Filter out stars from the main catalogs using both RA and DEC
    data_b_coords = list(zip(data_b['ALPHAWIN_J2000'], data_b['DELTAWIN_J2000']))
    matched_b_stars_coords = list(zip(matched_data_b_stars['ALPHAWIN_J2000'], matched_data_b_stars['DELTAWIN_J2000']))
    mask = np.array([(ra, dec) not in matched_b_stars_coords for ra, dec in data_b_coords])
    data_b_filtered = data_b[mask]

    data_g_coords = list(zip(data_g['ALPHAWIN_J2000'], data_g['DELTAWIN_J2000']))
    matched_g_stars_coords = list(zip(matched_data_g_stars['ALPHAWIN_J2000'], matched_data_g_stars['DELTAWIN_J2000']))
    mask = np.array([(ra, dec) not in matched_g_stars_coords for ra, dec in data_g_coords])
    data_g_filtered = data_g[mask]

    print(f"Band {bands[0]} - Total objects: {len(data_b)}, Stars: {len(star_data_b)}, Discarded: {len(matched_data_b_stars)}, Proceeding to next step: {len(data_b_filtered)}")
    print(f"Band {bands[1]} - Total objects: {len(data_g)}, Stars: {len(star_data_g)}, Discarded: {len(matched_data_g_stars)}, Proceeding to next step: {len(data_g_filtered)}")

    # Step 2: Interband Matching
    print("Interband matching is starting...")

    # Use SkyCoordMatcher for interband matching
    matcher_interband = SkyCoordMatcher(data_b_filtered, data_g_filtered, match_radius=tolerance_deg)
    matched_data_b, matched_data_g = matcher_interband.get_matched_pairs()

    print(f"Number of matches after interband matching: {len(matched_data_b)}")

    # Step 3: Combine Matched Data
    flux_b = matched_data_b['FLUX_AUTO']
    flux_g = matched_data_g['FLUX_AUTO']

    valid_flux = (flux_b > 0) & (flux_g > 0)
    print(f"Number of objects with positive flux in both bands: {np.sum(valid_flux)}")
    m_b = -2.5 * np.log10(flux_b[valid_flux])
    m_g = -2.5 * np.log10(flux_g[valid_flux])
    color_index = m_b - m_g

    # Step 4: Match NED Data
    ra_ned = np.radians(ned_cat['RA'])
    dec_ned = np.radians(ned_cat['DEC'])

    # Use SkyCoordMatcher for NED matching
    matcher_ned = SkyCoordMatcher(ned_cat, matched_data_b, cat1_ratag='RA', cat1_dectag='DEC',
                 cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', return_idx=True, match_radius=1 * tolerance_deg)
    matched_ned, matched_data_b_ned, idx1, idx2 = matcher_ned.get_matched_pairs()

    print(f"Number of matched galaxies with known redshifts: {len(matched_ned)}")

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

    print(f"Galaxies with z > {cluster_redshift_up}: {len(high_z_indices)}")
    print(f'Galaxies with {cluster_redshift_down} < z ≤ {cluster_redshift_up}: {len(mid_z_indices)}')
    print(f"Galaxies with z ≤ {cluster_redshift_down}: {len(low_z_indices)}")

    # Step 6: Compute Magnitudes for NED Matches
    flux_b_high = high_z_b['FLUX_AUTO']
    flux_g_high = matched_data_g[idx2][high_z_indices]['FLUX_AUTO']

    flux_b_low = low_z_b['FLUX_AUTO']
    flux_g_low = matched_data_g[idx2][low_z_indices]['FLUX_AUTO']

    flux_b_mid = matched_data_b_ned[mid_z_indices]['FLUX_AUTO']
    flux_g_mid = matched_data_g[idx2][mid_z_indices]['FLUX_AUTO']

    valid_flux_high = (flux_b_high > 0) & (flux_g_high > 0)
    valid_flux_low = (flux_b_low > 0) & (flux_g_low > 0)
    valid_flux_mid = (flux_b_mid > 0) & (flux_g_mid > 0)

    m_b_high = -2.5 * np.log10(flux_b_high[valid_flux_high])
    m_g_high = -2.5 * np.log10(flux_g_high[valid_flux_high])
    color_index_high = m_b_high - m_g_high

    m_b_low = -2.5 * np.log10(flux_b_low[valid_flux_low])
    m_g_low = -2.5 * np.log10(flux_g_low[valid_flux_low])
    color_index_low = m_b_low - m_g_low

    m_b_mid = -2.5 * np.log10(flux_b_mid[valid_flux_mid])
    m_g_mid = -2.5 * np.log10(flux_g_mid[valid_flux_mid])
    color_index_mid = m_b_mid - m_g_mid

    # Interband Star Matching
    matcher_b_stars = SkyCoordMatcher(matched_star_data_b, matched_star_data_g, match_radius=tolerance_deg)
    matched_star_data_b, matched_star_data_g = matcher_b_stars.get_matched_pairs()

    # Step 7: Compute Magnitudes for Matched Stars
    flux_stars_b = matched_star_data_b['FLUX_AUTO']
    flux_stars_g = matched_star_data_g['FLUX_AUTO']

    valid_flux_stars = (flux_stars_b > 0) & (flux_stars_g > 0)
    m_stars_b = -2.5 * np.log10(flux_stars_b[valid_flux_stars])
    m_stars_g = -2.5 * np.log10(flux_stars_g[valid_flux_stars])
    color_stars = m_stars_b - m_stars_g

    # Step 8: Plot the Color-Magnitude Diagram
    plt.figure(figsize=(8, 6))
    plt.scatter(m_b, color_index, s=5, alpha=0.10, color='blue', label='Galaxies')
    if args.plot_stars:
        plt.scatter(m_stars_b, color_stars, s=5, alpha=0.10, color='red', label='Stars')

    if args.plot_ned:
        plt.scatter(m_b_high, color_index_high, s=10, edgecolors='black', facecolors='orange', label=f'High-z (z > {cluster_redshift_up:.2f})')
        plt.scatter(m_b_mid, color_index_mid, s=10, edgecolors='black', facecolors='lime', label=f'Members ({cluster_redshift_down:.2f} < z ≤ {cluster_redshift_up:.2f})')
        plt.scatter(m_b_low, color_index_low, s=10, edgecolors='black', facecolors='red', label=f'Low-z (z ≤ {cluster_redshift_down:.2f})')

    plt.ylim(-4.2, 3.8)
    #plt.xlim(-20, -2)
    plt.xlabel(f'$m_{bands[0]}$')
    plt.ylabel(f'$m_{bands[0]} - m_{bands[1]}$')
    plt.title(f'{cluster_name}, Redshift={redshift}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine bands for a given cluster.")
    parser.add_argument("cluster_name", type=str, help="Name of the cluster (e.g., AbellS0592)")
    parser.add_argument("bands", type=str, nargs=2, help="Two bands to combine (e.g., b g)")
    parser.add_argument("--datadir", type=str, default="/work/mccleary_group/saha/data/", help="Directory containing the data files")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Angular tolerance in degrees (default: 1e-4)")
    parser.add_argument("--redshift", type=float, default=0.5, help="Redshift threshold for classification (default: 0.5)")
    parser.add_argument("--plot_stars", action="store_true", help="Plot stars in the color-magnitude diagram")
    parser.add_argument("--plot_ned", action="store_true", help="Plot NED galaxies in the color-magnitude diagram")
    args = parser.parse_args()

    main(args)