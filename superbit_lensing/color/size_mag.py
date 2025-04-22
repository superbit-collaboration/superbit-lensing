import os
import argparse
from astropy.io import fits
from astropy.table import Table, vstack, unique
import numpy as np
import matplotlib.pyplot as plt
from superbit_lensing.match import SkyCoordMatcher
from superbit_lensing.utils import gaia_query, get_sky_footprint_center_radius

def main(cluster_name, band, datadir, save_union_catalog, mag_low, mag_high, radius_low, tolerance_deg):
    print(f"Processing {cluster_name} for band {band} in {datadir}")
    # Define file paths
    sex_cat_path = f"{datadir}/{cluster_name}/{band}/coadd/{cluster_name}_coadd_{band}_cat.fits"
    sex_cat_fallback_path = f"{datadir}/{cluster_name}/sextractor_dualmode/cat/{cluster_name}_coadd_{band}_cat.fits"
    gaia_cat_path = f"{datadir}/catalogs/stars/{cluster_name}_gaia_starcat.fits"
    output_filename = f"{datadir}/{cluster_name}/{band}/coadd/{cluster_name}_coadd_{band}_starcat_union.fits"
    plot_filename = f"{datadir}/{cluster_name}/sextractor_dualmode/plots/{cluster_name}_coadd_{band}_size_mag_plot.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)

    # Check if sex_cat exists
    if os.path.exists(sex_cat_path):
        print(f"Using star catalog for band b: {sex_cat_path}")
    elif os.path.exists(sex_cat_fallback_path):
        sex_cat_path = sex_cat_fallback_path
        print(f"Warning: Using fallback star catalog for band b: {sex_cat_fallback_path}")
    else:
        raise FileNotFoundError(f"Star catalog not found for band b: {sex_cat_path}")
    # Load catalogs
    sex_cat = Table.read(sex_cat_path, hdu=2)

    try:
        gaia_cat = Table.read(gaia_cat_path, format='fits', hdu=2)
    except Exception as e:
        print(f"Failed to read star catalog for band b ({e}), Trying to query Gaia")
        try:
            gaia_cat = gaia_query(cluster_name)
            print(f"Gaia query successful, found {len(gaia_cat)} objects")
        except Exception as e:
            print(f"Gaia query failed ({e}), using empty star catalog")
            gaia_cat = Table(names=['ALPHAWIN_J2000', 'DELTAWIN_J2000'], dtype=['f8', 'f8'])
    print(f"Number of stars in the star catalog: {len(gaia_cat)}")

    matcher_b_stars = SkyCoordMatcher(gaia_cat, sex_cat, match_radius=tolerance_deg)
    _ , gaia_cat = matcher_b_stars.get_matched_pairs()
    print(f"Number of stars in the star catalog after matching: {len(gaia_cat)}")
    # Extract relevant columns
    flux_auto = sex_cat["FLUX_AUTO"]
    flux_radius = sex_cat["FLUX_RADIUS"]

    # Ensure both flux and flux_radius are positive
    valid_indices = (flux_auto > 0) & (flux_radius > 0)

    # Compute magnitudes only for valid entries
    #mag_calc = -2.5 * np.log10(flux_auto[valid_indices])
    mag_calc = sex_cat[valid_indices]["MAG_AUTO"]
    flux_radius_filtered = flux_radius[valid_indices]

    # Apply selection criteria
    condition1 = (mag_calc < mag_high) & (flux_radius_filtered < radius_low)
    condition2 = (mag_calc < mag_low)
    selected_indices = valid_indices.nonzero()[0][condition1 | condition2]

    # Create a new catalog with selected objects
    filtered_catalog = sex_cat[selected_indices]
    print(f"Filtered catalog saved with {len(filtered_catalog)} objects.")

    union_catalog = vstack([filtered_catalog, gaia_cat])  # Merge tables
    union_catalog = unique(union_catalog, keys=["ALPHAWIN_J2000", "DELTAWIN_J2000"])

    # Union catalog objects
    union_flux_radius = union_catalog["FLUX_RADIUS"]
    #union_mag = -2.5 * np.log10(union_catalog["FLUX_AUTO"])
    union_mag = union_catalog["MAG_AUTO"]

    # Create a primary HDU (empty, required for FITS format)
    primary_hdu = fits.PrimaryHDU()
    empty_hdu1 = fits.BinTableHDU(name="EMPTY_HDU_1")
    union_hdu = fits.BinTableHDU(union_catalog, name="UNION_CATALOG")

    hdul = fits.HDUList([primary_hdu, empty_hdu1, union_hdu])
    if save_union_catalog:
        hdul.writeto(output_filename, overwrite=True)
        print(f"Union catalog saved to '{output_filename}' in HDU 2 with {len(union_catalog)} objects.")

    # Plot the filtered catalog
    plt.figure(figsize=(8, 6))
    plt.scatter(flux_radius_filtered, mag_calc, s=5, alpha=0.3, color='blue', label="Original Data")
    plt.scatter(union_flux_radius, union_mag, s=5, alpha=0.3, color='red', label="Union Catalog")
    plt.scatter(gaia_cat["FLUX_RADIUS"], gaia_cat["MAG_AUTO"], s=5, alpha=0.3, color='green', label="Gaia Catalog")
    plt.axvline(x=radius_low, color='red', linestyle='--', label=f'Half-light radius={radius_low}')
    plt.axhline(y=mag_high, color='green', linestyle='--', label=f'Magnitude={mag_high}')
    plt.axhline(y=mag_low, color='orange', linestyle='--', label=f'Magnitude={mag_low}')
    plt.xlim(0, 50)
    #plt.ylim(-20, 2.5)
    plt.xlabel("Half-light radius")
    plt.ylabel(f"$m_{band}$")
    plt.title(f"{cluster_name}")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(plot_filename)
    print(f"Plot saved to '{plot_filename}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and merge astronomical catalogs.")
    
    parser.add_argument("cluster_name", type=str, help="Name of the cluster (e.g., AbellS0592)")
    parser.add_argument("bands", nargs="+", type=str, help="Bands (e.g., b g r i)")  # Allows any number of bands
    parser.add_argument("--datadir", type=str, default=os.getenv("DATADIR", ""), help="Directory containing the data files")
    parser.add_argument("--save_union_catalog", action='store_true', help="Flag to save the union catalog")
    parser.add_argument("--mag_low", type=float, default=15, help="Lower magnitude threshold")
    parser.add_argument("--mag_high", type=float, default=19.5, help="Upper magnitude threshold")
    parser.add_argument("--radius_low", type=float, default=3.0, help="Lower flux radius threshold")
    parser.add_argument("--tolerance_deg", type=float, default=1e-4, help="Matching tolerance in degrees")
    
    args = parser.parse_args()
    print("=== Arguments Passed to the Script ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================\n")
    # Run the script for each band provided
    for band in args.bands:
        main(args.cluster_name, band, args.datadir, args.save_union_catalog,
             args.mag_low, args.mag_high, args.radius_low, args.tolerance_deg)
