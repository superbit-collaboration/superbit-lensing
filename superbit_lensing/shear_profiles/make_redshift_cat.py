import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

def make_redshift_catalog(datadir, target, band, detect_cat_path):
    """
    Utility script to create a "redshift catalog" with spec-z's where they
    exist, a dummy value of 1 otherwise.

    Inputs
        datadir: basedir for unions
        target: cluster target name
        band: which bandpass are we measuring shear in?
        detect_cat_path: path to detection catalog
    """

    # Adjusted path for NED_redshifts
    ned_redshifts_path = \
        f"{datadir}/catalogs/redshifts/{target}_NED_redshifts.csv"
    ned_redshifts = pd.read_csv(ned_redshifts_path)

    # Path for detect_cat FITS file remains the same
    # detect_cat_path = f"{datadir}/{target}_{band}_coadd_cat.fits"
    with fits.open(detect_cat_path) as hdul:
        detect_cat = Table(hdul[2].data)

    # Create a dummy redshift column filled with ones
    redshift_col = np.ones(len(detect_cat))

    # Match detect_cat to NED_redshifts in RA and Dec
    ned_coords = SkyCoord(
        ra=ned_redshifts['RA']*u.degree, dec=ned_redshifts['DEC']*u.degree,
        unit='deg'
    )

    detect_cat_coords = SkyCoord(
        ra=detect_cat['ALPHAWIN_J2000']*u.degree,
        dec=detect_cat['DELTAWIN_J2000']*u.degree
    )

    idx, d2d, _ = detect_cat_coords.match_to_catalog_sky(ned_coords)
    max_sep = 1.0 * u.arcsec
    sep_constraint = d2d < max_sep
    matched_idx = idx[sep_constraint]

    # Update the redshift column in detect_cat for matched galaxies
    redshift_col[sep_constraint] = \
        ned_redshifts.iloc[matched_idx]['Redshift'].values

    # Create a new table with RA/Dec and new redshifts
    new_table = Table([
        detect_cat['ALPHAWIN_J2000'], detect_cat['DELTAWIN_J2000'],
        redshift_col], names=('RA', 'DEC', 'Redshift')
    )

    # Save the new table to the specified directory
    new_table_path = \
        f"{datadir}/catalogs/redshifts/{target}_{band}_with_redshifts.fits"
    new_table.write(new_table_path, format='fits', overwrite=True)

    print(f"Saved a redshift catalog to {new_table_path}")
    return new_table_path
