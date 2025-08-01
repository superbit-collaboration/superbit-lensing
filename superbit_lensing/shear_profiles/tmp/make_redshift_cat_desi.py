import pandas as pd
import os
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import superbit_lensing.utils as utils
from superbit_lensing.match import SkyCoordMatcher
import time

DESI_MASTER_FILE = "/projects/mccleary_group/superbit/desi_data/zall-pix-iron.fits"
desi_table = [
    'TARGETID', 'SURVEY', 'PROGRAM', 'OBJTYPE', 'SPECTYPE', 
    'TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR', 'ZWARN', 'ZCAT_NSPEC', 'ZCAT_PRIMARY'
]

def make_redshift_catalog(datadir, target, band, detect_cat_path, tolerance_deg=1/3600, vstack_desi=True):
    """
    Utility script to create a "redshift catalog" with spec-z's where they
    exist, a dummy value of 1 otherwise.

    Inputs
        datadir: basedir for unions
        target: cluster target name
        band: which bandpass are we measuring shear in?
        detect_cat_path: path to detection catalog
    """
    # Path for detect_cat FITS file remains the same
    # detect_cat_path = f"{datadir}/{target}_{band}_coadd_cat.fits"
    detect_cat = Table.read(detect_cat_path, format='fits', hdu=2)
    # Get footprint of the cluster
    center_ra, center_dec, radius = utils.get_sky_footprint_center_radius(detect_cat, buffer_fraction=0.2)    

    # Adjusted path for NED_redshifts
    ned_redshifts_path = \
        f"{datadir}/catalogs/redshifts/{target}_NED_redshifts.csv"
    lovoccs_path = f"{datadir}/catalogs/lovoccs/{target}_lovoccs_redshifts.fits"
    desi_path = f"{datadir}/catalogs/desi/{target}_desi_spectra.fits"
    # First try to load from file
    try:
        print(f"Attempting to load redshifts from {ned_redshifts_path}")
        ned_redshifts = Table.read(ned_redshifts_path, format='csv')
        print(f"Successfully loaded {len(ned_redshifts)} redshifts from file")
    except Exception as e:
        print(f"Could not load {ned_redshifts_path} because: {e}")
        
        # Fall back to NED query with three attempts
        print("Falling back to NED query with decreasing radius...")
        max_attempts = 3
        ned_query_success = False
        current_radius = radius  # Start with the original radius
        
        for attempt in range(max_attempts):
            try:
                print(f"Attempting NED query (attempt {attempt+1}/{max_attempts}) with radius={current_radius:.4f} deg...")
                ned_redshifts = utils.ned_query(rad_deg=current_radius, ra_center=center_ra, dec_center=center_dec)
                print(f"Successfully queried NED with {len(ned_redshifts)} results at radius {current_radius:.4f} deg")
                ned_query_success = True
                
                # Optionally save the results for future use
                try:
                    print(f"Saving results to {ned_redshifts_path}")
                    ned_redshifts.write(ned_redshifts_path, format='csv', overwrite=True)
                    print("Results saved successfully")
                except Exception as save_error:
                    print(f"Warning: Could not save NED results to file: {save_error}")
                
                break  # Exit retry loop if successful
                
            except Exception as e:
                print(f"NED query attempt {attempt+1} failed with radius {current_radius:.4f} deg: {e}")
                if attempt < max_attempts - 1:
                    wait_time = 5  # Longer wait for external API
                    # Reduce radius for next attempt
                    current_radius = current_radius / 1.06
                    print(f"Waiting {wait_time} seconds before retrying with smaller radius ({current_radius:.4f} deg)...")
                    time.sleep(wait_time)
        
        # If all NED query attempts failed, raise an error
        if not ned_query_success:
            error_msg = "All NED query attempts failed. Cannot proceed without redshift data."
            print(error_msg)
            raise RuntimeError(error_msg)

    try:
        print(f"Attempting to load redshifts from {desi_path}")
        desi = Table.read(desi_path, format='fits')
        print(f"Successfully loaded {len(desi)} redshifts from desi file")    
    except Exception as e:
        print(f"Could not load {desi_path} because: {e}")
        try:
            desi = utils.desi_query(rad_deg=radius, ra_center=center_ra, dec_center=center_dec)
        except Exception as e:
            print(f"desi query failed due to: {e}, creating an empty desi catalog")
            desi = Table(names=desi_table, dtype=['f8'] * len(desi_table))
        os.makedirs(os.path.dirname(desi_path), exist_ok=True)
        desi.write(desi_path, format='fits', overwrite=True)
    desi_filtered = desi['TARGET_RA', 'TARGET_DEC', 'Z', 'ZERR']
    desi_filtered.rename_column('TARGET_RA', 'RA')
    desi_filtered.rename_column('TARGET_DEC', 'DEC')    
    desi_filtered.rename_column('Z', 'Redshift')
    desi_filtered.rename_column('ZERR', 'Redshift_error')
    ned_redshifts = ned_redshifts['RA', 'DEC', 'Redshift']
    ned_redshifts['Redshift_error'] = -1.0
    # Save NED redshifts separately
    ned_redshift_path = f"{datadir}/catalogs/redshifts/{target}_ned_redshifts.fits"
    ned_redshifts.write(ned_redshift_path, format='fits', overwrite=True)
    print(f"Saved NED redshift catalog to {ned_redshift_path}")

    # Save DESI redshifts separately
    desi_redshift_path = f"{datadir}/catalogs/redshifts/{target}_desi_redshifts.fits"
    desi_filtered.write(desi_redshift_path, format='fits', overwrite=True)
    print(f"Saved DESI redshift catalog to {desi_redshift_path}")
    if vstack_desi:
        ned_redshifts = vstack([ned_redshifts, desi_filtered])
    combined_redshift_path = f"{datadir}/catalogs/redshifts/{target}_combined_redshifts.fits"
    ned_redshifts.write(combined_redshift_path, format='fits', overwrite=True)
    print(f"Saved combined redshift catalog to {combined_redshift_path}")
    print(f"Loaded {len(ned_redshifts)} redshifts from NED & DESI")
    matcher_ned = SkyCoordMatcher(ned_redshifts, detect_cat, cat1_ratag='RA', cat1_dectag='DEC',
                 cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', return_idx=True, match_radius=1 * tolerance_deg)
    matched_ned, matched_data_b_ned, idx1_ned, idx2_ned = matcher_ned.get_matched_pairs()    

    # Create a dummy redshift column filled with ones
    redshift_col = np.ones(len(detect_cat))
    redshift_col[idx2_ned] = matched_ned["Redshift"]

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
