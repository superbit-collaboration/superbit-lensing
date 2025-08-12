## This script assumes you've already run annular!
## Filter background objects from an annular catalog by removing foreground objects, calculate new response matrix and g_rinv
## Save as a fits file which you can then make a convergence map out of

import argparse
import os
import sys
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
from superbit_lensing.match import SkyCoordMatcher

## uses SkyCoordMatcher to match then filter out foreground

def filter_background_objects(annular_fits, foreground_fits, 
                             annular_ra_col='RA', annular_dec_col='DEC',
                             foreground_ra_col='RA', foreground_dec_col='DEC',
                             match_radius=1.0, verbose=True):
    """
    Filter background objects by removing foreground matches.
    
    Parameters
    ----------
    annular_fits : str
        Path to annular FITS file containing all objects
    foreground_fits : str
        Path to FITS file containing foreground objects
    annular_ra_col : str
        Name of RA column in annular catalog (default: 'RA')
    annular_dec_col : str
        Name of DEC column in annular catalog (default: 'DEC')
    foreground_ra_col : str
        Name of RA column in foreground catalog (default: 'RA')
    foreground_dec_col : str
        Name of DEC column in foreground catalog (default: 'DEC')
    match_radius : float
        Match radius in arcseconds (default: 1.0)
    verbose : bool
        Print progress information
        
    Returns
    -------
    background_catalog : astropy.table.Table
        Filtered catalog containing only background objects
    """
    
    # Load the catalogs
    if verbose:
        print(f"Loading annular catalog: {annular_fits}")
    annular_cat = Table.read(annular_fits)
    n_total = len(annular_cat)
    
    if verbose:
        print(f"  Total objects: {n_total:,}")
        print(f"\nLoading foreground catalog: {foreground_fits}")
    foreground_cat = Table.read(foreground_fits)
    n_foreground = len(foreground_cat)
    
    if verbose:
        print(f"  Foreground objects: {n_foreground:,}")
    
    # Check for required columns
    if annular_ra_col not in annular_cat.colnames:
        raise ValueError(f"Column '{annular_ra_col}' not found in annular catalog. "
                       f"Available columns: {annular_cat.colnames}")
    if annular_dec_col not in annular_cat.colnames:
        raise ValueError(f"Column '{annular_dec_col}' not found in annular catalog. "
                       f"Available columns: {annular_cat.colnames}")
    if foreground_ra_col not in foreground_cat.colnames:
        raise ValueError(f"Column '{foreground_ra_col}' not found in foreground catalog. "
                       f"Available columns: {foreground_cat.colnames}")
    if foreground_dec_col not in foreground_cat.colnames:
        raise ValueError(f"Column '{foreground_dec_col}' not found in foreground catalog. "
                       f"Available columns: {foreground_cat.colnames}")
    
    # Convert match radius from arcsec to degrees
    tolerance_deg = match_radius / 3600.0
    
    # Match coordinates using SkyCoordMatcher
    if verbose:
        print(f"\nMatching coordinates (radius: {match_radius} arcsec)...")
    
    # Create matcher instance
    matcher = SkyCoordMatcher(
        annular_cat, 
        foreground_cat,
        cat1_ratag=annular_ra_col,
        cat1_dectag=annular_dec_col,
        cat2_ratag=foreground_ra_col,
        cat2_dectag=foreground_dec_col,
        return_idx=True,
        match_radius=tolerance_deg
    )
    
    # Get matched pairs and indices
    matched_annular, matched_foreground, idx1, idx2 = matcher.get_matched_pairs()
    
    n_matched = len(idx1)
    
    if verbose:
        print(f"  Found {n_matched:,} matches")
    
    # Create mask for background objects (not matched to foreground)
    # idx1 contains the indices of matched objects in the annular catalog
    background_mask = np.ones(n_total, dtype=bool)
    if n_matched > 0:
        background_mask[idx1] = False
    
    # Filter to get background catalog
    background_catalog = annular_cat[background_mask]
    n_background = len(background_catalog)
    
    if verbose:
        print(f"\nFiltering results:")
        print(f"  Original objects: {n_total:,}")
        print(f"  Foreground matches removed: {n_matched:,}")
        print(f"  Background objects remaining: {n_background:,}")
        print(f"  Background fraction: {n_background/n_total:.3f}")
        
        if n_matched > 0:
            print(f"\nMatched annular catalog has {len(matched_annular)} objects")
            print(f"Available columns in matched catalog: {matched_annular.colnames[:5]}...")
    
    return background_catalog
