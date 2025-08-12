## This script assumes you've already run annular!
## Filter background objects from an annular catalog by removing foreground objects
## Calculate new response matrix and g_rinv
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

def remove_foreground_objects(annular_fits, foreground_fits, 
                             annular_ra_col='RA', annular_dec_col='DEC',
                             foreground_ra_col='RA', foreground_dec_col='DEC',
                             tolerance_degree=0.0001, verbose=True):
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
    tolerance_degree : float
        Match radius in degrees (default: 0.001)
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
    
    # Match coordinates using SkyCoordMatcher
    if verbose:
        print(f"\nMatching coordinates ({tolerance_degree} degree)...")
    
    # Create matcher instance
    matcher = SkyCoordMatcher(
        annular_cat, 
        foreground_cat,
        cat1_ratag=annular_ra_col,
        cat1_dectag=annular_dec_col,
        cat2_ratag=foreground_ra_col,
        cat2_dectag=foreground_dec_col,
        return_idx=True,
        match_radius=1 * tolerance_degree
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

def main():
    """Main function to run the script from command line."""
    
    parser = argparse.ArgumentParser(
        description="Filter foreground objects from annular catalog keeping background only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

        """
    )
    
    # Required arguments
    parser.add_argument('annular_fits', 
                       help='Path to annular FITS file with all objects')
    parser.add_argument('foreground_fits', 
                       help='Path to FITS file with foreground objects')
    
    # Optional but recommended argument
    parser.add_argument('-o', '--output', default=None,
                       help='Output FITS file for background catalog '
                            '(default: adds _background suffix)')
    
    # Fully optional
    parser.add_argument('--annular-ra', default='ra',
                       help='Name of RA column in annular catalog (default: RA)')
    parser.add_argument('--annular-dec', default='dec',
                       help='Name of DEC column in annular catalog (default: DEC)')
    parser.add_argument('--foreground-ra', default='ra',
                       help='Name of RA column in foreground catalog (default: RA)')
    parser.add_argument('--foreground-dec', default='dec',
                       help='Name of DEC column in foreground catalog (default: DEC)')
    parser.add_argument('--tolerance-degree', type=float, default=0.001,
                       help='Match radius in degrees (default: 0.001)')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.annular_fits):
        print(f"Error: Annular catalog not found: {args.annular_fits}")
        sys.exit(1)
    if not os.path.exists(args.foreground_fits):
        print(f"Error: Foreground catalog not found: {args.foreground_fits}")
        sys.exit(1)
    
    # Generate output filename if not specified
    if args.output is None:
        base = os.path.splitext(args.annular_fits)[0]
        args.output = f"{base}_background.fits"
    
    # Run the filtering
    try:
        background_catalog = remove_foreground_objects(
            args.annular_fits,
            args.foreground_fits,
            annular_ra_col=args.annular_ra,
            annular_dec_col=args.annular_dec,
            foreground_ra_col=args.foreground_ra,
            foreground_dec_col=args.foreground_dec,
            tolerance_degree=args.tolerance_degree,
            verbose=not args.quiet
        )
        
        # Save the result
        if not args.quiet:
            print(f"\nSaving background catalog: {args.output}")
        background_catalog.write(args.output, overwrite=True)
        
        if not args.quiet:
            print("Done!")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()