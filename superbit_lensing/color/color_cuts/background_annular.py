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

def calculate_shear_response(catalog, verbose=True):
    """
    Calculate shear response correction for the catalog.
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Catalog with r11, r22, and g_noshear columns
    verbose : bool
        Print progress information
        
    Returns
    -------
    catalog : astropy.table.Table
        Catalog with updated g1_rinv and g2_rinv columns
    """
    # Check for required columns
    required_cols = ['r11', 'r22', 'g_noshear']
    missing_cols = [col for col in required_cols if col not in catalog.colnames]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                       f"Available columns: {catalog.colnames}")
    
    if verbose:
        print("\nCalculating shear response correction...")
        print(f"  Found r11 column: Yes")
        print(f"  Found r22 column: Yes")
        print(f"  Found g_noshear column: Yes")
        
        # Check if g1_rinv and g2_rinv already exist
        has_g1_rinv = 'g1_rinv' in catalog.colnames
        has_g2_rinv = 'g2_rinv' in catalog.colnames
        
        if has_g1_rinv:
            print(f"  Found existing g1_rinv column - will update")
        else:
            print(f"  No g1_rinv column found - will create")
            
        if has_g2_rinv:
            print(f"  Found existing g2_rinv column - will update")
        else:
            print(f"  No g2_rinv column found - will create")
    
    # Calculate average response values
    r11_avg = np.mean(catalog['r11'])
    r22_avg = np.mean(catalog['r22'])
    
    if verbose:
        print(f"\nResponse matrix averages (from {len(catalog)} background objects):")
        print(f"  r11_avg = {r11_avg:.6f}")
        print(f"  r22_avg = {r22_avg:.6f}")
        print(f"  Response matrix R = [[{r11_avg:.6f}, 0], [0, {r22_avg:.6f}]]")
    
    # Extract g_noshear components
    # Handle different possible formats for g_noshear
    g_noshear = catalog['g_noshear']
    
    # Check if g_noshear is already a 2D array or needs to be parsed
    if len(g_noshear.shape) == 1:
        # It might be stored as tuples or strings, need to parse
        if verbose:
            print(f"\n  g_noshear format: {type(g_noshear[0])}")
        
        # Try to extract g1 and g2
        if isinstance(g_noshear[0], (list, tuple, np.ndarray)):
            g1_noshear = np.array([g[0] for g in g_noshear])
            g2_noshear = np.array([g[1] for g in g_noshear])
        else:
            # Might need different parsing
            raise ValueError(f"Unexpected g_noshear format: {type(g_noshear[0])}")
    else:
        # Already a 2D array
        g1_noshear = g_noshear[:, 0]
        g2_noshear = g_noshear[:, 1]
    
    # Calculate g_rinv = R * g_noshear
    # For diagonal matrix: g1_rinv = r11_avg * g1, g2_rinv = r22_avg * g2
    g1_rinv_new = r11_avg * g1_noshear
    g2_rinv_new = r22_avg * g2_noshear
    
    # Update or add g1_rinv and g2_rinv columns
    catalog['g1_rinv'] = g1_rinv_new
    catalog['g2_rinv'] = g2_rinv_new
    
    if verbose:
        print(f"\nShear correction applied:")
        print(f"  g1_rinv = r11_avg * g1_noshear")
        print(f"  g2_rinv = r22_avg * g2_noshear")
        print(f"  Sample values:")
        print(f"    g1_rinv[0] = {g1_rinv_new[0]:.6f}")
        print(f"    g2_rinv[0] = {g2_rinv_new[0]:.6f}")
        print(f"  Updated g1_rinv and g2_rinv columns")
    
    return catalog



def remove_foreground_objects(annular_fits, foreground_fits, 
                             annular_ra_col='RA', annular_dec_col='DEC',
                             foreground_ra_col='RA', foreground_dec_col='DEC',
                             tolerance_degree=0.0001, calculate_response=True, verbose=True):
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
        Match radius in degrees (default: 0.0001)
    calculate_response : bool
        Calculate shear response correction (default: True)
    verbose : bool
        Print progress information
        
    Returns
    -------
    background_catalog : astropy.table.Table
        Filtered catalog containing only background objects with updated g1_rinv and g2_rinv columns
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
    
    if calculate_response:
        try:
            background_catalog = calculate_shear_response(background_catalog, verbose=verbose)
        except ValueError as e:
            if verbose:
                print(f"\nWarning: Could not calculate shear response: {e}")
                print("Continuing without g_rinv calculation...")    
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
    parser.add_argument('--no-response', action='store_true',
                       help='Skip shear response calculation')
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
            calculate_response=not args.no_response,
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