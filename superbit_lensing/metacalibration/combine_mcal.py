import numpy as np
from astropy.table import Table
from argparse import ArgumentParser
from superbit_lensing.match import SkyCoordMatcher
from superbit_lensing.utils import separate_catalog_by_regions
import os
import ipdb

# hlr -> sigma conversion factor for a 2D Gaussian: sigma = hlr / sqrt(2 ln 2)
_HLR_TO_SIGMA = 1.0 / np.sqrt(2.0 * np.log(2.0))


def _has_col(table, name):
    """Works for both astropy Tables and FITS_rec / structured arrays."""
    names = getattr(table, "colnames", None)
    if names is None:
        names = table.dtype.names
    return name in names


def _T_from_hlr(table, t):
    """Compute (T, T_err) from the half-light radius in pars[4], assuming a
    Gaussian profile: sigma = hlr / sqrt(2 ln2), T = 2 sigma^2."""
    hlr = np.asarray(table[f"pars_{t}"])[:, 4]
    hlr_err = np.asarray(table[f"pars_err_{t}"])[:, 4]

    sigma = hlr * _HLR_TO_SIGMA
    sigma_err = hlr_err * _HLR_TO_SIGMA

    T = 2.0 * sigma**2
    T_err = 4.0 * sigma * sigma_err  # dT/dsigma = 4 sigma
    return T, T_err

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-nrun', type=int, default=50, help='Number of realizations to combine')
    parser.add_argument('-data_dir', type=str, required=True, help='Path to cluster data')
    parser.add_argument('-run_name', type=str, required=True, help='Cluster Name')
    parser.add_argument('-band', type=str, required=True, help='Band name')
    parser.add_argument('-outdir', type=str, help='Output directory')
    parser.add_argument('--file_ending', type=str, default='fits', help='File ending for mcal files (default: fits)')
    parser.add_argument('--isolate_stars', type=lambda x: x.lower() == 'true', default=True, help='Flag to isolate stars (default: True)')
    parser.add_argument('--isolate_redseq', type=lambda x: x.lower() == 'true', default=True, help='Flag to isolate red-seq galaxies (default: False)')
    return parser.parse_args()


def main(args):

    if not all([args.data_dir, args.run_name, args.band, args.outdir]):
        raise ValueError("All required arguments must be provided.")

    nrun = args.nrun
    data_dir = args.data_dir
    run_name = args.run_name
    band = args.band
    outdir = args.outdir if args.outdir else f"{data_dir}{run_name}/{band}/out"

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Generate file paths
    mcal_files = [f"{data_dir}/{run_name}/{band}/arr/run{i}/{run_name}_{band}_mcal.{args.file_ending}" for i in range(1, nrun+1)]

    # Check which files exist
    existing_files = [f for f in mcal_files if os.path.exists(f)]
    missing_files = [f for f in mcal_files if not os.path.exists(f)]

    # Print warnings for missing files
    print(f"Using {len(existing_files)} out of {nrun} expected files.")
    if missing_files:
        print("Warning: The following files are missing and will be skipped:")
        for f in missing_files:
            print(f"  - {f}")

    # Proceed only if at least one file exists
    if not existing_files:
        raise FileNotFoundError("No mcal files found. Cannot proceed.")

    # Load the mcal tables
    mcal_tables = [Table.read(f, format="fits") for f in existing_files]

    # Extract and find common "id" values across all files
    common_ids = set(mcal_tables[0]["id"])
    for table in mcal_tables[1:]:
        common_ids.intersection_update(set(table["id"]))

    print(f"Common objects found: {len(common_ids)}")

    # Filter each table to keep only common "id" values
    filtered_tables = []
    for table in mcal_tables:
        mask = np.isin(table["id"], list(common_ids))
        filtered_tables.append(table[mask])

    # Combine filtered tables
    mcal_combined = {}
    mcal_combined["id"] = filtered_tables[0]["id"]
    mcal_combined["meds_indx"] = filtered_tables[0]["meds_indx"]
    mcal_combined["ra"] = filtered_tables[0]["ra"]
    mcal_combined["dec"] = filtered_tables[0]["dec"]
    mcal_combined["XWIN_IMAGE"] = filtered_tables[0]["XWIN_IMAGE"]
    mcal_combined["YWIN_IMAGE"] = filtered_tables[0]["YWIN_IMAGE"]
    mcal_combined["ncutout"] = filtered_tables[0]["ncutout"]

    # Metacalibration types
    types = ["noshear", "1p", "1m", "2p", "2m", "1p_psf", "1m_psf", "2p_psf", "2m_psf"]

    # Number of realizations
    N = len(filtered_tables)

    # check if the T, T_err columns exist or not (absent for galsimfitter)
    test_table = filtered_tables[0]
    has_T = _has_col(test_table, f"T_{types[0]}")


    for t in types:
        g1_mean = np.median([table[f"g_{t}"][:, 0] for table in filtered_tables], axis=0)
        g2_mean = np.median([table[f"g_{t}"][:, 1] for table in filtered_tables], axis=0)
        mcal_combined[f"g_{t}"] = np.column_stack((g1_mean, g2_mean))
        mcal_combined[f"flux_{t}"] = np.median([table[f"flux_{t}"] for table in filtered_tables], axis=0)

        # T / T_err: use stored columns if present, otherwise derive from hlr in pars[4]
        if has_T:
            T_list = [table[f"T_{t}"] for table in filtered_tables]
            T_err_list = [table[f"T_err_{t}"] for table in filtered_tables]
        else:
            T_pairs = [_T_from_hlr(table, t) for table in filtered_tables]
            T_list = [p[0] for p in T_pairs]
            T_err_list = [p[1] for p in T_pairs]

        mcal_combined[f"T_{t}"] = np.median(T_list, axis=0)
        mcal_combined[f"T_err_{t}"] = np.median(T_err_list, axis=0)

        mcal_combined[f"g_cov_{t}"] = np.median([table[f"g_cov_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"flux_err_{t}"] = np.median([table[f"flux_err_{t}"] for table in filtered_tables], axis=0)
        
        # Unweighted average for Tpsf
        mcal_combined[f"Tpsf_{t}"] = np.median([table[f"Tpsf_{t}"] for table in filtered_tables], axis=0) 
        mcal_combined[f"gpsf_{t}"] = np.median([table[f"gpsf_{t}"] for table in filtered_tables], axis=0)

        # S/N: use s2n_{t} if present, otherwise fall back to s2n_r_{t}
        if _has_col(test_table, f"s2n_{types[0]}"):
            s2n_col = f"s2n_{t}"
            has_s2n_r = False
        elif _has_col(test_table, f"s2n_r_{types[0]}"):
            s2n_col = f"s2n_r_{t}"
            has_s2n_r = True
        else:
            raise KeyError(
                f"Neither 's2n_{types[0]}' nor 's2n_r_{types[0]}' found in table columns. "
                f"Available columns: {list(getattr(test_table, 'colnames', test_table.dtype.names))}"
            )
        mcal_combined[f"s2n_{t}"] = np.median([table[s2n_col] for table in filtered_tables], axis=0)

    if not has_T and has_s2n_r:
        print("INFO: T/T_err columns absent and s2n_r detected — looks like these mcal tables were produced with GalSimFitter.")

    # Convert to an Astropy table and save
    mcal_combined_table = Table(mcal_combined)

    mask_file_path = os.path.join(data_dir, "star_masks")
    mask_file = f"{mask_file_path}/{run_name}_{band}_starmask_physical.reg"
    if os.path.exists(mask_file):
        print(f"Using star mask file: {mask_file}")
        junks_mcal, mcal_combined_table, reg_mask = separate_catalog_by_regions(mask_file, mcal_combined_table)
        junks_mcal.write(f"{outdir}/{run_name}_{band}_mcal_junks.{args.file_ending}", format="fits", overwrite=True)
    else:
        print(f"WARNING: Star mask file '{mask_file}' not found, skipping the discarding process.")

    if args.isolate_stars:
        base_path = os.path.join(data_dir, run_name, band)
        file_stars_union = f"{base_path}/coadd/{run_name}_coadd_{band}_starcat_union.fits"
        file_stars_fallback = f"{base_path}/coadd/{run_name}_coadd_{band}_starcat.fits"
        if os.path.exists(file_stars_union):
            starcat = Table.read(file_stars_union, format="fits", hdu=2)
            print(f"Using union star catalog: {file_stars_union}")
        elif os.path.exists(file_stars_fallback):
            starcat = Table.read(file_stars_fallback, format="fits", hdu=2)
            print(f"Using fallback star catalog: {file_stars_fallback}")
        else:
            raise FileNotFoundError("Star catalog not found. Cannot proceed.")

        tolerance_deg = 2*1e-5

        matcher = SkyCoordMatcher(mcal_combined_table, starcat, cat1_ratag='ra', cat1_dectag='dec',
                cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', match_radius=5 * tolerance_deg)
        mcal_discard, matched_stars = matcher.get_matched_pairs()
        mcal_discard['MAG_AUTO'] = matched_stars["MAG_AUTO"]
        total_mcal = len(mcal_combined_table)
        total_discard = len(mcal_discard)

        mask = np.isin(mcal_combined_table['id'], mcal_discard['id'])
        num_discarded = np.sum(mask)
        num_remaining = total_mcal - num_discarded

        print(f"Total objects in mcal: {total_mcal}")
        print(f"Number of objects discarded as stars: {num_discarded}")
        print(f"Number of objects remaining: {num_remaining}")

        filtered_mcal = mcal_combined_table[~mask]
        mcal_combined_table = filtered_mcal
        mcal_discard.write(f"{outdir}/{run_name}_{band}_mcal_stars.{args.file_ending}", format="fits", overwrite=True)
        print(f"Stars Mcal saved to {outdir}/{run_name}_{band}_mcal_stars.{args.file_ending}")

    if args.isolate_redseq:
        base_path = os.path.join(data_dir, run_name, band)
        red_seq_file = f"{base_path}/coadd/{run_name}_coadd_redseq.fits"
        if os.path.exists(red_seq_file):
            redseq = Table.read(red_seq_file, format="fits")
            print(f"Using Red-seq galaxy catalog: {red_seq_file}")
            tolerance_deg = 1e-6
            try:
                matcher = SkyCoordMatcher(mcal_combined_table, redseq, cat1_ratag='ra', cat1_dectag='dec',
                        cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', match_radius=5 * tolerance_deg)
            except:
                matcher = SkyCoordMatcher(mcal_combined_table, redseq, cat1_ratag='ra', cat1_dectag='dec', match_radius=5 * tolerance_deg)
            mcal_discard, matched_galaxies = matcher.get_matched_pairs()
            total_mcal = len(mcal_combined_table)
            total_discard = len(mcal_discard)

            mask = np.isin(mcal_combined_table['id'], mcal_discard['id'])
            num_discarded = np.sum(mask)
            num_remaining = total_mcal - num_discarded

            print(f"Total objects in mcal: {total_mcal}")
            print(f"Number of objects discarded as red-seq galaxies: {num_discarded}")
            print(f"Number of objects remaining: {num_remaining}")

            filtered_mcal = mcal_combined_table[~mask]
            mcal_combined_table = filtered_mcal
            mcal_discard.write(f"{outdir}/{run_name}_{band}_mcal_redseq.{args.file_ending}", format="fits", overwrite=True)
            print(f"Red Seq Galaxies Mcal saved to {outdir}/{run_name}_{band}_mcal_redseq.{args.file_ending}")            
        else:
            print(f"WARNING: Red Seq Galaxy catalog '{red_seq_file}' not found, skipping the discarding process.")

    mcal_combined_table.write(f"{outdir}/{run_name}_{band}_mcal_combined.{args.file_ending}", format="fits", overwrite=True)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print("Final combined table saved successfully!")
    else:
        print(f'combine_mcal.py has failed w/ rc={rc}')
