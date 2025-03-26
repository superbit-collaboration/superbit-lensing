import numpy as np
from astropy.table import Table
from argparse import ArgumentParser
from superbit_lensing.match import SkyCoordMatcher
import os
import ipdb

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-nrun', type=int, default=50, help='Number of realizations to combine')
    parser.add_argument('-data_dir', type=str, required=True, help='Path to cluster data')
    parser.add_argument('-run_name', type=str, required=True, help='Cluster Name')
    parser.add_argument('-band', type=str, required=True, help='Band name')
    parser.add_argument('-outdir', type=str, help='Output directory')
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
    mcal_files = [f"{data_dir}/{run_name}/{band}/arr/run{i}/{run_name}_{band}_mcal.fits" for i in range(1, nrun+1)]

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
    mcal_combined["ra"] = filtered_tables[0]["ra"]
    mcal_combined["dec"] = filtered_tables[0]["dec"]
    mcal_combined["XWIN_IMAGE"] = filtered_tables[0]["XWIN_IMAGE"]
    mcal_combined["YWIN_IMAGE"] = filtered_tables[0]["YWIN_IMAGE"]

    # Metacalibration types
    types = ["noshear", "1p", "1m", "2p", "2m", "1p_psf", "1m_psf", "2p_psf", "2m_psf"]

    # Number of realizations
    N = len(filtered_tables)

    for t in types:
        g1_mean = np.median([table[f"g_{t}"][:, 0] for table in filtered_tables], axis=0)
        g2_mean = np.median([table[f"g_{t}"][:, 1] for table in filtered_tables], axis=0)
        mcal_combined[f"g_{t}"] = np.column_stack((g1_mean, g2_mean))
        mcal_combined[f"T_{t}"] = np.median([table[f"T_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"flux_{t}"] = np.median([table[f"flux_{t}"] for table in filtered_tables], axis=0)

        mcal_combined[f"g_cov_{t}"] = np.median([table[f"g_cov_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"T_err_{t}"] = np.median([table[f"T_err_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"flux_err_{t}"] = np.median([table[f"flux_err_{t}"] for table in filtered_tables], axis=0)
        
        # Unweighted average for Tpsf
        mcal_combined[f"Tpsf_{t}"] = np.median([table[f"Tpsf_{t}"] for table in filtered_tables], axis=0) 
        mcal_combined[f"gpsf_{t}"] = np.median([table[f"gpsf_{t}"] for table in filtered_tables], axis=0)

        # Average S/N
        s2n_avg = np.median([table[f"s2n_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"s2n_{t}"] = s2n_avg

    # Convert to an Astropy table and save
    mcal_combined_table = Table(mcal_combined)
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

        tolerance_deg = 1e-6

        matcher = SkyCoordMatcher(mcal_combined_table, starcat, cat1_ratag='ra', cat1_dectag='dec',
                cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000', match_radius=5 * tolerance_deg)
        mcal_discard, matched_stars = matcher.get_matched_pairs()
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
        mcal_discard.write(f"{outdir}/{run_name}_{band}_mcal_stars.fits", format="fits", overwrite=True)
        print(f"Stars Mcal saved to {outdir}/{run_name}_{band}_mcal_stars.fits")

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
            mcal_discard.write(f"{outdir}/{run_name}_{band}_mcal_redseq.fits", format="fits", overwrite=True)
            print(f"Red Seq Galaxies Mcal saved to {outdir}/{run_name}_{band}_mcal_redseq.fits")            
        else:
            print(f"WARNING: Red Seq Galaxy catalog '{red_seq_file}' not found, skipping the discarding process.")


    mcal_combined_table.write(f"{outdir}/{run_name}_{band}_mcal_combined.fits", format="fits", overwrite=True)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print("Final combined table saved successfully!")
    else:
        print(f'combine_mcal.py has failed w/ rc={rc}')
