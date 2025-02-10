import numpy as np
from astropy.table import Table
from argparse import ArgumentParser
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-nrun', type=int, default=50, help='Number of realisations to combine')
    parser.add_argument('-data_dir', type=str, required=True, help='Path to cluster data')
    parser.add_argument('-run_name', type=str, required=True, help='Cluster Name')
    parser.add_argument('-band', type=str, required=True, help='Band name')
    parser.add_argument('-outdir', type=str, help='Output directory')
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

    # Generate file paths for 50 runs
    mcal_files = [f"{data_dir}/{run_name}/{band}/arr/run{i}/{run_name}_{band}_mcal.fits" for i in range(1, nrun+1)]

    # Load the mcal tables
    mcal_tables = [Table.read(f, format="fits") for f in mcal_files]


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
        g1_mean = np.mean([table[f"g_{t}"][:, 0] for table in filtered_tables], axis=0)
        g2_mean = np.mean([table[f"g_{t}"][:, 1] for table in filtered_tables], axis=0)
        mcal_combined[f"g_{t}"] = np.column_stack((g1_mean, g2_mean))
        mcal_combined[f"T_{t}"] = np.mean([table[f"T_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"flux_{t}"] = np.mean([table[f"flux_{t}"] for table in filtered_tables], axis=0)

        mcal_combined[f"g_cov_{t}"] = np.median([table[f"g_cov_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"T_err_{t}"] = np.median([table[f"T_err_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"flux_err_{t}"] = np.median([table[f"flux_err_{t}"] for table in filtered_tables], axis=0)
        
        # Unweighted average for Tpsf
        mcal_combined[f"Tpsf_{t}"] = np.mean([table[f"Tpsf_{t}"] for table in filtered_tables], axis=0) 
        mcal_combined[f"gpsf_{t}"] = np.mean([table[f"gpsf_{t}"] for table in filtered_tables], axis=0)

        # Average S/N
        s2n_avg = np.median([table[f"s2n_{t}"] for table in filtered_tables], axis=0)
        mcal_combined[f"s2n_{t}"] = s2n_avg

    # Convert to an Astropy table and save
    mcal_combined_table = Table(mcal_combined)
    mcal_combined_table.write(f"{outdir}/{run_name}_{band}_mcal_combined.fits", format="fits", overwrite=True)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print("Final combined table saved successfully!")
    else:
        print(f'combine_mcal.py has failed w/ rc={rc}')    