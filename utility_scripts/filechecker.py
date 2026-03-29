import os
import argparse
from glob import glob
from astropy.io import fits
from astropy.table import Table

# ---------------------------
# Default data directory
# ---------------------------
DATA_DIR = "/n23data1/saha/data"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SUMMARY_TAB_FILE = os.path.join(PROJECT_ROOT, 'data', 'bit_exposure_summary.fits')

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_summary_table(path):
    table = Table.read(path)
    summary = {}

    for row in table:
        target = row["target"]
        summary[target] = {
            "u": row["u_good"],
            "b": row["b_good"],
            "g": row["g_good"],
        }

    return summary


def check_band(data_dir, cluster_name, band):
    path = os.path.join(data_dir, cluster_name, band, "cal", "*_clean.fits")
    files = glob(path)

    good = 0

    for f in files:
        try:
            with fits.open(f) as hdul:
                img_qual = hdul[0].header.get("IMG_QUAL", "UNKNOWN")

                if img_qual == "GOOD":
                    good += 1

        except Exception as e:
            print(f"[WARNING] Could not read {f}: {e}")

    return good


def main():
    parser = argparse.ArgumentParser(description="Check clean exposure quality per band.")
    parser.add_argument("cluster", help="Cluster name (e.g., Abell3411)")
    parser.add_argument(
        "--data_dir",
        default=DATA_DIR,
        help="Path to data directory (default: %(default)s)"
    )

    args = parser.parse_args()

    # ---------------------------
    # Load summary table
    # ---------------------------
    summary = load_summary_table(SUMMARY_TAB_FILE)

    if args.cluster not in summary:
        print(f"{RED}Cluster {args.cluster} not found in summary table!{RESET}")
        return

    expected = summary[args.cluster]

    bands = ["u", "b", "g"]

    print(f"\n{BOLD}Checking cluster:{RESET} {args.cluster}")
    print(f"Data directory: {args.data_dir}\n")

    for band in bands:
        actual_good = check_band(args.data_dir, args.cluster, band)
        expected_good = expected[band]

        # ---------------------------
        # Compare
        # ---------------------------
        if actual_good == expected_good:
            status = f"{GREEN}MATCH ✔{RESET}"
        else:
            status = f"{RED}MISMATCH ✘{RESET}"

        print(f"{BOLD}Band: {band}{RESET}")
        print(f"  GOOD (local)   : {actual_good}")
        print(f"  GOOD (expected): {expected_good}")
        print(f"  Status         : {status}")
        print("-" * 40)


if __name__ == "__main__":
    main()