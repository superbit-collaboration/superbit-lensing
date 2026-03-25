import os
import argparse
from glob import glob
from astropy.io import fits

# ---------------------------
# Default data directory
# ---------------------------
DATA_DIR = "/n23data1/saha/data"

GREEN = "\033[92m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def check_band(data_dir, cluster_name, band):
    """
    Check number of clean exposures and their quality for a given band.
    """
    path = os.path.join(data_dir, cluster_name, band, "cal", "*_clean.fits")
    files = glob(path)

    total = len(files)
    good = 0
    bad = 0
    unknown = 0

    for f in files:
        try:
            with fits.open(f) as hdul:
                header = hdul[0].header
                img_qual = header.get("IMG_QUAL", "UNKNOWN")

                if img_qual == "GOOD":
                    good += 1
                elif img_qual == "BAD":
                    bad += 1
                else:
                    unknown += 1

        except Exception as e:
            print(f"[WARNING] Could not read {f}: {e}")
            unknown += 1

    return {
        "band": band,
        "total": total,
        "good": good,
        "bad": bad,
        "unknown": unknown,
    }


def main():
    parser = argparse.ArgumentParser(description="Check clean exposure quality per band.")
    parser.add_argument("cluster", help="Cluster name (e.g., Abell3411)")
    parser.add_argument(
        "--data_dir",
        default=DATA_DIR,
        help="Path to data directory (default: %(default)s)"
    )

    args = parser.parse_args()

    bands = ["u", "b", "g"]

    print(f"\nChecking cluster: {args.cluster}")
    print(f"Data directory: {args.data_dir}\n")

    for band in bands:
        result = check_band(args.data_dir, args.cluster, band)

        print(f"{BOLD}Band: {band}{RESET}")
        print(f"  Total               : {result['total']}")
        print(f"  GOOD                : {GREEN}{result['good']}{RESET}")
        print(f"  BAD                 : {result['bad']}")
        print("-" * 40)


if __name__ == "__main__":
    main()