import os
import shutil
import argparse

DATADIR = "/n23data1/saha/data"  # <-- set this


def safe_rmtree(path):
    """Remove directory if it exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Removed directory: {path}")


def safe_remove_file(path):
    """Remove file if it exists."""
    if os.path.isfile(path):
        os.remove(path)
        print(f"Removed file: {path}")


def clean_band(cluster_path, band):
    band_path = os.path.join(cluster_path, band)

    if not os.path.isdir(band_path):
        print(f"Skipping missing band directory: {band_path}")
        return

    print(f"\nCleaning band: {band}")

    # 2. go to cal/ and delete everything except *_clean.fits
    cal_path = os.path.join(band_path, "cal")
    if os.path.isdir(cal_path):
        for fname in os.listdir(cal_path):
            fpath = os.path.join(cal_path, fname)
            if os.path.isfile(fpath) and not fname.endswith("_clean.fits"):
                safe_remove_file(fpath)

    # 3. delete everything inside coadd, cat, out
    for sub in ["coadd", "cat", "out"]:
        sub_path = os.path.join(band_path, sub)
        if os.path.isdir(sub_path):
            for item in os.listdir(sub_path):
                item_path = os.path.join(sub_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print(f"Cleared directory: {sub_path}")

    # 4. remove arr directory if present
    safe_rmtree(os.path.join(band_path, "arr"))


def main(args):
    datadir = args.data_dir
    clustername = args.clustername
    cluster_path = os.path.join(datadir, clustername)
    if not os.path.isdir(cluster_path):
        raise ValueError(f"Cluster path does not exist: {cluster_path}")

    print(f"Starting cleanup for cluster: {clustername}")
    safe_rmtree(os.path.join(cluster_path, "sextractor_dualmode"))

    for band in ["u", "g", "b"]:
        clean_band(cluster_path, band)

    print("\n✅ Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup cluster directories")
    parser.add_argument("clustername", help="Name of the cluster")
    parser.add_argument('--data_dir', type=str, default=DATADIR,
                        help='Base directory for data (default: /projects/mccleary_group/superbit/union)')
    args = parser.parse_args()

    main(args)