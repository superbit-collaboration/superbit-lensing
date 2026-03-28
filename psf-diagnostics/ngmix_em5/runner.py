from astropy.table import Table
from superbit_lensing.em5 import compute_em5_psfex_maps
import os
import math

def build_file_pairs(tbl, emp_psf_path):
    file_pairs = [
        (
            os.path.join(emp_psf_path, "all", os.path.basename(row["psf_model_file"])),
            os.path.join(emp_psf_path, "all_image_headers", os.path.basename(row["image_file"])),
        )
        for row in tbl
    ]
    return [(p, i) for p, i in file_pairs if os.path.exists(p) and os.path.exists(i)]

BASE_PATH = "/n23data1/saha/simulated_data/sim_utils/emp_psfs"


def main():
    tbl = Table.read(os.path.join(BASE_PATH, "psf_summary_table.fits"))
    tbl = tbl[tbl["does_exist"]]
    # tbl = tbl[tbl["cluster_name"] == "1E0657_Bullet"]
    # tbl = tbl[tbl["n_good_stars"] <= 500]

    emp_psf_path = os.path.join(BASE_PATH, "emp_psfs_all")
    output_dir = os.path.join(BASE_PATH, "emp_psfs_all", "em5_diags_output")
    os.makedirs(output_dir, exist_ok=True)

    file_pairs = build_file_pairs(tbl, emp_psf_path)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    ntasks  = int(os.environ["SLURM_ARRAY_TASK_COUNT"])  # set by SBATCH --array

    # split work evenly across array tasks
    n = len(file_pairs)
    chunk = math.ceil(n / ntasks)
    start = task_id * chunk
    end = min(n, start + chunk)

    print(f"Task {task_id}/{ntasks}: processing indices [{start}, {end}) out of {n}")

    for psf_f, img_f in file_pairs[start:end]:
        outfile = os.path.join(output_dir, os.path.basename(psf_f).replace(".psf", ".npz"))

        # ✅ keep your skip logic
        if os.path.exists(outfile):
            print(f"Skipping (exists): {outfile}")
            continue

        print(f"Processing: {psf_f} with {img_f}")
        m = compute_em5_psfex_maps(psf_f, image_file=img_f, step=100, margin=0)
        m.to_npz(outfile)
        print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()