"""
get_snr.py — Compute and plot SNR maps from shear catalogues.

Usage (standalone)
------------------
    python get_snr.py -c config.yaml [--plot_kappa] [--plot_error] [--save_fits]

Usage (from pipeline)
---------------------
    import get_snr
    get_snr.run(config_dict, plot_kappa=True, plot_error=True, save_fits=True)
"""

import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from utils import (
    read_config,
    load_shear_data,
    save_fits,
    calculate_field_boundaries,
    correct_RA_dec,
    correct_center,
    correct_box_boundary,
    create_shear_grid,
    create_count_grid,
    ks_inversion,
    generate_shuffled_shear_dfs,
    shear_grids_for_shuffled_dfs,
    ks_inversion_list,
    compute_snr,
    plot_convergence,
)

from ap_mass import ApertureMassCalculator, ApertureMassSNR


# ---------------------------------------------------------------------------
#  X-ray contour support
# ---------------------------------------------------------------------------
def read_ds9_ctr(filename):
    """Parse a DS9 .ctr contour file into a list of polylines."""
    contours = []
    current = []
    coord_system = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() in ('fk5', 'image', 'physical', 'wcs'):
                coord_system = line.lower()
            elif line.lower() == "line":
                if current:
                    contours.append(current)
                    current = []
            else:
                try:
                    parts = list(map(float, line.split()))
                    current.append(parts)
                except ValueError:
                    continue
        if current:
            contours.append(current)

    if coord_system is None:
        coord_system = 'fk5'
    return contours, coord_system


def load_xray_contours(config):
    """
    Read the X-ray contour file from config if present.

    Returns a list of (ra_array, dec_array) polylines, or None.
    """
    ctr_path = config.get("xray_contour_file")
    if not ctr_path or not os.path.isfile(ctr_path):
        return None

    print(f"Loading X-ray contours from {ctr_path}")
    contours, coord_system = read_ds9_ctr(ctr_path)
    print(f"  {len(contours)} contour segments  (coord_system={coord_system})")

    if coord_system != 'fk5':
        print(f"  Warning: contour coord system is '{coord_system}', "
              "expected 'fk5'. Coordinates may be wrong.")

    polylines = []
    for seg in contours:
        arr = np.array(seg)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            polylines.append((arr[:, 0], arr[:, 1]))  # (RA, Dec)
    return polylines


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process a shear catalog and compute the SNR map.")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--plot_kappa", action="store_true",
                        help="Plot E- and B-mode convergence maps.")
    parser.add_argument("--plot_error", action="store_true",
                        help="Plot the noise standard-deviation map.")
    parser.add_argument("--save_fits", action="store_true",
                        help="Save convergence / noise maps as FITS files.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Helpers for the two gridding branches
# ---------------------------------------------------------------------------
def _resolve_gridding(config, shear_df):
    """
    Return ``(coord1_col, coord2_col, resolution, ks_key, flip_ra)``
    depending on ``config['gridding']``.
    """
    gridding = config["gridding"]

    if gridding == "xy":
        # Compute the pixel ↔ sky scale factor
        x_factor = ((shear_df["x"].max() - shear_df["x"].min())
                     / (shear_df["ra"].max() - shear_df["ra"].min()))
        y_factor = ((shear_df["y"].max() - shear_df["y"].min())
                     / (shear_df["dec"].max() - shear_df["dec"].min()))
        factor = (x_factor + y_factor) / 2
        resolution = int(config["resolution"] * factor)
        return "x", "y", resolution, "x-y", False

    if gridding == "ra_dec":
        return "ra", "dec", config["resolution"], "ra-dec", True

    raise KeyError(f"Invalid gridding '{gridding}'. Use 'xy' or 'ra_dec'.")


def _save_noise_realisations(smoothed_stack, kappa_b_stack_smoothed,
                             boundaries, true_boundaries, config,
                             kernel, center_cl, num_sims):
    """Save every 250th noise realisation as FITS + PNG (E & B modes)."""
    noises_dir = os.path.join(config["output_path"], "noises")
    os.makedirs(noises_dir, exist_ok=True)

    cluster_band = f"{config['cluster']}_{config['band']}"
    res_str = f"(Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})"

    for i in range(num_sims):
        if i == 0 or i % 250 != 0:
            continue

        # E-mode
        fname = os.path.join(noises_dir,
                             f"noise_e_realisation_{i}_{cluster_band}.fits")
        save_fits(smoothed_stack[i], true_boundaries, fname)
        plot_convergence(
            smoothed_stack[i], boundaries, true_boundaries, config,
            invert_map=False,
            title=f"Noise E mode ({i}): {cluster_band} {res_str}",
            vmax=config["kmap_vmax"], vmin=config["kmap_vmin"],
            center_cl=center_cl,
            save_path=os.path.join(noises_dir,
                                   f"noise_e_realisation_{i}_{cluster_band}.png"),
        )

        # B-mode
        if kappa_b_stack_smoothed is not None:
            fname = os.path.join(noises_dir,
                                 f"noise_b_realisation_{i}_{cluster_band}.fits")
            save_fits(kappa_b_stack_smoothed[i], true_boundaries, fname)
            plot_convergence(
                kappa_b_stack_smoothed[i], boundaries, true_boundaries, config,
                invert_map=False,
                title=f"Noise B mode ({i}): {cluster_band} {res_str}",
                vmax=config["kmap_vmax"], vmin=config["kmap_vmin"],
                center_cl=center_cl,
                save_path=os.path.join(
                    noises_dir,
                    f"noise_b_realisation_{i}_{cluster_band}.png"),
            )


# ---------------------------------------------------------------------------
#  Core logic (callable from pipeline or CLI)
# ---------------------------------------------------------------------------
def run(config, plot_kappa=False, plot_error=False, save_fits_flag=False):
    """
    Run the full SNR pipeline.

    Parameters
    ----------
    config : dict
        Same schema as the YAML consumed by ``utils.read_config``.
    plot_kappa : bool
        Plot E- and B-mode convergence maps.
    plot_error : bool
        Plot noise standard-deviation map.
    save_fits_flag : bool
        Save convergence / noise maps as FITS files.
    """

    # --- Load data -----------------------------------------------------------
    shear_df = load_shear_data(
        config["input_path"],
        config["ra_col"], config["dec_col"],
        config["g1_col"], config["g2_col"],
        config["weight_col"],
        config["x_col"], config["y_col"],
    )
    print(f"Loaded shear data from {config['input_path']}")

    # --- Coordinate correction -----------------------------------------------
    true_boundaries = calculate_field_boundaries(shear_df["ra"], shear_df["dec"])
    shear_df, ra_0, dec_0 = correct_RA_dec(shear_df)
    boundaries = calculate_field_boundaries(shear_df["ra"], shear_df["dec"])
    print(true_boundaries)

    box_boundary = config.get("box_boundary")
    if box_boundary is not None:
        box_boundary = correct_box_boundary(box_boundary, ra_0, dec_0)

    center_cl = None  # config.get("center")
    if center_cl is not None:
        center_cl = correct_center(center_cl, ra_0, dec_0)

    # --- X-ray contour overlay -----------------------------------------------
    xray_contours = load_xray_contours(config)
    contour_kwargs = {}
    if xray_contours is not None:
        contour_kwargs = {
            "xray_contours":     xray_contours,
            "xray_contour_color": config.get("xray_contour_color", "cyan"),
            "xray_contour_s":    config.get("xray_contour_s", 0.3),
        }

    # --- Resolve gridding strategy -------------------------------------------
    c1, c2, resolution, ks_key, flip_ra = _resolve_gridding(config, shear_df)
    kernel = config["gaussian_kernel"]
    cluster_band = f"{config['cluster']}_{config['band']}"
    res_str = f"(Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})"

    # --- Build signal maps ---------------------------------------------------
    g1map, g2map, g_sq_map, res_list = create_shear_grid(
        shear_df[c1], shear_df[c2],
        shear_df["g1"], shear_df["g2"],
        resolution, shear_df["weight"], verbose=True,
    )

    # Save raw shear grids (xy branch only, matching original behaviour)
    # if config["gridding"] == "xy" and save_fits_flag:
    #     save_fits(g1map, true_boundaries,
    #               os.path.join(config["output_path"], f"g1_{cluster_band}.fits"))
    #     save_fits(g2map, true_boundaries,
    #               os.path.join(config["output_path"], f"g2_{cluster_band}.fits"))

    og_kappa_e, og_kappa_b = ks_inversion(g1map, g2map, key=ks_key)

    if flip_ra:
        og_kappa_e = og_kappa_e[:, ::-1]
        og_kappa_b = og_kappa_b[:, ::-1]

    og_kappa_e_sm = gaussian_filter(og_kappa_e, kernel)
    og_kappa_b_sm = gaussian_filter(og_kappa_b, kernel)

    os.makedirs(os.path.join(config["output_path"], "ks"), exist_ok=True)

    # --- Optional: plot convergence maps -------------------------------------
    if plot_kappa:
        plot_convergence(
            og_kappa_e_sm, boundaries, true_boundaries, config,
            invert_map=False,
            title=f"Convergence: {cluster_band} {res_str}",
            vmax=config["kmap_vmax"], vmin=config["kmap_vmin"],
            threshold=config.get("kmap_threshold"),
            center_cl=center_cl, box_boundary=box_boundary,
            save_path=os.path.join(config["output_path"], "ks",
                                   f"kappa_{cluster_band}.png"),
            **contour_kwargs,
        )
        plot_convergence(
            og_kappa_b_sm, boundaries, true_boundaries, config,
            invert_map=False,
            title=f"Convergence (B-modes): {cluster_band} {res_str}",
            vmax=config["kmap_vmax"], vmin=config["kmap_vmin"],
            center_cl=center_cl, box_boundary=box_boundary,
            save_path=os.path.join(config["output_path"], "ks",
                                   f"kappa_b_{cluster_band}.png"),
        )

    # --- Optional: save convergence FITS -------------------------------------

        # save_fits(og_kappa_b_sm, true_boundaries,
        #           os.path.join(config["output_path"], "ks", f"kappa_b_{cluster_band}.fits"))

        count_grid = create_count_grid(shear_df[c1], shear_df[c2],
                                       resolution, verbose=False)
        # save_fits(count_grid, true_boundaries,
        #           os.path.join(config["output_path"], "ks", f"count_{cluster_band}.fits"))

    # --- Noise realisations + SNR --------------------------------------------
    shuffle_type = config["shuffle_type"]
    if shuffle_type not in ("rotation", "spatial"):
        raise KeyError(f"Invalid shuffle_type '{shuffle_type}'. "
                       "Use 'rotation' or 'spatial'.")

    # TODO: implement a true spatial shuffle; currently both map to rotation.
    shuffled_dfs = generate_shuffled_shear_dfs(
        shear_df, config["num_sims"], seed=config["seed_sims"])

    grid_list = shear_grids_for_shuffled_dfs(
        shuffled_dfs, c1, c2, resolution)

    shuff_ke, shuff_kb = ks_inversion_list(grid_list, key=ks_key)

    ke_stack = np.stack(shuff_ke, axis=0)
    kb_stack = np.stack(shuff_kb, axis=0)

    _, std_map, snr_map, ke_smoothed = compute_snr(
        og_kappa_e, ke_stack, kernel, flip_ra=flip_ra)

    # --- Optional: save noise FITS + realisations ----------------------------
    if save_fits_flag:
        # save_fits(std_map, true_boundaries,
        #           os.path.join(config["output_path"],
        #                        f"error_{cluster_band}.fits"))

        # Also smooth B-mode noise for the realisation snapshots
        kb_smoothed = np.empty_like(kb_stack)
        for i in range(kb_stack.shape[0]):
            frame = kb_stack[i]
            if flip_ra:
                frame = frame[:, ::-1]
            kb_smoothed[i] = gaussian_filter(frame, kernel)

        # _save_noise_realisations(
        #     ke_smoothed, kb_smoothed,
        #     boundaries, true_boundaries, config,
        #     kernel, center_cl, config["num_sims"],
        # )

    # --- Optional: plot error map --------------------------------------------
    if plot_error:
        plot_convergence(
            std_map, boundaries, true_boundaries, config,
            invert_map=False,
            title=f"Error: {cluster_band} {res_str}",
            center_cl=center_cl,
            save_path=os.path.join(config["output_path"], "ks",
                                   f"error_{cluster_band}.png"),
        )

    # --- SNR map (always plotted) --------------------------------------------
    plot_convergence(
        snr_map, boundaries, true_boundaries, config,
        invert_map=False,
        title=(f" KS Mass SNR: {config['cluster']} {res_str}"),
        vmax=config["vmax"], vmin=config["vmin"],
        threshold=config["threshold"],
        center_cl=center_cl,
        save_path=os.path.join(config["output_path"], "ks",
                               f"snr_{config['cluster']}.png"),
        **contour_kwargs,
    )
    
    if save_fits_flag:
        save_fits(snr_map, true_boundaries,
                  os.path.join(config["output_path"], "ks", f"snr_ks_{cluster_band}.fits"))
    
    # ------- Aperture mass SNR (TODO) ------------------------------------------------
    os.makedirs(os.path.join(config["output_path"], "aperture_mass"), exist_ok=True)
    
    pixel_scale = 0.141 #arcsec per pixel
    Rs = config["aperture_mass_Rs"] # in image pixels
    bin_size = config["resolution"] * 60 / pixel_scale # convert arcmin to pixels
    snr = ApertureMassSNR(shear_df["g1"], shear_df["g2"], shear_df["weight"], shear_df["x"], shear_df["y"], bin_size=bin_size, Rs=Rs)
    SNR_E, SNR_B = snr.run(n_realizations=config["num_sims"], n_cpus=32, seed=config["seed_sims"])

    
    res_str = f"(Resolution: {config['resolution']:.2f} arcmin, Rs: {(Rs/bin_size):.2f})"
    
    plot_convergence(
        SNR_E, boundaries, true_boundaries, config,
        invert_map=False,
        title=(f" Aperture Mass SNR: {config['cluster']} {res_str}"),
        vmax=config["vmax"], vmin=config["vmin"],
        threshold=config["threshold"],
        center_cl=center_cl,
        save_path=os.path.join(config["output_path"], "aperture_mass",
                               f"snr_aperture_mass_{config['cluster']}.png"),
        **contour_kwargs,
    )
    
    plot_convergence(
        SNR_B, boundaries, true_boundaries, config,
        invert_map=False,
        title=(config["plot_title"] + f" Aperture Mass B-mode: {config['cluster']} {res_str}"),
        vmax=config["vmax"], vmin=config["vmin"],
        threshold=config["threshold"],
        center_cl=center_cl,
        save_path=os.path.join(config["output_path"], "aperture_mass",
                               f"M_B_aperture_mass_{config['cluster']}.png"),
        **contour_kwargs,
    )
    
    if save_fits_flag:
        save_fits(SNR_E, true_boundaries,
                  os.path.join(config["output_path"], "aperture_mass",
                               f"snr_aperture_mass_{config['cluster']}.fits"))
        # save_fits(SNR_B, true_boundaries,
        #           os.path.join(config["output_path"], "aperture_mass",
        #                        f"snr_b_aperture_mass_{config['cluster']}.fits"))
    # ------------ KS Plus ---------------
    # os.makedirs(os.path.join(config["output_path"], "ksplus"), exist_ok=True)
    # common = dict(
    #     data=config["input_path"],
    #     coord_system="radec",
    #     pixel_scale=0.4,
    #     g1_col=config["g1_col"],
    #     g2_col=config["g2_col"],
    #     weight_col=config["weight_col"],
    #     save_plots=False,
    # )
    # from smpy import map_ks_plus

    # result_ksp = map_ks_plus(**common, inpainting_iterations=200, reduced_shear_iterations=10)
    # maps = result_ksp["maps"]
    # e_signal = maps["E"]
    # b_signal = maps["B"]
    
    # plot_convergence(
    #     e_signal, boundaries, true_boundaries, config,
    #     invert_map=False,
    #     title=(f" KS+ Mass E: {cluster_band} {res_str}"),
    #     vmax=None, vmin=None,
    #     threshold=config["threshold"],
    #     center_cl=center_cl,
    #     save_path=os.path.join(config["output_path"], "ksplus",
    #                            f"M_E_ksplus_{cluster_band}.png"),
    #     **contour_kwargs,
    # )
    
    # plot_convergence(
    #     b_signal, boundaries, true_boundaries, config,
    #     invert_map=False,
    #     title=(config["plot_title"] + f" KS+ Mass B-mode: {cluster_band} {res_str}"),
    #     vmax=None, vmin=None,
    #     threshold=config["threshold"],
    #     center_cl=center_cl,
    #     save_path=os.path.join(config["output_path"], "ksplus",
    #                            f"M_B_ksplus_{cluster_band}.png"),
    #     **contour_kwargs,
    # )
    
    ## Now noise realisations for KS+ (TODO)
    


# ---------------------------------------------------------------------------
#  Standalone CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()
    config = read_config(args.config)
    run(config,
        plot_kappa=args.plot_kappa,
        plot_error=args.plot_error,
        save_fits_flag=args.save_fits)


if __name__ == "__main__":
    main()