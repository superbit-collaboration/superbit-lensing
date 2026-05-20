"""
run_pipeline.py

Unified entry point for the weak-lensing convergence pipeline.
Reads a single config file and optionally runs:
  1. target_separator  (foreground / low-z removal, shear responsivity)
  2. get_snr           (KS inversion, noise realisations, SNR map)

Usage:
    python run_pipeline.py pipeline_config.yaml
"""

import os
import sys
import yaml
from omegaconf import OmegaConf

# ── The two pipeline stages ──────────────────────────────────
import target_seperator as separator
import get_snr as snr


# ==============================================================
#  Config helpers
# ==============================================================

def load_pipeline_config(path):
    """Load the unified YAML and return a plain dict (OmegaConf-resolved)."""
    cfg = OmegaConf.load(path)
    # Resolve all ${...} interpolations into a plain dict
    return OmegaConf.to_container(cfg, resolve=True)


def build_separator_config(pcfg):
    """
    Extract the subset of keys that target_separator.load_config
    would normally produce, so we can call separator.run() directly.
    """
    cfg = OmegaConf.create({
        # Target
        "target_name":      pcfg["target_name"],
        "target_redshift":  pcfg["target_redshift"],
        "redshift_offset":  pcfg["redshift_offset"],

        # Paths
        "data_dir":         pcfg["data_dir"],
        "mega_file":        pcfg["mega_file"],
        "outfile":          pcfg["separator_outfile"],
        "overwrite":        pcfg["overwrite"],
        "foreground_file":  pcfg["foreground_file"],

        # Metacalibration
        "mcal_shear":       pcfg["mcal_shear"],

        # Matching
        "tolerance_arcsec": pcfg["tolerance_arcsec"],

        # Quality cuts
        "qual_cuts":        pcfg["qual_cuts"],
        
        # Boundary cuts
        "boundary_cuts":    pcfg["boundary_cuts"],

        # NED
        "ned":              pcfg["ned"],
    })

    # Derived quantities (mirrors target_separator.load_config)
    cfg["redshift_cut"]   = cfg["target_redshift"] + cfg["redshift_offset"]
    cfg["tolerance_deg"]  = cfg["tolerance_arcsec"] / 3600.0
    cfg["ned_redshifts_path"] = os.path.join(
        cfg["data_dir"], "catalogs", "redshifts",
        f"{cfg['target_name']}_NED_redshifts.csv",
    )
    cfg["qual_cuts"]["foreground_file"] = cfg["foreground_file"]

    return cfg


def build_snr_config(pcfg):
    """
    Build the dict that get_snr expects (same schema as config.yaml
    consumed by utils.read_config).
    """
    return {
        "cluster":          pcfg["target_name"],
        "band":             pcfg["band"],
        "input_path":       pcfg["separator_outfile"],
        "output_path":      pcfg["snr_output_path"],

        "resolution":       pcfg["resolution"],
        "gaussian_kernel":  pcfg["gaussian_kernel"],
        "aperture_mass_Rs": pcfg["aperture_mass_Rs"],
        "gridding":         pcfg["gridding"],

        "num_sims":         pcfg["num_sims"],
        "seed_sims":        pcfg["seed_sims"],
        "shuffle_type":     pcfg["shuffle_type"],

        "kmap_vmax":        pcfg["kmap_vmax"],
        "kmap_vmin":        pcfg["kmap_vmin"],
        "kmap_threshold":   pcfg["kmap_threshold"],
        "vmax":             pcfg["vmax"],
        "vmin":             pcfg["vmin"],
        "threshold":        pcfg["threshold"],

        "figsize":          pcfg["figsize"],
        "cmap":             pcfg["cmap"],
        "xlabel":           pcfg["xlabel"],
        "ylabel":           pcfg["ylabel"],
        "plot_title":       pcfg["plot_title"],
        "gridlines":        pcfg["gridlines"],

        "ra_col":           pcfg["ra_col"],
        "dec_col":          pcfg["dec_col"],
        "g1_col":           pcfg["g1_col"],
        "g2_col":           pcfg["g2_col"],
        "weight_col":       pcfg["weight_col"],
        "x_col":            pcfg["x_col"],
        "y_col":            pcfg["y_col"],

        "mode":             pcfg["mode"],
        "box_boundary":     pcfg.get("box_boundary"),

        # X-ray contour overlay
        "xray_contour_file":  pcfg.get("xray_contour_file"),
        "xray_contour_color": pcfg.get("xray_contour_color", "cyan"),
        "xray_contour_s":    pcfg.get("xray_contour_s", 0.3),
    }


# ==============================================================
#  Main
# ==============================================================

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pipeline_config.yaml>")
        sys.exit(1)

    pcfg = load_pipeline_config(sys.argv[1])

    # ── Stage 1: target separator ────────────────────────────
    if pcfg.get("run_separator", True):
        print("\n" + "=" * 60)
        print("  STAGE 1 — Target Separator")
        print("=" * 60)
        sep_cfg = build_separator_config(pcfg)
        separator.run(sep_cfg)
    else:
        print("\n[skip] Target separator disabled in config")
        print(f"       Using existing catalog: {pcfg['separator_outfile']}")

    # ── Stage 2: SNR mapping ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 — Convergence & SNR Mapping")
    print("=" * 60)

    snr_cfg = build_snr_config(pcfg)

    # Flags that used to be CLI arguments
    snr_flags = {
        "plot_kappa": pcfg.get("plot_kappa", False),
        "plot_error": pcfg.get("plot_error", False),
        "save_fits_flag":  pcfg.get("save_fits", False),
    }

    snr.run(snr_cfg, **snr_flags)


if __name__ == "__main__":
    main()