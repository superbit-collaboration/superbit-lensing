"""
rgb_tools.py — RGB composite image creation for SuperBIT galaxy cluster data.

Usage
-----
    # Raw kwargs (works exactly as before)
    make_rgb_image(u, b, g, stretch="sqrt", dpi=300)

    # Config object for reuse / saving
    cfg = RGBConfig(stretch="sqrt", dpi=300, xray_alpha=0.5)
    make_rgb_image(u, b, g, config=cfg)

    # Presets
    cfg = RGBConfig.from_preset("poster")
    make_rgb_image(u, b, g, config=cfg, output_file="a1689.png")

    # Save / reload a config you liked
    cfg.save("a1689_config.json")
    cfg = RGBConfig.load("a1689_config.json")
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from dataclasses import dataclass, fields
from typing import Optional, Tuple, Any, Dict
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize, MinMaxInterval,
    SqrtStretch, LogStretch, AsinhStretch,
)
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ═════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RGBConfig:
    """
    All non-file parameters for make_rgb_image() in one object.

    Pass as ``make_rgb_image(u, b, g, config=cfg)``.
    Any extra kwargs override the config values.
    """

    # Stretch / channel balance
    stretch: str = "asinh"
    percentile_limits: Tuple[float, float] = (0.5, 99.5)
    red_boost_factor: float = 1.1
    green_supression: float = 0.9
    blue_suppression: float = 0.9

    # Output
    output_file: Optional[str] = None
    dpi: int = 600
    format: str = "png"
    figsize: Tuple[int, int] = (12, 12)

    # RA/Dec crop
    ra_min: Optional[float] = None
    ra_max: Optional[float] = None
    dec_min: Optional[float] = None
    dec_max: Optional[float] = None

    # X-ray overlay
    xray_fits: Optional[str] = None
    xray_alpha: float = 0.45
    xray_smooth_sigma: float = 5
    xray_peak_gamma: float = 2.0
    xray_bin_factor: int = 150
    clip_sigma_xray: float = 20

    # Kappa overlay
    kappa_fits: Optional[str] = None
    kappa_alpha: float = 0.35
    kappa_smooth_sigma: float = 3
    kappa_peak_gamma: float = 2.0
    kappa_bin_factor: int = 1
    clip_sigma_kappa: float = 10
    r_inner_kappa: int = 1500
    r_outer_kappa: int = 2000

    # Overlay order: "kappa_first" (default) or "xray_first"
    overlay_order: str = "kappa_first"

    # Extras
    catalog: Any = None
    text: Optional[str] = None
    text_fontsize: int = 22
    text_color: str = "white"
    
    # External kappa contour overlay
    ext_kappa_fits: Optional[str] = None
    ext_kappa_err_fits: Optional[str] = None
    ext_kappa_smooth_sigma: float = 1.0       # in arcsec, converted to pixels internally
    ext_kappa_contour_levels: Optional[list] = None  # e.g. [1, 1.5, 2.2] for SNR levels
    ext_kappa_contour_color: str = "white"
    ext_kappa_show_footprint: bool = True

    # ── Presets ───────────────────────────────────────────────────────────

    _PRESETS: Dict[str, Dict[str, Any]] = None  # class-level, set below

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "RGBConfig":
        """
        Create a config from a named preset, with optional overrides.

            cfg = RGBConfig.from_preset("poster", dpi=150)
        """
        if name not in cls._PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. "
                f"Available: {', '.join(cls._PRESETS)}"
            )
        return cls(**{**cls._PRESETS[name], **overrides})

    # ── Helpers ──────────────────────────────────────────────────────────

    def to_kwargs(self) -> Dict[str, Any]:
        """Flatten to a dict suitable for **-unpacking into make_rgb_image."""
        skip = {"_PRESETS"}
        return {f.name: getattr(self, f.name)
                for f in fields(self) if f.name not in skip}

    def save(self, path: str):
        """Save to JSON (catalog excluded)."""
        kw = self.to_kwargs()
        kw.pop("catalog", None)
        with open(path, "w") as f:
            json.dump(kw, f, indent=2)
        print(f"Config saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RGBConfig":
        """Load from JSON."""
        with open(path) as f:
            return cls(**json.load(f))

    def __repr__(self):
        changed = []
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if val is not None and val != f.default:
                changed.append(f"  {f.name}={val!r}")
        body = ",\n".join(changed)
        return f"RGBConfig(\n{body}\n)" if changed else "RGBConfig()"


# Preset definitions
RGBConfig._PRESETS = {
    "default": {},
    "deep_faint": {
        "stretch": "asinh",
        "percentile_limits": (0.1, 99.9),
        "red_boost_factor": 1.0,
        "green_supression": 1.0,
        "blue_suppression": 1.0,
    },
    "high_contrast": {
        "stretch": "sqrt",
        "percentile_limits": (1.0, 99.0),
        "red_boost_factor": 1.2,
        "green_supression": 0.85,
        "blue_suppression": 0.85,
    },
    "poster": {
        "dpi": 300,
        "figsize": (16, 16),
        "stretch": "asinh",
        "percentile_limits": (0.3, 99.7),
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fix_tpv(header):
    """Return a copy of *header* with TPV projections replaced by TAN."""
    h = header.copy()
    if "CTYPE1" in h and "TPV" in h["CTYPE1"]:
        h["CTYPE1"] = h["CTYPE1"].replace("TPV", "TAN")
        h["CTYPE2"] = h["CTYPE2"].replace("TPV", "TAN")
    return h


def _get_header(fits_input):
    """Extract a FITS header from a filename or HDU object."""
    if isinstance(fits_input, str):
        return fits.getheader(fits_input)
    if hasattr(fits_input, "header"):
        return fits_input.header
    return None


def _load_data(fits_input):
    """Return (data_array, header_or_None) from a filename, HDU, or raw array."""
    if isinstance(fits_input, str):
        hdu = fits.open(fits_input)[0]
        return hdu.data, hdu.header
    if hasattr(fits_input, "data"):
        return fits_input.data, getattr(fits_input, "header", None)
    return fits_input, None


def _build_wcs(fits_input):
    """Build an astropy WCS (TAN) from a filename or HDU. Returns None on failure."""
    from astropy.wcs import WCS
    header = _get_header(fits_input)
    if header is None:
        return None
    return WCS(_fix_tpv(header))


def _normalize_channel(data, vmin, vmax, stretch):
    """Apply astropy stretch + normalization to a single channel."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    stretch_map = {
        "sqrt":  SqrtStretch(),
        "log":   LogStretch(),
        "asinh": AsinhStretch(),
    }
    s = stretch_map.get(stretch)
    if stretch == "log":
        vmin = max(vmin, 1e-10)
    if s is not None:
        norm = ImageNormalize(data, vmin=vmin, vmax=vmax,
                              interval=MinMaxInterval(), stretch=s)
    else:
        norm = ImageNormalize(data, vmin=vmin, vmax=vmax,
                              interval=MinMaxInterval())
    return norm(data)


def _reproject_overlay(fpath, target_header, target_shape,
                       smooth_sigma, bin_factor=8,
                       clip_sigma=3, peak_gamma=2.0):
    """Load a FITS map, reproject onto the RGB grid, smooth, and normalize."""
    from reproject import reproject_interp
    from astropy.stats import sigma_clip
    from astropy.nddata import block_reduce
    from scipy.ndimage import gaussian_filter, zoom

    with fits.open(fpath) as hdul:
        input_hdu = next(ext for ext in hdul
                         if ext.data is not None and ext.data.ndim == 2)
        reprojected, _ = reproject_interp(input_hdu, target_header,
                                          shape_out=target_shape)

    reprojected = np.nan_to_num(reprojected, nan=0.0)
    reprojected[reprojected < 0] = 0

    clipped = sigma_clip(reprojected, sigma=clip_sigma, maxiters=5, masked=True)
    clipped = clipped.filled(np.nan)

    if bin_factor > 1:
        binned = block_reduce(clipped, bin_factor, func=np.nanmean)
        binned = np.nan_to_num(binned, nan=0.0)
        zoom_factors = (target_shape[0] / binned.shape[0],
                        target_shape[1] / binned.shape[1])
        reprojected = zoom(binned, zoom_factors, order=3)

    if smooth_sigma > 0:
        reprojected = gaussian_filter(reprojected, sigma=smooth_sigma)

    pos = reprojected[reprojected > 0]
    vmin = np.percentile(pos, 1) if len(pos) else 0
    vmax = np.percentile(pos, 99.5) if len(pos) else 1
    reprojected = np.clip((reprojected - vmin) / (vmax - vmin), 0, 1)

    return np.power(reprojected, peak_gamma)


def _blend_overlay(rgb, overlay_map, color, alpha):
    """Alpha-blend a single-channel overlay map onto the RGB cube."""
    for c in range(3):
        rgb[:, :, c] = (rgb[:, :, c] * (1 - alpha * overlay_map)
                        + color[c] * alpha * overlay_map)


# ═════════════════════════════════════════════════════════════════════════════
#  Main function
# ═════════════════════════════════════════════════════════════════════════════

def make_rgb_image(
    u_fits, b_fits, g_fits,
    config=None,
    **kwargs,
):
    """
    Create an RGB composite from three FITS bands (g → R, b → G, u → B)
    with optional X-ray and convergence-map overlays.

    Parameters
    ----------
    u_fits, b_fits, g_fits : str or HDU or ndarray
        FITS files (or data) for the u, b, and g bands.
    config : RGBConfig, optional
        Configuration object. Any additional **kwargs override its values.
    **kwargs
        Any parameter from RGBConfig can be passed directly.
        See ``RGBConfig`` for the full list.

    Returns
    -------
    rgb_image : ndarray, shape (ny, nx, 3), float32, values in [0, 1]
    """

    # ── Resolve parameters: config → kwargs → defaults ───────────────────
    if config is not None:
        p = config.to_kwargs()
        p.update(kwargs)           # explicit kwargs win
    else:
        p = RGBConfig(**kwargs).to_kwargs()

    stretch          = p["stretch"]
    percentile_limits = p["percentile_limits"]
    red_boost_factor = p["red_boost_factor"]
    green_supression = p["green_supression"]
    blue_suppression = p["blue_suppression"]
    output_file      = p["output_file"]
    dpi              = p["dpi"]
    fmt              = p["format"]
    figsize          = p["figsize"]
    ra_min, ra_max   = p["ra_min"], p["ra_max"]
    dec_min, dec_max = p["dec_min"], p["dec_max"]
    xray_fits        = p["xray_fits"]
    kappa_fits       = p["kappa_fits"]
    xray_alpha       = p["xray_alpha"]
    kappa_alpha      = p["kappa_alpha"]
    xray_smooth_sigma = p["xray_smooth_sigma"]
    kappa_smooth_sigma = p["kappa_smooth_sigma"]
    xray_peak_gamma  = p["xray_peak_gamma"]
    kappa_peak_gamma = p["kappa_peak_gamma"]
    xray_bin_factor  = p["xray_bin_factor"]
    kappa_bin_factor = p["kappa_bin_factor"]
    clip_sigma_xray  = p["clip_sigma_xray"]
    clip_sigma_kappa = p["clip_sigma_kappa"]
    r_inner_kappa    = p["r_inner_kappa"]
    r_outer_kappa    = p["r_outer_kappa"]
    overlay_order    = p["overlay_order"]
    catalog          = p["catalog"]
    text             = p["text"]
    text_fontsize    = p["text_fontsize"]
    text_color       = p["text_color"]
    
    ext_kappa_fits        = p["ext_kappa_fits"]
    ext_kappa_err_fits    = p["ext_kappa_err_fits"]
    ext_kappa_smooth_sigma = p["ext_kappa_smooth_sigma"]
    ext_kappa_contour_levels = p["ext_kappa_contour_levels"]
    ext_kappa_contour_color  = p["ext_kappa_contour_color"]
    ext_kappa_show_footprint = p["ext_kappa_show_footprint"]

    # ── 1. Load band data ────────────────────────────────────────────────
    u_data, _ = _load_data(u_fits)
    b_data, _ = _load_data(b_fits)
    g_data, _ = _load_data(g_fits)

    wcs = _build_wcs(u_fits)
    if wcs is None and catalog is not None:
        print("Warning: cannot extract WCS — catalog overlay disabled.")
        catalog = None

    # ── 2. Crop to RA/Dec bounds ─────────────────────────────────────────
    if all(v is not None for v in [ra_min, ra_max, dec_min, dec_max]):
        if wcs is None:
            wcs = _build_wcs(u_fits)

        corner_ra  = [ra_min, ra_max, ra_min, ra_max]
        corner_dec = [dec_min, dec_min, dec_max, dec_max]
        x_pix, y_pix = wcs.all_world2pix(corner_ra, corner_dec, 0)

        x_lo = int(max(0, np.floor(x_pix.min())))
        x_hi = int(min(u_data.shape[1], np.ceil(x_pix.max())))
        y_lo = int(max(0, np.floor(y_pix.min())))
        y_hi = int(min(u_data.shape[0], np.ceil(y_pix.max())))

        u_data = u_data[y_lo:y_hi, x_lo:x_hi]
        b_data = b_data[y_lo:y_hi, x_lo:x_hi]
        g_data = g_data[y_lo:y_hi, x_lo:x_hi]

        wcs = wcs.deepcopy()
        wcs.wcs.crpix[0] -= x_lo
        wcs.wcs.crpix[1] -= y_lo
        print(f"Cropped to pixel region [{x_lo}:{x_hi}, {y_lo}:{y_hi}]")

    # ── 3. Build RGB cube (g→R, b→G, u→B) ───────────────────────────────
    bands = [g_data, b_data, u_data]
    channel_factors = [red_boost_factor, green_supression, blue_suppression]

    limits = []
    for data in bands:
        clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        pos = clean[clean > 0]
        limits.append(np.percentile(pos, percentile_limits) if len(pos) else (0, 1))

    rgb_image = np.zeros((*u_data.shape, 3), dtype=np.float32)
    for i, (data, (vmin, vmax)) in enumerate(zip(bands, limits)):
        rgb_image[:, :, i] = _normalize_channel(data, vmin, vmax, stretch)
        rgb_image[:, :, i] *= channel_factors[i]

    rgb_image = np.clip(rgb_image, 0, 1)

    # ── 4. Overlay X-ray and kappa maps ──────────────────────────────────
    if xray_fits is not None or kappa_fits is not None:
        if wcs is None:
            wcs = _build_wcs(u_fits)
        ref_header = wcs.to_header()

        target_header = fits.Header(ref_header)
        target_header["NAXIS1"] = rgb_image.shape[1]
        target_header["NAXIS2"] = rgb_image.shape[0]
        target_shape = (rgb_image.shape[0], rgb_image.shape[1])

        def _apply_kappa():
            if kappa_fits is None:
                return
            kappa_map = _reproject_overlay(
                kappa_fits, target_header, target_shape,
                kappa_smooth_sigma,
                bin_factor=kappa_bin_factor,
                clip_sigma=clip_sigma_kappa,
                peak_gamma=kappa_peak_gamma,
            )
            peak_y, peak_x = np.unravel_index(np.argmax(kappa_map), kappa_map.shape)
            yy, xx = np.mgrid[:kappa_map.shape[0], :kappa_map.shape[1]]
            dist = np.sqrt((xx - peak_x)**2 + (yy - peak_y)**2)
            taper = np.clip((r_outer_kappa - dist) / (r_outer_kappa - r_inner_kappa), 0, 1)
            kappa_map *= taper
            _blend_overlay(rgb_image, kappa_map,
                           color=np.array([0.2, 0.3, 1.0]), alpha=kappa_alpha)

        def _apply_xray():
            if xray_fits is None:
                return
            xray_map = _reproject_overlay(
                xray_fits, target_header, target_shape,
                xray_smooth_sigma,
                bin_factor=xray_bin_factor,
                clip_sigma=clip_sigma_xray,
                peak_gamma=xray_peak_gamma,
            )
            _blend_overlay(rgb_image, xray_map,
                           color=np.array([1.0, 0.2, 0.8]), alpha=xray_alpha)

        if overlay_order == "xray_first":
            _apply_xray()
            _apply_kappa()
        else:
            _apply_kappa()
            _apply_xray()

        rgb_image = np.clip(rgb_image, 0, 1)

    # ── 5. Final gamma stretch ───────────────────────────────────────────
    gamma = 0.6
    rgb_image = np.power(rgb_image, gamma)

    # ── 6. Plot with WCS axes ────────────────────────────────────────────
    if wcs is None:
        wcs = _build_wcs(u_fits)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_xlabel("RA", labelpad=0.6)
    ax.set_ylabel("Dec", labelpad=-1)
    ax.imshow(rgb_image, origin="lower", interpolation="nearest")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # ── External kappa footprint + contours ──────────────────────────
    if ext_kappa_fits is not None and ext_kappa_contour_levels is not None:

        from astropy.wcs import WCS as _WCS

        ext_kappa_data = fits.getdata(ext_kappa_fits)
        ext_kappa_hdr = fits.getheader(ext_kappa_fits)
        ext_wcs = _WCS(ext_kappa_hdr)

        # Smooth in pixel units: convert arcsec -> pixels
        pixel_scale_arcsec = np.abs(ext_kappa_hdr.get("CDELT1", ext_kappa_hdr.get("CD1_1", 1))) * 3600
        sigma_pix = ext_kappa_smooth_sigma / pixel_scale_arcsec

        ext_kappa_data = gaussian_filter(ext_kappa_data, sigma=sigma_pix)

        # If error map provided, contour SNR; otherwise contour kappa directly
        if ext_kappa_err_fits is not None:
            ext_err_data = fits.getdata(ext_kappa_err_fits)
            ext_err_data = gaussian_filter(ext_err_data, sigma=sigma_pix)
            contour_data = ext_kappa_data / ext_err_data
        else:
            contour_data = ext_kappa_data

        if ext_kappa_show_footprint:
            ny, nx = ext_kappa_data.shape
            # four corners in pixel coords of the ext kappa map
            corners_pix = np.array([[0, 0], [nx, 0], [nx, ny], [0, ny], [0, 0]])
            # convert to world coords
            corners_world = ext_wcs.all_pix2world(corners_pix[:, 0], corners_pix[:, 1], 0)
            # plot on the RGB axes
            ax.plot(
                corners_world[0], corners_world[1],
                color="white", linewidth=1.2, linestyle="dashed",
                transform=ax.get_transform("world"),
            )

        # Kappa contours
        ax.contour(
            contour_data, levels=ext_kappa_contour_levels,
            colors=ext_kappa_contour_color, linewidths=0.8,
            transform=ax.get_transform(ext_wcs),
        )

    # ── Text label (upper-right) ─────────────────────────────────────────
    if text is not None:
        ax.text(
            0.97, 0.97, text,
            transform=ax.transAxes,
            fontsize=text_fontsize,
            color=text_color,
            fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="black", alpha=0.5, edgecolor="none"),
        )

    # ── 7. Catalog overlay ───────────────────────────────────────────────
    if catalog is not None and wcs is not None:
        try:
            col_map = {c.lower(): c for c in catalog.columns}
            ra_col  = col_map.get("ra") or col_map.get("right_ascension")
            dec_col = col_map.get("dec") or col_map.get("declination")

            if ra_col is None or dec_col is None:
                print("Warning: 'ra'/'dec' columns not found in catalog.")
            else:
                ra_vals  = catalog[ra_col]
                dec_vals = catalog[dec_col]
                x_px, y_px = wcs.all_world2pix(ra_vals, dec_vals, 0)

                marker_radius = max(rgb_image.shape) / 1200
                ny, nx = rgb_image.shape[:2]
                for x, y in zip(x_px, y_px):
                    if 0 <= x < nx and 0 <= y < ny:
                        ax.add_patch(Circle(
                            (x, y), radius=marker_radius,
                            fill=False, edgecolor="cyan",
                            linewidth=0.1, alpha=1.0,
                        ))
                print(f"Marked {len(ra_vals)} catalog objects on the image")
        except Exception as e:
            print(f"Warning: catalog overlay failed — {e}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ── 8. Save or show ──────────────────────────────────────────────────
    if output_file:
        if "." in output_file and not fmt:
            fmt = output_file.split(".")[-1]
        if not output_file.endswith(f".{fmt}"):
            output_file = f"{output_file}.{fmt}"

        save_kw = dict(dpi=dpi, bbox_inches="tight", pad_inches=0, format=fmt)
        if fmt.lower() in ("jpg", "jpeg"):
            save_kw["quality"] = 100
        elif fmt.lower() == "tiff":
            save_kw["compression"] = "lzw"
        elif fmt.lower() == "png":
            save_kw["transparent"] = False

        plt.savefig(output_file, **save_kw)
        plt.close(fig)
        print(f"Saved {output_file} at {dpi} DPI")
    else:
        plt.tight_layout()
        plt.show()

    return rgb_image