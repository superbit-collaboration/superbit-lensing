"""
utils.py — shared utilities for weak-lensing convergence / SNR mapping.

Provides:
    - I/O helpers      : read_config, load_shear_data, save_fits
    - Coordinate tools : correct_RA_dec, correct_center, correct_box_boundary,
                         calculate_field_boundaries
    - Gridding         : create_shear_grid, create_count_grid
    - KS inversion     : ks_inversion, ks_inversion_list
    - Noise generation : shuffle_galaxy_rotation, generate_shuffled_shear_dfs,
                         shear_grids_for_shuffled_dfs
    - Smoothing / SNR  : compute_snr
    - Plotting         : plot_convergence
"""

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib import rc_context
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from multiprocessing import Pool

# ---------------------------------------------------------------------------
#  Plot style — set once, used everywhere via rc_context
# ---------------------------------------------------------------------------
PLOT_RC = {
    "axes.linewidth": 1.3,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.size": 8,
    "xtick.major.width": 1.3,
    "xtick.minor.visible": True,
    "xtick.minor.width": 1.0,
    "xtick.minor.size": 6,
    "xtick.direction": "in",
    "ytick.major.size": 8,
    "ytick.major.width": 1.3,
    "ytick.minor.visible": True,
    "ytick.minor.width": 1.0,
    "ytick.minor.size": 6,
    "ytick.direction": "in",
}


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------
def read_config(file_path):
    """Read a YAML configuration file and return a dict."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def load_shear_data(shear_cat_path, ra_col, dec_col, g1_col, g2_col,
                    weight_col, x_col, y_col):
    """
    Load a shear catalogue into a pandas DataFrame.

    If *weight_col* is ``None`` the ``'weight'`` column is filled with ones
    so that downstream weighted averages degrade gracefully to unweighted.
    """
    cat = Table.read(shear_cat_path)

    shear_df = pd.DataFrame({
        "ra":  np.asarray(cat[ra_col],  dtype=float),
        "dec": np.asarray(cat[dec_col], dtype=float),
        "g1":  np.asarray(cat[g1_col],  dtype=float),
        "g2":  np.asarray(cat[g2_col],  dtype=float),
        "x":   np.asarray(cat[x_col],   dtype=float),
        "y":   np.asarray(cat[y_col],   dtype=float),
    })

    if weight_col is not None:
        shear_df["weight"] = np.asarray(cat[weight_col], dtype=float)
    else:
        shear_df["weight"] = np.ones(len(cat))

    return shear_df

def find_peaks2d(image, threshold=None, ordered=True, mask=None,
                 include_border=False):
    """
    Copied from lenspack.peaks.find_peaks2d (https://github.com/CosmoStat/lenspack/blob/master/lenspack/peaks.py#L14)
    Identify peaks in an image (2D array) above a given threshold.

    A peak, or local maximum, is defined as a pixel of larger value than its
    eight neighbors. A mask may be provided to exclude certain regions from
    the search. The border is excluded by default.

    Parameters
    ----------
    image : array_like
        Two-dimensional input image.
    threshold : float, optional
        Minimum pixel amplitude to be considered as a peak. If not provided,
        the default value is set to the minimum of `image`.
    ordered : bool, optional
        If True, return peaks in decreasing order according to height.
    mask : array_like (same shape as `image`), optional
        Boolean array identifying which pixels of `image` to consider/exclude
        in finding peaks. A numerical array will be converted to binary, where
        only zero values are considered masked.
    include_border : bool, optional
        If True, include peaks found on the border of the image. Default is
        False.

    Returns
    -------
    X, Y, heights : tuple of 1D numpy arrays
        Pixel indices of peak positions and their associated heights.

    Notes
    -----
    The basic idea for this algorithm was provided by Chieh-An Lin.

    Examples
    --------
    ...

    """
    image = np.atleast_2d(image)

    # Deal with the mask first
    if mask is not None:
        mask = np.atleast_2d(mask)
        if mask.shape != image.shape:
            print("Warning: mask not compatible with image -> ignoring.")
            mask = np.ones(image.shape)
        else:
            # Make sure mask is binary, i.e. turn nonzero values into ones
            mask = mask.astype(bool).astype(float)
    else:
        mask = np.ones(image.shape)

    # Add 1 pixel padding if including border peaks
    if include_border:
        image = np.pad(image, pad_width=1, mode='constant',
                       constant_values=image.min())
        mask = np.pad(mask, pad_width=1, mode='constant', constant_values=1)

    # Determine threshold level
    if threshold is None:
        # threshold = image[mask.astype('bool')].min()
        threshold = image.min()
    else:
        threshold = max(threshold, image.min())

    # Shift everything to be positive to properly handle negative peaks
    offset = image.min()
    threshold = threshold - offset
    image = image - offset

    # Extract the center map
    map0 = image[1:-1, 1:-1]

    # Extract shifted maps
    map1 = image[0:-2, 0:-2]
    map2 = image[1:-1, 0:-2]
    map3 = image[2:, 0:-2]
    map4 = image[0:-2, 1:-1]
    map5 = image[2:, 1:-1]
    map6 = image[0:-2, 2:]
    map7 = image[1:-1, 2:]
    map8 = image[2:, 2:]

    # Compare center map with shifted maps
    merge = ((map0 > map1) & (map0 > map2) & (map0 > map3) & (map0 > map4) &
             (map0 > map5) & (map0 > map6) & (map0 > map7) & (map0 > map8))

    bordered = np.lib.pad(merge, (1, 1), 'constant', constant_values=(0, 0))
    peaksmap = image * bordered * mask
    X, Y = np.nonzero(peaksmap > threshold)

    # Extract peak heights
    heights = image[X, Y] + offset

    # Compensate for border padding
    if include_border:
        X = X - 1
        Y = Y - 1

    # Sort peaks according to height
    if ordered:
        inds = np.argsort(heights)[::-1]
        return X[inds], Y[inds], heights[inds]

    return X, Y, heights


def save_fits(data, true_boundaries, filename):
    """
    Save a 2D array as a FITS file with a TAN-projection WCS.

    Parameters
    ----------
    data : 2D ndarray
        Image data.
    true_boundaries : dict
        Keys ``ra_min``, ``ra_max``, ``dec_min``, ``dec_max`` in degrees.
    filename : str
        Output path.
    """
    ny, nx = data.shape
    ra_min  = true_boundaries["ra_min"]
    ra_max  = true_boundaries["ra_max"]
    dec_min = true_boundaries["dec_min"]
    dec_max = true_boundaries["dec_max"]

    ra_center  = (ra_max + ra_min) / 2
    dec_center = (dec_max + dec_min) / 2

    pixel_scale_dec = (dec_max - dec_min) / ny
    pixel_scale_ra  = (ra_max - ra_min) / nx * np.cos(np.deg2rad(dec_center))

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.cdelt = [-pixel_scale_ra, pixel_scale_dec]

    header = w.to_header()
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(filename, overwrite=True)
    print(f"Saved FITS file: {filename}")


# ---------------------------------------------------------------------------
#  Coordinate helpers
# ---------------------------------------------------------------------------
def calculate_field_boundaries(coord1, coord2):
    """Return ``{ra_min, ra_max, dec_min, dec_max}`` (or any two coords)."""
    return {
        "ra_min":  np.min(coord1),
        "ra_max":  np.max(coord1),
        "dec_min": np.min(coord2),
        "dec_max": np.max(coord2),
    }


def correct_RA_dec(shear_df):
    """
    Flatten RA/Dec onto a tangent plane centred on the field midpoint.

    Returns ``(corrected_df, ra_0, dec_0)``.
    """
    df = shear_df.copy()
    ra, dec = df["ra"].values, df["dec"].values

    ra_0  = (ra.max() + ra.min()) / 2
    dec_0 = (dec.max() + dec.min()) / 2

    df["ra"]  = (ra - ra_0) * np.cos(np.deg2rad(dec))
    df["dec"] = dec - dec_0
    return df, ra_0, dec_0


def correct_center(center_cl, ra_0, dec_0):
    """Project a cluster centre onto the tangent plane."""
    return {
        "ra_center":  (center_cl["ra_center"] - ra_0)
                      * np.cos(np.deg2rad(center_cl["dec_center"])),
        "dec_center": center_cl["dec_center"] - dec_0,
    }


def correct_box_boundary(box, ra_0, dec_0):
    """Project a bounding box onto the tangent plane."""
    return {
        "ra_min":  (box["ra_min"] - ra_0) * np.cos(np.deg2rad(box["dec_min"])),
        "ra_max":  (box["ra_max"] - ra_0) * np.cos(np.deg2rad(box["dec_max"])),
        "dec_min": box["dec_min"] - dec_0,
        "dec_max": box["dec_max"] - dec_0,
    }


# ---------------------------------------------------------------------------
#  Gridding
# ---------------------------------------------------------------------------


def create_shear_grid_v2(ra, dec, g1, g2, resolution, weight=None,
                      boundaries=None, verbose=False):
    """
    Bin shear components onto a 2-D pixel grid (weighted mean per pixel).

    Returns ``(g1map, g2map)`` — both transposed so that the first axis
    corresponds to Dec and the second to RA.
    """
    if boundaries is not None:
        ra_min, ra_max = boundaries["ra_min"], boundaries["ra_max"]
        dec_min, dec_max = boundaries["dec_min"], boundaries["dec_max"]
    else:
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)

    if weight is None:
        weight = np.ones_like(ra)

    npix_ra  = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    bins  = [npix_ra, npix_dec]
    rng   = [[ra_min, ra_max], [dec_min, dec_max]]

    wmap, xbins, ybins = np.histogram2d(ra, dec, bins=bins, range=rng,
                                        weights=weight)
    wmap[wmap == 0] = np.inf

    result = tuple(
        (np.histogram2d(ra, dec, bins=bins, range=rng,
                        weights=comp * weight)[0] / wmap).T
        for comp in [g1, g2]
    )

    if verbose:
        print(f"npix : [{npix_ra}, {npix_dec}]")
        print(f"extent : [{xbins[0]}, {xbins[-1]}, {ybins[0]}, {ybins[-1]}]")
        print(f"(dx, dy) : ({xbins[1] - xbins[0]}, {ybins[1] - ybins[0]})")

    return result


def create_shear_grid(ra, dec, g1, g2, resolution, weight=None,
                      boundaries=None, verbose=False):
    """
    Bin shear components onto a 2-D pixel grid (weighted mean per pixel).

    Returns ``(g1map, g2map)`` — both shaped (npix_dec, npix_ra).
    """
    if boundaries is not None:
        ra_min, ra_max = boundaries["ra_min"], boundaries["ra_max"]
        dec_min, dec_max = boundaries["dec_min"], boundaries["dec_max"]
    else:
        ra_min, ra_max = ra.min(), ra.max()
        dec_min, dec_max = dec.min(), dec.max()

    if weight is None:
        weight = np.ones_like(ra)

    npix_ra  = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))

    bins = [npix_ra, npix_dec]
    rng  = [[ra_min, ra_max], [dec_min, dec_max]]

    common = dict(x=ra, y=dec, bins=bins, range=rng, statistic="sum")

    g_sq = g1**2 + g2**2
    weight_sq = weight**2  
    
    wmap  = binned_statistic_2d(values=weight,          **common).statistic.T
    g1map = binned_statistic_2d(values=g1 * weight,     **common).statistic.T
    g2map = binned_statistic_2d(values=g2 * weight,     **common).statistic.T

    g_sq_map = binned_statistic_2d(values=g_sq * weight_sq, **common).statistic.T
    w_sq_map = binned_statistic_2d(values=weight_sq, **common).statistic.T
    w_sq_map[w_sq_map == 0] = np.inf
    g_sq_map = g_sq_map / w_sq_map
    

    wmap[wmap == 0] = np.inf
    g1map = g1map / wmap
    g2map = g2map / wmap
    # g1map = np.where(occupied, g1map / wmap, 0.0)
    # g2map = np.where(occupied, g2map / wmap, 0.0)

    if verbose:
        print(f"npix   : [{npix_ra}, {npix_dec}]")
        print(f"extent : [{ra_min}, {ra_max}, {dec_min}, {dec_max}]")
        print(f"(dx, dy) : ({(ra_max-ra_min)/npix_ra}, {(dec_max-dec_min)/npix_dec})")

    return g1map, g2map, g_sq_map, (npix_ra, npix_dec)

def create_count_grid(ra, dec, resolution, boundaries=None, verbose=False):
    """Bin galaxy positions into a 2-D count map."""
    if boundaries is not None:
        ra_min, ra_max = boundaries["ra_min"], boundaries["ra_max"]
        dec_min, dec_max = boundaries["dec_min"], boundaries["dec_max"]
    else:
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)

    npix_ra  = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    bins = [npix_ra, npix_dec]
    rng  = [[ra_min, ra_max], [dec_min, dec_max]]

    count, xbins, ybins = np.histogram2d(ra, dec, bins=bins, range=rng)

    if verbose:
        print(f"npix : [{npix_ra}, {npix_dec}]")
        print(f"extent : [{xbins[0]}, {xbins[-1]}, {ybins[0]}, {ybins[-1]}]")
        print(f"(dx, dy) : ({xbins[1] - xbins[0]}, {ybins[1] - ybins[0]})")

    return count.T


# ---------------------------------------------------------------------------
#  Kaiser-Squires inversion
# ---------------------------------------------------------------------------
def ks_inversion(g1_grid, g2_grid, key="x-y"):
    """
    Kaiser-Squires (1993) inversion for E- and B-mode convergence.

    Parameters
    ----------
    key : str
        ``'x-y'`` for pixel coordinates (g2 sign unchanged) or
        ``'ra-dec'`` for sky coordinates (g2 sign flipped).
    """
    if key == "ra-dec":
        g2_grid = -g2_grid
    elif key != "x-y":
        raise ValueError(f"Invalid key '{key}'. Use 'x-y' or 'ra-dec'.")

    npix_dec, npix_ra = g1_grid.shape

    g1_hat = np.fft.fft2(g1_grid)
    g2_hat = np.fft.fft2(g2_grid)

    k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
    k_sq = k1**2 + k2**2
    k_sq[k_sq == 0] = np.finfo(float).eps

    diff  = k1**2 - k2**2
    cross = 2 * k1 * k2

    kappa_e = np.real(np.fft.ifft2((diff * g1_hat + cross * g2_hat) / k_sq))
    kappa_b = np.real(np.fft.ifft2((diff * g2_hat - cross * g1_hat) / k_sq))
    return kappa_e, kappa_b


def ks_inversion_list(grid_list, key="x-y"):
    """
    Apply :func:`ks_inversion` to every ``(g1map, g2map)`` pair in a list.

    Returns ``(kappa_e_list, kappa_b_list)``.
    """
    kappa_e_list, kappa_b_list = [], []
    for g1map, g2map, gsq_map in grid_list:
        ke, kb = ks_inversion(g1map, g2map, key=key)
        kappa_e_list.append(ke)
        kappa_b_list.append(kb)
    return kappa_e_list, kappa_b_list


# ---------------------------------------------------------------------------
#  Noise realisations (shear randomisation)
# ---------------------------------------------------------------------------
def _shuffle_galaxy_rotation(shear_df):
    """Return a copy with per-galaxy shear orientations randomised."""
    df = shear_df.copy()
    g1, g2 = df["g1"].values, df["g2"].values

    angle = np.random.uniform(0, 2 * np.pi, len(g1))
    mag   = np.hypot(g1, g2)
    phase = np.arctan2(g2, g1) + angle

    df["g1"] = mag * np.cos(phase)
    df["g2"] = mag * np.sin(phase)
    return df


def generate_shuffled_shear_dfs(og_shear_df, num_shuffles=100, seed=42):
    """
    Generate *num_shuffles* shear catalogues with randomised orientations.

    Only the ``rotation`` strategy is currently implemented.
    """
    np.random.seed(seed)
    return [_shuffle_galaxy_rotation(og_shear_df) for _ in range(num_shuffles)]


def shear_grids_for_shuffled_dfs(list_of_dfs, coord1_col, coord2_col,
                                 resolution, boundaries=None):
    """
    Grid every shuffled catalogue and return a list of ``(g1map, g2map)``.

    *coord1_col* / *coord2_col* should be ``'ra'``/``'dec'`` or
    ``'x'``/``'y'``.
    """
    grids = []
    for df in list_of_dfs:
        g1map, g2map, gsq_map, _ = create_shear_grid(
            df[coord1_col], df[coord2_col],
            df["g1"], df["g2"],
            resolution, df["weight"],
            boundaries=boundaries,
        )
        grids.append((g1map, g2map, gsq_map))
    return grids


# ---------------------------------------------------------------------------
#  Smoothing + SNR helper
# ---------------------------------------------------------------------------
def compute_snr(og_kappa_e, shuffled_kappa_stack, kernel, flip_ra=False):
    """
    Smooth the signal and every noise realisation, then return
    ``(smoothed_signal, std_map, snr_map, smoothed_noise_stack)``.

    Parameters
    ----------
    flip_ra : bool
        If ``True`` each noise realisation is flipped along the RA axis
        (needed for the ``ra-dec`` gridding branch).
    """
    signal = gaussian_filter(og_kappa_e, kernel)

    n = shuffled_kappa_stack.shape[0]
    smoothed = np.empty_like(shuffled_kappa_stack)
    for i in range(n):
        frame = shuffled_kappa_stack[i]
        if flip_ra:
            frame = frame[:, ::-1]
        smoothed[i] = gaussian_filter(frame, kernel)

    std = np.std(smoothed, axis=0)
    snr = signal / std
    return signal, std, snr, smoothed


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def _nice_tick_step(data_range, n_ticks=5):
    """Pick a 'nice' step size from a fixed set."""
    steps = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    return steps[np.abs(data_range / n_ticks - steps).argmin()]

def _densify_polyline(ra, dec, point_spacing=0.0005):
    """
    Resample a polyline at *point_spacing* (in degree units) so that
    scatter markers appear as a continuous curve.
 
    Detects internal jumps (segments > 5× the median step) and splits
    the polyline there so no spurious straight lines are drawn across
    gaps.  Returns concatenated dense RA/Dec arrays with NaN separators
    removed.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if len(ra) < 2:
        return ra, dec
 
    dra = np.diff(ra)
    ddec = np.diff(dec)
    seg_len = np.sqrt(dra**2 + ddec**2)
 
    # Split at large jumps: whichever is smaller of
    #   3× median step length  OR  0.005 deg (~18 arcsec) absolute cap
    median_step = np.median(seg_len[seg_len > 0]) if np.any(seg_len > 0) else 1.0
    jump_threshold = min(3.0 * median_step, 0.005)
    break_idx = np.where(seg_len > jump_threshold)[0] + 1  # indices into ra/dec
 
    # Build list of sub-segments
    splits = np.split(np.arange(len(ra)), break_idx)
 
    ra_out, dec_out = [], []
    for idx in splits:
        if len(idx) < 2:
            continue
        r, d = ra[idx], dec[idx]
        dr = np.diff(r)
        dd = np.diff(d)
        sl = np.sqrt(dr**2 + dd**2)
        cum = np.concatenate(([0.0], np.cumsum(sl)))
        total = cum[-1]
        if total == 0:
            continue
        n_pts = max(int(total / point_spacing), len(r))
        t_new = np.linspace(0, total, n_pts)
        ra_out.append(np.interp(t_new, cum, r))
        dec_out.append(np.interp(t_new, cum, d))
 
    if not ra_out:
        return ra, dec
    return np.concatenate(ra_out), np.concatenate(dec_out)
 
def _draw_xray_contours(ax, xray_contours, true_boundaries, scaled_boundaries,
                         color="cyan", s=0.3):
    """
    Overlay X-ray contours on a convergence / SNR map using scatter.
 
    Each polyline is first densified (resampled at ~0.0005 deg spacing)
    so the scatter dots merge into a visually continuous curve.
    Adjust *s* (marker area in pt²) to control apparent thickness:
    0.1 = faint, 0.3 = default, 1.0 = bold.
 
    The contours arrive as a list of (ra_array, dec_array) polylines
    in true FK5 coordinates.  They are mapped to the plot's scaled
    coordinate system with the same linear interpolation used for
    tick labels.
    """
    for ra_true, dec_true in xray_contours:
        # Densify so gaps between vertices are filled
        ra_dense, dec_dense = _densify_polyline(ra_true, dec_true)
 
        # True FK5 → scaled plot coordinates
        ra_scaled = np.interp(
            ra_dense,
            [true_boundaries["ra_min"], true_boundaries["ra_max"]],
            [scaled_boundaries["ra_min"], scaled_boundaries["ra_max"]],
        )
        dec_scaled = np.interp(
            dec_dense,
            [true_boundaries["dec_min"], true_boundaries["dec_max"]],
            [scaled_boundaries["dec_min"], scaled_boundaries["dec_max"]],
        )
        ax.scatter(ra_scaled, dec_scaled,
                   s=s, c=color, marker='.', edgecolors='none', linewidths=0)

    ax.plot([], [], color=color, linewidth=1.5, label="Chandra X-ray")


def plot_convergence(convergence, scaled_boundaries, true_boundaries, config,
                     *, center_cl=None, smoothing=None, invert_map=True,
                     vmax=None, vmin=None, title=None, threshold=None,
                     box_boundary=None, save_path="output.png",
                     xray_contours=None,            # ← NEW
                     xray_contour_color="cyan",     # ← NEW
                     xray_contour_s=0.3          # ← NEW
                        ):
    """
    Plot a convergence (or SNR) map with peaks and optional bounding box.

    Returns ``(ra_peaks, dec_peaks, peak_values)``.
    """
    if smoothing is not None:
        filt = gaussian_filter(convergence, smoothing)
    else:
        filt = convergence

    ny, nx = filt.shape
    x_start, x_end = nx // 4, 3 * nx // 4
    y_start, y_end = ny // 4, 3 * ny // 4

    # --- peak detection (on the RA-flipped image) ---
    if threshold is not None:
        peaks = find_peaks2d(filt[:, ::-1], threshold=threshold,
                             include_border=False, ordered=False)
    else:
        peaks = ([], [], [])

    # Keep only central-50% peaks and shift to pixel centres
    keep = [i for i in range(len(peaks[0]))
            if y_start <= peaks[0][i] < y_end
            and x_start <= peaks[1][i] < x_end]
    py = [peaks[0][i] + 0.5 for i in keep]
    px = [peaks[1][i] + 0.5 for i in keep]
    pv = [peaks[2][i] for i in keep]
    print(f"Number of peaks: {len(py)}")

    if invert_map:
        px = [filt.shape[1] - x for x in px]

    # Convert pixel -> scaled coords
    ra_peaks = [
        scaled_boundaries["ra_min"]
        + x * (scaled_boundaries["ra_max"] - scaled_boundaries["ra_min"]) / nx
        for x in px
    ]
    dec_peaks = [
        scaled_boundaries["dec_min"]
        + y * (scaled_boundaries["dec_max"] - scaled_boundaries["dec_min"]) / ny
        for y in py
    ]

    if invert_map:
        filt = filt[:, ::-1]

    # --- plotting ---
    with rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=config["figsize"], tight_layout=True)

        extent = [
            scaled_boundaries["ra_max"], scaled_boundaries["ra_min"],
            scaled_boundaries["dec_min"], scaled_boundaries["dec_max"],
        ]

        im = ax.imshow(
            filt, cmap=config["cmap"], vmax=vmax, vmin=vmin,
            extent=extent, origin="lower",
        )

        # Peak markers
        if threshold is not None:
            for ra, dec, val in zip(ra_peaks, dec_peaks, pv):
                ax.scatter(ra, dec, s=50, facecolors="none",
                           edgecolors="g", linewidth=1.5)
                ax.text(ra - 0.002, dec + 0.002, f"{val:.2f}",
                        color="green", fontsize=10, ha="left", va="bottom")

        # Bounding box
        if box_boundary is not None:
            ra_corners = [
                box_boundary["ra_min"], box_boundary["ra_max"],
                box_boundary["ra_max"], box_boundary["ra_min"],
                box_boundary["ra_min"],
            ]
            dec_corners = [
                box_boundary["dec_min"], box_boundary["dec_min"],
                box_boundary["dec_max"], box_boundary["dec_max"],
                box_boundary["dec_min"],
            ]
            ax.plot(ra_corners, dec_corners, "w--", linewidth=2,
                    label="SuperBIT FOV")
            
#         # X-ray contours
        if xray_contours is not None:
            _draw_xray_contours(
                ax, xray_contours,
                true_boundaries, scaled_boundaries,
                color=xray_contour_color, s=xray_contour_s,
            )

        # Nice tick labels in true RA / Dec
        ra_step  = _nice_tick_step(
            true_boundaries["ra_max"] - true_boundaries["ra_min"])
        dec_step = _nice_tick_step(
            true_boundaries["dec_max"] - true_boundaries["dec_min"])

        x_ticks = np.arange(
            np.ceil(true_boundaries["ra_min"] / ra_step) * ra_step,
            np.floor(true_boundaries["ra_max"] / ra_step) * ra_step
            + ra_step / 2,
            ra_step,
        )
        y_ticks = np.arange(
            np.ceil(true_boundaries["dec_min"] / dec_step) * dec_step,
            np.floor(true_boundaries["dec_max"] / dec_step) * dec_step
            + dec_step / 2,
            dec_step,
        )

        scaled_x = np.interp(
            x_ticks,
            [true_boundaries["ra_min"], true_boundaries["ra_max"]],
            [scaled_boundaries["ra_min"], scaled_boundaries["ra_max"]],
        )
        scaled_y = np.interp(
            y_ticks,
            [true_boundaries["dec_min"], true_boundaries["dec_max"]],
            [scaled_boundaries["dec_min"], scaled_boundaries["dec_max"]],
        )

        ax.set_xticks(scaled_x)
        ax.set_yticks(scaled_y)
        ax.set_xticklabels([f"{t:.2f}" for t in x_ticks])
        ax.set_yticklabels([f"{t:.2f}" for t in y_ticks])

        ax.set_xlabel(config["xlabel"])
        ax.set_ylabel(config["ylabel"])
        ax.set_title(title)

        if box_boundary is not None:
            ax.legend(loc="upper left")
        
        if xray_contours is not None:
            ax.legend(loc="upper left")

        if config.get("gridlines", False):
            ax.grid(color="black")

        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        fig.colorbar(im, cax=cax)

        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return ra_peaks, dec_peaks, pv


def create_coord_list(npix_ra, npix_dec):
    xv, yv = np.meshgrid(np.arange(npix_ra), np.arange(npix_dec))
    coord_list = list(zip(yv.flatten(), xv.flatten() ) )
    return xv, yv, coord_list

def Schirmer_weight(r, Rs):
    x = r/Rs

    a = 6.
    b = 150.
    c = 47.
    d = 50.
    xc = 0.15

    Q =  1./(1. + np.exp(a-b*x) + np.exp(d*x-c) )
    ratio = x/xc
    safe_ratio = np.where(ratio == 0, 1.0, ratio)
    factor = np.where(ratio == 0, 1.0, np.tanh(safe_ratio) / safe_ratio)
    Q *= factor

    return Q
