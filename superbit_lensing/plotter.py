from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.wcs.utils import pixel_to_skycoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.coordinates import SkyCoord
from astropy.visualization import ZScaleInterval
import matplotlib.patches as patches
from astropy.visualization import (MinMaxInterval, SqrtStretch, 
                                  AsinhStretch, LogStretch,
                                  ImageNormalize)
from matplotlib.colors import LogNorm
from scipy.interpolate import splprep, splev
from reproject import reproject_interp
import astropy.units as u
from astropy.table import Table
import random
import os
from superbit_lensing.match import SkyCoordMatcher
import matplotlib.colors as colors
from matplotlib.path import Path
from superbit_lensing.utils import build_clean_tan_wcs, read_ds9_ctr, get_cluster_info

def plot_comparison(cat, 
                   reference_key, 
                   compare_keys, 
                   noshear_key=None, 
                   noshear_index=0,
                   error_allowed=0.01, 
                   figsize=(18, 7), 
                   save_path=None,
                   colors=['blue', 'purple'],
                   point_size=0.7,
                   point_alpha=0.5):
    """
    Create comparison plots for measurements in a catalog.
    
    Parameters:
    -----------
    cat : dict or similar
        The catalog containing the measurements
    reference_key : str
        The key to use as the reference/x-axis value
    compare_keys : list or str
        The key(s) to compare against the reference key
        If a single key is provided, only the left plot will be used
    noshear_key : str, optional
        If provided, a key for a 2D array containing no-shear measurements
    noshear_index : int, default=0
        Index to use for the noshear array if noshear_key is provided
    error_allowed : float, default=0.01
        Allowed error threshold for differences
    figsize : tuple, default=(18, 7)
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, save the figure to this path
    colors : list, default=['blue', 'purple']
        Colors to use for the scatter plots
    point_size : float, default=0.7
        Size of scatter plot points
    point_alpha : float, default=0.5
        Alpha/transparency of scatter plot points
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axes : list of matplotlib.axes.Axes
        The axes of the subplots
    fit_coeffs : dict
        Dictionary containing the fit coefficients for both plots
    """
    # Handle the case where a single comparison key is provided
    if isinstance(compare_keys, str):
        compare_keys = [compare_keys]
        
    # Determine if we need one or two subplots
    n_plots = 1 if noshear_key is None and len(compare_keys) <= 1 else 2
    
    # Calculate the range for x-axis
    x_min = np.min(cat[reference_key])
    x_max = np.max(cat[reference_key])
    
    # Create subplots
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    
    # Make sure axes is always a list/array for consistent indexing
    if n_plots == 1:
        axes = [axes]
    
    # X-axis values for smooth plotting
    x_fit = np.linspace(x_min, x_max, 500)
    
    # Dictionary to store fit coefficients
    fit_coeffs = {}
    
    # First plot - always use the first comparison key
    diff_1 = cat[compare_keys[0]] - cat[reference_key]
    
    # Quadratic fit with covariance matrix
    coeffs1_quad, cov1 = np.polyfit(cat[reference_key], diff_1, 2, cov=True)
    quad_fit1 = np.poly1d(coeffs1_quad)
    errors1 = np.sqrt(np.diag(cov1))
    
    # Fit text for first plot
    fit_text1 = (
        #f"Quadratic Fit:\n"
        f"m = {coeffs1_quad[1]:.3f} ± {errors1[1]:.3f}\n"
        f"a = {coeffs1_quad[0]:.3f} ± {errors1[0]:.3f}\n"
        f"c = {coeffs1_quad[2]:.3f} ± {errors1[2]:.3f}"
    )
    
    # Store coefficients in dictionary
    plot1_key = f"{compare_keys[0]}_vs_{reference_key}"
    fit_coeffs[plot1_key] = {
        'quadratic': coeffs1_quad,
        'errors': errors1
    }
    
    # Create first plot
    axes[0].scatter(cat[reference_key], diff_1, alpha=point_alpha, s=point_size, color=colors[0])
    axes[0].axhline(y=0, color='r', linestyle='--', label="y = 0")
    #axes[0].axhline(y=error_allowed, color='r', linestyle=':')
    #axes[0].axhline(y=-error_allowed, color='r', linestyle=':')
    axes[0].fill_between([x_min, x_max],
                     [-error_allowed, -error_allowed],
                     [error_allowed, error_allowed],
                     color='red', alpha=0.2)
    
    axes[0].plot(x_fit, quad_fit1(x_fit), color='c', lw=1.5,
             label=f"Quadratic: y = {coeffs1_quad[0]:.3f}x² + {coeffs1_quad[1]:.3f}x + {coeffs1_quad[2]:.3f}")
    
    axes[0].set_xlabel(reference_key, fontsize=14)
    axes[0].set_ylabel("Difference", fontsize=14)
    
    # Add text box to first plot
    axes[0].text(
        0.05, 0.20, fit_text1,
        transform=axes[0].transAxes,
        fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        family='monospace'
    )
    
    # Set title for first plot
    first_title = f"{compare_keys[0]} - {reference_key}"
    axes[0].set_title(first_title, fontsize=14)
        
    # Second plot - if we have a second comparison key or noshear key
    if n_plots > 1:
        # Determine what to plot in the second subplot
        if len(compare_keys) > 1:
            # Use the second comparison key
            diff_2 = cat[compare_keys[1]] - cat[reference_key]
            second_title = f"{compare_keys[1]} - {reference_key}"
            plot2_key = f"{compare_keys[1]}_vs_{reference_key}"
        else:
            # Use the noshear key
            diff_2 = cat[noshear_key][:, noshear_index] - cat[reference_key]
            second_title = f"{noshear_key}[:,{noshear_index}] - {reference_key}"
            plot2_key = f"{noshear_key}_{noshear_index}_vs_{reference_key}"
        
        # Quadratic fit for second plot
        coeffs2_quad, cov2 = np.polyfit(cat[reference_key], diff_2, 2, cov=True)
        quad_fit2 = np.poly1d(coeffs2_quad)
        errors2 = np.sqrt(np.diag(cov2))
        
        # Store coefficients
        fit_coeffs[plot2_key] = {
            'quadratic': coeffs2_quad,
            'errors': errors2
        }
        
        # Fit text for second plot
        fit_text2 = (
            #f"Quadratic Fit:\n"
            f"m = {coeffs2_quad[1]:.3f} ± {errors2[1]:.3f}\n"
            f"a = {coeffs2_quad[0]:.3f} ± {errors2[0]:.3f}\n"
            f"c = {coeffs2_quad[2]:.3f} ± {errors2[2]:.3f}"
        )
        
        # Create second plot
        axes[1].scatter(cat[reference_key], diff_2, alpha=point_alpha, s=point_size, color=colors[1])
        axes[1].axhline(y=0, color='r', linestyle='--', label="y = 0")
        #axes[1].axhline(y=error_allowed, color='r', linestyle=':')
        #axes[1].axhline(y=-error_allowed, color='r', linestyle=':')
        axes[1].fill_between([x_min, x_max],
                         [-error_allowed, -error_allowed],
                         [error_allowed, error_allowed],
                         color='red', alpha=0.2)
        
        axes[1].plot(x_fit, quad_fit2(x_fit), color='c',
                 label=f"Quadratic: y = {coeffs2_quad[0]:.3f}x² + {coeffs2_quad[1]:.3f}x + {coeffs2_quad[2]:.3f}")
        
        axes[1].set_xlabel(reference_key, fontsize=14)
        
        # Add text box to second plot
        axes[1].text(
            0.05, 0.20, fit_text2,
            transform=axes[1].transAxes,
            fontsize=14, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            family='monospace'
        )
        
        axes[1].set_title(second_title, fontsize=14)
    
    # General layout adjustments
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Return the figure, axes, and fit coefficients
    return fig, axes, fit_coeffs

def plot_gridwise_residuals_vs_truth(g1_grid, g2_grid, g1_grid_truth, g2_grid_truth, error_allowed=0.01, colors=['blue', 'purple'],
                   point_size=0.7,
                   point_alpha=0.5):
    """
    Plot residuals (measured - truth) for g1 and g2 grid values,
    with quadratic fits and annotated coefficients.

    Parameters:
    -----------
    g1_grid : 2D array
        Measured g1 values on grid
    g2_grid : 2D array
        Measured g2 values on grid
    g1_grid_truth : 2D array
        Truth/reference g1 values on grid
    g2_grid_truth : 2D array
        Truth/reference g2 values on grid
    """

    # Flatten and mask valid grid points
    mask = ~np.isnan(g1_grid) & ~np.isnan(g1_grid_truth)

    g1_diff = g1_grid[mask] - g1_grid_truth[mask]
    g2_diff = g2_grid[mask] - g2_grid_truth[mask]

    g1_truth_flat = g1_grid_truth[mask]
    g2_truth_flat = g2_grid_truth[mask]

    # Fit quadratic: y = a x² + m x + c
    coeffs1, cov1 = np.polyfit(g1_truth_flat, g1_diff, 2, cov=True)
    coeffs2, cov2 = np.polyfit(g2_truth_flat, g2_diff, 2, cov=True)

    quad_fit1 = np.poly1d(coeffs1)
    quad_fit2 = np.poly1d(coeffs2)

    errors1 = np.sqrt(np.diag(cov1))
    errors2 = np.sqrt(np.diag(cov2))

    x_fit1 = np.linspace(np.min(g1_truth_flat), np.max(g1_truth_flat), 500)
    x_fit2 = np.linspace(np.min(g2_truth_flat), np.max(g2_truth_flat), 500)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

    # g1 plot
    x_min = np.min(g1_truth_flat)
    x_max = np.max(g1_truth_flat)
    axes[0].scatter(g1_truth_flat, g1_diff, alpha=point_alpha, s=point_size, color='blue')
    axes[0].plot(x_fit1, quad_fit1(x_fit1), color='orange', lw=2, label="Quadratic Fit")
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].fill_between([x_min, x_max],
                     [-error_allowed, -error_allowed],
                     [error_allowed, error_allowed],
                     color='red', alpha=0.2)
    axes[0].set_xlabel("g1 truth", fontsize=14)
    axes[0].set_ylabel("Residual (Rinv - truth)", fontsize=14)
    axes[0].set_title("g1 Grid Residuals", fontsize=14)
    axes[0].legend()

    fit_text1 = (
        f"a = {coeffs1[0]:.3f} ± {errors1[0]:.3f}\n"
        f"m = {coeffs1[1]:.3f} ± {errors1[1]:.3f}\n"
        f"c = {coeffs1[2]:.3f} ± {errors1[2]:.3f}"
    )
    axes[0].text(
        0.05, 0.05, fit_text1,
        transform=axes[0].transAxes,
        fontsize=12, verticalalignment='bottom',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        family='monospace'
    )

    # g2 plot
    x_min = np.min(g2_truth_flat)
    x_max = np.max(g2_truth_flat)
    axes[1].scatter(g2_truth_flat, g2_diff, alpha=point_alpha, s=point_size, color='purple')
    axes[1].plot(x_fit2, quad_fit2(x_fit2), color='orange', lw=2, label="Quadratic Fit")
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].fill_between([x_min, x_max],
                         [-error_allowed, -error_allowed],
                         [error_allowed, error_allowed],
                         color='red', alpha=0.2)
    axes[1].set_xlabel("g2 truth", fontsize=14)
    axes[1].set_title("g2 Grid Residuals", fontsize=14)
    axes[1].legend()

    fit_text2 = (
        f"a = {coeffs2[0]:.3f} ± {errors2[0]:.3f}\n"
        f"m = {coeffs2[1]:.3f} ± {errors2[1]:.3f}\n"
        f"c = {coeffs2[2]:.3f} ± {errors2[2]:.3f}"
    )
    axes[1].text(
        0.05, 0.05, fit_text2,
        transform=axes[1].transAxes,
        fontsize=12, verticalalignment='bottom',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        family='monospace'
    )

    plt.suptitle("Grid-wise Shear Residuals vs Truth with Quadratic Fit", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_kappa_with_counts(kappa_file, count_file, figsize=(20, 16), plot_title='Convergence (Kappa) with Galaxy Counts', vmin=-0.1, vmax=0.1, draw_rect=True, frac=0.25**0.5):
    # Load data from FITS files
    with fits.open(count_file) as hdul:
        count_data = hdul[0].data

    with fits.open(kappa_file) as hdul:
        kappa_data = hdul[0].data

    # Get shape of count data
    ny, nx = count_data.shape

    # Compute 50% area bounds
    dx = int(nx * frac / 2)
    dy = int(ny * frac / 2)
    x_center, y_center = nx // 2, ny // 2
    x1, x2 = x_center - dx, x_center + dx
    y1, y2 = y_center - dy, y_center + dy

    # Plot kappa
    plt.figure(figsize=figsize)
    plt.imshow(kappa_data, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Kappa')

    # Overlay counts as text
    for y in range(ny):
        for x in range(nx):
            count = count_data[y, x]
            if count > 0:  # Only label non-zero counts
                plt.text(x, y, f'{int(count)}', ha='center', va='center', fontsize=12, color='white')

    # Optional: Draw rectangle for central 50% area (Uncomment to enable)
    if draw_rect:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)

    # Title and labels
    plt.title(plot_title, fontsize=12)
    plt.xlabel('X Pixel', fontsize=12)
    plt.ylabel('Y Pixel', fontsize=12)
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_kappa_with_coadd_v2(kappa_file, coadd_file, figsize=(10, 10), kappa_vmin=None, kappa_vmax=None, sample_order=1, alpha=0.5, title='Abell3411', ctr_file=None):
    # Load FITS files
    with fits.open(kappa_file) as hdul:
        kappa_data = hdul[0].data
        kappa_hdr = hdul[0].header

    # Extract data and WCS
    with fits.open(coadd_file) as hdul:
        coadd_hdr = hdul[0].header
        coadd_data = hdul[0].data
        
    coadd_wcs = build_clean_tan_wcs(coadd_hdr)
    kappa_wcs = WCS(kappa_hdr)

    center = SkyCoord(ra = kappa_wcs.wcs.crval[0], dec = kappa_wcs.wcs.crval[1], unit = (u.deg, u.deg), frame = 'icrs')
    ny, nx = kappa_data.shape
    corners = [[0, 0], [0, nx], [ny, 0], [ny, nx]]
    sky_corners = pixel_to_skycoord([y for y, x in corners], [x for y, x in corners], kappa_wcs)

    # Estimate size in pixels using WCS
    # Note: this assumes approximate scale is constant
    pixel_scale = np.abs(coadd_wcs.pixel_scale_matrix[1,1])  # deg/pix
    size_deg_x = np.max([corner.ra.degree for corner in sky_corners]) - np.min([corner.ra.degree for corner in sky_corners])
    size_deg_y = np.max([corner.dec.degree for corner in sky_corners]) - np.min([corner.dec.degree for corner in sky_corners])
    size_pix = (int(size_deg_x / pixel_scale), int(size_deg_y / pixel_scale))
    cutout = Cutout2D(data=coadd_data, position=center, size=size_pix, wcs=coadd_wcs)

    # zscale for both
    z1 = ZScaleInterval()
    vmin1, vmax1 = z1.get_limits(cutout.data)
    if kappa_vmin is None or kappa_vmax is None:
        kappa_vmin, kappa_vmax = z1.get_limits(kappa_data)
    vmin2, vmax2 = kappa_vmin, kappa_vmax

    # Normalize coadd
    coadd_norm = np.clip((cutout.data - vmin1) / (vmax1 - vmin1), 0, 1)
    
    # Use reproject to properly align kappa with coadd WCS
    kappa_reprojected, footprint = reproject_interp(
        (kappa_data, kappa_wcs),
        cutout.wcs,
        shape_out=cutout.shape,
        order=sample_order
    )

    # Fill areas where reprojection failed with extrapolated values
    # This ensures full coverage while maintaining proper alignment
    valid_mask = footprint > 0
    if not np.all(valid_mask):
        # Get the median value of valid pixels for filling
        valid_values = kappa_reprojected[valid_mask]
        fill_value = np.median(valid_values) if len(valid_values) > 0 else kappa_vmin
        
        # Fill invalid areas with this value
        kappa_reprojected[~valid_mask] = fill_value

    # Normalize kappa after reprojection
    kappa_norm = np.clip((kappa_reprojected - vmin2) / (vmax2 - vmin2), 0, 1)

    from astropy.visualization.wcsaxes import WCSAxes
    # Plot with WCSAxes
    wcs = cutout.wcs
    fig = plt.figure(figsize=(figsize))
    ax = plt.subplot(projection=wcs)
    ax.coords[0].set_format_unit('deg')

    # Coadd background
    ax.imshow(coadd_norm, origin='lower', cmap='gray')
    ax.imshow(kappa_norm, origin='lower', cmap='magma', alpha=alpha)

    if ctr_file is not None:
        contours, coord_type = read_ds9_ctr(ctr_file)
        label_added = False  # To avoid duplicate legend entries
        for contour in contours:
            contour = np.array(contour)
            if coord_type == 'fk5':
                sky = SkyCoord(ra=contour[:,0]*u.deg, dec=contour[:,1]*u.deg)
                pix = wcs.world_to_pixel(sky)
                if not label_added:
                    ax.scatter(pix[0], pix[1], color='cyan', s=1, alpha=0.5, label='X-ray contours')
                    label_added = True
                else:
                    ax.scatter(pix[0], pix[1], color='cyan', s=1, alpha=0.5, linewidth=1)
            else:
                # Pixel space contours
                if not label_added:
                    ax.scatter(contour[:,0], contour[:,1], color='cyan', linewidth=1, label='X-ray contours')
                    label_added = True
                else:
                    ax.scatter(contour[:,0], contour[:,1], color='cyan', linewidth=1)

    # Labels and ticks
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.grid(color='white', ls='dotted')
    ax.coords.grid(True, color='white', ls='dotted')
    ax.set_title(title)
    if ctr_file is not None:
        ax.legend(loc='upper right', fontsize='medium')
    plt.show()

def plot_kappa_with_coadd(kappa_file, coadd_file, figsize=(10, 10), kappa_vmin=None, kappa_vmax=None, sample_order=1, alpha=0.5, title='Abell3411', ctr_file=None):
    # Load FITS files
    with fits.open(kappa_file) as hdul:
        kappa_data = hdul[0].data
        kappa_hdr = hdul[0].header

    # Extract data and WCS
    with fits.open(coadd_file) as hdul:
        coadd_hdr = hdul[0].header
        coadd_data = hdul[0].data
        
    coadd_wcs = build_clean_tan_wcs(coadd_hdr)
    kappa_wcs = WCS(kappa_hdr)

    center = SkyCoord(ra = kappa_wcs.wcs.crval[0], dec = kappa_wcs.wcs.crval[1], unit = (u.deg, u.deg), frame = 'icrs')
    ny, nx = kappa_data.shape
    corners = [[0, 0], [0, nx], [ny, 0], [ny, nx]]
    sky_corners = pixel_to_skycoord([y for y, x in corners], [x for y, x in corners], kappa_wcs)

    # Estimate size in pixels using WCS
    # Note: this assumes approximate scale is constant
    pixel_scale = np.abs(coadd_wcs.pixel_scale_matrix[1,1])  # deg/pix
    size_deg_x = np.max([corner.ra.degree for corner in sky_corners]) - np.min([corner.ra.degree for corner in sky_corners])
    size_deg_y = np.max([corner.dec.degree for corner in sky_corners]) - np.min([corner.dec.degree for corner in sky_corners])
    size_pix = (int(size_deg_x / pixel_scale), int(size_deg_y / pixel_scale))
    cutout = Cutout2D(data=coadd_data, position=center, size=size_pix, wcs=coadd_wcs)

    # zscale for both
    z1 = ZScaleInterval()
    vmin1, vmax1 = z1.get_limits(cutout.data)
    if kappa_vmin is None or kappa_vmax is None:
        kappa_vmin, kappa_vmax = z1.get_limits(kappa_data)
    vmin2, vmax2 = kappa_vmin, kappa_vmax

    # Normalize both
    coadd_norm = np.clip((cutout.data - vmin1) / (vmax1 - vmin1), 0, 1)
    kappa_norm = np.clip((kappa_data - vmin2) / (vmax2 - vmin2), 0, 1)

    # Resize kappa to match coadd cutout shape (you can use proper reproject for real alignment)
    zoom_y = coadd_norm.shape[0] / kappa_norm.shape[0]
    zoom_x = coadd_norm.shape[1] / kappa_norm.shape[1]
    kappa_resized = zoom(kappa_norm, (zoom_y, zoom_x), order=sample_order)

    from astropy.visualization.wcsaxes import WCSAxes
    # Plot with WCSAxes
    wcs = cutout.wcs
    fig = plt.figure(figsize=(figsize))
    ax = plt.subplot(projection=wcs)
    ax.coords[0].set_format_unit('deg')

    # Coadd background
    ax.imshow(coadd_norm, origin='lower', cmap='gray')
    ax.imshow(kappa_resized, origin='lower', cmap='magma', alpha=alpha)

    if ctr_file is not None:
        contours, coord_type = read_ds9_ctr(ctr_file)
        label_added = False  # To avoid duplicate legend entries
        for contour in contours:
            contour = np.array(contour)
            if coord_type == 'fk5':
                sky = SkyCoord(ra=contour[:,0]*u.deg, dec=contour[:,1]*u.deg)
                pix = wcs.world_to_pixel(sky)
                if not label_added:
                    ax.scatter(pix[0], pix[1], color='cyan', s=1, alpha=0.5, label='X-ray contours')
                    label_added = True
                else:
                    ax.scatter(pix[0], pix[1], color='cyan', s=1, alpha=0.5, linewidth=1)
            else:
                # Pixel space contours
                if not label_added:
                    ax.scatter(contour[:,0], contour[:,1], color='cyan', linewidth=1, label='X-ray contours')
                    label_added = True
                else:
                    ax.scatter(contour[:,0], contour[:,1], color='cyan', linewidth=1)

    # Labels and ticks
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")
    ax.grid(color='white', ls='dotted')
    ax.coords.grid(True, color='white', ls='dotted')
    ax.set_title(title)
    if ctr_file is not None:
        ax.legend(loc='upper right', fontsize='medium')
    plt.show()

def make_rgb_image(u_fits, b_fits, g_fits, stretch='asinh', output_file=None, 
                   percentile_limits=(0.5, 99.5),
                   red_boost_factor=1.1, green_supression=0.9, blue_suppression=0.9,
                   dpi=600, format='png', figsize=(12, 12)):
    """
    Create an RGB image from three FITS files with enhanced red coloration for galaxies.
    
    Parameters
    ----------
    u_fits, b_fits, g_fits : str or HDUList or HDU
        FITS files for the u, b, and g bands
    stretch : str, optional
        Stretching function to apply ('linear', 'sqrt', 'log', 'asinh')
    output_file : str, optional
        If provided, save the image to this file
    percentile_limits : tuple, optional
        Lower and upper percentiles for scaling (default: 0.5, 99.5)
    red_boost_factor : float, optional
        Extra boost to the red channel (default: 1.1)
    green_suppression: float, optional
        Factor to reduce green channel (default: 0.9)
    blue_suppression : float, optional
        Factor to reduce blue channel (default: 0.9)
    dpi : int, optional
        Resolution in dots per inch for saved file (default: 600)
    format : str, optional
        File format for saved image ('png', 'tiff', 'jpg', etc.) (default: 'png')
    figsize : tuple, optional
        Figure size in inches (default: (12, 12))
    """
    # Load the data
    if isinstance(u_fits, str):
        u_data = fits.getdata(u_fits)
    else:
        u_data = u_fits
        
    if isinstance(b_fits, str):
        b_data = fits.getdata(b_fits)
    else:
        b_data = b_fits
        
    if isinstance(g_fits, str):
        g_data = fits.getdata(g_fits)
    else:
        g_data = g_fits
    
    # Create an empty RGB image array
    rgb_image = np.zeros((u_data.shape[0], u_data.shape[1], 3), dtype=np.float32)
    
    # Map data to RGB channels: g→Red, b→Green, u→Blue
    data_bands = [g_data, b_data, u_data]
    
    # Calculate lower/upper limits for scaling based on each band's data
    limits = []
    for data in data_bands:
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data_positive = data_clean[data_clean > 0]
        if len(data_positive) > 0:
            limits.append(np.percentile(data_positive, percentile_limits))
        else:
            limits.append((0, 1))
    
    # Apply stretching/normalization for each channel
    for i, (data, (vmin, vmax)) in enumerate(zip(data_bands, limits)):
        # Handle NaNs and replace with zeros
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply the selected stretch
        if stretch == 'sqrt':
            norm = ImageNormalize(data, vmin=vmin, vmax=vmax, 
                                 interval=MinMaxInterval(), stretch=SqrtStretch())
        elif stretch == 'log':
            # Ensure vmin is positive for log stretch
            vmin = max(vmin, 1e-10)
            norm = ImageNormalize(data, vmin=vmin, vmax=vmax, 
                                 interval=MinMaxInterval(), stretch=LogStretch())
        elif stretch == 'asinh':
            norm = ImageNormalize(data, vmin=vmin, vmax=vmax, 
                                 interval=MinMaxInterval(), stretch=AsinhStretch())
        else:  # linear
            norm = ImageNormalize(data, vmin=vmin, vmax=vmax, 
                                 interval=MinMaxInterval())
        
        # Normalize the data
        rgb_image[:, :, i] = norm(data)
    
    # Apply red boost and blue suppression
    rgb_image[:, :, 0] *=  red_boost_factor  # Boost red channel
    rgb_image[:, :, 1] *=  green_supression # Keep green as is
    rgb_image[:, :, 2] *=  blue_suppression  # Reduce blue channel
    
    # Clip values to the range [0, 1]
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # Create figure with high resolution
    plt.figure(figsize=figsize, dpi=dpi)
    
    # Use interpolation='nearest' for astronomical images to preserve details
    plt.imshow(rgb_image, origin='lower', interpolation='nearest')
    plt.axis('off')
    
    if output_file:
        # Determine format from filename if not explicitly provided
        if '.' in output_file and not format:
            format = output_file.split('.')[-1]
            
        # Ensure the output_file has the correct extension
        if not output_file.endswith(f'.{format}'):
            output_file = f"{output_file}.{format}"
            
        # Save with maximum quality
        if format.lower() == 'jpg' or format.lower() == 'jpeg':
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                       format=format, quality=100)
        elif format.lower() == 'tiff':
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                       format=format, compression='lzw')
        elif format.lower() == 'png':
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                       format=format, transparent=False, optimize=True)
        else:
            plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                       format=format)
            
        plt.close()
        print(f"High quality RGB image saved to {output_file} at {dpi} DPI")
    else:
        plt.tight_layout()
        plt.show()
    
    return rgb_image


class gtan_cross_1D:
    """
    Class for computing tangential and cross shear profiles from catalog data.
    
    This class bins galaxy shapes on a grid, calculates tangential/cross components,
    and produces radial profiles with error estimates.
    """
    def __init__(self, cat, g1_tag="g1_Rinv", g2_tag="g2_Rinv",
                 x_tag="X_IMAGE", y_tag="Y_IMAGE", weight_tag=None,
                 x_c=4800, y_c=3211, resolution=0.61, pix_scale=0.141):
        """
        Initialize the shear profile calculator.
        
        Parameters:
        -----------
        cat : dict-like
            Catalog containing galaxy data with shape measurements
        g1_tag, g2_tag : str
            Column names for g1 and g2 ellipticity components
        x_tag, y_tag : str
            Column names for x and y positions
        weight_tag : str, optional
            Column name for shape weights
        x_c, y_c : float
            Center coordinates for the profile
        resolution : float
            Bin size in arcminutes
        pix_scale : float
            Pixel scale in arcsec/pixel
        """        
        self.cat = cat
        self.x_tag = x_tag
        self.y_tag = y_tag
        self.x_c = x_c
        self.y_c = y_c
        self.x_col = cat[x_tag]
        self.y_col = cat[y_tag]
        self.g1_col = cat[g1_tag]
        self.g2_col = cat[g2_tag]
        self.weight_col = cat[weight_tag] if weight_tag else np.ones_like(self.g1_col)
        self.resolution = resolution
        self.pix_scale = pix_scale

        # resolution in pixels (arcmin to pixels)
        self.resolution_pix = int(np.ceil((resolution * 60) / pix_scale))

    def create_gtan_gcross_col(self):
        dx = self.x_col - self.x_c
        dy = self.y_col - self.y_c        

    def create_g1_g2_grid(self):
        # Shift coordinates relative to center
        dx = self.x_col - self.x_c
        dy = self.y_col - self.y_c

        # Calculate grid indices
        ix = np.floor(dx / self.resolution_pix).astype(int)
        iy = np.floor(dy / self.resolution_pix).astype(int)

        # Normalize indices to positive grid
        ix -= ix.min()
        iy -= iy.min()

        grid_shape = (iy.max() + 1, ix.max() + 1)

        # Initialize accumulators
        g1_grid = np.zeros(grid_shape)
        g2_grid = np.zeros(grid_shape)
        weight_grid = np.zeros(grid_shape)

        # Accumulate weighted values
        for x, y, g1, g2, w in zip(ix, iy, self.g1_col, self.g2_col, self.weight_col):
            g1_grid[y, x] += g1 * w
            g2_grid[y, x] += g2 * w
            weight_grid[y, x] += w

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            g1_grid = np.where(weight_grid > 0, g1_grid / weight_grid, np.nan)
            g2_grid = np.where(weight_grid > 0, g2_grid / weight_grid, np.nan)

        return g1_grid, g2_grid, weight_grid

    def compute_gtan_gx_grid(self, g1_grid, g2_grid):
        ny, nx = g1_grid.shape
        # Construct grid coordinate arrays (in pixels)
        y_indices = np.arange(ny)
        x_indices = np.arange(nx)
        X, Y = np.meshgrid(x_indices, y_indices)

        # Convert grid indices to physical pixel positions
        # Undo the binning shift: grid center is at (x_c, y_c)
        # Recall that each grid cell represents `resolution_pix` pixels
        X_phys = (X * self.resolution_pix) + self.x_col.min()
        Y_phys = (Y * self.resolution_pix) + self.y_col.min()

        # Vector from center to each pixel
        dx = X_phys - self.x_c
        dy = Y_phys - self.y_c
        phi = np.arctan2(dy, dx)  # angle between center and each point

        # Compute 2φ
        cos2phi = np.cos(2 * phi)
        sin2phi = np.sin(2 * phi)

        # Tangential and cross shear
        g_tan = -g1_grid * cos2phi - g2_grid * sin2phi
        g_x = g1_grid * sin2phi - g2_grid * cos2phi

        return g_tan, g_x

    def compute_radial_profile(self, g_tan, g_x, weight_grid, n_bins=20, smooth_sigma=None):
        """
        Computes 1D radial profiles of g_tan and g_x with error bars.

        Returns:
            bin_centers: arcmin
            g_tan_profile, g_tan_err: profile and error
            g_x_profile, g_x_err: profile and error
            counts: weight sum per bin
        """
        from scipy.ndimage import gaussian_filter

        if smooth_sigma is not None:
            g_tan = gaussian_filter(g_tan, sigma=smooth_sigma, mode='nearest')
            g_x = gaussian_filter(g_x, sigma=smooth_sigma, mode='nearest')
            weight_grid = gaussian_filter(weight_grid, sigma=smooth_sigma, mode='nearest')

        ny, nx = g_tan.shape
        y_indices = np.arange(ny)
        x_indices = np.arange(nx)
        X, Y = np.meshgrid(x_indices, y_indices)

        X_phys = (X * self.resolution_pix) + self.x_col.min()
        Y_phys = (Y * self.resolution_pix) + self.y_col.min()

        dx = X_phys - self.x_c
        dy = Y_phys - self.y_c
        r = np.sqrt(dx**2 + dy**2) * self.pix_scale / 60  # arcmin

        # Compute max_radius that fits a full circle in image
        max_radius = min(
            (self.x_c - self.x_col.min()) * self.pix_scale / 60,
            (self.x_col.max() - self.x_c) * self.pix_scale / 60,
            (self.y_c - self.y_col.min()) * self.pix_scale / 60,
            (self.y_col.max() - self.y_c) * self.pix_scale / 60,
        )

        # Flatten
        r_flat = r.flatten()
        g_tan_flat = g_tan.flatten()
        g_x_flat = g_x.flatten()
        w_flat = weight_grid.flatten()

        valid = np.isfinite(g_tan_flat) & np.isfinite(g_x_flat) & (w_flat > 0)
        r_flat = r_flat[valid]
        g_tan_flat = g_tan_flat[valid]
        g_x_flat = g_x_flat[valid]
        w_flat = w_flat[valid]

        # Bins
        r_max = min(r_flat.max(), max_radius)
        bins = np.linspace(0, r_max, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        g_tan_profile = np.zeros(n_bins)
        g_x_profile = np.zeros(n_bins)
        g_tan_err = np.zeros(n_bins)
        g_x_err = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for i in range(n_bins):
            in_bin = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(in_bin):
                w = w_flat[in_bin]
                gt = g_tan_flat[in_bin]
                gx = g_x_flat[in_bin]

                # Weighted means
                g_tan_profile[i] = np.average(gt, weights=w)
                g_x_profile[i] = np.average(gx, weights=w)
                counts[i] = np.sum(w)

                # Weighted standard error
                def weighted_se(x, w, mean):
                    w_sum = np.sum(w)
                    w_sq_sum = np.sum(w ** 2)
                    var = np.sum(w * (x - mean)**2)
                    denom = w_sum ** 2 - w_sq_sum
                    return np.sqrt(var / denom) if denom > 0 else np.nan

                g_tan_err[i] = weighted_se(gt, w, g_tan_profile[i])
                g_x_err[i] = weighted_se(gx, w, g_x_profile[i])
            else:
                g_tan_profile[i] = np.nan
                g_x_profile[i] = np.nan
                g_tan_err[i] = np.nan
                g_x_err[i] = np.nan
                counts[i] = 0

        return bin_centers, g_tan_profile, g_tan_err, g_x_profile, g_x_err, counts

    def calculate_shear_bias(self, g_tan, g_true, g_tan_err, r_bins=None, weights=None):
        """
        Calculate the shear bias parameter α and its uncertainty using the Cramér-Rao bound.
        
        Parameters:
        -----------
        g_tan : numpy.ndarray
            Measured tangential shear profile
        g_true : numpy.ndarray
            True tangential shear profile
        g_tan_err : numpy.ndarray
            Error on the tangential shear measurements (standard deviation)
        r_bins : numpy.ndarray, optional
            Radial bins for optional weighting (not used if weights are provided)
        weights : numpy.ndarray, optional
            Optional weights for the calculation. If None, uses inverse variance weights.
            
        Returns:
        --------
        alpha : float
            Shear bias parameter
        alpha_err : float
            Uncertainty on the shear bias parameter
        """
        # Ensure inputs are numpy arrays
        g_tan = np.asarray(g_tan)
        g_true = np.asarray(g_true)
        g_tan_err = np.asarray(g_tan_err)
        
        # Remove any NaN values
        valid = np.isfinite(g_tan) & np.isfinite(g_true) & np.isfinite(g_tan_err) & (g_tan_err > 0) & (g_true<0.06)
        g_tan = g_tan[valid]
        g_true = g_true[valid]
        g_tan_err = g_tan_err[valid]
        
        if len(g_tan) == 0:
            return np.nan, np.nan
        
        # Create the covariance matrix (diagonal with variance terms)
        # C^-1 is simply a diagonal matrix with 1/σ^2 terms
        C_inv = np.diag(1.0 / (g_tan_err**2))
        
        # Calculate g_true^T C^-1 g_tan  (numerator)
        g_true_T_C_inv_g_tan = g_true @ C_inv @ g_tan
        
        # Calculate g_true^T C^-1 g_true (denominator)
        g_true_T_C_inv_g_true = g_true @ C_inv @ g_true
        
        # Calculate α
        alpha = g_true_T_C_inv_g_tan / g_true_T_C_inv_g_true
        
        # Calculate uncertainty on α using the Cramér-Rao bound
        alpha_err = np.sqrt(1.0 / g_true_T_C_inv_g_true)
        
        return alpha, alpha_err

    def plot_shear_bias(self, r, g_tan_profile, g_tan_err, g_true_profile, counts=None, save_path=None):
        """
        Calculate and plot the shear bias and its uncertainty with the bias value printed on the plot.
        
        Parameters:
        -----------
        r : numpy.ndarray
            Radial bins
        g_tan_profile : numpy.ndarray
            Measured tangential shear profile
        g_tan_err : numpy.ndarray
            Error on the tangential shear measurements
        g_true_profile : numpy.ndarray
            True tangential shear profile
        counts : numpy.ndarray, optional
            Number of objects in each bin for error scaling
        save_path : str, optional
            Path to save the figure. If None, the figure is not saved.
            
        Returns:
        --------
        alpha : float
            Shear bias parameter
        alpha_err : float
            Uncertainty on the shear bias parameter
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Scale errors if counts are provided
        if counts is not None:
            scaled_err = g_tan_err * np.sqrt(counts)
        else:
            scaled_err = g_tan_err
        
        # Calculate shear bias and its uncertainty
        alpha, alpha_err = self.calculate_shear_bias(g_tan_profile, g_true_profile, scaled_err)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                    gridspec_kw={'height_ratios': [2, 1]})
        
        # Add a title with the shear bias value
        fig.suptitle(rf'Shear Bias: $\alpha = {alpha:.4f} \pm {alpha_err:.4f}$', 
                    fontsize=16, y=0.95)
        
        # Plot profiles in top panel
        ax1.errorbar(r, g_tan_profile, yerr=scaled_err, fmt='o-', color='blue', 
                    label=r'$g_{\mathrm{tan}}$ (measured)', capsize=4)
        ax1.fill_between(r, g_tan_profile-scaled_err, g_tan_profile+scaled_err, color='blue', alpha=0.2, zorder=1)
        ax1.plot(r, g_true_profile, 'k--', label=r'$g_{\mathrm{tan}}$ (true)')
        #ax1.plot(r, alpha * g_true_profile, 'r-', 
        #        label=rf'$\alpha \cdot g_{{true}}$')
        
        # Add a text box with the shear bias in the upper right corner
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        bias_text = rf'$\alpha = {alpha:.4f} \pm {alpha_err:.4f}$'
        ax1.text(0.95, 0.95, bias_text, transform=ax1.transAxes, fontsize=14,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax1.set_ylabel('Shear', fontsize=12)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Tangential Shear Profiles', fontsize=14)
        
        # Calculate chi residuals (measured - model) / error
        # For our case: (g_tan - α·g_true) / σ_tan
        chi_residuals = (g_tan_profile - alpha * g_true_profile) / scaled_err
        valid = g_true_profile < 0.06
        # Calculate chi-square and reduced chi-square
        chi_sq = np.sum(chi_residuals[valid]**2)
        dof = len(chi_residuals[valid]) - 1  # Degrees of freedom (number of points - number of parameters)
        reduced_chi_sq = chi_sq / dof if dof > 0 else np.nan

        # Plot chi residuals in bottom panel
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        ax2.axhline(y=1, color='grey', linestyle=':', alpha=0.5)
        ax2.axhline(y=-1, color='grey', linestyle=':', alpha=0.5)
        ax2.scatter(r, chi_residuals, color='blue', s=40)
        ax2.errorbar(r, chi_residuals, yerr= scaled_err, fmt='none', 
                    ecolor='blue', alpha=0.3, capsize=4)
        
        # Add a text box with chi-square information
        chi_text = rf'$\chi^2 = {chi_sq:.2f}$, $\chi^2/\mathrm{{dof}} = {reduced_chi_sq:.2f}$'
        ax2.text(0.95, 0.95, chi_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax2.set_ylabel(r'$\chi = (g_{\mathrm{tan}} - \alpha \cdot g_{\mathrm{true}})/\sigma$', fontsize=12)
        ax2.set_xlabel('Radius (arcmin)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Chi Residuals', fontsize=14)
        
        # Set reasonable y-limits for the residual plot
        y_max = max(3, np.max(np.abs(chi_residuals)) * 1.2)
        ax2.set_ylim(-y_max, y_max)
        
        # Tighten the layout but leave room for the overall title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return alpha, alpha_err, chi_sq

class ClusterRedSequenceAnalysis:
    """
    Class to perform red sequence analysis and cluster member identification
    for galaxy clusters.
    """
    
    def __init__(self, cluster_name, datadir=None, datafilename=None, delz=0.02):
        """
        Initialize the cluster analysis.
        
        Parameters:
        -----------
        cluster_name : str
            Name of the cluster
        datadir : str
            Base directory containing cluster data
        delz : float
            Redshift tolerance for cluster membership (default: 0.02)
        """
        if datafilename is None and datadir is None:
            raise ValueError("Either datafilename or datadir must be provided.")        
        self.cluster_name = cluster_name
        self.datadir = datadir
        self.datafilename = datafilename
        self.delz = delz
        
        # Initialize attributes that will be populated
        self.ra_center = None
        self.dec_center = None
        self.cluster_redshift = None
        self.cm_cat = None
        self.red_sequence_mask = None
        self.cluster_member_indices = None
        
    def load_data(self):
        """
        Load all necessary data for the cluster.
        """
        # Get cluster info
        self.ra_center, self.dec_center, self.cluster_redshift = get_cluster_info(self.cluster_name)
        
        # Load catalog
        if self.datafilename is not None:
            color_mag_file = self.datafilename
        else:
            color_mag_file = os.path.join(
                self.datadir, 
                f'{self.cluster_name}/sextractor_dualmode/out/{self.cluster_name}_colors_mags.fits'
            )
        
        self.cm_cat = Table.read(color_mag_file)
        
        # Filter valid detections
        valid = (self.cm_cat['FLUX_AUTO_b'] > 0) & \
                (self.cm_cat["FLUX_AUTO_g"] > 0) & \
                (self.cm_cat["FLUX_AUTO_u"] > 0)
        self.cm_cat = self.cm_cat[valid]
        
        # Extract magnitudes and colors
        self.m_b = self.cm_cat["m_b"]
        self.m_g = self.cm_cat["m_g"]
        self.color_index = self.cm_cat["color_bg"]
        
        # Process redshift data
        self._process_redshift_data()
        
    def _process_redshift_data(self):
        """Process redshift data and classify galaxies by redshift."""
        z = self.cm_cat['Z_best']
        z_matched = z[~np.isnan(z)]
        matched_data_b_ned = self.cm_cat[~np.isnan(z)]
        
        # Redshift boundaries
        self.cluster_redshift_up = self.cluster_redshift + self.delz
        self.cluster_redshift_down = self.cluster_redshift - self.delz
        
        # Classify by redshift
        high_z_indices = np.where(z_matched > self.cluster_redshift_up)[0]
        low_z_indices = np.where(z_matched <= self.cluster_redshift_down)[0]
        mid_z_indices = np.where((z_matched > self.cluster_redshift_down) & 
                                (z_matched <= self.cluster_redshift_up))[0]
        
        # Extract data for each redshift class
        self.high_z_b = matched_data_b_ned[high_z_indices]
        self.low_z_b = matched_data_b_ned[low_z_indices]
        self.mid_z_b = matched_data_b_ned[mid_z_indices]
        
        print(f"Galaxies with z > {self.cluster_redshift_up:0.2f}: {len(high_z_indices)}")
        print(f'Galaxies with {self.cluster_redshift_down:0.2f} < z ≤ {self.cluster_redshift_up:0.2f}: {len(mid_z_indices)}')
        print(f"Galaxies with z ≤ {self.cluster_redshift_down:0.2f}: {len(low_z_indices)}")
        
        # Store colors and magnitudes for each class
        self.color_index_high = self.high_z_b["color_bg"]
        self.m_b_high = self.high_z_b["m_b"]
        self.color_index_mid = self.mid_z_b["color_bg"]
        self.m_b_mid = self.mid_z_b["m_b"]
        self.color_index_low = self.low_z_b["color_bg"]
        self.m_b_low = self.low_z_b["m_b"]
        
    def compute_red_sequence(self, a=0.0, b=1.55, tolerance=0.1, resolution=0.5, sigma=1.5, save_path=None):
        """
        Compute the red sequence mask.
        
        Parameters:
        -----------
        a : float
            Slope of red sequence line
        b : float
            Intercept of red sequence line
        tolerance : float
            Tolerance for red sequence selection
        """
        self.a = a
        self.b = b
        self.tolerance = tolerance
        
        # Get RA/Dec boundaries
        self.ra_max = np.max(self.cm_cat["ra"])
        self.ra_min = np.min(self.cm_cat["ra"])
        self.dec_max = np.max(self.cm_cat["dec"])
        self.dec_min = np.min(self.cm_cat["dec"])
        self.ra_center_inverted = self.ra_max + self.ra_min - self.ra_center
        
        # Compute red sequence mask
        predicted_color = self.a * self.m_b + self.b
        self.red_sequence_mask = np.abs(self.color_index - predicted_color) < self.tolerance
        print(f"Number of objects in the red sequence: {np.sum(self.red_sequence_mask)}")
        
        # Extract red sequence galaxy positions
        self.ra_red = self.cm_cat["ra"][self.red_sequence_mask]
        self.dec_red = self.cm_cat["dec"][self.red_sequence_mask]

        self.create_density_map(resolution=resolution, sigma=sigma)
        self.plot_red_sequence_analysis(save_path=save_path)
        
    def create_density_map(self, resolution=0.5, sigma=1.5):
        """
        Create smoothed density map of red sequence galaxies.
        
        Parameters:
        -----------
        resolution : float
            Spatial resolution in arcmin
        sigma : float
            Gaussian smoothing kernel size
        """
        self.resolution = resolution
        self.sigma = sigma
        
        # Define RA and Dec boundaries
        n_bins_ra = int(np.ceil((self.ra_max - self.ra_min) * 60 / resolution))
        n_bins_dec = int(np.ceil((self.dec_max - self.dec_min) * 60 / resolution))
        
        # Create the 2D histogram
        self.hist, self.xedges, self.yedges = np.histogram2d(
            self.ra_red, self.dec_red, bins=[n_bins_ra, n_bins_dec]
        )
        
        # Apply Gaussian smoothing
        self.smoothed_hist = gaussian_filter(self.hist, sigma=sigma)
        
    def plot_red_sequence_analysis(self, save_path=None):
        """Plot the red sequence analysis with spatial distribution."""
        # Create side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # First plot: Color-Magnitude Diagram
        axes[0].scatter(self.m_b, self.color_index, s=5, alpha=0.1, color='blue', label="All Galaxies")
        axes[0].scatter(self.m_b[self.red_sequence_mask], self.color_index[self.red_sequence_mask], 
                       s=15, color='red', edgecolors='black', label="Red Sequence Galaxies")
        m_b_range = np.linspace(min(self.m_b), max(self.m_b), 100)
        axes[0].plot(m_b_range, self.a * m_b_range + self.b, color='red', linestyle='--', 
                    label=f"red-seq line, $m_b - m_g$ = {self.b}")
        axes[0].scatter(self.m_b_high, self.color_index_high, s=12, edgecolors='black', 
                       facecolors='orange', label=f'High-z (z > {self.cluster_redshift_up:.2f})')
        axes[0].scatter(self.m_b_mid, self.color_index_mid, s=12, edgecolors='black', 
                       facecolors='lime', label=f'Members ({self.cluster_redshift_down:.2f} < z ≤ {self.cluster_redshift_up:.2f})')
        axes[0].scatter(self.m_b_low, self.color_index_low, s=12, edgecolors='black', 
                       facecolors='red', label=f'Low-z (z ≤ {self.cluster_redshift_down:.2f})')
        axes[0].set_xlabel(r"$m_b$")
        axes[0].set_ylabel(r"$m_b - m_g$")
        axes[0].set_title(f"Red Sequence in {self.cluster_name}")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Second plot: Spatial distribution of red-sequence galaxies
        im = axes[1].imshow(self.smoothed_hist.T[:, ::-1], origin='lower', aspect='auto', cmap='magma',
                           extent=[self.ra_max, self.ra_min, self.dec_min, self.dec_max])
        axes[1].scatter(self.ra_center_inverted, self.dec_center, color='lime', marker='x', 
                       s=50, label="X-ray Center")
        axes[1].set_xlabel("Right Ascension (deg)")
        axes[1].set_ylabel("Declination (deg)")
        axes[1].set_title(f"Resolution: {self.resolution} arcmin, kernel: {self.sigma}")
        axes[1].legend()
        fig.colorbar(im, ax=axes[1], label="Count")
        n_red_seq = np.sum(self.red_sequence_mask)
        fig.suptitle(f"Red Sequence Evolution (b = {self.b:.1f}, {n_red_seq} galaxies selected)", 
                    fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved red sequence analysis plot to: {save_path}")
                
        plt.show()
        
    def identify_cluster_members(self, percentiles=[85, 98], sigma_smooth=2.5, save_path=None):
        """
        Identify cluster members using contour selection.
        
        Parameters:
        -----------
        percentiles : list
            Percentile levels for contour selection
        sigma_smooth : float
            Smoothing kernel for contour finding (can be different from density map)
            
        Returns:
        --------
        cluster_catalog : astropy.table.Table
            Catalog of identified cluster member galaxies
        """
        # Re-smooth if different sigma requested
        if sigma_smooth != self.sigma:
            smoothed_for_contours = gaussian_filter(self.hist, sigma=sigma_smooth)
        else:
            smoothed_for_contours = self.smoothed_hist
            
        # Find contour levels
        contour_levels = [np.percentile(smoothed_for_contours[smoothed_for_contours > 0], p) 
                         for p in percentiles]
        
        # Create meshgrid for contour finding
        X, Y = np.meshgrid(self.xedges[:-1], self.yedges[:-1])
        
        # Create figure (only spatial plot)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Spatial distribution with contours
        im = ax.imshow(smoothed_for_contours.T[:, ::-1], origin='lower', aspect='auto', cmap='magma',
                      extent=[self.ra_max, self.ra_min, self.dec_min, self.dec_max])
        
        # Draw contours
        cs = ax.contour(X, Y, smoothed_for_contours.T, levels=contour_levels, 
                       colors=['cyan', 'yellow'], linewidths=2)
        
        # Find galaxies within the highest density contour
        innermost_contour = cs.collections[-1].get_paths()[0]
        contour_polygon = Path(innermost_contour.vertices)
        
        # Check which red sequence galaxies are inside this contour
        points = np.column_stack((self.ra_red, self.dec_red))
        inside_contour = contour_polygon.contains_points(points)
        
        # Get cluster member indices
        self.cluster_member_indices = np.where(self.red_sequence_mask)[0][inside_contour]
        n_cluster_members = len(self.cluster_member_indices)
        
        # Plot cluster members
        ax.scatter(self.ra_red[inside_contour], self.dec_red[inside_contour], 
                  color='yellow', s=30, edgecolors='black', 
                  label=f"Cluster Members (n={n_cluster_members})")
        ax.scatter(self.ra_red[~inside_contour], self.dec_red[~inside_contour], 
                  color='red', s=10, alpha=0.5, label="Red Seq (field)")
        
        ax.set_xlabel("Right Ascension (deg)")
        ax.set_ylabel("Declination (deg)")
        ax.set_title(f"Cluster Members (within {percentiles[-1]}% density contour)")
        ax.legend()
        fig.colorbar(im, ax=ax, label="Count")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cluster member identification plot to: {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\nCluster Member Selection Summary:")
        print(f"================================")
        print(f"Total red sequence galaxies: {np.sum(self.red_sequence_mask)}")
        print(f"Galaxies within cluster contour: {n_cluster_members}")
        print(f"Fraction in cluster: {n_cluster_members/np.sum(self.red_sequence_mask)*100:.1f}%")
        
        # Return cluster member catalog
        cluster_catalog = self.cm_cat[self.cluster_member_indices]
        return cluster_catalog
    
    def save_cluster_catalog(self, output_path, format='fits'):
        """
        Save the cluster member catalog to disk.
        
        Parameters:
        -----------
        output_path : str
            Path where to save the catalog
        format : str
            Format for saving ('fits', 'csv', 'ascii', etc.)
        """
        if self.cluster_member_indices is None:
            print("No cluster members identified yet. Run identify_cluster_members() first.")
            return
        
        cluster_catalog = self.cm_cat[self.cluster_member_indices]
        
        # Add a column to mark these as cluster members
        cluster_catalog['is_cluster_member'] = True
        
        # Save the catalog
        cluster_catalog.write(output_path, format=format, overwrite=True)
        print(f"Saved cluster member catalog to: {output_path}")
        print(f"Total cluster members saved: {len(cluster_catalog)}")
        
    def update_original_catalog(self):
        """
        Update the original color_mag_file with cluster membership information.
        This overwrites the original file with the new columns added.
        """
        if self.cluster_member_indices is None:
            print("No cluster members identified yet. Run identify_cluster_members() first.")
            return
        
        # Path to original file
        original_file = os.path.join(self.datadir, 
                                    f'{self.cluster_name}/sextractor_dualmode/out/{self.cluster_name}_colors_mags.fits')
        
        # Create backup first
        backup_file = original_file.replace('.fits', '_backup.fits')
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(original_file, backup_file)
            print(f"Created backup: {backup_file}")
        
        # Read the original full catalog (before any filtering)
        full_catalog = Table.read(original_file)
        
        # Initialize new columns
        full_catalog['is_cluster_member'] = False
        full_catalog['is_red_sequence'] = False
        
        # Find which indices in the full catalog correspond to our filtered catalog
        # Match based on multiple columns to ensure correct identification
        for idx, obj in enumerate(self.cm_cat):
            mask = (full_catalog['ra'] == obj['ra']) & \
                   (full_catalog['dec'] == obj['dec']) & \
                   (full_catalog['id'] == obj['id'])
            
            if np.sum(mask) == 1:  # Unique match found
                full_idx = np.where(mask)[0][0]
                
                # Mark if red sequence
                if self.red_sequence_mask[idx]:
                    full_catalog['is_red_sequence'][full_idx] = True
                
                # Mark if cluster member
                if idx in self.cluster_member_indices:
                    full_catalog['is_cluster_member'][full_idx] = True
        
        # Save the updated catalog
        full_catalog.write(original_file, format='fits', overwrite=True)
        print(f"Updated original catalog: {original_file}")
        print(f"Total red sequence galaxies marked: {np.sum(full_catalog['is_red_sequence'])}")
        print(f"Total cluster members marked: {np.sum(full_catalog['is_cluster_member'])}")
        
    def get_cluster_catalog(self):
        """
        Get the catalog of identified cluster members.
        
        Returns:
        --------
        cluster_catalog : astropy.table.Table or None
            Catalog of cluster members if identified, None otherwise
        """
        if self.cluster_member_indices is not None:
            return self.cm_cat[self.cluster_member_indices]
        else:
            print("No cluster members identified yet. Run identify_cluster_members() first.")
            return None


from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path
from IPython.display import display
import ipywidgets as widgets
from IPython.display import clear_output

class InteractiveRedSequenceAnalysis:
    """
    Interactive version of ClusterRedSequenceAnalysis that allows
    real-time adjustment of red sequence parameters.
    """
    
    def __init__(self, analyzer):
        """
        Initialize with an existing ClusterRedSequenceAnalysis instance.
        
        Parameters:
        -----------
        analyzer : ClusterRedSequenceAnalysis
            An instance that has already loaded data
        """
        self.analyzer = analyzer
        self.current_b = 1.55
        self.current_a = 0.0
        self.current_tolerance = 0.1
        
    def interactive_matplotlib(self, b_range=(1.0, 2.0), tolerance_range=(0.05, 0.3)):
        """
        Create an interactive plot using matplotlib sliders.
        """
        # Initial computation
        self.analyzer.compute_red_sequence(a=self.current_a, b=self.current_b, 
                                         tolerance=self.current_tolerance)
        self.analyzer.create_density_map(resolution=0.5, sigma=1.5)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid for subplots and sliders
        gs = fig.add_gridspec(3, 2, height_ratios=[8, 1, 1], hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax_slider_b = fig.add_subplot(gs[1, :])
        ax_slider_tol = fig.add_subplot(gs[2, :])
        
        # Create sliders
        slider_b = Slider(ax_slider_b, 'Red Sequence b', b_range[0], b_range[1], 
                         valinit=self.current_b, valstep=0.01)
        slider_tol = Slider(ax_slider_tol, 'Tolerance', tolerance_range[0], 
                           tolerance_range[1], valinit=self.current_tolerance, valstep=0.01)
        
        def update(val):
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Update parameters
            self.current_b = slider_b.val
            self.current_tolerance = slider_tol.val
            
            # Recompute red sequence
            self.analyzer.compute_red_sequence(a=self.current_a, b=self.current_b, 
                                             tolerance=self.current_tolerance)
            self.analyzer.create_density_map(resolution=0.5, sigma=1.5)
            
            # Plot Color-Magnitude Diagram
            ax1.scatter(self.analyzer.m_b, self.analyzer.color_index, s=5, alpha=0.1, 
                       color='blue', label="All Galaxies")
            ax1.scatter(self.analyzer.m_b[self.analyzer.red_sequence_mask], 
                       self.analyzer.color_index[self.analyzer.red_sequence_mask], 
                       s=15, color='red', edgecolors='black', label="Red Sequence")
            
            m_b_range = np.linspace(min(self.analyzer.m_b), max(self.analyzer.m_b), 100)
            ax1.plot(m_b_range, self.current_a * m_b_range + self.current_b, 
                    color='red', linestyle='--', linewidth=2)
            
            # Show tolerance bands
            ax1.fill_between(m_b_range, 
                           self.current_a * m_b_range + self.current_b - self.current_tolerance,
                           self.current_a * m_b_range + self.current_b + self.current_tolerance,
                           color='red', alpha=0.2, label='Tolerance band')
            
            ax1.set_xlabel(r"$m_b$")
            ax1.set_ylabel(r"$m_b - m_g$")
            ax1.set_title(f"Red Sequence (b={self.current_b:.2f}, tol={self.current_tolerance:.2f})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot spatial distribution
            im = ax2.imshow(self.analyzer.smoothed_hist.T[:, ::-1], origin='lower', 
                           aspect='auto', cmap='magma',
                           extent=[self.analyzer.ra_max, self.analyzer.ra_min, 
                                  self.analyzer.dec_min, self.analyzer.dec_max])
            
            # Find and draw contours
            percentiles = [85, 95]
            contour_levels = [np.percentile(self.analyzer.smoothed_hist[self.analyzer.smoothed_hist > 0], p) 
                             for p in percentiles]
            X, Y = np.meshgrid(self.analyzer.xedges[:-1], self.analyzer.yedges[:-1])
            cs = ax2.contour(X, Y, self.analyzer.smoothed_hist.T, levels=contour_levels, 
                            colors=['cyan', 'yellow'], linewidths=2)
            
            ax2.set_xlabel("Right Ascension (deg)")
            ax2.set_ylabel("Declination (deg)")
            n_red = np.sum(self.analyzer.red_sequence_mask)
            ax2.set_title(f"Spatial Distribution ({n_red} red sequence galaxies)")
            
            plt.draw()
        
        # Connect slider to update function
        slider_b.on_changed(update)
        slider_tol.on_changed(update)
        
        # Initial plot
        update(None)
        
        plt.show()
        
    def interactive_jupyter(self, b_range=(1.0, 2.0), tolerance_range=(0.05, 0.3), 
                          percentiles=[85, 95], sigma_smooth=2.5):
        """
        Create an interactive widget for Jupyter notebooks using ipywidgets.
        """
        # Create widgets
        b_slider = widgets.FloatSlider(
            value=self.current_b,
            min=b_range[0],
            max=b_range[1],
            step=0.01,
            description='b value:',
            continuous_update=False
        )
        
        tolerance_slider = widgets.FloatSlider(
            value=self.current_tolerance,
            min=tolerance_range[0],
            max=tolerance_range[1],
            step=0.01,
            description='Tolerance:',
            continuous_update=False
        )
        
        update_button = widgets.Button(description="Identify Cluster Members")
        
        output = widgets.Output()
        
        def update_plots(change):
            with output:
                clear_output(wait=True)
                
                # Update parameters
                self.current_b = b_slider.value
                self.current_tolerance = tolerance_slider.value
                
                # Recompute
                self.analyzer.compute_red_sequence(a=self.current_a, b=self.current_b, 
                                                 tolerance=self.current_tolerance)
                self.analyzer.create_density_map(resolution=0.5, sigma=1.5)
                
                # Create plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # CMD plot
                ax1.scatter(self.analyzer.m_b, self.analyzer.color_index, s=5, alpha=0.1, 
                           color='blue', label="All Galaxies")
                ax1.scatter(self.analyzer.m_b[self.analyzer.red_sequence_mask], 
                           self.analyzer.color_index[self.analyzer.red_sequence_mask], 
                           s=15, color='red', edgecolors='black', label="Red Sequence")
                
                m_b_range = np.linspace(min(self.analyzer.m_b), max(self.analyzer.m_b), 100)
                ax1.plot(m_b_range, self.current_a * m_b_range + self.current_b, 
                        color='red', linestyle='--', linewidth=2)
                ax1.fill_between(m_b_range, 
                               self.current_a * m_b_range + self.current_b - self.current_tolerance,
                               self.current_a * m_b_range + self.current_b + self.current_tolerance,
                               color='red', alpha=0.2, label='Tolerance band')
                
                ax1.set_xlabel(r"$m_b$")
                ax1.set_ylabel(r"$m_b - m_g$")
                ax1.set_title(f"Red Sequence Selection")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Spatial plot with contours
                im = ax2.imshow(self.analyzer.smoothed_hist.T[:, ::-1], origin='lower', 
                               aspect='auto', cmap='magma',
                               extent=[self.analyzer.ra_max, self.analyzer.ra_min, 
                                      self.analyzer.dec_min, self.analyzer.dec_max])
                
                # Add contours
                contour_levels = [np.percentile(self.analyzer.smoothed_hist[self.analyzer.smoothed_hist > 0], p) 
                                 for p in percentiles]
                X, Y = np.meshgrid(self.analyzer.xedges[:-1], self.analyzer.yedges[:-1])
                cs = ax2.contour(X, Y, self.analyzer.smoothed_hist.T, levels=contour_levels, 
                                colors=['cyan', 'yellow'], linewidths=2)
                
                ax2.set_xlabel("Right Ascension (deg)")
                ax2.set_ylabel("Declination (deg)")
                n_red = np.sum(self.analyzer.red_sequence_mask)
                ax2.set_title(f"Spatial Distribution ({n_red} galaxies)")
                plt.colorbar(im, ax=ax2, label="Count")
                
                fig.suptitle(f"b = {self.current_b:.2f}, tolerance = {self.current_tolerance:.2f}")
                plt.tight_layout()
                plt.show()
                
                # Print statistics
                print(f"Red sequence galaxies: {n_red}")
                print(f"Parameters: b={self.current_b:.2f}, tolerance={self.current_tolerance:.2f}")
        
        def identify_members(b):
            with output:
                print("\n" + "="*50)
                print("Identifying cluster members with current parameters...")
                
                # Re-smooth for contour finding if needed
                smoothed_for_contours = gaussian_filter(self.analyzer.hist, sigma=sigma_smooth)
                
                # Find contour levels
                contour_levels = [np.percentile(smoothed_for_contours[smoothed_for_contours > 0], p) 
                                 for p in percentiles]
                
                # Create meshgrid
                X, Y = np.meshgrid(self.analyzer.xedges[:-1], self.analyzer.yedges[:-1])
                
                # Find contours (matplotlib contour finding)
                fig_temp, ax_temp = plt.subplots()
                cs = ax_temp.contour(X, Y, smoothed_for_contours.T, levels=contour_levels)
                plt.close(fig_temp)
                
                # Get innermost contour
                innermost_contour = cs.collections[-1].get_paths()[0]
                contour_polygon = Path(innermost_contour.vertices)
                
                # Check which galaxies are inside
                points = np.column_stack((self.analyzer.ra_red, self.analyzer.dec_red))
                inside_contour = contour_polygon.contains_points(points)
                
                cluster_member_indices = np.where(self.analyzer.red_sequence_mask)[0][inside_contour]
                n_members = len(cluster_member_indices)
                
                print(f"Cluster members identified: {n_members}")
                print(f"Fraction of red sequence: {n_members/np.sum(self.analyzer.red_sequence_mask)*100:.1f}%")
                
                # Show final plot
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(smoothed_for_contours.T[:, ::-1], origin='lower', 
                              aspect='auto', cmap='magma',
                              extent=[self.analyzer.ra_max, self.analyzer.ra_min, 
                                     self.analyzer.dec_min, self.analyzer.dec_max])
                
                cs = ax.contour(X, Y, smoothed_for_contours.T, levels=contour_levels, 
                               colors=['cyan', 'yellow'], linewidths=2)
                
                ax.scatter(self.analyzer.ra_red[inside_contour], 
                          self.analyzer.dec_red[inside_contour], 
                          color='yellow', s=30, edgecolors='black', 
                          label=f"Cluster Members (n={n_members})")
                ax.scatter(self.analyzer.ra_red[~inside_contour], 
                          self.analyzer.dec_red[~inside_contour], 
                          color='red', s=10, alpha=0.5, label="Red Seq (field)")
                
                ax.set_xlabel("Right Ascension (deg)")
                ax.set_ylabel("Declination (deg)")
                ax.set_title(f"Final Cluster Selection")
                ax.legend()
                plt.colorbar(im, ax=ax, label="Count")
                plt.show()
        
        # Connect events
        b_slider.observe(update_plots, names='value')
        tolerance_slider.observe(update_plots, names='value')
        update_button.on_click(identify_members)
        
        # Create layout
        controls = widgets.VBox([b_slider, tolerance_slider, update_button])
        
        # Initial plot
        update_plots(None)
        
        # Display
        display(controls, output)
