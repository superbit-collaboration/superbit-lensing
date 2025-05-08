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

import astropy.units as u
from astropy.table import Table
import random
from superbit_lensing.match import SkyCoordMatcher
import matplotlib.colors as colors
from superbit_lensing.utils import build_clean_tan_wcs, read_ds9_ctr

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

def plot_kappa_with_counts(kappa_file, count_file, figsize=(20, 16), vmin=-0.1, vmax=0.1, draw_rect=True, frac=0.25**0.5):
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
    plt.title('Convergence (Kappa) with Galaxy Counts', fontsize=12)
    plt.xlabel('X Pixel', fontsize=12)
    plt.ylabel('Y Pixel', fontsize=12)
    
    # Display the plot
    plt.tight_layout()
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
