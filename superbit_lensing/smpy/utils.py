from lenspack.peaks import find_peaks2d
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import yaml
import pandas as pd


def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_field_boundaries_v2(ra, dec):
    """
    Calculate the boundaries of the field in right ascension (RA) and declination (Dec).
    
    :param ra: Dataframe column containing the right ascension values.
    :param dec: Dataframe column containing the declination values.
    :param resolution: Resolution of the map in arcminutes.
    :return: A dictionary containing the corners of the map {'ra_min', 'ra_max', 'dec_min', 'dec_max'}.
    """
    boundaries = {
        'ra_min': np.min(ra),
        'ra_max': np.max(ra),
        'dec_min': np.min(dec),
        'dec_max': np.max(dec)
    }
    
    return boundaries

def create_shear_grid(ra, dec, g1, g2, resolution, weight=None, boundaries = None, verbose=False):
    '''
    Bin values of shear data according to position on the sky with an option of not having a specified boundary.
    
    Args:
    - ra, dec, g1, g2, weight: numpy arrays of the same length containing the shear data.
    - resolution: Resolution of the map in arcminutes.
    - boundaries: Dictionary containing 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    - verbose: If True, print details of the binning.
    Returns:
    - A tuple of two 2D numpy arrays containing the binned g1 and g2 values.
    '''
    
    if boundaries is not None:
        ra_min, ra_max = boundaries['ra_min'], boundaries['ra_max']
        dec_min, dec_max = boundaries['dec_min'], boundaries['dec_max']
    else:
        ra_min, ra_max = np.min(ra), np.max(ra)
        dec_min, dec_max = np.min(dec), np.max(dec)
        
    if weight is None:
        weight = np.ones_like(ra)
    #print(ra_min, ra_max, dec_min, dec_max)
    # Calculate number of pixels based on field size and resolution
    npix_ra = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    
    #print(npix_ra, npix_dec)
    
    # Initialize the grids
    wmap, xbins, ybins = np.histogram2d(ra, dec, bins=[npix_ra, npix_dec], range=[[ra_min, ra_max], [dec_min, dec_max]],
                                            weights=weight)
    
    wmap[wmap == 0] = np.inf
    # Compute mean values per pixel
    result = tuple((np.histogram2d(ra, dec, bins=[npix_ra, npix_dec], range=[[ra_min, ra_max], [dec_min, dec_max]],
                    weights=(vv * weight))[0] / wmap).T for vv in [g1, g2])
    
    if verbose:
        print("npix : {}".format([npix_ra, npix_dec]))
        print("extent : {}".format([xbins[0], xbins[-1], ybins[0], ybins[-1]]))
        print("(dx, dy) : ({}, {})".format(xbins[1] - xbins[0],
                                           ybins[1] - ybins[0]))
        
    return result

# Function to save a FITS file
def save_fits(data, true_boundaries, filename):
    """
    Save a 2D array as a FITS file with proper WCS information.

    - data: 2D numpy array containing the map.
    - true_boundaries: Dictionary with 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    - filename: Output filename.
    """
    hdu = fits.PrimaryHDU(data)
    header = hdu.header

    ny, nx = data.shape
    ra_min, ra_max = true_boundaries['ra_min'], true_boundaries['ra_max']
    dec_min, dec_max = true_boundaries['dec_min'], true_boundaries['dec_max']

    pixel_scale_ra = (ra_max - ra_min) / nx
    pixel_scale_dec = (dec_max - dec_min) / ny

    header["CTYPE1"] = "RA---TAN"
    header["CUNIT1"] = "deg"
    header["CRVAL1"] = (ra_max + ra_min) / 2
    header["CRPIX1"] = nx / 2
    header["CD1_1"]  = -pixel_scale_ra
    header["CD1_2"]  = 0.0

    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT2"] = "deg"
    header["CRVAL2"] = (dec_max + dec_min) / 2
    header["CRPIX2"] = ny / 2
    header["CD2_1"]  = 0.0
    header["CD2_2"]  = pixel_scale_dec

    hdu.writeto(filename, overwrite=True)
    print(f"Saved FITS file: {filename}")

def plot_convergence(convergence, scaled_boundaries, true_boundaries, config,  output_name="Converenge map", center_cl=None, smoothing=None, invert_map=True, vmax=None, vmin=None, title=None, threshold = None, con_peaks=None, box_boundary=None, save_path="output.png"):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    
    if smoothing is not None:
        filtered_convergence = gaussian_filter(convergence, smoothing)
    else:
        filtered_convergence = convergence
    
    # Determine the central 50% area
    ny, nx = filtered_convergence.shape
    x_start, x_end = nx // 4, 3 * nx // 4
    y_start, y_end = ny // 4, 3 * ny // 4
   
    peaks = find_peaks2d(filtered_convergence[:,::-1], threshold=threshold, include_border=False, ordered=False) if threshold is not None else ([], [], [])
    
    # Find peaks which are in the central 50% area
    filtered_indices = [i for i in range(len(peaks[0])) if y_start <= peaks[0][i] < y_end and x_start <= peaks[1][i] < x_end]
    peaks = ([peaks[0][i] for i in filtered_indices], [peaks[1][i] for i in filtered_indices], [peaks[2][i] for i in filtered_indices])
    
    # find the center of the peaks by adding 0.5 with every pixel
    peaks = ([x+0.5 for x in peaks[0]], [y+0.5 for y in peaks[1]], peaks[2])
    
    print(f"Number of peaks: {len(peaks[0])}")
    
    if invert_map:
        #peaks = find_peaks2d(filtered_convergence, threshold=threshold, include_border=False) if threshold is not None else ([], [], [])
        xcr = []
        for x in peaks[1]:
            xcr.append(filtered_convergence.shape[1] - x)
        peaks = (peaks[0], xcr, peaks[2])
    #else:
#        peaks = find_peaks2d(filtered_convergence[:,::-1], threshold=threshold, include_border=False) if threshold is not None else ([], [], [])
    #    peaks = ([x for x in peaks[0]], [y-1.0 for y in peaks[1]], peaks[2])
    ra_peaks = [scaled_boundaries['ra_min'] + (x) * (scaled_boundaries['ra_max'] - scaled_boundaries['ra_min']) / filtered_convergence.shape[1] for x in peaks[1]]
    dec_peaks = [scaled_boundaries['dec_min'] + (y) * (scaled_boundaries['dec_max'] - scaled_boundaries['dec_min']) / filtered_convergence.shape[0] for y in peaks[0]]        
    if invert_map:
        filtered_convergence = filtered_convergence[:, ::-1]
        
    # Find peaks of convergence


    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    if threshold is not None:
        for ra, dec, peak_value in zip(ra_peaks, dec_peaks, peaks[2]):
            ax.scatter(ra, dec, s=50, facecolors='none', edgecolors='g', linewidth=1.5)
            ax.text(
            ra - 0.002, dec + 0.002,  # Adjust these offsets as needed
            f"{peak_value:.2f}",
            color='green',
            fontsize=10,
            ha='left',
            va='bottom'
        )
    extent = [scaled_boundaries['ra_max'], 
                scaled_boundaries['ra_min'], 
                scaled_boundaries['dec_min'], 
                scaled_boundaries['dec_max']]
    
    #extent = [scaled_boundaries['ra_min'], scaled_boundaries['ra_max'], scaled_boundaries['dec_min'], scaled_boundaries['dec_max']]
    
    im = ax.imshow(
        filtered_convergence, 
        cmap=config['cmap'],
        vmax=vmax, 
        vmin=vmin,
        extent=extent,
        origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
    )
    
    # Mark cluster center if specified
    cluster_center = center_cl
    ra_center = None
    dec_center = None
    
    if cluster_center == 'auto':
        ra_center = (scaled_boundaries['ra_max'] + scaled_boundaries['ra_min']) / 2
        dec_center = (scaled_boundaries['dec_max'] + scaled_boundaries['dec_min']) / 2
    elif isinstance(cluster_center, dict):
        ra_center = cluster_center['ra_center']
        dec_center = cluster_center['dec_center']
    elif cluster_center is not None:
        print("Unrecognized cluster_center format, skipping marker.")
        ra_center = dec_center = None

    if ra_center is not None:
        if invert_map:
            ra_center =  (scaled_boundaries['ra_max'] - np.array(ra_center)) + scaled_boundaries['ra_min']
        ax.scatter(ra_center, dec_center, marker='x', color='lime', s=50, label='FOV Center')
        #ax.axhline(y=dec_center, color='w', linestyle='--')
        #ax.axvline(x=ra_center, color='w', linestyle='--')
        
    if box_boundary is not None:
        ra_box_max = box_boundary['ra_max']
        ra_box_min = box_boundary['ra_min']
        dec_box_max = box_boundary['dec_max']
        dec_box_min = box_boundary['dec_min']
        ra_corners = [ra_box_min, ra_box_max, ra_box_max, ra_box_min, ra_box_min]
        dec_corners = [dec_box_min, dec_box_min, dec_box_max, dec_box_max, dec_box_min]
        ax.plot(ra_corners, dec_corners, color='w', linestyle='--', linewidth=2, label="SuperBIT FOV")


    # Determine nice step sizes based on the range
    ra_range = true_boundaries['ra_max'] - true_boundaries['ra_min']
    dec_range = true_boundaries['dec_max'] - true_boundaries['dec_min']

    # Choose step size (0.01, 0.05, 0.1, 0.25, 0.5) based on range size
    possible_steps = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    ra_step = possible_steps[np.abs(ra_range/5 - possible_steps).argmin()]
    dec_step = possible_steps[np.abs(dec_range/5 - possible_steps).argmin()]

    # Generate ticks
    x_ticks = np.arange(np.ceil(true_boundaries['ra_min']/ra_step)*ra_step,
                        np.floor(true_boundaries['ra_max']/ra_step)*ra_step + ra_step/2,
                        ra_step)
    y_ticks = np.arange(np.ceil(true_boundaries['dec_min']/dec_step)*dec_step,
                        np.floor(true_boundaries['dec_max']/dec_step)*dec_step + dec_step/2,
                        dec_step)

    # Convert to scaled coordinates
    scaled_x_ticks = np.interp(x_ticks, 
                            [true_boundaries['ra_min'], true_boundaries['ra_max']], 
                            [scaled_boundaries['ra_min'], scaled_boundaries['ra_max']])
    scaled_y_ticks = np.interp(y_ticks, 
                            [true_boundaries['dec_min'], true_boundaries['dec_max']], 
                            [scaled_boundaries['dec_min'], scaled_boundaries['dec_max']])

    # Set the ticks
    ax.set_xticks(scaled_x_ticks)
    ax.set_yticks(scaled_y_ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
      
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(title)
    ax.legend(loc='upper left')

    # Is there a better way to force something to be a boolean?
    if config['gridlines'] == True:
        ax.grid(color='black')

    # Add colorbar; turn off minor axes first
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)

    # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
    fig.tight_layout() 
    #plt.show(block=True)
#    plt.show()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return ra_peaks, dec_peaks, peaks[2]

# Function to load shear data
def load_shear_data(shear_cat_path, ra_col, dec_col, g1_col, g2_col, weight_col, x_col, y_col):
    shear_catalog = Table.read(shear_cat_path)
    
    shear_df = pd.DataFrame({
        'ra': shear_catalog[ra_col],
        'dec': shear_catalog[dec_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
        'x': shear_catalog[x_col],
        'y': shear_catalog[y_col],
    })
    
    if weight_col is not None:
        shear_df['weight'] = shear_catalog[weight_col]
    else:
        shear_df['weight'] = None  # or np.nan if you prefer numerical NaN values
    
    return shear_df

# Function to correct cluster center
def correct_center(center_cl, ra_0, dec_0):
    center_c = {}
    center_c["ra_center"] = (center_cl["ra_center"] - ra_0) * np.cos(np.deg2rad(center_cl["dec_center"]))
    center_c["dec_center"] = center_cl["dec_center"] - dec_0
    return center_c

def correct_box_boundary(box_boundary, ra_0, dec_0):
    box_boundary_c = {}
    box_boundary_c["ra_min"] = (box_boundary["ra_min"] - ra_0) * np.cos(np.deg2rad(box_boundary["dec_min"]))
    box_boundary_c["ra_max"] = (box_boundary["ra_max"] - ra_0) * np.cos(np.deg2rad(box_boundary["dec_max"]))
    box_boundary_c["dec_min"] = box_boundary["dec_min"] - dec_0
    box_boundary_c["dec_max"] = box_boundary["dec_max"] - dec_0
    return box_boundary_c

def ks_inversion(g1_grid, g2_grid, key="x-y"):
    """
    Perform the Kaiser-Squires inversion to obtain both E-mode and B-mode convergence maps from shear components.
    """
    # Get the dimensions of the input grids
    npix_dec, npix_ra = g1_grid.shape
    
    if key == "x-y":
        g2_grid = g2_grid
    elif key == "ra-dec":
        g2_grid = -g2_grid
    else:    
        raise ValueError("Invalid key. Must be either 'x-y' or 'ra-dec'.")

    # Fourier transform the shear components
    g1_hat = np.fft.fft2(g1_grid)
    g2_hat = np.fft.fft2(g2_grid)

    # Create a grid of wave numbers
    k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
    k_squared = k1**2 + k2**2

    # Avoid division by zero by replacing zero values with a small number
    k_squared = np.where(k_squared == 0, np.finfo(float).eps, k_squared)

    # Kaiser-Squires inversion in Fourier space
    kappa_e_hat = (1 / k_squared) * ((k1**2 - k2**2) * g1_hat + 2 * k1 * k2 * g2_hat)
    kappa_b_hat = (1 / k_squared) * ((k1**2 - k2**2) * g2_hat - 2 * k1 * k2 * g1_hat)

    # Inverse Fourier transform to get the convergence maps
    kappa_e_grid = np.real(np.fft.ifft2(kappa_e_hat))
    kappa_b_grid = np.real(np.fft.ifft2(kappa_b_hat))

    return kappa_e_grid, kappa_b_grid