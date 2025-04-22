import numpy as np
import matplotlib.pyplot as plt

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