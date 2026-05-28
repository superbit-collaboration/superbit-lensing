import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
from astropy.table import Table

from superbit_lensing.utils import get_cluster_info, get_sky_footprint_center_radius
from superbit_lensing.smpy.utils import save_fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TARGET_LIST = os.path.join(PROJECT_ROOT, 'data', 'SuperBIT_target_list.csv')


class ClusterRedSequenceAnalysis:
    """
    Perform red sequence analysis and cluster member identification
    for galaxy clusters.
    """

    VALID_COLOR_INDICES = {
        'bg': 'color_bg',
        'ug': 'color_ug',
        'ub': 'color_ub',
    }

    COLOR_LABELS = {
        'bg': r"$b - g$",
        'ug': r"$u - g$",
        'ub': r"$u - b$",
    }

    def __init__(
        self, cluster_name, datadir=None, megafilename=None,
        datafilename=None, delz=0.02, color_index='bg', radius_th=-1,
        cluster_redshift=None, only_specz=True
    ):
        if datafilename is None and datadir is None and megafilename is None:
            raise ValueError(
                "Either datafilename, datadir, or megafilename must be provided."
            )

        if color_index not in self.VALID_COLOR_INDICES:
            raise ValueError(
                f"Invalid color index '{color_index}'. "
                f"Choose from {list(self.VALID_COLOR_INDICES)}."
            )

        self.cluster_name = cluster_name
        self.datadir = './' if (datafilename or megafilename) else datadir
        self.datafilename = datafilename
        self.megafilename = megafilename
        self.delz = delz
        self.radius_th = radius_th
        self.only_specz = only_specz
        self.color_index_col = self.VALID_COLOR_INDICES[color_index]
        self.color_index_key = color_index

        # Populated by load_data / downstream methods
        self.ra_center = None
        self.dec_center = None
        self.cluster_redshift = cluster_redshift
        self.cm_cat = None
        self.red_sequence_mask = None
        self.cluster_member_indices = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        """Load all necessary data for the cluster."""

        # Resolve the color-mag catalog path
        if self.datafilename is not None:
            self.color_mag_file = self.datafilename
        elif self.megafilename is not None:
            print("Using mega file:", self.megafilename)
            mega_cm_cat = Table.read(self.megafilename)
            cm_cat = mega_cm_cat[mega_cm_cat['CLUSTER'] == self.cluster_name]
            self.color_mag_file = os.path.join(
                self.datadir, f'{self.cluster_name}_colors_mags.fits'
            )
            cm_cat.write(self.color_mag_file, overwrite=True)
        else:
            self.color_mag_file = os.path.join(
                self.datadir, self.cluster_name,
                'sextractor_dualmode', 'out',
                f'{self.cluster_name}_colors_mags.fits',
            )

        self.plot_out_path = os.path.join(
            self.datadir, self.cluster_name, 'sextractor_dualmode', 'plots',
        )
        os.makedirs(self.plot_out_path, exist_ok=True)

        self.coadd_path = os.path.join(
            self.datadir, self.cluster_name, 'sextractor_dualmode', 'coadd',
        )

        self.cm_cat = Table.read(self.color_mag_file)
        ### check is there are keys for the color index column
        print("Color index column:", self.color_index_col)
        if self.color_index_col not in self.cm_cat.colnames:
            raise ValueError(
                f"Color index column '{self.color_index_col}' not found in catalog. "
                f"Available columns: {self.cm_cat.colnames}"
            )

        # Cluster centre & redshift
        cluster_data = pd.read_csv(TARGET_LIST)
        idx = cluster_data['SuperBIT_name'] == self.cluster_name
        if not idx.any():
            self.ra_center, self.dec_center, _ = (
                get_sky_footprint_center_radius(self.cm_cat)
            )
        else:
            self.ra_center, self.dec_center, self.cluster_redshift = (
                get_cluster_info(self.cluster_name)
            )
            
        # get the center of the focv
        self.ra_center, self.dec_center, _ = (
                get_sky_footprint_center_radius(self.cm_cat)
            )
        
        print(f"Cluster center: RA={self.ra_center:.4f} deg, Dec={self.dec_center:.4f} deg")

        # Filter valid detections
        valid = (
            (self.cm_cat['FLUX_AUTO_b'] > 0)
            & (self.cm_cat['FLUX_AUTO_g'] > 0)
            & (self.cm_cat['FLUX_AUTO_u'] > 0)
            & (self.cm_cat['R_b'] > self.radius_th)
        )
        self.cm_cat = self.cm_cat[valid]

        self.m_b = self.cm_cat['m_b']
        self.m_g = self.cm_cat['m_g']
        self.color_index = self.cm_cat[self.color_index_col]

        self._process_redshift_data()

    def _process_redshift_data(self):
        """Process redshift data and classify galaxies by redshift."""

        z = self.cm_cat['Z_best']
        z_source = self.cm_cat['Z_source']

        if self.only_specz:
            valid_z_mask = (
                ((z_source == 'DESI') | (z_source == 'NED'))
                & np.isfinite(z)
            )
        else:
            valid_z_mask = np.isfinite(z)

        z_matched = z[valid_z_mask]
        matched_data = self.cm_cat[valid_z_mask]

        # Redshift boundaries
        self.cluster_redshift_up = self.cluster_redshift + self.delz
        self.cluster_redshift_down = self.cluster_redshift - self.delz

        high_z_indices = np.where(z_matched > self.cluster_redshift_up)[0]
        low_z_indices = np.where(z_matched <= self.cluster_redshift_down)[0]
        mid_z_indices = np.where(
            (z_matched > self.cluster_redshift_down)
            & (z_matched <= self.cluster_redshift_up)
        )[0]

        self.n_high_z = len(high_z_indices)
        self.n_low_z = len(low_z_indices)
        self.n_mid_z = len(mid_z_indices)

        self.high_z_b = matched_data[high_z_indices]
        self.low_z_b = matched_data[low_z_indices]
        self.mid_z_b = matched_data[mid_z_indices]

        print(f"Galaxies with z > {self.cluster_redshift_up:.2f}: {self.n_high_z}")
        print(
            f"Galaxies with {self.cluster_redshift_down:.2f} < z "
            f"≤ {self.cluster_redshift_up:.2f}: {self.n_mid_z}"
        )
        print(f"Galaxies with z ≤ {self.cluster_redshift_down:.2f}: {self.n_low_z}")

        # Per-class colours and magnitudes
        self.color_index_high = self.high_z_b[self.color_index_col]
        self.m_b_high = self.high_z_b['m_b']
        self.color_index_mid = self.mid_z_b[self.color_index_col]
        self.m_b_mid = self.mid_z_b['m_b']
        self.color_index_low = self.low_z_b[self.color_index_col]
        self.m_b_low = self.low_z_b['m_b']

    # ------------------------------------------------------------------
    # Red sequence fitting & selection
    # ------------------------------------------------------------------

    def fit_red_sequence_line(self, sigma_clip=2., max_iter=5, fixed_intercept=True):
        """Fit a line to the known members in the colour-magnitude diagram.

        Parameters
        ----------
        sigma_clip : float
            Reject points whose residual exceeds this many standard
            deviations from the current fit.
        max_iter : int
            Maximum number of clipping iterations.
        """
        known_color = np.array(self.color_index_mid)
        known_m_b = np.array(self.m_b_mid)

        if len(known_color) < 2:
            print("Not enough known members to fit a line. Using default values.")
            self.a = 0.0
            self.b = 1.55
            return

        mask = np.ones(len(known_color), dtype=bool)

        for i in range(max_iter):
            coeffs = np.polyfit(known_m_b[mask], known_color[mask], 1)
            residuals = known_color - (coeffs[0] * known_m_b + coeffs[1])
            sigma = np.std(residuals[mask])

            new_mask = np.abs(residuals) < sigma_clip * sigma
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

            if np.sum(mask) < 2:
                print("Sigma clipping left fewer than 2 points; using last valid fit.")
                break
            
        zp = 29.146
        weights = 10 ** ((zp - known_m_b[mask]) / 2.5)

        if fixed_intercept:
            self.a = 0.0
            self.b = np.average(known_color[mask], weights=weights)
        else:
            coeffs = np.polyfit(known_m_b[mask], known_color[mask], 1, w=weights)
            self.a = coeffs[0]
            self.b = coeffs[1]
            
        n_rejected = len(known_color) - np.sum(mask)
        print(
            f"Fitted red sequence line: color = {self.a:.3f} * m_b + {self.b:.3f} "
            f"({n_rejected} outlier(s) clipped in {i + 1} iteration(s))"
        )

    def compute_red_sequence(
        self, a=None, b=None, tolerance=0.1, resolution=0.5,
        sigma=1.5, save_path=None, fixed_intercept=False, density_mode="number", m_upper_limit=None,
        m_lower_limit=None
    ):
        """
        Compute the red sequence mask.

        Parameters
        ----------
        a, b : float, optional
            Slope / intercept of the red-sequence line.  When *None* the
            values are determined via ``fit_red_sequence_line()``.
        tolerance : float
            Half-width of the colour band around the fitted line.
        resolution : float
            Spatial resolution in arcmin for the density map.
        sigma : float
            Gaussian smoothing kernel size for the density map.
        save_path : str, optional
            File path for saving the analysis plot.
        """
        self.m_lower_limit = m_lower_limit
        self.m_upper_limit = m_upper_limit
        if a is None or b is None:
            self.fit_red_sequence_line(fixed_intercept=fixed_intercept)
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        self.tolerance = tolerance

        # RA / Dec boundaries
        self.ra_max = np.max(self.cm_cat['ra'])
        self.ra_min = np.min(self.cm_cat['ra'])
        self.dec_max = np.max(self.cm_cat['dec'])
        self.dec_min = np.min(self.cm_cat['dec'])
        self.ra_center_inverted = self.ra_max + self.ra_min - self.ra_center

        # Red-sequence mask
        predicted_color = self.a * self.m_b + self.b
        self.red_sequence_mask = np.abs(self.color_index - predicted_color) < self.tolerance
        if self.m_upper_limit is not None:
            self.red_sequence_mask &= self.m_b < self.m_upper_limit
        if self.m_lower_limit is not None:
            self.red_sequence_mask &= self.m_b > self.m_lower_limit
        print(f"Number of objects in the red sequence: {np.sum(self.red_sequence_mask)}")

        self.ra_red = self.cm_cat['ra'][self.red_sequence_mask]
        self.dec_red = self.cm_cat['dec'][self.red_sequence_mask]
        self.m_b_red = self.cm_cat['m_b'][self.red_sequence_mask]

        self.create_density_map(resolution=resolution, sigma=sigma, mode=density_mode)
        self.plot_red_sequence_analysis(save_path=save_path)

    # ------------------------------------------------------------------
    # Density map
    # ------------------------------------------------------------------

    def create_density_map(self, resolution=0.5, sigma=1.5, mode="number"):
        """
        Create a smoothed density map of red-sequence galaxies.

        Parameters
        ----------
        resolution : float
            Spatial resolution in arcmin.
        sigma : float
            Gaussian smoothing kernel size in pixels.
        mode : str
            'number' for galaxy number density (count / arcmin^2),
            'luminosity' for luminosity density (flux / arcmin^2).
        """
        if mode not in ("number", "luminosity"):
            raise ValueError(f"mode must be 'number' or 'luminosity', got '{mode}'")

        self.resolution = resolution
        self.sigma = sigma
        self.density_mode = mode

        # --- Tangent-plane projection centred on the field ---
        mean_dec = 0.5 * (self.dec_min + self.dec_max)
        mean_ra  = 0.5 * (self.ra_min + self.ra_max)

        ra_proj  = (self.ra_red - mean_ra) * np.cos(np.radians(mean_dec)) * 60  # arcmin
        dec_proj = (self.dec_red - mean_dec) * 60  # arcmin

        ra_proj_min,  ra_proj_max  = ra_proj.min(),  ra_proj.max()
        dec_proj_min, dec_proj_max = dec_proj.min(), dec_proj.max()

        n_bins_ra  = int(np.ceil((ra_proj_max - ra_proj_min) / resolution))
        n_bins_dec = int(np.ceil((dec_proj_max - dec_proj_min) / resolution))

        if mode == "luminosity":
            zp = 29.146
            weights = 10 ** (-0.4 * (self.m_b_red - zp))
        else:
            weights = None

        self.hist, self.xedges, self.yedges = np.histogram2d(
            ra_proj, dec_proj,
            bins=[n_bins_ra, n_bins_dec],
            weights=weights,
        )

        # Pixel area is just resolution^2 — cos(dec) already in the projection
        pixel_area_arcmin2 = resolution ** 2

        self.density = self.hist / pixel_area_arcmin2
        self.smoothed_density = gaussian_filter(self.density, sigma=sigma)
        self.smoothed_hist = gaussian_filter(self.hist, sigma=sigma)

        # Store projection info for plotting
        self._ra_center = mean_ra
        self._dec_center = mean_dec
        
    def save_density_fits(self, save_path, smoothed=True):
        """
        Save the current density map as a FITS file with WCS.

        Parameters
        ----------
        save_path : str
            Output FITS file path.
        smoothed : bool
            If True, save the smoothed density map; otherwise the raw.
        """
        dirname = os.path.dirname(save_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        data = self.smoothed_density if smoothed else self.density

        true_boundaries = {
            "ra_min":  self.ra_min,
            "ra_max":  self.ra_max,
            "dec_min": self.dec_min,
            "dec_max": self.dec_max,
        }

        # save_fits expects (ny, nx) — our hist is (n_ra, n_dec),
        # so transpose to put Dec on axis 0 (FITS row) and RA on axis 1
        save_fits(data.T[:, ::-1], true_boundaries, save_path)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_red_sequence_analysis(self, save_path=None):
        """Plot the red-sequence analysis with spatial distribution."""

        if save_path is None:
            save_path = os.path.join(
                self.plot_out_path,
                f'{self.cluster_name}_red_sequence_analysis.png',
            )

        color_label = self.COLOR_LABELS[self.color_index_key]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        # --- Left: colour-magnitude diagram ---
        ax_cmd = axes[0]

        # Background cloud
        ax_cmd.scatter(
            self.m_b, self.color_index,
            s=5, alpha=0.08, color='#393939', label='All Galaxies',
        )

        # Selection band
        m_lo = (self.m_lower_limit if self.m_lower_limit is not None
                else float(np.min(self.m_b)))
        m_hi = (self.m_upper_limit if self.m_upper_limit is not None
                else float(np.max(self.m_b)))
        m_b_band = np.linspace(m_lo, m_hi, 100)
        line_band = self.a * m_b_band + self.b
        ax_cmd.fill_between(
            m_b_band, line_band - self.tolerance, line_band + self.tolerance,
            color='firebrick', alpha=0.18, edgecolor='firebrick',
            linewidth=0.8, label='Red Sequence Selection',
        )

        # Known members
        ax_cmd.scatter(
            self.m_b_mid, self.color_index_mid,
            s=14, edgecolors='k', linewidths=0.4,
            facecolors='#FF1A7D', label='Known Members',
        )

        ax_cmd.set_xlabel(r"$b$")
        ax_cmd.set_ylabel(rf"{color_label}")
        ax_cmd.set_ylim(bottom=-1.2, top=0.6)
        ax_cmd.set_xlim(left=16, right=24.5)

        leg = ax_cmd.legend(
            loc='best', frameon=True, fancybox=False,
            edgecolor=None, framealpha=1.0,
            handletextpad=0.4, borderpad=0.5,
            markerscale=1.3, scatterpoints=1,
        )
        leg.legend_handles[0].set_alpha(0.6)
        leg.get_frame().set_edgecolor('none')

        #leg.get_frame().set_linewidth(0.5)
        ax_cmd.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_cmd.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # --- Right: spatial density ---
        ax_spa = axes[1]

        im = ax_spa.imshow(
            self.smoothed_density.T[:, ::-1],
            origin='lower', aspect='equal', cmap='turbo',
            interpolation='bicubic',
            extent=[self.ra_max, self.ra_min, self.dec_min, self.dec_max],
        )
        ax_spa.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_spa.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_spa.set_xlabel("RA (deg)")
        ax_spa.set_ylabel("Dec (deg)")

        if self.density_mode == "luminosity":
            cbar_label = r"Luminosity density (counts arcmin$^{-2}$)"
        else:
            cbar_label = r"Number density (galaxies arcmin$^{-2}$)"

        divider = make_axes_locatable(ax_spa)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)


        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved red sequence analysis plot to: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    # Cluster member identification
    # ------------------------------------------------------------------

    def identify_cluster_members(
        self, percentiles=[85, 98], sigma_smooth=2.5, save_path=None,
    ):
        """
        Identify cluster members using contour selection.

        Parameters
        ----------
        percentiles : list
            Percentile levels for contour selection.
        sigma_smooth : float
            Smoothing kernel for contour finding (can differ from the
            density map kernel).
        save_path : str, optional
            File path for saving the plot.

        Returns
        -------
        cluster_catalog : astropy.table.Table
            Catalog of identified cluster-member galaxies.
        """
        if save_path is None:
            save_path = os.path.join(
                self.plot_out_path,
                f'{self.cluster_name}_cluster_members.png',
            )

        # Re-smooth if a different sigma is requested
        if sigma_smooth != self.sigma:
            smoothed_for_contours = gaussian_filter(self.hist, sigma=sigma_smooth)
        else:
            smoothed_for_contours = self.smoothed_hist

        contour_levels = [
            np.percentile(smoothed_for_contours[smoothed_for_contours > 0], p)
            for p in percentiles
        ]

        X, Y = np.meshgrid(self.xedges[:-1], self.yedges[:-1])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.imshow(
            smoothed_for_contours.T[:, ::-1],
            origin='lower', aspect='equal', cmap='magma',
            interpolation='bicubic',
            extent=[self.ra_max, self.ra_min, self.dec_min, self.dec_max],
        )

        cs = ax.contour(
            X, Y, smoothed_for_contours.T,
            levels=contour_levels, colors=['cyan', 'yellow'], linewidths=2,
        )

        # Galaxies inside the innermost density contour
        innermost_contour = cs.get_paths()[-1]
        contour_polygon = Path(innermost_contour.vertices)
        points = np.column_stack((self.ra_red, self.dec_red))
        inside_contour = contour_polygon.contains_points(points)

        self.cluster_member_indices = np.where(self.red_sequence_mask)[0][inside_contour]
        n_cluster_members = len(self.cluster_member_indices)

        ax.scatter(
            self.ra_red[inside_contour], self.dec_red[inside_contour],
            color='yellow', s=30, edgecolors='black',
            label=f"Cluster Members (n={n_cluster_members})",
        )
        ax.scatter(
            self.ra_red[~inside_contour], self.dec_red[~inside_contour],
            color='red', s=10, alpha=0.5, label="Red Seq (field)",
        )

        ax.set_xlabel("Right Ascension (deg)")
        ax.set_ylabel("Declination (deg)")
        ax.set_title(f"Cluster Members (within {percentiles[-1]}% density contour)")
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cluster member identification plot to: {save_path}")
        plt.show()

        # Summary
        n_red_seq = np.sum(self.red_sequence_mask)
        print(f"\nCluster Member Selection Summary:")
        print(f"================================")
        print(f"Total red sequence galaxies: {n_red_seq}")
        print(f"Galaxies within cluster contour: {n_cluster_members}")
        print(f"Fraction in cluster: {n_cluster_members / n_red_seq * 100:.1f}%")

        self.cluster_catalog = self.cm_cat[self.cluster_member_indices]
        return self.cluster_catalog

    # ------------------------------------------------------------------
    # Catalog I/O
    # ------------------------------------------------------------------

    def save_cluster_catalog(self, output_path=None, format='fits'):
        """
        Save the cluster-member catalog to disk.

        Parameters
        ----------
        output_path : str, optional
            Destination path.  Defaults to the standard pipeline location.
        format : str
            Astropy table write format ('fits', 'csv', 'ascii', …).
        """
        if self.cluster_member_indices is None:
            print("No cluster members identified yet. Run identify_cluster_members() first.")
            return

        if output_path is None:
            output_path = os.path.join(
                self.datadir, self.cluster_name,
                'sextractor_dualmode', 'out',
                f'{self.cluster_name}_coadd_redseq.fits',
            )

        cluster_catalog = self.cm_cat[self.cluster_member_indices]
        cluster_catalog['is_cluster_member'] = True
        cluster_catalog.write(output_path, format=format, overwrite=True)

        print(f"Saved cluster member catalog to: {output_path}")
        print(f"Total cluster members saved: {len(cluster_catalog)}")

    def update_original_catalog(self, file_name=None):
        """
        Update the original color_mag_file with cluster membership flags.

        Writes a new FITS file (or overwrites the given path) containing
        ``is_cluster_member`` and ``is_red_sequence`` boolean columns.
        """
        if self.cluster_member_indices is None:
            print("No cluster members identified yet. Run identify_cluster_members() first.")
            return

        if file_name is None:
            file_name = self.color_mag_file.replace('.fits', '_updated.fits')

        full_catalog = Table.read(self.color_mag_file)
        full_catalog['is_cluster_member'] = False
        full_catalog['is_red_sequence'] = False

        cluster_member_set = set(self.cluster_member_indices)

        for idx, obj in enumerate(self.cm_cat):
            mask = (
                (full_catalog['ra'] == obj['ra'])
                & (full_catalog['dec'] == obj['dec'])
                & (full_catalog['id'] == obj['id'])
            )
            if np.sum(mask) != 1:
                continue

            full_idx = np.where(mask)[0][0]

            if self.red_sequence_mask[idx]:
                full_catalog['is_red_sequence'][full_idx] = True
            if idx in cluster_member_set:
                full_catalog['is_cluster_member'][full_idx] = True

        full_catalog.write(file_name, format='fits', overwrite=True)

        print(f"Updated original catalog: {file_name}")
        print(f"Total red sequence galaxies marked: {np.sum(full_catalog['is_red_sequence'])}")
        print(f"Total cluster members marked: {np.sum(full_catalog['is_cluster_member'])}")

    def get_cluster_catalog(self):
        """Return the catalog of identified cluster members, or *None*."""
        if self.cluster_member_indices is not None:
            return self.cm_cat[self.cluster_member_indices]
        print("No cluster members identified yet. Run identify_cluster_members() first.")
        return None