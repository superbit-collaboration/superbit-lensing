"""
Self-contained aperture mass S/N calculator.

Usage
-----
    snr = ApertureMassSNR(g1, g2, weight, x, y, bin_size=200., Rs=4500.)
    SNR_E, SNR_B = snr.run(n_realizations=200, n_cpus=8)
    Author: Shenming Fu, KIPAC (shenming.fu.astro@gmail.com)
"""

import numpy as np
from scipy import stats
from multiprocessing import Pool
from scipy.signal import fftconvolve


# ═══════════════════════════════════════════════════════════════════
# Schirmer filter (module-level for pickle)
# ═══════════════════════════════════════════════════════════════════

def _schirmer_weight(r, Rs):
    x = r / Rs
    a, b, c, d, xc = 6.0, 150.0, 47.0, 50.0, 0.15
    Q = 1.0 / (1.0 + np.exp(a - b * x) + np.exp(d * x - c))
    ratio = x / xc
    safe = np.where(ratio == 0, 1.0, ratio)
    Q *= np.where(ratio == 0, 1.0, np.tanh(safe) / safe)
    return Q


# ═══════════════════════════════════════════════════════════════════
# Module-level worker for Pool.map (pickle requirement)
# ═══════════════════════════════════════════════════════════════════

def _run_one_realization(args):
    """Compute E-mode aperture mass for a single noise realization."""
    e_amp, weight, x, y, x_bin, y_bin, nrow, ncol, Rs, seed = args

    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2 * np.pi, size=len(e_amp))
    e1_r = e_amp * np.cos(2 * phi)
    e2_r = e_amp * np.sin(2 * phi)

    e1_b  = _bin_weighted_mean(x, y, e1_r,               weight,    x_bin, y_bin)
    e2_b  = _bin_weighted_mean(x, y, e2_r,               weight,    x_bin, y_bin)
    esq_b = _bin_weighted_mean(x, y, e1_r**2 + e2_r**2,  weight**2, x_bin, y_bin)

    M_E, _, _ = _aperture_mass_grid(nrow, ncol, e1_b, e2_b, esq_b, Rs)
    return M_E


def _bin_weighted_mean(x, y, values, weight, x_bin, y_bin):
    kw = dict(statistic="sum", bins=[x_bin, y_bin])
    wsum = stats.binned_statistic_2d(x, y, values * weight, **kw).statistic
    wmap = stats.binned_statistic_2d(x, y, weight,          **kw).statistic
    wmap[wmap == 0] = np.inf
    return (wsum / wmap).T


def _aperture_mass_grid_v2(nrow, ncol, e1_b, e2_b, esq_b, Rs):
    """Loop over every pixel and return (M_E, M_B, n_M)."""
    xv, yv = np.meshgrid(np.arange(ncol), np.arange(nrow))
    M_E = np.empty((nrow, ncol))
    M_B = np.empty((nrow, ncol))
    n_M = np.empty((nrow, ncol))

    for row in range(nrow):
        for col in range(ncol):
            dx, dy = xv - col, yv - row
            Q = _schirmer_weight(np.sqrt(dx**2 + dy**2), Rs)

            angle = np.arctan2(dy, dx)
            cos2a = np.cos(2.0 * angle)
            sin2a = np.sin(2.0 * angle)

            et = -e1_b * cos2a - e2_b * sin2a
            ex =  e1_b * sin2a - e2_b * cos2a

            M_E[row, col] = np.nansum(Q * et)
            M_B[row, col] = np.nansum(Q * ex)
            n_M[row, col] = np.sqrt(np.nansum(Q**2 * esq_b)) / np.sqrt(2)

    return M_E, M_B, n_M

def _aperture_mass_grid(nrow, ncol, e1_b, e2_b, esq_b, Rs):
    """Convolution-based aperture mass. Returns (M_E, M_B, n_M)."""

    # --- build kernels (once) ---
    # Schirmer sigmoid kills Q beyond r ~ Rs; pad slightly to be safe
    half = int(np.ceil(Rs)) + 1
    size = 2 * half + 1
    coords = np.arange(size) - half
    X_k, Y_k = np.meshgrid(coords, coords)
    R_k = np.sqrt(X_k**2 + Y_k**2)
    Phi_k = np.arctan2(Y_k, X_k)

    Q_k = _schirmer_weight(R_k, Rs)

    K1 = Q_k * np.cos(2.0 * Phi_k)
    K2 = Q_k * np.sin(2.0 * Phi_k)
    K_noise = Q_k**2

    # --- handle NaN pixels (masked regions) ---
    mask = np.isnan(e1_b) | np.isnan(e2_b)
    e1 = np.where(mask, 0.0, e1_b)
    e2 = np.where(mask, 0.0, e2_b)
    esq = np.where(mask | np.isnan(esq_b), 0.0, esq_b)

    # --- E-mode: conv(-e1, K1) + conv(-e2, K2) ---
    M_E = fftconvolve(-e1, K1, mode='same') + \
          fftconvolve(-e2, K2, mode='same')

    # --- B-mode: conv(e1, K2) + conv(-e2, K1) ---
    M_B = fftconvolve(e1, K2, mode='same') + \
          fftconvolve(-e2, K1, mode='same')

    # --- noise: sqrt(conv(esq, Q^2)) / sqrt(2) ---
    n_M = np.sqrt(fftconvolve(esq, K_noise, mode='same')) / np.sqrt(2)

    return M_E, M_B, n_M


# ═══════════════════════════════════════════════════════════════════
# Public class
# ═══════════════════════════════════════════════════════════════════

class ApertureMassSNR:
    """
    End-to-end aperture mass S/N from a shape catalog.

    Parameters
    ----------
    g1, g2 : array_like
        Shear components per galaxy.
    weight : array_like
        Shape measurement weight per galaxy.
    x, y : array_like
        Pixel coordinates per galaxy.
    bin_size : float
        Spatial bin width in pixels.
    Rs : float
        Schirmer filter scale in pixels (converted to bin units internally).
    """

    def __init__(self, g1, g2, weight, x, y, bin_size, Rs):
        self.g1     = np.asarray(g1, dtype=float)
        self.g2     = np.asarray(g2, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.x      = np.asarray(x, dtype=float)
        self.y      = np.asarray(y, dtype=float)
        self.e_amp  = np.sqrt(self.g1**2 + self.g2**2)

        self.bin_size = float(bin_size)
        self.Rs       = float(Rs) # / self.bin_size   # bin units

        # grid setup
        self.x_min, self.x_max = self.x.min(), self.x.max()
        self.y_min, self.y_max = self.y.min(), self.y.max()
        self.x_bin = np.arange(self.x_min, self.x_max + bin_size, bin_size)
        self.y_bin = np.arange(self.y_min, self.y_max + bin_size, bin_size)
        self.ncol  = int(np.ceil((self.x_max - self.x_min) / bin_size))
        self.nrow  = int(np.ceil((self.y_max - self.y_min) / bin_size))

    def run(self, n_realizations=200, n_cpus=1, seed=12000, return_kappa_n_noise=True):
        """
        Compute S/N maps.

        Returns
        -------
        SNR_E, SNR_B : ndarray, shape (nrow, ncol)
        """
        # ── real signal ──────────────────────────────────────────
        e1_b  = _bin_weighted_mean(self.x, self.y, self.g1,
                                   self.weight, self.x_bin, self.y_bin)
        e2_b  = _bin_weighted_mean(self.x, self.y, self.g2,
                                   self.weight, self.x_bin, self.y_bin)
        esq_b = _bin_weighted_mean(self.x, self.y,
                                   self.g1**2 + self.g2**2,
                                   self.weight**2, self.x_bin, self.y_bin)

        M_E_real, M_B_real, _ = _aperture_mass_grid(
            self.nrow, self.ncol, e1_b, e2_b, esq_b, self.Rs)

        # ── noise realizations (parallel over realizations) ──────
        base_rng = np.random.RandomState(seed)
        seeds = base_rng.randint(0, 2**31, size=n_realizations)

        args = [(self.e_amp, self.weight, self.x, self.y,
                 self.x_bin, self.y_bin, self.nrow, self.ncol,
                 self.Rs, s) for s in seeds]

        if n_cpus > 1:
            with Pool(n_cpus) as pool:
                M_E_stack = np.array(pool.map(_run_one_realization, args))
        else:
            M_E_stack = np.array([_run_one_realization(a) for a in args])

        # ── S/N ──────────────────────────────────────────────────
        noise_map = np.std(M_E_stack, axis=0)
        SNR_E = M_E_real / noise_map
        SNR_B = M_B_real / noise_map
        if return_kappa_n_noise:
            return SNR_E, SNR_B, M_E_real, noise_map
        else:
            return SNR_E, SNR_B
    

class ApertureMassCalculator:
    """
    Compute Schirmer aperture mass (E/B/noise) over a 2-D pixel grid.

    Parameters
    ----------
    nrow, ncol : int
        Grid dimensions (dec, ra).
    e1_binned, e2_binned, e_sq_binned : ndarray, shape (nrow, ncol)
        Weighted-mean binned shear components and squared ellipticity.
    Rs : float
        Filter scale in bin units.
    """

    def __init__(self, nrow, ncol, e1_binned, e2_binned, e_sq_binned, Rs):
        self.nrow = nrow
        self.ncol = ncol
        self.e1_binned = e1_binned
        self.e2_binned = e2_binned
        self.e_sq_binned = e_sq_binned
        self.Rs = Rs
        self.xv, self.yv = np.meshgrid(np.arange(ncol), np.arange(nrow))

    def update_shear(self, e1_binned, e2_binned, e_sq_binned):
        """Swap in new binned shear maps (e.g. for a noise realization)."""
        self.e1_binned = e1_binned
        self.e2_binned = e2_binned
        self.e_sq_binned = e_sq_binned

    def _at_pixel(self, row, col):
        """Aperture mass (E, B, noise) at a single grid pixel."""
        dx, dy = self.xv - col, self.yv - row
        Q = _schirmer_weight(np.sqrt(dx**2 + dy**2), self.Rs)

        angle = np.arctan2(dy, dx)
        cos2a = np.cos(2.0 * angle)
        sin2a = np.sin(2.0 * angle)

        et = -self.e1_binned * cos2a - self.e2_binned * sin2a
        ex =  self.e1_binned * sin2a - self.e2_binned * cos2a

        M_E = np.nansum(Q * et)
        M_B = np.nansum(Q * ex)
        n_M = np.sqrt(np.nansum(Q**2 * self.e_sq_binned)) / np.sqrt(2)

        return M_E, M_B, n_M

    def run(self):
        """
        Returns ``(M_E, M_B, n_M)`` — each shaped ``(nrow, ncol)``.
        """
        M_E = np.empty((self.nrow, self.ncol))
        M_B = np.empty((self.nrow, self.ncol))
        n_M = np.empty((self.nrow, self.ncol))

        for row in range(self.nrow):
            for col in range(self.ncol):
                M_E[row, col], M_B[row, col], n_M[row, col] = \
                    self._at_pixel(row, col)

        return M_E, M_B, n_M
    
    @staticmethod
    def run_realizations(nrow, ncol, Rs, grid_list, n_cpus=1):
        """
        Run aperture mass E-mode for many noise realizations in parallel.

        Parameters
        ----------
        grid_list : iterable of (g1map, g2map, gsq_map)
        n_cpus : int

        Returns
        -------
        M_E_stack : ndarray, shape (n_realizations, nrow, ncol)
        """
        args = [(nrow, ncol, Rs, g1, g2, gsq)
                for g1, g2, gsq in grid_list]

        with Pool(n_cpus) as pool:
            M_E_stack = pool.map(_run_single_realization, args)

        return np.array(M_E_stack)


def _run_single_realization(args):
    """Module-level function so pickle can find it."""
    nrow, ncol, Rs, g1map, g2map, gsq_map = args
    calc = ApertureMassCalculator(nrow, ncol, g1map, g2map, gsq_map, Rs)
    M_E, _, _ = calc.run()
    return M_E