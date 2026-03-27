import os
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from dust_extinction.parameter_averages import G23
from dustmaps.csfd import CSFDQuery


# ----------------------------
# Paths (keep your structure)
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INSTRUMENT_PATH = os.path.join(PROJECT_ROOT, 'data', 'photometry', 'instrument')

imx455_csv = os.path.join(INSTRUMENT_PATH, 'camera', 'imx455.csv')

b_transmission_file = os.path.join(INSTRUMENT_PATH, 'bandpass', 'b_2023.csv')
g_transmission_file = os.path.join(INSTRUMENT_PATH, 'bandpass', 'g_2023.csv')
u_transmission_file = os.path.join(INSTRUMENT_PATH, 'bandpass', 'u_2023.csv')


# ----------------------------
# Load transmissions ONCE
# ----------------------------
B_TRANSMISSION = np.genfromtxt(b_transmission_file, delimiter=',')[:, 2][1:]
G_TRANSMISSION = np.genfromtxt(g_transmission_file, delimiter=',')[:, 2][1:]
U_TRANSMISSION = np.genfromtxt(u_transmission_file, delimiter=',')[:, 2][1:]


def load_wavelengths(camera_name='imx455'):
    if camera_name == 'imx455':
        data = np.genfromtxt(imx455_csv, delimiter=',')
        wavelengths = data[:, 0][1:]  # assumed nm
        return wavelengths
    else:
        raise ValueError("Invalid camera name")


# ============================
# Main class
# ============================

class DustCorrector:
    """
    Fast dust extinction corrector for u, b, g bands.
    Uses Gordon et al. (2023) [https://arxiv.org/abs/2304.01991] Milky Way R(V) dependent model, with default Rv=3.1
    Uses dust map: SFD dust map of Chiang (2023) [https://arxiv.org/abs/2306.03926]
    """

    def __init__(self, Rv=3.1, camera_name='imx455'):
        self.Rv = Rv

        # models / heavy objects (init once)
        self.model = G23(Rv=Rv)
        self.wavelengths = load_wavelengths(camera_name) * u.nm
        self.csfd = CSFDQuery()

        # precompute bandpass-weighted Aλ/AV
        self.band_AxAv = {}
        for band in ['u', 'b', 'g']:
            self.band_AxAv[band] = self._compute_band_AxAv(band)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _get_band_transmission(self, band):
        if band == 'b':
            return B_TRANSMISSION
        elif band == 'g':
            return G_TRANSMISSION
        elif band == 'u':
            return U_TRANSMISSION
        else:
            raise ValueError("Invalid band")

    def _compute_band_AxAv(self, band):
        """
        Compute bandpass-weighted <Aλ/A_V>
        """
        T = self._get_band_transmission(band)

        lam = self.wavelengths
        AxAv = self.model(lam)

        lam_val = lam.value

        num = np.trapezoid(T * AxAv * lam_val, lam_val)
        den = np.trapezoid(T * lam_val, lam_val)

        return num / den

    # ----------------------------
    # Public API
    # ----------------------------
    def get_Av(self, ra, dec):
        """
        Get A_V from CSFD dust map
        """
        coords = SkyCoord(ra, dec, unit='deg')
        ebv = self.csfd(coords)
        return ebv * self.Rv

    def get_Ax(self, band, ra, dec):
        """
        Get extinction in a single band
        """
        Av = self.get_Av(ra, dec)
        AxAv = self.band_AxAv[band]
        return Av * AxAv

    def get_Ax_all(self, ra, dec, return_type='dict'):
        """
        Get extinction for all bands (u, b, g)
        """
        Av = self.get_Av(ra, dec)

        Ax_u = Av * self.band_AxAv['u']
        Ax_b = Av * self.band_AxAv['b']
        Ax_g = Av * self.band_AxAv['g']

        if return_type == 'dict':
            return {'u': Ax_u, 'b': Ax_b, 'g': Ax_g}

        elif return_type == 'array':
            return np.vstack([Ax_u, Ax_b, Ax_g]).T  # shape (N, 3)

        else:
            raise ValueError("return_type must be 'dict' or 'array'")

    def __call__(self, ra, dec, bands='all', return_type='dict'):
        """
        Main interface

        Examples:
        ---------
        dust(ra, dec)              -> all bands
        dust(ra, dec, 'b')         -> single band
        dust(ra, dec, ['u','g'])   -> subset
        """
        if bands == 'all':
            return self.get_Ax_all(ra, dec, return_type=return_type)

        elif isinstance(bands, str):
            return self.get_Ax(bands, ra, dec)

        elif isinstance(bands, (list, tuple)):
            Av = self.get_Av(ra, dec)
            result = {b: Av * self.band_AxAv[b] for b in bands}
            return result

        else:
            raise ValueError("Invalid bands input")