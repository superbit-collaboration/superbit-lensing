import numpy as np
import fitsio
import piff
import galsim
from galsim.des import DES_PSFEx
import superbit_lensing.utils as utils

import ipdb

def psf_extender(mode, stamp_size, **kwargs):
    '''
    Utility function to add the get_rec function expected
    by the MEDS package

    psf: PSF object of unknown class
        A python object representing a PSF
    mode: str
        The PSF mode you are using.
        Current valid types: ['piff', 'true'].
        NOTE: PSFEx does not need an extender
    stamp_size: int
        The size of the piff PSF cutout
    kwargs: kwargs dict
        Anything that should be passed to the corresponding PSF extender
    '''

    # PSFEx does not need an extender
    valid_modes = ['piff', 'true']

    if mode == 'piff':
        piff_file = kwargs['piff_file']
        psf_extended = _piff_extender(piff_file, stamp_size)
    elif mode == 'true':
        psf = kwargs['psf']
        psf_pix_scale = kwargs['psf_pix_scale']
        psf_extended = _true_extender(psf, stamp_size, psf_pix_scale)
    else:
        raise KeyError(f'{mode} is not one of the valid PSF modes: ' +\
                       f'{valid_modes}')

    return psf_extended

def _piff_extender(piff_file, stamp_size):
    '''
    Utility function to add the get_rec function expected
    by the MEDS package to a PIFF PSF

    piff_file: str
        The piff filename
    stamp_size: int
        The size of the piff PSF cutout
    '''

    psf = piff.read(piff_file)
    type_name = type(psf)

    class PiffExtender(type_name):
        '''
        A helper class that adds functions expected by MEDS
        '''

        def __init__(self, psf):

            self.psf = psf
            self.single_psf = type_name

            return

        def get_rec(self, row, col):

            fake_pex = self.psf.draw(
                x=col, y=row, stamp_size=stamp_size
                ).array

            return fake_pex

        def get_center(self, row, col):

            psf_shape = self.psf.draw(
                x=col, y=row, stamp_size=stamp_size
                ).array.shape
            cenpix_row = (psf_shape[0]-1)/2
            cenpix_col = (psf_shape[1]-1)/2
            cen = np.array([cenpix_row, cenpix_col])

            return cen

    psf_extended = PiffExtender(psf)

    return psf_extended

def _true_extender(psf, stamp_size, psf_pix_scale):
    '''
    Utility function to add the get_rec function expected
    by the MEDS package to a True GalSim PSF

    psf: galsim.GSObject
        The true GSObject used for the PSF in image simulation rendering
    stamp_size: int
        The size of the piff PSF cutout
    psf_pix_scale: float
        The pixel scale in arcsec/pixel
    '''

    type_name = type(psf)

    class TrueExtender(type_name):
        '''
        A helper class that adds functions expected by MEDS to a
        GalSim PSF
        '''

        def __init__(self, psf):

            self.psf = psf
            self.type_name = type_name
            self.psf_pix_scale = psf_pix_scale

            self.wcs = galsim.PixelScale(psf_pix_scale)

            return

        # def get_wcs(self):
        #     return self.wcs

        def get_rec(self, row, col, method='real_space'):
            '''
            Reconstruct the PSF image at the specified location

            NOTE: For a constant True PSF across the image, row & col
            are not used
            NOTE: k-space integration will cause issues for rendering
            our tiny PSF
            '''

            image = galsim.Image(
                stamp_size, stamp_size, scale=self.psf_pix_scale
                )

            psf_im = self.psf.drawImage(image, method=method).array
            # psf = galsim.Gaussian(flux=1, fwhm=0.24)
            # psf_im = psf.drawImage(image, method='real_space').array

            # from matplotlib.colors import LogNorm
            # import matplotlib.pyplot as plt
            # plt.subplot(121)
            # plt.imshow(psf_im, origin='lower', norm=LogNorm(vmin=1e-8, vmax=1e-1))
            # plt.colorbar()
            # plt.title('Gauss only')

            # plt.subplot(122)
            # psf_im = self.psf.drawImage(image, method='real_space').array
            # plt.imshow(psf_im, origin='lower', norm=LogNorm(vmin=1e-8, vmax=1e-1))
            # plt.colorbar()
            # plt.title('Conv[Gauss, delta]')

            # plt.gcf().set_size_inches(9,4)

            # plt.show()

            return psf_im

        def get_center(self, row, col):

            psf_shape = self.get_rec(col, row).shape
            cenpix_row = (psf_shape[0] - 1) / 2
            cenpix_col = (psf_shape[1] - 1) / 2
            cen = np.array([cenpix_row, cenpix_col])

            return cen

    psf_extended = TrueExtender(psf)

    return psf_extended

class PSFWrapper(dict):
    """
    Helper wrapper that provides the PSFEx-style methods expected by MEDS.

    This implementation is inspired by the DES/DECADE/DELVE PSFWrapper:
    https://github.com/delve-survey/mcal_sim_test/blob/main/psf_wrapper.py

    Parameters
    ----------
    psf_file : str
        Path to the PSFEx `.psf` file.
    image_file : str
        Path to the image file that provides the WCS.
    npix : int
        Size of the PSF cutout (npix x npix).
    """

    def __init__(self, psf_file: str, image_file: str | None = None, npix: int = 51):
        super().__init__()

        self.psf_file = psf_file

        if image_file is not None:
            self.wcs = self.galsim_wcs(image_file)
        else:
            self.wcs = utils.get_galsim_tanwcs()

        self.npix = int(npix)

        self.load()

    def load(self) -> None:
        hdr = fitsio.read_header(self.psf_file, ext=1)
        self["filename"] = self.psf_file
        self["poldeg"] = int(hdr["POLDEG1"])
        self["masksize"] = np.array([hdr["PSFAXIS1"], hdr["PSFAXIS2"], hdr["PSFAXIS3"]], dtype=int)
        self["contextoffset"] = np.array([hdr["POLZERO1"], hdr["POLZERO2"]], dtype=float)
        self["contextscale"] = np.array([hdr["POLSCAL1"], hdr["POLSCAL2"]], dtype=float)
        self["psf_samp"] = float(hdr["PSF_SAMP"])
        self.psfex_model = DES_PSFEx(self.psf_file, wcs=self.wcs)

    def get_rec(self, row, col):
        """
        Return a PSF image (npix x npix) at the requested detector location.

        Notes
        -----
        `row, col` are assumed to be 0-indexed image coordinates. We add +1
        when building the GalSim PositionD to match FITS/WCS convention.
        """
        row = float(row)
        col = float(col)

        img_pos = galsim.PositionD(col + 1.0, row + 1.0)
        wcs_loc = self.wcs.local(img_pos)

        psf_local = self.psfex_model.getPSF(img_pos)
        psf_im = psf_local.drawImage(
            nx=self.npix,
            ny=self.npix,
            wcs=wcs_loc,
            method="no_pixel",
        ).array
        return psf_im

    def get_center(self, row, col):
        # MEDS expects the center of the returned cutout in (row, col) order.
        cen = (self.npix - 1.0) / 2.0
        return cen, cen

    def get_rec_shape(self, row, col):
        return (self.npix, self.npix)

    @staticmethod
    def galsim_wcs(image_file, image_ext=0):
        hd = fitsio.read_header(image_file, ext=image_ext)
        hd = {k.upper(): hd[k] for k in hd if k is not None}
        wcs = galsim.FitsWCS(header=hd)
        assert not isinstance(wcs, galsim.PixelScale)  # sanity check
        return wcs
    
