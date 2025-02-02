import numpy as np
import piff
import galsim

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
