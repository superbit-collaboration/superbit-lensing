import numpy as np
import galsim
import ngmix
import sys
from scipy.signal import convolve2d
from .utils import fft_ifft, get_memory_usage #, convolve2d
import ipdb

psf_fnmae = '/projects/mccleary_group/saha/codes/.empty/psf_cutouts_superbit.npy'
weight_fname = '/projects/mccleary_group/saha/codes/.empty/weights_cutouts_superbit.npy'

def generate_dataset(samples, psf_fwhm, npix=53, scale=0.2, exp='ideal'):
    images = []
    labels = []

    for seed in range(samples):
        g1, g2 = np.random.uniform(-0.1, 0.1, size=2)  # Random shears
        if exp == 'superbit':
            obj_obs = sim_func_superbit(g1, g2, seed, npix=npix, scale=scale)
        elif exp == 'ideal':
            obj_obs = sim_func(g1, g2, seed, psf_fwhm, npix=npix, scale=scale)
        else:
            raise ValueError("For now only supported experiments are 'ideal' or 'superbit'")
        
        images.append(obj_obs.image)
        labels.append([g1, g2])  # e1, e2 labels

    return np.array(images), np.array(labels)

def sim_func(g1, g2, sigma=1.0, flux=1.0, psf_sigma=0.5, nse_sd = 1e-5,  type='exp', npix=53, scale=0.141, seed=42, exp="ideal", superbit_psf_fname=psf_fnmae):

    rng = np.random.RandomState(seed=seed)

    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    try:
        my_gaussian_image = gal.drawImage(nx=npix, ny=npix, scale=scale)
        my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)
        g1_admom, g2_admom = my_moments.observed_shape.g1, my_moments.observed_shape.g2
        sigma_admom = my_moments.moments_sigma * scale
        flag = 1
    except Exception as e:
        #print(e)
        g1_admom, g2_admom, sigma_admom = 0, 0, 0
        flag = 0

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = gal.shift(dx, dy)

    # Convolve with PSF
    if exp == 'ideal':
        psf = galsim.Gaussian(sigma=psf_sigma).shear(g1=-0.09190977, g2=0.03838577)
        obj = galsim.Convolve(sheared_gal, psf)

        # Draw images
        obj_im = obj.drawImage(nx=npix, ny=npix, scale=scale).array
        psf_im = psf.drawImage(nx=npix, ny=npix, scale=scale).array
    elif exp == 'superbit':
        try:
            sheared_im = sheared_gal.drawImage(nx=npix, ny=npix, scale=scale).array
        except Exception as e:
            print(e)
            sheared_im = np.zeros((npix, npix))
            flag = 2 
        psf_images = np.load(superbit_psf_fname)
        random_psf_index = rng.randint(0, psf_images.shape[0])  # Random index in the range [0, n)
        psf_im = psf_images[random_psf_index].copy()
        obj_im = convolve2d(sheared_im, psf_im, mode='same', boundary='wrap')

    else:
        raise ValueError("For now only supported experiments are 'ideal' or 'superbit'")

    # Add noise
    nse = rng.normal(size=obj_im.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.shape, scale=nse_sd)

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_im**2)) / target_psf_s2n
    #print(target_psf_noise)
    psf_obs = ngmix.Observation(
        image=psf_im,
        weight=np.ones_like(psf_im) / target_psf_noise**2,
        jacobian=psf_jac,
    )
    obj_obs = ngmix.Observation(
        image=obj_im + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )
    return obj_obs, g1_admom, g2_admom, sigma_admom, flag

def sim_func_superbit(g1, g2, sigma=1.0, flux=1.0, nse_sd = 1e-5,  type='exp', npix=53, scale=0.141, seed=42, superbit_psf_fnmae=psf_fnmae):

    rng = np.random.RandomState(seed=seed)
    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    try:
        my_gaussian_image = gal.drawImage(nx=npix, ny=npix, scale=scale)
        my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)
        g1_admom, g2_admom = my_moments.observed_shape.g1, my_moments.observed_shape.g2
        sigma_admom = my_moments.moments_sigma * scale
        flag = 1
    except Exception as e:
        #print(e)
        g1_admom, g2_admom, sigma_admom = 0, 0, 0
        flag = 0

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2

    sheared_gal = gal.shift(dx, dy)

    obj_im = sheared_gal.drawImage(nx=npix, ny=npix, scale=scale).array
    psf_images = np.load(superbit_psf_fnmae)

    random_psf_index = rng.randint(0, psf_images.shape[0])  # Random index in the range [0, n)
    psf_image = psf_images[random_psf_index]

    # Convolve the object image with the PSF
    psf_convolved = convolve2d(obj_im, psf_image, mode='same', boundary='wrap')

    nse = rng.normal(size=obj_im.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.shape, scale=nse_sd)
    final_image = psf_convolved + nse_im

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_image**2)) / target_psf_s2n
    #print(target_psf_noise)
    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=np.ones_like(psf_image) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    obj_obs = ngmix.Observation(
        image=final_image,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs, g1_admom, g2_admom, sigma_admom, flag
