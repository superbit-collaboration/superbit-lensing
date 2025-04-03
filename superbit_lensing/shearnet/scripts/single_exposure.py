from superbit_lensing.shearnet import dataset
from superbit_lensing.shearnet import utils
from astropy.io import fits
import ngmix
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from tqdm import tqdm
import os
import argparse
import sys

def main(args):
    print("=== Arguments Passed to the Script ===")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================\n")
    num_samples = args.num_samples
    nobj = num_samples
    flux=args.flux
    jitter_fwhm = args.psf_fwhm
    psf_sigma = jitter_fwhm/2.355
    seed=args.seed
    npix, scale = args.npix, args.scale
    nse_sd = args.nse_sd
    sim_type= args.sim_type
    sim_exp = args.sim_exp
    ngmix_model_psf = args.ngmix_model_psf
    ngmix_model_gal = args.ngmix_model_gal
    outfilename = args.outfilename

    g1_selected, g2_selected, sigma_selected = utils.g1_g2_sigma_sample(num_samples=num_samples)

    obs_list = []
    g1_list = []
    g2_list = []
    g1_admom_list = []
    g2_admom_list = []
    sigma_admom_list = []
    sigma_list = []
    flag_list = []
    for i in tqdm(range(nobj)):
        try:
            obs, g1_admom, g2_admom, sigma_admom, flag = dataset.sim_func(g1_selected[i], g2_selected[i], sigma=sigma_selected[i], flux=flux, psf_sigma=psf_sigma, nse_sd = nse_sd,  type=sim_type, npix=npix, scale=scale, seed=seed + 10*i, exp=sim_exp)
            #size_gb = sys.getsizeof(obs.image) / (1024**2)
            #print(f"Memory occupied by x: {size_gb:.6f} MB")
            obs_list.append(obs)
            g1_list.append(g1_selected[i])
            g2_list.append(g2_selected[i])
            g1_admom_list.append(g1_admom)
            g2_admom_list.append(g2_admom)
            sigma_admom_list.append(sigma_admom)
            sigma_list.append(sigma_selected[i])
            flag_list.append(flag)
        except Exception as e:
            print(f"error in sim_func for {e}")

    prior = utils._get_priors(seed)
    rng = np.random.RandomState(seed)

    data_list = utils.mp_fit_one(obs_list, prior, rng, psf_model=ngmix_model_psf, gal_model=ngmix_model_gal)

    mcal_shear = 0.01

    r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list = utils.response_calculation(data_list, mcal_shear)

    g_noshear_list = []
    g_cov_noshear_list = []
    T_noshear_list = []
    Tpsf_noshear_list = []
    for i in tqdm(range(len(obs_list))):
        g_noshear = data_list[i][0][3]
        T_noshear = data_list[i][0][4]
        Tpsf_noshear = data_list[i][0][5]
        g_cov_noshear = data_list[i][0][6]
        g_noshear_list.append(g_noshear)
        T_noshear_list.append(T_noshear)
        Tpsf_noshear_list.append(Tpsf_noshear)
        g_cov_noshear_list.append(g_cov_noshear)

    # Convert lists to NumPy arrays
    r11_array = np.array(r11_list)
    r22_array = np.array(r22_list)
    r12_array = np.array(r12_list)
    r21_array = np.array(r21_list)
    c1_array = np.array(c1_list)
    c2_array = np.array(c2_list)
    c1_psf_array = np.array(c1_psf_list)
    c2_psf_array = np.array(c2_psf_list)
    g_noshear_array = np.array(g_noshear_list)
    g_cov_noshear_array = np.array(g_cov_noshear_list)
    T_noshear_array = np.array(T_noshear_list)
    Tpsf_noshear_array = np.array(Tpsf_noshear_list)

    # New columns
    g1_array = np.array(g1_list)
    g2_array = np.array(g2_list)
    g1_admom_array = np.array(g1_admom_list)
    g2_admom_array = np.array(g2_admom_list)
    sigma_admom_array = np.array(sigma_admom_list)
    flag_array = np.array(flag_list)
    sigma_array = np.array(sigma_list)
    true_T_array = 2 * sigma_array**2  # Compute true_T

    # Define FITS columns
    cols = [
        fits.Column(name='r11', format='E', array=r11_array),
        fits.Column(name='r22', format='E', array=r22_array),
        fits.Column(name='r12', format='E', array=r12_array),
        fits.Column(name='r21', format='E', array=r21_array),
        fits.Column(name='c1', format='E', array=c1_array),
        fits.Column(name='c2', format='E', array=c2_array),
        fits.Column(name='c1_psf', format='E', array=c1_psf_array),
        fits.Column(name='c2_psf', format='E', array=c2_psf_array),
        fits.Column(name='g_noshear', format='2E', array=g_noshear_array),  # Assuming g_noshear has 2 elements per entry
        fits.Column(name="g_cov_noshear", format="4E", array=g_cov_noshear_array),
        fits.Column(name='T_noshear', format='E', array=T_noshear_array),
        fits.Column(name='Tpsf_noshear', format='E', array=Tpsf_noshear_array),
        fits.Column(name='true_g1', format='E', array=g1_array),
        fits.Column(name='true_g2', format='E', array=g2_array),
        fits.Column(name='admom_g1', format='E', array=g1_admom_array),
        fits.Column(name='admom_g2', format='E', array=g2_admom_array),
        fits.Column(name='admom_sigma', format='E', array=sigma_admom_array),
        fits.Column(name='flag', format='I', array=flag_array),
        fits.Column(name='true_T', format='E', array=true_T_array),
    ]

    # Create FITS table
    hdu = fits.BinTableHDU.from_columns(cols)

    # Save FITS file
    hdu.writeto(outfilename, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create simulations for shear net")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--flux", type=float, default=1, help="Flux of the galaxy")
    parser.add_argument("--psf_fwhm", type=float, default=0.5, help="PSF FWHM")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--npix", type=int, default=63, help="Number of pixels")
    parser.add_argument("--scale", type=float, default=0.141, help="Scale factor")
    parser.add_argument("--nse_sd", type=float, default=1e-4, help="Noise standard deviation")
    parser.add_argument("--sim_type", type=str, default="gauss", help="Simulation type")
    parser.add_argument("--sim_exp", type=str, default="ideal", help="Simulation experiment")
    parser.add_argument("--ngmix_model_psf", type=str, default="gauss", help="NGMIX model for PSF")
    parser.add_argument("--ngmix_model_gal", type=str, default="gauss", help="NGMIX model for galaxy")
    parser.add_argument("--outfilename", type=str, default="sim.fits", help="Output FITS file name")
    args = parser.parse_args()
    main(args)