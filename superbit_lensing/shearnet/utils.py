import numpy as np
import ngmix
import galsim
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import sys
from pympler import asizeof


# Function to remove NaN values and count them
def clean_and_report_nans(data_list, name):
    clean_list = np.array(data_list)
    nan_count = np.isnan(clean_list).sum()
    
    if nan_count > 0:
        print(f"Removed {nan_count} NaN values from {name}.")
    
    return clean_list[~np.isnan(clean_list)]  # Return array without NaNs

def fourier_transform(psf):
    # Compute the Fourier Transform
    ft_psf = np.fft.fft2(psf)
    ft_psf_shifted = np.fft.fftshift(ft_psf)  # Shift zero frequency to center
    return ft_psf_shifted

def inverse_fourier_transform(ft_psf_shifted):
    # Shift back and compute Inverse Fourier Transform
    ft_psf = np.fft.ifftshift(ft_psf_shifted)
    psf_reconstructed = np.fft.ifft2(ft_psf)
    return np.abs(psf_reconstructed)

def fft_ifft(psf):
    # Compute the Fourier Transform
    ft_psf = np.fft.fft2(psf)
    ft_psf_shifted = np.fft.fftshift(ft_psf)  # Shift zero frequency to center
    # Shift back and compute Inverse Fourier Transform
    ft_psf = np.fft.ifftshift(ft_psf_shifted)
    psf_reconstructed = np.fft.ifft2(ft_psf)
    return np.abs(psf_reconstructed)

def convolve2d(image, psf, mode='same', boundary='wrap'):
    """
    Convolve an image with a PSF using 2D FFT-based convolution.
    Always returns an image of the same dimensions as the input.

    Parameters:
    image (np.ndarray): The input image.
    psf (np.ndarray): The PSF.
    mode (str): Not used in FFT implementation, kept for API consistency.
    boundary (str): Not used in FFT implementation, kept for API consistency.

    Returns:
    np.ndarray: The convolved image with same dimensions as input.
    """
    # Ensure PSF is centered in an array of the same size as the image
    psf_padded = np.zeros(image.shape, dtype=psf.dtype)
    psf_center = np.array(psf.shape) // 2
    image_center = np.array(image.shape) // 2
    
    # Calculate the corner positions for placing the PSF
    top = image_center[0] - psf_center[0]
    left = image_center[1] - psf_center[1]
    
    # Handle PSFs larger than the image
    psf_top = max(0, -top)
    psf_left = max(0, -left)
    psf_bottom = min(psf.shape[0], image.shape[0] - top)
    psf_right = min(psf.shape[1], image.shape[1] - left)
    
    # Handle image boundaries
    img_top = max(0, top)
    img_left = max(0, left)
    img_bottom = min(image.shape[0], top + psf.shape[0])
    img_right = min(image.shape[1], left + psf.shape[1])
    
    # Place the PSF in the padded array
    psf_padded[img_top:img_bottom, img_left:img_right] = psf[psf_top:psf_bottom, psf_left:psf_right]
    
    # Perform FFT convolution
    ft_image = np.fft.fft2(image)
    ft_psf = np.fft.fft2(psf_padded)
    ft_result = ft_image * ft_psf
    result = np.abs(np.fft.ifft2(ft_result))
    
    return np.fft.fftshift(result)

def sample_half_gaussian(size=1, sigma=0.5):
    """
    Generate samples from a half-Gaussian distribution with given sigma.
    Ensures that all values are strictly greater than zero.

    Parameters:
    size (int): Number of samples to generate.
    sigma (float): Standard deviation of the full Gaussian.

    Returns:
    np.ndarray: Array of sampled values.
    """
    samples = np.abs(np.random.normal(loc=0, scale=sigma, size=size))
    samples = samples[samples > 0.14]  # Remove zeros
    while len(samples) < size:
        extra_samples = np.abs(np.random.normal(loc=0, scale=sigma, size=size - len(samples)))
        extra_samples = extra_samples[extra_samples > 0.14]
        samples = np.concatenate((samples, extra_samples))
    return samples

def g1_g2_sigma_sample(num_samples=10000):
    """
    Generate samples for g1, g2, and sigma from a half-Gaussian distribution.

    Parameters:
    num_samples (int): Number of samples to generate.

    Returns:
    tuple: Arrays of sampled g1, g2, and sigma values.
    """

    sigma = sample_half_gaussian(size=num_samples, sigma=0.5)
    # Generate g1 and g2 ensuring |g1 + 1j * g2| <= 1
    g1_selected = np.zeros(num_samples)
    g2_selected = np.zeros(num_samples)

    for i in range(num_samples):
        while True:
            g1, g2 = np.random.uniform(-1, 1, 2)  # Generate g1 and g2
            if np.abs(g1 + 1j * g2) <= 0.5:  # Check the constraint
                g1_selected[i] = g1
                g2_selected[i] = g2
                break  # Accept only valid values

    return g1_selected, g2_selected, sigma

def make_struct(res, obs, shear_type):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', 'T', and 'g_cov'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i4'),
        ('shear_type', 'U7'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
        ('g_cov', 'f8', (2, 2)),
    ]
    data = np.zeros(1, dtype=dt)
    data['shear_type'] = shear_type
    data['flags'] = res['flags']

    if res['flags'] == 0:
        data['s2n'] = res['s2n']
        # for Gaussian moments we are actually measuring e, the ellipticity
        try:
            data['g'] = res['e']
        except KeyError:
            data['g'] = res['g']
        data['T'] = res['T']
        data['g_cov'] = res.get('g_cov', np.nan * np.ones((2, 2)))
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['Tpsf'] = np.nan
        data['g_cov'] = np.nan * np.ones((2, 2))

    # Get the psf T by averaging over epochs (and eventually bands)
    tpsf_list = []
    
    try:
        tpsf_list.append(obs.psf.meta['result']['T'])
    except:
        print(f"No PSF T found for observation")
            
    data['Tpsf'] = np.mean(tpsf_list)

    return data

def _get_priors(seed):

    # This bit is needed for ngmix v2.x.x
    # won't work for v1.x.x
    rng = np.random.RandomState(seed)

    # prior on ellipticity.  The details don't matter, as long
    # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014

    g_sigma = 0.3
    g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

    # 2-d gaussian prior on the center
    # row and column center (relative to the center of the jacobian, which would be zero)
    # and the sigma of the gaussians

    # units same as jacobian, probably arcsec
    row, col = 0.0, 0.0
    row_sigma, col_sigma = 0.2, 0.2 # a bit smaller than pix size of SuperBIT
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

    # T prior.  This one is flat, but another uninformative you might
    # try is the two-sided error function (TwoSidedErf)

    Tminval = -1.0 # arcsec squared
    Tmaxval = 1000
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

    # similar for flux.  Make sure the bounds make sense for
    # your images

    Fminval = -1.e1
    Fmaxval = 1.e5
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

    # now make a joint prior.  This one takes priors
    # for each parameter separately
    priors = ngmix.joint_prior.PriorSimpleSep(
    cen_prior,
    g_prior,
    T_prior,
    F_prior)

    return priors

def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

def get_coellip_ngauss(name):
    ngauss=int( name[7:] )
    return ngauss

def process_obs(obs, boot):
    resdict, obsdict = boot.go(obs)
    dlist = [make_struct(res=sres, obs=obsdict[stype], shear_type=stype) for stype, sres in resdict.items()]
    return np.hstack(dlist)

def mp_fit_one(obslist, prior, rng, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
    """
    Multiprocessing version of original _fit_one()

    Method to perfom metacalibration on an object. Returns the unsheared ellipticities
    of each galaxy, as well as entries for each shear step

    inputs:
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - mcal_pars: mcal running parameters

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """
    # get image pixel scale (assumes constant across list)
    jacobian = obslist[0]._jacobian
    Tguess = 4*jacobian.get_scale()**2
    ntry = 20
    lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    psf_lm_pars={'maxfev': 4000, 'xtol':5.0e-5,'ftol':5.0e-5}

    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    # psf fitting
    if 'em' in psf_model:
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}
        psf_ngauss = get_em_ngauss(psf_model)
        psf_fitter = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in psf_model:
        psf_ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif psf_model == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    else:
        raise ValueError('psf_model must be one of emn, coellipn, or gauss')

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)

    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

    #types = ['noshear', '1p', '1m', '2p', '2m']
    psf = mcal_pars['psf']
    mcal_shear = mcal_pars['mcal_shear']
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        step = mcal_shear,
        #types=types,
    )



    num_cores = max(18, cpu_count())
    print(f"Using {num_cores} cores out of {cpu_count()} available.")

    with Pool(num_cores) as pool:
        data_list = list(tqdm(pool.starmap(process_obs, [(obs, boot) for obs in obslist]), total=len(obslist)))

    return data_list

def mp_fit_one_single(obslist, prior, rng, psf_model='gauss', gal_model='gauss', mcal_pars= {'psf': 'dilate', 'mcal_shear': 0.01}):
    """
    Multiprocessing version of original _fit_one()

    Method to perfom metacalibration on an object. Returns the unsheared ellipticities
    of each galaxy, as well as entries for each shear step

    inputs:
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - mcal_pars: mcal running parameters

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """
    # get image pixel scale (assumes constant across list)
    jacobian = obslist[0]._jacobian
    Tguess = 4*jacobian.get_scale()**2
    ntry = 20
    lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    psf_lm_pars={'maxfev': 4000, 'xtol':5.0e-5,'ftol':5.0e-5}

    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    # psf fitting
    if 'em' in psf_model:
        em_pars={'tol': 1.0e-6, 'maxiter': 50000}
        psf_ngauss = get_em_ngauss(psf_model)
        psf_fitter = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in psf_model:
        psf_ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif psf_model == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    else:
        raise ValueError('psf_model must be one of emn, coellipn, or gauss')

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)

    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

    #types = ['noshear', '1p', '1m', '2p', '2m']
    psf = mcal_pars['psf']
    mcal_shear = mcal_pars['mcal_shear']
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        step = mcal_shear,
        #types=types,
    )



    num_cores = max(18, cpu_count())
    print(f"Using {num_cores} cores out of {cpu_count()} available.")

    data_list  = []
    for i in tqdm(range(len(obslist))):
        resdict, obsdict = boot.go(obslist[i])
        dlist = []
        for stype, sres in resdict.items():
            st = make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
            dlist.append(st)

        data = np.hstack(dlist)
        data_list.append(data) 

    return data_list

def get_memory_usage(obj):
    """Prints the memory usage of each attribute in an object in MB."""
    memory_usage = {attr: asizeof.asizeof(getattr(obj, attr)) / (1024 * 1024) for attr in obj.__dict__}
    
    for attr, size in memory_usage.items():
        print(f"{attr}: {size:.6f} MB")
    
    total_memory = sum(memory_usage.values())
    print(f"\nTotal memory used by instance: {total_memory:.6f} MB")

def response_calculation(data_list, mcal_shear):
    r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list = [], [], [], [], [], [], [], []

    for i in tqdm(range(len(data_list))):
        g_noshear = data_list[i][0][3]
        g_1p = data_list[i][1][3]
        g_1m = data_list[i][2][3]
        g_2p = data_list[i][3][3]
        g_2m = data_list[i][4][3]
        g_1p_psf =  data_list[i][5][3]
        g_1m_psf =  data_list[i][6][3]
        g_2p_psf =  data_list[i][7][3]
        g_2m_psf =  data_list[i][8][3]
        r11 = (g_1p[0] - g_1m[0])/(2. * mcal_shear)
        r22 = (g_2p[1] - g_2m[1])/(2. * mcal_shear)
        r12 = (g_2p[0] - g_2m[0])/(2. * mcal_shear)
        r21 = (g_1p[1] - g_1m[1])/(2. * mcal_shear)
        c1 = (g_1p[0] + g_1m[0])/2 - g_noshear[0]
        c2 = (g_2p[1] + g_2m[1])/2 - g_noshear[1]
        c1_psf = (g_1p_psf[0] + g_1m_psf[0])/2 - g_noshear[0]
        c2_psf = (g_2p_psf[1] + g_2m_psf[1])/2 - g_noshear[1]
        r11_list.append(r11)
        r22_list.append(r22)
        r12_list.append(r12)
        r21_list.append(r21)
        c1_list.append(c1)
        c2_list.append(c2)
        c1_psf_list.append(c1_psf)
        c2_psf_list.append(c2_psf)

    return r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list