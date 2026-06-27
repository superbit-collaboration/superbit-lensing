import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
import warnings

def compute_shear_bias(profile_tab, gtan_max=None, col_prefix=None, vb=True, include_psf_leakage=False, include_constant=False):
    '''
    profile_tab: astropy.Table
        Profile table used to compute the shear bias
    col_prefix: str
        Prefix to add to "alpha" & "sig_alpha" saved in metadata
    vb: bool
        Set to True to turn on prints
    include_psf_leakage: bool
        Set to True to include PSF leakage in the calculation

    Function to compute the max. likelihood estimator for the bias of a shear profile
    relative to the input NFW profile and the uncertainty on the bias.

    Saves the shear bias estimator ("alpha") and the uncertainty on the bias ("asigma")
    within the meta of the input profile_tab

    The following columns are assumed present in profile_tab:

    :gtan:      tangential shear of data
    :err_gtan:  RMS uncertainty for tangential shear data
    :nfw_gtan:  reference (NFW) tangential shear
    '''

    if not isinstance(profile_tab, Table):
        raise TypeError('profile_tab must be an astropy Table!')
    

    zeta, alpha, cbias, alpha_cross, sig_zeta, sig_alpha, sig_c, sig_alpha_cross,  r_at_gtan_max = _compute_shear_bias(profile_tab, gtan_max, include_psf_leakage, include_constant)

    if vb is True:
        print('# ')
        print(f'# shear bias is {zeta:.4f} +/- {sig_zeta:.3f}')
        print(f'# PSf leakage is {alpha:.3f} +/- {sig_alpha:.3f}')
        print(f'# additive bias is {cbias:.4f} +/- {sig_c:.3f}')
        print(f'# PSF leakage (from cross shear) is {alpha_cross:.3f} +/- {sig_alpha_cross:.3f}')
        print('# ')

    # Add this information to profile_tab metadata
    zeta_col = 'zeta'
    alpha_col = 'alpha'
    c_col = 'c'
    sig_zeta_col = 'sig_zeta'
    sig_alpha_col = 'sig_alpha'
    sig_c_col = 'sig_c'
    alpha_cross_col = 'alpha_cross'
    sig_alpha_cross_col = 'sig_alpha_cross'
    r_col = 'r_gtan_max'
    if col_prefix is not None:
        zeta_col = f'{col_prefix}_{zeta_col}'
        alpha_col = f'{col_prefix}_{alpha_col}'
        sig_zeta_col = f'{col_prefix}_{sig_zeta_col}'
        sig_alpha_col = f'{col_prefix}_{sig_alpha_col}'
        c_col = f'{col_prefix}_{c_col}'
        sig_c_col = f'{col_prefix}_{sig_c_col}'
        alpha_cross_col = f'{col_prefix}_{alpha_cross_col}'
        sig_alpha_cross_col = f'{col_prefix}_{sig_alpha_cross_col}'
        r_col = f'{col_prefix}_{r_col}'

    profile_tab.meta.update({
        zeta_col: zeta,
        alpha_col: alpha,
        sig_zeta_col: sig_zeta,
        sig_alpha_col: sig_alpha,
        c_col: cbias,
        sig_c_col: sig_c,
        alpha_cross_col: alpha_cross,
        sig_alpha_cross_col: sig_alpha_cross,
        r_col: r_at_gtan_max
    })

    return

def _compute_shear_bias(profile_tab, gtan_max=None, include_psf_leakage=False, include_constant=False):
    '''
    Compute alpha & sig_alpha from a table or rec_array.
    Optionally filter bins where mean_nfw_gtan < gtan_max.
    Returns (alpha, sig_alpha).
    '''

    try:
        T = profile_tab['mean_nfw_gtan']
        D = profile_tab['mean_gtan']
        D_cross = profile_tab['mean_gcross']
        errbar_gcross = profile_tab['err_gcross']
        errbar = profile_tab['err_gtan']
        midpoints_r = profile_tab['midpoint_r']
        if 'mean_gtan_psf' in profile_tab.columns:
            PSF = profile_tab['mean_gtan_psf']
        else:
            PSF = np.zeros_like(D)
        if 'mean_gcross_psf' in profile_tab.columns:
            PSF_cross = profile_tab['mean_gcross_psf']
        else:
            PSF_cross = np.zeros_like(D_cross)

    except KeyError as kerr:
        print('Shear bias calculation:')
        print('required columns not found; check input names?')
        raise kerr

    r_at_gtan_max = None
    # Filter bins if gtan_max is provided
    if gtan_max is not None:
        # Sort T for interpolation
        sort_idx = np.argsort(T)
        T_sorted = T[sort_idx]
        r_sorted = midpoints_r[sort_idx]

        # Interpolate r corresponding to gtan_max
        gtan_max_clipped = np.clip(gtan_max, T_sorted.min(), T_sorted.max())
        r_at_gtan_max = np.interp(gtan_max_clipped, T_sorted, r_sorted)
        print(f"Interpolated radius corresponding to gtan_max={gtan_max}: r = {r_at_gtan_max}")

        # Mask bins for calculation
        mask = T < gtan_max
        n_before = len(T)
        n_after = np.count_nonzero(mask)
        if n_after < n_before:
            warnings.warn(
                f"gtan_max={gtan_max} applied: "
                f"using {n_after}/{n_before} bins with mean_nfw_gtan < gtan_max",
                RuntimeWarning
            )
        T = T[mask]
        D = D[mask]
        D_cross = D_cross[mask]
        errbar_gcross = errbar_gcross[mask]
        PSF = PSF[mask]
        PSF_cross = PSF_cross[mask]
        errbar = errbar[mask]

    alpha = np.nan
    sig_alpha = np.nan
    corr = np.nan
    alpha_cross, sig_alpha_cross = np.nan, np.nan
    cbias, sig_c = np.nan, np.nan
    
    alpha_cross, sig_alpha_cross = alpha_from_psf_cross(PSF_cross, D_cross, errbar_gcross)
    
    if include_psf_leakage:
        zeta, alpha, sig_zeta, sig_alpha, corr = estimate_zeta_alpha(T, PSF, D, errbar)
        print(f"correlation between zeta and alpha: {np.sqrt(corr)}")
        #zeta, alpha, sig_zeta, sig_alpha, corr = estimate_zeta_alpha_diag(T, D, errbar)
    
    elif include_constant:
        zeta, alpha, cbias, sig_zeta, sig_alpha, sig_c, corr_za, corr_zc, corr_ac = estimate_zeta_alpha_c(T, PSF, D, errbar)
        
    else:
        print("Using standard estimation method.")
        # Compute covariance, alpha, and uncertainty
        Cinv = np.diag(1.0 / errbar**2)
        numer = T.T.dot(Cinv).dot(D)
        denom = T.T.dot(Cinv).dot(T)
        zeta = numer / denom

        # sig_alpha: Cramer-Rao bound uncertainty on alpha
        sig_zeta = 1.0 / np.sqrt(denom)

    return zeta, alpha, cbias, alpha_cross, sig_zeta, sig_alpha, sig_c, sig_alpha_cross,  r_at_gtan_max


def estimate_zeta_alpha_v2(gamma_true, gamma_psf, gamma_hat, sigma):
    """
    ML / weighted-least-squares fit of
        gamma_hat = zeta * gamma_true + alpha * gamma_psf + noise
    for independent Gaussian per-bin errors sigma_k, via scipy.curve_fit.
    """
    def model(X, zeta, alpha):
        gt, gp = X
        return zeta * gt + alpha * gp

    X = np.vstack([gamma_true, gamma_psf])
    popt, pcov = curve_fit(model, X, gamma_hat,
                           p0=[1.0, 0.0],         # zeta ~ 1, alpha ~ 0
                           sigma=sigma,
                           absolute_sigma=True)   # treat sigma as true errors

    zeta, alpha = popt
    sig_zeta, sig_alpha = np.sqrt(np.diag(pcov))
    corr = pcov[0, 1] / (sig_zeta * sig_alpha)
    return zeta, alpha, sig_zeta, sig_alpha, corr

def estimate_zeta_alpha(gamma_true, gamma_psf, gamma_hat, sigma):
    """
    ML / weighted-least-squares fit of
        gamma_hat = zeta * gamma_true + alpha * gamma_psf + noise
    for independent Gaussian per-bin errors sigma_k.
    Your original function is the special case gamma_psf = ones_like(...).
    """
    w = 1.0 / sigma**2

    Sxx = np.sum(w * gamma_true**2)
    Szz = np.sum(w * gamma_psf**2)
    Sxz = np.sum(w * gamma_true * gamma_psf)
    Sxy = np.sum(w * gamma_true * gamma_hat)
    Szy = np.sum(w * gamma_psf  * gamma_hat)

    Delta = Sxx * Szz - Sxz**2

    zeta  = (Szz * Sxy - Sxz * Szy) / Delta
    alpha = (Sxx * Szy - Sxz * Sxy) / Delta

    var_zeta  = Szz / Delta
    var_alpha = Sxx / Delta
    cov_za    = -Sxz / Delta

    corr = cov_za / np.sqrt(var_zeta * var_alpha)
    return zeta, alpha, np.sqrt(var_zeta), np.sqrt(var_alpha), corr


def estimate_zeta_alpha_c(gamma_true, gamma_psf, gamma_hat, sigma):
    """
    ML / weighted-least-squares fit of
        gamma_hat = zeta * gamma_true + alpha * gamma_psf + c + noise
    for independent Gaussian per-bin errors sigma_k.
    The two-parameter estimate_zeta_alpha is the special case where the
    constant column is dropped.
    """
    w = 1.0 / sigma**2

    # weighted design moments  S_ab = sum_k w_k a_k b_k   (u_k = 1)
    Sxx = np.sum(w * gamma_true**2)
    Szz = np.sum(w * gamma_psf**2)
    Suu = np.sum(w)
    Sxz = np.sum(w * gamma_true * gamma_psf)
    Sxu = np.sum(w * gamma_true)
    Szu = np.sum(w * gamma_psf)
    # right-hand side
    Sxy = np.sum(w * gamma_true * gamma_hat)
    Szy = np.sum(w * gamma_psf  * gamma_hat)
    Suy = np.sum(w * gamma_hat)

    # normal-equations matrix M = [[Sxx,Sxz,Sxu],[Sxz,Szz,Szu],[Sxu,Szu,Suu]]
    a, b, c = Sxx, Sxz, Sxu
    d, e, f = Szz, Szu, Suu

    Delta = a*(d*f - e**2) - b*(b*f - e*c) + c*(b*e - d*c)

    # entries of M^{-1} (= parameter covariance), symmetric cofactors
    Izz = (d*f - e**2) / Delta          # var_zeta
    Iza = (c*e - b*f) / Delta           # cov(zeta, alpha)
    Izc = (b*e - c*d) / Delta           # cov(zeta, c)
    Iaa = (a*f - c**2) / Delta          # var_alpha
    Iac = (b*c - a*e) / Delta           # cov(alpha, c)
    Icc = (a*d - b**2) / Delta          # var_c

    # theta = M^{-1} @ rhs
    zeta  = Izz*Sxy + Iza*Szy + Izc*Suy
    alpha = Iza*Sxy + Iaa*Szy + Iac*Suy
    cbias = Izc*Sxy + Iac*Szy + Icc*Suy

    sig_zeta  = np.sqrt(Izz)
    sig_alpha = np.sqrt(Iaa)
    sig_c     = np.sqrt(Icc)

    corr_za = Iza / (sig_zeta  * sig_alpha)
    corr_zc = Izc / (sig_zeta  * sig_c)
    corr_ac = Iac / (sig_alpha * sig_c)

    return (zeta, alpha, cbias,
            sig_zeta, sig_alpha, sig_c,
            corr_za, corr_zc, corr_ac)
    
    
def alpha_from_psf_cross(gamma_psf, gamma_hat, sigma):
    Cinv = np.diag(1.0 / sigma**2)
    numer = gamma_psf.T.dot(Cinv).dot(gamma_hat)
    denom = gamma_psf.T.dot(Cinv).dot(gamma_psf)
    alpha = numer / denom

    # sig_alpha: Cramer-Rao bound uncertainty on alpha
    sig_alpha = 1.0 / np.sqrt(denom)
    
    return alpha, sig_alpha

def estimate_zeta_alpha_diag(gamma_true, gamma_hat, sigma):
    """
    Weighted least-squares fit of  gamma_hat = zeta * gamma_true + alpha * gamma_psf
    for diagonal (uncorrelated) bin errors.

    gamma_true, gamma_hat, sigma : (K,) arrays
        sigma = per-bin 1-sigma error (sqrt of the diagonal of C)
    """
    w = 1.0 / sigma**2                 # per-bin weights

    a = np.sum(w * gamma_true**2)
    b = np.sum(w * gamma_true)
    d = np.sum(w)
    p = np.sum(w * gamma_true * gamma_hat)
    q = np.sum(w * gamma_hat)

    Delta = a * d - b**2

    zeta  = (d * p - b * q) / Delta
    alpha = (a * q - b * p) / Delta

    var_zeta  = d / Delta
    var_alpha = a / Delta
    cov_za    = -b / Delta

    return (zeta, alpha,
            np.sqrt(var_zeta), np.sqrt(var_alpha),
            cov_za / np.sqrt(var_zeta * var_alpha))   # last term = correlation