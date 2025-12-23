import numpy as np
from astropy.table import Table
import warnings

def compute_shear_bias(profile_tab, gtan_max=None, col_prefix=None, vb=True):
    '''
    profile_tab: astropy.Table
        Profile table used to compute the shear bias
    col_prefix: str
        Prefix to add to "alpha" & "sig_alpha" saved in metadata
    vb: bool
        Set to True to turn on prints

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

    alpha, sig_alpha, r_at_gtan_max = _compute_shear_bias(profile_tab, gtan_max)

    if vb is True:
        print('# ')
        print(f'# shear bias is {alpha:.4f} +/- {sig_alpha:.3f}')
        print('# ')

    # Add this information to profile_tab metadata
    alpha_col = 'alpha'
    sig_alpha_col = 'sig_alpha'
    r_col = 'r_gtan_max'
    if col_prefix is not None:
        alpha_col = f'{col_prefix}_{alpha_col}'
        sig_alpha_col = f'{col_prefix}_{sig_alpha_col}'
        r_col = f'{col_prefix}_{r_col}'

    profile_tab.meta.update({
        alpha_col: alpha,
        sig_alpha_col: sig_alpha,
        r_col: r_at_gtan_max
    })

    return

def _compute_shear_bias(profile_tab, gtan_max=None):
    '''
    Compute alpha & sig_alpha from a table or rec_array.
    Optionally filter bins where mean_nfw_gtan < gtan_max.
    Returns (alpha, sig_alpha).
    '''

    try:
        T = profile_tab['mean_nfw_gtan']
        D = profile_tab['mean_gtan']
        errbar = profile_tab['err_gtan']
        midpoints_r = profile_tab['midpoint_r']

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
        errbar = errbar[mask]

    # Compute covariance, alpha, and uncertainty
    Cinv = np.diag(1.0 / errbar**2)
    numer = T.T.dot(Cinv).dot(D)
    denom = T.T.dot(Cinv).dot(T)
    alpha = numer / denom

    # sig_alpha: Cramer-Rao bound uncertainty on alpha
    sig_alpha = 1.0 / np.sqrt(denom)

    return alpha, sig_alpha, r_at_gtan_max
