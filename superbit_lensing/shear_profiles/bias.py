import numpy as np
from astropy.table import Table

def compute_shear_bias(profile_tab, col_prefix=None, vb=True):
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

    alpha, sig_alpha = _compute_shear_bias(profile_tab)

    if vb is True:
        print('# ')
        print(f'# shear bias is {alpha:.4f} +/- {sig_alpha:.3f}')
        print('# ')

    # add this information to profile_tab metadata
    alpha_col = 'alpha'
    sig_alpha_col = 'sig_alpha'
    if col_prefix is not None:
        alpha_col = f'{col_prefix}_{alpha_col}'
        sig_alpha_col = f'{col_prefix}_{sig_alpha_col}'
    profile_tab.meta.update({
        alpha_col: alpha,
        sig_alpha_col: sig_alpha
        })

    return

def _compute_shear_bias(profile_tab):
    '''
    Compute alpha & sig_alpha from a table or rec_array, but without the
    frills & metadata saving. Returns the values instead.
    '''

    try:
        T = profile_tab['mean_nfw_gtan']
        D = profile_tab['mean_gtan']
        errbar = profile_tab['err_gtan']

    except KeyError as kerr:
        print('Shear bias calculation:')
        print('required columns not found; check input names?')
        raise kerr

    # C = covariance, alpha = shear bias maximum likelihood estimator
    C = np.diag(errbar**2)
    numer = T.T.dot(np.linalg.inv(C)).dot(D)
    denom = T.T.dot(np.linalg.inv(C)).dot(T)
    alpha = numer/denom

    # sig_alpha: Cramer-Rao bound uncertainty on Ahat
    sig_alpha = 1. / np.sqrt((T.T.dot(np.linalg.inv(C)).dot(T)))

    return alpha, sig_alpha
