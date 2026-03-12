from astropy.table import Table
from dustmaps.csfd import CSFDQuery
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.config import config
from dust_extinction.parameter_averages import G23
import numpy as np
import os


def deredden_catalog(
    catname,                # path to the input FITS catalog
    outname,                # path to output/corrected file
    ra_colname,         
    dec_colname,  
    band_names=None,         # list of band names  
    wavelengths=None,        # list of u b g superbit wavelengths, just set to None unless you don't want it to be [3950, 4760, 5970]     
    cluster=None,            # cluster name in cat file, if cluster = None this runs on the whole catalog 
    dust_map=False,          # You need to download the CSFD dust map in the same directory as whereever you call this function
                             # if you have it downloaded, put True. If not, leave it as False. 
    dust_map_dir='/projects/mccleary_group/amit.m/dust/dust/', 
    Rv=3.1                   # ratio of total to selective extinction
):

    # from jmac dust correction code, modified to: 
    # - add correction column instead of overwriting
    # - have global csfd directory 
    # - correct wavelngth units in place (no more params)


    # Read catalog as an astropy Table 
    print(f"Loading catalog: {catname}")
    tab = Table.read(catname)

    # Filter by cluster if specified
    if cluster is not None:
        print(f"Filtering for cluster: {cluster}")
        cluster_names = np.array([s.strip() for s in tab['CLUSTER']])
        tab = tab[cluster_names == cluster]
        print(f"  Found {len(tab)} objects in {cluster}")
    else:
        print(f"  Processing all {len(tab)} objects")

    # grab stuff
    ra = tab[ra_colname]
    dec = tab[dec_colname]
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    print("Got coordinates")

    # Get our E(B-V) values
    if dust_map == False:
        config['data_dir'] = dust_map_dir
    csfd = CSFDQuery()  # works either way 
    ebv = csfd(coords)
    print("Queried dust map")

    # Convert to Av values
    Av_values = ebv * Rv

    # Define and add units to wavelengths
    if wavelengths is None: 
        wavelengths = [3950, 4760, 5970]
    wavelengths_w_units = np.array(wavelengths) * u.AA

    # Do dust modeling
    gordon23 = G23(Rv=Rv)
    AxAv = gordon23(wavelengths_w_units)

    # Calculate offsets
    Ax = Av_values[:, np.newaxis] * AxAv
    Ax = Ax.T
    print("Obtained Ax for catalog")

    # Do corrected magnitudes
    print("Beginning magnitude corrections...")

    if band_names is None: 
        band_names = ['m_u', 'm_b', 'm_g']

    for i in range(len(band_names)):
        this_Ax = Ax[i,:]
        corr_band = np.array(tab[band_names[i]], dtype=float) - this_Ax
        key_name = f'{band_names[i]}_corr_csfd'
        tab[key_name] = corr_band
        print(f"  Corrected {band_names[i]} -> {key_name}")

    # Build output filename
    if outname is None:
        base, ext = os.path.splitext(catname)
        outname = f"{base}_dereddened{ext}"

    # Write new file
    print(f"Writing output to: {outname}")
    tab.write(outname, format='fits', overwrite=True)

    print("Done!")