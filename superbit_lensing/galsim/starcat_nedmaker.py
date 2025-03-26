import argparse
from astropy.io import fits
from astropy.table import Table
import os
import yaml

def process_fits(run_name, band, datadir, codedir):
    fname = f"{datadir}/{run_name}/{band}/cal/{run_name}_truth_{band}.fits"
    ned_outname = f"{datadir}/catalogs/redshifts/{run_name}_NED_redshifts.csv"
    starcat_outname = f"{datadir}/catalogs/stars/{run_name}_gaia_starcat.fits"
    config_path = f"{codedir}/superbit_lensing/medsmaker/configs/{run_name}_{band}_starparams.yaml"

    with fits.open(fname) as hdul:
        data = hdul[1].data  # Access table data from HDU=1
        
        # Ensure 'obj_class' column exists
        if "obj_class" not in data.names:
            raise ValueError("'obj_class' column not found in the FITS file.")

        # Filter only 'star' and 'galaxy' objects
        star_mask = data["obj_class"] == "star"
        gal_mask = data["obj_class"] == "gal"
        filtered_data = data[star_mask]
        galaxy = data[gal_mask]

        # Rename columns for stars
        new_columns = []
        for col in hdul[1].columns:
            if col.name == "ra":
                new_columns.append(fits.Column(name="ALPHAWIN_J2000", array=filtered_data["ra"], format=col.format))
            elif col.name == "dec":
                new_columns.append(fits.Column(name="DELTAWIN_J2000", array=filtered_data["dec"], format=col.format))
            else:
                new_columns.append(fits.Column(name=col.name, array=filtered_data[col.name], format=col.format))

        # Create new table HDU with updated columns for stars
        new_hdu = fits.BinTableHDU.from_columns(new_columns, header=hdul[1].header)

        # Create a new HDUList: Keep HDU 0, add HDU 2 (filtered & renamed table)
        new_hdul = fits.HDUList([hdul[0], fits.ImageHDU(), new_hdu])  # HDU=1 is a placeholder ImageHDU

        # Save the new FITS file for stars
        new_hdul.writeto(starcat_outname, overwrite=True)

        # Process galaxy table for CSV output
        galaxy_table = Table()
        if "ra" in galaxy.dtype.names:
            galaxy_table["RA"] = galaxy["ra"]
        if "dec" in galaxy.dtype.names:
            galaxy_table["DEC"] = galaxy["dec"]
        if "redshift" in galaxy.dtype.names:
            galaxy_table["Redshift"] = galaxy["redshift"]

        # Save galaxy table as CSV
        galaxy_table.write(ned_outname, format="csv", overwrite=True)

    # Create config file
    config_data = {
        "MIN_MAG": 27,
        "MAX_MAG": 21,
        "MIN_SIZE": 2,
        "MAX_SIZE": 3.5,
        "MIN_SNR": 20,
        "MAG_KEY": "MAG_AUTO",
        "SIZE_KEY": "FWHM_IMAGE",
        "SNR_KEY": "SNR_WIN",
        "use_truthstars": True,
        "truth_filename": starcat_outname,
        "truth_ra_key": "ALPHAWIN_J2000",
        "truth_dec_key": "DELTAWIN_J2000",
        "cat_hdu": 2
    }

    with open(config_path, "w") as config_file:
        yaml.dump(config_data, config_file, default_flow_style=False, sort_keys=False)

    print(f"Filtered and updated files saved:\nStars: {starcat_outname}\nGalaxies: {ned_outname}\nConfig: {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FITS files for star and galaxy catalogs.")
    parser.add_argument("run_name", type=str, help="Name of the simulation run")
    parser.add_argument("band", type=str, help="Name of the band")
    parser.add_argument("datadir", type=str, default=os.getenv("DATADIR", ""), help="Path to the data directory")
    parser.add_argument("codedir", type=str, default=os.getenv("CODEDIR", ""), help="Path to the code directory")
    
    args = parser.parse_args()
    process_fits(args.run_name, args.band, args.datadir, args.codedir)
