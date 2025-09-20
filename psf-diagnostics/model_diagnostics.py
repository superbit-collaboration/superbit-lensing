import os
import glob
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
# import astropy.io.fits
# import astropy.units
import numpy as np
from datetime import datetime
import glob
import re
# import warnings
import ngmix
import filecmp
from superbit_lensing.utils import gaia_query, make_psfex_model, radec_to_xy, extract_vignette, add_admom_columns
from superbit_lensing.match import SkyCoordMatcher
import multiprocessing

DATA_DIR = "/scratch/sa.saha/data"
CODE_DIR = "/projects/mccleary_group/saha/codes/superbit-lensing"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_PLOT_DIR = os.path.join(DATA_DIR, f"plots_mag_t_{timestamp}")

clusters = [
    "Abell3526",
    "Abell2163",
    "SMACSJ2031d8m4036",
    "MS1008d1m1224",
    "Abell2384a",
    # "MACSJ1931d8m2635",
    "RXCJ1514d9m1523",
    "Abell2345",
    # "SPTCLJ0411",
    "AbellS780",
    "AbellS0592",
    "Abell3411",
    "1E0657_Bullet",
    "Abell141",
    "Abell1689",
    "Abell2384b",
    "Abell3365",
    "Abell3571",
    "Abell3716S",
    "Abell3827",
    "COSMOS113",
    "COSMOSa",
    "COSMOSb",
    "COSMOSg",
    "COSMOSo",
    "MACSJ0723d3_7327_JWST",
    "MACSJ1105d7m1014",
    "MS2137m2353",
    "PLCKG287d0p32d9",
    "RXCJ1314d4m2515",
    "RXCJ2003d5m2323",
    "Z20_SPT_CLJ0135m5904",
    "Abell3192",
    "Abell3667",
    "ACT_CL_J0012_0855_J0012_0857",
    "COSMOSk",
    "MACSJ0416d1m2403",
    "RXJ1347d5m1145"
]

def process_cluster(cluster):
    cluster_seed = hash(cluster) % (2**32)  # Convert cluster name to seed
    np.random.seed(cluster_seed)
    rng = np.random.RandomState(cluster_seed)
    cluster_tables = []
    tolerance_deg = 1e-4
    config_path = os.path.join(CODE_DIR, "superbit_lensing/medsmaker/superbit/astro_config/")
    base_dir = DATA_DIR
    base_plot_dir = BASE_PLOT_DIR

    os.makedirs(base_plot_dir, exist_ok=True)

    cluster_dir = os.path.join(base_plot_dir, cluster.replace(" ", "_"))
    os.makedirs(cluster_dir, exist_ok=True)

    em_pars={'tol': 1.0e-6, 'maxiter': 50000}

    print(f"\nProcessing cluster: {cluster}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cat_folder = os.path.join(base_dir, cluster, "b", "cat")
    cal_folder = os.path.join(base_dir, cluster, "b", "cal")
    cat_files = sorted(glob.glob(os.path.join(cat_folder, "*_clean_cat.fits")))
    if cluster=="MACSJ1931d8m2635":
        cat_files = cat_files[:36]
    diag_folder = os.path.join(base_dir, cluster, "b", "psf_diags")
    os.makedirs(diag_folder, exist_ok=True)

    print(f"Found {len(cat_files)} exposure files")
    print(f"Looking stars for {cluster}...")
    star_file = os.path.join(base_dir, "stars" ,f"{cluster}_gaia_dr3.fits")
    try:
        gaia_stars = Table.read(star_file)
    except Exception as e:
        print(f'[WARNING] Could not open star file : {e}, doing a fresh new query')
        gaia_stars = gaia_query(cluster)
        try:
            gaia_stars.write(star_file, format='fits', overwrite=True)
            print(f'[INFO] Saved the queried file to {star_file}')
        except Exception as e:
            print(f'[WARNING] could not save it to fits file for future use')

    print(f"Found {len(gaia_stars)} GAIA stars")


    for exp_idx, cat_name in enumerate(cat_files):
        print(f"\n  [{exp_idx+1}/{len(cat_files)}] Processing exposure: {os.path.basename(cat_name)}")
        basename = os.path.basename(cat_name).replace('_clean_cat.fits', '')
        sub_name = os.path.join(cal_folder, basename+'_clean.sub.fits')
        cat = add_admom_columns(cat_name, mode="galsim")
        ss_fits = fits.open(cat_name)
        if len(ss_fits) == 3:
            # It is an ldac
            ext = 2
        else:
            ext = 1
        cat = ss_fits[ext].data
        print(f"    Catalog contains {len(cat)} objects")
        
        print(f"    Matching with GAIA stars...")
        matcher = SkyCoordMatcher(cat, gaia_stars,
                                  cat1_ratag='ALPHAWIN_J2000', cat1_dectag='DELTAWIN_J2000',
                                  cat2_ratag='ALPHAWIN_J2000', cat2_dectag='DELTAWIN_J2000',
                                  return_idx=True, match_radius=1 * tolerance_deg)

        all_stars, matched2, idx1, idx2 = matcher.get_matched_pairs()

        print(f"    Matched {len(all_stars)} stars")
        # Initial cuts
        valid = (all_stars['MAG_AUTO'] > 16.5) & (all_stars['SNR_WIN'] > 20)

        # Filter, sigma-clip T_ADMOM, and remove high-ellipticity stars in one go
        T_clipped = sigma_clip(all_stars['T_ADMOM'][valid], sigma=2.5, maxiters=None)
        # ellipticity = np.sqrt(all_stars['E1_ADMOM'][valid]**2 + all_stars['E2_ADMOM'][valid]**2)
        good_stars = all_stars[valid] #[~T_clipped.mask]
        bad_stars = all_stars[~valid]

        print(f"    Selected {len(good_stars)} good stars after T cuts")

        N = len(good_stars)
        indices = np.random.permutation(N)

        # Compute split point
        split = int(0.8 * N)

        # Split into train/test sets
        train_stars = good_stars[indices[:split]]
        test_stars = good_stars[indices[split:]]
        psf_catname = os.path.join(diag_folder, basename+'_clean_starcat.fits')
        ss_fits[ext].data = train_stars
        ss_fits.writeto(psf_catname, overwrite=True)

        model = make_psfex_model(psf_catname, config_path, overwrite=True)
        test_stars = Table(test_stars)
        bad_stars = Table(bad_stars)
        test_stars = vstack([test_stars, bad_stars])

        train_stars = Table(train_stars)
        print(f"    → Final train stars: {len(train_stars)}")
        print(f"    → Final test stars: {len(test_stars)} (includes {len(bad_stars)} 'bad' stars)")        
        train_stars['CLUSTER_NAME'] = cluster
        train_stars['EXP_NUM'] = exp_idx + 1
        train_stars['SET'] = 'train'

        test_stars['CLUSTER_NAME'] = cluster
        test_stars['EXP_NUM'] = exp_idx + 1
        test_stars['SET'] = 'test'
        test_stars_name = os.path.join(diag_folder, basename+'_clean_test_stars.fits')
        test_stars.write(test_stars_name, overwrite=True)

        with fits.open(sub_name) as hdul:
            bg_image_data = hdul[0].data
            bg_header = hdul[0].header
        wcs_header = WCS(bg_header)
        pixel_scales_deg = proj_plane_pixel_scales(wcs_header)

        # Convert to arcsec/pixel
        pixel_scales_arcsec = pixel_scales_deg * 3600  # 1 degree = 3600 arcseconds
        scale = np.mean(pixel_scales_arcsec)

        print(f"    Pixel scale: {scale:.3f} arcsec/pixel")

        # ---- Admom shape measurement for train_stars ----
        print(f"    Measuring shapes for {len(train_stars)} train stars...")
        n_objects = len(train_stars)
        e1_admom_obs = np.full(n_objects, np.nan)
        e2_admom_obs = np.full(n_objects, np.nan)
        T_admom_obs = np.full(n_objects, np.nan)

        e1_admom_model = np.full(n_objects, np.nan)
        e2_admom_model = np.full(n_objects, np.nan)
        T_admom_model = np.full(n_objects, np.nan)

        e1_em_obs = np.full(n_objects, np.nan)
        e2_em_obs = np.full(n_objects, np.nan)
        T_em_obs = np.full(n_objects, np.nan)

        e1_em_model = np.full(n_objects, np.nan)
        e2_em_model = np.full(n_objects, np.nan)
        T_em_model = np.full(n_objects, np.nan)

        am = ngmix.admom.AdmomFitter()
        em = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=5)

        for i in range(n_objects):
            if i % 200 == 0:
                print(f"      Progress: {i}/{n_objects} stars processed")
            x_image, y_image, ra, dec = train_stars[i]["XWIN_IMAGE"], train_stars[i]['YWIN_IMAGE'], train_stars[i]["ALPHAWIN_J2000"], train_stars[i]["DELTAWIN_J2000"]
            center = model.get_center(y_image, x_image)
            psf_model_im = model.get_rec(y_image, x_image)
            xim, yim = radec_to_xy(bg_header, ra, dec)
            star_im, meta_im = extract_vignette(bg_image_data, bg_header, xim, yim, size=51)
            jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=center[1], col=center[0])
            norm = np.sum(star_im[star_im > 0])
            if norm == 0:
                continue
            obs_im = ngmix.Observation(image=(star_im / norm), jacobian=jac)
            obs_model = ngmix.Observation(image=psf_model_im, jacobian=jac)

            try:
                res_obs = am.go(obs_im, guess=0.5)
                res_model = am.go(obs_model, guess=0.5)
                # res_obs = get_admoms(star_im, scale=scale, mode="galsim", reduced=False)
                # res_model = get_admoms(psf_model_im, scale=scale, mode="galsim", reduced=False)

                # res_em_obs = em.go(obs=obs_im, guess=psf_guesser._get_guess(obs_im))
                # res_em_model = em.go(obs=obs_model, guess=psf_guesser._get_guess(obs_model))
                if res_obs["flags"]==0:
                    e1_admom_obs[i] = res_obs['e1']
                    e2_admom_obs[i] = res_obs['e2']
                    T_admom_obs[i] = res_obs['T']
                if res_model["flags"]==0:
                    e1_admom_model[i] = res_model['e1']
                    e2_admom_model[i] = res_model['e2']
                    T_admom_model[i] = res_model['T']

                # if res_em_obs['flags'] == 0:
                #     e1e2T_obs = res_em_obs.get_gmix().get_e1e2T()
                #     e1_em_obs[i] = e1e2T_obs[0]
                #     e2_em_obs[i] = e1e2T_obs[1]
                #     T_em_obs[i] = e1e2T_obs[2]

                # if res_em_model['flags'] == 0:
                #     e1e2T_model = res_em_model.get_gmix().get_e1e2T()
                #     e1_em_model[i] = e1e2T_model[0]
                #     e2_em_model[i] = e1e2T_model[1]
                #     T_em_model[i] = e1e2T_model[2]                                    

            except Exception as e:
                # leave values as NaN
                continue    

        train_stars["e1_admom_obs"] = e1_admom_obs
        train_stars["e2_admom_obs"] = e2_admom_obs
        train_stars["T_admom_obs"] = T_admom_obs
        train_stars['fwhm_obs'] =  2.355 * np.sqrt(T_admom_obs / 2)
        train_stars["e1_admom_model"] = e1_admom_model
        train_stars["e2_admom_model"] = e2_admom_model
        train_stars["T_admom_model"] = T_admom_model
        train_stars['fwhm_model'] =  2.355 * np.sqrt(T_admom_model / 2)

        train_stars['e1_em_obs'] = e1_em_obs
        train_stars['e2_em_obs'] = e2_em_obs
        train_stars['T_em_obs'] = T_em_obs
        train_stars['e1_em_model'] = e1_em_model
        train_stars['e2_em_model'] = e2_em_model
        train_stars['T_em_model'] = T_em_model


        # ---- Admom shape measurement for test_stars ----
        print(f"    Measuring shapes for {len(test_stars)} test stars...")
        n_test = len(test_stars)
        e1_admom_obs_test = np.full(n_test, np.nan)
        e2_admom_obs_test = np.full(n_test, np.nan)
        T_admom_obs_test = np.full(n_test, np.nan)

        e1_admom_model_test = np.full(n_test, np.nan)
        e2_admom_model_test = np.full(n_test, np.nan)
        T_admom_model_test = np.full(n_test, np.nan)

        e1_em_obs = np.full(n_test, np.nan)
        e2_em_obs = np.full(n_test, np.nan)
        T_em_obs = np.full(n_test, np.nan)

        e1_em_model = np.full(n_test, np.nan)
        e2_em_model = np.full(n_test, np.nan)
        T_em_model = np.full(n_test, np.nan)

        am = ngmix.admom.AdmomFitter()
        em = ngmix.em.EMFitter(maxiter=em_pars['maxiter'], tol=em_pars['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=5)
        for i in range(n_test):
            if i % 200 == 0:
                print(f"      Progress: {i}/{n_test} stars processed")
            x_image = test_stars[i]["XWIN_IMAGE"]
            y_image = test_stars[i]["YWIN_IMAGE"]
            ra = test_stars[i]["ALPHAWIN_J2000"]
            dec = test_stars[i]["DELTAWIN_J2000"]
            
            center = model.get_center(y_image, x_image)
            psf_model_im = model.get_rec(y_image, x_image)
            xim, yim = radec_to_xy(bg_header, ra, dec)
            star_im, meta_im = extract_vignette(bg_image_data, bg_header, xim, yim, size=51)
            
            jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=center[1], col=center[0])
            norm = np.sum(star_im[star_im > 0])
            if norm == 0:
                continue

            obs_im = ngmix.Observation(image=(star_im / norm), jacobian=jac)
            obs_model = ngmix.Observation(image=psf_model_im, jacobian=jac)

            try:
                res_obs = am.go(obs_im, guess=0.5)
                res_model = am.go(obs_model, guess=0.5)

                # res_em_obs = em.go(obs=obs_im, guess=psf_guesser._get_guess(obs_im))
                # res_em_model = em.go(obs=obs_model, guess=psf_guesser._get_guess(obs_model))

                e1_admom_obs_test[i] = res_obs['e1']
                e2_admom_obs_test[i] = res_obs['e2']
                T_admom_obs_test[i] = res_obs['T']

                e1_admom_model_test[i] = res_model['e1']
                e2_admom_model_test[i] = res_model['e2']
                T_admom_model_test[i] = res_model['T']

                # if res_em_obs['flags'] == 0:
                #     e1e2T_obs = res_em_obs.get_gmix().get_e1e2T()
                #     e1_em_obs[i] = e1e2T_obs[0]
                #     e2_em_obs[i] = e1e2T_obs[1]
                #     T_em_obs[i] = e1e2T_obs[2]

                # if res_em_model['flags'] == 0:
                #     e1e2T_model = res_em_model.get_gmix().get_e1e2T()
                #     e1_em_model[i] = e1e2T_model[0]
                #     e2_em_model[i] = e1e2T_model[1]
                #     T_em_model[i] = e1e2T_model[2]      
            except Exception:
                continue

        # Assign results to test_stars
        test_stars["e1_admom_obs"] = e1_admom_obs_test
        test_stars["e2_admom_obs"] = e2_admom_obs_test
        test_stars["T_admom_obs"] = T_admom_obs_test
        test_stars["fwhm_obs"] = 2.355 * np.sqrt(T_admom_obs_test / 2)

        test_stars["e1_admom_model"] = e1_admom_model_test
        test_stars["e2_admom_model"] = e2_admom_model_test
        test_stars["T_admom_model"] = T_admom_model_test
        test_stars["fwhm_model"] = 2.355 * np.sqrt(T_admom_model_test / 2)

        test_stars['e1_em_obs'] = e1_em_obs
        test_stars['e2_em_obs'] = e2_em_obs
        test_stars['T_em_obs'] = T_em_obs
        test_stars['e1_em_model'] = e1_em_model
        test_stars['e2_em_model'] = e2_em_model
        test_stars['T_em_model'] = T_em_model

        # After computing these:
        n_good_stars = len(good_stars)
        valid_fwhm = train_stars['fwhm_obs'][np.isfinite(train_stars['fwhm_obs'])]
        # Calculate percentiles for outlier removal
        lower_percentile = np.percentile(valid_fwhm, 0.5)
        upper_percentile = np.percentile(valid_fwhm, 99.5)

        # Filter to remove outliers
        fwhm_no_outliers = valid_fwhm[(valid_fwhm >= lower_percentile) & (valid_fwhm <= upper_percentile)]

        # Calculate statistics on filtered data
        median_fwhm = np.median(fwhm_no_outliers)
        std_fwhm = np.std(fwhm_no_outliers)

        # Assign to both train and test stars
        for tbl in [train_stars, test_stars]:
            tbl['N_GOOD_STARS'] = n_good_stars
            tbl['FWHM_MEDIAN'] = median_fwhm
            tbl['FWHM_STD'] = std_fwhm

        all_stars = vstack([train_stars, test_stars])
        
        # -------------- Plot --------------------------
        plt.figure(figsize=(10, 8))
        plt.scatter(test_stars['MAG_AUTO'] - 1.33, test_stars["T_admom_obs"], s=5, alpha=0.5, 
                    color='blue', label='Test Stars')
        plt.scatter(train_stars['MAG_AUTO'] - 1.33, train_stars["T_admom_obs"], 
                    s=8, alpha=0.7, color='red', label='Train Stars', edgecolors='darkred', linewidth=0.5)
        
        plt.xlabel("MAG_AUTO", fontsize=12)
        plt.ylim([0, 1.2])
        plt.xlim([10, 23])
        plt.ylabel("T_admom (arcsec²)", fontsize=12)
        plt.title(f"MAG_AUTO vs T_admom: Target- {cluster}, Exposure {exp_idx+1}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        plt.tight_layout()

        # Save with cluster and exposure in filename
        plotfile = os.path.join(cluster_dir, f"{cluster}_exp{exp_idx+1}_mag_vs_t.png")
        plt.savefig(plotfile, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: {plotfile}")
        # --------------- Plot --------------------------

        cluster_tables.append(all_stars)
    if cluster_tables:
        return vstack(cluster_tables)
    else:
        return None

def main():
    print("=== SLURM Environment Variables ===", flush=True)
    for key, value in os.environ.items():
        if key.startswith('SLURM'):
            print(f"{key}: {value}", flush=True)
    print("===================================", flush=True)
    
    # Get SLURM allocation
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_processes = int(os.environ['SLURM_CPUS_PER_TASK'])
    elif 'SLURM_NTASKS' in os.environ:
        n_processes = int(os.environ['SLURM_NTASKS'])
    else:
        n_processes = os.cpu_count()
    
    n_processes = min(len(clusters), n_processes)
    
    print(f"========================================", flush=True)
    print(f"Processing {len(clusters)} clusters using {n_processes} processes", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"========================================", flush=True)
    
    completed = 0
    results = []

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add timestamp to your filenames
    output_path = os.path.join(DATA_DIR, f"all_clusters_star_catalog_withem_{timestamp}.fits")
    output_path_2 = os.path.join(DATA_DIR, f"all_clusters_star_catalog_withem_{timestamp}_temp.fits")
    
    temp_path = output_path_2 + ".tmp"
    
    # Check if we already have partial results from a previous run
    
    with multiprocessing.Pool(processes=n_processes) as pool:
        # imap_unordered returns results as they complete
        for result in pool.imap_unordered(process_cluster, clusters):
            completed += 1
            
            if result is not None:
                results.append(result)
                
                # Save after each successful cluster
                try:
                    final_catalog = vstack(results)
                    
                    # Write to temp file first (safer)
                    final_catalog.write(temp_path, format='fits', overwrite=True)
                    
                    # Atomic rename
                    os.rename(temp_path, output_path_2)
                    
                    print(f"[MAIN] Progress: {completed}/{len(clusters)} clusters completed ({100*completed/len(clusters):.1f}%) - Temp catalog saved with {len(final_catalog)} stars", flush=True)
                    
                except Exception as e:
                    print(f"[WARNING] Failed to save intermediate results: {e}", flush=True)
                    print(f"[MAIN] Progress: {completed}/{len(clusters)} clusters completed ({100*completed/len(clusters):.1f}%)", flush=True)
            else:
                print(f"[MAIN] Progress: {completed}/{len(clusters)} clusters completed ({100*completed/len(clusters):.1f}%) - Cluster returned None", flush=True)
    
    print(f"========================================", flush=True)
    print(f"All processing complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # Final save with summary
    results = [r for r in results if r is not None]
    
    if results:
        final_catalog = vstack(results)
        final_catalog.write(output_path, overwrite=True)
        print(f"Final catalog saved: {output_path}")
        print(f"Temp catalog available at: {output_path_2}")
        print(f"Total stars in catalog: {len(final_catalog)}")
        print(f"Successfully processed: {len(results)}/{len(clusters)} clusters")
    else:
        print("No data processed.")

    # Check if files exist
    if os.path.exists(output_path) and os.path.exists(output_path_2):
        # Compare binary contents
        if filecmp.cmp(output_path, output_path_2, shallow=False):
            print("Files are identical. Deleting temp file...")
            os.remove(output_path_2)
        else:
            print("Files differ. Keeping both.")
    else:
        print("One or both files do not exist.")

if __name__ == '__main__':
    main()