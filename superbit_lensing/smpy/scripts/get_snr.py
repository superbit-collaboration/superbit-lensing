import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import yaml
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from superbit_lensing.smpy import utils

# Command-line argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a shear catalog and compute the SNR map.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('--plot_kappa', action='store_true', help='Flag to plot convergence map')
    parser.add_argument('--plot_error', action='store_true', help='Flag to plot the std map')
    parser.add_argument('--save_fits', action='store_true', help='Flag to save the convergence map as a FITS file')  # New flag
    return parser.parse_args()

# Main script logic
if __name__ == "__main__":
    args = parse_arguments()
    config = read_config(args.config)

    # Load and preprocess shear data
    shear_df = utils.load_shear_data(
        config['input_path'], 
        config['ra_col'], 
        config['dec_col'], 
        config['g1_col'], 
        config['g2_col'], 
        config['weight_col'], 
        config['x_col'], 
        config['y_col']
    )
    
    print(f"Loaded shear data from {config['input_path']}")

    true_boundaries = utils.calculate_field_boundaries_v2(shear_df['ra'], shear_df['dec'])
    shear_df, ra_0, dec_0 = utils.correct_RA_dec(shear_df)
    boundaries = utils.calculate_field_boundaries_v2(shear_df['ra'], shear_df['dec'])
    boundaries_xy = utils.calculate_field_boundaries_v2(shear_df['x'], shear_df['y'])
    print(true_boundaries)

    box_boundary = config['box_boundary']
    if box_boundary is not None:
        box_boundary = correct_box_boundary(box_boundary, ra_0, dec_0)


    if config['center'] is not None:
        center_cl = correct_center(config["center"], ra_0, dec_0)
    else:
        center_cl = None

    x_factor, y_factor = (
        (np.max(shear_df['x']) - np.min(shear_df['x'])) / (np.max(shear_df['ra']) - np.min(shear_df['ra'])), 
        (np.max(shear_df['y']) - np.min(shear_df['y'])) / (np.max(shear_df['dec']) - np.min(shear_df['dec']))
    )
    factor = (x_factor + y_factor) / 2
    resolution_xy = int(config["resolution"] * factor)

    #print(shear_df['weight'])


    if config["gridding"] == "xy":
        g1map_og_2, g2map_og_2 = utils.create_shear_grid_v2(
            shear_df['x'], shear_df['y'], shear_df['g1'], shear_df['g2'], resolution_xy, shear_df['weight'], verbose=True
        )
        fits_filename = config['output_path'] + f"g1_{config['cluster']}_{config['band']}.fits"
        save_fits(g1map_og_2, true_boundaries, fits_filename)
        fits_filename = config['output_path'] + f"g2_{config['cluster']}_{config['band']}.fits"
        save_fits(g2map_og_2, true_boundaries, fits_filename)
        og_kappa_e_2, og_kappa_b_2 = kaiser_squires.ks_inversion(g1map_og_2, g2map_og_2, key='x-y')
    elif config["gridding"] == "ra_dec":
        g1map_og_2, g2map_og_2 = utils.create_shear_grid_v2(
            shear_df['ra'], shear_df['dec'], shear_df['g1'], shear_df['g2'], config["resolution"], shear_df['weight'], verbose=True
        )
        og_kappa_e_2, og_kappa_b_2 = kaiser_squires.ks_inversion(g1map_og_2, g2map_og_2, key='ra-dec')
    else:
        KeyError("Invalid gridding type. Must be either 'xy' or 'ra-dec'.")

    
    
    kernel = config['gaussian_kernel']
    og_kappa_e_2_smoothed = gaussian_filter(og_kappa_e_2, kernel)
    og_kappa_b_2_smoothed = gaussian_filter(og_kappa_b_2, kernel)


    if args.plot_kappa:
        plot_kmap.plot_convergence_v4(
            og_kappa_e_2_smoothed, 
            boundaries, 
            true_boundaries, 
            config, 
            invert_map=False, 
            title="Convergence: "+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
            vmax= config['kmap_vmax'],
            vmin= config['kmap_vmin'], 
            #threshold=config['threshold'],
            center_cl=center_cl,
            box_boundary=box_boundary,
            save_path=config['output_path']+"kappa_"+config['cluster']+"_"+config['band']+".png"
        )
        plot_kmap.plot_convergence_v4(
            og_kappa_b_2_smoothed, 
            boundaries, 
            true_boundaries, 
            config, 
            invert_map=False, 
            title="Convergence (b_modes): "+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
            vmax= config['kmap_vmax'],
            vmin= config['kmap_vmin'], 
            #threshold=config['threshold'],
            center_cl=center_cl,
            box_boundary=box_boundary,
            save_path=config['output_path']+"kappa_b_"+config['cluster']+"_"+config['band']+".png"
        )        
    if args.save_fits:
        fits_filename = config['output_path'] + f"kappa_{config['cluster']}_{config['band']}.fits"
        save_fits(og_kappa_e_2_smoothed, true_boundaries, fits_filename)
        fits_filename = config['output_path'] + f"kappa_b_{config['cluster']}_{config['band']}.fits"
        save_fits(og_kappa_b_2_smoothed, true_boundaries, fits_filename)
        count_grid = utils.create_count_grid(shear_df['x'], shear_df['y'], resolution_xy, verbose=True)
        fits_filename = config['output_path'] + f"count_{config['cluster']}_{config['band']}.fits"
        save_fits(count_grid, true_boundaries, fits_filename)
        fits_filename = config['output_path'] + f"error_{config['cluster']}_{config['band']}.fits"
        
    
    if config["shuffle_type"] == "rotation":
        shuffled_dfs = utilsv2.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    elif config["shuffle_type"] == "spatial":
        shuffled_dfs = utils.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    else:
        KeyError("Invalid shuffle type. Must be either 'rotation' or 'spatial'.")

    #shuffled_dfs = utilsv2.generate_multiple_shear_dfs(shear_df, config['num_sims'], seed=config['seed_sims'])
    if config["gridding"] == 'xy':
        g1_g2_map_list_xy = utils.shear_grids_for_shuffled_dfs_xy(shuffled_dfs, resolution_xy)
        shuff_kappa_e_list_xy, shuff_kappa_b_list_xy = utils.ks_inversion_list(g1_g2_map_list_xy, 'xy')

        kappa_e_stack_xy = np.stack(shuff_kappa_e_list_xy, axis=0)
        kappa_e_stack_smoothed_xy = np.zeros_like(kappa_e_stack_xy)
        for i in range(kappa_e_stack_xy.shape[0]):
            kappa_e_stack_smoothed_xy[i] = gaussian_filter(kappa_e_stack_xy[i], kernel)
    
        std_xy = np.std(kappa_e_stack_smoothed_xy, axis=0)
        if args.save_fits:
            save_fits(std_xy, true_boundaries, fits_filename)
        snr_xy = gaussian_filter(og_kappa_e_2, kernel) / std_xy
        if args.plot_error:
            plot_kmap.plot_convergence_v4(
                std_xy,
                boundaries,
                true_boundaries,
                config,
                invert_map=False,
                title="Error: "+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
                #vmax=config['vmax'],
                #threshold=config['threshold'],
                center_cl=center_cl,
                save_path=config['output_path']+"error_"+config['cluster']+"_"+config['band']+".png"
            )

        # Plotting SNR map
        plot_kmap.plot_convergence_v4(
            snr_xy, 
            boundaries, 
            true_boundaries, 
            config, 
            invert_map=False, 
            title=config['plot_title']+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
            vmax=config['vmax'], 
            vmin=config['vmin'],
            threshold=config['threshold'],
            center_cl=center_cl,
            save_path=config['output_path']+"snr_"+config['cluster']+"_"+config['band']+".png"
        )
    elif config["gridding"] == 'ra_dec':
        g1_g2_map_list = utils.shear_grids_for_shuffled_dfs(shuffled_dfs, config)
        shuff_kappa_e_list, shuff_kappa_b_list = utils.ks_inversion_list(g1_g2_map_list, key='ra_dec')
        kappa_e_stack = np.stack(shuff_kappa_e_list, axis=0)
        kappa_e_stack_smoothed = np.zeros_like(kappa_e_stack)
        for i in range(kappa_e_stack.shape[0]):
            kappa_e_stack_smoothed[i] = gaussian_filter(kappa_e_stack[i], kernel)
        std = np.std(kappa_e_stack_smoothed, axis=0)
        snr = gaussian_filter(og_kappa_e_2, kernel) / std
        plot_kmap.plot_convergence_v4(
            snr, 
            boundaries, 
            true_boundaries, 
            config, 
            invert_map=False, 
            title=config['plot_title']+config['cluster']+"_"+config['band'] + f" (Resolution: {config['resolution']:.2f} arcmin, Kernel: {kernel:.2f})",
            vmax=config['vmax'], 
            vmin=config['vmin'],
            threshold=config['threshold'],
            center_cl=center_cl,
            save_path=config['output_path']+"snr_"+config['cluster']+"_"+config['band']+".png"
        )    

    else:
        KeyError("Invalid gridding type. Must be either 'xy' or 'ra_dec'.")