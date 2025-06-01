#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import argparse
from astropy.io import fits

def create_cluster_directory(cluster_name, data_dir="/projects/mccleary_group/superbit/union"):
    """
    Create a new cluster directory based on the template directory or create it manually if template doesn't exist.
    
    Parameters:
    -----------
    cluster_name : str
        Name of the cluster (e.g., Abell3411)
    data_dir : str
        Base directory for data
    
    Returns:
    --------
    tuple
        (Path to the newly created cluster directory, Path to the preliminary directory)
    """
    # Define paths
    template_dir = os.path.join(data_dir, "template")
    cluster_dir = os.path.join(data_dir, cluster_name)
    
    # Check if the cluster directory already exists
    if os.path.exists(cluster_dir):
        print(f"Directory already exists: {cluster_dir}")
    else:
        print(f"Creating directory: {cluster_dir}")
        
        # Check if template directory exists
        if os.path.exists(template_dir):
            # Copy template directory structure
            print(f"Copying structure from template directory: {template_dir}")
            shutil.copytree(template_dir, cluster_dir)
        else:
            # Create directory structure manually
            print("Template directory not found. Creating directory structure manually.")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Create band directories
            for band in ['u', 'b', 'g']:
                for subdir in ['cal', 'cat', 'coadd', 'out']:
                    path = os.path.join(cluster_dir, band, subdir)
                    os.makedirs(path, exist_ok=True)
                    print(f"Created: {path}")
            
            # Create detection directories
            for subdir in ['cat', 'coadd']:
                path = os.path.join(cluster_dir, 'det', subdir)
                os.makedirs(path, exist_ok=True)
                print(f"Created: {path}")
    
    # Create preliminary directory for downloading data
    prelim_dir = os.path.join(cluster_dir, "preliminary")
    os.makedirs(prelim_dir, exist_ok=True)
    
    return cluster_dir, prelim_dir

def download_data(cluster_name, prelim_dir, username=None):
    """
    Download data from the hen server for the given cluster.
    
    Parameters:
    -----------
    cluster_name : str
        Name of the cluster (e.g., Abell3411)
    prelim_dir : str
        Directory where data will be initially downloaded
    username : str, optional
        Username for hen.astro.utoronto.ca. If None, will prompt user for input.
    """
    # Change to the preliminary directory
    original_dir = os.getcwd()
    os.chdir(prelim_dir)
    
    # If username is not provided, ask for it interactively
    if username is None:
        username = input("Enter your username for hen.astro.utoronto.ca: ")
    
    # Download data using scp with full hostname and username
    print(f"Downloading data for cluster '{cluster_name}' from hen.astro.utoronto.ca...")
    print("You will be prompted for your password.")
    
    # Full server path with username
    server_path = f"{username}@hen.astro.utoronto.ca:/data/analysis/superbit_2023/cleaned_images/{cluster_name}*"
    
    # Using subprocess.run with shell=False for better security and handling
    scp_command = ["scp", server_path, "."]
    
    try:
        # Pass-through stdin, stdout, and stderr so the user can enter credentials
        process = subprocess.run(
            scp_command, 
            shell=False, 
            check=True,
            stdin=sys.stdin,  # Allow input for password
            stdout=sys.stdout,  # Show download progress
            stderr=sys.stderr   # Show any errors
        )
        print("Data download completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        os.chdir(original_dir)
        return False
    
    # Return to original directory
    os.chdir(original_dir)
    return True


def filter_fits_files(directory):
    """
    Filters FITS files based on IMG_QUAL header value.
    
    Parameters:
    -----------
    directory : str
        Directory containing FITS files
    
    Returns:
    --------
    tuple
        Paths to good and bad files directories
    """
    # Create directories to separate files
    good_dir = os.path.join(directory, 'GOOD_FILES')
    bad_dir = os.path.join(directory, 'BAD_FILES')
    
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    
    # Initialize counters
    good_count = 0
    bad_count = 0
    error_count = 0
    
    # Change to the directory
    original_dir = os.getcwd()
    os.chdir(directory)
    
    # Iterate through all FITS files in the directory
    for file in os.listdir('.'):
        if file.endswith('.fits'):
            filepath = os.path.abspath(file)
            try:
                with fits.open(filepath) as hdul:
                    header = hdul[0].header  # Access primary header
                    img_qual = header.get('IMG_QUAL', 'BAD')  # Default to 'BAD' if not present
                    if img_qual == 'GOOD':
                        shutil.move(filepath, os.path.join(good_dir, file))
                        good_count += 1
                    else:
                        shutil.move(filepath, os.path.join(bad_dir, file))
                        bad_count += 1
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                error_count += 1
    
    # Print summary
    print(f"\nFile filtering complete:")
    print(f"  Good files: {good_count}")
    print(f"  Bad files: {bad_count}")
    if error_count > 0:
        print(f"  Errors: {error_count}")
    print(f"  Total processed: {good_count + bad_count}")
    
    # Return to original directory
    os.chdir(original_dir)
    return good_dir, bad_dir

def delete_bad_files(bad_dir):
    """
    Delete files in the BAD_FILES directory.
    
    Parameters:
    -----------
    bad_dir : str
        Directory containing bad FITS files
    """
    print(f"Deleting files in {bad_dir}...")
    for file in os.listdir(bad_dir):
        file_path = os.path.join(bad_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    # Remove the empty directory
    try:
        os.rmdir(bad_dir)
        print(f"Removed directory: {bad_dir}")
    except:
        print(f"Could not remove directory: {bad_dir}")

def organize_by_band(good_dir, cluster_dir, cluster_name):
    """
    Organize files by band (u, b, g).
    
    Parameters:
    -----------
    good_dir : str
        Directory containing good FITS files
    cluster_dir : str
        Base directory for the cluster
    cluster_name : str
        Name of the cluster
    """
    # Create band directories if they don't exist
    u_dir = os.path.join(cluster_dir, "u", "cal")
    b_dir = os.path.join(cluster_dir, "b", "cal")
    g_dir = os.path.join(cluster_dir, "g", "cal")
    
    os.makedirs(u_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)
    os.makedirs(g_dir, exist_ok=True)
    
    # Initialize counters
    u_count = 0
    b_count = 0
    g_count = 0
    unknown_count = 0
    
    # Move files to appropriate directories
    for file in os.listdir(good_dir):
        file_path = os.path.join(good_dir, file)
        
        if not os.path.isfile(file_path):
            continue
        
        # Band 0 (u)
        if f"{cluster_name}_0" in file:
            shutil.move(file_path, os.path.join(u_dir, file))
            u_count += 1
        # Band 1 (b)
        elif f"{cluster_name}_1" in file:
            shutil.move(file_path, os.path.join(b_dir, file))
            b_count += 1
        # Band 2 (g)
        elif f"{cluster_name}_2" in file:
            shutil.move(file_path, os.path.join(g_dir, file))
            g_count += 1
        else:
            print(f"Warning: Could not determine band for {file}")
            unknown_count += 1
    
    # Print summary statistics
    print(f"\nBand organization complete:")
    print(f"  u-band exposures: {u_count}")
    print(f"  b-band exposures: {b_count}")
    print(f"  g-band exposures: {g_count}")
    if unknown_count > 0:
        print(f"  Unknown band: {unknown_count}")
    print(f"  Total organized: {u_count + b_count + g_count}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download and organize superbit data for a cluster.')
    parser.add_argument('cluster_name', type=str, help='Name of the cluster (e.g., Abell3411)')
    parser.add_argument('--username', type=str, help='Username for hen.astro.utoronto.ca')
    parser.add_argument('--data-dir', type=str, default="/projects/mccleary_group/superbit/union",
                        help='Base directory for data (default: /projects/mccleary_group/superbit/union)')
    
    args = parser.parse_args()
    
    try:
        # Create cluster directory
        print(f"Processing cluster: {args.cluster_name}")
        cluster_dir, prelim_dir = create_cluster_directory(args.cluster_name, args.data_dir)
        
        # Download data with username (could be None, will be handled in download_data)
        if not download_data(args.cluster_name, prelim_dir, args.username):
            print("Download failed. Exiting.")
            return 1
        
        # Filter FITS files
        print("Filtering FITS files...")
        good_dir, bad_dir = filter_fits_files(prelim_dir)
        
        # Delete bad files
        delete_bad_files(bad_dir)
        
        # Organize files by band
        print("Organizing files by band...")
        organize_by_band(good_dir, cluster_dir, args.cluster_name)
        
        print("Remove Preliminary Directory..")
        shutil.rmtree(prelim_dir)

        print(f"Successfully processed data for cluster: {args.cluster_name}")
        print(f"Data organized in: {cluster_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())