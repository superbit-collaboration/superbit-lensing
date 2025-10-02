import os
import re
import subprocess
import argparse
import shutil

def normalize_path(path):
    """Normalize a path by removing trailing slashes and handling edge cases."""
    # Strip whitespace
    path = path.strip()
    
    # Remove trailing slashes, but preserve root '/'
    if path != '/':
        path = path.rstrip('/')
    
    # Normalize the path (resolves '..' and '.' components)
    path = os.path.normpath(path)
    
    return path

def get_shell_config_file():
    """Determine the appropriate shell configuration file."""
    shell = os.getenv('SHELL', '/bin/bash')
    home_dir = os.path.expanduser('~')
    
    if 'zsh' in shell:
        rc_file = os.path.join(home_dir, '.zshrc')
    elif 'bash' in shell:
        # Prefer .bashrc if it exists, otherwise use .bash_profile
        if os.path.exists(os.path.join(home_dir, '.bashrc')):
            rc_file = os.path.join(home_dir, '.bashrc')
        else:
            rc_file = os.path.join(home_dir, '.bash_profile')
    else:
        rc_file = os.path.join(home_dir, '.profile')  # Fallback for other shells
    
    return rc_file

def add_bit_download_alias(current_dir, datadir, username=None):
    """Add bit-download alias to shell configuration file."""
    rc_file = get_shell_config_file()
    
    # Construct the alias command
    script_path = os.path.join(current_dir, 'utility_scripts', 'data_downloader.py')
    
    if username:
        alias_cmd = f'alias bit-download="python {script_path} --data-dir {datadir} --username {username}"'
    else:
        alias_cmd = f'alias bit-download="python {script_path} --data-dir {datadir}"'
    
    # Check if alias already exists
    alias_exists = False
    if os.path.exists(rc_file):
        with open(rc_file, 'r') as f:
            content = f.read()
            if 'alias bit-download=' in content:
                alias_exists = True
    
    if alias_exists:
        print(f"\nbit-download alias already exists in {rc_file}")
        update = input("Do you want to update it with new settings? (yes/no): ").strip().lower()
        if update not in ['yes', 'y']:
            return
        
        # Remove old alias
        with open(rc_file, 'r') as f:
            lines = f.readlines()
        
        with open(rc_file, 'w') as f:
            for line in lines:
                if line.strip().startswith('alias bit-download='):
                    continue
                if line.strip() == '# SuperBIT data download alias (added by post_installation.py)':
                    continue
                f.write(line)

    
    # Add the alias
    try:
        with open(rc_file, 'a') as f:
            f.write(f"\n# SuperBIT data download alias (added by post_installation.py)\n")
            f.write(f"{alias_cmd}\n")
        
        print(f"\nAdded bit-download alias to {rc_file}")
        print("\nTo download data in the future, use:")
        print("  bit-download <cluster_name>")
        print("\nNote: You need to reload your shell or run:")
        print(f"  source {rc_file}")
        
    except Exception as e:
        print(f"\nError adding alias to {rc_file}: {e}")

def update_config_sh(config_file, datadir, codedir, env_name=None):
    """Update the DATADIR and CODEDIR in the config.sh file."""
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found.")
        return
    
    with open(config_file, 'r') as file:
        lines = file.readlines()
    
    updated_datadir = False
    updated_codedir = False
    
    with open(config_file, 'w') as file:
        for line in lines:
            # Update DATADIR line
            if line.strip().startswith('export DATADIR='):
                file.write(f'export DATADIR="{datadir}"\n')
                print(f"Updated DATADIR to: {datadir}")
                updated_datadir = True
            # Update CODEDIR line
            elif line.strip().startswith('export CODEDIR='):
                file.write(f'export CODEDIR="{codedir}"\n')
                print(f"Updated CODEDIR to: {codedir}")
                updated_codedir = True
            # Update CONDA_ENV line (if env_name provided)
            elif env_name is not None and line.strip().startswith('export CONDA_ENV='):
                file.write(f'export CONDA_ENV="{env_name}"\n')
                print(f"Updated CONDA_ENV to: {env_name}")
                updated_env = True
            else:
                file.write(line)

        # If env_name provided but no line existed, append it
        if env_name is not None and not updated_env:
            file.write(f'export CONDA_ENV="{env_name}"\n')
            print(f"Added CONDA_ENV: {env_name}")
    
    if not updated_datadir:
        print("Warning: DATADIR line not found in config.sh")
    if not updated_codedir:
        print("Warning: CODEDIR line not found in config.sh")

def update_truth_filename(yaml_file, datadir):
    """Update the truth_filename line in the given YAML file without modifying other content."""
    with open(yaml_file, 'r') as file:
        lines = file.readlines()

    with open(yaml_file, 'w') as file:
        for line in lines:
            if line.strip().startswith("truth_filename:"):
                # Match the truth_filename line, capturing any quotes around the path
                match = re.match(r'^(truth_filename:\s*)(["\']?)(.*?/catalogs/stars/)([^"\']+)(["\']?)\s*$', line)
                
                if match:
                    prefix, opening_quote, _, filename, closing_quote = match.groups()
                    
                    # Ensure that the new line retains the original quote style (single/double) if present
                    quote = opening_quote if opening_quote else "'"  # Default to single quote if none exist
                    
                    new_line = f"{prefix}{quote}{datadir}/catalogs/stars/{filename}{quote}\n"
                    file.write(new_line)
                    print(f"Updated {yaml_file}: {new_line.strip()}")
                else:
                    file.write(line)
            else:
                file.write(line)

def download_catalogs(datadir):
    """Download catalogs from hen server."""
    print("\n" + "="*60)
    print("CATALOG DOWNLOAD SETUP")
    print("="*60)
    
    response = input("\nDo you want to set up your data directory by downloading star catalogs, NED catalogs, etc. from hen? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Skipping catalog download.")
        return None
    
    print("\nTo download catalogs from hen, you need an account on hen.astro.utoronto.ca")
    print("If you don't have an account, please contact Emaad Paracha (emaad.paracha@mail.utoronto.ca)")
    
    has_account = input("\nDo you have an account on hen? (yes/no): ").strip().lower()
    
    if has_account not in ['yes', 'y']:
        print("\nPlease get an account first, then run this script again.")
        return None
    
    username = input("\nEnter your hen username: ").strip()
    
    if not username:
        print("Username cannot be empty. Skipping download.")
        return None
    
    # Construct the scp command
    source = f"{username}@hen.astro.utoronto.ca:/data/analysis/superbit_2023/shape_cats/catalogs {username}@hen.astro.utoronto.ca:/data/analysis/superbit_2023/shape_cats/star_masks"
    destination = datadir

    print(f"\nWill download from: {source}")
    print(f"To: {destination}")
    print("\nNote: You will be prompted for your hen password.")

    # Use rsync for better handling of directories and resume capability
    cmd = ["rsync", "-avz", "--progress"] + source.split() + [destination]

    try:
        print("\nStarting download...")
        result = subprocess.run(cmd, check=True)
        print("\nCatalog download completed successfully!")
        return username  # Return username for alias creation
    except subprocess.CalledProcessError as e:
        print(f"\nError during download: {e}")
        print("Please check your username, password, and network connection.")
    except FileNotFoundError:
        print("\nError: rsync not found. Trying with scp instead...")
        # Fallback to scp if rsync is not available
        cmd = ["scp", "-r"] + source.split() + [destination]
    
    return username  # Return username even if download failed

def update_galsim_config(sim_path, current_dir):
    """
    Update specific paths in the galsim_config.yaml file (cosmosdir, datadir, emp_psf_path)
    to point to the local simulation directory.

    Parameters:
        sim_path (str): Path to the local simulation directory.
        current_dir (str): Current working directory where the script is run.
    """
    config_file = os.path.join(current_dir, "job_sims", "galsim_config.yaml")
    
    if not os.path.exists(config_file):
        print(f"galsim_config.yaml not found at {config_file}")
        return
    
    new_lines = []
    with open(config_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("cosmosdir:"):
                line = f"cosmosdir:         '{os.path.join(sim_path, 'sim_utils/galsim_data/galsim_cosmos_catalogs')}' # Path to COSMOS data directory\n"
            elif stripped.startswith("datadir:"):
                line = f"datadir:           '{os.path.join(sim_path, 'sim_utils/galsim_data')}' # Path to repo/galsim data directory\n"
            elif stripped.startswith("emp_psf_path:"):
                line = f"emp_psf_path:      '{os.path.join(sim_path, 'sim_utils/emp_psfs_order5')}'\n"
            new_lines.append(line)
    
    with open(config_file, "w") as f:
        f.writelines(new_lines)
    
    print(f"galsim_config.yaml updated with sim_path = {os.path.join(sim_path, 'sim_utils')}")


def update_simblaster(sim_path, current_dir):
    """
    Update DATADIR and CODEDIR in the SimBlaster.sh script.
    
    Parameters:
        sim_path (str): Path to the local simulation directory.
        current_dir (str): Current working directory where the code resides.
    """
    simblaster_file = os.path.join(current_dir, "utility_scripts", "SimBlaster.sh")
    
    if not os.path.exists(simblaster_file):
        print(f"SimBlaster.sh not found at {simblaster_file}")
        return
    
    new_lines = []
    with open(simblaster_file, "r") as f:
        for line in f:
            if line.strip().startswith("DATADIR="):
                line = f'DATADIR="{sim_path}"\n'
            elif line.strip().startswith("CODEDIR="):
                line = f'CODEDIR="{current_dir}"\n'
            new_lines.append(line)
    
    with open(simblaster_file, "w") as f:
        f.writelines(new_lines)
    
    print(f"SimBlaster.sh updated with DATADIR={sim_path} and CODEDIR={current_dir}")

def download_sim_data(username, sim_path):
    """
    Download the sim_utils directory from the remote server to the local simulation path.
    
    Parameters:
        username (str): Your username on the remote server.
        sim_path (str): Local path where the sim_utils directory should be saved.
    """
    source = f"{username}@hen.astro.utoronto.ca:/data/analysis/superbit_2023/shape_cats/sim_utils"
    destination = sim_path

    print(f"\nWill download sim_utils from: {source}")
    print(f"To: {destination}")
    print("\nNote: You will be prompted for your hen password.")

    # Use rsync for directories and resume capability
    cmd = ["rsync", "-avz", "--progress", source, destination]

    try:
        print("\nStarting download of sim_utils...")
        subprocess.run(cmd, check=True)
        print("\nsim_utils download completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError during download: {e}")
        print("Please check your username, password, and network connection.")
    except FileNotFoundError:
        print("\nError: rsync not found. Trying with scp instead...")
        cmd = ["scp", "-r", source, destination]
        subprocess.run(cmd, check=True)
        print("\nsim_utils download completed successfully via scp!")

def update_slurm_partitions(current_dir):
    """
    Prompt user for SLURM partition preference and update submission scripts accordingly.
    """
    print("\n" + "="*60)
    print("SLURM PARTITION SETUP")
    print("="*60)

    response = input("\nDo you want to specify a partition for your HPC jobs? (yes/no): ").strip().lower()

    # Files to update
    sims_dir = os.path.join(current_dir, "job_sims", "scripts")
    jobs_dir = os.path.join(current_dir, "job_scripts", "scripts")
    files_to_update = [
        os.path.join(jobs_dir, "color_color_run.sh"),
        os.path.join(jobs_dir, "make_meds.sh"),
        os.path.join(jobs_dir, "ngmix_job_template.sh"),
        os.path.join(sims_dir, "gen_mocks.sh"),
        os.path.join(sims_dir, "make_meds_sims.sh"),
        os.path.join(sims_dir, "ngmix_job_template.sh"),
    ]

    # Collect partition if needed
    partition = None
    if response in ["yes", "y"]:
        partition = input("Enter the partition name you want to use: ").strip()
        if not partition:
            print("No partition name provided. Skipping partition update.")
            return

    # Process each file
    for filepath in files_to_update:
        if not os.path.exists(filepath):
            continue

        new_lines = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip().startswith("#SBATCH --partition="):
                    if partition:  # replace with new partition
                        new_lines.append(f"#SBATCH --partition={partition}\n")
                    else:  # remove the line
                        continue
                else:
                    new_lines.append(line)

        with open(filepath, "w") as f:
            f.writelines(new_lines)

        if partition:
            print(f"Updated partition in {filepath} → {partition}")
        else:
            print(f"Removed partition line in {filepath}")

def setup_job_submission_dir(current_dir):
    """
    Set up a job submission directory for real data jobs.
    - Ask the user if they want to set it up.
    - Default location: ../jobs
    - Copy job_scripts → <chosen_dir>/base
    """
    print("\n" + "="*60)
    print("JOB SUBMISSION DIRECTORY SETUP")
    print("="*60)

    response = input("\nDo you want to set up your job-submission directory? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Skipping job-submission directory setup.")
        return

    # Default directory is ../jobs relative to current_dir
    default_dir = os.path.abspath(os.path.join(current_dir, "..", "jobs"))
    user_input = input(f"Enter the path for real data jobs (default: {default_dir}): ").strip()

    if not user_input:
        jobs_dir = default_dir
    else:
        jobs_dir = os.path.abspath(user_input)

    # Create the directory if needed
    try:
        os.makedirs(jobs_dir, exist_ok=True)
        print(f"Created/verified job-submission directory: {jobs_dir}")
    except Exception as e:
        print(f"Error creating directory {jobs_dir}: {e}")
        return

    # Source job_scripts
    src = os.path.join(current_dir, "job_scripts")
    dst = os.path.join(jobs_dir, "base")

    if not os.path.exists(src):
        print(f"Error: job_scripts folder not found in {current_dir}")
        return

    # If base already exists, ask user
    if os.path.exists(dst):
        overwrite = input(f"Directory {dst} already exists. Overwrite? (yes/no): ").strip().lower()
        if overwrite not in ["yes", "y"]:
            print("Skipping copy of job_scripts.")
            return
        shutil.rmtree(dst)

    try:
        shutil.copytree(src, dst)
        print(f"Copied {src} → {dst}")
    except Exception as e:
        print(f"Error copying job_scripts: {e}")

def main(env_name=None):
    # Resolve the absolute path of the current directory
    current_dir = os.getcwd()
    default_path = os.path.join(current_dir, 'data')
    
    # Get user input
    user_input = input(f"Enter the absolute path of your data directory (default: {default_path}): ").strip()
    
    # Use the default path if the user doesn't provide input
    if not user_input:
        save_path = default_path
    else:
        # Normalize the user-provided path
        save_path = normalize_path(user_input)
    
    # Convert to absolute path if relative path was given
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    
    print(f"Using data directory: {save_path}")

    # Ensure the directory exists
    try:
        os.makedirs(save_path, exist_ok=True)
        print(f"Created/verified directory: {save_path}")
    except Exception as e:
        print(f"Error creating directory {save_path}: {e}")
        return

    # Set the correct directory for YAML config files
    config_dir = os.path.join(current_dir, 'superbit_lensing', 'medsmaker', 'configs')

    # Loop through all YAML files and update truth_filename
    if os.path.exists(config_dir):
        yaml_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        
        if yaml_files:
            print(f"\nFound {len(yaml_files)} YAML files to update:")
            for yaml_file in yaml_files:
                update_truth_filename(os.path.join(config_dir, yaml_file), save_path)
        else:
            print(f"No YAML files found in {config_dir}")
    else:
        print(f"Config directory {config_dir} does not exist.")
        print("Please make sure you're running this script from the superbit-lensing root directory.")

    # Update config.sh file
    print("\nUpdating config.sh file...")
    config_sh_path = os.path.join(current_dir, 'job_scripts', 'config.sh')
    update_config_sh(config_sh_path, save_path, current_dir, env_name=env_name)
    
    # Show completion message
    print("\n" + "="*60)
    print("Configuration update completed!")
    print("="*60)
    
    # Ask about downloading catalogs
    username = download_catalogs(save_path)

    # Ask about setting up simulation branch
    setup_sim = input("\nDo you want to set up a simulation branch? (y/n): ").strip().lower()
    if setup_sim == 'y':
        sim_default_path = os.path.join(current_dir, 'simulated_data')
        sim_path_input = input(f"Enter the path for simulated data (default: {sim_default_path}): ").strip()
        
        if not sim_path_input:
            sim_path = sim_default_path
        else:
            sim_path = normalize_path(sim_path_input)
        
        if not os.path.isabs(sim_path):
            sim_path = os.path.abspath(sim_path)
        
        try:
            os.makedirs(sim_path, exist_ok=True)
            print(f"Simulation directory created/verified: {sim_path}")
        except Exception as e:
            print(f"Error creating simulation directory {sim_path}: {e}")
            return

        config_sim_sh_path = os.path.join(current_dir, 'job_sims', 'config.sh')
        update_config_sh(config_sim_sh_path, sim_path, current_dir, env_name=env_name)
        update_galsim_config(sim_path, current_dir)
        update_simblaster(sim_path, current_dir)
        download_sim_data(username, sim_path)

    # Update SLURM partitions
    update_slurm_partitions(current_dir)
    # Set up job submission directory
    setup_job_submission_dir(current_dir)
    # Add bit-download alias
    print("\n" + "="*60)
    print("SETTING UP BIT-DOWNLOAD ALIAS")
    print("="*60)
    add_bit_download_alias(current_dir, save_path, username)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup script for SuperBIT lensing configs.")
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Optional conda environment name to add to config.sh"
    )
    args = parser.parse_args()
    main(env_name=args.env_name)