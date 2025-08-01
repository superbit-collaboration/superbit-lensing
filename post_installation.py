import os
import re

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

def update_config_sh(config_file, datadir, codedir):
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
            else:
                file.write(line)
    
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

def main():
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
    update_config_sh(config_sh_path, save_path, current_dir)

if __name__ == "__main__":
    main()
