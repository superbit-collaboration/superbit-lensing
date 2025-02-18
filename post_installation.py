import os
import re

def update_truth_filename(yaml_file, datadir):
    """Update the truth_filename line in the given YAML file without modifying other content."""
    with open(yaml_file, 'r') as file:
        lines = file.readlines()

    with open(yaml_file, 'w') as file:
        for line in lines:
            if line.strip().startswith("truth_filename:"):
                # Replace anything before `/catalogs/stars/` with `datadir`
                new_line = re.sub(r'^truth_filename:\s*.*?/catalogs/stars/', 
                                  f'truth_filename: {datadir}/catalogs/stars/', line)
                file.write(new_line)
                print(f"Updated {yaml_file}: {new_line.strip()}")
            else:
                file.write(line)

def main():
    # Resolve the absolute path of the current directory
    current_dir = os.getcwd()
    default_path = os.path.join(current_dir, 'data')
    save_path = input(f"Enter the directory to save trained models and plots (default: {default_path}): ").strip()
    
    # Use the default path if the user doesn't provide input
    if not save_path:
        save_path = default_path

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Set the correct directory for YAML config files
    config_dir = os.path.join(current_dir, 'superbit_lensing', 'medsmaker', 'configs')

    # Loop through all YAML files and update truth_filename
    if os.path.exists(config_dir):
        for yaml_file in os.listdir(config_dir):
            if yaml_file.endswith('.yaml'):
                update_truth_filename(os.path.join(config_dir, yaml_file), save_path)
    else:
        print(f"Config directory {config_dir} does not exist.")

if __name__ == "__main__":
    main()
