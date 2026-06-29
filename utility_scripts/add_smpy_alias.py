import os
from ..post_installation import get_shell_config_file

def main():
    rc_file = get_shell_config_file()
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pipeline_runner_pth = os.path.join(PROJECT_ROOT, "smpy", "run_pipeline.py")
    alias_key = "run_smpy"
    
    alias_exists = False
    if os.path.exists(rc_file):
        with open(rc_file, 'r') as f:
            content = f.read()
            if alias_key in content:
                alias_exists = True
                
    if alias_exists:
        print(f"\n run_smpy alias already exist in {rc_file}")
        update = input("Do you want to update it? (yes/no): ").strip().lower()
        if update not in ['yes', 'y']:
            return
        
        # Remove ALL related aliases + header
        with open(rc_file, 'r') as f:
            lines = f.readlines()
        
        with open(rc_file, 'w') as f:
            for line in lines:
                stripped = line.strip()
                
                if stripped.startswith(alias_key):
                    continue

                f.write(line)
                
    # add the new alias
    with open(rc_file, 'a') as f:
        f.write(f"\n# Alias for running the smpy pipeline\n")
        f.write(f"alias {alias_key}='python {pipeline_runner_pth}'\n")
        
if __name__ == "__main__":
    main()