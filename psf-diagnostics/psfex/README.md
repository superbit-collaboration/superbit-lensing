# PSFEx Model Diagnostics

## Usage

### Prerequisites

Make sure you have run `medsmaker` for all the clusters listed in `model_diagnostics.py`.

### Step 1: Run Model Diagnostics

Submit the diagnostics job to HPC:

```bash
sbatch submit_diagnostics.sh
```
### Step 2: Generate Plots

After the model diagnostics run finishes, create the analysis plots:

```bash
python psf_analysis_runner.py
```
## Files

- `model_diagnostics.py` - Main diagnostics script
- `submit_diagnostics.sh` - HPC submission script
- `psf_analysis_runner.py` - Plot generation script