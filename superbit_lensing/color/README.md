This folder contains scripts to generate color-magnitude and size-magnitude plots for galaxy clusters. Below are the instructions to set up and run the scripts.

## Setup

1. **Export the Data Directory**
    
    Before running the scripts, ensure you export the path to your data directory. Replace `/work/mccleary_group/saha/codes/superbit-lensing/data` with the actual path to your data directory.
    
    ```bash
    export DATADIR="/work/mccleary_group/saha/codes/superbit-lensing/data"
    ```
    

## Usage

### 1. Color-Magnitude Plot

To generate a color-magnitude plot for a specific cluster, use the `color_mag.py` script. The script requires the cluster name, filter bands, and optional flags to plot stars and NED data.

```bash
python color_mag.py AbellS0592 b g --datadir=$DATADIR --plot_star --plot_ned
```

### Arguments:

- `AbellS0592`: Replace with the name of the cluster you want to analyze.
- `b g`: Specify the filter bands (e.g., `b` and `g`).
- `-datadir=$DATADIR`: Path to the data directory (exported earlier).
- `-plot_star`: (Optional) Include stars in the plot.
- `-plot_ned`: (Optional) Include NED data in the plot.

---

### 2. Size-Magnitude Plot

To generate a size-magnitude plot for a specific cluster, use the `size_mag.py` script. The script requires the cluster name and a single filter band.

```bash
python size_mag.py AbellS0592 b --datadir=$DATADIR
```

### Arguments:

- `AbellS0592`: Replace with the name of the cluster you want to analyze.
- `b`: Specify the filter band (e.g., `b`).
- `-datadir=$DATADIR`: Path to the data directory (exported earlier).

---

## Notes

- Ensure the data directory (`$DATADIR`) contains the necessary files for the specified cluster and filters.
- Replace `AbellS0592` with the name of the cluster you are analyzing.
- The optional flags (`-plot_star` and `-plot_ned`) in the `color_mag.py` script can be omitted if not needed.