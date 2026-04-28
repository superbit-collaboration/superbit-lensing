# source_selection

Color-cut source selection module for the SuperBIT weak gravitational lensing pipeline. Separates foreground cluster member galaxies from background source galaxies using a pixel mask approach in B-G vs U-B color-color space.

## Overview

The module trains a pixel voting map on spectroscopic objects with known redshifts (NED and DESI sources), classifies each color-color pixel as foreground- or background-dominated based on a purity threshold τ, and applies the mask to a full photometric catalog to produce per-cluster foreground and background catalogs. Contamination rates are estimated via 5-fold stratified cross-validation.

## Directory Structure

```
source_selection/
├── __init__.py
├── color_cuts.py                  # Core pixel mask logic
├── validation.py                  # CV contamination estimation and tau sweep
├── make_background_catalog.py     # Main driver script
├── plot_cluster_comparison.py     # Multi-cluster comparison plots
├── configs/
│   └── default_source_selection.yaml
└── README.md
```

## Method

### Pixel Mask

The color-color space (B-G on x, U-B on y) is divided into pixels of size `pixel_size`. For each pixel, the weighted foreground fraction (purity) is computed using inverse-variance weights (1/σ²) from the spectroscopic training set. Pixels where purity ≥ τ are classified as foreground-dominated. Only pixels with ≥ `min_count` training objects are included in the hard mask.

### Training Set

Objects are selected from the input catalog where:
- `Z_source` is NED or DESI
- Color errors are below `err_thresh` in both B-G and U-B
- No NaN values in colors or redshift

Red sequence members (flagged in the RS catalog) are optionally excluded from training before building the pixel mask.

### Contamination Estimation

Background contamination is estimated via 5-fold stratified cross-validation on the global training set:

- **bg_contam** = FN / (TN + FN) — foreground objects that leaked into the background sample
- **fg_contam** = FP / (TP + FP) — background objects incorrectly classified as foreground  
- **total_contam** = (FP + FN) / N — total misclassification rate

When `tau_sweep: false`, contamination estimates at the chosen `purity_threshold` are written into the background catalog FITS header.

## Usage

```bash
python -m superbit_lensing.source_selection.make_background_catalog \
    -c superbit_lensing/source_selection/configs/default_source_selection.yaml \
    --overwrite --vb
```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `-c` / `--config` | Path to YAML config file |
| `--overwrite` | Overwrite existing output files |
| `--vb` | Verbose output |

## Configuration

The YAML config controls all pipeline behavior. Key parameters:

### Catalogs

```yaml
# Single mega catalog — training set = NED/DESI rows with Z_best
color_catalog: /path/to/FINAL_mega_color_mag_catalog.fits
```

The same catalog serves as both the training set (rows with NED/DESI redshifts) and the full photometric sample for object selection.

### Clusters

```yaml
clusters:
  - name: Abell3411
    redshift: 0.170
    shear_catalog: /path/to/annular_combined.fits  # only needed if shapes: true
```

Each cluster must match the `CLUSTER` column in the catalog exactly. The redshift threshold is set to `redshift + 0.025`.

### Pixel Mask Parameters

```yaml
pixel_size: 0.05       # pixel size in color space (mag)
min_count: 10          # minimum training objects per mask pixel
err_thresh: 0.5        # maximum allowed color error (mag)
xlim: [-1.5, 1.0]     # B-G axis limits
ylim: [-1.0, 3.0]     # U-B axis limits
weighting: true        # use inverse-variance weighting
```

### Tau Settings

```yaml
tau_sweep: false        # if true, sweep over tau_values and plot
purity_threshold: 0.5   # tau used for final catalogs
tau_values: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
n_folds: 5
```

When `tau_sweep: true`, a contamination vs source density plot is saved per cluster, and final catalogs are still produced at `purity_threshold`.

### Shapes Mode

```yaml
shapes: false   # if true, requires per-cluster shear_catalog
```

When `shapes: false` (default), the background catalog is built from the color catalog and contains photometric columns only. When `shapes: true`, foreground objects are removed from a separate per-cluster annular shear catalog and shear response is recalculated.

### Red Sequence

```yaml
with_redsequence: true
redseq_catalog: /path/to/combined_colors_mags.fits
```

RS members (`is_red_sequence = True`) are excluded from the pixel mask training set. Clusters not present in the RS catalog have RS removal silently skipped.

## Outputs

For each cluster, outputs are written to `{output_dir}/{cluster_name}/`:

| File | Description |
|------|-------------|
| `{cluster}_foreground.fits` | Objects classified as foreground by the pixel mask |
| `{cluster}_background.fits` | Remaining objects (background catalog) |
| `{cluster}_pixel_mask.png` | Diagnostic image of the color-color pixel map |
| `{cluster}_contamination_vs_density.png` | Tau sweep plot (tau_sweep only) |

The background catalog FITS header contains:

| Keyword | Description |
|---------|-------------|
| `TAU` | Purity threshold used |
| `Z_THRESH` | Redshift threshold (z_cluster + 0.025) |
| `BG_CONT` / `BG_CSTD` | Mean/std BG contamination from CV |
| `FG_CONT` / `FG_CSTD` | Mean/std FG contamination from CV |
| `TOT_CONT` / `TOT_CSTD` | Mean/std total misclassification from CV |
| `DENS_AM2` | Background source density (obj/arcmin²) |

## Multi-Cluster Comparison Plot

```bash
python superbit_lensing/source_selection/plot_cluster_comparison.py \
    -c superbit_lensing/source_selection/configs/default_source_selection.yaml \
    --outdir /path/to/output
```

Produces two plots overlaying results across clusters:
- `cluster_comparison_bg_contam_vs_tau.png`
- `cluster_comparison_density_vs_tau.png`

## Notes

- Low-redshift clusters (z < 0.1) tend to have degenerate mask behavior due to class imbalance in the training set. Their contamination estimates should be treated with caution.
- The pixel mask is trained globally on all clusters combined, not per-cluster.
- CV training uses `random_state=42` for reproducibility.
- Source density is computed using the central 50% of the field (matching `utils.analyze_mcal_fits`).
