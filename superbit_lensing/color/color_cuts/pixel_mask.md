## Overview

This notebook implements a pixel-based voting method to identify foreground galaxies in galaxy cluster fields.
Unlike traditional geometric cuts (e.g., parabolic), this method adapts to the actual galaxy distribution in color-color space.

## Prerequisites

### Required Data Files

1. **Redshift Catalog** (`mega_color_mag_with_redshift.fits`): Contains galaxies with known redshifts for training
2. **Full Catalog** (`mega_color_mag_catalog.fits`): Complete photometric catalog for all clusters
3. **Red Sequence Member Catalogs** (optional):
    - `{cluster}_actual_members_catalog.fits`: Confirmed cluster members
    - `{cluster}_imposter_members_catalog.fits`: Misidentified members
  
### `plot_pixel_voting_map` Parameters:

- **pixel_size**: Resolution of color-color grid (0.05-0.1 mag recommended)
- **purity_threshold**: Minimum fraction of foreground galaxies for pixel to be "blue" (default 0.75)
- **show_scatter**: Display individual galaxy points
- **show_counts**: Label pixels with object counts
- **show_mask**: Apply high-count threshold (≥10 objects)
- **blue_counts_only**: Only show counts for blue pixels
- **show_cluster_objects**: Overlay all cluster galaxies in grey
- **redshift_seq**: Show red sequence members (imposters + actual)

### Color Coding:

- **Blue pixels**: Foreground-dominated (purity ≥ threshold)
- **Red pixels**: Background-dominated (purity < threshold)
- **Green pixels**: Blue pixels with ≥10 objects (when show_mask=True)
- **Magenta pixels**: Green pixels containing actual cluster members

## How `apply_pixel_mask_to_catalog` Works

**Pixel Colors:**

- **Green pixels**: Blue pixels (foreground-dominated) with ≥10 objects
- **Magenta pixels**: Green pixels that also contain actual cluster members

**Function Process:**

1. Filters full catalog for specified cluster
2. Checks each galaxy's color to see which pixel it falls in
3. Keeps galaxies in green or magenta pixels only

**What Gets Saved:**

- `pixel_method_only=True`: Only galaxies in green/magenta pixels → `{cluster}_foreground_pixel.fits`
- `pixel_with_redseq=True`: Green/magenta pixels + actual red sequence members → `{cluster}_foreground_pixel_and_redseq.fits`

The pixel method identifies foreground regions based on color-color voting, while the red sequence method adds known cluster members regardless of their pixel location
