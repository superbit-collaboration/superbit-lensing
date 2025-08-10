# Color Cuts

This directory contains tools and analysis for implementing color-color cuts to select background galaxies for weak lensing measurements.

## Motivation

In weak lensing studies, accurate mass measurements depend critically on having a clean sample of background source galaxies. 
Contamination from cluster member galaxies and foreground objects can significantly bias and dilute the lensing signal. 
This leads to underestimated cluster masses.  


## Workflow

### 1. Color-Color Space Selection
- Plot galaxies in multi-band color space (for us that will be u-b and b-g)
- Identify regions dominated by cluster red-sequence galaxies and foreground galaxies
- Define conservative cuts to avoid contaminated regions

### 2. Optimization - to be implemented 
- Test different color cut boundaries using cluster samples
- Measure lensing signal as a function of color cut limits
- Select cuts that minimize signal dilution while maintaining statistical power

### 3. Application
- Apply optimized cuts to full cluster catalog (like Abell3411)
- Generate background-only and foreground-only catalogs
- Create convergence maps with the background-only catalog

## Files
- `Color_Cutter_Notebook.ipynb` - This is the first iteration of the parabola analysis notebook implementing color-color cuts. It implements a singular parabolic cut

- `Pixel_Mask_Notebook.ipynb` - This is the first iteration of the pixel mask analysis notebook implementing color-cuts. It uses a mask to cut out the forgeround objects. 

## References

Based on methods developed in:
- Medezinski et al. (2018) - "Source selection for cluster weak lensing measurements in the Hyper Suprime-Cam survey"
