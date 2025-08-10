# Color Cuts

This directory contains tools and analysis for implementing color-color cuts to select background galaxies for weak lensing measurements.

## Motivation

In weak lensing studies, accurate mass measurements depend critically on having a clean sample of background source galaxies. 
Contamination from cluster member galaxies and foreground objects can significantly bias and dilute the lensing signal. 
This leads to underestimated cluster masses.  

## Files
- `Color_Cutter_Notebook.ipynb` - This is the first iteration of the parabola analysis notebook implementing color-color cuts. It implements a singular parabolic cut

- `Pixel_Mask_Notebook.ipynb` - This is the first iteration of the pixel mask analysis notebook implementing color-cuts. It uses a mask to cut out the forgeround objects. 

## References

Based on methods developed in:
- Medezinski et al. (2018) - "Source selection for cluster weak lensing measurements in the Hyper Suprime-Cam survey"
