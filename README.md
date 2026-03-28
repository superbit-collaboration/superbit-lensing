# superbit-lensing

A pipeline for weak gravitational lensing analysis of SuperBIT observations, performing ngmix fits (including metacalibration) on real and simulated images.

[![arXiv](https://img.shields.io/badge/arXiv-2603.18376-b31b1b.svg)](https://arxiv.org/abs/2603.18376)

## Pipeline overview

The pipeline is organized into five submodules within `superbit_lensing`, each of which can be used independently:

| Module | Description |
|---|---|
| `galsim` | Generates simulated SuperBIT observations for validation with Gaussian and SuperBIT PSFs |
| `medsmaker` | Builds coadd images, runs SExtractor & PSFEx, and produces MEDS files |
| `metacalibration` | Runs ngmix/metacalibration on MEDS files from Medsmaker |
| `shear-profiles` | Computes tangential/cross shear profiles and outputs results to file |
| `color` | Makes registered coadds for u, b, g bands and obtains photometry via SExtractor dual-mode |

Each module directory contains more detailed documentation.

## Data access

The data is not yet public. It currently lives on the UofT server `hen`. Contact [Emaad Paracha](mailto:emaad.paracha@mail.utoronto.ca) to request an account. Once you have access, the pipeline is plug-and-play.

## Installation

> **Disk space:** setup will ask for a `data_dir` and `simulation_dir`. Make sure you have ~50 GB available before starting.

```bash
git clone https://github.com/superbit-collaboration/superbit-lensing.git
cd superbit-lensing
make install ENV_NAME=sblens
conda activate sblens
```

`make install` automatically runs `post_installation.py`, which will prompt you to configure your data and simulation paths and download the required PSF files and catalogs.

## Questions / issues

Open a [GitHub issue](https://github.com/superbit-collaboration/superbit-lensing/issues) or reach out to any contributor.
