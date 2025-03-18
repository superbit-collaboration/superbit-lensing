#!/bin/sh
#SBATCH -t 13:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J color
#SBATCH -v
#SBATCH -o colorout.log
#SBATCH -e colorerr.log

# Load configuration file
source "$SLURM_SUBMIT_DIR/config.sh"
date
which python

python $CODEDIR/superbit_lensing/color/color_color_v2.py ${cluster_name} \
--datadir=$DATADIR \
--redshift=$cluster_redshift \
--config_dir=$CODEDIR/superbit_lensing/medsmaker/superbit/astro_config \
--plot_color_mag \
--save_fits \
--plot_ned #--plot_lovoccs
# Optional flags: --overwrite_coadds --overwrite_cats