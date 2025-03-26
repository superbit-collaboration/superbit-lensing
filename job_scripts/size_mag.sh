source "./config.sh"

python /work/mccleary_group/saha/codes/superbit-lensing/superbit_lensing/color/size_mag.py ${cluster_name} ${band_name} \
--datadir=$DATADIR \
--save_union_catalog 