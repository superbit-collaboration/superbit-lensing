#!/bin/bash

## Remove intermediate files from a series of realizations

## By default, $run_name_???.sub.fits, *sgm*, single-exposure catalogs,
## mask_files, and weight_files
## deep_clean gets rid of everything except raw galsim exposures
## piranha_clean gets rid of everything except configs

# Check input arguments and print a help statement if they are missing

if [ $# -lt 2 ]; then
  echo "   "
  echo "Usage: "
  echo "  $0 <run_name> <directory1> [<directory2> ...] [--deep] [--piranha] [--echo]"
  echo "   "
  echo "    --deep:     deep_clean; delete all files except raw galsim exposures"
  echo "    --piranha:  piranha scrub; delete everything except for config files"
  echo "    --echo:     list all files that would be deleted (dry run)"
  echo "   "
  exit 1
fi

# Get file extension, directory and flag options from command line
run_name="$1"
deep_clean=false
piranha_clean=false
echo_only=false


# check for flags
for i in "$@"; do
  if [[ $i == "--deep" ]]; then
    deep_clean=true
  elif [[ $i == "--echo" ]]; then
    echo_only=true
  elif [[ $i == "--piranha" ]]; then
    piranha_clean=true
  elif [[ $i != $run_name ]]; then
    directories+=("$i")
  fi
done

# Delete all files with specified extension in specified directory

for directory in ${directories[@]}; do
  if [[ $deep_clean == true ]]; then
    if [[ $echo_only == true ]]; then
      ls "$directory"/*sgm* "$directory"/"$run_name"_???.sub.fits \
          "$directory"/"$run_name"_???_cat.ldac \
          "$directory"/"$run_name"_???_catstars.ldac "$directory"/weight_files \
          "$directory"/mask_files
      ls "$directory"/*mock_coadd* "$directory"/*joined* "$directory"/*shear* \
          "$directory"/*sub* "$directory"/*.p* "$directory"/piff-output \
          "$directory"/*meds* "$directory"/*mcal* "$directory"/*annular* \
          "$directory"/*log
    else
      rm -rf "$directory"/*sgm* "$directory"/"$run_name"_???.sub.fits \
            "$directory"/"$run_name"_???_cat.ldac "$directory"/weight_files \
            "$directory"/"$run_name"_???_catstars.ldac  "$directory"/mask_files
      rm -rf "$directory"/*mock_coadd* "$directory"/*joined* \
            "$directory"/*shear* \
            "$directory"/*sub* "$directory"/*.p* "$directory"/piff-output \
            "$directory"/*meds* "$directory"/*mcal* "$directory"/*annular* \
            "$directory"/*log
    fi
  elif [[ $piranha == true ]]; then
    if [[ $echo_only == true ]]; then
      ls "$directory"/*fits "$directory"/*ldac "$directory"/*png \
          "$directory"/*pdf "$directory"/*weight_files \
          "$directory"/*mask_files "$directory"/piff-output
    else
      rm -rf "$directory"/*fits "$directory"/*ldac "$directory"/*png \
            "$directory"/*pdf "$directory"/*weight_files \
            "$directory"/*mask_files "$directory"/piff-output
    fi
  else
    if [[ $echo_only == true ]]; then
      ls "$directory"/*sgm* "$directory"/"$run_name"_???.sub.fits \
          "$directory"/"$run_name"_???_cat.ldac \
          "$directory"/"$run_name"_???_catstars.ldac "$directory"/weight_files \
          "$directory"/mask_files
    else
      rm -rf "$directory"/*sgm* "$directory"/"$run_name"_???.sub.fits \
            "$directory"/"$run_name"_???_cat.ldac \
            "$directory"/"$run_name"_???_catstars.ldac "$directory"/weight_files \
            "$directory"/mask_files
    fi
  fi
done
