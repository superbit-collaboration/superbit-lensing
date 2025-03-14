import os

def _run_sextractor_dual(image_file1, image_file2, cat_dir, config_dir, diag_dir=None, back_type='AUTO'):
    '''
    Utility method to invoke Source Extractor on supplied detection file
    Returns: file path of catalog
    '''
    if diag_dir is None:
        diag_dir = cat_dir
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    cat_name = os.path.basename(image_file2).replace('.fits','_cat.fits')
    cat_file = os.path.join(cat_dir, cat_name)

    image_arg  = f'"{image_file1}[0], {image_file2}[0]"'
    name_arg   = '-CATALOG_NAME ' + cat_file
    config_arg = f'-c {os.path.join(config_dir, "sextractor.real.config")}'
    param_arg  = f'-PARAMETERS_NAME {os.path.join(config_dir, "sextractor.param")}'
    nnw_arg    = f'-STARNNW_NAME {os.path.join(config_dir, "default.nnw")}'
    filter_arg = f'-FILTER_NAME {os.path.join(config_dir, "gauss_2.0_3x3.conv")}'
    bg_sub_arg = f'-BACK_TYPE {back_type}'

    bkg_file   = os.path.basename(image_file2).replace('.fits','.sub.fits')
    bkg_name   = os.path.join(diag_dir, bkg_file) 
    seg_file   = os.path.basename(image_file2).replace('.fits','.sgm.fits')
    seg_name   = os.path.join(diag_dir, seg_file)
    rms_file   = os.path.basename(image_file2).replace('.fits','.bkg_rms.fits')
    rms_name   = os.path.join(diag_dir, rms_file)
    #apt_file   = os.path.basename(image_file2).replace('.fits','.apt.fits')
    #apt_name   = os.path.join(diag_dir, apt_file)
    checkname_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name},{rms_name}'

    weight_arg = f'-WEIGHT_IMAGE "{image_file1}[1], {image_file2}[1]" ' + \
                    '-WEIGHT_TYPE MAP_WEIGHT'

    cmd = ' '.join([
                'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                ])

    print("sex cmd is " + cmd)
    os.system(cmd)

    print(f'cat_name is {cat_file} \n')
    return cat_file

def _run_sextractor_single(image_file1, cat_dir, config_dir, diag_dir=None, back_type='AUTO'):
    '''
    Utility method to invoke Source Extractor on supplied detection file
    Returns: file path of catalog
    '''
    os.makedirs(cat_dir, exist_ok=True)
    if diag_dir is None:
        diag_dir = cat_dir

    cat_name = os.path.basename(image_file1).replace('.fits','_cat.fits')
    cat_file = os.path.join(cat_dir, cat_name)

    image_arg  = f'"{image_file1}[0]"'
    name_arg   = '-CATALOG_NAME ' + cat_file
    config_arg = f'-c {os.path.join(config_dir, "sextractor.real.config")}'
    param_arg  = f'-PARAMETERS_NAME {os.path.join(config_dir, "sextractor.param")}'
    nnw_arg    = f'-STARNNW_NAME {os.path.join(config_dir, "default.nnw")}'
    filter_arg = f'-FILTER_NAME {os.path.join(config_dir, "gauss_2.0_3x3.conv")}'
    bg_sub_arg = f'-BACK_TYPE {back_type}'

    bkg_file   = os.path.basename(image_file1).replace('.fits','.sub.fits')
    bkg_name   = os.path.join(diag_dir, bkg_file) 
    seg_file   = os.path.basename(image_file1).replace('.fits','.sgm.fits')
    seg_name   = os.path.join(diag_dir, seg_file)
    rms_file   = os.path.basename(image_file1).replace('.fits','.bkg_rms.fits')
    rms_name   = os.path.join(diag_dir, rms_file)
    #apt_file   = os.path.basename(image_file1).replace('.fits','.apt.fits')
    #apt_name   = os.path.join(diag_dir, apt_file)
    checkname_arg = f'-CHECKIMAGE_NAME  {bkg_name},{seg_name},{rms_name}'

    weight_arg = f'-WEIGHT_IMAGE "{image_file1}[1]" ' + \
                    '-WEIGHT_TYPE MAP_WEIGHT'

    cmd = ' '.join([
                'sex', image_arg, weight_arg, name_arg,  checkname_arg,
                param_arg, nnw_arg, filter_arg, bg_sub_arg, config_arg
                ])

    print("sex cmd is " + cmd)
    os.system(cmd)

    print(f'cat_name is {cat_file} \n')
    return cat_file

