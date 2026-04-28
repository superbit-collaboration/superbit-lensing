from astropy.table import Table
import numpy as np
t = Table.read('/projects/mccleary_group/superbit/color/FINAL_mega_color_mag_catalog_20260402_114934.fits')
print('Columns:', t.colnames)
print('Total rows:', len(t))
if 'Z_best' in t.colnames:
    has_z = np.sum(~np.isnan(t['Z_best'].astype(float)))
    print(f'Rows with Z_best: {has_z} ({has_z/len(t)*100:.1f}%)')
    print(f'Rows without Z_best: {len(t) - has_z}')
    if 'Z_source' in t.colnames:
        print('Z_source values:', set(t['Z_source'][~np.isnan(t['Z_best'].astype(float))]))
else:
    print('No Z_best column found')

