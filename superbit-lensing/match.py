'''
File with classes & functions useful for matching catalogs
'''

import numpy as np
import esutil.htm as htm
from astropy.table import Table, hstack

class MatchedCatalog(object):

    def __init__(self, cat1_file, cat2_file,
                 cat1_ratag='ra', cat1_dectag='dec',
                 cat2_ratag='ra', cat2_dectag='dec',
                 cat1_hdu=1, cat2_hdu=1,
                 match_radius=1.0/3600, depth=14):
        '''
        match_radius is in deg, same as htm
        '''

        self.cat1_file = cat1_file
        self.cat2_file = cat2_file

        self.cat1_ratag  = cat1_ratag
        self.cat2_ratag  = cat2_ratag
        self.cat1_dectag = cat1_dectag
        self.cat2_dectag = cat2_dectag

        self.match_radius = match_radius
        self.depth = depth

        self.cat2 = None
        self.cat1 = None
        self.cat = None # matched catalog
        self.Nobjs = 0

        self._match()

        return

    def _match(self):
        cat1_cat, cat2_cat = self._load_cats()

        h = htm.HTM(self.depth)

        self.matcher = htm.Matcher(
            depth=self.depth,
            ra=cat1_cat[self.cat1_ratag],
            dec=cat1_cat[self.cat1_dectag]
            )

        id_m, id_t, dist = self.matcher.match(
            ra=cat2_cat[self.cat2_ratag],
            dec=cat2_cat[self.cat2_dectag],
            radius=self.match_radius
            )

        self.cat1 = cat1_cat[id_t]
        self.cat2 = cat2_cat[id_m]
        self.cat2['separation'] = dist
        self.dist = dist

        assert len(self.cat1) == len(self.cat2)

        self.cat = hstack([self.cat1, self.cat2])

        self.Nobjs = len(self.cat)

        return

    def _load_cats(self):
        cat1_cat = Table.read(self.cat1_file)
        cat2_cat = Table.read(self.cat2_file)

        return cat1_cat, cat2_cat

class MatchedTruthCatalog(MatchedCatalog):
    '''
    Same as MatchedCatalog, where one cat holds
    truth information

    Must pass truth cat first
    '''

    def __init__(self, truth_file, meas_file, **kwargs):

        cat1, cat2 = truth_file, meas_file
        super(MatchedTruthCatalog, self).__init__(
            cat1, cat2, **kwargs
            )

        return

    @property
    def true_file(self):
        return self.cat1_file

    @property
    def meas_file(self):
        return self.cat2_file

    @property
    def true(self):
        return self.cat1

    @property
    def meas(self):
        return self.cat2
