'''
File with classes & functions useful for matching catalogs
'''

import numpy as np
import esutil.htm as htm
from astropy.table import Table, hstack
from tqdm import tqdm
import ipdb

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

class SkyCoordMatcher:
    def __init__(self, cat1, cat2,
                 cat1_ratag='ALPHAWIN_J2000', cat1_dectag='DELTAWIN_J2000',
                 cat2_ratag=None, cat2_dectag=None, return_idx=False,
                 match_radius=1.0/3600):
        """
        Initialize the SkyCoordMatcher with catalogs and matching parameters.

        Parameters:
        - cat1, cat2: Input catalogs (structured arrays or Astropy Tables).
        - cat1_ratag, cat1_dectag: Column names for RA and DEC in cat1.
        - cat2_ratag, cat2_dectag: Column names for RA and DEC in cat2 (defaults to same as cat1 if not provided).
        - match_radius: Matching radius in degrees.
        """
        self.cat1 = cat1
        self.cat2 = cat2

        self.cat1_ratag = cat1_ratag
        self.cat1_dectag = cat1_dectag
        self.cat2_ratag = cat2_ratag if cat2_ratag is not None else cat1_ratag
        self.cat2_dectag = cat2_dectag if cat2_dectag is not None else cat1_dectag

        self.match_radius = match_radius
        self.return_idx = return_idx

        self.matched_cat1 = None
        self.matched_cat2 = None
        self.matched_cat = None
        self.Nobjs = 0

        self._match()

    def _match(self):
        """
        Perform the matching between the two catalogs based on RA and DEC coordinates.
        """
        # Convert RA/DEC to radians
        ra1 = np.radians(self.cat1[self.cat1_ratag])
        dec1 = np.radians(self.cat1[self.cat1_dectag])
        ra2 = np.radians(self.cat2[self.cat2_ratag])
        dec2 = np.radians(self.cat2[self.cat2_dectag])

        # Convert match radius to radians
        tolerance_rad = np.radians(self.match_radius)

        print("Matching objects...")
        matches = -np.ones(len(self.cat1), dtype=int)
        distances = np.full(len(self.cat1), np.inf)

        for i, (ra1_i, dec1_i) in tqdm(enumerate(zip(ra1, dec1)), total=len(ra1), desc="Matching objects"):
            cos_distance = (
                np.sin(dec1_i) * np.sin(dec2) +
                np.cos(dec1_i) * np.cos(dec2) * np.cos(ra1_i - ra2)
            )
            angular_distance = np.arccos(np.clip(cos_distance, -1, 1))

            closest_index = np.argmin(angular_distance)
            closest_distance = angular_distance[closest_index]

            if closest_distance < tolerance_rad:
                matches[i] = closest_index
                distances[i] = closest_distance

        print(f"Number of matches after initial pass: {np.sum(matches != -1)}")

        # Reassign best matches
        best_match_for_2 = {}
        for i, match in enumerate(matches):
            if match != -1:
                if match not in best_match_for_2 or distances[i] < distances[best_match_for_2[match]]:
                    best_match_for_2[match] = i
        final_matches1 = list(best_match_for_2.values())
        final_matches2 = list(best_match_for_2.keys())
        print(f"Number of matches after reassignment: {len(final_matches1)}")

        self.matched_cat1 = self.cat1[final_matches1]
        self.matched_cat2 = self.cat2[final_matches2]
        self.matched_cat2['separation'] = distances[final_matches1]
        self.matched_cat = hstack([self.matched_cat1, self.matched_cat2])
        self.match_idx1 = final_matches1
        self.match_idx2 = final_matches2

        self.Nobjs = len(self.matched_cat)

    def get_matched_catalog(self):
        """
        Return the matched catalog.

        Returns:
        - matched_cat: Combined catalog of matched objects.
        """
        return self.matched_cat

    def get_matched_pairs(self):
        """
        Return the matched pairs of objects.

        Returns:
        - matched_cat1, matched_cat2: Matched subsets of the input catalogs.
        """
        if not self.return_idx:
            return self.matched_cat1, self.matched_cat2
        return self.matched_cat1, self.matched_cat2, self.match_idx1, self.match_idx2