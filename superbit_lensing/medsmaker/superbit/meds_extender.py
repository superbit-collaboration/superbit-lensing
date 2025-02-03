from meds import MEDS
from ngmix.medsreaders import NGMixMEDS
import numpy as np

class MEDSExtender(MEDS):
    '''
    This class extends MEDS to allow for different image & PSF WCS's
    & jacobians
    '''
    pass

class NGMixMEDSExtender(MEDS):
    '''
    This class extends NGMixMEDS to allow for different image & PSF
    WCS's & jacobians
    '''
    pass
