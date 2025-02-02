import sys
import getopt
import re
import os
import math
import numpy as np
from numpy import r_, c_
from numpy import linalg as la
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
#import nfwtools as nt
from astropy.io import fits
from astropy.table import Table
import pdb

class Annular():


    def __init__(self,cat_info = None, annular_info = None, use_resp = False):

        """
        :incat:   table that will be read in
        :outcat:  name of output shear profile catalog
        :g1/2:    reduced-shear components
        :gcross:  cross-shear (B-mode)
        :gtan:    tangential shear (E-mode)
        :x/y:     x and y positions
        :r:       distance of galaxy to NFW center
        :nbins:   number of radial bins for averaging (default = 5)
        :start/endrad: region to consider

        """

        self.cat_info = cat_info
        self.cat = None
        self.annular_info = annular_info
        self.x = None
        self.y = None
        self.r = None
        self.g1 = None
        self.g2 = None
        self.weight = None
        self.gcross = None
        self.gtan = None
        self.Rgamma = None
        self.Rcross = None
        self.Rs = None


        """
        self.xc = annular_info['nfw_center'][0]
        self.yc = annular_info['nfw_center'][1]
        self.startrad = annular_info['shear_args'][0]
        self.endrad = annular_info['shear_args'][1]
        self.nbins = annular_info['nbins']
        """


    def open_table(self,cat_info):

        try:
            self.cat = Table.read(cat_info['incat'],format='fits')
        except:
            self.cat = Table.read(cat_info['incat'],format='ascii')

        try:
            self.x = self.cat[cat_info['xy_args'][0]]
            self.y = self.cat[cat_info['xy_args'][1]]
            self.g1 = self.cat[cat_info['shear_args'][0]]
            self.g2 = self.cat[cat_info['shear_args'][1]]
            self.weight = self.cat[cat_info['weight_arg']]
            #self.weight=np.ones_like(self.cat[cat_info['xy_args'][0]])

        except:
            print("Could not load xy/g1g2 columns; check supplied column names?")
            pdb.set_trace()

        if np.shape(self.g1) == (len(self.x),2):
            self.g1 = self.g1[:,0]
            self.g2 = self.g2[:,1]

        return

    def get_r_gtan(self,write=False,use_resp=False):
        """
        Populates self.r with radial distance of galaxies from (xc,yc)
        and self.gtan/gcross with tangential and cross ellipticities

        Failed shape measurements with g1/g2 = -999999 get filtered out
        """

        xc = self.annular_info['nfw_center'][0]
        yc = self.annular_info['nfw_center'][1]

        #wg=(self.mu>1)
        g = np.sqrt(self.g1**2 + self.g2**2)
        wg = (g>0)

        print("## %d galaxies actually used for calculation" % len(wg.nonzero()[0]))

        g1=self.g1[wg]
        g2=self.g2[wg]
        x=self.x[wg]
        y=self.y[wg]

        self.r=np.sqrt( ((x-xc)**2.0) + ((y-yc)**2))

        phi = np.arctan2((y-yc),(x-xc))

        print("## Mean g: %f sigma_g: %f" % (np.mean(g),np.std(g)))
        self.gtan= -1.0*(g1*np.cos(2.0*phi) + g2*np.sin(2.0*phi))
        self.gcross = g1*np.sin(2.0*phi) - g2*np.cos(2.0*phi) # note that annular.c has opposite sign convention

        if use_resp == True:

            r11 = self.cat['r11'] ; r12 = self.cat['r12']
            r21 = self.cat['r21'] ; r22 = self.cat['r22']

            R11_S = self.cat['R11_S'] ; R22_S = self.cat['R22_S']

            Rtan_gamma = r11*(np.cos(2*phi)**2) + r22*(np.sin(2*phi)**2) + (r12+r21)*(np.sin(2*phi)*np.cos(2*phi))
            #Rcross_gamma = r11*(np.sin(2*phi)**2) - r22*(np.cos(2*phi)**2) + (r12-r21)*(np.sin(2*phi)*np.cos(2*phi))
            Rtan_S = np.ones_like(self.cat['r11'])*((R11_S + R22_S)*0.5)

            # The Rtan_S is really more of a placeholder, doesn't get called under normal circumstances
            self.Rgamma = Rtan_gamma
            #self.Rcross= Rcross_gamma
            self.Rs = Rtan_S

        if write==True:
            newtab=Table()
            newtab.add_columns([x,y,self.r,self.gtan,self.gcross],names=['x','y','r','gcross','gtan'])
            if use_resp==True:
                newtab.add_columns([Rtan_gamma,Rtan_S], names=['Rtan_gamma','Rtan_S'])
            newtab.write('annular_inputs_resp.csv',format='csv',overwrite=True)

        return

    def get_Rs_annulus(self,annulus):
        """
        min_Tpsf = 1.1 # orig 1.15
        max_sn = 1000
        min_sn = 5 # orig 8 for ensemble
        min_T = 0.07 # orig 0.05
        max_T = 10 # orig inf
        covcut=3E-3 # orig 1 for ensemble
        """
        min_Tpsf = 1.2 # orig 1.15
        max_sn = 1000
        min_sn =10 # orig 8 for ensemble
        min_T = 0.04 # orig 0.05
        max_T = 10 # orig inf
        covcut=7e-3 # orig 1 for ensemble

        qualcuts=str('#\n# cuts applied: Tpsf_ratio>%.2f SN>%.1f T>%.2f covcut=%.1e\n#\n' \
                         % (min_Tpsf,min_sn,min_T,covcut))
        #print(qualcuts)

        this_cat = self.cat[annulus]

        noshear_selection = this_cat[(this_cat['T_noshear']>=min_Tpsf*this_cat['Tpsf_noshear'])\
                                        & (this_cat['T_noshear']<max_T)\
                                        & (this_cat['T_noshear']>=min_T)\
                                        & (this_cat['s2n_r_noshear']>min_sn)\
                                        & (this_cat['s2n_r_noshear']<max_sn)\
                                        & (np.array(this_cat['pars_cov0_noshear'].tolist())[:,0,0]<covcut)\
                                        & (np.array(this_cat['pars_cov0_noshear'].tolist())[:,1,1]<covcut)
                                           ]

        selection_1p = this_cat[(this_cat['T_1p']>=min_Tpsf*this_cat['Tpsf_1p'])\
                                      & (this_cat['T_1p']<=max_T)\
                                      & (this_cat['T_1p']>=min_T)\
                                      & (this_cat['s2n_r_1p']>min_sn)\
                                      & (this_cat['s2n_r_1p']<max_sn)\
                                      & (np.array(this_cat['pars_cov0_1p'].tolist())[:,0,0]<covcut)\
                                      & (np.array(this_cat['pars_cov0_1p'].tolist())[:,1,1]<covcut)
                                       ]

        selection_1m = this_cat[(this_cat['T_1m']>=min_Tpsf*this_cat['Tpsf_1m'])\
                                      & (this_cat['T_1m']<=max_T)\
                                      & (this_cat['T_1m']>=min_T)\
                                      & (this_cat['s2n_r_1m']>min_sn)\
                                      & (this_cat['s2n_r_1m']<max_sn)\
                                      & (np.array(this_cat['pars_cov0_1m'].tolist())[:,0,0]<covcut)\
                                      & (np.array(this_cat['pars_cov0_1m'].tolist())[:,1,1]<covcut)
                                     ]

        selection_2p = this_cat[(this_cat['T_2p']>=min_Tpsf*this_cat['Tpsf_2p'])\
                                      & (this_cat['T_2p']<=max_T)\
                                      & (this_cat['T_2p']>=min_T)\
                                      & (this_cat['s2n_r_2p']>min_sn)\
                                      & (this_cat['s2n_r_2p']<max_sn)\
                                      & (np.array(this_cat['pars_cov0_2p'].tolist())[:,0,0]<covcut)\
                                      & (np.array(this_cat['pars_cov0_2p'].tolist())[:,1,1]<covcut)
                                      ]

        selection_2m = this_cat[(this_cat['T_2m']>=min_Tpsf*this_cat['Tpsf_2m'])\
                                      & (this_cat['T_2m']<=max_T)\
                                      & (this_cat['T_2m']>=min_T)\
                                      & (this_cat['s2n_r_2m']>min_sn)\
                                      & (this_cat['s2n_r_2m']<max_sn)\
                                      & (np.array(this_cat['pars_cov0_2m'].tolist())[:,0,0]<covcut)\
                                      & (np.array(this_cat['pars_cov0_2m'].tolist())[:,1,1]<covcut)
                                    ]


        r11_gamma=(np.mean(noshear_selection['g_1p'][:,0]) -np.mean(noshear_selection['g_1m'][:,0]))/0.02
        r22_gamma=(np.mean(noshear_selection['g_2p'][:,1]) -np.mean(noshear_selection['g_2m'][:,1]))/0.02

        # assuming delta_shear in ngmix_fit_superbit is 0.01
        r11_S = (np.mean(selection_1p['g_noshear'][:,0])-np.mean(selection_1m['g_noshear'][:,0]))/0.02
        r22_S = (np.mean(selection_2p['g_noshear'][:,1])-np.mean(selection_2m['g_noshear'][:,1]))/0.02

        Rtan_S = 0.5*(r11_S + r22_S)
        #print("# mean values <r11_gamma> = %f <r22_gamma> = %f" % (r11_gamma,r22_gamma))
        #print("# mean values <r11_S> = %f <r22_S> = %f" % (r11_S,r22_S))
        #print("\n")


        return Rtan_S

    def do_annular(self,use_resp=False):

        """
        workhorse; this will contain the actual azimuthal binning
        """
        minrad = self.annular_info['rad_args'][0]
        maxrad = self.annular_info['rad_args'][1]
        num_bins = self.annular_info['nbins']

        number,bins = np.histogram(self.r,bins=num_bins,range=(minrad,maxrad))


        print("##\n r       n        gtan      err_gtan     gcross     err_gcross")



        for i in range(len(bins)-1):
            annulus = (self.r>=bins[i]) & (self.r<bins[i+1])
            midpoint_r = np.mean([bins[i],bins[i+1]])
            n = number[i]

            if (use_resp==True):
                this_Rs = self.get_Rs_annulus(annulus)
                #r11_S,r22_S=self.get_Rs_annulus(annulus)
            else:
                #this_Rs = self.Rs[0]
                this_Rs = 1

            sum_gtan = np.sum(self.gtan[annulus]*self.weight[annulus])
            denom_gtan = np.sum(self.weight[annulus] * self.Rgamma[annulus]) + np.sum(self.weight[annulus]*this_Rs)
            gtan_mean = sum_gtan/denom_gtan

            #sum_gcross = np.sum(self.gcross[annulus]*self.weight[annulus])
            #denom_gcross = np.sum(self.weight[annulus] * self.Rcross[annulus]) + np.sum(self.weight[annulus]*this_Rs)
            #gcross_mean = sum_gcross/denom_gcross


            #gtan_mean = np.mean(self.gtan[annulus])
            gcross_mean=np.mean(self.gcross[annulus])

            #gtan_err = np.std(self.gtan[annulus])/np.sqrt(n)
            #gcross_err = np.std(self.gcross[annulus])/np.sqrt(n)

            gtan_err = np.std(self.gtan[annulus]/(np.mean(self.Rgamma[annulus])+self.Rs[annulus]))/np.sqrt(n)
            gcross_err = np.std(self.gcross[annulus]/(np.mean(self.Rgamma[annulus])+self.Rs[annulus]))/np.sqrt(n)
            #gcross_err = np.std(self.gcross[annulus]/(np.mean(self.Rcross[annulus])+self.Rs[annulus]))/np.sqrt(n)

            #gtan_err = np.std(gtan_mean)/np.sqrt(n)

            print("%.3f   %d   %f   %f   %f   %f" % (midpoint_r,n,gtan_mean,gtan_err,gcross_mean,gcross_err))

        return


    def run(self,write=False,use_resp=False):

        self.open_table(self.cat_info)
        self.get_r_gtan(write=write,use_resp=use_resp)
        self.do_annular(use_resp=use_resp)

        return


def print_header(args):

    print("###\n### Shear profile calculation\n###")
    print("### created with:")
    print("### %s\n###" % (' '.join(args)))
    print("## r: midpoint radius of annulus")
    print("## n: number of objects for shear calculation")
    print("## gtan: tangential (E-mode) reduced-shear")
    print("## err_gtan: standard error of gtan")
    print("## gcross: cross (B-mode) reduced-shear")
    print("## err_gcross: standard error of gcross")

    return

def main(args):

    """
    TO DO: clean up input arguments, using -o --option format
    and make startrad/endrad options as well, rather than magic numbers

    """

    if ( ( len(args) < 3) or (args == '-h') or (args == '--help') ):
        print("\n### \n### annular_jmac is a routine which takes an ascii table and x/y/g1/g2 column names as its input and returns shear profiles")
        print("### Note that at present, annular_jmac assumes that the shear arguments [g1, g2] are *reduced shear*, not image ellipticities\n###\n")
        print("### if '-resp' is set, the tangential shear will be responsivity-corrected following McClintock et al. 2019\n###\n")
        print(" python annular_jmac.py incat g1_arg g2_arg start_rad end_rad n_bins [-resp]\n \n")

    else:
        pass

    # Define catalog args
    incat = args[1]
    g1_arg = args[2]
    g2_arg = args[3]
    startrad = float(args[4])
    endrad = float(args[5])
    num_bins = int(args[6])

    # Perform responsivity correction?
    try:
        if(args[7]!= None):
            use_resp = True

    except:
        use_resp = False


    # Define annular args
    x_arg = 'XWIN_IMAGE_mcal'
    y_arg = 'YWIN_IMAGE_mcal'
    weight_arg = 'weight'

    nfw_center = [5031,3353]
    #nfw_center = [4784.0,3190.0]


    print_header(args)

    cat_info={'incat':incat, 'xy_args':[x_arg,y_arg], 'shear_args': [g1_arg,g2_arg], 'weight_arg': weight_arg}
    annular_info={'rad_args': [startrad, endrad],'nfw_center': nfw_center, 'nbins': num_bins, 'use_resp': use_resp}

    annular = Annular(cat_info,annular_info)
    annular.run(write=True,use_resp=use_resp)


if __name__ == '__main__':

    """
    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    """
    main(sys.argv)
