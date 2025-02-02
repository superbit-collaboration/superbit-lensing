import sys
import getopt
import re
import os
import math
import string
from math import sqrt
import numpy as np
from numpy import r_, c_
from numpy import linalg as la
import matplotlib.pyplot as plt
from fiattools import *
import nfwtools as nt
import pdb



def get_r_etan_alt(xc,yc,functor):
    """
    Populates functor.r[] with radial distance of galaxies from (xc,yc)
    and functor.etan[] with tangential ellipticities
    """
    """x = np.array(catalog.data["xer"])
       y = np.array(catalog.data["yer"])
       xc=np.float(xc);yc=np.float(yc) """ #All of this was for catalog of galaxies
    
    #Create an array of distances and redshifts! 
    functor.r= np.arange(10.0, 14000.0, 20.0)
    num_entries = list(range(len(functor.r)))
    #z=[0.3 for i in range(num_entries)]
    #functor.z = np.array(z)
    
    return num_entries


def get_sigmacrit(functor,zl):
    """
    Populates functor.sigmacrit[] with values of sigma crit for each galaxy
    based on their redshifts

    :param functor: Functor() object that stores parameters of NFW fit
    :param catalog: FiatCatalog() storing observed galaxy properties
    :param zl     : redshift of lens (i.e., cluster)

    """

    zlist=functor.z
    
    # First, get redshifts (thaaaanks Rich!)
    Dd=nt.dist(0,zl)
    Ds=[]
    Dds=[]
    for z in zlist:
        Ds.append(nt.dist(0,z))
        Dds.append(nt.dist(zl,z))
    Ds_over_Dds = np.array(Ds)/np.array(Dds)

    # Now turn this into a critical density
    functor.sigmacrit=(c**2/(4*3.1415926*G))*Ds_over_Dds/Dd

    return

 
def main(argv):

    global G
    G = 6.673e-11
    global h
    #h=0.678
    h =0.7
    global omega_m
    omega_m=0.306
    global omega_lambda
    omega_lambda=0.692
    global omega_k
    omega_k=0
    global H_0
    H_0=100*h*1e3/3.0857e22#*mpc_to_m
    global c
    c=299792458
    global m_solar
    m_solar=1.989e30
    global mpc_to_m
    mpc_to_m=1e6*3.0856e16
    global rho_crit
    rho_crit=3*(H_0**2)/(8*3.1415926*G)#in kg/m^3
    global D_H
    D_H=c/H_0
    global tol
    tol = 1.0E-6
    
    

    # Should all probably be command line options:
    """
    zl = 0.0591       # Redshift of lensing cluster
    xcenter=16628.8  # x-center of image in pixels
    ycenter=14554.95  # y-center of image in pixels
    nt.pixscale = 0.257 # Detector pixel scale (for distances!)
    m1 = 1.39E14
    """
    zl = 0.17       # Redshift of lensing cluster
    xcenter=3333  # x-center of image in pixels
    ycenter=2227  # y-center of image in pixels
    nt.pixscale = 0.206 # Detector pixel scale (for distances!)
    m200 = 1E15/h
       
    # Load in galaxy params
    gals = FiatCatalog()
    catalog='/Users/jemcclea/Research/SuperBIT/shear_profiles/truth_shear_forNFWplot.fiat'
    print("xcenter=%f ycenter=%f cat=%s" %(xcenter,ycenter, catalog))
    gals.load_columns(catalog,["x_image","y_image","g1_meas","g2_meas","redshift"])
    mean_z=np.median(gals.data['redshift'])


    # Initialize Functor object that will store NFW fit 
    nfw = nt.Functor()
    nfw.zl=zl
    arr=get_r_etan_alt(xcenter,ycenter,nfw)
    # sigmacrit needs a redshift! Here put all points of evaluation 
    # at mean z of sample
    nfw.z=[mean_z for i in range(len(nfw.r))]
    get_sigmacrit(nfw,zl)
 
    # Here is WL part: 
    #m200 = 1.59E14   
    Rs,del_c = nt.nfw_params(m200,zl)
    sigma=nt.sigma_nfw(nfw,Rs,del_c,arr)
    kappa = sigma/nfw.sigmacrit[arr]
    shear,g_value,xvalue=nt.shear_nfw(nfw,Rs,del_c,arr)
    g=shear/(1-kappa)
    
    
    # OK, time to write out values
    f=open("nfw_plotted.txt",'w')
    f.write("#xcenter=%f ycenter=%f cat=%s\n#\n#\n" %(xcenter,ycenter, catalog))
    
    f.write("# radius reduced_shear g_no_prefactor x_value\n")
    for i in range(len(g)):
        f.write("%.4e  %.12g  %.4e  %.4e\n" % (nfw.r[i],g[i],g_value[i],xvalue[i]))
    f.close()
    
    # Should you wish to plot, you could uncomment this section
    # and change some variables
    """
    plt.figure(1)
    plt.plot(marr,garr,'-b')
    #plt.xlim([1.0E14,1.0E15])
    plt.xlabel("Mass (Msol)")
    plt.ylabel("NFW reduced shear")
    plt.savefig("nfwg_weird.png")

    plt.figure(2)
    plt.plot(marr,xarr,'-b')
    #plt.xlim([1.0E14,1.0E15])
    plt.xlabel("Mass (Msol)")
    plt.ylabel("NFW chi-squared")
    plt.savefig("nfwchisq_weird.png")

    plt.figure(3)
    plt.plot(xsclarr,gvalarr,'-b',plt.xscale('log'),plt.yscale('log'))
    #plt.xlim([-0.1,0.2])
    plt.xlabel("log(R/r_s) ")
    plt.ylabel("log(g<)")
    plt.savefig("nfwgval_weird.png")
    """ 
if __name__ == "__main__":
    main(sys.argv[1:])
