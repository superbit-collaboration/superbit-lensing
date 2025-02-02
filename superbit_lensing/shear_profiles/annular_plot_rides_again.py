### VIP: START WITH $ module load texlive/2018 first! (on CCV)

import numpy as np
from matplotlib import rc,rcParams
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:pwd
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'serif'})
rc('text', usetex=True)
import matplotlib.pyplot as plt
from astropy.table import Table
import pdb

global pixscale
pixscale=0.144
global minrad
minrad = 50
global maxrad
maxrad=5100
nfw_truth_file='/Users/jemcclea/Research/SuperBIT/forecasting-analysis/cluster5/truth_shear_cl5_full.annular_jmac'
#cat2name = '/Volumes/PassportWD/SuperBIT/forecasting-analysis/tests/param_tests2/average_combine/shear_profiles/SN5_T0.02_1.2Tpsf_1.5e-3cov_minellip.annular'

cat1name = 'testcuts.annular_jmac'
outname = 'testcuts.annular_jmac.pdf'
#cat1name = 'SN7_1.0PSF_Tmin0.05_1e-2cov.annular_jmac'
#outname = 'SN7_1.0PSF_Tmin0.05_1e-2cov.annular_jmac.pdf'

label1=r'annular'
#label1=r'SN$>$5 0.3arcsec gaussPSF TPV avg combine 5e-3 cov'
nfw_label=r'7E14/h Msol NFW z=0.25'

title = r'S2N\_R$>$5 T$>$ 1.2*TPsf $0.02<T<10$ 1E-2 covcut'
plt.ion()


def covar_calculations(nfw,radius,etan,variance):

    # set some definitions; this will obviously throw horrible errors if it receives the wrong object

    nfwr=nfw[0]
    nfw_shear=nfw[1]
    C = np.diag(variance**2)
    D = etan

    # build theory array to match data points with a kludge
    # radius defines the length of our dataset
    T = []
    for rstep in radius:
        T.append(np.interp(rstep, nfwr, nfw_shear))
    T=np.array(T)

    # Ok, I think we are ready
    Ahat = T.T.dot(np.linalg.inv(C)).dot(D)/ (T.T.dot(np.linalg.inv(C)).dot(T))
    sigma_A = 1./np.sqrt((T.T.dot(np.linalg.inv(C)).dot(T)))

    return Ahat, sigma_A

data=Table.read(cat1name,format='ascii')

try:
    data.sort('col1') # get in descending order
    ## uncomment below if it's needed
    #data.remove_rows([0])
    radius=data['col1']
    radius=radius*pixscale/60.
except ValueError:
    data.sort('r') # get in descending order
    ## uncomment below if it's needed
    #data.remove_rows([0])
    radius=data['r']
    radius=radius*pixscale/60.

try:
    etan=data['gtan']
    ecross=data['gcross']
    shear1err=data['err_gtan']
    shear2err=data['err_gcross']

except KeyError:

    etan=data['col3']
    ecross=data['col4']
    shear1err=data['col5']
    shear2err=data['col6']



# So far, this is looking great!!! Now, let's remember how to make bin width error bars
minrad = minrad*pixscale/60 #pixels --> arcmin
maxrad = maxrad*pixscale/60 #pixels --> arcmin

upper_err=np.zeros_like(radius)
for e in range(len(radius)-1):
    this_err=(radius[e+1]-radius[e])*0.5
    upper_err[e]=this_err
upper_err[-1]=(maxrad-radius[-1])*0.5

lower_err=np.zeros_like(radius)
for e in (np.arange(len(radius)-1)+1):
    this_err=(radius[e]-radius[e-1])*0.5
    lower_err[e]=this_err

lower_err[0]=(radius[0]-minrad)*0.5

# And scene
rad_err=np.vstack([lower_err,upper_err])

rcParams['axes.linewidth'] = 1.3
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['xtick.minor.visible'] = True
rcParams['xtick.minor.width'] = 1
rcParams['xtick.direction'] = 'inout'
rcParams['ytick.minor.visible'] = True
rcParams['ytick.minor.width'] = 1
rcParams['ytick.direction'] = 'out'


fig, axs = plt.subplots(2,1,figsize=(10,7),sharex=True)#,sharey=True)
fig.subplots_adjust(hspace=0.1)


axs[0].errorbar(radius,etan,yerr=shear1err,xerr=rad_err,fmt='-o',capsize=5,color='cornflowerblue',label=label1)
axs[0].axhline(y=0,c="black",alpha=0.4,linestyle='--')
axs[0].set_ylabel(r'$g_{+}(\theta)$',fontsize=16)
axs[0].tick_params(which='major',width=1.3,length=8)
axs[0].tick_params(which='minor',width=0.8,length=4)
axs[0].set_title(title,fontsize=14)
axs[0].set_ylim(-0.05,0.60)

axs[1].errorbar(radius,ecross,xerr=rad_err,yerr=shear2err,fmt='d',capsize=5,color='cornflowerblue',alpha=0.5,label=label1)
axs[1].axhline(y=0,c="black",alpha=0.4,linestyle='--')
axs[1].set_xlabel(r'$\theta$ (arcmin)',fontsize=16)
axs[1].set_ylabel(r'$g_{\times}(\theta)$',fontsize=16)
axs[1].tick_params(which='major',width=1.3,length=8)
axs[1].tick_params(which='minor',width=0.8,length=4)
axs[1].set_ylim(-0.1,0.1)
axs[1].legend()



#####
##### Overplotting? Do it again!
#####
"""
data2=Table.read(cat2name,format='ascii')
try:
    data2.sort('col1') # get in descending order
    radius=data2['col1']
    radius2=radius2*pixscale/60.
except ValueError:
    data2.sort('r') # get in descending order
    radius2=data2['r']
    radius2=radius2*pixscale/60.

try:
    etan2=data2['gtan']
    ecross2=data2['gcross']
    shear1err2=data2['err_gtan']
    shear2err2=data2['err_gcross']

except KeyError:

    etan2=data2['col3']
    ecross2=data2['col4']
    shear1err2=data2['col5']
    shear2err2=data2['col6']



# So far, this is looking great!!! Now, let's remember how to make bin width error bars
minrad2 = minrad*pixscale/60 #pixels --> arcmin
maxrad2 = maxrad*pixscale/60 #pixels --> arcmin

upper_err=np.zeros_like(radius2)
for e in range(len(radius2)-1):
    this_err=(radius2[e+1]-radius2[e])*0.5
    upper_err[e]=this_err
upper_err[-1]=(maxrad-radius2[-1])*0.5

lower_err=np.zeros_like(radius2)
for e in (np.arange(len(radius2)-1)+1):
    this_err=(radius2[e]-radius2[e-1])*0.5
    lower_err[e]=this_err

lower_err[0]=(radius2[0]-minrad)*0.5

# And scene
rad_err2=np.vstack([lower_err,upper_err])


axs[0].errorbar(radius2,etan2,yerr=shear1err2,xerr=rad_err2,fmt='-o',capsize=5,label=label2,color='darkorange')
axs[1].errorbar(radius2,ecross2,xerr=rad_err2,yerr=shear2err2,fmt='s',capsize=5,alpha=0.5,label=label2,color='darkorange')
"""
#####
##### Overplot the NFW, do covar calculations
#####

nfw_plotted2=Table.read(nfw_truth_file,format='ascii')

try:
    nfw_plotted2.sort('col1')
    nfw_r2=nfw_plotted2['col1']

except:
    nfw_plotted2.sort('r')
    nfw_r2=nfw_plotted2['r']

nfw_r2= nfw_r2*pixscale/60.

try:
    nfw_etan=nfw_plotted2['col3']
    nfw_ecross=nfw_plotted2['col4']
    nfw_shear1err=nfw_plotted2['col5']
    nfw_shea2err=nfw_plotted2['col6']

except KeyError:
    nfw_etan=nfw_plotted2['gtan']
    nfw_ecross=nfw_plotted2['gcross']
    nfw_shear1err=nfw_plotted2['err_gtan']
    nfw_shea2err=nfw_plotted2['err_gcross']

etan_smooth=np.convolve(nfw_etan, np.ones(5)/5, mode='valid')
r_smooth=np.convolve(nfw_r2,np.ones(5)/5, mode='valid')
nfw_array=[nfw_r2,nfw_etan]
#axs[0].plot(nfw_r2,nfw_etan,'-r',label=nfw_label)
axs[0].plot(r_smooth,etan_smooth,'-r',label=nfw_label)
axs[0].set_xlim(0.1,13)


###
### Do covariance calculations for dataset 1
###
### TODO: make more general

alpha,sigma_alpha = covar_calculations(nfw=nfw_array,radius=radius,etan=etan,variance=shear1err)

txt = str(r'$\hat{\alpha}=%.4f~\sigma_{\hat{\alpha}}=%.4f$' % (alpha,sigma_alpha))
ann = axs[0].annotate(txt, xy=[0.1,0.9], xycoords='axes fraction',fontsize=12,\
    bbox=dict(facecolor='white', edgecolor='cornflowerblue',alpha=0.8,boxstyle='round,pad=0.3'))

try:
    alpha2,sigma_alpha2 = covar_calculations(nfw=nfw_array,radius=radius2,etan=etan2,variance=shear1err2)
    txt2 = str(r'$\hat{\alpha}=%.4f~\sigma_{\hat{\alpha}}=%.4f$' % (alpha2,sigma_alpha2))
    ann2 = axs[0].annotate(txt2, xy=[0.35,0.9], xycoords='axes fraction',fontsize=12,\
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkorange',boxstyle='round,pad=0.3'))
except:
    #pdb.set_trace()
    pass

axs[1].legend()
axs[0].legend()


# some sort of plt.annotate nonsense would go here; maybe in title or subtitle?
fig.savefig(outname)
