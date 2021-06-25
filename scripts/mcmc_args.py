import sys
from skylens.utils import *
from skylens.survey_utils import *

"""tracer pairs for correlations"""
#tracers currently supported: galaxy,shear,kappa
corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')

corrs=[corr_ll,corr_ggl,corr_gg] #tracer pairs to use. This is passed to skylens

    
"""C_ell"""
lmax_cl=2000
lmin_cl=20
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl
lmax_cl_Bins=lmax_cl
Nl_bins=25
l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
lb=0.5*(l_bins[1:]+l_bins[:-1])
print('n ell bins: ',lb.shape)
l=l0
# l=np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell

bin_cl=True
do_pseudo_cl=True
use_binned_l=True

"""correlation functions"""
do_xi=False
bin_xi=True
use_binned_theta=True
theta_min=5/60
theta_max=250./60
n_theta_bins=20
theta_bins=np.logspace(np.log10(theta_min),np.log10(theta_max),n_theta_bins+1)
theta=np.logspace(np.log10(theta_min),np.log10(theta_max),n_theta_bins*10)
thb=.5*(theta_bins[1:]+theta_bins[:-1])

#Hankel Transform setup
WT_kwargs={'l':l0,'theta':theta,'s1_s2':[(2,2),(2,-2),(0,0),(0,2),(2,0)]}
WT=wigner_transform(**WT_kwargs)

"""window calculations"""
#relevant for pseudo_cl, correlation functions and covariances.
use_window=True
nside=1024 #nside is not used by skylens.
window_l=None
window_lmax=nside #this is used to generate window_l=np.arange(window_lmax) if window_l is none. 
                    # for serious analysis, window_lmax=2*lmax_cl

store_win=True #store coupling matrices and other relevant quantities.
                #if False, these are computed along with the C_ell.
                #False is not recommended right now.
        
clean_tracer_window=True #remove tracer windows from memory once coupling matrices are done

wigner_files={} #wigner file to get pseudo_cl coupling matrices.
                #these can be gwenerated using Gen_wig_m0.py and Gen_wig_m2.py
                #these are large files and are stored as compressed arrays, using zarr package.
wig_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/'
wigner_files[0]= wig_home+'dask_wig3j_l3500_w2100_0_reorder.zarr'
wigner_files[2]= wig_home+'dask_wig3j_l3500_w2100_2_reorder.zarr'

"""covariance"""
do_cov=True
SSV_cov=False
tidal_SSV_cov=False
Tri_cov=tidal_SSV_cov
sparse_cov=True #store covariances as sparse matrices
do_sample_variance=True #if false, only shot noise is used in gaussian covariance
xi_SN_analytical=True #use analytical expression for correlation function shot noise. if False, Cl shot noise is transformed.

f_sky=0.35 #if there is no window. This can also be a dictionary for different correlation pairs, to account for partial overlaps, etc.
            # e.g. f_sky[corr_gg][(0,0)]=0.35, f_sky[corr_ggl][(0,0)]=0.1
            # for covariance, f_sky[corr_gg+corr_ggl][(0,0,0,1)]=0  (0,0,0,1)==(0,0)+(0,1)

"""generate simulated samples"""
#for this example. You should define your own tracer_zbins
nzbins=5
shear_zbins=lsst_source_tomo_bins(nbins=nzbins,use_window=use_window,nside=nside)
galaxy_zbins=shear_zbins

"""cosmology and power spectra"""
from astropy.cosmology import Planck15 as cosmo
cosmo_params=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'OmR':cosmo.Ogamma0+cosmo.Onu0,'w':-1,'wa':0,'Tcmb':cosmo.Tcmb0})
cosmo_params['Oml']=1.-cosmo_params['Om']-cosmo_params['Omk']
pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':500,'scenario':'dmo','pk_func':'camb_pk_too_many_z','halofit_version':'takahashi','use_astropy':True}

z_PS=None #redshifts at which to compute power spectra.
nz_PS=100 #number of redshifts to sample power spectra. used if z_PS is none
log_z_PS=2 #grid to generate nz_PS redshifts. 0==linear, 1==log, 2==log+linear. used if z_PS is none
