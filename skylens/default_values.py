

corrs=None
corr_indxs=None
stack_indxs=None
name=''

"""
Dask
"""
scheduler_info=None
njobs_submit_per_worker=10

"""
C_ell and pseudo-C_ell
"""
l=None
l_cl=None
cl_func_names={}
do_pseudo_cl=True

"""
Correlation functions
"""
do_xi=False
WT_kwargs=None
WT=None

"""
Binning
"""
l_bins=None
l_bins_center=None
bin_cl=False

bin_xi=False
theta_bins=None
theta_bins_center=None
use_binned_l=False
use_binned_theta=False

"""
Covariance and window
"""
do_cov=False
SSV_cov=False
tidal_SSV_cov=False
do_sample_variance=True
Tri_cov=False
sparse_cov=False
xi_SN_analytical=True

use_window=True
window_lmax=None
window_l=None
store_win=True
Win=None
f_sky=None
wigner_step=None
wigner_files=None
clean_tracer_window=True

"""
Tracers
"""
shear_zbins=None
kappa_zbins=None
galaxy_zbins=None
zkernel_func_names={tracer:'set_kernel' for tracer in ['shear','galaxy','kappa']}


"""cosmology and power spectra"""
from astropy.cosmology import Planck15 as cosmo

cosmo_h=cosmo.clone(H0=100)

cosmo_params=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'OmR':cosmo.Ogamma0+cosmo.Onu0,'w':-1,'wa':0,'Tcmb':cosmo.Tcmb0,'z_max':4,'use_astropy':True})
cosmo_params['Oml']=1.-cosmo_params['Om']-cosmo_params['Omk']

cosmo_params['astropy_cosmo']=cosmo

pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':500,'scenario':'dmo','pk_func':'camb_pk_too_many_z','halofit_version':'takahashi'}# see power_spectra.py 

z_PS=None #redshifts at which to compute power spectra.
nz_PS=100 #number of redshifts to sample power spectra. used if z_PS is none
log_z_PS=2 #grid to generate nz_PS redshifts. 0==linear, 1==log, 2==log+linear. used if z_PS is none



"""
Skylens sub-classes
"""
Ang_PS=None
cov_utils=None
tracer_utils=None
logger=None
stack_data=False

Skylens_default_kwargs=locals()