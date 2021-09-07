import sys
from skylens.utils import *
from skylens.survey_utils import *

"""to start a dask cluster"""
LC,scheduler_info=start_client(Scheduler_file=None,local_directory='../temp/',ncpu=None,n_workers=1,
                                threads_per_worker=None,
                                memory_limit='20gb',dashboard_address=8801)
client=client_get(scheduler_info=scheduler_info)

"""tracer pairs for correlations"""
#tracers currently supported: galaxy,shear,kappa
corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
corr_kkl=('galaxy','kappa')
corr_kk=('kappa','kappa')

corrs=[corr_ll,corr_ggl,corr_gg] #tracer pairs to use. This is passed to skylens

stack_indxs=None #zbin pairs for each tracer pair to correlate. by default, use all possible pairs.
                # e.g.
                #stack_indxs[corr_gg]=[(0,0),(0,1),(1,1)]
                #stack_indxs[corr_ggl]=[(0,0),(0,1),(1,0),(1,1)]

cl_func_names={} # if we want to use custom functions for cl computations. must be in globals()
                 # e.g. cl_func_names[corr_gg]='calc_cl',  cl_func_names[corr_ggl]='calc_cl'
                 # default is calc_cl defined in skylens_main.py file.
                # calc_cl gets following inputs: 
                # zbin1={}, zbin2={},corr=('shear','shear'),cosmo_params=None,clz=None,Ang_PS=None
                # Ang_PS contains cosmology object, i.e. Ang_PS.PS, which also has power spectra, p(z,k).

zkernel_func_names={} #if we want to use custom functions for redshift kernels. must be in globals()
                    # e.g. zkernel_func_names['shear']='set_kernel',
                    # if you have a custom cl function and prefer not to compute a redshift kernel,
                    # define a function that returns the input z_bin, which will be passed along to 
                    # custom cl_calc. z_bin is user input and should have the required parameters such
                    # as galaxy bias. kernel_func gets following inputs:
                    # l,cosmo_h=cosmology_object,zl=redshifts_at which power spectra are computed,tracer=None,z_bin=dictionary containing the tracer_zbin[i]
    
"""C_ell"""
lmax_cl=100
lmin_cl=2
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl+1
lmax_cl_Bins=lmax_cl-1
Nl_bins=20
l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
lb=np.sqrt(l_bins[1:]*l_bins[:-1])

l=l0
# l=np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell

bin_cl=True
do_pseudo_cl=True
use_binned_l=True

"""correlation functions"""
do_xi=False
bin_xi=True
use_binned_theta=True
theta_min=2.5/60
theta_max=25./60
n_theta_bins=15
theta_bins=np.logspace(np.log10(theta_min),np.log10(theta_max),n_theta_bins+1)
theta=np.logspace(np.log10(theta_min),np.log10(theta_max),n_theta_bins*10)
thb=.5*(theta_bins[1:]+theta_bins[:-1])

#Hankel Transform setup
WT_kwargs={'l':l0,'theta':theta,'s1_s2':[(2,2),(2,-2),(0,0),(0,2),(2,0)],'scheduler_info':scheduler_info}
WT=None #wigner_transform(**WT_kwargs)


"""window calculations"""
#relevant for pseudo_cl, correlation functions and covariances.
use_window=True
nside=16 #nside is not used by skylens.
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
wig_home='./tests/'
wigner_files[0]= wig_home+'dask_wig3j_l100_w100_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l100_w100_2_reorder.zarr'

"""covariance"""
do_cov=True
SSV_cov=True
tidal_SSV_cov=False
Tri_cov=True
sparse_cov=True #store covariances as sparse matrices
do_sample_variance=True #if false, only shot noise is used in gaussian covariance
xi_SN_analytical=True #use analytical expression for correlation function shot noise. if False, Cl shot noise is transformed.

f_sky=0.35 #if there is no window. This can also be a dictionary for different correlation pairs, to account for partial overlaps, etc.
            # e.g. f_sky[corr_gg][(0,0)]=0.35, f_sky[corr_ggl][(0,0)]=0.1
            # for covariance, f_sky[corr_gg+corr_ggl][(0,0,0,1)]=0  (0,0,0,1)==(0,0)+(0,1)

"""generate simulated samples"""
#for this example. You should define your own tracer_zbins
nzbins=2
shear_zbins=lsst_source_tomo_bins(nbins=nzbins,use_window=use_window,nside=nside,scheduler_info=scheduler_info,n_zs=100)
galaxy_zbins=shear_zbins
"""tracer_zbins are expected to be nested dictionaries with following structure:
tracer_zbin={ n_bins: 2, #total number of z_bins
                SN: [], shot noise for covariance (if needed). this should be a 3D matrix of form n_ell X n_bins X n_bins
                i:{ #dictionary for the ith z_bin. i goes from 0->n_bins-1. with following items
                        'pz':[], #dn/dz
                        'z':[], #redshift at which dn/dz is sampled
                        'dz': np.gradient(z),
                        'pzdz':pz*dz, #we save this because it is used repeadtedly in the calculations
                        #other relevant parameters, such as
                        'galaxy_bias_func':None, #custom galaxy bias function if defined. must be in globals(). should return b(z,l). 
                        'b1':1 , #linear galaxy bias, for galaxies
                        'bz1':None, #linear galaxy bias varied with z. should pass 'galaxy_bias_func'=linear_bias_z' or a custom function
                        'mag_fact':0,# magnification bias prefactor , for galaxies
                        'AI':0, #IA amplitude, for shear 
                        'shear_m_bias':1, shear multiplicative bias (1+m), default is 1 (no bias). 
                    }
               }
"""


"""cosmology and power spectra"""
from astropy.cosmology import Planck15 as cosmo
cosmo_params=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'OmR':cosmo.Ogamma0+cosmo.Onu0,'w':-1,'wa':0,'Tcmb':cosmo.Tcmb0})
cosmo_params['Oml']=1.-cosmo_params['Om']-cosmo_params['Omk']
pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':500,'scenario':'dmo','pk_func':'camb_pk_too_many_z','halofit_version':'takahashi'}

z_PS=None #redshifts at which to compute power spectra.
nz_PS=100 #number of redshifts to sample power spectra. used if z_PS is none
log_z_PS=2 #grid to generate nz_PS redshifts. 0==linear, 1==log, 2==log+linear. used if z_PS is none

# from skylens.parse_input import parse_dict
#skylens_kwargs=parse_dict(locals()) #in case we want to get a dictionary of arguments for skylens from this file. skylens will use the same function after running this file

if __name__=='__main__':
    from skylens import *
    kappa0=Skylens(python_inp_file='./tests/skylens_args.py')
#     kappa0=Skylens(**skylens_kwargs) #this can be used if you uncomment line above defining skylens_kwargs