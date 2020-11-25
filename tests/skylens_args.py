import sys
import pickle
import camb

from skylens import *
from skylens.survey_utils import *

LC,scheduler_info=start_client(Scheduler_file=None,local_directory='../temp/',ncpu=None,n_workers=1,threads_per_worker=None,
                              memory_limit='120gb',dashboard_address=8801)
client=client_get(scheduler_info=scheduler_info)

wigner_files={}
wig_home='./tests/'
wigner_files[0]= wig_home+'dask_wig3j_l100_w100_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l100_w100_2_reorder.zarr'

#setup parameters
lmax_cl=100
lmin_cl=2
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl+1
lmax_cl_Bins=lmax_cl-1
Nl_bins=20
l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
lb=np.sqrt(l_bins[1:]*l_bins[:-1])

l=np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell

do_cov=True
bin_cl=True

SSV_cov=True
tidal_SSV_cov=False
Tri_cov=True

bin_xi=True

nside=16
window_lmax=nside

use_window=True
do_cov=True

use_binned_l=True

store_win=True

SSV_cov=True
bin_cl=True
do_covs=True
Tri_cov=True

do_pseudo_cl=True

do_xi=False

bin_xi=True
bin_cl=True
th_min=2.5/60
th_max=25./60
n_th_bins=15
th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins*10)
thb=np.sqrt(th_bins[1:]*th_bins[:-1])

#Hankel Transform setup
WT_kwargs={'l':l0,'theta':th,'s1_s2':[(2,2),(2,-2),(0,0),(0,2),(2,0)]}
WT=wigner_transform(**WT_kwargs)

z0=1 #1087
nzbins=1
zs_bin1=lsst_source_tomo_bins(nbins=nzbins,use_window=use_window,nside=nside)
f_sky=0.35
sparse_cov=True

corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
corrs=[corr_ll,corr_ggl,corr_gg]

nz_PS=30

Skylens_kwargs={'zs_bins':zs_bin1,'do_cov':do_cov,'bin_cl':bin_cl,'l_bins':l_bins,'l':l0, 'zg_bins':zs_bin1,'nz_PS':nz_PS,
                'use_window':use_window,'Tri_cov':Tri_cov,'use_binned_l':use_binned_l,'wigner_files':wigner_files,
                 'SSV_cov':SSV_cov,'tidal_SSV_cov':SSV_cov,'f_sky':f_sky,'store_win':store_win,
                'window_lmax':window_lmax,'sparse_cov':sparse_cov,'corrs':corrs,'do_xi':do_xi,'bin_xi':bin_xi,
                'theta_bins':th_bins,'WT':WT,'use_binned_theta':use_binned_l}
