import sys
import traceback
# import pyccl as ccl
import pickle
import camb
sys.path.insert(0,'../skylens/')

from distributed import LocalCluster
from dask.distributed import Client  # we already had this above
#http://distributed.readthedocs.io/en/latest/_modules/distributed/worker.html

from skylens import *
from survey_utils import *

#only for python3
import importlib
reload=importlib.reload


LC,scheduler_info=start_client(Scheduler_file=None,local_directory='../temp/',ncpu=None,n_workers=1,threads_per_worker=None,
                              memory_limit='120gb',dashboard_address=8801)
client=client_get(scheduler_info=scheduler_info)


wigner_files={}
# wigner_files[0]= '/Users/Deep/dask_temp/dask_wig3j_l3500_w2100_0_reorder.zarr'
# wigner_files[2]= '/Users/Deep/dask_temp/dask_wig3j_l3500_w2100_2_reorder.zarr'
wig_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/'
wigner_files[0]= wig_home+'dask_wig3j_l3500_w2100_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l3500_w2100_2_reorder.zarr'

#setup parameters
lmax_cl=200
lmin_cl=2
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl+10
lmax_cl_Bins=lmax_cl-10
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
theta_bins=np.logspace(np.log10(1./60),1,20)

nside=32
window_lmax=nside

use_window=True
do_cov=True


# In[29]:


do_xi=True
bin_xi=True
bin_cl=True
th_min=2.5/60
th_max=250./60
n_th_bins=20
th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins*40)
thb=np.sqrt(th_bins[1:]*th_bins[:-1])

#Hankel Transform setup
WT_kwargs={'l':l0,'theta':th,'s1_s2':[(2,2),(2,-2),(0,0),(0,2),(2,0)]}
WT=wigner_transform(**WT_kwargs)


# In[24]:


z0=1 #1087
# zs_bin1=source_tomo_bins(zp=[z0],p_zp=np.array([1]),ns=30,use_window=use_window,nside=nside)
zs_bin1=lsst_source_tomo_bins(nbins=2,use_window=use_window,nside=nside)


corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
corrs=[corr_ll,corr_ggl,corr_gg]

use_binned_ls=[False,True]

store_wins=[False,True]

# In[21]:


SSV_covs=[False,True]
bin_cl=True
do_covs=[True,False]
# Tri_cov=[False,True]

do_pseudo_cls=[False,True]
do_xis=[True,True]
use_windows=[True,False]

passed=0
failed=0
failed_tests={}
traceback_tests={}
for do_xi in do_xis:
    for do_pseudo_cl in do_pseudo_cls:
        if do_xi==do_pseudo_cl:
            continue
        for use_window in use_windows:
            for do_cov in do_covs:
                for SSV_cov in SSV_covs:
                    Tri_cov=SSV_cov
                    for use_binned_l in use_binned_ls:
                        for store_win in store_wins:
                            s=''
                            s=s+' do_xi ' if do_xi else s+' do_cl '
                            s=s+' use_window ' if use_window else s
                            s=s+' do_cov ' if do_xi else s
                            s=s+' SSV_cov ' if do_xi else s
                            s=s+' use_binned_l ' if do_xi else s
                            s=s+' store_win ' if do_xi else s
                            print("\n","\n")
                            print('passed failed: ',passed, failed, ' now testing ',s)
                            print('tests that failed: ',failed_tests)
                            print("\n","\n")
                            try:
                                kappa0=Skylens(shear_zbins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, galaxy_zbins=zs_bin1,
                                               use_window=use_window,Tri_cov=Tri_cov,
                                               use_binned_l=use_binned_l,wigner_files=wigner_files,
                                               SSV_cov=SSV_cov,tidal_SSV_cov=SSV_cov,f_sky=0.35,
                                               store_win=store_win,window_lmax=window_lmax,
                                               sparse_cov=True,corrs=corrs,
                                               do_xi=do_xi,bin_xi=bin_xi,theta_bins=th_bins,WT=WT,
                                                use_binned_theta=use_binned_l
                                               )
                                if do_xi:
                                    G=kappa0.xi_tomo()
                                    xi_bin_utils=client.gather(kappa0.xi_bin_utils)
                                else:
                                    G=kappa0.cl_tomo()
                                cc=client.compute(G['stack']).result()
                                
                                kappa0.gather_data()
#                                 kappa0.scatter_data()
                                xi_bin_utils=kappa0.xi_bin_utils
                                cl_bin_utils=kappa0.cl_bin_utils
# #                                 kappa0.Ang_PS.clz=client.gather(kappa0.Ang_PS.clz)
#                                 kappa0.WT.gather_data()
                                WT_binned=kappa0.WT_binned
                                
                                cS=delayed(kappa0.tomo_short)(cosmo_params=kappa0.Ang_PS.PS.cosmo_params,Win=kappa0.Win,WT=kappa0.WT,
                                                    WT_binned=WT_binned,Ang_PS=kappa0.Ang_PS,zkernel=G['zkernel'],xi_bin_utils=xi_bin_utils,
                                                             cl_bin_utils=cl_bin_utils,z_bins=kappa0.z_bins)
                                cc=client.compute(cS).result()
                                passed+=1
                                
                            except Exception as err:
                                print(s,' failed with error ',err)
                                print(traceback.format_exc())
                                failed_tests[failed]=s+' failed with error '+str(err)
                                traceback_tests[failed]=str(traceback.format_exc())
                                failed+=1
#                                 crash

for i in failed_tests.keys():
    print(failed_tests[i])
    print(traceback_tests[i])