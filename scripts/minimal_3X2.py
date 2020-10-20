import sys, os, gc, threading, subprocess
sys.path.insert(0,'/verafs/scratch/phy200040p/sukhdeep/project/skylens/skylens/')

from distributed import LocalCluster
from dask.distributed import Client

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--scheduler", "-s", help="Scheduler file")
args = parser.parse_args()
Scheduler_file=args.scheduler
print('scheduler ',Scheduler_file)
if Scheduler_file is None:
    worker_kwargs={'memory_spill_fraction':.75,'memory_target_fraction':.99,'memory_pause_fraction':1}
    LC=LocalCluster(n_workers=10,processes=False,memory_limit='20GB',threads_per_worker=1,
                local_dir='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/NGL-worker/', **worker_kwargs,
                #scheduler_port=12234,
#                 dashboard_address=8801
                diagnostics_port=8801,
#                memory_monitor_interval='2000ms')
               )
    client=Client(LC,)#diagnostics_port=8801,)
else:
    client=Client(scheduler_file=Scheduler_file)
#    client.restart()
print(client)

from skylens import *
from survey_utils import *

wigner_files={}
# wigner_files[0]= '/Users/Deep/dask_temp/dask_wig3j_l3500_w2100_0_reorder.zarr'
# wigner_files[2]= '/Users/Deep/dask_temp/dask_wig3j_l3500_w2100_2_reorder.zarr'

wigner_files[0]= '/home/deep/data/repos/SkyLens/temp/dask_wig3j_l6500_w2100_0_reorder.zarr'
wigner_files[2]= '/home/deep/data/repos/SkyLens/temp/dask_wig3j_l3500_w2100_2_reorder.zarr'


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

store_win=True
window_lmax=200

use_window=False
do_cov=True

nside=128

z0=1 #1087
zs_bin1=source_tomo_bins(zp=[z0],p_zp=np.array([1]),ns=30,use_window=use_window,nside=nside)

print('got z bin')

use_binned_l=True
store_win=True
SSV_cov=False
bin_cl=False
do_cov=True
Tri_cov=False

#use all ell
kappa0=Skylens(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=None,
               use_window=use_window,Tri_cov=Tri_cov,
               use_binned_l=use_binned_l,wigner_files=wigner_files,
               SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=0.35,
               store_win=store_win,window_lmax=window_lmax,
               sparse_cov=True
)

print('kappa0 done')

cl0G=kappa0.cl_tomo()
print('graph done')

cl0=cl0G['stack'].compute()

print('stack done')
