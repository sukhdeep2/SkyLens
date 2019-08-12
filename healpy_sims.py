import numpy as np
import sys

nside=1024
nsim=10
use_shot_noise=np.bool(sys.argv[1]) 
unit_window=np.bool(sys.argv[2])
print('doing ',use_shot_noise,unit_window)
window_lmax=1000  #0.5 % difference between 100 and 400, for unit window, 2% for realistic window
use_cosmo_power=True
use_window=True
f_sky=0.3
store_win=True

n_source_bins=1
sigma_gamma=0.3944/np.sqrt(2.)  #*2**0.25

wigner_files={}
wigner_files[0]='/global/cscratch1/sd/sukhdeep/temp/dask_wig3j_l6500_w1100_0_reorder.zarr'
wigner_files[2]='/global/cscratch1/sd/sukhdeep/temp/dask_wig3j_l6500_w1100_2_reorder.zarr'


            
import pickle
from cov_3X2 import *
from lsst_utils import *
from scipy.stats import norm,mode,skew,kurtosis,percentileofscore

import tracer_utils
import window_utils
import importlib

import multiprocessing
ncpu=multiprocessing.cpu_count()

from distributed import LocalCluster
from dask.distributed import Client  # we already had this above
#http://distributed.readthedocs.io/en/latest/_modules/distributed/worker.html
worker_kwargs={'memory_spill_fraction':.75,'memory_target_fraction':.99,'memory_pause_fraction':1}
LC=LocalCluster(n_workers=1,processes=False,memory_limit='100gb',threads_per_worker=ncpu,
                local_dir='./temp/NGL-worker/',
               **worker_kwargs,
                #scheduler_port=12234,
#                 dashboard_address=8801
                diagnostics_port=8801,
#                memory_monitor_interval='2000ms')
               )
client=Client(LC,diagnostics_port=8801,)

#LC.close()
#client.close()

lmax_cl=int(nside*2.9)
lmin_cl=0
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl+10
lmax_cl_Bins=lmax_cl-10
Nl_bins=40
l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
# l_bins=np.int64(np.linspace(lmin_cl_Bins,lmax_cl_Bins,Nl_bins))
lb=(l_bins[1:]+l_bins[:-1])*.5

l=l0 #np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell

do_cov=True
bin_cl=True

SSV_cov=False
tidal_SSV_cov=False

do_xi=False

corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')

window_cl_fact=1
if unit_window:
    window_cl_fact=0

l0w=np.arange(512)+1
ww=1000*np.exp(-(l0w-150)**2/50**2)


z0=0.5
zl_bin1=lsst_source_tomo_bins(zp=np.array([z0]),p_zp=np.array([1]),ns=10,use_window=use_window,nbins=1,window_cl_fact=window_cl_fact,
                              f_sky=f_sky,nside=nside,unit_win=False,use_shot_noise=use_shot_noise)
zl_bin1w=lsst_source_tomo_bins(zp=np.array([z0]),p_zp=np.array([1]),ns=30,use_window=use_window,window_cl_fact=window_cl_fact*(1+ww)+1,
                               f_sky=f_sky,nbins=n_source_bins,nside=nside,use_shot_noise=use_shot_noise)

z0=1 #1087
zs_bin1=lsst_source_tomo_bins(zp=np.array([z0]),p_zp=np.array([1]),ns=30,use_window=use_window,window_cl_fact=window_cl_fact,
                              f_sky=f_sky,nbins=n_source_bins,nside=nside,unit_win=False,use_shot_noise=use_shot_noise)

zs_bin1w=lsst_source_tomo_bins(zp=np.array([z0]),p_zp=np.array([1]),ns=30,use_window=use_window,window_cl_fact=(1+ww)*window_cl_fact+1,
                               f_sky=f_sky,nbins=n_source_bins,nside=nside,use_shot_noise=use_shot_noise)

th_min=1./60
th_max=600./60
n_th_bins=20
th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min*0.98),np.log10(1),n_th_bins*30)
th2=np.linspace(1,th_max*1.02,n_th_bins*30)
# th2=np.logspace(np.log10(1),np.log10(th_max),60*6)
th=np.unique(np.sort(np.append(th,th2)))
thb=np.sqrt(th_bins[1:]*th_bins[:-1])
bin_xi=True
# HT_kwargs={'l_min':l_min,  'l_max':l_max,
#                         'theta_min':th_min*d2r*.9, 'theta_max':th_max*d2r,
#                         'n_zeros':40000, 'prune_theta':prune_theta, 'm1_m2':[(2,2),(2,-2),(0,2),(0,0)]}
l0_win=np.arange(lmax_cl)
WT_L_kwargs={'l': l0_win,'theta': th*d2r,'m1_m2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}
WT_L=None
if do_xi:
    WT_L=wigner_transform(**WT_L_kwargs)

corrs=[corr_ll,corr_ggl,corr_gg]

t1=time.time()
kappa_win=cov_3X2(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1,
            use_window=use_window,store_win=store_win,window_lmax=window_lmax,corrs=corrs,
            SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=f_sky,
                  HT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi,wigner_files=wigner_files,
                 #Win=kappa_win.Win.Win
                 )

t2=time.time()
print('W done',t2-t1)
clG_win=kappa_win.cl_tomo(corrs=corrs) 
cl0_win=clG_win['stack'].compute()

if do_xi:
    xiWG_L=kappa_win.xi_tomo()
    xiW_L=xiWG_L['stack'].compute()


kappa_win_w=cov_3X2(zs_bins=zs_bin1w,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1w,
            use_window=use_window,store_win=store_win,window_lmax=window_lmax,corrs=corrs,
            SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=f_sky,
            HT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi,wigner_files=wigner_files
                   #Win=kappa_win_w.Win.Win
                   )

clG_win_w=kappa_win_w.cl_tomo(corrs=corrs) 
cl0_win_w=clG_win_w['stack'].compute()

if do_xi:
    xiWG_L_w=kappa_win_w.xi_tomo()
    xiW_L_w=xiWG_L_w['stack'].compute()

kappa0=cov_3X2(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1,
            use_window=False,store_win=store_win,corrs=corrs,
            SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=f_sky,
               HT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi,
               wigner_files=wigner_files)

clG0=kappa0.cl_tomo(corrs=corrs) 
cl0=clG0['stack'].compute()

if do_xi:
    xiG_L0=kappa0.xi_tomo()
    xi_L0=xiG_L0['stack'].compute()

bi=(0,0)
cl0={'cl_b':{},'cov':{},'cl':{}}
cl0_win_w={'cl_b':{},'cov':{}}
cl0_win={'cl_b':{},'cov':{}}
for corr in corrs:
    cl0['cl_b'][corr]=clG0['cl_b'][corr][bi].compute()
    cl0['cl'][corr]=clG0['cl'][corr][bi].compute()
    cl0['cov'][corr]=clG0['cov'][corr+corr][bi+bi].compute()
    
    cl0_win['cl_b'][corr]=clG_win['cl_b'][corr][bi].compute()
    cl0_win['cov'][corr]=clG_win['cov'][corr+corr][bi+bi].compute()['final_b']
    
    cl0_win_w['cl_b'][corr]=clG_win_w['cl_b'][corr][bi].compute()
    cl0_win_w['cov'][corr]=clG_win_w['cov'][corr+corr][bi+bi].compute()['final_b']

#client.restart()

from binning import *
M_binnings={}
M_binning_utils={}
Mw_binning_utils={}
for corr in corrs:
    M_binnings[corr]=binning()
    wt_b=1./cl0['cl_b'][corr]
    wt0=cl0['cl'][corr]
    M_binning_utils[corr]=M_binnings[corr].bin_utils(r=kappa0.l,r_bins=kappa0.l_bins,
                                                r_dim=2,mat_dims=[1,2],wt_b=wt_b,wt0=wt0)
    Mw_binning_utils[corr]=M_binnings[corr].bin_utils(r=kappa0.l,r_bins=kappa0.l_bins,
                                                r_dim=2,mat_dims=[1,2],wt_b=wt_b,wt0=wt0)

def corr_matrix(cov_mat=[]): #correlation matrix
    diag=np.diag(cov_mat)
    return cov_mat/np.sqrt(np.outer(diag,diag))

def bin_coupling_M(kappa_class,coupling_M): #following https://arxiv.org/pdf/astro-ph/0105302.pdf 
#construct coupling matrix for the binned c_ell. This assumes that the C_ell within a bin follows powerlaw. 
#Without this assumption we cannot undo the effects of binning
    l=kappa_class.l
    bin_M=kappa_win.cl_bin_utils['binning_mat']
    l2=l*(l+1)
    x=l==0
    l2[x]=1
    Q=bin_M.T*np.pi*2/(l2)
    P=bin_M.T*(l2)/(np.pi*2)
    P=P.T/(kappa_class.l_bins[1:]-kappa_class.l_bins[:-1])
    return P.T@coupling_M@Q.T

def bin_coupling_M2(kappa_class,coupling_M): #following https://arxiv.org/pdf/astro-ph/0105302.pdf 
#construct coupling matrix for the binned c_ell. This assumes that the C_ell within a bin follows powerlaw. 
#Without this assumption we cannot undo the effects of binning
    l=kappa_class.l
    bin_M=kappa_win.cl_bin_utils['binning_mat']
    
    l2=l*(l+1)
    x=l==0
    l2[x]=1
    
    Q=bin_M.T*np.pi*2/(l2)**2
    P=bin_M.T*(l2)**2/(np.pi*2)
    P=P.T/(kappa_class.l_bins[1:]-kappa_class.l_bins[:-1])
    return P.T@coupling_M@Q.T

seed=12334
def get_clsim2(clg0,window,mask,kappa_class,coupling_M,coupling_M_inv,ndim,i):
    
    local_state = np.random.RandomState(seed+i)
    cl_map=hp.synfast(clg0,nside=nside,RNG=local_state,new=True,pol=True)

    if ndim>1:
        cl_map[0]*=window['galaxy']
        cl_map[0][mask['galaxy']]=hp.UNSEEN
        cl_map[1]*=window['shear'] #shear_1
        cl_map[2]*=window['shear']#shear_2
        cl_map[1][mask['shear']]=hp.UNSEEN
        cl_map[2][mask['shear']]=hp.UNSEEN
        clpi=hp.anafast(cl_map, lmax=max(l),pol=True) #TT, EE, BB, TE, EB, TB for polarized input map
        clpi=clpi[:,l]
        clpi=clpi[[0,1,3],:]
#             for i in np.arange(6):

    else:
        cl_map*=window
        cl_map[mask]=hp.UNSEEN
        clpi=hp.anafast(cl_map, lmax=max(l),pol=True)[l]
        
    del cl_map

    if ndim>1:
        clpi[0]-=(np.ones_like(clpi[0])*kappa_class.SN[corr_gg][:,0,0])@coupling_M[corr_gg]*use_shot_noise
        clpi[1]-=(np.ones_like(clpi[1])*kappa_class.SN[corr_ll][:,0,0])@coupling_M[corr_ll]*use_shot_noise
        clpi[1]-=(np.ones_like(clpi[1])*kappa_class.SN[corr_ll][:,0,0])@coupling_M['shear_B']*use_shot_noise #remove B-mode leakage

        clgi=[clpi[0]@coupling_M_inv[corr_gg],
              clpi[1]@coupling_M_inv[corr_ll],
              clpi[2]@coupling_M_inv[corr_ggl]]
    else:
        clpi-=(np.ones_like(clpi)*shot_noise)@coupling_M
        clgi=clpi@coupling_M_inv
    clgi=np.array(clgi)
    return [clpi.T,clgi.T]


def calc_sim_stats(sim=[],sim_truth=[],PC=False):
    sim_stats={}
    sim_stats['std']=np.std(sim,axis=0)    
    sim_stats['mean']=np.mean(sim,axis=0)
    sim_stats['median']=np.median(sim,axis=0)
    sim_stats['percentile']=np.percentile(sim,[16,84],axis=0)
    sim_stats['skew']=skew(sim,axis=0)
    sim_stats['kurt']=kurtosis(sim,axis=0)
    sim_stats['cov']=np.cov(sim,rowvar=0)
    
    if not PC:
        try:
            sim_stats['cov_ev'],sim_stats['cov_evec']=np.linalg.eig(sim_stats['cov'])
            sim_stats['PC']={}
            sim_stats['PC']['data']=(sim_stats['cov_evec'].T@sim.T).T
            sim_stats['PC']['stats']=calc_sim_stats(sim=sim_stats['PC']['data'],PC=True)
        except Exception as err:
            print(err)
            sim_stats['PC']=err
    else:
        sim_truth=sim_stats['mean']
    
    sim_stats['percetile_score']=np.zeros_like(sim_stats['std'])
    if len(sim_stats['std'].shape)==1:
        for i in np.arange(len(sim_stats['std'])):
            sim_stats['percetile_score'][i]=percentileofscore(sim[:,i],sim_truth[i])
    elif len(sim_stats['std'].shape)==2:
        for i in np.arange(len(sim_stats['std'])):
            for i_dim in np.arange(2):
                for j_dim in np.arange(2):
                    sim_stats['percetile_score'][i][i_dim,j_dim]=percentileofscore(sim[:,i,i_dim,j_dim],
                                                                                   sim_truth[i,i_dim,j_dim])
    else:
        sim_stats['percetile_score']='not implemented for ndim>2'
    return sim_stats
    
def sim_cl_xi(Rsize=150,do_norm=False,cl0=None,kappa_class=None,fsky=f_sky,zbins=None,use_shot_noise=True,
             convolve_win=False,nside=nside,use_cosmo_power=True):
    l=kappa_class.l
    shear_lcut=l>=2
    
    l_bins=kappa_class.l_bins
    dl=l_bins[1:]-l_bins[:-1]
    nu=(2.*l+1.)*fsky
    
    coupling_M={}
    coupling_M4={}
    coupling_M_binned={}
    coupling_M_binned2={}
    coupling_M_binned2wt={}
    coupling_M4_binned={}
    coupling_M4_binned2={}
    
    coupling_M_inv={}
    coupling_M_binned_inv={}
    coupling_M_binned2_inv={}
    coupling_M_binned2wt_inv={}
    mask={}
    window={}
    if convolve_win:
        nu=2.*l+1.
        
        for tracer in kappa_class.z_bins.keys():
            window[tracer]=kappa_class.z_bins[tracer][0]['window']
            mask[tracer]=window[tracer]==hp.UNSEEN
        for corr in corrs:
            coupling_M[corr]=kappa_class.Win.Win['cl'][corr][(0,0)]['M']
            if corr==corr_ll:
                coupling_M['shear_B']=kappa_class.Win.Win['cl'][corr][(0,0)]['M_B']
            coupling_M_binned[corr]=bin_coupling_M(kappa_class,coupling_M[corr])
            coupling_M_binned2[corr]=kappa_class.binning.bin_2d(cov=coupling_M[corr],bin_utils=kappa_class.cl_bin_utils) 
            coupling_M_binned2[corr]*=dl
            
#             coupling_M_binned2wt[corr]=M_binnings[corr].bin_2d(cov=coupling_M[corr],bin_utils=M_binning_utils[corr])
#             coupling_M_binned2wt[corr]*=dl
            
            coupling_M_binned2wt[corr]=M_binnings[corr].bin_2d_coupling(cov=coupling_M[corr].T,bin_utils=M_binning_utils[corr])
            coupling_M_binned2wt[corr]=coupling_M_binned2wt[corr].T  #to keep the same order in dot product later. Remeber that the coupling matrix is not symmetric.

            
#             coupling_M4=kappa_win.Win.Win['cov'][corr+corr][(0,0,0,0)]['M1324'][s] #*2
#             coupling_M4_binned[corr]=bin_coupling_M(kappa_class,coupling_M4[corr])
#             coupling_M4_binned2[corr]=kappa_class.binning.bin_2d(cov=coupling_M4[corr],bin_utils=kappa_class.cl_bin_utils) 
#             coupling_M4_binned2[corr]*=dl
#         kappa_class.binning.bin_2d(cov=coupling_M,bin_utils=kappa_win.cl_bin_utils)
#             print(corr,coupling_M[corr])
            
            cut=l>0
            if 'shear' in corr:
                cut=shear_lcut 
            coupling_M_inv[corr]=np.zeros_like(coupling_M[corr])
            coupling_M_inv[corr][:,cut][cut,:]=np.linalg.inv(coupling_M[corr][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
            
            coupling_M_binned_inv[corr]=np.linalg.inv(coupling_M_binned[corr])
            coupling_M_binned2_inv[corr]=np.linalg.inv(coupling_M_binned2[corr])
            coupling_M_binned2wt_inv[corr]=np.linalg.inv(coupling_M_binned2wt[corr])
    outp={}
    win=0
    clp_shear_B=None
    if cl0 is None:
        cl0={}
        clp0={}
        clG0=kappa_class.cl_tomo() 
        for corr in kappa_class.corrs:
            cl0[corr]=clG0['cl'][corr][(0,0)].compute()
            clp0[corr]=clG0['cl_b'][corr][(0,0)].compute()
    clg0={}
    for corr in kappa_class.corrs: #ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
        shot_noise=0
        if corr[0]==corr[1]:
            shot_noise=kappa_class.SN[corr][:,0,0]
        shot_noise=shot_noise*use_shot_noise
        clg0[corr]=cl0[corr]*use_cosmo_power+shot_noise
        if corr==corr_ll:
            clg0['shear_B']=cl0[corr]*0+shot_noise
            clp_shear_B=shot_noise@coupling_M[corr_ll]+cl0[corr_ll]@coupling_M['shear_B']
    ndim=len(kappa_class.corrs)
    print('ndim:',ndim)
    outp['clg0_0']=clg0.copy()
    outp['ndim']=ndim
    if ndim>1:
        clg0=(clg0[corr_gg],clg0[corr_ll],clg0['shear_B'],clg0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
    else:
        clg0=clg0[corr_gg]
    
    SN=kappa_class.SN
    sim_cl_shape=(Rsize,len(kappa_class.l),ndim)
    
    clp=np.zeros(sim_cl_shape,dtype='float32')
    clg=np.zeros(sim_cl_shape,dtype='float32')
    clpB=np.zeros(sim_cl_shape,dtype='float32')
    lmax=max(l)
    lmin=min(l)
    
    
    clg_b=None
    clp_b=None
    clpB_b=None
    nu_b=None
    if l_bins is not None:
        clg0_b={corr: kappa_class.binning.bin_1d(xi=cl0[corr],bin_utils=kappa_class.cl_bin_utils) for corr in kappa_class.corrs} 
        ll=kappa_class.cl_bin_utils['bin_center']
        sim_clb_shape=(Rsize,len(ll),ndim)
        nu_b=(2.*ll+1.)*fsky*(l_bins[1:]-l_bins[:-1])
        clg_b=np.zeros(sim_clb_shape,dtype='float32')
        clg_bM=np.zeros(sim_clb_shape,dtype='float32') #master
        clg_bM2=np.zeros(sim_clb_shape,dtype='float32') #different master
        clg_bM2wt=np.zeros(sim_clb_shape,dtype='float32')# weighted different master
        clp_b=np.zeros(sim_clb_shape,dtype='float32')
        clpB_b=np.zeros(sim_clb_shape,dtype='float32')
        
        binning_func=kappa_class.binning.bin_1d
        binning_utils=kappa_class.cl_bin_utils
        
        clp_shear_B_b=binning_func(xi=clp_shear_B,bin_utils=binning_utils)
        
    corr_t=[corr_gg,corr_ll,corr_ggl] #order in which sim corrs are output.
    seed=12334
    def get_clsim(i):
        print('doing map: ',i)
        local_state = np.random.RandomState(seed+i)
        cl_map=hp.synfast(clg0,nside=nside,RNG=local_state,new=True,pol=True)
        
        if ndim>1:
            cl_map[0]*=window['galaxy']
            cl_map[0][mask['galaxy']]=hp.UNSEEN
            cl_map[1]*=window['shear'] #shear_1
            cl_map[2]*=window['shear']#shear_2
            cl_map[1][mask['shear']]=hp.UNSEEN
            cl_map[2][mask['shear']]=hp.UNSEEN
            clpi=hp.anafast(cl_map, lmax=max(l),pol=True) #TT, EE, BB, TE, EB, TB for polarized input map
            clpi=clpi[:,l]
            clpi_B=clpi[[2,4,5],:]
            clpi=clpi[[0,1,3],:]
            
#             for i in np.arange(6):
            
            
        else:
            cl_map*=window
            cl_map[mask]=hp.UNSEEN
            clpi=hp.anafast(cl_map, lmax=max(l),pol=True)[l]

        if ndim>1:
            clpi[0]-=(np.ones_like(clpi[0])*SN[corr_gg][:,0,0])@coupling_M[corr_gg]*use_shot_noise
            clpi[1]-=(np.ones_like(clpi[1])*SN[corr_ll][:,0,0])@coupling_M[corr_ll]*use_shot_noise
            clpi[1]-=(np.ones_like(clpi[1])*SN[corr_ll][:,0,0])@coupling_M['shear_B']*use_shot_noise #remove B-mode 
            
            clgi=[clpi[0]@coupling_M_inv[corr_gg],
                  clpi[1]@coupling_M_inv[corr_ll],
                  clpi[2]@coupling_M_inv[corr_ggl]]
        else:
            clpi-=(np.ones_like(clpi)*shot_noise)@coupling_M
            clgi=clpi@coupling_M_inv
        clgi=np.array(clgi)
        if l_bins is not None:
            corr_t=[corr_gg,corr_ll,corr_ggl]
            sim_clb_shape=(len(ll),ndim)
            clg_b=np.zeros(sim_clb_shape,dtype='float32')
            clg_bM=np.zeros(sim_clb_shape,dtype='float32')
            clg_bM2=np.zeros(sim_clb_shape,dtype='float32')
            clg_bM2wt=np.zeros(sim_clb_shape,dtype='float32')
            clp_b=np.zeros(sim_clb_shape,dtype='float32')
            clpB_b=np.zeros(sim_clb_shape,dtype='float32')
            for ii in np.arange(ndim):
                clg_b[:,ii]=binning_func(xi=clgi[ii],bin_utils=binning_utils)
                clp_b[:,ii]=binning_func(xi=clpi[ii],bin_utils=binning_utils)
                clpB_b[:,ii]=binning_func(xi=clpi_B[ii],bin_utils=binning_utils)
                clg_bM[:,ii]=clp_b[:,ii]@coupling_M_binned_inv[corr_t[ii]] #be careful with ordering as coupling matrix is not symmetric
                clg_bM2[:,ii]=clp_b[:,ii]@coupling_M_binned2_inv[corr_t[ii]]
                clg_bM2wt[:,ii]=clp_b[:,ii]@coupling_M_binned2wt_inv[corr_t[ii]]
            return clpi.T,clgi.T,clg_b,clp_b,clg_bM,clg_bM2,clg_bM2wt,clpi_B.T,clpB_b
        else:
            return clpi.T,clgi.T
    
    def comb_maps(futures):
        for i in np.arange(Rsize):
            x=futures[i]#.compute()
            clp[i,:,:]+=x[0]
            clg[i,:,:]+=x[1]
        return clp,clg 
    
    print('generating maps')
    if convolve_win:
        futures={}
#         for i in np.arange(Rsize):
#             futures[i]=dask.delayed(get_clsim)(i)  
#         print(futures)
#         clpg=dask.delayed(comb_maps)(futures)
#         clpg.compute()
        i=0
        j=0
        step=min(5,Rsize)
        #funct=partial(get_clsim2,clg0,window,mask,SN,coupling_M,coupling_M_inv,ndim)
        while j<Rsize:
            futures={}
            for ii in np.arange(step):
                futures[ii]=delayed(get_clsim)(i+ii)  
            futures=client.compute(futures)
            futures=futures.result()
            #print('got the futures')
            for ii in np.arange(step):
                if l_bins is None:
                    clp[i,:],clg[i,:]=futures[ii]#.result()[ii]
                else:
                    clp[i,:],clg[i,:],clg_b[i,:],clp_b[i,:],clg_bM[i,:],clg_bM2[i,:],clg_bM2wt[i,:],clpB[i,:],clpB_b[i,:]=futures[ii]#.result()[ii]
                i+=1
            print('done map ',i)
            del futures
            #client.restart()
            j+=step
        
    print('done generating maps')
    
            
    outp['clg_b_stats']=client.compute({corr_t[ii]: delayed(calc_sim_stats)(sim=clg_b[:,:,ii],sim_truth=clg0_b[corr_t[ii]]) for ii in np.arange(ndim)})
    outp['clg_bM_stats']=client.compute({corr_t[ii]: delayed(calc_sim_stats)(sim=clg_bM[:,:,ii],sim_truth=clg0_b[corr_t[ii]]) for ii in np.arange(ndim)})
    outp['clg_bM2_stats']=client.compute({corr_t[ii]: delayed(calc_sim_stats)(sim=clg_bM2[:,:,ii],sim_truth=clg0_b[corr_t[ii]]) for ii in np.arange(ndim)})
    outp['clg_bM2wt_stats']=client.compute({corr_t[ii]: delayed(calc_sim_stats)(sim=clg_bM2wt[:,:,ii],sim_truth=clg0_b[corr_t[ii]]) for ii in np.arange(ndim)})
    outp['clp_b_stats']=client.compute({corr_t[ii]: delayed(calc_sim_stats)(sim=clp_b[:,:,ii],sim_truth=clp_b[:,:,ii].mean(axis=0)) for ii in np.arange(ndim)})
    outp['clpB_b_stats']=client.compute({corr_t[ii]: delayed(calc_sim_stats)(sim=clpB_b[:,:,ii],sim_truth=clpB_b[:,:,ii].mean(axis=0)) for ii in np.arange(ndim)})
    
    outp['clg_b_stats']=outp['clg_b_stats'].result()
    outp['clg_bM_stats']=outp['clg_bM_stats'].result()
    outp['clg_bM2_stats']=outp['clg_bM2_stats'].result()
    outp['clg_bM2wt_stats']=outp['clg_bM2wt_stats'].result()
    outp['clp_b_stats']=outp['clp_b_stats'].result()
    outp['clpB_b_stats']=outp['clpB_b_stats'].result()
        
#         outp['clp_b_stats']=calc_sim_stats(sim=clp_b,sim_truth=clp_b.mean(axis=0))
#     xiN=np.zeros((Rsize,len(xi)))
#     xig=np.zeros((Rsize,len(xi)))
#     xigB=np.zeros((Rsize,len(r_bins)-1))
#     xiNB=np.zeros((Rsize,len(r_bins)-1))
#     for i in np.arange(Rsize):
#         r,xig[i,:]=HT.projected_correlation(k_pk=l,pk=clg[i,:],j_nu=0,taper=True,**taper_kw)
#         rb,xigB[i,:]=HT.bin_mat(r=r,mat=xig[i,:],r_bins=r_bins)
#         if do_clN:
#             r,xiN[i,:]=HT.projected_correlation(k_pk=l,pk=clN[i,:],j_nu=0,taper=True,**taper_kw)
#             rb,xiNB[i,:]=HT.bin_mat(r=r,mat=xiN[i,:],r_bins=r_bins)
#     outp['xi_truth']=xi_truth
#    outp['rb']=rb
    outp['clg']=clg
    outp['clp']=clp
    outp['clpB']=clpB
    outp['clg_b']=clg_b
    outp['clg_bM']=clg_bM
    outp['clg_bM2']=clg_bM2
    outp['clg_bM2wt']=clg_bM2wt
    outp['clp_b']=clp_b
    outp['clpB_b']=clpB_b
    outp['clg0']=clg0
    outp['cl0']=cl0
    outp['clp0']=clp0
    outp['clp_shear_B_b']=clp_shear_B_b
    outp['clp_shear_B']=clp_shear_B
#     outp['clN']=clN
#     outp['xig']=xig
#     outp['xigB']=xigB
#     outp['xiNB']=xiNB
#     outp['xiN']=xiN
    clg0_2=np.array(clg0)[[0,1,3],:]
    outp['clg_stats']={corr_t[ii]: calc_sim_stats(sim=clg[:,:,ii],sim_truth=clg0_2[ii]) for ii in np.arange(ndim)}#calc_sim_stats(sim=clg,sim_truth=clg0)
    outp['clp_stats']={corr_t[ii]: calc_sim_stats(sim=clp[:,:,ii],sim_truth=clp[:,:,ii].mean(axis=0)) for ii in np.arange(ndim)}#     calc_sim_stats(sim=clp,sim_truth=clp.mean(axis=0))

#     outp['xig_stats']=calc_sim_stats(sim=xig,sim_truth=xi)
#     if convolve_win:
#         outp['xig_stats0']=calc_sim_stats(sim=xig,sim_truth=xi0)
#     rb,xiB=HT.bin_mat(r=r,mat=xi_truth,r_bins=r_bins)
#     outp['xigB_stats']=calc_sim_stats(sim=xigB,sim_truth=xiB)
#     if do_clN:
#         outp['xiN_stats']=calc_sim_stats(sim=xiN,sim_truth=xi_truth)
#         outp['xiNB_stats']=calc_sim_stats(sim=xiNB,sim_truth=xiB)

    outp['size']=Rsize
    outp['fsky']=fsky
    outp['nu']=nu
    outp['nu_b']=nu_b
    outp['l_bins']=l_bins
    
    outp['coupling_M']=coupling_M
    outp['coupling_M_binned']=coupling_M_binned
    outp['coupling_M_binned2']=coupling_M_binned2
    outp['coupling_M_binned2wt']=coupling_M_binned2wt
    
    outp['coupling_M_inv']=coupling_M_inv
    outp['coupling_M_binned_inv']=coupling_M_binned_inv
    outp['coupling_M_binned2_inv']=coupling_M_binned2_inv
    outp['coupling_M_binned2wt_inv']=coupling_M_binned2wt_inv
    outp['use_shot_noise']=use_shot_noise
    
    
    return outp

#cov=np.diag(cl**2/nu)

fname='./tests/non_gaussian_likeli_sims'+str(nsim)+'_ns'+str(nside)+'_lmax'+str(lmax_cl)+'_wlmax'+str(window_lmax)+'_fsky'+str(f_sky)
if not use_shot_noise:
    fname+='_noSN'
if unit_window:
    fname+='_unit_window'
fname+='.pkl'

print(fname)

cl_sim_W=sim_cl_xi(Rsize=nsim,do_norm=False,#cl0=clG0['cl'][corrs[0]][(0,0)].compute(),
          kappa_class=kappa_win,fsky=f_sky,use_shot_noise=use_shot_noise,use_cosmo_power=use_cosmo_power,
             convolve_win=True,nside=nside)

cl_sim_Ww=sim_cl_xi(Rsize=nsim,do_norm=False,#cl0=clG0['cl'][corrs[0]][(0,0)].compute(),
          kappa_class=kappa_win_w,fsky=f_sky,use_shot_noise=use_shot_noise,
             convolve_win=True,nside=nside)



outp={}
outp['simW']=cl_sim_W
outp['zs_bin1']=zs_bin1
outp['zl_bin1']=zl_bin1
outp['cl0']=cl0
outp['cl0_win']=cl0_win

outp['simWw']=cl_sim_Ww
outp['zs_bin1w']=zs_bin1w
outp['zl_bin1w']=zl_bin1w
outp['cl0_win_w']=cl0_win_w


with open(fname,'wb') as f:
    pickle.dump(outp,f)
written=True
