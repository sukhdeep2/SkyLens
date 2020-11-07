#FIXME: 
# 1. need to save SN from kappa_class
# 2. save window correlation from treecorr and theory

import sys, os, gc, threading, subprocess
sys.path.insert(0,'/verafs/scratch/phy200040p/sukhdeep/project/skylens/skylens/')
from thread_count import *

from resource import getrusage, RUSAGE_SELF
import psutil
from distributed.utils import format_bytes


debug=False
if debug:
    import faulthandler; faulthandler.enable()
    #in case getting some weird seg fault, run as python -Xfaulthandler run_sim_jk.py
    # problem is likely to be in some package

import skylens
import pickle
import treecorr
from skylens import *
#from survey_utils import *
from jk_utils import *
from binning import *

from scipy.stats import norm,mode,skew,kurtosis,percentileofscore

import sys
import tracemalloc

from dask_mpi import initialize as dask_initialize
from distributed import Client
from distributed import LocalCluster
from dask.distributed import Client  # we already had this above
#http://distributed.readthedocs.io/en/latest/_modules/distributed/worker.html

import argparse

test_run=True
parser = argparse.ArgumentParser()
parser.add_argument("--cw", "-cw",type=int, help="use complicated window")
parser.add_argument("--uw", "-uw",type=int, help="use unit window")
parser.add_argument("--lognormal", "-l",type=int, help="use complicated window")
parser.add_argument("--blending", "-b",type=int, help="use complicated window")
parser.add_argument("--ssv", "-ssv",type=int, help="use complicated window")
parser.add_argument("--noise", "-sn",type=int, help="use complicated window")
parser.add_argument("--xi", "-xi",type=int, help="do_xi, i.e. compute correlation functions")
parser.add_argument("--pseudo_cl", "-pcl",type=int, help="do_pseudo_cl, i.e. compute power spectra functions")
parser.add_argument("--njk", "-njk",type=int, help="number of jackknife regions, default=0 (no jackknife)")
parser.add_argument("--scheduler", "-s", help="Scheduler file")
parser.add_argument("--dask_dir", "-Dd", help="dask log directory")
args = parser.parse_args()


# Read arguments from the command line
args = parser.parse_args()

gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
gc.enable()

use_complicated_window=False if not args.cw else np.bool(args.cw)
unit_window=False if args.uw is None else np.bool(args.uw)
lognormal=False if not args.lognormal else np.bool(args.lognormal)

do_blending=False if not args.blending else np.bool(args.blending)
do_SSV_sim=False if not args.ssv else np.bool(args.ssv)
use_shot_noise=True if args.noise is None else np.bool(args.noise)
Scheduler_file=args.scheduler
dask_dir=args.dask_dir

# Scheduler_file=None

print(use_complicated_window,unit_window,lognormal,do_blending,do_SSV_sim,use_shot_noise)
print('scheduler: ',Scheduler_file)
# print(args.cw,args.uw,lognormal,do_blending,do_SSV_sim,use_shot_noise)

do_pseudo_cl=False if not args.pseudo_cl else np.bool(args.pseudo_cl)
do_xi=True if args.xi is None else np.bool(args.xi)

# if args.noise is None: #because 0 and None both result in same bool
#     use_shot_noise=True


njk=0 if args.njk is None else args.njk

njk1=njk
njk2=njk1
njk=njk1*njk2

if njk>0:
    nsim=10 #time / memory 
else:
    nsim=1000

subsample=False
do_cov_jk=False #compute covariance coupling matrices


lognormal_scale=2

nside=1024
lmax_cl=1000#
window_lmax=2000 #0

if not do_pseudo_cl:
    lmax_cl=np.int(3*nside-1)
    window_lmax=np.int(lmax_cl*1.)

Nl_bins=37 #40

use_cosmo_power=True
use_window=True
f_sky=0.3

n_source_bins=1
sigma_gamma=0.3944/np.sqrt(2.)  #*2**0.25

store_win=True
smooth_window=False

if test_run:
    nside=128
    lmax_cl=int(nside*2.9)
    window_lmax=50
#    window_lmax=nside*3-1
    Nl_bins=7 #40
    njk=4 #4
    nsim=10
    print('this will be test run')

    
wigner_files={}
# wig_home='/global/cscratch1/sd/sukhdeep/dask_temp/'
#wig_home='/Users/Deep/dask_temp/'
#wig_home='/home/deep/data/repos/SkyLens/temp/'
#home='/physics2/sukhdees/skylens/'
home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/'
wig_home=home+'temp/'
wigner_files[0]= wig_home+'/dask_wig3j_l3500_w2100_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l3500_w2100_2_reorder.zarr'

l0w=np.arange(3*nside-1)

memory='240gb'#'120gb'
import multiprocessing

ncpu=multiprocessing.cpu_count() - 1
# ncpu=20 #4
if test_run:
    memory='50gb'
    ncpu=8
worker_kwargs={'memory_spill_fraction':.75,'memory_target_fraction':.99,'memory_pause_fraction':1}
print('initializing dask')
if Scheduler_file is None:
#     worker_kwargs={'memory_spill_fraction':.75,'memory_target_fraction':.99,'memory_pause_fraction':1}
    LC=LocalCluster(n_workers=1,processes=False,memory_limit=memory,threads_per_worker=ncpu,
                local_directory=dask_dir, **worker_kwargs,
                #scheduler_port=12234,
#                 dashboard_address=8801
                diagnostics_port=8801,
#                memory_monitor_interval='2000ms')
               )
    client=Client(LC,)#diagnostics_port=8801,)
    Scheduler_file=client.scheduler_info()['address']
#     dask_initialize(nthreads=27,local_directory=dask_dir)
#     client = Client()
else:
    client=Client(scheduler_file=Scheduler_file,processes=True)
#    client.restart()
scheduler_info=client.scheduler_info()
scheduler_info['file']=Scheduler_file
print('client: ',client,dask_dir,scheduler_info)


#setup parameters
lmin_cl=0
l0=np.arange(lmin_cl,lmax_cl)

lmin_cl_Bins=lmin_cl+10
lmax_cl_Bins=lmax_cl-10
l_bins=np.unique(np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins)))
lb=(l_bins[1:]+l_bins[:-1])*.5

Nl_bins=len(lb)
# if not do_pseudo_cl:
#     Nl_bins=0
#     l_bins=None

l=l0 #np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell

do_cov=True
bin_cl=True

SSV_cov=False
tidal_SSV_cov=False


xi_win_approx=True

    
corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
corrs=[corr_ll,corr_ggl,corr_gg]


th_min=10./60
th_max=120./60
n_th_bins=15
th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min*0.98),np.log10(1),n_th_bins*30)
th2=np.linspace(1,th_max*1.02,n_th_bins*30)
th=np.unique(np.sort(np.append(th,th2)))
thb=np.sqrt(th_bins[1:]*th_bins[:-1])
corr_config = {'min_sep':th_min*60, 'max_sep':th_max*60, 'nbins':n_th_bins, 'sep_units':'arcmin','metric':'Arc','bin_slop':False}#0.01}
bin_xi=True

l0_win=np.arange(lmax_cl)
WT_L_kwargs={'l': l,'theta': th*d2r,'s1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}
WT_L=None
if do_xi:
    WT_L=wigner_transform(**WT_L_kwargs)
    
mean=150
sigma=50
ww=1000*np.exp(-(l0w-mean)**2/sigma**2)

print('getting win')
z0=0.5
zl_bin1=lsst_source_tomo_bins(zp=np.array([z0]),ns0=10,use_window=use_window,nbins=1,
                            window_cl_fact=(1+ww*use_complicated_window),scheduler_info=scheduler_info,
                            f_sky=f_sky,nside=nside,unit_win=unit_window,use_shot_noise=True)

z0=1 #1087
zs_bin1=lsst_source_tomo_bins(zp=np.array([z0]),ns0=30,use_window=use_window,
                              scheduler_info=scheduler_info,
                                    window_cl_fact=(1+ww*use_complicated_window),
                                    f_sky=f_sky,nbins=n_source_bins,nside=nside,
                                    unit_win=unit_window,use_shot_noise=True)

if njk>0:
    f_sky_jk=f_sky*(njk-1.)/njk
    if subsample:
        f_sky_jk=f_sky/njk

mask=zs_bin1[0]['window']>-1000
mask=mask.astype('bool')
jkmap=jk_map(mask=mask,nside=nside,njk1=njk1,njk2=njk2)

print('zbins, jkmap done')#,thread_count())
if not use_shot_noise:
    for t in zs_bin1['SN'].keys():
        zs_bin1['SN'][t]*=0
        zl_bin1['SN'][t]*=0
client.restart()
cl_func_names={corr:'calc_cl2' for corr in corrs}
# cl_func_names=None
kappa_win=Skylens(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1,
            use_window=use_window,store_win=store_win,window_lmax=window_lmax,corrs=corrs,
            SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=f_sky,
                  cl_func_names=cl_func_names,
            WT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi,scheduler_info=scheduler_info,
            wigner_files=wigner_files,do_pseudo_cl=do_pseudo_cl,xi_win_approx=xi_win_approx,
            clean_tracer_window=False,
)

clG_win=kappa_win.cl_tomo(corrs=corrs)
print(clG_win['cl'][corr_gg])
cl0_win=client.compute(clG_win['stack']).result()#.compute()
#client.restart()
if do_xi:
    xiWG_L=kappa_win.xi_tomo()
    xiW_L=client.compute(xiWG_L['stack']).result() #.compute()  #####mem crash

kappa0=Skylens(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1,
            use_window=False,store_win=store_win,corrs=corrs,window_lmax=window_lmax,
               scheduler_info=scheduler_info,
            SSV_cov=True,tidal_SSV_cov=True,f_sky=f_sky,do_pseudo_cl=do_pseudo_cl,xi_win_approx=xi_win_approx,
            WT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi)

clG0=kappa0.cl_tomo(corrs=corrs) 
cl0=client.compute(clG0['stack']).result()#.compute()


if do_xi:
     xiG_L0=kappa0.xi_tomo()
     xi_L0=client.compute(xiG_L0['stack']).result() #.compute()

bi=(0,0)
cl0={'cl_b':{},'cov':{},'cl':{}}
cl0_win={'cl_b':{},'cov':{}}

for corr in corrs:
    cl0['cl'][corr]=clG0['cl'][corr][bi].compute()

    cl0['cl_b'][corr]=clG0['pseudo_cl_b'][corr][bi].compute()
#     cl0['cov'][corr]=clG0['cov'][corr+corr][bi+bi].compute()

    cl0_win['cl_b'][corr]=clG_win['pseudo_cl_b'][corr][bi].compute()
#     cl0_win['cov'][corr]=clG_win['cov'][corr+corr][bi+bi].compute()['final_b']

print('Skylens done, binning coupling matrices')

M_binning_utils={}
Mp_binning_utils={}

M_binning=binning()

if do_pseudo_cl:
    for corr in corrs:
        wt_b=1./cl0['cl_b'][corr]
        wt0=cl0['cl'][corr]
        M_binning_utils[corr]=M_binning.bin_utils(r=kappa0.l,r_bins=kappa0.l_bins,
                                                    r_dim=2,mat_dims=[1,2],wt_b=wt_b,wt0=wt0)
        wt_b=1./clG_win['cl_b'][corr][bi].compute()
        wt0=clG_win['pseudo_cl'][corr][bi].compute()
        Mp_binning_utils[corr]=M_binning.bin_utils(r=kappa0.l,r_bins=kappa0.l_bins,
                                                    r_dim=2,mat_dims=[1,2],wt_b=wt_b,wt0=wt0)
    
    
mask=zs_bin1[0]['window']>-1.e-20

print('binning coupling matrices done')


def get_coupling_matrices(kappa_class=None): 
    coupling_M={}
    coupling_M_N={}
    coupling_M_binned={'iMaster':{}} #{'Master':{},'nMaster':{},'iMaster':{}}    
    coupling_M_inv={}
    coupling_M_binned_inv={'iMaster':{}} #,'nMaster':{},'Master':{}}
    
    corrs=kappa_class.corrs
    l_bins=kappa_class.l_bins
    fsky=kappa_class.f_sky[corr_gg][(0,0)]
    dl=l_bins[1:]-l_bins[:-1]
    l=kappa_class.l
    shear_lcut=l>=2
    nu=(2.*l+1.)*fsky
    for corr in corrs:
        coupling_M[corr]=kappa_class.Win.Win['cl'][corr][(0,0)]['M']
        coupling_M_N[corr]=kappa_class.Win.Win['cl'][corr][(0,0)]['M_noise']
        if corr==corr_ll:
            coupling_M['shear_B']=kappa_class.Win.Win['cl'][corr][(0,0)]['M_B']
            coupling_M_N['shear_B']=kappa_class.Win.Win['cl'][corr][(0,0)]['M_B_noise']
        # coupling_M_binned['Master'][corr]=bin_coupling_M(kappa_class,coupling_M[corr])
        # coupling_M_binned['nMaster'][corr]=kappa_class.binning.bin_2d(cov=coupling_M[corr],bin_utils=kappa_class.cl_bin_utils) 
        # coupling_M_binned['nMaster'][corr]*=dl

        coupling_M_binned['iMaster'][corr]=M_binning.bin_2d_coupling(cov=coupling_M[corr].T,bin_utils=M_binning_utils[corr])
        coupling_M_binned['iMaster'][corr]=coupling_M_binned['iMaster'][corr].T  #to keep the same order in dot product later. Remeber that the coupling matrix is not symmetric.

        if corr==corr_ll:
            coupling_M_binned['iMaster']['shear_B']=M_binning.bin_2d_coupling(cov=coupling_M['shear_B'].T,bin_utils=M_binning_utils[corr])
            coupling_M_binned['iMaster']['shear_B']=coupling_M_binned['iMaster']['shear_B'].T  #to keep the same order in dot product later. Remeber that the coupling matrix is not symmetric.

        cut=l>=0
        if 'shear' in corr:
            cut=shear_lcut 
        coupling_M_inv[corr]=np.zeros_like(coupling_M[corr])
#             coupling_M_inv[corr][:,cut][cut,:]+=np.linalg.inv(coupling_M[corr][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
        MT=np.linalg.inv(coupling_M[corr][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
        coupling_M_inv[corr]=np.pad(MT,((~cut).sum(),0),mode='constant',constant_values=0)

        for k in coupling_M_binned.keys():
            coupling_M_binned_inv[k][corr]=np.linalg.inv(coupling_M_binned[k][corr])
        if corr==corr_ll:
            coupling_M_inv['shear_B']=np.zeros_like(coupling_M['shear_B'])
            coupling_M_inv['shear_B'][:,cut][cut,:]=np.linalg.inv(coupling_M['shear_B'][cut,:][:,cut]) #otherwise we get singular matrix since for shear l<2 is not defined.
            coupling_M_binned_inv['iMaster']['shear_B']=np.linalg.inv(coupling_M_binned['iMaster']['shear_B'])

    outp={}
    outp['coupling_M_N']=coupling_M_N
    outp['coupling_M_binned']=coupling_M_binned
    outp['coupling_M_inv']=coupling_M_inv
    outp['coupling_M_binned_inv']=coupling_M_binned_inv
    gc.collect()
    return outp


def get_treecorr_cat_args(maps,masks=None):
    tree_cat_args={}
    if masks is None:
        masks={}
        for tracer in maps.keys():
            masks[tracer]=maps[tracer]==hp.UNSEEN
    for tracer in maps.keys():
        seen_indices = np.where( ~masks[tracer] )[0]
        theta, phi = hp.pix2ang(nside, seen_indices)
        ra = np.degrees(np.pi*2.-phi)
        dec = -np.degrees(theta-np.pi/2.)
        tree_cat_args[tracer] = {'ra':ra, 'dec':dec, 'ra_units':'degrees', 'dec_units':'degrees'}
    return tree_cat_args


def get_xi_window_norm(window=None):
    window_norm={corr:{} for corr in corrs}
    mask={}
    for tracer in window.keys():
        # window[tracer]=kappa_class.z_bins[tracer][0]['window']
        mask[tracer]=window[tracer]==hp.UNSEEN
        # window[tracer]=window[tracer][~mask[tracer]]
    fsky=mask[tracer].mean()
    cat0={'fullsky':np.ones_like(mask)}
    tree_cat_args0=get_treecorr_cat_args(cat0,masks=None)

    tree_cat0=treecorr.Catalog(**tree_cat_args0['fullsky'])
    tree_corrs0=treecorr.NNCorrelation(**corr_config)
    _=tree_corrs0.process(tree_cat0,tree_cat0)
    npairs0=tree_corrs0.npairs*fsky
    del cat0,tree_cat0,tree_corrs0
    
    tree_cat_args=get_treecorr_cat_args(window,masks=mask)
    tree_cat= {tracer: treecorr.Catalog(w=window[tracer][~mask[tracer]], **tree_cat_args[tracer]) for tracer in window.keys()}
    del mask
    for corr in corrs:
        tree_corrs=treecorr.NNCorrelation(**corr_config)
        _=tree_corrs.process(tree_cat[corr[0]],tree_cat[corr[1]])
        window_norm[corr]['weight']=tree_corrs.weight
        window_norm[corr]['npairs']=tree_corrs.npairs 
        window_norm[corr]['npairs0']=npairs0
    del tree_cat,tree_corrs
    return window_norm

def get_xi_window_norm_jk(kappa_class=None):
    window_norm={}
    window={}
    for tracer in kappa_class.z_bins.keys():
        window[tracer]=kappa_class.tracer_utils.z_win[tracer][0]['window']
    window_norm['full']=get_xi_window_norm(window=window)
    
    for ijk in np.arange(njk):
        window_i={}
        x=jkmap==ijk
        for tracer in window.keys():
            window_i[tracer]=window[tracer]*1.
            window_i[tracer][x]=hp.UNSEEN
        window_norm[ijk]=get_xi_window_norm(window=window_i)
        del window_i,x
        gc.collect()
        print('window norm ',ijk,' done',)#window_norm[ijk][corr_ll].shape,window_norm[ijk].keys())
    return window_norm


def get_xi(map,window_norm,mask=None):
    
    maps={'galaxy':map[0]}
    maps['shear']={0:map[1],1:map[2]}
    if mask is None:
        mask={}
        mask['galaxy']=maps['galaxy']==hp.UNSEEN
        mask['shear']=maps['shear'][0]==hp.UNSEEN
    tree_cat_args=get_treecorr_cat_args(maps,masks=mask)
    tree_cat={}
    tree_cat['galaxy']=treecorr.Catalog(w=maps['galaxy'][~mask['galaxy']], **tree_cat_args['galaxy']) 
    tree_cat['shear']=treecorr.Catalog(g1=maps['shear'][0][~mask['shear']],g2=maps['shear'][1][~mask['shear']], **tree_cat_args['shear'])
    del mask
    ndim=3 #FIXME
    xi=np.zeros(n_th_bins*(ndim+1))
    th_i=0
    tree_corrs={}
    for corr in corrs:#note that in treecorr npairs includes pairs with 0 weights. That affects this calc
        if corr==corr_ggl:
            tree_corrs[corr]=treecorr.NGCorrelation(**corr_config)
            tree_corrs[corr].process(tree_cat['galaxy'],tree_cat['shear'])
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].xi*tree_corrs[corr].weight/window_norm[corr]['weight']*-1 #sign convention 
                #
            th_i+=n_th_bins
        if corr==corr_ll:
            tree_corrs[corr]=treecorr.GGCorrelation(**corr_config)
            tree_corrs[corr].process(tree_cat['shear'])
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].xip*tree_corrs[corr].npairs/window_norm[corr]['weight']
            th_i+=n_th_bins
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].xim*tree_corrs[corr].npairs/window_norm[corr]['weight']
            th_i+=n_th_bins
        if corr==corr_gg:
            tree_corrs[corr]=treecorr.NNCorrelation(**corr_config)
            tree_corrs[corr].process(tree_cat['galaxy'])
            xi[th_i:th_i+n_th_bins]=tree_corrs[corr].weight/tree_corrs[corr].npairs/window_norm[corr]['weight']  #
#             xi[th_i:th_i+n_th_bins]=tree_corrs[corr].weight/window_norm[corr]
            th_i+=n_th_bins
    del tree_cat,tree_corrs
    gc.collect()
    return xi


def get_coupling_matrices_jk(kappa_class=None): 
    coupling_M={}
    coupling_M['full']=get_coupling_matrices(kappa_class=kappa_win)
    for ijk in np.arange(njk):
        zs_binjk=copy.deepcopy(kappa_class.z_bins['shear'])
        zl_binjk=copy.deepcopy(kappa_class.z_bins['galaxy'])

        x=jkmap==ijk
        for i in np.arange(zs_binjk['n_bins']):
            if subsample:
                zs_binjk[i]['window'][~x]=hp.UNSEEN
            else:
                zs_binjk[i]['window'][x]=hp.UNSEEN
            zs_binjk[i]['window_alm']=hp.map2alm(zs_binjk[i]['window'])
            zs_binjk[i]['window_cl']=None
        for i in np.arange(zl_binjk['n_bins']):
            if subsample:
                zl_binjk[i]['window'][~x]=hp.UNSEEN
            else:
                zl_binjk[i]['window'][x]=hp.UNSEEN
            zl_binjk[i]['window_alm']=hp.map2alm(zl_binjk[i]['window'])
            zl_binjk[i]['window_cl']=None

        kappa_win_JK=Skylens(zs_bins=zs_binjk,do_cov=do_cov_jk,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_binjk,
                use_window=use_window,store_win=store_win,window_lmax=window_lmax,corrs=corrs,
                SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=f_sky_jk,
                WT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=False,
                      wigner_files=wigner_files,
    #                  Win=kappa_win.Win.Win
                     )
        coupling_M[ijk]=get_coupling_matrices(kappa_class=kappa_win_JK)
        del kappa_win_JK
        del zs_binjk
        del zl_binjk
        gc.collect()
        print('coupling M jk ',ijk,'done')
    return coupling_M


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
def get_clsim2(cl0,window,mask,kappa_class,coupling_M,coupling_M_inv,ndim,i):
    print(i)
    local_state = np.random.RandomState(seed+i)
    cl_map=hp.synfast(cl0,nside=nside,rng=local_state,new=True,pol=True,verbose=False)

    if ndim>1:
        cl_map[0]*=window['galaxy']
        cl_map[0][mask['galaxy']]=hp.UNSEEN
        cl_map[1]*=window['shear'] #shear_1
        cl_map[2]*=window['shear']#shear_2
        cl_map[1][mask['shear']]=hp.UNSEEN
        cl_map[2][mask['shear']]=hp.UNSEEN
        pcli=hp.anafast(cl_map, lmax=max(l),pol=True) #TT, EE, BB, TE, EB, TB for polarized input map
        pcli=pcli[:,l]
        pcli=pcli[[0,1,3],:]
#             for i in np.arange(6):

    else:
        cl_map*=window
        cl_map[mask]=hp.UNSEEN
        pcli=hp.anafast(cl_map, lmax=max(l),pol=True)[l]
        
    del cl_map

    if ndim>1:
        pcli[0]-=(np.ones_like(pcli[0])*kappa_class.SN[corr_gg][:,0,0])@coupling_M[corr_gg]*use_shot_noise
        pcli[1]-=(np.ones_like(pcli[1])*kappa_class.SN[corr_ll][:,0,0])@coupling_M[corr_ll]*use_shot_noise
        pcli[1]-=(np.ones_like(pcli[1])*kappa_class.SN[corr_ll][:,0,0])@coupling_M['shear_B']*use_shot_noise #remove B-mode leakage

        cli=[pcli[0]@coupling_M_inv[corr_gg],
              pcli[1]@coupling_M_inv[corr_ll],
              pcli[2]@coupling_M_inv[corr_ggl]]
    else:
        pcli-=(np.ones_like(pcli)*shot_noise)@coupling_M
        cli=pcli@coupling_M_inv
    cli=np.array(cli)
    return [pcli.T,cli.T]


def get_xi_cljk(cl_map,lmax=np.int(l0.max()),pol=True,coupling_M={},xi_window_norm={}):
    pcl_jk={}
    xi_jk={}
    if do_pseudo_cl:
        pcl_jk['full']=hp.anafast(cl_map, lmax=max(l),pol=True) #TT, EE, BB, TE, EB, TB for polarized input map
    if do_xi:
        xi_jk['full']=get_xi(cl_map, window_norm=xi_window_norm['full'])
    for ijk in np.arange(njk):
        x=jkmap==ijk
        cl_map_i=copy.deepcopy(cl_map)
        if subsample:
            cl_map_i[:,~x]=hp.UNSEEN
        else:
            cl_map_i[:,x]=hp.UNSEEN

        if do_pseudo_cl:
            pcl_jk[ijk]=hp.anafast(cl_map_i,lmax=lmax,pol=pol)
        if do_xi:
            xi_jk[ijk]=get_xi(cl_map_i, window_norm=xi_window_norm[ijk])
        del cl_map_i
        gc.collect()
    return pcl_jk,xi_jk

def jk_mean(p={},njk=njk):
    if njk==0:
        return p
    p2={}
    nn=np.arange(njk)
    for i in nn: #p.keys():
        #if i in nn:
        p2[i]=p[i]
    jk_vals=np.array(list(p2.values()))
    mean=np.mean(jk_vals,axis=0)
    var=np.var(jk_vals,axis=0,ddof=0)
    try:
        cov=np.cov(jk_vals,rowvar=0)
    except Exception as err:
#         print(jk_vals.shape, err)
        cov=np.cov(jk_vals.reshape(njk, jk_vals.shape[2]*jk_vals.shape[1]),rowvar=0)
    if subsample:
        var/=njk
        cov/=njk
    else:
        cov*=(njk-1)
    p['jk_mean']=mean
    p['jk_err']=np.sqrt(var) 
    p['jk_cov']=cov
    p['jk_corr']=corr_matrix(cov_mat=cov)
    return p

def sample_mean(p={},nsamp=100):
#     if check_empty(p):
#         print ('sample-mean: got empty dict')
#         return p
    p2={}
    nn=np.arange(nsamp)
    for i in nn: #p.keys():
        #if i in nn:
        p2[i]=p[i]
    jk_vals=np.array(list(p2.values()))
    mean=np.mean(jk_vals,axis=0)
    #print mean
    var=np.var(jk_vals,axis=0,ddof=0)
    p['mean']=mean
    p['err']=np.sqrt(var)
    try:
        cov=np.cov(jk_vals,rowvar=0)
        p['cov']=cov
        p['corr']=corr_matrix(cov_mat=cov)
    except Exception as err:
        p['cov']=err
        p['corr']=err
    return p



def calc_sim_stats(sim=[],sim_truth=[],PC=False):
    sim_stats={}
    sim_stats['std']=np.std(sim,axis=0)    
    sim_stats['mean']=np.mean(sim,axis=0)
    sim_stats['median']=np.median(sim,axis=0)
    sim_stats['percentile']=np.percentile(sim,[16,84],axis=0)
    sim_stats['cov']=np.cov(sim,rowvar=0)
    
    sim_stats['percentile_score']=np.zeros_like(sim_stats['std'])
    if len(sim_stats['std'].shape)==1:
        for i in np.arange(len(sim_stats['std'])):
            sim_stats['percentile_score'][i]=percentileofscore(sim[:,i],sim_truth[i])
    elif len(sim_stats['std'].shape)==2:
        for i in np.arange(len(sim_stats['std'])):
            for i_dim in np.arange(2):
                for j_dim in np.arange(2):
                    sim_stats['percentile_score'][i][i_dim,j_dim]=percentileofscore(sim[:,i,i_dim,j_dim],
                                                                                   sim_truth[i,i_dim,j_dim])
    else:
        sim_stats['percentile_score']='not implemented for ndim>2'
    return sim_stats
    

def sim_cl_xi(nsim=150,do_norm=False,cl0=None,kappa_class=None,fsky=f_sky,zbins=None,use_shot_noise=True,
             convolve_win=False,nside=nside,use_cosmo_power=True,lognormal=False,lognormal_scale=1,
             coupling_M=None,add_SSV=True,add_tidal_SSV=True,
              add_blending=False,blending_coeff=-2,fiber_coll_coeff=-1):
    ndim=len(kappa_class.corrs)
    xi_window_norm={}
    coupling_M={}
    sim_clb_shape=None
    sim_xib_shape=None
    if do_pseudo_cl:
        coupling_M=get_coupling_matrices_jk(kappa_class=kappa_class)
        sim_clb_shape=(nsim,Nl_bins*(ndim+1)) #shear-shear gives 2 corrs, EE and BB..    
    if do_xi:
        xi_window_norm=get_xi_window_norm_jk(kappa_class=kappa_class)
        sim_xib_shape=(nsim,n_th_bins*(ndim+1)) #shear-shear gives 2 corrs, xi+ and xi-

    l=kappa_class.l
    shear_lcut=l>=2
    
    l_bins=kappa_class.l_bins
    dl=l_bins[1:]-l_bins[:-1]
    nu=(2.*l+1.)*fsky
    
    mask={}
    window={}
    if convolve_win:
        nu=2.*l+1.
        for tracer in kappa_class.z_bins.keys():
            window[tracer]=kappa_class.tracer_utils.z_win[tracer][0]['window']
            mask[tracer]=window[tracer]==hp.UNSEEN
#             print(coupling_M_inv.keys())
    outp={}
    win=0
    pcl_shear_B=None
    if cl0 is None:
        cl0={}
        pcl0={}
        clG0=kappa_class.cl_tomo() 
        for corr in kappa_class.corrs:
            cl0[corr]=clG0['cl'][corr][(0,0)].compute()
            pcl0[corr]=clG0['cl_b'][corr][(0,0)].compute()
    clN0={}
    for corr in kappa_class.corrs: #ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
        shot_noise=kappa_class.SN[corr_gg][:,0,0]*0
        if corr[0]==corr[1]:
            shot_noise=kappa_class.SN[corr][:,0,0]
        shot_noise=shot_noise*use_shot_noise
        clN0[corr]=shot_noise
        cl0[corr]=cl0[corr]*use_cosmo_power#+shot_noise
        if corr==corr_ll:
            cl0['shear_B']=cl0[corr]*0
            clN0['shear_B']=shot_noise
            #pcl_shear_B=shot_noise@coupling_M[corr_ll]+(cl0[corr_ll]+shot_noise)@coupling_M['shear_B']
#             pcl_shear_B=cl0[corr_ll]@coupling_M['full']['coupling_M']['shear_B']
    
    print('ndim:',ndim)
    outp['cl0_0']=cl0.copy()
    outp['clN0_0']=clN0.copy()
    outp['ndim']=ndim
    
    SN=kappa_class.SN
    # sim_cl_shape=(nsim,len(kappa_class.l),ndim)
    
    jk_stat_keys=['jk_mean','jk_err','jk_cov']
    if njk==0:
        jk_stat_keys=[]
#     pcl={'full':np.zeros(sim_cl_shape,dtype='float32')}
#     pcl.update({jks:{} for jks in jk_stat_keys})
#     cl={'full':np.zeros(sim_cl_shape,dtype='float32')}
#     cl.update({jks:{} for jks in jk_stat_keys})
#     pclB={'full':np.zeros(sim_cl_shape,dtype='float32')}
#     pclB.update({jks:{} for jks in jk_stat_keys})
#     clB={'full':np.zeros(sim_cl_shape,dtype='float32')}
#     clB.update({jks:{} for jks in jk_stat_keys})
    cl_maps={}
    lmax=max(l)
    lmin=min(l)

    SSV_sigma={}
    SSV_cov={}
    SSV_kernel={}
    SSV_response={}
    
    if add_SSV:
        SSV_response0=kappa0.Ang_PS.clz['clsR']
        if add_tidal_SSV:
            SSV_response0=kappa0.Ang_PS.clz['clsR']+kappa0.Ang_PS.clz['clsRK']/6.
            
        for corr in kappa_class.corrs:
            zs1_indx=0
            zs2_indx=0
            SSV_kernel[corr]=kappa_class.z_bins[corr[0]][zs1_indx]['kernel_int']
            SSV_kernel[corr]=SSV_kernel[corr]*kappa_class.z_bins[corr[1]][zs2_indx]['kernel_int']
            SSV_response[corr]=SSV_kernel[corr]*SSV_response0.T
        SSV_sigma=kappa_class.cov_utils.sigma_win_calc(clz=kappa0.Ang_PS.clz,Win=kappa_class.Win.Win,tracers=corr_ll+corr_ll,zs_indx=(0,0,0,0)) #use of kappa0 is little dangerous
        SSV_cov=np.diag(SSV_sigma**2)
    
    cl0_b={corr: kappa_class.binning.bin_1d(xi=cl0[corr],bin_utils=kappa_class.cl_bin_utils) for corr in kappa_class.corrs} 
        
    Master_algs=['iMaster'] #['unbinned','Master','nMaster','iMaster']

    cl_b=None;pcl_b=None;xi_b=None
    if do_pseudo_cl:    
        cl_b={'full':np.zeros(sim_clb_shape,dtype='float32')}   #  {im:np.zeros(sim_clb_shape,dtype='float32') for im in Master_algs}}
        cl_b.update({jks:{} for jks in jk_stat_keys})  #{im:{} for im in Master_algs} for jks in jk_stat_keys})
        
        pcl_b={'full':np.zeros(sim_clb_shape,dtype='float32')}
        pcl_b.update({jks:{} for jks in jk_stat_keys})
    if do_xi:    
        xi_b={'full':np.zeros(sim_xib_shape,dtype='float32')}   #  {im:np.zeros(sim_clb_shape,dtype='float32') for im in Master_algs}}
        xi_b.update({jks:{} for jks in jk_stat_keys})  #{im:{} for im in Master_algs} for jks in jk_stat_keys})
    
    binning_func=kappa_class.binning.bin_1d
    binning_utils=kappa_class.cl_bin_utils
    
#         pcl_shear_B_b=binning_func(xi=pcl_shear_B,bin_utils=binning_utils)
    
    if ndim>1:
        cl0=(cl0[corr_gg],cl0[corr_ll],cl0['shear_B'],cl0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
        clN0=(clN0[corr_gg],clN0[corr_ll],clN0['shear_B'],clN0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
    else:
        cl0=cl0[corr_gg]
        clN0=clN0[corr_gg]
    
    gamma_trans_factor=0
    if lognormal:
        l_t=np.arange(nside*3-1+1)
        gamma_trans_factor = np.array([np.sqrt( (li+2)*(li-1)/(li*(li+1))) for li in l_t   ] )
        gamma_trans_factor[0] = 0.
        gamma_trans_factor[1] = 0.
        l_alm,m_alm=hp.sphtfunc.Alm.getlm(l_t.max())
        l_alm=np.int32(l_alm)
        m_alm=0
        gamma_trans_factor=gamma_trans_factor[l_alm]
        l_alm=0
        
    def kappa_to_shear_map(kappa_map=[]):
        kappa_alm = hp.map2alm(kappa_map,pol=False)        
        gamma_alm = []
        gamma_alm=kappa_alm*gamma_trans_factor#[l_alm]
        k_map, g1_map, g2_map = hp.sphtfunc.alm2map( [kappa_alm,gamma_alm,kappa_alm*0 ], nside=nside,pol=True  )
        return g1_map,g2_map
    
    
    def process_pcli(pcli,coupling_M={}):
        if ndim>1:
            # pcli_B=pcli[[2,4,5],:]
            pcli=pcli[[0,1,3,2],:] #2==BB
            corr_ti=[corr_gg,corr_ll,corr_ggl,'shear_B']
            if use_shot_noise:
                pcli[0]-=(np.ones_like(pcli[0])*SN[corr_gg][:,0,0])@coupling_M['coupling_M_N'][corr_gg]*use_shot_noise
                pcli[1]-=(np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N'][corr_ll]*use_shot_noise
                pcli[1]-= (np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N']['shear_B']*use_shot_noise #remove B-mode 
                pcli[3]-=(np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N']['shear_B']*use_shot_noise #remove B-mode 
                pcli[3]-=(np.ones_like(pcli[1])*SN[corr_ll][:,0,0])@coupling_M['coupling_M_N'][corr_ll]*use_shot_noise
            
        else:
            pcli-=(np.ones_like(pcli)*shot_noise)@coupling_M['coupling_M']
            cli=pcli@coupling_M['coupling_M_inv']
        
        pcli=pcli[[3,1,2,0],:]#corr_t=['shear_B',corr_ll,corr_ggl,corr_gg] #FIXME: this should be dynamic, based on corr_t and corr_ti
        
        sim_clb_shape=(Nl_bins*(ndim+1))
        pcl_b=np.zeros(sim_clb_shape,dtype='float32')
        
        cl_b=np.zeros(sim_clb_shape,dtype='float32')#{'unbinned':np.zeros(sim_clb_shape,dtype='float32'),
                # 'iMaster':np.zeros(sim_clb_shape,dtype='float32')}                    
        
        pcl_b=np.zeros(sim_clb_shape,dtype='float32')

        li=0

        for ii in np.arange(ndim)+1:
            #cl_b['unbinned'][:,ii]=binning_func(xi=cli[ii],bin_utils=binning_utils)
            pcl_b[li:li+Nl_bins]=binning_func(xi=pcli[ii],bin_utils=binning_utils)
            # pclB_b[:,ii]=binning_func(xi=pcli_B[ii],bin_utils=binning_utils)
            # for k in coupling_M['coupling_M_binned_inv'].keys():
            k='iMaster'
            cl_b[li:li+Nl_bins]=pcl_b[li:li+Nl_bins]@coupling_M['coupling_M_binned_inv'][k][corr_t[ii]] #be careful with ordering as coupling matrix is not symmetric
            # if corr_t[ii]==corr_ll:
            #     clB_b['iMaster'][:,ii]=pclB_b[:,ii]@coupling_M['coupling_M_binned_inv']['iMaster']['shear_B']
#             return pcli.T,cli.T,pcl_b,pcli_B.T,cl_b,pclB_b,clB_b
        return pcl_b,cl_b#,pclB_b,clB_b
    
    # corr_t=[corr_gg,corr_ll,corr_ggl,'shear_B'] #order in which sim corrs are output.
    corr_t=['shear_B',corr_ll,corr_ggl,corr_gg] #order in which sim corrs are output... some shuffling done in process_pcli
    seed=12334
    def get_clsim(i):
        mapi=i*1.
        print('doing map: ',i)
        local_state = np.random.RandomState(seed+i)
        cl0i=copy.deepcopy(cl0)
#         cl_map=hp.synfast(cl0,nside=nside,RNG=local_state,new=True,pol=True)
        if add_SSV:
            SSV_delta=np.random.multivariate_normal(mean=SSV_sigma*0,cov=SSV_cov,size=1)[0]
            # print('adding SSV')
            SSV_delta2=SSV_delta*kappa_class.Ang_PS.clz['dchi']
#                 print('SSV delta shape',SSV_delta2.shape,SSV_response[corr_gg].shape)
            tt=SSV_response[corr_gg]@SSV_delta2
#                 print('SSV delta shape',SSV_delta2.shape,tt.shape)
            cl0i[0]+=(SSV_response[corr_gg]@SSV_delta2)@coupling_M[corr_gg]
            cl0i[1]+=(SSV_response[corr_ll]@SSV_delta2)@coupling_M[corr_ll]
            cl0i[2]+=(SSV_response[corr_ggl]@SSV_delta2)@coupling_M[corr_ggl]
        if lognormal:
            cl_map=hp.synfast(cl0i,nside=nside,rng=local_state,new=True,pol=False,verbose=False)
            cl_map_min=np.absolute(cl_map.min(axis=1))
            lmin_match=10
            lmax_match=100
            scale_f=lognormal_scale
            v0=np.std(cl_map.T/cl_map_min*scale_f,axis=0)
            cl_map=np.exp(cl_map.T/cl_map_min*scale_f)*np.exp(-0.5*v0**2)-1 #https://arxiv.org/pdf/1306.4157.pdf
            cl_map*=cl_map_min/scale_f
            cl_map=cl_map.T
            cl_map[1,:],cl_map[2,:]=kappa_to_shear_map(kappa_map=cl_map[1])#,nside=nside)

        else:
            cl_map=hp.synfast(cl0i,nside=nside,rng=local_state,new=True,pol=True,verbose=False)
    
        N_map=0
        if use_shot_noise:
            N_map=hp.synfast(clN0,nside=nside,rng=local_state,new=True,pol=True,verbose=False)
        
        tracers=['galaxy','shear','shear']
        if ndim>1:
            for i in np.arange(3):
                tracer=tracers[i]
                if lognormal:
                    cl_map[i]-=cl_map[i].mean()

                window2=window[tracer]
                
                if add_blending:
                    if tracer=='shear':
                        window2=window[tracer]+cl_map[i]*blending_coeff
                        
                    if tracer=='galaxy':
                        window2=window[tracer]+cl_map[i]*fiber_coll_coeff
                        window2+=cl_map[1]*blending_coeff
                        
                    window2[window2<0]=0
                    window2/=window2[~mask[tracer]].mean()
                    window2[mask[tracer]]=hp.UNSEEN
                    # print('adding blending ',tracer,blending_coeff,fiber_coll_coeff)
                    # print( 'window2',window2.max(),window2.min(),window2[mask[tracer]].min(),
                    #         window2[mask[tracer]],window[tracer])
                # print(np.all(window2==window[tracer]))   
                cl_map[i]*=window2

                if use_shot_noise:
                    N_map[i]*=np.sqrt(window[tracer])
                    cl_map[i]+=N_map[i]
                    N_map[i][mask[tracer]]=hp.UNSEEN
                cl_map[i][mask[tracer]]=hp.UNSEEN
            del N_map    
            pcli_jk,xi_jk=get_xi_cljk(cl_map,lmax=max(l),pol=True,xi_window_norm=xi_window_norm)         
            del cl_map
        else:
            cl_map*=window
            cl_map[mask]=hp.UNSEEN
            pcli_jk=get_cljk(cl_map,lmax=max(l),pol=True)
            pcli_jk['full']=hp.anafast(cl_map, lmax=max(l),pol=True)[l]        

        pcl_b_jk={};cl_b_jk={}
        if do_pseudo_cl:
            
            # cl_b_jk={'iMaster':{}} #{'unbinned':{},'Master':{},'nMaster':{},'iMaster':{}}    
                          
            for ijk in pcli_jk.keys():
                pcl_b_jk[ijk],cl_b_jk[ijk]=process_pcli(pcli_jk[ijk],coupling_M=coupling_M[ijk])
                # for k in cl_b_jk_i.keys():
                #     cl_b_jk[k][ijk]=cl_b_jk_i[k]
                # clB_b_jk['iMaster'][ijk]=clB_b_jk_i['iMaster']
            # pcli_jk=jk_mean(pcli_jk,njk=njk)
            # cli_jk=jk_mean(cli_jk,njk=njk)
            # pcli_B_jk=jk_mean(pcli_B_jk,njk=njk)
            # if do_pseudo_cl:
            pcl_b_jk=jk_mean(pcl_b_jk,njk=njk)
            cl_b_jk=jk_mean(cl_b_jk,njk=njk)
        if do_xi:
            xi_jk=jk_mean(xi_jk,njk=njk)
            # for k in cl_b_jk.keys():
            #     cl_b_jk[k]=jk_mean(cl_b_jk[k],njk=njk)
#             return pcli_jk,cli_jk,pcl_b_jk,pcli_B_jk,cl_b_jk,pclB_b_jk,clB_b_jk
#         gc.collect()
        return pcl_b_jk,cl_b_jk,xi_jk
        # else:
        #     for ijk in pcl_jk.keys():
        #         pcli_jk[ijk],cli_jk[ijk]=process_pcli(pcli_jk[ijk],coupling_M=coupling_M[ijk])
        #     return pcli_jk,cli_jk
    
    def comb_maps(futures):
        for i in np.arange(nsim):
            x=futures[i]#.compute()
            pcl[i,:,:]+=x[0]
            cl[i,:,:]+=x[1]
        return pcl,cl 
    
    print('generating maps')
    gc.disable()
    if convolve_win:
        i=0
        j=0
        step= 1 # min(nsim,len(client.scheduler_info()['workers']))
        while j<nsim:
            futures={}
            for ii in np.arange(step):
                futures[ii]=delayed(get_clsim)(i+ii)  
            futures=client.compute(futures).result()
            for ii in np.arange(step):
                tt=futures[ii]
                if do_pseudo_cl:
                    pcl_b[i]=tt[0]
                    cl_b[i]=tt[1]
                if do_xi:
                    xi_b[i]=tt[2]
                        
                i+=1
            proc = psutil.Process()
            print('done map ',i, thread_count(),'mem, peak mem: ',format_bytes(proc.memory_info().rss),
                 int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.)
                 )
            del futures
            gc.collect()
            print('done map ',i, thread_count(),'mem, peak mem: ',format_bytes(proc.memory_info().rss),
                 int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.)
                 )
            # client.restart() #this can sometimes fail... useful for clearing memory on cluster.
            j+=step
    print('done generating maps')
    gc.enable()
    def get_full_samp(cljk={}):
        k=None
        try:
            k=cljk['full'].keys()
        except:
            pass
        for i in np.arange(nsim):            
            if k is None:
                cljk['full'][i,:]=cljk[i]['full']
                for jks in jk_stat_keys:
                    cljk[jks][i]=cljk[i][jks]
            else:
                for ki in k:
                    cljk['full'][ki][i,:]=cljk[i][ki]['full']
                    for jks in jk_stat_keys:
                        cljk[jks][ki][i]=cljk[i][ki][jks]
        if k is None:
            for jks in jk_stat_keys:
                cljk[jks]=sample_mean(cljk[jks],nsim)
        else:
            for ki in k:
                for jks in jk_stat_keys:
                    cljk[jks][ki]=sample_mean(cljk[jks][ki],nsim)
        return cljk
    
    if do_pseudo_cl:
        cl_b=get_full_samp(cl_b)
        pcl_b=get_full_samp(pcl_b)

        cl0_b['shear_B']=cl0_b[corr_ll]*0
        cl0_b=np.array([cl0_b[corr] for corr in corr_t]).flatten()
        outp['cl_b_stats']=client.compute(delayed(calc_sim_stats)(sim=cl_b['full'],sim_truth=cl0_b))
        outp['pcl_b_stats']=client.compute(delayed(calc_sim_stats)(sim=pcl_b['full'],sim_truth=pcl_b['full'].mean(axis=0)))
    
        outp['cl_b_stats']=outp['cl_b_stats'].result()
        outp['pcl_b_stats']=outp['pcl_b_stats'].result()
        
        outp['cl_b']=cl_b
        outp['pcl_b']=pcl_b
    if do_xi:
        xi_b=get_full_samp(xi_b)
        outp['xi_b_stats']=client.compute(delayed(calc_sim_stats)(sim=xi_b['full'],sim_truth=xi_b['full'].mean(axis=0)))
        outp['xi_b_stats']=outp['xi_b_stats'].result()
        outp['xi0']=xi_L0
        outp['xiW0']=xiW_L
        outp['xi_b']=xi_b
        outp['xi_window_norm']=xi_window_norm

    outp['cl0']=cl0
    outp['clN0']=clN0
    outp['cl0']=cl0
    outp['pcl0']=pcl0

    outp['size']=nsim
    outp['fsky']=fsky
    outp['l_bins']=l_bins
    
    outp['coupling_M']=coupling_M
    outp['use_shot_noise']=use_shot_noise
    
    return outp


test_home=home+'/tests/'
#test_home='/home/deep/data/repos/SkyLens/tests/'

f0='/sims'
if do_pseudo_cl:
    f0+='_cl'
if do_xi:
    f0+='_xi'
if njk>0:
    f0+='_jk'
fname=test_home+f0+'_newN'+str(nsim)+'_ns'+str(nside)+'_lmax'+str(lmax_cl)+'_wlmax'+str(window_lmax)+'_fsky'+str(f_sky)
if lognormal:
    fname+='_lognormal'+str(lognormal_scale)
if not use_shot_noise:
    fname+='_noSN'
if do_SSV_sim:
    fname+='_SSV'
if do_blending:
    fname+='_blending'

if unit_window:
    fname+='_unit_window'
if use_complicated_window:
    fname+='_cWin'
if smooth_window:
    fname+='_smooth_window'
fname+='.pkl'

print(fname)
# dask.config.set(scheduler='single-threaded')
#client.restart()
#client.close()
cl_sim_W=sim_cl_xi(nsim=nsim,do_norm=False,#cl0=clG0['cl'][corrs[0]][(0,0)].compute(),
          kappa_class=kappa_win,fsky=f_sky,use_shot_noise=use_shot_noise,use_cosmo_power=use_cosmo_power,
             convolve_win=True,nside=nside,
             lognormal=lognormal,lognormal_scale=lognormal_scale,
             add_SSV=do_SSV_sim,add_tidal_SSV=do_SSV_sim,add_blending=do_blending)

outp={}
outp['simW']=cl_sim_W
outp['zs_bin1']=zs_bin1
outp['zl_bin1']=zl_bin1
outp['cl0']=cl0
outp['cl0_win']=cl0_win

with open(fname,'wb') as f:
    pickle.dump(outp,f)
written=True

print(fname)
print('all done')
# client.shutdown()
# try:
#     if Scheduler_file is None:
#         LC.close()
# except Exception as err:
#     print('LC close error:', err)

gc.collect()
sys.exit(0)