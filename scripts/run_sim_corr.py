import sys, os, gc, threading, subprocess
sys.path.insert(0,'../skylens/')
from thread_count import *
os.environ['OMP_NUM_THREADS'] = '20'
# import libpython
pid=os.getpid()
print('pid: ',pid, sys.version)

#thread_count()
# sys,settrace
import pickle
import treecorr
from skylens import *
from survey_utils import *
from scipy.stats import norm,mode,skew,kurtosis,percentileofscore

import sys
import tracemalloc

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

gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
args = parser.parse_args()

use_complicated_window=False if not args.cw else np.bool(args.cw)
unit_window=False if not args.uw else np.bool(args.cw)
lognormal=False if not args.lognormal else np.bool(args.lognormal)

do_blending=False if not args.blending else np.bool(args.blending)
do_SSV_sim=False if not args.ssv else np.bool(args.ssv)
use_shot_noise=False if not args.noise else np.bool(args.noise)
print(use_complicated_window,unit_window,lognormal,do_blending,do_SSV_sim,use_shot_noise)

nsim=5

lognormal_scale=2

nside=1024
lmax_cl=3*nside-1#
window_lmax=3*nside-1 #0
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
    lmax_cl=int(nside*3-1)
    window_lmax=lmax_cl
#    window_lmax=nside*3-1
    Nl_bins=7 #40
    nsim=100
    print('this will be test run')

    
wigner_files={}
# wig_home='/global/cscratch1/sd/sukhdeep/dask_temp/'
wig_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/'
# wig_home='/media/deep/repos/Skylens/temp/'
wigner_files[0]= wig_home+'/dask_wig3j_l6500_w2100_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l3500_w2100_2_reorder.zarr'

l0w=np.arange(3*nside-1)

memory='55gb'#'120gb'
import multiprocessing

ncpu=multiprocessing.cpu_count() - 2
#ncpu=28 #4
if test_run:
    memory='20gb'
    ncpu=4
worker_kwargs={'memory_spill_fraction':.75,'memory_target_fraction':.99,'memory_pause_fraction':1}
LC=LocalCluster(n_workers=1,processes=False,memory_limit=memory,threads_per_worker=ncpu,
                local_dir=wig_home+'/NGL-worker/', **worker_kwargs,
                #scheduler_port=12234,
#                 dashboard_address=8801
                diagnostics_port=8801,
#                memory_monitor_interval='2000ms')
               )
client=Client(LC,)#diagnostics_port=8801,)
print(client)

#setup parameters
lmin_cl=0
l0=np.arange(lmin_cl,lmax_cl)
# l0=np.unique(np.int32(np.linspace(lmin_cl,lmax_cl,800)))

lmin_cl_Bins=lmin_cl+10
lmax_cl_Bins=lmax_cl-10
l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
lb=(l_bins[1:]+l_bins[:-1])*.5
# l_bins=None

l=l0 #np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell
window_l=l #[l<2100] #FIXME .. issues with wig_3j
do_cov=True
bin_cl=False

SSV_cov=False
tidal_SSV_cov=False

do_xi=True
do_pseudo_cl=False
xi_win_approx=True


w_smooth_lmax=1.e7 #some large number
if smooth_window:
    w_smooth_lmax=1000
window_cl_fact=np.cos(np.pi/2*(l0w/w_smooth_lmax)**10)
x=window_cl_fact<0
x+=l0w>w_smooth_lmax
window_cl_fact[x]=0
if unit_window:
    window_cl_fact=0
    
corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
corrs=[corr_ll,corr_ggl,corr_gg]


th_min=10./60
th_max=600./60
n_th_bins=20
th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
th=np.logspace(np.log10(th_min*0.98),np.log10(1),n_th_bins*30)
th2=np.linspace(1,th_max*1.02,n_th_bins*30)
th=np.unique(np.sort(np.append(th,th2)))
thb=np.sqrt(th_bins[1:]*th_bins[:-1])

bin_xi=True

l0_win=l0 #np.arange(lmax_cl)
WT_L_kwargs={'l': l0_win,'theta': th*d2r,'s1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}
WT_L=None
if do_xi:
    WT_L=wigner_transform(**WT_L_kwargs)
    
mean=150
sigma=50
ww=1000*np.exp(-(l0w-mean)**2/sigma**2)

print('getting win')
z0=0.5
zl_bin1=lsst_source_tomo_bins(zp=np.array([z0]),ns0=10,use_window=use_window,nbins=1,
                            window_cl_fact=window_cl_fact*(1+ww*use_complicated_window),
                            f_sky=f_sky,nside=nside,unit_win=unit_window,use_shot_noise=True)

z0=1 #1087
zs_bin1=lsst_source_tomo_bins(zp=np.array([z0]),ns0=30,use_window=use_window,
                                window_cl_fact=window_cl_fact*(1+ww*use_complicated_window),
                                f_sky=f_sky,nbins=n_source_bins,nside=nside,
                                unit_win=unit_window,use_shot_noise=True)

print('zbins done')#,thread_count())
if not use_shot_noise:
    for t in zs_bin1['SN'].keys():
        zs_bin1['SN'][t]*=0
        zl_bin1['SN'][t]*=0

kappa_win=Skylens(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1,
            use_window=use_window,store_win=store_win,window_lmax=window_lmax,corrs=corrs,window_l=window_l,
            SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=f_sky,
            WT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi,
            wigner_files=wigner_files,do_pseudo_cl=do_pseudo_cl,xi_win_approx=xi_win_approx
)

clG_win=kappa_win.cl_tomo(corrs=corrs)
# cl0_win=clG_win['stack'].compute()

if do_xi:
    xiWG_L=kappa_win.xi_tomo()
    xiW_L=xiWG_L['stack'].compute()

l=kappa_win.window_l
Om_W=np.pi*4*f_sky
theta_win=np.sqrt(Om_W/np.pi)
l_th=l*theta_win
Win0=2*jn(1,l_th)/l_th
Win0=np.nan_to_num(Win0)

kappa0=Skylens(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zl_bin1,
            use_window=False,store_win=store_win,corrs=corrs,window_lmax=window_lmax,
            SSV_cov=True,tidal_SSV_cov=True,f_sky=f_sky,do_pseudo_cl=do_pseudo_cl,
            WT=WT_L,bin_xi=bin_xi,theta_bins=th_bins,do_xi=do_xi)

clG0=kappa0.cl_tomo(corrs=corrs) 
# cl0=clG0['stack'].compute()


if do_xi:
     xiG_L0=kappa0.xi_tomo()
     xi_L0=xiG_L0['stack'].compute()

bi=(0,0)
cl0={'cl_b':{},'cov':{},'cl':{}}
cl0_win={'cl_b':{},'cov':{}}

for corr in corrs:
    cl0['cl'][corr]=clG0['cl'][corr][bi].compute()

#     cl0['cl_b'][corr]=clG0['pseudo_cl_b'][corr][bi].compute()
#     cl0['cov'][corr]=clG0['cov'][corr+corr][bi+bi].compute()

#     cl0_win['cl_b'][corr]=clG_win['pseudo_cl_b'][corr][bi].compute()
#     cl0_win['cov'][corr]=clG_win['cov'][corr+corr][bi+bi].compute()['final_b']

print('kappa done, binning coupling matrices')
from binning import *
seed=12334
def get_clsim2(clg0,window,mask,kappa_class,coupling_M,coupling_M_inv,ndim,i):
    print(i)
    local_state = np.random.RandomState(seed+i)
    cl_map=hp.synfast(clg0,nside=nside,rng=local_state,new=True,pol=True,verbose=False)

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
#     sim_stats['skew']=skew(sim,axis=0)
#     sim_stats['kurt']=kurtosis(sim,axis=0)
    sim_stats['cov']=np.cov(sim,rowvar=0)
    
#     if not PC:
#         try:
#             sim_stats['cov_ev'],sim_stats['cov_evec']=np.linalg.eig(sim_stats['cov'])
#             sim_stats['PC']={}
#             sim_stats['PC']['data']=(sim_stats['cov_evec'].T@sim.T).T
#             sim_stats['PC']['stats']=calc_sim_stats(sim=sim_stats['PC']['data'],PC=True)
#         except Exception as err:
#             print(err)
#             sim_stats['PC']=err
#     else:
#         sim_truth=sim_stats['mean']
    
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
    
def sim_cl_xi(Rsize=150,do_norm=False,cl0=None,kappa_class=None,fsky=f_sky,zbins=None,
            use_shot_noise=True,
             convolve_win=False,nside=nside,use_cosmo_power=True,lognormal=False,
             lognormal_scale=1, add_SSV=True,add_tidal_SSV=True,
              add_blending=False,blending_coeff=-2,fiber_coll_coeff=-1):
    print('running sim_cl_xi','lognormal:',lognormal,'blending', add_blending, 'SSV:',add_SSV)
    l=kappa_class.l
    shear_lcut=l>=2
    
    l_bins=kappa_class.l_bins
    dl=l_bins[1:]-l_bins[:-1]
    nu=(2.*l+1.)*fsky
    
    mask={}
    window={}
    tree_cat={}
    tree_cat_args={}
    if convolve_win:
        nu=2.*l+1.
        for tracer in kappa_class.z_bins.keys():
            window[tracer]=kappa_class.z_bins[tracer][0]['window']
            mask[tracer]=window[tracer]==hp.UNSEEN
            seen_indices = np.where( ~mask[tracer] )[0]
            theta, phi = hp.pix2ang(nside, seen_indices)
            del seen_indices
            ra = np.degrees(np.pi*2.-phi)
            dec = -np.degrees(theta-np.pi/2.)
            tree_cat_args[tracer] = {'ra':ra, 'dec':dec, 'ra_units':'degrees', 'dec_units':'degrees'}
            tree_cat[tracer] = treecorr.Catalog(w=window[tracer][~mask[tracer]], **tree_cat_args[tracer])

            
    tree_corrs={}
    window_norm={}
    corr_config = {'min_sep':th_min*60, 'max_sep':th_max*60, 'nbins':n_th_bins, 'sep_units':'arcmin'}
    for corr in kappa_class.corrs:
        if corr==corr_gg:
            tree_corrs[corr]=treecorr.NNCorrelation(**corr_config)
            _=tree_corrs[corr].process(tree_cat['galaxy'])
        if corr==corr_ll:
            tree_corrs[corr]=treecorr.NNCorrelation(**corr_config)
            _=tree_corrs[corr].process(tree_cat['shear'])
            print(tree_corrs[corr],tree_cat['shear'])
        if corr==corr_ggl:
            tree_corrs[corr]=treecorr.NNCorrelation(**corr_config)
            _=tree_corrs[corr].process(tree_cat['galaxy'],tree_cat['shear'])
        window_norm[corr]=tree_corrs[corr].weight*1.
        
    
    del tree_cat
    del tree_corrs
        
    outp={}
    win=0
    clp_shear_B=None
    if cl0 is None:
        cl0={}
        clG0=kappa_class.cl_tomo() 
        for corr in kappa_class.corrs:
            cl0[corr]=clG0['cl'][corr][(0,0)].compute()
    clg0={}
    clN0={}
    for corr in kappa_class.corrs: #ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
        shot_noise=kappa_class.SN[corr_gg][:,0,0]*0
        if corr[0]==corr[1]:
            shot_noise=kappa_class.SN[corr][:,0,0]
        shot_noise=shot_noise*use_shot_noise
        clN0[corr]=shot_noise
        clg0[corr]=cl0[corr]*use_cosmo_power#+shot_noise
        if corr==corr_ll:
            clg0['shear_B']=cl0[corr]*0
            clN0['shear_B']=shot_noise
            #clp_shear_B=shot_noise@coupling_M[corr_ll]+(cl0[corr_ll]+shot_noise)@coupling_M['shear_B']
#             clp_shear_B=cl0[corr_ll]@coupling_M['shear_B']
    ndim=len(kappa_class.corrs)
    if corr_ll in kappa_class.corrs:#2xi for ll
        ndim+=1
    
    outp['clg0_0']=clg0.copy()
    outp['clN0_0']=clN0.copy()
    outp['ndim']=ndim
    if ndim>1:
        clg0=(clg0[corr_gg],clg0[corr_ll],clg0['shear_B'],clg0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
        clN0=(clN0[corr_gg],clN0[corr_ll],clN0['shear_B'],clN0[corr_ggl])#ordering: TT, EE, BB, TE if 4 cl as input.. use newbool=True
    else:
        clg0=clg0[corr_gg]
        clN0=clN0[corr_gg]
    
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
        
    
    SN=kappa_class.SN
    sim_xi_shape=(Rsize,n_th_bins*ndim)
    xi=np.zeros(sim_xi_shape)
    lmax=max(l)
    lmin=min(l)
    nu_b=None
    
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
    
    
    corr_t=[corr_gg,corr_ll,corr_ggl] #order in which sim corrs are output.
    seed=12334
    def get_clsim(i,):
        tracemalloc.clear_traces()
        tracemalloc.start()


        # print('doing map: ',i,thread_count(), 'lognormal:',lognormal,'blending', add_blending)
        local_state = np.random.RandomState(seed+i)
#         cl_map=hp.synfast(clg0,nside=nside,RNG=local_state,new=True,pol=True)
        if lognormal:
            print('doing lognormal')
            cl_map=hp.synfast(clg0,nside=nside,rng=local_state,new=True,pol=False,verbose=False)
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
            cl_map=hp.synfast(clg0,nside=nside,rng=local_state,new=True,pol=True,verbose=False)
            

        N_map=0
        if use_shot_noise:
            N_map=hp.synfast(clN0,nside=nside,rng=local_state,new=True,pol=True,verbose=False)
        if add_SSV:
            SSV_delta=np.random.multivariate_normal(mean=SSV_sigma*0,cov=SSV_cov,size=1)[0]
       
        tracers=['galaxy','shear','shear']
        tree_cat={}
        tree_corrs={}
        
        xi=np.zeros((1,n_th_bins*ndim))
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
                    window2[mask[tracer]]=0#hp.UNSEEN
                    # print('adding blending ',tracer,blending_coeff,fiber_coll_coeff)
                    # print( 'window2',window2.max(),window2.min(),window2[mask[tracer]].min(),
                    #         window2[mask[tracer]],window[tracer])
                # print(np.all(window2==window[tracer]))   
                cl_map[i]*=window2
                if use_shot_noise:
                    N_map[i]*=np.sqrt(window2)
                    cl_map[i]+=N_map[i]
#                     N_map[i][mask[tracer]]=hp.UNSEEN
                cl_map[i][mask[tracer]]=hp.UNSEEN
                
            tree_cat['galaxy'] = treecorr.Catalog(w=cl_map[i][~mask['galaxy']], **tree_cat_args['galaxy'])
            tree_cat['shear'] = treecorr.Catalog(g1=cl_map[1][~mask['shear']],g2=cl_map[2][~mask['shear']], **tree_cat_args['shear'])
            th_i=0
            for corr in [corr_ll,corr_ggl,corr_gg]:
                if corr==corr_ggl:
                    tree_corrs[corr]=treecorr.NGCorrelation(**corr_config)
                    tree_corrs[corr_ggl].process(tree_cat['galaxy'],tree_cat['shear'])
                    xi[0,th_i:th_i+n_th_bins]=tree_corrs[corr_ggl].xi*(tree_corrs[corr_ggl].weight/window_norm[corr_ggl])
                    th_i+=n_th_bins
                if corr==corr_ll:
                    tree_corrs[corr]=treecorr.GGCorrelation(**corr_config)
                    tree_corrs[corr_ll].process(tree_cat['shear'])
                    xi[0,th_i:th_i+n_th_bins]=tree_corrs[corr_ll].xip*(tree_corrs[corr_ll].weight/window_norm[corr_ll])
                    th_i+=n_th_bins
                    xi[0,th_i:th_i+n_th_bins]=tree_corrs[corr_ll].xim*(tree_corrs[corr_ll].weight/window_norm[corr_ll])
                    th_i+=n_th_bins
                if corr==corr_gg:
                    tree_corrs[corr]=treecorr.NNCorrelation(**corr_config)
                    tree_corrs[corr_gg].process(tree_cat['galaxy'])
                    xi[0,th_i:th_i+n_th_bins]=tree_corrs[corr_gg].weight/window_norm[corr_gg]
                    th_i+=n_th_bins
            del tree_corrs
            del tree_cat
            del cl_map
            del N_map
            return xi
        else:
            cl_map*=window
            cl_map[mask]=hp.UNSEEN
            clpi=hp.anafast(cl_map, lmax=max(l),pol=True)[l]
            crash
            return clpi.T,clgi.T
    
    def comb_maps(futures):
        for i in np.arange(Rsize):
            x=futures[i]#.compute()
            clp[i,:,:]+=x[0]
            clg[i,:,:]+=x[1]
        return clp,clg 
    
    # print('generating maps', thread_count())
    if convolve_win:
        futures={}
#         for i in np.arange(Rsize):
#             futures[i]=dask.delayed(get_clsim)(i)  
#         print(futures)
#         clpg=dask.delayed(comb_maps)(futures)
#         clpg.compute()
        i=0
        j=0
        step=min(np.int(5),Rsize)
#         funct=partial(get_clsim2,clg0,window,mask,SN,coupling_M,coupling_M_inv,ndim)
        while j<Rsize:
            futures={}
            #client=Client(LC,)
            for ii in np.arange(step):
                futures[ii]=delayed(get_clsim)(i+ii,)  
            futures=client.compute(futures)
            for ii in np.arange(step):
                xi[i]=futures.result()[ii]                        
                i+=1
            print('done map ',i, thread_count())
            del futures
            gc.collect()
            #client.restart()
            #client.close()
            # print('done map ',i, thread_count())
            j+=step
        
    print('done generating maps',xi.mean(axis=0).shape)
    #client=Client(LC,)    
    outp['xi_stats']={}
    outp['xi_stats']=client.compute(delayed(calc_sim_stats)(sim=xi,sim_truth=xi.mean(axis=0))).result()
    outp['mask']=mask

    outp['xi']=xi
    outp['clg0']=clg0
    outp['clN0']=clN0
    outp['cl0']=cl0
    outp['window_norm']=window_norm
    outp['window']=window
    outp['size']=Rsize
    outp['fsky']=fsky
    outp['nu']=nu
    outp['nu_b']=nu_b
    outp['l_bins']=l_bins
        
    outp['use_shot_noise']=use_shot_noise
    
    #client.close()
    return outp
#test_home=wig_home+'/tests/'
# test_home='/media/deep/data/repos/cosmic_shear/tests/'
test_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/tests/'
fname=test_home+'/corr_sims_newN'+str(nsim)+'_ns'+str(nside)+'_lmax'+str(lmax_cl)+'_wlmax'+str(window_lmax)+'_fsky'+str(f_sky)
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

#client.restart()
#client.close()
cl_sim_W=sim_cl_xi(Rsize=nsim,do_norm=False,#cl0=clG0['cl'][corrs[0]][(0,0)].compute(),
          kappa_class=kappa_win,fsky=f_sky,use_shot_noise=use_shot_noise,use_cosmo_power=use_cosmo_power,
             convolve_win=True,nside=nside,lognormal=lognormal,lognormal_scale=lognormal_scale,
             add_SSV=do_SSV_sim,add_tidal_SSV=do_SSV_sim,add_blending=do_blending)

outp={}
outp['simW']=cl_sim_W
outp['zs_bin1']=zs_bin1

outp['zl_bin1']=zl_bin1
outp['cl0']=cl0
outp['cl0_win']=cl0_win
outp['xi_win']=xiW_L
outp['xi0']=xi_L0

with open(fname,'wb') as f:
    pickle.dump(outp,f)
written=True

print(fname)
print('all done')

LC.close()
