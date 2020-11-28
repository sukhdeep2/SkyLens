import os
os.environ["OMP_NUM_THREADS"] = "5" #need to be set early

import sys
import emcee

from skylens import *
# from skylens.utils import *
from resource import getrusage, RUSAGE_SELF
import psutil
from distributed.utils import format_bytes

from dask.distributed import Lock

import faulthandler; faulthandler.enable()
    #in case getting some weird seg fault, run as python -Xfaulthandler something.py
    # problem is likely to be in some package

import multiprocessing
from distributed import LocalCluster
from dask.distributed import Client

import argparse

if __name__=='__main__':
    
    
    test_run=False
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_xi", "-do_xi",type=int, help="")
    parser.add_argument("--fix_cosmo", "-fc",type=int, help="use unit window")
    parser.add_argument("--bin_l", "-bl",type=int, help="use complicated window")
    parser.add_argument("--scheduler", "-s", help="Scheduler file")
    parser.add_argument("--dask_dir", "-Dd", help="dask log directory")
    args = parser.parse_args()

    args = parser.parse_args()

    fix_cosmo=False if args.fix_cosmo is None else np.bool(args.fix_cosmo)
    do_xi=False if args.do_xi is None else np.bool(args.do_xi)
    use_binned_l=True if args.bin_l is None else np.bool(args.bin_l)

    print('Doing mcmc',fix_cosmo,do_xi,use_binned_l,test_run) #err  True False True True     False True False True

    Scheduler_file=args.scheduler
    dask_dir=args.dask_dir


    do_pseudo_cl=not do_xi
    use_binned_theta=use_binned_l

    nzbins=5
    lmax_cl=2000


    nwalkers=100
    nsteps=100

    ncpu=multiprocessing.cpu_count()-1
    vmem=psutil.virtual_memory()
    mem=str(vmem.total/(1024**3)*0.95)+'GB'
    memory='120gb'#'120gb'

    if test_run:
        nzbins=2
        lmax_cl=200
    #     memory='20gb'
    #     ncpu=4
        nwalkers=10
        nsteps=10
 
    # os.environ["OMP_NUM_THREADS"] = "1"
    LC,scheduler_info=start_client(Scheduler_file=Scheduler_file,local_directory=dask_dir,ncpu=None,n_workers=25,threads_per_worker=1,
                                  memory_limit=memory,dashboard_address=8801,processes=True)
    client=client_get(scheduler_info=scheduler_info)
    client.wait_for_workers(n_workers=1)
    print('client: ',client)#,dask_dir,scheduler_info)

    lock = None #Lock(name="Why_Camb_Why",client=client)
    #We get segmentation fault if camb is not under lock and multiple instances are called (tested on single node).
    # Why camb is called, multiple instances run, but they all rely on same underlying fortran objects and there are race conditions.
    # I tried pickling camb, but Camb cannot be pickled because it's a fortran wrapper and pickle doesn't really know what to do with it. deepcopy throws same error too.
    # deepcopy of Ang_PS doesn't capture it because there is no camb object stored as part of PS. It is simply called.

    wigner_files={}
    wig_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/'
    wigner_files[0]= wig_home+'dask_wig3j_l3500_w2100_0_reorder.zarr'
    wigner_files[2]= wig_home+'/dask_wig3j_l3500_w2100_2_reorder.zarr'

    bin_cl=True
    lmin_cl=2
    l0=np.arange(lmin_cl,lmax_cl)

    lmin_cl_Bins=lmin_cl+10
    lmax_cl_Bins=lmax_cl-10
    Nl_bins=25
    l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins+1))
    lb=np.sqrt(l_bins[1:]*l_bins[:-1])

    l=np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell
    bin_cl=True

    xi_win_approx=True
    bin_xi=True
    xi_SN_analytical=True
    th_min=25/60
    th_max=250./60
    n_th_bins=20
    th_bins=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins+1)
    th=np.logspace(np.log10(th_min),np.log10(th_max),n_th_bins*40)
    thb=np.sqrt(th_bins[1:]*th_bins[:-1])

    WT_kwargs={'l':l0,'theta':th*d2r,'s1_s2':[(2,2),(2,-2),(0,0),(0,2)]}
    WT=wigner_transform(wig_d_taper_order_low=6,wig_d_taper_order_high=8,**WT_kwargs)

    do_cov=True

    SSV_cov=False
    tidal_SSV_cov=SSV_cov
    Tri_cov=SSV_cov

    use_window=True
    store_win=True

    nside=32
    window_lmax=nside

    zs_bin=lsst_source_tomo_bins(nbins=nzbins,use_window=use_window,nside=nside)

    do_cov=True

    SSV_cov=False
    bin_cl=True
    do_cov=True
    Tri_cov=False

    corr_ggl=('galaxy','shear')
    corr_gg=('galaxy','galaxy')
    corr_ll=('shear','shear')
    corrs=[corr_ll,corr_ggl,corr_gg]

    power_spectra_kwargs={} #{'pk_lock':lock}

    kappa0=Skylens(zs_bins=zs_bin,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zs_bin,
                   use_window=use_window,Tri_cov=Tri_cov,
                   use_binned_l=True,wigner_files=wigner_files, #binned_l here is true because we cannot fit everything in memory otherwise
                   SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=0.35,
                   store_win=store_win,window_lmax=window_lmax,
                   sparse_cov=True,corrs=corrs,
                   do_xi=do_xi,bin_xi=bin_xi,theta_bins=th_bins,WT=WT,
                    use_binned_theta=True,
                    nz_PS=100,do_pseudo_cl=do_pseudo_cl,xi_win_approx=True,
                    xi_SN_analytical=xi_SN_analytical,power_spectra_kwargs=power_spectra_kwargs,
                    scheduler_info=scheduler_info
                   )
    print('kappa0 size',get_size_pickle(kappa0))#,kappa0.Win)
    print('kappa0.Ang_PS size',get_size_pickle(kappa0.Ang_PS))
    # client.restart() #can't do this with futures
    #xi0t=kappa0.tomo_short()
    if do_xi:
        print('MCMC getting xi0G')
        xi0G=kappa0.xi_tomo()
        print('MCMC getting xi0 stack')
        xi_cov=client.compute(xi0G['stack']).result()
        cov_inv=np.linalg.inv(xi_cov['cov'].todense())
        data=xi_cov['xi']
    #     for k in xi0G.keys():
    #         client.cancel(xi0G[k])
    #     del xi0G
    else:
        print('MCMC getting cl0G')
        cl0G=kappa0.cl_tomo()
    #     print('MCMC getting pcl',cl0G.keys(),kappa0.Win['cl'][corr_ll][(0,0)])
    #     pcl=client.compute(cl0G['pseudo_cl_b']).result()
        print('MCMC getting stack')
        cl_cov=client.compute(cl0G['stack']).result()
        cov_inv=np.linalg.inv(cl_cov['cov'].todense())
        data=cl_cov['pcl_b']
    #     del cl0G
    kappa0.gather_data()
    print('Got data and cov')
    if not np.all(np.isfinite(data)):
        x=np.isfinite(data)
        print('data problem',data[~x],np.where(x))
    #Win={'cl':client.gather(kappa0.Win['cl'])}
    Win=None
    del kappa0
#     client.restart()
#     client.wait_for_workers(n_workers=1)
    do_cov=False
    #kappa0.do_cov=False
    kappa0=Skylens(zs_bins=zs_bin,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0, zg_bins=zs_bin,
                   use_window=use_window,Tri_cov=Tri_cov,
                    use_binned_l=use_binned_l,wigner_files=wigner_files,#covariance is false, binned_l false should be ok
                    SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,f_sky=0.35,
                    store_win=store_win,window_lmax=window_lmax,
                    sparse_cov=True,corrs=corrs,
                    do_xi=do_xi,bin_xi=bin_xi,theta_bins=th_bins,WT=WT,
                     use_binned_theta=use_binned_theta,
                     nz_PS=100,do_pseudo_cl=do_pseudo_cl,xi_win_approx=True,
                     xi_SN_analytical=xi_SN_analytical,power_spectra_kwargs=power_spectra_kwargs,
                    Win=Win,scheduler_info=scheduler_info
                    )
    # kappa0.Win={'cl':client.gather(kappa0.Win['cl'])}
    Win=client.gather(kappa0.Win)
    kappa0.gather_data()

    WT=client.scatter(kappa0.WT,broadcast=True)

    WT_binned=client.gather(kappa0.WT_binned)
    WT_binned=client.scatter(WT_binned,broadcast=True)
    data=client.scatter(data,broadcast=True)
    cov_inv=client.scatter(cov_inv,broadcast=True)

    cosmo_fid=kappa0.Ang_PS.PS.cosmo_params

    params_order=['b1_{i}'.format(i=i) for i in np.arange(zs_bin['n_bins'])]#,'Ase9','Om']

    priors_max=np.ones(len(params_order))*2
    priors_min=np.ones(len(params_order))*.5
    if not fix_cosmo:
        params_order+=['Ase9','Om']
        pf=np.array([cosmo_fid[k] for k in ['Ase9','Om']])
        priors_max=np.append(priors_max,pf*2)
        priors_min=np.append(priors_min,pf*.5)

    zs_bin1=copy.deepcopy(zs_bin)
    del_k=['window','window_cl']
    for k in del_k:
        if zs_bin1.get(k) is not None:
            del zs_bin1[k]
        for i in np.arange(zs_bin1['n_bins']):
            if zs_bin1[i].get(k) is not None:
                del zs_bin1[i][k]

    zs_bin1=scatter_dict(zs_bin1,scheduler_info=scheduler_info,broadcast=True) 
    cl_bin_utils=scatter_dict(kappa0.cl_bin_utils,broadcast=True)
    xi_bin_utils=scatter_dict(kappa0.xi_bin_utils,broadcast=True)

    if fix_cosmo:
        kappa0.Ang_PS.angular_power_z()
    else:
        kappa0.Ang_PS.reset()
    kappa0=client.scatter(kappa0,broadcast=True)

    proc = psutil.Process()
    print('starting mcmc ', 'mem, peak mem: ',format_bytes(proc.memory_info().rss),
                     int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.)
                     )

    def get_priors(params):#assume flat priors for now
        x=np.logical_or(np.any(params>priors_max,axis=1),np.any(params<priors_min,axis=1))
        p=np.zeros(len(params))
        p[x]=-np.inf
        return p

    def assign_zparams(zbins={},p_name='',p_value=None):
        pp=p_name.split('_')
        p_n=pp[0]
        bin_indx=np.int(pp[1])
        zbins[bin_indx][p_n]=p_value
        return zbins

    def get_params(params,kappa0,z_bins,log_prior):
        cosmo_params=copy.deepcopy(cosmo_fid)
#         Ang_PS=copy.deepcopy(kappa0.Ang_PS)
        Ang_PS=kappa0.Ang_PS
        if not np.isfinite(log_prior):
            return cosmo_params,z_bins,Ang_PS
        zbins=copy.deepcopy(z_bins)
        i=0
        for p in params_order:
            if cosmo_params.get(p) is not None:
                cosmo_params[p]=params[i]
            else:
                zbins=assign_zparams(zbins=zbins,p_name=p,p_value=params[i])
            i+=1
        zbins={'galaxy':zbins,'shear':zbins}
#         model=kappa.tomo_short(cosmo_params=cosmo_params,z_bins=zbins,Ang_PS=Ang_PS,pk_lock=pk_lock)
        return cosmo_params,zbins,Ang_PS

    def get_model(params,data,cov_inv,kappa0,z_bins,log_prior,indx,Win,WT,WT_binned,cl_bin_utils,xi_bin_utils):
        if not np.isfinite(log_prior):
            return -np.inf #np.zeros_like(data)
        cosmo_params,z_bins,Ang_PS=get_params(params,kappa0,z_bins,log_prior)
#         cosmo_params,z_bins,Ang_PS=params
#         Ang_PS.angular_power_z(cosmo_params=cosmo_params)
    #     Ang_PS=copy.deepcopy(Ang_PS)
#         if not np.isfinite(log_prior):
#             return np.zeros_like(data)
        model=kappa0.tomo_short(cosmo_params=cosmo_params,z_bins=z_bins,Ang_PS=Ang_PS,Win=Win,WT=WT,WT_binned=WT_binned,
                                cl_bin_utils=cl_bin_utils,xi_bin_utils=xi_bin_utils)#,pk_lock=pk_lock)
        loss=data-model
        chisq=-0.5*loss@cov_inv@loss
        chisq+=log_prior
        return chisq #model
    
    def get_PS(kappa0,args_p):  
        ap=args_p[2]
        ap.angular_power_z(cosmo_params=args_p[0])
        return args_p

    def chi_sq(params,data,cov_inv,kappa0,z_bins,pk_lock):
        t1=time.time()
        params=np.atleast_2d(params)
        log_priors=get_priors(params)
        n_params=len(params)
        models={}
        chisq={i:delayed(get_model)(params[i],data,cov_inv,kappa0,z_bins,log_priors[i],i,Win,WT,WT_binned,cl_bin_utils,xi_bin_utils) for i in np.arange(n_params)}
        chisq=client.compute(chisq).result()
        chisq=[chisq[i]for i in np.arange(n_params)]
            #         chisq=np.zeros(n_params)-np.inf
            #         for i in np.arange(n_params):
            #             chisq[i]=-0.5*loss[i]@cov_inv@loss[i].T
            #         chisq+=log_priors
        print('chisq',time.time()-t1)
        return chisq

    def ini_walkers():
        ndim=len(params_order)
        p0=np.zeros(ndim)
        p0f=np.zeros(ndim)
        i=0
        for p in params_order:
            if cosmo_fid.get(p) is not None:
                p0[i]=cosmo_fid[p]
                p0f=p0[i]*.5
            else:
                pp=p.split('_')
                p_n=pp[0]
                bin_indx=np.int(pp[1])
    #             print(bin_indx,p_n,zs_bin1[bin_indx].keys())
                p0[i]=zs_bin[bin_indx][p_n]
                p0f=.2
            i+=1
        return p0,p0f,ndim

    nsteps_burn=1
    thin=1

    def sample_params(fname=''):
        p00,p0_f,ndim=ini_walkers()
        p0 = np.random.uniform(-1,1,ndim * nwalkers).reshape((nwalkers, ndim))*p0_f*p00 + p00

        outp={}
        sampler = emcee.EnsembleSampler(nwalkers, ndim,chi_sq,threads=ncpu,a=2,vectorize=True,args=(data,cov_inv,kappa0,zs_bin1,lock))
                                                                    #a=2 default, 5 ok

        t1=time.time()

        pos, prob, state = sampler.run_mcmc(p0, nsteps_burn,store=False)
        print('done burn in '+str(time.time()-t1)+'  '+str((time.time()-t1)/3600.)+'  '+
        str(np.mean(sampler.acceptance_fraction)))

        sampler.reset()

        step_size=nsteps
        steps_taken=0
    #     if step_size%thin!=0 or step_size==0:
    #         step_size=max(1,int(step_size/thin)*thin+thin)

    #         step_size=nsteps/10 #30
    #         if step_size%thin!=0 or step_size==0:
    #             step_size=max(1,int(step_size/thin)*thin+thin)
    #         #print 'step-size',step_size,thin
    #         steps_taken=0
    #     while steps_taken<nsteps:
        pos, prob, state =sampler.run_mcmc(pos, step_size,thin=thin)
        steps_taken+=step_size
        outp['chain']=sampler.flatchain
        outp['p0']=p00
        outp['params']=params_order
        outp['ln_prob']=sampler.lnprobability.flatten()
        outp['acceptance_fraction']=np.mean(sampler.acceptance_fraction)
        outp['pos']=pos
        outp['prob']=prob
        outp['nsteps']=nsteps
        outp['nwalkers']=nwalkers
        outp['burnin']=nsteps_burn
        outp['thin']=thin
        outp['time']=time.time()-t1

        print('Done steps '+str(steps_taken)+ ' acceptance fraction ' +str(outp['acceptance_fraction'])+'  '
        'time'+str(time.time()-t1)+str((time.time()-t1)/3600.))
        return outp

    if not do_xi and not use_binned_l:
        print('unbinned cl will not work in this example because covariance is always binned')
        client.shutdown()
        sys.exit(0)
    outp=sample_params()
    print('calcs done')

    outp['l0']=l0
    outp['l_bins']=l_bins
    outp['do_xi']=do_xi
    outp['do_pseudo_cl']=do_pseudo_cl
    outp['use_binned_l']=use_binned_l
    outp['use_binned_theta']=use_binned_theta
    outp['data']=data
    outp['zbins']=zs_bin1
    outp['cov_inv']=cov_inv
    outp['params_order']=params_order

    file_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/tests/imaster/'
    zs_bin1=client.gather(zs_bin1)
    if do_xi:
        fname_out='xi_{nz}_bl{bl}_bth{bth}_nw{nw}_ns{ns}_camb{fc}.pkl'.format(nz=zs_bin1['n_bins'],bl=np.int(use_binned_l),
                                                                              bth=np.int(use_binned_theta),ns=nsteps,nw=nwalkers,fc=int(fix_cosmo))
    if do_pseudo_cl:
        fname_out='pcl_{nz}_bl{bl}_bth{bth}_nw{nw}_ns{ns}_camb{fc}.pkl'.format(nz=zs_bin1['n_bins'],bl=np.int(use_binned_l),
                                                                               bth=np.int(use_binned_theta),ns=nsteps,nw=nwalkers,fc=int(fix_cosmo))

    fname_out=file_home+fname_out
    with open(fname_out, 'wb') as f:
        pickle.dump(outp,f)
    print('file written: ',fname_out)
    client.shutdown()
    sys.exit(0)
