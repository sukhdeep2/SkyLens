import os
os.environ["OMP_NUM_THREADS"] = "4" #need to be set early

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
    
    test_run=True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_xi", "-do_xi",type=int, help="")
    parser.add_argument("--eh_pk", "-eh_pk",type=int, help="")
    parser.add_argument("--fix_cosmo", "-fc",type=int, help="use unit window")
    parser.add_argument("--bin_l", "-bl",type=int, help="use complicated window")
    parser.add_argument("--scheduler", "-s", help="Scheduler file")
    parser.add_argument("--dask_dir", "-Dd", help="dask log directory")
    args = parser.parse_args()

    args = parser.parse_args()

    fix_cosmo=False if args.fix_cosmo is None else np.bool(args.fix_cosmo)
    do_xi=False if args.do_xi is None else np.bool(args.do_xi)
    eh_pk=True if args.eh_pk is None else np.bool(args.eh_pk)
    use_binned_l=True if args.bin_l is None else np.bool(args.bin_l)

    print('Doing mcmc',fix_cosmo,do_xi,use_binned_l,eh_pk,test_run) #err  True False True True     False True False True

    Scheduler_file=args.scheduler
    dask_dir=args.dask_dir


    do_pseudo_cl=not do_xi
    use_binned_theta=use_binned_l

    nwalkers=100
    nsteps=100
    yml_file='./mcmc.yml'
    python_file='mcmc_args.py'
    if test_run:
        nzbins=2
        nwalkers=10
        nsteps=10
        yml_file='./mcmc_test.yml'
        python_file='mcmc_test_args.py'
     
    ncpu=25
    # os.environ["OMP_NUM_THREADS"] = "1"
    LC,scheduler_info=start_client(Scheduler_file=Scheduler_file,local_directory=dask_dir,ncpu=None,n_workers=ncpu,threads_per_worker=1,
                                  memory_limit='120gb',dashboard_address=8801,processes=True)
    client=client_get(scheduler_info=scheduler_info)
    print('client: ',client)#,dask_dir,scheduler_info)
 
    lock = None #Lock(name="Why_Camb_Why",client=client)
    #We get segmentation fault if camb is not under lock and multiple instances are called (tested on single node).
    # Why camb is called, multiple instances run, but they all rely on same underlying fortran objects and there are race conditions.
    # I tried pickling camb, but Camb cannot be pickled because it's a fortran wrapper and pickle doesn't really know what to do with it. deepcopy throws same error too.
    # deepcopy of Ang_PS doesn't capture it because there is no camb object stored as part of PS. It is simply called.


#     skylens_args=parse_yaml(file_name=yml_file)
    skylens_args=parse_python(file_name=python_file)
    skylens_args['do_xi']=do_xi
    skylens_args['do_pseudo_cl']=do_pseudo_cl
    skylens_args['use_binned_theta']=use_binned_theta
    skylens_args['use_binned_l']=use_binned_l
    if eh_pk:
        print('mcmc will use eh_pk')
        skylens_args['pk_params']['pk_func']='eh_pk'
    skylens_args['scheduler_info']=scheduler_info
    zs_bin=skylens_args['shear_zbins']
    file_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/tests/imaster/'
    if do_xi:
        fname_out='mcmc_dat_xi_{nz}_bl{bl}_bth{bth}_eh{eh_pk}.pkl'.format(nz=zs_bin['n_bins'],bl=int(use_binned_l),
                                                                              bth=int(use_binned_theta),eh_pk=int(eh_pk))
    if do_pseudo_cl:
        fname_out='mcmc_dat_pcl_{nz}_bl{bl}_bth{bth}_eh{eh_pk}.pkl'.format(nz=zs_bin['n_bins'],bl=int(use_binned_l),
                                                                               bth=int(use_binned_theta),eh_pk=int(eh_pk))

    get_cov=False
    try:
        fname_cl=file_home+fname_out
        with open(fname_cl,'rb') as of:
            cl_all=pickle.load(of)
        Win=cl_all['Win']
        if do_pseudo_cl:
            cl_cov=cl_all['cov']
            cov_inv=np.linalg.inv(cl_cov.todense())
            data=cl_all['pcl_b']
        elif do_xi:
            xi_cov=cl_all['cov']
            cov_inv=np.linalg.inv(xi_cov.todense())
            data=cl_all['xi']
        zs_bin=cl_all['zs_bin']
        skylens_args['shear_zbins']=zs_bin
        print('read cl / cov from file: ',fname_cl)
    except Exception as err:
        get_cov=True
        print('cl not found. Will compute',fname_cl,err)

    outp={}
    if get_cov:    
        kappa0=Skylens(**skylens_args)
        print('kappa0 size',get_size_pickle(kappa0))#,kappa0.Win)
        print('kappa0.Ang_PS size',get_size_pickle(kappa0.Ang_PS))

        if do_xi:
            print('MCMC getting xi0G')
            xi0G=kappa0.xi_tomo()
            print('MCMC getting xi0 stack')
            xi_cov=client.compute(xi0G['stack']).result()
            cov_inv=np.linalg.inv(xi_cov['cov'])#.todense())
            data=xi_cov['xi']
        else:
            print('MCMC getting cl0G')
            cl0G=kappa0.cl_tomo()
            print('MCMC getting stack')
            cl_cov=client.compute(cl0G['stack']).result()
            cov_inv=np.linalg.inv(cl_cov['cov'])
            data=cl_cov['pcl_b']

        kappa0.gather_data()
        Win=kappa0.Win
        outp['Win']=kappa0.Win
        outp['zs_bin']=skylens_args['shear_zbins']
        with open(fname_cl,'wb') as of:
            pickle.dump(outp,of)
        del kappa0

    def check_finite(dat=None,prefix=None):
        if prefix is None:
            prefix='Check_finite: '
        all_finite=True
        if isinstance(dat,dict):
            for k in dat.keys():
                all_finite=np.logical_and(all_finite,check_finite(dat=dat[k],prefix=prefix+' dict key '+str(k)))
        else:
            try:
                x=np.isfinite(dat)
                if not np.all(x):
                    print(prefix,' not finite',data[~x],np.where(x))
                    all_finite=False
            except Exception as err:
                print('Check_finite error',err,dat)
        return all_finite


    print('Got data and cov')
    all_finite=check_finite(dat=data,prefix='data problem ')
    if not all_finite:
        check_finite(dat=outp['zs_bin'],prefix='data problem, zbins: ')
        check_finite(dat=outp['Win'],prefix='data problem, Win: ')

    Win['cov']=None
    skylens_args['do_cov']=False
    skylens_args['Win']=Win

    kappa0=Skylens(**skylens_args)
    # kappa0.Win={'cl':client.gather(kappa0.Win['cl'])}
    Win=client.gather(kappa0.Win)
    kappa0.gather_data()

    WT=client.scatter(kappa0.WT,broadcast=True)

    WT_binned=client.gather(kappa0.WT_binned)
    WT_binned=client.scatter(WT_binned,broadcast=True)
    data=client.scatter(data,broadcast=True)
    cov_inv=client.scatter(cov_inv,broadcast=True)

    cosmo_fid=copy.deepcopy(kappa0.Ang_PS.PS.cosmo_params)
    cosmo_fid.pop('astropy_cosmo')
    params_order=['b1_{i}'.format(i=i) for i in np.arange(kappa0.tracer_utils.z_bins['galaxy']['n_bins'])]#,'Ase9','Om']

    priors_max=np.ones(len(params_order))*2
    priors_min=np.ones(len(params_order))*.5
    if not fix_cosmo:
        params_order+=['Ase9','Om']
        pf=np.array([cosmo_fid[k] for k in ['Ase9','Om']])
        priors_max=np.append(priors_max,pf*2)
        priors_min=np.append(priors_min,pf*.5)

    zs_bin1=copy.deepcopy(client.gather(kappa0.tracer_utils.z_bins['shear']))
    zs_bin=copy.deepcopy(zs_bin1)
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
    print('kappa0 pk',kappa0.Ang_PS.PS.pk_func)
    
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
        bin_indx=int(pp[1])
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
        if np.isnan(chisq):
            print('chisq problem: ',chisq,model)
            check_finite(dat=z_bins,prefix='chisq problem: zbins: ')
            chisq=-np.inf
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
                bin_indx=int(pp[1])
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
        'time'+str(time.time()-t1)+str((time.time()-t1)/3600.), 'nsteps: ',nsteps, 'chain shape',outp['chain'].shape)
        return outp

    if not do_xi and not use_binned_l:
        print('unbinned cl will not work in this example because covariance is always binned')
        client.shutdown()
        sys.exit(0)
    outp=sample_params()
    print('calcs done')

    outp['skylens_args']=skylens_args
    # outp['l_bins']=
    # outp['do_xi']=do_xi
    # outp['do_pseudo_cl']=do_pseudo_cl
    # outp['use_binned_l']=use_binned_l
    # outp['use_binned_theta']=use_binned_theta
    # outp['zbins']=zs_bin1
    outp['data']=data
    outp['cov_inv']=cov_inv
    outp['params_order']=params_order

    file_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/tests/imaster/'
    zs_bin1=client.gather(zs_bin1)
    if do_xi:
        fname_out='xi_{nz}_bl{bl}_bth{bth}_nw{nw}_ns{ns}_camb{fc}.pkl'.format(nz=zs_bin1['n_bins'],bl=int(use_binned_l),
                                                                              bth=int(use_binned_theta),ns=nsteps,nw=nwalkers,fc=int(fix_cosmo))
    if do_pseudo_cl:
        fname_out='pcl_{nz}_bl{bl}_bth{bth}_nw{nw}_ns{ns}_camb{fc}.pkl'.format(nz=zs_bin1['n_bins'],bl=int(use_binned_l),
                                                                               bth=int(use_binned_theta),ns=nsteps,nw=nwalkers,fc=int(fix_cosmo))

    fname_out=file_home+fname_out
    with open(fname_out, 'wb') as f:
        pickle.dump(outp,f)
    print('file written: ',fname_out)
    client.shutdown()
    sys.exit(0)
