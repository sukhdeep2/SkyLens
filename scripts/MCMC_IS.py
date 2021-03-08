import os
os.environ["OMP_NUM_THREADS"] = "4" #need to be set early

import sys
import scipy,qmcpy

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

    N_points=512**2
    grid='Sobol'
    sample_dist='t'
    yml_file='./mcmc.yml'
    python_file='mcmc_args.py'
    if test_run:
        N_points=512
        nzbins=2
        nwalkers=10
        nsteps=10
        yml_file='./mcmc_test.yml'
        python_file='mcmc_test_args.py'
     
    ncpu=10
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
        fname_out='mcmc_dat_xi_{nz}_bl{bl}_bth{bth}_eh{eh_pk}.pkl'.format(nz=zs_bin['n_bins'],bl=np.int(use_binned_l),
                                                                              bth=np.int(use_binned_theta),eh_pk=int(eh_pk))
    if do_pseudo_cl:
        fname_out='mcmc_dat_pcl_{nz}_bl{bl}_bth{bth}_eh{eh_pk}.pkl'.format(nz=zs_bin['n_bins'],bl=np.int(use_binned_l),
                                                                               bth=np.int(use_binned_theta),eh_pk=int(eh_pk))

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
        skylens_args['galaxy_zbins']=zs_bin
        print('read cl / cov from file: ',fname_cl)
    except Exception as err:
        get_cov=True
        print('cl not found. Will compute',fname_cl,err)
        

    if get_cov:
        kappa0=Skylens(**skylens_args)
        print('kappa0 size',get_size_pickle(kappa0))#,kappa0.Win)
        print('kappa0.Ang_PS size',get_size_pickle(kappa0.Ang_PS))
        
        if do_xi:
            print('MCMC getting xi0G')
            xi0G=kappa0.xi_tomo()
            print('MCMC getting xi0 stack')
            xi_cov=client.compute(xi0G['stack']).result()
            cov_inv=np.linalg.inv(xi_cov['cov'].todense())
            data=xi_cov['xi']
            outp=xi_cov
        else:
            print('MCMC getting cl0G')
            cl0G=kappa0.cl_tomo()
            print('MCMC getting stack')
            cl_cov=client.compute(cl0G['stack']).result()
            cov_inv=np.linalg.inv(cl_cov['cov'].todense())
            data=cl_cov['pcl_b']
            outp=cl_cov

        kappa0.gather_data()
        Win=kappa0.Win
        outp['Win']=kappa0.Win
        outp['zs_bin']=kappa0.tracer_utils.z_bins['shear']
        with open(fname_cl,'wb') as of:
            pickle.dump(outp,of)

        del kappa0

    print('Got data and cov')
    if not np.all(np.isfinite(data)):
        x=np.isfinite(data)
        print('data problem',data[~x],np.where(x))

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

    cosmo_fid=kappa0.Ang_PS.PS.cosmo_params

    params_order=['b1_{i}'.format(i=i) for i in np.arange(kappa0.tracer_utils.z_bins['galaxy']['n_bins'])]#,'Ase9','Om']
    ndim=len(params_order)
    priors_mean=np.ones(ndim)
    priors_err=np.ones(ndim)*.1
    if not fix_cosmo:
        params_order+=['Ase9','Om']
        ndim=len(params_order)
        pf=np.array([cosmo_fid[k] for k in ['Ase9','Om']])
        priors_mean=np.append(priors_mean,pf)
        priors_err=np.append(priors_err,[.1,.1])
    
    priors_cov=np.diag(priors_err**2)
    
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

    def get_priors(params):#assume gaussian priors for now
        p=scipy.stats.multivariate_normal.pdf(params,mean=priors_mean,cov=priors_cov)
        return p

    def assign_zparams(zbins={},p_name='',p_value=None):
        pp=p_name.split('_')
        p_n=pp[0]
        bin_indx=np.int(pp[1])
        zbins[bin_indx][p_n]=p_value
        return zbins

    def get_params(params,kappa0,z_bins):
        cosmo_params=copy.deepcopy(cosmo_fid)
#         Ang_PS=copy.deepcopy(kappa0.Ang_PS)
        Ang_PS=kappa0.Ang_PS
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

    def get_model(params,data,cov_inv,kappa0,z_bins,Win,WT,WT_binned,cl_bin_utils,xi_bin_utils):
        prior=get_priors(params)
#         if not np.isfinite(log_prior):
        if prior==0 or np.any(params<=0):
            return -np.inf #np.zeros_like(data)
        log_prior=np.log(prior)
        cosmo_params,z_bins,Ang_PS=get_params(params,kappa0,z_bins)
        
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

    def chi_sq(params,data,cov_inv,kappa0,z_bins,npartitions=100):
        t1=time.time()
        params=np.atleast_2d(params)
#         log_priors=get_priors(params)
        n_params=len(params)
        models={}
        p_bag=dask.bag.from_sequence(params,npartitions=npartitions)
        chisq=p_bag.map(get_model,data,cov_inv,kappa0,z_bins,Win,WT,WT_binned,cl_bin_utils,xi_bin_utils)
        chisq=client.compute(chisq).result()
        print('chisq',time.time()-t1,)
        return chisq

#for student-t, not guaranteed to work at df<3
    class Sample_dist():
        def __init__(self,mean,cov,dist='norm',ndim=2,samples_uniform=None,dist_kwargs={},dist_args=[],df=3,grid='Sobol',N_points=512**2):
            self.__dict__.update(locals())
            self.dist1d=getattr(scipy.stats,dist)
            if dist=='t':
                x=self.samples_uniform==0
                if np.any(x):
                    print('moving 0 in uniform sample to 1.e-20 to avoid -np.inf')
                    self.samples_uniform[x]=1.e-20
            if samples_uniform is None:
                self.grid=getattr(qmcpy,grid)
                self.grid=self.grid(dimension=ndim,randomize=True) 
                self.samples_uniform=self.grid.gen_samples(N_points)
            self.cov_diag()
    #         self.samples_transform()
        def cov_diag(self):
            self.eig_val,self.eig_vec=np.linalg.eig(self.cov)
            self.eig_vec_inv=np.linalg.inv(self.eig_vec)
            self.cov_diag=self.eig_vec.T@self.cov@self.eig_vec
            self.err_diag=np.sqrt(np.diag(self.cov_diag))

        def samples_transform_norm(self):
            samples_trans=np.zeros_like(self.samples_uniform)
            for i in np.arange(self.ndim):
                samples_trans[:,i]=scipy.stats.norm.ppf(self.samples_uniform[:,i],loc=0,scale=self.err_diag[i])#,**self.dist_kwargs)
            samples_trans=samples_trans@self.eig_vec_inv #FIXME: some problem with t dist at df<=2 (covariance is undefined)
            samples_trans+=self.mean
            self.samples_trans=samples_trans

        def samples_transform_t2(self):#appears not to work well at low df with non-diag covariance, though pdf test is passed
                                        #problem is small errors in ppf function, primarily due to numerical errors in numpy.
                                        #mpmath helps, but is very slow.
            """
                    pv=np.linspace(-10**4,10**4,100)
                    qv=scipy.stats.t.cdf(pv,df=30,loc=0,scale=1)
                    pv2=scipy.stats.t.ppf(qv,df=30,loc=0,scale=1)
                    np.all(np.isclose(pv,pv2)),np.all(pv==pv2)
                    np.all(np.isclose(np.tan(np.arctan(pv)),pv)) #this passes
                    np.all(np.tan(np.arctan(pv))==pv)#this fails
            """
            samples_trans=np.zeros_like(self.samples_uniform)
            for i in np.arange(self.ndim):
                samples_trans[:,i]=student_t_ppf(self.samples_uniform[:,i],loc=0,scale=self.err_diag[i],df=self.df)
            self.prob_t2=scipy.stats.multivariate_t.pdf(samples_trans,loc=self.mean*0,df=self.df,shape=self.cov_diag)

            samples_trans=samples_trans@self.eig_vec_inv #FIXME: some problem with t dist at df<=2 (covariance is undefined)
            samples_trans+=self.mean

            self.samples_trans_t2=samples_trans
            self.prob_t22=scipy.stats.multivariate_t.pdf(samples_trans,loc=self.mean,df=self.df,shape=self.cov)
            self.prob_norm_t2=scipy.stats.multivariate_normal.pdf(samples_trans,mean=self.mean,cov=self.cov)
            print('samples_transform_t2, pdf test:',np.all(np.isclose(self.prob_t2,self.prob_t22)), np.all(self.prob_t2==self.prob_t22))

        def samples_transform_t(self):
            self.samples_transform_norm()

            self.chi2=scipy.stats.chi2.ppf(self.samples_uniform[:,0],df=self.df,loc=0,scale=1)
            self.chi2/=self.df

            self.samples_trans-=self.mean
            self.samples_trans/=np.random.permutation(np.sqrt(self.chi2[:,None]))
            self.samples_trans+=self.mean

            self.prob=scipy.stats.multivariate_t.pdf(self.samples_trans,loc=self.mean,df=self.df,shape=self.cov)
            self.prob_norm=scipy.stats.multivariate_normal.pdf(self.samples_trans,mean=self.mean,cov=self.cov)

    def get_samples(sample_dist='t',N_points=512**2,grid='Sobol'):
        ndim=len(params_order)
        print('getting parameters: ',ndim,N_points,sample_dist)
        p0=priors_mean
        cov=priors_cov
        sd=Sample_dist(grid=grid,N_points=N_points,dist=sample_dist,mean=priors_mean,cov=cov,ndim=ndim)
        if sample_dist=='t':
            sd.samples_transform_t()
        if sample_dist=='norm':
            sd.samples_transform_norm()
        return sd
            
    def sample_params(fname=''):
        Sample_dist=get_samples(N_points=N_points,grid=grid,sample_dist=sample_dist)
        print('got samples ',Sample_dist.samples_trans.shape)
        chisq_wts=chi_sq(Sample_dist.samples_trans,data,cov_inv,kappa0,zs_bin1)
        outp={}
        outp['chain']=Sample_dist.samples_trans
        outp['p0']=Sample_dist.prob
        outp['params']=params_order
        outp['ln_prob']=chisq_wts
        return outp

    if not do_xi and not use_binned_l:
        print('unbinned cl will not work in this example because covariance is always binned')
        client.shutdown()
        sys.exit(0)
    outp=sample_params()
    print('calcs done')

    kappa0=client.gather(kappa0)
    outp['l0']=kappa0.l0
    outp['l_bins']=kappa0.l_bins
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
        fname_out='IS{np}_xi_{nz}_bl{bl}_bth{bth}_camb{fc}.pkl'.format(nz=zs_bin1['n_bins'],bl=np.int(use_binned_l),np=int(N_points),
                                                                              bth=np.int(use_binned_theta),fc=int(fix_cosmo))
    if do_pseudo_cl:
        fname_out='IS{np}_pcl_{nz}_bl{bl}_bth{bth}_camb{fc}.pkl'.format(nz=zs_bin1['n_bins'],bl=np.int(use_binned_l),np=int(N_points),
                                                                               bth=np.int(use_binned_theta),fc=int(fix_cosmo))

    fname_out=file_home+fname_out
    with open(fname_out, 'wb') as f:
        pickle.dump(outp,f)
    print('file written: ',fname_out)
    client.shutdown()
    sys.exit(0)
