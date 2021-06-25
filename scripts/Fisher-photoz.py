#To do:
# 1. Add cmb lensing
import numpy as np
import sys
import gc
import time
from resource import getrusage, RUSAGE_SELF
from skylens.parse_input import parse_dict
from Fisher_photoz_functions import * #get_cl_ells, d2r,corrs
import psutil

import argparse

debug=True
# if debug:
import faulthandler; faulthandler.enable()
    #in case getting some weird seg fault, run as python -Xfaulthandler run_sim_jk.py
    # problem is likely to be in some package

if __name__=='__main__':
    test=True

    Fmost=False

    eh_pk=False

    parser = argparse.ArgumentParser()
    parser.add_argument("--dask_dir", "-Dd", help="dask log directory")
    parser.add_argument("--scheduler", "-s", help="Scheduler file")

    parser.add_argument("--desi", "-desi",type=int, help="use DESI samples")
    parser.add_argument("--train", "-t", type=int, help="use training samples")
    parser.add_argument("--train_spectra", "-ts", type=int, help="total number of training spectra")
    parser.add_argument("--train_area", "-ta", type=int, help="area of training spectra, deg^2")

    args = parser.parse_args()

    dask_dir=args.dask_dir
    Scheduler_file=args.scheduler

    Desi=True if args.desi is None else np.bool(args.desi)
    train_sample=True if args.train is None else np.bool(args.train)

    n_train_spectra=np.int32(1e6) if args.train_spectra is None else np.int32(args.train_spectra)
    area_train=150 if args.train_area is None else np.int32(args.train_area) #deg^2

    file_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/tests/fisher/'

    #redshift bins
    z_min=0.1
    z_max=3.5 #max photoz
    z_max_galaxy=1.5 #max photoz
    zmin=z_min
    zmax=z_max
    area=15000 #same for both DESI and LSST
    area_overlap=5000 #overlap area b/w DESI and LSST

    area_overlap=area_overlap*1./area

    shear_n_zbins=5
    galaxy_n_zbins=5
    galaxyD_n_zbins={} #galaxyD==DESI
    galaxyD_n_zbins_tot=0
    if Desi:
        galaxyD_n_zbins['elg']=5
        galaxyD_n_zbins['lrg']=4
        galaxyD_n_zbins['BG']=2
        galaxyD_n_zbins['qso']=6
        galaxyD_n_zbins_tot=np.int(np.sum(list(galaxyD_n_zbins.values())))

    nz_shear=26 #galaxy/arcmin^-2
    nz_galaxy=3

    train_n_zbins=np.int((z_max-z_min)/.1)

    train_sample_missed=1
    nz_shear_missed=6  #arcmin^-2

    nz_shear_train=n_train_spectra/area_train/3600 #arcmin^-2
    if not train_sample or n_train_spectra*area_train==0:
        if area_train>0 or n_train_spectra>0 or train_sample:
            raise Exception('miss matched args for training sample:',train_sample, area_train,n_train_spectra)
    if area_train==0 or n_train_spectra==0 or not train_sample:
        area_train=0
        n_train_spectra=0
        nz_shear_train=0
        train_n_zbins=0

    if train_sample_missed==0:
        nz_shear_missed=0  #arcmin^-2

    nz_shear-=nz_shear_missed

    sigma_gamma=0.26

    n_zPS=100
    z_max_PS=5
    z_true_max=z_max_PS
    z_PS=np.logspace(np.log10(z_min),np.log10(z_max_PS),50)
    z_PS2=np.linspace(z_min,z_max_PS,np.int((z_max_PS-z_min)/.05 ))
    z_PS=np.sort(np.unique(np.around(np.append(z_PS,z_PS2),decimals=3)))
    n_zPS=len(z_PS)
    nz_PS=n_zPS

    n_zs_shear=np.int((z_true_max-z_min)/.1)  #this will be nz(or pz) params as well
    n_zs_galaxy=np.int((z_true_max-z_min)/.1)


    #Cl args
    lmax_cl=1000   #[1000,2000,5000]
    lmin_cl=20
    Nl_bins=12
    bin_cl=True #False
    use_binned_l=True
    do_pseudo_cl=True
    l0,l_bins,l=get_cl_ells(lmax_cl=lmax_cl,Nl_bins=Nl_bins,lmin_cl=lmin_cl,bin_cl=bin_cl)
    lb=l*1
    if bin_cl:
        lb=0.5*(l_bins[1:]+l_bins[:-1])
    print('n ell bins: ',lb.shape)

    #xi args
    do_xi=False

    #window
    use_window=do_pseudo_cl
    unit_window=False
    nside=1024 #32
    window_lmax=nside #30

    print('doing nside',nside,window_lmax,use_binned_l)
    wig_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/'
    wig_home=wig_home+'temp/'
    wigner_files={}
    wigner_files[0]= wig_home+'/dask_wig3j_l2200_w4400_0_reorder.zarr'
    wigner_files[2]= wig_home+'/dask_wig3j_l2200_w4400_2_reorder.zarr'

    store_win=True
    save_win=True

    #covariance and area
    SSV_cov=False #[False,True]

    do_cov=True
    sparse_cov=True
    tidal_SSV_cov=SSV_cov
    Tri_cov=False
    do_sample_variance=True

    f_sky=area*d2r**2/4/np.pi
    f_sky_train=area_train*d2r**2/4/np.pi

    #baryons & cosmology
    bary_nQ=0   #[0,2,1,3,5]
    if bary_nQ>0:
        pk_func='baryon_pk'
    else:
        pk_func='camb_pk_too_many_z'
    if eh_pk:
        pk_func='eh_pk'
        print('warning, power spectra in eh_pk')
    cosmo_parameters=np.atleast_1d(['As','Om','w','wa'])
    pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':2000,'scenario':'dmo','pk_func':pk_func}# 'pk_func':'camb_pk_too_many_z'} #baryon_pk
    power_spectra_kwargs={'pk_params':pk_params}

    corrs=[corr_ggl,corr_gg,corr_ll]

    if test:
        lmax_cl=200
        l0,l_bins,l=get_cl_ells(lmax_cl=lmax_cl,Nl_bins=Nl_bins,lmin_cl=lmin_cl,bin_cl=bin_cl)
        shear_n_zbins=2
        galaxy_n_zbins=2
        train_n_zbins=shear_n_zbins
        nside=32
        window_lmax=nside
        if Desi:
            galaxyD_n_zbins['elg']=1
            galaxyD_n_zbins['lrg']=0
            galaxyD_n_zbins['BG']=0
            galaxyD_n_zbins['qso']=0
            galaxyD_n_zbins_tot=np.int(np.sum(list(galaxyD_n_zbins.values())))

    WIN={'full':None,'lsst':None}

    proc = psutil.Process()
    print(format_bytes(proc.memory_info().rss))

    fname_out='{ns}_{nsm}_{nl}_{nlD}_nlb{nlb}_lmax{lmax}_z{zmin}-{zmax}_zlmax{zlmax}_bary{bary_nQ}_AT{at}_NT{NT}.pkl'
    if bin_cl and use_binned_l:
        fname_out='binnedL_'+fname_out
    elif bin_cl and not use_binned_l:
        fname_out='binned_'+fname_out
    if use_window and not unit_window:
        fname_out='win'+str(nside)+'_'+fname_out
    elif use_window and unit_window:
        fname_out='unit_win'+str(nside)+'_'+fname_out
    if SSV_cov:
        fname_out='SSV_'+fname_out
    if eh_pk:
        fname_out='eh_pk_'+fname_out

    ncpu=5 #multiprocessing.cpu_count()-1
    LC,scheduler_info=start_client(Scheduler_file=Scheduler_file,local_directory=dask_dir,ncpu=None,n_workers=ncpu,threads_per_worker=1,
                                      memory_limit='120gb',dashboard_address=8801,processes=True)
    client=client_get(scheduler_info=scheduler_info)
    print('client: ',client)#,dask_dir,scheduler_info)

    fname_out=fname_out.format(ns=shear_n_zbins,nsm=train_sample_missed,nl=galaxy_n_zbins,nlD=galaxyD_n_zbins_tot,
                               nlb=Nl_bins,lmax=lmax_cl,bary_nQ=bary_nQ,
                               zmin=z_min,zmax=z_max,zlmax=z_max_galaxy,at=area_train,NT=n_train_spectra)
    fname_win=file_home+'/win_'+fname_out

    Skylens_kwargs=parse_dict(locals())

    from Fisher_photoz_functions import * #to prevent some skylens galaxies getting assigned to Skylens_kwargs

    # fname='temp/win_D_{ns}{nl}{nlD}.pkl'.format(ns=nbins,nl=galaxy_n_zbins,nlD=n_lensD_bins)
    if use_window and store_win:
        fname_out='win_'+fname_out
        try:
#             crash
            with open(fname_win,'rb') as of:
                WIN=pickle.load(of)
            print('window read')
        except Exception as err:
            save_win=True
            print('window not found. Will compute',err)

    clean_tracer_window=False
    fname_cl=file_home+'/cl_cov_'+fname_out

    try:
#         crash
        with open(fname_cl,'rb') as of:
            cl_all=pickle.load(of)
        cl_L=cl_all['cl_L']
        #cl_L_lsst=cl_all['cl_L_lsst']
        z_bins_kwargs=cl_all['z_bins']
        #z_bins_lsst_kwargs=cl_all['z_bins_lsst']
        print('read cl / cov from file: ',fname_cl)
    except Exception as err:
        print('cl not found. Will compute',fname_cl,err)

        kappa_class,z_bins_kwargs=init_fish(z_min=z_min,z_max=z_max,SSV=SSV_cov,nz_shear=nz_shear,f_sky=f_sky,nside=nside,
                                            nz_galaxy=nz_galaxy,n_zs_shear=n_zs_shear,n_zs_galaxy=n_zs_galaxy,
                                            corrs=corrs,unit_window=unit_window,nz_shear_missed=nz_shear_missed,
                                            nz_shear_train=nz_shear_train,z_max_galaxy=z_max_galaxy,use_window=use_window,
                                            z_true_max=z_true_max,area_train=area_train,train_sample_missed=train_sample_missed,
                                            train_n_zbins=train_n_zbins,area_overlap=area_overlap,
                                            shear_n_zbins=shear_n_zbins,galaxy_n_zbins=galaxy_n_zbins,galaxyD_n_zbins=galaxyD_n_zbins,
                                              Win=WIN['full'],store_win=store_win,pk_params=pk_params,Skylens_kwargs=Skylens_kwargs)

        cl0G=kappa_class.cl_tomo(corrs=corrs,stack_corr_indxs=z_bins_kwargs['corr_indxs'])

        proc = psutil.Process()
        print('graphs done, memory: ',format_bytes(proc.memory_info().rss))

        cl_L=None
#         cl_L_lsst=None

        if cl_L is None:
            #get_ipython().run_line_magic('time', "
            cl_L=cl0G['stack'].compute()

#         if cl_L_lsst is None:
#             #get_ipython().run_line_magic('time', "
#             cl_L_lsst=cl0G_lsst['stack'].compute()

        cl_all={'cl_L':cl_L,'z_bins':z_bins_kwargs}#,'cl_L_lsst':cl_L_lsst,'z_bins_lsst':z_bins_lsst_kwargs}


        with open(fname_cl,'wb') as of:
            pickle.dump(cl_all,of)

        if save_win:
            win_all={'full':gather_dict(kappa_class.Win,scheduler_info=kappa_class.scheduler_info)}#,'lsst':client.gather(kappa_class_lsst.Win)}
            with open(fname_win,'wb') as of:
                pickle.dump(win_all,of)
            WIN=win_all
        del kappa_class  #,kappa_class_lsst

    if sparse_cov:
        cov_p_inv_test1=np.linalg.inv(cl_L['cov'].todense())
#         cov_p_inv_test2=np.linalg.inv(cl_L_lsst['cov'].todense())
        del cov_p_inv_test1,#cov_p_inv_test2
    proc = psutil.Process()
    print('cl, cov done. memory:',format_bytes(proc.memory_info().rss),
          "Peak memory (gb):",
          int(getrusage(RUSAGE_SELF).ru_maxrss/1024./1024.))

    WIN['full']['cov']=None
#     WIN['lsst']['cov']=None

    Skylens_kwargs['do_cov']=False
    Skylens_kwargs['clean_tracer_window']=True
    Skylens_kwargs['Win']=WIN['full']

    kappa_class,z_bins_kwargs=init_fish(z_min=z_min,z_max=z_max,SSV=SSV_cov,nz_shear=nz_shear,f_sky=f_sky,nside=nside,nz_galaxy=nz_galaxy,
                                        unit_window=unit_window,z_true_max=z_true_max,area_train=area_train,train_sample_missed=train_sample_missed,
                                        shear_n_zbins=shear_n_zbins,galaxy_n_zbins=galaxy_n_zbins,galaxyD_n_zbins=galaxyD_n_zbins,corrs=corrs,
                                        nz_shear_missed=nz_shear_missed,nz_shear_train=nz_shear_train,use_window=use_window,
                                        store_win=store_win,do_cov=do_cov,z_bins_kwargs=z_bins_kwargs,
                                        n_zs_shear=n_zs_shear,n_zs_galaxy=n_zs_galaxy,z_max_galaxy=z_max_galaxy,pk_params=pk_params,
                                        Skylens_kwargs=Skylens_kwargs)#reset after cl,cov calcs

#     kappa_class_lsst,z_bins_lsst_kwargs=init_fish(z_min=z_min,z_max=z_max,SSV=SSV_cov,nz_shear=nz_shear,f_sky=f_sky,nside=nside,nz_galaxy=nz_galaxy,
#                                                   use_window=use_window,pk_params=pk_params,
#                                                   unit_window=unit_window,z_true_max=z_true_max,area_train=area_train,train_sample_missed=train_sample_missed,
#                                                 shear_n_zbins=shear_n_zbins,galaxy_n_zbins=galaxy_n_zbins,galaxyD_n_zbins=None,corrs=corrs,
#                                                   nz_shear_missed=nz_shear_missed,nz_shear_train=nz_shear_train,z_max_galaxy=z_max_galaxy,
#                                               Win=WIN['lsst'],store_win=store_win,do_cov=do_cov,n_zs_shear=n_zs_shear,n_zs_galaxy=n_zs_galaxy,
#                                               z_bins_kwargs=z_bins_lsst_kwargs,Skylens_kwargs=Skylens_kwargs)#reset after cl,cov calcs

    # sigma_68=-0.1*(1+z) + 0.12*(1+z)**2 #https://arxiv.org/pdf/1708.01532.pdf
    priors={}

    priors['Ase9']=np.inf
    priors['Om']=np.inf
    priors['w']=np.inf
    priors['wa']=np.inf

    priors['pz_B_s']=0.001 #bias =B*(1+z)
    priors['pz_B_sm']=priors['pz_B_s']*10
    priors['pz_B_l']=0.0001 #bias =B*(1+z)
    for i in np.arange(10): #photo-z bias
        priors['pz_b_s_'+str(i)]=0.001
        priors['pz_b_sm_'+str(i)]=0.001*10

    for i in np.arange(10): #shear multiplicative bias
        priors['s_m_s_'+str(i)]=0.001
        priors['s_m_sm_'+str(i)]=0.001

    for i in np.arange(10): #IA bias
        priors['AI_s_'+str(i)]=1
        priors['AI_sm_'+str(i)]=1

    for i in np.arange(10): #photo-z bias
        priors['pz_b_l_'+str(i)]=0.0001

    pp_s={}
    for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins']): #photo-z bias
        pp_s[i]=sigma_photoz(z_bins_kwargs['shear_zbins'][i])
        for j in np.arange(n_zs_shear): #photo-z bias
    #         priors['nz_s_'+str(i)+'_'+str(j)]=0.01
            priors['nz_s_'+str(i)+'_'+str(j)]=pp_s[i][j]

    for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins']): #photo-z bias
        pp_s[i]=sigma_photoz(z_bins_kwargs['shear_zbins_missed'][i]) #FIXME
        for j in np.arange(n_zs_shear): #photo-z bias
    #         priors['nz_s_'+str(i)+'_'+str(j)]=0.01
            priors['nz_sm_'+str(i)+'_'+str(j)]=pp_s[i][j]


    pp_l={}
    for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins']): #photo-z bias
        pp_l[i]=sigma_photoz(z_bins_kwargs['galaxy_zbins'][i])/5.
        for j in np.arange(n_zs_galaxy): #photo-z bias
    #         priors['nz_l_'+str(i)+'_'+str(j)]=0.01
            priors['nz_l_'+str(i)+'_'+str(j)]=pp_l[i][j]


    for i in np.arange(50):#magnification bias
        priors['mag_s_'+str(i)]=0.05
        priors['mag_sm_'+str(i)]=0.05
        priors['mag_l_'+str(i)]=0.05
        priors['mag_lD_'+str(i)]=0.05

    for i in np.arange(50):#galaxy bias
        priors['g_b_s_1_'+str(i)]=1
        priors['g_b_sm_1_'+str(i)]=1
        for k in np.arange(n_zs_shear):
            priors['g_bz_s_'+str(k)+'_1_'+str(i)]=1
            priors['g_bz_sm_'+str(k)+'_1_'+str(i)]=1
    #         print(priors.keys())

    for i in np.arange(50): #galaxy bias, b2
        priors['g_b_s_2_'+str(i)]=1
        priors['g_b_sm_2_'+str(i)]=1

    for i in np.arange(50):#galaxy bias
        priors['g_b_l_1_'+str(i)]=1
        priors['g_b_lD_1_'+str(i)]=1
        for k in np.arange(n_zs_galaxy):
            priors['g_bz_l_'+str(k)+'_1_'+str(i)]=1
            priors['g_bz_lD_'+str(k)+'_1_'+str(i)]=1
    for i in np.arange(50): #galaxy bias, b2
        priors['g_b_l_2_'+str(i)]=1
        priors['g_b_lD_2_'+str(i)]=1

    for i in np.arange(10): #baryon PCA
        priors['Q'+str(i)]=100

    if nz_shear_train>0 and area_train>0:
        priors['nz_ana']=photoz_prior(kappa_class=kappa_class,Skylens_kwargs0=Skylens_kwargs,z_bins_kwargs=z_bins_kwargs,key_label='nz_s_')
        for k in priors.keys():
            if k=='nz_ana' or 'nz_s_' in k:
                continue
            if priors['nz_ana'].get(k) is not None:
                continue
            priors['nz_ana'][k]=priors[k]

    cosmo_params=cosmo_parameters

    fishes={}
    fishes['priors']=priors

    z_bins_kwargs['use_window']=use_window

    # if use_window:
    #     kappa_class.Win.store_win=True

    cosmo_fid=kappa_class.Ang_PS.PS.cosmo_params.copy()

    cosmo_params=np.atleast_1d(['Ase9','Om','w','wa'])

    fishes['f_C0']=fisher_calc(cosmo_params=cosmo_params #np.atleast_1d(['Ase9']) #,'Om','Omb','Omd','OmR'])
                               ,z_params=[],
                               galaxy_params=[],kappa_class=kappa_class,scheduler_info=scheduler_info,
                        clS=cl_L,z_bins_kwargs=z_bins_kwargs,priors=priors,baryon_params=[])

    print('fisher cosmo only done.')

    baryon_params=['Q{i}'.format(i=i) for i in np.arange(bary_nQ)]
    pz_params=['pz_b_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    pz_params+=['pz_b_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    # pz_params+=['pz_b_l_{j}'.format(j=i) for i in np.arange(2)]
    galaxy_params=[]#['g_b_s_1{j}'.format(j=i) for i in np.arange(kappa_class.z_bins['shear']['n_bins'])]
    # galaxy_params=['g_b_l_1{j}'.format(j=i) for i in np.arange(4)]



    # fishes['f0']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,kappa_class=kappa_class,scheduler_info=scheduler_info,
    #                         clS=cl_L,z_bins_kwargs=z_bins_kwargs,priors=priors,baryon_params=baryon_params)

    pz_params=['nz_s_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
    pz_params+=['nz_sm_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins']) for k in np.arange(n_zs_shear)]
    # pz_params+=['nz_l_{j}_8'.format(j=i) for i in np.arange(2)]
    # galaxy_params=['g_b_s_1_{j}'.format(j=i) for i in np.arange(kappa_class.z_bins['shear']['n_bins'])]
    # galaxy_params=['g_b_l_1{j}'.format(j=i) for i in np.arange(4)]
    galaxy_params=[]
    # print(pz_params)

    # fishes['f_nz0']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,scheduler_info=scheduler_info,
    #                 kappa_class=kappa_class,clS=cl_L,z_bins_kwargs=z_bins_kwargs,priors=priors,baryon_params=baryon_params)

    gc.collect()
    print('Priors done')
    print('zs bins:',z_bins_kwargs['shear_zbins']['n_bins'])
    pz_params=['pz_b_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    pz_params+=['pz_b_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    pz_params+=['pz_b_l_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]
    galaxy_params=['g_b_s_1_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    galaxy_params+=['g_b_sm_1_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    galaxy_params+=['g_b_l_1_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]

    galaxy_params+=['mag_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    galaxy_params+=['mag_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    galaxy_params+=['mag_l_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]

    shear_params=['s_m_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    shear_params+=['s_m_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    shear_params+=['AI_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    shear_params+=['AI_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    if Desi:
        galaxy_params+=['g_b_lD_1_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxyD_zbins']['n_bins'])]
        galaxy_params+=['mag_lD_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxyD_zbins']['n_bins'])]


    fishes['f_all']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,scheduler_info=scheduler_info,
                        kappa_class=kappa_class,clS=cl_L,z_bins_kwargs=z_bins_kwargs,priors=priors,baryon_params=baryon_params)

    pz_params=['nz_s_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
    pz_params+=['nz_sm_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins']) for k in np.arange(n_zs_shear)]
    pz_params+=['nz_l_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins']) for k in np.arange(n_zs_galaxy)]

    fishes['f_nz_all']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,scheduler_info=scheduler_info,
                        kappa_class=kappa_class,clS=cl_L,z_bins_kwargs=z_bins_kwargs,priors=priors,baryon_params=baryon_params)

    pz_params=['nz_s_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
    pz_params+=['nz_sm_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins']) for k in np.arange(n_zs_shear)]
    pz_params+=['nz_l_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins']) for k in np.arange(n_zs_galaxy)]
    galaxy_params=['g_bz_s_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
    galaxy_params+=['g_bz_sm_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])for k in np.arange(n_zs_shear)]
    galaxy_params+=['g_bz_l_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])for k in np.arange(n_zs_galaxy)]
    galaxy_params+=['mag_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
    galaxy_params+=['mag_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
    galaxy_params+=['mag_l_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]
    if Desi:
        galaxy_params+=['g_bz_lD_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_kwargs['galaxyD_zbins']['n_bins'])for k in np.arange(n_zs_galaxy)]
        galaxy_params+=['mag_lD_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxyD_zbins']['n_bins'])]

    fishes['f_nz_bz_all']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,shear_params=shear_params,scheduler_info=scheduler_info,
                        kappa_class=kappa_class,clS=cl_L,z_bins_kwargs=z_bins_kwargs,priors=priors,baryon_params=baryon_params)



#     pz_params=['pz_b_s_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['shear_zbins']['n_bins'])]
#     pz_params+=['pz_b_sm_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['shear_zbins_missed']['n_bins'])]
#     pz_params+=['pz_b_l_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['galaxy_zbins']['n_bins'])]
#     galaxy_params=['g_b_s_1_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['shear_zbins']['n_bins'])]
#     galaxy_params+=['g_b_sm_1_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['shear_zbins_missed']['n_bins'])]
#     galaxy_params+=['g_b_l_1_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['galaxy_zbins']['n_bins'])]
#     galaxy_params+=['mag_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
#     galaxy_params+=['mag_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
#     galaxy_params+=['mag_l_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]

#     fishes['f_all_lsst']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,kappa_class=kappa_class_lsst,scheduler_info=scheduler_info,
#                                     shear_params=shear_params,clS=cl_L_lsst,z_bins_kwargs=z_bins_lsst_kwargs,priors=priors,baryon_params=baryon_params)

#     pz_params=['nz_s_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
#     pz_params+=['nz_sm_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['shear_zbins_missed']['n_bins']) for k in np.arange(n_zs_shear)]
#     pz_params+=['nz_l_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['galaxy_zbins']['n_bins']) for k in np.arange(n_zs_galaxy)]
#     galaxy_params=['g_b_s_1_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['shear_zbins']['n_bins'])]
#     galaxy_params+=['g_b_sm_1_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['shear_zbins_missed']['n_bins'])]
#     galaxy_params+=['g_b_l_1_{j}'.format(j=i) for i in np.arange(z_bins_lsst_kwargs['galaxy_zbins']['n_bins'])]
#     galaxy_params+=['mag_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
#     galaxy_params+=['mag_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
#     galaxy_params+=['mag_l_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]

#     fishes['f_nz_all_lsst']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,kappa_class=kappa_class_lsst,scheduler_info=scheduler_info,
#                                         clS=cl_L_lsst,z_bins_kwargs=z_bins_lsst_kwargs,priors=priors,baryon_params=baryon_params)

    # f_nz_all_lsst=fish_apply_priors(fish=f_nz_all_lsst,priors=priors)


#     pz_params=['nz_s_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
#     pz_params+=['nz_sm_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['shear_zbins_missed']['n_bins']) for k in np.arange(n_zs_shear)]
#     pz_params+=['nz_l_{j}_{k}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['galaxy_zbins']['n_bins']) for k in np.arange(n_zs_galaxy)]

#     galaxy_params=['g_bz_s_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['shear_zbins']['n_bins']) for k in np.arange(n_zs_shear)]
#     galaxy_params+=['g_bz_sm_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['shear_zbins_missed']['n_bins'])for k in np.arange(n_zs_shear)]
#     galaxy_params+=['g_bz_l_{k}_1_{j}'.format(j=i,k=k) for i in np.arange(z_bins_lsst_kwargs['galaxy_zbins']['n_bins'])for k in np.arange(n_zs_galaxy)]
#     galaxy_params+=['mag_s_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins']['n_bins'])]
#     galaxy_params+=['mag_sm_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['shear_zbins_missed']['n_bins'])]
#     galaxy_params+=['mag_l_{j}'.format(j=i) for i in np.arange(z_bins_kwargs['galaxy_zbins']['n_bins'])]


#     fishes['f_nz_bz_all_lsst']=fisher_calc(cosmo_params=cosmo_params,z_params=pz_params,galaxy_params=galaxy_params,
#                                            kappa_class=kappa_class_lsst,scheduler_info=scheduler_info,
#                                            clS=cl_L_lsst,z_bins_kwargs=z_bins_lsst_kwargs,priors=priors,baryon_params=baryon_params)

    fishes['cov_file']=fname_cl

    fishes['cosmo_fid']=cosmo_fid

    # fishes['z_bins_lsst_kwargs']=z_bins_lsst_kwargs
    # fishes['z_bins_kwargs']=z_bins_kwargs


    fname_fish=file_home+'/fisher_'+fname_out

    with open(fname_fish,'wb') as of:
        pickle.dump(fishes,of)

    client.shutdown()
    LC.close()

    # import plot_fisher_tool
    # reload(plot_fisher_tool)
    # from plot_fisher_tool import *


    # fish1 = fisher_tool(Fishers={0:f_nz_all_lsst['cov_p_inv'],1:f_nz_all['cov_p_inv']},
    #                     pars={0:f_nz_all_lsst['params'],1:f_nz_all['params']},
    #                     par_cen=cosmo_fid#{'Om':0.28374511,'Ase9':0.80351633}
    #                     ,fisher_titles={0:'LSST',1:'LSST+DESI'})
    # f=fish1.plot_fish(pars=['Ase9','Om','w','wa'])


    # fish1 = fisher_tool(Fishers={0:f_nz_all_lsst['prior']['cov_p_inv'],1:f_nz_all['prior']['cov_p_inv']},
    #                     pars={0:f_nz_all_lsst['params'],1:f_nz_all['params']},
    #                     par_cen=cosmo_fid#{'Om':0.28374511,'Ase9':0.80351633}
    #                     ,fisher_titles={0:'LSST',1:'LSST+DESI'})
    # f=fish1.plot_fish(pars=['Ase9','Om','w','wa'])


    # fish1 = fisher_tool(Fishers={0:f_nz_all_lsst['cov_p_inv'],1:f_nz_all['cov_p_inv'],
    #                             2:f_nz_all_lsst['prior']['cov_p_inv'],3:f_nz_all['prior']['cov_p_inv']},
    #                     pars={0:f_nz_all_lsst['params'],1:f_nz_all['params'],
    #                          2:f_nz_all_lsst['params'],3:f_nz_all['params']},
    #                     par_cen=cosmo_fid#{'Om':0.28374511,'Ase9':0.80351633}
    #                     ,fisher_titles={0:'LSST',1:'LSST+DESI',
    #                                    2:'LSST+p(z) prior',3:'LSST+DESI+p(z) prior'},
    #                    print_par_error=False)
    # f=fish1.plot_fish(pars=['Ase9','Om','w','wa'])
