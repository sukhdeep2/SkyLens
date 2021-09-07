import sys
import pickle
import camb
from importlib import reload
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import default_cosmology
from astropy import units
import astropy
import gc
import time

import psutil,os
from distributed.utils import format_bytes

from skylens import *
from skylens.utils import *
from PCA_shear import *
from skylens.survey_utils import *
import argparse

from dask_mpi import initialize as dask_initialize
from distributed import Client

def cosmo_w0_wa(cosmo=None,w0=-1,wa=0):
    attrs=['H0','Om0', 'Ode0','Tcmb0', 'Neff', 'm_nu', 'Ob0']
    args={}
    args['w0']=w0
    args['wa']=wa
    for a in attrs:
        args[a]=getattr(cosmo,a)
    cosmo_w=astropy.cosmology.w0waCDM(**args)
    return cosmo_w
cosmo_w0_wa(cosmo=cosmo)

def cosmo_h_set(cosmo=None,cosmo_params={}):
    cosmo2=cosmo.clone(H0=cosmo_params['h']*100,
                    Om0=cosmo_params['Om'],
                   Ob0=cosmo_params['Omb'],
#                        Odm0=cosmo_params['Omd'],
#                        Ok0=cosmo_params['Omk'],
#                        w=cosmo_params['w'],
                   m_nu=[0,0,cosmo_params['mnu']]*units.eV
                )
    if cosmo_params['wa']!=0:
        cosmo2=cosmo_w0_wa(cosmo=cosmo2,w0=cosmo_params['w'],wa=cosmo_params['wa'])
    return cosmo2

def get_x_var(x0=None,dx_max=0.01,do_log=False,Nx=2):
    Dx=np.linspace((1-dx_max),(1+dx_max),Nx)
    if do_log:
        x0=np.absolute(x0)
        x_vars=x0**Dx
        if x0==1:
            x_vars=(2.**Dx)/2. # 1**x=1
        if x0==0:
            x_vars=np.logspace(-3,-2,Nx)
        x_vars*=np.sign(cosmo_fid[p])
        x_grad=np.gradient(np.log(x_vars))
    else:
        x_vars=x0*Dx #np.linspace(x0*(1-dx_max),x0*(1+dx_max),Nx)
        if x0==0:
            x_vars=Dx-1
        x_grad=np.gradient(x_vars)
    return x_vars,x_grad


def fish_cosmo_model(p='As',Nx=2,dx_max=0.01,do_log=False,kappa_class=None,do_cov=False,z_bins_kwargs={},Win=None,Ang_PS=None):
    t0=time.time()
    x0=cosmo_fid[p]
    shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs)
    z_bins={'shear':shear_zbins_comb,'galaxy':galaxy_zbins_comb}
    models={}
    covs={}
    x_vars,x_grad=get_x_var(x0=x0,dx_max=dx_max,do_log=do_log,Nx=Nx) #FIXME: NX
    print(p,x_vars)
    Ang_PS=copy.deepcopy(Ang_PS)
    for i in np.arange(Nx):
        cosmo_t=copy.deepcopy(cosmo_fid)#.copy()

        cosmo_t[p]=x_vars[i]

        if p=='s8':
            s80=cosmo_fid['s8']
            cosmo_t['As']*=(cosmo_t['s8']/s80)**2
            kappa_class.Ang_PS.PS.get_pk(z=[0],cosmo_params=cosmo_t,return_s8=True)
            kappa_class.Ang_PS.PS.pk=None
            x_vars[i]=kappa_class.Ang_PS.PS.s8
#                 print(x_vars[p][i],s80,cosmo_t['s8'])
#         cosmo_h2=cosmo_h_set(cosmo=cosmo,cosmo_params=cosmo_t)
        t2=time.time()
        cl0G=kappa_class.tomo_short(cosmo_params=cosmo_t,z_bins=z_bins,Ang_PS=Ang_PS,Win=Win,stack_corr_indxs=z_bins_kwargs['corr_indxs']) #cosmo_h=cosmo_h2,
        dt=time.time()-t2
        print(p,'one calc done',dt,cl0G.shape)
        models[i]=cl0G
#         cl_t=cl0G['stack'].compute()
#         models[i]=cl_t['pcl_b']
#         covs[i]=cl_t['cov']
        if not np.all(np.isfinite(models[i])):
            print('nan crash',p,x_vars,cosmo_t,)
            print(models[i],models[i][~np.isfinite(models[i])],np.where(~np.isfinite(models[i])))
            crash
#         kappa_class.Ang_PS.reset()
    print(p,'done',time.time()-t0)
    return models,covs,x_vars,x_grad

def pz_update(z_bins={},bin_indx=None,z_indx=None,pz_new=None):
    z_bins[bin_indx]['pz'][z_indx]=pz_new
    z_bins=zbin_pz_norm(shear_zbins=z_bins,bin_indx=bin_indx,zs=z_bins[bin_indx]['z'],p_zs=z_bins[bin_indx]['pz'],ns=z_bins[bin_indx]['ns'])
    return z_bins


def set_zbin_bias0(shear_zbins={},bias=0):
    bias0=np.copy(shear_zbins['zp_bias'])
    bias0=bias*(1+shear_zbins['zp'])
    return bias0

def set_zbin_bias(shear_zbins={},bias=0,bin_id=0):
#     bias0=np.copy(shear_zbins['zp_bias'])
#     idx=np.digitize(shear_zbins['zp'],shear_zbins['z_bins'])-1
#     bias0[np.where(idx==bin_id)]=bias
    indx=shear_zbins['zp'].searchsorted(shear_zbins['z_bins'][bin_id:bin_id+2])
    zs,p_zs=ztrue_given_pz_Gaussian(zp=shear_zbins['zp'][indx[0]:indx[1]],p_zp=shear_zbins['pz'][indx[0]:indx[1]],
                            bias=np.ones_like(shear_zbins['zp'][indx[0]:indx[1]])*bias,
                            sigma=shear_zbins['zp_sigma'][indx[0]:indx[1]],zs=shear_zbins['zs'])
    i=bin_id
    shear_zbins=zbin_pz_norm(shear_zbins=shear_zbins,bin_indx=i,zs=zs,p_zs=p_zs,ns=shear_zbins[i]['ns'],bg1=shear_zbins[i]['b1'],
                             AI=shear_zbins[i]['AI'],AI_z=shear_zbins[i]['AI_z'],mag_fact=shear_zbins[i]['mag_fact'],k_max=shear_zbins[i]['k_max'])
    return shear_zbins

def set_zbin_sigma(shear_zbins={},zsigma_frac=1,bin_id=0):
    zp_sigma=np.copy(shear_zbins['zp_sigma'])
    idx=np.digitize(shear_zbins['zp'],shear_zbins['z_bins'])-1
    zp_sigma[np.where(idx==bin_id)]=zp_sigma[np.where(idx==bin_id)]*zsigma_frac
    return zp_sigma

def fish_z_model(p='pz_b_s_1',Nx=2,dx_max=0.01,kappa_class=None,do_cov=False,do_log=False,z_bins_kwargs0={},Win=None,Ang_PS=None):
    t1=time.time()
    z_bins_kwargs=copy.deepcopy(z_bins_kwargs0)#.copy()
    shear_zbins=z_bins_kwargs['shear_zbins']
    shear_zbins_missed=z_bins_kwargs['shear_zbins_missed']
    galaxy_zbins=z_bins_kwargs['galaxy_zbins']
    galaxyD_zbins=z_bins_kwargs['galaxyD_zbins']
    if 'pz_b' in p:
        x0=0
    elif 'nz' in p:
        pp=p.split('_')
        bin_id=np.int(pp[-2])
        z_id=np.int(pp[-1])
#         print(p,pp,bin_id,z_id)
        if 'nz_s' in p:
            x0=shear_zbins[bin_id]['pz'][z_id]
        if 'nz_sm' in p:
            x0=shear_zbins_missed[bin_id]['pz'][z_id]
        elif 'nz_l' in p:
            x0=galaxy_zbins[bin_id]['pz'][z_id]

        if x0<1.e-5:
            x0=0
    else:
        x0=1

    x_vars,x_grad=get_x_var(x0=x0,dx_max=dx_max,do_log=do_log,Nx=Nx)

    models={}
    covs={}
    print(p,x_vars)
    for i in np.arange(Nx):
        shear_zbins_i=z_bins_kwargs['shear_zbins']
        galaxy_zbins_i=z_bins_kwargs['galaxy_zbins']
        zsm_bins_i=z_bins_kwargs['shear_zbins_missed']
        if 'pz_B' in p:
            crash
            bias=set_zbin_bias0(shear_zbins=shear_zbins_i,bias=x_vars[i])
            shear_zbins_kwargs['z_bias']=bias
            shear_zbins_i=lsst_source_tomo_bins(**shear_zbins_kwargs)#wrong
        elif 'pz_b_s' in p:
            bin_id=np.int(p[-1])
            if 'pz_b_sm' in p:
                zsm_bins_i=set_zbin_bias(shear_zbins=zsm_bins_i,bias=x_vars[i],bin_id=bin_id)
            else:
                shear_zbins_i=set_zbin_bias(shear_zbins=shear_zbins_i,bias=x_vars[i],bin_id=bin_id)
        elif 'pz_b_l' in p:
            bin_id=np.int(p[-1])
            galaxy_zbins_i=set_zbin_bias(shear_zbins=galaxy_zbins_i,bias=x_vars[i],bin_id=bin_id)
        elif 'nz_s' in p:
            if 'nz_sm' in p:
                zsm_bins_i=pz_update(z_bins=zsm_bins_i,bin_indx=bin_id,z_indx=z_id,pz_new=x_vars[i])
            else:
                shear_zbins_i=pz_update(z_bins=shear_zbins_i,bin_indx=bin_id,z_indx=z_id,pz_new=x_vars[i])
        elif 'nz_l' in p:
            galaxy_zbins_i=pz_update(z_bins=galaxy_zbins_i,bin_indx=bin_id,z_indx=z_id,pz_new=x_vars[i])

        elif 'sig' in p:
            crash
            bin_id=np.int(p[-1])
            zp_sigma=set_zbin_sigma(shear_zbins=shear_zbins_i,zsigma_frac=x_vars[i],bin_id=0)
            shear_zbins_kwargs['z_sigma']=zp_sigma
            shear_zbins_i=lsst_source_tomo_bins(**shear_zbins_kwargs)#wrong
#         print(p,'doing z',time.time()-t1)
#         t1=time.time()
        shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs)
        z_bins={'shear':shear_zbins_comb,'galaxy':galaxy_zbins_comb}
#         kappa_class.update_zbins(z_bins=shear_zbins_comb,tracer='shear')
#         kappa_class.update_zbins(z_bins=galaxy_zbins_comb,tracer='galaxy')
#         print(p,'z done',time.time()-t1)
#         t1=time.time()
        cl0G=kappa_class.tomo_short(stack_corr_indxs=z_bins_kwargs0['corr_indxs'],z_bins=z_bins,Ang_PS=Ang_PS,Win=Win)
#         print(p,'graph done',time.time()-t1)
#         t1=time.time()
        cl_t=cl0G
#         cl_t=client.compute(cl0G['stack']).result()
#         cl_t=client.compute(cl0G['pseudo_cl_b']).result()
#         print(p,'compute done',time.time()-t1,cl_t)
        models[i]=cl_t#['pcl_b']
        covs[i]=None #cl_t['cov']

#     shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs0)
#     kappa_class.update_zbins(z_bins=shear_zbins_comb,tracer='shear')
#     kappa_class.update_zbins(z_bins=galaxy_zbins_comb,tracer='galaxy')
    print(p,'done',time.time()-t1)
    return models,covs,x_vars,x_grad

def fish_galaxy_model(p='g_b_l_1_1',Nx=2,dx_max=0.01,kappa_class=None,do_cov=False,do_log=False,
                      z_bins_kwargs0={},Win=None,Ang_PS=None):
    t0=time.time()
    z_bins_kwargs=copy.deepcopy(z_bins_kwargs0)#.copy()
    shear_zbins=z_bins_kwargs['shear_zbins']
    shear_zbins_missed=z_bins_kwargs['shear_zbins_missed']
    galaxy_zbins=z_bins_kwargs['galaxy_zbins']
    galaxyD_zbins=z_bins_kwargs['galaxyD_zbins']

    Dx=np.linspace((1-dx_max),(1+dx_max),Nx)
    p2=p.split('_')
    z_indx=np.nan
    models={}
    covs={}
    if 'bz' in p:
#         print(p,p2,p2[-3])
        p_n='bz'+p2[-2]
        bin_indx=np.int(p2[-1])
        z_indx=np.int(p2[-3])
        if 'l' in p:
            if 'lD' in p:
                x0=galaxyD_zbins[bin_indx][p_n][z_indx]
                pz=galaxyD_zbins[bin_indx]['pz'][z_indx]
            else:
#                 print(galaxy_zbins[bin_indx][p_n].shape,n_zl)
                x0=galaxy_zbins[bin_indx][p_n][z_indx]
                pz=galaxy_zbins[bin_indx]['pz'][z_indx]
        if 's' in p:
            if 'sm' in p:
                x0=shear_zbins_missed[bin_indx][p_n][z_indx]
                pz=shear_zbins_missed[bin_indx]['pz'][z_indx]
            else:
                x0=shear_zbins[bin_indx][p_n][z_indx]
                pz=shear_zbins[bin_indx]['pz'][z_indx]
        print(p,'pz:',pz)

        if pz==0:
            print('pz zero, ignoring ',p)
            return models,covs,[],[]
    elif 'b' in p:
        bin_indx=np.int(p2[-1])
        p_n='b'+p2[-2]
        if 'l' in p:
            if 'lD' in p:
                x0=galaxyD_zbins[bin_indx][p_n]
            else:
                x0=galaxy_zbins[bin_indx][p_n]
        if 's' in p:
            if 'sm' in p:
                x0=shear_zbins_missed[bin_indx][p_n]
            else:
                x0=shear_zbins[bin_indx][p_n]
    elif 'mag' in p:
        bin_indx=np.int(p2[-1])
        p_n='mag_fact'
        if 'l' in p:
            if 'lD' in p:
                x0=galaxyD_zbins[bin_indx][p_n]
            else:
                x0=galaxy_zbins[bin_indx][p_n]
        if 's' in p:
            if 'sm' in p:
                x0=shear_zbins_missed[bin_indx][p_n]
            else:
                x0=shear_zbins[bin_indx][p_n]
    else:
        pass
    x_vars,x_grad=get_x_var(x0=x0,dx_max=dx_max,do_log=do_log,Nx=Nx)

    print(p,x_vars)
    for i in np.arange(Nx):
#         z_bins_kwargs=copy.deepcopy(z_bins_kwargs0)
        shear_zbins=z_bins_kwargs['shear_zbins']
        galaxy_zbins=z_bins_kwargs['galaxy_zbins']
        galaxyD_zbins=z_bins_kwargs['galaxyD_zbins']
        zsm_bins=z_bins_kwargs['shear_zbins_missed']
        zst_bins=z_bins_kwargs['shear_zbins_train']

        if 'bz' in p:
            if 'l' in p:
                if 'lD' in p:
                    galaxyD_zbins[bin_indx][p_n][z_indx]=x_vars[i]
                    galaxyD_zbins['bias_func']='linear_bias_z'
                else:
                    galaxy_zbins[bin_indx][p_n][z_indx]=x_vars[i]
                    galaxy_zbins['bias_func']='linear_bias_z'
            if 's' in p:
                if 'sm' in p:
                    zsm_bins[bin_indx][p_n][z_indx]=x_vars[i]
                    zsm_bins['bias_func']='linear_bias_z'
                else:
                    shear_zbins[bin_indx][p_n][z_indx]=x_vars[i]
                    shear_zbins['bias_func']='linear_bias_z'
                    if zst_bins['n_bins']>0:
                        zst_bins[bin_indx][p_n][z_indx]=x_vars[i]
                        zst_bins['bias_func']='linear_bias_z'
        elif 'b' in p:
            if 'l' in p:
                if 'lD' in p:
                    galaxyD_zbins[bin_indx][p_n]=x_vars[i]
                    galaxyD_zbins['bias_func']='constant_bias'
                else:
                    galaxy_zbins[bin_indx][p_n]=x_vars[i]
                    galaxy_zbins['bias_func']='constant_bias'
            if 's' in p:
                if 'sm' in p:
                    zsm_bins[bin_indx][p_n]=x_vars[i]
                    zsm_bins['bias_func']='constant_bias'
                else:
                    shear_zbins[bin_indx][p_n]=x_vars[i]
                    shear_zbins['bias_func']='constant_bias'
                    if zst_bins['n_bins']>0:
                        zst_bins[bin_indx][p_n]=x_vars[i]
                        zst_bins['bias_func']='constant_bias'
        elif 'mag' in p:
            if 'l' in p:
                if 'lD' in p:
                    galaxyD_zbins[bin_indx][p_n]=x_vars[i]
                else:
                    galaxy_zbins[bin_indx][p_n]=x_vars[i]
            if 's' in p:
                if 'sm' in p:
                    zsm_bins[bin_indx][p_n]=x_vars[i]
                else:
                    shear_zbins[bin_indx][p_n]=x_vars[i]
                    if zst_bins['n_bins']>0:
                        zst_bins[bin_indx][p_n]=x_vars[i]

        shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs)
        z_bins={'shear':shear_zbins_comb,'galaxy':galaxy_zbins_comb}
#         kappa_class.update_zbins(z_bins=shear_zbins_comb,tracer='shear')
#         kappa_class.update_zbins(z_bins=galaxy_zbins_comb,tracer='galaxy')

        cl0G=kappa_class.tomo_short(stack_corr_indxs=z_bins_kwargs0['corr_indxs'],z_bins=z_bins,
                                    Ang_PS=Ang_PS,Win=Win)
        models[i]=cl0G
#         cl_t=cl0G['stack'].compute()
#         models[i]=cl_t['pcl_b']
#         covs[i]=cl_t['cov']
#         kappa_class.Ang_PS.reset()#FIXME


#     shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs0)
#     kappa_class.update_zbins(z_bins=shear_zbins_comb,tracer='shear')
#     kappa_class.update_zbins(z_bins=galaxy_zbins_comb,tracer='galaxy')
    print(p,'done',time.time()-t0)
    return models,covs,x_vars,x_grad


def fish_shear_model(p='s_m_s_1',Nx=2,dx_max=0.01,kappa_class=None,do_cov=False,do_log=False,
                      z_bins_kwargs0={},Win=None,Ang_PS=None):
    t0=time.time()
    z_bins_kwargs=copy.deepcopy(z_bins_kwargs0)#.copy()
    shear_zbins=z_bins_kwargs['shear_zbins']
    shear_zbins_missed=z_bins_kwargs['shear_zbins_missed']
    galaxy_zbins=z_bins_kwargs['galaxy_zbins']
    galaxyD_zbins=z_bins_kwargs['galaxyD_zbins']

    Dx=np.linspace((1-dx_max),(1+dx_max),Nx)
    p2=p.split('_')
    z_indx=np.nan
    models={}
    covs={}
    if 's_m' in p:
        bin_indx=np.int(p2[-1])
        p_n='shear_m_bias'
        if 'sm' in p:
            x0=shear_zbins_missed[bin_indx][p_n]
        else:
            x0=shear_zbins[bin_indx][p_n]
    elif 'AI' in p:
        bin_indx=np.int(p2[-1])
        p_n='AI'
        if 'sm' in p:
            x0=shear_zbins_missed[bin_indx][p_n]
        else:
            x0=shear_zbins[bin_indx][p_n]

    x_vars,x_grad=get_x_var(x0=x0,dx_max=dx_max,do_log=do_log,Nx=Nx)

    print(p,x_vars)
    for i in np.arange(Nx):
#         z_bins_kwargs=copy.deepcopy(z_bins_kwargs0)
        shear_zbins=z_bins_kwargs['shear_zbins']
        galaxy_zbins=z_bins_kwargs['galaxy_zbins']
        galaxyD_zbins=z_bins_kwargs['galaxyD_zbins']
        zsm_bins=z_bins_kwargs['shear_zbins_missed']
        zst_bins=z_bins_kwargs['shear_zbins_train']

        if 'sm' in p:
            zsm_bins[bin_indx][p_n]=x_vars[i]
        else:
            shear_zbins[bin_indx][p_n]=x_vars[i]

        shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs)
        z_bins={'shear':shear_zbins_comb,'galaxy':galaxy_zbins_comb}
#         kappa_class.update_zbins(z_bins=shear_zbins_comb,tracer='shear')
#         kappa_class.update_zbins(z_bins=galaxy_zbins_comb,tracer='galaxy')
        cl0G=kappa_class.tomo_short(stack_corr_indxs=z_bins_kwargs0['corr_indxs'],z_bins=z_bins,Ang_PS=Ang_PS,Win=Win)
        models[i]=cl0G
#         cl_t=cl0G['stack'].compute()
#         models[i]=cl_t['pcl_b']
#         covs[i]=cl_t['cov']

#     shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs0)
#     kappa_class.update_zbins(z_bins=shear_zbins_comb,tracer='shear')
#     kappa_class.update_zbins(z_bins=galaxy_zbins_comb,tracer='galaxy')
    print(p,'done',time.time()-t0)
    return models,covs,x_vars,x_grad


def fish_baryon_model(p='Q1',Nx=2,dx_max=0.01,kappa_class=None,clS=None,cl0=None,do_cov=False,do_log=False,
                      z_bins_kwargs0={},NmarQ=2,Win=None,Ang_PS=None):
    Dx=np.linspace((1-dx_max),(1+dx_max),Nx)

    if 'Q' in p:
        Q_indx=np.int(p[-1])
        x0=0
    else:
        pass
    x_vars,x_grad=get_x_var(x0=x0,dx_max=dx_max,do_log=do_log,Nx=Nx)

    models={}
    covs={}
    cov_F=1.#/np.median(np.diag(clS['cov']))
    print(p,x_vars)

    PCS=PCA_shear(kappa_class=kappa_class,NmarQ=NmarQ,clS=clS)

    for i in np.arange(Nx):
        Q0=[0]*NmarQ
        Q0[Q_indx]=x_vars[i]

        cl_t=PCS.compute_modelv_bary(cosmo_params=cosmo_fid,Q=Q0)
        models[i]=np.zeros_like(cl0) #this is needed to keep everything in right shape. Baryon stuff is only for shear right now.
        models[i][:len(cl_t)]+=cl_t #['cl_b']
        models[i][len(cl_t):]+=cl0[len(cl_t):]
        print(models[i].shape,PCS.COV.shape)
        covs[i]=PCS.COV
        kappa_class.Ang_PS.reset()
    return models,covs,x_vars,x_grad

def fisher_calc(cosmo_params=['As'],z_params=[],galaxy_params=[],baryon_params=[],shear_params=[],
                Nx=2,dx_max=0.01,do_log=False,scheduler_info=None,
                kappa_class=None,do_cov=False,baryon_PCA_nQ=2,clS=None,
               Skylens_kwargs={},z_bins_kwargs={},ell_bin_kwargs={},cl0=None,priors=None):
    t1=time.time()
    if kappa_class is None:
        shear_zbins=lsst_source_tomo_bins(**shear_zbins_kwargs)
        l0,l_bins,l=get_cl_ells(**ell_bin_kwargs)
        kappa_class=Skylens(l=l0,l_bins=l_bins,shear_zbins=shear_zbins,**Skylens_kwargs)

    do_cov0=np.copy(kappa_class.do_cov)
    kappa_class.do_cov=do_cov
    cl0G=kappa_class.cl_tomo()

    if clS is None:
        kappa_class.do_cov=True #need covariance to start things.
        cl0G=kappa_class.cl_tomo()

        clS=cl0G['stack'].compute()
#     cl_t=client.submit(cl0G['stack'])
    cl0=clS['pcl_b']

    zk=['shear_zbins','galaxy_zbins','galaxyD_zbins','shear_zbins_missed','shear_zbins_train','corr_indxs']
    cc=[]
    for k in z_bins_kwargs['corr_indxs'].keys():
        cc+=z_bins_kwargs['corr_indxs'][k]
    z_bins_kwargs2={k:copy.deepcopy(z_bins_kwargs[k]) for k in zk}
    t2=time.time()
    print('fisher_calc cl0G done',cl0G['cov'],t2-t1,kappa_class.Ang_PS.PS.cosmo_params)#get_size_pickle(kappa_class),get_size_pickle(z_bins_kwargs2))
    del cc
    cosmo_fid=copy.deepcopy(kappa_class.Ang_PS.PS.cosmo_params)#.copy()
    cosmo_h=None #kappa_class.Ang_PS.PS.cosmo_h.clone()
    cov=clS['cov']

    ndim=len(cosmo_params)+len(z_params)+len(galaxy_params)+len(baryon_params)+len(shear_params)
    params_all=np.append(np.append(cosmo_params,z_params),galaxy_params)
    params_all=np.append(params_all,shear_params)
    params_all=np.append(params_all,baryon_params)

    x_vars={}
    models={}
    model_derivs={}
    covs={}
    cov_derivs={}
    x_grads={}
    t=time.time()
    gc.disable()
    outp={}
    t0=time.time()

    client=client_get(scheduler_info)
    kappa_class.gather_data()
    z_bins_kwargs2=scatter_dict(z_bins_kwargs2,broadcast=True,scheduler_info=scheduler_info,depth=2)
    Win=scatter_dict(kappa_class.Win,broadcast=True,scheduler_info=scheduler_info,depth=1)
    Ang_PS=client.scatter(kappa_class.Ang_PS,broadcast=True)

    kappa_class=client.scatter(kappa_class,broadcast=True)
    print('fisher calc, scattered kappa_class',kappa_class)

    for p in cosmo_params:
#         models[p],covs[p],x_vars[p],x_grads[p]=fish_cosmo_model(p=p,Nx=Nx,dx_max=dx_max,do_log=do_log,
        outp[p]=delayed(fish_cosmo_model)(p=p,Nx=Nx,dx_max=dx_max,do_log=do_log,Win=Win,Ang_PS=Ang_PS,
                                          kappa_class=kappa_class,do_cov=do_cov,z_bins_kwargs=z_bins_kwargs2)
        t1=time.time()
        print(p,'time: ',t1-t)
        t=t1
#     gc.enable()
#     gc.collect()
#     gc.disable()
    for p in z_params:
#         models[p],covs[p],x_vars[p],x_grads[p]=fish_z_model(p=p,Nx=Nx,dx_max=dx_max,
        outp[p]=delayed(fish_z_model)(p=p,Nx=Nx,dx_max=dx_max,Win=Win,Ang_PS=Ang_PS,
                                                 kappa_class=kappa_class,
                                                 do_cov=do_cov,z_bins_kwargs0=z_bins_kwargs2)
        t1=time.time()
        print(p,'time: ',t1-t)
        t=t1
#     gc.enable()
#     gc.collect()
#     gc.disable()

    for p in galaxy_params:
#         models[p],covs[p],x_vars[p],x_grads[p]=fish_galaxy_model(p=p,Nx=Nx,dx_max=dx_max,
        outp[p]=delayed(fish_galaxy_model)(p=p,Nx=Nx,dx_max=dx_max,Win=Win,Ang_PS=Ang_PS,
                                                 kappa_class=kappa_class,
                                                 do_cov=do_cov,z_bins_kwargs0=z_bins_kwargs2)
#         if models[p]=={}:
#             x=params_all!=p
#             params_all=params_all[x]
        t1=time.time()
        print(p,'time: ',t1-t)
        t=t1
#     gc.enable()
#     gc.collect()
#     gc.disable()

    for p in shear_params:
#         models[p],covs[p],x_vars[p],x_grads[p]=fish_shear_model(p=p,Nx=Nx,dx_max=dx_max,
        outp[p]=delayed(fish_shear_model)(p=p,Nx=Nx,dx_max=dx_max,Win=Win,Ang_PS=Ang_PS,
                                                 kappa_class=kappa_class,
                                                 do_cov=do_cov,z_bins_kwargs0=z_bins_kwargs2)
#         if models[p]=={}:
#             x=params_all!=p
#             params_all=params_all[x]
        t1=time.time()
        print(p,'time: ',t1-t)
        t=t1
#     gc.enable()
#     gc.collect()
#     gc.disable()

    for p in baryon_params:
#         models[p],covs[p],x_vars[p],x_grads[p]=
        outp[p]=delayed(fish_baryon_model)(p=p,Nx=Nx,dx_max=dx_max,Win=Win,Ang_PS=Ang_PS,
                                                 kappa_class=kappa_class,clS=cl_shear,cl0=cl0,
                                                 do_cov=do_cov,NmarQ=baryon_PCA_nQ)
        t1=time.time()
        print(p,'time: ',t1-t)
        t=t1
#     gc.enable()
#     gc.collect()

    ndim=len(params_all)
    print(ndim,params_all)
    client=client_get(scheduler_info=scheduler_info)
    outp=client.compute(outp).result()
    print(p,'all done time: ',time.time()-t0)

    kappa_class=client.gather(kappa_class)
    z_bins_kwargs2=gather_dict(z_bins_kwargs2,scheduler_info=scheduler_info)

    params_missing=[]
    for p in params_all:
        models[p],covs[p],x_vars[p],x_grads[p]=outp[p]
        if models[p]=={}:
#             x=params_all!=p
#             params_all=params_all[x]
            params_missing+=[p]
            continue
        model_derivs[p]=np.gradient(np.array([models[p][i] for i in np.arange(Nx)]),axis=0).T
        model_derivs[p]/=x_grads[p]
        model_derivs[p]=model_derivs[p][:,np.int(Nx/2)]
        if 'bz' in p and np.all(model_derivs[p]==0):
            print('model derivs zero ',p, np.all(model_derivs[p]==0))
            print('model0:', models[p][0],models[p][0]-models[p][1])
            print('z:',kappa_class.Ang_PS.z,z_bins_kwargs['galaxy_zbins'][0]['z'])

        if do_cov:
            cov_derivs[p]=np.gradient(np.array([covs[p][i] for i in np.arange(Nx)]),axis=0).T
            cov_derivs[p]/=x_grads[p]
#             print(cov_derivs[p].shape,x_grad.shape)
            cov_derivs[p]=cov_derivs[p][:,:,np.int(Nx/2)]

    for p in params_missing:
        x=params_all!=p
        params_all=params_all[x]
    ndim=len(params_all)

#     if kappa_class.sparse_cov:
#         cov=cov.todense()
    cov_inv=np.linalg.inv(cov)

    cov_p_inv=np.zeros([ndim]*2)
    i1=0
    for p1 in params_all:
        i2=0
        for p2 in params_all:
            cov_p_inv[i1,i2]=np.dot(model_derivs[p1],np.dot(cov_inv,model_derivs[p2]))
#             if np.all(cov_p_inv[i1,i2]==0):
#                 print('',p1,p2,model_derivs[p1],model_derivs[p2])
            if do_cov:
#                 print(cov_p_inv[i1,i2],0.5*np.trace(cov_inv@cov_derivs[p1]@cov_inv@cov_derivs[p2]))
                cov_p_inv[i1,i2]+=0.5*np.trace(cov_inv@cov_derivs[p1]@cov_inv@cov_derivs[p2])
            i2+=1
        i1+=1
    out={}
    out['cov_p_inv']=np.copy(cov_p_inv)
    out['params_all']=params_all
    out['params_missing']=params_missing
    if priors is not None:
        out=fish_apply_priors(fish=out,priors=priors)
        if priors.get('nz_ana') is not None:
            out=fish_apply_priors(fish=out,priors=priors['nz_ana'])
#         i2=0
#         for p1 in params_all:
#             cov_p_inv[i2,i2]+=1./priors[p1]**2
#             i2+=1
    try:
        out['cov_p']=np.linalg.inv(cov_p_inv)
        out['error']=np.sqrt(np.diag(out['cov_p']))
    except Exception as err:
        print(err,cov_p_inv,np.linalg.matrix_rank(cov_p_inv),cov_p_inv.shape)

    out['cov_deriv']=cov_derivs
    out['model_deriv']=model_derivs
    out['cov']=cov
    out['cov_inv']=cov_inv
    out['model']=models
    out['x_vars']=x_vars
    out['params']=params_all
    kappa_class.do_cov=do_cov0
    print('Fisher calc done')
    return out

def fish_apply_priors(priors=None,fish=[]):
#     ndim=len(cosmo_params)+len(z_params)+len(galaxy_params)+len(baryon_params)
#     params_all=np.append(np.append(cosmo_params,z_params),galaxy_params)
#     params_all=np.append(params_all,baryon_params)
    params_all=fish['params_all']
    ndim=len(params_all)

    cov_p_inv=np.copy(fish['cov_p_inv'])
    fish['prior']={}
    priors2={}
    i2=0
    for p1 in params_all:
        priors2[p1]=priors[p1]
        cov_p_inv[i2,i2]+=1./priors[p1]**2
        i2+=1
    fish['prior']['cov_p_inv']=cov_p_inv
    fish['prior']['prior']=priors2
    try:
        fish['prior']['cov_p']=np.linalg.inv(cov_p_inv)
        fish['prior']['error']=np.sqrt(np.diag(fish['prior']['cov_p']))
    except Exception as err:
        print(err)
    return fish

k='corrs3_zmin0_barynQ2_pkfclass_pk_SSVFalse'


def get_cl_ells(lmax_cl=None,lmin_cl=None,Nl_bins=None,bin_cl=None):
    l0=np.arange(lmin_cl,lmax_cl)

    lmin_cl_Bins=lmin_cl+10
    lmax_cl_Bins=lmax_cl-10
    l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins))
    l=l0 #np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*20))) #if we want to use fewer ell
    if not bin_cl:
        lb=np.int64(0.5*(l_bins[1:]+l_bins[:-1]))
        l0=lb*1
        l=lb*1
        l_bins=None
        print('no binning, l=',l)
    return l0,l_bins,l


corr_ggl=('galaxy','shear')
corr_gg=('galaxy','galaxy')
corr_ll=('shear','shear')
Fmost=False
def get_z_bins(zmin,zmax,shear_n_zbins,galaxy_n_zbins,galaxyD_n_zbins=None,nz_shear=None,nz_galaxy=None,nz_galaxyD=None,use_window=None,
               nside=None,shear_zsigma=0.05,galaxy_zsigma=0.01,area_overlap=0.24,z_max_galaxy=None,f_sky=None,AI=0,AI_z=0,
               nz_shear_missed=0,area_train=0,nz_shear_train=0,train_n_zbins=0,
               mag_fact=0,unit_window=False,z_true_max=None,train_sample_missed=None,n_zs_shear=0,n_zs_galaxy=0): #,**kwargs):

    shear_zbins=lsst_source_tomo_bins(zmin=zmin,zmax=zmax,nside=nside,ns0=nz_shear,nbins=shear_n_zbins,f_sky=f_sky,
                                  z_sigma_power=1,z_sigma=shear_zsigma,ztrue_func=ztrue_given_pz_Gaussian,
                                      n_zs=n_zs_shear,
                                  use_window=use_window,unit_win=unit_window,AI=AI,AI_z=AI_z,z_true_max=z_true_max)
    shear_zbins_missed={'n_bins':0}
    if train_sample_missed!=0:
        shear_zbins_missed=lsst_source_tomo_bins(zmin=zmin,zmax=zmax,n_zs=n_zs_shear,nside=nside,ns0=nz_shear_missed,nbins=train_sample_missed,f_sky=f_sky,
                                  z_sigma_power=1,z_sigma=shear_zsigma*3,ztrue_func=ztrue_given_pz_Gaussian,
                                             use_window=use_window,AI=AI,AI_z=AI_z,unit_win=unit_window,z_true_max=z_true_max)
#         shear_zbins=combine_zbins(z_bins1=shear_zbins,z_bins2=shear_zbins_missed)
    shear_zbins_train={'n_bins':0}
    if area_train>0:
        f_sky_train=area_train*d2r**2/4/np.pi
        shear_zbins_train=lsst_source_tomo_bins(zmin=zmin,zmax=zmax,n_zs=n_zs_shear,nside=nside,ns0=nz_shear_train,nbins=train_n_zbins,f_sky=f_sky_train,
                                  z_sigma_power=1,z_sigma=shear_zsigma,ztrue_func=ztrue_given_pz_Gaussian,unit_win=unit_window,
                                  use_window=use_window,AI=AI,AI_z=AI_z,z_true_max=z_true_max)#exactly same config as shear_zbins, including shear_zsigma as we want same p(zs)

    galaxy_zbins=lsst_source_tomo_bins(zmin=zmin,zmax=z_max_galaxy,n_zs=n_zs_galaxy,ns0=nz_galaxy,nbins=galaxy_n_zbins,nside=nside,f_sky=f_sky,
                         ztrue_func=ztrue_given_pz_Gaussian,use_window=use_window,mag_fact=mag_fact,unit_win=unit_window,
                        z_sigma=galaxy_zsigma,z_true_max=z_true_max)
    galaxyD_zbins={}
    galaxyD_zbins['n_bins']=0
    galaxyD_zbins['n_binsF']=0
    galaxyD_zbins['n_binsF1']=0
    galaxyD_zbins['n_binsF2']=0
    galaxyD_zbins['area_overlap']=area_overlap
    if galaxyD_n_zbins is not None:
        galaxyD_zbins=DESI_z_bins(nside=nside,f_sky=f_sky,use_window=use_window,mag_fact=mag_fact,
                             n_zs=n_zs_galaxy,zmin=zmin,zmax=zmax,unit_win=unit_window,nbins=galaxyD_n_zbins,
                             mask_start_pix=np.int32(hp.nside2npix(nside)*f_sky*(1-area_overlap)),z_true_max=z_true_max)
        galaxyD_zbins['n_binsF']=0
        galaxyD_zbins['n_binsF1']=0
        galaxyD_zbins['n_binsF2']=0
        if Fmost:
            galaxyD_zbins2=DESI_z_bins(datasets=['lrg']#,'qso']
                                  ,nside=nside,f_sky=f_sky4,use_window=use_window,mag_fact=mag_fact,unit_win=unit_window,
                                  mask_start_pix=0,z_true_max=z_true_max,n_zs=n_zs_galaxy)
            galaxyD_zbins3=DESI_z_bins(datasets=['elg'],nside=nside,f_sky=1000./40000,use_window=use_window,mag_fact=mag_fact,
                                  mask_start_pix=0,unit_win=unit_window,z_true_max=z_true_max,n_zs=n_zs_galaxy)
            galaxyD_zbins=combine_zbins(z_bins1=galaxyD_zbins,z_bins2=galaxyD_zbins2)
            galaxyD_zbins=combine_zbins(z_bins1=galaxyD_zbins,z_bins2=galaxyD_zbins3)
            galaxyD_zbins['n_binsF1']=galaxyD_zbins2['n_bins']
            galaxyD_zbins['n_binsF2']=galaxyD_zbins3['n_bins']
            galaxyD_zbins['n_binsF']=galaxyD_zbins3['n_bins']+galaxyD_zbins2['n_bins']
    return shear_zbins,shear_zbins_train,shear_zbins_missed,galaxy_zbins,galaxyD_zbins

def combine_z_bins_all(z_bins_kwargs={}):
    #FIXME: add training sample to shear also.
    if z_bins_kwargs['shear_zbins_missed']['n_bins']>0:
        shear_zbins_comb=combine_zbins(z_bins1=z_bins_kwargs['shear_zbins'],z_bins2=z_bins_kwargs['shear_zbins_missed'])
    else:
        shear_zbins_comb=z_bins_kwargs['shear_zbins']
    if z_bins_kwargs['galaxyD_zbins']['n_bins']>0:
        galaxy_zbins_comb=combine_zbins(z_bins1=z_bins_kwargs['galaxy_zbins'],z_bins2=z_bins_kwargs['galaxyD_zbins'])
        galaxyD_n_zbins2=z_bins_kwargs['galaxyD_zbins']['n_bins']
    else:
        galaxy_zbins_comb=z_bins_kwargs['galaxy_zbins']
    galaxy_zbins_comb=combine_zbins(z_bins1=shear_zbins_comb,z_bins2=galaxy_zbins_comb)
    if z_bins_kwargs['shear_zbins_train']['n_bins']>0:
        galaxy_zbins_comb=combine_zbins(z_bins1=galaxy_zbins_comb,z_bins2=z_bins_kwargs['shear_zbins_train'])
        shear_zbins_comb=combine_zbins(z_bins1=shear_zbins_comb,z_bins2=z_bins_kwargs['shear_zbins_train'])
    

    return shear_zbins_comb,galaxy_zbins_comb

def mask_comb(win1,win2):
    """
    combined the mask from two windows which maybe partially overlapping.
    Useful for some covariance calculations, specially SSC, where we assume a uniform window.
    """
    W=win1*win2
    x=np.logical_or(win1==hp.UNSEEN, win2==hp.UNSEEN)
    W[x]=hp.UNSEEN
    W[~x]=1. #mask = 0,1
    fsky=(~x).mean()
    return fsky,W#.astype('int16')


def photoz_prior(kappa_class=None,Skylens_kwargs0={},z_bins_kwargs={},key_label={}):
    client=client_get(scheduler_info=Skylens_kwargs0['scheduler_info'])
#     kappa_class.gather_data()
    window_l=np.arange(Skylens_kwargs0['window_lmax']+1)
    zbins=client.gather(z_bins_kwargs['shear_zbins_train']) #dictionary of z_bins of the training sample, in skylens format

    #we need new skylens object to compute power spectra at window_l
#     Skylens_kwargs=copy.deepcopy(Skylens_kwargs0)
#     Skylens_kwargs['galaxy_zbins']=zbins
#     Skylens_kwargs['shear_zbins']=None
#     Skylens_kwargs['l']=window_l
#     Skylens_kwargs['window_l']=window_l
#     Skylens_kwargs['use_binned_l']=False
#     Skylens_kwargs['use_window']=True
#     Skylens_kwargs['do_cov']=False
#     Skylens_kwargs['corrs']=[corr_gg]
#     kappa_class2=Skylens(**Skylens_kwargs)
#     kappa_class2.gather_data()
    AP2=copy.deepcopy(kappa_class.Ang_PS)
    AP2.l=window_l
    AP2.reset()

    f_sky=z_bins_kwargs['area_train']*d2r**2/4/np.pi
    area_train_arcmin=z_bins_kwargs['area_train']*3600
    Om_w12=4*np.pi*f_sky

    sigma_win={}

    #following computes cl for each histogram bin, centered on zl.
#     clz=kappa_class2.Ang_PS.angular_power_z() #this is dimensionaless power spectra, normalized with some factors of chi and on ell grid.
    clz=AP2.angular_power_z()
    cosmo_h=AP2.PS
    zl=AP2.z
    cH=cosmo_h.Dh/cosmo_h.efunc(zl)
    kernel=1./cH # missing galaxy bias, add later
    cls=clz['cls'].T*clz['dchi']*kernel**2
    cls=cls.T

    prior={}
    for bi in np.arange(zbins['n_bins']):
        for bj in np.arange(bi,zbins['n_bins']):
#             win_f=kappa_class2.Win['cl'][corr_gg][(bi,bj)][12]['cl']
            win_f=hp.anafast(map1=zbins[bi]['window'],map2=zbins[bj]['window'],lmax=Skylens_kwargs0['window_lmax'])
            for zi in np.arange(len(zl)):
                clij=cls[zi]
                sigma_win=np.dot(win_f*np.gradient(window_l)*(2*window_l+1),clij) #under limber clij is non-zero for zi=zj
                sigma_win/=Om_w12**1 #FIXME: Check norm
#                 print('photoz_prior: ',sigma_win,win_f.shape,window_l.shape,clij.shape,kappa_class.window_l,kappa_class.l)
                for zj in np.arange(len(zl)):
                    var=0
                    if zi==zj:
                        try:
                            pzf=zbins[bi]['pz'][zi]*zbins[bj]['pz'][zj] #some of 'pz'=0 bins are thrown away
                        except:
                            continue
                        var+=sigma_win*pzf
                    if bi==bj and zi==zj:
                        var+=1*zbins[bi]['pz'][zi]/(zbins[bi]['ns']*area_train_arcmin) #denominator is the total number of galaxies
                        kk=key_label+str(bi)+'_'+str(zj)
                        prior[kk]=np.sqrt(var) #FIXME: this is neglecting cross correlations between bins
                        print('photoz_prior: ',kk,prior[kk],zbins[bi]['pz'][zi],zbins[bi]['ns']*area_train_arcmin,sigma_win*pzf)
    return prior



def init_fish(z_min=None,z_max=None,corrs=None,SSV=None,do_cov=None,
              pk_func=None,nz_shear=None,shear_n_zbins=None,f_sky=0.3,area_overlap=0.2,
              z_true_max=None,train_sample_missed=0,area_train=0,nz_shear_missed=0,train_n_zbins=0,
              store_win=None,mag_fact=0,nside=None,unit_window=False,
             galaxy_n_zbins=None,galaxyD_n_zbins=None,nz_galaxy=None,nz_galaxyD=None,z_bins_kwargs=None,pk_params=None,
              n_zs_shear=None,n_zs_galaxy=None,nz_shear_train=0,z_max_galaxy=None,use_window=True,
             Skylens_kwargs=None,Win=None):

    pk_params2=copy.deepcopy(pk_params)
    pk_params2['pk_func']=pk_func
#     power_spectra_kwargs2={'pk_params':pk_params2}

#     Skylens_kwargs={'do_cov':do_cov,'bin_cl':bin_cl,'use_binned_l':use_binned_l, #'l':l0,'l_bins':l_bins,
#             'SSV_cov':SSV,'tidal_SSV_cov':SSV,'do_xi':False,'use_window':use_window,'window_lmax':window_lmax,
#             'f_sky':f_sky,'corrs':corrs,'store_win':store_win,'Win':Win, 'wigner_files':wigner_files, #'sigma_gamma':sigma_gamma
#             'do_sample_variance':do_sample_variance,'pk_params':pk_params2,'f_sky':f_sky,
#             'bin_xi':bin_xi,'sparse_cov':sparse_cov,'nz_PS':nz_PS,'z_PS':z_PS,'scheduler_info':scheduler_info,
#             'clean_tracer_window':clean_tracer_window#'client':client
#             }
#     ell_bin_kwargs={'lmax_cl':l_max,'lmin_cl':l_min,'Nl_bins':Nl_bins}
#     l0,l_bins,l=get_cl_ells(**ell_bin_kwargs)

    if z_bins_kwargs is None:
        z_bins_kwargs={'zmin':z_min,'zmax':z_max,'nz_shear':nz_shear,'shear_n_zbins':shear_n_zbins,'nside':nside,'z_true_max':z_true_max,
                       'galaxy_n_zbins':galaxy_n_zbins,'galaxyD_n_zbins':galaxyD_n_zbins,'unit_window':unit_window,
                       'train_sample_missed':train_sample_missed,'area_train':area_train,'z_max_galaxy':z_max_galaxy,
                       'n_zs_shear':n_zs_shear,'n_zs_galaxy':n_zs_galaxy,'nz_shear_train':nz_shear_train,'use_window':use_window,
                        'use_window':True,'nz_galaxy':nz_galaxy,'nz_galaxyD':nz_galaxyD,'nz_shear_missed':nz_shear_missed,
                       'area_overlap':area_overlap, 'f_sky':f_sky,'galaxy_zsigma':0.01,
                       'mag_fact':mag_fact,'shear_zsigma':0.05,'train_n_zbins':train_n_zbins}

        shear_zbins,shear_zbins_train,shear_zbins_missed,galaxy_zbins,galaxyD_zbins=get_z_bins(**z_bins_kwargs)

        z_bins_kwargs['shear_zbins']=shear_zbins
        z_bins_kwargs['galaxy_zbins']=galaxy_zbins
        z_bins_kwargs['galaxyD_zbins']=galaxyD_zbins
        z_bins_kwargs['shear_zbins_train']=shear_zbins_train
        z_bins_kwargs['shear_zbins_missed']=shear_zbins_missed

        shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs)

        print('nbins',shear_zbins['n_bins'],galaxy_zbins['n_bins'],galaxyD_zbins['n_bins'],galaxy_zbins_comb['n_bins'],shear_zbins_comb['n_bins'])

        n_bins_shear_photo=shear_zbins['n_bins']+shear_zbins_missed['n_bins']
        
        ii_galaxy_photo=np.arange(n_bins_shear_photo)
        ii_shear_photo=np.arange(n_bins_shear_photo)
        ii_galaxy_spec=np.arange(n_bins_shear_photo,galaxy_zbins_comb['n_bins']) #DESI+training
        ii_shear_spec=np.arange(n_bins_shear_photo,shear_zbins_comb['n_bins']) #training
        ii_galaxy_train=np.arange(n_bins_shear_photo+galaxyD_zbins['n_bins'],galaxy_zbins_comb['n_bins']) #training

        z_bins_kwargs['corr_indxs']={corr_gg:{},corr_ll:{},corr_ggl:{}}
        z_bins_kwargs['corr_indxs'][corr_gg]=[(i,i) for i in ii_galaxy_spec] #auto for spec
        z_bins_kwargs['corr_indxs'][corr_gg]+=[(i,j) for i in ii_galaxy_photo for j in ii_galaxy_spec] #cross between photo and spec samples
        z_bins_kwargs['corr_indxs'][corr_gg]+=[(i,j) for i in ii_galaxy_photo for j in np.arange(i,n_bins_shear_photo)] #auto and cross b/w photo samples

        z_bins_kwargs['corr_indxs'][corr_ll]=[(i,j) for i in ii_shear_photo for j in np.arange(i,n_bins_shear_photo)] #auto and cross with photo samples only.

        z_bins_kwargs['corr_indxs'][corr_ggl]=[(j,i) for i in ii_shear_photo for j in ii_galaxy_spec ] #shear: photo, lens: spec
        z_bins_kwargs['corr_indxs'][corr_ggl]+=[(j,i) for i in ii_shear_photo for j in np.arange(i,n_bins_shear_photo)]#shear: photo, lens: photo
        z_bins_kwargs['corr_indxs'][corr_ggl]+=[(j,i) for i in ii_shear_spec for j in ii_galaxy_train]#shear: spec, lens: spec. auto only, for IA

    #     print('init fish, indxs:',z_bins_kwargs['corr_indxs'])

        Skylens_kwargs['galaxy_zbins']=galaxy_zbins_comb
        Skylens_kwargs['shear_zbins']=shear_zbins_comb

    #     print('running Skylens',l0.max(),galaxy_zbins_comb['n_bins'],Skylens_kwargs.keys())
        use_window=copy.deepcopy(Skylens_kwargs['use_window'])
        Skylens_kwargs['use_window']=False #not needed in this particular calc. removed for speed
        Skylens_kwargs['stack_indxs']=z_bins_kwargs['corr_indxs']
#         for k in Skylens_kwargs.keys():
#             print('init fish',k,Skylens_kwargs[k])
#             print(get_size_pickle(Skylens_kwargs[k]))
        kappa_class=Skylens(**Skylens_kwargs)
        Skylens_kwargs['use_window']=use_window
        print('init fish, use window',use_window,Skylens_kwargs['store_win'])
        #Following is not used unless use_window=False. Note that for z_bins, we always have use_window=True.
        #Easier to keep track of f_sky overlaps.
        f_sky={}
        sc={}
        kappa_class.gather_data()
        zkernels={}
        for tracer in ['galaxy','shear']:
            zkernels[tracer]=kappa_class.tracer_utils.set_kernels(Ang_PS=kappa_class.Ang_PS,delayed_compute=False,tracer=tracer)
        for corr in z_bins_kwargs['corr_indxs']:
            indxs=z_bins_kwargs['corr_indxs'][corr]
            f_sky[corr]={}
            f_sky[corr[::-1]]={}
            sc[corr]={}
            for (i,j) in indxs:
                zs1=kappa_class.tracer_utils.z_win[corr[0]][i]#.copy() #we will modify these locally
                zs2=kappa_class.tracer_utils.z_win[corr[1]][j]
                f_sky_ij,mask12=mask_comb(zs1['window'],zs2['window'])
                zs1=kappa_class.z_bins[corr[0]][i]#.copy() #we will modify these locally
                zs2=kappa_class.z_bins[corr[1]][j]
                zs1=zkernels[corr[0]][i]#.copy() #we will modify these locally
                zs2=zkernels[corr[1]][j]
                sc_ij=np.sum(zs1['kernel_int']*zs2['kernel_int'])
                if np.isnan(sc_ij):
                    print(corr,i,j,'nan',sc_ij,zs1['kernel_int'],zs2['kernel_int'])
                if f_sky_ij==0 or sc_ij==0:
                    print('Fish init: ',corr,(i,j),'removed because fsky=',f_sky_ij,' kernel product=',sc_ij)
                    z_bins_kwargs['corr_indxs'][corr].remove((i,j))
                    try:
                        z_bins_kwargs['corr_indxs'][corr[::-1]].remove((j,i))
                    except:
                        pass
                else:
                    f_sky[corr][(i,j)]=f_sky_ij
                    f_sky[corr[::-1]][(j,i)]=f_sky_ij
                    sc[corr][(i,j)]=sc_ij
        if not use_window:
            for corr in kappa_class.cov_indxs:#FIXME: should be parallelized.
                f_sky[corr]={}
                for (i,j,k,l) in kappa_class.cov_indxs[corr]:
                    zs1=kappa_class.tracer_utils.z_win[corr[0]][i]#.copy() #we will modify these locally
                    zs2=kappa_class.tracer_utils.z_win[corr[1]][j]
                    zs3=kappa_class.tracer_utils.z_win[corr[2]][k]#.copy() #we will modify these locally
                    zs4=kappa_class.tracer_utils.z_win[corr[3]][l]
                    f_sky_12,mask12=mask_comb(zs1['window'],zs2['window'])
                    f_sky_12,mask34=mask_comb(zs3['window'],zs4['window'])

                    f_sky_1234,mask1234=mask_comb(mask12,mask34)
                    f_sky[corr][(i,j,k,l)]=f_sky_1234
        z_bins_kwargs['f_sky']=f_sky
        z_bins_kwargs['kernel_product']=sc
        kappa_class.gather_data()
        del kappa_class
    else:
        print('fisher_init using provided zbins',z_bins_kwargs['shear_zbins'][0]['window'].shape,z_bins_kwargs['shear_zbins'][0]['window'].mask.shape)
        shear_zbins_comb,galaxy_zbins_comb=combine_z_bins_all(z_bins_kwargs=z_bins_kwargs)
        Skylens_kwargs['galaxy_zbins']=galaxy_zbins_comb
        Skylens_kwargs['shear_zbins']=shear_zbins_comb
    
    Skylens_kwargs['f_sky']=None
    if not use_window:
        Skylens_kwargs['f_sky']=z_bins_kwargs['f_sky']
    Skylens_kwargs['stack_indxs']=z_bins_kwargs['corr_indxs']
    Skylens_kwargs['Win']=Win
    print('fisher_init getting final skylens')
    kappa_class=Skylens(**Skylens_kwargs)
    return kappa_class,z_bins_kwargs

def DESI_z_bins(datasets=['elg','lrg','BG','qso']
                ,nbins=None,**kwargs):
    zbins={}
    i=0
    for d in datasets:
        if nbins[d]<1:
            continue
        d_bins=DESI_lens_bins(dataset=d,nbins=nbins[d],**kwargs)
        if i==0:
            zbins=copy.deepcopy(d_bins)
        else:
            zbins=combine_zbins(z_bins1=zbins,z_bins2=d_bins)
        i+=1
        print(i,zbins['n_bins'],d_bins['n_bins'])
    return zbins

def sigma_photoz(z_bin={}):
    zm=(z_bin['z']*z_bin['pzdz']).sum()
    z=z_bin['z']
    ddz=np.absolute(z-zm)
    p=z_bin['nz']*(-0.1*(1+ddz) + 0.12*(1+ddz)**2+.3/np.sqrt(z_bin['ns']*z_bin['nz']))
    x=np.isnan(p)
    p[x]=z_bin['nz'].max()*0.01
    return p
