import camb
from camb import model, initialpower
#import pyccl
import os,sys
from classy import Class
sys.path.insert(0,'./')

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
from scipy.integrate import quad as scipy_int1d
import pandas as pd
from scipy import interpolate


cosmo_h=cosmo.clone(H0=100)
#c=c.to(u.km/u.second)

cosmo_fid=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'w':-1,'wa':0})
cosmo_fid['Oml']=1.-cosmo_fid['Om']-cosmo_fid['Omk']
pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':5000,'scenario':'eagle'}

# baryonic scenario option:
# "owls_AGN","owls_DBLIMFV1618","owls_NOSN","owls_NOSN_NOZCOOL","owls_NOZCOOL","owls_REF","owls_WDENS"
# "owls_WML1V848","owls_WML4","illustris","mb2","eagle","HzAGN"


Bins_z_HzAGN   = np.array([4.9285,4.249,3.7384,3.33445,
                           3.00295,1.96615,1.02715,0.519195,0.22878,0.017865,0.0])
Bins_z_mb2     = np.array([3.5,3.25,2.8,2.45,2.1,2.0,1.8,1.7,1.6,1.4,1.2,1.1,1.0,0.8,0.7,
                           0.6,0.4,0.35,0.2,0.0625,0.0])
Bins_z_ill1    = np.array([3.5,3.49,3.28,3.08,2.90,2.73,2.44,2.1,2.0,1.82,1.74,1.6,1.41,
                           1.21,1.04,1.0,0.79,0.7,0.6,0.4,0.35,0.2,0.0])
Bins_z_eagle   = np.array([3.53,3.02,2.48,2.24,2.01,1.74,1.49,1.26,1.0,0.74,0.5,0.27,0.0])
Bins_z_OWLS    = np.array([3.5,3.25,3.0,2.75,2.25,2.00,1.75,1.50,1.25,
                           1.00,0.75,0.50,0.375,0.25,0.125,0.0])
Bins_z_NOSN    = np.array([3.5,3.25,3.0,2.75,2.25,2.00,1.75,1.50,
                           1.25,1.00,0.75,0.50,0.375,0.25,0.0])

zbin_logPkR = {"owls_AGN":Bins_z_OWLS,
               "owls_DBLIMFV1618":Bins_z_OWLS,
               "owls_NOSN":Bins_z_NOSN,
               "owls_NOSN_NOZCOOL":Bins_z_OWLS,
               "owls_NOZCOOL":Bins_z_OWLS,
               "owls_REF":Bins_z_OWLS,
               "owls_WDENS":Bins_z_OWLS,
               "owls_WML1V848":Bins_z_OWLS,
               "owls_WML4":Bins_z_OWLS,
               "illustris":Bins_z_ill1,
               "mb2":Bins_z_mb2,
               "eagle":Bins_z_eagle,
               "HzAGN":Bins_z_HzAGN
              }


class_accuracy_settings={ #from Vanessa. To avoid class errors due to compiler issues.
                          #https://github.com/lesgourg/class_public/issues/193
            "k_min_tau0":0.002, #you could try change this
            "k_max_tau0_over_l_max":3., #you can also try 5 here
            "k_step_sub":0.015,
            "k_step_super":0.0001,
            "k_step_super_reduction":0.1,
            'k_per_decade_for_pk': 20,
#             'k_output_values': 2.0,
            'perturb_sampling_stepsize':0.01,
            'tol_perturb_integration':1.e-4,
            'halofit_k_per_decade': 3000. #you could try change this
            }

class Power_Spectra():
    def __init__(self,cosmo_params=cosmo_fid,pk_params=pk_params,cosmo=cosmo,
                 silence_camb=False,pk_func=None,SSV_cov=False,scenario=None,
                 logger=None):
        self.logger=logger
        self.cosmo_params=cosmo_params
        self.pk_params=pk_params
        self.cosmo=cosmo
        self.silence_camb=silence_camb
        self.cosmo_h=cosmo.clone(H0=100)
        #self.pk_func=self.camb_pk_too_many_z if pk_func is None else pk_func
        #self.pk_func=self.ccl_pk if pk_func is None else pk_func
        self.pk_func=self.class_pk if pk_func is None else getattr(self,pk_func)
        self.SSV_cov=SSV_cov
        self.scenario = scenario
        self.pk=None
        if not pk_params is None:
            self.kh=np.logspace(np.log10(pk_params['kmin']),np.log10(pk_params['kmax']),
            pk_params['nk'])


    def get_pk(self,z,cosmo_params=None,pk_params=None,return_s8=False):
        pk_func=self.pk_func

        if pk_params is not None:
            if pk_params.get('pk_func'):
                pk_func=getattr(self,pk_params['pk_func'])
        if return_s8:
            self.pk,self.kh,self.s8=pk_func(z,cosmo_params=cosmo_params,
                            pk_params=pk_params,return_s8=return_s8)
        else:
            self.pk,self.kh=pk_func(z,cosmo_params=cosmo_params,
                            pk_params=pk_params,return_s8=return_s8)
        if self.SSV_cov:
            self.get_SSV_terms(z,cosmo_params=cosmo_params,
                            pk_params=pk_params)

    def get_SSV_terms(self,z,cosmo_params=None,pk_params=None):
        pk_params_lin=self.pk_params.copy() if pk_params is None else pk_params.copy()
        pk_params_lin['non_linear']=0
        self.pk_lin,self.kh=self.pk_func(z,cosmo_params=cosmo_params,
                        pk_params=pk_params_lin,return_s8=False)
        self.R1=self.R1_calc(k=self.kh,pk=self.pk_lin,axis=1)
        self.Rk=self.R_K_calc(k=self.kh,pk=self.pk_lin,axis=1)


    def DZ_int(self,z=[0],cosmo=None): #linear growth factor.. full integral.. eq 63 in Lahav and suto
        if not cosmo:
            cosmo=self.cosmo
        def intf(z):
            return (1+z)/(cosmo.H(z).value)**3
        j=0
        Dz=np.zeros_like(z,dtype='float32')

        for i in z:
            Dz[j]=cosmo.H(i).value*scipy_int1d(intf,i,np.inf,epsrel=1.e-6,epsabs=1.e-6)[0]
            j=j+1
        Dz*=(2.5*cosmo.Om0*cosmo.H0.value**2)
        return Dz/Dz[0]

    def ccl_pk(self,z,cosmo_params=None,pk_params=None,return_s8=False):
        if not cosmo_params:
            cosmo_params=self.cosmo_params
        if not pk_params:
            pk_params=self.pk_params

        cosmo_ccl=pyccl.Cosmology(h=cosmo_params['h'],Omega_c=cosmo_params['Omd'],
                                Omega_b=cosmo_params['Omb'],
                                A_s=cosmo_params['Ase9']*1.e-9,n_s=cosmo_params['ns'],
                                m_nu=cosmo_params['mnu'])
        kh=np.logspace(np.log10(pk_params['kmin']),np.log10(pk_params['kmax']),pk_params['nk'])
        nz=len(z)
        ps=np.zeros((nz,pk_params['nk']))
        ps0=[]
        z0=9.#PS(z0) will be rescaled using growth function when CCL fails.

        pyccl_pkf=pyccl.linear_matter_power
        if pk_params['non_linear']==1:
            pyccl_pkf=pyccl.nonlin_matter_power
        for i in np.arange(nz):
            try:
                ps[i]= pyccl_pkf(cosmo_ccl,kh,1./(1+z[i]))
            except Exception as err:
                self.logger.error ('CCL err %s %s',err,z[i])
                if not np.any(ps0):
                    ps0=pyccl.linear_matter_power(cosmo_ccl,kh,1./(1.+z0))
                Dz=self.DZ_int(z=[z0,z[i]])
                ps[i]=ps0*(Dz[1]/Dz[0])**2
        return ps*cosmo_params['h']**3,kh/cosmo_params['h'] #factors of h to get in same units as camb output

    def camb_pk(self,z,cosmo_params=None,pk_params=None,return_s8=False):
        #Set up a new set of parameters for CAMB
        if cosmo_params is None:
            cosmo_params=self.cosmo_params
        if pk_params is None:
            pk_params=self.pk_params

        pars = camb.CAMBparams()
        h=cosmo_params['h']

        pars.set_cosmology(H0=h*100,
                            ombh2=cosmo_params['Omb']*h**2,
                            omch2=(cosmo_params['Om']-cosmo_params['Omb'])*h**2,
                            mnu=cosmo_params['mnu'],tau=cosmo_params['tau']
                            ) #    omk=cosmo_params['Omk'], )

        if cosmo_params['w']!=-1:
            pars.set_dark_energy(cosmo_params['w'])

        #stdout=np.copy(sys.stdout)
        #sys.stdout = open(os.devnull, 'w')
        if self.silence_camb:
            sys.stdout = open(os.devnull, 'w')
        pars.InitPower.set_params(ns=cosmo_params['ns'], r=0,As =cosmo_params['Ase9']*1.e-9) #
        if return_s8:
            z_t=np.sort(np.unique(np.append([0],z).flatten()))
        else:
            z_t=np.array(z)
        pars.set_matter_power(redshifts=z_t,kmax=pk_params['kmax'])
        if self.silence_camb:
            sys.stdout = sys.__stdout__
        #sys.stdout = sys.__stdout__
        #sys.stdout=stdout

        if pk_params['non_linear']==1:
            pars.NonLinear = model.NonLinear_both
        else:
            pars.NonLinear = model.NonLinear_none

        results = camb.get_results(pars) #This is the time consuming part.. pk add little more (~5%).. others are negligible.

        kh, z2, pk =results.get_matter_power_spectrum(minkh=pk_params['kmin'],
                                                        maxkh=pk_params['kmax'],
                                                        npoints =pk_params['nk'])
        if not np.all(z2==z_t):
            raise Exception('CAMB changed z order',z2,z_mocks)

        if return_s8:
            s8=results.get_sigma8()
            if len(s8)>len(z):
                return pk[1:],kh,s8[-1]
            else:
                return pk,kh,s8[-1]
        else:
            return pk,kh

    def camb_pk_too_many_z(self,z,cosmo_params=None,pk_params=None,return_s8=False):
        i=0
        pk=None #np.array([])
        z_step=140 #camb cannot handle more than 150 redshifts
        nz=len(z)

        while i<nz:
            p=self.camb_pk(z=z[i:i+z_step],pk_params=pk_params,cosmo_params=cosmo_params,return_s8=return_s8)
            pki=p[0]
            kh=p[1]
            pk=np.vstack((pk,pki)) if pk is not None else pki
            i+=z_step

        if return_s8:
            return pk,kh,p[2]
        return pk,kh

    def class_pk(self,z,cosmo_params=None,pk_params=None,return_s8=False):
        if  cosmo_params is None:
            cosmo_params=self.cosmo_params
        if pk_params is None:
            pk_params=self.pk_params
        cosmoC=Class()
        h=cosmo_params['h']
        class_params={'h':h,'omega_b':cosmo_params['Omb']*h**2,
                            'omega_cdm':(cosmo_params['Om']-cosmo_params['Omb'])*h**2,
                            'A_s':cosmo_params['Ase9']*1.e-9,'n_s':cosmo_params['ns'],
                            'output': 'mPk','z_max_pk':100, #max(z)*2, #to avoid class error.
                                                      #Allegedly a compiler error, whatever that means
                            'P_k_max_1/Mpc':pk_params['kmax']*h*1.1,
                    }
        if pk_params['non_linear']==1:
            class_params['non linear']='halofit'

        class_params['N_ur']=3.04 #ultra relativistic species... neutrinos
        if cosmo_params['mnu']!=0:
            class_params['N_ur']-=1 #one massive neutrino
            class_params['m_ncdm']=cosmo_params['mnu']
        class_params['N_ncdm']=3.04-class_params['N_ur']

        if cosmo_params['w']!=-1 or cosmo_params['wa']!=0:
            class_params['Omega_fld']=cosmo_params['Oml']
            class_params['w0_fld']=cosmo_params['w']
            class_params['wa_fld']=cosmo_params['wa']


        for ke in class_accuracy_settings.keys():
            class_params[ke]=class_accuracy_settings[ke]

        cosmoC=Class()
        cosmoC.set(class_params)
        try:
            cosmoC.compute()
        except Exception as err:
            print(class_params, err)
            raise Exception('Class crashed')

        k=self.kh*h
        pkC=np.array([[cosmoC.pk(ki,zj) for ki in k ]for zj in z])
        pkC*=h**3
        s8=cosmoC.sigma8()
        cosmoC.struct_cleanup()
        if return_s8:
            return pkC,self.kh,s8
        else:
            return pkC,self.kh

    def baryon_pk(self,z,cosmo_params=None,pk_params=None,return_s8=False):

        if pk_params is None:
            pk_params=self.pk_params
        if cosmo_params is None:
            cosmo_params=self.cosmo_params
        scenario = pk_params['scenario']

        out=self.class_pk(z,cosmo_params=cosmo_params,pk_params=pk_params,return_s8=return_s8)
        pk=out[0]
        kh=out[1]

        non_linear=pk_params['non_linear'] if pk_params.get('non_linear') is not None else 1

        if scenario is None or scenario is 'dmo' or non_linear==0:
            if return_s8:
                s8=out[2]
                return pk,kh,s8
            else:
                return pk,kh

        data_dir = "./Pk_ratio/"
        infile_logPkRatio = data_dir + "logPkRatio_" + scenario + ".dat"
        Arr2D_logPkratio = pd.read_csv(infile_logPkRatio,sep='\s+')
        Bins_log_k_Mpc = np.array(Arr2D_logPkratio["logk"])
        del Arr2D_logPkratio["logk"]
        Arr2D_logPkratio = np.array(Arr2D_logPkratio)
        f_logPkRatio = interpolate.interp2d(zbin_logPkR[scenario], Bins_log_k_Mpc, Arr2D_logPkratio, kind='linear')

        PkR    = 10**(f_logPkRatio(z,np.log10(kh)))
        pkbary = pk*PkR.T

        if return_s8:
            s8=out[2]
            return pkbary,kh,s8
        else:
            return pkbary,kh  #,pk,PkR.T


    def R1_calc(self,k=None,pk=None,k_NonLinear=3.2,axis=0): #eq 2.5, R1, Barriera+ 2017
        G1=26./21.*np.ones_like(k)
        x=k>k_NonLinear
#         G1[x]*=k_NonLinear/k[x]
        B1=0.75
        G1[x]=B1+(G1[x]-B1)*(k_NonLinear/k[x])**0.5
        dpk=np.gradient(np.log(pk),axis=axis)/np.gradient(np.log(k),axis=0)
        R=1 - 1./3*dpk + G1
        return R

    def R_K_calc(self,k=None,pk=None,k_NonLinear=3.2,axis=0): #eq 2.5, R1
        GK=8./7.*np.ones_like(k)
        x=k>k_NonLinear
        BK=2.2
        GK[x]=BK+(GK[x]-BK)*(k_NonLinear/k[x])**0.5
        dpk=np.gradient(np.log(pk),axis=axis)/np.gradient(np.log(k),axis=0)
#         R=12./13*G1-dpk
        R=GK-dpk
        return R

    def reset(self):
        self.pk=None
        self.s8=None

if __name__ == "__main__":
    PS=Power_Spectra()
