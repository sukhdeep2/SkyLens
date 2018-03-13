import camb
from camb import model, initialpower
import os,sys
#import pyccl

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
#from astropy.constants import c,G
from astropy import units as u
from scipy.integrate import quad as scipy_int1d

cosmo_h=cosmo.clone(H0=100)
#c=c.to(u.km/u.second)

cosmo_fid=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'As':2.12e-09,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965})
pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':5000}

class Power_Spectra():
    def __init__(self,cosmo_params=cosmo_fid,pk_params=pk_params,cosmo=cosmo,cosmo_h=None,
                 silence_camb=False):
        self.cosmo_params=cosmo_params
        self.pk_params=pk_params
        self.cosmo=cosmo
        self.silence_camb=silence_camb

        if not cosmo_h:
            self.cosmo_h=cosmo.clone(H0=100)
        else:
            self.cosmo_h=cosmo_h

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

    def ccl_pk(self,z,cosmo_params=None,pk_params=None):
        if not cosmo_params:
            cosmo_params=self.cosmo_params
        if not pk_params:
            pk_params=self.pk_params

        cosmo_ccl=pyccl.Cosmology(h=cosmo_params['h'],Omega_c=cosmo_params['Omd'],Omega_b=cosmo_params['Omb'],
                              A_s=cosmo_params['As'],n_s=cosmo_params['ns'],m_nu=cosmo_params['mnu'])
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
                print ('CCL err',err,z[i])
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

        #stdout=np.copy(sys.stdout)
        #sys.stdout = open(os.devnull, 'w')
        if self.silence_camb:
            sys.stdout = open(os.devnull, 'w')
        pars.InitPower.set_params(ns=cosmo_params['ns'], r=0,As =cosmo_params['As']) #
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

    def camb_pk_too_many_z(self,z,cosmo_params=None,pk_params=None):
        i=0
        pk=None #np.array([])
        z_step=140 #camb cannot handle more than 150 redshifts
        nz=len(z)
        
        while i<nz:
            pki,kh=self.camb_pk(z=z[i:i+z_step],pk_params=pk_params,cosmo_params=cosmo_params,return_s8=False)
            pk=np.vstack((pk,pki)) if pk is not None else pki
            i+=z_step
        return pk,kh

    def R1(self,k=None,pk=None,k_NonLinear=3.2,axis=0): #eq 2.5, R1, Barriera+ 2017
        G1=26./21.*np.ones_like(k)
        x=k>k_NonLinear
        G1[x]*=k_NonLinear/k[x]
        dpk=np.gradient(np.log(pk),axis=axis)/np.gradient(np.log(k),axis=0)
        R=1-1./3*dpk+G1
        return R

    def R_K(self,k=None,pk=None,k_NonLinear=3.2,axis=0): #eq 2.5, R1
        G1=26./21.*np.ones_like(k)
        x=k>k_NonLinear
        G1[x]*=k_NonLinear/k[x]
        dpk=np.gradient(np.log(pk),axis=axis)/np.gradient(np.log(k),axis=0)
        R=12./13*G1-dpk
        return R

if __name__ == "__main__":
    PS=Power_Spectra()
