import os,sys

from power_spectra import *
from hankel_transform import *
from binning import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d

d2r=np.pi/180.
c=c.to(u.km/u.second)

class Angular_power_spectra():
    def __init__(self,silence_camb=False,l=np.arange(2,2001),power_spectra_kwargs={},
                zl=None,n_zl=100,SSV_cov=False,tracer='kappa',cov_utils=None):
        self.PS=Power_Spectra(silence_camb=silence_camb,SSV_cov=SSV_cov,**power_spectra_kwargs)
        self.l=l
        self.tracer=tracer
        
        self.SSV_cov=SSV_cov

        self.DC=None #these should be cosmology depdendent. set to none before when varying cosmology
        self.clz=None
        self.cov_utils=cov_utils
        self.zl=zl

    
    def angular_power_z(self,z=None,pk_params=None,cosmo_h=None,
                    cosmo_params=None,pk_func=None):
        """
             This function outputs p(l=k/chi,z) / chi(z)^2, where z is the lens redshifts. The shape of the output is l,n_z, where n_z is the number of z bins.
        """
        if self.clz is not None:
            return 
        if cosmo_h is None:
            cosmo_h=self.PS.cosmo_h
        
        l=self.l

        if z is None:
            z=self.zl

        nz=len(z)
        nl=len(l)

        #XXX At some point this should be moved to power spectra, pk and SSV, especially if doing 3X2
        if self.PS.pk is None:
            self.PS.get_pk(z=z,pk_params=pk_params,cosmo_params=cosmo_params)
        cls=np.zeros((nz,nl),dtype='float32')#*u.Mpc#**2

        Rls=None #pk response functions, used for SSV calculations
        RKls=None
        cls_lin=None #cls from linear power spectra, to compute \delta_window for SSV
        if self.SSV_cov: #things needed to compute SSV cov
            Rls=np.zeros((nz,nl),dtype='float32')
            RKls=np.zeros((nz,nl),dtype='float32')
            cls_lin=np.zeros((nz,nl),dtype='float32')#*u.Mpc#**2

        cH=c/(cosmo_h.efunc(self.zl)*cosmo_h.H0)
        cH=cH.value

        def k_to_l(l,lz,f_k): #take func from k to l space
            fk_int=interp1d(lz,f_k,bounds_error=False,fill_value=0)
            return fk_int(l)

        kh=self.PS.kh
        pk=self.PS.pk
        
        for i in np.arange(nz):
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value#because camb k in h/mpc
            lz=kh*DC_i-0.5
            cls[i][:]+=k_to_l(l,lz,pk[i]/DC_i**2)
            if self.SSV_cov:
                Rls[i][:]+=k_to_l(l,lz,self.PS.R1[i]) 
                RKls[i][:]+=k_to_l(l,lz,self.PS.Rk[i]) 
                cls_lin[i][:]+=k_to_l(l,lz,self.PS.pk_lin[i]/DC_i**2)
        
        f=(l+0.5)**2/(l*(l+1.)) # cl correction from Kilbinger+ 2017
            #cl*=2./np.pi #comparison with CAMB requires this.
        self.clz={'cls':cls,'l':l,'cH':cH,'f':f}
        if self.SSV_cov:
            self.cov_utils.sigma_win_calc(cls_lin=cls_lin)
            self.clz.update({'clsR':cls*Rls,'clsRK':cls*RKls})