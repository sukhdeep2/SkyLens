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
                z_PS=None,nz_PS=100,log_z_PS=False,z_PS_max=None,logger=None,window_l=None,
                SSV_cov=False,tracer='kappa',cov_utils=None):
        self.logger=logger
        self.PS=Power_Spectra(silence_camb=silence_camb,SSV_cov=SSV_cov,**power_spectra_kwargs)
        self.l=l
        self.window_l=window_l
        self.cl_f=(l+0.5)**2/(l*(l+1.)) # cl correction from Kilbinger+ 2017
        self.tracer=tracer

        self.SSV_cov=SSV_cov

        self.DC=None #these should be cosmology depdendent. set to none before when varying cosmology
        self.clz=None
        self.cov_utils=cov_utils
        self.set_z_PS(z=z_PS,nz=nz_PS,log_z=log_z_PS,z_max=z_PS_max)

        self.dz=np.gradient(self.z)

    def set_z_PS(self,z=None,nz=10,log_z=False,z_max=None):
        """
            Define redshifts where we compute the matter power spectra.
            These can be input when intializing the class or set here.
            Can be set in log or linear space.
            Redshift range here will be from z=0 to z_max
        """
        z_min=0
        if z_max is None:
            z_max=3

        if z is None:
            if log_z:#bins for z_lens.
                self.z=np.logspace(np.log10(max(z_min,1.e-4)),np.log10(z_max),nz)
            else:
                self.z=np.linspace(z_min,z_max,nz)
        else:
            self.z=z
        self.dz=np.gradient(self.z)

    def angular_power_z(self,z=None,pk_params=None,cosmo_h=None,
                    cosmo_params=None,pk_func=None):
        """
             This function outputs p(l=k/chi,z) / chi(z)^2, where z is the lens redshifts.
             The shape of the output is l,nz, where nz is the number of z bins.
        """

        if cosmo_h is None:
            cosmo_h=self.PS.cosmo_h

        l=self.l

        z=self.z

        nz=len(z)
        nl=len(l)

        self.PS.get_pk(z=z,pk_params=pk_params,cosmo_params=cosmo_params)
        cls=np.zeros((nz,nl),dtype='float32')#*u.Mpc#**2

        Rls=None #pk response functions, used for SSV calculations
        RKls=None
        cls_lin=None #cls from linear power spectra, to compute \delta_window for SSV
        if self.SSV_cov: #things needed to compute SSV cov
            nl_w=len(self.window_l)
            Rls=np.zeros((nz,nl),dtype='float32')
            RKls=np.zeros((nz,nl),dtype='float32')
            cls_lin=np.zeros((nz,nl_w),dtype='float32')#*u.Mpc#**2

        cH=c/(cosmo_h.efunc(self.z)*cosmo_h.H0)
        cH=cH.value

        def k_to_l(l,lz,f_k): #take func from k to l space
            fk_int=interp1d(lz,f_k,bounds_error=False,fill_value=0)
            return fk_int(l)

        kh=self.PS.kh
        pk=self.PS.pk
        chi=cosmo_h.comoving_transverse_distance(z).value
        for i in np.arange(nz):
            DC_i=chi[i] #cosmo_h.comoving_transverse_distance(z[i]).value#because camb k in h/mpc
            lz=kh*DC_i-0.5
            cls[i][:]+=k_to_l(l,lz,pk[i]/DC_i**2)
            if self.SSV_cov:
                Rls[i][:]+=k_to_l(l,lz,self.PS.R1[i])
                RKls[i][:]+=k_to_l(l,lz,self.PS.Rk[i])
                cls_lin[i][:]+=k_to_l(self.window_l,lz,self.PS.pk_lin[i]/DC_i**2)


            #cl*=2./np.pi #comparison with CAMB requires this.
        self.clz={'cls':cls,'l':l,'cH':cH,'dchi':cH*self.dz,'chi':chi,'dz':self.dz}
        if self.SSV_cov:
            self.cov_utils.sigma_win_calc(cls_lin=cls_lin)
            self.clz.update({'clsR':cls*Rls,'clsRK':cls*RKls})
            
    def reset(self):
        self.clz=None
        self.PS.reset()
