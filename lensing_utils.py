import os,sys

from power_spectra import *
from angular_power_spectra import *
from hankel_transform import *
from binning import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d

d2r=np.pi/180.
c=c.to(u.km/u.second)

class Lensing_utils():
    def __init__(self,sigma_gamma=0.3):
        #self.ns=ns
        ns=1 #different for different z_source. should be property of z_source
        self.sigma_gamma=sigma_gamma
        self.SN0=sigma_gamma**2/(ns*3600./d2r**2)   
        #Gravitaional const to get Rho crit in right units
        self.G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2) 
        self.G2*=8*np.pi/3.     
    
    def Rho_crit(self,cosmo_h=None):
        #G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        #rc=3*cosmo_h.H0**2/(8*np.pi*G2)
        rc=cosmo_h.H0**2/(self.G2) #factors of pi etc. absorbed in self.G2
        rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
        return rc.value

    def sigma_crit(self,zl=[],zs=[],cosmo_h=None):
        ds=cosmo_h.comoving_transverse_distance(zs)
        dl=cosmo_h.comoving_transverse_distance(zl)
        ddls=1.-np.multiply.outer(1./ds,dl)#(ds-dl)/ds
        w=(3./2.)*((cosmo_h.H0/c)**2)*(1+zl)*dl/self.Rho_crit(cosmo_h) 
        sigma_c=1./(ddls*w)
        x=ddls<=0 #zs<zl
        sigma_c[x]=np.inf
        return sigma_c.value

    def shape_noise_calc(self,zs1=None,zs2=None):
        if not np.array_equal(zs1['z'],zs2['z']):
            return 0
        if np.any(np.isinf(zs1['nz'])) or np.any(np.isinf(zs2['nz'])):
            return 0
        SN=self.SN0*np.sum(zs1['W']*zs2['W']*zs1['nz']) #FIXME: this is probably wrong.
        #Assumption: ns(z)=ns*pzs*dzs
        SN/=np.sum(zs1['nz']*zs1['W'])
        SN/=np.sum(zs2['nz']*zs2['W'])
        return SN
        # XXX Make sure pzs are properly normalized