"""
Class with utility functions for defining tracer properties, e.g. redshift kernels, galaxy bias, IA amplitude etc.
"""

import os,sys
import copy
from skylens import *
#from skylens.power_spectra import *
#from angular_power_spectra import *
#from hankel_transform import *
#from binning import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import loggamma

d2r=np.pi/180.
c=c.to(u.km/u.second)

class Tracer_utils():
    def __init__(self,zs_bins=None,zg_bins=None,zk_bins=None,logger=None,l=None,
                ):
        self.logger=logger
        self.l=l
        #Gravitaional const to get Rho crit in right units
        self.G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        self.G2*=8*np.pi/3.
    
        self.SN={}

        self.n_bins={}
        self.z_bins={}
        self.spin={'galaxy':0,'kappa':0,'shear':2}
        
        self.set_zbins(z_bins=zs_bins,tracer='shear')
        self.set_zbins(z_bins=zg_bins,tracer='galaxy')
        self.set_zbins(z_bins=zk_bins,tracer='kappa')
        self.tracers=list(self.z_bins.keys())
        
        if not self.tracers==[]:
            self.set_z_PS_max()
        
        
    def set_z_PS_max(self):
        """
            Get max z for power spectra computations.
        """
        self.z_PS_max=0
        z_max_all=np.array([self.z_bins[tracer]['zmax'] for tracer in self.tracers])
        self.z_PS_max=np.amax(z_max_all).item() +0.01

    def set_zbins(self,z_bins={},tracer=None):
        """
        Set tracer z_bins as class property.
        """
        if z_bins is not None:
            self.z_bins[tracer]=copy.deepcopy(z_bins)
            self.n_bins[tracer]=self.z_bins[tracer]['n_bins']
            self.set_noise(tracer=tracer)
    
    def Rho_crit(self,cosmo_h=None):
        #G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        #rc=3*cosmo_h.H0**2/(8*np.pi*G2)
        rc=cosmo_h.H0**2/(self.G2) #factors of pi etc. absorbed in self.G2
        rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
        return rc.value

    def sigma_crit(self,zl=[],zs=[],cosmo_h=None):
        """
        Inverse of lensing kernel.
        """
        ds=cosmo_h.comoving_transverse_distance(zs)
        dl=cosmo_h.comoving_transverse_distance(zl)
        ddls=1.-np.multiply.outer(1./ds,dl)#(ds-dl)/ds
        w=(3./2.)*((cosmo_h.H0/c)**2)*(1+zl)*dl/self.Rho_crit(cosmo_h)
        sigma_c=1./(ddls*w)
        x=ddls<=0 #zs<zl
        sigma_c[x]=np.inf
        return sigma_c.value

    def get_z_bins(self,tracer=None):
        return self.z_bins[tracer]
        # if tracer=='shear':
        #      return self.zs_bins
        # if tracer=='galaxy':
        #     return self.zg_bins
        # if tracer=='kappa':
        #     return self.zk_bins

    def set_noise(self,tracer=None):
        """
        Setting the noise of the tracers. We assume noise is in general a function of ell.
        """
        z_bins=self.get_z_bins(tracer=tracer)
        n_bins=z_bins['n_bins']
        self.SN[tracer]=np.zeros((len(self.l),n_bins,n_bins)) #if self.do_cov else None

        for i in np.arange(n_bins):
            self.SN[tracer][:,i,i]+=z_bins['SN'][tracer][:,i,i]

    def NLA_amp_z(self,z=[],z_bin={},cosmo_h=None):
        """
        Redshift dependent intrinsic alignment amplitude. This is assumed to be a function of ell in general, 
        though here we set it as constant.
        """
        AI=z_bin['AI']
        AI_z=z_bin['AI_z']
        return np.outer(AI*(1+z)**AI_z,np.ones_like(self.l)) #FIXME: This might need to change to account
    
    
    def constant_bias(self,z=[],z_bin={},cosmo_h=None):
        """
        Galaxy bias, assumed to be constant (ell and z independent).
        """
        b=z_bin['b1']
        lb_m=z_bin['lm']
        lm=np.ones_like(self.l)
#         x=self.l>lb_m  #if masking out modes based on kmax. lm is attribute of the z_bins, that is based on kmax.
#         lm[x]=0        
        return b*np.outer(np.ones_like(z),lm)

    def linear_bias_z(self,z=[],z_bin={},cosmo_h=None):
        """
        linear Galaxy bias, assumed to be constant in ell and specified at every z.
        """
#         b=z_bin['bz1']
        b=np.interp(z,z_bin['z'],z_bin['bz1'],left=0,right=0) #this is linear interpolation
        
        lb_m=z_bin['lm']
        lm=np.ones_like(self.l)
#         x=self.l>lb_m  #if masking out modes based on kmax. lm is attribute of the z_bins, that is based on kmax.
#         lm[x]=0        
        return np.outer(b,lm)

    def linear_bias_powerlaw(self,z_bin={},cosmo_h=None):
        """
        Galaxy bias, assumed to be constant (ell independent). Varies as powerlaw with redshift. This is useful
        for some photo-z tests.
        """
        b1=z_bin['b1']
        b2=z_bin['b2']
        lb_m=z_bin['lm']
        lm=np.ones_like(self.l)+self.l/self.l[-1]
#         x=self.l>lb_m
#         lm[x]=0        
        return np.outer(b1*(1+z_bin['z'])**b2,lm) #FIXME: This might need to change to account
    
    def spin_factor(self,l=None,tracer=None):
        """
        Spin of tracers. Needed for wigner transforms and pseudo-cl calculations.
        """
        if l is None:
            l=self.l
        if tracer is None:
            return np.nan
        if tracer=='galaxy':
            s=0
            return np.ones_like(l,dtype='float32')
        
        if tracer=='shear':
            s=2 #there is (-1)**s factor, so sign is same of +/- 2 spin
        if tracer=='kappa':
            s=1  #see Kilbinger+ 2017
        
        F=loggamma(l+s+1)
        F-=loggamma(l-s+1)
        F=np.exp(1./s*F) # units should be l**2, hence sqrt for s=2... comes from angular derivative of the potential
        F/=(l+0.5)**2 #when writing potential to delta_m, we get 1/k**2, which then results in (l+0.5)**2 factor
        x=l-s<0
        F[x]=0

        return F.astype('float32')
    
    def set_kernel(self,cosmo_h=None,zl=None,tracer=None):
        """
        Set the tracer kernels. This includes the local kernel, e.g. galaxy density, IA and also the lensing 
        kernel. Galaxies have magnification bias.
        """
        self.set_lensing_kernel(cosmo_h=cosmo_h,zl=zl,tracer=tracer)
        self.set_galaxy_kernel(cosmo_h=cosmo_h,zl=zl,tracer=tracer)
        z_bins=self.get_z_bins(tracer=tracer)
        n_bins=z_bins['n_bins']
        for i in np.arange(n_bins):
            z_bins[i]['kernel_int']=z_bins[i]['Gkernel_int']+z_bins[i]['gkernel_int']
        
    def set_lensing_kernel(self,cosmo_h=None,zl=None,tracer=None):
        """
            Compute rho/Sigma_crit for each source bin at every lens redshift where power spectra is computed.
            cosmo_h: cosmology to compute Sigma_crit
        """
        #We need to compute these only once in every run
        # i.e not repeat for every ij combo
        
        z_bins=self.get_z_bins(tracer=tracer)
        n_bins=z_bins['n_bins']
        rho=self.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0
        mag_fact=1
        spin_tracer=tracer
        for i in np.arange(n_bins):
            if tracer=='galaxy':
                mag_fact=z_bins[i]['mag_fact']
                spin_tracer='kappa'
            spin_fact=self.spin_factor(tracer=spin_tracer)
            
            z_bins[i]['Gkernel']=mag_fact*rho/self.sigma_crit(zl=zl,
                                                        zs=z_bins[i]['z'],
                                                        cosmo_h=cosmo_h)
            z_bins[i]['Gkernel_int']=np.dot(z_bins[i]['pzdz'],z_bins[i]['Gkernel'])
            z_bins[i]['Gkernel_int']/=z_bins[i]['Norm']
            if z_bins[i]['Norm']==0:#FIXME
                z_bins[i]['Gkernel_int'][:]=0
            z_bins[i]['Gkernel_int']=np.outer(spin_fact,z_bins[i]['Gkernel_int'])

    def set_galaxy_kernel(self,cosmo_h=None,zl=None,tracer=None):
        """
            Compute rho/Sigma_crit for each source bin at every lens redshift where power spectra is computed.
            cosmo_h: cosmology to compute Sigma_crit
        """
        IA_const=0.0134*cosmo_h.Om0
        b_const=1
        z_bins=self.get_z_bins(tracer=tracer)
        if tracer=='shear' or tracer=='kappa': # kappa maps can have AI. For CMB, set AI=0 in the z_bin properties.
            bias_func=self.NLA_amp_z
            b_const=IA_const
        if tracer=='galaxy':
            bias_func_t=z_bins.get('bias_func')
            bias_func=self.constant_bias if bias_func_t is None else getattr(self,bias_func_t)
#             bias_func=self.constant_bias #FIXME: Make it flexible to get other bias functions
        
        spin_fact=self.spin_factor(tracer=tracer)
        
        dzl=np.gradient(zl)
        cH=c/(cosmo_h.efunc(zl)*cosmo_h.H0)
        cH=cH.value
        
        n_bins=z_bins['n_bins']
        for i in np.arange(n_bins):
            z_bins[i]['gkernel']=b_const*bias_func(z=zl,z_bin=z_bins[i],cosmo_h=cosmo_h)
            z_bins[i]['gkernel']=(z_bins[i]['gkernel'].T/cH).T  #cH factor is for conversion of n(z) to n(\chi).. n(z)dz=n(\chi)d\chi
            z_bins[i]['gkernel']*=spin_fact
            
            #pz_int=interp1d(z_bins[i]['z'],z_bins[i]['pz'],bounds_error=False,fill_value=0)
            #pz_zl=pz_int(zl)

            if len(z_bins[i]['pz'])==1: #interpolation doesnot work well when only 1 point
                bb=np.digitize(z_bins[i]['z'],zl)
                
                pz_zl=np.zeros_like(zl)
                if bb<len(pz_zl):
                    pz_zl[bb]=z_bins[i]['pz']  #assign to nearest zl
                    pz_zl/=np.sum(pz_zl*dzl)
            else:
                pz_zl=np.interp(zl,z_bins[i]['z'],z_bins[i]['pz'],left=0,right=0) #this is linear interpolation
                if not np.sum(pz_zl*dzl)==0: #FIXME
                    pz_zl/=np.sum(pz_zl*dzl)
                else:
                    print('Apparently empty bin',i,zl,z_bins[i]['z'],z_bins[i]['pz'])
                  
            z_bins[i]['gkernel_int']=z_bins[i]['gkernel'].T*pz_zl #dzl multiplied later
            z_bins[i]['gkernel_int']/=np.sum(pz_zl*dzl)
          
            if np.sum(pz_zl*dzl)==0: #FIXME
                z_bins[i]['gkernel_int'][:]=0
                
    def reset_zs(self):
        """
            Reset cosmology dependent values for each source bin
        """
        for i in np.arange(self.ns_bins):
            self.zs_bins[i]['kernel']=None
            self.zs_bins[i]['kernel_int']=None
