#deprecated.. see tracer utils
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
    def __init__(self,sigma_gamma=0.3,zs_bins=None,logger=None,l=None):
        #self.ns=ns
        ns=1 #different for different z_source. should be property of z_source
        self.sigma_gamma=sigma_gamma
        self.logger=logger
        self.SN0=sigma_gamma**2/(ns*3600./d2r**2) #FIXME: Note that for correlation function,
                                                    #sigma_gamma should be per component.
        self.l=l
        #Gravitaional const to get Rho crit in right units
        self.G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        self.G2*=8*np.pi/3.
        self.zs_bins=zs_bins
        if zs_bins is not None: #sometimes we call this class just to access some of the functions
            self.set_zbins(zs_bins)

    def set_zbins(self,z_bins={}):
        self.zs_bins=z_bins
        self.set_shape_noise()

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

    def shape_noise_calc(self,zs1=None,zs2=None):#this function is deprecated. shape noise should be input
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


    def set_shape_noise(self,cross_PS=True):
        """
            Setting source redshift bins in the format used in code.
            Need
            zs (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                        with values for each bin
            pzs: redshift distribution. same format as zs
            z_bins: if zs and pzs are for whole survey, then bins to divide the sample. If
                    tomography is based on lens redshift, then this arrays contains those redshifts.
            ns: The number density for each bin to compute shape noise.
        """
        self.ns_bins=self.zs_bins['n_bins']
        self.SN=np.zeros((len(self.l),self.ns_bins,self.ns_bins)) #if self.do_cov else None

        for i in np.arange(self.ns_bins):
#             self.zs_bins[i]['SN']=self.shape_noise_calc(zs1=self.zs_bins[i],
#                                                                     zs2=self.zs_bins[i])
            self.SN[:,i,i]+=self.zs_bins['SN']['shear'][:,i,i]

#         if not cross_PS: # if we are not computing cross_PS, then we assume that sources overlap in different bins. Hence we need to compute the cross shape noise... this should be input.
#             for i in np.arange(self.ns_bins):
#                 for j in np.arange(i,self.ns_bins):
#                     self.SN[:,i,j]=self.shape_noise_calc(zs1=self.zs_bins[i],
#                                                                         zs2=self.zs_bins[j])
#                     self.SN[:,j,i]=self.SN[:,i,j]
#                     #FIXME: this shape noise calc is probably wrong
    def NLA_amp_z(self,z=[],AI=1,AI_z=0,cosmo_h=None):
        return np.outer(AI*(1+z)**AI_z*cosmo_h.H(z)/c,np.ones_like(self.l)) #FIXME: This might need to change to account

    def set_zs_sigc(self,cosmo_h=None,zl=None):
        """
            Compute rho/Sigma_crit for each source bin at every lens redshift where power spectra is computed.
            cosmo_h: cosmology to compute Sigma_crit
        """
        #We need to compute these only once in every run
        # i.e not repeat for every ij combo

        rho=self.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0
        IA_const=0.0134*cosmo_h.Om0
        dzl=np.gradient(zl)

        cH=c/(cosmo_h.efunc(zl)*cosmo_h.H0)
        cH=cH.value

        for i in np.arange(self.ns_bins):
            self.zs_bins[i]['Gkernel']=rho/self.sigma_crit(zl=zl,
                                                        zs=self.zs_bins[i]['z'],
                                                        cosmo_h=cosmo_h)
            self.zs_bins[i]['Gkernel_int']=np.dot(self.zs_bins[i]['pzdz'],self.zs_bins[i]['Gkernel'])
            self.zs_bins[i]['Gkernel_int']/=self.zs_bins[i]['Norm']

            self.zs_bins[i]['IA_kernel']=IA_const*self.NLA_amp_z(z=zl,AI=self.zs_bins[i]['AI'],
                                                                    AI_z=self.zs_bins[i]['AI_z'],cosmo_h=cosmo_h)
            pz_int=interp1d(self.zs_bins[i]['z'],self.zs_bins[i]['pz'],bounds_error=False,fill_value=0)
            pz_zl=pz_int(zl)
            self.zs_bins[i]['IA_kernel_int']=self.zs_bins[i]['IA_kernel'].T*pz_zl #dzl multiplied later
#             self.zs_bins[i]['IA_kernel_int']/=np.sum(pz_zl*dzl)
#             print('pzl',np.sum(pz_zl*dzl))

            self.zs_bins[i]['kernel_int']=self.zs_bins[i]['Gkernel_int']+self.zs_bins[i]['IA_kernel_int']

    def reset_zs(self):
        """
            Reset cosmology dependent values for each source bin
        """
        for i in np.arange(self.ns_bins):
            self.zs_bins[i]['kernel']=None
            self.zs_bins[i]['kernel_int']=None
