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

class Galaxy_utils():
    def __init__(self,zg_bins=None,bias_func=None,logger=None,l=None,z_th=None):
        self.l=l
        self.z_th=z_th
        self.zg_bins=zg_bins
        if zg_bins is not None: #sometimes we call this class just to access some of the functions
            self.set_zg_to_zth()
            self.set_shot_noise()
            self.bias_func=bias_func
            if bias_func is None:
                self.bias_func=self.linear_bias_powerlaw

    def set_zg_to_zth(self):
        dz_th=np.gradient(self.z_th)
        nbins=self.zg_bins['n_bins']
        for i in np.arange(nbins):
            zb=self.zg_bins[i]
            pz_int=interp1d(zb['z'],zb['pz'],bounds_error=False,fill_value=0)
            pz_zth=pz_int(self.z_th)
            self.zg_bins[i]['z']=self.z_th
            self.zg_bins[i]['dz']=dz_th
            norm=np.sum(self.z_th*dz_th*pz_zth)
            self.zg_bins[i]['pz']=pz_zth/norm
            self.zg_bins[i]['pzdz']=dz_th*self.zg_bins[i]['pz'] #FIXME: This can mess things up
            self.zg_bins[i]['Norm']=1
            self.zg_bins[i]['lens_kernel']=None
            if hasattr(self.zg_bins[i]['W'],"__len__"):
                W_int=interp1d(zb[i]['z'],W,bounds_error=False,fill_value=0)
                self.zg_bins[i]['W']=W_int(self.z_th)




    def shot_noise_calc(self,zg1=None,zg2=None):
        if not np.array_equal(zg1['z'],zg2['z']):
            return 0
        if np.any(np.isinf(zg1['nz'])) or np.any(np.isinf(zg2['nz'])):
            return 0
        SN=np.sum(zg1['W']*zg2['W']*zg1['nz']) #FIXME: Check this
        #Assumption: ns(z)=ns*pzg*dzg
        SN/=np.sum(zg1['nz']*zg1['W'])
        SN/=np.sum(zg2['nz']*zg2['W'])
        return SN
        # XXX Make sure pzg are properly normalized


    def set_shot_noise(self,cross_PS=True):
        """
            Setting source redshift bins in the format used in code.
            Need
            zg (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                        with values for each bin
            pzg: redshift distribution. same format as zg
            z_bins: if zg and pzg are for whole survey, then bins to divide the sample. If
                    tomography is based on lens redshift, then this arrays contains those redshifts.
            ns: The number density for each bin to compute shot noise.
        """
        self.ng_bins=self.zg_bins['n_bins']
        self.SN=np.zeros((1,self.ng_bins,self.ng_bins)) #if self.do_cov else None

        for i in np.arange(self.ng_bins):
            self.zg_bins[i]['SN']=self.shot_noise_calc(zg1=self.zg_bins[i],
                                                                    zg2=self.zg_bins[i])
            self.SN[:,i,i]=self.zg_bins[i]['SN']


    def linear_bias_powerlaw(self,z=[],cosmo_h=None,b1=None,b2=None):
        return np.outer(b1*(1+z)**b2,np.ones_like(self.l)+self.l/self.l[-1]) #FIXME: This might need to change to account

    def set_zg_bias(self,cosmo_h=None,bias_kwargs={},bias_func=None,zl=[]):
        """
            Compute rho/Sigma_crit for each source bin at every lens redshift where power spectra is computed.
            cosmo_h: cosmology to compute Sigma_crit
        """
        #We need to compute these only once in every run
        # i.e not repeat for every ij combo

        for i in np.arange(self.ng_bins):
            self.zg_bins[i]['kernel']=self.bias_func(z=self.zg_bins[i]['z'],
                                                        cosmo_h=cosmo_h,**bias_kwargs)
            # self.zg_bins[i]['kernel_int']=np.dot(self.zg_bins[i]['pzdz'],self.zg_bins[i]['kernel'])
            # self.zg_bins[i]['kernel_int']/=self.zg_bins[i]['Norm']
            self.zg_bins[i]['kernel_int']=self.zg_bins[i]['kernel'].T*self.zg_bins[i]['pzdz']
            # self.zg_bins[i]['kernel_int']=self.zg_bins[i]['kernel_int'].T

    def reset_zg(self):
        """
            Reset cosmology dependent values for each source bin
        """
        for i in np.arange(self.ng_bins):
            self.zg_bins[i]['kernel']=None
            self.zg_bins[i]['kernel_int']=None
