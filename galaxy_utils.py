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

class galaxy_utils():
    def __init__(self,zg_bins=None):
        if zg_bins is not None: #sometimes we call this class just to access some of the functions
            self.zg_bins=zg_bins
            self.set_shot_noise()

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
            ns: The number density for each bin to compute shape noise.
        """
        self.ns_bins=self.zg_bins['n_bins']
        self.SN=np.zeros((1,self.ns_bins,self.ns_bins)) #if self.do_cov else None
                
        for i in np.arange(self.ns_bins):
            self.zg_bins[i]['SN']=self.shape_noise_calc(zg1=self.zg_bins[i],
                                                                    zg2=self.zg_bins[i])
            self.SN[:,i,i]=self.zg_bins[i]['SN']