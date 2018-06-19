import os,sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import jn, jn_zeros


d2r=np.pi/180.
sky_area=np.pi*4/(d2r)**2 #in degrees


class Covariance_utils():
    def __init__(self,f_sky=0,l=None):
        self.l=l
        self.f_sky=f_sky
        self.set_window_params(f_sky=self.f_sky)
        self.window_func(theta_win=self.theta_win,f_sky=self.f_sky)

        self.gaussian_cov_norm=(2.*l+1.)*f_sky*np.gradient(l) #need Delta l here. Even when 
                                                                    #binning later
        
    def set_window_params(self,f_sky=None):
        self.theta_win=np.sqrt(f_sky*sky_area)
        self.Win=self.window_func(theta_win=self.theta_win,f_sky=f_sky) 
        self.Om_W=4*np.pi*f_sky
        self.Win/=self.Om_W

    def window_func(self,theta_win=None,f_sky=None):
        l=self.l
        theta_win*=d2r
        l_th=l*theta_win
        W=2*jn(1,l_th)/l_th*4*np.pi*f_sky
        return W

    def sigma_win_calc(self,cls_lin):
        self.sigma_win=np.dot(self.Win**2*np.gradient(self.l)*self.l,cls_lin.T)

    def corr_matrix(self,cov=[]):
        diag=np.diag(cov)
        return cov/np.sqrt(np.outer(diag,diag))
    
    
    def gaussian_cov_auto(self,cls,SN,z_indx,do_xi):
        G1324= ( cls[:,z_indx[0], z_indx[2] ] + SN[:,z_indx[0], z_indx[2] ] )
        G1324*=( cls[:,z_indx[1], z_indx[3] ] + SN[:,z_indx[1], z_indx[3] ] )

        G1423= ( cls[:,z_indx[0], z_indx[3] ] + SN[:,z_indx[0], z_indx[3] ] )
        G1423*=( cls[:,z_indx[1], z_indx[2] ] + SN[:,z_indx[1], z_indx[2] ] )

        G=None
        if not do_xi:
            G=np.diag(G1423+G1324)
            G/=self.gaussian_cov_norm
        return G,G1324,G1423
    