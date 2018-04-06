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
        
    def set_window_params(self,f_sky=None):
        self.theta_win=np.sqrt(f_sky*sky_area)
        self.Win=self.window_func(theta_win=self.theta_win,f_sky=f_sky) 
        self.Om_W=4*np.pi*f_sky

    def window_func(self,theta_win=None,f_sky=None):
        l=self.l
        theta_win*=d2r
        l_th=l*theta_win
        W=2*jn(1,l_th)/l_th*4*np.pi*f_sky
        return W

    def sigma_win_calc(self,cls_lin):
        self.sigma_win=np.dot(self.Win**2*np.gradient(self.l)*self.l,cls_lin.T)