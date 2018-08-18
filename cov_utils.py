import os,sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import jn, jn_zeros
from torch_utils import *

d2r=np.pi/180.
sky_area=np.pi*4/(d2r)**2 #in degrees


class Covariance_utils():
    def __init__(self,f_sky=0,l=None,logger=None,l_cut_jnu=None,do_sample_variance=True,use_window=True):
        self.logger=logger
        self.l=l
        self.l_cut_jnu=l_cut_jnu #this is needed for hankel_transform case for xi. Need separate sigma_window calc.
        self.f_sky=f_sky

        self.use_window=use_window
        self.sample_variance_f=tc.tensor([1.],dtype=l.dtype,device=l.device)
        if not do_sample_variance:
            self.sample_variance_f*=0 #remove sample_variance from gaussian part

        self.set_window_params(f_sky=self.f_sky)
        self.window_func(theta_win=self.theta_win,f_sky=self.f_sky)

        self.gaussian_cov_norm=(2.*l+1.)*f_sky*tc_gradient(l) #need Delta l here. Even when
                                                                    #binning later

    def set_window_params(self,f_sky=None):
        self.theta_win=np.sqrt(f_sky*sky_area)
        if self.use_window:
            self.Win=self.window_func(theta_win=self.theta_win,f_sky=f_sky)
        else:
            self.Win=np.ones_like(self.l,dtype='float32')
        self.Om_W=4*np.pi*f_sky
        self.Win/=self.Om_W #FIXME: This thing has been forgotten and not used anywhere in the code.

    def window_func(self,theta_win=None,f_sky=None):
        l=self.l
        theta_win*=d2r
        l_th=l*theta_win
        W=2*jn(1,l_th)/l_th*4*np.pi*f_sky
        return W

    def sigma_win_calc(self,cls_lin):
        if self.l_cut_jnu is None:
            #self.sigma_win=np.dot(self.Win**2*tc_gradient(self.l)*self.l,cls_lin.transpose(1,0))
            self.sigma_win=(self.Win**2*tc_gradient(self.l)*self.l*cls_lin).sum(1)
        else: #FIXME: This is ugly. Only needed for hankel transform (not wigner). Remove if HT is deprecated.
            self.sigma_win={}
            for m1_m2 in self.l_cut_jnu['m1_m2s']:
                lc=self.l_cut_jnu[m1_m2]
                self.sigma_win[m1_m2]=np.dot(self.Win[lc]**2*tc_gradient(self.l[lc])*self.l[lc],cls_lin[:,lc])
        #FIXME: This is ugly

    def corr_matrix(self,cov=[]):
        diag=cov.diag()
        return cov/tc.sqrt(tc.ger(diag,diag))


    def gaussian_cov_auto(self,cls,SN,tracers,z_indx,do_xi):
        """
        This is 'auto' covariance for a particular power spectra, but the power spectra
        itself could a cross-correlation, eg. galaxy-lensing cross correlations.
        For auto correlation, eg. lensing-lensing, cls1,cls2,cl12 should be same. Same for shot noise
        SN.

        """
        # print(cls[(tracers[0],tracers[2])].keys())
        G1324= ( cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f
             # + (SN.get((tracers[0],tracers[2]))[:,z_indx[0], z_indx[2] ]  or 0)
             + (SN[(tracers[0],tracers[2])][:,z_indx[0], z_indx[2] ] if SN.get((tracers[0],tracers[2])) is not None else
                                                                                                        tc.tensor(0,dtype=tc.double))
                )
             #get returns None if key doesnot exist. or 0 adds 0 is SN is none

        G1324*=( cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f
              # +(SN.get((tracers[1],tracers[3]))[:,z_indx[1], z_indx[3] ] or 0)
              + (SN[(tracers[1],tracers[3])][:,z_indx[1], z_indx[3] ] if SN.get((tracers[1],tracers[3])) is not None else
                                                                                                                tc.tensor(0,dtype=tc.double))
              )

        G1423= ( cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f
              # + (SN.get((tracers[0],tracers[3]))[:,z_indx[0], z_indx[3] ] or 0)
              + (SN[(tracers[0],tracers[3])][:,z_indx[0], z_indx[3] ] if SN.get((tracers[0],tracers[3])) is not None else
                                                                                                                tc.tensor(0,dtype=tc.double))
              )

        G1423*=( cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f
             # + (SN.get((tracers[1],tracers[2]))[:,z_indx[1], z_indx[2] ] or 0)
             + (SN[(tracers[1],tracers[2])][:,z_indx[1], z_indx[2] ] if SN.get((tracers[1],tracers[2])) is not None else
                                                                                                                tc.tensor(0,dtype=tc.double))
                )

        G=None
        if not do_xi:
            G=(G1423+G1324).diag()
            G/=self.gaussian_cov_norm
        return G,G1324,G1423
