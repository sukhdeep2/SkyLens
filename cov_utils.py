import os,sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import jn, jn_zeros
from wigner_functions import *
import healpy as hp

d2r=np.pi/180.
sky_area=np.pi*4/(d2r)**2 #in degrees


class Covariance_utils():
    def __init__(self,f_sky=0,l=None,logger=None,l_cut_jnu=None,do_sample_variance=True,
                 use_window=True,window_l=None,window_file=None,wig_3j=None, do_xi=False,
                ):
        self.logger=logger
        self.l=l
        self.window_l=window_l
        self.window_file=window_file
        # self.l_cut_jnu=l_cut_jnu #this is needed for hankel_transform case for xi. Need separate sigma_window calc.
        self.f_sky=f_sky
        self.do_xi=do_xi
#         self.pseudo_cl=pseudo_cl

        self.use_window=use_window
        self.wig_3j=wig_3j
        self.sample_variance_f=1
        if not do_sample_variance:
            self.sample_variance_f=0 #remove sample_variance from gaussian part

        self.set_window_params()

        self.gaussian_cov_norm=(2.*l+1.)*np.gradient(l) #need Delta l here. Even when
                                                        #binning later
                                                        #take care of f_sky later
        self.gaussian_cov_norm_2D=np.outer(np.sqrt(self.gaussian_cov_norm),np.sqrt(self.gaussian_cov_norm))

    def set_window_params(self):
        self.Om_W=4*np.pi  #multiply f_sky later.
        self.window_func()

        self.Win/=self.Om_W
        self.Win0/=self.Om_W

    def window_func(self):
        if self.window_file is not None:
            W=np.genfromtxt(self.window_file,names=('l','cl'))
            window_l=W['l']
            self.Win=W['cl']
            win_i=interp1d(window_l,self.Win,bounds_error=False,fill_value=0)
            self.Win0=win_i(self.window_l) #this will be useful for SSV
            return

        if self.window_l is None:
            self.window_l=np.arange(100)

        l=self.window_l
        theta_win=np.sqrt(self.Om_W/np.pi)
        l_th=l*theta_win
        Win0=2*jn(1,l_th)/l_th
        Win0=np.nan_to_num(Win0)

        win_i=interp1d(l,Win0,bounds_error=False,fill_value=0)
        self.Win=win_i(self.window_l) #this will be useful for SSV
        self.Win0=win_i(self.l)
        return 0

    def sigma_win_calc(self,cls_lin):
        self.sigma_win=np.dot(self.Win**2*np.gradient(self.window_l)*self.window_l,cls_lin.T)

    def corr_matrix(self,cov=[]):
        diag=np.diag(cov)
        return cov/np.sqrt(np.outer(diag,diag))

    def get_SN(self,SN,tracers,z_indx):
            #get shot noise/shape noise for given set of tracer and z_bins
        SN2={}
        SN2[13]=SN[(tracers[0],tracers[2])][:,z_indx[0], z_indx[2] ] if SN.get((tracers[0],tracers[2])) is not None else np.zeros_like(self.l)
        SN2[24]=SN[(tracers[1],tracers[3])][:,z_indx[1], z_indx[3] ] if SN.get((tracers[1],tracers[3])) is not None else np.zeros_like(self.l)
        SN2[14]=SN[(tracers[0],tracers[3])][:,z_indx[0], z_indx[3] ] if SN.get((tracers[0],tracers[3])) is not None else np.zeros_like(self.l)
        SN2[23]=SN[(tracers[1],tracers[2])][:,z_indx[1], z_indx[2] ] if SN.get((tracers[1],tracers[2])) is not None else np.zeros_like(self.l)

        return SN2

    def gaussian_cov_window(self,cls,SN,tracers,z_indx,do_xi):

        SN2=self.get_SN(SN,tracers,z_indx)
        G1324= ( cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f
             + SN2[13]
                )#/self.gaussian_cov_norm
             #get returns None if key doesnot exist. or 0 adds 0 is SN is none

        G1324=np.outer(G1324,( cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f
              + SN2[24]))

        G1423= ( cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f
              + SN2[14]
              )#/self.gaussian_cov_norm

        G1423=np.outer(G1423,(cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2])
                                                          ]*self.sample_variance_f
                    + SN2[23]))

        return G1324,G1423

    def gaussian_cov(self,cls,SN,tracers,z_indx,do_xi):

        SN2=self.get_SN(SN,tracers,z_indx)

        G1324= ( cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f
             + SN2[13]
                )#/self.gaussian_cov_norm
             #get returns None if key doesnot exist. or 0 adds 0 is SN is none

        G1324*=( cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f
              + SN2[24])

        G1423= ( cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f
              + SN2[14]
              )#/self.gaussian_cov_norm

        G1423*=(cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f
             + SN2[23]
                )

#         G1423/=self.cov_utils.gaussian_cov_norm
        G1423=np.diag(G1423)
        G1324=np.diag(G1324)
#         G=np.diag(G1423+G1324)
# #         if not do_xi:
#         G/=self.gaussian_cov_norm
        return G1324,G1423



    def shear_SN(self,SN,tracers,z_indx):
        SN2=self.get_SN(SN,tracers,z_indx)
        SN1324=np.outer(SN2[13],SN2[24])
        SN1423=np.outer(SN2[14],SN2[23])
        return SN1324,SN1423
