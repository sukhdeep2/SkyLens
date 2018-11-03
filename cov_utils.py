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
                pseudo_cl=False):
        self.logger=logger
        self.l=l
        self.window_l=window_l
        self.window_file=window_file
        self.l_cut_jnu=l_cut_jnu #this is needed for hankel_transform case for xi. Need separate sigma_window calc.
        self.f_sky=f_sky
        self.do_xi=do_xi
        self.pseudo_cl=pseudo_cl

        self.use_window=use_window
        self.wig_3j=wig_3j
        self.sample_variance_f=1
        if not do_sample_variance:
            self.sample_variance_f=0 #remove sample_variance from gaussian part

        self.set_window_params(f_sky=self.f_sky)

        self.gaussian_cov_norm=(2.*l+1.)*f_sky*np.gradient(l) #need Delta l here. Even when
                                                                    #binning later

    def set_window_params(self,f_sky=None):
        self.Om_W=4*np.pi*f_sky
        if self.use_window:
            self.window_func()
            if not self.do_xi and self.pseudo_cl:
                if self.wig_3j is None: #FIXME: not using for correlation function for now. Memeory issues among others.
                    m_1=0 #FIXME: Use proper spins (m_i) here
                    m_2=0
                    self.wig_3j=Wigner3j_parallel( m_1, m_2, 0, self.l, self.l, self.window_l)
                    print('wg_3j max:',self.wig_3j.todense().max())
                self.coupling_M=np.dot(self.wig_3j**2,self.Win*(2*self.window_l+1))
        else:
            self.Win=np.zeros_like(self.l,dtype='float32')
            x=self.l==0
            self.Win[x]=1.
            self.Win0=np.copy(self.Win)

        self.Win/=self.Om_W #FIXME: This thing has been forgotten and not used anywhere in the code.
        self.Win0/=self.Om_W

    def window_func(self):
        if self.window_file is not None:
            W=np.genfromtxt(self.window_file,names=('l','cl'))
            self.window_l=W['l']
            self.Win=W['cl']
            win_i=interp1d(self.window_l,self.Win,bounds_error=False,fill_value=0)
            self.Win0=win_i(self.l) #this will be useful for SSV
            return

        if self.window_l is None:
            self.window_l=np.arange(100)
        NP=hp.nside2npix(256)
        M=np.zeros(NP)
        M[:np.int(NP*self.f_sky)]=1
        Win0=hp.sphtfunc.anafast(M)
        l=np.arange(len(Win0))

#         l=np.logspace(-2,2,1000)#self.l
#         theta_win=self.theta_win*d2r
#         l_th=l*theta_win
#         Win0=2*jn(1,l_th)/l_th*4*np.pi*self.f_sky
        win_i=interp1d(l,Win0,bounds_error=False,fill_value=0)
        self.Win=win_i(self.window_l) #this will be useful for SSV
        self.Win0=win_i(self.l)
        return 0

    def sigma_win_calc(self,cls_lin):
        if self.l_cut_jnu is None:
            self.sigma_win=np.dot(self.Win0**2*np.gradient(self.l)*self.l,cls_lin.T)
        else: #FIXME: This is ugly. Only needed for hankel transform (not wigner). Remove if HT is deprecated.
            self.sigma_win={}
            for m1_m2 in self.l_cut_jnu['m1_m2s']:
                lc=self.l_cut_jnu[m1_m2]
                self.sigma_win[m1_m2]=np.dot(self.Win0[lc]**2*np.gradient(self.l[lc])*self.l[lc],cls_lin[:,lc].T)
        #FIXME: This is ugly

    def corr_matrix(self,cov=[]):
        diag=np.diag(cov)
        return cov/np.sqrt(np.outer(diag,diag))


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
             + (SN[(tracers[0],tracers[2])][:,z_indx[0], z_indx[2] ] if SN.get((tracers[0],tracers[2])) is not None else 0)
                )
             #get returns None if key doesnot exist. or 0 adds 0 is SN is none

        G1324*=( cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f
              # +(SN.get((tracers[1],tracers[3]))[:,z_indx[1], z_indx[3] ] or 0)
              + (SN[(tracers[1],tracers[3])][:,z_indx[1], z_indx[3] ] if SN.get((tracers[1],tracers[3])) is not None else 0)
              )

        G1423= ( cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f
              # + (SN.get((tracers[0],tracers[3]))[:,z_indx[0], z_indx[3] ] or 0)
              + (SN[(tracers[0],tracers[3])][:,z_indx[0], z_indx[3] ] if SN.get((tracers[0],tracers[3])) is not None else 0)
              )

        G1423*=( cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f
             # + (SN.get((tracers[1],tracers[2]))[:,z_indx[1], z_indx[2] ] or 0)
             + (SN[(tracers[1],tracers[2])][:,z_indx[1], z_indx[2] ] if SN.get((tracers[1],tracers[2])) is not None else 0)
                )
        
        if do_xi and np.all(np.array(tracers)=='shear'): #FIXME: Temporary fix for shear-shear. Check ggl as well
            G1324+=(SN[(tracers[0],tracers[2])][:,z_indx[0], z_indx[2] ] if SN.get((tracers[0],tracers[2])) is not None else 0)*(SN[(tracers[1],tracers[3])][:,z_indx[1], z_indx[3] ] if SN.get((tracers[1],tracers[3])) is not None else 0)
            G1423+=(SN[(tracers[0],tracers[3])][:,z_indx[0], z_indx[3] ] if SN.get((tracers[0],tracers[3])) is not None else 0)*(SN[(tracers[1],tracers[2])][:,z_indx[1], z_indx[2] ] if SN.get((tracers[1],tracers[2])) is not None else 0)
                
        G=None
        if not do_xi:
            G=np.diag(G1423+G1324)
            G/=self.gaussian_cov_norm
        return G,G1324,G1423
