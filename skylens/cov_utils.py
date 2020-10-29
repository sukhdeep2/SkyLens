"""
This file contains a class with helper functions for covariance calculations.
"""

import os,sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d
from scipy.special import jn, jn_zeros
from skylens.wigner_functions import *
from skylens.binning import *
import healpy as hp

d2r=np.pi/180.
sky_area=np.pi*4/(d2r)**2 #in degrees


class Covariance_utils():
    def __init__(self,f_sky=0,l=None,logger=None,l_cut_jnu=None,do_sample_variance=True,
                 use_window=True,window_l=None,window_file=None,wig_3j=None, do_xi=False,
                ):
        self.__dict__.update(locals()) #assign all input args to the class as properties
        self.binning=binning()
        self.sample_variance_f=1
        if not do_sample_variance:
            self.sample_variance_f=0 #remove sample_variance from gaussian part

        self.set_window_params()

        self.gaussian_cov_norm=(2.*l+1.)*np.gradient(l) #need Delta l here. Even when
                                                        #binning later
                                                        #take care of f_sky later
        self.gaussian_cov_norm_2D=np.outer(np.sqrt(self.gaussian_cov_norm),np.sqrt(self.gaussian_cov_norm))

    def set_window_params(self):
        if isinstance(self.f_sky,float):
            self.Om_W=4*np.pi*self.f_sky
            self.Win,self.Win0=self.window_func()
        else:
            self.Om_W={k1:{}for k1 in self.f_sky.keys()}
            self.Win={k1:{}for k1 in self.f_sky.keys()}
            self.Win0={k1:{}for k1 in self.f_sky.keys()}
            for k1 in self.f_sky.keys():
                for k2 in self.f_sky[k1].keys():
                    self.Om_W[k1][k2]=4*np.pi*self.f_sky[k1][k2]
                    self.Win[k1][k2],self.Win0[k1][k2]=self.window_func(self.Om_W[k1][k2])

    def window_func(self,Om_W=None):
        """
        Set a default unit window used in some calculations (when survey window is not supplied.)
        """
        if Om_W is None:
            Om_W=self.Om_W
        if self.window_file is not None:
            W=np.genfromtxt(self.window_file,names=('l','cl'))
            window_l=W['l']
            self.Win=W['cl']
            win_i=interp1d(window_l,self.Win,bounds_error=False,fill_value=0)
            self.Win0=win_i(self.window_l) #this will be useful for SSV
            return

        #same as eq. 5.4 of https://arxiv.org/pdf/1711.07467.pdf
        if self.window_l is None:
            self.window_l=np.arange(100)

        l=self.window_l
        theta_win=np.sqrt(Om_W/np.pi)
        l_th=l*theta_win
        Win0=2*jn(1,l_th)/l_th
        Win0=np.nan_to_num(Win0)
        Win0=Win0**2 #p-Cl equivalent
#         Win0*=self.f_sky  #FIXME???? Missing 4pi? Using Om_w doesnot match healpix calculation.
        
        win_i=interp1d(l,Win0,bounds_error=False,fill_value=0)
        Win=win_i(self.window_l) #this will be useful for SSV
        Win0=win_i(self.l)
        return Win,Win0

    def sigma_win_calc(self,clz,Win=None,tracers=None,zs_indx=[]):#cls_lin, Win_cl=None,Om_w12=None,Om_w34=None):
        """
        compute mass variance on the scale of the survey window.
        """
        cls_lin=clz['cls_lin']
        if Win is None: #use defaulr f_sky window
            if isinstance(self.f_sky,float):
                Win_cl=self.Win
                Om_w12=self.Om_W
                Om_w34=self.Om_W
            else:
                Win_cl=self.Win[tracers][zs_indx]#FIXME: Need 4 window term here
                Om_w12=self.Om_W[(tracers[0],tracers[1])][(zs_indx[0],zs_indx[1])]
                Om_w34=self.Om_W[(tracers[2],tracers[3])][(zs_indx[2],zs_indx[3])]
        else:
            Win_cl=Win['mask_comb_cl']  #['cov'][tracers][zs_indx]
            Om_w12=Win['Om_w12'] #[tracers][zs_indx]
            Om_w34=Win['Om_w34'] #[tracers][zs_indx]
            
        sigma_win=np.dot(Win_cl*np.gradient(self.window_l)*(2*self.window_l+1),cls_lin.T)
        sigma_win/=Om_w12*Om_w34
        return sigma_win

    def corr_matrix(self,cov=[]):
        """
        convert covariance matrix into correlation matrix.
        """
        diag=np.diag(cov)
        return cov/np.sqrt(np.outer(diag,diag))

    def get_SN(self,SN,tracers,z_indx):
        """
            get shot noise/shape noise for given set of tracer and z_bins.
            Note that we assume shot noise is function of ell in general. 
            Default is to set that function to constant. For e.x., CMB lensing
            noise is function of ell. Galaxy shot noise can also vary with ell, 
            especially on small scales.
        """
        SN2={}
        SN2[13]=SN[(tracers[0],tracers[2])][:,z_indx[0], z_indx[2] ] if SN.get((tracers[0],tracers[2])) is not None else np.zeros_like(self.l)
        SN2[24]=SN[(tracers[1],tracers[3])][:,z_indx[1], z_indx[3] ] if SN.get((tracers[1],tracers[3])) is not None else np.zeros_like(self.l)
        SN2[14]=SN[(tracers[0],tracers[3])][:,z_indx[0], z_indx[3] ] if SN.get((tracers[0],tracers[3])) is not None else np.zeros_like(self.l)
        SN2[23]=SN[(tracers[1],tracers[2])][:,z_indx[1], z_indx[2] ] if SN.get((tracers[1],tracers[2])) is not None else np.zeros_like(self.l)

        return SN2

    def get_CV_cl(self,cls,tracers,z_indx):
        """
        Get the tracer power spectra, C_ell, for covariance calculations.
        """
        CV2={}
        CV2[13]=cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f
        CV2[24]=cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f
        CV2[14]=cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f
        CV2[23]=cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f
        return CV2

    def gaussian_cov_window(self,cls,SN,tracers,z_indx,do_xi,Win,Bmode_mf=1,bin_window=False,bin_utils=None):
        """
        Computes the power spectrum gaussian covariance, with proper factors of window. We have separate function for the 
        case when window is not supplied.
        """
        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        CV_B=self.get_CV_B_cl(cls,tracers,z_indx)
        
        G={1324:0,1423:0}
        cv_indxs={1324:(13,24),1423:(14,23)}
        
        add_EB=1
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            add_EB+=1
            
        for corr_i in [1324,1423]:
            W_pm=Win['W_pm'][corr_i]
            c1=cv_indxs[corr_i][0]
            c2=cv_indxs[corr_i][1]

            for k in Win['M'][corr_i].keys():
                for wp in W_pm:
                    CV2=CV
                    for a_EB in np.arange(add_EB):
                        if wp<0 or a_EB>0:
                            CV2=CV_B
                        if k=='clcl': 
                            G_t=np.outer(CV2[c1],CV2[c2])
                        if k=='Ncl': 
                            G_t=np.outer(SN2[c1],CV2[c2])
                        if k=='clN': 
                            G_t=np.outer(CV2[c1],SN2[c2])
                        if k=='NN': 
                            G_t=np.outer(SN2[c1],SN2[c2])
                        if a_EB>0:
                            G_t*=Bmode_mf #need to -1 for xi+/- cross covariance
                        # if bin_window:
                        #     G_t=self.binning.bin_2d(cov=G_t,bin_utils=bin_utils)
                        G[corr_i]+=G_t*Win['M'][corr_i][k][wp]

        return G[1324],G[1423]
        
    def get_CV_B_cl(self,cls,tracers,z_indx): #
        """ 
            Return power spectra and noise contributions for shear B-mode covariance. We 
            assume that the shear B-mode power spectra is zero (there is still E-mode contribution 
            due to leakage caused by window).
        """
        sv_f={13:1,24:1,14:1,23:1}
        if tracers[0]=='shear' and tracers[2]=='shear':
            sv_f[13]=0
        if tracers[1]=='shear' and tracers[3]=='shear':
            sv_f[24]=0
        if tracers[0]=='shear' and tracers[3]=='shear':
            sv_f[14]=0
        if tracers[1]=='shear' and tracers[2]=='shear':
            sv_f[23]=0
            
        CV2={}
        CV2[13]=cls[(tracers[0],tracers[2])] [(z_indx[0], z_indx[2]) ]*self.sample_variance_f*sv_f[13]
        CV2[24]=cls[(tracers[1],tracers[3])][(z_indx[1], z_indx[3]) ]*self.sample_variance_f*sv_f[24]
        CV2[14]=cls[(tracers[0],tracers[3])][(z_indx[0], z_indx[3]) ]*self.sample_variance_f*sv_f[14]
        CV2[23]=cls[(tracers[1],tracers[2])][(z_indx[1], z_indx[2]) ]*self.sample_variance_f*sv_f[23]
        return CV2
            
    def gaussian_cov(self,cls,SN,tracers,z_indx,do_xi,f_sky,Bmode_mf=1): #no-window covariance
        """
        Gaussian covariance for the case when no window is supplied and only f_sky is used.
        """

        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        
        def get_G4(CV,SN2):
            """
            return the two sub covariance matrix.
            """
            G1324= ( CV[13]+ SN2[13])#/self.gaussian_cov_norm
                 #get returns None if key doesnot exist. or 0 adds 0 is SN is none

            G1324*=( CV[24]+ SN2[24])

            G1423= ( CV[14]+ SN2[14])#/self.gaussian_cov_norm

            G1423*=(CV[23]+ SN2[23])
            return G1324,G1423
            
        G1324,G1423=get_G4(CV,SN2)
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            CVB=self.get_CV_B_cl(cls,tracers,z_indx)
            G1324_B,G1423_B=get_G4(CVB,SN2)
            G1324+=G1324_B*Bmode_mf
            G1423+=G1423_B*Bmode_mf

        G1423=np.diag(G1423)
        G1324=np.diag(G1324)
        
        Norm=np.pi*4
        
        fs1324=1
        fs0=1
        fs1423=1
        if f_sky is not None:
            if isinstance(self.f_sky,float):
                fs1324=self.f_sky
                fs0=self.f_sky**2
                fs1423=self.f_sky
            else:
                fs1324=f_sky[tracers][z_indx]#np.sqrt(f_sky[tracers[0],tracers[2]][z_indx[0],z_indx[2]]*f_sky[tracers[1],tracers[3]][z_indx[1],z_indx[3]])
                fs0=f_sky[tracers[0],tracers[1]][z_indx[0],z_indx[1]] * f_sky[tracers[2],tracers[3]][z_indx[2],z_indx[3]]
                fs1423=f_sky[tracers][z_indx]#np.sqrt(f_sky[tracers[0],tracers[3]][z_indx[0],z_indx[3]]*f_sky[tracers[1],tracers[2]][z_indx[1],z_indx[2]])
        
        if not self.do_xi:
            G1324/=self.gaussian_cov_norm_2D/fs1324*fs0
            G1423/=self.gaussian_cov_norm_2D/fs1423*fs0
        else:
            G1324*=fs1324/fs0/Norm
            G1423*=fs1423/fs0/Norm

        if np.all(np.array(tracers)=='shear') and Bmode_mf<0:
            G1324*=Bmode_mf
            G1423*=Bmode_mf
            
        return G1324,G1423

    def xi_gaussian_cov_window_approx(self,cls,SN,tracers,z_indx,do_xi,Win,WT,WT_kwargs,Bmode_mf=1):
        """
        This returns correlation function gaussian covariance. Here window is assumed to decouple from the
        covariance and the product of the two is taken.
        """
        #FIXME: Need to implement the case when we are only using bin centers.
        SN2=self.get_SN(SN,tracers,z_indx)
        CV=self.get_CV_cl(cls,tracers,z_indx)
        CV_B=self.get_CV_B_cl(cls,tracers,z_indx)
        
        G={1324:0,1423:0}
        cv_indxs={1324:(13,24),1423:(14,23)}
        
        Norm=np.pi*4
        
        add_EB=1
        if self.do_xi and np.all(np.array(tracers)=='shear'):
            add_EB+=1
            
        for corr_i in [1324,1423]:
            W_pm=Win['W_pm'][corr_i]
            c1=cv_indxs[corr_i][0]
            c2=cv_indxs[corr_i][1]

            for k in Win['M'][corr_i].keys():
                for wp in W_pm:
                    CV2=CV
                    for a_EB in np.arange(add_EB):
                        if wp<0 or a_EB>0:
                            CV2=CV_B
                        if k=='clcl': 
                            G_t=np.outer(CV2[c1],CV2[c2])
                        if k=='Ncl': 
                            G_t=np.outer(SN2[c1],CV2[c2])
                        if k=='clN': 
                            G_t=np.outer(CV2[c1],SN2[c2])
                        if k=='NN': 
                            G_t=np.outer(SN2[c1],SN2[c2])
                        th,G_t=WT.projected_covariance2(cl_cov=G_t,**WT_kwargs)
                        G_t*=Win['xi'][corr_i][k]
                        G_t/=Norm
                        if a_EB>0:
                            G_t*=Bmode_mf #need to -1 for xi+/- cross covariance
                        G[corr_i]+=G_t

        return G[1324]+G[1423]
