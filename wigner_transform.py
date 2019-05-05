from scipy.special import jn, jn_zeros,jv
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.optimize import fsolve
from wigner_functions import *
import numpy as np
import itertools

class wigner_transform():
    def __init__(self,theta=[],l=[],m1_m2=[(0,0)],logger=None,ncpu=None,use_window=False,**kwargs):
        self.name='Wigner'
        self.logger=logger
        self.l=l
        self.grad_l=np.gradient(l)
        self.norm=(2*l+1.)/(4.*np.pi) #ignoring some factors of -1,
                                                    #assuming sum and differences of m1,m2
                                                    #are even for all correlations we need.
        self.wig_d={}
        self.wig_3j={}
        self.m1_m2s=m1_m2
        self.theta={}
        # self.theta=theta
        for (m1,m2) in m1_m2:
            self.wig_d[(m1,m2)]=wigner_d_parallel(m1,m2,theta,self.l,ncpu=ncpu)
            # self.wig_d[(m1,m2)]*=self.norm #this works for covariance and correlation function
            self.theta[(m1,m2)]=theta #FIXME: Ugly


    def cl_grid(self,l_cl=[],cl=[],taper=False,**kwargs):
        if taper:
            sself.taper_f=self.taper(l=l,**kwargs)
            cl=cl*taper_f
        # if l==[]:#In this case pass a function that takes k with kwargs and outputs cl
        #     cl2=cl(l=self.l,**kwargs)
        # else:
        cl_int=interp1d(l_cl,cl,bounds_error=False,fill_value=0,
                        kind='linear')
        cl2=cl_int(self.l)
        return cl2

    def cl_cov_grid(self,l_cl=[],cl_cov=[],taper=False,**kwargs):
        if taper:#FIXME there is no check on change in taper_kwargs
            if self.taper_f2 is None or not np.all(np.isclose(self.taper_f['l'],cl)):
                self.taper_f=self.taper(l=l,**kwargs)
                taper_f2=np.outer(self.taper_f['taper_f'],self.taper_f['taper_f'])
                self.taper_f2={'l':l,'taper_f2':taper_f2}
            cl=cl*self.taper_f2['taper_f2']
        if l_cl_cl==[]:#In this case pass a function that takes k with kwargs and outputs cl
            cl2=cl_cov(l=self.l,**kwargs)
        else:
            cl_int=RectBivariateSpline(l_cl,l_cl,cl_cov,)#bounds_error=False,fill_value=0,
                            #kind='linear')
                    #interp2d is slow. Make sure l_cl is on regular grid.
            cl2=cl_int(self.l,self.l)
        return cl2

    def projected_correlation(self,l_cl=[],cl=[],m1_m2=[],taper=False,**kwargs):
        cl2=self.cl_grid(l_cl=l_cl,cl=cl,taper=taper,**kwargs)
        w=np.dot(self.wig_d[m1_m2]*self.grad_l*self.norm,cl2)
        return self.theta[m1_m2],w

    def projected_covariance(self,l_cl=[],cl_cov=[],m1_m2=[],m1_m2_cross=None,
                            taper=False,**kwargs):
        if m1_m2_cross is None:
            m1_m2_cross=m1_m2
        #when cl_cov can be written as vector, eg. gaussian covariance
        cl2=self.cl_grid(l_cl=l_cl,cl=cl_cov,taper=taper,**kwargs)
        cov=np.einsum('rk,k,sk->rs',self.wig_d[m1_m2]*np.sqrt(self.norm),cl2*self.grad_l,
                    self.wig_d[m1_m2_cross]*np.sqrt(self.norm),optimize=True)
        #FIXME: Check normalization
        return self.theta[m1_m2],cov

    def projected_covariance2(self,l_cl=[],cl_cov=[],m1_m2=[],m1_m2_cross=None,
                                taper=False,**kwargs):
        #when cl_cov is a 2-d matrix
        if m1_m2_cross is None:
            m1_m2_cross=m1_m2
        cl_cov2=cl_cov  #self.cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,m1_m2=m1_m2,taper=taper,**kwargs)

        cov=np.einsum('rk,kk,sk->rs',self.wig_d[m1_m2]*np.sqrt(self.norm)*self.grad_l,cl_cov2,
                    self.wig_d[m1_m2_cross]*np.sqrt(self.norm),optimize=True)
#         cov=np.dot(self.wig_d[m1_m2]*self.grad_l*np.sqrt(self.norm),np.dot(self.wig_d[m1_m2_cross]*np.sqrt(self.norm),cl_cov2).T)
        # cov*=self.norm
        #FIXME: Check normalization
        return self.theta[m1_m2],cov

    def taper(self,l=[],large_k_lower=10,large_k_upper=100,low_k_lower=0,low_k_upper=1.e-5):
        #FIXME there is no check on change in taper_kwargs
        if self.taper_f is None or not np.all(np.isclose(self.taper_f['k'],k)):
            taper_f=np.zeros_like(k)
            x=k>large_k_lower
            taper_f[x]=np.cos((k[x]-large_k_lower)/(large_k_upper-large_k_lower)*np.pi/2.)
            x=k<large_k_lower and k>low_k_upper
            taper_f[x]=1
            x=k<low_k_upper
            taper_f[x]=np.cos((k[x]-low_k_upper)/(low_k_upper-low_k_lower)*np.pi/2.)
            self.taper_f={'taper_f':taper_f,'k':k}
        return self.taper_f

    def diagonal_err(self,cov=[]):
        return np.sqrt(np.diagonal(cov))

    def skewness(self,l_cl=[],cl1=[],cl2=[],cl3=[],m1_m2=[],taper=False,**kwargs):
        cl1=self.cl_grid(l_cl=l_cl,cl=cl1,m1_m2=m1_m2,taper=taper,**kwargs)
        cl2=self.cl_grid(l_cl=l_cl,cl=cl2,m1_m2=m1_m2,taper=taper,**kwargs)
        cl3=self.cl_grid(l_cl=l_cl,cl=cl3,m1_m2=m1_m2,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.wig_d[m1_m2],self.wig_d[m1_m2],
                        self.wig_d[m1_m2]*cl1*cl2*cl3)
        skew*=self.norm
        #FIXME: Check normalization
        return self.theta[m1_m2],skew
