from scipy.special import jn, jn_zeros,jv
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy.optimize import fsolve
from skylens.wigner_functions import *
from scipy.special import jn, jn_zeros,jv
import numpy as np
import itertools

class wigner_transform():
    def __init__(self,theta=[],l=[],s1_s2=[(0,0)],logger=None,ncpu=None,wig_d_taper_order_low=6,wig_d_taper_order_high=8,**kwargs):
        self.__dict__.update(locals())
        self.name='Wigner'
        self.logger=logger
        self.grad_l=np.gradient(l)
        self.norm=(2*l+1.)/(4.*np.pi) 
        self.wig_d={}
        self.theta={}
        self.s1_s2s=s1_s2
#         self.theta=theta
        for (m1,m2) in s1_s2:
            self.wig_d[(m1,m2)]=wigner_d_parallel(m1,m2,theta,self.l,ncpu=ncpu)
#             self.wig_d[(m1,m2)]=wigner_d_recur(m1,m2,theta,self.l)
            self.theta[(m1,m2)]=theta #FIXME: Ugly
            self.wig_d_smoothing(s1_s2=(m1,m2))

    def reset_theta_l(self,theta=None,l=None):
        """
        In case theta ell values are changed. This can happen when we implement the binning scheme.
        """
        if theta is None:
            theta=self.theta
        if l is None:
            l=self.l
        self.__init__(theta=theta,l=l,s1_s2=self.s1_s2s,logger=self.logger)
    
    def wig_d_smoothing(self,s1_s2):
        if self.wig_d_taper_order_low<=0:
            return
        if self.wig_d_taper_order_high<=0:
            self.wig_d_taper_order_high=self.wig_d_taper_order_low+2
        bessel_order=np.absolute(s1_s2[0]-s1_s2[1])
        zeros=jn_zeros(bessel_order,max(self.wig_d_taper_order_low,self.wig_d_taper_order_high))
        l_max_low=zeros[self.wig_d_taper_order_low-1]/self.theta[s1_s2]
        if l_max_low.max()>self.l.max():
            print('Wigner ell max too low for theta_min. Recommendation based on first few zeros of bessel ',s1_s2,' :',zeros[:5]/self.theta[s1_s2].min())
        l_max_high=zeros[self.wig_d_taper_order_high-1]/self.theta[s1_s2]
        if self.wig_d_taper_order_high==0:
            l_max_high[:]=self.l.max()
        l_max_low[l_max_low>self.l.max()]=self.l.max()
        l_max_high[l_max_high>self.l.max()]=self.l.max()
        taper_f=np.cos((self.l[None,:]-l_max_low[:,None])/(l_max_high[:,None]-l_max_low[:,None])*np.pi/2.)
        x=self.l[None,:]>=l_max_low[:,None]
        y=self.l[None,:]>=l_max_high[:,None]
        taper_f[~x]=1
        taper_f[y]=0
        self.wig_d[s1_s2]=self.wig_d[s1_s2]*taper_f
    
    def cl_grid(self,l_cl=[],cl=[],taper=False,**kwargs):
        """
        Interpolate a given C_ell onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
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
        """
        Interpolate a given C_ell covariance onto the grid of ells for which WT is intialized. 
        This is to generalize in case user doesnot want to compute C_ell at every ell.
        Also apply tapering if needed.
        """
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

    def projected_correlation(self,l_cl=[],cl=[],s1_s2=[],taper=False,wig_d=None,**kwargs):
        """
        Get the projected correlation function from given c_ell.
        """
        if wig_d is None: #when using default wigner matrices, interpolate to ensure grids match.
            cl2=self.cl_grid(l_cl=l_cl,cl=cl,taper=taper,**kwargs)
            w=np.dot(self.wig_d[s1_s2]*self.grad_l*self.norm,cl2)
        else:
            w=np.dot(wig_d,cl)
        return self.theta[s1_s2],w

    def projected_covariance(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                             wig_d1=None,wig_d2=None,
                            taper=False,**kwargs):
        if wig_d1 is not None:
            return self.theta[s1_s2],wig_d1@np.diag(cl_cov)@wig_d2
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        #when cl_cov can be written as vector, eg. gaussian covariance
        cl2=self.cl_grid(l_cl=l_cl,cl=cl_cov,taper=taper,**kwargs)
        cov=np.einsum('rk,k,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm),cl2*self.grad_l,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
        #FIXME: Check normalization
        #FIXME: need to allow user to input wigner matrices.
        return self.theta[s1_s2],cov

    def projected_covariance2(self,l_cl=[],cl_cov=[],s1_s2=[],s1_s2_cross=None,
                              wig_d1=None,wig_d2=None,
                                taper=False,**kwargs):
        if wig_d1 is not None:
            print(wig_d1.shape,cl_cov.shape)
            return self.theta[s1_s2],wig_d1@cl_cov@wig_d2.T
        #when cl_cov is a 2-d matrix
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        cl_cov2=cl_cov  #self.cl_cov_grid(l_cl=l_cl,cl_cov=cl_cov,s1_s2=s1_s2,taper=taper,**kwargs)

        cov=np.einsum('rk,kk,sk->rs',self.wig_d[s1_s2]*np.sqrt(self.norm)*self.grad_l,cl_cov2,
                    self.wig_d[s1_s2_cross]*np.sqrt(self.norm),optimize=True)
#         cov=np.dot(self.wig_d[s1_s2]*self.grad_l*np.sqrt(self.norm),np.dot(self.wig_d[s1_s2_cross]*np.sqrt(self.norm),cl_cov2).T)
        # cov*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],cov

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

    def skewness(self,l_cl=[],cl1=[],cl2=[],cl3=[],s1_s2=[],taper=False,**kwargs):
        """
        Because we can do 6 point functions as well :). 
        """
        cl1=self.cl_grid(l_cl=l_cl,cl=cl1,s1_s2=s1_s2,taper=taper,**kwargs)
        cl2=self.cl_grid(l_cl=l_cl,cl=cl2,s1_s2=s1_s2,taper=taper,**kwargs)
        cl3=self.cl_grid(l_cl=l_cl,cl=cl3,s1_s2=s1_s2,taper=taper,**kwargs)
        skew=np.einsum('ji,ki,li',self.wig_d[s1_s2],self.wig_d[s1_s2],
                        self.wig_d[s1_s2]*cl1*cl2*cl3)
        skew*=self.norm
        #FIXME: Check normalization
        return self.theta[s1_s2],skew
