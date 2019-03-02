import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.stats import norm

iblack  = list(np.array([  0,  0,  0])/256.)
iblue   = list(np.array([ 33, 79,148])/256.)
ired    = list(np.array([204,  2,  4])/256.)
iorange = list(np.array([255,169,  3])/256.)
igray   = list(np.array([130,130,120])/256.)
igreen  = list(np.array([  8,153  ,0])/256.)

def cal_variance_FlatDist(a,b):
    return (b-a)**2/12.

def tune_xtick(ax):
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
    return ax

def tune_ytick(ax):
    for tick in ax.get_yticklabels():
        tick.set_rotation(45)
    for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
    return ax

class fisher_tool():
    def __init__(self,Fisher,par_cen,par_prior_variance=None):
        self.init_fisher(Fisher=Fisher,par_prior_variance=par_prior_variance)
        self.par_cen       = par_cen
        self.Npar = len(self.par_cen)
        self.cal_Cov_par()
        self.cal_Par_err()
        self.cal_Corr_par()
        self.set_alpha_ellip(nsigma_2Dposterior=1)  # 1 : 68.3% 2D contour, 2 : 95.4%, 3 : 99.7%
        
    def init_fisher(self,Fisher,par_prior_variance=None):
        if par_prior_variance is None:
            self.Fisher = Fisher
        else:
            self.Fisher_0prior = Fisher
            self.Fisher_prior = np.diag(1./np.array(par_prior_variance))
            self.Fisher        = self.Fisher_0prior + self.Fisher_prior
    
    def cal_Cov_par(self):
        self.Cov_par = np.linalg.inv(self.Fisher)
    
    def cal_Par_err(self):
        self.par_sigma1D = np.sqrt(np.diag(self.Cov_par))
    
    def cal_Corr_par(self):
        #compute parameter correlation coefficient
        self.Corr_par = np.zeros([self.Npar,self.Npar])
        for i in range(self.Npar):
            for j in range(self.Npar):
                self.Corr_par[i][j]=self.Cov_par[i][j]/np.sqrt(self.Cov_par[i][i]*self.Cov_par[j][j])

        
    def get_2D_ellips_info(self,id_1,id_2): # id number respect to Fisher matrix id

        Cov_par2D = self.Cov_par[[id_1,id_2],:][:,[id_1,id_2]]
        ellip_sigma2, v = np.linalg.eig(Cov_par2D)
        id_rank = np.argsort(ellip_sigma2)[::-1]
        ellip_sigma2 = ellip_sigma2[id_rank]  ;  v = v[id_rank]    # change the ranking of eigenvalues and corr. eigenvectors (bigger first)
        a = np.sqrt(ellip_sigma2[0])
        b = np.sqrt(ellip_sigma2[1])
        theta_rad = np.arctan2(2*Cov_par2D[0,1],Cov_par2D[0,0]-Cov_par2D[1,1])/2.
        theta_deg = theta_rad*180/np.pi

        return a, b, theta_deg
    
    def set_alpha_ellip(self,nsigma_2Dposterior):
        sigma_alpha = {1:1.52,2:2.48,3:3.44}  #Table5 of arXiv0906.4123
        self.alpha  = sigma_alpha[nsigma_2Dposterior]

    def plot_ellips2D(self,ax,id_1,id_2,color=iblue,alpha=0.2,ls="-",fid=None,axlim=None):
        a, b, theta_deg = self.get_2D_ellips_info(id_1,id_2)
        ells = Ellipse((self.par_cen[id_1], self.par_cen[id_2]), 2*a*self.alpha, 2*b*self.alpha, theta_deg)
        
        ax.add_artist(ells)
        ells.set_clip_box(ax.bbox)
        ells.set_edgecolor(tuple(color+[1.]))
        ells.set_facecolor(tuple(color+[alpha]))
        
        ells.set_linestyle(ls)
        ells.set_linewidth(2)
        
        corr_coe="%.2f"%self.Corr_par[id_1][id_2]
        ax.text(0.75,0.82, corr_coe ,transform=ax.transAxes,color='blue',fontsize=15)
        
        if fid is not None:
            ax.axvline(x=fid[0],color='silver',zorder=-1)
            ax.axhline(y=fid[1],color='silver',zorder=-2)
            
        if axlim is not None:
            ax.set_xlim(axlim[0])
            ax.set_ylim(axlim[1])
        
        return ax
    
    def plot_gauss1D(self,ax,id_1,color=iblue,ls='-',fid=None,axlim=None):
        mu    = self.par_cen[id_1]
        sigma = self.par_sigma1D[id_1]

        x     = np.linspace(mu-3*sigma, mu+3*sigma, 300)
        y     = norm.pdf(x,mu,sigma)
        x_1s  = np.linspace(mu-1*sigma, mu+1*sigma, 300)
        y_1s  = norm.pdf(x_1s,mu,sigma)

        ax.plot(x,y,linestyle=ls,linewidth=2,color=color)
        ax.fill_between(x_1s,[0.]*len(x_1s), y_1s, facecolor=color, alpha=0.2)
        
        if fid is not None:
            ax.axvline(x=fid,color='silver',zorder=-1)
    
        if axlim is not None:
            ax.set_xlim(axlim)

        