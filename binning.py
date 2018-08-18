from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
import numpy as np
import itertools
from torch_utils import *

class binning():
    def __init__(self,rmin=0.1,rmax=100,kmax=10,kmin=1.e-4,n_zeros=1000,n_zeros_step=1000,
                 j_nu=[0],prune_r=0,prune_log_space=True):
        pass

    def bin_utils(self,r=[],r_bins=[],r_dim=2,wt=1,mat_dims=None):
        bu={}
        bu['bin_center']=0.5*(r_bins[1:]+r_bins[:-1])
        bu['n_bins']=len(r_bins)-1
        bu['bin_indx']=np.digitize(r,r_bins)-1

        binning_mat=tc.zeros((len(r),bu['n_bins']),dtype=r.dtype,device=r.device)
        for i in tc.arange(len(r)):
            if bu['bin_indx'][i]<0 or bu['bin_indx'][i]>=bu['n_bins']:
                continue
            binning_mat[i,bu['bin_indx'][i]]=1.
        bu['binning_mat']=binning_mat

        r2=tc.sort(tc.unique(tc.cat((r,r_bins))))[0] #this takes care of problems around bin edges
        dr=tc_gradient(r2)
        r2_idx=[i for i in tc.arange(len(r2)) if r2[i] in r]
        dr=dr[r2_idx]
        bu['r_dr']=r**(r_dim-1)*dr
        bu['r_dr']*=wt
        # bu['norm']=bu['r_dr'].mm(binning_mat)
        bu['norm']=(binning_mat.transpose(1,0)*bu['r_dr']).sum(1)

        x=r_bins[1:]<=r.min()#+r_bins[:-1]>=r.max()
        bu['norm'][x]=np.inf
        x=r_bins[:-1]>=r.max()
        bu['norm'][x]=np.inf #FIXME: This should be done in 1-2 lines


        if mat_dims is not None:
            bu['r_dr_m']={}
            bu['norm_m']={}
            ls=['i','j','k','l','m']
            for ndim in mat_dims:
                s1=ls[0]
                s2=ls[0]
                r_dr_m=bu['r_dr'].clone()
                norm_m=bu['norm'].clone()
                for i in tc.arange(ndim-1):
                    s1=s2+','+ls[i+1]
                    s2+=ls[i+1]
                    r_dr_m=tc.einsum(s1+'->'+s2,(r_dr_m,bu['r_dr']))#works ok for 2-d case
                    norm_m=tc.einsum(s1+'->'+s2,(norm_m,bu['norm']))#works ok for 2-d case
                bu['r_dr_m'][ndim]=r_dr_m
                bu['norm_m'][ndim]=norm_m
        return bu

    def bin_1d(self,xi=[],bin_utils=None):
        xi_b=(bin_utils['binning_mat'].transpose(1,0)*bin_utils['r_dr']*xi).sum(1)
        #np.dot(xi*bin_utils['r_dr'],bin_utils['binning_mat'])
        xi_b/=bin_utils['norm']
        return xi_b

    def bin_2d(self,r=[],cov=[],r_bins=[],r_dim=2,bin_utils=None):
        #r_dr=bin_utils['r_dr']
        #cov_r_dr=cov*bin_utils['r_dr_m'][2]#np.outer(r_dr,r_dr)
        binning_mat=bin_utils['binning_mat']
        # cov_b=np.dot(binning_mat.T, np.dot(cov*bin_utils['r_dr_m'][2],binning_mat) )
        cov_b=binning_mat.transpose(1,0).mm((cov*bin_utils['r_dr_m'][2]).mm(binning_mat))
        cov_b/=bin_utils['norm_m'][2]
        return cov_b

    def bin_mat(self,r=[],mat=[],r_bins=[],r_dim=2,bin_utils=None):#works for cov and skewness
        ndim=len(mat.shape)
        n_bins=bin_utils['n_bins']
        bin_idx=bin_utils['bin_indx']#np.digitize(r,r_bins)-1
        r_dr=bin_utils['r_dr']
        r_dr_m=bin_utils['r_dr_m'][ndim]

        mat_int=np.zeros([n_bins]*ndim,dtype='float64')
        norm_int=np.zeros([n_bins]*ndim,dtype='float64')

        mat_r_dr=mat*r_dr_m # same as cov_r_dr=cov*np.outer(r_dr,r_dr)
        norm_ijk=bin_utils['norm_m'][ndim]
        for indxs in itertools.product(tc.arange(min(bin_idx),n_bins),repeat=ndim):
            x={}#np.zeros_like(mat_r_dr,dtype='bool')
            mat_t=[]
            for nd in tc.arange(ndim):
                slc = [slice(None)] * (ndim)
                #x[nd]=bin_idx==indxs[nd]
                slc[nd]=bin_idx==indxs[nd]
                if nd==0:
                    mat_t=mat_r_dr[slc]
                else:
                    mat_t=mat_t[slc]
            mat_int[indxs]=np.sum(mat_t)
        mat_int/=norm_ijk
        return mat_int




#more basic binning code for testing.
def bin_cov(r=[],cov=[],r_bins=[]):
    bin_center=tc.sqrt(r_bins[1:]*r_bins[:-1])
    n_bins=len(bin_center)
    cov_int=tc.zeros((n_bins,n_bins))
    bin_idx=np.digitize(r,r_bins)-1
    r2=tc.sort(tc.unique(tc.cat(r,r_bins)))[0] #this takes care of problems around bin edges
    dr=tc_gradient(r2)
    r2_idx=[i for i in tc.arange(len(r2)) if r2[i] in r]
    dr=dr[r2_idx]
    r_dr=r*dr
    cov_r_dr=cov*tc.ger(r_dr,r_dr)
    for i in tc.arange(min(bin_idx+1),n_bins):
        xi=bin_idx==i
        for j in tc.arange(min(bin_idx),n_bins):
            xj=bin_idx==j
            norm_ij=tc.sum(r_dr[xi])*tc.sum(r_dr[xj])
            if i==j:
                print( i,j,norm_ij)
            if norm_ij==0:
                continue
            cov_int[i][j]=tc.sum(cov_r_dr[xi,:][:,xj])/norm_ij
    #cov_int=np.nan_to_num(cov_int)
#         print np.diag(cov_r_dr)
    return cov_int
