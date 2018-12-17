import dask
from dask import delayed
from wigner_transform import *
from binning import *
from cov_utils import *
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import warnings,logging
from distributed import LocalCluster
from dask.distributed import Client  
import h5py
import zarr


class window_utils():
    def __init__(self,window_l=None,window_lmax=None,l=None,corrs=None,m1_m2s=None,use_window=None,f_sky=None,
                    do_cov=False,cov_utils=None,corr_indxs=None,z_bins=None,HT=None,xi_bin_utils=None,do_xi=False):
        self.Win=None
        self.wig_3j=None
        self.window_lmax=window_lmax
        self.window_l=window_l
        self.l=l
        self.HT=HT
        self.corrs=corrs
        self.m1_m2s=m1_m2s
        self.use_window=use_window
        self.do_cov=do_cov
        self.do_xi=do_xi
        self.xi_bin_utils=xi_bin_utils
        self.binning=binning()
        self.cov_utils=cov_utils
        self.corr_indxs=corr_indxs
        self.z_bins=z_bins   
        self.f_sky=f_sky

        self.set_wig3j()
        
        if not use_window:
            self.coupling_M=np.diag(np.ones_like(self.l))
            self.coupling_G=np.diag(1./self.cov_utils.gaussian_cov_norm)
#         else:
        self.set_window(corrs=self.corrs,corr_indxs=self.corr_indxs)
#         if use_window:
#             self.wig_DB.close()

    def coupling_matrix(self,win,wig_3j_1,wig_3j_2):
        return np.dot(wig_3j_1*wig_3j_2,win*(2*self.window_l+1)   )/4./np.pi #FIXME: check the order of division by l.

    def coupling_matrix_large(self,win,wig_3j_1,wig_3j_2,step=100):
        nl=len(self.l)
        M=np.zeros((nl,nl))
        lm=0
        while lm<nl:
#             wig=wig_3j_1[self.l[lm:lm+step],:,:self.window_lmax+1][:,np.int32(self.l),:]
#             wig=wig*wig_3j_2[self.l[lm:lm+step],:,:self.window_lmax+1][:,np.int32(self.l),:]
#             M[lm:lm+step,:]=np.dot(win*(2*self.window_l+1),wig )/4./np.pi #FIXME: check the order of division by l.

            wig=wig_3j_1.oindex[np.int32(self.window_l),np.int32(self.l[lm:lm+step]),np.int32(self.l)]
            wig=wig*wig_3j_2.oindex[np.int32(self.window_l),np.int32(self.l[lm:lm+step]),np.int32(self.l)]
            M[lm:lm+step,:]=np.einsum('ijk,i->jk',wig, win*(2*self.window_l+1), optimize=True )/4./np.pi #FIXME: check the order of division by l.
            lm+=step
        return M

    
    def set_wig3j(self,wig_file='temp/wigner_test.h5'):
        self.wig_3j={}
        if not self.use_window:
            return

        m_s=np.concatenate([np.abs(i).flatten() for i in self.m1_m2s.values()])
        m_s=np.unique(m_s)
        
#         self.wig_DB=h5py.File(wig_file, 'r')
        fname='temp/dask_wig3j_l5000_w500_{m}_asym50.zarr'
        for m in m_s:
#             self.wig_3j[m]=Wigner3j_parallel( m, -m, 0, self.l, self.l, self.window_l)
#             self.wig_3j[m]=self.wig_DB[str(m)]
            self.wig_3j[m]=zarr.open(fname.format(m=m))
        print('wigner done',self.wig_3j.keys())


    def multiply_window(self,win1,win2):
        W=win1*win2
        x=np.logical_or(win1==hp.UNSEEN, win2==hp.UNSEEN)
        W[x]=hp.UNSEEN
        return W

    def get_window_power(self,corr={},indxs={}):
        win={}
        if not self.use_window:
            win={'cl':self.f_sky, 'M':self.coupling_M,'xi':1,'xi_b':1}
            if len(indxs)>2:
                win={'cl1324':self.f_sky,'M1324':self.coupling_G, 'M1423':self.coupling_G, 'cl1423':self.f_sky}
            return win

        m1m2=np.absolute(self.m1_m2s[(corr[0],corr[1])]).flatten()
#         print(m1m2[0],self.wig_3j)
        wig_3j_1=self.wig_3j[m1m2[0]]
        wig_3j_2=self.wig_3j[m1m2[1]]

        z_bin1=self.z_bins[corr[0]][indxs[0]]
        z_bin2=self.z_bins[corr[1]][indxs[1]]
        alm1=z_bin1['window_alm']
        alm2=z_bin2['window_alm']
        do_cov=False

        def cov_m1m2s(corrs):
            m1m2=np.absolute(self.m1_m2s[corrs]).flatten()
            if m1m2[0]==m1m2[1]:
                return m1m2[0]
            else:
                return 0
            
        if len(indxs)>2:
            m1m2=cov_m1m2s(corrs=(corr[0],corr[2]))
            wig_3j13=self.wig_3j[m1m2]
            m1m2=cov_m1m2s(corrs=(corr[1],corr[3]))
            wig_3j24=self.wig_3j[m1m2]

            m1m2=cov_m1m2s(corrs=(corr[0],corr[3]))
            wig_3j14=self.wig_3j[m1m2]
            m1m2=cov_m1m2s(corrs=(corr[1],corr[2]))
            wig_3j23=self.wig_3j[m1m2]
            
            z_bin3=self.z_bins[corr[2]][indxs[2]]
            z_bin4=self.z_bins[corr[3]][indxs[3]]
            alm13=hp.map2alm(self.multiply_window(z_bin1['window'],z_bin3['window']))
            alm24=hp.map2alm(self.multiply_window(z_bin2['window'],z_bin4['window']))

            alm14=hp.map2alm(self.multiply_window(z_bin1['window'],z_bin4['window']))
            alm23=hp.map2alm(self.multiply_window(z_bin2['window'],z_bin3['window']))
            do_cov=True

        if not do_cov:
            win['cl']=hp.alm2cl(alms1=alm1,alms2=alm2,lmax_out=self.window_lmax) #This is f_sky*cl.
            win['M']=self.coupling_matrix_large(win['cl'], wig_3j_1,wig_3j_2)*(2*self.l[:,None]+1) #FIXME: check ordering
            #Note that this matrix leads to pseudo cl, which differs by factor of f_sky from true cl
            if self.do_xi:
                th,win['xi']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl'])
                win['xi_b']=self.binning.bin_1d(xi=win['xi'],bin_utils=self.xi_bin_utils[(0,0)])
        if do_cov:
            win['cl1324']=hp.alm2cl(alms1=alm13,alms2=alm24,lmax_out=self.window_lmax) #This is f_sky*cl.
            win['cl1423']=hp.alm2cl(alms1=alm14,alms2=alm23,lmax_out=self.window_lmax)
            win['M1324']=self.coupling_matrix_large(win['cl1324'], wig_3j13 , wig_3j24) #/np.gradient(self.l)
            win['M1423']=self.coupling_matrix_large(win['cl1423'], wig_3j14 , wig_3j23) #/np.gradient(self.l)

#             win['M1324']/=self.f_sky**3 #FIXME: Where is this factor coming from
#             win['M1423']/=self.f_sky**3
        return win

    def set_window(self,corrs=None,corr_indxs=None,client=None):
        if client is None:
            LC=LocalCluster(n_workers=1,processes=False,memory_limit='30gb',threads_per_worker=8,memory_spill_fraction=.99,
               memory_monitor_interval='2000ms')
            client=Client(LC)
        
        if self.Win is None:
            self.Win={'cl':{},'xi':{}}

        for corr in corrs:
#             self.Win['cl'][corr]={}
#             self.Win['xi'][corr]={}
            corr2=corr[::-1]
            self.Win[corr]={}
            self.Win[corr2]={}

            for (i,j) in corr_indxs[corr]:
                zb1=self.z_bins[corr[0]][i]
                zb2=self.z_bins[corr[1]][j]
                self.Win[corr][(i,j)]=delayed(self.get_window_power)(corr,(i,j))
#                 self.Win[corr][(i,j)]=client.submit(self.get_window_power,corr,(i,j))
                self.Win[corr2][(j,i)]=self.Win[corr][(i,j)]

        if self.do_cov:
            self.Win['cov']={'cl':{},'xi':{}}
            for ic1 in np.arange(len(corrs)):
                corr1=corrs[ic1]
                indxs_1=corr_indxs[corr1]
                n_indx1=len(indxs_1)

                for ic2 in np.arange(ic1,len(corrs)):
                    corr2=corrs[ic2]
                    indxs_2=corr_indxs[corr2]
                    n_indx2=len(indxs_2)

                    corr=corr1+corr2
                    corr2=corr2+corr1
                    self.Win['cov'][corr]={}
                    self.Win['cov'][corr2]={}

                    for i1 in np.arange(n_indx1):
                        start2=0
                        indx1=indxs_1[i1]
                        if corr1==corr2:
                            start2=i1
                        for i2 in np.arange(start2,n_indx2):
                            indx2=indxs_2[i2]
                            indxs=indx1+indx2
                            indxs2=indx2+indx1
                            zb1=self.z_bins[corr1[0]][indx1[0]]
                            zb2=self.z_bins[corr1[1]][indx1[1]]
                            zb3=self.z_bins[corr2[0]][indx2[0]]
                            zb4=self.z_bins[corr2[1]][indx2[1]]
                            self.Win['cov'][corr][indxs]=delayed(self.get_window_power)(corr,indxs)
#                             self.Win['cov'][corr][indxs]=client.submit(self.get_window_power,corr,indxs)
                            self.Win['cov'][corr][indxs2]=self.Win['cov'][corr][indxs]
                            self.Win['cov'][corr2][indxs2]=self.Win['cov'][corr][indxs]
                            self.Win['cov'][corr2][indxs]=self.Win['cov'][corr][indxs]

#         for corr in corrs:
#                 #for indx in self.Win[corr].keys():
#             for indx in corr_indxs[corr]:
# #                     self.Win[corr][indx]=self.Win[corr][indx].compute()
#                     self.Win[corr][indx]=self.Win[corr][indx].result()
#                     (i,j)=indx
#                     corr2=corr[::-1]
#                     self.Win[corr2][(j,i)]=self.Win[corr][(i,j)]

#         if self.do_cov:
#             for corr in self.Win['cov'].keys():
#                 if 'cl' in corr or 'xi' in corr or 'cov' in corr:
#                     continue
#                 for indx in self.Win['cov'][corr].keys():
# #                     self.Win['cov'][corr][indx]=self.Win['cov'][corr][indx].compute()
#                     self.Win['cov'][corr][indx]=self.Win['cov'][corr][indx].result()
        return self.Win
