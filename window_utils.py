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
from dask.distributed import Client,get_client
import h5py
import zarr
from dask.threaded import get


class window_utils():
    def __init__(self,window_l=None,window_lmax=None,l=None,corrs=None,m1_m2s=None,use_window=None,f_sky=None,
                do_cov=False,cov_utils=None,corr_indxs=None,z_bins=None,HT=None,xi_bin_utils=None,do_xi=False,
                store_win=False,Win=None):
        self.Win=Win
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
        self.store_win=store_win

        self.set_wig3j()
        
#         if self.Win is None:
#             if not use_window:
#                 self.coupling_M=np.diag(np.ones_like(self.l))
#                 self.coupling_G=np.diag(1./self.cov_utils.gaussian_cov_norm)
        if self.use_window:
            self.set_window(corrs=self.corrs,corr_indxs=self.corr_indxs)
        
#         if self.store_win:
#             self.Win=self.store_win_func(Win=self.Win,corrs=self.corrs,corr_indxs=self.corr_indxs)

    def coupling_matrix(self,win,wig_3j_1,wig_3j_2):
        return np.dot(wig_3j_1*wig_3j_2,win*(2*self.window_l+1)   )/4./np.pi #FIXME: check the order of division by l.

    def coupling_matrix_large(self,win,wig_3j_1,wig_3j_2,step=1000):
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

    def get_window_power_cl(self,corr={},indxs={}):
        win={}
        if not self.use_window:
            win={'cl':self.f_sky, 'M':self.coupling_M,'xi':1,'xi_b':1}
            return win

        m1m2=np.absolute(self.m1_m2s[(corr[0],corr[1])]).flatten()
#         print(m1m2[0],self.wig_3j)
        wig_3j_1=self.wig_3j[m1m2[0]]
        wig_3j_2=self.wig_3j[m1m2[1]]

        z_bin1=self.z_bins[corr[0]][indxs[0]]
        z_bin2=self.z_bins[corr[1]][indxs[1]]
        alm1=z_bin1['window_alm']
        alm2=z_bin2['window_alm']

        win['cl']=hp.alm2cl(alms1=alm1,alms2=alm2,lmax_out=self.window_lmax) #This is f_sky*cl.
        win['M']=self.coupling_matrix_large(win['cl'], wig_3j_1,wig_3j_2)*(2*self.l[:,None]+1) #FIXME: check ordering
                #Note that this matrix leads to pseudo cl, which differs by factor of f_sky from true cl
        if self.do_xi:
            th,win['xi']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl'])
            win['xi_b']=self.binning.bin_1d(xi=win['xi'],bin_utils=self.xi_bin_utils[(0,0)])

        del alm1
        del alm2
        return win

    def get_window_power_cov(self,corr={},indxs={}):
        win={}
        if not self.use_window:
            win={'cl1324':self.f_sky,'M1324':self.coupling_G, 'M1423':self.coupling_G, 'cl1423':self.f_sky}
            return win

        def cov_m1m2s(corrs):
            m1m2=np.absolute(self.m1_m2s[corrs]).flatten()
            if m1m2[0]==m1m2[1]:
                return m1m2[0]
            else:
                return 0
            

        m1m2=cov_m1m2s(corrs=(corr[0],corr[2]))
        wig_3j13=self.wig_3j[m1m2]
        m1m2=cov_m1m2s(corrs=(corr[1],corr[3]))
        wig_3j24=self.wig_3j[m1m2]

        m1m2=cov_m1m2s(corrs=(corr[0],corr[3]))
        wig_3j14=self.wig_3j[m1m2]
        m1m2=cov_m1m2s(corrs=(corr[1],corr[2]))
        wig_3j23=self.wig_3j[m1m2]
        
        z_bin1=self.z_bins[corr[0]][indxs[0]]
        z_bin2=self.z_bins[corr[1]][indxs[1]]
        z_bin3=self.z_bins[corr[2]][indxs[2]]
        z_bin4=self.z_bins[corr[3]][indxs[3]]
        
        alm={}
        alm[13]=hp.map2alm(self.multiply_window(z_bin1['window'],z_bin3['window']))
        alm[24]=hp.map2alm(self.multiply_window(z_bin2['window'],z_bin4['window']))

        alm[14]=hp.map2alm(self.multiply_window(z_bin1['window'],z_bin4['window']))
        alm[23]=hp.map2alm(self.multiply_window(z_bin2['window'],z_bin3['window']))


        win['cl1324']=hp.alm2cl(alms1=alm[13],alms2=alm[24],lmax_out=self.window_lmax) #This is f_sky*cl.
        win['cl1423']=hp.alm2cl(alms1=alm[14],alms2=alm[23],lmax_out=self.window_lmax)
        win['M1324']=self.coupling_matrix_large(win['cl1324'], wig_3j13 , wig_3j24) #/np.gradient(self.l)
        win['M1423']=self.coupling_matrix_large(win['cl1423'], wig_3j14 , wig_3j23) #/np.gradient(self.l)
        if self.do_xi:
            th,win['xi1324']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl1324'])
            th,win['xi1423']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl1423'])
            win['xi_b1324']=self.binning.bin_1d(xi=win['xi1324'],bin_utils=self.xi_bin_utils[(0,0)])
            win['xi_b1423']=self.binning.bin_1d(xi=win['xi1423'],bin_utils=self.xi_bin_utils[(0,0)])
        
        del alm
        
        return win

    
    def return_dict_cl(self,result,corrs,corr_indxs):
            dic={}
            i=0
            for corr in corrs:
                dic[corr]={}
                dic[corr[::-1]]={}
                for (i,j) in corr_indxs[corr]:
                    dic[corr][(i,j)]=result[i]
                    dic[corr[::-1]][(j,i)]=result[i]
                    i+=1
            return dic
        
    def return_dict_cov(self,result,win_cov_tuple):
        dic={}
        i=0
        for (corr1,corr2,indx1,indx2) in win_cov_tuple:
            corr=corr1+corr2
            corr21=corr2+corr1
            indxs=indx1+indx2
            indxs2=indx2+indx1
            
            if dic.get(corr) is None:
                dic[corr]={}
            if dic.get(corr21) is None:
                dic[corr21]={}
            
            dic[corr][indxs]=result[i]
                                
            dic[corr][indxs2]=result[i]
            dic[corr21][indxs2]=result[i]
            dic[corr21][indxs]=result[i]
            i+=1
        return dic
    
    def set_window(self,corrs=None,corr_indxs=None,client=None):

        if self.store_win and client is None:
            client=get_client()
#             LC=LocalCluster(n_workers=1,processes=False,memory_limit='30gb',threads_per_worker=8,memory_spill_fraction=.99,
#                memory_monitor_interval='2000ms')
#             client=Client(LC)
        print(client)                
        if self.store_win:
            self.Win_cl={corr+indx: (self.get_window_power_cl,corr,indx) for corr in corrs for indx in corr_indxs[corr]}
        else:
            self.Win_cl={corr+indx: delayed(self.get_window_power_cl)(corr,indx) for corr in corrs for indx in corr_indxs[corr]}
        
        self.Win_cl.update({'W1': (self.return_dict_cl, [corr+indx for corr in corrs for indx in corr_indxs[corr]],corrs,corr_indxs)})
        self.Win=get(self.Win_cl,'W1')
        
        if self.do_cov:
            self.Win_cov={} 
            self.win_cov_tuple=None
            for ic1 in np.arange(len(corrs)):
                corr1=corrs[ic1]
                indxs_1=corr_indxs[corr1]
                n_indx1=len(indxs_1)

                for ic2 in np.arange(ic1,len(corrs)):
                    corr2=corrs[ic2]
                    indxs_2=corr_indxs[corr2]
                    n_indx2=len(indxs_2)

                    corr=corr1+corr2
                    
                    for i1 in np.arange(n_indx1):
                        start2=0
                        indx1=indxs_1[i1]
                        if corr1==corr2:
                            start2=i1
                        for i2 in np.arange(start2,n_indx2):
                            indx2=indxs_2[i2]
                            indxs=indx1+indx2
                            

                            if self.store_win:
                                self.Win_cov.update({corr+indxs:(self.get_window_power_cov,corr,indxs)})
                                
                            else:
                                self.Win_cov.update({corr+indxs: delayed(self.get_window_power_cov)(corr,indxs)})
                                
                            if self.win_cov_tuple is None:
                                self.win_cov_tuple=[(corr1,corr2,indx1,indx2)]
                            else:
                                self.win_cov_tuple.append((corr1,corr2,indx1,indx2))
                                
            self.Win_cov.update({'W1':(self.return_dict_cov,[corr1+corr2+indx1+indx2 for (corr1,corr2,indx1,indx2) in self.win_cov_tuple], self.win_cov_tuple)})
            self.Win['cov']=get(self.Win_cov,'W1')
        return self.Win

    def store_win_func(self,Win,corrs=None,corr_indxs=None):
        Win2={}
        for corr in corrs:
            corr2=corr[::-1]
            Win2[corr]={}
            Win2[corr2]={}
            for indx in corr_indxs[corr]:
                Win2[corr][indx]=Win[corr][indx].result()
#                         self.Win[corr][indx]=self.Win[corr][indx].result()
                (i,j)=indx
                Win2[corr2][(j,i)]=Win2[corr][(i,j)]

        if self.do_cov:
            Win2['cov']={}
            for corr in Win['cov'].keys():
                Win2['cov'][corr]={}
                if 'cl' in corr or 'xi' in corr or 'cov' in corr:
                    continue
                for indx in Win['cov'][corr].keys():
                    Win2['cov'][corr][indx]=Win['cov'][corr][indx].result()
#                         self.Win['cov'][corr][indx]=self.Win['cov'][corr][indx].result()
        return Win2