import dask
from dask import delayed
import sparse
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
import time
from multiprocessing import Pool,cpu_count



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

    def coupling_matrix(self,win,wig_3j_1,wig_3j_2,W_pm=0):
        #need to add E/B things
        return np.dot(wig_3j_1*wig_3j_2,win*(2*self.window_l+1)   )/4./np.pi #FIXME: check the order of division by l.

    def coupling_matrix_large(self,win,m1m2,wig_3j_2,lm=None,W_pm=0): 
        nl=len(self.l)
#         M=np.zeros((nl,nl))
#         return M

        nwl=len(self.window_l)
        step=self.step
        
        m1m2=np.sort(m1m2)

        t1=time.time()

        wig=wig_3j_2['w2']#[str(m1m2[0])+str(m1m2[1])] #[lm]#.compute()

        t2=time.time()
        mf=1
#         mf=sparse.COO(np.atleast_1d([1]))
        if W_pm!=0:
            if W_pm==2: #W_+
                mf=wig_3j_2['mf_p']
            if W_pm==-2: #W_-
                mf=wig_3j_2['mf_n']

        t3=time.time()        

        t4=time.time()
        #M[lm:lm+step,:]
        M=np.einsum('ijk,i->jk',wig.todense()*mf, win*(2*self.window_l+1), optimize=True )/4./np.pi #FIXME: check the order of division by l.
#         M=np.zeros((min(nl,step),nl))
#             Mi=np.einsum('ijk,i->jk',wig, win*(2*self.window_l+1), optimize=True )/4./np.pi #FIXME: check the order of division by l.
#         lm+=step
        t5=time.time()
        return M

    def set_wig3j_step_multiplied(self,m1=0,m2=0,lm=None,step=10):
        wig_temp={}
        
        wig_temp[m1]=self.wig_3j[m1].oindex[np.int32(self.window_l),np.int32(self.l[lm:lm+step]),np.int32(self.l)]
        if m1==m2:
            wig_temp[m2]=wig_temp[m1]
        else:
            wig_temp[m2]=self.wig_3j[m2].oindex[np.int32(self.window_l),np.int32(self.l[lm:lm+step]),np.int32(self.l)]
            
        out={'w2':sparse.COO(wig_temp[m1]*wig_temp[m2])} #sparse leads to small hit in in time when doing dot products but helps with the memory overall.
        
#         out={'w2':np.zeros((len(self.window_l),min(self.step,len(self.l)),len(self.l)),dtype='float32')}
#         print(out['w2'].shape)
        
        if m1>=0 or m2>=0: #used for some cov windows as well, where we otherwise assume spin 0
            li1=np.int32(self.window_l).reshape(len(self.window_l),1,1)
            li3=np.int32(self.l).reshape(1,1,len(self.l))
            li2=np.int32(self.l[lm:lm+step]).reshape(1,len(self.l[lm:lm+step]),1)
            mf=(-1)**(li1+li2+li3)
            
            out['mf_p']=np.int8((1.+mf)/2.) #memory hog... bool doesn't help, as it is only byte size in numpy. This should be ok for now. 
#             print(np.array_equal(out['mf_p'], out['mf_p'].astype(bool)) ) #check if array is 0,1
            
            if m1>=0 and m2>=0:#used for some cov windows as well, where we otherwise assume spin 0
                out['mf_n']=np.int8((1.-mf)/2.) #memory hog
        return out


    
    def set_wig3j(self,wig_file='temp/wigner_test.h5',step=None):
        self.wig_3j={}
        if not self.use_window:
            return

        m_s=np.concatenate([np.abs(i).flatten() for i in self.m1_m2s.values()])
        self.m_s=np.sort(np.unique(m_s))
        
#         self.wig_DB=h5py.File(wig_file, 'r')
        fname='temp/dask_wig3j_l5000_w500_{m}_asym50.zarr'
        for m in self.m_s:
#             self.wig_3j[m]=Wigner3j_parallel( m, -m, 0, self.l, self.l, self.window_l)
#             self.wig_3j[m]=self.wig_DB[str(m)]
            self.wig_3j[m]=zarr.open(fname.format(m=m))
        
        nl=len(self.l)
        
        nwl=len(self.window_l)*1.0
        if step is None:
            step=np.int32(500.*(1000./nl)*(100./nwl)) #small step is useful for lower memory load
        self.step=step
        self.lms=np.arange(nl,step=step)
        
        self.wig_3j_2={}
        client=get_client()
        for lm in self.lms:
            self.wig_3j_2[lm]={}
            mi=0
            for m1 in self.m_s:
                for m2 in self.m_s[mi:]:
#                     self.wig_3j_2[str(m1)+str(m2)]={}              
                    self.wig_3j_2[lm][str(m1)+str(m2)]=delayed(self.set_wig3j_step_multiplied)(m1=m1,m2=m2,lm=lm,step=self.step)
            self.wig_3j_2[lm]=client.compute(self.wig_3j_2[lm]) #computing here helps with memory. Otherwise sometimes there are multiple calls in parallel, not sure why.
            mi+=1
            
        for lm in self.lms:
            self.wig_3j_2[lm]=self.wig_3j_2[lm].result()
        
        self.wig_m1m2s={}
        for corr in self.corrs:
            mi=np.sort(np.absolute(self.m1_m2s[corr]).flatten())
            self.wig_m1m2s[corr]=str(mi[0])+str(mi[1])
        print('wigner done',self.wig_3j.keys())


    def multiply_window(self,win1,win2):
        W=win1*win2
        x=np.logical_or(win1==hp.UNSEEN, win2==hp.UNSEEN)
        W[x]=hp.UNSEEN
        return W

    def get_window_power_cl(self,corr={},indxs={}):
#         print('cl window doing',corr,indxs)
        win={}
        win['corr']=corr
        win['indxs']=indxs
        if not self.use_window:
            win={'cl':self.f_sky, 'M':self.coupling_M,'xi':1,'xi_b':1}
            return win

        m1m2=np.absolute(self.m1_m2s[corr]).flatten()
        W_pm=0
        if np.sum(m1m2)!=0:
            W_pm=2 #we only deal with E mode\

        z_bin1=self.z_bins[corr[0]][indxs[0]]
        z_bin2=self.z_bins[corr[1]][indxs[1]]
        alm1=z_bin1['window_alm']
        alm2=z_bin2['window_alm']

        win['cl']=hp.alm2cl(alms1=alm1,alms2=alm2,lmax_out=self.window_lmax) #This is f_sky*cl.
        win['W_pm']=W_pm
        win['m1m2']=m1m2
        if self.do_xi:
            th,win['xi']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl'])
            win['xi_b']=self.binning.bin_1d(xi=win['xi'],bin_utils=self.xi_bin_utils[(0,0)])
        
        win['M']={} #self.coupling_matrix_large(win['cl'], m1m2,wig_3j_2=wig_3j_2,W_pm=W_pm)*(2*self.l[:,None]+1) #FIXME: check ordering
        win['M_B']=None
        return win

    def get_cl_coupling_lm(self,win,lm,wig_3j_2):
        win2={'M':{},'M_B':{}}
        if lm==0:
            win2=win
        win2['M'][lm]=self.coupling_matrix_large(win['cl'], win['m1m2'],wig_3j_2=wig_3j_2,lm=lm,W_pm=win['W_pm'])
#         win2['M'][lm]=np.zeros((min(self.step,len(self.l)), len(self.l)))
        
        if win['corr']==('shear','shear') and win['indxs'][0]==win['indxs'][1]:
#             win2['M_B']={lm:np.zeros((min(self.step,len(self.l)), len(self.l)))}
            win2['M_B']={lm:self.coupling_matrix_large(win['cl'], win['m1m2'], wig_3j_2,lm=lm,W_pm=-2)}
                #Note that this matrix leads to pseudo cl, which differs by factor of f_sky from true cl
            
        return win2

    def return_dict_cl(self,result,corrs):
        dic={}
        nl=len(self.l)
        
        for corr in corrs:
            dic[corr]={}
            dic[corr[::-1]]={}
        
        for ii in list(result[0].keys()):
            
            result_ii=result[0][ii]
            corr=result_ii['corr']
            indxs=result_ii['indxs']

            result0={}
            for k in result_ii.keys():
                result0[k]=result_ii[k]
                
            result0['M']=np.zeros((nl,nl))
            if corr==('shear','shear') and indxs[0]==indxs[1]:
                result0['M_B']=np.zeros((nl,nl))
                
            for lm in self.lms:
                result0['M'][lm:lm+self.step,:]+=result[lm][ii]['M'][lm]
                if corr==('shear','shear') and indxs[0]==indxs[1]:
                    result0['M_B'][lm:lm+self.step,:]+=result[lm][ii]['M_B'][lm]
                
                del result[lm][ii]

            dic[corr][indxs]=result0
            dic[corr[::-1]][indxs[::-1]]=result0

        return dic

    def cov_m1m2s(self,corr): #when spins are not same, we set them to 0. Should be ok for l>~50 ish
            m1m2=np.absolute(self.m1_m2s[corr]).flatten()
            if m1m2[0]==m1m2[1]:
                return m1m2[0]
            else:
                return 0
    def get_window_power_cov(self,corr1=None,corr2=None,indxs1=None,indxs2=None):
        win={}
        corr=corr1+corr2
        indxs=indxs1+indxs2
        win['corr1']=corr1
        win['corr2']=corr2
        win['indxs1']=indxs1
        win['indxs2']=indxs2
        if not self.use_window:
            win={'cl1324':self.f_sky,'M1324':self.coupling_G, 'M1423':self.coupling_G, 'cl1423':self.f_sky}
            return win
       
        def get_window_spins(cov_indxs=[(0,2),(1,3)]):    #W +/- factors based on spin
            W_pm=[0]
#             corr1=(corr[cov_indxs[0][0]],corr[cov_indxs[0][1]])
#             corr2=(corr[cov_indxs[1][0]],corr[cov_indxs[1][1]])
            s=[np.sum(self.m1_m2s[corr1]),np.sum(self.m1_m2s[corr2])]

            if s[0]==2 and s[1]==2: #gE,gE
                W_pm=[2]
            elif 4 in s and 2 in s: #EE,gE
                W_pm=[2]
            elif 0 in s and 2 in s: #gg,gE
                W_pm=[2]
            elif 4 in s and 0 in s: #EE,gg
                W_pm=[2]
                for i in np.arange(2):
                    if indxs[cov_indxs[i][0]]==indxs[cov_indxs[i][1]] and s[i]==4: #auto correlation, include B modes
                        W_pm=[2,-2]
            elif s[0]==4 and s[1]==4: #EE,EE
                W_pm=[2]
                for i in np.arange(2):
                    if indxs[cov_indxs[i][0]]==indxs[cov_indxs[i][1]] and s[i]==4: #auto correlation, include B modes
                        W_pm=[2,-2]

            return W_pm
            
            
        m1m2s={}
        
        m1m2s[1324]=np.array([self.cov_m1m2s(corr=(corr[0],corr[2])), #13
                              self.cov_m1m2s(corr=(corr[1],corr[3])) #24
                    ])
        
        m1m2s[1423]=np.array([self.cov_m1m2s(corr=(corr[0],corr[3])), #14
                              self.cov_m1m2s(corr=(corr[1],corr[2])) #23
                    ])
        
        W_pm={} #W +/- factors based on spin
        W_pm[1324]=get_window_spins(cov_indxs=[(0,2),(1,3)])
        W_pm[1423]=get_window_spins(cov_indxs=[(0,3),(1,2)])

        z_bin1=self.z_bins[corr[0]][indxs[0]]
        z_bin2=self.z_bins[corr[1]][indxs[1]]
        z_bin3=self.z_bins[corr[2]][indxs[2]]
        z_bin4=self.z_bins[corr[3]][indxs[3]]
        
        win['cl1324']=hp.anafast(map1=self.multiply_window(z_bin1['window'],z_bin3['window']),
                                 map2=self.multiply_window(z_bin2['window'],z_bin4['window']),
                                 lmax=self.window_lmax
                        )
        win['cl1423']=hp.anafast(map1=self.multiply_window(z_bin1['window'],z_bin4['window']),
                                 map2=self.multiply_window(z_bin2['window'],z_bin3['window']),
                                 lmax=self.window_lmax
                            )
        
        win['M1324']={wp:{} for wp in W_pm[1324]}
        win['M1423']={wp:{} for wp in W_pm[1423]}
            
        if self.do_xi:
            th,win['xi1324']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl1324'])
            th,win['xi1423']=self.HT.projected_correlation(l_cl=self.window_l,m1_m2=(0,0),cl=win['cl1423'])
            win['xi_b1324']=self.binning.bin_1d(xi=win['xi1324'],bin_utils=self.xi_bin_utils[(0,0)])
            win['xi_b1423']=self.binning.bin_1d(xi=win['xi1423'],bin_utils=self.xi_bin_utils[(0,0)])
                    
        win['W_pm']=W_pm
        win['m1m2']=m1m2s
        return win
    
    def get_cov_coupling_lm(self,win,lm,wig_3j_2_1324,wig_3j_2_1423):
        for wp in win['W_pm'][1324]:
            win['M1324'][wp][lm]=self.coupling_matrix_large(win['cl1324'], win['m1m2'][1324],lm=lm,wig_3j_2=wig_3j_2_1324,W_pm=wp)
    
        for wp in win['W_pm'][1423]:
            win['M1423'][wp][lm]=self.coupling_matrix_large(win['cl1423'], win['m1m2'][1423],lm=lm,wig_3j_2=wig_3j_2_1423,W_pm=wp) #/np.gradient(self.l)
        return win
        
    def return_dict_cov(self,result,win_cov_tuple): #to compute the covariance graph generated in set window
        dic={}
        nl=len(self.l)
        
        for ii in list(result[0].keys()):#np.arange(len(result)):
            result0={}

            for k in result[0][ii].keys():
                result0[k]=result[0][ii][k]
            
            W_pm=result[0][ii]['W_pm']
            corr1=result[0][ii]['corr1']
            corr2=result[0][ii]['corr2']
            indx1=result[0][ii]['indxs1']
            indx2=result[0][ii]['indxs2']
            
            result0['M1324']={wp: np.zeros((nl,nl)) for wp in W_pm[1324]}
            result0['M1423']={wp: np.zeros((nl,nl)) for wp in W_pm[1423]}
                
            for lm in self.lms:
                for wp in W_pm[1324]:
                    result0['M1324'][wp][lm:lm+self.step,:]+=result[lm][ii]['M1324'][wp][lm]
                for wp in W_pm[1423]:
                    result0['M1423'][wp][lm:lm+self.step,:]+=result[lm][ii]['M1423'][wp][lm]
                    
                del result[lm][ii]
            
            corr=corr1+corr2
            corr21=corr2+corr1
            indxs=indx1+indx2
            indxs2=indx2+indx1
            
            if dic.get(corr) is None:
                dic[corr]={}
            if dic.get(corr21) is None:
                dic[corr21]={}
            
            dic[corr][indxs]=result0
                                
            dic[corr][indxs2]=result0
            dic[corr21][indxs2]=result0
            dic[corr21][indxs]=result0

        return dic
    
    def set_window(self,corrs=None,corr_indxs=None,client=None):

        self.Win={'cl':{}}
        if self.store_win and client is None:
            client=get_client()
            
        print('setting windows',client)                
    
        self.Win_cl={corr+indx: delayed(self.get_window_power_cl)(corr,indx) for corr in corrs for indx in corr_indxs[corr]}
#         self.Win_cl={corr+indx: self.get_window_power_cl(corr,indx) for corr in corrs for indx in corr_indxs[corr]}
        
        self.Win_cl_lm={}
        
        for lm in self.lms:
            self.Win_cl_lm[lm]={}
            for k in self.Win_cl.keys():
                corr=(k[0],k[1])
                #                 if self.store_win:
                #                   self.Win_cl_lm[lm][k]=self.get_cl_coupling_lm(self.Win_cl[k],lm,self.wig_3j_2[lm][self.wig_m1m2s[corr]])
                self.Win_cl_lm[lm][k]=delayed(self.get_cl_coupling_lm)(self.Win_cl[k],lm,self.wig_3j_2[lm][self.wig_m1m2s[corr]])
#             if self.store_win:
#                 self.Win_cl_lm[lm]=client.compute(self.Win_cl_lm[lm]).result()
                
                
        self.Win_cl=delayed(self.return_dict_cl)(self.Win_cl_lm,corrs)
        if self.store_win:
            self.Win['cl']=client.compute(self.Win_cl).result()
            
        else:
            self.Win['cl']=self.Win_cl
                
        print('Cl windows done, now to covariance')#,self.Win[('galaxy','galaxy')][(0,0)]['M'][0,0])       

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
                            
                            self.Win_cov.update({corr+indxs: delayed(self.get_window_power_cov)(corr1,corr2,indx1,indx2)})
                                
                            if self.win_cov_tuple is None:
                                self.win_cov_tuple=[(corr1,corr2,indx1,indx2)]
                            else:
                                self.win_cov_tuple.append((corr1,corr2,indx1,indx2))
                  
            self.Win_cov_lm={}
            for lm in self.lms:
                self.Win_cov_lm[lm]={}
                for k in self.Win_cov.keys():
                    corr=(k[0],k[1],k[2],k[3])
                    m1m2s={}
                    m1m2s[1324]=np.sort(np.array([self.cov_m1m2s(corr=(corr[0],corr[2])), #13
                                          self.cov_m1m2s(corr=(corr[1],corr[3])) #24
                                        ]))
                    m1m2s[1324]=str(m1m2s[1324][0])+str(m1m2s[1324][1])
                    m1m2s[1423]=np.sort(np.array([self.cov_m1m2s(corr=(corr[0],corr[3])), #14
                                          self.cov_m1m2s(corr=(corr[1],corr[2])) #23
                                        ]))
                    m1m2s[1423]=str(m1m2s[1423][0])+str(m1m2s[1423][1])

                    self.Win_cov_lm[lm][k]=delayed(self.get_cov_coupling_lm)(self.Win_cov[k],lm,self.wig_3j_2[lm][m1m2s[1324]],self.wig_3j_2[lm][m1m2s[1423]] )
                    
#                 if self.store_win: #might help with memory
#                     self.Win_cov_lm[lm]=client.compute(self.Win_cov_lm[lm]).result()
            self.Win_cov=delayed(self.return_dict_cov)(self.Win_cov_lm,self.win_cov_tuple)
            if self.store_win:
#                 self.Win['cov']=self.Win_cov.compute() #apparently client.compute has better memeory manangement than simple compute https://distributed.dask.org/en/latest/memory.html
                self.Win['cov']=client.compute(self.Win_cov).result()
            else:
                self.Win['cov']=self.Win_cov
            if self.store_win:
                self.wig_3j_2={}
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