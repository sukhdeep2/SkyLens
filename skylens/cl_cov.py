import os,sys
sys.path.append('/verafs/scratch/phy200040p/sukhdeep/project/skylens/skylens/')
import dask
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
import warnings,logging
import copy
import multiprocessing,psutil
import sparse
import gc
import dask.bag
from dask import delayed

from skylens.power_spectra import *
from skylens.angular_power_spectra import *
from skylens.hankel_transform import *
from skylens.wigner_transform import *
from skylens.binning import *
from skylens.cov_utils import *
from skylens.tracer_utils import *
from skylens.window_utils import *
from skylens.cov_tri import *
from skylens.thread_count import *

d2r=np.pi/180.
c=c.to(u.km/u.second)

class Skylens():
    def __init__(self,l=np.arange(2,2001),WT=None,Ang_PS=None,
                cov_utils=None,logger=None,tracer_utils=None,#lensing_utils=None,galaxy_utils=None,
                zs_bins=None,zk_bins=None,zg_bins=None,galaxy_bias_func=None,
                power_spectra_kwargs={},WT_kwargs=None,
                z_PS=None,nz_PS=100,log_z_PS=True,
                do_cov=False,SSV_cov=False,tidal_SSV_cov=False,do_sample_variance=True,
                Tri_cov=False,sparse_cov=False,
                use_window=True,window_lmax=None,window_l=None,store_win=False,Win=None,
                f_sky=None,wigner_step=None,
                l_bins=None,bin_cl=False,use_binned_l=False,do_pseudo_cl=True,
                stack_data=False,bin_xi=False,do_xi=False,theta_bins=None,
                use_binned_theta=False, xi_win_approx=False,
                corrs=None,corr_indxs=None,stack_indxs=None,
                wigner_files=None,name='',
                client=None,scheduler_info=None):

        self.__dict__.update(locals()) #assign all input args to the class as properties
        self.l0=l*1.
        
#         self.set_client()
        self.set_bin_params()
        self.set_binned_measure(locals())

        if logger is None:
            self.logger=logging.getLogger() #not really being used right now
            self.logger.setLevel(level=logging.DEBUG)
            logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                                level=logging.DEBUG, datefmt='%I:%M:%S')

        if tracer_utils is None:
            self.tracer_utils=Tracer_utils(zs_bins=zs_bins,zg_bins=zg_bins,zk_bins=zk_bins,
                                            logger=self.logger,l=self.l)

        self.set_corr_indxs(corr_indxs=corr_indxs,stack_indxs=stack_indxs)
        
        self.window_lmax=30 if window_lmax is None else window_lmax
        self.window_l=np.arange(self.window_lmax+1) if window_l is None else window_l

        self.set_WT_spins()
        self.set_WT_binned()

        self.z_bins=self.tracer_utils.z_bins
        self.set_fsky(f_sky)
        
        if cov_utils is None:
            self.cov_utils=Covariance_utils(f_sky=f_sky,l=self.l,logger=self.logger,
                                            do_xi=self.do_xi,
                                            do_sample_variance=do_sample_variance,
                                            use_window=use_window,use_binned_l=self.use_binned_l,
                                            window_l=self.window_l)

        if Ang_PS is None:
            self.Ang_PS=Angular_power_spectra(
                                SSV_cov=self.SSV_cov,l=self.l,logger=self.logger,
                                power_spectra_kwargs=power_spectra_kwargs,
                                cov_utils=self.cov_utils,window_l=self.window_l,
                                z_PS=z_PS,nz_PS=nz_PS,log_z_PS=log_z_PS,
                                z_PS_max=self.tracer_utils.z_PS_max)
                        #FIXME: Need a dict for these args

        self.Win={}
        
        if self.do_xi and not self.xi_win_approx: #FIXME: Since the `aprrox' is actually the correct way, change the notation.
            self.do_pseudo_cl=True #we will use pseudo_cl transform to get correlation functions.

        
        self.Win=window_utils(window_l=self.window_l,l=self.l0,l_bins=self.l_bins,corrs=self.corrs,s1_s2s=self.s1_s2s,
                        cov_indxs=self.cov_indxs,client=self.client,scheduler_info=self.scheduler_info,
                        use_window=use_window,do_cov=do_cov,cov_utils=self.cov_utils,
                        f_sky=f_sky,corr_indxs=self.stack_indxs,z_bins=self.z_bins,
                        window_lmax=self.window_lmax,Win=Win,WT=self.WT,do_xi=self.do_xi,
                        xi_win_approx=self.xi_win_approx,do_pseudo_cl=self.do_pseudo_cl,
                        kappa_class0=self.kappa0,kappa_class_b=self.kappa_b,wigner_step=wigner_step,
                        xi_bin_utils=self.xi_bin_utils,store_win=store_win,wigner_files=wigner_files,
                        bin_window=self.use_binned_l)
        self.bin_window=self.Win.bin_window
        
        print('Window done. Size:',get_size(self.Win.Win)/1.e6)
        if self.Tri_cov:
            self.CTR=cov_matter_tri(k=self.l)
    
    def set_client(self):
        self.LC=None
        if self.scheduler_info is not None:
            self.client=get_client(address=self.scheduler_info['address'])
        if self.client is None:
            ncpu=multiprocessing.cpu_count()-1
            vmem=psutil.virtual_memory()
            mem=str(vmem.total/(1024**3)*0.9)+'GB'
            self.LC=LocalCluster(n_workers=1,processes=False,memory_limit=mem,threads_per_worker=ncpu,memory_spill_fraction=.99,
               memory_monitor_interval='2000ms')
            self.client=Client(self.LC)

    def clean_client(self):
        if not self.LC is None:
            self.client.shutdown()
            self.LC.close()
    def set_binned_measure(self,local_args):
        """
            If we only want to run computations at effective bin centers, then we 
            need to bin the windows and wigner matrices properly, for which unbinned
            quantities need to be computed once. This function sets up the unbinned
            computations, which are used later for binning window coupling and wigner
            matrices. 
            This is useful when running multiple computations for chains etc. For 
            covariance and one time calcs, may as well just do the full computation.
        """
        if self.use_binned_l or self.use_binned_theta:
            inp_args={}
            for k in local_args.keys():
                if k=='self' or k=='client':
                    continue
                inp_args[k]=copy.deepcopy(local_args[k])
#             print('inp_args:',inp_args.keys())
            self.lb=np.int32((self.l_bins[1:]+self.l_bins[:-1])*.5)
            inp_args['use_binned_l']=False
            inp_args['use_binned_theta']=False
            inp_args['use_window']=False
            inp_args['do_cov']=False 
            inp_args['bin_xi']=False
            inp_args['name']='S0'
            if self.do_cov:
                inp_args['corr_indxs']=None
                inp_args['stack_indxs']=None
#             del inp_args['self']
            inp_args2=copy.deepcopy(inp_args)
            inp_args['client']=self.client
            inp_args2['client']=self.client
            self.kappa0=Skylens(**inp_args)  #to get unbinned c_ell and xi

            inp_args2['l']=self.lb
            inp_args2['name']='S_b'
            inp_args2['l_bins']=None
            inp_args2['bin_cl']=False
            inp_args2['do_xi']=False
            self.kappa_b=Skylens(**inp_args2) #to get binned c_ell

            if self.do_xi and self.use_binned_theta:
                theta_bins=inp_args['theta_bins']
                self.thb=(theta_bins[1:]+theta_bins[:-1])*.5 #FIXME:this may not be effective theta of meaurements
                inp_args_xi=copy.deepcopy(inp_args)
                inp_args_xi['name']='S_b_xi'
                
                inp_args_xi['WT'].reset_theta_l(theta=self.thb)
                self.kappa_b_xi=Skylens(**inp_args_xi) #to get binned xi. 
                
                self.xi0=self.kappa0.xi_tomo()['xi']
                self.xi_b=self.kappa_b_xi.xi_tomo()['xi']
            self.l=self.lb*1.
            self.c_ell0=self.kappa0.cl_tomo()['cl']
            self.c_ell_b=self.kappa_b.cl_tomo()['cl']
        else:
            self.kappa_b=self
            self.kappa0=self

    def set_corr_indxs(self,corr_indxs=None,stack_indxs=None):
        """
        set up the indexes for correlations. indexes= tracer and bin ids. 
        User can input the corr_indxs which will be the ones computed (called stack_indxs later). 
        However, when doing covariances, we may need to compute the 
        aiddtional correlations, hence those are included added to the corr_indxs.
        corr_indxs are used for constructing full compute graph but only the stack_indxs 
        is actually computed when stack_dat is called. 
        """
        self.stack_indxs=stack_indxs
        self.corr_indxs=corr_indxs
        self.cov_indxs={}

        if self.corrs is None:
            if bool(self.stack_indxs):
                self.corrs=list(corr_indxs.keys())
            else:
                nt=len(self.tracer_utils.tracers)
                self.corrs=[(self.tracer_utils.tracers[i],self.tracer_utils.tracers[j])
                            for i in np.arange(nt)
                            for j in np.arange(i,nt)
                            ]
                
        if not self.do_cov and self.corr_indxs is None:
            self.corr_indxs=self.stack_indxs    
        if not self.do_cov and  (not self.corr_indxs is None):
            print('not setting corr_indxs',self.do_cov , bool(self.corr_indxs))
            return
        else:
            self.corr_indxs={}
        for tracer in self.tracer_utils.tracers:
            self.corr_indxs[(tracer,tracer)]=[j for j in itertools.combinations_with_replacement(
                                                    np.arange(self.tracer_utils.n_bins[tracer]),2)]

#             if tracer=='galaxy' and not self.do_cov:
#                 self.corr_indxs[(tracer,tracer)]=[(i,i) for i in np.arange(self.tracer_utils.n_bins[tracer])] 
                #by default, assume no cross correlations between galaxy bins

        for tracer1 in self.tracer_utils.tracers:#zbin-indexs for cross correlations
            for tracer2 in self.tracer_utils.tracers:
                if tracer1==tracer2:
                    continue
                if self.corr_indxs.get((tracer1,tracer2)) is not None:
                    continue
                self.corr_indxs[(tracer1,tracer2)]=[ k for l in [[(i,j) for i in np.arange(
                                        self.tracer_utils.n_bins[tracer1])] 
                                        for j in np.arange(self.tracer_utils.n_bins[tracer2])] for k in l]
        
        if self.stack_indxs is None:# or not bool(self.stack_indxs):
            self.stack_indxs=self.corr_indxs

        if self.do_cov:
            stack_corr_indxs=self.stack_indxs
            corrs=self.corrs
            corrs_iter=[(corrs[i],corrs[j]) for i in np.arange(len(corrs)) for j in np.arange(i,len(corrs))]
            for (corr1,corr2) in corrs_iter:
                corr1_indxs=stack_corr_indxs[(corr1[0],corr1[1])]
                corr2_indxs=stack_corr_indxs[(corr2[0],corr2[1])]
                if corr1==corr2:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for j in np.arange(i,
                                     len(corr1_indxs))] for i in np.arange(len(corr2_indxs))] for k in l]
                else:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for i in np.arange(
                                    len(corr1_indxs))] for j in np.arange(len(corr2_indxs))] for k in l]
                self.cov_indxs[corr1+corr2]=cov_indxs_iter

    def set_fsky(self,f_sky):
        """
        We assume different tracers can have partial
        overlap, in which case f_sky will be a dictionary with varying values for different two point correlations.
        This function sets that dictionary if only single f_sky is passed.
        """
        self.f_sky=f_sky
        if np.isscalar(self.f_sky): #this is dict because we allow for partial overlap between different tracers.
            f_temp=np.copy(self.f_sky)
            self.f_sky={}
            for kk in self.corr_indxs.keys():
#                 n_indx=len(self.corr_indxs[kk])
                indxs=self.corr_indxs[kk]
                self.f_sky[kk]={}
                self.f_sky[kk[::-1]]={}
                for idx in indxs:
                    self.f_sky[kk][idx]=f_temp #*np.ones((n_indx,n_indx))
                    self.f_sky[kk[::-1]][idx[::-1]]=f_temp

    def set_WT_spins(self):
        self.s1_s2s={}
        for tracer1 in self.tracer_utils.tracers:#zbin-indexs for cross correlations
            for tracer2 in self.tracer_utils.tracers:
                self.s1_s2s[(tracer1,tracer2)]=[(self.tracer_utils.spin[tracer1],self.tracer_utils.spin[tracer2])]
        if 'shear' in self.tracer_utils.tracers:
            self.s1_s2s[('shear','shear')]=[(2,2),(2,-2)]
        self.s1_s2s[('window')]=[(0,0)]

    def set_WT_binned(self):
        """
        If we only want to compute at bin centers, wigner transform matrices need to be binned.
        """
        if not self.do_xi:
            return 
        WT=self.WT
        self.WT_binned={corr:{} for corr in self.corrs} #intialized later.
        if self.do_xi and (self.use_binned_l or self.use_binned_theta):
            for corr in self.corrs:
                s1_s2s=self.s1_s2s[corr]
                self.WT_binned[corr]={s1_s2s[im]:{} for im in np.arange(len(s1_s2s))}
                for indxs in self.corr_indxs[corr]:    
                    cl0=self.c_ell0[corr][indxs].compute()
                    cl_b=self.c_ell_b[corr][indxs].compute()
                    for im in np.arange(len(s1_s2s)):
                        s1_s2=s1_s2s[im]
                        self.WT_binned[corr][s1_s2][indxs]=self.binning.bin_2d_WT(
                                                            wig_mat=self.WT.wig_d[s1_s2]*self.WT.grad_l*self.WT.norm,
                                                                wt0=cl0,wt_b=1./cl_b,bin_utils_cl=self.cl_bin_utils,
                                                                bin_utils_xi=self.xi_bin_utils[s1_s2])
                        

    def update_zbins(self,z_bins={},tracer='shear'):
        """
        If the tracer bins need to be updated. Ex. when running chains with varying photo-z params.
        """
        self.tracer_utils.set_zbins(z_bins,tracer=tracer)
        self.z_bins=self.tracer_utils.z_bins
        return


    def set_bin_params(self):
        """
            Setting up the binning functions to be used in binning the data
        """
        self.binning=binning()
        if self.bin_cl:
            self.cl_bin_utils=self.binning.bin_utils(r=self.l0,r_bins=self.l_bins,
                                                r_dim=2,mat_dims=[1,2])
        self.xi_bin_utils={}
        if self.do_xi and self.bin_xi:
            for s1_s2 in self.WT.s1_s2s:
                self.xi_bin_utils[s1_s2]=self.binning.bin_utils(r=self.WT.theta[s1_s2]/d2r,
                                                    r_bins=self.theta_bins,
                                                    r_dim=2,mat_dims=[1,2])
            
    def calc_cl(self,zs1_indx=-1, zs2_indx=-1,corr=('shear','shear')):
        """
            Compute the angular power spectra, Cl between two source bins
            zs1, zs2: Source bins. Dicts containing information about the source bins
        """

        zs1=self.z_bins[corr[0]][zs1_indx]#.copy() #we will modify these locally
        zs2=self.z_bins[corr[1]][zs2_indx]#.copy()

        clz=self.Ang_PS.clz
        cls=clz['cls']
        f=self.Ang_PS.cl_f
        sc=zs1['kernel_int']*zs2['kernel_int']

        dchi=np.copy(clz['dchi'])
        cl=np.dot(cls.T*sc,dchi)
                # cl*=2./np.pi #FIXME: needed to match camb... but not CCL
        return cl

    def cov_four_kernels(self,zs_indx=[],tracers=[]):
        zs1=self.z_bins[tracers[0]][zs_indx[0]]
        zs2=self.z_bins[tracers[1]][zs_indx[1]]
        zs3=self.z_bins[tracers[2]][zs_indx[2]]
        zs4=self.z_bins[tracers[3]][zs_indx[3]]
#                 sig_cL=zs1['kernel_int']*zs2['kernel_int']*zs3['kernel_int']*zs4['kernel_int']
        sig_cL=zs1['Gkernel_int']*zs2['Gkernel_int']*zs3['Gkernel_int']*zs4['Gkernel_int']#Only use lensing kernel... not implemented for galaxies (galaxies have magnification, which is included)
        sig_cL*=self.Ang_PS.clz['dchi']
        return sig_cL
    
    def cl_cov(self,zs_indx,cls=None, tracers=[],Win_cov=None,Win_cl=None):
        """
            Computes the covariance between any two tomographic power spectra.
            cls: tomographic cls already computed before calling this function
            zs_indx: 4-d array, noting the indices of the source bins involved
            in the tomographic cls for which covariance is computed.
            For ex. covariance between 12, 56 tomographic cross correlations
            involve 1,2,5,6 source bins
        """
        cov={}
        cov['z_indx']=zs_indx
        cov['tracers']=tracers
        cov['final']=None

        cov['G']=None
        cov['G1324_B']=None;cov['G1423_B']=None
        
        Win=None
        if self.use_window and self.store_win:
            Win_cov=self.Win.Win['cov'][tracers] #passed as none to save memory
            Win_cl=self.Win.Win['cl']
        if Win_cov is not None:
            Win=Win_cov[zs_indx]

        if self.use_window and self.do_pseudo_cl:
            cov['G1324'],cov['G1423']=self.cov_utils.gaussian_cov_window(cls,
                                            self.SN,tracers,zs_indx,self.do_xi,Win,
                                            binned_l=self.use_binned_l)#bin_window=self.bin_window,bin_utils=self.cl_bin_utils)
        else:
            fs=self.f_sky
            if self.do_xi and self.xi_win_approx and self.use_window : #in this case we need to use a separate function directly from xi_cov
                cov['G1324']=0
                cov['G1423']=0
            else:
                cov['G1324'],cov['G1423']=self.cov_utils.gaussian_cov(cls,
                                            self.SN,tracers,zs_indx,self.do_xi,fs)
        cov['G']=cov['G1324']+cov['G1423']
        cov['final']=cov['G']

        if not self.do_xi:
            cov['G1324']=None #save memory
            cov['G1423']=None

        cov['SSC']=0
        cov['Tri']=0

        if self.Tri_cov or self.SSV_cov:
            sig_cL=self.cov_four_kernels(zs_indx=zs_indx,tracers=tracers)

        if self.SSV_cov :
            clz=self.Ang_PS.clz
            sigma_win=self.cov_utils.sigma_win_calc(clz=clz,Win=Win,tracers=tracers,zs_indx=zs_indx)

            clr=self.Ang_PS.clz['clsR']
            if self.tidal_SSV_cov:
                clr=self.Ang_PS.clz['clsR']+ self.Ang_PS.clz['clsRK']/6.

            sig_F=np.sqrt(sig_cL*sigma_win) #kernel is function of l as well due to spin factors
            clr=clr*sig_F.T
            cov['SSC']=np.dot(clr.T,clr)

        if self.Tri_cov:
            cov['Tri']=self.CTR.cov_tri_zkernel(P=self.Ang_PS.clz['cls'],z_kernel=sig_cL/self.Ang_PS.clz['chi']**2,chi=self.Ang_PS.clz['chi']) #FIXME: check dimensions, get correct factors of length.. chi**2 is guessed from eq. A3 of https://arxiv.org/pdf/1601.05779.pdf ... note that cls here is in units of P(k)/chi**2
            fs0=self.f_sky[tracers[0],tracers[1]][zs_indx[0],zs_indx[1]]
            fs0*=self.f_sky[tracers[2],tracers[3]][zs_indx[2],zs_indx[3]]
            fs0=np.sqrt(fs0)
#             cov['Tri']/=self.cov_utils.gaussian_cov_norm_2D**2 #Since there is no dirac delta, there should be 2 factor of (2l+1)dl... eq. A3 of https://arxiv.org/pdf/1601.05779.pdf
            cov['Tri']/=fs0 #(2l+1)f_sky.. we didnot normalize gaussian covariance in trispectrum computation.

        if self.use_window and (self.SSV_cov or self.Tri_cov) and self.do_pseudo_cl: #Check: This is from writing p-cl as M@cl... cov(p-cl)=M@cov(cl)@M.T ... separate  M when different p-cl
            M1=Win_cl[(tracers[0],tracers[1])][(zs_indx[0],zs_indx[1])]['M'] #12
            M2=Win_cl[(tracers[2],tracers[3])][(zs_indx[2],zs_indx[3])]['M'] #34
            if self.bin_window:
                for k in ['SSC','Tri']:
                    cov[k]=self.bin_cl_func(cov=cov[k])
            cov['final']=cov['G']+ M1@(cov['SSC']+cov['Tri'])@M2.T
        else:
            cov['final']=cov['G']+cov['SSC']+cov['Tri']

        
        for k in ['final','G','SSC','Tri']:#no need to bin G1324 and G1423
            if self.bin_cl:
                cov[k+'_b']=self.bin_cl_func(cov=cov[k])
            else:
                cov[k+'_b']=cov[k]
            
            if self.sparse_cov and cov[k+'_b'] is not None:
                if k!='final':
                    # print('deleting',k)
                    cov[k+'_b']=None
                    continue
                cov[k+'_b']=sparse.COO(cov[k+'_b'])
            if not self.do_xi and self.bin_cl:
                del cov[k]
        # gc.collect() #this can slow down the code.
        return cov

    def bin_cl_func(self,cl=None,cov=None):
        """
            bins the tomographic power spectra
            results: Either cl or covariance
            bin_cl: if true, then results has cl to be binned
            bin_cov: if true, then results has cov to be binned
            Both bin_cl and bin_cov can be true simulatenously.
        """
        cl_b=None
        cov_b=None
#         if self.bin_cl:
        if not cl is None:
            if self.use_binned_l or not self.bin_cl:
                cl_b=cl*1.
            else:
                cl_b=self.binning.bin_1d(xi=cl,bin_utils=self.cl_bin_utils)
            return cl_b
        if not cov is None:
            if self.use_binned_l or not self.bin_cl:
                cov_b=cov*1.
            else:
                cov_b=self.binning.bin_2d(cov=cov,bin_utils=self.cl_bin_utils)
            return cov_b

    def calc_pseudo_cl(self,cl,Win,zs1_indx=-1, zs2_indx=-1,corr=('shear','shear')):
        pcl=cl@Win['M']
#         if np.any(~np.isfinite(pcl)):
#             print('pseudo cl not finite:', corr,zs1_indx,zs2_indx, cl,Win['M'])
        return  pcl

    def cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,
                corrs=None,bias_kwargs={},bias_func=None,stack_corr_indxs=None):
        """
         Computes full tomographic power spectra and covariance, including shape noise. output is
         binned also if needed.
         Arguments are for the power spectra  and sigma_crit computation,
         if it needs to be called from here.
         source bins are already set. This function does set the sigma crit for sources.
        """

        l=self.l
        if corrs is None:
            corrs=self.corrs
        if stack_corr_indxs is None:
            stack_corr_indxs=self.stack_indxs

        tracers=np.unique([j for i in corrs for j in i])
        
        corrs2=corrs.copy()
        if self.do_cov:
            for i in np.arange(len(tracers)):
                for j in np.arange(i,len(tracers)):
                    if (tracers[i],tracers[j]) not in corrs2 and (tracers[j],tracers[i]) in corrs2:
                        corrs2+=[(tracers[i],tracers[j])]
                        print('added extra corr calc for covariance',corrs2)

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h

        self.SN={}
        if 'shear' in tracers:
            self.tracer_utils.set_kernel(cosmo_h=cosmo_h,zl=self.Ang_PS.z,tracer='shear')
            self.SN[('shear','shear')]=self.tracer_utils.SN['shear']
        if 'kappa' in tracers:
            self.tracer_utils.set_kernel(cosmo_h=cosmo_h,zl=self.Ang_PS.z,tracer='kappa')
            self.SN[('kappa','kappa')]=self.tracer_utils.SN['kappa']
        if 'galaxy' in tracers:
            if bias_func is None:
                bias_func='constant_bias'
                bias_kwargs={'b1':1,'b2':1}
            self.tracer_utils.set_kernel(cosmo_h=cosmo_h,zl=self.Ang_PS.z,tracer='galaxy')
            self.SN[('galaxy','galaxy')]=self.tracer_utils.SN['galaxy']

        self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,
                                cosmo_params=cosmo_params)

        out={}
        cl={corr:{} corr in corrs2}.update({corr[::-1]:{} corr in corrs2})
        pcl={corr:{} corr in corrs2}.update({corr[::-1]:{} corr in corrs2}) #pseudo_cl
        cl_b={corr:{} corr in corrs2}.update({corr[::-1]:{} corr in corrs2})
        pcl_b={corr:{} corr in corrs2}.update({corr[::-1]:{} corr in corrs2})
        
        cov={}
        for corr in corrs2:
            corr2=corr[::-1]
            corr_indxs=self.corr_indxs[(corr[0],corr[1])]#+self.cov_indxs
            for (i,j) in corr_indxs:#FIXME: we might want to move to map, like covariance. will be useful to define the tuples in forzenset then.
                # out[(i,j)]
                cl[corr][(i,j)]=delayed(self.calc_cl)(zs1_indx=i,zs2_indx=j,corr=corr) 
                cl_b[corr][(i,j)]=delayed(self.bin_cl_func)(cl=cl[corr][(i,j)],cov=None)
                if self.use_window and self.do_pseudo_cl and (i,j) in self.stack_indxs[corr]:
                    if not self.bin_window:
                        pcl[corr][(i,j)]=delayed(self.calc_pseudo_cl)(cl[corr][(i,j)],Win=self.Win.Win['cl'][corr][(i,j)],zs1_indx=i,
                                                zs2_indx=j,corr=corr)
                        pcl_b[corr][(i,j)]=delayed(self.bin_cl_func)(cl=pcl[corr][(i,j)],cov=None)
                    else:
                        pcl[corr][(i,j)]=None
                        pcl_b[corr][(i,j)]=delayed(self.calc_pseudo_cl)(cl_b[corr][(i,j)],Win=self.Win.Win['cl'][corr][(i,j)],zs1_indx=i,
                                                zs2_indx=j,corr=corr)
                else:
                    pcl[corr][(i,j)]=cl[corr][(i,j)]
                    pcl_b[corr][(i,j)]=cl_b[corr][(i,j)]
                cl[corr2][(j,i)]=cl[corr][(i,j)]#useful in gaussian covariance calculation.
                pcl[corr2][(j,i)]=pcl[corr][(i,j)]#useful in gaussian covariance calculation.
                cl_b[corr2][(j,i)]=cl_b[corr][(i,j)]#useful in gaussian covariance calculation.
                pcl_b[corr2][(j,i)]=pcl_b[corr][(i,j)]#useful in gaussian covariance calculation.
    
        print('cl dict done')
        if self.do_cov:
            # t1=time.time()
            cii_t=0
            start_j=0
            Win_cov=None
            Win_cl=None
            corrs_iter=[(corrs[i],corrs[j]) for i in np.arange(len(corrs)) for j in np.arange(i,len(corrs))]
            cov_indxs={}
            for (corr1,corr2) in corrs_iter:
                cov[corr1+corr2]={}
                cov[corr2+corr1]={}

                corr1_indxs=stack_corr_indxs[(corr1[0],corr1[1])]
                corr2_indxs=stack_corr_indxs[(corr2[0],corr2[1])]
                if corr1==corr2:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for j in np.arange(i,
                                     len(corr1_indxs))] for i in np.arange(len(corr2_indxs))] for k in l]
                else:
                    cov_indxs_iter=[ k for l in [[corr1_indxs[i]+corr2_indxs[j] for i in np.arange(
                                    len(corr1_indxs))] for j in np.arange(len(corr2_indxs))] for k in l]
                cov_indxs[corr1+corr2]=cov_indxs_iter #because in principle we allow stack_indxs to be different than self.stack_indxs
                
                if self.use_window and not self.store_win:
                    Win_cov=self.Win.Win['cov'][corr1+corr2] # we only want to pass this if it is a graph. Otherwise, read within function
                    Win_cl=self.Win.Win['cl']
                cov[corr1+corr2]=dask.bag.from_sequence(cov_indxs_iter).map(self.cl_cov,cls=cl,Win_cov=Win_cov,tracers=corr1+corr2,Win_cl=Win_cl)
            cov['cov_indxs']=cov_indxs

            print('cov dict done')

        out_stack=delayed(self.stack_dat)({'cov':cov,'pcl_b':pcl_b,'est':'pcl_b'},corrs=corrs,
                                          corr_indxs=stack_corr_indxs)
        # gc.collect()
        return {'stack':out_stack,'cl_b':cl_b,'cov':cov,'cl':cl,'pseudo_cl':pcl,'pseudo_cl_b':pcl_b}

    def xi_cov(self,cov_cl,cls={},s1_s2=None,s1_s2_cross=None,#clr=None,clrk=None,
#                indxs_1=[],indxs_2=[],
               corr1=[],corr2=[], Win_cov=None,Win_cl=None):
        """
            Computes covariance of xi, by performing 2-D hankel transform on covariance of Cl.
            In current implementation of hankel transform works only for s1_s2=s1_s2_cross.
            So no cross covariance between xi+ and xi-.
        """

#         z_indx=indxs_1+indxs_2
        z_indx=cov_cl['z_indx']
        indxs_1=(z_indx[0],z_indx[1])
        indxs_2=(z_indx[2],z_indx[3])
#         cov_cl=cov_cl0[z_indx]
        tracers=corr1+corr2
        if s1_s2_cross is None:
            s1_s2_cross=s1_s2
        cov_xi={}

        if self.WT.name=='Hankel' and s1_s2!=s1_s2_cross:
            n=len(self.theta_bins)-1
            cov_xi['final']=np.zeros((n,n))
            return cov_xi

        SN1324=0
        SN1423=0

        Win=None
        if self.use_window and self.store_win:
            Win_cov=self.Win.Win['cov'][tracers]
        if Win_cov is not None:
            Win=Win_cov[z_indx]
        
        if np.all(np.array(tracers)=='shear') and  s1_s2!=s1_s2_cross and not self.xi_win_approx: #cross between xi+ and xi-
            if self.use_window:
                G1324,G1423=self.cov_utils.gaussian_cov_window(cls,self.SN,tracers,z_indx,self.do_xi,Win,Bmode_mf=-1)
#             elif self.use_window and self.xi_win_approx:
#                 bf=-1
#                 G1324,G1423=self.cov_utils.xi_gaussian_cov_window_approx(cls,self.SN,tracers,z_indx,self.do_xi,Win['cov'][tracers][z_indx],self.WT,WT_kwargs,bf)
            else:
                if not self.xi_win_approx:
                    G1324,G1423=self.cov_utils.gaussian_cov(cls,self.SN,tracers,z_indx,self.do_xi,self.f_sky,Bmode_mf=-1)
                else:
                    G1324=0
                    G1423=0
            cov_cl_G=G1324+G1423
        else:
            cov_cl_G=cov_cl['G1324']+cov_cl['G1423'] #FIXME: needs Bmode for shear


        if self.use_window and self.xi_win_approx: 
            WT_kwargs={'l_cl':self.l,'s1_s2':s1_s2,'s1_s2_cross':s1_s2_cross}
            bf=1
            if np.all(np.array(tracers)=='shear') and not s1_s2==s1_s2_cross: #cross between xi+ and xi-
                bf=-1
#             try:
            cov_xi['G']=self.cov_utils.xi_gaussian_cov_window_approx(cls,self.SN,tracers,z_indx,self.do_xi,Win,self.WT,WT_kwargs,bf)
#             except Exception as err:
#                 print('error', err, tracers, z_indx,Win['cov'][tracers].keys())
#                 crash
        else:
            th0,cov_xi['G']=self.WT.projected_covariance2(l_cl=self.l,s1_s2=s1_s2,
                                                      s1_s2_cross=s1_s2_cross,
                                                      cl_cov=cov_cl_G)


        cov_xi['G']=self.binning.bin_2d(cov=cov_xi['G'],bin_utils=self.xi_bin_utils[s1_s2])
        #binning is cheap

#         cov_xi['final']=cov_xi['G']
        cov_xi['SSC']=0
        cov_xi['Tri']=0

        if self.SSV_cov:
            th0,cov_xi['SSC']=self.WT.projected_covariance2(l_cl=self.l,s1_s2=s1_s2,
                                                            s1_s2_cross=s1_s2_cross,
                                                            cl_cov=cov_cl['SSC'])
            cov_xi['SSC']=self.binning.bin_2d(cov=cov_xi['SSC'],bin_utils=self.xi_bin_utils[s1_s2])
        if self.Tri_cov:
            th0,cov_xi['Tri']=self.WT.projected_covariance2(l_cl=self.l,s1_s2=s1_s2,
                                                            s1_s2_cross=s1_s2_cross,
                                                            cl_cov=cov_cl['Tri'])
            cov_xi['Tri']=self.binning.bin_2d(cov=cov_xi['Tri'],bin_utils=self.xi_bin_utils[s1_s2])

        cov_xi['final']=cov_xi['G']+cov_xi['SSC']+cov_xi['Tri']
        #         if self.use_window: #pseudo_cl:
        if self.use_window and self.xi_win_approx:
            cov_xi['G']/=(Win_cl[corr1][indxs_1]['xi_b']*Win_cl[corr2][indxs_2]['xi_b'])
            cov_xi['final']/=(Win_cl[corr1][indxs_1]['xi_b']*Win_cl[corr2][indxs_2]['xi_b'])

        return cov_xi

    def get_xi(self,cls={},s1_s2=[],corr=None,indxs=None,Win=None):
        cl=cls[corr][indxs] #this should be pseudo-cl when using window
        wig_m=None
        if self.use_binned_l or self.use_binned_theta:
            wig_m=self.WT_binned[corr][s1_s2][indxs]
        th,xi=self.WT.projected_correlation(l_cl=self.l,s1_s2=s1_s2,cl=cl,wig_d=wig_m)
        if self.use_window and self.xi_win_approx: 
            xi=xi*Win['cl'][corr][indxs]['xi']

        xi_b=xi
        if self.bin_xi and not self.use_binned_theta:
            xi_b=self.binning.bin_1d(xi=xi,bin_utils=self.xi_bin_utils[s1_s2])
        
        if self.use_window or self.xi_win_approx:
            xi_b/=(Win['cl'][corr][indxs]['xi_b'])
        return xi_b

    def xi_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None,
                corrs=None):
        """
            Computed tomographic angular correlation functions. First calls the tomographic
            power spectra and covariance and then does the hankel transform and  binning.
        """
        """
            For hankel transform is done on l-theta grid, which is based on s1_s2. So grid is
            different for xi+ and xi-.
            In the init function, we combined the ell arrays for all s1_s2. This is not a problem
            except for the case of SSV, where we will use l_cut to only select the relevant values
        """

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h
        if corrs is None:
            corrs=self.corrs

        #Donot use delayed here. Leads to error/repeated calculations
        cls_tomo_nu=self.cl_tomo(cosmo_h=cosmo_h,cosmo_params=cosmo_params,
                            pk_params=pk_params,corrs=corrs)

        cl=cls_tomo_nu['cl'] #Note that if window is turned off, pseudo_cl=cl
        cov_xi={}
        xi={}
        out={}
        # for s1_s2 in self.s1_s2s:
        for corr in corrs:
            s1_s2s=self.s1_s2s[corr]
            xi[corr]={}
            for im in np.arange(len(s1_s2s)):
                s1_s2=s1_s2s[im]
                xi[corr][s1_s2]={}
                for indx in self.corr_indxs[corr]:
                    xi[corr][s1_s2][indx]=delayed(self.get_xi)(cls=cl,corr=corr,indxs=indx,
                                                        s1_s2=s1_s2,Win=self.Win.Win)
        if self.do_cov:
            corrs_iter=[(corrs[i],corrs[j]) for i in np.arange(len(corrs)) for j in np.arange(i,len(corrs))]
            cov_indxs={}
            for (corr1,corr2) in corrs_iter:
                s1_s2s_1=self.s1_s2s[corr1]
                s1_s2s_2=self.s1_s2s[corr2]

                corr=corr1+corr2
                cov_xi[corr]={}
                cov_xi['cov_indxs']=cls_tomo_nu['cov']['cov_indxs']

                cov_cl=cls_tomo_nu['cov'][corr]#.compute()
                cov_iter=cls_tomo_nu['cov']['cov_indxs'][corr]

                Win_cov=None
                Win_cl=None
                if self.use_window or self.xi_win_approx:
                    if not self.store_win:
                        Win_cov=self.Win.Win['cov'][corr]
                    Win_cl=self.Win.Win['cl']
                for im1 in np.arange(len(s1_s2s_1)):
                    s1_s2=s1_s2s_1[im1]
                    start2=0
                    if corr1==corr2:
                        start2=im1
                    for im2 in np.arange(start2,len(s1_s2s_2)):
                        s1_s2_cross=s1_s2s_2[im2]
                        cov_xi[corr][s1_s2+s1_s2_cross]=dask.bag.from_sequence(cov_cl).map(self.xi_cov,
                                                                                        cls=cl,s1_s2=s1_s2,
                                                                                        s1_s2_cross=s1_s2_cross,#clr=clr,
                                                                                        Win_cov=Win_cov,
                                                                                        Win_cl=Win_cl,
                                                                                        corr1=corr1,corr2=corr2
                                                                                        )

        out['stack']=delayed(self.stack_dat)({'cov':cov_xi,'xi':xi,'est':'xi'},corrs=corrs)
        out['xi']=xi
        out['cov']=cov_xi
        out['cl']=cls_tomo_nu
        return out


    def stack_dat(self,dat,corrs,corr_indxs=None):
        """
            outputs from tomographic caluclations are dictionaries.
            This fucntion stacks them such that the cl or xi is a long
            1-d array and the covariance is N X N array.
            dat: output from tomographic calculations.
            XXX: reason that outputs tomographic bins are distionaries is that
            it make is easier to
            handle things such as binning, hankel transforms etc. We will keep this structure for now.
        """

        if corr_indxs is None:
            corr_indxs=self.stack_indxs

        est=dat['est']
        if est=='xi':
            len_bins=len(self.theta_bins)-1
        else:
            #est='cl_b'
            if self.l_bins is not None:
                len_bins=len(self.l_bins)-1
            else:
                len_bins=len(self.l)

        n_bins=0
        for corr in corrs:
            n_s1_s2=1
            if est=='xi':
                n_s1_s2=len(self.s1_s2s[corr])
            n_bins+=len(corr_indxs[corr])*n_s1_s2 #np.int64(nbins*(nbins-1.)/2.+nbins)
        D_final=np.zeros(n_bins*len_bins)
        i=0
        for corr in corrs:
            n_s1_s2=1
            if est=='xi':
                s1_s2=self.s1_s2s[corr]
                n_s1_s2=len(s1_s2)

            for im in np.arange(n_s1_s2):
                if est=='xi':
                    dat_c=dat[est][corr][s1_s2[im]]
                else:
                    dat_c=dat[est][corr]#[corr] #cl_b gets keys twice. dask won't allow standard dict merge.. should be fixed
                    
                for indx in corr_indxs[corr]:
                    D_final[i*len_bins:(i+1)*len_bins]=dat_c[indx]
                    i+=1
        if not self.do_cov:
            out={'cov':None}
            out[est]=D_final
            return out

        
        cov_final=np.zeros((len(D_final),len(D_final)))#-999.#np.int(nD2*(nD2+1)/2)
        if self.sparse_cov:
            cov_final=sparse.DOK(cov_final)

        indx0_c1=0
        for ic1 in np.arange(len(corrs)):
            corr1=corrs[ic1]
            indxs_1=corr_indxs[corr1]
            n_indx1=len(indxs_1)

            indx0_c2=indx0_c1
            for ic2 in np.arange(ic1,len(corrs)):
                corr2=corrs[ic2]
                indxs_2=corr_indxs[corr2]
                n_indx2=len(indxs_2)

                corr=corr1+corr2
                n_s1_s2_1=1
                n_s1_s2_2=1
                if est=='xi':
                    s1_s2_1=self.s1_s2s[corr1]
                    s1_s2_2=self.s1_s2s[corr2]
                    n_s1_s2_1=len(s1_s2_1)
                    n_s1_s2_2=len(s1_s2_2)

                for im1 in np.arange(n_s1_s2_1):
                    start_m2=0
                    if corr1==corr2:
                        start_m2=im1
                    for im2 in np.arange(start_m2,n_s1_s2_2):
                        indx0_m1=(im1)*n_indx1*len_bins
                        indx0_m2=(im2)*n_indx2*len_bins
                        for i1 in np.arange(n_indx1):
                            start2=0
                            if corr1==corr2:
                                start2=i1
                            for i2 in np.arange(start2,n_indx2):
                                indx0_1=(i1)*len_bins
                                indx0_2=(i2)*len_bins
                                indx=indxs_1[i1]+indxs_2[i2]
#                                 i_here=np.where(self.cov_indxs[corr]==indx)[0]
                                i_here=dat['cov']['cov_indxs'][corr].index(indx)
#                                 i_here=indx
                                if est=='xi':
                                    cov_here=dat['cov'][corr][s1_s2_1[im1]+s1_s2_2[im2]][i_here]['final']
                                else:
                                    cov_here=dat['cov'][corr][i_here]['final_b']

                                if self.sparse_cov:
                                    cov_here=cov_here.todense()
                                # if im1==im2:
                                i=indx0_c1+indx0_1+indx0_m1
                                j=indx0_c2+indx0_2+indx0_m2

                                cov_final[i:i+len_bins,j:j+len_bins]=cov_here
                                cov_final[j:j+len_bins,i:i+len_bins]=cov_here.T

                                if im1!=im2 and corr1==corr2:
                                    i=indx0_c1+indx0_1+indx0_m2
                                    j=indx0_c2+indx0_2+indx0_m1
                                    cov_final[i:i+len_bins,j:j+len_bins]=cov_here.T
                                    cov_final[j:j+len_bins,i:i+len_bins]=cov_here
                                    gc.collect()

                indx0_c2+=n_indx2*len_bins*n_s1_s2_2
            indx0_c1+=n_indx1*len_bins*n_s1_s2_1

        out={'cov':cov_final}
        out[est]=D_final
        return out


if __name__ == "__main__":
    import cProfile
    import pstats

    import dask,dask.multiprocessing
    dask.config.set(scheduler='processes')
    # dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler..
                                            # Works as usual single threaded worload. Useful for profiling.
    # see minimal notebook for example.
