import os,sys
import dask
from dask import delayed
from dask.distributed import Client
from power_spectra import *
from angular_power_spectra import *
from lsst_utils import *
from hankel_transform import *
from binning import *
from cov_utils import *
from galaxy_utils import *
from lensing_utils import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d

d2r=np.pi/180.
c=c.to(u.km/u.second)

class g_Kappa():
    def __init__(self,silence_camb=False,l=np.arange(2,2001),HT=None,Ang_PS=None,
                lensing_utils=None,cov_utils=None,galaxy_utils=None,
                power_spectra_kwargs={},HT_kwargs=None,zs_bins=None,zl_bins=None,
                zl=None,n_zl=100,log_zl=True,zl_max=None,
                do_cov=False,SSV_cov=False,tidal_SSV_cov=False,
                sigma_gamma=0.3,f_sky=0.3,l_bins=None,bin_cl=False,
                stack_data=False,bin_xi=False,do_xi=False,theta_bins=None,tracer='kappa'):
        self.lensing_utils=lensing_utils
        self.do_cov=do_cov
        self.SSV_cov=SSV_cov
        self.tidal_SSV_cov=tidal_SSV_cov

        if lensing_utils is None:
            self.lensing_utils=Lensing_utils(sigma_gamma=sigma_gamma,zs_bins=zs_bins)
        
        self.set_lens_bins(zl=zl,n_zl=n_zl,log_zl=log_zl,zl_max=zl_max)
        self.l=l
        
        self.cov_utils=cov_utils
        if cov_utils is None:
            self.cov_utils=Covariance_utils(f_sky=f_sky,l=self.l)

        self.Ang_PS=Ang_PS
        if Ang_PS is None:
            self.Ang_PS=Angular_power_spectra(silence_camb=silence_camb,SSV_cov=SSV_cov,l=self.l,
                        power_spectra_kwargs=power_spectra_kwargs,cov_utils=self.cov_utils,
                        zl=self.zl,n_zl=n_zl)

        self.zs_bins=self.lensing_utils.zs_bins
        self.l_bins=l_bins
        self.stack_data=stack_data
        self.theta_bins=theta_bins
        self.do_xi=do_xi
        self.bin_utils=None
        self.tracer=tracer

        self.HT=HT
        if HT is None and do_xi:
            if HT_kwargs is None:
                th_min=1./60. if theta_bins is None else np.amin(theta_bins)
                th_max=5 if theta_bins is None else np.amax(theta_bins)
                HT_kwargs={'kmin':min(l),'kmax':max(l),
                            'rmin':th_min*d2r,'rmax':th_max*d2r,
                            'n_zeros':2000,'prune_r':2,'j_nu':[0]}
            self.HT=hankel_transform(**HT_kwargs)
            self.j_nus=self.HT.j_nus

        self.bin_cl=bin_cl
        self.bin_xi=bin_xi
        self.set_bin_params()
        self.cov_indxs=[]
        self.corr_indxs=[j for j in itertools.combinations_with_replacement(np.arange(self.zs_bins  ['n_bins']),2)]
        
    def set_bin_params(self):
        """
            Setting up the binning functions to be used in binning the data
        """
        self.binning=binning()
        if self.bin_cl:
            self.cl_bin_utils=self.binning.bin_utils(r=self.l,r_bins=self.l_bins,
                                                r_dim=2,mat_dims=[1,2])
        if self.do_xi and self.bin_xi:
            self.xi_bin_utils={}
            for j_nu in self.j_nus:
                self.xi_bin_utils[j_nu]=self.binning.bin_utils(r=self.HT.r[j_nu]/d2r,
                                                    r_bins=self.theta_bins,
                                                    r_dim=2,mat_dims=[1,2])

    def calc_galaxy_lens_cl(self,zs=None,zg=None):
        """
            Compute the angular power spectra, Cl between two source bins
            zs1, zs2: Source bins. Dicts containing information about the source bins
        """
        clz=self.Ang_PS.clz
        cls=clz['cls']
        f=clz['f']    
        sc=zs1['sig_c_int']*zg['dndz_int']

        cl=np.dot(sc*clz['dchi'],cls)
        
        # cl/=zs2['Norm']*zs1['Norm'] #absorbed into sig_c_int
        cl/=f**2# cl correction from Kilbinger+ 2017
        return cl
    
    #@delayed
    def galaxy_kappa_cl(self,zs_indx=-1, zg_indx=-1,
                pk_func=None,pk_params=None,cosmo_h=None,cosmo_params=None):
        """
            Wrapper for calc_lens_lens_cl. Checks to make sure quantities such as power spectra and cosmology 
            are available otherwise sets them to some default values.
            zs1_indx, zs2_indx: Indices of the source bins to be correlated.
            Others are arguments to be passed to power spectra function is it needs to be computed
        """
        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h
        
        l=self.l
        zs=self.zs_bins[zs1_indx]#.copy() #we will modify these locally
        zg=self.zg_bins[zg_indx]#.copy()
        if zs1['sig_c'] is None: 
            self.lensing_utils.set_zs_sigc(cosmo_h=cosmo_h,zl=self.Ang_PS.z)
        if zg['dndz'] is None:
            self.galaxy_utils.set_zs_sigc(cosmo_h=cosmo_h,zl=self.Ang_PS.z)
        
        if self.Ang_PS.clz is None:
            self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                                cosmo_params=cosmo_params)
    
        cl=self.calc_galaxy_lens_cl(zs1=zs1,zg=zg)
        out={'l':l,'cl':cl}
        if self.bin_cl:
            out['binned']=self.bin_kappa_cl(results=out,bin_cl=True)
            #need unbinned cl for covariance
        return out

    #@delayed
    def galaxy_kappa_cl_cov(self,cls=None, zs_indx=[],zg_indx=[]):
        """
            Computes the covariance between any two tomographic power spectra.
            cls: tomographic cls already computed before calling this function
            zs_indx: 4-d array, noting the indices of the source bins involved in the tomographic 
                    cls for which covariance is computed. For ex. covariance between 12, 56 tomographic cross correlations involve 1,2,5,6 source bins
        """
        cov={}
        l=self.l 

        #galaxy-galaxy*lensing-lensing XXX: need galaxy-galaxy and lensing-lensing auto correlations
        cov['G1324']=(cls_g[:,zg_indx[0],zg_indx[1]]+self.galaxy_utils.SN[:,zg_indx[0],zg_indx[1]]) 
        cov['G1324']*=(cls_s[:,zs_indx[0],zs_indx[1]]+self.lensing_utils.SN[:,zs_indx[0],zs_indx[1]])
        cov['G1324']/=self.cov_utils.gaussian_cov_norm

        cov['G1423']=(cls[:,zg_indx[0],zs_indx[1]])  #galaxy-lensing term
        cov['G1423']*=(cls[:,zg_indx[1],zs_indx[0]])
        cov['G1423']/=self.cov_utils.gaussian_cov_norm

        cov['final']=None
        if not self.do_xi:
            cov['G']=np.diag(cov['G1423']+cov['G1324'])# this can be expensive with large l
            # cov['G']/=(2.*l+1.)*self.cov_utils.f_sky#*np.gradient(l) #need Delta l here. Even when 
                                                                    #binning later
            cov['final']=cov['G']
        if self.SSV_cov: #FIXME: Need the correct formalism for g-l correlations
            clz=self.Ang_PS.clz
            zs1=self.zs_bins[zs_indx[0]]
            zs2=self.zs_bins[zs_indx[1]]
            zg1=self.zg_bins[zg_indx[0]]
            zg2=self.zg_bins[zg_indx[1]]
            sigma_win=self.cov_utils.sigma_win
            
            sig_cL=zs1['sig_c_int']*zs2['sig_c_int']
            sig_cL*=zg1['dndz_int']*zg2['dndz_int']

            sig_cL*=self.Ang_PS.clz['dchi']
            sig_cL*=sigma_win

            clr1=self.Ang_PS.clz['clsR']
           
            cov['SSC_dd']=np.dot((clr1).T*sig_cL,clr1)
            # cov['SSC_dd']=np.einsum('i,jki->jk',sig_cL,clr1)
            cov['final']+=cov['SSC_dd']
        
            if self.tidal_SSV_cov:
                #sig_cL will be divided by some factors to account for different sigma_win
                clrk=self.Ang_PS.clz['clsRK']
                
                cov['SSC_kk']=np.dot((clrk).T*sig_cL/36.,clrk)
                cov['SSC_dk']=np.dot((clr1).T*sig_cL/6.,clrk)
                
                cov['final']+=cov['SSC_kk']+cov['SSC_dk']*2. #+cov['SSC_kd']    
        if self.bin_cl:
            cov=self.bin_kappa_cl(results=cov,bin_cov=True)
        return cov
    
    def bin_kappa_cl(self,results=None,bin_cl=False,bin_cov=False):
        """
            bins the tomographic power spectra
            results: Either cl or covariance
            bin_cl: if true, then results has cl to be binned
            bin_cov: if true, then results has cov to be binned
            Both bin_cl and bin_cov can be true simulatenously. 
        """
        results_b={}
        if bin_cl:
            results_b['cl']=self.binning.bin_1d(r=self.l,xi=results['cl'],
                                        r_bins=self.l_bins,r_dim=2,bin_utils=self.cl_bin_utils)
        if bin_cov:
            keys=['final']
            keys=['G','final']
            if self.SSV_cov:
                keys=np.append(keys,'SSC_dd')
                if self.tidal_SSV_cov:
                    keys=np.append(keys,['SSC_kk','SSC_dk']) #SSC_kd
            for k in keys:
                results_b[k]=self.binning.bin_2d(r=self.l,cov=results[k],r_bins=self.l_bins,r_dim=2
                                        ,bin_utils=self.cl_bin_utils)
            #results_b=cov_b
        return results_b

    def combine_cl_tomo(self,cl_compute_dict={}):
        cl=np.zeros((len(self.l),self.ns_bins,self.ns_bins))
        for (i,j) in self.corr_indxs+self.cov_indxs:#we need unbinned cls for covariance
            clij=cl_compute_dict[(i,j)]#.compute()
            cl[:,i,j]=clij['cl']
            cl[:,j,i]=clij['cl']
            
        if self.bin_cl:
            cl_b=np.zeros((len(self.l_bins)-1,self.ns_bins,self.ns_bins))
            for (i,j) in self.corr_indxs+self.cov_indxs:
                cl_b[:,i,j]=clij['binned']['cl']
                cl_b[:,j,i]=clij['binned']['cl']
        else:
            cl_b=cl
        return {'cl':cl,'cl_b':cl_b}
            

    def kappa_cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None):
        """
         Computes full tomographic power spectra and covariance, including shape noise. output is 
         binned also if needed.
         Arguments are for the power spectra  and sigma_crit computation, 
         if it needs to be called from here.
         source bins are already set. This function does set the sigma crit for sources. 
        """
        nbins=self.ns_bins
        l=self.l 
        
        self.set_zs_sigc(cosmo_h=cosmo_h,set_zs_sigc) 

        out={}
        cov={}
        for (i,j) in self.corr_indxs+self.cov_indxs:
            out[(i,j)]=delayed(self.kappa_cl)(zs1_indx=i,zs2_indx=j,cosmo_h=cosmo_h,
                                    cosmo_params=cosmo_params,pk_params=pk_params,
                                    pk_func=pk_func)
        
        cl=delayed(self.combine_cl_tomo)(out)
        if self.do_xi:
            return cl.compute()
        else:     
            cl_b=cl['cl_b']
            cl=cl['cl']
            if self.do_cov: 
            #need large l range for xi which leads to memory issues.. donot do cov here for xi
                for i in self.corr_indxs: #np.arange(len(indxs)):
                    for j in self.corr_indxs: #np.arange(i,len(indxs)):
                        indx=i+j #indxs[i]+indxs[j]#np.append(indxs[i],indxs[j])
                        cov[indx]=delayed(self.kappa_cl_cov)(cls=cl, zs_indx=indx)
            out_stack=delayed(self.stack_dat)({'cov':cov,'cl':cl_b})
            return {'stack':out_stack,'cl0':cl_b,'cov0':cov}
        

    def compute_cov_tomo(self,covG):
        cov={}
        for i in covG.keys():
            cov[i]=covG[i].compute()
        return cov

    def cut_clz_lxi(self,clz=None,l_xi=None):
        """
            For hankel transform is done on l-theta grid, which is based on j_nu. So grid is 
            different for xi+ and xi-.
            When computing a given xi, we need to cut cls only to l values which are defined on the 
            grid for j_nu relevant to that xi. This function does that. 
        """
        x=np.isin(self.l,l_xi)
        clz['f']=(l_xi+0.5)**2/(l_xi*(l_xi+1.)) # cl correction from Kilbinger+ 2017
        clz['cls']=clz['cls'][:,x]
        return clz

    def xi_cov(self,cov_cl={},j_nu=None,j_nu2=None):
        """
            Computes covariance of xi, by performing 2-D hankel transform on covariance of Cl.
            In current implementation of hankel transform works only for j_nu=j_nu2. So no cross covariance between xi+ and xi-.
        """
        #FIXME: Implement the cross covariance
        cov_xi={}
        Norm= self.Om_W
        th0,cov_xi['G1423']=self.HT.projected_covariance(k_pk=self.l,j_nu=j_nu,
                                                     pk1=cov_cl['G1423'],pk2=cov_cl['G1423'])
                                                     
        th2,cov_xi['G1324']=self.HT.projected_covariance(k_pk=self.l,j_nu=j_nu2,
                                                        pk1=cov_cl['G1324'],pk2=cov_cl['G1324'],)
                                                     
        cov_xi['G1423']=self.binning.bin_2d(r=th0,cov=cov_xi['G1423'],r_bins=self.theta_bins,
                                                r_dim=2,bin_utils=self.xi_bin_utils[j_nu])
        cov_xi['G1324']=self.binning.bin_2d(r=th2,cov=cov_xi['G1324'],r_bins=self.theta_bins,
                                                r_dim=2,bin_utils=self.xi_bin_utils[j_nu2])
        cov_xi['G']=cov_xi['G1423']+cov_xi['G1324']
        cov_xi['G']/=Norm
        cov_xi['final']=cov_xi['G']
        if self.SSV_cov:
            keys=['SSC_dd']
            if self.tidal_SSV_cov:
                keys=np.append(keys,['SSC_kk','SSC_dk','SSC_kd'])
            for k in keys:
                th,cov_xi[k]=self.HT.projected_covariance2(k_pk=self.l,j_nu=j_nu,
                                                            pk_cov=cov_cl[k])
                cov_xi[k]=self.binning.bin_2d(r=th,cov=cov_xi[k],r_bins=self.theta_bins,
                                                r_dim=2,bin_utils=self.xi_bin_utils[j_nu])
                cov_xi[k]/=Norm
                cov_xi['final']+=cov_xi[k]
        return cov_xi

    def combine_xi_tomo(self,cl=[],j_nu=0):
        xi=np.zeros((len(self.theta_bins)-1,self.ns_bins,self.ns_bins))
        l_nu=self.HT.k[j_nu]
        for (i,j) in self.corr_indxs:
            th,xi_ij=self.HT.projected_correlation(k_pk=l_nu,j_nu=j_nu,pk=cl[:,i,j])
            xi[:,i,j]=self.binning.bin_1d(r=th/d2r,xi=xi_ij,
                                        r_bins=self.theta_bins,r_dim=2,
                                        bin_utils=self.xi_bin_utils[j_nu])
            xi[:,j,i]=xi[:,i,j]
        return xi

    def xi_tomo_cov(self,cl=[],j_nu1=0,j_nu2=0):
        cov_xi={}
        for i in self.corr_indxs: #np.arange(ni):
            for j in self.corr_indxs:#np.arange(i,ni):
                indx=i+j #indxs[i]+indxs[j]
                cov_cl_i=delayed(self.kappa_cl_cov)(cls=cl,zs_indx=indx)
                                        #need large l range for xi which leads to memory issues
                cov_xi[indx]=delayed(self.xi_cov)(cov=cov_cl_i,j_nu=j_nu1,j_nu2=j_nu2)
        return cov_xi

    def kappa_xi_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None):
        """
            Computed tomographic angular correlation functions. First calls the tomographic 
            power spectra and covariance and then does the hankel transform and  binning.
        """
        self.l=np.sort(np.unique(np.hstack((self.HT.k[i] for i in self.j_nus))))
                
        nbins=self.ns_bins
        cov_xi={}
        xi={}
        out={}
        for j_nu in [0]: #self.j_nus:
            l_nu=self.HT.k[j_nu]
            self.l=l_nu
            self.Ang_PS.l=l_nu

            cls_tomo_nu=delayed(self.kappa_cl_tomo)(cosmo_h=cosmo_h,cosmo_params=cosmo_params,
                                                     pk_params=pk_params,pk_func=pk_func)

            #cl=delayed(self.combine_cl_tomo)(cls_tomo_nu)
            #cl=cl['cl']
            cl=cls_tomo_nu['cl']
            xi[j_nu]=delayed(self.combine_xi_tomo)(cl=cl,j_nu=j_nu)

            if self.do_cov:
                j_nu2=j_nu
                if j_nu==0 and self.tracer=='shear':
                    j_nu2=4
                cov_xi[j_nu]=delayed(self.xi_tomo_cov)(cl=cl,j_nu1=0,j_nu2=0)

            out[j_nu]={}
            out[j_nu]['stack']=delayed(self.stack_dat)({'cov':cov_xi[j_nu],'xi':xi[j_nu]})        
        out['xi']=xi
        out['cov']=cov_xi
        
        return out

    def stack_dat(self,dat):
        """
            outputs from tomographic caluclations are dictionaries. This fucntion stacks them such that the cl or xi is a long 1-d array and the covariance is N X N array. 
            dat: output from tomographic calculations.
            XXX: reason that outputs tomographic bins are distionaries is that it make is easier to 
            handle things such as binning, hankel transforms etc. We will keep this structure for now.
        """
        nbins=self.ns_bins
        nD=len(self.corr_indxs) #np.int64(nbins*(nbins-1.)/2.+nbins)
        nD2=1
        est='cl'
        if self.do_xi:
            est='xi'
        
        nX=len(dat[est])
        D_final=np.zeros(nD*nX*nD2)
        cov_final=np.zeros((nD*nX*nD2,nD*nX*nD2))
        # print( D_final.shape)
        ij=0
        for iD2 in np.arange(nD2):
            dat2=dat[est]
            # if self.do_xi:
            #     dat2=dat[est][d_k[iD2]]
            
            D_final[nD*nX*iD2:nD*nX*(iD2+1)]=np.hstack((dat2[:,i,j] for (i,j) in self.corr_indxs))

            if not self.do_cov:
                cov_final=None
                continue
            dat2=dat['cov']
            # if self.do_xi:
            #     dat2=dat['cov'][d_k[iD2]]
            
            i_indx=0
            for i in np.arange(len(self.corr_indxs)):
                for j in np.arange(i,len(self.corr_indxs)):
                    indx=self.corr_indxs[i]+self.corr_indxs[j]
                    #print(indx,self.corr_indxs[i],self.corr_indxs[j])
                    # print(dat2[indx]['final'])
                    cov_final[ i*nX : (i+1)*nX , j*nX : (j+1)*nX] = dat2[indx]['final']
                    cov_final[ j*nX : (j+1)*nX , i*nX : (i+1)*nX] = dat2[indx]['final']
        out={'cov':cov_final}
        out[est]=D_final
        return out
            
if __name__ == "__main__":
    import cProfile
    import pstats

    