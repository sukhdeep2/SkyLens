import os,sys

from power_spectra import *
from angular_power_spectra import *
from hankel_transform import *
from binning import *
from cov_utils import *
from lensing_utils import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d

d2r=np.pi/180.
c=c.to(u.km/u.second)

class Kappa():
    def __init__(self,silence_camb=False,l=np.arange(2,2001),HT=None,Ang_PS=None,
                lensing_utils=None,cov_utils=None,
                power_spectra_kwargs=None,HT_kwargs=None,zs=None,pzs=None,ns=None,z_bins=None,
                lens_weight=False,zl=None,n_zl=100,log_zl=True,do_cov=False,SSV_cov=False,
                tidal_SSV_cov=False,sigma_gamma=0.3,f_sky=0.3,l_bins=None,bin_cl=False,
                stack_data=False,bin_xi=False,do_xi=False,theta_bins=None,tracer='kappa'):

        self.lens_weight=lens_weight
        self.lensing_utils=lensing_utils
        if lensing_utils is None:
            self.lensing_utils=Lensing_utils(sigma_gamma=sigma_gamma)
        
        self.set_ls_zbins(zs=zs,pzs=pzs,z_bins=z_bins,zl=zl,n_zl=n_zl,log_zl=log_zl,ns=ns)
        self.l=l
        
        self.cov_utils=cov_utils
        if cov_utils is None:
            self.cov_utils=Cov_utils(f_sky=f_sky,l=self.l)

        self.Ang_PS=Ang_PS
        if Ang_PS is None:
            self.Ang_PS=Angular_power_spectra(silence_camb=silence_camb,SSV_cov=SSV_cov,l=self.l,
                        power_spectra_kwargs=power_spectra_kwargs,cov_utils=cov_utils,
                        zl=self.zl,n_zl=n_zl)

        
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
        
        self.do_cov=do_cov
        self.SSV_cov=SSV_cov
        self.tidal_SSV_cov=tidal_SSV_cov
        
    def set_ls_zbins(self,zs=None,pzs=None,z_bins=None,zl=None,n_zl=10,log_zl=False,ns=None):
        zl_min=0
        zl_max=np.amax(zs.values()) if isinstance(zs,dict) else np.amax(zs)

        if zl is None:
            if log_zl:#bins for z_lens.
                self.zl=np.logspace(np.log10(max(zl_min,1.e-4)),np.log10(zl_max),n_zl)
            else:
                self.zl=np.linspace(zl_min,zl_max,n_zl)
        else:
            self.zl=zl
        self.dzl=np.gradient(self.zl)
    
        self.zs_bins={}
        if z_bins is None:
            self.ns_bins=len(zs.keys()) #pass zs bins as dicts
            k=zs.keys()
            for i in np.arange(self.ns_bins):
                self.zs_bins[i]={}
                self.zs_bins[i]['z']=np.array(zs[k[i]])
                self.zs_bins[i]['pz']=np.array(pzs[k[i]])
                self.zs_bins[i]['ns']=np.array(ns[k[i]])
                self.zs_bins[i]['pz0']=self.zs_bins[i]['pz']
                self.zs_bins[i]['W']=1.
        else:
            self.ns_bins=len(z_bins)-1
            if self.lens_weight:
                self.ns_bins=len(z_bins)
            for i in np.arange(self.ns_bins):
                self.zs_bins[i]={}
                if self.lens_weight:
                    self.zs_bins[i]['z']=zs
                    self.zs_bins[i]['ns']=ns
                    self.zs_bins[i]['W']=1./self.lensing_utils.sigma_crit(zl=z_bins[i],zs=zs,
                                                            cosmo_h=self.PS.cosmo_h)
                    self.zs_bins[i]['pz']=pzs*self.zs_bins[i]['W']
                    self.zs_bins[i]['pz0']=pzs
                else:
                    xi=zs>z_bins[i]
                    xi*=zs<z_bins[i+1]
                    self.zs_bins[i]['z']=zs[xi]
                    self.zs_bins[i]['pz']=pzs[xi]
                    self.zs_bins[i]['ns']=ns[xi]
                    self.zs_bins[i]['pz0']=self.zs_bins[i]['pz']
                    self.zs_bins[i]['W']=1.
        for i in self.zs_bins.keys():
            self.zs_bins[i]['dz']= np.gradient(self.zs_bins[i]['z']) if len(self.zs_bins[i]['z'])>1 else np.array([1])
            self.zs_bins[i]['pzdz']=self.zs_bins[i]['pz']*self.zs_bins[i]['dz']
            self.zs_bins[i]['Norm']=np.sum(self.zs_bins[i]['pzdz'])
            self.zs_bins[i]['SN']=self.lensing_utils.shape_noise_calc(zs1=self.zs_bins[i],zs2=self.zs_bins[i])
        self.reset_zs()

    def set_bin_params(self):
        self.binning=binning()
        if self.bin_cl:
            self.cl_bin_utils=self.binning.bin_utils(r=self.l,r_bins=self.l_bins,
                                                r_dim=2,mat_dims=[1,2])
        if self.bin_xi:
            self.xi_bin_utils={}
            for j_nu in self.j_nus:
                self.xi_bin_utils[j_nu]=self.binning.bin_utils(r=self.HT.r[j_nu]/d2r,
                                                    r_bins=self.theta_bins,
                                                    r_dim=2,mat_dims=[1,2])

    def reset_zs(self):
        for i in np.arange(self.ns_bins):
            self.zs_bins[i]['sig_c']=None
            self.zs_bins[i]['sig_c_int']=None 
            #this is cosmology dependent. Need to reset after every cosmo dependent run

    def set_zs_sigc(self,cosmo_h=None):
        #We need to compute these only once in every run 
        # i.e not repeat for every ij combo
        if cosmo_h is None:
            cosmo_h=self.PS.cosmo_h
        rho=self.lensing_utils.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0 
            
        for i in np.arange(self.ns_bins):
            self.zs_bins[i]['sig_c']=rho/self.lensing_utils.sigma_crit(zl=self.zl,
                                                        zs=self.zs_bins[i]['z'],
                                                        cosmo_h=cosmo_h)
            self.zs_bins[i]['sig_c_int']=np.dot(self.zs_bins[i]['pzdz'],self.zs_bins[i]['sig_c'])
    
    def kappa_cl(self,clz_dict=None, return_clz=False,zs1_indx=-1, zs2_indx=-1,
                pk_func=None,pk_params=None,cosmo_h=None,cosmo_params=None,DC=None):
        if cosmo_h is None:
            cosmo_h=self.A_PS.PS.cosmo_h
        
        l=self.l
        zs1=self.zs_bins[zs1_indx]#.copy() #we will modify these locally
        zs2=self.zs_bins[zs2_indx]#.copy()
        if zs1['sig_c'] is None or zs2['sig_c'] is None:
            self.set_zs_sigc(cosmo_h=cosmo_h)
        
        if self.Ang_PS.clz is None:
            self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                                cosmo_params=cosmo_params)
    
        cl=self.Ang_PS.calc_cl(clz=clz_dict,zs1=zs1,zs2=zs2)
        out={'l':l,'cl':cl}
        if self.bin_cl:
            out['binned']=self.bin_kappa_cl(results=out,bin_cl=True)
        return out

    def kappa_cl_cov(self,cls=None,SN=None, zs_indx=[]):
        cov={}
        l=self.l 
        cov['G1324']=(cls[:,zs_indx[0],zs_indx[2]]+SN[:,zs_indx[0],zs_indx[2]])
        cov['G1324']*=(cls[:,zs_indx[1],zs_indx[3]]+SN[:,zs_indx[1],zs_indx[3]])

        cov['G1423']=(cls[:,zs_indx[0],zs_indx[3]]+SN[:,zs_indx[0],zs_indx[3]])
        cov['G1423']*=(cls[:,zs_indx[1],zs_indx[2]]+SN[:,zs_indx[1],zs_indx[2]])

        cov['final']=0
        if not self.do_xi:
            cov['G']=np.diag(cov['G1423']+cov['G1324'])# this can be expensive with large l
            cov['G']/=(2.*l+1.)*self.cov_utils.f_sky
            cov['final']=cov['G']
        if self.SSV_cov:
            clz=self.Ang_PS.clz
            zs1=self.zs_bins[zs_indx[0]]
            zs2=self.zs_bins[zs_indx[1]]
            zs3=self.zs_bins[zs_indx[2]]
            zs4=self.zs_bins[zs_indx[3]]
            sigma_win=clz['sigma_win']
            
            sig_cL=zs1['sig_c_int']*zs2['sig_c_int']*zs3['sig_c_int']*zs4['sig_c_int']

            sig_cL*=self.dzl*clz_dict['cH']
            sig_cL/=self.Om_W**2
            sig_cL*=sigma_win
            sig_cL/=zs1['Norm']*zs2['Norm']*zs3['Norm']*zs4['Norm']

            clr1=clz_dict['clsR']
           
            cov['SSC_dd']=np.dot((clr1).T*sig_cL,clr1)
            cov['final']=cov['SSC_dd']+cov['final']
            #print np.all(np.isclose(cov['SSC_dd2'],cov['SSC_dd']))
        
            if self.tidal_SSV_cov:
                #sig_cL will be divided by some factors to account for different sigma_win
                clrk=clz_dict['clsRK']
                cov['SSC_kk']=np.dot((clrk).T*sig_cL/36.,clrk)
                cov['SSC_dk']=np.dot((clr1).T*sig_cL/6.,clrk)
                cov['SSC_kd']=np.dot((clrk).T*sig_cL/6.,clr1)

                cov['SSC_tidal']=cov['SSC_kd']+cov['SSC_dk']+cov['SSC_kk']    
            
                cov['final']+=cov['SSC_tidal']
        if self.bin_cl:
            cov=self.bin_kappa_cl(results=cov,bin_cov=True)
        return cov
    
    def bin_kappa_cl(self,results=None,bin_cl=False,bin_cov=False):
        results_b={}
        if bin_cl:
            results_b['cl']=self.binning.bin_1d(r=self.l,xi=results['cl'],
                                        r_bins=self.l_bins,r_dim=2,bin_utils=self.cl_bin_utils)
        if bin_cov:
            cov_b={}
            keys=['final']
            keys=['G','final']
            if self.SSV_cov:
                keys=np.append(keys,'SSC_dd')
                if self.tidal_SSV_cov:
                    keys=np.append(keys,['SSC_kk','SSC_dk','SSC_kd'])
            for k in keys:
                cov_b[k]=self.binning.bin_2d(r=self.l,cov=results[k],r_bins=self.l_bins,r_dim=2
                                        ,bin_utils=self.cl_bin_utils)
            results_b=cov_b
        return results_b
    

    def kappa_cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None,
                        return_clz=False,clz_dict=None):
        nbins=self.ns_bins
        l=self.l 
        
        if not self.do_xi: #this should already be set in xi function
            self.set_zs_sigc(cosmo_h=cosmo_h) 

        cl=np.zeros((len(l),nbins,nbins))
        if self.bin_cl:
            cl_b=np.zeros((len(self.l_bins)-1,nbins,nbins))#we need unbinned cls for covariance
        SNij=None
        cov={}

        SN=np.zeros((1,nbins,nbins)) if self.do_cov else None

        if clz_dict is None:
            clz_dict=self.cl_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                        cosmo_params=cosmo_params)
        
        #following can be parallelized 
        for i in np.arange(nbins):
            for j in np.arange(i,nbins): #we assume i,j ==j,i
                out=self.kappa_cl(zs1_indx=i,zs2_indx=j,clz_dict=clz_dict,cosmo_h=cosmo_h,
                                    cosmo_params=cosmo_params,pk_params=pk_params,
                                    pk_func=pk_func,return_clz=False)
                cl[:,i,j]=out['cl']
                cl[:,j,i]=out['cl']
                if self.bin_cl:
                    cl_b[:,i,j]=out['binned']['cl']
                    cl_b[:,j,i]=out['binned']['cl']
                if self.do_cov:
                    if i==j:
                        SN[:,i,j]=self.zs_bins[i]['SN']
                    elif self.lens_weight:
                        SN[:,i,j]=self.shape_noise_calc(zs1=self.zs_bins[i],zs2=self.zs_bins[j])
                        SN[:,i,j]=SN[:,j,i]     

        if self.do_cov and not self.do_xi: #need large l range for xi which leads to memory issues
            cov={}
            indxs=[j for j in itertools.combinations_with_replacement(np.arange(nbins),2)]
            for i in np.arange(len(indxs)):
                for j in np.arange(i,len(indxs)):
                    indx=indxs[i]+indxs[j]#np.append(indxs[i],indxs[j])
                    cov[indx]=self.kappa_cl_cov(clz_dict=clz_dict,cls=cl,SN=SN, 
                                        zs_indx=indx)

        cl=cl_b if self.bin_cl else cl
        l=self.cl_bin_utils['bin_center'] if self.bin_cl else self.l
        out={'l':l,'cl':cl,'SN':SN,'cov':cov}
        
        if return_clz:
            out['clz_dict']=clz_dict
        if not self.do_xi:
            self.reset_zs()
        return out

    def cut_clz_lxi(self,clz=None,l_xi=None):
        x=np.isin(self.l,l_xi)
        clz['f']=(l_xi+0.5)**2/(l_xi*(l_xi+1.)) # cl correction from Kilbinger+ 2017
        clz['cls']=clz['cls'][:,x]
        return clz

    def xi_cov(self,cov={},j_nu=None,j_nu2=None):
        cov_xi={}
        Norm= self.Om_W
        th0,cov_xi['G1423']=self.HT.projected_covariance(k_pk=self.l,j_nu=j_nu,
                                                     pk1=cov['G1423'],pk2=cov['G1423'])
                                                     
        th2,cov_xi['G1324']=self.HT.projected_covariance(k_pk=self.l,j_nu=j_nu2,
                                                        pk1=cov['G1324'],pk2=cov['G1324'],)
                                                     
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
                                                            pk_cov=cov[k])
                cov_xi[k]=self.binning.bin_2d(r=th,cov=cov_xi[k],r_bins=self.theta_bins,
                                                r_dim=2,bin_utils=self.xi_bin_utils[j_nu])
                cov_xi[k]/=Norm
                cov_xi['final']+=cov_xi[k]
        return cov_xi

    def kappa_xi_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None):
        self.l=np.sort(np.unique(np.hstack((self.HT.k[i] for i in self.j_nus))))
        self.l=np.append(self.l,[20000,50000])
        print 'l changed for xi',self.l.shape
        self.set_zs_sigc(cosmo_h=cosmo_h) 
        clz_dict=self.cl_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                        cosmo_params=cosmo_params)
        nbins=self.ns_bins
        cov_xi={}
        xi={}
        
        for j_nu in [0]: #self.j_nus:
            l_nu=self.HT.k[j_nu]
            xi[j_nu]=np.zeros((len(self.theta_bins)-1,nbins,nbins))
            cls_tomo_nu=clz_dict.copy()
            cls_tomo_nu=self.cut_clz_lxi(clz=clz_dict,l_xi=l_nu)
            self.l=l_nu

            cls_tomo_nu=self.kappa_cl_tomo(cosmo_h=cosmo_h,cosmo_params=cosmo_params,
                                    pk_params=pk_params,pk_func=pk_func,
                                    clz_dict=cls_tomo_nu,return_clz=False)


            for i in np.arange(nbins):
                for j in np.arange(i,nbins): # we assume i,j==j,i
                    th,xi_ij=self.HT.projected_correlation(k_pk=l_nu,j_nu=j_nu,
                                                            pk=cls_tomo_nu['cl'][:,i,j])
                    xi[j_nu][:,i,j]=self.binning.bin_1d(r=th/d2r,xi=xi_ij,
                                        r_bins=self.theta_bins,r_dim=2,
                                        bin_utils=self.xi_bin_utils[j_nu])
                    xi[j_nu][:,j,i]=xi[j_nu][:,i,j]

            if self.do_cov:
                cov_xi[j_nu]={}

                j_nu2=j_nu
                if j_nu==0 and self.tracer=='shear':
                    j_nu2=4

                indxs=[j for j in itertools.combinations_with_replacement(np.arange(nbins),2)]
                ni=len(indxs)
                for i in np.arange(ni):
                    for j in np.arange(i,ni):
                        indx=indxs[i]+indxs[j]
                        cov_cl_i=self.kappa_cl_cov(clz_dict=cls_tomo_nu,cls=cls_tomo_nu['cl'],
                                                SN=cls_tomo_nu['SN'],zs_indx=indx)
                        cov_xi[j_nu][indx]=self.xi_cov(cov=cov_cl_i,j_nu=j_nu,j_nu2=j_nu2)
        out={}
        out['xi']=xi
        out['cov']=cov_xi
        self.reset_zs()
        return out

    def stack_dat(self,dat):
        nbins=self.ns_bins
        nD=np.int64(nbins*(nbins-1.)/2.+nbins)
        nD2=1
        est='cl'
        if self.do_xi:
            est='xi'
            d_k=dat[est].keys()
            nD2=len(d_k)
            nX=len(dat[est][d_k[0]])
        else:
            nX=len(dat[est])
        D_final=np.zeros(nD*nX*nD2)
        cov_final=np.zeros((nD*nX*nD2,nD*nX*nD2))
        
        ij=0
        for iD2 in np.arange(nD2):
            dat2=dat[est]
            if self.do_xi:
                dat2=dat[est][d_k[iD2]]
            indxs=itertools.combinations_with_replacement(np.arange(nbins),2)
            D_final[nD*nX*iD2:nD*nX*(iD2+1)]=np.hstack((dat2[:,i,j] for (i,j) in indxs))

            dat2=dat['cov']
            if self.do_xi:
                dat2=dat['cov'][d_k[iD2]]
            indxs=[j for j in itertools.combinations_with_replacement(np.arange(nbins),2)]
            i_indx=0
            for i in np.arange(len(indxs)):
                for j in np.arange(i,len(indxs)):
                    indx=indxs[i]+indxs[j]
                    cov_final[ i*nX : (i+1)*nX , j*nX : (j+1)*nX] = dat2[indx]['final']
                    cov_final[ j*nX : (j+1)*nX , i*nX : (i+1)*nX] = dat2[indx]['final']
        out={'cov':cov_final}
        out[est]=D_final
        return out
            

def lsst_zsource(alpha=1.24,z0=0.51,beta=1.01,z=[]):
    pzs=z**alpha*np.exp(-(z/z0)**beta)
    pzs/=np.sum(np.gradient(z)*pzs)
    return pzs

if __name__ == "__main__":
    import cProfile
    import pstats

    # kappa_fn=Kappa(zs={0:[1100]},pzs={0:[1]})
    #         #    l,cl=PS.kappa_cl(n_zl=140,log_zl=True,zl_min=1.e-4,zl_max=1100) #camb
    # cl2=kappa_fn.kappa_cl(zs1_indx=0, zs2_indx=0) #pk_func=kappa_fn.PS.ccl_pk)
    # fname='kappa_cl_cmb'
    #         #np.savetxt(fname+'_camb.dat',np.column_stack((l,cl)))
    #         #    np.savetxt(fname+'_ccl.dat',np.column_stack((l,cl2)))

    z=np.linspace(0,5,200)
    pzs=lsst_zsource(z=z)
    x=z<2
    z=z[x]
    pzs=pzs[x]

    ns0=26
    ns=ns0*pzs

    nbins=3
    zs_bins=np.linspace(0.1,2,nbins+1)
    zl_bins=np.linspace(0.5,1.5,nbins+1)
    lmax_cl=2000
    lmin_cl=2
    l=np.arange(lmin_cl,lmax_cl)
    l_bins=np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),20))
    do_cov=True
    bin_cl=True
    bin_xi=True
    do_xi=False
    theta_bins=np.logspace(np.log10(1./60),1,20)

    cProfile.run('kappa_fn = Kappa(zs=z,pzs=pzs,l=l,z_bins=zs_bins,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,bin_xi=bin_xi,theta_bins=theta_bins,do_xi=do_xi,ns=ns)', 'output_stats')
                                            # globals(), locals()) #runctx
    p = pstats.Stats('output_stats')
    p.sort_stats('tottime').print_stats(2)

    cProfile.run('clS=kappa_fn.kappa_cl_tomo()','output_stats2')
    p2 = pstats.Stats('output_stats2')
    p2.sort_stats('tottime').print_stats(10)
    
    # cProfile.run('clS=kappa_fn.stack_dat(clS)','output_stats3')
    # p2 = pstats.Stats('output_stats3')
    # p2.sort_stats('tottime').print_stats(10)

    bin_cl=True

    cProfile.run('kappa_fn = Kappa(zs=z,pzs=pzs,l=l,z_bins=zl_bins,lens_weight=True,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins)', 'output_statsL')
                                            # globals(), locals()) #runctx
    p = pstats.Stats('output_statsL')
    p.sort_stats('tottime').print_stats(2)

    cProfile.run('clL=kappa_fn.kappa_cl_tomo()','output_statsL2')
    p2 = pstats.Stats('output_statsL2')
    p2.sort_stats('tottime').print_stats(10)

    # kappa_fn=Kappa(zs=z,pzs=pzs,l=l,z_bins=zl_bins,lens_weight=True)
    # clL=kappa_fn.kappa_cl_many_bins()

    #kappa_fn=Kappa(zs=z,pzs=pzs,l=l,z_bins=zs_bins)
    #clS=kappa_fn.kappa_cl_many_bins()
