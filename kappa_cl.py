import os,sys

from power_spectra import *
from hankel_transform import *
from binning import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad as scipy_int1d

d2r=np.pi/180.
sky_area=np.pi*4/(d2r)**2 #in degrees
c=c.to(u.km/u.second)

class Kappa():
    def __init__(self,silence_camb=False,l=np.arange(2,2001),HT=None,power_spectra_kwargs={},
                HT_kwargs=None,zs=None,pzs=None,z_bins=None,lens_weight=False,
                zl=None,n_zl=100,log_zl=True,do_cov=False,SSV_cov=False,tidal_SSV_cov=False,
                ns=26,sigma_gamma=0.3,f_sky=0.3,l_bins=None,bin_cl=False,
                bin_xi=False,do_xi=False,theta_bins=None,tracer='kappa'):
        self.PS=Power_Spectra(silence_camb=silence_camb,**power_spectra_kwargs)
        self.l=l
        self.l_bins=l_bins
        self.theta_bins=theta_bins
        self.do_xi=do_xi
        self.bin_utils=None
        self.HT=HT
        self.tracer=tracer
        self.G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        self.G2*=8*np.pi/3.
        if HT is None:
            if HT_kwargs is None:
                if theta_bins is None:
                    th_min=1./60.
                    th_max=5
                else:
                    th_min=np.amin(theta_bins)
                    th_max=np.amax(theta_bins)
                HT_kwargs={'kmin':min(l),'kmax':max(l),'rmin':th_min*d2r,
                                    'rmax':th_max*d2r,
                            'n_zeros':2000,'prune_r':2,'j_nu':[0]}
            self.HT=hankel_transform(**HT_kwargs)
        self.j_nus=self.HT.j_nus

        self.bin_cl=bin_cl
        self.bin_xi=bin_xi
        self.set_bin_params()

        self.f_sky=f_sky
        self.ns=ns
        self.sigma_gamma=sigma_gamma
        self.SN0=sigma_gamma**2/(ns*3600./d2r**2)
        self.do_cov=do_cov
        self.SSV_cov=SSV_cov
        self.tidal_SSV_cov=tidal_SSV_cov

        self.lens_weight=lens_weight

        self.set_ls_params(zs=zs,pzs=pzs,z_bins=z_bins,zl=zl,n_zl=n_zl,log_zl=log_zl)
        self.set_window_params(f_sky=f_sky)
        self.DC=None #this should be cosmology depdendent. set to none before when varying cosmology

    def set_ls_params(self,zs=None,pzs=None,z_bins=None,zl=None,n_zl=10,log_zl=False):
        zl_min=0
        if z_bins is None:
            zl_max=np.amax(zs.values())
        else:
            zl_max=np.amax(zs)

        if zl is None:
            if log_zl:#bins for z_lens.
                self.zl=np.logspace(np.log10(max(zl_min,1.e-4)),np.log10(zl_max),n_zl)
            else:
                self.zl=np.linspace(zl_min,zl_max,n_zl)
            self.dzl=np.gradient(self.zl)

        self.zs_bins={}
        if z_bins is None:
            self.ns_bins=len(zs.keys()) #pass zs bins as dicts
            k=zs.keys()
            for i in np.arange(self.ns_bins):
                self.zs_bins[i]={}
                self.zs_bins[i]['z']=np.array(zs[k[i]])
                self.zs_bins[i]['pz']=np.array(pzs[k[i]])
                self.zs_bins[i]['W']=1.
        else:
            self.ns_bins=len(z_bins)-1
            for i in np.arange(self.ns_bins):
                self.zs_bins[i]={}
                if self.lens_weight:
                    self.zs_bins[i]['z']=zs
                    self.zs_bins[i]['W']=1./self.sigma_crit(zl=z_bins[i],zs=zs,
                                                            cosmo_h=self.PS.cosmo_h)
                    self.zs_bins[i]['pz']=pzs*self.zs_bins[i]['W']
                else:
                    xi=zs>z_bins[i]
                    xi*=zs<z_bins[i+1]
                    self.zs_bins[i]['z']=zs[xi]
                    self.zs_bins[i]['pz']=pzs[xi]
                    self.zs_bins[i]['W']=1.
        for i in self.zs_bins.keys():
            self.zs_bins[i]['dz']= np.gradient(self.zs_bins[i]['z']) if len(self.zs_bins[i]['z'])>1 else np.array([1])
            self.zs_bins[i]['pzdz']=self.zs_bins[i]['pz']*self.zs_bins[i]['dz']
            self.zs_bins[i]['Norm']=np.sum(self.zs_bins[i]['pzdz'])
            self.zs_bins[i]['SN']=self.shape_noise_calc(zs1=self.zs_bins[i],zs2=self.zs_bins[i])
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
            #this is cosmology dependent. Need to reset after every cosmo dependent run

    def set_zs_sigc(self,cosmo_h=None):
        #We need to compute these only once in every run 
        # i.e not repeat for every ij combo
        if cosmo_h is None:
            cosmo_h=self.PS.cosmo_h
        rho=self.Rho_crit(cosmo_h=cosmo_h)*cosmo_h.Om0 
            
        for i in np.arange(self.ns_bins):
            self.zs_bins[i]['sig_c']=rho/self.sigma_crit(zl=self.zl,zs=self.zs_bins[i]['z'],
                                                        cosmo_h=cosmo_h)

    def set_window_params(self,f_sky=0):
        self.theta_win=np.sqrt(f_sky*sky_area/np.pi)
        self.Win=self.window_func(theta_win=self.theta_win,f_sky=f_sky) 
        self.Om_W=4*np.pi*f_sky

    def Rho_crit(self,cosmo_h=None):
        #G2=G.to(u.Mpc/u.Msun*u.km**2/u.second**2)
        #rc=3*cosmo_h.H0**2/(8*np.pi*G2)
        rc=cosmo_h.H0**2/(self.G2) #factors of pi etc. absorbed in selg.G2
        rc=rc.to(u.Msun/u.pc**2/u.Mpc)# unit of Msun/pc^2/mpc
        return rc.value

    def sigma_crit(self,zl=[],zs=[],cosmo_h=None):
        ds=cosmo_h.comoving_transverse_distance(zs)
        dl=cosmo_h.comoving_transverse_distance(zl)
        ddls=1.-np.multiply.outer(1./ds,dl)#(ds-dl)/ds
        w=(3./2.)*((cosmo_h.H0/c)**2)*(1+zl)*dl/self.Rho_crit(cosmo_h) 
        sigma_c=1./(ddls*w)
        x=ddls<=0 #zs<zl
        sigma_c[x]=np.inf
        return sigma_c.value
    
    def window_func(self,theta_win=10,f_sky=None):
        l=self.l
        theta_win*=d2r
        l_th=l*theta_win
        W=2*jn(1,l_th)/l_th*4*np.pi*f_sky
        return W

    def shape_noise_calc(self,zs1=None,zs2=None):
        if not np.array_equal(zs1['z'],zs2['z']):
            return 0
        SN=self.SN0*np.sum(zs1['dz']*np.sqrt(zs1['pz']*zs2['pz']))
        SN/=zs1['Norm']*zs2['Norm']
        return SN
        # XXX Make sure pzs are properly normalized

#return c/H*P(k*chi)/chi**2
    def cl_z(self,z=None,pk_params=None,cosmo_h=None,
                    cosmo_params=None,pk_func=None):
        if cosmo_h is None:
            cosmo_h=self.PS.cosmo_h
        
        if pk_func is None:
            pk_func=self.PS.camb_pk_too_many_z

        l=self.l

        if z is None:
            z=self.zl

        nz=len(z)
        nl=len(l)

        pk,kh=pk_func(z=z,pk_params=pk_params,cosmo_params=cosmo_params)
        cls=np.zeros((nz,nl),dtype='float32')#*u.Mpc#**2

        Rls=None #pk response functions
        RKls=None
        cls_lin=None #cls from linear power spectra, to compute \delta_window for SSV
        if self.do_cov and self.SSV_cov: #things needed to compute SSV cov
            R=self.PS.R1(k=kh,pk=pk,axis=1)
            Rk=self.PS.R_K(k=kh,pk=pk,axis=1)
            Rls=np.zeros((nz,nl),dtype='float32')
            RKls=np.zeros((nz,nl),dtype='float32')

            pk_params2=self.PS.pk_params.copy()
            pk_params2['non_linear']=0
            pk_lin,kh_lin=pk_func(z=z,pk_params=pk_params2)
            cls_lin=np.zeros((nz,nl),dtype='float32')#*u.Mpc#**2

        cH=c/(cosmo_h.efunc(self.zl)*cosmo_h.H0)
        cH=cH.value

        def k_to_l(l,lz,f_k): #take func from k to l space
            fk_int=interp1d(lz,f_k,bounds_error=False,fill_value=0)
            return fk_int(l)

        for i in np.arange(nz):
            DC_i=cosmo_h.comoving_transverse_distance(z[i]).value#because camb k in h/mpc
            lz=kh*DC_i-0.5
            cls[i][:]+=k_to_l(l,lz,pk[i]/DC_i**2)
            if self.SSV_cov:
                Rls[i][:]+=k_to_l(l,lz,R[i]) 
                RKls[i][:]+=k_to_l(l,lz,Rk[i]) 
                cls_lin[i][:]+=k_to_l(l,lz,pk_lin[i]/DC_i**2)
        
        sigma_win=np.dot(self.Win**2*np.gradient(self.l)*self.l,cls_lin.T) if self.SSV_cov else None

        f=(l+0.5)**2/(l*(l+1.)) # cl correction from Kilbinger+ 2017
            #cl*=2./np.pi #comparison with CAMB requires this.
        out={'cls':cls,'l':l,'cH':cH,'f':f}
        if self.SSV_cov:
            out.update({'clsR':cls*Rls,'clsRK':cls*RKls,'sigma_win':sigma_win})
        return out

    def calc_cl(self,clz=None,zs1=None,zs2=None):
        cls=clz['cls']
        f=clz['f']    
        sc=np.dot(zs1['pzdz'],zs1['sig_c']) #cosmo dependent
        sc*=np.dot(zs2['pzdz'],zs2['sig_c'])
        cl=np.dot(sc*self.dzl*clz['cH'],cls)
        # cl_zs_12=np.einsum('ji,ki,il',zs2['sig_c'],zs1['sig_c']*self.dzl*clz['cH'],cls)#integrate over zl..
        # cl=np.dot(zs2['pzdz'],np.dot(zs1['pzdz'],cl_zs_12))

        cl/=zs2['Norm']*zs1['Norm']
        cl/=f**2# cl correction from Kilbinger+ 2017
        return cl
        
    def kappa_cl(self,clz_dict=None, return_clz=False,zs1_indx=-1, zs2_indx=-1,
                pk_func=None,pk_params=None,cosmo_h=None,cosmo_params=None,DC=None):
        if cosmo_h is None:
            cosmo_h=self.PS.cosmo_h
        
        l=self.l
        zs1=self.zs_bins[zs1_indx]#.copy() #we will modify these locally
        zs2=self.zs_bins[zs2_indx]#.copy()
        if zs1['sig_c'] is None or zs2['sig_c'] is None:
            self.set_zs_sigc(cosmo_h=cosmo_h)
        
        if clz_dict is None:
            clz_dict=self.cl_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                                cosmo_params=cosmo_params)
    
        cl=self.calc_cl(clz=clz_dict,zs1=zs1,zs2=zs2)
        out={'l':l,'cl':cl,'clz':clz_dict}
        if self.bin_cl:
            out['binned']=self.bin_kappa_cl(results=out,bin_cl=True)
        # if return_clz:
        #     out['clz']=clz_dict
        return out

    def kappa_cl_cov(self,clz_dict=None,cls=None,SN=None, zs_indx=[]):
        cov={}
        l=self.l 
        cov['G1324']=(cls[:,zs_indx[0],zs_indx[2]]+SN[:,zs_indx[0],zs_indx[2]])
        cov['G1324']*=(cls[:,zs_indx[1],zs_indx[3]]+SN[:,zs_indx[1],zs_indx[3]])

        cov['G1423']=(cls[:,zs_indx[0],zs_indx[3]]+SN[:,zs_indx[0],zs_indx[3]])
        cov['G1423']*=(cls[:,zs_indx[1],zs_indx[2]]+SN[:,zs_indx[1],zs_indx[2]])

        cov['final']=0
        if not self.do_xi:
            cov['G']=np.diag(cov['G1423']+cov['G1324'])# this can be expensive with large l
            cov['G']/=(2.*l+1.)*self.f_sky
            cov['final']=cov['G']
        if self.SSV_cov:
            clz=clz_dict['cls']
            zs1=self.zs_bins[zs_indx[0]]
            zs2=self.zs_bins[zs_indx[1]]
            zs3=self.zs_bins[zs_indx[2]]
            zs4=self.zs_bins[zs_indx[3]]
            sigma_win=clz['sigma_win']
            # sig_cL=np.einsum('ji,ki->i',(p_zs2*dzs2)[:,None]*sigma_c2**2,
            #                 (p_zs1*dzs1)[:,None]*sigma_c1**2*dzl*cH)
            # sig_cL/=(np.sum(p_zs2*dzs2)*np.sum(p_zs1*dzs1))
            sig_cL=np.dot(zs1['pzdz'],zs1['sig_c'])
            sig_cL*=np.dot(zs2['pzdz'],zs2['sig_c'])
            sig_cL*=np.dot(zs3['pzdz'],zs3['sig_c'])
            sig_cL*=np.dot(zs4['pzdz'],zs4['sig_c'])

            sig_cL*=self.dzl*clz['cH']
            sig_cL/=self.Om_W**2
            sig_cL*=sigma_win

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
            cov['binned']=self.bin_kappa_cl(results=cov,bin_cov=True)
        return cov

    def kappa_cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None,
                        return_clz=False,clz_dict=None):
        nbins=self.ns_bins
        l=self.l 
        
        if not self.do_xi: #this should already be set in xi function
            self.set_zs_sigc(cosmo_h=cosmo_h) 

        cl=np.zeros((len(l),nbins,nbins))
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
        out={'l':out['l'],'cl':cl,'SN':SN,'cov':cov}
        if return_clz:
            out['clz_dict']=clz_dict
        if not self.do_xi:
            self.reset_zs()
        return out

    def bin_kappa_cl(self,results=None,bin_cl=False,bin_cov=False):
        results_b={}
        if bin_cl:
            results_b['cl']=self.binning.bin_1d(r=self.l,xi=results['cl'],
                                        r_bins=self.l_bins,r_dim=2,bin_utils=self.cl_bin_utils)
        cov_b=None
        if bin_cov:
            cov_b={}
            keys=['G','final']
            if self.SSV_cov:
                keys=np.append(keys,'SSC_dd')
                if self.tidal_SSV_cov:
                    keys=np.append(keys,['SSC_kk','SSC_dk','SSC_kd'])
            for k in keys:
                cov_b[k]=self.binning.bin_2d(r=l,cov=results[k],r_bins=self.l_bins,r_dim=2
                                        ,bin_utils=self.cl_bin_utils)
        results_b['cov']=cov_b
        return results_b
    
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

    nbins=5
    zs_bins=np.linspace(0.1,2,nbins+1)
    zl_bins=np.linspace(0.5,1.5,nbins+1)
    lmax_cl=2000
    lmin_cl=2
    l=np.arange(lmin_cl,lmax_cl)
    l_bins=np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),20))
    do_cov=True
    bin_cl=False
    bin_xi=True
    theta_bins=np.logspace(np.log10(1./60),1,20)

    cProfile.run('kappa_fn = Kappa(zs=z,pzs=pzs,l=l,z_bins=zs_bins,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,bin_xi=bin_xi,theta_bins=theta_bins,do_xi=True)', 'output_stats')
                                            # globals(), locals()) #runctx
    p = pstats.Stats('output_stats')
    p.sort_stats('tottime').print_stats(2)

    cProfile.run('clS=kappa_fn.kappa_xi_tomo()','output_stats2')
    p2 = pstats.Stats('output_stats2')
    p2.sort_stats('tottime').print_stats(10)
    
    cProfile.run('clS=kappa_fn.stack_dat(clS)','output_stats3')
    p2 = pstats.Stats('output_stats3')
    p2.sort_stats('tottime').print_stats(10)

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
