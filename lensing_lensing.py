import os,sys
import dask
from dask import delayed
from power_spectra import *
from angular_power_spectra import *
from lsst_utils import *
from hankel_transform import *
from binning import *
from cov_utils import *
from lensing_utils import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d

d2r=np.pi/180.
c=c.to(u.km/u.second)

class Kappa():
    def __init__(self,silence_camb=False,l=np.arange(2,2001),HT=None,Ang_PS=None,
                lensing_utils=None,cov_utils=None,
                power_spectra_kwargs={},HT_kwargs=None,zs_bins=None,cross_PS=True,
                z_PS=None,nz_PS=100,log_z_PS=True,z_PS_max=None,
                do_cov=False,SSV_cov=False,tidal_SSV_cov=False,
                sigma_gamma=0.3,f_sky=0.3,l_bins=None,bin_cl=False,
                stack_data=False,bin_xi=False,do_xi=False,theta_bins=None,
                tracer='kappa'):

        self.cross_PS=cross_PS
        self.lensing_utils=lensing_utils
        self.do_cov=do_cov
        self.SSV_cov=SSV_cov
        self.tidal_SSV_cov=tidal_SSV_cov

        if lensing_utils is None:
            self.lensing_utils=Lensing_utils(sigma_gamma=sigma_gamma,zs_bins=zs_bins,
                                            )

        # self.set_lens_bins(zl=zl,n_zl=n_zl,log_zl=log_zl,zl_max=zl_max)
        self.l=l
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
        if do_xi:
            # self.l=np.sort(np.unique(np.hstack((self.HT.k[i] for i in self.j_nus))))
            self.l=np.hstack((self.HT.k[i] for i in self.j_nus))
            self.l_cut_jnu={}
            for j_nu in self.j_nus:
                self.l_cut_jnu[j_nu]=np.isin(self.l,(self.HT.k[j_nu]))

        self.cov_utils=cov_utils
        if cov_utils is None:
            self.cov_utils=Covariance_utils(f_sky=f_sky,l=self.l)

        self.Ang_PS=Ang_PS
        if Ang_PS is None:
            self.Ang_PS=Angular_power_spectra(silence_camb=silence_camb,
                                SSV_cov=SSV_cov,l=self.l,
                                power_spectra_kwargs=power_spectra_kwargs,
                                cov_utils=self.cov_utils,
                                z_PS=z_PS,nz_PS=nz_PS,log_z_PS=log_z_PS,
                                z_PS_max=z_PS_max)
                        #FIXME: Need a dict for these args

        self.zs_bins=self.lensing_utils.zs_bins
        self.ns_bins=self.zs_bins['n_bins']
        self.l_bins=l_bins
        self.stack_data=stack_data
        self.theta_bins=theta_bins
        self.do_xi=do_xi
        self.bin_utils=None
        self.tracer=tracer

        self.bin_cl=bin_cl
        self.bin_xi=bin_xi
        self.set_bin_params()
        self.cov_indxs=[]
        if self.cross_PS:
            self.corr_indxs=[j for j in itertools.combinations_with_replacement(np.arange(self.zs_bins  ['n_bins']),2)]
        else:
            self.corr_indxs=[(i,i) for i in np.arange(self.zs_bins['n_bins'])]
            if self.do_cov:
                self.cov_indxs=[j for j in itertools.combinations(np.arange(self.zs_bins['n_bins']),2)]
            self.lensing_utils.set_shape_noise(cross_PS=self.cross_PS)
            #need cross terms when doing covariance


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

    def calc_lens_lens_cl(self,zs1=None,zs2=None):
        """
            Compute the angular power spectra, Cl between two source bins
            zs1, zs2: Source bins. Dicts containing information about the source bins
        """
        clz=self.Ang_PS.clz
        cls=clz['cls']
        f=self.Ang_PS.cl_f
        sc=zs1['sig_c_int']*zs2['sig_c_int']

        cl=np.dot(sc*clz['dchi'],cls)

        cl/=self.Ang_PS.cl_f**2# cl correction from Kilbinger+ 2017
        return cl

#     @jit
    def kappa_cl(self,zs1_indx=-1, zs2_indx=-1,
                pk_func=None,pk_params=None,cosmo_h=None,cosmo_params=None):
        """
            Wrapper for calc_lens_lens_cl. Checks to make sure quantities such as power spectra and cosmology
            are available otherwise sets them to some default values.
            zs1_indx, zs2_indx: Indices of the source bins to be correlated.
            Others are arguments to be passed to power spectra function if it needs to be computed
        """
        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h

        # print('kappa_cl indx: ',zs1_indx,zs2_indx)

        l=self.l
        zs1=self.zs_bins[zs1_indx]#.copy() #we will modify these locally
        zs2=self.zs_bins[zs2_indx]#.copy()
        if zs1['sig_c'] is None or zs2['sig_c'] is None:
            self.lensing_utils.set_zs_sigc(cosmo_h=cosmo_h,zl=self.Ang_PS.z)

        if self.Ang_PS.clz is None:
            self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                                cosmo_params=cosmo_params)

        cl=self.calc_lens_lens_cl(zs1=zs1,zs2=zs2)

        return cl

#     @jit#(nopython=True)
    def kappa_cl_cov(self,cls=None, zs_indx=[]):
        """
            Computes the covariance between any two tomographic power spectra.
            cls: tomographic cls already computed before calling this function
            zs_indx: 4-d array, noting the indices of the source bins involved
            in the tomographic cls for which covariance is computed.
            For ex. covariance between 12, 56 tomographic cross correlations
                    involve 1,2,5,6 source bins
        """
        cov={}
        cov['final']=None
        l=self.l
        cov['G'],cov['G1324'],cov['G1423']=self.cov_utils.gaussian_cov_auto(cls,
                                        self.lensing_utils.SN,zs_indx,self.do_xi)

        cov['final']=cov['G']


        if self.SSV_cov:
            clz=self.Ang_PS.clz
            zs1=self.zs_bins[zs_indx[0]]
            zs2=self.zs_bins[zs_indx[1]]
            zs3=self.zs_bins[zs_indx[2]]
            zs4=self.zs_bins[zs_indx[3]]
            sigma_win=self.cov_utils.sigma_win

            sig_cL=zs1['sig_c_int']*zs2['sig_c_int']*zs3['sig_c_int']*zs4['sig_c_int']

            sig_cL*=self.Ang_PS.clz['dchi']
            sig_cL*=sigma_win

            if self.do_xi:
                cov['sig_cL']=sig_cL
                return cov

            clr1=self.Ang_PS.clz['clsR']
            if self.tidal_SSV_cov:
                clr1+=self.Ang_PS.clz['clsRK']/6.

            # cov['SSC_dd']=np.dot((clr1).T*sig_cL,clr1)
            cov['SSC']=np.dot((clr1).T*sig_cL,clr1)
            cov['final']+=cov['SSC']

            # if self.tidal_SSV_cov:
            #     #sig_cL will be divided by some factors to account for different sigma_win
            #     clrk=self.Ang_PS.clz['clsRK']
            #
            #     cov['SSC_kk']=np.dot((clrk).T*sig_cL/36.,clrk)
            #     cov['SSC_dk']=np.dot((clr1).T*sig_cL/6.,clrk)
            #
            #     cov['final']+=cov['SSC_kk']+cov['SSC_dk']*2. #+cov['SSC_kd']
        if self.bin_cl:
            for k in cov.keys():
                cl_none,cov[k]=self.bin_kappa_cl(cov=cov[k],bin_cov=True)
        return cov

    def bin_kappa_cl(self,cl=None,cov=None,bin_cl=False,bin_cov=False):
        """
            bins the tomographic power spectra
            results: Either cl or covariance
            bin_cl: if true, then results has cl to be binned
            bin_cov: if true, then results has cov to be binned
            Both bin_cl and bin_cov can be true simulatenously.
        """
        # results_b={}
        cl_b=None
        cov_b=None
        if bin_cl:
            cl_b=self.binning.bin_1d(r=self.l,xi=cl,
                                        r_bins=self.l_bins,r_dim=2,bin_utils=self.cl_bin_utils)
        if bin_cov:
             cov_b=self.binning.bin_2d(r=self.l,cov=cov,r_bins=self.l_bins,r_dim=2
                                            ,bin_utils=self.cl_bin_utils)
        return cl_b,cov_b

    def combine_cl_tomo(self,cl_compute_dict={}):

        cl_b={}
        for (i,j) in self.corr_indxs+self.cov_indxs:
            clij=cl_compute_dict[(i,j)]
            cl_b[(i,j)],cov_none=self.bin_kappa_cl(cl=clij,cov=None,bin_cl=True,
                                    bin_cov=False)
        return cl_b


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

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h

        self.lensing_utils.set_zs_sigc(cosmo_h=cosmo_h,zl=self.Ang_PS.z)
        if self.Ang_PS.clz is None:
            self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                                cosmo_params=cosmo_params)

        out={}
        cl={}
        cov={}
        cl_b=None
        for (i,j) in self.corr_indxs+self.cov_indxs:
            # out[(i,j)]
            cl[(i,j)]=delayed(self.kappa_cl)(zs1_indx=i,zs2_indx=j,cosmo_h=cosmo_h,
                                    cosmo_params=cosmo_params,pk_params=pk_params,
                                    pk_func=pk_func)
            cl[(j,i)]=cl[(i,j)]
        if self.bin_cl:
            cl_b=delayed(self.combine_cl_tomo)(cl)
        else:
            cl_b=cl

        if self.do_cov:
            for i in self.corr_indxs:
                for j in self.corr_indxs:
                    indx=i+j
                    cov[indx]=delayed(self.kappa_cl_cov)(cls=cl, zs_indx=indx)

        out_stack=delayed(self.stack_dat)({'cov':cov,'cl':cl_b})
        return {'stack':out_stack,'cl_b':cl_b,'cov':cov,'cl':cl}

    def xi_cov(self,cov_cl={},j_nu=None,j_nu2=None,clr=None,clrk=None):
        """
            Computes covariance of xi, by performing 2-D hankel transform on covariance of Cl.
            In current implementation of hankel transform works only for j_nu=j_nu2. So no cross covariance between xi+ and xi-.
        """
        #FIXME: Implement the cross covariance
        cov_xi={}

        Norm= self.cov_utils.Om_W
        th0,cov_xi['G']=self.HT.projected_covariance(k_pk=self.l,j_nu=j_nu,
                                                     pk_cov=cov_cl['G1423']+cov_cl['G1324'])

        cov_xi['G']=self.binning.bin_2d(r=th0,cov=cov_xi['G'],r_bins=self.theta_bins,
                                                r_dim=2,bin_utils=self.xi_bin_utils[j_nu])
        #binning is cheap

        cov_xi['G']/=Norm
        cov_xi['final']=cov_xi['G']
        if self.SSV_cov:
            sig_cL=cov_cl['sig_cL']

            #tidal term is added to clr in the calling function
            cov_SSC=np.einsum('rk,kz,zl,sl->rs',self.HT.J[j_nu]/self.HT.J_nu1[j_nu]**2,
                            (clr).T*sig_cL,clr,self.HT.J[j_nu],optimize=True)

            cov_SSC*=(2.*self.HT.kmax**2/self.HT.zeros[j_nu][-1]**2)/(2*np.pi)/Norm
            # cov_SSC/=Norm
            cov_xi['SSC']=self.binning.bin_2d(r=th0,cov=cov_SSC,r_bins=self.theta_bins,
                                                    r_dim=2,bin_utils=self.xi_bin_utils[j_nu])
            cov_xi['final']+=cov_xi['SSC']

        return cov_xi

    def get_xi(self,cl=[],j_nu=0):
        th,xi=self.HT.projected_correlation(k_pk=self.l,j_nu=j_nu,pk=cl)
        xi_b=self.binning.bin_1d(r=th/d2r,xi=xi,
                                    r_bins=self.theta_bins,r_dim=2,
                                    bin_utils=self.xi_bin_utils[j_nu])
        return xi_b

    def combine_xi_tomo(self,cl=[],j_nu=0):
        xi={}
        for (i,j) in self.corr_indxs:
            xi[(i,j)]=self.get_xi(cl=cl[(i,j)],j_nu=j_nu)
            # xi[:,j,i]=xi[:,i,j]
        return xi

    def xi_tomo_cov(self,cov_cl=[],j_nu1=0,j_nu2=0):
        cov_xi={}
        for i in self.corr_indxs: #np.arange(ni):
            for j in self.corr_indxs:#np.arange(i,ni):
                indx=i+j #indxs[i]+indxs[j]
                cov_cl_i=cov_cl[indx] #delayed(self.kappa_cl_cov)(cls=cl,zs_indx=indx)
                                        #need large l range for xi which leads to memory issues
                cov_xi[indx]=delayed(self.xi_cov)(cov=cov_cl_i,j_nu=j_nu1,j_nu2=j_nu2)
        return cov_xi

    def kappa_xi_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None):
        """
            Computed tomographic angular correlation functions. First calls the tomographic
            power spectra and covariance and then does the hankel transform and  binning.
        """
        """
            For hankel transform is done on l-theta grid, which is based on j_nu. So grid is
            different for xi+ and xi-.
            In the init function, we combined the ell arrays for all j_nu. This is not a problem
            except for the case of SSV, where we will use l_cut to only select the relevant values
        """

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h

        nbins=self.ns_bins

        #Donot use delayed here. Leads to error/repeated calculations
        cls_tomo_nu=self.kappa_cl_tomo(cosmo_h=cosmo_h,cosmo_params=cosmo_params,
                                                 pk_params=pk_params,pk_func=pk_func)

        cl=cls_tomo_nu['cl']
        cov_xi={}
        xi={}
        out={}
        for j_nu in [0]: #self.j_nus:
            # l_nu=self.HT.k[j_nu]
            # self.l=l_nu
            # self.Ang_PS.l=l_nu
            # self.cov_utils.l=l_nu
            xi[j_nu]={}
            for (i,j) in self.corr_indxs:
                xi[j_nu][(i,j)]=delayed(self.get_xi)(cl=cl[(i,j)]#.compute()
                                                    ,j_nu=j_nu)
            if self.do_cov:
                l_cut=self.l_cut_jnu[j_nu]
                cov_cl=cls_tomo_nu['cov']#.compute()
                j_nu2=j_nu

                cov_xi[j_nu]={}

                clr=None
                clrk=None
                # SSC_z_temp={}
                if self.SSV_cov:#this can be slower
                    clr=self.Ang_PS.clz['clsR'][:,l_cut]
                    if self.tidal_SSV_cov:
                        clr+=self.Ang_PS.clz['clsRK'][:,l_cut]/6
                for i in self.corr_indxs: #np.arange(ni):
                    for j in self.corr_indxs:#np.arange(i,ni):
                        indx=i+j #indxs[i]+indxs[j]
                        cov_xi[j_nu][indx]=delayed(self.xi_cov)(cov_cl=cov_cl[indx]#.compute()
                                                        ,j_nu=j_nu,j_nu2=j_nu2,clr=clr)
        out['stack']=delayed(self.stack_dat)({'cov':cov_xi,'xi':xi})
        out['xi']=xi
        out['cov']=cov_xi

        return out


    def stack_dat(self,dat):
        """
            outputs from tomographic caluclations are dictionaries.
            This fucntion stacks them such that the cl or xi is a long
            1-d array and the covariance is N X N array.
            dat: output from tomographic calculations.
            XXX: reason that outputs tomographic bins are distionaries is that
            it make is easier to
            handle things such as binning, hankel transforms etc. We will keep this structure for now.
        """
        nbins=self.ns_bins
        nD=len(self.corr_indxs) #np.int64(nbins*(nbins-1.)/2.+nbins)


        if self.do_xi:
            est='xi'
            D_keys=list(dat[est].keys())
            nX=len(dat[est][D_keys[0]][self.corr_indxs[0]])
            nD2=len(D_keys)
        else:
            est='cl'
            nX=len(dat[est][self.corr_indxs[0]])
            nD2=1

        D_final=np.zeros(nD*nX*nD2)
        cov_final=np.zeros((nD*nX*nD2,nD*nX*nD2))
        ij=0
        for iD2 in np.arange(nD2):
            dat2=dat[est]
            if self.do_xi:
                dat2=dat[est][D_keys[iD2]]

            D_final[nD*nX*iD2:nD*nX*(iD2+1)]=np.hstack((dat2[(i,j)] for (i,j) in self.corr_indxs))

            if not self.do_cov:
                cov_final=None
                continue

            dat2=dat['cov']
            if self.do_xi:
                dat2=dat['cov'][D_keys[iD2]]

            i_indx=0
            indx0=iD2*nX*nD
            for i in np.arange(len(self.corr_indxs)):
                for j in np.arange(i,len(self.corr_indxs)):
                    indx=self.corr_indxs[i]+self.corr_indxs[j]
                    cov_final[ indx0+i*nX : indx0+(i+1)*nX , indx0+j*nX : indx0+(j+1)*nX] = dat2[indx]['final']
                    cov_final[ indx0+j*nX : indx0+(j+1)*nX , indx0+i*nX : indx0+(i+1)*nX] = dat2[indx]['final']
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


    zs_bin1=source_tomo_bins(zp=[1],p_zp=np.array([1]),ns=26)

    lmax_cl=2000
    lmin_cl=2
    l_step=3 #choose odd number

#     l=np.arange(lmin_cl,lmax_cl,step=l_step) #use fewer ell if lmax_cl is too large
    l0=np.arange(lmin_cl,lmax_cl)

    lmin_clB=lmin_cl+10
    lmax_clB=lmax_cl-10
    Nl_bins=40
    l_bins=np.int64(np.logspace(np.log10(lmin_clB),np.log10(lmax_clB),Nl_bins))
    l=np.unique(np.int64(np.logspace(np.log10(lmin_cl),np.log10(lmax_cl),Nl_bins*30)))

    bin_xi=True
    theta_bins=np.logspace(np.log10(1./60),1,20)


    do_cov=True
    bin_cl=True
    SSV_cov=True
    tidal_SSV_cov=True
    stack_data=True

    # kappa0=Kappa(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0,
    #            stack_data=stack_data,SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,)
    #
    # cl_G=kappa0.kappa_cl_tomo() #make the compute graph
    # cProfile.run("cl0=cl_G['stack'].compute()",'output_stats_1bin')
    # cl=cl0['cl']
    # cov=cl0['cov']
    #
    # p = pstats.Stats('output_stats_1bin')
    # p.sort_stats('tottime').print_stats(10)

##############################################################
    do_xi=True
    bin_cl=not do_xi
    zmin=0.3
    zmax=2
    z=np.linspace(0,5,200)
    pzs=lsst_pz_source(z=z)
    x=z<zmax
    x*=z>zmin
    z=z[x]
    pzs=pzs[x]

    ns0=26#+np.inf # Total (cumulative) number density of source galaxies, arcmin^-2.. setting to inf turns off shape noise
    nbins=5 #number of tomographic bins
    z_sigma=0.01
    zs_bins=source_tomo_bins(zp=z,p_zp=pzs,ns=ns0,nz_bins=nbins,
                             ztrue_func=ztrue_given_pz_Gaussian,
                             zp_bias=np.zeros_like(z),
                            zp_sigma=z_sigma*np.ones_like(z))


    kappaS = Kappa(zs_bins=zs_bins,l=l,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,
                    stack_data=stack_data,SSV_cov=SSV_cov,
                    tidal_SSV_cov=tidal_SSV_cov,do_xi=do_xi,bin_xi=bin_xi,
                    theta_bins=theta_bins)#ns=np.inf)

    if not do_xi:
        clSG=kappaS.kappa_cl_tomo()#make the compute graph
        cProfile.run("cl0=clSG['stack'].compute(num_workers=4)",'output_stats_3bins')
        cl=cl0['cl']
        cov=cl0['cov']
    else:
        xiSG=kappaS.kappa_xi_tomo()#make the compute graph
        cProfile.run("xi0=xiSG['stack'].compute(num_workers=4)",'output_stats_3bins')


    p = pstats.Stats('output_stats_3bins')
    p.sort_stats('tottime').print_stats(10)
