import os,sys
import dask
from dask import delayed
from power_spectra import *
from angular_power_spectra import *
from lsst_utils import *
from hankel_transform import *
from wigner_transform import *
from binning import *
from cov_utils import *
from lensing_utils import *
from galaxy_utils import *
from astropy.constants import c,G
from astropy import units as u
import numpy as np
from scipy.interpolate import interp1d
import warnings,logging

d2r=np.pi/180.
c=c.to(u.km/u.second)

#corrs=['gg','gl_p','gl_k','ll_p','ll_m','ll_k','ll_kp']

class cov_3X2():
    def __init__(self,silence_camb=False,l=np.arange(2,2001),HT=None,Ang_PS=None,
                cov_utils=None,logger=None,lensing_utils=None,galaxy_utils=None,
                zs_bins=None,zg_bins=None,power_spectra_kwargs={},HT_kwargs=None,
                z_PS=None,nz_PS=100,log_z_PS=True,#z_PS_max=None,
                do_cov=False,SSV_cov=False,tidal_SSV_cov=False,do_sample_variance=True,
                use_window=True,
                sigma_gamma=0.3,f_sky=0.3,l_bins=None,bin_cl=False,
                stack_data=False,bin_xi=False,do_xi=False,theta_bins=None,
                corrs=[('shear','shear')]):
        self.logger=logger
        self.cov_SSC_nobin={}
        if logger is None:
            self.logger=logging.getLogger()
            self.logger.setLevel(level=logging.DEBUG)
            logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                                level=logging.DEBUG, datefmt='%I:%M:%S')
            # format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            # ch = logging.StreamHandler(sys.stdout)
            # ch.setFormatter(format)
            # self.logger.addHandler(ch)

        self.do_cov=do_cov
        self.SSV_cov=SSV_cov
        self.tidal_SSV_cov=tidal_SSV_cov
        self.l=l
        self.do_xi=do_xi
        self.corrs=corrs
        self.l_cut_jnu=None
        z_PS_max=zs_bins['zmax']
        if do_xi:
            self.set_HT(HT=HT,HT_kwargs=HT_kwargs)

        self.lensing_utils=lensing_utils
        if lensing_utils is None:
            self.lensing_utils=Lensing_utils(sigma_gamma=sigma_gamma,zs_bins=zs_bins,
                                            logger=self.logger)

        self.cov_utils=cov_utils
        if cov_utils is None:
            self.cov_utils=Covariance_utils(f_sky=f_sky,l=self.l,logger=self.logger,
                                            l_cut_jnu=self.l_cut_jnu,
                                            do_sample_variance=do_sample_variance,use_window=use_window)

        self.Ang_PS=Ang_PS
        if Ang_PS is None:
            self.Ang_PS=Angular_power_spectra(silence_camb=silence_camb,
                                SSV_cov=SSV_cov,l=self.l,logger=self.logger,
                                power_spectra_kwargs=power_spectra_kwargs,
                                cov_utils=self.cov_utils,
                                z_PS=z_PS,nz_PS=nz_PS,log_z_PS=log_z_PS,
                                z_PS_max=z_PS_max)
                        #FIXME: Need a dict for these args

        self.galaxy_utils=galaxy_utils
        if galaxy_utils is None:
            self.galaxy_utils=Galaxy_utils(zg_bins=zg_bins,logger=self.logger,l=self.l,
                                            z_th=self.Ang_PS.z)
        self.z_bins={}
        self.z_bins['shear']=self.lensing_utils.zs_bins
        #self.z_bins['kappa']=self.lensing_utils.zk_bins
        self.z_bins['galaxy']=self.galaxy_utils.zg_bins
        self.l_bins=l_bins
        self.stack_data=stack_data
        self.theta_bins=theta_bins
        self.bin_utils=None

        self.bin_cl=bin_cl
        self.bin_xi=bin_xi
        self.set_bin_params()
        self.cov_indxs=[]
        self.corr_indxs={}
        self.m1_m2s={}

        n_s_bins=self.z_bins['shear']['n_bins']
        n_g_bins=0
        if self.galaxy_utils.zg_bins is not None:
            n_g_bins=self.z_bins['galaxy']['n_bins']

        self.corr_indxs[('shear','shear')]=[j for j in itertools.combinations_with_replacement(
                                                    np.arange(n_s_bins),2)]

        self.corr_indxs[('galaxy','galaxy')]=[(i,i) for i in np.arange(n_g_bins)]

        if self.do_cov: #gg cross terms are needed for covariance
            self.corr_indxs[('galaxy','galaxy')]=[j for j in itertools.combinations_with_replacement(np.arange(n_g_bins),2)]

        self.corr_indxs[('galaxy','shear')]=[ k for l in [[(i,j) for i in np.arange(
                                        n_g_bins)] for j in np.arange(n_s_bins)] for k in l]

        self.corr_indxs[('shear','galaxy')]=[ k for l in [[(i,j) for i in np.arange(
                                        n_s_bins)] for j in np.arange(n_g_bins)] for k in l]

        self.m1_m2s[('shear','shear')]=[(2,2),(2,-2)]
        self.m1_m2s[('galaxy','shear')]=[(0,2)] #FIXME: check the order in covariance case
        self.m1_m2s[('galaxy','galaxy')]=[(0,0)]

    def set_HT(self,HT=None,HT_kwargs=None):
        self.HT=HT
        if HT is None:
            if HT_kwargs is None:
                th_min=1./60. if theta_bins is None else np.amin(theta_bins)
                th_max=5 if theta_bins is None else np.amax(theta_bins)
                HT_kwargs={'l_min':min(l),'l_max':max(l),
                            'theta_min':th_min*d2r,'theta_max':th_max*d2r,
                            'n_zeros':2000,'prune_theta':2,'m1_m2':[(0,0)]}
            HT_kwargs['logger']=self.logger
            self.HT=hankel_transform(**HT_kwargs)

        self.l_cut_jnu={}
        self.m1_m2s=self.HT.m1_m2s
        self.l_cut_jnu['m1_m2s']=self.m1_m2s
        self.m1_m2s=self.HT.m1_m2s
        if self.HT.name=='Hankel':
            self.l=np.hstack((self.HT.l[i] for i in self.m1_m2s))
            for m1_m2 in self.m1_m2s:
                self.l_cut_jnu[m1_m2]=np.isin(self.l,(self.HT.l[m1_m2]))

        if self.HT.name=='Wigner':
            self.l=self.HT.l
            for m1_m2 in self.m1_m2s:
                self.l_cut_jnu[m1_m2]=np.isin(self.l,(self.l))
            #FIXME: This is ugly

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
            for m1_m2 in self.m1_m2s:
                self.xi_bin_utils[m1_m2]=self.binning.bin_utils(r=self.HT.theta[m1_m2]/d2r,
                                                    r_bins=self.theta_bins,
                                                    r_dim=2,mat_dims=[1,2])

    def calc_cl(self,zs1=None,zs2=None):
        """
            Compute the angular power spectra, Cl between two source bins
            zs1, zs2: Source bins. Dicts containing information about the source bins
        """
        clz=self.Ang_PS.clz
        cls=clz['cls']
        f=self.Ang_PS.cl_f
        sc=zs1['kernel_int']*zs2['kernel_int']

        # cl=np.dot(sc*clz['dchi'],cls)
        cl=np.dot(cls.T*sc,clz['dchi'])

        cl/=self.Ang_PS.cl_f**2# cl correction from Kilbinger+ 2017
        # cl*=2./np.pi #FIXME: needed to match camb... but not CCL
        return cl

#     @jit
    def get_cl(self,zs1_indx=-1, zs2_indx=-1,corr=('shear','shear'),
                pk_func=None,pk_params=None,cosmo_h=None,cosmo_params=None):
        """
            Wrapper for calc_lens_lens_cl. Checks to make sure quantities such as power spectra and cosmology
            are available otherwise sets them to some default values.
            zs1_indx, zs2_indx: Indices of the source bins to be correlated.
            Others are arguments to be passed to power spectra function if it needs to be computed
        """
        # if cosmo_h is None:
        #     cosmo_h=self.Ang_PS.PS.cosmo_h

        zs1=self.z_bins[corr[0]][zs1_indx]#.copy() #we will modify these locally
        zs2=self.z_bins[corr[1]][zs2_indx]#.copy()

        # if zs1['kernel'] is None or zs2['kernel'] is None:
        #     self.lensing_utils.set_zs_sigc(cosmo_h=cosmo_h,zl=self.Ang_PS.z)
        #
        # if self.Ang_PS.clz is None:
        #     self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
        #                         cosmo_params=cosmo_params)

        cl=self.calc_cl(zs1=zs1,zs2=zs2)

        return cl

#     @jit#(nopython=True)
    def cl_cov(self,cls=None, zs_indx=[],tracers=[]):
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
                                                            self.SN,tracers,zs_indx,self.do_xi)

        cov['final']=cov['G']

        cov['SSC']=None
        if self.SSV_cov:
            clz=self.Ang_PS.clz
            zs1=self.z_bins[tracers[0]][zs_indx[0]]
            zs2=self.z_bins[tracers[1]][zs_indx[1]]
            zs3=self.z_bins[tracers[2]][zs_indx[2]]
            zs4=self.z_bins[tracers[3]][zs_indx[3]]
            sigma_win=self.cov_utils.sigma_win

            sig_cL=zs1['kernel_int']*zs2['kernel_int']*zs3['kernel_int']*zs4['kernel_int']
            # sig_cL*=zs3['kernel_int']*zs4['kernel_int']

            sig_cL*=self.Ang_PS.clz['dchi']

            if self.do_xi:
                cov['kernel']=sig_cL
                return cov

            sig_cL*=sigma_win

            clr=self.Ang_PS.clz['clsR']
            if self.tidal_SSV_cov:
                clr+=self.Ang_PS.clz['clsRK']/6.

            # cov['SSC_dd']=np.dot((clr1).T*sig_cL,clr1)
            cov['SSC']=np.dot((clr).T*sig_cL,clr)
            cov['final']+=cov['SSC']

        for k in ['G','SSC','final']:#no need to bin G1324 and G1423
            cl_none,cov[k]=self.bin_cl_func(cov=cov[k])
        return cov

    def bin_cl_func(self,cl=None,cov=None):
        """
            bins the tomographic power spectra
            results: Either cl or covariance
            bin_cl: if true, then results has cl to be binned
            bin_cov: if true, then results has cov to be binned
            Both bin_cl and bin_cov can be true simulatenously.
        """
        # results_b={}
        cl_b=cl
        cov_b=cov
        if self.bin_cl:
            if not cl is None:
                cl_b=self.binning.bin_1d(xi=cl,bin_utils=self.cl_bin_utils)
            if not cov is None:
                cov_b=self.binning.bin_2d(r=self.l,cov=cov,r_bins=self.l_bins,r_dim=2
                                            ,bin_utils=self.cl_bin_utils)
        return cl_b,cov_b

    def combine_cl_tomo(self,cl_compute_dict={},corr=None):
        corr2=corr[::-1]
        cl_b={corr:{},corr2:{}}

        for (i,j) in self.corr_indxs[corr]+self.cov_indxs:
            clij=cl_compute_dict[(i,j)]
            cl_b[corr][(i,j)],cov_none=self.bin_cl_func(cl=clij,cov=None)
            cl_b[corr2][(j,i)]=cl_b[corr][(i,j)]
        return cl_b


    def cl_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None,
                corrs=[('shear','shear')],bias_kwargs={},bias_func=None):
        """
         Computes full tomographic power spectra and covariance, including shape noise. output is
         binned also if needed.
         Arguments are for the power spectra  and sigma_crit computation,
         if it needs to be called from here.
         source bins are already set. This function does set the sigma crit for sources.
        """

        l=self.l

        #tracers=[j for i in corrs for j in i]
        tracers=np.unique([j for i in corrs for j in i])

        corrs2=corrs
        if self.do_cov:#make sure we compute cl for all cross corrs necessary for covariance
                        #FIXME: If corrs are gg and ll only, this will lead to uncessary gl. This
                        #        is an unlikely use case though
            corrs2=[]
            for i in np.arange(len(tracers)):
                for j in np.arange(i,len(tracers)):
                    corrs2+=[(tracers[i],tracers[j])]

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h

        self.SN={}
        # self.SN[('galaxy','shear')]={}
        if 'shear' in tracers:
            self.lensing_utils.set_zs_sigc(cosmo_h=cosmo_h,zl=self.Ang_PS.z)
            self.SN[('shear','shear')]=self.lensing_utils.SN
        if 'galaxy' in tracers:
            if bias_func is None:
                bias_func=self.galaxy_utils.linear_bias_powerlaw
                bias_kwargs={'b1':1,'b2':1}
            self.galaxy_utils.set_zg_bias(cosmo_h=cosmo_h,zl=self.Ang_PS.z,bias_func=bias_func,
                                          bias_kwargs=bias_kwargs)
            self.SN[('galaxy','galaxy')]=self.galaxy_utils.SN

#         if self.Ang_PS.clz is None:
        self.Ang_PS.angular_power_z(cosmo_h=cosmo_h,pk_params=pk_params,pk_func=pk_func,
                                cosmo_params=cosmo_params)

        out={}
        cl={}
        cov={}
        cl_b={}
        for corr in corrs2:
            corr2=corr[::-1]
            cl[corr]={}
            cl[corr2]={}
            corr_indxs=self.corr_indxs[(corr[0],corr[1])]#+self.cov_indxs
            for (i,j) in corr_indxs:
                # out[(i,j)]
                cl[corr][(i,j)]=delayed(self.get_cl)(zs1_indx=i,zs2_indx=j,cosmo_h=cosmo_h,
                                        cosmo_params=cosmo_params,pk_params=pk_params,
                                        pk_func=pk_func,corr=corr)

                cl[corr2][(j,i)]=cl[corr][(i,j)]#useful in gaussian covariance calculation.
            cl_b=delayed(self.combine_cl_tomo)(cl[corr],corr=corr)

        if self.do_cov:
            for corr1 in corrs:
                corr1_indxs=self.corr_indxs[(corr1[0],corr1[1])]
                for corr2 in corrs:
                    cov[corr1+corr2]={}
                    corr2_indxs=self.corr_indxs[(corr2[0],corr2[1])]
                    for i in np.arange(len(corr1_indxs)):
                        start_j=0
                        if corr1==corr2:
                            start_j=i
                        for j in np.arange(start_j,len(corr2_indxs)):
                            indx=corr1_indxs[i]+corr2_indxs[j]
                            cov[corr1+corr2][indx]=delayed(self.cl_cov)(cls=cl, zs_indx=indx,
                                                                        tracers=corr1+corr2)

        out_stack=delayed(self.stack_dat)({'cov':cov,'cl':cl_b},corrs=corrs)
        return {'stack':out_stack,'cl_b':cl_b,'cov':cov,'cl':cl}

    def xi_cov(self,cov_cl={},m1_m2=None,m1_m2_cross=None,clr=None,clrk=None):
        """
            Computes covariance of xi, by performing 2-D hankel transform on covariance of Cl.
            In current implementation of hankel transform works only for m1_m2=m1_m2_cross.
            So no cross covariance between xi+ and xi-.
        """
        #FIXME: Implement the cross covariance
        if m1_m2_cross is None:
            m1_m2_cross=m1_m2
        cov_xi={}

        if self.HT.name=='Hankel' and m1_m2!=m1_m2_cross:
            n=len(self.theta_bins)-1
            cov_xi['final']=np.zeros((n,n))
            return cov_xi

        Norm= self.cov_utils.Om_W
        th0,cov_xi['G']=self.HT.projected_covariance(l_cl=self.l,m1_m2=m1_m2,m1_m2_cross=m1_m2_cross,
                                                     cl_cov=cov_cl['G1423']+cov_cl['G1324'])

        cov_xi['G']=self.binning.bin_2d(r=th0/d2r,cov=cov_xi['G'],r_bins=self.theta_bins,
                                                r_dim=2,bin_utils=self.xi_bin_utils[m1_m2])
        #binning is cheap

        cov_xi['G']/=Norm
        cov_xi['final']=cov_xi['G']
        if self.SSV_cov:
            sig_cL=cov_cl['kernel']*self.cov_utils.sigma_win[m1_m2]

            #tidal term is added to clr in the calling function
            if self.HT.name=='Hankel':
                cov_SSC=np.einsum('rk,kz,zl,sl->rs',self.HT.J[m1_m2]/self.HT.J_nu1[m1_m2]**2,
                                    (clr).T*sig_cL,clr,self.HT.J[m1_m2_cross],optimize=True)

                self.cov_SSC_nobin[m1_m2]=sig_cL
                cov_SSC*=(2.*self.HT.l_max[m1_m2]**2/self.HT.zeros[m1_m2][-1]**2)/(2*np.pi)/Norm

            elif self.HT.name=='Wigner':
                cov_SSC=np.einsum('rk,kz,zl,sl->rs',self.HT.wig_d[m1_m2]*np.sqrt(self.HT.norm)/Norm,
                                (clr).T*sig_cL,clr,np.sqrt(self.HT.wig_d[m1_m2_cross]),optimize=True)
                                #FIXME: This is likely to be broken.

            cov_xi['SSC']=self.binning.bin_2d(r=th0/d2r,cov=cov_SSC,r_bins=self.theta_bins,
                                                    r_dim=2,bin_utils=self.xi_bin_utils[m1_m2])
            cov_xi['final']+=cov_xi['SSC']
            # cov_xi['final']=cov_xi['SSC']+cov_xi['G']

        return cov_xi

    def get_xi(self,cl=[],m1_m2=[]):
        th,xi=self.HT.projected_correlation(l_cl=self.l,m1_m2=m1_m2,cl=cl)
        xi_b=self.binning.bin_1d(xi=xi,bin_utils=self.xi_bin_utils[m1_m2])
        return xi_b

    def xi_tomo(self,cosmo_h=None,cosmo_params=None,pk_params=None,pk_func=None,
                corrs=[('shear','shear')]):
        """
            Computed tomographic angular correlation functions. First calls the tomographic
            power spectra and covariance and then does the hankel transform and  binning.
        """
        """
            For hankel transform is done on l-theta grid, which is based on m1_m2. So grid is
            different for xi+ and xi-.
            In the init function, we combined the ell arrays for all m1_m2. This is not a problem
            except for the case of SSV, where we will use l_cut to only select the relevant values
        """

        if cosmo_h is None:
            cosmo_h=self.Ang_PS.PS.cosmo_h

        #Donot use delayed here. Leads to error/repeated calculations
        cls_tomo_nu=self.cl_tomo(cosmo_h=cosmo_h,cosmo_params=cosmo_params,
                            pk_params=pk_params,pk_func=pk_func,
                            corrs=corrs)

        cl=cls_tomo_nu['cl']
        cov_xi={}
        xi={}
        out={}
        self.clr={}
        # for m1_m2 in self.m1_m2s:
        for corr in corrs:
            m1_m2s=self.m1_m2s[corr]
            xi[corr]={}
            for im in np.arange(len(m1_m2s)):
                m1_m2=m1_m2s[im]
                xi[corr][m1_m2]={}
                for (i,j) in self.corr_indxs[corr]:
                    xi[corr][m1_m2][(i,j)]=delayed(self.get_xi)(cl=cl[corr][(i,j)]#.compute()
                                                        ,m1_m2=m1_m2)
        if self.do_cov:
            for corr1 in corrs:
                for corr2 in corrs:

                    m1_m2s_1=self.m1_m2s[corr1]
                    indxs_1=self.corr_indxs[corr1]
                    m1_m2s_2=self.m1_m2s[corr2]
                    indxs_2=self.corr_indxs[corr2]

                    corr=corr1+corr2
                    cov_xi[corr]={}

                    for im1 in np.arange(len(m1_m2s_1)):
                        m1_m2=m1_m2s_1[im1]
                        l_cut=self.l_cut_jnu[m1_m2]
                        cov_cl=cls_tomo_nu['cov'][corr]#.compute()
                        clr=None
                        if self.SSV_cov:
                            clr=self.Ang_PS.clz['clsR'][:,l_cut]#this is mainly for Hankel transform.
                                                                # Which doesnot work for cross correlations
                                                                # Does not impact Wigner.

                            if self.tidal_SSV_cov:
                                clr+=self.Ang_PS.clz['clsRK'][:,l_cut]/6.

                        start2=0
                        if corr1==corr2:
                            start2=im1
                        for im2 in np.arange(start2,len(m1_m2s_2)):
                            m1_m2_cross=m1_m2s_2[im2]
                            cov_xi[corr][m1_m2+m1_m2_cross]={}

                            for i1 in np.arange(len(indxs_1)):
                                start2=0
                                if corr1==corr2:# and m1_m2==m1_m2_cross:
                                    start2=i1
                                for i2 in np.arange(start2,len(indxs_2)):
                                    indx=indxs_1[i1]+indxs_2[i2]
                                    cov_xi[corr][m1_m2+m1_m2_cross][indx]=delayed(self.xi_cov)(cov_cl=cov_cl[indx]#.compute()
                                                                    ,m1_m2=m1_m2,m1_m2_cross=m1_m2_cross,clr=clr)
        out['stack']=delayed(self.stack_dat)({'cov':cov_xi,'xi':xi},corrs=corrs)
        out['xi']=xi
        out['cov']=cov_xi
        out['cl']=cls_tomo_nu
        return out


    def stack_dat(self,dat,corrs):
        """
            outputs from tomographic caluclations are dictionaries.
            This fucntion stacks them such that the cl or xi is a long
            1-d array and the covariance is N X N array.
            dat: output from tomographic calculations.
            XXX: reason that outputs tomographic bins are distionaries is that
            it make is easier to
            handle things such as binning, hankel transforms etc. We will keep this structure for now.
        """



        if self.do_xi:
            est='xi'
            len_bins=len(self.theta_bins)-1
        else:
            est='cl'
            n_m1_m2=1
            len_bins=len(self.l_bins)-1

        n_bins=0
        for corr in corrs:
            if self.do_xi:
                n_m1_m2=len(self.m1_m2s[corr])
            n_bins+=len(self.corr_indxs[corr])*n_m1_m2 #np.int64(nbins*(nbins-1.)/2.+nbins)
#         print(n_bins,len_bins,n_m1_m2)
        D_final=np.zeros(n_bins*len_bins)

        i=0
        for corr in corrs:
            n_m1_m2=1
            if self.do_xi:
                m1_m2=self.m1_m2s[corr]
                n_m1_m2=len(m1_m2)

            for im in np.arange(n_m1_m2):
                dat_c=dat[est][corr]
                if self.do_xi:
                    dat_c=dat[est][corr][m1_m2[im]]

                for indx in self.corr_indxs[corr]:
                    # print(len_bins,dat_c[indx].shape)
                    D_final[i*len_bins:(i+1)*len_bins]=dat_c[indx]
                    i+=1


        if not self.do_cov:
            out={'cov':None}
            out[est]=D_final
            return out

        cov_final=np.zeros((len(D_final),len(D_final)))-999.#np.int(nD2*(nD2+1)/2)

        indx0_c1=0
        for ic1 in np.arange(len(corrs)):
            corr1=corrs[ic1]
            indxs_1=self.corr_indxs[corr1]
            n_indx1=len(indxs_1)
            # indx0_c1=(ic1)*n_indx1*len_bins

            indx0_c2=indx0_c1
            for ic2 in np.arange(ic1,len(corrs)):
                corr2=corrs[ic2]
                indxs_2=self.corr_indxs[corr2]
                n_indx2=len(indxs_2)
                # indx0_c2=(ic2)*n_indx2*len_bins

                corr=corr1+corr2
                n_m1_m2_1=1
                n_m1_m2_2=1
                if self.do_xi:
                    m1_m2_1=self.m1_m2s[corr1]
                    m1_m2_2=self.m1_m2s[corr2]
                    n_m1_m2_1=len(m1_m2_1)
                    n_m1_m2_2=len(m1_m2_2)

                for im1 in np.arange(n_m1_m2_1):
                    start_m2=0
                    if corr1==corr2:
                        start_m2=im1
                    for im2 in np.arange(start_m2,n_m1_m2_2):
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

                                if self.do_xi:
                                    cov_here=dat['cov'][corr][m1_m2_1[im1]+m1_m2_2[im2]][indx]['final']
                                else:
                                    cov_here=dat['cov'][corr][indx]['final']

                                # if im1==im2:
                                i=indx0_c1+indx0_1+indx0_m1
                                j=indx0_c2+indx0_2+indx0_m2

                                cov_final[i:i+len_bins,j:j+len_bins]=cov_here
                                cov_final[j:j+len_bins,i:i+len_bins]=cov_here.T

                                if im1!=im2 and corr1==corr2:
                                    # i=indx0_c1+indx0_1+indx0_m1
                                    # j=indx0_c2+indx0_2+indx0_m2
                                    # cov_final[i:i+len_bins,j:j+len_bins]=cov_here
                                    # cov_final[j:j+len_bins,i:i+len_bins]=cov_here.T

                                    i=indx0_c1+indx0_1+indx0_m2
                                    j=indx0_c2+indx0_2+indx0_m1
                                    cov_final[i:i+len_bins,j:j+len_bins]=cov_here.T
                                    cov_final[j:j+len_bins,i:i+len_bins]=cov_here

                indx0_c2+=n_indx2*len_bins*n_m1_m2_2
            indx0_c1+=n_indx1*len_bins*n_m1_m2_1

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

    lmax_cl=5000
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

    # kappa0=lensing_lensing(zs_bins=zs_bin1,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,l=l0,
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


    if not do_xi:
        kappaS = lensing_lensing(zs_bins=zs_bins,l=l,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,
                    stack_data=stack_data,SSV_cov=SSV_cov,
                    tidal_SSV_cov=tidal_SSV_cov,do_xi=do_xi,bin_xi=bin_xi,
                    theta_bins=theta_bins)#ns=np.inf)
        clSG=kappaS.kappa_cl_tomo()#make the compute graph
        cProfile.run("cl0=clSG['stack'].compute(num_workers=4)",'output_stats_3bins')
        cl=cl0['cl']
        cov=cl0['cov']
    else:
        l_max=2e4
        l_W=np.arange(2,l_max,dtype='int')
        WT_kwargs={'l':l_W ,'theta': np.logspace(-1,1,200)*d2r,'m1_m2':[(2,2),(2,-2)]}
        cProfile.run("WT=wigner_transform(**WT_kwargs)",'output_stats_3bins')
        kappaS = lensing_lensing(zs_bins=zs_bins,l=l,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,
                    stack_data=stack_data,SSV_cov=SSV_cov,HT=WT,
                    tidal_SSV_cov=tidal_SSV_cov,do_xi=do_xi,bin_xi=bin_xi,
                    theta_bins=theta_bins)#ns=np.inf)
        xiSG=kappaS.xi_tomo()#make the compute graph
        cProfile.run("xi0=xiSG['stack'].compute(num_workers=4)",'output_stats_3bins')


    p = pstats.Stats('output_stats_3bins')
    p.sort_stats('tottime').print_stats(10)
