import sys
sys.path.insert(0,'../')
from cov_3X2 import *

# only for python3
import importlib
reload=importlib.reload



class PCA_shear():
    def __init__(self,PCA_simName,NmarQ):
        self.PCA_simName = PCA_simName  # training PCA scenario e.g. PCA_simName = ["owls_AGN","owls_DBLIMFV1618","owls_NOSN","owls_NOSN_NOZCOOL","owls_NOZCOOL","owls_REF","owls_WDENS","owls_WML1V848","owls_WML4"]       
        self.compute_mock_obs_data(scenario="eagle",cosmo_params=cosmo_fid) # setting up mock datav
        self.compute_COV(cosmo_params=cosmo_fid,scenario="dmo")             # compute COV, setting at certain pco, dmo based
        self.cal_invL(self.COV)
        self.Nscenario = len(self.PCA_simName)
        self.Ndata     = len(self.modelv_dmo_fid)
        self.build_Ratio()
        self.build_Delta()
        self.build_wDelta()
        self.SVD(in_Delta = self.wDelta) # use weighted difference matrix to perform SVD
        self.NmarQ        = NmarQ    # number of PC modes to perform marginalization
    
    def compute_modelv(self,cosmo_params=cosmo_fid,do_cov=False,scenario='dmo'):
        #setup parameters
        lmax_cl=2000
        lmin_cl=2
        l0=np.arange(lmin_cl,lmax_cl)

        lmin_cl_Bins=lmin_cl+10
        lmax_cl_Bins=lmax_cl-10
        Nl_bins=40
        l_bins=np.int64(np.logspace(np.log10(lmin_cl_Bins),np.log10(lmax_cl_Bins),Nl_bins+1))
        lb=np.sqrt(l_bins[1:]*l_bins[:-1])

        do_cov=do_cov
        bin_cl=True
    
        SSV_cov=False
        tidal_SSV_cov=False

        bin_xi=True

        #Setup redshift bins
        zmin=0.3
        zmax=2

        z=np.linspace(0,5,200)
        pzs=lsst_pz_source(z=z)
        x=z<zmax
        x*=z>zmin
        z=z[x]
        pzs=pzs[x]

        ns0=26#+np.inf #ns=inf means shape noise is zero
        nbins=3  # Number of tomographic bins
        z_sigma=0.01
        zs_bins=source_tomo_bins(zp=z,p_zp=pzs,ns=ns0,nz_bins=nbins,
                                 ztrue_func=ztrue_given_pz_Gaussian,zp_bias=np.zeros_like(z),
                                zp_sigma=z_sigma*np.ones_like(z))
                            
        pk_params={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':5000,'scenario':scenario}
    
        kappaS = cov_3X2(zs_bins=zs_bins,l=l0,do_cov=do_cov,bin_cl=bin_cl,l_bins=l_bins,zg_bins=None,
                       SSV_cov=SSV_cov,tidal_SSV_cov=tidal_SSV_cov,do_xi=False,use_window=False,
                         do_sample_variance=True,power_spectra_kwargs={'pk_func':'bary_pk','pk_params':pk_params,'cosmo_params':cosmo_params},
                       bin_xi=bin_xi)#ns=np.inf)
                   
        clSG=kappaS.cl_tomo(pk_params=pk_params)
        clS=clSG['stack'].compute()
    
        return clS
    
    def compute_mock_obs_data(self,scenario,cosmo_params=cosmo_fid):
        clS = self.compute_modelv(cosmo_params=cosmo_params,do_cov=False,scenario=scenario)
        self.datav = clS["cl"]
    
    def compute_COV(self,cosmo_params=cosmo_fid,scenario="dmo"):
        clS = self.compute_modelv(cosmo_params=cosmo_params,do_cov=True,scenario=scenario)
        self.COV = clS["cov"]
        self.modelv_dmo_fid = clS["cl"]
        self.invCOV = np.linalg.inv(self.COV)
        
    def cal_invL(self,COV):
        self.L = np.linalg.cholesky(COV)
        self.invL = np.linalg.inv(self.L)
        
    def build_Ratio(self):
        # build Ratio Matrix (at pco_fid)
        self.Ratio     = np.zeros((self.Ndata,self.Nscenario))
        
        for j in range(self.Nscenario):
            modelv_bary_fid = self.compute_modelv(scenario=self.PCA_simName[j])["cl"]
            self.Ratio.T[j] = modelv_bary_fid/self.modelv_dmo_fid
        
        self.Ratio_1 = self.Ratio - 1.
    
    def build_Delta(self):
        # build Difference Matrix at pco_fid
        DeltaT = self.Ratio.T*self.modelv_dmo_fid-self.modelv_dmo_fid
        self.Delta = DeltaT.T
        
    def build_wDelta(self):
        # build weighted Difference Matrix at pco_fid
        self.wDelta = np.dot(self.invL,self.Delta)
    
    def SVD(self,in_Delta):
        # PC1 = self.U.T[0] ; PC2 =self.U.T[1]
        self.U, self.Sdig, VT = np.linalg.svd(in_Delta,full_matrices=True)
        
    def cal_Qexp(self,scenario):
        modelv_bary_fid = self.compute_modelv(cosmo_params=cosmo_fid,do_cov=False,scenario=scenario)["cl"]
        diff = modelv_bary_fid - self.modelv_dmo_fid
        wdiff = np.dot(self.invL,diff)
        Qexp = np.dot(self.U.T,wdiff)
        return Qexp[0:self.Nscenario]
        
    def compute_modelv_bary(self,cosmo_params=cosmo_fid,Q=None):
        if Q is None:
            Q=[0]*self.NmarQ
            
        modelv_dmo = self.compute_modelv(cosmo_params=cosmo_params,do_cov=False,scenario="dmo")["cl"]
        
        sumPC = 0.
        for j in range(self.NmarQ):
            sumPC = sumPC + Q[j]*self.U.T[j]
            
        modelv_bary = modelv_dmo + np.dot(self.L,sumPC)
        return modelv_bary
