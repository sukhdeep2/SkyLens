from scipy.stats import norm as gaussian
import copy
import numpy as np
from lensing_utils import *
from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
cosmo_h_PL=cosmo.clone(H0=100)
from cov_3X2 import *
import healpy as hp

def lsst_pz_source(alpha=2,z0=0.11,beta=0.68,z=[]): #alpha=1.24,z0=0.51,beta=1.01,z=[]
    p_zs=z**alpha*np.exp(-(z/z0)**beta)
    p_zs/=np.sum(np.gradient(z)*p_zs)
    return p_zs

def ztrue_given_pz_Gaussian(zp=[],p_zp=[],bias=[],sigma=[],zs=None,ns=0):
    """
        zp: photometrix redshift bins
        p_zp: Probability p(zp) of galaxies in photometric bins
        bias: bias in the true redshift distribution relative to zp.. gaussian will be centered at
                pz+bias
        sigma: scatter in the true redshift distribution relative to zp.. sigma error of the
                gaussian
        zs: True z_source, for which to compute p(z_source)
        ns: Source number density. Needed to return n(zs)
    """
    if zs is None:
        zs=np.linspace(min(zp-sigma*5),max(zp-sigma*5),500)

    y=np.tile(zs, (len(zp),1)  )
    pdf=gaussian.pdf(y.T,loc=zp+bias,scale=sigma).T
    # pdf=np.zeros((len(zp),len(zs)))
    # for i in np.arange(len(zp)):
    #     pdf[i,:]=gaussian.pdf(zs,loc=zp[i]+bias[i],scale=sigma[i])

    dzp=np.gradient(zp)

    p_zs=np.dot(dzp*p_zp,pdf)

    dzs=np.gradient(zs)
    p_zs/=np.sum(p_zs*dzs)
    nz=dzs*p_zs*ns
    return zs,p_zs,nz

def set_window(zs_bins={},f_sky=0.3,nside=256):
    l0=np.linspace(0,512,dtype='int')
    corr=('galaxy','galaxy')
    kappa0=cov_3X2(zg_bins=zs_bins,do_cov=False,bin_cl=False,l_bins=None,l=l0, zs_bins=None,use_window=False,
                   corrs=[corr])
    npix0=hp.nside2npix(nside)

    npix=np.int(npix0*f_sky)
    cl0G=kappa0.cl_tomo()
    mask=np.ones(npix0,dtype='bool')
    mask[int(npix):]=0

    cl_map0=hp.ma(np.ones(npix0))
    cl_map0[~mask]=hp.UNSEEN

    zs_bins['window0']=cl_map0
    zs_bins['window0_alm']=hp.map2alm(cl_map0)

    for i in np.arange(zs_bins['n_bins']):
        cl_i=cl0G['cl'][corr][(i,i)].compute()
        cl_map=hp.ma(1+hp.synfast(cl_i,nside=nside))
        cl_map[~mask]=hp.UNSEEN
        # cl_map.mask=mask
        zs_bins[i]['window']=cl_map
        zs_bins[i]['window_alm']=hp.map2alm(cl_map)

    return zs_bins

def source_tomo_bins(zp=None,p_zp=None,nz_bins=None,ns=26,ztrue_func=None,zp_bias=None,
                    zp_sigma=None,zs=None,z_bins=None,f_sky=0.3,nside=256,use_window=False):
    """
        Setting source redshift bins in the format used in code.
        Need
        zs (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zs
        z_bins: if zs and p_zs are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        ns: The number density for each bin to compute shape noise.
    """
    zs_bins={}

    if nz_bins is None:
        nz_bins=1

    if z_bins is None:
        z_bins=np.linspace(min(zp)-0.0001,max(zp)+0.0001,nz_bins+1)

    if zs is None:
        zs=np.linspace(0,max(z_bins)+1,100)
    dzs=np.gradient(zs)
    dzp=np.gradient(zp) if len(zp)>1 else [1]
    zp=np.array(zp)

    zl_kernel=np.linspace(0,max(zs),50)
    lu=Lensing_utils()
    cosmo_h=cosmo_h_PL

    zmax=max(z_bins)

    for i in np.arange(nz_bins):
        zs_bins[i]={}
        indx=zp.searchsorted(z_bins[i:i+2])

        if ztrue_func is None:
            if indx[0]==indx[1]:
                indx[1]=-1
            zs=zp[indx[0]:indx[1]]
            p_zs=p_zp[indx[0]:indx[1]]
            nz=ns*p_zs*dzp[indx[0]:indx[1]]
            print(indx,zs,z_bins)
        else:
            ns_i=ns*np.sum(p_zp[indx[0]:indx[1]]*dzp[indx[0]:indx[1]])
            zs,p_zs,nz=ztrue_func(zp=zp[indx[0]:indx[1]],p_zp=p_zp[indx[0]:indx[1]],
                            bias=zp_bias[indx[0]:indx[1]],
                            sigma=zp_sigma[indx[0]:indx[1]],zs=zs,ns=ns_i)
        x= p_zs>1.e-10
        zs_bins[i]['z']=zs[x]
        zs_bins[i]['dz']=np.gradient(zs_bins[i]['z']) if len(zs_bins[i]['z'])>1 else 1
        zs_bins[i]['nz']=nz[x]
        zs_bins[i]['W']=1.
        zs_bins[i]['pz']=p_zs[x]*zs_bins[i]['W']
        zs_bins[i]['pzdz']=zs_bins[i]['pz']*zs_bins[i]['dz']
        zs_bins[i]['Norm']=np.sum(zs_bins[i]['pzdz'])
        sc=1./lu.sigma_crit(zl=zl_kernel,zs=zs[x],cosmo_h=cosmo_h)
        zs_bins[i]['lens_kernel']=np.dot(zs_bins[i]['pzdz'],sc)
        zs_bins[i]['b1']=1

        zmax=max([zmax,max(zs[x])])
    zs_bins['n_bins']=nz_bins #easy to remember the counts
    zs_bins['z_lens_kernel']=zl_kernel
    zs_bins['zmax']=zmax
    zs_bins['zp']=zp
    zs_bins['pz']=p_zp
    zs_bins['z_bins']=z_bins
    zs_bins['zp_sigma']=zp_sigma
    zs_bins['zp_bias']=zp_bias
    if use_window:
        zs_bins=set_window(zs_bins=zs_bins,f_sky=f_sky,nside=nside)
    return zs_bins

def lens_wt_tomo_bins(zp=None,p_zp=None,nz_bins=None,ns=26,ztrue_func=None,zp_bias=None,
                        zp_sigma=None,cosmo_h=None,z_bins=None):
    """
        Setting source redshift bins in the format used in code.
        Need
        zs (array): redshift bins for every source bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zs
        z_bins: if zs and p_zs are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        ns: The number density for each bin to compute shape noise.
    """
    if nz_bins is None:
        nz_bins=1

    z_bins=np.linspace(min(zp),max(zp)*0.9,nz_bins) if z_bins is None else z_bins

    zs_bins0=source_tomo_bins(zp=zp,p_zp=p_zp,zp_bias=zp_bias,zp_sigma=zp_sigma,ns=ns,nz_bins=1)
    lu=Lensing_utils()

    if cosmo_h is None:
        cosmo_h=cosmo_h_PL

    zs=zs_bins0[0]['z']
    p_zs=zs_bins0[0]['z']
    dzs=zs_bins0[0]['dz']
    zs_bins={}

    zl=np.linspace(0,2,50)
    sc=1./lu.sigma_crit(zl=zl,zs=zs,cosmo_h=cosmo_h)
    scW=1./np.sum(sc,axis=1)

    for i in np.arange(nz_bins):
        i=np.int(i)
        zs_bins[i]=copy.deepcopy(zs_bins0[0])
        zs_bins[i]['W']=1./lu.sigma_crit(zl=z_bins[i],zs=zs,cosmo_h=cosmo_h)
        zs_bins[i]['W']*=scW
        zs_bins[i]['pz']*=zs_bins[i]['W']

        x= zs_bins[i]['pz']>-1 #1.e-10 # FIXME: for shape noise we check equality of 2 z arrays. Thats leads to null shape noise when cross the bins in covariance
        for v in ['z','pz','dz','W','nz']:
            zs_bins[i][v]=zs_bins[i][v][x]

        zs_bins[i]['pzdz']=zs_bins[i]['pz']*zs_bins[i]['dz']
        zs_bins[i]['Norm']=np.sum(zs_bins[i]['pzdz'])
        # zs_bins[i]['pz']/=zs_bins[i]['Norm']
        zs_bins[i]['lens_kernel']=np.dot(zs_bins[i]['pzdz'],sc)/zs_bins[i]['Norm']
    zs_bins['n_bins']=nz_bins #easy to remember the counts
    zs_bins['z_lens_kernel']=zl
    zs_bins['z_bins']=z_bins
    return zs_bins


def galaxy_tomo_bins(zp=None,p_zp=None,nz_bins=None,ns=10,ztrue_func=None,zp_bias=None,
                    zp_sigma=None,zg=None,f_sky=0.3,nside=256,use_window=False):
    """
        Setting source redshift bins in the format used in code.
        Need
        zg (array): redshift bins for every galaxy bin. if z_bins is none, then dictionary with
                    with values for each bin
        p_zs: redshift distribution. same format as zs
        z_bins: if zg and p_zg are for whole survey, then bins to divide the sample. If
                tomography is based on lens redshift, then this arrays contains those redshifts.
        ns: The number density for each bin to compute shot noise.
    """
    zg_bins={}

    if nz_bins is None:
        nz_bins=1
    z_bins=np.linspace(min(zp)-0.0001,max(zp)+0.0001,nz_bins+1)

    if zg is None:
        zg=np.linspace(0,1.5,100)
    dzg=np.gradient(zg)
    dzp=np.gradient(zp) if len(zp)>1 else [1]
    zp=np.array(zp)

    zl_kernel=np.linspace(0,2,50)
    lu=Lensing_utils()
    cosmo_h=cosmo_h_PL

    for i in np.arange(nz_bins):
        zg_bins[i]={}
        indx=zp.searchsorted(z_bins[i:i+2])

        if ztrue_func is None:
            if indx[0]==indx[1]:
                indx[1]=-1
            zg=zp[indx[0]:indx[1]]
            p_zg=p_zp[indx[0]:indx[1]]
            nz=ns*p_zg*dzp[indx[0]:indx[1]]
            print(indx,zg,z_bins)
        else:
            ns_i=ns*np.sum(p_zp[indx[0]:indx[1]]*dzp[indx[0]:indx[1]])
            zg,p_zg,nz=ztrue_func(zp=zp[indx[0]:indx[1]],p_zp=p_zp[indx[0]:indx[1]],
                            bias=zp_bias[indx[0]:indx[1]],
                            sigma=zp_sigma[indx[0]:indx[1]],zg=zg,ns=ns_i)
        x= p_zg>1.e-10
        zg_bins[i]['z']=zg[x]
        zg_bins[i]['dz']=np.gradient(zg_bins[i]['z']) if len(zg_bins[i]['z'])>1 else 1
        zg_bins[i]['nz']=nz[x]
        zg_bins[i]['W']=1.
        zg_bins[i]['pz']=p_zg[x]*zg_bins[i]['W']
        zg_bins[i]['pzdz']=zg_bins[i]['pz']*zg_bins[i]['dz']
        zg_bins[i]['Norm']=np.sum(zg_bins[i]['pzdz'])
    zg_bins['n_bins']=nz_bins #easy to remember the counts
    if use_window:
        zg_bins=set_window(zs_bins=zg_bins,f_sky=f_sky,nside=nside)
    return zg_bins


def lsst_source_tomo_bins(zmin=0.3,zmax=3,ns0=27,nbins=3,z_sigma=0.03,z_bias=None,z_bins=None,
                          ztrue_func=ztrue_given_pz_Gaussian,z_sigma_power=1,f_sky=0.3,nside=256,use_window=False):

    z=np.linspace(0,3.5,200)
    pzs=lsst_pz_source(z=z)
    N1=np.sum(pzs)

    x=z>zmin
    x*=z<zmax
    z=z[x]
    pzs=pzs[x]
    ns0=ns0*np.sum(pzs)/N1
    print('ns0: ',ns0)

    if z_bins is None:
        z_bins=np.linspace(zmin, min(2,zmax), nbins+1)
        z_bins[-1]=zmax

    if z_bias is None:
        z_bias=np.zeros_like(z)
    else:
        try:
            zb=interp1d(z_bias['z'],z_bias['b'],bounds_error=False,fill_value=0)
            z_bias=zb(z)
        except:#FIXME: Ugly
            do_nothing=1
    if np.isscalar(z_sigma):
        z_sigma=z_sigma*((1+z)**z_sigma_power)
    else:
        try:
            zs=interp1d(z_sigma['z'],z_sigma['b'],bounds_error=False,fill_value=0)
            z_sigma=zs(z)
        except: #FIXME: Ugly
            do_nothing=1

    return source_tomo_bins(zp=z,p_zp=pzs,ns=ns0,nz_bins=nbins,
                         ztrue_func=ztrue_func,zp_bias=z_bias,
                        zp_sigma=z_sigma,z_bins=z_bins,f_sky=f_sky,nside=nside,
                           use_window=use_window)


def DES_lens_bins(fname='~/Cloud/Dropbox/DES/2pt_NG_mcal_final_7_11.fits'):
    z_bins={}
    try:
        t=Table.read(fname,format='fits',hdu=6)
        dz=t['Z_HIGH']-t['Z_LOW']
        zmax=max(t['Z_HIGH'])
    except:
        t=np.genfromtxt(fname,names=('Z_MID','BIN1','BIN2','BIN3','BIN4','BIN5'))
        dz=np.gradient(t['Z_MID'])
        zmax=max(t['Z_MID'])+dz[-1]/2.

    nz_bins=5
    nz=[0.0134,0.0343,0.0505,0.0301,0.0089]
    bz=[1.44,1.70,1.698,1.997,2.058]
    for i in np.arange(nz_bins):
        z_bins[i]={}
        z_bins[i]['z']=t['Z_MID']
        z_bins[i]['dz']=dz
        z_bins[i]['nz']=nz[i]
        z_bins[i]['b1']=bz[i]
        z_bins[i]['pz']=t['BIN'+str(i+1)]
        z_bins[i]['W']=1.
        z_bins[i]['pzdz']=z_bins[i]['pz']*z_bins[i]['dz']
        z_bins[i]['Norm']=np.sum(z_bins[i]['pzdz'])
    z_bins['n_bins']=nz_bins
    z_bins['nz']=nz
    z_bins['zmax']=zmax
    return z_bins

def DES_bins(fname='~/Cloud/Dropbox/DES/2pt_NG_mcal_final_7_11.fits'):
    z_bins={}
    try:
        t=Table.read(fname,format='fits',hdu=6)
        dz=t['Z_HIGH']-t['Z_LOW']
        zmax=max(t['Z_HIGH'])
    except:
        t=np.genfromtxt(fname,names=('Z_MID','BIN1','BIN2','BIN3','BIN4'))
        dz=np.gradient(t['Z_MID'])
        zmax=max(t['Z_MID'])+dz[-1]/2.

    nz_bins=4
    nz=[1.496,1.5189,1.5949,0.7949]
    for i in np.arange(nz_bins):
        z_bins[i]={}
        z_bins[i]['z']=t['Z_MID']
        z_bins[i]['dz']=dz
        z_bins[i]['nz']=nz[i]
        z_bins[i]['pz']=t['BIN'+str(i+1)]
        z_bins[i]['W']=1.
        z_bins[i]['pzdz']=z_bins[i]['pz']*z_bins[i]['dz']
        z_bins[i]['Norm']=np.sum(z_bins[i]['pzdz'])
    z_bins['n_bins']=nz_bins
    z_bins['nz']=nz
    z_bins['zmax']=zmax
    return z_bins


def Kids_bins(kids_fname='/home/deep/data/KiDS-450/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z{zl}t{zh}.asc'):
    zl=[0.1,0.3,0.5,0.7]
    zh=[0.3,0.5,0.7,0.9]
    z_bins={}
    nz_bins=4
    nz=[1.94, 1.59, 1.52, 1.09]
    for i in np.arange(nz_bins):
        z_bins[i]={}
        t=np.genfromtxt(kids_fname.format(zl=zl[i],zh=zh[i]),names=('z','pz','pz_err'))
        z_bins[i]['z']=t['z']
        z_bins[i]['dz']=0.05
        z_bins[i]['nz']=nz[i]
        z_bins[i]['pz']=t['pz']
        z_bins[i]['W']=1.
        z_bins[i]['pzdz']=z_bins[i]['pz']*z_bins[i]['dz']
        z_bins[i]['Norm']=np.sum(z_bins[i]['pzdz'])
    z_bins['n_bins']=nz_bins
    z_bins['nz']=nz
    z_bins['zmax']=max(t['z'])+0.05
    return z_bins
