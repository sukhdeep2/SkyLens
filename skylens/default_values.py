from astropy.cosmology import Planck15 as cosmo



cosmo_h=cosmo.clone(H0=100)

cosmo_fid=dict({'h':cosmo.h,'Omb':cosmo.Ob0,'Omd':cosmo.Om0-cosmo.Ob0,'s8':0.817,'Om':cosmo.Om0,
                'Ase9':2.2,'mnu':cosmo.m_nu[-1].value,'Omk':cosmo.Ok0,'tau':0.06,'ns':0.965,
                'OmR':cosmo.Ogamma0+cosmo.Onu0,'w':-1,'wa':0,'Tcmb':cosmo.Tcmb0,'z_max':4,'use_astropy':True})
cosmo_fid['Oml']=1.-cosmo_fid['Om']-cosmo_fid['Omk']

pk_params_default={'non_linear':1,'kmax':30,'kmin':3.e-4,'nk':500,'scenario':'dmo','pk_func':'camb_pk_too_many_z','halofit_version':'takahashi'}
