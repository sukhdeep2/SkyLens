from kappa_cl import *

def fisher():
  def __init__(self,):
    pass

  def calc_derivative(self,func=None,params=None,var=None,N_deriv=3,dx=0.01,do_log=False,
                      func_kwargs={}):
    derivs={}
    v0=params[var]
    v_sign=np.sign(v0)
    dv0s=0

    if do_log and v0!=0:
      log_v0=np.log10(np.absolute(v0))
      v0s=np.linspace(log_v0*(1.-dx),log_v0*(1.+dx),N_deriv)
      dv0s=v_sign*(v0s[1]-v0s[0])
      v0s=v_sign*10**v0s
    else:
      v0s=np.linspace(v0*(1.-dx),v0*(1.+dx),N_deriv)
      dv0s=(v0s[1]-v0s[0])
    
    Fs={}
    for i in np.arange(N_deriv):
      params_i=params.copy()
      params_i[v]=v0s[i]
      Fs[i]=func(**params_i)

    Fs['deriv']=(Fs[N_deriv-1]-Fs[0])/dv0s/(N_deriv-1)
    Fs['v0s']=v0s
    Fs['dv0s']=dv0s
    return Fs

  def calc_fisher_fix_cov(self,func=None,params=None,vars=None,N_deriv=3,dx=0.01,
                  do_log=False,func_kwargs={},cov=[]):
        
    derivs={}
    nvar=lens(vars)
    for v in vars:
      derivs[v]=self.calc_derivative(func=func,params=params,var=v,N_deriv=N_deriv,dx=dx,
                                  do_log=do_log,func_kwargs=func_kwargs)
    
    fisher=np.zeros((nvar,nvar))
    cov_inv=np.linalg.inv(cov)
    for i in np.arange(nvar):
      di=derivs[vars[i]]['deriv']
      for j in np.arange(nvar):
        dj=derivs[vars[i]]['deriv']
        fisher[i][j]=np.dot(di,np.dot(cov_inv,dj))
    out={'fisher':fisher,'derivs':derivs}
    out['cov']=np.linalg.inv(fisher)
    out['error']=np.sqrt(np.diag(out['cov']))
    return out