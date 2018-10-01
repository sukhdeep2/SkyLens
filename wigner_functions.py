import numpy as np
from scipy.special import binom,jn,loggamma
from scipy.special import eval_jacobi as jacobi
from multiprocessing import Pool,cpu_count
from functools import partial
import sparse

def wigner_d(m1,m2,theta,l,l_use_bessel=1.e4):
    l0=np.copy(l)
    if l_use_bessel is not None:
    #FIXME: This is not great. Due to a issues with the scipy hypergeometric function, jacobi can output nan for large ell, l>1.e4
    # As a temporary fix, for ell>1.e4, we are replacing the wigner function with the bessel function. Fingers and toes crossed!!!
    # mpmath is slower and also has convergence issues at large ell.
    #https://github.com/scipy/scipy/issues/4446
        l=np.atleast_1d(l)
        x=l<l_use_bessel
        l=np.atleast_1d(l[x])
    k=np.amin([l-m1,l-m2,l+m1,l+m2],axis=0)
    a=np.absolute(m1-m2)
    lamb=0 #lambda
    if m2>m1:
        lamb=m2-m1
    b=2*l-2*k-a
    d_mat=(-1)**lamb
    d_mat*=np.sqrt(binom(2*l-k,k+a)) #this gives array of shape l with elements choose(2l[i]-k[i], k[i]+a)
    d_mat/=np.sqrt(binom(k+b,b))
    d_mat=np.atleast_1d(d_mat)
    x=k<0
    d_mat[x]=0

    d_mat=d_mat.reshape(1,len(d_mat))
    theta=theta.reshape(len(theta),1)
    d_mat=d_mat*((np.sin(theta/2.0)**a)*(np.cos(theta/2.0)**b))
    d_mat*=jacobi(l,a,b,np.cos(theta))

    if l_use_bessel is not None:
        l=np.atleast_1d(l0)
        x=l>=l_use_bessel
        l=np.atleast_1d(l[x])
#         d_mat[:,x]=jn(m1-m2,l[x]*theta)
        d_mat=np.append(d_mat,jn(m1-m2,l*theta),axis=1)
    return d_mat


def wigner_d_parallel(m1,m2,theta,l,ncpu=None,l_use_bessel=1.e4):
    if ncpu is None:
        ncpu=cpu_count()
    p=Pool(ncpu)
    d_mat=np.array(p.map(partial(wigner_d,m1,m2,theta,l_use_bessel=l_use_bessel),l))
    return d_mat[:,:,0].T

def log_factorial(n):
    return loggamma(n+1)

def Wigner3j(m_1, m_2, m_3,j_1, j_2, j_3):
    """Calculate the Wigner 3j symbol `Wigner3j(j_1,j_2,j_3,m_1,m_2,m_3)`

    This function is inspired from implementation in
    sympy.physics.Wigner, as written by Jens Rasch.
    https://docs.sympy.org/latest/modules/physics/wigner.html

    We have modified the implementation to use log_factorial so as to
    avoid dealing with large numbers. This function also accepts
    j_1,j_2,j_3 as 1d arrays (can be of different size) and returns
    a sparse matrix of size n_1 X n_2 X n_3, where n_i is the length of j_i.
    m_i should be integer scalars.
    For sparse package, see https://pypi.org/project/sparse/

    Following from sympy implementation:
    The inputs must be integers.  (Half integer arguments are
    sacrificed so that we can use numba.)  Nonzero return quantities
    only occur when the `j`s obey the triangle inequality (any two
    must add up to be as big as or bigger than the third).

    Examples
    ========

    >>> from spherical_functions import Wigner3j
    >>> Wigner3j_log(2, 6, 4, 0, 0, 0)
    0.186989398002
    >>> Wigner3j_log(2, 6, 4, 0, 0, 1)
    0
    """
    j_1=j_1.reshape(len(np.atleast_1d(j_1)),1,1)
    j_2=j_2.reshape(1,len(np.atleast_1d(j_2)),1)
    j_3=j_3.reshape(1,1,len(np.atleast_1d(j_3)))

    if (m_1 + m_2 + m_3 != 0):
        return np.zeros_like(j_1+j_2+j_3,dtype='float64')

    x0=np.logical_not(np.any([  j_1 + j_2 - j_3<0, #triangle inequalities
                                j_1 - j_2 + j_3<0,
                                -j_1 + j_2 + j_3<0,
                                abs(m_1) > j_1+j_2*0+j_3*0, #|m_i|<j_i
                                abs(m_2) > j_2+j_1*0+j_3*0,
                                abs(m_3) > j_3+j_2*0+j_1*0
                             ],axis=0))

    a={1:(j_1 + j_2 - j_3)[x0]}

    m_3 = -m_3

    log_argsqrt =(  log_factorial(j_1 - m_1) +
                    log_factorial(j_1 + m_1) +
                    log_factorial(j_2 - m_2) +
                    log_factorial(j_2 + m_2) +
                    log_factorial(j_3 - m_3) +
                    log_factorial(j_3 + m_3)
                 )[x0]

    log_argsqrt+=(log_factorial(a[1]) +
                log_factorial(( j_1 - j_2 + j_3)[x0]) +
                log_factorial((-j_1 + j_2 + j_3)[x0]) - log_factorial((j_1+j_2+j_3)[x0]+ 1))

    log_ressqrt=0.5*log_argsqrt
    log_argsqrt=None

#     imin = max(-j_3 + j_1 + m_2, max(-j_3 + j_2 - m_1, 0))
    imin_t=(-j_3 + j_2 - m_1 +j_1*0 ).clip(min=0)[x0]
    imin = (-j_3 + j_1 + m_2 +j_2*0)[x0]
    imin[imin<imin_t]=imin_t[imin<imin_t]
    imin_t=None

#     imax = min(j_2 + m_2, min(j_1 - m_1, j_1 + j_2 - j_3))
    imax_t=(j_1 - m_1 + j_2*0+j_3*0)[x0]
    imax =(j_1 + j_2 - j_3)[x0]
    imax[imax>imax_t]=imax_t[imax>imax_t]
    imax_t=(j_2 + m_2 + j_1*0+j_3*0)[x0]
    imax[imax>imax_t]=imax_t[imax>imax_t]
    imax_t=None


    iis=np.arange(np.amin(imin), np.amax(imax) + 1) #no need to use x0 here. Can also lead to somewhat wrong answers
    sgns=np.ones_like(iis,dtype='int')*-1
    sgns[iis%2==0]=1

    b1=(j_3 - j_1 - m_2 +j_2*0)[x0]
    b2=(j_2 + m_2 +j_1*0+j_3*0)[x0]
    b3=(j_1-m_1 +j_2*0+j_3*0)[x0]
    b4=(j_3 - j_2 + m_1 +j_1*0)[x0]
    sumres_t=np.zeros_like(b1,dtype='float')

    for i in np.arange(len(iis)):
        ii=iis[i]
        x=np.logical_not(np.logical_or(ii<imin,ii>imax))
        log_den =( log_factorial(ii) +
                    log_factorial( b1[x] + ii ) +
                    log_factorial( b2[x] - ii) +
                    log_factorial( b3[x] - ii) +
                    log_factorial( b4[x] + ii ) +
                    log_factorial(a[1][x] - ii) )
        sumres_ii=np.exp(log_ressqrt[x]-log_den)*sgns[i]
        sumres_t[x]+=sumres_ii

    prefid = np.ones_like(x0,dtype='int8') # (1 if (j_1 - j_2 - m_3) % 2 == 0 else -1)
    prefid[(j_1 - j_2 - m_3+j_3*0) % 2 == 1]=-1
    return sparse.COO(np.where(x0),data=sumres_t*prefid[x0])    #ressqrt taken inside sumres calc

def Wigner3j_parallel( m_1, m_2, m_3,j_1, j_2, j_3,ncpu=None):
    if ncpu is None:
        ncpu=cpu_count()-2
    p=Pool(ncpu)
    d_mat=sparse.stack(p.map( partial(Wigner3j, m_1, m_2, m_3,j_1, j_2), j_3,
                                 chunksize=max(1,np.int(len(j_3)/ncpu/10))
                        )  )
    p.close()
    return d_mat[:,:,:,0].transpose((1,2,0))
