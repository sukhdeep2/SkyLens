import numpy as np
from scipy.special import binom,jn,loggamma
from scipy.special import eval_jacobi as jacobi
from multiprocessing import Pool,cpu_count
from functools import partial
import sparse
from sympy import Integer
from sympy import sqrt as sy_sqrt
from sympy import exp as sy_exp
from sympy import log as sy_log

from mpmath import exp as mp_exp
from mpmath import log as mp_log
from sympy.physics.wigner import wigner_3j



def wigner_d(m1,m2,theta,l,l_use_bessel=1.e4):
    l0=np.copy(l)
    if l_use_bessel is not None:
    #FIXME: This is not great. Due to a issues with the scipy hypergeometric function,
    #jacobi can output nan for large ell, l>1.e4
    # As a temporary fix, for ell>1.e4, we are replacing the wigner function with the
    # bessel function. Fingers and toes crossed!!!
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
    *Has problems due to rounding errors when numbers get large.*
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

    x0=np.logical_not(np.any([  j_1 + j_2 - j_3<0, #triangle inequalities
                                j_1 - j_2 + j_3<0,
                                -j_1 + j_2 + j_3<0,
                                abs(m_1) > j_1+j_2*0+j_3*0, #|m_i|<j_i
                                abs(m_2) > j_2+j_1*0+j_3*0,
                                abs(m_3) > j_3+j_2*0+j_1*0
                             ],axis=0))

    if (m_1 + m_2 + m_3 != 0 or x0.sum()==0):
        return np.zeros_like(j_1+j_2+j_3,dtype='float64')


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
        sumres_ii=np.exp(sumres_t-log_den)*sgns[i] #FIXME: This has numerical issues.
        sumres_t[x]+=sumres_ii

#     sumres_t=np.exp(log_ressqrt+np.log(np.absolute(sumres_t)))*np.sign(sumres_t)

    prefid = np.ones_like(x0,dtype='int8') # (1 if (j_1 - j_2 - m_3) % 2 == 0 else -1)
    prefid[(j_1 - j_2 - m_3+j_3*0) % 2 == 1]=-1
    return sparse.COO(np.where(x0),data=sumres_t*prefid[x0])    #ressqrt taken inside sumres calc

# def Wigner3j_parallel( m_1, m_2, m_3,j_1, j_2, j_3,ncpu=None):
#     if ncpu is None:
#         ncpu=cpu_count()-2
#     p=Pool(ncpu)
#     d_mat=sparse.stack(p.map( partial(Wigner3j, m_1, m_2, m_3,j_1, j_2), j_3,
#                                  chunksize=max(1,np.int(len(j_3)/ncpu/10))
#                         )  )
#     p.close()
#     return d_mat[:,:,:,0].transpose((1,2,0))



def wigner_3j_2(j_1, j_2, j_3, m_1, m_2, m_3): #this and some helper functions below is a cpoy-paste of sympy function
    r"""
    Calculate the Wigner 3j symbol `\operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)`.

    INPUT:

    -  ``j_1``, ``j_2``, ``j_3``, ``m_1``, ``m_2``, ``m_3`` - integer or half integer

    OUTPUT:

    Rational number times the square root of a rational number.

    Examples
    ========

    >>> from sympy.physics.wigner import wigner_3j
    >>> wigner_3j(2, 6, 4, 0, 0, 0)
    sqrt(715)/143
    >>> wigner_3j(2, 6, 4, 0, 0, 1)
    0

    It is an error to have arguments that are not integer or half
    integer values::

        sage: wigner_3j(2.1, 6, 4, 0, 0, 0)
        Traceback (most recent call last):
        ...
        ValueError: j values must be integer or half integer
        sage: wigner_3j(2, 6, 4, 1, 0, -1.1)
        Traceback (most recent call last):
        ...
        ValueError: m values must be integer or half integer

    NOTES:

    The Wigner 3j symbol obeys the following symmetry rules:

    - invariant under any permutation of the columns (with the
      exception of a sign change where `J:=j_1+j_2+j_3`):

      .. math::

         \begin{aligned}
         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
          &=\operatorname{Wigner3j}(j_3,j_1,j_2,m_3,m_1,m_2) \\
          &=\operatorname{Wigner3j}(j_2,j_3,j_1,m_2,m_3,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_3,j_2,j_1,m_3,m_2,m_1) \\
          &=(-1)^J \operatorname{Wigner3j}(j_1,j_3,j_2,m_1,m_3,m_2) \\
          &=(-1)^J \operatorname{Wigner3j}(j_2,j_1,j_3,m_2,m_1,m_3)
         \end{aligned}

    - invariant under space inflection, i.e.

      .. math::

         \operatorname{Wigner3j}(j_1,j_2,j_3,m_1,m_2,m_3)
         =(-1)^J \operatorname{Wigner3j}(j_1,j_2,j_3,-m_1,-m_2,-m_3)

    - symmetric with respect to the 72 additional symmetries based on
      the work by [Regge58]_

    - zero for `j_1`, `j_2`, `j_3` not fulfilling triangle relation

    - zero for `m_1 + m_2 + m_3 \neq 0`

    - zero for violating any one of the conditions
      `j_1 \ge |m_1|`,  `j_2 \ge |m_2|`,  `j_3 \ge |m_3|`

    ALGORITHM:

    This function uses the algorithm of [Edmonds74]_ to calculate the
    value of the 3j symbol exactly. Note that the formula contains
    alternating sums over large factorials and is therefore unsuitable
    for finite precision arithmetic and only useful for a computer
    algebra system [Rasch03]_.

    REFERENCES:

    .. [Regge58] 'Symmetry Properties of Clebsch-Gordan Coefficients',
      T. Regge, Nuovo Cimento, Volume 10, pp. 544 (1958)

    .. [Edmonds74] 'Angular Momentum in Quantum Mechanics',
      A. R. Edmonds, Princeton University Press (1974)

    AUTHORS:

    - Jens Rasch (2009-03-24): initial version
    """
    if int(j_1 * 2) != j_1 * 2 or int(j_2 * 2) != j_2 * 2 or \
            int(j_3 * 2) != j_3 * 2:
        raise ValueError("j values must be integer or half integer")
    if int(m_1 * 2) != m_1 * 2 or int(m_2 * 2) != m_2 * 2 or \
            int(m_3 * 2) != m_3 * 2:
        raise ValueError("m values must be integer or half integer")
    if m_1 + m_2 + m_3 != 0:
        return 0
    prefid = Integer((-1) ** int(j_1 - j_2 - m_3))
    m_3 = -m_3
    a1 = j_1 + j_2 - j_3
    if a1 < 0:
        return 0
    a2 = j_1 - j_2 + j_3
    if a2 < 0:
        return 0
    a3 = -j_1 + j_2 + j_3
    if a3 < 0:
        return 0
    if (abs(m_1) > j_1) or (abs(m_2) > j_2) or (abs(m_3) > j_3):
        return 0

    maxfact = max(j_1 + j_2 + j_3 + 1, j_1 + abs(m_1), j_2 + abs(m_2),
                  j_3 + abs(m_3))
    _calc_factlist(int(maxfact))

    argsqrt = Integer(_Factlist[int(j_1 + j_2 - j_3)] *
                     _Factlist[int(j_1 - j_2 + j_3)] *
                     _Factlist[int(-j_1 + j_2 + j_3)] *
                     _Factlist[int(j_1 - m_1)] *
                     _Factlist[int(j_1 + m_1)] *
                     _Factlist[int(j_2 - m_2)] *
                     _Factlist[int(j_2 + m_2)] *
                     _Factlist[int(j_3 - m_3)] *
                     _Factlist[int(j_3 + m_3)]) / \
        _Factlist[int(j_1 + j_2 + j_3 + 1)]
    ressqrt = sy_sqrt(argsqrt)
    if ressqrt.is_complex:
        ressqrt = ressqrt.as_real_imag()[0]

    imin = max(-j_3 + j_1 + m_2, -j_3 + j_2 - m_1, 0)
    imax = min(j_2 + m_2, j_1 - m_1, j_1 + j_2 - j_3)
    sumres = 0
    for ii in range(int(imin), int(imax) + 1):
        den = _Factlist[ii] * \
            _Factlist[int(ii + j_3 - j_1 - m_2)] * \
            _Factlist[int(j_2 + m_2 - ii)] * \
            _Factlist[int(j_1 - ii - m_1)] * \
            _Factlist[int(ii + j_3 - j_2 + m_1)] * \
            _Factlist[int(j_1 + j_2 - j_3 - ii)]
        sumres = sumres + Integer((-1) ** ii) / den
    res = ressqrt * sumres * prefid
    return res

_Factlist=[1,1]
def _calc_factlist(nn):
    r"""
    Function calculates a list of precomputed factorials in order to
    massively accelerate future calculations of the various
    coefficients.

    INPUT:

    -  ``nn`` -  integer, highest factorial to be computed

    OUTPUT:

    list of integers -- the list of precomputed factorials

    EXAMPLES:

    Calculate list of factorials::

        sage: from sage.functions.wigner import _calc_factlist
        sage: _calc_factlist(10)
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]
    """
    global _Factlist
    if nn >= len(_Factlist):
        for ii in range(len(_Factlist), int(nn + 1)):
            _Factlist.append(_Factlist[ii - 1] * ii)
    return _Factlist[:int(nn) + 1]


def wigner_3j_asym(j_1,j_2,j_3,m_1,m_2,m_3): #assume j1,j2>>j3... not very accurate.. only seems to work when j1==j2
    sj=(j_1+j_2+1)
    th=np.arccos((m_1-m_2)/sj)
    wd=wigner_d(m_3,j_2-j_1,np.atleast_1d(th),j_3)[0,0]
    return ((-1)**(j_2+m_2))*wd/np.sqrt(sj)

def wigner_3j_000(j_1,j_2,j_3,m_1,m_2,m_3): #m1=m2=m3=0.. Hivon+ 2002
    J=j_1+j_2+j_3
    logwj=log_factorial(J/2)
    logwj-=log_factorial(J/2-j_1)
    logwj-=log_factorial(J/2-j_2)
    logwj-=log_factorial(J/2-j_3)
    logwj-=0.5*log_factorial(J+1)
    logwj+=0.5*log_factorial(J-2*j_1)
    logwj+=0.5*log_factorial(J-2*j_2)
    logwj+=0.5*log_factorial(J-2*j_3)
    wj=(-1)**(J/2)*np.exp(logwj)
#     x=J%2==1 #already applied in calling functions
#     wj[x]=0
    return np.real(wj)

def wigner_3j_3(asym_fact,m1,m2,m3,js):
    if np.all(np.array(js)>np.absolute([m1,m2,m3])*asym_fact) and np.sum(js)%2==0:
        return np.float32(wigner_3j_000(js[0],js[1],js[2],m1,m2,m3))
    return np.float32(wigner_3j(js[0],js[1],js[2],m1,m2,m3)) #.evalf()

from itertools import product as Comb
import time
def Wigner3j_parallel( m_1, m_2, m_3,j_1, j_2, j_3,ncpu=None,asym_fact=np.inf):
    if ncpu is None:
        ncpu=cpu_count()-2

    t1=time.time()
    j_max=np.amax(j_1.max()+j_2.max()+j_3.max()+1)
    _calc_factlist(j_max)

    n1=len(j_1)
    n2=len(j_2)
    n3=len(j_3)

    c=np.array(np.meshgrid(j_1,j_2,j_3,indexing='ij')).T.reshape(-1,3) #only needed to put cuts below. Otherwise Comb is better

#     print('cmax',np.amax(c,axis=0))

    x=c[:,0]+c[:,1]-c[:,2]>=0
    x*=c[:,0]-c[:,1]+c[:,2]>=0
    x*=-c[:,0]+c[:,1]+c[:,2]>=0

    marr=np.array([m_1,m_2,m_3])

#     x*=(c>=np.abs(marr)).prod(axis=1)
    x*=abs(m_1) <= c[:,0]
    x*=abs(m_2) <= c[:,1]
    x*=abs(m_3) <= c[:,2]

    if np.all(marr==0):
        x*=(c[:,0]+c[:,1]+c[:,2])%2==0
    elif np.all(c>=np.absolute(marr)*asym_fact):
        x*=(c[:,0]+c[:,1]+c[:,2])%2==0

    c=c[x]

    x2=c>=np.absolute(marr)*asym_fact
    x2=x2.prod(axis=1)==1

    t2=time.time()
    t3=t2
    dd=np.zeros((n1,n2,n3),dtype='float32')

    if np.all(marr==0) or np.all(x2):
        d_mat=wigner_3j_000(c[:,0],c[:,1],c[:,2],m_1,m_2,m_3)
        indx1=np.searchsorted(j_1,c[:,0])
        indx2=np.searchsorted(j_2,c[:,1])
        indx3=np.searchsorted(j_3,c[:,2])
        dd[indx1,indx2,indx3]=d_mat

    else:
        d_mat=wigner_3j_000(c[x2][:,0],c[x2][:,1],c[x2][:,2],m_1,m_2,m_3)
        indx1=np.searchsorted(j_1,c[x2][:,0]) #FIXME: check this
        indx2=np.searchsorted(j_2,c[x2][:,1])
        indx3=np.searchsorted(j_3,c[x2][:,2])
        dd[indx1,indx2,indx3]=d_mat

        t3=time.time()

        if not np.all(x2):
            c=c[~x2]
            p=Pool(ncpu)
            d_mat2=p.map(partial(wigner_3j_3,asym_fact, m_1, m_2, m_3),c,chunksize=100)
            p.close()
            indx1=np.searchsorted(j_1,c[:,0])
            indx2=np.searchsorted(j_2,c[:,1])
            indx3=np.searchsorted(j_3,c[:,2])
            dd[indx1,indx2,indx3]=d_mat2
            t4=time.time()
            print(j_3,j_1.max(),j_2.max(),'done','wig time,size: ',t4-t3,c.size,np.amax(c,axis=0))
    tf=time.time()
#     print(j_3,j_1.max(),j_2.max(),'done',t2-t1,t3-t2,tf-t3,tf-t1)
    return dd
#     c=Comb(j_1,j_2,j_3) #slower
#     d_mat=p.map(partial(wigner_3j_3, m_1, m_2, m_3),c,chunksize=100)
#     d_mat=np.array(d_mat).reshape(n1,n2,n3) #when not putting any cuts on c
#     return d_mat
