lmax=2000 #1e4
wlmax=5e2
m1=-2
m2=2
m3=0
asym_fact=100


lmax=np.int(lmax+1)
wlmax=np.int(wlmax+1)

fname='temp/dask_wig3j_l{lmax}_w{wlmax}_{i}.zarr'
if asym_fact is not np.inf:
    fname='temp/dask_wig3j_l{lmax}_w{wlmax}_{i}_asym'+str(asym_fact)+'.zarr'
fname=fname.format(i=m2,lmax=lmax,wlmax=wlmax)
print('will save to ',fname)

ncpu=8
l_step=10 #not used with dask
w_l=np.arange(wlmax)
l=np.arange(lmax)


from wigner_functions import *
import zarr

import time

import dask
import dask.array as da
from dask import delayed

from distributed import LocalCluster
from dask.distributed import Client  # we already had this above
#http://distributed.readthedocs.io/en/latest/_modules/distributed/worker.html
LC=LocalCluster(n_workers=1,processes=False,memory_limit='50gb',threads_per_worker=ncpu,memory_spill_fraction=.99,
               memory_monitor_interval='2000ms')
client=Client(LC)

def wig3j_map(m1,m2,m3,j1,j2,j3,asym_fact=np.inf):
    xx=Wigner3j_parallel( m1, m2, m3,np.atleast_1d(j1), np.atleast_1d(j2) ,np.atleast_1d(j3),ncpu=1,
                             asym_fact=asym_fact)
    return xx

step=np.int(lmax/100)
lb=np.arange(lmax+1,step=step)
lb[-1]=lmax

j3_b=np.arange(0,wlmax+1,step=100)
j3_b[-1]=wlmax

arrs=[da.hstack([da.vstack([da.from_delayed(delayed(wig3j_map)(m1,m2,m3,l[lb[i]:lb[i+1]],l[lb[k]:lb[k+1]],
                                                               np.atleast_1d(w_l[j3_b[j3]:j3_b[j3+1] ]),asym_fact),
                    shape=(lb[i+1]-lb[i],lb[k+1]-lb[k],j3_b[j3+1]-j3_b[j3]),
                                            dtype='float32') 
                    for i in np.arange(len(lb)-1)]) 
                     for k in np.arange(len(lb)-1)])
                      for j3 in np.arange(len(j3_b)-1)]
arrs2=da.concatenate(arrs,axis=-1)
arrs2=arrs2.rechunk(chunks=(np.int(lmax/10),np.int(lmax/10),np.int(wlmax/10)))

t1=time.time()

arrs2.to_zarr(fname,overwrite=True)
t2=time.time()

print(fname, 'took time ',(t2-t1)/60, ' mins' )

LC.close()
