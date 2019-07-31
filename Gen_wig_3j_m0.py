from wigner_functions import *
import zarr
import time
lmax=3000 #1e4
wlmax=500 
m1=0
m2=0
m3=0

lmax=np.int(lmax)
wlmax=np.int(wlmax)

fname='temp/dask_wig3j_l{lmax}_w{wlmax}_{i}_reorder.zarr'

fname=fname.format(i=m2,lmax=lmax,wlmax=wlmax)
print('will save to ',fname)

lmax+=1
wlmax+=1
ncpu=12
l_step=100 #not used with dask
w_l=np.arange(wlmax)
l=np.arange(lmax)

lb=np.arange(lmax,step=l_step)
z1 = zarr.open(fname, mode='w', shape=(wlmax,lmax,lmax), #0-lmax
               chunks=(wlmax/10, lmax/10,lmax/10),
               dtype='float32',overwrite=True)

j_max=np.amax(lmax+lmax+wlmax+10)
calc_factlist(j_max)

j3=np.arange(wlmax)
    
from multiprocessing import Pool

def wig3j_recur_2d(j1b,m1,m2,m3,j3_outmax,step,z1_out,j2b):
    if j2b<j1b: #we exploit j1-j2 symmetry and hence only compute for j2>=j1
        return 0
    if np.absolute(j2b-j1b-step-1)>j3_outmax: #given j1-j2, there is a min j3 for non-zero values. If it falls outside the required j3 range, nothing to compute
        return 0
    #out= np.zeros((j3_outmax,min(step,lmax-j1b),min(step,lmax-j2b)))

    j1=np.arange(j1b,min(lmax,j1b+step))
    j2=np.arange(j2b,min(lmax,j2b+step))

    j1s=j1.reshape(1,len(j1),1)
    j2s=j2.reshape(1,1,len(j2))
    j3s=j3.reshape(len(j3),1,1)

    out=wigner_3j_000(j1s,j2s,j3s,0,0,0)
                    
    z1[:,j1b:j1b+step,j2b:j2b+step]+=out
    
    for j1i in np.arange(len(j1)):
        for j2i in np.arange(len(j2)):
            if j2[j2i]==j1[j1i]:
                out[:,j1i,j2i]*=0 #don't want to add diagonal twice below.
    z1[:,j2b:j2b+step,j1b:j1b+step]+=out.transpose(0,2,1) #exploit j1-j2 symmetry
    t3=time.time()
    print('done ',j1b,j2b,t3-t1)
    return 0

t0=time.time()
for lb1 in lb:
    ww_out={}
    t1=time.time()
    funct=partial(wig3j_recur_2d,lb1,m1,m2,m3,wlmax,l_step,z1)
    pool=Pool(ncpu)
    out_ij=pool.map(funct,lb,chunksize=1)
    pool.close()
    
    t2=time.time()
    print('done',lb1,t2-t1)
t2=time.time()
print('done all',t2-t0)
