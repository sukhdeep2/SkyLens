import sys, os, gc, threading, subprocess,pickle,multiprocessing,dask
import numpy as np
from dask.distributed import Client,get_client
from distributed import LocalCluster
# print('pid: ',pid, sys.version)
def thread_count():
    pid=os.getpid()
    nlwp=subprocess.run(['ps', '-o', 'nlwp', str(pid)], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[1]
    nlwp=int(nlwp)
    thc=threading.active_count()
    current_th=threading.current_thread()
    #print(pid, ' thread count, os:',nlwp, 'py:', thc)
    #print('thread id, os: ',os.getpid(), 'py: ' , current_th, threading.get_native_id() )

    return nlwp, thc


def get_size(obj, seen=None): #https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
        """Recursively finds size of objects"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size
    
def get_size_pickle(obj):
    yy=pickle.dumps(obj)
    return np.around(sys.getsizeof(yy)/1.e6,decimals=3)

def dict_size_pickle(obj,print_prefact=''): #useful for some memory diagnostics
    print(print_prefact,'dict full size ',get_size_pickle(obj))
    for k in obj.keys():
        if isinstance(obj[k],dict):
            dict_size_pickle(obj[k])
        else:
            print(print_prefact,'dict obj size: ',k, get_size_pickle(dict_size_pickle(obj[k])))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    lst2=[]
    for i in range(0, len(lst), n):
#         yield lst[i:i + n]
        lst2+=[lst[i:i + n]]
    return lst2

def pickle_deepcopy(obj):
       return pickle.loads(pickle.dumps(obj, -1))

def scatter_dict(dic,scheduler_info=None,depth=10,broadcast=False,return_workers=False,workers=None): #FIXME: This needs some improvement to ensure data stays on same worker. Also allow for broadcasting.
    """
        depth: Need to think this through. It appears dask can see one level of depth when scattering and gathering, but not more.
    """
    if dic is None:
        print('scatter_dict got empty dictionary')
    else:
        client=client_get(scheduler_info=scheduler_info)
        for k in dic.keys():
            if isinstance(dic[k],dict) and depth>0:
                dic[k],workers=scatter_dict(dic[k],scheduler_info=scheduler_info,depth=depth-1,broadcast=broadcast,return_workers=True,workers=workers)
            else:
                dic[k]=client.scatter(dic[k],broadcast=broadcast,workers=workers)
                workers=list(client.who_has(dic[k]).values())[0]
    #             print('scatter-dict ',k,workers)
    if return_workers:
        return dic,workers
    return dic

def gather_dict(dic,scheduler_info=None,depth=0): #FIXME: This needs some improvement to ensure data stays on same worker. Also allow for broadcasting.
                                                    #we can use client.who_has()
    """
        depth: Need to think this through. It appears dask can see one level of depth when scattering and gathering, but not more.
    """
    if dic is None:
        print('gather_dict got empty dictionary')
        return dic
    client=client_get(scheduler_info=scheduler_info)
    for k in dic.keys():
        if isinstance(dic[k],dict) and depth>0:
            dic[k]=gather_dict(dic[k],scheduler_info=scheduler_info,depth=depth-1)
        else:
            dic[k]=client.gather(dic[k])
    return dic

def client_get(scheduler_info=None):
    if scheduler_info is not None:
        client=get_client(address=scheduler_info['address'])
    else:
        client=get_client()
    return client

worker_kwargs={}#{'memory_spill_fraction':.95,'memory_target_fraction':.95,'memory_pause_fraction':1}
def start_client(Scheduler_file=None,local_directory=None,ncpu=None,n_workers=1,threads_per_worker=None,
                  worker_kwargs=worker_kwargs,LocalCluster_kwargs={},dashboard_address=8801,
                 memory_limit='120gb',processes=False):
    LC=None
    if local_directory is None:
        local_directory='./temp_skylens/pid'+str(os.getpid())+'/'
    try:  
        os.makedirs(local_directory)  
    except Exception as error:  
        print('error in creating local directory: ',local_directory,error) 
    if threads_per_worker is None:
        if ncpu is None:
            ncpu=multiprocessing.cpu_count()-1
        threads_per_worker=ncpu
    if n_workers is None:
        n_workers=1
    if Scheduler_file is None:
                #     dask_initialize(nthreads=27,local_directory=dask_dir)
                #     client = Client()
#         dask.config.set(scheduler='threads')
        LC=LocalCluster(n_workers=n_workers,processes=processes,threads_per_worker=threads_per_worker,
                        local_directory=local_directory,dashboard_address=dashboard_address,
                        memory_limit=memory_limit,**LocalCluster_kwargs,**worker_kwargs
                   )
        client=Client(LC)
    else:
        client=Client(scheduler_file=Scheduler_file,processes=False)
    scheduler_info=client.scheduler_info()
    scheduler_info['file']=Scheduler_file
    return LC,scheduler_info #client can be obtained from client_get

