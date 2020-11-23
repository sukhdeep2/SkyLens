import sys, os, gc, threading, subprocess,pickle
import numpy as np
from dask.distributed import Client,get_client
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

def client_get(scheduler_info=None):
    if scheduler_info is not None:
        client=get_client(address=scheduler_info['address'])
    else:
        client=get_client()
    return client

def scatter_dict(dic,scheduler_info=None):
    client=client_get(scheduler_info=scheduler_info)
    for k in dic.keys():
        if isinstance(dic[k],dict):
            dic[k]=scatter_dict(dic[k],scheduler_info=scheduler_info)
        else:
            dic[k]=client.scatter(dic[k])
    return dic