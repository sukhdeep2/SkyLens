import sys, os, gc, threading, subprocess
pid=os.getpid()
print('pid: ',pid, sys.version)
def thread_count():
    nlwp=subprocess.run(['ps', '-o', 'nlwp', str(pid)], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[1]
    nlwp=int(nlwp)
    thc=threading.active_count()
    current_th=threading.current_thread()
    #print(pid, ' thread count, os:',nlwp, 'py:', thc)
    #print('thread id, os: ',os.getpid(), 'py: ' , current_th, threading.get_native_id() )

    return nlwp, thc
