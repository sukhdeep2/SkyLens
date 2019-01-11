#Created by Yu Feng
conda_env py36
OPTS=`getopt -o p:P:T:M:s:m:h --long port:nprocs:,nthreads:,memory-limit:,socks:,monitor:,help -- "$@"`

usage () {
    echo "usage : bash dask-cluster.sh -M memory-limit -P nprocs -T nthreads -s SOCKSPORT -m MONITORHOST -p MONITORPORT [ clientcommand ... ] "
    echo
    echo launch a transient dask cluster via SLURM. The purpose of the cluster is to run a single dask
    echo client.
    echo
    echo The script will setup a SOCKS proxy at localhost:SOCKSPORT on MONITORHOST to access the cluster.
    echo via the ssh service at MONITORHOST:MONITORPORT
    echo After the proxy is set up, all web interfaces inside the cluster can be accessed by the browser
    echo on the MONITORHOST.
    echo 
    echo memory-limit : memory limit per node in MB \( not per process \)
    echo nprocs : number of worker processes per node
    echo nthreads : number of threads in the pool, per process
    echo SOCKSPORT : port on the MONITORHOST for the socks proxy to access the cluster
    echo MONITORHOST : host that will have access to the cluster, must allow ssh login service. can be username@hostname.
    echo MONITORPORT : port on the MONITORHOST that has ssh service
    echo clientcmd : command that runs the dask-client. If empty, a jupyter-notebook at 8080 is launched.

    echo dask-scheculer and client are both ran on the first node of the SLURM reservation, which is
    echo assumed to be the head node that the batch script is ran.
    echo one dask-worker is launched for every other node in the reservation. 
    echo 
    echo Note: prepare the python environment before invoking this inside the sbatch script.
    echo
}

if [ $? != 0 ] ; then usage; exit 1; fi

export PYTHONPATH=$PYTHONPATH:$PWD

eval set -- "$OPTS"
XDG_RUNTIME_DIR=$HOME/.run 

SOCKSPORT=1080
CMD="jupyter lab --no-browser --ip `hostname` --port=8080"
MONITORHOST=genie

NPROCS=1
NTHREADS=16
MEMORYLIMIT=`free -t -m| awk '/^Total/ {print $2}'`

while true; do
    case $1 in
        -s | --socks) SOCKSPORT=$2 ; shift 2;;
        -P | --nprocs) NPROCS=$2 ; shift 2;;
        -T | --nthreads) NTHREADS=$2 ; shift 2;;
        -m | --monitor) MONITORHOST=$2; shift 2;;
        -M | --memory-limit) MEMORYLIMIT=$2; shift 2;;
        -h | --help ) usage; exit 0;;
        -- ) shift ; break ;;
        * ) break ;;
    esac
done

if ! [ -e $1 ] ; then
    CMD="$*"
fi

echo $CMD

if ! which dask-worker; then
    echo dask and distributed is not properly installed. Configure your python environment
    exit 1

fi

SCHEFILE=$CSCRATCH/${SLURM_JOB_ID}.dasksche.json
echo 'Scheduler file  '$SCHEFILE

WORKSPACE=$CSCRATCH/${SLURM_JOB_ID}/dask-local
CONTROLFILE=$CSCRATCH/${SLURM_JOB_ID}.control
MEM=$(($MEMORYLIMIT / $NPROCS))MB

finalize () {
    if ! [ -e $MONITORHOST ]; then
        ssh -O exit -oControlPath=$CONTROLFILE $MONITORHOST
    fi
    kill -- -$$
}

start_monitor () {
    ssh -f -N -oControlMaster=auto -oControlPath=$CONTROLFILE \
    -R$SOCKSPORT:localhost:1080 $MONITORHOST

    # 8080 : jupyter
    # 8787 : brokeh
    # 8786 : scheduler

    if ! which pproxy ; then
        pip install pproxy
    fi

    pproxy -i http+socks://localhost:1080/ &

    echo on $MONITORHOST SOCKS://localhost:$SOCKSPORT is a proxy to access the private network of the cluster
    echo    Bokeh server for job monitoring : http://`hostname`:8787
    echo    Jupyter notebook server: http://`hostname`:8785
}

trap "trap - SIGTERM && finalize" SIGINT SIGTERM EXIT

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

rm -rf $SCHEFILE
rm -rf $WORKSPACE

mkdir -p $WORKSPACE

if ! [ -e $MONITORHOST ]; then
    start_monitor
fi

echo  Working directory : $WORKSPACE
echo  Memory per node : $MEMORYLIMIT

DASKWORKER=`which dask-scheduler`
DASKWORKER=`which dask-worker`

# avoid thread oversubscription
export OMP_NUM_THREADS=1

# set -x
# launch the scheduler, and reserve the first node 
srun -w `hostname` --cpu-bind=none -n 1 -N 1 -l \
    --output=$WORKSPACE/scheduler.log \
    python `which dask-scheduler` \
    --scheduler-file=$SCHEFILE --local-directory=$WORKSPACE &

NWORKER=$(($SLURM_NNODES - 1))

while ! [ -f $SCHEFILE ]; do
    sleep 3
    echo -n .
done
echo 'Scheduler booted, launching worker and client'

srun -x `hostname` --cpu-bind=none -l -N $NWORKER -n $NWORKER \
 --output=$WORKSPACE/worker-%t.log \
 python `which dask-worker` \
 --nthreads $NTHREADS \
 --nprocs $NPROCS \
 --memory-limit $MEM \
 --scheduler-file=${SCHEFILE} \
 --local-directory=${WORKSPACE}  &

    #--no-bokeh \

# run the command
#srun --cpu-bind=none -r 0 -n 1 $CMD
( eval $CMD )
# will terminate the job when CMD finishes

ssh -O cancel -f -N -oControlMaster=auto -oControlPath=$CONTROLFILE \
    -R$SOCKSPORT:localhost:1080 $MONITORHOST
