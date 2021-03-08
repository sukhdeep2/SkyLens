# https://github.com/willirath/dask_jobqueue_workshop_materials/tree/master/notebooks
# https://jobqueue.dask.org/en/latest/howitworks.html

#njob=$1
temp_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/temp/'
CSCRATCH=$1  #$temp_home'/scheduler_'${SLURM_ARRAY_JOB_ID}'/'$njob'/'

SCHEFILE=$CSCRATCH/Scheduler.dasksche.json

WORKSPACE=$CSCRATCH/dask-local
CONTROLFILE=$CSCRATCH/dask.control

NCPU=$SLURM_CPUS_ON_NODE
NPROCS_worker_node=4 #$NPCU
NTHREADS=1 #$NPCU
NWORKER=$(($SLURM_NNODES -1 ))

if [ -z "$SLURM_NNODES" ]
then
    NWORKER=1
fi

echo 'worker_args:' $NWORKER $NPROCS_worker_node $NTHREADS $NCPU

MEMORYLIMIT=$(free -t -m| awk '/^Total/ {print $2}')
WORKER_MEM=50000MB #$(($MEMORYLIMIT / $NPROCS_worker_node))MB
echo 'memory ' $WORKER_MEM $MEMORYLIMIT

scheduler_ncpu=2
NPROCS_scheduler_node=2  #$(($NCPU -$scheduler_ncpu ))

IFS='.' read -r -a _hostname <<< $(hostname)  #$(hostname)
_hostname=$(echo "$_hostname")
# _hostname=$SLURMD_NODENAME
echo 'hostname:' $_hostname $(hostname)

rm -rf $SCHEFILE
rm -rf $WORKSPACE

mkdir -p $CSCRATCH
mkdir -p $WORKSPACE

monitor_port=8801

echo 'booting Scheduler' $scheduler_ncpu $NPROCS_scheduler_node $_hostname
echo 'scheduler file ' $SCHEFILE
# env|grep SLURM
srun -w $_hostname -O -N 1 -n 1 --cpus-per-task=$scheduler_ncpu --cpu-bind=none --mem-bind=none dask-scheduler --scheduler-file=$SCHEFILE \
             --dashboard-address=$monitor_port & #--exclusive
while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo . #>>$log_file
done

echo 'Scheduler booted, launching worker and client' $NWORKER'  '$NTHREADS'  '$NPROCS '  ' $SCHEFILE #>>$log_file

srun -w $_hostname -O -N 1 -n 1 --cpus-per-task=$NPROCS_scheduler_node --cpu-bind=none --mem-bind=none dask-worker --scheduler-file=$SCHEFILE \
        --nprocs $NPROCS_scheduler_node --nthreads $NTHREADS --local-directory=$WORKSPACE --memory-limit $WORKER_MEM &

srun -x $_hostname -N $NWORKER -n $NWORKER --cpu-bind=none dask-worker --scheduler-file=$SCHEFILE \
        --nprocs $NPROCS_worker_node --nthreads $NTHREADS --local-directory=$WORKSPACE --memory-limit $WORKER_MEM & #--ntasks-per-node=1 --cpus-per-task=$NPROCS_worker_node 

wait
