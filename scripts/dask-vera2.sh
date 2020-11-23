njob=$1
temp_home='/verafs/home/sukhdeep/temp/'
CSCRATCH=$temp_home'/dask/scheduler_'${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID}'/'$njob'/'

SCHEFILE=$CSCRATCH/Scheduler.dasksche.json

WORKSPACE=$CSCRATCH/dask-local
CONTROLFILE=$CSCRATCH/dask.control
MEM=$(($MEMORYLIMIT / $NPROCS))MB

NPROCS=1
NCPU=$SLURM_CPUS_ON_NODE
NTHREADS=$(($SLURM_CPUS_ON_NODE -1))
NWORKER=1 #$(($SLURM_NNODES ))

rm -rf $SCHEFILE
rm -rf $WORKSPACE

mkdir -p $CSCRATCH
mkdir -p $WORKSPACE

monitor_port=8801

dask-scheduler --scheduler-file=$SCHEFILE --dashboard-address=$monitor_port &

echo 'starting scheduler ' $SCHEFILE $WORKSPACE $NWORKER'  '$NTHREADS'  '$NPROCS '  ' $SCHEFILE #>>$log_file

while ! [ -f $SCHEFILE ]; do
    sleep 3
    echo -n . #>>$log_file
done
echo 'Scheduler booted, launching worker and client' $NWORKER'  '$NTHREADS'  '$NPROCS '  ' $SCHEFILE #>>$log_file

dask-worker  --nthreads=$NTHREADS \
	     --scheduler-file=$SCHEFILE \
	     #--no-nanny \
	     --local-directory=$WORKSPACE &

worker_log=$CSCRATCH/dask-local/worker-0.log
while ! [ -f $worker_log ]; do
    sleep 3
    echo -n . #>>$log_file
done
echo 'worker booted' $NWORKER'  '$NTHREADS #>>$log_file


wait
