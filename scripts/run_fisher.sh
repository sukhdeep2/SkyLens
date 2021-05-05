#!/bin/bash -l
#SBATCH -q RM
#SBATCH -J fish
#SBATCH -e /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/fisher_many%A_%a.err
#SBATCH -o /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/fisher_many%A_%a.out
#SBATCH -t 60:00:00
#SBATCH -N 5
##SBATCH -n 28
#SBATCH --ntasks-per-node=27
##SBATCH --mem=128G
##SBATCH -A phy200040p
#SBATCH --array=1-1

total_job=$SLURM_ARRAY_TASK_COUNT
i=$SLURM_ARRAY_TASK_ID

njob=$i

home='/verafs/scratch/phy200040p/sukhdeep/project/skylens/scripts/'
#temp_home='/verafs/home/sukhdeep/temp/dask/'
temp_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/temp/'

echo '=============================================================='
echo 'begining::'  'on date: ' $(date)

CSCRATCH=$temp_home'/scheduler_'${SLURM_ARRAY_JOB_ID}'/'
mkdir $CSCRATCH
CSCRATCH=$CSCRATCH''$njob'/'
rm -rf $CSCRATCH
mkdir $CSCRATCH
SCHEFILE=$CSCRATCH/Scheduler.dasksche.json
WORKSPACE=$CSCRATCH/dask-local
worker_log=$WORKSPACE/worker-*
echo 'run_fisher expects scheduler at ' $SCHEFILE
./dask-vera2.sh $CSCRATCH &

while ! [ -f $SCHEFILE ]; do
    sleep 1
    echo . #>>$log_file
done

python3 -Xfaulthandler Fisher-photoz.py --dask_dir=$WORKSPACE --scheduler=$SCHEFILE 
# fg
echo 'Finished::' 'on date: ' $(date) >>$logfil
