#!/bin/bash -l
#SBATCH -q RM
#SBATCH -J test
#SBATCH -e /verafs/scratch/phy200040p/sukhdeep/job_logs/do_something.err
#SBATCH -o /verafs/scratch/phy200040p/sukhdeep/job_logs/do_something.out
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 28
##SBATCH -A phy200040p
##SBATCH --array=1-1

#total_job=$SLURM_ARRAY_TASK_COUNT
#i=$SLURM_ARRAY_TASK_ID

home='/verafs/scratch/phy200040p/sukhdeep/project/skylens/scripts/'

logfile='/verafs/scratch/phy200040p/sukhdeep/job_logs/fisher_py.log'

echo '=============================================================='>>$logfile
echo 'begining::'  'on date: ' $(date)>>$logfile

python $home'Fisher-photoz.py' | cat>>$logfile


echo 'Finished::' 'on date: ' $(date) >>$logfil
