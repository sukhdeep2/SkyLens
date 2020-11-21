#!/bin/bash -l
#SBATCH -q RM
#SBATCH -J mc
#SBATCH -e /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/run_mcmc_many%A_%a.err
#SBATCH -o /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/run_mcmc_many%A_%a.out
#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 28
##SBATCH --ntasks-per-node=2
#SBATCH --mem=128G
#SBATCH -A phy200040p
#SBATCH --array=1-8

ID=$SLURM_ARRAY_JOB_ID

total_job=$SLURM_ARRAY_TASK_COUNT
job_id=$SLURM_ARRAY_TASK_ID

home='/verafs/scratch/phy200040p/sukhdeep/project/skylens/scripts/'

cd $home

fix_cosmos=( 1 0 )
do_xis=( 0 1 )
bin_ls=( 1 0 )


tmp_file="/verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/""$ID""$job_id"".tmp"

echo 0 > $tmp_file

echo 'doing'
for do_xi in "${do_xis[@]}"
do
    (    
    for bin_l in "${bin_ls[@]}"
    do
    (
        for fix_cosmo in "${fix_cosmos[@]}"
        do
        (       
            njob=$(cat $tmp_file)
            echo $njob
            njob=$(( $njob + 1 ))
            echo $njob > $tmp_file
       ###donot delete
            if [ "$njob" -ne "$job_id" ]
            then
                echo 'exiting' $njob $job_id $total_job #>> $log_file
                exit
            fi
#                         ./dask-vera2.sh &
               CSCRATCH='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/scheduler_'${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID}'/'
               rm -rf $CSCRATCH
               mkdir $CSCRATCH
               killall python
               CSCRATCH=$CSCRATCH$njob'/'
               SCHEFILE=$CSCRATCH/Scheduler.dasksche.json
               worker_log=$CSCRATCH/dask-local/worker-*
               echo $worker_log
#                        while ! [ -f $SCHEFILE ]; do #redundant
#                            sleep 3
#                            echo -n . #>>$log_file
#                        done

                echo 'ids' $njob $job_id #>> $log_file
                #conda_env py36
                echo 'doing'  $lognormal $do_blending $do_SSV_sim $use_shot_noise 
                echo '==============================================================' #>>$log_file
                echo 'begining::' $(date) #>>$log_file 

                python -Xfaulthandler MCMC_emcee.py  --do_xi=$do_xi --bin_l=$bin_l --fix_cosmo=$fix_cosmo --dask_dir=$CSCRATCH #--scheduler=$SCHEFILE #|cat>>$log_file

                echo 'Finished::' $(date) #>>$log_file                                                                                                                                                                    
                echo '================================================' #>>$log_file          
                )
                done
            )
            done
        )
        done
rm $tmp_file
exit
# wait
