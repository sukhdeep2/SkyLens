#!/bin/bash -l
#SBATCH -q RM
#SBATCH -J sims
#SBATCH -e /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/run_sim_many%A_%a.err
#SBATCH -o /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/run_sim_many%A_%a.out
#SBATCH -t 70:00:00
#SBATCH -N 1
##SBATCH -n 28
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128G
#SBATCH -A phy200040p
#SBATCH --array=1-4

ID=$SLURM_ARRAY_JOB_ID

total_job=$SLURM_ARRAY_TASK_COUNT
job_id=$SLURM_ARRAY_TASK_ID

home='/verafs/scratch/phy200040p/sukhdeep/project/skylens/scripts/'

cd $home

use_complicated_windows=( 0 )
unit_windows=( 0 ) #( 0 1 )

lognormals=( 0 )  #( 0 1 )
do_blendings=( 0 ) #( 0 1 ) 
do_SSV_sims=( 0 )
use_shot_noises=( 1 )

tmp_file="/verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/""$ID""$job_id"".tmp"

echo 0 > $tmp_file

echo 'doing'
for use_complicated_window in "${use_complicated_windows[@]}"
do
    (    
    for unit_window in "${unit_windows[@]}"
    do
    (
        for lognormal in "${lognormals[@]}"
        do
	    (       
            for do_blending in "${do_blendings[@]}"
            do
		(		
                for do_SSV_sim in "${do_SSV_sims[@]}"
                do
                    (   
                    for use_shot_noise in "${use_shot_noises[@]}"
                    do
			(	
                        # if [ "$do_blending" -eq "$do_SSV_sim" ] && [ "$lognormal" -eq "0" ]
                        # then 
                        #     #echo $do_blending $do_SSV_sim
                        #     #continue
                        #     exit 
                        # fi
                        # if [ "$do_blending" -eq "$lognormal" ] && [ "$do_SSV_sim" -eq "0" ]
                        # then 
                        #     #echo 'BL' $do_blending $lognormal $do_SSV_sim
                        #     #continue
                        #     exit 
                        # fi
                        # if [ "$do_blending" -eq "$do_SSV_sim" ] && [ "$lognormal" -eq "$do_SSV_sim" ]
                        # then 
                        #     #echo $do_blending $do_SSV_sim
                        #     #continue
                        #     exit 
                        # fi

                       if [ "$use_complicated_window" -eq "1" ] && [ "$unit_window" -eq "1" ]
                       then 
                            echo 'window exit' $use_complicated_window $unit_window
                            continue
                           exit 
                       fi
                            njob=$(cat $tmp_file)
                            echo $njob
                            njob=$(( $njob + 1 ))
                            echo $njob > $tmp_file

                        #log_file=/verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/run_sim_many%A_%a.out
                        #log_file=$home'../temp/log/run_sim'$use_complicated_window$unit_window$lognormal$do_blending$do_SSV_sim$use_shot_noise".log"
                        #touch $log_file
                   ###donot delete
                        if [ "$njob" -ne "$job_id" ]
                        then
                            echo 'exiting' $njob $job_id $total_job #>> $log_file
                            exit
                        fi
#                         ./dask-vera2.sh &
                        CSCRATCH='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/scheduler_'${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID}'/'
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
                        
                       python run_sim.py  --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise  --dask_dir=$CSCRATCH #--scheduler=$SCHEFILE #|cat>>$log_file
                        
                        #---------------------------------------------------
                        echo '==========================================================================================' #|cat>>$log_file
                        echo 'begining - jk ::' $(date) #>>$log_file
#                         killall dask-mpi
#                         ./dask-vera2.sh &
                        #python run_sim_jk.py --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise --dask_dir=$CSCRATCH #--scheduler=$SCHEFILE #|cat>>$log_file

                        #---------------------------------------------------
                        echo 'done'  $lognormal $do_blending $do_SSV_sim $use_shot_noise
                        echo 'logfile' $log_file 

                        echo 'Finished::' $(date) #>>$log_file                                                                                                                                                                    
                        echo '================================================' #>>$log_file          
#                         killall python
#                         killall pproxy
#                         killall srun
#                         pkill -f dask-vera.sh
#                         mv $worker_log $worker_log$use_complicated_window$unit_window$lognormal$do_blending$do_SSV_sim$use_shot_noise
                )
                        done
                )
                done
            )
            done
        )
        done
    )
    done
)
done
rm $tmp_file
exit
# wait
