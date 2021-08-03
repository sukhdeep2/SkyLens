#!/bin/bash -l
#SBATCH -q RM
#SBATCH -J fish
#SBATCH -e /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/fisher_many%A_%a.err
#SBATCH -o /verafs/scratch/phy200040p/sukhdeep/project/skylens/temp/log/fisher_many%A_%a.out
#SBATCH -t 60:00:00
#SBATCH -N 5
##SBATCH -n 28
#SBATCH --ntasks-per-node=28
##SBATCH --mem=128G
##SBATCH -A phy200040p
#SBATCH --array=1-1

ID=$SLURM_ARRAY_JOB_ID

total_job=$SLURM_ARRAY_TASK_COUNT
job_id=$SLURM_ARRAY_TASK_ID

home='/verafs/scratch/phy200040p/sukhdeep/project/skylens/scripts/'
temp_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/temp/'

mkdir $temp_home
cd $home

desis=( 1  )
trains=( 1 )
train_spectras=( '100000' )
train_areas=( '150' )

tmp_file=$temp_home"/""$ID""$job_id"".tmp"

echo 0 > $tmp_file

echo 'doing'

for desi in "${desis[@]}"
do
(
    for train in "${trains[@]}"
    do
        (    
        for train_spectra in "${train_spectras[@]}"
        do
        (
            for train_area in "${train_areas[@]}"
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

                python3 -Xfaulthandler -u Fisher-photoz.py --dask_dir=$WORKSPACE --scheduler=$SCHEFILE --desi=$desi --train=$train --train_spectra=$train_spectra --train_area=$train_area
                # fg
                echo 'Finished::' 'on date: ' $(date) >>$logfil
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
