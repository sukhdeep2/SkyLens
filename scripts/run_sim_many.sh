use_complicated_windows=( 0 1 )
unit_windows=( 0 1 ) #( 0 1 )

lognormals=( 0 )  #( 0 1 )
do_blendings=( 0 ) #( 0 1 ) 
do_SSV_sims=( 0 )
use_shot_noises=( 0 1 )

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

                        # if [ "$do_SSV_sim" -eq "1" ] && [ "$use_shot_noise" -eq "0" ]
                        # then 
                        #     #echo $do_blending $do_SSV_sim
                        #     #continue
                        #     exit 
                        # fi

                        log_file='../temp/log/run_sim'$use_complicated_window$unit_window$lognormal$do_blending$do_SSV_sim$use_shot_noise".log"

                        #conda_env py36
                        echo 'doing'  $lognormal $do_blending $do_SSV_sim $use_shot_noise 
                        echo 'logfile' $log_file
                        echo '=============================================================='>>$log_file
                        echo 'begining::' $(date)>>$log_file 
                        # python run_sim.py $use_complicated_window $unit_window $lognormal $do_blending $do_SSV_sim $use_shot_noise |cat>>$log_file
                        python3 run_sim.py  --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise |cat>>$log_file
                        
                        #---------------------------------------------------
                        echo '=========================================================================================='|cat>>$log_file
                        echo 'begining - jk ::' $(date)>>$log_file

                        python3 run_sim_jk.py --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise |cat>>$log_file
                        #---------------------------------------------------
                        echo 'done'  $lognormal $do_blending $do_SSV_sim $use_shot_noise
                        echo 'logfile' $log_file 

                        echo 'Finished::' $(date)>>$log_file                                                                                                                                                                    
                        echo '================================================' >>$log_file          

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

# wait
