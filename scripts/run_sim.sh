use_complicated_window=0
unit_window=0

lognormal=0
do_blending=1
do_SSV_sim=0
use_shot_noise=0

log_file='./temp/run_sim'$use_complicated_window$unit_window$lognormal$do_blending$do_SSV_sim$use_shot_noise".log"

#conda_env py36
python run_sim.py $use_complicated_window $unit_window $lognormal $do_blending $do_SSV_sim $use_shot_noise #|cat>>$log_file
#---------------------------------------------------
