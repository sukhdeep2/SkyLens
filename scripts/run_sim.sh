use_complicated_window=0
unit_window=0

lognormal=0
do_blending=0
do_SSV_sim=0
use_shot_noise=0

log_file='./temp/log/run_sim'$use_complicated_window$unit_window$lognormal$do_blending$do_SSV_sim$use_shot_noise".log"

#conda_env py36
# python run_sim_jk.py --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise |cat>>$log_file
python3 run_sim.py    --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise #|cat>>$log_file

# use_shot_noise=1
# python run_sim_jk.py --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise #|cat>>$log_file

# unit_window=1
# use_shot_noise=0
# python run_sim_jk.py --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise #|cat>>$log_file

# use_shot_noise=1
# python run_sim_jk.py --cw=$use_complicated_window --uw=$unit_window --lognormal=$lognormal --blending=$do_blending --ssv=$do_SSV_sim --noise=$use_shot_noise #|cat>>$log_file

#---------------------------------------------------
