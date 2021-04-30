import sys
from skylens import *
from survey_utils import *

import seaborn as sns
colors=sns.color_palette()
markers=['o','s','^','>','v']

#start local dask cluster
ncpu=2
LC,scheduler_info=start_client(Scheduler_file=None,local_directory=None,ncpu=None,n_workers=ncpu,threads_per_worker=1,
                              memory_limit='120gb',dashboard_address=8811,processes=True)
client=client_get(scheduler_info=scheduler_info)

#set path to wigner-3j files.
wig_home='/verafs/scratch/phy200040p/sukhdeep/physics2/skylens/temp/'
wigner_files={}
wigner_files[0]= wig_home+'dask_wig3j_l5000_w5000_0_reorder.zarr'
wigner_files[2]= wig_home+'/dask_wig3j_l2200_w4400_2_reorder.zarr'
wigner_step=100

#directory to save the figures
fig_home='./figures/'

#spin factors for different correlations.
s1_s2s={}
s1_s2s[corr_gg]=(0,0)
s1_s2s[corr_ll]=[(2,2),(2,-2)]
s1_s2s[corr_ggl]=(0,2)

#plot labels for different correlations
corr_labels={corr:{} for corr in s1_s2s.keys()}
corr_labels[corr_ll][(2,2)]=r'$\xi_+$'
corr_labels[corr_ll][(2,-2)]=r'$\xi_-$'
corr_labels[corr_gg][(0,0)]=r'$gg$'
corr_labels[corr_ggl][(0,2)]=r'$g\gamma$'

#arguments to append to filenames for different correlations
corr_fnames={corr:{} for corr in s1_s2s.keys()}
corr_fnames[corr_ll][(2,2)]='llp'
corr_fnames[corr_ll][(2,-2)]='llm'
corr_fnames[corr_gg][(0,0)]='gg'
corr_fnames[corr_ggl][(0,2)]='ggl'

#do not compute non-gaussian covariance.
SSV_cov=False
tidal_SSV_cov=False
Tri_cov=False