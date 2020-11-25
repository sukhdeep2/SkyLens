from skylens_args import *

use_binned_ls=[False,True]

store_wins=[False,True]

SSV_covs=[False,True]
bin_cl=True
do_covs=[True,False]
# Tri_cov=[False,True]

do_pseudo_cls=[False,True]
do_xis=[True,True]
use_windows=[True,False]

keys_del=['do_xi','do_pseudo_cl','use_window','use_binned_l','use_binned_theta','store_win','SSV_cov',
          'do_cov','Tri_cov','tidal_SSV_cov']
for k in keys_del:
    del Skylens_kwargs[k]

passed=0
failed=0
failed_tests={}
traceback_tests={}
for do_xi in do_xis:
    for do_pseudo_cl in do_pseudo_cls:
        if do_xi==do_pseudo_cl:
            continue
        for use_window in use_windows:
            for do_cov in do_covs:
                for SSV_cov in SSV_covs:
                    Tri_cov=SSV_cov
                    for use_binned_l in use_binned_ls:
                        for store_win in store_wins:
                            s=''
                            s=s+' do_xi ' if do_xi else s+' do_cl '
                            s=s+' use_window ' if use_window else s
                            s=s+' do_cov ' if do_xi else s
                            s=s+' SSV_cov ' if do_xi else s
                            s=s+' use_binned_l ' if do_xi else s
                            s=s+' store_win ' if do_xi else s
                            print("\n","\n")
                            print('passed failed: ',passed, failed, ' now testing ',s)
                            print('tests that failed: ',failed_tests)
                            print("\n","\n")
                            try:
                                kappa0=Skylens(do_cov=do_cov,use_window=use_window,Tri_cov=Tri_cov,
                                               use_binned_l=use_binned_l,SSV_cov=SSV_cov,
                                               tidal_SSV_cov=SSV_cov,store_win=store_win,do_xi=do_xi,
                                                use_binned_theta=use_binned_l,
                                               **Skylens_kwargs
                                               )
                                if do_xi:
                                    G=kappa0.xi_tomo()
                                    xi_bin_utils=client.gather(kappa0.xi_bin_utils)
                                else:
                                    G=kappa0.cl_tomo()
                                cc=client.compute(G['stack']).result()
                                
                                kappa0.gather_data()
#                                 kappa0.scatter_data()
                                xi_bin_utils=kappa0.xi_bin_utils
                                cl_bin_utils=kappa0.cl_bin_utils
# #                                 kappa0.Ang_PS.clz=client.gather(kappa0.Ang_PS.clz)
#                                 kappa0.WT.gather_data()
                                WT_binned=kappa0.WT_binned
                                
                                cS=delayed(kappa0.tomo_short)(cosmo_params=kappa0.Ang_PS.PS.cosmo_params,Win=kappa0.Win,WT=kappa0.WT,
                                                    WT_binned=WT_binned,Ang_PS=kappa0.Ang_PS,zkernel=G['zkernel'],xi_bin_utils=xi_bin_utils,
                                                             cl_bin_utils=cl_bin_utils,z_bins=kappa0.z_bins)
                                cc=client.compute(cS).result()
                                passed+=1
                                
                            except Exception as err:
                                print(s,' failed with error ',err)
                                print(traceback.format_exc())
                                failed_tests[failed]=s+' failed with error '+str(err)
                                traceback_tests[failed]=str(traceback.format_exc())
                                failed+=1
#                                 crash

for i in failed_tests.keys():
    print(failed_tests[i])
    print(traceback_tests[i])