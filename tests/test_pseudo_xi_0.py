from skylens_args import *

Skylens_kwargs['do_xi']=True
Skylens_kwargs['do_pseudo_cl']=False

kappa0=Skylens(**Skylens_kwargs)

G=kappa0.xi_tomo()

cc=client.compute(G['stack']).result()
