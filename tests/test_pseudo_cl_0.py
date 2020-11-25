from skylens_args import *

Skylens_kwargs['do_xi']=False
Skylens_kwargs['do_pseudo_cl']=True

kappa0=Skylens(**Skylens_kwargs)

G=kappa0.cl_tomo()

cc=client.compute(G['stack']).result()
