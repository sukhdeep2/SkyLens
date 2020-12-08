from skylens_args import *
from skylens import *
from skylens.parse_input import *

Skylens_kwargs=parse_python(file_name='./tests/skylens_args.py')
print(Skylens_kwargs['scheduler_info'])
Skylens_kwargs['do_xi']=True
Skylens_kwargs['do_pseudo_cl']=False

kappa0=Skylens(**Skylens_kwargs)

G=kappa0.xi_tomo()

client=client_get(Skylens_kwargs['scheduler_info'])
cc=client.compute(G['stack']).result()
