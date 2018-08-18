import torch as tc
import numpy as np

def tc_gradient(x):
    return tc.as_tensor(np.gradient(x),dtype=x.dtype,device=x.device)
