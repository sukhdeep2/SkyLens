import torch as tc
import numpy as np

def tc_gradient(x):
    return tc.as_tensor(np.gradient(x),dtype=x.dtype,device=x.device)

def tc_isin(ar1, ar2):
    return tc.from_numpy(np.isin(ar1,ar2).astype('uint8'))