import numpy as np
from nn.modules.upsample import Upsample
from nn.tensor import Tensor
import nn.functional as F
import torch

def main(): 
    t = np.arange(1, 21, 1).reshape(-1, 5)
    s = np.arange(21, 41, 1).reshape(-1, 5)
    ts = Tensor(np.array([[t, t, t], [s, s, s]]))
    us = Upsample(scale_factor=2, mode = 'nearest')
    print('input shape: ', ts.shape)
    uts = us(ts)
    print(uts[0, 0:2, 0:3, 0:2])
    print(uts.shape)

if __name__ == '__main__': 
    t = np.arange(1, 21, 1).reshape(-1, 5)
    s = np.arange(21, 41, 1).reshape(-1, 5)
    ts = Tensor(np.array([[t, t, t, t], [s, s, s, s]]))
    print(ts.shape)
    ts = F.upsample(ts, 2)
    print(ts.shape)
    print(ts[0])

    
    
    