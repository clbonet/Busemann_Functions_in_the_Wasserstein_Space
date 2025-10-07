import torch
import scipy

import numpy as np

from torch.autograd import Function
from mpmath import *


def sqrtm(a):
    # From POT https://github.com/PythonOT/POT/blob/master/ot/backend.py

    L, V = torch.linalg.eigh(a)
    L = torch.sqrt(torch.clamp(L, 1e-8))
    # Q[...] = V[...] @ diag(L[...])
    Q = torch.einsum('...jk,...k->...jk', V, L)
    # R[...] = Q[...] @ V[...].T
    return torch.einsum('...jk,...kl->...jl', Q,
                        torch.transpose(V, -1, -2))
