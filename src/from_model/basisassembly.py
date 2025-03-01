import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import ToyDataSet as d
import torch
import torch.nn as nn
import numpy as np


def b1b2_grid(nb1, nb2):
    """
    Generate a grid of indices for basis 1 and basis 2
    """
    b1 = np.arange(nb1)
    b2 = np.arange(nb2)
    bb1, bb2 = np.meshgrid(b1, b2)
    return bb1, bb2

class PassBasisAssembly(nn.Module):
    nbasis = 1
    def __init__(self,):
        super().__init__()
    def forward(self,x):
        return x
        
class ElectronHoleBasisAssembly_TensorProduct(nn.Module):
    nbasis = 2
    finite_momentum = False
    def __init__(self,):
        """
        Tensorproduct basis assembly: |cvk> = |ck> x |vk>
        input: x1, x2 with shape (batch, nk, nband, d_model)
        output: x with shape (batch, nk, x2_band, x1_band, d_model^2)
        """
        super().__init__()
    def forward(self, x1, x2):
        assert x1.shape[-3] == x2.shape[-3], "The number of k-points should be the same"
        d_model1 = x1.shape[-1]
        d_model2 = x2.shape[-1]
        assert d_model1 == d_model2, "The dimension of the model should be the same"
        bb1, bb2 = b1b2_grid(x1.shape[-2], x2.shape[-2])
        cvk = torch.einsum("bkcvi, bkcvj -> bkcvij" , x1[..., bb1, :], x2[..., bb2, :])
        return cvk.view(*cvk.shape[:-2], -1)

class ElectronHoleBasisAssembly_Concatenate(nn.Module):
    nbasis = 2
    finite_momentum = False
    def __init__(self,):
        """
        Tensorproduct basis assembly: |cvk> = |ck> + |vk>
        input: x1, x2 with shape (batch, nk, nband, d_model)
        output: x with shape (batch, nk, x2_band, x1_band, d_model*2)
        """
        super().__init__()
    
    def forward(self, x1, x2):
        assert x1.shape[-3] == x2.shape[-3], "The number of k-points should be the same"
        bb1, bb2 = b1b2_grid(x1.shape[-2], x2.shape[-2])
        cvk = torch.cat([x1[..., bb1, :], 
                      x2[..., bb2, :]], dim=-1)

        return cvk


