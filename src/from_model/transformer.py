import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import ToyDataSet as d
import torch
import torch.nn as nn
import numpy as np
from posemb import PositionalEmbeddings_band_energy_kpt
from basisassembly import PassBasisAssembly, ElectronHoleBasisAssembly_Concatenate, ElectronHoleBasisAssembly_TensorProduct

class MBformerEncoder(nn.Module):
    def __init__(self, d_output: int = 1, d_model: int = 480, nhead: int = 2, num_encoder_layers: int =3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5, norm_first: bool = False, bias: bool = True, 
                 max_band: int = 30, kpt_dim: int = 2, base_kpt: int = 10000, base_energy: int = 10000,
                 BasisAssembly: nn.Module = PassBasisAssembly,):
        """
        Encoder only Transformer: encode ground-state propertes, such as independent electron, electron-hole pairs.
        Note: only support independent electron (nbasis=1), and electron-hole pairs (nbasis=2).
        """
        super().__init__()

        self.kpt_dim = kpt_dim
        self.max_band = max_band
        
        batch_first: bool = True # we use batch_first
        assert activation in ["relu", "gelu", "silu", 'leaky_relu'], f"activation should be relu, gelu or glu, but got {activation}"

        self.BasisAssembly = BasisAssembly()
        self.posembedding_kpt_band_energy = PositionalEmbeddings_band_energy_kpt(d_model=d_model, max_band=max_band, kpt_dim=kpt_dim, base_kpt=base_kpt, base_energy=base_energy)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias,
                                                   activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, d_output)


    def forward(self, data1, data2=None) -> Tensor:
        """
        data: [src, kpt, band, energy]
            src: (batch, nk, nb, d_model), nbasis is the number of basis, nk is the number of k-points, nb is the number of bands.
            kpt: (batch, nk, nb, 3), 3 is the dimension of k-point.
            band: (batch, nk, nb, 1), nb is the number of bands.
            energy: (batch, nk, nb, 1), nb is the number of bands.
        args: other data
        """

        # TODO: use mask for nb and nk
        batch1, nk1, nb1, d_model1 = data1[0].shape
        assert self.max_band > data1[1].shape[-2]
        x1_emb = data1[0] + self.posembedding_kpt_band_energy(data1[1][...,:self.kpt_dim], data1[2], data1[3])

        if data2 is not None:
            assert self.max_band > data2[1].shape[-2]
            assert self.BasisAssembly.nbasis == 2, f"nbasis should be 2, but got {self.BasisAssembly.nbasis}"
            batch2, nk2, nb2, d_model2 = data2[0].shape
            x2_emb = data2[0] + self.posembedding_kpt_band_energy(data2[1][...,:self.kpt_dim], data2[2], data2[3])
            x_emb = self.BasisAssembly(x1_emb, x2_emb)
        else:
            assert self.BasisAssembly.nbasis == 1, f"nbasis should be 1, but got {self.BasisAssembly.nbasis}"
            x_emb = self.BasisAssembly(x1_emb)

        return x_emb


if __name__ == "__main__":
    ele = d.get_ele_data_batch()
    hole = d.get_hole_data_batch()
    enc1 = MBformerEncoder(d_model=d.d_model)
    enc2 = MBformerEncoder(d_model=d.d_model, BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    enc3 = MBformerEncoder(d_model=d.d_model, BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)
    print('enc1(ele) shape:',enc1(ele).shape)
    print('enc2(ele, hole) shape:',enc2(ele, hole).shape)
    print('enc3(ele, hole) shape:',enc3(ele, hole).shape)