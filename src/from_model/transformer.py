import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import ToyDataSet as d
import torch
import torch.nn as nn
import numpy as np
import logging
from e2vae import print_model_size
from posemb import PositionalEmbeddings_band_energy_kpt
from basisassembly import PassBasisAssembly, ElectronHoleBasisAssembly_Concatenate, ElectronHoleBasisAssembly_TensorProduct

class MBformerEncoder(nn.Module):
    def __init__(self, d_input: int = 24 ,d_output: int = 1, d_model: int = 576, nhead: int = 2, num_encoder_layers: int =3, dim_feedforward: int = 2048, dropout: float = 0.1,
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
        self.posembedding_kpt_band_energy = PositionalEmbeddings_band_energy_kpt(d_model=d_input, max_band=max_band, kpt_dim=kpt_dim, base_kpt=base_kpt, base_energy=base_energy)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias,
                                                   activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, d_output)
        self.summary()

    def summary(self):
        print_model_size(self)

    def forward(self, datas: list[list[Tensor, Tensor, Tensor, Tensor]]) -> torch.Tensor:

        """
        datas: [data1. data2, ...]
        data: [src, kpt, band, energy]
            src: (batch, nk, nb, d_model), nbasis is the number of basis, nk is the number of k-points, nb is the number of bands.
            kpt: (batch, nk, nb, 3), 3 is the dimension of k-point.
            band: (batch, nk, nb, 1), nb is the number of bands.
            energy: (batch, nk, nb, 1), nb is the number of bands.
        """

        assert len(datas) <= 2, "Support only 1 or 2 inputs for now."
        assert self.BasisAssembly.nbasis == 2 if len(datas) == 2 else 1, f"nbasis should be 2 if len(datas) == 2, but got {self.BasisAssembly.nbasis}"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Positional Embedding
        start.record()
        x_emb = [datas[i][0] + self.posembedding_kpt_band_energy(datas[i][1][..., :self.kpt_dim], datas[i][2], datas[i][3]) for i in range(len(datas))]
        end.record(); torch.cuda.synchronize()
        logging.debug(f"Embedding computation: {start.elapsed_time(end)*1e-3:.3f} s")

        # Basis Assembly
        start.record()
        x_emb = self.BasisAssembly(*x_emb)
        end.record(); torch.cuda.synchronize()
        logging.debug(f"Basis Assembly ({len(datas)} inputs): {start.elapsed_time(end)*1e-3:.3f} s")

        # Encoder
        start.record()
        x_emb_shape = x_emb.shape
        x_emb = x_emb.view(x_emb.shape[0], -1, x_emb.shape[-1])
        y = self.encoder(x_emb)
        end.record(); torch.cuda.synchronize()
        logging.debug(f"Encoder computation: {start.elapsed_time(end)*1e-3:.3f} s")

        # Linear
        start.record()
        y = self.fc(y)
        end.record(); torch.cuda.synchronize()
        logging.debug(f"Linear computation: {start.elapsed_time(end)*1e-3:.3f} s")
        
        # Reshape to output (batch, nk, (nb1, nb2...), d_output)
        y_emb_shape = x_emb_shape[:-1] + (y.shape[-1],)
        y = y.view(*y_emb_shape)
        return y

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.basicConfig(level=logging.INFO)

    ele = d.get_ele_data_batch()
    hole = d.get_hole_data_batch()
    enc1 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model)
    enc2 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    enc3 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model**2, num_encoder_layers=3,
                           BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)
    # print('ele.shape:',ele[0].shape)
    # print('hole.shape:',hole[0].shape)
    # print('enc1(ele) shape:',enc1([ele]).shape)
    # print('enc2(ele, hole) shape:',enc2([ele, hole]).shape)
    print('enc3(ele, hole) shape:',enc3([ele, hole]).shape)