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
from functools import wraps

def timeCudaWatch(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        
        logging.debug(f"{func.__name__} execution time: {start.elapsed_time(end) * 1e-3:.3f} s")
        return result
    return wrapper

class MBformerEncoder(nn.Module):
    def __init__(self, d_input: int = 24 ,d_output: int = 1, d_model: int = 576, nhead: int = 2, num_encoder_layers: int =3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5, norm_first: bool = False, bias: bool = True, 
                 max_band: int = 30, kpt_dim: int = 2, base_kpt: int = 10000, base_energy: int = 10000,
                 BasisAssembly: nn.Module = PassBasisAssembly):
        """
        Encoder only Transformer: encode ground-state propertes, such as independent electron, electron-hole pairs.
        d_input: int, the dimension of input (raw VAE latent space).
        d_model: int, the dimension of model
            d_model depends on how BasisAssembly is implemented.
            For PassBasisAssembly, d_model = d_input
            For ElectronHoleBasisAssembly_Concatenate, d_model = d_input * 2
            For ElectronHoleBasisAssembly_TensorProduct, d_model = d_input ** 2
        d_output: int, the dimension of output data (depends on the task).
        Other parameters are the same as nn.TransformerEncoderLayer.
        """
        super().__init__()

        self.kpt_dim = kpt_dim
        self.max_band = max_band
        batch_first: bool = True # we use batch_first
        assert activation in ["relu", "gelu", "silu", 'leaky_relu'], f"activation should be relu, gelu or glu, but got {activation}"

        # Modules
        self.raw_vae_emb = nn.Linear(d_input, d_input)
        self.BasisAssembly = BasisAssembly()
        self.posembedding_kpt_band_energy = PositionalEmbeddings_band_energy_kpt(d_model=d_input, max_band=max_band, kpt_dim=kpt_dim, base_kpt=base_kpt, base_energy=base_energy)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias,
                                                   activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # last layer calculate attention, which is used to predict wavefunction (e.g. AcvkS)
        self.final_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=batch_first)        
        self.fc = nn.Linear(d_model, d_output)

        # softmax for raw_vae_emb and self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.summary()

    def summary(self):
        print_model_size(self)

    @timeCudaWatch
    def compute_embedding(self, datas: list[list[Tensor, Tensor, Tensor, Tensor]]) -> list[Tensor]:
        return [self.softmax(self.raw_vae_emb(data[0])) +
                self.posembedding_kpt_band_energy(data[1][..., :self.kpt_dim], data[2], data[3])
                for data in datas]
    @timeCudaWatch
    def assemble_basis(self, x_emb: list[Tensor]) -> Tensor:
        return self.BasisAssembly(*x_emb)


    @timeCudaWatch
    def encode(self, x_emb: Tensor) -> Tensor:
        x_emb = x_emb.view(x_emb.shape[0], -1, x_emb.shape[-1])
        x_emb = self.encoder(x_emb)
        return x_emb
    
    @timeCudaWatch
    def calculate_attention(self, y: Tensor) -> Tensor:
        return self.final_attention(y, y, y)
    
    @timeCudaWatch
    def apply_final_linear(self, y: Tensor) -> Tensor:
        return self.fc(y)

    def forward(self, datas: list[list[Tensor, Tensor, Tensor, Tensor]]) -> torch.Tensor:

        """
        Input:
            datas: [data1. data2, ...], len(datas) = number of basis (1 or 2)
            data: [vae_raw_emb, kpt, band, energy]
                vae_raw_emb: (batch, nk, nb, d_model), nk is the number of k-points, nb is the number of bands.
                kpt: (batch, nk, nb, 3), 3 is the dimension of k-point.
                band: (batch, nk, nb, 1), nb is the number of bands.
                energy: (batch, nk, nb, 1), nb is the number of bands.
        Output:
            y(default): (batch, nk, (nb1, nb2...), d_output)
                nb1, nb2... is the number of bands for different basis.
                e.g. (batch, nk, 1, d_output) for independent electron, 
                     (batch, nk, 2, d_output) for electron-hole pairs.
            attention(default): (batch, (nk, nb1, nb2...), (nk, nb1, nb2...)) 
        """

        assert len(datas) <= 2, "Support only 1 or 2 inputs for now."
        assert self.BasisAssembly.nbasis == 2 if len(datas) == 2 else 1, f"nbasis should be 2 if len(datas) == 2, but got {self.BasisAssembly.nbasis}"

        x_emb = self.compute_embedding(datas) #-> list[Tensor], (batch, nk, nb1, d_model)
        x_emb = self.assemble_basis(x_emb) # -> Tensor, (batch, nk, (nb1, nb2..)., d_model)
        y = self.encode(x_emb) # -> Tensor, (batch, nk*nb1*nb2.., d_model)

        # calculate attention and output
        _, attn_weights = self.calculate_attention(y)  # -> Tensor, (batch, nk*nb1*nb2.., nk*nb1*nb2..)
        y = self.apply_final_linear(y)
        
        y_emb_shape = x_emb.shape[:-1] + (y.shape[-1],)
        attn_weights_shape = attn_weights.shape[0:1] + x_emb.shape[1:-1] + x_emb.shape[1:-1] 
        return y.view(*y_emb_shape), attn_weights.view(*attn_weights_shape)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s')
    # logging.basicConfig(level=logging.INFO)
    ele = d.get_ele_data_batch()
    hole = d.get_hole_data_batch()
    enc1 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model)
    enc2 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    enc3 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model**2, num_encoder_layers=3,
                           BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)
    val, atten = enc3([ele, hole])
    print('enc3(ele, hole) shape:',val.shape)
    print('enc3(ele, hole) shape:',atten.shape)