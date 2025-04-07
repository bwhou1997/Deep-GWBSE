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
from model_util import print_model_size, timeCudaWatch
from posemb import PositionalEmbeddings_band_energy_kpt
from basisassembly import PassBasisAssembly, ElectronHoleBasisAssembly_Concatenate, ElectronHoleBasisAssembly_TensorProduct


class MBformerEncoder(nn.Module):
    def __init__(self, d_input: int = 24, d_output: int = 1, d_model: int = 576, 
                 nhead: int = 2, num_encoder_layers: int =3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5, norm_first: bool = False, bias: bool = True, 
                 max_band: int = 30, kpt_dim: int = 2, base_kpt: int = 10000, base_energy: int = 10000,
                 BasisAssembly: nn.Module = PassBasisAssembly):
        """
        TODO: use **kwargs to simplify the parameters
        Encoder only Transformer: encode ground-state propertes, such as independent electron, electron-hole pairs.
        d_input: int, the dimension of input (raw VAE latent space).
        d_model: int, the dimension of model
            d_model depends on how BasisAssembly is selected.
            if tensorProduct, d_model should be a square number, and its square root could be devisiable by 6
            if concatenate, d_model/2 should be divisiable by 6
            if other assembly, d_model should be divisiable by 6
        d_output: int, the dimension of output data (depends on the task).
        Note: d_model and d_input has been decoupled by d_fixed, which is calculated by BasisAssembly.
        Other parameters are the same as nn.TransformerEncoderLayer.
        """
        super().__init__()

        self.kpt_dim = kpt_dim
        self.max_band = max_band
        assert activation in ["relu", "gelu"], f"activation should be relu, gelu but got {activation}"

        # BasisAssembly: get d_input based on BasisAssembly and d_model
        # e.g. if assmebly is tensor product, then d_fixed**2 == d_model
        # e.g. if assembly is concatenate, then d_fixed*2 == d_model
        self.BasisAssembly = BasisAssembly()
        # d_fixed is used to decouple the input and model dimension
        d_fixed = self.BasisAssembly.get_dinput_from_dmodel(d_model)
        logging.debug(f"d_input, d_fixed, d_model: {d_input}, {d_fixed}, {d_model}")

        # Modules
        self.raw_vae_emb = nn.Linear(d_input, d_fixed)
        self.posembedding_kpt_band_energy = PositionalEmbeddings_band_energy_kpt(d_model=d_fixed, max_band=max_band, kpt_dim=kpt_dim, base_kpt=base_kpt, base_energy=base_energy)

        # Transformer-Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias,
                                                   activation=activation, layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # last layer calculate attention, which is used to predict wavefunction (e.g. AcvkS)
        self.final_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)        
        self.fc = nn.Linear(d_model, d_output)

        # softmax for raw_vae_emb and self-attention
        self.softmax = nn.Softmax(dim=-1)
        self.summary()

    def summary(self,):
        print_model_size(model=self, model_name=self._get_name())

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

    def forward(self, src_datas: list[list[Tensor, Tensor, Tensor, Tensor]]) -> torch.Tensor:

        """
        Input:
            src_datas: [basis_data1, basis_data2, ...], len(datas) = number of basis (1 or 2)
            basis_data: [vae_raw_emb, kpt, band, energy]
                vae_raw_emb: (batch, nk, nb, d_input), nk is the number of k-points, nb is the number of bands.
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

        assert len(src_datas) <= 2, "Support only 1 or 2 inputs for now."
        assert self.BasisAssembly.nbasis == 2 if len(src_datas) == 2 else 1, f"nbasis should be 2 if len(datas) == 2, but got {self.BasisAssembly.nbasis}"

        # Embedding src_datas to memory_like tensor
        x_emb = self.compute_embedding(src_datas) #-> list[Tensor], (batch, nk, nb1, d_fixed)
        x_emb = self.assemble_basis(x_emb) # -> Tensor, (batch, nk, (nb1, nb2..)., d_model)
        
        # Encode src
        y = self.encode(x_emb) # -> Tensor, (batch, nk*nb1*nb2.., d_model)

        # calculate attention and output
        _, attn_weights = self.calculate_attention(y)  # -> Tensor, (batch, nk*nb1*nb2.., nk*nb1*nb2..)
        y = self.apply_final_linear(y)
        
        # reshape y and attn_weights
        y_emb_shape = x_emb.shape[:-1] + (y.shape[-1],)
        attn_weights_shape = attn_weights.shape[0:1] + x_emb.shape[1:-1] + x_emb.shape[1:-1] 
        return y.view(*y_emb_shape), attn_weights.view(*attn_weights_shape)
    
class MBformerDecoder(MBformerEncoder):
    def __init__(self, d_input: int = 24 ,d_output: int = 1, d_model: int = 576, 
                 nhead: int = 2, num_decoder_layers: int =3, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5, norm_first: bool = False, bias: bool = True, 
                 max_band: int = 30, kpt_dim: int = 2, base_kpt: int = 10000, base_energy: int = 10000,
                 BasisAssembly: nn.Module = PassBasisAssembly):
        """
        Decoder only Transformer: decode to many-body properties.
        Other parameters are the same as MBformerEncoder.
        """    
        super().__init__(d_input=d_input, d_output=d_output, d_model=d_model, nhead=nhead, num_encoder_layers=0, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, bias=bias, max_band=max_band, kpt_dim=kpt_dim, base_kpt=base_kpt, base_energy=base_energy,
                         BasisAssembly=BasisAssembly)
        # Transformer-Decoder: overwrite the encoder_layer as None
        self.encoder_layer = None
        self.encoder = None
    
        # Transformer-Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias,
                                                        activation=activation, layer_norm_eps=layer_norm_eps, batch_first=True, norm_first=norm_first)
        self.deoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

    @timeCudaWatch
    def decode(self, tgt: Tensor, src: Tensor) -> Tensor:
        tgt = tgt.view(tgt.shape[0], -1, tgt.shape[-1])
        src = src.view(src.shape[0], -1, src.shape[-1])
        out = self.deoder(tgt, src)
        return out

    def forward(self, tgt_datas:list[list[Tensor, Tensor, Tensor, Tensor]], memory: Tensor) -> torch.Tensor:
        """
        Input:
            memory: Tensor, the output of the encoder.
            tgt_datas: [basis_data1, basis_data2, ...], len(datas) = number of basis (1 or 2) (currently)
                basis_data: [vae_raw_emb, kpt, band, energy]
        Output:
            y(default): (batch, nk, (nb1, nb2...), d_output)
            attention(default): (batch, (nk, nb1, nb2...), (nk, nb1, nb2...)) 
        See MBformerEncoder for details.
        """
        # Embedding tgt_data to memory_like tensor
        assert len(tgt_datas) <= 2, "Support only 1 or 2 inputs for now."
        assert self.BasisAssembly.nbasis == 2 if len(tgt_datas) == 2 else 1, f"nbasis should be 2 if len(datas) == 2, but got {self.BasisAssembly.nbasis}"
        tgt_emb = self.compute_embedding(tgt_datas) #-> list[Tensor], (batch, nk, nb1, d_fixed)
        tgt_emb = self.assemble_basis(tgt_emb) # -> Tensor, (batch, nk, (nb1, nb2..)., d_model)

        # Decode src and tgt
        assert tgt_emb.shape[-1] == memory.shape[-1], "The dimension of the model should be the same"
        out = self.decode(tgt_emb, memory) # -> Tensor, (batch, nk*nb1*nb2.., d_model)
    
        # Calculate attention and output
        _, attn_weights = self.calculate_attention(out)  # -> Tensor, (batch, nk*nb1*nb2.., nk*nb1*nb2..)
        out = self.apply_final_linear(out)        

        # Reshape output and attention
        out_shape = tgt_emb.shape[:-1] + (out.shape[-1],)
        atten_weights_shape = attn_weights.shape[0:1] + tgt_emb.shape[1:-1] + tgt_emb.shape[1:-1]
        return out.view(out_shape), attn_weights.view(atten_weights_shape)


class MBformer(nn.Module):
    def __init__(self, d_input_src: int = 24,  num_encoder_layers: int =3,
                 d_input_tgt: int = 24, num_decoder_layers: int =3, 
                 BasisAssembly: nn.Module = PassBasisAssembly, d_output: int = 1, d_model: int = 576, nhead: int = 2, 
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu", layer_norm_eps: float = 1e-5, 
                 norm_first: bool = False, bias: bool = True,  max_band: int = 30, kpt_dim: int = 2, 
                 base_kpt: int = 10000, base_energy: int = 10000,
                 ):
        """
        d_input_src(tgt): int, the dimension of the input source(target) data
        BasisAssembly_src(tgt): nn.Module, the basis assembly module for source(target) data
        num_encoder_layers_src(tgt): int, the number of encoder layers for sourc(target)e data

        Note: the source and target data are treated separately in the model, they can have different dimension
        
        """
        super(MBformer, self).__init__()
        self.encoder = MBformerEncoder(
            d_input = d_input_src, ## different from decoder
            d_output = d_model, ## different from decoder
            d_model = d_model,
            nhead = nhead,
            num_encoder_layers = num_encoder_layers, ## different from decoder
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            norm_first = norm_first,
            bias = bias,
            max_band = max_band,
            kpt_dim = kpt_dim,
            base_kpt = base_kpt,
            base_energy = base_energy,
            BasisAssembly = BasisAssembly,
        )

        self.decoder = MBformerDecoder(
            d_input = d_input_tgt, ## different from encoder
            d_output = d_output, ## different from encoder
            d_model = d_model,
            nhead = nhead,
            num_decoder_layers = num_decoder_layers, ## different from encoder
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            layer_norm_eps = layer_norm_eps,
            norm_first = norm_first,
            bias = bias,
            max_band = max_band,
            kpt_dim = kpt_dim,
            base_kpt = base_kpt,
            base_energy = base_energy,
            BasisAssembly = BasisAssembly,
        )
        self.summary()
    def summary(self):
        print_model_size(model=self, model_name=self._get_name())
    
    def forward(self, tgt_datas:list[list[Tensor, Tensor, Tensor, Tensor]], src_datas:list[list[Tensor, Tensor, Tensor, Tensor]]):
        """
        Input:
            tgt_datas: [basis_data1, basis_data2, ...], len(tgt_datas) = number of basis (1 or 2) (currently)
                data: [vae_raw_emb, kpt, band, energy]
            src_datas: [basis_data1, basis_data2, ...], len(datas) = number of basis (1 or 2) (currently)
                data: [vae_raw_emb, kpt, band, energy]
        Output:
            y(default): (batch, nk, (nb1, nb2...), d_output)
            attention(default): (batch, (nk, nb1, nb2...), (nk, nb1, nb2...)) 
        """
        memory, _ = self.encoder(src_datas)
        output = self.decoder(tgt_datas, memory)
        return output


def unit_test_MBformerEncoder():
    # logging.basicConfig(level=logging.INFO)
    print("Unit test for MBformerEncoder")
    ele = d.get_ele_data_batch()
    hole = d.get_hole_data_batch()
    enc1 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model)
    enc2 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    enc3 = MBformerEncoder(d_input=d.d_model, d_model=d.d_model**2, num_encoder_layers=3,
                           BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)
    enc4 = MBformerEncoder(d_input=d.d_model, d_model=144, num_encoder_layers=3,
                            BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)
                           
    val, atten = enc1([ele])
    val, atten = enc2([ele, hole])
    val, atten = enc4([ele, hole])
    val, atten = enc3([ele, hole])
    assert atten.shape == (d.batch_size, d.nk_max, d.nv_max, d.nc_max, d.nk_max, d.nv_max, d.nc_max)
    print('enc3(ele, hole) shape:',val.shape)
    print('enc3(ele, hole) shape:',atten.shape)
    print("Test passed")

def unit_test_MBformerDecoder():
    print("Unit test for MBformerDecoder")
    ele = d.get_ele_data_batch()
    hole = d.get_hole_data_batch()

    ground_emb1 = torch.rand(torch.Size([10, 144, 2, 8, 24]))
    ground_emb2 = torch.rand(torch.Size([10, 144, 2, 8, 24*2]))
    ground_emb3 = torch.rand(torch.Size([10, 144, 2, 8, 24*24]))
    ground_emb4 = torch.rand(torch.Size([10, 144, 2, 8, 144]))

    dec1 = MBformerDecoder(d_input=d.d_model, d_model=d.d_model)
    dec2 = MBformerDecoder(d_input=d.d_model, d_model=d.d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    dec3 = MBformerDecoder(d_input=d.d_model, d_model=d.d_model**2, num_decoder_layers=3,
                           BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)
    dec4 = MBformerDecoder(d_input=d.d_model, d_model=144, num_decoder_layers=3,
                            BasisAssembly=ElectronHoleBasisAssembly_TensorProduct)

    val, atten = dec1([ele], ground_emb1)
    val, atten = dec2([ele, hole], ground_emb2)
    val, atten = dec4([ele, hole], ground_emb4)
    val, atten = dec3([ele, hole], ground_emb3)
    assert atten.shape == (d.batch_size, d.nk_max, d.nv_max, d.nc_max, d.nk_max, d.nv_max, d.nc_max)
    print('enc3(ele, hole) shape:',val.shape)
    print('enc3(ele, hole) shape:',atten.shape)
    print("Test passed")

def unit_test_MBFormer():
    ele = d.get_ele_data_batch()
    hole = d.get_hole_data_batch()

    t1 = MBformer(d_input_src=d.d_model,
                  d_input_tgt=d.d_model, 
                  d_model=d.d_model)

    t2 = MBformer(d_input_src=d.d_model,
                 d_input_tgt=d.d_model,  
                 d_model=d.d_model*4, 
                 BasisAssembly=ElectronHoleBasisAssembly_Concatenate)

    t3 = MBformer(d_input_src=d.d_model,
                 d_input_tgt=d.d_model,  
                 d_model=d.d_model**2, num_decoder_layers=3,
                 BasisAssembly=ElectronHoleBasisAssembly_TensorProduct
                 )
    t4 = MBformer(d_input_src=d.d_model,
                 d_input_tgt=d.d_model,  
                 d_model=144, num_decoder_layers=3,
                 BasisAssembly=ElectronHoleBasisAssembly_TensorProduct
                 )

    val, atten = t1([ele], [ele])
    val, atten = t2([ele, hole], [ele, hole])
    val, atten = t4([ele, hole], [ele, hole])
    val, atten = t3([ele, hole], [ele, hole])
    assert atten.shape == (d.batch_size, d.nk_max, d.nv_max, d.nc_max, d.nk_max, d.nv_max, d.nc_max)
    print('enc3(ele, hole) shape:',val.shape)
    print('enc3(ele, hole) shape:',atten.shape)
    print("Test passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s')
    unit_test_MBformerEncoder()
    unit_test_MBformerDecoder()
    unit_test_MBFormer()