import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import ToyDataSet as d

import torch
import torch.nn as nn
import numpy as np


"""
This PositionalEmbedding is modified from APET:
paper:  https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02036
        https://arxiv.org/abs/2411.16483 
github: https://github.com/emotionor/APET/tree/main
"""
##############################################################################################################
def get_emb(sin_inp_x, sin_inp_y, sin_inp_z):
    """
    Gets a base embedding for three dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp_x.sin(), sin_inp_x.cos(), 
                       sin_inp_y.sin(), sin_inp_y.cos(), 
                       sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
    #emb2 = emb.transpose(2, 3)
    return torch.flatten(emb, -2, -1)

def get_emb_2(sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model: int):
        """
        TODO: add a learnable layer?
        :param d_model: state embedding size (d_model)
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_d_model = d_model

        assert d_model%6==0

        d_model = int(np.ceil(d_model / 6)) # why *2?
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 1).float() / d_model))

        self.register_buffer("inv_freq", inv_freq)
        self.d_model = d_model
        self.cached_penc = None
        
    def forward(self, pos):
        """
        :param tensor: A 5d tensor of size (batch_size, nk, nb, 3)
        :return: Positional Encoding Matrix of size (batch_size, nk, nb, d_model)
        """

        self.cached_penc = None
        # batch_size, nk, nb, orig_ch = tensor.shape

        pos_x = pos[:,:,:,0]*2*np.pi
        pos_y = pos[:,:,:,1]*2*np.pi
        pos_z = pos[:,:,:,2]*2*np.pi

        sin_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("ijl,k->ijlk", pos_z, self.inv_freq)
        emb_tot = get_emb(sin_inp_x, sin_inp_y, sin_inp_z)
        return emb_tot
##############################################################################################################

class BandPositionalEmbeddings(nn.Module):
    """
    learnable positional embeddings for bands
    Conduction index [1, 2, 3, ...]
    Valence index [-1, -2, -3, ...]
    Here we use two embeddings to represent the position of condution and valence bands respectively
    Note: 0 never appears in the band index
    """
    def __init__(self, d_model: int, max_len: int):
        """
        max_len: maximum band index for conduction or valence bands, not total number of bands
        The total number of bands is 2*max_len
        """
        super().__init__()
        self.positional_embeddings_pos = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line
        self.positional_embeddings_neg = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line

        self.d_model = d_model

    def forward(self, pos: Tensor) -> Tensor:
        """
        input: pos: (batch_size, nk, nb, 1)
        return:     (batch_size, nk, nb, d_model)
        """
        assert (pos != 0).all(), "Band index should not contain 0"

        batch_size, nk, nb, _ = pos.shape
        pos = pos.squeeze(-1).reshape(batch_size*nk, nb) # we treat batch_size*nk as a new batch_size, nb is the sequence length

        # Treat positive and negative indices separately
        if (pos > 0).all():
            res = self.positional_embeddings_pos(pos)
        elif (pos < 0).all():
            res =  self.positional_embeddings_neg(-pos)
        else:
            positive_mask = pos > 0
            negative_mask = pos < 0
            pos_emb = self.positional_embeddings_pos(pos[positive_mask])
            neg_emb = self.positional_embeddings_neg(-pos[negative_mask])
            # create a new tensor with the same shape as pos
            emb = torch.zeros((pos.shape[-1], self.d_model), device=pos.device)

            emb[positive_mask] = pos_emb
            emb[negative_mask] = neg_emb
            res = emb
        
        res = res.reshape(batch_size, nk, nb, self.d_model)
        return res
        

class PositionalEmbeddings_band_kpt(nn.Module):
    """
    learnable positional embeddings for bands
    """
    def __init__(self, d_model: int, max_band: int):
        super().__init__()
        self.kpt_pos_emb = PositionalEncoding3D(d_model)
        self.band_pos_emb = BandPositionalEmbeddings(d_model, max_band)

    def forward(self, kpt_pos: Tensor, band_pos: Tensor) -> Tensor:
        """
        input pos : (batch_size, nk, nb, 1) or (batch_size, nk, nb, 3)
        return:     (batch_size, nk, nb, d_model)       
        """
        # TODO: think if adding the two embeddings is the best way to combine them
        return self.kpt_pos_emb(kpt_pos) + self.band_pos_emb(band_pos)

if __name__ == "__main__":
    kpt_emb = PositionalEncoding3D(d.d_model)
    band_emb = BandPositionalEmbeddings(d.d_model, d.nb_max)
    pos_emb = PositionalEmbeddings_band_kpt(d.d_model, d.nb_max)
    k_eb = kpt_emb( d.cond_kpt)
    b_eb = band_emb(d.cond_band_index)
    kb_eb = pos_emb(d.cond_kpt, d.cond_band_index)
