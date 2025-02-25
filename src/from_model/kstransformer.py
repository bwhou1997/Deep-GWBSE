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
paper: https://pubs.acs.org/doi/10.1021/acs.jpclett.3c02036
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
    def __init__(self, channels, sep=True):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels

        assert channels%12==0
        self.sep=sep
        if sep:
            channels = int(np.ceil(channels / 12) * 2)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 1).float() / channels))
        else:
            channels = int(np.ceil(channels / 4) * 2)
            inv_freq = 1.0 / (10000 ** (torch.arange(0, channels*3, 1).float() / channels*3))

        #inv_freq = 1.0 / (10000 ** (torch.arange(0, channels*3, 1).float() / channels*3))
        #inv_freq = 1.0 / (2 ** (torch.arange(0, channels, 1).float() ))
        self.register_buffer("inv_freq", inv_freq)
        self.channels = channels
        self.cached_penc = None
        

    def forward(self, tensor, pos):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """

        # if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
        #     return self.cached_penc

        self.cached_penc = None
        batch_size, nk, nb, orig_ch = tensor.shape

        pos_x = pos[:,:,:,0]*2*np.pi
        pos_y = pos[:,:,:,1]*2*np.pi
        pos_z = pos[:,:,:,2]*2*np.pi
        #print(pos_x.size())
        #lenp = len(pos_x)
        #pos_x = torch.rand(pos_x.size()).to(tensor.device)
        #pos_y = torch.rand(pos_y.size()).to(tensor.device)
        #pos_z = torch.rand(pos_z.size()).to(tensor.device)
        if self.sep:
            sin_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq)
            sin_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq)
            sin_inp_z = torch.einsum("ijl,k->ijlk", pos_z, self.inv_freq)
            emb_tot = get_emb(sin_inp_x, sin_inp_y, sin_inp_z).type(tensor.type())
        else:   
            sin_inp_x = torch.einsum("ijl,k->ijlk", pos_x, self.inv_freq[:self.channels])
            sin_inp_y = torch.einsum("ijl,k->ijlk", pos_y, self.inv_freq[self.channels:self.channels*2])
            sin_inp_z = torch.einsum("ijl,k->ijlk", pos_z, self.inv_freq[self.channels*2:])
            emb_x = get_emb_2(sin_inp_x)
            emb_y = get_emb_2(sin_inp_y)
            emb_z = get_emb_2(sin_inp_z)
            emb_tot = (emb_x + emb_y + emb_z).type(tensor.type())
        

        res = tensor + emb_tot
        return res
##############################################################################################################

class BandPositionalEmbeddings(nn.Module):
    """
    learnable positional embeddings for bands
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.positional_embeddings = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line

    def forward(self, x: Tensor) -> Tensor:
        return self.positional_embeddings(x)   # overwrite this line




if __name__ == "__main__":
    pose_emb = PositionalEncoding3D(d.d_model)
    r, e =pose_emb(d.cond_embedding, d.cond_kpt)
    # plot the positional embeddings
    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(e[0,:,4,:].detach().numpy().T)
    plt.xlabel("kpt")
    plt.ylabel('feature')
    plt.colorbar()
    print(e.shape)