import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset

class ToyDataSet(Dataset):
    def __init__(self, ):
        pass

class BandPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.positional_embeddings = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line

    def forward(self, x: Tensor) -> Tensor:
        return self.positional_embeddings(x)   # overwrite this line


if __name__ == "__main__":
    d_model = 12
    batch_size = 1 #
    nk_max = 12*12
    nc_max = 20
    nv_max = 20
    nb_max = nc_max + nv_max
    d_latent = 12

    cond_embedding = torch.rand((batch_size, nk_max, nc_max, d_latent))
    val_embedding = torch.rand((batch_size, nk_max, nv_max, d_latent))
    cond_band_index = torch.arange(nc_max)[None, None, :]
    val_band_index = torch.arange(nb_max)[None, None, :]