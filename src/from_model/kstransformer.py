import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from data import ToyDataSet as d


class BandPositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.positional_embeddings = nn.Embedding(num_embeddings=max_len, embedding_dim=d_model)  # overwrite this line

    def forward(self, x: Tensor) -> Tensor:
        return self.positional_embeddings(x)   # overwrite this line


if __name__ == "__main__":
    pass