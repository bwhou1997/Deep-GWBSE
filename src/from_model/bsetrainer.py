import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.interpolate import griddata
from interface import wfn
import torch.nn.functional as F
from scipy.ndimage import zoom
import time
import torch
from scipy.interpolate import LinearNDInterpolator
from data import ManyBodyData, ToyDataSet
from torch.utils.data import DataLoader
from collect_tool import check_flows_status
from trainer import Trainer
from transformer import MBformerEncoder
from basisassembly import ElectronHoleBasisAssembly_Concatenate

def toy_wfn_latent_create(dataset, wfn_latent_dim=24):
    nk, nb = dataset[0]['src']['wfn'].shape[:2]
    for i in range(len(dataset)):
        dataset[i]['src']['latent'] = np.random.random((nk, nb, wfn_latent_dim))
    return dataset

class BSETransformerTrainer(Trainer):
    def __ini__(self, model, loss ,optimizer, model_name="bse_transformer", **kwargs):
        super().__init__(model, optimizer, loss, model_name=model_name, **kwargs)

    def get_loss(self, input)->torch.Tensor:
        pass

    def evaluate(self, input, **kwargs):
        self.model.eval()
        pass

def bse_collate_fn(batch):
    # electron and hole partition
    src, label = batch[0]['src'], batch[0]['label']
    # print(src, label)

    nk = src['band_indices'].shape[0]
    nc = sum(src['band_indices'][0]>0)[0]
    nv = sum(src['band_indices'][0]<0)[0]

    # for key, value in src.items():
    ele_partition = np.where(src['band_indices']>0, True, False).squeeze()
    hole_partition = np.where(src['band_indices']<0, True, False).squeeze()

    ele = [torch.from_numpy(src['latent'][ele_partition].reshape(1,nk, nc, -1)).float(),
           torch.from_numpy(src['kpt'][ele_partition].reshape(1,nk, nc, -1)).float(), 
           torch.from_numpy(src['band_indices'][ele_partition].reshape(1,nk, nc, -1)).int(), 
           torch.from_numpy(src['el'][ele_partition].reshape(1,nk, nc, -1)).float()]
    hole = [torch.from_numpy(src['latent'][hole_partition].reshape(1,nk, nv, -1)).float(),
           torch.from_numpy(src['kpt'][hole_partition].reshape(1,nk, nv, -1)).float(), 
           torch.from_numpy(src['band_indices'][hole_partition].reshape(1,nk, nv, -1)).int(), 
           torch.from_numpy(src['el'][hole_partition].reshape(1,nk, nv, -1)).float()]

    return ele, hole, label['eigenvalues'], label['eigenvectors']


if __name__ == "__main__":  
    d_model = 24
    bsedata = ToyDataSet.get_bse_dataset()
    bsedata = toy_wfn_latent_create(bsedata, 
                                    wfn_latent_dim=d_model)

    dataloader = DataLoader(bsedata, 
                            batch_size=1, 
                            collate_fn=bse_collate_fn)

    enc2 = MBformerEncoder(d_input=d_model, 
                           d_model=d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)

