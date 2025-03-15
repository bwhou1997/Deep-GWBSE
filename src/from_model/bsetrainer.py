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
from enum import Enum

class BSEPredictTask(Enum):
    eigenvalues = 1
    eigenvectors = 2
    all_bse = 3

def toy_wfn_embedder(dataset, wfn_latent_dim=24):
    nk, nb = dataset[0]['src']['wfn'].shape[:2]
    for i in range(len(dataset)):
        dataset[i]['src']['latent'] = np.random.random((nk, nb, wfn_latent_dim))
    return dataset

class BSETransformerTrainer(Trainer):
    def __init__(self, model, loss ,optimizer, model_name="bse_transformer", 
                task:BSEPredictTask=BSEPredictTask.eigenvalues ,**kwargs):
        super().__init__(model, optimizer, loss, model_name=model_name, **kwargs)   

        self.task = task
        print(f"Trainer Task: {task}")
        if self.task == BSEPredictTask.all_bse:
            print('Not recommended to use all_bse task, use eigenvalues or eigenvectors instead')


    def get_loss(self, input)->torch.Tensor:
        
        ele, hole, eigenvalues, eigenvectors = input
        ele = [x.to(self.device) for x in ele]
        hole = [x.to(self.device) for x in hole]
        eigenvalues = eigenvalues.to(self.device)
        eigenvectors = eigenvectors.to(self.device)

        value, atten = self.model([ele, hole])

        # return value, atten

        print("get value and atten")

        if self.task == BSEPredictTask.eigenvalues:
            return self.loss(value, eigenvalues)
        elif self.task == BSEPredictTask.eigenvectors:
            raise NotImplementedError("Not implemented")
        elif self.task == BSEPredictTask.all_bse:
            pass
        else:
            raise NotImplementedError("Task not implemented")
        

    def evaluate(self, input, **kwargs):
        self.model.eval()
        pass

    def evaluate_input_parser(self, input):
        pass

def bse_collate_fn(batch):
    """
    input: {'src', 'label'}
           'src': see data/ManyBodyData.py "WFN" datapoint
           'laebel': {'eigenvalues', 'eigenvectors'}
    output: ele, hole, eigenvalues, eigenvectors
        ele: [latent, kpt, band_indices, el]
        hole: [latent, kpt, band_indices, el]
        eigenvalues: [nS, 1]
        eigenvectors: [nS, nk, nc, nv]
    """
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

    eigenvalues = (torch.from_numpy(label['eigenvalues']).float())[None,...]
    eigenvectors = (torch.from_numpy(label['eigenvectors']).float())[None,...]

    assert ele[0].shape[1] == nk, f"ele[0].shape[1]: {ele[0].shape[1]}, nk: {nk}"
    assert ele[0].shape[2] == nc, f"ele[0].shape[2]: {ele[0].shape[2]}, nc: {nc}"
    assert hole[0].shape[1] == nk, f"hole[0].shape[1]: {hole[0].shape[1]}, nk: {nk}"
    assert hole[0].shape[2] == nv, f"hole[0].shape[2]: {hole[0].shape[2]}, nv: {nv}"
    assert eigenvectors.shape[2] == nk, f"eigenvectors.shape[2]: {eigenvectors.shape[2]}, nk: {nk}"
    assert eigenvectors.shape[3] == nc, f"eigenvectors.shape[3]: {eigenvectors.shape[3]}, nc: {nc}"
    assert eigenvectors.shape[4] == nv, f"eigenvectors.shape[4]: {eigenvectors.shape[4]}, nv: {nv}"

    return ele, hole, eigenvalues, eigenvectors


if __name__ == "__main__":  
    d_model = 24
    num_epoches = 10
    bsedata = ToyDataSet.get_bse_dataset()
    bsedata = toy_wfn_embedder(bsedata, 
                               wfn_latent_dim=d_model)

    dataloader = DataLoader(bsedata, 
                            batch_size=1, 
                            collate_fn=bse_collate_fn)

    enc2 = MBformerEncoder(d_input=d_model, 
                           d_model=d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    
    optimizer = torch.optim.Adam(enc2.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()

    bse_trainer_eigval = BSETransformerTrainer(enc2, loss, optimizer,  
                                                model_name="bse_transformer", 
                                                task=BSEPredictTask.eigenvalues,
                                                overwrite=True)

    # TODO: sort eigenvalue and eigenvector based on basisassembly order.
    assert False, "Figure out the order of ele-hole pair (basisassembly.py, how to order state?) and uncomment the following line"