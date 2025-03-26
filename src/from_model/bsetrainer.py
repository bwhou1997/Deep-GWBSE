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
from basisassembly import ElectronHoleBasisAssembly_Concatenate, sort_exciton_eigenvalues_by_eh_pair_energy, b1b2_grid
from wfnembedder import ManyBodyData_WFN_Embedder_pretrained, SimpleSumXYEmbedder
from enum import Enum
from sklearn.metrics import mean_absolute_error


class BSEPredictTask(Enum):
    eigenvalues = 1
    eigenvectors = 2
    # all_bse = 3

# def toy_wfn_embedder(dataset, wfn_latent_dim=24):
#     nk, nb = dataset[0]['src']['wfn'].shape[:2]
#     for i in range(len(dataset)):
#         dataset[i]['src']['latent'] = np.random.random((nk, nb, wfn_latent_dim))
#     return dataset

class BSETransformerTrainer(Trainer):
    def __init__(self, model, loss ,optimizer, model_name="bse_transformer", 
                task:BSEPredictTask=BSEPredictTask.eigenvalues ,**kwargs):
        super().__init__(model, optimizer, loss, model_name=model_name, **kwargs)   

        self.task = task
        print(f"Trainer Task: {task}")

    def get_loss(self, input:list)->torch.Tensor:
        """
        input: [ele, hole, eigenvalues, eigenvectors]
               see bse_collate_fn for details
        """
        
        ele, hole, eigenvalues, eigenvectors = input
        ele = [x.to(self.device) for x in ele]
        hole = [x.to(self.device) for x in hole]
        eigenvalues = eigenvalues.to(self.device)
        eigenvectors = eigenvectors.to(self.device)

        value, atten = self.model([ele, hole])

        if self.task == BSEPredictTask.eigenvalues:
            # eigenvalues has shape of (batch, nS, 1), here we reorder the eigenvalues based on electron-hole pair energy
            eigenvalues_sorted_by_eh_pair_energy, _ = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues)
            assert value.shape == eigenvalues_sorted_by_eh_pair_energy.shape, f"value.shape: {value.shape}, eigenvalues_sorted_by_eh_pair_energy.shape: {eigenvalues_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            return self.loss(value, eigenvalues_sorted_by_eh_pair_energy)
        
        elif self.task == BSEPredictTask.eigenvectors:
            _, eigenvectors_sorted_by_eh_pair_energy = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, eigenvectors) 
            assert atten.shape == eigenvectors_sorted_by_eh_pair_energy.shape, f"atten.shape: {atten.shape}, eigenvectors_sorted_by_eh_pair_energy.shape: {eigenvectors_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            return self.loss(atten, eigenvectors_sorted_by_eh_pair_energy)
        else:
            raise NotImplementedError("Task not implemented")
        
    @torch.no_grad()
    def evaluate(self, input=None, **kwargs):
        """
        input:
        - None: use the validation dataloader
        - data_labled: [ele, hole, eigenvalues, eigenvectors]
        - data_unlabled: [ele, hole]
        """
        self.model.eval()

        if input is None:
            assert self.validation_dataloader is not None, "Must have a non-empty input"
            for data in self.validation_dataloader:
                ele, hole, _, _ = data
                break  # By default, get only one batch
        elif isinstance(input,list) or isinstance(input, tuple):
            assert len(input)==2 or len(input)==4, f"Input should be of length 2 or 4, but got {len(input)}"
            ele, hole = input[:2]

        else:
            raise NotImplementedError("Input type not implemented")
    
        ele = [x.to(self.device) for x in ele]
        hole = [x.to(self.device) for x in hole]

        value, atten = self.model([ele, hole])

        if self.task == BSEPredictTask.eigenvalues:
            return value.cpu().numpy()
        elif self.task == BSEPredictTask.eigenvectors:
            return atten.cpu().numpy()
        else:
            raise NotImplementedError("Task not implemented")


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


class bse_training_flow:
    """
    1) create and read dataset: ManyBodyData
    2) get wfn embedding
    3) create model
    4) create trainer
    5) train model
    6) evaluate model
    """

if __name__ == "__main__":  
    
    d_model = 24
    num_epoches = 1000
    bsedata = ToyDataSet.get_bse_dataset()
    # bsedata = toy_wfn_embedder(bsedata, 
    #                            wfn_latent_dim=d_model)
    eb = ManyBodyData_WFN_Embedder_pretrained(d_model, SimpleSumXYEmbedder)
    bsedata = eb.create_latent_for_ManyBodyData(bsedata, del_wfn_original=True)

    dataloader = DataLoader(bsedata, 
                            batch_size=1, 
                            collate_fn=bse_collate_fn)

    enc2 = MBformerEncoder(d_input=d_model, 
                           d_model=d_model*2, 
                           BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    
    optimizer = torch.optim.Adam(enc2.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()

    bse_trainer_eigval = BSETransformerTrainer(enc2, loss, optimizer,  
                                                model_name="bse_transformer_eval", 
                                                task=BSEPredictTask.eigenvalues)
    bse_trainer_eigval.load_model(load_best=True)
    bse_trainer_eigval.train(num_epoches, dataloader, dataloader, continued=True)



    bse_trainer_eigvec = BSETransformerTrainer(enc2, loss, optimizer,
                                               model_name="bse_transformer_evec",
                                               task=BSEPredictTask.eigenvectors)
    bse_trainer_eigvec.load_model(load_best=True)
    bse_trainer_eigvec.train(num_epoches, dataloader, dataloader, continued=True)


    for d in dataloader:
        ele, hole, eigenvalues, eigenvectors = d
        eigval_sort, eigvec_sort = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, eigenvectors) 
        break

    bse_trainer_eigval.load_model(load_best=True)
    print("eigenval:", mean_absolute_error(bse_trainer_eigval.evaluate(d).ravel(), eigval_sort.ravel()), 'eV')

    bse_trainer_eigvec.load_model(load_best=True)
    print('eigenvec:', mean_absolute_error(bse_trainer_eigvec.evaluate(d).ravel(), eigvec_sort.ravel()))