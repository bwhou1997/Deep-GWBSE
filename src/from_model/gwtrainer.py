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
from transformer import MBformerEncoder, MBformer
from basisassembly import ElectronHoleBasisAssembly_Concatenate, sort_exciton_eigenvalues_by_eh_pair_energy, b1b2_grid, PassBasisAssembly
from enum import Enum
from sklearn.metrics import mean_absolute_error
from wfnembedder import ManyBodyData_WFN_Embedder_pretrained, SimpleSumXYEmbedder
from functools import partial

class GWTransformerTrainer(Trainer):
    def __init__(self, model, loss ,optimizer, model_name="gw_transformer", additional_metrics=None, **kwargs):
        if additional_metrics is not None:
            assert hasattr(self, "get_additional_loss"), "get_additional_loss function not implemented"
            # additional_metrics = additional_metrics.to(self.device)

        super().__init__(model, optimizer, loss, model_name=model_name, additional_metrics=additional_metrics,**kwargs)  
    def get_loss(self, input:list)->torch.Tensor:
        """
        input: [src_data, tgt_data, corr]

        """
        src_data, tgt_data, corr = input
        src_data = [x.to(self.device) for x in src_data]
        tgt_data = [x.to(self.device) for x in tgt_data]
        corr = corr.to(self.device)

        self.value, self.atten = self.model([tgt_data],[src_data])

        return self.loss(self.value, corr)
    
    @torch.no_grad()
    def evaluate(self, input=None, **kwargs):
        self.model.eval()

        src_data, tgt_data, corr = input


        src_data = [x.to(self.device) for x in src_data]
        tgt_data = [x.to(self.device) for x in tgt_data]
        corr = corr.to(self.device)
        value, atten = self.model([tgt_data],[src_data])

        return value.cpu().numpy()   


def gw_collate_fn(batch):
    src, tgt, label = batch[0]['src'], batch[0]['tgt'], batch[0]['label']

    nk = src['band_indices'].shape[0]
    nc = sum(src['band_indices'][0]>0)[0]
    nv = sum(src['band_indices'][0]<0)[0]
    nc_sigma = sum(tgt['band_indices'][0]>0)[0]
    nv_sigma = sum(tgt['band_indices'][0]<0)[0]

    src_data = [torch.from_numpy(src['latent'][()].reshape(1,nk, nc+nv, -1)).float(),
           torch.from_numpy(src['kpt'][()].reshape(1,nk, nc+nv, -1)).float(), 
           torch.from_numpy(src['band_indices'][()].reshape(1,nk, nc+nv, -1)).int(), 
           torch.from_numpy(src['el'][()].reshape(1,nk, nc+nv, -1)).float()]
    
    tgt_data = [torch.from_numpy(tgt['latent'][()].reshape(1,nk, nc_sigma+nv_sigma, -1)).float(),
           torch.from_numpy(tgt['kpt'][()].reshape(1,nk, nc_sigma+nv_sigma, -1)).float(), 
           torch.from_numpy(tgt['band_indices'][()].reshape(1,nk, nc_sigma+nv_sigma, -1)).int(), 
           torch.from_numpy(tgt['el'][()].reshape(1,nk, nc_sigma+nv_sigma, -1)).float()]
    
    corr = torch.from_numpy(label['corr'][()].reshape(1,nk, nc_sigma+nv_sigma, -1)).float()

    return src_data, tgt_data, corr




class gw_training_flow:
    # ignore this class for now
    pass

if __name__ == "__main__":
    d_model = 24
    num_epoches = 1000
    train_val_split = 0.7

    gwdata = ManyBodyData.from_existing_dataset('./dataset/dataset_GW.h5')
    eb = ManyBodyData_WFN_Embedder_pretrained(d_model, SimpleSumXYEmbedder)
    gwdata = eb.create_latent_for_ManyBodyData(gwdata, del_wfn_original=True)

    gwdata_train = gwdata[:int(len(gwdata)*train_val_split)]
    gwdata_val = gwdata[int(len(gwdata)*train_val_split):]
    dataloader_train = DataLoader(gwdata_train, batch_size=1, collate_fn=gw_collate_fn)
    dataloader_val = DataLoader(gwdata_val, batch_size=1, collate_fn=gw_collate_fn)
       
    enc2 = MBformer(d_input_src=d_model,d_input_tgt=d_model, d_model=d_model)
    
    optimizer = torch.optim.Adam(enc2.parameters(), lr=4e-4)
    loss = torch.nn.MSELoss()
    #additional_metrics = partial(torch.nn.functional.l1_loss, reduction='mean')

    gw_trainer_sigma = GWTransformerTrainer(enc2, loss, optimizer,  
                                                model_name="gw_transformer_sigma", 
                                                additional_metrics=None)    
    gw_trainer_sigma.load_model(True)
    gw_trainer_sigma.train(num_epoches, dataloader_train, dataloader_val, continued=False)

    loss = 0
    gw_trainer_sigma.load_model(load_best=True)

    for d in dataloader_val:
        src, tgt, corr = d
        # breakc2
        loss += mean_absolute_error(gw_trainer_sigma.evaluate(d).ravel(), corr.ravel())

        print("sigma:", mean_absolute_error(gw_trainer_sigma.evaluate(d).ravel(), corr.ravel()), 'eV')

    print('MAE:', loss/len(dataloader_val))