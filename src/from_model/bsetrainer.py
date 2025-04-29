import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.interpolate import griddata
from from_model.interface import wfn
import torch.nn.functional as F
from scipy.ndimage import zoom
import time
import torch
from scipy.interpolate import LinearNDInterpolator
from from_model.data import ManyBodyData, ToyDataSet
from torch.utils.data import DataLoader
# from collect_tool import check_flows_status
from from_model.trainer import Trainer
from from_model.transformer import MBformerEncoder
from from_model.basisassembly import ElectronHoleBasisAssembly_Concatenate, sort_exciton_eigenvalues_by_eh_pair_energy, b1b2_grid
from from_model.wfnembedder import ManyBodyData_WFN_Embedder_pretrained, SimpleSumXYEmbedder, E2VAEEmbedder
from enum import Enum
from from_model.e2vae import EquivariantVAE
# from torchmetrics.regression import MeanAbsoluteError
from sklearn.metrics import mean_absolute_error, r2_score
from functools import partial
import os
import shutil


class BSEPredictTask(Enum):
    eigenvalues = 1
    eigenvectors = 2
    dipole = 3

# def toy_wfn_embedder(dataset, wfn_latent_dim=24):
#     nk, nb = dataset[0]['src']['wfn'].shape[:2]
#     for i in range(len(dataset)):
#         dataset[i]['src']['latent'] = np.random.random((nk, nb, wfn_latent_dim))
#     return dataset

class BSETransformerTrainer(Trainer):
    def __init__(self, model, loss ,optimizer, model_name="bse_transformer", 
                task:BSEPredictTask=BSEPredictTask.eigenvalues, additional_metrics=None, **kwargs):
        
        if additional_metrics is not None:
            assert hasattr(self, "get_additional_loss"), "get_additional_loss function not implemented"
            # additional_metrics = additional_metrics.to(self.device)

        super().__init__(model, optimizer, loss, model_name=model_name, additional_metrics=additional_metrics,**kwargs)   

        self.task = task
        print(f"Trainer Task: {task}")


    def get_loss(self, input:list)->torch.Tensor:
        """
        input: [ele, hole, eigenvalues, eigenvectors]
               see bse_collate_fn for details
        """

        ele, hole, eigenvalues, dipole, eigenvectors = input
        ele = [x.to(self.device) for x in ele]
        hole = [x.to(self.device) for x in hole]
        eigenvalues = eigenvalues.to(self.device)
        dipole = dipole.to(self.device)
        eigenvectors = eigenvectors.to(self.device)
        kcv_prod = np.prod(eigenvalues.shape[-3:])
        kcv = eigenvalues.shape[-3:]

        self.value, self.atten = self.model([ele, hole])

        if self.task == BSEPredictTask.eigenvalues:
            # eigenvalues has shape of (batch, nS, 1), here we reorder the eigenvalues based on electron-hole pair energy
            self.eigenvalues_sorted_by_eh_pair_energy, _, _ = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, None, None)
            assert self.value.shape == self.eigenvalues_sorted_by_eh_pair_energy.shape, f"value.shape: {self.value.shape}, eigenvalues_sorted_by_eh_pair_energy.shape: {eigenvalues_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            return self.loss(self.value, self.eigenvalues_sorted_by_eh_pair_energy)
        
        elif self.task == BSEPredictTask.eigenvectors:
            _, self.eigenvectors_sorted_by_eh_pair_energy, _ = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, eigenvectors, None) 
            assert self.atten.shape == self.eigenvectors_sorted_by_eh_pair_energy.shape, f"atten.shape: {self.atten.shape}, eigenvectors_sorted_by_eh_pair_energy.shape: {eigenvectors_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            
            # soft_atten = F.softmax(self.atten.reshape(kcv_prod, kcv_prod), dim=-1)
            # log_atten = (self.atten.reshape(kcv_prod, kcv_prod)).log() ###?????
            atten = (self.atten.reshape(kcv_prod, kcv_prod))
            target = self.eigenvectors_sorted_by_eh_pair_energy.reshape(kcv_prod, kcv_prod)
            return self.loss(atten, target)
    
        elif self.task == BSEPredictTask.dipole:
            _, _, self.dipole_sorted_by_eh_pair_energy = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, None, dipole)
            assert self.value.shape == self.dipole_sorted_by_eh_pair_energy.shape, f"value.shape: {self.value.shape}, dipole_sorted_by_eh_pair_energy.shape: {self.dipole_sorted_by_eh_pair_energy.shape}. Make sure [ele, hole] order right"
            value =torch.nn.functional.log_softmax(self.value + 1e-7, dim=1)
            return self.loss(value, self.dipole_sorted_by_eh_pair_energy)

        else:
            raise NotImplementedError("Task not implemented")


    def get_additional_loss(self)->float:
        """
        This function provide additional metrics for the model
        For efficiency, we only calculate the additional metrics from last round of get_loss (validation)
        """
        if self.additional_metrics is None:
            return ""
        else:
            if self.task == BSEPredictTask.eigenvalues:
                return self.additional_metrics(self.value.ravel(), self.eigenvalues_sorted_by_eh_pair_energy.ravel())
            elif self.task == BSEPredictTask.eigenvectors:
                return self.additional_metrics(self.atten.ravel(), self.eigenvectors_sorted_by_eh_pair_energy.ravel())
                # raise NotImplementedError("Only support eigenvalues additional metrics task for now")
            elif self.task == BSEPredictTask.dipole:
                return self.additional_metrics(torch.exp(self.value.ravel()), self.dipole_sorted_by_eh_pair_energy.ravel())
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
                ele, hole, _, _, _ = data
                break  # By default, get only one batch
        elif isinstance(input,list) or isinstance(input, tuple):
            assert len(input)>=2, f"Input should be larger than 2, but got {len(input)}"
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
        elif self.task == BSEPredictTask.dipole:
            return value.cpu().numpy()
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
        dipole: [nS, 1]
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
    dipole = (torch.from_numpy(label['dipole_squared']).float()).reshape(*eigenvalues.shape)
    eigenvectors = (torch.from_numpy(label['eigenvectors']).float())[None,...]

    # normalize eigenvectors
    # eigenvectors = eigenvectors / eigenvectors.amax(dim=(2, 3, 4), keepdim=True)
    dipole = dipole / dipole.sum(dim=(1, 2), keepdim=True) # this is not a distribution, don't normalize it to one
    eigenvectors = eigenvectors / eigenvectors.sum(axis=(2,3,4), keepdim=True)

    # eigenvectors = torch.log(eigenvectors + 1e-7)  # Avoid log(0)

    assert ele[0].shape[1] == nk, f"ele[0].shape[1]: {ele[0].shape[1]}, nk: {nk}"
    assert ele[0].shape[2] == nc, f"ele[0].shape[2]: {ele[0].shape[2]}, nc: {nc}"
    assert hole[0].shape[1] == nk, f"hole[0].shape[1]: {hole[0].shape[1]}, nk: {nk}"
    assert hole[0].shape[2] == nv, f"hole[0].shape[2]: {hole[0].shape[2]}, nv: {nv}"
    assert eigenvectors.shape[2] == nk, f"eigenvectors.shape[2]: {eigenvectors.shape[2]}, nk: {nk}"
    assert eigenvectors.shape[3] == nc, f"eigenvectors.shape[3]: {eigenvectors.shape[3]}, nc: {nc}"
    assert eigenvectors.shape[4] == nv, f"eigenvectors.shape[4]: {eigenvectors.shape[4]}, nv: {nv}"

    return ele, hole, eigenvalues, dipole, eigenvectors


class bse_training_manager_for_deepgwbse_paper:
    """
    Note: This class is designed to generate data and figure for deep-gwbse paper
    functionality:
     - train
     - evaluate and plot figure xx
    """
    def __init__(self, d_model:int, model_name:str, task:BSEPredictTask
                 , BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                 optimizer=torch.optim.Adam, loss=torch.nn.MSELoss, lr=1e-3,
                 **kwargs):
        self.d_model = d_model
        self.task = task
        self.model = MBformerEncoder(d_input=d_model,
                                    d_model=d_model*2, 
                                    BasisAssembly=BasisAssembly,
                                    **kwargs)
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss = loss()
        self.additional_metrics = partial(torch.nn.functional.l1_loss, reduction='mean')
        self.trainer = BSETransformerTrainer(self.model, self.loss, self.optimizer,
                                            model_name=model_name, 
                                            task=self.task,
                                            additional_metrics=self.additional_metrics)

    def load_data(self, dataset_dir:str, dataset_fname:str,  dataset_latent_fname_suffix:str,
                 eb:ManyBodyData_WFN_Embedder_pretrained, data_slice:slice=None, train_val_split:float=0.2,**kwargs):

        self.dataset_dir = dataset_dir
        self.dataset_fname = dataset_fname
        self.dataset_latent_fname = dataset_fname.split('.')[0] + dataset_latent_fname_suffix

        if not os.path.exists(os.path.join(self.dataset_dir, self.dataset_latent_fname)):
            print(f"latent dataset not found, creating new one")
            # create latent_dataset
            self.bsedata = ManyBodyData.from_existing_dataset(os.path.join(self.dataset_dir, self.dataset_fname))
            self.bsedata = eb.create_latent_for_ManyBodyData_h5(self.bsedata, dataset_dir=self.dataset_dir, dataset_fname=self.dataset_latent_fname)
        else:
            print(f"latent dataset found, using {os.path.join(self.dataset_dir, self.dataset_latent_fname)}")
            # directly read the latent_dataset
            self.bsedata = ManyBodyData.from_existing_dataset(os.path.join(self.dataset_dir, self.dataset_latent_fname), data_slice=data_slice)

        self.bsedata_train = self.bsedata[:int(len(self.bsedata)*train_val_split)]
        self.bsedata_val = self.bsedata[int(len(self.bsedata)*train_val_split):]
        self.dataloader_train = DataLoader(self.bsedata_train, batch_size=1, collate_fn=bse_collate_fn)
        self.dataloader_val = DataLoader(self.bsedata_val, batch_size=1, collate_fn=bse_collate_fn)
        return self.dataloader_train, self.dataloader_val

    def train(self, num_epoches:int, continued:bool=True):
        self.trainer.load_model(True)
        self.trainer.train(num_epoches, self.dataloader_train, self.dataloader_val, continued=continued)

    def evaluate_dataset(self, dataloader=None):
        # see test r2
        # if valid_dataset:
        #     self.dataloader = self.dataloader_val
        # else:
        #     self.dataloader = self.dataloader_train
        assert dataloader is not None, "dataloader is None, please load data first"
        loss = 0
        r2 = 0
        eval_original = torch.tensor([])
        eval_pred = np.array([])
        self.trainer.load_model(load_best=True)
        for d in dataloader:
            ele, hole, eigenvalues, dipole, eigenvectors = d
            eigval_sort, eigvec_sort, dipole = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, None, dipole) 
            # break

            if self.trainer.task == BSEPredictTask.eigenvalues:    
                original = eigval_sort
            elif self.trainer.task == BSEPredictTask.dipole:
                original = dipole
            else:
                raise NotImplementedError("Task not implemented")

            loss += mean_absolute_error(self.trainer.evaluate(d).ravel(), original.ravel())
            r2 += r2_score(self.trainer.evaluate(d).ravel(), original.ravel())

            eval_original = torch.cat((eval_original, original.ravel()))
            eval_pred = np.concatenate((eval_pred, self.trainer.evaluate(d).ravel()), axis=0)

            print('\n')
            print("eigenval:", mean_absolute_error(self.trainer.evaluate(d).ravel(), original.ravel()), 'eV')
            print('r2:', r2_score(self.trainer.evaluate(d).ravel(), original.ravel()))

        print('\nMAE:', loss/len(dataloader))
        print('R2:', r2/len(dataloader))
        np.savetxt('data1.dat', eval_original)
        np.savetxt('data2.dat', eval_pred)

        # For plotting energy vs energy (ML)!
        # cnt = 0
        # for d in self.dataloader_val:
        #     ele, hole, eigenvalues, eigenvectors = d
        #     eigval_sort, eigvec_sort = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, eigenvectors) 
        #     plt.figure(figsize=(5,5))
        #     plt.scatter(np.sort(self.bse_trainer_eigval.evaluate(d).ravel())[:], np.sort(eigval_sort.ravel())[:], s=1, alpha=1)
        #     plt.xlim(5,15)
        #     plt.ylim(5,15)
        #     if cnt == 1:
        #         break
        #     else:
        #         cnt += 2

        # np.savetxt('data.dat',self.bse_trainer_eigval.evaluate(d).ravel())
        # np.savetxt('data.dat',eigval_sort.ravel())

    def fig_a(self):
        # #f = h5.File(os.path.join(dataset_dir, dataset_latent_fname),'r')
        # # r2_score(f[f['info/mat_id'][100]]['label']['eigenvalues'][:3600], f[f['info/mat_id'][440]]['label']['eigenvalues'][:3600])
        pass


if __name__ == "__main__":  
    
    """A Basic way to train and evaluate the model"""
    # d_model = 24
    # num_epoches = 200
    # train_val_split = 0.5
    # dataset_dir = './dataset'
    # dataset_fname = 'dataset_BSE.h5'
    # dataset_latent_fname = dataset_fname.split('.')[0] + '_latent.h5'
    # # data_slice = slice(0,-1)
    # data_slice = None

    # if not os.path.exists(os.path.join(dataset_dir, dataset_latent_fname)):
    #     print(f"latent dataset not found, creating new one")
    #     # create latent_dataset
    #     bsedata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_fname))
    #     eb = ManyBodyData_WFN_Embedder_pretrained(d_model, SimpleSumXYEmbedder)
    #     bsedata = eb.create_latent_for_ManyBodyData_h5(bsedata, dataset_dir=dataset_dir, dataset_fname=dataset_latent_fname)
    #     # bsedata = eb.create_latent_for_ManyBodyData(bsedata, del_wfn_original=True)

    # else:
    #     print(f"latent dataset found, using {os.path.join(dataset_dir, dataset_latent_fname)}")
    #     # directly read the latent_dataset
    #     bsedata = ManyBodyData.from_existing_dataset(os.path.join(dataset_dir, dataset_latent_fname), data_slice=data_slice)

    # print('loaded latent dataset')

    # bsedata_train = bsedata[:int(len(bsedata)*train_val_split)]
    # bsedata_val = bsedata[int(len(bsedata)*train_val_split):]
    # dataloader_train = DataLoader(bsedata_train, batch_size=1, collate_fn=bse_collate_fn)
    # dataloader_val = DataLoader(bsedata_val, batch_size=1, collate_fn=bse_collate_fn)

    # # dataloader = DataLoader(bsedata, 
    # #                         batch_size=1, 
    # #                         collate_fn=bse_collate_fn)

    # enc2 = MBformerEncoder(d_input=d_model, 
    #                        d_model=d_model*2, 
    #                        BasisAssembly=ElectronHoleBasisAssembly_Concatenate)
    
    # optimizer = torch.optim.Adam(enc2.parameters(), lr=1e-3)
    # loss = torch.nn.MSELoss()
    # # additional_metrics=MeanAbsoluteError()  # Ensure it's on GPU if needed
    # additional_metrics = partial(torch.nn.functional.l1_loss, reduction='mean')

    # # bse_trainer_eigval = BSETransformerTrainer(enc2, loss, optimizer,  
    # #                                             model_name="bse_transformer_eval", 
    # #                                             task=BSEPredictTask.eigenvalues,
    # #                                             additional_metrics=additional_metrics)
    # # bse_trainer_eigval.load_model(True)
    # # bse_trainer_eigval.train(num_epoches, dataloader_train, dataloader_val, continued=True)


    # bse_trainer_eigvec = BSETransformerTrainer(enc2, loss, optimizer,
    #                                            model_name="bse_transformer_evec",
    #                                            task=BSEPredictTask.eigenvectors,
    #                                            additional_metrics=additional_metrics)
    # bse_trainer_eigvec.load_model(load_best=True)
    # bse_trainer_eigvec.train(num_epoches, dataloader_train, dataloader_val, continued=True)

    # loss = 0
    # r2 = 0
    # bse_trainer_eigval.load_model(load_best=True)
    # for d in dataloader_val:
    #     ele, hole, eigenvalues, eigenvectors = d
    #     eigval_sort, eigvec_sort = sort_exciton_eigenvalues_by_eh_pair_energy(ele, hole, eigenvalues, eigenvectors) 
    #     # break
    #     loss += mean_absolute_error(bse_trainer_eigval.evaluate(d).ravel(), eigval_sort.ravel())
    #     r2 += r2_score(bse_trainer_eigval.evaluate(d).ravel(), eigval_sort.ravel())

    #     print('\n')
    #     print("eigenval:", mean_absolute_error(bse_trainer_eigval.evaluate(d).ravel(), eigval_sort.ravel()), 'eV')
    #     print('r2:', r2_score(bse_trainer_eigval.evaluate(d).ravel(), eigval_sort.ravel()))

    # print('MAE:', loss/len(dataloader_val))
    # print('R2:', r2/len(dataloader_val))
    # bse_trainer_eigval.load_model(load_best=True)
    # print('eigenvec:', mean_absolute_error(bse_trainer_eigval.evaluate(d).ravel(), eigvec_sort.ravel()))

        
    vae = Trainer.configure_model(EquivariantVAE, "./vae_e2_wfn.save")
    eb = ManyBodyData_WFN_Embedder_pretrained(48, E2VAEEmbedder, model=vae, model_name='vae_e2_wfn', 
                                              save_path='../vae_e2_wfn.save')

    dataset_kwargs = {
        'dataset_dir': './dataset',
        'dataset_fname': 'dataset_BSE.h5',
        'dataset_latent_fname_suffix': '_latent_vae.h5',
    }


    """Training for BSE eigenvalues with pretrained VAE"""
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, model_name='bse_transformer_eval_vae_test', task=BSEPredictTask.eigenvalues,
                                            BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                            optimizer=torch.optim.Adam, loss=torch.nn.MSELoss, lr=1e-3)


    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_eval_vae_test.save")

    """Training for BSE eigenvectors with pretrained VAE"""
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, model_name='bse_transformer_evec_vae_test', task=BSEPredictTask.eigenvectors,
                                            BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                            optimizer=torch.optim.Adam, loss=torch.nn.CrossEntropyLoss, lr=1e-3)
    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    # bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_evec_vae_test.save")

    """Training for BSE dipole with pretrained VAE"""
    bse = bse_training_manager_for_deepgwbse_paper(d_model=48, model_name='bse_transformer_dipole_vae_test', task=BSEPredictTask.dipole,
                                            BasisAssembly=ElectronHoleBasisAssembly_Concatenate,
                                            optimizer=torch.optim.Adam, loss=torch.nn.MSELoss, lr=1e-3)
    bse.load_data(eb=eb, data_slice=None, train_val_split=0.5, **dataset_kwargs)
    bse.train(5, continued=False)

    _, dataloader_val = bse.load_data(eb=eb, data_slice=slice(-20,-1), train_val_split=0.5, **dataset_kwargs)
    bse.evaluate_dataset(dataloader_val)

    shutil.rmtree("bse_transformer_dipole_vae_test.save")