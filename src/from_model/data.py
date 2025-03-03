import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from interface import wfn
from model_util import time_watch, memory_watch
import os
from tqdm import tqdm
from os.path import join as pjoin
import h5py as h5
import logging
"""
Author: Bowen Hou
Developer: Bowen Hou, Xian Xu
Date: 2025-03-03
"""

class ManyBodyData(Dataset):
    """
raw_data_dir(flows)/
├── mat-1
|   ├──02-wfn
|   ├──13-sigma
|   |   └── eqp1.dat # (G0W0 corr.)
|   | ...
|   ├──17-wfn_fi
|   ├──18-kernel 
|   ├──19-absorption 
├── mat-2
|   └──  ...
└── ..."""

    """Output if workflow is 'WFN':
Note: N_bands_i, N_kpoints_i, Rx_i could be different for different materials
wfndata.h5
├── mat-1/data (data.shape = (N_bands_1, N_kpoints_1, Rx_1, Ry_1, Rz_1_truncated))
├── mat-2/data
├── mat-3/data
├── mat-4/...
    """

    def __init__(self, flows_dir: str, dataset_dir: str, dataset_type: str='WFN',
                 dataset_name: str='', multiprocessing: bool = False, load_dataset: bool = True, 
                 **kwargs):
        """
        :param flows_dir: Path to the raw data directory (flows)
        :param dataset_dir: Path to the dataset directory
        :param dataset_type: Workflow to process data, support ['WFN', 'GW','BSE'] now.

            'WFN': used to train VAE model (unsupervised)
                -  required dir: '02-wfn'
                -  required kwargs: nc_wfn, nv_wfn
                -  optional kwargs: cutoff, useWigner

            'GW': used to train GW-Transformer (supervised)
                -  required dir: '01-density','02-wfn', '13-sigma'
                -  required kwargs: 
                -  optional kwargs: 

            'BSE': used to train BSE-Transformer (supervised)
                -  required dir: '01-density','17-wfn_fi', '18-kernel', '19-absorption'
                -  required kwargs: 
                -  optional kwargs: 

        :param dataset_name: Name of the dataset
        :param multiprocessing: Whether to use multiprocessing to process data
        :param load_dataset: Whether to load existing dataset
        :param **kwargs:
            'GW'
            nc_wfn:
            nv_wfn:
        """
        super(ManyBodyData, self).__init__()
        assert dataset_type in ['WFN','GW','BSE'], f"dataset_type should be one of ['WFN','GW','BSE']"

        self.multiprocessing = multiprocessing
        self.flows_dir = flows_dir
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        
        # dataset and hyperparameters
        # - required:
        self.data = None

        # - optional:
        # WFN:
        self.isWFNDataset = False
        self.nc_wfn = None
        self.nv_wfn = None
        self.cutoff = None
        self.useWigner = None

        # GW:
        self.isGWDataset = False
        
        # BSE:
        self.isBSEDataset = False


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    @classmethod
    def from_existing_dataset(cls, existing_dataset_dir: str) -> 'ManyBodyData':
        return cls()

    def process_worker_WFN(self, folder, wfn_dir='02-wfn'):
        wfn_fname = pjoin(pjoin(folder, wfn_dir, "wfn.h5"))
        dateset_h5_fname = pjoin(self.dataset_dir, self.dataset_type+f'_datadet_{self.dataset_name}.h5')
        mat_id = os.path.basename(folder)
        assert wfn_dir in ['02-wfn'], f"Only support wfn_dir = '02-wfn' now"
        assert os.path.exists(wfn_fname), f"{wfn_fname} does not exist"
        assert os.path.exists(dateset_h5_fname), f"Dataset h5 file does not exist"
        assert {"nc_wfn","nv_wfn"} <= set(self.kwargs.keys()), f"nc_wfn and nv_wfn are required kwargs for WFN dataset"

        nc, nv = self.kwargs.get('nc_wfn'), self.kwargs.get('nv_wfn')
        # TODO: use **kwargs when inteface.py/wfn.get_wfn_dataset() is updated
        wf = wfn(wfn_fname)

        datapoint =  wf.get_wfn_dataset(nc=nc, nv=nv)

        with h5.File(dateset_h5_fname, 'a') as f:
            if mat_id in f:
                del f[mat_id]
            f.create_group(mat_id)
            for key, val in datapoint.items():
                f[mat_id].create_dataset(key, data=val)

        return datapoint
        
    def process(self):
        """
        folder_list: List["flow-mat-1", "flow-mat-2"]
        """
        # get materials list
        folder_list = []
        print(f'Looking for flows data under: {os.path.abspath(self.flows_dir)}')
        for root, dirs, files in os.walk(self.flows_dir):
            if "01-density" in dirs: # scf is foundation for all workflows
                folder_list.append(root)
        assert len(folder_list) > 0, f"No data found under {self.flows_dir}"
        print(f"Found {len(folder_list)} materials")

        # initialize dataset h5 file
        os.makedirs(self.dataset_dir, exist_ok=True)
        with h5.File(pjoin(self.dataset_dir, self.dataset_type+f'_datadet_{self.dataset_name}.h5'), 'w') as f:
            print(f"Creating dataset file: {os.path.abspath(f.filename)} \n")

        #==================Create Dataset==================#
        if self.dataset_type == 'WFN':
            self.data_list = [self.process_worker_WFN(folder) for folder in tqdm(folder_list, desc='Processing WFN data')]
        elif self.dataset_type == 'GW':
            raise NotImplementedError
        elif self.dataset_type == 'BSE':
            raise NotImplementedError


    def summary(self):
        pass


class ToyDataSet(Dataset):

    """
    For testing purposes, we will use a toy dataset

    Note: each material is a "sentenece" in the transformer model
    nk*nb: number of "words" in the "sentence"
    d_latent: dimension of the "word" embedding

    """
    d_model = 24 # divisible by 24
    batch_size = 10 #
    nk_max = 12*12
    nc_max = 8
    nv_max = 2
    nb_max = nc_max + nv_max
    d_latent = 12

    # BSE data
    cond_embedding = torch.rand((batch_size, nk_max, nc_max, d_model)) # after VAE-Embeeding
    val_embedding = torch.rand((batch_size, nk_max, nv_max, d_model)) # after VAE-Embeeding
    cond_band_index = torch.arange(1,nc_max+1)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    val_band_index = torch.arange(-1,-nv_max-1,-1)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    cond_band_energy = torch.rand(nc_max)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    val_band_energy = torch.rand(nv_max)[None, None, :, None].repeat(batch_size, nk_max, 1,1)

    cond_kpt = torch.rand((batch_size, nk_max, 1, 3)).repeat(1, 1, nc_max, 1)
    val_kpt = torch.rand((batch_size, nk_max, 1, 3)).repeat(1, 1, nv_max, 1)
    cond_kpt_weight = torch.rand((batch_size, nk_max, 1, 1)).repeat(1, 1, nc_max, 1)
    val_kpt_weight = torch.rand((batch_size, nk_max, 1, 1)).repeat(1, 1, nv_max, 1)

    @classmethod
    def get_ele_data_batch(cls):
        return [cls.cond_embedding, cls.cond_kpt, cls.cond_band_index, cls.cond_band_energy]

    @classmethod
    def get_hole_data_batch(cls):
        return [cls.val_embedding, cls.val_kpt, cls.val_band_index, cls.val_band_energy]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    wfdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='WFN', dataset_name='',
                          nc_wfn=4, nv_wfn=2)    
    wfdata.process()

