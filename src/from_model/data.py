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
import numpy as np
"""
Author: Bowen Hou
Date: 2025-03-03
"""

class DataSetInfo:
    """
    Basic Info for the dataset
    dataset_type = WFN, GW, BSE
    """
    def __init__(self, dataset_type: str='WFN', **kwargs):
        if isinstance(dataset_type, bytes):
            dataset_type = dataset_type.decode('utf-8')
        if dataset_type == 'WFN':
            self.dataset_type = 'WFN'
            assert {"nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc_wfn and nv_wfn are required kwargs for WFN dataset"
            self.nc_wfn = kwargs.get('nc_wfn')
            self.nv_wfn = kwargs.get('nv_wfn')
            self.cutoff = kwargs.get('cutoff', np.nan)
            self.useWigner = kwargs.get('useWigner', np.nan)
        
        if dataset_type == 'GW':
            self.dataset_type = 'GW'

        if dataset_type == 'BSE':
            self.dataset_type = 'BSE'
        
        # common attributes
        # update after loading dataset
        self.mat_id = kwargs.get('mat_id', [])
    
    def show_info(self,):
        print(f"\n{str(self.dataset_type)} Dataset Info:")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("Total number of data: ", len(self.mat_id))


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
    └── ...
    """

    """dataset general format:
    dataset.h5
    ├── info: see DataSetInfo
    ├── mat-1/{datapoint1}
    ├── mat-2/{datapoint2}
    ├── mat-3/...
    """

    def __init__(self, flows_dir: str, dataset_dir: str, dataset_type: str='WFN',
                 dataset_fname: str='dataset.h5', multiprocessing: bool = False, load_dataset: bool = True, 
                 **kwargs):
        """
        :param flows_dir: Path to the raw data directory (flows)
        :param dataset_dir: Path to the dataset directory
        :param dataset_type: Workflow to process data, support ['WFN', 'GW','BSE'] now.

            'WFN': used to train VAE model (unsupervised)
                -  required dir: '02-wfn'
                -  required kwargs: nc_wfn, nv_wfn
                -  optional kwargs: cutoff, useWigner
                -  datapoint: {'wfn': (nk, nc+nv, Rx, Ry, Rz(cutoff)), 'kpt': (nk, 3), 'band_indices': (nk, nc+nv, 1), 
                               'el': (nk, nc+nv, 1), 'kpt_weights': (nk, nc+nv, 1), 'kpt': (nk, nc+nv, 3)}

            'GW': used to train GW-Transformer (supervised)
                -  required dir: '01-density','02-wfn', '13-sigma'
                -  required kwargs: 
                -  optional kwargs: 

            'BSE': used to train BSE-Transformer (supervised)
                -  required dir: '01-density','17-wfn_fi', '18-kernel', '19-absorption'
                -  required kwargs: 
                -  optional kwargs: 

        :param dataset_name: name of the dataset
        :param multiprocessing: Whether to use multiprocessing to process data
        :param load_dataset: Whether to load existing dataset
        Output: 
            self.data: [datapoint1, datapoint2, ...]
            self.info: DataSetInfo
            dataset.h5:
        """
        super(ManyBodyData, self).__init__()
        assert dataset_type in ['WFN','GW','BSE'], f"dataset_type should be one of ['WFN','GW','BSE']"

        self.multiprocessing = multiprocessing
        self.flows_dir = flows_dir
        self.dataset_dir = dataset_dir
        self.dataset_type = dataset_type
        self.dataset_fname = dataset_fname
        self.kwargs = kwargs
        
        # dataset and hyperparameters
        # - required:
        self.data = None
        self.info = DataSetInfo(dataset_type=dataset_type, **kwargs)

        if load_dataset and os.path.exists(pjoin(dataset_dir, dataset_fname)):
            print(f"Loading existing dataset: {os.path.abspath(pjoin(dataset_dir, dataset_fname))}")
            self.load_dataset()
        else:
            if not os.path.exists(pjoin(dataset_dir, dataset_fname)):
                print(f"Dataset file not found: {os.path.abspath(pjoin(dataset_dir, dataset_fname))}")
            print(f"Creating new dataset: {os.path.abspath(pjoin(dataset_dir, dataset_fname))}")
            self.process()
        
        assert self.data is not None, "Data is not loaded or processed"
        self.info.show_info()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @classmethod
    def from_existing_dataset(cls, existing_dataset_fname: str) -> 'ManyBodyData':
        assert os.path.exists(existing_dataset_fname), f"{existing_dataset_fname} does not exist"
        dataset_dir, dataset_fname = os.path.dirname(existing_dataset_fname), os.path.basename(existing_dataset_fname)
        print('Loading dataset info')
        info_dict = {}
        with h5.File(existing_dataset_fname, 'r') as f:
            for key, value in f['info'].items():
                info_dict[key] = value[()]

        dataset_type = info_dict.pop('dataset_type')
        info = DataSetInfo(dataset_type,
                           **info_dict)
        # info.show_info()
        return cls(flows_dir=None, 
                   dataset_dir=dataset_dir, 
                   dataset_type=info.dataset_type, 
                   load_dataset=True, 
                   dataset_fname=dataset_fname,
                   **info_dict)

    def load_dataset(self):
        """
        load existing dataset
        """
        self.data = []
        with h5.File(pjoin(self.dataset_dir, self.dataset_fname), 'r') as f:
            print("updating info from existing dataset")
            for key, value in f['info'].items():
                if key == 'dataset_type':
                    assert self.info.dataset_type == value[()].decode('utf-8'), f"Dataset type mismatch: set {self.info.dataset_type}, get {value[()].decode('utf-8')}"
                self.info.__dict__[key] = value[()]
            print("loading data")
            for mat_id in self.info.mat_id:
                datapoint = {}
                for key, val in f[mat_id].items():
                    datapoint[key] = val[()]
                self.data.append(datapoint)

        # print(f"Loading existing dataset: {os.path.abspath(self.data.filename)}")

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
        self.info.mat_id = np.array([os.path.basename(folder) for folder in folder_list], dtype='S')

        # initialize dataset h5 file
        os.makedirs(self.dataset_dir, exist_ok=True)    
        with h5.File(pjoin(self.dataset_dir, self.dataset_fname), 'w') as f:
            # put info dict into h5 file
            f.create_group('info')
            for key, value in self.info.__dict__.items():
                f['info'].create_dataset(key, data=value)
            print(f"Creating dataset file: {os.path.abspath(f.filename)} \n")

        #==================Create Dataset==================#
        if self.dataset_type == 'WFN':
            self.data = [self.process_worker_WFN(folder) for folder in tqdm(folder_list, desc='Processing WFN data')]
        elif self.dataset_type == 'GW':
            raise NotImplementedError
        elif self.dataset_type == 'BSE':
            raise NotImplementedError


    def process_worker_WFN(self, folder, wfn_dir='02-wfn'):
        """
        This function processes the WFN data for a single material
        """
        wfn_fname = pjoin(pjoin(folder, wfn_dir, "wfn.h5"))
        dateset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        mat_id = os.path.basename(folder)
        assert wfn_dir in ['02-wfn'], f"Only support wfn_dir = '02-wfn' now"
        assert os.path.exists(wfn_fname), f"{wfn_fname} does not exist"
        assert os.path.exists(dateset_h5_fname), f"Dataset h5 file does not exist"

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

    """Usage"""
    # 1. Create new dataset
    wfdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='WFN',
                          load_dataset=False, nc_wfn=4, nv_wfn=2)    

    # 2. Load existing dataset
    # Recommend: use classmethod from_existing_dataset() to load existing dataset
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset.h5')

    # Not recommended: set load_dataset=True
    #   wfdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='WFN',
    #                       load_dataset=True, nc_wfn=4, nv_wfn=2)    