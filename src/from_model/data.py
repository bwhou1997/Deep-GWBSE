import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from interface import wfn
from model_util import time_watch, memory_watch
from pathos.multiprocessing import ProcessingPool as Pool
import os
from tqdm import tqdm
from os.path import join as pjoin
import h5py as h5
import logging
import numpy as np
import copy
"""
Author: Bowen Hou
Date: 2025-03-03
"""

class DataSetInfo:
    """
    Basic Info for the dataset
    dataset_type = WFN, GW, BSE

    This class is used as variable register for different dataset types
    The registered attributes will be saved into info dict
    """
    def __init__(self, dataset_type: str='WFN', **kwargs):
        if isinstance(dataset_type, bytes):
            dataset_type = dataset_type.decode('utf-8')

        if dataset_type == 'WFN':
            self.dataset_type = 'WFN'
            self.wfn_base_set(**kwargs)

        if dataset_type == 'GW':
            self.dataset_type = 'GW'
            self.eqp_base_set(**kwargs)

            # for src and tgt
            if kwargs.get('from_dft'):
                self.wfn_base_set(**kwargs)
            else:
                raise NotImplementedError("GW dataset from non-DFT is not implemented yet")
     
        if dataset_type == 'BSE':
            self.dataset_type = 'BSE'
        
        # common attributes
        self.mat_id = kwargs.get('mat_id', []) # updated after processing
    
    def wfn_base_set(self, **kwargs):
        """
        see interface.py/wfn.get_wfn_dataset()
        """
        assert {"nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc_wfn and nv_wfn are required kwargs for WFN dataset"
        self.nc_wfn = kwargs.get('nc_wfn')
        self.nv_wfn = kwargs.get('nv_wfn')
        self.useWignerXY = kwargs.get('useWignerXY', np.nan)
        self.cell_slab_truncation = kwargs.get('cell_slab_truncation', np.nan) # Required for Wigner
        self.AngstromPerPixel = kwargs.get('AngstromPerPixel', np.nan) # Required for Wigner
        self.AngstromPerPixel_z = kwargs.get('AngstromPerPixel_z', np.nan) # Required for Wigner
        self.upsampling_factor = kwargs.get('upsampling_factor', np.nan) # Required for Wigner

    def eqp_base_set(self, **kwargs):
        assert {"nc_sigma","nv_sigma","from_dft"} <= set(kwargs.keys()), f"nc_wfn, nv_wfn, from_dft are required kwargs for WFN dataset"
        self.nc_wfn = kwargs.get('nc_sigma')
        self.nv_wfn = kwargs.get('nv_sigma')      
        self.from_dft = kwargs.get('from_dft', False)
        self.prdict_only = kwargs.get('predict_only', False)


    def show_info(self,):
        print(f"\n======{str(self.dataset_type)} Dataset Info:=======")
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("Total number of data: ", len(self.mat_id),'\n\n')


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
    
    dataset general format:
    dataset.h5
    ├── info: see DataSetInfo
    ├── mat-1/{datapoint1}
    ├── mat-2/{datapoint2}
    ├── mat-3/...
    """

    def __init__(self, flows_dir: str, dataset_dir: str, dataset_type: str='WFN',
                 dataset_fname: str='dataset.h5', multiprocessing: bool = False, load_dataset: bool = True, 
                 onlySave:bool=False, **kwargs):
        """
        :param **kwargs: all parameters related to specific dataset ['WFN','GW','BSE'], see DataSetInfo
        :param flows_dir: Path to the raw data directory (flows)
        :param dataset_dir: Path to the dataset directory
        :param dataset_type: Workflow to process data, support ['WFN', 'GW','BSE'] now.

            'WFN': used to train VAE model (unsupervised)
                -  required dir: '02-wfn'
                -  required kwargs: nc_wfn, nv_wfn
                -  [optional] kwargs : useWignerXY, cell_slab_truncation, AngstromPerPixel, AngstromPerPixel_z
                                     upsample_factor (This is highly recommened for fast_cK)
                -  datapoint (see interface.py/wfn.get_wfn_dataset()):
                            {'wfn': (nk, nc_wfn+nv_wfn, Rx, Ry, Rz(cutoff)), 'kpt': (nk, 3), 'occ': (nk, nc_wfn_nv_wfn, 1),
                             'el': (nk, nc_wfn+nv_wfn 1), 'kpt_weights': (nk, nc_wfn+nv_wfn, 1), 'kpt': (nk, nc_wfn+nv_wfn, 3),
                             'band_indices': (nk, nc_wfn+nv_wfn, 1), 'band_indices_abs':(nk, nc_wfn+nv_wfn, 1)}

            'GW': used to train GW-Transformer (supervised)
                -  required dir: '02-wfn', '13-sigma', 
                -  [optional] dir: '05-band'[optional: predict_only]
                -  required kwargs: nc_wfn, nv_wfn, nc_sigma, nv_sigma, from_dft: bool
                -  [optional] kwargs: predict_only:bool=False, 
                                      other parameters are same as WFN
                -  datapoint: 
                    from_dft:
                    - {'src':dict, 'tgt':dict, 'label':dict[optional: not created if predict_only]}
                    -  src: same as "WFN" datapoint
                    -  tgt: same as "WFN" datapoint, with nc_sigma, nv_sigma instead of nc_wfn, nv_wfn
                    -  label: {'mf': (nk, nc_sigma+nv_sigma, 1), 'qp': (nk, nc_sigma+nv_sigma, 1), 'corr': (nk, nc_sigma+nv_sigma, 1)}
                    not from_dft:
                    - {'src':dict, 'tgt':dict, 'label':dict[optional: not created if predict_only]}
                    - src: not implemented yet
                    - tgt: not implemented yet
                    - label: same as from_dft
                    
            'BSE': used to train BSE-Transformer (supervised)
                -  required dir: '01-density','17-wfn_fi', '18-kernel', '19-absorption'
                -  required kwargs: 
                -  [optional] kwargs: 

        :param dataset_name: name of the dataset
        :param multiprocessing: Whether to use multiprocessing to process data
        :param load_dataset: Whether to load existing dataset
        :param onlySave (TODO): Whether to only save the dataset without loading (used for large dataset)

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
                   multiprocessing=False,
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
                    self.info.__dict__[key] = value[()].decode('utf-8')
                    continue
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
        #==================General Setting==================#
        #====Filter valid folders===
        folder_list, self.info.mat_id = self.mat_statistics(self.flows_dir, self.dataset_type)

        #===initialize dataset h5 file===
        self.init_dataset_h5(self.multiprocessing)

        #==================Dataset Specific Setting==================#
        #===Get processor===
        if self.dataset_type == 'WFN':
            processor = self.process_worker_WFN
        elif self.dataset_type == 'GW':
            raise NotImplementedError
        elif self.dataset_type == 'BSE':
            raise NotImplementedError
        
        #===Process data===
        if self.multiprocessing:
            with Pool() as pool:
                self.data = list(tqdm(pool.imap(processor, folder_list), total=len(folder_list), desc='Processing WFN data'))
                self.merge_dataset_h5(list(map(lambda x: x.decode('utf-8'), self.info.mat_id)), save_original=False, dataset_fname=self.dataset_fname)
        else:
            self.data = [processor(folder) for folder in tqdm(folder_list, desc='Processing WFN data')]

    def init_dataset_h5(self, multiprocessing: bool = False):
        """
        multiprocessing: 
            True: create dataset files for each material
                  h5: mat_id+dataset_fname
            False: create one dataset file for all materials
                  h5: dataset_fname
        """
        os.makedirs(self.dataset_dir, exist_ok=True)    
        if not multiprocessing:
            with h5.File(pjoin(self.dataset_dir, self.dataset_fname), 'w') as f:
                # put info dict into h5 file
                f.create_group('info')
                for key, value in self.info.__dict__.items():
                    f['info'].create_dataset(key, data=value)
                print(f"[Series]: creating dataset file: {os.path.abspath(f.filename)}")

        else:
            print(f"[Pool]: creating dataset files for {len(self.info.mat_id)} material")
            mat_id_list = list(map(lambda x: x.decode('utf-8'), self.info.mat_id))
            for mat_id in mat_id_list:
                with h5.File(pjoin(self.dataset_dir, mat_id+self.dataset_fname), 'w') as f:
                    # put info dict into h5 file
                    f.create_group('info')
                    for key, value in self.info.__dict__.items():
                        f['info'].create_dataset(key, data=value)
                    # print(f"Creating dataset file: {os.path.abspath(f.filename)}")

    def merge_dataset_h5(self, mat_id_list: list, save_original: bool = False, dataset_fname: str='dataset.h5'):
        """
        Merge dataset h5 files into one
        """
        print("Merging dataset h5 files", [mat_id+dataset_fname for mat_id in mat_id_list])

        self.init_dataset_h5(multiprocessing=False)

        for mat_id in mat_id_list:
            with h5.File(pjoin(self.dataset_dir, mat_id+dataset_fname), 'r') as f:
                with h5.File(pjoin(self.dataset_dir, dataset_fname), 'a') as f_new:
                    # make sure info is the same
                    for key, value in f['info'].items():
                        if key == 'dataset_type':
                            assert self.info.dataset_type == value[()].decode('utf-8'), f"Dataset type mismatch: set {self.info.dataset_type}, get {value[()].decode('utf-8')}"
                            continue
                        assert (self.info.__dict__[key] == value[()]).all(), f"Info mismatch: set {self.info.__dict__[key]}, get {value[()]}"
                    if mat_id in f_new:
                        del f_new[mat_id]
                    f_new.create_group(mat_id)
                    for key, val in f[mat_id].items():
                        f_new[mat_id].create_dataset(key, data=val[()])
            if not save_original:
                os.remove(pjoin(self.dataset_dir, mat_id+dataset_fname))
                
    @classmethod
    def mat_statistics(cls, flows_dir:str, dataset_type:type='WFN')-> tuple[list, np.ndarray]:
        """
        classmethod:
            Get the statistics of the dataset
            For different task, the required folders are different (see __init__)
            return: 
                folder_list: List["flow-mat-1", "flow-mat-2"]
                self.info.mat_id: np.array(["mat-1", "mat-2"], dtype='S')
        """
        folder_list = []

        print(f'Looking for flows data under: {os.path.abspath(flows_dir)}')
        for root, dirs, files in os.walk(flows_dir):
            # add rules to filter valid folders

            if dataset_type == 'WFN':
                if '02-wfn' in dirs: # scf is foundation for all workflows
                    if os.path.exists(pjoin(pjoin(root, "02-wfn/wfn.h5"))):
                        folder_list.append(root)

            elif dataset_type == 'GW':
                raise NotImplementedError
            
            elif dataset_type == 'BSE':
                raise NotImplementedError
            
        assert len(folder_list) > 0, f"No data found under {flows_dir}"
        print(f"Found {len(folder_list)} materials")

        mat_id = np.array([os.path.basename(folder) for folder in folder_list], dtype='S')

        return folder_list, mat_id

    def process_worker_WFN(self, folder:str)-> dict:
        """
        This function processes the WFN data for a single material
        folder: flow folder (not flows)
        """
        wfn_dir='02-wfn'
        wfn_fname = pjoin(pjoin(folder, wfn_dir, "wfn.h5"))
        mat_id = os.path.basename(folder)

        # if use multiprocessing, save data to mat_id+dataset_fname
        # else save data to dataset_fname
        if not self.multiprocessing:
            dataset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        else:
            dataset_h5_fname = pjoin(self.dataset_dir, mat_id+self.dataset_fname)

        assert os.path.exists(wfn_fname), f"{wfn_fname} does not exist"
        assert os.path.exists(dataset_h5_fname), f"Dataset h5 file does not exist"

        kwargs = copy.deepcopy(self.kwargs)

        nc, nv = kwargs.pop('nc_wfn'), kwargs.pop('nv_wfn')
        # TODO: use **kwargs when inteface.py/wfn.get_wfn_dataset() is updated
        wf = wfn(wfn_fname)

        datapoint =  wf.get_wfn_dataset(nc=nc, nv=nv, **kwargs)

        with h5.File(dataset_h5_fname, 'a') as f:
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
                          load_dataset=False, nc_wfn=4, nv_wfn=2, cell_slab_truncation=30, useWignerXY=True, 
                        AngstromPerPixel=0.1, AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True)    

    # 2. Load existing dataset: classmethod (Recommend)
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset.h5')

    # 3. Load existing dataset: using load_dataset=True (Not recommend)
    # wfdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='WFN',
    #                       load_dataset=True, nc_wfn=4, nv_wfn=2)    

    """Unit Test"""
    assert abs(wfdata[1]['wfn'][0,0,14,13,15] - 2.1230801376011337e-06 < 1e-10), "Unit Test Failed"
    print("Unit Test Passed")