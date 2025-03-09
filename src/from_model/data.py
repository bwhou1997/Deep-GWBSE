import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from interface import wfn, eqp
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

    This class saves all the hyperparameters (required and optional) for the dataset
    All the hyperparameters are saved in the __dict__ attribute and with assigned default values
    """
    def __init__(self, dataset_type: str='WFN', **kwargs):
        if isinstance(dataset_type, bytes):
            dataset_type = dataset_type.decode('utf-8')

        if dataset_type == 'WFN':
            self.dataset_type = 'WFN'
            self.wfn_base_set(**kwargs)

        if dataset_type == 'GW':
            self.dataset_type = 'GW'
            self.gw_base_set(**kwargs)

            # for src and tgt
            if kwargs.get('from_dft'):
                self.wfn_base_set(**kwargs)
            else:
                raise NotImplementedError("GW dataset from non-DFT is not implemented yet")
                self.vae_base_set(**kwargs)
     
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
        self.useWignerXY = kwargs.get('useWignerXY', False)
        self.cell_slab_truncation = kwargs.get('cell_slab_truncation', 40) # Required for Wigner
        self.AngstromPerPixel = kwargs.get('AngstromPerPixel', 0.1) # Required for Wigner
        self.AngstromPerPixel_z = kwargs.get('AngstromPerPixel_z', 0.1) # Required for Wigner
        self.upsampling_factor = kwargs.get('upsampling_factor', 1) # Required for Wigner

    def gw_base_set(self, **kwargs):
        assert {"nc_sigma","nv_sigma","nc_wfn","nv_wfn"} <= set(kwargs.keys()), f"nc, nv are required kwargs for GW dataset"
        self.nc_sigma = kwargs.get('nc_sigma')
        self.nv_sigma = kwargs.get('nv_sigma')      
        self.nc_wfn = kwargs.get('nc_wfn')
        self.nv_wfn = kwargs.get('nv_wfn')
        self.from_dft = kwargs.get('from_dft', True)
        self.predict_only = kwargs.get('predict_only', False)
    
    def vae_base_set(self, **kwargs):
        pass


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
                -  required kwargs: nc_wfn, nv_wfn, nc_sigma, nv_sigma 
                -  [optional] kwargs: from_dft: bool=True, # save wfn instead of VAE latent space
                                      predict_only:bool=False, 
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
                   onlySave=False,
                   **info_dict)

    def load_dataset(self):
        """
        load existing dataset
        """

        with h5.File(pjoin(self.dataset_dir, self.dataset_fname), 'r') as f:
            print("updating info from existing dataset")
            for key, value in f['info'].items():
                if key == 'dataset_type':
                    assert self.info.dataset_type == value[()].decode('utf-8'), f"Dataset type mismatch: set {self.info.dataset_type}, get {value[()].decode('utf-8')}"
                    self.info.__dict__[key] = value[()].decode('utf-8')
                    continue
                self.info.__dict__[key] = value[()]
        
        print("loading data")
        self.data = [self.datapoint2h5(pjoin(self.dataset_dir, self.dataset_fname), mat_id, mode='r') for mat_id in self.info.mat_id]

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
            processor = self.process_worker_GW
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
                # This is not a class method, so we don't need further info check
                self.datapoint2h5(pjoin(self.dataset_dir, dataset_fname), mat_id, f[mat_id], mode='a')

            if not save_original:
                os.remove(pjoin(self.dataset_dir, mat_id+dataset_fname))


    def mat_statistics(self, flows_dir:str, dataset_type:type='WFN')-> tuple[list, np.ndarray]:
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
                if not self.info.from_dft:
                    raise NotImplementedError
                else:
                    if '02-wfn' in dirs: # scf is foundation for all workflows
                        if not self.info.predict_only:
                            if (os.path.exists(pjoin(pjoin(root, "02-wfn/wfn.h5"))) and\
                                os.path.exists(pjoin(pjoin(root, "13-sigma/eqp1.dat")))):
                                    folder_list.append(root)
                        else:
                            if (os.path.exists(pjoin(pjoin(root, "02-wfn/wfn.h5"))) and\
                                os.path.exists(pjoin(pjoin(root, "05-band/wfn.h5")))):
                                    folder_list.append(root)

            
            elif dataset_type == 'BSE':
                raise NotImplementedError
        
            else:
                raise Exception(f"Dataset type {dataset_type} is not supported")
            
        assert len(folder_list) > 0, f"No data found under {flows_dir}"
        print(f"Found {len(folder_list)} materials")

        mat_id = np.array([os.path.basename(folder) for folder in folder_list], dtype='S')

        return folder_list, mat_id

    def process_worker_WFN(self, folder:str)-> dict:
        """
        This function processes the WFN data for a single material
        this func: get kwargs -> get wfn file (diff dir should be considered in future) -> create datapoint -> save data to h5 file
        folder: flow folder (not flows)
        """
        # get info
        mat_id = os.path.basename(folder)
        info = copy.deepcopy(dict(self.info.__dict__))
        nc, nv = info.pop('nc_wfn'), info.pop('nv_wfn')

        # get wfn file
        wfn_fname = pjoin(pjoin(folder, '02-wfn', "wfn.h5"))

        # create datapoint
        wf = wfn(wfn_fname)
        datapoint =  wf.get_wfn_dataset(nc=nc, nv=nv, **info)

        # save data to h5 file
        # if use multiprocessing, save data to mat_id+dataset_fname
        # else save data to dataset_fname
        if not self.multiprocessing:
            dataset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        else:
            dataset_h5_fname = pjoin(self.dataset_dir, mat_id+self.dataset_fname)

        self.datapoint2h5(dataset_h5_fname, mat_id, datapoint, mode='a')

        return datapoint

    @classmethod
    def datapoint2h5(cls, dataset_h5_fname: str, mat_id: str, datapoint=None, mode: str = 'a'):
        """
        Save or load a datapoint to/from an HDF5 file.
        
        Parameters:
        - datapoint(two types, only required in "a"): 
            dict -> Nested dictionary structure containing data
            h5.Group(dict like structure) -> HDF5 group object
        - dataset_h5_fname: str -> Path to the HDF5 file.
        - mat_id: str -> Identifier for the dataset inside HDF5.(first level dict)
        - mode: str -> 'a' for append/write, 'r' for read.
        
        Structure:
        - Unsupervised: {"xx": np.array, "yy": np.array}
        - Supervised: {"src": {"xx": np.array, ...}, "tgt": {...}, "label": {...}}
        
        HDF5 format:
        ```
        dataset.h5
        ├── info
        └── mat_id
            ├── xx
            ├── yy
            ├── src
            │   ├── xx
            │   ├── ...
            ├── tgt
            ├── label
        ```
        """

        assert mode in ['a', 'r'], "mode should be 'a' or 'r'"

        if mode == 'a':
            with h5.File(dataset_h5_fname, mode) as f:
                if mat_id in f:
                    del f[mat_id]
                f.create_group(mat_id)

                def write_data(group, data):
                    """Recursively writes data to HDF5, handling nested dictionaries."""
                    for key, val in data.items():
                        if isinstance(val, dict) or isinstance(val, h5.Group):  # Nested dictionary
                            subgroup = group.create_group(key)
                            write_data(subgroup, val)
                        else:
                            group.create_dataset(key, data=val)

                write_data(f[mat_id], datapoint)
            return

        elif mode == 'r':
            datapoint = {}

            def read_data(group):
                """Recursively reads HDF5 data into a nested dictionary."""
                data_dict = {}
                for key, item in group.items():
                    if isinstance(item, h5.Group):  # If it's a group, recurse
                        data_dict[key] = read_data(item)
                    else:
                        data_dict[key] = item[()]  # Read dataset
                return data_dict

            with h5.File(dataset_h5_fname, 'r') as f:
                if mat_id in f:
                    datapoint = read_data(f[mat_id])

            return datapoint


    def process_worker_GW(self, folder:str)-> dict:
        """
        This function processes the GW data for a single material
        """

        # get kwargs
        datapoint = {}
        mat_id = os.path.basename(folder)
        info = copy.deepcopy(self.info.__dict__)
        nc_wfn, nv_wfn, nc_sigma, nv_sigma = info.pop('nc_wfn'), info.pop('nv_wfn'), \
                                     info.pop('nc_sigma'), info.pop('nv_sigma')

        if info.get('from_dft'):
            # build src
            wfn_fname = pjoin(pjoin(folder, '02-wfn', "wfn.h5"))
            wf = wfn(wfn_fname)
            datapoint_src =  wf.get_wfn_dataset(nc=nc_wfn, nv=nv_wfn, **info)
            datapoint['src'] = datapoint_src

            # build tgt & label
            if info.get('predict_only'):
                raise NotImplementedError
                wfn_fname = pjoin(pjoin(folder, '05-band', "wfn.h5"))
                wf = wfn(wfn_fname)
                datapoint_tgt = wf.get_wfn_dataset(nc=nc_sigma, nv=nv_sigma, **info)

            else:
                datapoint_tgt = wf.get_wfn_dataset(nc=nc_sigma, nv=nv_sigma, **info)
                eqp1 = eqp(pjoin(pjoin(folder, '13-sigma'), "eqp1.dat"))
                datapoint_eqp = eqp1.get_eqp_dataset()

                _, tgt_idx, label_idx = np.intersect1d(datapoint_tgt['band_indices_abs'][0], datapoint_eqp['band_indices_abs'][0], return_indices=True)
                assert len(tgt_idx) == len(datapoint_tgt['band_indices_abs'][0]), "selected nc_sigma, nv_sigma are not in the label"

                # select the same band indices for tgt and label
                for key, val in datapoint_eqp.items():
                    datapoint_eqp[key] = val[:,label_idx,:]
                
                datapoint['tgt'], datapoint['label'] = datapoint_tgt, datapoint_eqp

        else: # read from vae output 
            raise NotImplementedError

        # save data to h5 file
        if not self.multiprocessing:
            dataset_h5_fname = pjoin(self.dataset_dir, self.dataset_fname)
        else:
            dataset_h5_fname = pjoin(self.dataset_dir, mat_id+self.dataset_fname)
        
        self.datapoint2h5(dataset_h5_fname, mat_id, datapoint, mode='a')

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

    """WFN Usage"""
    # 1. Create new dataset
    wfdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='WFN', dataset_fname='dataset_WFN.h5',
                          load_dataset=False, cell_slab_truncation=30, useWignerXY=True, AngstromPerPixel=0.1,
                          AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                          nc_wfn=4, nv_wfn=2, )   # required line

    # 2. Load existing dataset: classmethod (Recommend)
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')

    # 3. Load existing dataset: using load_dataset=True (Not recommend)
    # wfdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='WFN',
    #                       load_dataset=True, nc_wfn=4, nv_wfn=2)    

    """WFN Unit Test"""
    assert abs(wfdata[1]['wfn'][0,0,14,13,15] - 2.1230801376011337e-06) < 1e-10, "Unit Test Failed"
    print("WFN: unit test passed")


    """GW Usage"""
    gwdata = ManyBodyData(flows_dir='../../examples/flows', dataset_dir='./dataset', dataset_type='GW', dataset_fname='dataset_GW.h5',
                          load_dataset=False, cell_slab_truncation=30, useWignerXY=True,  AngstromPerPixel=0.1, 
                          AngstromPerPixel_z=0.2, upsampling_factor=2, multiprocessing=True,
                          nc_wfn=4, nv_wfn=2,nc_sigma=1, nv_sigma=1, from_dft=True, predict_only=False,)    


    gwdata = ManyBodyData.from_existing_dataset('./dataset/dataset_GW.h5')

    assert abs(gwdata[1]['src']['wfn'][0,0,14,13,15] - 2.1230801376011337e-06) < 1e-10, "Unit Test Failed"
    assert abs(gwdata[1]['tgt']['wfn'][0,0,14,13,15] - 1.261505271449588e-07) < 1e-10, "Unit Test Failed"
    print("GW: unit test passed")