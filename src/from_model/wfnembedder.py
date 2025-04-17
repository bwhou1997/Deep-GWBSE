from abc import ABC, abstractmethod
from from_model.data import ManyBodyData
import numpy as np
from from_model.model_util import H5ls
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from os.path import join as pjoin

# class ManyBodyData_WFN_Embedder_pretrained:
#     """
#     This Embedder Class only support static pretrained embedder embedder
#     i.e. it cannot be integrated downstream model for training.
#     """
#     def __init__(self, latent_dim, latent_embedder , **kwargs):
#         """
#         Initialize the WFNEmbedder with any necessary parameters.
#         latent_dim (int): The dimension of the latent space.
#         latent_embedder (LatentEmbedderBASE): The latent embedder class.
#         kwargs:
#             ...
#         """
#         self.latent_embedder = latent_embedder(latent_dim, **kwargs)

#         pass
#     def create_latent_for_ManyBodyData(self, manybodydata: ManyBodyData, del_wfn_original=False)->ManyBodyData:
#         """
#         Abstract method to perform wavefunction embedding from ManyBodyData.
#         Args:
#             manybodydata (ManyBodyData): The ManyBodyData object, see ManyBodyData.py "WFN" datapoint for details
#                 ...
#                 wfn_datapoint = {...}
#                 ...
#             del_wfn_original (bool): Whether to delete the original WFN data (nk, nc, nx, ny, nz) after embedding.
                
#         Returns:
#             manybodydata (ManyBodyData): Modified ManyBodyData object.
#                 ...
#                 wfn_datapoint = {..., 'latent': (nk, nc_wfn+nv_wfn, latent) ,...}
#                 ...
#         Note: support `WFN`, `GW`, and `BSE`
#         """
#         assert manybodydata.info.dataset_type in ['WFN', 'GW', 'BSE'], "only support dataset of `WFN`, `GW`, and `BSE`"

#         # This function will add a "latent" key to each wfn_data point and this is IN-PLACE!
#         update_wfn_data = lambda wfn_data: wfn_data.update({'latent': self.latent_embedder.embed(wfn_data['wfn'])}) 
#         update_src_data = lambda data: data['src'].update({'latent': self.latent_embedder.embed(data['src']['wfn'])}) 
#         update_tgt_data = lambda data: data['tgt'].update({'latent': self.latent_embedder.embed(data['tgt']['wfn'])})

#         if manybodydata.info.dataset_type == 'WFN':
#             list(map(update_wfn_data, tqdm(manybodydata, desc='Embedding WFN'))) 
#             if del_wfn_original:
#                 list(map(lambda data: data.pop('wfn'), manybodydata))

#         elif manybodydata.info.dataset_type == 'GW':
#             list(map(update_src_data, tqdm(manybodydata, desc='Embedding GW src WFN')))
#             list(map(update_tgt_data, tqdm(manybodydata, desc='Embedding GW tgt WFN')))
#             if del_wfn_original:
#                 list(map(lambda data: data['src'].pop('wfn'), manybodydata))
#                 list(map(lambda data: data['tgt'].pop('wfn'), manybodydata))

#         elif manybodydata.info.dataset_type == 'BSE':
#             list(map(update_src_data, tqdm(manybodydata, desc='Embedding BSE src WFN')))   
#             if del_wfn_original:
#                 list(map(lambda data: data['src'].pop('wfn'), manybodydata)) 

#         else:
#             raise NotImplementedError("Task not implemented")

#         return manybodydata

#     def create_latent_for_ManyBodyData_h5(self, h5file, **kwargs)->ManyBodyData:
#         """
#         Abstract method to perform wavefunction embedding from HDF5 file.
#         Args:
#             h5file (str): Path to the HDF5 file.
#         Returns:
#         """
#         raise NotImplementedError

class ManyBodyData_WFN_Embedder_pretrained:
    def __init__(self, latent_dim, latent_embedder, **kwargs):
        self.latent_embedder = latent_embedder(latent_dim, **kwargs)
        self.del_wfn_original = False

    def _embed_wfn(self, wfn_data):
        """Helper function to embed wavefunction data."""
        wfn_data['latent'] = self.latent_embedder.embed(wfn_data['wfn'])
        if self.del_wfn_original:
            wfn_data.pop('wfn', None)
        return wfn_data

    def _embed_src(self, data):
        """Helper function to embed source wavefunction."""
        data['src']['latent'] = self.latent_embedder.embed(data['src']['wfn'])
        if self.del_wfn_original:
            data['src'].pop('wfn', None)
        return data

    def _embed_tgt(self, data):
        """Helper function to embed target wavefunction."""
        data['tgt']['latent'] = self.latent_embedder.embed(data['tgt']['wfn'])
        if self.del_wfn_original:
            data['tgt'].pop('wfn', None)
        return data

    def create_latent_for_ManyBodyData(self, manybodydata, del_wfn_original=False)->ManyBodyData:
        """
        manybodydata -> manybodydata (latent created)
        """
        self.del_wfn_original = del_wfn_original
        assert manybodydata.info.dataset_type in ['WFN', 'GW', 'BSE'], "Only support dataset of `WFN`, `GW`, and `BSE`"
        
        with Pool(processes=32) as pool:
            if manybodydata.info.dataset_type == 'WFN':
                manybodydata = list(tqdm(pool.imap(self._embed_wfn, manybodydata), total=len(manybodydata), desc='Embedding WFN'))

            elif manybodydata.info.dataset_type == 'GW':
                manybodydata = list(tqdm(pool.imap(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding GW src WFN'))
                manybodydata = list(tqdm(pool.imap(self._embed_tgt, manybodydata), total=len(manybodydata), desc='Embedding GW tgt WFN'))

            elif manybodydata.info.dataset_type == 'BSE':
                manybodydata = list(tqdm(pool.imap(self._embed_src, manybodydata), total=len(manybodydata), desc='Embedding BSE src WFN'))
                
        return manybodydata

    def create_latent_for_ManyBodyData_h5(self, manybodydata:ManyBodyData, dataset_dir:str='./', dataset_fname:str='./latent_mbdata.h5'):
        """
        manybodydata -> manybodydata, manybody_h5file (latent replacing wfn)
        """
        info = manybodydata.info
        self.info = info
        info.latent_created = True
        manybodydata = self.create_latent_for_ManyBodyData(manybodydata, del_wfn_original=True)

        # Create a new HDF5 file with the updated data
        ManyBodyData.init_dataset_h5(dataset_dir, dataset_fname, info, multiprocessing=False)

        # info.mats_id has the same order as manybodydata
        # see data.py ManyBodyData.load_dataset, ManyBodyData.mat_statistics, and ManyBodyData.process for details
        for i, mat_id in enumerate(info.mat_id):
            ManyBodyData.datapoint_interface_h5(pjoin(dataset_dir, dataset_fname), mat_id, manybodydata[i], mode='a')
        
        return manybodydata

class ManyBodyData_WFN_Embedder_trainable:
    """
    You should put this to your model!
    """
    def __init__(self): 
        raise NotImplementedError

class LatentEmbedderBASE(ABC):
    """
    Abstract base class for latent embedding methods.
    """
    @abstractmethod
    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform embedding on the input data.
        Args:
            wfn_data: The input WFN data to be embedded. each batch is a material
                (nk, nc_wfn+nv_wfn, nx, ny, nz)
        Returns:
            representation of wfn_data.
                (nk, nc_wfn+nv_wfn, latent_dim)
        """
        pass

# Here is a simple example!
class SimpleSumXYEmbedder(LatentEmbedderBASE):
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        self.kwargs = kwargs

    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        """
        This is just an example, we don't consider efficiency and physical meaning at all!
        input: wfn_data, see LateEmbedderBASE
        return: latent, see LateEmbedderBASE
        """
        wfn_data = np.copy(wfn_data) # don't change the original data
        assert len(wfn_data.shape) == 5, f"len(wfn_data.shape): {len(wfn_data.shape)}"
        nk, nc_nv, nx, ny, nz = wfn_data.shape
        # mid = np.argmax(np.nan_to_num(wfn_data,0).sum(axis=(0,1,2,3)))
        mid = nz //2
        left, right = mid - self.latent_dim//2, mid + self.latent_dim//2
        right = right if self.latent_dim%2 == 0 else right + 1
        assert left >= 0, f"left: {left}, nz: {nz}"
        assert right <= nz, f"right: {right}, nz: {nz}"
        # return wfn_data[..., left:right].sum(axis=(2,3))
        wfn_data = np.nan_to_num(wfn_data, 0)
        extracted = wfn_data[..., left:right].sum(axis=(2, 3))
        # norm_factor = np.sum(wfn_data) / np.sum(extracted) if np.sum(extracted) != 0 else 1
        # return np.ones((nk, nc_nv, self.latent_dim))
        return extracted / extracted.sum(axis=2, keepdims=True)

class OtherEmbedder(LatentEmbedderBASE):
    def __init__(self, latent_dim, **kwargs):
        pass
    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        pass



if __name__ == "__main__":
    wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')
    gwdata = ManyBodyData.from_existing_dataset('./dataset/dataset_GW.h5')
    bsedata = ManyBodyData.from_existing_dataset('./dataset/dataset_BSE.h5')

    eb = ManyBodyData_WFN_Embedder_pretrained(24, SimpleSumXYEmbedder)
    eb.create_latent_for_ManyBodyData_h5(wfdata, dataset_dir='./dataset', dataset_fname='dataset_WFN_latent.h5')
    eb.create_latent_for_ManyBodyData_h5(gwdata, dataset_dir='./dataset', dataset_fname='dataset_GW_latent.h5')
    eb.create_latent_for_ManyBodyData_h5(bsedata, dataset_dir='./dataset', dataset_fname='dataset_BSE_latent.h5')

    wfdata = eb.create_latent_for_ManyBodyData(wfdata, del_wfn_original=True)
    gwdata = eb.create_latent_for_ManyBodyData(gwdata, del_wfn_original=True)
    bsedata = eb.create_latent_for_ManyBodyData(bsedata, del_wfn_original=True)

    wfdata_h5 = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN_latent.h5')
    gwdata_h5 = ManyBodyData.from_existing_dataset('./dataset/dataset_GW_latent.h5')
    bsedata_h5 = ManyBodyData.from_existing_dataset('./dataset/dataset_BSE_latent.h5')

    " unit test "
    assert np.allclose(gwdata[0]['src']['latent'],  wfdata[0]['latent'])
    assert abs(gwdata[0]['tgt']['latent'].sum() -  4) < 1e-6
    assert np.allclose(gwdata[0]['src']['latent'],  gwdata_h5[0]['src']['latent'])
    assert np.allclose(gwdata[0]['tgt']['latent'],  gwdata_h5[0]['tgt']['latent'])
    assert np.allclose(bsedata[0]['src']['latent'],  bsedata_h5[0]['src']['latent'])
    assert np.allclose(wfdata[0]['latent'],  wfdata_h5[0]['latent'])

    print('unit test passed!')


