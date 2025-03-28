from abc import ABC, abstractmethod
from data import ManyBodyData
import numpy as np
from model_util import H5ls
import matplotlib.pyplot as plt
from tqdm import tqdm

class ManyBodyData_WFN_Embedder_pretrained:
    """
    This Embedder Class only support static pretrained embedder embedder
    i.e. it cannot be integrated downstream model for training.
    """
    def __init__(self, latent_dim, latent_embedder , **kwargs):
        """
        Initialize the WFNEmbedder with any necessary parameters.
        latent_dim (int): The dimension of the latent space.
        latent_embedder (LatentEmbedderBASE): The latent embedder class.
        kwargs:
            ...
        """
        self.latent_embedder = latent_embedder(latent_dim, **kwargs)

        pass
    def create_latent_for_ManyBodyData(self, manybodydata: ManyBodyData, del_wfn_original=False)->ManyBodyData:
        """
        Abstract method to perform wavefunction embedding from ManyBodyData.
        Args:
            manybodydata (ManyBodyData): The ManyBodyData object, see ManyBodyData.py "WFN" datapoint for details
                ...
                wfn_datapoint = {...}
                ...
            del_wfn_original (bool): Whether to delete the original WFN data (nk, nc, nx, ny, nz) after embedding.
                
        Returns:
            manybodydata (ManyBodyData): Modified ManyBodyData object.
                ...
                wfn_datapoint = {..., 'latent': (nk, nc_wfn+nv_wfn, latent) ,...}
                ...
        Note: support `WFN`, `GW`, and `BSE`
        """
        assert manybodydata.info.dataset_type in ['WFN', 'GW', 'BSE'], "only support dataset of `WFN`, `GW`, and `BSE`"

        # This function will add a "latent" key to each wfn_data point and this is IN-PLACE!
        update_wfn_data = lambda wfn_data: wfn_data.update({'latent': self.latent_embedder.embed(wfn_data['wfn'])}) 
        update_src_data = lambda data: data['src'].update({'latent': self.latent_embedder.embed(data['src']['wfn'])}) 
        update_tgt_data = lambda data: data['tgt'].update({'latent': self.latent_embedder.embed(data['tgt']['wfn'])})

        if manybodydata.info.dataset_type == 'WFN':
            list(map(update_wfn_data, tqdm(manybodydata, desc='Embedding WFN'))) 
            if del_wfn_original:
                list(map(lambda data: data.pop('wfn'), manybodydata))

        elif manybodydata.info.dataset_type == 'GW':
            list(map(update_src_data, tqdm(manybodydata, desc='Embedding GW src WFN')))
            list(map(update_tgt_data, tqdm(manybodydata, desc='Embedding GW tgt WFN')))
            if del_wfn_original:
                list(map(lambda data: data['src'].pop('wfn'), manybodydata))
                list(map(lambda data: data['tgt'].pop('wfn'), manybodydata))

        elif manybodydata.info.dataset_type == 'BSE':
            list(map(update_src_data, tqdm(manybodydata, desc='Embedding BSE src WFN')))   
            if del_wfn_original:
                list(map(lambda data: data['src'].pop('wfn'), manybodydata)) 

        else:
            raise NotImplementedError("Task not implemented")

        return manybodydata

    def create_latent_for_ManyBodyData_h5(self, h5file, **kwargs)->ManyBodyData:
        """
        Abstract method to perform wavefunction embedding from HDF5 file.
        Args:
            h5file (str): Path to the HDF5 file.
        Returns:
        """
        raise NotImplementedError


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

    wfdata = eb.create_latent_for_ManyBodyData(wfdata, del_wfn_original=True)
    gwdata = eb.create_latent_for_ManyBodyData(gwdata, del_wfn_original=True)
    bsedata = eb.create_latent_for_ManyBodyData(bsedata, del_wfn_original=True)

    " unit test "
    assert np.allclose(gwdata[0]['src']['latent'],  wfdata[0]['latent'])
    assert abs(gwdata[0]['tgt']['latent'].sum() -  3.563698976979375) < 1e-6

    print('unit test passed!')


