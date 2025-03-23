from abc import ABC, abstractmethod
from data import ManyBodyData
import numpy as np
from model_utils import H5ls

class WFNEmbedder:
    """
    Abstract base class for wavefunction embedding methods.
    """
    def __init__(self, latent_dim, latent_embedder , **kwargs):
        """
        Initialize the WFNEmbedder with any necessary parameters.
        """
        self.latent_embedder = latent_embedder(latent_dim, **kwargs)

        pass
    def create_latent_for_ManyBodyData(self, manybodydata: ManyBodyData)->ManyBodyData:
        """
        Abstract method to perform wavefunction embedding from ManyBodyData.
        Args:
            manybodydata (ManyBodyData): The ManyBodyData object to embed.
        Returns:
            ManyBodyData: The embedded ManyBodyData object.
        """
        # TODO
        return

    def create_latent_for_ManyBodyData_h5(self, h5file, **kwargs)->ManyBodyData:
        """
        Abstract method to perform wavefunction embedding from HDF5 file.
        Args:
            h5file (str): Path to the HDF5 file.
            **kwargs: Additional arguments for embedding.
        Returns:
            ManyBodyData: The embedded ManyBodyData object.
        """
        # TODO
        return


class LatentEmbedder(ABC):
    """
    Abstract base class for latent embedding methods.
    """
    @abstractmethod
    def __init__(self, latent_dim, **kwargs):
        """
        Initialize the LatentEmbedder with any necessary parameters.
        Args:
            latent_dim (int): The dimension of the latent space.
            **kwargs: Additional arguments for embedding.
        """
        pass

    @abstractmethod
    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform embedding on the input data.
        Args:
            wfn_data: The input WFN data to be embedded.
                (nk, nc_wfn+nv_wfn, nx, ny, nz)
        Returns:
            representation of wfn_data.
                (nk, nc_wfn+nv_wfn, latent_dim)
        """
        pass

# Here is a simple example!
class SimpleSumEmbedder(LatentEmbedder):
    def __init__(self, latent_dim, **kwargs):
        self.latent_dim = latent_dim
        self.kwargs = kwargs

    def embed(self, wfn_data: np.ndarray) -> np.ndarray:
        """
        This is just an example, we don't consider efficiency and physical meaning at all!
        """
        wfn_z = np.sum(wfn_data, axis=(2,3)) # (nk, nc_wfn+nv_wfn, nz)
        # find the index of z max, then only preserve the window centered on z max (the window size is latent_dim)
        z_max_index = np.argmax(wfn_z, axis=2)
        wfn_truncated = np.zeros((wfn_z.shape[0], wfn_z.shape[1], self.latent_dim))
        for i in range(wfn_z.shape[0]):
            for j in range(wfn_z.shape[1]):
                z_max = z_max_index[i,j]
                if z_max < self.latent_dim//2:
                    wfn_truncated[i,j,:] = wfn_z[i,j,:self.latent_dim]
                elif z_max > (wfn_z.shape[2]-self.latent_dim//2):
                    wfn_truncated[i,j,:] = wfn_z[i,j,-self.latent_dim:]
                else:
                    wfn_truncated[i,j,:] = wfn_z[i,j,z_max-self.latent_dim//2:z_max+self.latent_dim//2]
        return wfn_truncated


if __name__ == "__main__":
    pass



