import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset


class ManyBodyData(Dataset):
    """
raw_data_dir(flows)/
├── mat-1
|   ├──02-wfn
|   ├──0x-wfn-kernel (todo)
|   ├──13-sigma
|   |   ├── eqp1.dat # (G0W0 corr.)
|   └── └── ...
|   ├──kernel (todo)
|   ├──absorption (todo)
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

    def __init__(self, raw_data_dir: str, dataset_dir: str, workflow: str,
                 dataset_name: str, multiprocessing: bool = False, load_dataset: bool = True,
                 N_bands: int = 20):
        """
        :param raw_data_dir: Path to the raw data directory
        :param dataset_dir: Path to the dataset directory
        :param workflow: Workflow to process data, support ['WFN', 'GW','BSE'] now.
            'WFN': used to train VAE model (unsupervised)
                - '02-wfn': wavefunction data
            'GW': used to train GW-Transformer (supervised)
                - '01-density': charge density data
                - '02-wfn': wavefunction data
                - '13-sigma': sigma data
            'BSE': used to train BSE-Transformer (supervised)
                - '01-density': charge density data
                - '02-wfn': wavefunction data
                - 'kernel': kernel data
                - 'absorption': absorption data
        :param dataset_name: Name of the dataset
        :param multiprocessing: Whether to use multiprocessing to process data
        :param load_dataset: Whether to load existing dataset
        :param N_bands: Number of bands to use
        """
        super(ManyBodyData, self).__init__()

        assert workflow in ['WFN','GW','BSE']
        self.data = None
        self.multiprocessing = multiprocessing


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    

    def process_worker_WFN(self):
        pass
        wfnfft()
        ...

    def process(self):
        pass
        # parallel over here
        for i in range(self.multiprocessing):
            self.process_worker_WFN(i)

    def summary(self):
        pass

def wfnfft():
    pass


class ToyDataSet(Dataset):

    """
    For testing purposes, we will use a toy dataset

    Note: each material is a "sentenece" in the transformer model
    nk*nb: number of "words" in the "sentence"
    d_latent: dimension of the "word" embedding

    """
    d_model = 96
    batch_size = 1 #
    nk_max = 12*12
    nc_max = 20
    nv_max = 20
    nb_max = nc_max + nv_max
    d_latent = 12

    # BSE data
    cond_embedding = torch.rand((batch_size, nk_max, nc_max, d_model)) # after VAE-Embeeding
    val_embedding = torch.rand((batch_size, nk_max, nv_max, d_model)) # after VAE-Embeeding
    cond_band_index = torch.arange(1,nc_max+1)[None, None, :, None].repeat(batch_size, nk_max, 1,1)
    val_band_index = torch.arange(-1,-nv_max-1,-1)[None, None, :, None].repeat(batch_size, nk_max, 1,1)

    cond_kpt = torch.rand((batch_size, nk_max, 1, 3)).repeat(1, 1, nc_max, 1)
    val_kpt = torch.rand((batch_size, nk_max, 1, 3)).repeat(1, 1, nv_max, 1)
    cond_kpt_weight = torch.rand((batch_size, nk_max, 1, 1)).repeat(1, 1, nc_max, 1)
    val_kpt_weight = torch.rand((batch_size, nk_max, 1, 1)).repeat(1, 1, nv_max, 1)

    @classmethod
    def get_BSE_data_batch(cls):
        return cls.cond_embedding, cls.val_embedding, cls.cond_band_index, cls.val_band_index
    
