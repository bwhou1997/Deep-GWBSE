import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset


class ManyBodyData(Dataset):
    """
raw_data_dir/
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    

    def process_worker(self):
        pass

    def process(self):
        pass

    def summary(self):
        pass
