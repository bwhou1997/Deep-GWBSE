import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.interpolate import griddata
from interface import wfn
import torch.nn.functional as F
from scipy.ndimage import zoom
import time
import torch
from scipy.interpolate import LinearNDInterpolator
from data import ManyBodyData, ToyDataSet
from torch.utils.data import DataLoader
from collect_tool import check_flows_status
from trainer import Trainer
from transformer import MBformerEncoder
from basisassembly import ElectronHoleBasisAssembly_Concatenate, sort_exciton_eigenvalues_by_eh_pair_energy, b1b2_grid
from enum import Enum
from sklearn.metrics import mean_absolute_error


class GWPredictTask(Enum):
    eigenvalues = 1
    eigenvectors = 2

def toy_wfn_embedder(dataset, wfn_latent_dim=24):
    pass

class GWTransformerTrainer(Trainer):
    pass
    # see bsetrainer.py for details

def gw_collate_fn(batch):
    pass
    # see bse_collate_fn for details

class gw_training_flow:
    # ignore this class for now
    pass

if __name__ == "__main__":
    # see bsetrainer.py for details
    pass