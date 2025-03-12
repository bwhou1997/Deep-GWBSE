#%%
import math
from trainer import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as e2nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from e2vae import EquivariantVAE, vae_loss
from torch.utils.data import DataLoader
from data import ManyBodyData

class WFNVAETrainer(Trainer):
    """
    For options in `kwargs`, see the `__init__` function of `Trainer`.
    """
    def __init__(self, model, optimizer, beta: float, 
                 model_name="vae_e2",
                 **kwargs):
        loss = lambda recon_x, x, mu, logvar: vae_loss(recon_x, x, mu, logvar, beta=beta)
        super().__init__(model, optimizer, loss, model_name=model_name, **kwargs)
        self.beta = beta

    def get_loss(self, input)->torch.Tensor:
        x, mask = input
        x = x.to(self.device)
        mask = ~mask.to(self.device)
        x_recon, mu, logvar = self.model(x)
        return self.loss(x_recon*mask, x*mask, mu, logvar)
        
    def evaluate(self, input=None, mask=None, **kwargs):
        self.model.eval()
        
        with torch.no_grad():
            if input is None:
                assert self.validation_dataloader is not None, "Must have a non-empty input"
                for x, loaded_mask in self.validation_dataloader:
                    input = x
                    mask = loaded_mask
                    break # By default, get only one batch
            if isinstance(input, DataLoader):
                for x, loaded_mask in input:
                    input = x
                    mask = loaded_mask
                    break # By default, get only one batch
            
            input = input.to(self.device)
            x_recon, _, _ = self.model(input)
            if mask is not None:
                mask_nan = torch.where(mask, np.nan, 1.0).to(self.device)
                x_recon = mask_nan * x_recon
                input = input.to(self.device)
                input = mask_nan * input
        
        input = input.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        return input, x_recon

wfdata = ManyBodyData.from_existing_dataset('./dataset/dataset_WFN.h5')

def wfn_collate_fn(batch):
    assert len(batch)==1, "Batch size should be 1 for WFN data"
    wfn = batch[0]["wfn"]
    nk, nb, X, Y, C = wfn.shape
    # This moves the z coordinate dimension to the second dimension,
    # which, in the eyes of torchvision, is the channel dimension, which is desired.
    # The first dimension, whose size is nk*nb,
    # is the batch dimension:
    # thus one material is one batch, and its Kohn-Sham states are the samples in that batch.
    # We have to treat a material as a batch,
    # because the sizes of wave functions from different materials differ. 
    wfn = (wfn.reshape(nk*nb, X, Y, C)).transpose(0, 3, 1, 2)
    wfn = torch.from_numpy(wfn).float()
    scaling_factor = 4 # TODO: make the process determining the scaling factor automatic
    delta_X = scaling_factor * math.ceil(X / scaling_factor) - X
    delta_Y = scaling_factor * math.ceil(Y / scaling_factor) - Y
    wfn = F.pad(wfn, (0, delta_Y, 0, delta_X), mode="constant", value=np.nan)
    # Add one batch dimension; the size of the batch dimension is one,
    # because the mask is the same for all samples in one batch.
    # We rely on broadcasting to expand it to all batches.
    mask = torch.isnan(wfn[0])[None, ...] 
    # The only reason we keep nans in WFN is to demarcate the boundary of one primitive unit cell;
    # after the mask is obtained, all nans can be set to zero.
    wfn = torch.nan_to_num(wfn, nan=0.0)
    return wfn, mask

# Here batch_size is set to one, so one material in wfdata corresponds to one batch.
# Yet a material in wfdata is not a tensor:
# it is a dict containing keys like "wfn".
# In wfn_collate_fn, the wave functions are extracted from the material,
# and reorganized to conform to the standards of torchvision,
# like the channel dimension being the second dimension.
dataloader = DataLoader(wfdata, batch_size=1, collate_fn=wfn_collate_fn)

num_epoches = 500
beta = 0.02
vae = EquivariantVAE(input_channels=wfdata.info.cell_slab_truncation,
                        hidden_cnn_channels=[60,60,48,48,4],
                        hidden_pooling=[-1,0.66,-1,-1,0.66],
                        kernel_size=[7,5,5,3,3])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)


vae_trainer = WFNVAETrainer(vae, optimizer, beta=beta, model_name="vae_e2_wfn", overwrite=True, checkpoint=True, best_model=True)
vae_trainer.train(num_epoches, dataloader, dataloader)

#%%

x, x_recon = vae_trainer.evaluate(dataloader)
n_batch_sampling = 12
i_channel = 3
fig, axes = plt.subplots(2, n_batch_sampling, figsize=(n_batch_sampling, 3))
for i in range(n_batch_sampling):
    axes[0, i].imshow(x[i, i_channel, :, :])
    axes[0, i].axis("off")
    axes[1, i].imshow(x_recon[i, i_channel, :, :])
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")
plt.show()
# %%
