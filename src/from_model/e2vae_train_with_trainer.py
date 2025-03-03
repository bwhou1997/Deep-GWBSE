#%%
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

class VAETrainer(Trainer):
    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, beta: float, 
                 save_path=os.getcwd(),
                 model_name="vae_e2",
                 overwrite=False):
        loss = lambda recon_x, x, mu, logvar: vae_loss(recon_x, x, mu, logvar, beta=beta)
        super().__init__(model, training_dataloader, validation_dataloader, optimizer, loss, save_path=save_path, model_name=model_name, overwrite=overwrite)
        self.beta = beta

    def get_loss(self, x: torch.Tensor)->torch.Tensor:
        x_recon, mu, logvar = self.model(x)
        return self.loss(x_recon, x, mu, logvar)

    def train_each_epoch(self, epoch_idx: int):
        total_loss = 0.0

        for x, _ in tqdm(self.training_dataloader, f"Epoch {epoch_idx+1}"):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            this_loss = self.get_loss(x)
            this_loss.backward()
            self.optimizer.step()
            total_loss += this_loss.item()
        
        self.record(total_loss / len(self.training_dataloader))
        
    def evaluate(self, input=None):
        self.model.eval()
        
        with torch.no_grad():
            if input is None:
                for x, _ in test_loader:
                    input = x
                    break # By default, get only one batch
            
            input = input.to(self.device)
            x_recon, _, _ = self.model(input)
        
        input = input.cpu().numpy()
        x_recon = x_recon.cpu().numpy()
        return input, x_recon

#%%

# Model and training strategies
num_epochs = 10
beta = 0.02
vae = EquivariantVAE(input_channels=1,
                    hidden_cnn_channels=[60,60,48,48,4],
                    hidden_pooling=[-1,0.66,-1,-1,0.66],
                    kernel_size=[7,5,5,3,3])
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# Data preparation
# The images are normalized and converted to tensors
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single channel
    transforms.ToTensor()
])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Start training!
vae_trainer = VAETrainer(vae, train_loader, test_loader, optimizer, beta=beta, model_name="vae_e2_minst")
vae_trainer.train(num_epochs)
    
    
# %%
# Mini-testing
x, x_recon = vae_trainer.evaluate()
# Plot original vs reconstructed images
fig, axes = plt.subplots(2, 10, figsize=(10, 3))
for i in range(10):
    axes[0, i].imshow(x[i, 0], cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(x_recon[i, 0], cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("Reconstructed")
plt.show()
# %%
